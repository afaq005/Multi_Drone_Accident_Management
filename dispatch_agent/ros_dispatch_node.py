#!/usr/bin/env python3
"""
dispatch_agent/ros_dispatch_node.py
====================================

ROS wrapper for A_disp.

Subscribes to:

    /description_agent/fused_report
        std_msgs/String containing the consolidated incident report.

For each new incident, the node:

    1. Computes the severity score using Eq. 24.
    2. Selects the nearest rescue center using Eq. 25.
    3. Synthesizes the incident summary with Piper TTS.
    4. Publishes the routed JSON alert over MQTT.
    5. Publishes the final routed alert on:

           /dispatch_agent/alert

       for logging or additional ROS consumers.

Coordinate handling
-------------------

coordinate_mode="gps":
    location must contain:
        {"lat": ..., "lon": ...}

    Haversine distance is used and the default GPS rescue centers from
    model/config.py may be used.

coordinate_mode="local":
    location must contain:
        {"x": ..., "y": ...}

    Euclidean distance is used. Local rescue-center coordinates must be
    supplied explicitly because they depend on the user's Gazebo/world
    coordinate frame.

Example:

    python dispatch_agent/ros_dispatch_node.py \
        --centers-json config/local_rescue_centers.json

Example local rescue-center JSON:

    [
        ["Rescue Center A", -150.0, 20.0],
        ["Rescue Center B", 80.0, 15.0]
    ]
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from dispatch_agent.rescue_center_router import (  # noqa: E402
    build_alert_payload,
    route_alert,
)
from dispatch_agent.tts_dispatch import (  # noqa: E402
    publish_mqtt,
    synthesize_audio,
)
from model.utils import get_logger  # noqa: E402

logger = get_logger(
    "dispatch_agent.ros"
)


def load_centers_json(
    path: str,
) -> List[Tuple[str, float, float]]:
    """
    Load rescue-center coordinates from JSON.

    Expected format:

        [
            ["Rescue Center A", -150.0, 20.0],
            ["Rescue Center B", 80.0, 15.0]
        ]
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Rescue-center file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        raw = json.load(f)

    if not isinstance(
        raw,
        list,
    ):
        raise ValueError(
            "Rescue-center JSON must contain a list."
        )

    centers = []

    for item in raw:

        if (
            not isinstance(
                item,
                (list, tuple),
            )
            or len(item) != 3
        ):
            raise ValueError(
                "Each rescue center must have the form "
                "[name, coordinate_a, coordinate_b]."
            )

        name, coord_a, coord_b = item

        centers.append(
            (
                str(name),
                float(coord_a),
                float(coord_b),
            )
        )

    if not centers:
        raise ValueError(
            "At least one rescue center is required."
        )

    return centers


def fused_report_to_alert(
    report: dict,
    centers: Optional[
        List[
            Tuple[
                str,
                float,
                float,
            ]
        ]
    ] = None,
) -> dict:
    """
    Convert one fused description report into a routed dispatch alert.

    The primary view supplies:
        - YOLO confidence
        - BLIP-2 mean token log-likelihood

    for the Eq. 24 severity calculation.
    """

    if not isinstance(
        report,
        dict,
    ):
        raise ValueError(
            "Fused report must be a dictionary."
        )

    required = {
        "incident_id",
        "incident_type",
        "coordinate_mode",
        "location",
        "summary",
        "severity_confidence",
        "severity_log_likelihood",
    }

    missing = (
        required
        - set(
            report.keys()
        )
    )

    if missing:
        raise ValueError(
            "Fused report missing required fields: "
            f"{sorted(missing)}"
        )

    coordinate_mode = report[
        "coordinate_mode"
    ]

    location = report[
        "location"
    ]

    if not isinstance(
        location,
        dict,
    ):
        raise ValueError(
            "Report location must be a dictionary."
        )

    if coordinate_mode == "gps":

        if (
            "lat" not in location
            or "lon" not in location
        ):
            raise ValueError(
                "GPS fused report requires "
                "location.lat and location.lon."
            )

        coord_a = float(
            location["lat"]
        )

        coord_b = float(
            location["lon"]
        )

    elif coordinate_mode == "local":

        if (
            "x" not in location
            or "y" not in location
        ):
            raise ValueError(
                "Local fused report requires "
                "location.x and location.y."
            )

        if centers is None:
            raise ValueError(
                "Local-coordinate dispatch requires "
                "explicit rescue-center coordinates."
            )

        coord_a = float(
            location["x"]
        )

        coord_b = float(
            location["y"]
        )

    else:
        raise ValueError(
            "coordinate_mode must be either "
            "'gps' or 'local'."
        )

    alert = build_alert_payload(
        incident_type=str(
            report[
                "incident_type"
            ]
        ),
        summary=str(
            report[
                "summary"
            ]
        ),
        coord_a=coord_a,
        coord_b=coord_b,
        conf_yolo=float(
            report[
                "severity_confidence"
            ]
        ),
        log_p_blip2=float(
            report[
                "severity_log_likelihood"
            ]
        ),
        coordinate_mode=(
            coordinate_mode
        ),

        # Use the stable incident identifier instead of generating
        # another unrelated event identifier.
        event_id=str(
            report[
                "incident_id"
            ]
        ),
    )

    routed = route_alert(
        alert,
        centers=centers,
    )

    # Preserve mission-level traceability.
    routed[
        "mission_id"
    ] = report.get(
        "mission_id"
    )

    routed[
        "incident_id"
    ] = report[
        "incident_id"
    ]

    routed[
        "incident_index"
    ] = report.get(
        "incident_index"
    )

    routed[
        "primary_drone"
    ] = report.get(
        "primary_drone"
    )

    routed[
        "num_views"
    ] = report.get(
        "num_views",
        1,
    )

    routed[
        "fusion_method"
    ] = report.get(
        "fusion_method",
        "single_view",
    )

    return routed


def run_ros_node(
    centers_json: Optional[str] = None,
    audio_dir: str = (
        "dispatch_agent/audio_out"
    ),
):
    """Run the automatic dispatch stage as a ROS node."""

    import rospy

    from std_msgs.msg import String

    rospy.init_node(
        "dispatch_agent_node",
        anonymous=False,
    )

    centers = None

    if centers_json:

        centers = load_centers_json(
            centers_json
        )

        logger.info(
            f"Loaded {len(centers)} custom "
            "rescue-center coordinates from "
            f"{centers_json}"
        )

    os.makedirs(
        audio_dir,
        exist_ok=True,
    )

    alert_pub = rospy.Publisher(
        "/dispatch_agent/alert",
        String,
        queue_size=10,
    )

    # Prevent accidental repeat dispatch if the same fused report is
    # delivered more than once.
    dispatched_incidents = set()

    def _fused_report_cb(
        msg,
    ):

        try:

            report = json.loads(
                msg.data
            )

        except json.JSONDecodeError:

            logger.error(
                "Received malformed JSON on "
                "/description_agent/fused_report"
            )

            return

        incident_id = report.get(
            "incident_id"
        )

        if not incident_id:

            logger.error(
                "Received fused report without "
                "incident_id."
            )

            return

        mission_id = report.get(
            "mission_id"
        )

        dispatch_key = (
            str(mission_id),
            str(incident_id),
        )

        if dispatch_key in (
            dispatched_incidents
        ):

            logger.warning(
                f"Ignoring duplicate fused "
                f"report for incident "
                f"{incident_id}"
            )

            return

        try:

            routed = (
                fused_report_to_alert(
                    report,
                    centers=centers,
                )
            )

        except (
            ValueError,
            TypeError,
            KeyError,
        ) as exc:

            logger.error(
                f"Unable to route incident "
                f"{incident_id}: {exc}"
            )

            return

        summary = routed[
            "summary"
        ]

        audio_path = os.path.join(
            audio_dir,
            f"{incident_id}.wav",
        )

        synthesized = (
            synthesize_audio(
                summary,
                audio_path,
            )
        )

        routed[
            "audio_file"
        ] = synthesized

        mqtt_success = (
            publish_mqtt(
                routed
            )
        )

        routed[
            "mqtt_published"
        ] = bool(
            mqtt_success
        )

        # Publish on ROS regardless of whether the external MQTT
        # broker is currently reachable.
        alert_pub.publish(
            String(
                data=json.dumps(
                    routed
                )
            )
        )

        dispatched_incidents.add(
            dispatch_key
        )

        logger.info(
            f"Dispatched incident "
            f"{incident_id} "
            f"to "
            f"{routed['rescue_center']} "
            f"(severity="
            f"{routed['severity']:.4f})"
        )

    rospy.Subscriber(
        "/description_agent/fused_report",
        String,
        _fused_report_cb,
        queue_size=10,
    )

    logger.info(
        "Dispatch agent listening on "
        "/description_agent/fused_report"
    )

    logger.info(
        "Final alerts published on "
        "/dispatch_agent/alert"
    )

    if centers is None:

        logger.info(
            "No custom rescue-center file "
            "was supplied. GPS reports may "
            "use the configured GPS centers; "
            "local reports will be rejected "
            "until local centers are provided."
        )

    rospy.spin()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "ROS Dispatch Agent "
            "(severity + routing + "
            "Piper TTS + MQTT)"
        )
    )

    parser.add_argument(
        "--centers-json",
        type=str,
        default=None,
        help=(
            "JSON file containing rescue-center "
            "coordinates. Required for local "
            "Cartesian/Gazebo dispatch."
        ),
    )

    parser.add_argument(
        "--audio-dir",
        type=str,
        default=(
            "dispatch_agent/audio_out"
        ),
    )

    args = parser.parse_args()

    try:

        run_ros_node(
            centers_json=(
                args.centers_json
            ),
            audio_dir=(
                args.audio_dir
            ),
        )

    except ImportError:

        logger.error(
            "ROS dispatch mode requires rospy."
        )


if __name__ == "__main__":
    main()
