#!/usr/bin/env python3
"""
dispatch_agent/rescue_center_router.py
=======================================

Implements A_disp's routing logic from Section 6.6:

  Eq. 24 — severity score:
      severity = sigmoid(alpha * conf_YOLO + beta * log p_BLIP2(y_hat | I))

  Eq. 25 — nearest-rescue-center selection:
      r* = argmin_r d(g_k, l_r)

      For local Cartesian coordinates (e.g. Gazebo), Euclidean distance
      is used. For outdoor GPS coordinates, Haversine distance is used.

Also builds the minimal JSON alert payload shown in Section 6.6.
"""

import argparse
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from model.config import (  # noqa: E402
    RESCUE_CENTERS,
    SEVERITY_ALPHA,
    SEVERITY_BETA,
)
from model.utils import (  # noqa: E402
    get_logger,
    haversine_distance_m,
    sigmoid,
)

logger = get_logger(
    "dispatch_agent.router"
)


def compute_severity(
    conf_yolo: float,
    log_p_blip2: float,
    alpha: float = SEVERITY_ALPHA,
    beta: float = SEVERITY_BETA,
) -> float:
    """Eq. 24."""
    return sigmoid(
        alpha * conf_yolo
        + beta * log_p_blip2
    )


def nearest_rescue_center(
    coord_a: float,
    coord_b: float,
    centers: Optional[
        List[
            Tuple[
                str,
                float,
                float,
            ]
        ]
    ] = None,
    coordinate_mode: str = "gps",
) -> Tuple[str, float]:
    """
    Eq. 25: select the nearest rescue center.

    coordinate_mode="gps":
        coord_a = latitude
        coord_b = longitude
        Haversine distance is used.

    coordinate_mode="local":
        coord_a = local x-coordinate
        coord_b = local y-coordinate
        Euclidean distance is used.

    Returns:
        (center_name, distance)
    """

    if coordinate_mode not in {
        "gps",
        "local",
    }:
        raise ValueError(
            "coordinate_mode must be either "
            "'gps' or 'local'"
        )

    # GPS mode may use the configured default geographic centers.
    # Local mode must explicitly receive centers defined in the same
    # Cartesian coordinate frame as the incident.
    if centers is None:

        if coordinate_mode == "gps":
            centers = RESCUE_CENTERS

        else:
            raise ValueError(
                "Local coordinate mode requires explicit "
                "rescue-center coordinates defined in the "
                "same Cartesian frame."
            )

    if not centers:
        raise ValueError(
            "At least one rescue center is required."
        )

    best_name = None
    best_dist = float("inf")

    for (
        name,
        center_a,
        center_b,
    ) in centers:

        if coordinate_mode == "gps":

            dist = haversine_distance_m(
                coord_a,
                coord_b,
                center_a,
                center_b,
            )

        else:

            dist = math.hypot(
                coord_a - center_a,
                coord_b - center_b,
            )

        if dist < best_dist:
            best_name = name
            best_dist = dist

    return (
        best_name,
        best_dist,
    )


def build_alert_payload(
    incident_type: str,
    summary: str,
    coord_a: float,
    coord_b: float,
    conf_yolo: float,
    log_p_blip2: float,
    coordinate_mode: str = "gps",
    image_paths: Optional[List[str]] = None,
    event_id: Optional[str] = None,
) -> dict:
    """Build the minimal JSON alert payload."""

    if coordinate_mode not in {
        "gps",
        "local",
    }:
        raise ValueError(
            "coordinate_mode must be either "
            "'gps' or 'local'"
        )

    severity = compute_severity(
        conf_yolo,
        log_p_blip2,
    )

    payload = {
        "event_id": (
            event_id
            or uuid.uuid4().hex[:8].upper()
        ),
        "type": incident_type,
        "summary": summary,
        "coordinate_mode": (
            coordinate_mode
        ),
        "severity": round(
            severity,
            4,
        ),
        "timestamp": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "images": (
            image_paths
            or []
        ),
    }

    if coordinate_mode == "gps":

        payload["lat"] = (
            coord_a
        )

        payload["lon"] = (
            coord_b
        )

    else:

        payload["x"] = (
            coord_a
        )

        payload["y"] = (
            coord_b
        )

    return payload


def route_alert(
    payload: dict,
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
    """Attach the chosen rescue center (Eq. 25) to an alert payload."""

    coordinate_mode = (
        payload.get(
            "coordinate_mode",
            "gps",
        )
    )

    if coordinate_mode == "gps":

        coord_a = payload[
            "lat"
        ]

        coord_b = payload[
            "lon"
        ]

    elif coordinate_mode == "local":

        coord_a = payload[
            "x"
        ]

        coord_b = payload[
            "y"
        ]

    else:

        raise ValueError(
            "Unknown coordinate_mode "
            "in payload"
        )

    center_name, distance = (
        nearest_rescue_center(
            coord_a,
            coord_b,
            centers=centers,
            coordinate_mode=(
                coordinate_mode
            ),
        )
    )

    payload = dict(
        payload
    )

    payload[
        "rescue_center"
    ] = center_name

    if coordinate_mode == "gps":

        payload[
            "distance_to_center_m"
        ] = round(
            distance,
            1,
        )

        logger.info(
            f"Routed alert "
            f"{payload['event_id']} "
            f"(severity="
            f"{payload['severity']}) "
            f"to {center_name} "
            f"({distance:.0f} m away)"
        )

    else:

        payload[
            "distance_to_center_local"
        ] = round(
            distance,
            3,
        )

        logger.info(
            f"Routed alert "
            f"{payload['event_id']} "
            f"(severity="
            f"{payload['severity']}) "
            f"to {center_name} "
            f"(local distance="
            f"{distance:.2f})"
        )

    return payload


def load_centers_json(
    path: str,
) -> List[
    Tuple[
        str,
        float,
        float,
    ]
]:
    """
    Load rescue-center coordinates from JSON.

    Expected format:

        [
            ["Rescue Center A", -150.0, 20.0],
            ["Rescue Center B", 80.0, 15.0]
        ]
    """

    if not os.path.isfile(
        path
    ):
        raise FileNotFoundError(
            f"Rescue-center file "
            f"not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        raw = json.load(
            f
        )

    if not isinstance(
        raw,
        list,
    ):
        raise ValueError(
            "Rescue-center JSON "
            "must contain a list."
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
                "Each rescue center must "
                "have the form "
                "[name, coordinate_a, "
                "coordinate_b]."
            )

        (
            name,
            coord_a,
            coord_b,
        ) = item

        centers.append(
            (
                str(name),
                float(
                    coord_a
                ),
                float(
                    coord_b
                ),
            )
        )

    if not centers:
        raise ValueError(
            "At least one rescue "
            "center is required."
        )

    return centers


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Dispatch Agent routing "
            "(Eq. 24-25)"
        )
    )

    parser.add_argument(
        "--centers-json",
        type=str,
        default=None,
        help=(
            "Optional JSON file containing "
            "rescue-center coordinates. "
            "Required when "
            "--coordinate-mode local."
        ),
    )

    parser.add_argument(
        "--type",
        type=str,
        default="accident",
    )

    parser.add_argument(
        "--summary",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--coordinate-mode",
        choices=[
            "gps",
            "local",
        ],
        default="gps",
        help=(
            "Use 'gps' for lat/lon "
            "or 'local' for Cartesian "
            "x/y coordinates."
        ),
    )

    parser.add_argument(
        "--lat",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--lon",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--x",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--y",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.85,
        help=(
            "YOLO detection confidence"
        ),
    )

    parser.add_argument(
        "--log-p-blip2",
        type=float,
        default=-0.5,
        help=(
            "BLIP-2 caption "
            "log-likelihood"
        ),
    )

    args = parser.parse_args()

    if args.coordinate_mode == "gps":

        if (
            args.lat is None
            or args.lon is None
        ):
            parser.error(
                "--lat and --lon are "
                "required when "
                "--coordinate-mode gps"
            )

        coord_a = args.lat
        coord_b = args.lon

        centers = (
            load_centers_json(
                args.centers_json
            )
            if args.centers_json
            else None
        )

    else:

        if (
            args.x is None
            or args.y is None
        ):
            parser.error(
                "--x and --y are required "
                "when --coordinate-mode local"
            )

        if not args.centers_json:
            parser.error(
                "--centers-json is required "
                "when --coordinate-mode local"
            )

        coord_a = args.x
        coord_b = args.y

        centers = load_centers_json(
            args.centers_json
        )

    payload = build_alert_payload(
        incident_type=args.type,
        summary=args.summary,
        coord_a=coord_a,
        coord_b=coord_b,
        conf_yolo=args.conf,
        log_p_blip2=(
            args.log_p_blip2
        ),
        coordinate_mode=(
            args.coordinate_mode
        ),
    )

    routed = route_alert(
        payload,
        centers=centers,
    )

    print(
        json.dumps(
            routed,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
