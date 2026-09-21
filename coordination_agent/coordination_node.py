#!/usr/bin/env python3
"""
coordination_agent/coordination_node.py
========================================

ROS wrapper around assignment_solver.py.

Subscribes to the planning agent's `/planning_agent/incidents` topic,
resolves the drone-to-incident allocation (Eq. 2, honoring any
operator-directed mapping), and publishes:

    /droneN/waypoints
        geometry_msgs/PoseArray
        Resolved waypoint(s) for the assigned drone.

    /droneN/assigned_incident
        std_msgs/String
        JSON mission/incident metadata for downstream perception,
        description, and dispatch agents.

Waypoint commands are rebroadcast at WAYPOINT_PUBLISH_HZ for
WAYPOINT_REPEAT_COUNT cycles to improve command-delivery reliability.
"""

import json
import os
import sys
import time
import uuid

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from model.config import (  # noqa: E402
    DRONE_NAMESPACES,
    WAYPOINT_PUBLISH_HZ,
    WAYPOINT_REPEAT_COUNT,
)
from model.utils import get_logger  # noqa: E402
from coordination_agent.assignment_solver import (  # noqa: E402
    build_waypoint_payload,
    solve_assignment,
)

logger = get_logger(
    "coordination_node"
)


def _prepare_incidents_payload(
    incidents_payload: dict,
) -> dict:
    """
    Ensure every mission and incident has a stable identifier.

    If the planning agent already supplied mission_id / incident_id,
    those identifiers are preserved. Otherwise they are created here.

    Existing coordinate fields are left unchanged so that the current
    assignment solver remains compatible.
    """

    prepared = dict(
        incidents_payload
    )

    mission_id = prepared.get(
        "mission_id"
    )

    if not mission_id:
        mission_id = (
            uuid.uuid4()
            .hex[:8]
            .upper()
        )

    prepared[
        "mission_id"
    ] = mission_id

    default_coordinate_mode = (
        prepared.get(
            "coordinate_mode",
            "local",
        )
    )

    prepared_incidents = []

    for index, incident in enumerate(
        prepared.get(
            "incidents",
            []
        )
    ):
        inc = dict(
            incident
        )

        if not inc.get(
            "incident_id"
        ):
            inc[
                "incident_id"
            ] = (
                f"{mission_id}-"
                f"I{index + 1:02d}"
            )

        inc[
            "coordinate_mode"
        ] = inc.get(
            "coordinate_mode",
            default_coordinate_mode,
        )

        prepared_incidents.append(
            inc
        )

    prepared[
        "incidents"
    ] = prepared_incidents

    return prepared


def _build_assignment_metadata(
    prepared_payload: dict,
    assignments: dict,
) -> dict:
    """
    Build one structured assignment message per assigned drone.

    Returns:
        {
            "/drone1": {...},
            "/drone2": {...}
        }
    """

    incidents = prepared_payload[
        "incidents"
    ]

    metadata = {}

    for (
        drone,
        incident_index,
    ) in assignments.items():

        incident = incidents[
            incident_index
        ]

        coordinate_mode = (
            incident.get(
                "coordinate_mode",
                "local",
            )
        )

        coord_a = float(
            incident["lat"]
        )

        coord_b = float(
            incident["lon"]
        )

        altitude = float(
            incident.get(
                "alt",
                0.0,
            )
        )

        if coordinate_mode == "gps":

            target = {
                "lat": coord_a,
                "lon": coord_b,
                "alt": altitude,
            }

        else:

            # The current planning/assignment interface historically
            # stores the first two local Cartesian coordinates in the
            # fields named `lat` and `lon`. Downstream metadata exposes
            # them explicitly as x/y when coordinate_mode="local".
            target = {
                "x": coord_a,
                "y": coord_b,
                "z": altitude,
            }

        assignment = {
            "mission_id": (
                prepared_payload[
                    "mission_id"
                ]
            ),

            "incident_id": (
                incident[
                    "incident_id"
                ]
            ),

            "incident_index": (
                incident_index
            ),

            "drone_id": drone,

            "coordinate_mode": (
                coordinate_mode
            ),

            "target": target,
        }

        if (
            "intent"
            in prepared_payload
        ):
            assignment[
                "intent"
            ] = prepared_payload[
                "intent"
            ]

        if "type" in incident:
            assignment[
                "incident_type"
            ] = incident[
                "type"
            ]

        metadata[
            drone
        ] = assignment

    return metadata


def resolve_and_broadcast(
    incidents_payload: dict,
    drone_positions: dict,
    publish_fn,
    assignment_publish_fn=None,
):
    """
    Resolve assignments and broadcast waypoint commands.

    Parameters
    ----------
    incidents_payload:
        Planning-agent JSON payload.

    drone_positions:
        Current local UAV positions keyed by namespace.

    publish_fn:
        Callable:
            publish_fn(namespace, waypoints)

    assignment_publish_fn:
        Optional callable:
            assignment_publish_fn(namespace, metadata)

        Used to publish structured incident identifiers and assignment
        metadata for downstream agents.

    Returns
    -------
    dict
        The existing waypoint payload format is retained for backward
        compatibility.
    """

    prepared_payload = (
        _prepare_incidents_payload(
            incidents_payload
        )
    )

    incidents = prepared_payload.get(
        "incidents",
        [],
    )

    if not incidents:
        logger.warning(
            "No incidents to resolve; "
            "nothing published."
        )
        return {}

    assignments = solve_assignment(
        drone_positions,
        incidents,
    )

    if not assignments:
        logger.warning(
            "No drone-to-incident "
            "assignments were produced."
        )
        return {}

    payload = build_waypoint_payload(
        assignments,
        incidents,
    )

    assignment_metadata = (
        _build_assignment_metadata(
            prepared_payload,
            assignments,
        )
    )

    # Publish structured assignment metadata once. In ROS mode the
    # publisher is latched, so downstream nodes that start later can
    # still retrieve the current assignment.
    if assignment_publish_fn is not None:

        for (
            namespace,
            metadata,
        ) in assignment_metadata.items():

            assignment_publish_fn(
                namespace,
                metadata,
            )

    # Preserve the existing waypoint rebroadcast behavior.
    for _ in range(
        WAYPOINT_REPEAT_COUNT
    ):

        for (
            topic,
            waypoints,
        ) in payload.items():

            namespace = (
                topic.replace(
                    "/waypoints",
                    "",
                )
            )

            publish_fn(
                namespace,
                waypoints,
            )

        time.sleep(
            1.0
            / WAYPOINT_PUBLISH_HZ
        )

    logger.info(
        "Mission %s assignments: %s",
        prepared_payload[
            "mission_id"
        ],
        json.dumps(
            assignment_metadata
        ),
    )

    logger.info(
        "Broadcast waypoints: %s",
        json.dumps(
            payload
        ),
    )

    return payload


def run_ros_node():

    import rospy

    from geometry_msgs.msg import (
        Pose,
        PoseArray,
        PoseStamped,
    )
    from std_msgs.msg import String

    rospy.init_node(
        "coordination_agent_node",
        anonymous=False,
    )

    # Existing waypoint publishers.
    waypoint_publishers = {
        ns: rospy.Publisher(
            f"{ns}/waypoints",
            PoseArray,
            queue_size=10,
        )
        for ns in DRONE_NAMESPACES
    }

    # New assignment-metadata publishers.
    #
    # latch=True ensures that a perception node started after the
    # assignment was issued can still obtain its current incident ID.
    assignment_publishers = {
        ns: rospy.Publisher(
            f"{ns}/assigned_incident",
            String,
            queue_size=10,
            latch=True,
        )
        for ns in DRONE_NAMESPACES
    }

    # Current drone positions are updated from MAVROS local pose topics.
    drone_positions = {
        ns: (
            0.0,
            0.0,
            0.0,
        )
        for ns in DRONE_NAMESPACES
    }

    def _make_pose_callback(
        ns,
    ):
        def _cb(msg):

            drone_positions[
                ns
            ] = (
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            )

        return _cb

    for ns in DRONE_NAMESPACES:

        rospy.Subscriber(
            f"{ns}/mavros/"
            f"local_position/pose",
            PoseStamped,
            _make_pose_callback(
                ns
            ),
            queue_size=10,
        )

    def publish_fn(
        namespace: str,
        waypoints: list,
    ):

        pub = (
            waypoint_publishers.get(
                namespace
            )
        )

        if pub is None:
            logger.error(
                f"No waypoint publisher "
                f"configured for "
                f"{namespace}"
            )
            return

        pose_array = PoseArray()

        pose_array.header.stamp = (
            rospy.Time.now()
        )

        for wp in waypoints:

            pose = Pose()

            (
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ) = (
                float(wp[0]),
                float(wp[1]),
                float(wp[2]),
            )

            pose_array.poses.append(
                pose
            )

        pub.publish(
            pose_array
        )

    def assignment_publish_fn(
        namespace: str,
        metadata: dict,
    ):

        pub = (
            assignment_publishers.get(
                namespace
            )
        )

        if pub is None:
            logger.error(
                f"No assignment publisher "
                f"configured for "
                f"{namespace}"
            )
            return

        pub.publish(
            String(
                data=json.dumps(
                    metadata
                )
            )
        )

        logger.info(
            f"Published assignment for "
            f"{namespace}: "
            f"incident_id="
            f"{metadata['incident_id']}"
        )

    def _incidents_callback(
        msg: "String",
    ):

        try:
            incidents_payload = (
                json.loads(
                    msg.data
                )
            )

        except json.JSONDecodeError:

            logger.error(
                f"Malformed incidents "
                f"payload: {msg.data!r}"
            )

            return

        resolve_and_broadcast(
            incidents_payload,
            drone_positions,
            publish_fn,
            assignment_publish_fn=(
                assignment_publish_fn
            ),
        )

    rospy.Subscriber(
        "/planning_agent/incidents",
        String,
        _incidents_callback,
        queue_size=10,
    )

    logger.info(
        "Coordination agent listening on "
        "/planning_agent/incidents"
    )

    logger.info(
        "Publishing per-drone assignment "
        "metadata on "
        "/droneN/assigned_incident"
    )

    rospy.spin()


if __name__ == "__main__":

    try:
        run_ros_node()

    except ImportError:

        logger.warning(
            "rospy not available. "
            "Use assignment_solver.py "
            "directly for standalone testing, e.g.:\n"
            "  python "
            "coordination_agent/"
            "assignment_solver.py "
            "--incidents "
            '\'[{"lat":-165,'
            '"lon":-1.43,'
            '"alt":10}]\'' 
        )
