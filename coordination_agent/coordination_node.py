#!/usr/bin/env python3
"""
coordination_agent/coordination_node.py
========================================

ROS wrapper around assignment_solver.py.

Subscribes:
    /planning_agent/incidents

Publishes:
    /droneN/waypoints
    /droneN/assigned_incident

The demonstrated ROS/MAVROS closed-loop pipeline uses local Cartesian
coordinates. GPS-mode planning payloads are therefore rejected by this
node rather than being mixed with MAVROS local-position coordinates.
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
    """Add stable mission and incident identifiers if absent."""

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

    prepared["mission_id"] = mission_id

    default_mode = prepared.get(
        "coordinate_mode",
        "local",
    )

    prepared_incidents = []

    for index, incident in enumerate(
        prepared.get(
            "incidents",
            [],
        )
    ):
        inc = dict(
            incident
        )

        if not inc.get(
            "incident_id"
        ):
            inc["incident_id"] = (
                f"{mission_id}-I{index + 1:02d}"
            )

        inc["coordinate_mode"] = inc.get(
            "coordinate_mode",
            default_mode,
        )

        prepared_incidents.append(
            inc
        )

    prepared["incidents"] = (
        prepared_incidents
    )

    return prepared


def _validate_local_pipeline(
    prepared_payload: dict,
) -> None:
    """
    Ensure the ROS coordination pipeline uses only local coordinates.

    Drone positions in this node come from:
        /droneN/mavros/local_position/pose

    so mixing those positions with geographic latitude/longitude would
    produce invalid assignment costs.
    """

    payload_mode = prepared_payload.get(
        "coordinate_mode",
        "local",
    )

    if payload_mode != "local":
        raise ValueError(
            "The ROS coordination node currently supports only "
            "coordinate_mode='local' because drone positions are "
            "obtained from MAVROS local_position/pose."
        )

    for incident in prepared_payload.get(
        "incidents",
        [],
    ):
        if incident.get(
            "coordinate_mode",
            payload_mode,
        ) != "local":
            raise ValueError(
                "GPS incident coordinates cannot be mixed with "
                "MAVROS local-position coordinates."
            )


def _build_assignment_metadata(
    prepared_payload: dict,
    assignments: dict,
) -> dict:
    """Build one structured assignment message per assigned drone."""

    incidents = prepared_payload[
        "incidents"
    ]

    metadata = {}

    for drone, incident_index in assignments.items():
        incident = incidents[
            incident_index
        ]

        target = {
            "x": float(
                incident["lat"]
            ),
            "y": float(
                incident["lon"]
            ),
            "z": float(
                incident.get(
                    "alt",
                    0.0,
                )
            ),
        }

        assignment = {
            "mission_id": prepared_payload[
                "mission_id"
            ],
            "incident_id": incident[
                "incident_id"
            ],
            "incident_index": incident_index,
            "drone_id": drone,
            "coordinate_mode": "local",
            "target": target,
        }

        if "intent" in prepared_payload:
            assignment["intent"] = (
                prepared_payload["intent"]
            )

        if "type" in incident:
            assignment["incident_type"] = (
                incident["type"]
            )

        metadata[drone] = assignment

    return metadata


def resolve_and_broadcast(
    incidents_payload: dict,
    drone_positions: dict,
    publish_fn,
    assignment_publish_fn=None,
):
    """Resolve assignments and broadcast local Cartesian waypoints."""

    prepared_payload = (
        _prepare_incidents_payload(
            incidents_payload
        )
    )

    _validate_local_pipeline(
        prepared_payload
    )

    incidents = prepared_payload.get(
        "incidents",
        [],
    )

    if not incidents:
        logger.warning(
            "No incidents to resolve; nothing published."
        )
        return {}

    assignments = solve_assignment(
        drone_positions,
        incidents,
        coordinate_mode="local",
    )

    if not assignments:
        logger.warning(
            "No assignments were produced."
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

    if assignment_publish_fn is not None:
        for namespace, metadata in (
            assignment_metadata.items()
        ):
            assignment_publish_fn(
                namespace,
                metadata,
            )

    for _ in range(
        WAYPOINT_REPEAT_COUNT
    ):
        for topic, waypoints in payload.items():
            namespace = topic.replace(
                "/waypoints",
                "",
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
        prepared_payload["mission_id"],
        json.dumps(
            assignment_metadata
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

    waypoint_publishers = {
        ns: rospy.Publisher(
            f"{ns}/waypoints",
            PoseArray,
            queue_size=10,
        )
        for ns in DRONE_NAMESPACES
    }

    assignment_publishers = {
        ns: rospy.Publisher(
            f"{ns}/assigned_incident",
            String,
            queue_size=10,
            latch=True,
        )
        for ns in DRONE_NAMESPACES
    }

    drone_positions = {
        ns: (0.0, 0.0, 0.0)
        for ns in DRONE_NAMESPACES
    }

    def _make_pose_callback(ns):
        def _cb(msg):
            drone_positions[ns] = (
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            )

        return _cb

    for ns in DRONE_NAMESPACES:
        rospy.Subscriber(
            f"{ns}/mavros/local_position/pose",
            PoseStamped,
            _make_pose_callback(ns),
            queue_size=10,
        )

    def publish_fn(
        namespace: str,
        waypoints: list,
    ):
        pub = waypoint_publishers.get(
            namespace
        )

        if pub is None:
            logger.error(
                f"No waypoint publisher configured for {namespace}"
            )
            return

        pose_array = PoseArray()
        pose_array.header.stamp = (
            rospy.Time.now()
        )
        pose_array.header.frame_id = (
            "map"
        )

        for wp in waypoints:
            pose = Pose()

            pose.position.x = float(
                wp[0]
            )
            pose.position.y = float(
                wp[1]
            )
            pose.position.z = float(
                wp[2]
            )

            pose.orientation.w = 1.0

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
        pub = assignment_publishers.get(
            namespace
        )

        if pub is None:
            logger.error(
                f"No assignment publisher configured for {namespace}"
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
            f"Published assignment for {namespace}: "
            f"incident_id={metadata['incident_id']}"
        )

    def _incidents_callback(msg):
        try:
            incidents_payload = json.loads(
                msg.data
            )
        except json.JSONDecodeError:
            logger.error(
                f"Malformed incidents payload: {msg.data!r}"
            )
            return

        try:
            resolve_and_broadcast(
                incidents_payload,
                drone_positions,
                publish_fn,
                assignment_publish_fn,
            )
        except ValueError as exc:
            logger.error(
                f"Mission rejected: {exc}"
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

    rospy.spin()


if __name__ == "__main__":
    try:
        run_ros_node()
    except ImportError:
        logger.warning(
            "rospy not available. "
            "Use assignment_solver.py for standalone testing."
        )
