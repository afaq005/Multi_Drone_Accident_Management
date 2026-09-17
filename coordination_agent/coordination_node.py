#!/usr/bin/env python3
"""
coordination_agent/coordination_node.py
========================================
ROS wrapper around assignment_solver.py. Subscribes to the planning
agent's `/planning_agent/incidents` topic, resolves the drone-to-incident
allocation (Eq. 2, honoring any operator-directed mapping), and publishes
a geometry_msgs/PoseArray on `/droneN/waypoints` for each namespace,
rebroadcast at WAYPOINT_PUBLISH_HZ for WAYPOINT_REPEAT_COUNT cycles to
overcome packet loss, exactly as described in Section 6.2.
"""
import json
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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

logger = get_logger("coordination_node")


def resolve_and_broadcast(incidents_payload: dict, drone_positions: dict, publish_fn):
    """Core logic, decoupled from ROS so it is unit-testable.

    `publish_fn(namespace: str, waypoints: list)` is called once per
    drone/namespace with its resolved waypoint list.
    """
    incidents = incidents_payload.get("incidents", [])
    if not incidents:
        logger.warning("No incidents to resolve; nothing published.")
        return {}

    assignments = solve_assignment(drone_positions, incidents)
    payload = build_waypoint_payload(assignments, incidents)

    for _ in range(WAYPOINT_REPEAT_COUNT):
        for topic, waypoints in payload.items():
            namespace = topic.replace("/waypoints", "")
            publish_fn(namespace, waypoints)
        time.sleep(1.0 / WAYPOINT_PUBLISH_HZ)

    logger.info(f"Broadcast waypoints: {json.dumps(payload)}")
    return payload


def run_ros_node():
    import rospy
    from geometry_msgs.msg import Pose, PoseArray
    from std_msgs.msg import String

    rospy.init_node("coordination_agent_node", anonymous=False)

    publishers = {
        ns: rospy.Publisher(f"{ns}/waypoints", PoseArray, queue_size=10)
        for ns in DRONE_NAMESPACES
    }

    # In a full deployment, drone_positions would be updated from each
    # drone's /droneN/mavros/local_position/pose topic. For clarity this
    # node keeps a simple in-memory cache seeded at the origin and updated
    # via ROS subscribers wired up below.
    drone_positions = {ns: (0.0, 0.0, 0.0) for ns in DRONE_NAMESPACES}

    def _make_pose_callback(ns):
        def _cb(msg):
            drone_positions[ns] = (
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            )
        return _cb

    from geometry_msgs.msg import PoseStamped
    for ns in DRONE_NAMESPACES:
        rospy.Subscriber(
            f"{ns}/mavros/local_position/pose", PoseStamped, _make_pose_callback(ns)
        )

    def publish_fn(namespace: str, waypoints: list):
        pub = publishers.get(namespace)
        if pub is None:
            logger.error(f"No publisher configured for {namespace}")
            return
        pose_array = PoseArray()
        for wp in waypoints:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = wp[0], wp[1], wp[2]
            pose_array.poses.append(pose)
        pub.publish(pose_array)

    def _incidents_callback(msg: "String"):
        try:
            incidents_payload = json.loads(msg.data)
        except json.JSONDecodeError:
            logger.error(f"Malformed incidents payload: {msg.data!r}")
            return
        resolve_and_broadcast(incidents_payload, drone_positions, publish_fn)

    rospy.Subscriber("/planning_agent/incidents", String, _incidents_callback)
    logger.info("Coordination agent listening on /planning_agent/incidents ...")
    rospy.spin()


if __name__ == "__main__":
    try:
        run_ros_node()
    except ImportError:
        logger = get_logger("coordination_node")
        logger.warning(
            "rospy not available. Use assignment_solver.py directly for "
            "standalone testing, e.g.:\n"
            "  python assignment_solver.py --incidents "
            '\'[{"lat":-165,"lon":-1.43,"alt":10},{"lat":102.2,"lon":4.1,"alt":10}]\''
        )
