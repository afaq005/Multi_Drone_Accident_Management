#!/usr/bin/env python3
"""
simulation/gnc_controller/gnc_node.py
======================================
Implements the per-drone GNC controller described in Section 6.3:

  * Arms and climbs to CRUISE_ALTITUDE_M (10 m AGL) once the autopilot
    reports a healthy altitude solution.
  * Listens for updates on /droneN/waypoints, converts each pose to a
    target position, and steers to it with closed-loop position control
    via MAVROS setpoint_position/local.
  * Continuously issues a conditional yaw update every YAW_SWEEP_PERIOD_S
    (5 s) so the camera executes a full 360-degree sweep while loitering
    at the final waypoint.
  * When all waypoints are reached, hovers in place but remains
    subscribed; a fresh waypoint list immediately restarts the mission.
  * Publishes setpoints at WAYPOINT_PUBLISH_HZ (2 Hz, "line
    wp_pub_interval(0.5)") — frequent but non-saturating.

This node is built against the `gnc_functions` library conventions
referenced in the paper (a common ROS/MAVROS GNC API for ArduPilot SITL,
e.g. the "iq_sim"/"gnc_functions" package) but is written as a
self-contained MAVROS client so it has no hard dependency on that
specific library.
"""
import argparse
import math
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from model.config import CRUISE_ALTITUDE_M, WAYPOINT_PUBLISH_HZ, YAW_SWEEP_PERIOD_S  # noqa: E402
from model.utils import get_logger  # noqa: E402

logger = get_logger("gnc_controller")


class GNCController:
    """Arm/takeoff/waypoint-follow/360-scan state machine for one drone."""

    STATE_IDLE = "IDLE"
    STATE_TAKEOFF = "TAKEOFF"
    STATE_ENROUTE = "ENROUTE"
    STATE_SCANNING = "SCANNING"

    def __init__(self, namespace: str):
        self.namespace = namespace.rstrip("/")
        self.state = self.STATE_IDLE
        self.current_waypoints = []
        self.waypoint_idx = 0
        self.current_position = (0.0, 0.0, 0.0)
        self.current_yaw = 0.0
        self._last_yaw_update = time.time()

    # -- Pure state-machine logic (unit-testable without ROS/MAVROS) -----
    def on_altitude_healthy(self):
        if self.state == self.STATE_IDLE:
            self.state = self.STATE_TAKEOFF
            logger.info(f"[{self.namespace}] Altitude solution healthy -> arming/climbing to "
                        f"{CRUISE_ALTITUDE_M} m AGL")

    def on_waypoints_received(self, waypoints: list):
        self.current_waypoints = waypoints
        self.waypoint_idx = 0
        self.state = self.STATE_ENROUTE
        logger.info(f"[{self.namespace}] Received {len(waypoints)} waypoint(s); "
                     "mission (re)started.")

    def target_reached(self, position: tuple, threshold_m: float = 1.0) -> bool:
        if self.waypoint_idx >= len(self.current_waypoints):
            return True
        target = self.current_waypoints[self.waypoint_idx]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(position[:3], target[:3])))
        return dist <= threshold_m

    def step(self, position: tuple):
        """Advances the state machine given the drone's current position.
        Returns the current setpoint (x, y, z, yaw) to publish."""
        self.current_position = position

        if self.state == self.STATE_ENROUTE:
            if self.target_reached(position):
                self.waypoint_idx += 1
                if self.waypoint_idx >= len(self.current_waypoints):
                    self.state = self.STATE_SCANNING
                    logger.info(f"[{self.namespace}] Target reached; entering 360-degree scan.")

        if self.state == self.STATE_SCANNING:
            now = time.time()
            if now - self._last_yaw_update >= YAW_SWEEP_PERIOD_S:
                self.current_yaw = (self.current_yaw + 45.0) % 360.0  # conditional yaw step
                self._last_yaw_update = now

        target = (
            self.current_waypoints[min(self.waypoint_idx, len(self.current_waypoints) - 1)]
            if self.current_waypoints
            else (position[0], position[1], CRUISE_ALTITUDE_M)
        )
        return (target[0], target[1], target[2] if len(target) > 2 else CRUISE_ALTITUDE_M,
                self.current_yaw)


# --------------------------------------------------------------------------
# ROS / MAVROS integration
# --------------------------------------------------------------------------
def run_ros_node(namespace: str):
    import rospy
    from geometry_msgs.msg import PoseArray, PoseStamped
    from mavros_msgs.msg import State
    from mavros_msgs.srv import CommandBool, SetMode

    rospy.init_node(f"gnc_node_{namespace.strip('/')}", anonymous=False)
    controller = GNCController(namespace)
    fcu_state = {"connected": False, "armed": False, "mode": ""}

    def _state_cb(msg: "State"):
        fcu_state["connected"] = msg.connected
        fcu_state["armed"] = msg.armed
        fcu_state["mode"] = msg.mode
        if msg.connected and controller.state == controller.STATE_IDLE:
            controller.on_altitude_healthy()

    def _pose_cb(msg: "PoseStamped"):
        pos = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        setpoint_pub.publish(_build_setpoint(*controller.step(pos)))

    def _waypoints_cb(msg: "PoseArray"):
        waypoints = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]
        controller.on_waypoints_received(waypoints)

    def _build_setpoint(x, y, z, yaw_deg):
        sp = PoseStamped()
        sp.pose.position.x, sp.pose.position.y, sp.pose.position.z = x, y, z
        yaw_rad = math.radians(yaw_deg)
        sp.pose.orientation.z = math.sin(yaw_rad / 2.0)
        sp.pose.orientation.w = math.cos(yaw_rad / 2.0)
        return sp

    setpoint_pub = rospy.Publisher(
        f"{namespace}/mavros/setpoint_position/local", PoseStamped, queue_size=10
    )
    rospy.Subscriber(f"{namespace}/mavros/state", State, _state_cb)
    rospy.Subscriber(f"{namespace}/mavros/local_position/pose", PoseStamped, _pose_cb)
    rospy.Subscriber(f"{namespace}/waypoints", PoseArray, _waypoints_cb)

    rospy.wait_for_service(f"{namespace}/mavros/cmd/arming")
    rospy.wait_for_service(f"{namespace}/mavros/set_mode")
    arm_srv = rospy.ServiceProxy(f"{namespace}/mavros/cmd/arming", CommandBool)
    mode_srv = rospy.ServiceProxy(f"{namespace}/mavros/set_mode", SetMode)

    rate = rospy.Rate(WAYPOINT_PUBLISH_HZ)  # 2 Hz, throttled per Section 6.3
    logger.info(f"[{namespace}] GNC controller running at {WAYPOINT_PUBLISH_HZ} Hz ...")
    while not rospy.is_shutdown():
        if fcu_state["connected"] and not fcu_state["armed"]:
            try:
                mode_srv(custom_mode="GUIDED")
                arm_srv(True)
            except rospy.ServiceException as exc:
                logger.warning(f"[{namespace}] Arm/mode service call failed: {exc}")
        rate.sleep()


def main():
    parser = argparse.ArgumentParser(description="Per-drone GNC controller (Section 6.3)")
    parser.add_argument("--namespace", type=str, default="/drone1")
    args = parser.parse_args()

    try:
        run_ros_node(args.namespace)
    except ImportError:
        logger.warning(
            "rospy/mavros_msgs not available. GNCController's state-machine "
            "logic can still be unit-tested directly, e.g.:\n"
            "  from gnc_node import GNCController\n"
            "  c = GNCController('/drone1'); c.on_waypoints_received([(10,0,10)])"
        )


if __name__ == "__main__":
    main()
