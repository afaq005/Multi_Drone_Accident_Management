#!/usr/bin/env python3
"""
coordination_agent/mavros_waypoint_adapter.py
==============================================

Vehicle-control adapter between the coordination agent and MAVROS /
ArduPilot.

Input:

    /droneN/waypoints
        geometry_msgs/PoseArray

Output:

    /droneN/mavros/setpoint_position/local
        geometry_msgs/PoseStamped

The coordination agent remains vehicle-independent. This adapter converts
its generic local Cartesian waypoint representation into MAVROS local
position setpoints.

The adapter also monitors:

    /droneN/mavros/local_position/pose

and advances through a waypoint list when the UAV reaches the current
target within the configured tolerance.

Optional command-line flags can request GUIDED mode and vehicle arming.
These actions are disabled by default because arming and flight-mode
changes are deployment-specific operations.

The complete simulation pipeline uses local Cartesian coordinates. Native
GPS waypoint control is not implemented by this adapter.

Example:

    python coordination_agent/mavros_waypoint_adapter.py \
        --namespace /drone1

Optional:

    python coordination_agent/mavros_waypoint_adapter.py \
        --namespace /drone1 \
        --set-guided \
        --auto-arm \
        --rate 5 \
        --tolerance 1.5
"""

import argparse
import math
import os
import sys
import threading
from typing import List, Optional, Tuple

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from model.utils import get_logger  # noqa: E402

logger = get_logger(
    "coordination_agent.mavros_adapter"
)


def distance_3d(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    """Return 3-D Euclidean distance between two local positions."""

    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


class WaypointTracker:
    """
    Dependency-light waypoint state machine.

    This class is separated from ROS so its waypoint progression logic can
    be unit-tested independently.
    """

    def __init__(
        self,
        tolerance_m: float = 1.5,
    ):
        if tolerance_m <= 0:
            raise ValueError(
                "Waypoint tolerance must be positive."
            )

        self.tolerance_m = float(
            tolerance_m
        )

        self.waypoints: List[
            Tuple[float, float, float]
        ] = []

        self.current_index = 0

    def set_waypoints(
        self,
        waypoints: List[
            Tuple[float, float, float]
        ],
    ) -> None:
        """Replace the current mission waypoint list."""

        self.waypoints = [
            (
                float(wp[0]),
                float(wp[1]),
                float(wp[2]),
            )
            for wp in waypoints
        ]

        self.current_index = 0

    def current_waypoint(
        self,
    ) -> Optional[
        Tuple[float, float, float]
    ]:
        """Return the current target or None when no target remains."""

        if (
            not self.waypoints
            or self.current_index
            >= len(self.waypoints)
        ):
            return None

        return self.waypoints[
            self.current_index
        ]

    def update_position(
        self,
        position: Tuple[
            float, float, float
        ],
    ) -> bool:
        """
        Update waypoint progression.

        Returns True if the current waypoint was reached during this call.
        """

        target = self.current_waypoint()

        if target is None:
            return False

        dist = distance_3d(
            position,
            target,
        )

        if dist <= self.tolerance_m:

            self.current_index += 1

            return True

        return False

    def complete(
        self,
    ) -> bool:
        """Return True when all current waypoints have been reached."""

        return (
            bool(self.waypoints)
            and self.current_index
            >= len(self.waypoints)
        )


def run_ros_node(
    namespace: str,
    publish_rate_hz: float = 5.0,
    tolerance_m: float = 1.5,
    set_guided: bool = False,
    auto_arm: bool = False,
):
    """
    Run the MAVROS local-position waypoint adapter.
    """

    import rospy

    from geometry_msgs.msg import (
        PoseArray,
        PoseStamped,
    )

    if publish_rate_hz <= 0:
        raise ValueError(
            "publish_rate_hz must be positive."
        )

    node_name = (
        "mavros_waypoint_adapter_"
        + namespace.strip("/")
    )

    rospy.init_node(
        node_name,
        anonymous=False,
    )

    tracker = WaypointTracker(
        tolerance_m=tolerance_m
    )

    lock = threading.Lock()

    state = {
        "position": (
            0.0,
            0.0,
            0.0,
        ),
        "have_pose": False,
    }

    setpoint_topic = (
        f"{namespace}/"
        "mavros/setpoint_position/local"
    )

    waypoint_topic = (
        f"{namespace}/waypoints"
    )

    pose_topic = (
        f"{namespace}/"
        "mavros/local_position/pose"
    )

    setpoint_pub = rospy.Publisher(
        setpoint_topic,
        PoseStamped,
        queue_size=10,
    )

    def _waypoint_cb(
        msg,
    ):
        waypoints = []

        for pose in msg.poses:

            waypoints.append(
                (
                    float(
                        pose.position.x
                    ),
                    float(
                        pose.position.y
                    ),
                    float(
                        pose.position.z
                    ),
                )
            )

        if not waypoints:

            logger.warning(
                f"[{namespace}] Received "
                "empty waypoint list."
            )

            return

        with lock:

            tracker.set_waypoints(
                waypoints
            )

        logger.info(
            f"[{namespace}] Received "
            f"{len(waypoints)} waypoint(s): "
            f"{waypoints}"
        )

    def _pose_cb(
        msg,
    ):
        position = (
            float(
                msg.pose.position.x
            ),
            float(
                msg.pose.position.y
            ),
            float(
                msg.pose.position.z
            ),
        )

        with lock:

            state[
                "position"
            ] = position

            state[
                "have_pose"
            ] = True

            previous_target = (
                tracker.current_waypoint()
            )

            reached = (
                tracker.update_position(
                    position
                )
            )

            next_target = (
                tracker.current_waypoint()
            )

        if reached:

            logger.info(
                f"[{namespace}] Reached "
                f"waypoint "
                f"{previous_target}"
            )

            if next_target is not None:

                logger.info(
                    f"[{namespace}] Advancing "
                    f"to waypoint "
                    f"{next_target}"
                )

            else:

                logger.info(
                    f"[{namespace}] Waypoint "
                    "mission complete."
                )

    rospy.Subscriber(
        waypoint_topic,
        PoseArray,
        _waypoint_cb,
        queue_size=10,
    )

    rospy.Subscriber(
        pose_topic,
        PoseStamped,
        _pose_cb,
        queue_size=10,
    )

    # --------------------------------------------------------------
    # Optional MAVROS vehicle services
    # --------------------------------------------------------------

    if (
        set_guided
        or auto_arm
    ):

        try:

            from mavros_msgs.srv import (
                CommandBool,
                SetMode,
            )

        except ImportError:

            logger.error(
                "mavros_msgs is required "
                "for --set-guided or "
                "--auto-arm."
            )

            raise

        if set_guided:

            mode_service = (
                f"{namespace}/"
                "mavros/set_mode"
            )

            logger.info(
                f"[{namespace}] Waiting "
                f"for {mode_service}"
            )

            rospy.wait_for_service(
                mode_service
            )

            try:

                set_mode = (
                    rospy.ServiceProxy(
                        mode_service,
                        SetMode,
                    )
                )

                response = set_mode(
                    base_mode=0,
                    custom_mode="GUIDED",
                )

                if response.mode_sent:

                    logger.info(
                        f"[{namespace}] "
                        "GUIDED mode request "
                        "accepted."
                    )

                else:

                    logger.warning(
                        f"[{namespace}] "
                        "GUIDED mode request "
                        "was not accepted."
                    )

            except rospy.ServiceException as exc:

                logger.error(
                    f"[{namespace}] "
                    f"set_mode failed: "
                    f"{exc}"
                )

        if auto_arm:

            arm_service = (
                f"{namespace}/"
                "mavros/cmd/arming"
            )

            logger.info(
                f"[{namespace}] Waiting "
                f"for {arm_service}"
            )

            rospy.wait_for_service(
                arm_service
            )

            try:

                arm = (
                    rospy.ServiceProxy(
                        arm_service,
                        CommandBool,
                    )
                )

                response = arm(
                    True
                )

                if response.success:

                    logger.info(
                        f"[{namespace}] "
                        "Arming request "
                        "accepted."
                    )

                else:

                    logger.warning(
                        f"[{namespace}] "
                        "Arming request "
                        "was not accepted."
                    )

            except rospy.ServiceException as exc:

                logger.error(
                    f"[{namespace}] "
                    f"arming failed: "
                    f"{exc}"
                )

    # --------------------------------------------------------------
    # Continuous local setpoint publisher
    # --------------------------------------------------------------

    def _publish_timer(
        _event,
    ):

        with lock:

            target = (
                tracker.current_waypoint()
            )

        if target is None:
            return

        msg = PoseStamped()

        msg.header.stamp = (
            rospy.Time.now()
        )

        # The exact local frame is supplied by the MAVROS/environment
        # configuration. We do not assume a fixed Gazebo world.
        msg.header.frame_id = (
            "map"
        )

        msg.pose.position.x = (
            target[0]
        )

        msg.pose.position.y = (
            target[1]
        )

        msg.pose.position.z = (
            target[2]
        )

        # Neutral orientation. Vehicle yaw control can be supplied by
        # a deployment-specific controller if required.
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        setpoint_pub.publish(
            msg
        )

    rospy.Timer(
        rospy.Duration(
            1.0
            / publish_rate_hz
        ),
        _publish_timer,
    )

    logger.info(
        f"MAVROS waypoint adapter "
        f"for {namespace} running."
    )

    logger.info(
        f"Input waypoints: "
        f"{waypoint_topic}"
    )

    logger.info(
        f"Current pose: "
        f"{pose_topic}"
    )

    logger.info(
        f"MAVROS setpoints: "
        f"{setpoint_topic}"
    )

    logger.info(
        f"Setpoint rate: "
        f"{publish_rate_hz:.1f} Hz"
    )

    logger.info(
        f"Waypoint tolerance: "
        f"{tolerance_m:.2f} m"
    )

    if not set_guided:

        logger.info(
            "Automatic GUIDED-mode "
            "selection is disabled."
        )

    if not auto_arm:

        logger.info(
            "Automatic arming is disabled."
        )

    rospy.spin()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "MAVROS local-position "
            "waypoint adapter"
        )
    )

    parser.add_argument(
        "--namespace",
        type=str,
        default="/drone1",
        help=(
            "Drone ROS namespace, "
            "e.g. /drone1"
        ),
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=5.0,
        help=(
            "Continuous MAVROS setpoint "
            "publication rate in Hz."
        ),
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.5,
        help=(
            "Waypoint-reached distance "
            "threshold in meters."
        ),
    )

    parser.add_argument(
        "--set-guided",
        action="store_true",
        help=(
            "Request ArduPilot GUIDED mode "
            "through MAVROS at startup."
        ),
    )

    parser.add_argument(
        "--auto-arm",
        action="store_true",
        help=(
            "Request vehicle arming through "
            "MAVROS at startup."
        ),
    )

    args = parser.parse_args()

    try:

        run_ros_node(
            namespace=args.namespace,
            publish_rate_hz=args.rate,
            tolerance_m=args.tolerance,
            set_guided=args.set_guided,
            auto_arm=args.auto_arm,
        )

    except ImportError:

        logger.error(
            "This adapter requires rospy, "
            "geometry_msgs, and MAVROS."
        )


if __name__ == "__main__":
    main()
