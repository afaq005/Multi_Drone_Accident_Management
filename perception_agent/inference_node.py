#!/usr/bin/env python3
"""
perception_agent/inference_node.py
===================================

Implements A_perc^(j) from Section 6.4, Eq. 21 and Eq. 22.

Each drone runs an independent instance of this node against its own
camera topic.

For every frame:

1. Run YOLOv11n inference using tau_conf = 0.5 (Eq. 21).
2. Maintain a rolling window of recent detected annotated frames.
3. Apply the spatial deduplication rule of Eq. 22.
4. Publish a persisted event only when a valid incident assignment
   is available from the coordination agent.

ROS inputs:
    /droneN/mavros/local_position/pose
    /droneN/assigned_incident
    /droneN_camera/image_raw

ROS outputs:
    /droneN/yolo_detection/detected
    /droneN/yolo_detection/annotated
    /droneN/yolo_detection/event

Standalone:
    python perception_agent/inference_node.py \
        --source path/to/video.mp4 \
        --namespace /drone1
"""

import argparse
import json
import os
import sys
import time
import uuid
from collections import deque
from typing import Optional, Tuple

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from model.config import (  # noqa: E402
    DETECTION_DEDUP_DISTANCE_M,
    DETECTION_EVENT_WINDOW,
    YOLO_CONF_THRESHOLD,
    YOLO_IMG_SIZE,
    YOLO_WEIGHTS_PATH,
)
from model.utils import (  # noqa: E402
    euclidean_distance,
    get_logger,
)

logger = get_logger("perception_agent.inference")


class PerceptionAgent:
    """Stateful per-drone perception agent implementing Eq. 21 and Eq. 22."""

    def __init__(
        self,
        namespace: str,
        weights_path: str = YOLO_WEIGHTS_PATH,
        conf_threshold: float = YOLO_CONF_THRESHOLD,
        dedup_distance_m: float = DETECTION_DEDUP_DISTANCE_M,
    ):
        self.namespace = namespace
        self.conf_threshold = conf_threshold
        self.dedup_distance_m = dedup_distance_m

        self.last_detection_pos: Optional[
            Tuple[float, float, float]
        ] = None

        self.frame_buffer = deque(
            maxlen=DETECTION_EVENT_WINDOW
        )

        self.first_run = True

        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error(
                "ultralytics not installed. "
                "Install it with `pip install ultralytics`."
            )
            raise

        if not os.path.exists(weights_path):
            logger.warning(
                f"Weights not found at {weights_path}. "
                "Train the model first or provide --weights."
            )
            self.model = None
        else:
            self.model = YOLO(weights_path)
            logger.info(
                f"[{self.namespace}] Loaded YOLO weights: {weights_path}"
            )

    def reset_event_state(self):
        """
        Reset event-deduplication state when the drone receives a
        different incident assignment.
        """
        self.last_detection_pos = None
        self.frame_buffer.clear()
        self.first_run = True

        logger.info(
            f"[{self.namespace}] Detection state reset "
            "for new incident assignment."
        )

    def _max_class_confidence(
        self,
        result,
    ) -> Tuple[float, Optional[str]]:
        """Return max_c conf_c(frame) and its corresponding class."""

        if result.boxes is None or len(result.boxes) == 0:
            return 0.0, None

        confs = result.boxes.conf.tolist()
        cls_ids = result.boxes.cls.tolist()

        best_idx = max(
            range(len(confs)),
            key=lambda i: confs[i],
        )

        class_id = int(cls_ids[best_idx])

        class_name = result.names.get(
            class_id,
            "unknown",
        )

        return float(confs[best_idx]), class_name

    def process_frame(
        self,
        frame,
        drone_position: Tuple[float, float, float],
    ) -> dict:
        """Run Eq. 21 detection and Eq. 22 spatial deduplication."""

        if self.model is None:
            return {
                "namespace": self.namespace,
                "detected": False,
                "persisted": False,
                "reason": "no_weights_loaded",
            }

        results = self.model.predict(
            source=frame,
            imgsz=YOLO_IMG_SIZE,
            conf=self.conf_threshold,
            verbose=False,
        )

        result = results[0]

        max_conf, class_name = self._max_class_confidence(
            result
        )

        # Eq. 21
        detected = (
            max_conf >= self.conf_threshold
        )

        annotated_frame = result.plot()

        outcome = {
            "namespace": self.namespace,
            "detected": bool(detected),
            "confidence": max_conf,
            "class": class_name,
            "annotated_frame": annotated_frame,
            "persisted": False,
        }

        if not detected:
            return outcome

        timestamp = time.time()

        # Store recent detected frames only.
        self.frame_buffer.append(
            {
                "frame": annotated_frame,
                "confidence": max_conf,
                "class": class_name,
                "position": drone_position,
                "timestamp": timestamp,
            }
        )

        # Eq. 22
        should_persist = (
            self.first_run
            or (
                self.last_detection_pos is not None
                and euclidean_distance(
                    drone_position,
                    self.last_detection_pos,
                )
                > self.dedup_distance_m
            )
        )

        if not should_persist:
            return outcome

        event_id = uuid.uuid4().hex[:8].upper()

        self.last_detection_pos = drone_position
        self.first_run = False

        outcome["persisted"] = True
        outcome["event_id"] = event_id
        outcome["event_frame_window"] = list(
            self.frame_buffer
        )

        outcome["event"] = {
            "event_id": event_id,
            "drone_id": self.namespace,
            "class": class_name,
            "confidence": round(
                max_conf,
                6,
            ),
            "coordinate_mode": "local",
            "position": {
                "x": float(drone_position[0]),
                "y": float(drone_position[1]),
                "z": float(drone_position[2]),
            },
            "timestamp": timestamp,
            "frame_window_size": len(
                self.frame_buffer
            ),
        }

        logger.info(
            f"[{self.namespace}] Persisted event {event_id}: "
            f"{class_name} "
            f"(conf={max_conf:.2f}) "
            f"at {drone_position}; "
            f"frame_window={len(self.frame_buffer)}"
        )

        return outcome


def run_ros_node(
    namespace: str,
    weights_path: str = YOLO_WEIGHTS_PATH,
):
    import rospy

    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, String

    rospy.init_node(
        f"perception_agent_{namespace.strip('/')}",
        anonymous=False,
    )

    bridge = CvBridge()

    agent = PerceptionAgent(
        namespace,
        weights_path=weights_path,
    )

    state = {
        "position": (0.0, 0.0, 0.0),
        "assignment": None,
    }

    detection_pub = rospy.Publisher(
        f"{namespace}/yolo_detection/detected",
        Bool,
        queue_size=10,
    )

    annotated_pub = rospy.Publisher(
        f"{namespace}/yolo_detection/annotated",
        Image,
        queue_size=1,
    )

    event_pub = rospy.Publisher(
        f"{namespace}/yolo_detection/event",
        String,
        queue_size=10,
    )

    def _pose_cb(msg):
        state["position"] = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        )

    def _assignment_cb(msg):
        try:
            assignment = json.loads(
                msg.data
            )
        except json.JSONDecodeError:
            logger.error(
                f"[{namespace}] Invalid assignment JSON: {msg.data!r}"
            )
            return

        assigned_drone = assignment.get(
            "drone_id"
        )

        if (
            assigned_drone is not None
            and assigned_drone != namespace
        ):
            logger.warning(
                f"[{namespace}] Ignoring assignment "
                f"intended for {assigned_drone}"
            )
            return

        new_incident_id = assignment.get(
            "incident_id"
        )

        if not new_incident_id:
            logger.error(
                f"[{namespace}] Assignment missing incident_id."
            )
            return

        previous_assignment = state.get(
            "assignment"
        )

        previous_incident_id = (
            previous_assignment.get("incident_id")
            if previous_assignment
            else None
        )

        if new_incident_id != previous_incident_id:
            agent.reset_event_state()

        state["assignment"] = assignment

        logger.info(
            f"[{namespace}] Assignment received: "
            f"mission_id={assignment.get('mission_id')}, "
            f"incident_id={new_incident_id}"
        )

    def _image_cb(msg):
        frame = bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8",
        )

        outcome = agent.process_frame(
            frame,
            state["position"],
        )

        annotated_frame = outcome.get(
            "annotated_frame"
        )

        if annotated_frame is not None:
            annotated_msg = bridge.cv2_to_imgmsg(
                annotated_frame,
                encoding="bgr8",
            )
            annotated_msg.header = msg.header
            annotated_pub.publish(
                annotated_msg
            )

        if not outcome.get(
            "persisted",
            False,
        ):
            return

        assignment = state.get(
            "assignment"
        )

        if (
            assignment is None
            or not assignment.get(
                "incident_id"
            )
        ):
            logger.warning(
                f"[{namespace}] Detection was persisted locally, "
                "but no valid assigned_incident metadata is available; "
                "downstream publication skipped."
            )
            return

        event = outcome["event"]

        event["mission_id"] = assignment.get(
            "mission_id"
        )

        event["incident_id"] = assignment[
            "incident_id"
        ]

        event["incident_index"] = assignment.get(
            "incident_index"
        )

        event["assigned_target"] = assignment.get(
            "target"
        )

        event_pub.publish(
            String(
                data=json.dumps(
                    event
                )
            )
        )

        detection_pub.publish(
            Bool(data=True)
        )

    rospy.Subscriber(
        f"{namespace}/mavros/local_position/pose",
        PoseStamped,
        _pose_cb,
        queue_size=10,
    )

    rospy.Subscriber(
        f"{namespace}/assigned_incident",
        String,
        _assignment_cb,
        queue_size=10,
    )

    rospy.Subscriber(
        f"{namespace}_camera/image_raw",
        Image,
        _image_cb,
        queue_size=1,
    )

    logger.info(
        f"Perception agent for {namespace} running."
    )

    logger.info(
        f"Assignment topic: {namespace}/assigned_incident"
    )

    logger.info(
        f"Camera topic: {namespace}_camera/image_raw"
    )

    logger.info(
        f"Event topic: {namespace}/yolo_detection/event"
    )

    rospy.spin()


def main():
    parser = argparse.ArgumentParser(
        description="Perception Agent (YOLOv11n)"
    )

    parser.add_argument(
        "--namespace",
        type=str,
        default="/drone1",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=YOLO_WEIGHTS_PATH,
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Image/video path for standalone testing.",
    )

    args = parser.parse_args()

    if args.source:
        import cv2

        agent = PerceptionAgent(
            args.namespace,
            weights_path=args.weights,
        )

        cap = cv2.VideoCapture(
            args.source
        )

        frame_idx = 0

        while cap.isOpened():
            ok, frame = cap.read()

            if not ok:
                break

            fake_position = (
                frame_idx * 0.5,
                0.0,
                10.0,
            )

            outcome = agent.process_frame(
                frame,
                fake_position,
            )

            if outcome.get(
                "detected",
                False,
            ):
                print(
                    f"frame {frame_idx}: "
                    f"{outcome['class']} "
                    f"conf={outcome['confidence']:.2f} "
                    f"persisted={outcome['persisted']}"
                )

                if outcome.get(
                    "persisted",
                    False,
                ):
                    print(
                        json.dumps(
                            outcome["event"],
                            indent=2,
                        )
                    )

            frame_idx += 1

        cap.release()
        return

    try:
        run_ros_node(
            args.namespace,
            weights_path=args.weights,
        )
    except ImportError:
        logger.warning(
            "rospy/cv_bridge not available. "
            "Use --source <video_path> for standalone testing."
        )


if __name__ == "__main__":
    main()
