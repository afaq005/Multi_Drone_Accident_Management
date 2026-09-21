#!/usr/bin/env python3
"""
perception_agent/inference_node.py
===================================

Implements A_perc^(j) from Section 6.4, Eq. 21 and Eq. 22.

Each drone runs an independent instance of this node against its own
camera topic (e.g. /drone1_camera/image_raw). For every frame:

  1. Runs YOLOv11n inference, thresholded at tau_conf = 0.5 (Eq. 21):

         a_perc = 1
         if max_c conf_c(frame) >= tau_conf
         else 0

  2. Maintains a rolling window of up to 3 recent detected,
     annotated frames.

  3. If a detection fires AND the drone has moved more than
     d_min = 10 m since the previous persisted event (Eq. 22),
     the event is persisted and published for downstream
     description processing.

ROS outputs:
    /droneN/yolo_detection/detected
        std_msgs/Bool
        Published True only when a new deduplicated event is persisted.

    /droneN/yolo_detection/annotated
        sensor_msgs/Image
        Latest YOLO-annotated camera frame.

    /droneN/yolo_detection/event
        std_msgs/String
        JSON metadata for the persisted event.

Run standalone on a video for testing:

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

logger = get_logger(
    "perception_agent.inference"
)


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

        # Stores the most recent detected annotated frames.
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
                "Train the model first with train_yolov11n.py "
                "or provide a valid path with --weights."
            )

            self.model = None

        else:
            self.model = YOLO(
                weights_path
            )

            logger.info(
                f"[{self.namespace}] Loaded YOLO weights: "
                f"{weights_path}"
            )

    def _max_class_confidence(
        self,
        result,
    ) -> Tuple[float, Optional[str]]:
        """
        Return max_c conf_c(frame) and its corresponding class name,
        as used by Eq. 21.
        """

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            return 0.0, None

        confs = (
            result.boxes.conf.tolist()
        )

        cls_ids = (
            result.boxes.cls.tolist()
        )

        best_idx = max(
            range(len(confs)),
            key=lambda i: confs[i],
        )

        class_id = int(
            cls_ids[best_idx]
        )

        class_name = result.names.get(
            class_id,
            "unknown",
        )

        return (
            float(confs[best_idx]),
            class_name,
        )

    def process_frame(
        self,
        frame,
        drone_position: Tuple[
            float,
            float,
            float,
        ],
    ) -> dict:
        """
        Run Eq. 21 detection and Eq. 22 spatial deduplication.

        Parameters
        ----------
        frame:
            NumPy BGR image from OpenCV or cv_bridge.

        drone_position:
            Current local Cartesian UAV position (x, y, z).

        Returns
        -------
        dict
            Detection state and, when applicable, persisted-event
            metadata for downstream processing.
        """

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

        max_conf, class_name = (
            self._max_class_confidence(
                result
            )
        )

        # Eq. 21
        a_perc = int(
            max_conf
            >= self.conf_threshold
        )

        annotated_frame = (
            result.plot()
        )

        outcome = {
            "namespace": self.namespace,
            "detected": bool(a_perc),
            "confidence": max_conf,
            "class": class_name,
            "annotated_frame": annotated_frame,
            "persisted": False,
        }

        # No incident detection in this frame.
        if not a_perc:
            return outcome

        timestamp = time.time()

        # Keep only detected frames in the event window.
        self.frame_buffer.append(
            {
                "frame": annotated_frame,
                "confidence": max_conf,
                "class": class_name,
                "position": drone_position,
                "timestamp": timestamp,
            }
        )

        # Eq. 22:
        # persist the first detection, then persist subsequent events
        # only after sufficient spatial displacement.
        should_persist = (
            self.first_run
            or (
                self.last_detection_pos
                is not None
                and euclidean_distance(
                    drone_position,
                    self.last_detection_pos,
                )
                > self.dedup_distance_m
            )
        )

        if not should_persist:
            return outcome

        event_id = (
            uuid.uuid4()
            .hex[:8]
            .upper()
        )

        self.last_detection_pos = (
            drone_position
        )

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
                "x": float(
                    drone_position[0]
                ),
                "y": float(
                    drone_position[1]
                ),
                "z": float(
                    drone_position[2]
                ),
            },
            "timestamp": timestamp,
            "frame_window_size": len(
                self.frame_buffer
            ),
        }

        logger.info(
            f"[{self.namespace}] "
            f"Persisted event {event_id}: "
            f"{class_name} "
            f"(conf={max_conf:.2f}) "
            f"at {drone_position}; "
            f"frame_window="
            f"{len(self.frame_buffer)}"
        )

        return outcome


# --------------------------------------------------------------------------
# ROS integration
# --------------------------------------------------------------------------

def run_ros_node(
    namespace: str,
    weights_path: str = YOLO_WEIGHTS_PATH,
):
    import rospy

    from cv_bridge import CvBridge
    from geometry_msgs.msg import (
        PoseStamped,
    )
    from sensor_msgs.msg import Image
    from std_msgs.msg import (
        Bool,
        String,
    )

    rospy.init_node(
        f"perception_agent_"
        f"{namespace.strip('/')}",
        anonymous=False,
    )

    bridge = CvBridge()

    agent = PerceptionAgent(
        namespace,
        weights_path=weights_path,
    )

    state = {
        "position": (
            0.0,
            0.0,
            0.0,
        )
    }

    # Event trigger for downstream description processing.
    detection_pub = rospy.Publisher(
        f"{namespace}/"
        f"yolo_detection/detected",
        Bool,
        queue_size=10,
    )

    # YOLO visualization output.
    annotated_pub = rospy.Publisher(
        f"{namespace}/"
        f"yolo_detection/annotated",
        Image,
        queue_size=1,
    )

    # Structured event metadata for downstream agents.
    event_pub = rospy.Publisher(
        f"{namespace}/"
        f"yolo_detection/event",
        String,
        queue_size=10,
    )

    def _pose_cb(msg):

        state["position"] = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
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

        annotated_frame = (
            outcome.get(
                "annotated_frame"
            )
        )

        # Republish the latest annotated image for visualization
        # and downstream description processing.
        if annotated_frame is not None:

            annotated_msg = (
                bridge.cv2_to_imgmsg(
                    annotated_frame,
                    encoding="bgr8",
                )
            )

            annotated_msg.header = (
                msg.header
            )

            annotated_pub.publish(
                annotated_msg
            )

        # Trigger downstream processing ONLY for a newly persisted
        # event, not for every repeated detection frame.
        if outcome.get(
            "persisted",
            False,
        ):

            event = outcome["event"]

            event_pub.publish(
                String(
                    data=json.dumps(
                        event
                    )
                )
            )

            detection_pub.publish(
                Bool(
                    data=True
                )
            )

    rospy.Subscriber(
        f"{namespace}/"
        f"mavros/local_position/pose",
        PoseStamped,
        _pose_cb,
        queue_size=10,
    )

    rospy.Subscriber(
        f"{namespace}_camera/"
        f"image_raw",
        Image,
        _image_cb,
        queue_size=1,
    )

    logger.info(
        f"Perception agent for "
        f"{namespace} running."
    )

    logger.info(
        f"Camera topic: "
        f"{namespace}_camera/image_raw"
    )

    logger.info(
        f"Event topic: "
        f"{namespace}/"
        f"yolo_detection/event"
    )

    rospy.spin()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Perception Agent "
            "(YOLOv11n)"
        )
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
        help=(
            "Image/video path for "
            "standalone testing "
            "(no ROS)."
        ),
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

            # Synthetic straight-line motion used only for
            # standalone testing without ROS.
            fake_position = (
                frame_idx * 0.5,
                0.0,
                10.0,
            )

            outcome = (
                agent.process_frame(
                    frame,
                    fake_position,
                )
            )

            if outcome.get(
                "detected",
                False,
            ):

                print(
                    f"frame {frame_idx}: "
                    f"{outcome['class']} "
                    f"conf="
                    f"{outcome['confidence']:.2f} "
                    f"persisted="
                    f"{outcome['persisted']}"
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
            "Use --source <video_path> "
            "for standalone testing "
            "without ROS."
        )


if __name__ == "__main__":
    main()
