#!/usr/bin/env python3
"""
perception_agent/inference_node.py
===================================
Implements A_perc^(j) from Section 6.4, Eq. 21 and Eq. 22.

Each drone runs an independent instance of this node against its own
camera topic (e.g. /drone1_camera/image_raw). For every frame:

  1. Runs YOLOv11n inference, thresholded at tau_conf = 0.5 (Eq. 21):
         a_perc = 1  if max_c conf_c(frame) >= tau_conf   else 0
  2. If a detection fires AND the drone has moved > d_min = 10 m since the
     last stored detection (Eq. 22), persists a rolling window of 3
     annotated frames and publishes True on /droneN/yolo_detection/detected
     to trigger the description agent.

Run standalone on a single image/video for testing:
    python inference_node.py --source path/to/video.mp4 --namespace /drone1
"""
import argparse
import os
import sys
import time
from collections import deque
from typing import Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.config import (  # noqa: E402
    DETECTION_DEDUP_DISTANCE_M,
    DETECTION_EVENT_WINDOW,
    YOLO_CLASSES,
    YOLO_CONF_THRESHOLD,
    YOLO_IMG_SIZE,
    YOLO_WEIGHTS_PATH,
)
from model.utils import euclidean_distance, get_logger  # noqa: E402

logger = get_logger("perception_agent.inference")


class PerceptionAgent:
    """Stateful per-drone perception agent (Eq. 21 + Eq. 22)."""

    def __init__(self, namespace: str, weights_path: str = YOLO_WEIGHTS_PATH,
                 conf_threshold: float = YOLO_CONF_THRESHOLD,
                 dedup_distance_m: float = DETECTION_DEDUP_DISTANCE_M):
        self.namespace = namespace
        self.conf_threshold = conf_threshold
        self.dedup_distance_m = dedup_distance_m
        self.last_detection_pos: Optional[Tuple[float, float, float]] = None
        self.frame_buffer = deque(maxlen=DETECTION_EVENT_WINDOW)
        self.first_run = True

        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. `pip install ultralytics`.")
            raise

        if not os.path.exists(weights_path):
            logger.warning(
                f"Weights not found at {weights_path}. Train the model first with "
                "train_yolov11n.py, or point --weights at a valid checkpoint."
            )
        self.model = YOLO(weights_path) if os.path.exists(weights_path) else None

    def _max_class_confidence(self, result) -> Tuple[float, Optional[str]]:
        """Returns (max_c conf_c(frame), class_name) per Eq. 21."""
        if result.boxes is None or len(result.boxes) == 0:
            return 0.0, None
        confs = result.boxes.conf.tolist()
        cls_ids = result.boxes.cls.tolist()
        best_idx = max(range(len(confs)), key=lambda i: confs[i])
        class_name = result.names.get(int(cls_ids[best_idx]), "unknown")
        return confs[best_idx], class_name

    def process_frame(self, frame, drone_position: Tuple[float, float, float]) -> dict:
        """Runs Eq. 21 (detection trigger) and Eq. 22 (dedup persistence).

        `frame` is a numpy BGR image (as read by OpenCV / a ROS image
        bridge). Returns a dict describing the outcome for logging/testing.
        """
        if self.model is None:
            return {"detected": False, "reason": "no_weights_loaded"}

        results = self.model.predict(
            source=frame, imgsz=YOLO_IMG_SIZE, conf=self.conf_threshold, verbose=False
        )
        result = results[0]
        max_conf, class_name = self._max_class_confidence(result)

        a_perc = 1 if max_conf >= self.conf_threshold else 0  # Eq. 21

        annotated_frame = result.plot()

        outcome = {
            "namespace": self.namespace,
            "detected": bool(a_perc),
            "confidence": max_conf,
            "class": class_name,
            "annotated_frame": annotated_frame,
            "persisted": False,
        }
        
        # Maintain a rolling window of the three most recent annotated frames.
        self.frame_buffer.append(
            {
                "frame": annotated_frame,
                "confidence": max_conf,
                "class": class_name,
                "position": drone_position,
                "timestamp": time.time(),
            }
        )

if not a_perc:
    return outcome

        # Eq. 22: persist only if drone has moved > d_min since last
        # stored detection, or this is the first detection ever.
        should_persist = self.first_run or (
            self.last_detection_pos is not None
            and euclidean_distance(drone_position, self.last_detection_pos)
            > self.dedup_distance_m
        )
        if should_persist:
            self.last_detection_pos = drone_position
            self.first_run = False
            outcome["persisted"] = True
            outcome["event_frame_window"] = list(self.frame_buffer)
        
            logger.info(
                f"[{self.namespace}] Persisted {class_name} detection "
                f"(conf={max_conf:.2f}) at {drone_position}; "
                f"frame_window={len(self.frame_buffer)}"
            )
        # if should_persist:
        #     self.frame_buffer.append(
        #         {
        #             "frame": outcome["annotated_frame"],
        #             "confidence": max_conf,
        #             "class": class_name,
        #             "position": drone_position,
        #             "timestamp": time.time(),
        #         }
        #     )
        #     self.last_detection_pos = drone_position
        #     self.first_run = False
        #     outcome["persisted"] = True
        #     logger.info(
        #         f"[{self.namespace}] Persisted {class_name} detection "
        #         f"(conf={max_conf:.2f}) at {drone_position}"
        #     )

        return outcome


# --------------------------------------------------------------------------
# ROS integration
# --------------------------------------------------------------------------
def run_ros_node(namespace: str):
    import rospy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool

    rospy.init_node(f"perception_agent_{namespace.strip('/')}", anonymous=False)
    bridge = CvBridge()
    agent = PerceptionAgent(namespace)
    state = {"position": (0.0, 0.0, 0.0)}

    detection_pub = rospy.Publisher(
        f"{namespace}/yolo_detection/detected", Bool, queue_size=10
    )
    annotated_pub = rospy.Publisher(f"{namespace}/yolo_detection/annotated",    Image, queue_size=1,)
    def _pose_cb(msg):
        state["position"] = (
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        )

    def _image_cb(msg):
      frame = bridge.imgmsg_to_cv2(
          msg,
          desired_encoding="bgr8"
      )
  
      outcome = agent.process_frame(
          frame,
          state["position"]
      )
  
      annotated_frame = outcome.get("annotated_frame")
  
      if annotated_frame is not None:
          annotated_msg = bridge.cv2_to_imgmsg(
              annotated_frame,
              encoding="bgr8",
          )
          annotated_msg.header = msg.header
          annotated_pub.publish(annotated_msg)
  
      if outcome.get("detected", False):
          detection_pub.publish(Bool(data=True))

    rospy.Subscriber(f"{namespace}/mavros/local_position/pose", PoseStamped, _pose_cb)
    rospy.Subscriber(f"{namespace}_camera/image_raw", Image, _image_cb)
    logger.info(f"Perception agent for {namespace} running ...")
    rospy.spin()


def main():
    parser = argparse.ArgumentParser(description="Perception Agent (YOLOv11n)")
    parser.add_argument("--namespace", type=str, default="/drone1")
    parser.add_argument("--weights", type=str, default=YOLO_WEIGHTS_PATH)
    parser.add_argument("--source", type=str, default=None,
                         help="Image/video path for standalone testing (no ROS).")
    args = parser.parse_args()

    if args.source:
        import cv2

        agent = PerceptionAgent(args.namespace, weights_path=args.weights)
        cap = cv2.VideoCapture(args.source)
        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            # Simulated straight-line drone motion for standalone testing.
            fake_position = (frame_idx * 0.5, 0.0, 10.0)
            outcome = agent.process_frame(frame, fake_position)
            if outcome["detected"]:
                print(
                    f"frame {frame_idx}: {outcome['class']} "
                    f"conf={outcome['confidence']:.2f} persisted={outcome['persisted']}"
                )
            frame_idx += 1
        cap.release()
        return

    try:
        run_ros_node(args.namespace)
    except ImportError:
        logger.warning(
            "rospy/cv_bridge not available. Use --source <video_path> for "
            "standalone testing without ROS."
        )


if __name__ == "__main__":
    main()
