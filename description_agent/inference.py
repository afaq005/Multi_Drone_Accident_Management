#!/usr/bin/env python3
"""
description_agent/inference.py
===============================

Implements A_desc from Section 6.5.

The agent generates one scene caption from a persisted perception event
and publishes a structured per-view description report.

For downstream severity calculation, the report includes the mean token
log-likelihood of the actually selected generated caption.

Standalone:
    python description_agent/inference.py --image frame.jpg

ROS:
    python description_agent/inference.py \
        --ros \
        --namespace /drone1
"""

import argparse
import json
import os
import sys
import time
from threading import Lock

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from model.config import (  # noqa: E402
    BLIP2_BASE_MODEL,
    BLIP2_MAX_NEW_TOKENS,
    BLIP2_WEIGHTS_PATH,
)
from model.utils import get_logger  # noqa: E402
from description_agent.fusion import (  # noqa: E402
    View,
    fuse_captions,
)

logger = get_logger(
    "description_agent.inference"
)


class DescriptionAgent:
    """Fine-tuned BLIP-2 description agent."""

    def __init__(
        self,
        weights_path: str = BLIP2_WEIGHTS_PATH,
    ):
        try:
            import torch

            from transformers import (
                Blip2ForConditionalGeneration,
                Blip2Processor,
            )
        except ImportError:
            logger.error(
                "transformers/torch not installed."
            )
            raise

        self._torch = torch

        if os.path.isdir(
            weights_path
        ):
            model_source = weights_path
        else:
            model_source = (
                BLIP2_BASE_MODEL
            )

            logger.warning(
                f"Fine-tuned weights not found at {weights_path}; "
                f"loading base checkpoint {BLIP2_BASE_MODEL}."
            )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        logger.info(
            f"Loading description model "
            f"{model_source} on {self.device}"
        )

        self.processor = (
            Blip2Processor.from_pretrained(
                model_source
            )
        )

        self.model = (
            Blip2ForConditionalGeneration
            .from_pretrained(
                model_source,
                torch_dtype=self.dtype,
            )
            .to(self.device)
        )

        self.model.eval()

    def caption_image(
        self,
        image,
        prompt: str = "Accident Scene Summary:",
    ) -> dict:
        """
        Generate a caption and the mean log-probability of the selected
        generated tokens.
        """

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        for key, value in inputs.items():
            if key == "pixel_values":
                inputs[key] = value.to(
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                inputs[key] = value.to(
                    self.device
                )

        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=(
                    BLIP2_MAX_NEW_TOKENS
                ),
                output_scores=True,
                return_dict_in_generate=True,
            )

        decoded = self.processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
        )[0].strip()

        # Some decoder-only variants include the prompt in decoded output.
        if decoded.lower().startswith(
            prompt.lower()
        ):
            caption = decoded[
                len(prompt):
            ].strip()
        else:
            caption = decoded

        mean_log_prob = 0.0

        if outputs.scores:
            beam_indices = getattr(
                outputs,
                "beam_indices",
                None,
            )

            transition_scores = (
                self.model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    beam_indices=beam_indices,
                    normalize_logits=True,
                )
            )

            selected_scores = (
                transition_scores[0]
            )

            if selected_scores.numel() > 0:
                mean_log_prob = (
                    selected_scores
                    .float()
                    .mean()
                    .item()
                )

        return {
            "caption": caption,
            "log_likelihood": (
                mean_log_prob
            ),
        }

    def process_multi_view(
        self,
        detections: list,
    ) -> dict:
        """Caption and fuse several views of the same incident."""

        if not detections:
            raise ValueError(
                "process_multi_view requires at least one detection."
            )

        views = []
        log_likelihoods = {}

        for det in detections:
            result = self.caption_image(
                det["image"]
            )

            views.append(
                View(
                    drone_id=det["drone_id"],
                    confidence=float(
                        det["confidence"]
                    ),
                    caption=result[
                        "caption"
                    ],
                )
            )

            log_likelihoods[
                det["drone_id"]
            ] = result[
                "log_likelihood"
            ]

        fused = fuse_captions(
            views
        )

        primary_drone = fused[
            "primary_drone"
        ]

        primary_log_likelihood = (
            log_likelihoods[
                primary_drone
            ]
        )

        fused[
            "log_likelihoods"
        ] = log_likelihoods

        fused[
            "primary_log_likelihood"
        ] = primary_log_likelihood

        fused[
            "severity_log_likelihood"
        ] = primary_log_likelihood

        return fused


def run_ros_node(
    namespace: str,
    weights_path: str = BLIP2_WEIGHTS_PATH,
):
    import rospy

    from cv_bridge import CvBridge
    from PIL import Image as PILImage
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    rospy.init_node(
        "description_agent_"
        + namespace.strip("/"),
        anonymous=False,
    )

    bridge = CvBridge()

    agent = DescriptionAgent(
        weights_path=weights_path
    )

    state_lock = Lock()

    state = {
        "latest_image": None,
        "latest_image_time": None,
    }

    report_pub = rospy.Publisher(
        "/description_agent/view_report",
        String,
        queue_size=10,
    )

    annotated_topic = (
        f"{namespace}/yolo_detection/annotated"
    )

    event_topic = (
        f"{namespace}/yolo_detection/event"
    )

    def _image_cb(msg):
        try:
            rgb_frame = bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="rgb8",
            )

            pil_image = PILImage.fromarray(
                rgb_frame
            )

            with state_lock:
                state[
                    "latest_image"
                ] = pil_image.copy()

                state[
                    "latest_image_time"
                ] = time.time()

        except Exception as exc:
            logger.error(
                f"Failed to cache annotated image: {exc}"
            )

    def _event_cb(msg):
        try:
            event = json.loads(
                msg.data
            )
        except json.JSONDecodeError:
            logger.error(
                f"Invalid JSON on {event_topic}"
            )
            return

        incident_id = event.get(
            "incident_id"
        )

        if not incident_id:
            logger.error(
                "Received perception event without incident_id."
            )
            return

        with state_lock:
            cached_image = state[
                "latest_image"
            ]

            cached_time = state[
                "latest_image_time"
            ]

            image = (
                cached_image.copy()
                if cached_image is not None
                else None
            )

        if image is None:
            logger.warning(
                f"Event {event.get('event_id')} received "
                "before an annotated image was available."
            )
            return

        if cached_time is not None:
            age = (
                time.time()
                - cached_time
            )

            if age > 2.0:
                logger.warning(
                    f"Cached image is {age:.2f}s old "
                    f"for event {event.get('event_id')}."
                )

        result = agent.caption_image(
            image
        )

        report = {
            "event_id": event.get(
                "event_id"
            ),
            "mission_id": event.get(
                "mission_id"
            ),
            "incident_id": incident_id,
            "incident_index": event.get(
                "incident_index"
            ),
            "drone_id": event.get(
                "drone_id",
                namespace,
            ),
            "incident_type": event.get(
                "class",
                "unknown",
            ),
            "confidence": float(
                event.get(
                    "confidence",
                    0.0,
                )
            ),
            "coordinate_mode": event.get(
                "coordinate_mode",
                "local",
            ),
            "position": event.get(
                "position",
                {},
            ),
            "assigned_target": event.get(
                "assigned_target"
            ),
            "caption": result[
                "caption"
            ],
            "log_likelihood": result[
                "log_likelihood"
            ],
            "source_event_timestamp": event.get(
                "timestamp"
            ),
            "description_timestamp": time.time(),
        }

        report_pub.publish(
            String(
                data=json.dumps(
                    report
                )
            )
        )

        logger.info(
            f"Published description for "
            f"incident {incident_id}: "
            f"{report['caption']}"
        )

    rospy.Subscriber(
        annotated_topic,
        Image,
        _image_cb,
        queue_size=1,
    )

    rospy.Subscriber(
        event_topic,
        String,
        _event_cb,
        queue_size=10,
    )

    logger.info(
        f"Description agent for {namespace} running."
    )

    rospy.spin()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Description Agent "
            "(fine-tuned BLIP-2 inference)"
        )
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--image",
        type=str,
    )

    mode.add_argument(
        "--ros",
        action="store_true",
    )

    parser.add_argument(
        "--namespace",
        type=str,
        default="/drone1",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=BLIP2_WEIGHTS_PATH,
    )

    args = parser.parse_args()

    if args.ros:
        try:
            run_ros_node(
                args.namespace,
                args.weights,
            )
        except ImportError:
            logger.error(
                "ROS mode requires rospy and cv_bridge."
            )
        return

    from PIL import Image

    if not os.path.isfile(
        args.image
    ):
        raise FileNotFoundError(
            f"Image not found: {args.image}"
        )

    agent = DescriptionAgent(
        args.weights
    )

    with Image.open(
        args.image
    ) as img:
        image = img.convert(
            "RGB"
        ).copy()

    result = agent.caption_image(
        image
    )

    print(
        f"Caption: {result['caption']}"
    )

    print(
        "Mean token log-likelihood: "
        f"{result['log_likelihood']:.4f}"
    )


if __name__ == "__main__":
    main()
