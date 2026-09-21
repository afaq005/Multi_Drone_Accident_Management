#!/usr/bin/env python3
"""
description_agent/inference.py
===============================

Implements A_desc from Section 6.5.

Given a triggered detection frame from the perception agent, the
description agent generates a single-sentence scene report using the
fine-tuned BLIP-2 model.

For multiple views of the same incident, per-view captions can be combined
using the confidence-prioritized fusion implemented in fusion.py.

The generated caption's mean token log-likelihood is retained for the
dispatch severity calculation in Eq. 24.

Standalone usage:

    python description_agent/inference.py \
        --image detected_frame.jpg

ROS usage:

    python description_agent/inference.py \
        --ros \
        --namespace /drone1

ROS inputs:

    /droneN/yolo_detection/annotated
        sensor_msgs/Image

    /droneN/yolo_detection/event
        std_msgs/String containing JSON event metadata

ROS output:

    /description_agent/view_report
        std_msgs/String containing JSON description metadata
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
    """
    Fine-tuned BLIP-2 description agent.

    If the fine-tuned checkpoint is not available, the base BLIP-2 model
    is loaded with a warning so that the software pipeline remains
    executable. Paper-matching results require the fine-tuned checkpoint.
    """

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
                "transformers/torch not installed. "
                "Install them with "
                "`pip install transformers torch`."
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
                f"Fine-tuned weights not found at "
                f"{weights_path}; loading base "
                f"checkpoint {BLIP2_BASE_MODEL}. "
                "Run finetune_blip2.py first for "
                "paper-matching results."
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
        prompt: str = (
            "Accident Scene Summary:"
        ),
    ) -> dict:
        """
        Generate a caption and its mean token log-likelihood.

        The returned `log_likelihood` is the length-normalized mean
        log-probability of the generated caption tokens. This scalar is
        used by the dispatch agent's severity calculation.
        """

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        # Move tensors to the model device while preserving integer
        # input_ids/attention_mask. Only image pixels should be cast
        # to the model floating-point dtype.
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

        decoded = (
            self.processor.batch_decode(
                outputs.sequences,
                skip_special_tokens=True,
            )[0]
            .strip()
        )

        # Decoder-only BLIP-2 variants may include the supplied prompt
        # in the decoded sequence. Remove it when present.
        if decoded.lower().startswith(
            prompt.lower()
        ):
            caption = decoded[
                len(prompt):
            ].strip()

        else:
            caption = decoded

        # Compute the mean token log-likelihood over generated tokens.
        mean_log_prob = 0.0

        if outputs.scores:

            generated_token_ids = (
                outputs.sequences[0][
                    -len(outputs.scores):
                ]
            )

            total_log_prob = 0.0

            for (
                step_scores,
                token_id,
            ) in zip(
                outputs.scores,
                generated_token_ids,
            ):

                log_probs = (
                    self._torch.log_softmax(
                        step_scores[0],
                        dim=-1,
                    )
                )

                total_log_prob += (
                    log_probs[
                        token_id
                    ].item()
                )

            mean_log_prob = (
                total_log_prob
                / len(outputs.scores)
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
        """
        Caption and fuse several drone views of the same incident.

        Expected input:

            [
                {
                    "drone_id": "/drone1",
                    "confidence": 0.91,
                    "image": <PIL image>
                },
                ...
            ]

        The confidence-prioritized fusion selects the highest-confidence
        view as the primary caption and supplements it with salient
        information from secondary views.

        The primary view's mean token log-likelihood is retained as the
        scalar language-model confidence used by the downstream severity
        calculation. A new BLIP-2 likelihood is not fabricated for the
        heuristic fused text.
        """

        if not detections:
            raise ValueError(
                "process_multi_view requires "
                "at least one detection."
            )

        views = []
        log_likelihoods = {}

        for det in detections:

            result = self.caption_image(
                det["image"]
            )

            views.append(
                View(
                    drone_id=(
                        det["drone_id"]
                    ),
                    confidence=float(
                        det["confidence"]
                    ),
                    caption=(
                        result["caption"]
                    ),
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

        # Explicit field for the downstream severity calculation.
        fused[
            "severity_log_likelihood"
        ] = primary_log_likelihood

        return fused


# --------------------------------------------------------------------------
# ROS integration
# --------------------------------------------------------------------------

def run_ros_node(
    namespace: str,
    weights_path: str = BLIP2_WEIGHTS_PATH,
):
    """
    Run the description agent as a ROS node.

    The node caches the latest YOLO-annotated image from the perception
    agent. When a persisted-event message arrives, that image is captioned
    and a structured per-view report is published for downstream
    aggregation/fusion and dispatch.
    """

    import rospy

    from cv_bridge import CvBridge
    from PIL import Image as PILImage
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    node_name = (
        "description_agent_"
        + namespace.strip("/")
    )

    rospy.init_node(
        node_name,
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
        f"{namespace}/"
        f"yolo_detection/annotated"
    )

    event_topic = (
        f"{namespace}/"
        f"yolo_detection/event"
    )

    def _image_cb(msg):

        try:
            # Convert directly to RGB so BLIP-2 receives the expected
            # channel ordering.
            rgb_frame = (
                bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="rgb8",
                )
            )

            pil_image = (
                PILImage.fromarray(
                    rgb_frame
                )
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
                f"Failed to cache "
                f"annotated image: {exc}"
            )

    def _event_cb(msg):

        try:
            event = json.loads(
                msg.data
            )

        except json.JSONDecodeError:
            logger.error(
                "Received invalid JSON on "
                f"{event_topic}"
            )
            return

        with state_lock:

            cached_image = (
                state["latest_image"]
            )

            cached_time = (
                state[
                    "latest_image_time"
                ]
            )

            if cached_image is not None:
                image = (
                    cached_image.copy()
                )
            else:
                image = None

        if image is None:
            logger.warning(
                f"Received event "
                f"{event.get('event_id')} "
                "before an annotated image "
                "was available."
            )
            return

        if cached_time is not None:

            age = (
                time.time()
                - cached_time
            )

            if age > 2.0:
                logger.warning(
                    f"Latest annotated image "
                    f"is {age:.2f} s old for "
                    f"event "
                    f"{event.get('event_id')}."
                )

        logger.info(
            f"Generating description for "
            f"event "
            f"{event.get('event_id')} "
            f"from "
            f"{event.get('drone_id', namespace)}"
        )

        result = (
            agent.caption_image(
                image
            )
        )

        report = {
            "event_id": event.get(
                "event_id"
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

            "coordinate_mode": (
                event.get(
                    "coordinate_mode",
                    "local",
                )
            ),

            "position": event.get(
                "position",
                {},
            ),

            "caption": result[
                "caption"
            ],

            # Mean generated-token log-probability.
            "log_likelihood": result[
                "log_likelihood"
            ],

            "source_event_timestamp": (
                event.get(
                    "timestamp"
                )
            ),

            "description_timestamp": (
                time.time()
            ),
        }

        report_pub.publish(
            String(
                data=json.dumps(
                    report
                )
            )
        )

        logger.info(
            f"Published description "
            f"report for event "
            f"{report['event_id']}: "
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
        f"Description agent for "
        f"{namespace} running."
    )

    logger.info(
        f"Annotated-image topic: "
        f"{annotated_topic}"
    )

    logger.info(
        f"Event topic: "
        f"{event_topic}"
    )

    logger.info(
        "Report topic: "
        "/description_agent/view_report"
    )

    rospy.spin()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Description Agent "
            "(fine-tuned BLIP-2 inference)"
        )
    )

    mode = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    mode.add_argument(
        "--image",
        type=str,
        default=None,
        help=(
            "Single image for standalone "
            "inference."
        ),
    )

    mode.add_argument(
        "--ros",
        action="store_true",
        help=(
            "Run as a ROS description node."
        ),
    )

    parser.add_argument(
        "--namespace",
        type=str,
        default="/drone1",
        help=(
            "Drone namespace used in "
            "ROS mode."
        ),
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
                namespace=args.namespace,
                weights_path=args.weights,
            )

        except ImportError:
            logger.error(
                "ROS mode requires rospy "
                "and cv_bridge."
            )

        return

    from PIL import Image

    if not os.path.isfile(
        args.image
    ):
        raise FileNotFoundError(
            f"Image not found: "
            f"{args.image}"
        )

    agent = DescriptionAgent(
        weights_path=args.weights
    )

    image = (
        Image.open(
            args.image
        )
        .convert("RGB")
    )

    result = agent.caption_image(
        image
    )

    print(
        f"Caption: "
        f"{result['caption']}"
    )

    print(
        "Mean token log-likelihood: "
        f"{result['log_likelihood']:.4f}"
    )


if __name__ == "__main__":
    main()
