#!/usr/bin/env python3
"""
description_agent/inference.py
===============================
Implements A_desc from Section 6.5: given one or more triggered detection
frames (from the perception agent's rolling window, Eq. 22), generates a
single-sentence scene report using the fine-tuned BLIP-2 model, then fuses
multi-drone views of the same incident via fusion.py (Eq. 23).

Also exposes `caption_log_likelihood`, used by the dispatch agent's
severity score (Eq. 24: severity = sigmoid(alpha*conf_YOLO + beta*log p_BLIP2)).

Usage (single image):
    python inference.py --image frame.jpg

Usage (ROS): triggered by /droneN/yolo_detection/detected == True.
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.config import (  # noqa: E402
    BLIP2_BASE_MODEL,
    BLIP2_MAX_NEW_TOKENS,
    BLIP2_WEIGHTS_PATH,
)
from model.utils import get_logger  # noqa: E402
from description_agent.fusion import View, fuse_captions  # noqa: E402

logger = get_logger("description_agent.inference")


class DescriptionAgent:
    """Loads the fine-tuned BLIP-2 model (falls back to the base
    checkpoint with a warning if fine-tuned weights are not present, so the
    pipeline is still runnable end-to-end before fine-tuning is complete)."""

    def __init__(self, weights_path: str = BLIP2_WEIGHTS_PATH):
        try:
            import torch
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
        except ImportError:
            logger.error("transformers/torch not installed. `pip install transformers torch`.")
            raise

        self._torch = torch
        model_source = weights_path if os.path.isdir(weights_path) else BLIP2_BASE_MODEL
        if model_source == BLIP2_BASE_MODEL:
            logger.warning(
                f"Fine-tuned weights not found at {weights_path}; loading base "
                f"checkpoint {BLIP2_BASE_MODEL} instead. Run finetune_blip2.py first "
                "for paper-matching results."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained(model_source)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def caption_image(self, image, prompt: str = "Accident Scene Summary:") -> dict:
        """Generates a caption and its approximate log-likelihood, needed
        for the severity score in Eq. 24 (log p_BLIP2(y_hat | I))."""
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device, self._torch.float16 if self.device == "cuda" else self._torch.float32
        )
        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=BLIP2_MAX_NEW_TOKENS,
                output_scores=True,
                return_dict_in_generate=True,
            )
        caption = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )[0].strip()

        # Approximate sequence log-likelihood from per-token generation scores.
        log_prob = 0.0
        if outputs.scores:
            for step_scores, token_id in zip(outputs.scores, outputs.sequences[0][-len(outputs.scores):]):
                log_probs = self._torch.log_softmax(step_scores[0], dim=-1)
                log_prob += log_probs[token_id].item()
            log_prob /= max(len(outputs.scores), 1)  # length-normalized

        return {"caption": caption, "log_likelihood": log_prob}

    def process_multi_view(self, detections: list) -> dict:
        """detections: list of {"drone_id", "confidence", "image"} dicts
        for the same fused incident (multiple drones observing it). Returns
        the fused report dict (Eq. 23) plus per-view log-likelihoods for
        the dispatch agent's severity score (Eq. 24)."""
        views = []
        log_likelihoods = {}
        for det in detections:
            result = self.caption_image(det["image"])
            views.append(
                View(drone_id=det["drone_id"], confidence=det["confidence"],
                     caption=result["caption"])
            )
            log_likelihoods[det["drone_id"]] = result["log_likelihood"]

        fused = fuse_captions(views)
        fused["log_likelihoods"] = log_likelihoods
        return fused


def main():
    parser = argparse.ArgumentParser(description="Description Agent (fine-tuned BLIP-2) inference")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--weights", type=str, default=BLIP2_WEIGHTS_PATH)
    args = parser.parse_args()

    from PIL import Image

    agent = DescriptionAgent(weights_path=args.weights)
    image = Image.open(args.image).convert("RGB")
    result = agent.caption_image(image)
    print(f"Caption: {result['caption']}")
    print(f"Log-likelihood (length-normalized): {result['log_likelihood']:.4f}")


if __name__ == "__main__":
    main()
