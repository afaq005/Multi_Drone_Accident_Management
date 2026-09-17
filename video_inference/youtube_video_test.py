#!/usr/bin/env python3
"""
video_inference/youtube_video_test.py
=======================================
Reproduces the real-world video evaluation of Section 5.3 / Fig. 5 / Table 3:
runs YOLOv11n detection on a video clip, then generates scene descriptions
with both the fine-tuned BLIP-2 model and SmolVLM for a side-by-side
inference-time comparison (BLIP-2 FT was ~2-2.7x faster in the paper).

This script operates on a *local* video file — download the source clip
yourself (see the README's note on reproducibility: the paper's YouTube
test clips are not redistributed here) and pass its path via --video.

Usage:
    python youtube_video_test.py --video accident_clip.mp4 --compare-smolvlm
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.config import YOLO_CONF_THRESHOLD, YOLO_IMG_SIZE, YOLO_WEIGHTS_PATH  # noqa: E402
from model.utils import get_logger  # noqa: E402

logger = get_logger("video_inference")


def run_yolo_on_video(video_path: str, weights_path: str, sample_every_n: int = 15):
    """Runs YOLOv11n over sampled frames and returns the detected crops
    (frame + bounding box) worth captioning, mirroring how the perception
    agent triggers the description agent in the live system (Eq. 21)."""
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        logger.error("opencv-python/ultralytics not installed.")
        raise SystemExit(1)

    if not os.path.exists(weights_path):
        logger.error(f"Weights not found at {weights_path}. Train perception_agent first.")
        raise SystemExit(1)

    model = YOLO(weights_path)
    cap = cv2.VideoCapture(video_path)
    detections = []
    frame_idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every_n == 0:
            results = model.predict(source=frame, imgsz=YOLO_IMG_SIZE,
                                     conf=YOLO_CONF_THRESHOLD, verbose=False)
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                best_idx = result.boxes.conf.argmax().item()
                cls_name = result.names[int(result.boxes.cls[best_idx].item())]
                conf = result.boxes.conf[best_idx].item()
                detections.append({
                    "frame_idx": frame_idx,
                    "frame": result.plot(),
                    "class": cls_name,
                    "confidence": conf,
                })
                logger.info(f"Frame {frame_idx}: {cls_name} ({conf:.2f})")
        frame_idx += 1

    cap.release()
    logger.info(f"Processed {frame_idx} frames, {len(detections)} detection events.")
    return detections


def caption_with_blip2(detections, weights_path):
    from PIL import Image
    from description_agent.inference import DescriptionAgent

    agent = DescriptionAgent(weights_path=weights_path)
    results = []
    for det in detections:
        image = Image.fromarray(det["frame"][:, :, ::-1])  # BGR -> RGB
        start = time.time()
        caption_result = agent.caption_image(image)
        elapsed = time.time() - start
        results.append({**det, "caption": caption_result["caption"],
                         "engine": "BLIP2-Finetuned", "seconds": elapsed})
        logger.info(f"[BLIP2-FT {elapsed:.3f}s] {caption_result['caption']}")
    return results


def caption_with_smolvlm(detections):
    try:
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor
        import torch
    except ImportError:
        logger.error("transformers/torch not installed for SmolVLM comparison.")
        return []

    model_id = "HuggingFaceTB/SmolVLM-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    results = []
    for det in detections:
        image = Image.fromarray(det["frame"][:, :, ::-1])
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this accident or fire scene in one sentence."},
            ]}
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

        start = time.time()
        output_ids = model.generate(**inputs, max_new_tokens=64)
        elapsed = time.time() - start
        caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        results.append({**det, "caption": caption, "engine": "SmolVLM", "seconds": elapsed})
        logger.info(f"[SmolVLM {elapsed:.3f}s] {caption}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Real-world video test (Fig. 5 / Table 3)")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--weights", type=str, default=YOLO_WEIGHTS_PATH)
    parser.add_argument("--blip2-weights", type=str,
                         default="description_agent/weights/blip2_finetuned")
    parser.add_argument("--compare-smolvlm", action="store_true")
    parser.add_argument("--sample-every-n", type=int, default=15)
    args = parser.parse_args()

    detections = run_yolo_on_video(args.video, args.weights, args.sample_every_n)
    if not detections:
        logger.warning("No detections found in video.")
        return

    blip2_results = caption_with_blip2(detections, args.blip2_weights)

    if args.compare_smolvlm:
        smolvlm_results = caption_with_smolvlm(detections)
        avg_blip2 = sum(r["seconds"] for r in blip2_results) / len(blip2_results)
        avg_smol = (
            sum(r["seconds"] for r in smolvlm_results) / len(smolvlm_results)
            if smolvlm_results else float("nan")
        )
        logger.info(f"Average BLIP2-Finetuned time: {avg_blip2:.3f}s")
        logger.info(f"Average SmolVLM time: {avg_smol:.3f}s")


if __name__ == "__main__":
    main()
