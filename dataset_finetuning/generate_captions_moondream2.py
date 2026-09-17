#!/usr/bin/env python3
"""
dataset_finetuning/generate_captions_moondream2.py
====================================================
Reproduces the BLIP-2 fine-tuning data generation step of Section 4.1:
"we utilized moondream2 visual language model VLM to generate descriptions
of training, validation and testing images" — 1,783 train / 511 val / 254
test image-caption pairs (2,548 total, matching the YOLO dataset before
augmentation).

Outputs JSONL files consumable by description_agent/finetune_blip2.py:
    {"image": "path/to/frame.jpg", "caption": "..."}

Usage:
    python generate_captions_moondream2.py \
        --split-dir dataset_finetuning/compiled/train/images \
        --output dataset_finetuning/blip2_data/train.jsonl
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.utils import get_logger  # noqa: E402

logger = get_logger("dataset_finetuning.moondream2")

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CAPTION_PROMPT = "Describe this highway accident or fire scene in one concise sentence."


def load_moondream2():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("transformers not installed. `pip install transformers`.")
        raise SystemExit(1)

    model_id = "vikhyatk/moondream2"
    logger.info(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer


def caption_directory(split_dir: str, output_path: str):
    from PIL import Image

    model, tokenizer = load_moondream2()

    image_files = sorted(
        f for f in os.listdir(split_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    )
    logger.info(f"Captioning {len(image_files)} images from {split_dir}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for fname in image_files:
            image_path = os.path.join(split_dir, fname)
            try:
                image = Image.open(image_path).convert("RGB")
                encoded_image = model.encode_image(image)
                caption = model.answer_question(encoded_image, CAPTION_PROMPT, tokenizer)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to caption {fname}: {exc}; skipping.")
                continue

            out_f.write(json.dumps({"image": image_path, "caption": caption.strip()}) + "\n")
            written += 1
            if written % 100 == 0:
                logger.info(f"  ... {written}/{len(image_files)} captioned")

    logger.info(f"Wrote {written} image-caption pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate BLIP-2 fine-tuning captions via moondream2")
    parser.add_argument("--split-dir", type=str, required=True,
                         help="Directory of images for one split (train/val/test)")
    parser.add_argument("--output", type=str, required=True,
                         help="Output JSONL path, e.g. dataset_finetuning/blip2_data/train.jsonl")
    args = parser.parse_args()
    caption_directory(args.split_dir, args.output)


if __name__ == "__main__":
    main()
