#!/usr/bin/env python3
"""
dataset_finetuning/generate_captions_moondream2.py
==================================================

Generates the BLIP-2 fine-tuning image-caption pairs described in
Section 4.1 using moondream2.

The reported BLIP-2 dataset uses the original image splits only:

    train: 1,783
    validation: 511
    test: 254
    total: 2,548

YOLO-only augmented training images whose filenames contain "_aug_"
are excluded by default.

Output JSONL format:

    {
        "image": "../../compiled/train/images/example.jpg",
        "caption": "..."
    }

Image paths are written relative to the output JSONL file so the
repository/archive can be moved without rewriting absolute paths.

Usage:

    python dataset_finetuning/generate_captions_moondream2.py \
        --split-dir dataset_finetuning/compiled/train/images \
        --output dataset_finetuning/blip2_data/train.jsonl
"""

import argparse
import json
import os
import sys

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from model.utils import get_logger  # noqa: E402

logger = get_logger(
    "dataset_finetuning.moondream2"
)

IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

MOONDREAM2_MODEL_ID = (
    "vikhyatk/moondream2"
)

CAPTION_PROMPT = (
    "Describe this highway accident or fire "
    "scene in one concise sentence."
)


def load_moondream2():
    """Load the moondream2 caption-generation model."""

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

    except ImportError:
        logger.error(
            "transformers not installed. "
            "Install it with "
            "`pip install transformers`."
        )
        raise SystemExit(1)

    logger.info(
        f"Loading {MOONDREAM2_MODEL_ID} ..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MOONDREAM2_MODEL_ID,
            trust_remote_code=True,
        )
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MOONDREAM2_MODEL_ID,
            trust_remote_code=True,
        )
    )

    return model, tokenizer


def caption_directory(
    split_dir: str,
    output_path: str,
    include_augmented: bool = False,
):
    """
    Generate one caption for each selected image in a dataset split.

    By default, filenames containing "_aug_" are excluded because the
    BLIP-2 dataset uses only the original train/validation/test images.
    """

    from PIL import Image

    if not os.path.isdir(
        split_dir
    ):
        raise FileNotFoundError(
            f"Image directory not found: "
            f"{split_dir}"
        )

    all_image_files = sorted(
        filename
        for filename in os.listdir(
            split_dir
        )
        if os.path.splitext(
            filename
        )[1].lower()
        in IMG_EXTENSIONS
    )

    if include_augmented:

        image_files = (
            all_image_files
        )

    else:

        image_files = [
            filename
            for filename in all_image_files
            if "_aug_" not in filename
        ]

    excluded = (
        len(all_image_files)
        - len(image_files)
    )

    logger.info(
        f"Found {len(all_image_files)} "
        f"images in {split_dir}"
    )

    if excluded:

        logger.info(
            f"Excluded {excluded} augmented "
            "YOLO images from BLIP-2 "
            "caption generation."
        )

    logger.info(
        f"Captioning {len(image_files)} "
        "selected images."
    )

    model, tokenizer = (
        load_moondream2()
    )

    output_path = (
        os.path.abspath(
            output_path
        )
    )

    output_dir = (
        os.path.dirname(
            output_path
        )
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    written = 0

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as out_f:

        for filename in image_files:

            image_path = (
                os.path.abspath(
                    os.path.join(
                        split_dir,
                        filename,
                    )
                )
            )

            try:

                with Image.open(
                    image_path
                ) as img:

                    image = (
                        img.convert("RGB")
                        .copy()
                    )

                encoded_image = (
                    model.encode_image(
                        image
                    )
                )

                caption = (
                    model.answer_question(
                        encoded_image,
                        CAPTION_PROMPT,
                        tokenizer,
                    )
                )

            except Exception as exc:  # noqa: BLE001

                logger.warning(
                    f"Failed to caption "
                    f"{filename}: "
                    f"{exc}; skipping."
                )

                continue

            caption = str(
                caption
            ).strip()

            if not caption:

                logger.warning(
                    f"Empty caption returned "
                    f"for {filename}; skipping."
                )

                continue

            # Store a path relative to the JSONL directory rather than
            # an absolute machine-specific path.
            relative_image_path = (
                os.path.relpath(
                    image_path,
                    start=output_dir,
                )
            )

            record = {
                "image": (
                    relative_image_path
                ),
                "caption": caption,
            }

            out_f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            written += 1

            if written % 100 == 0:

                logger.info(
                    f"... {written}/"
                    f"{len(image_files)} "
                    "captioned"
                )

    logger.info(
        f"Wrote {written} "
        "image-caption pairs to "
        f"{output_path}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate BLIP-2 fine-tuning "
            "captions using moondream2"
        )
    )

    parser.add_argument(
        "--split-dir",
        type=str,
        required=True,
        help=(
            "Image directory for one "
            "train/validation/test split."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Output JSONL path, e.g. "
            "dataset_finetuning/"
            "blip2_data/train.jsonl"
        ),
    )

    parser.add_argument(
        "--include-augmented",
        action="store_true",
        help=(
            "Also caption filenames containing "
            "'_aug_'. Disabled by default because "
            "the reported BLIP-2 dataset uses only "
            "the original images."
        ),
    )

    # Important: parse only after every argument has been registered.
    args = parser.parse_args()

    caption_directory(
        split_dir=args.split_dir,
        output_path=args.output,
        include_augmented=(
            args.include_augmented
        ),
    )


if __name__ == "__main__":
    main()
