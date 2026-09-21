#!/usr/bin/env python3
"""
description_agent/finetune_blip2.py
===================================

Fine-tunes Salesforce/blip2-opt-2.7b on the image-caption dataset
described in Section 4.1 of the paper.

Training configuration
----------------------
The base BLIP-2 architecture was originally pretrained using
image-text contrastive (ITC), image-text matching (ITM), and
image-grounded text-generation (ITG) objectives. These original
pretraining objectives are not recomputed separately in this
task-specific fine-tuning script.

For the present accident/fire caption-generation task:

    - Vision encoder: frozen
    - Q-Former: frozen
    - Language projection: trainable
    - OPT language-generation model: trainable
    - Training objective: conditional-generation cross-entropy loss

Expected JSONL format
---------------------

Each line must contain:

    {
        "image": "path/to/image.jpg",
        "caption": "A multi-vehicle collision is visible..."
    }

Example usage
-------------

    python description_agent/finetune_blip2.py \
        --train train.jsonl \
        --val val.jsonl \
        --epochs 5 \
        --batch-size 4 \
        --lr 5e-5 \
        --seed 42 \
        --output description_agent/weights/blip2_finetuned
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Allow imports from the repository root when this file is run directly.
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

try:
    from model.config import (
        BLIP2_BASE_MODEL,
        BLIP2_FINETUNE_EPOCHS,
    )
    from model.utils import get_logger

    logger = get_logger("description_agent.finetune")

except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(
        "description_agent.finetune"
    )

    BLIP2_BASE_MODEL = (
        "Salesforce/blip2-opt-2.7b"
    )

    BLIP2_FINETUNE_EPOCHS = 5


class CaptionDataset(Dataset):
    """
    Dataset wrapper around a JSONL file containing image-caption pairs.

    Expected record:

        {
            "image": "path/to/image.jpg",
            "caption": "Description of the incident."
        }
    """

    def __init__(self, jsonl_path: str):

        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(
                f"JSONL dataset not found: {jsonl_path}"
            )

        self.records = []

        jsonl_dir = os.path.dirname(
            os.path.abspath(jsonl_path)
        )

        with open(
            jsonl_path,
            "r",
            encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at "
                        f"{jsonl_path}:{line_number}"
                    ) from exc

                if "image" not in record:
                    raise ValueError(
                        f"{jsonl_path}:{line_number} "
                        f"is missing the 'image' field."
                    )

                if "caption" not in record:
                    raise ValueError(
                        f"{jsonl_path}:{line_number} "
                        f"is missing the 'caption' field."
                    )

                image_path = record["image"]

                # Allow image paths to be either absolute or
                # relative to the JSONL file.
                if not os.path.isabs(image_path):
                    candidate = os.path.join(
                        jsonl_dir,
                        image_path,
                    )

                    if os.path.exists(candidate):
                        image_path = candidate

                if not os.path.isfile(image_path):
                    raise FileNotFoundError(
                        f"Image not found at "
                        f"{jsonl_path}:{line_number}: "
                        f"{record['image']}"
                    )

                caption = str(
                    record["caption"]
                ).strip()

                if not caption:
                    raise ValueError(
                        f"Empty caption at "
                        f"{jsonl_path}:{line_number}"
                    )

                self.records.append(
                    {
                        "image": image_path,
                        "caption": caption,
                    }
                )

        if not self.records:
            raise ValueError(
                f"No valid records found in {jsonl_path}"
            )

        logger.info(
            f"Loaded {len(self.records)} "
            f"image-caption pairs from {jsonl_path}"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        record = self.records[idx]

        # Copy the converted image into memory so the
        # underlying file handle can be safely closed.
        with Image.open(record["image"]) as img:
            image = img.convert("RGB").copy()

        return {
            "image": image,
            "caption": record["caption"],
        }


class Blip2DataCollator:
    """
    Convert raw PIL images and captions into BLIP-2 model inputs.

    Padding locations are replaced by -100 in the labels so they do
    not contribute to the cross-entropy generation loss.
    """

    def __init__(
        self,
        processor: Blip2Processor,
    ):
        self.processor = processor

    def __call__(self, batch):

        images = [
            item["image"]
            for item in batch
        ]

        captions = [
            item["caption"]
            for item in batch
        ]

        inputs = self.processor(
            images=images,
            text=captions,
            padding=True,
            return_tensors="pt",
        )

        labels = (
            inputs["input_ids"]
            .clone()
        )

        # Mask padding positions rather than masking by
        # pad_token_id directly. This is safer for OPT tokenizers
        # because EOS may also be used as the padding token.
        if "attention_mask" in inputs:
            labels[
                inputs["attention_mask"] == 0
            ] = -100

        inputs["labels"] = labels

        return inputs


def configure_trainable_parameters(model):
    """
    Configure the BLIP-2 modules used during task-specific fine-tuning.

    Frozen:
        - vision encoder
        - Q-Former

    Trainable:
        - language projection
        - OPT language-generation model
    """

    # Freeze everything first so the configuration is explicit.
    for param in model.parameters():
        param.requires_grad = False

    # Explicitly keep the visual encoder frozen.
    for param in model.vision_model.parameters():
        param.requires_grad = False

    # Explicitly keep the Q-Former frozen.
    for param in model.qformer.parameters():
        param.requires_grad = False

    # Train the projection from Q-Former representation
    # into the language-model embedding space.
    for param in model.language_projection.parameters():
        param.requires_grad = True

    # Train the OPT language-generation model.
    for param in model.language_model.parameters():
        param.requires_grad = True

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    percentage = (
        100.0
        * trainable_parameters
        / total_parameters
    )

    logger.info(
        "BLIP-2 parameter configuration:"
    )

    logger.info(
        "  Vision encoder: frozen"
    )

    logger.info(
        "  Q-Former: frozen"
    )

    logger.info(
        "  Language projection: trainable"
    )

    logger.info(
        "  Language model: trainable"
    )

    logger.info(
        f"Trainable parameters: "
        f"{trainable_parameters:,} / "
        f"{total_parameters:,} "
        f"({percentage:.2f}%)"
    )

    return model


def finetune(
    train_path: str,
    val_path: str,
    epochs: int,
    output_dir: str,
    batch_size: int,
    lr: float,
    seed: int,
):
    """
    Fine-tune BLIP-2 using conditional-generation cross-entropy loss.
    """

    set_seed(seed)

    logger.info(
        f"Random seed: {seed}"
    )

    logger.info(
        f"Loading BLIP-2 processor: "
        f"{BLIP2_BASE_MODEL}"
    )

    processor = (
        Blip2Processor.from_pretrained(
            BLIP2_BASE_MODEL
        )
    )

    # OPT tokenizers may not define a separate padding token.
    if (
        processor.tokenizer.pad_token
        is None
    ):
        processor.tokenizer.pad_token = (
            processor.tokenizer.eos_token
        )

    logger.info(
        f"Loading base model: "
        f"{BLIP2_BASE_MODEL}"
    )

    model = (
        Blip2ForConditionalGeneration
        .from_pretrained(
            BLIP2_BASE_MODEL,
            torch_dtype=torch.float32,
        )
    )

    # Synchronize tokenizer/model padding configuration.
    model.config.pad_token_id = (
        processor.tokenizer.pad_token_id
    )

    if hasattr(
        model,
        "generation_config",
    ):
        model.generation_config.pad_token_id = (
            processor.tokenizer.pad_token_id
        )

    model = configure_trainable_parameters(
        model
    )

    train_dataset = CaptionDataset(
        train_path
    )

    val_dataset = (
        CaptionDataset(val_path)
        if val_path
        else None
    )

    logger.info(
        f"Training pairs: "
        f"{len(train_dataset)}"
    )

    logger.info(
        f"Validation pairs: "
        f"{len(val_dataset) if val_dataset else 0}"
    )

    collator = Blip2DataCollator(
        processor
    )

    training_args = TrainingArguments(
        output_dir=output_dir,

        num_train_epochs=epochs,

        per_device_train_batch_size=(
            batch_size
        ),

        per_device_eval_batch_size=(
            batch_size
        ),

        learning_rate=lr,

        # Compatible with Transformers 4.40+.
        evaluation_strategy=(
            "epoch"
            if val_dataset is not None
            else "no"
        ),

        save_strategy="epoch",

        logging_steps=10,

        # The reported experiment uses float32.
        fp16=False,
        bf16=False,

        # Explicitly use PyTorch AdamW.
        optim="adamw_torch",

        # Trainer defaults retained for reproducibility.
        weight_decay=0.0,
        warmup_ratio=0.0,
        lr_scheduler_type="linear",

        report_to=[],

        save_total_limit=2,

        load_best_model_at_end=(
            val_dataset is not None
        ),

        # Required because the dataset contains custom
        # "image" and "caption" fields consumed by our collator.
        remove_unused_columns=False,

        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    logger.info(
        "Starting BLIP-2 "
        "caption-generation fine-tuning."
    )

    logger.info(
        f"Epochs: {epochs}"
    )

    logger.info(
        f"Batch size: {batch_size}"
    )

    logger.info(
        f"Learning rate: {lr}"
    )

    logger.info(
        "Objective: conditional-generation "
        "cross-entropy loss"
    )

    trainer.train()

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # Save final/best loaded model.
    trainer.save_model(
        output_dir
    )

    processor.save_pretrained(
        output_dir
    )

    # Save the training configuration alongside the model
    # to improve reproducibility.
    training_config = {
        "base_model": BLIP2_BASE_MODEL,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "precision": "float32",
        "optimizer": "AdamW",
        "vision_encoder_frozen": True,
        "qformer_frozen": True,
        "language_projection_trainable": True,
        "language_model_trainable": True,
        "objective": (
            "conditional-generation "
            "cross-entropy loss"
        ),
        "train_pairs": len(
            train_dataset
        ),
        "validation_pairs": (
            len(val_dataset)
            if val_dataset is not None
            else 0
        ),
    }

    config_path = os.path.join(
        output_dir,
        "finetuning_config.json",
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            training_config,
            f,
            indent=2,
        )

    logger.info(
        f"Fine-tuned BLIP-2 model "
        f"saved to: {output_dir}"
    )

    logger.info(
        f"Fine-tuning configuration "
        f"saved to: {config_path}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the BLIP-2 "
            "description agent."
        )
    )

    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to training JSONL file.",
    )

    parser.add_argument(
        "--val",
        type=str,
        default=None,
        help=(
            "Optional path to validation "
            "JSONL file."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=BLIP2_FINETUNE_EPOCHS,
        help="Number of fine-tuning epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducible "
            "training."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "description_agent/weights/"
            "blip2_finetuned"
        ),
        help=(
            "Directory in which to save "
            "the fine-tuned model."
        ),
    )

    args = parser.parse_args()

    finetune(
        train_path=args.train,
        val_path=args.val,
        epochs=args.epochs,
        output_dir=args.output,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
