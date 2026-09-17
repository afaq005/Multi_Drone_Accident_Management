#!/usr/bin/env python3
"""
description_agent/finetune_blip2.py
====================================
Fine-tunes Salesforce/blip2-opt-2.7b on image-caption pairs using float32 precision
to match the model configuration.

Expected data format — a JSONL file with one record per line:
    {"image": "path/to/frame.jpg", "caption": "A description of the scene..."}

Usage:
    python finetune_blip2.py --train data/train.jsonl --val data/val.jsonl \
        --epochs 5 --output description_agent/weights/blip2_finetuned
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
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from model.config import BLIP2_BASE_MODEL, BLIP2_FINETUNE_EPOCHS
    from model.utils import get_logger
    logger = get_logger("description_agent.finetune")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("description_agent.finetune")
    BLIP2_BASE_MODEL = "Salesforce/blip2-opt-2.7b"
    BLIP2_FINETUNE_EPOCHS = 5


class CaptionDataset(Dataset):
    """Dataset wrapper around a JSONL file containing {image, caption} records."""

    def __init__(self, jsonl_path: str):
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record["image"]).convert("RGB")
        caption = record["caption"]
        return {"image": image, "caption": caption}


class Blip2DataCollator:
    """Collates and processes raw dataset items using Blip2Processor."""

    def __init__(self, processor: Blip2Processor):
        self.processor = processor

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        captions = [item["caption"] for item in batch]

        # Process images and text targets
        inputs = self.processor(
            images=images,
            text=captions,
            padding=True,
            return_tensors="pt"
        )

        labels = inputs["input_ids"].clone()
        # Replace padding token id with -100 to ignore it in loss calculation
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        inputs["labels"] = labels
        return inputs


def freeze_vision_encoder(model):
    """Freezes the ViT vision encoder AND Q-Former to train language adaptation layers."""
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.qformer.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,}")
    return model


def finetune(train_path: str, val_path: str, epochs: int, output_dir: str,
             batch_size: int, lr: float):
    logger.info(f"Loading base model {BLIP2_BASE_MODEL}...")
    processor = Blip2Processor.from_pretrained(BLIP2_BASE_MODEL)
    
    # Force tokenizer pad_token if not defined (common with OPT models)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Load in float32 precision matching config torch_dtype
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_BASE_MODEL,
        torch_dtype=torch.float32,
    )
    model = freeze_vision_encoder(model)

    train_dataset = CaptionDataset(train_path)
    val_dataset = CaptionDataset(val_path) if val_path else None
    logger.info(f"Train pairs: {len(train_dataset)}, Val pairs: "
                f"{len(val_dataset) if val_dataset else 0}")

    collator = Blip2DataCollator(processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        logging_steps=10,
        fp16=False,  # Enforce float32 as defined in config
        report_to=[],
        save_total_limit=2,
        load_best_model_at_end=bool(val_dataset),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    logger.info(f"Starting fine-tuning for {epochs} epochs...")
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Fine-tuned BLIP-2 model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BLIP-2 description agent")
    parser.add_argument("--train", type=str, required=True, help="Path to train.jsonl")
    parser.add_argument("--val", type=str, default=None, help="Path to val.jsonl")
    parser.add_argument("--epochs", type=int, default=BLIP2_FINETUNE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output", type=str, default="description_agent/weights/blip2_finetuned")
    args = parser.parse_args()
    finetune(args.train, args.val, args.epochs, args.output, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
