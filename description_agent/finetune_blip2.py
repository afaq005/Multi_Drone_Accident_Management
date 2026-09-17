#!/usr/bin/env python3
"""
description_agent/finetune_blip2.py
====================================
Fine-tunes Salesforce/blip2-opt-2.7b on the 1,783 train / 511 val / 254 test
image-caption pairs described in Section 4.1, with the visual encoder frozen
to preserve pretrained feature representations (paper abstract / Section 5.2).

Implements the combined loss of Eq. 18:
    L_T = L_ITC + L_ITM + L_ITG
(handled internally by BLIP-2's pretraining objective; here we fine-tune
stage-2 generation, Eq. 19-20, since Q-Former + vision encoder stay frozen
and only the language-generation head/adapter is updated against the
image-captioning cross-entropy loss L_gen = L_CE(LLM_lang(Y_q), txt)).

Expected data format — a JSONL file with one record per line:
    {"image": "path/to/frame.jpg", "caption": "A car accident scene on the highway..."}

Usage:
    python finetune_blip2.py --train data/train.jsonl --val data/val.jsonl \
        --epochs 5 --output description_agent/weights/blip2_finetuned
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.config import BLIP2_BASE_MODEL, BLIP2_FINETUNE_EPOCHS  # noqa: E402
from model.utils import get_logger  # noqa: E402

logger = get_logger("description_agent.finetune")


class CaptionDataset:
    """Minimal torch Dataset wrapper around a JSONL {image, caption} file."""

    def __init__(self, jsonl_path: str, processor):
        import torch  # noqa: F401
        from PIL import Image

        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.processor = processor
        self._Image = Image

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = self._Image.open(record["image"]).convert("RGB")
        encoding = self.processor(
            images=image, text=record["caption"], padding="max_length",
            truncation=True, return_tensors="pt",
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()
        return encoding


def freeze_vision_encoder(model):
    """Freezes the ViT vision encoder AND the Q-Former, matching the
    paper's statement that fine-tuning "holds its visual encoder frozen
    to preserve pre-trained feature representations." Only the
    language-model adapter/projection layers remain trainable."""
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
    try:
        import torch
        from transformers import (
            Blip2ForConditionalGeneration,
            Blip2Processor,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        logger.error("transformers/torch not installed. `pip install transformers torch`.")
        raise SystemExit(1)

    logger.info(f"Loading base model {BLIP2_BASE_MODEL} ...")
    processor = Blip2Processor.from_pretrained(BLIP2_BASE_MODEL)
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model = freeze_vision_encoder(model)

    train_dataset = CaptionDataset(train_path, processor)
    val_dataset = CaptionDataset(val_path, processor) if val_path else None
    logger.info(f"Train pairs: {len(train_dataset)}, Val pairs: "
                f"{len(val_dataset) if val_dataset else 0}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to=[],
        save_total_limit=2,
        load_best_model_at_end=bool(val_dataset),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logger.info(f"Starting fine-tuning for {epochs} epochs (Eq. 19-20 objective) ...")
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
