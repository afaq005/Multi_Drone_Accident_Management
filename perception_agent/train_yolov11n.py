#!/usr/bin/env python3
"""
perception_agent/train_yolov11n.py
===================================
Trains the YOLOv11n incident-detection model on the 2-class
{accident, fire} dataset described in Section 4.1 (2,548 base images,
split into 1,783 train / 511 validation / 254 test images before
augmentation. Augmentation is applied only to the training partition,
yielding 3,566 train / 511 validation / 254 test images at 640x640
(4,331 images total).

Uses Ultralytics' YOLO API directly, matching Table 1's evaluation
(P, R, mAP@50, inference time, params, GFLOPs) so results are directly
comparable to the paper.

Usage:
    python train_yolov11n.py \
    --data dataset_finetuning/compiled/data.yaml \
    --epochs 100
"""
import argparse
import os
import sys

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from model.config import YOLO_IMG_SIZE  # noqa: E402
from model.utils import get_logger  # noqa: E402

logger = get_logger("perception_agent.train")


def train(data_yaml: str, epochs: int, batch: int, imgsz: int, weights: str, project: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. `pip install ultralytics`.")
        raise SystemExit(1)

    if not os.path.exists(data_yaml):
        logger.error(
            f"Dataset config {data_yaml} not found. Generate it with "
            "dataset_finetuning/prepare_dataset.py first."
        )
        raise SystemExit(1)

    model = YOLO(weights)  # "yolo11n.pt" pretrained checkpoint as starting point
    logger.info(f"Training YOLOv11n on {data_yaml} for {epochs} epochs @ {imgsz}px")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name="yolov11n_accident_fire",
        classes=None,  # dataset yaml defines 'accident', 'fire'
        patience=20,
        exist_ok=True,
    )

    # Validation mirrors Table 1's metrics (P, R, mAP@50).
    # metrics = model.val()
    # logger.info(f"mAP@50 (all classes): {metrics.box.map50:.3f}")
    # logger.info(f"Precision (mean): {metrics.box.mp:.3f}")
    # logger.info(f"Recall (mean): {metrics.box.mr:.3f}")

    # best_weights = os.path.join(project, "yolov11n_accident_fire", "weights", "best.pt")
    best_weights = os.path.join(
    project,
    "yolov11n_accident_fire",
    "weights",
    "best.pt",)

    # Evaluate the best checkpoint on the held-out TEST partition,
    # matching Table 1 of the paper.
    best_model = YOLO(best_weights)
    
    metrics = best_model.val(
        data=data_yaml,
        split="test",
        imgsz=imgsz,
    )
    
    logger.info(
        f"Test mAP@50 (all classes): "
        f"{metrics.box.map50:.3f}"
    )
    logger.info(
        f"Test Precision (mean): "
        f"{metrics.box.mp:.3f}"
    )
    logger.info(
        f"Test Recall (mean): "
        f"{metrics.box.mr:.3f}"
    )

    logger.info(f"Best weights saved to: {best_weights}")
    logger.info(
    "To use this newly trained checkpoint for inference, either set "
    "MDAM_YOLO_WEIGHTS to this path or copy it to "
    "Yolov11n_Model_Weights/best.pt."
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11n perception agent")
    parser.add_argument("--data", type=str, default="dataset_finetuning/compiled/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=YOLO_IMG_SIZE)
    parser.add_argument("--weights", type=str, default="yolo11n.pt",
                         help="Starting checkpoint (pretrained COCO weights)")
    parser.add_argument("--project", type=str, default="perception_agent/runs")
    args = parser.parse_args()
    train(args.data, args.epochs, args.batch, args.imgsz, args.weights, args.project)


if __name__ == "__main__":
    main()
