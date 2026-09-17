#!/usr/bin/env python3
"""
dataset_finetuning/prepare_dataset.py
======================================
Reproduces the dataset compilation pipeline of Section 4.1 / Fig. 3:

  1. Merge the two public Roboflow datasets [41, 42] (1,034 'Accident' +
     1,052 'Fire' images) with the prior-study dataset [43] (582 images)
     -> 2,548 base images after manual review.
  2. Apply data augmentation: horizontal/vertical flip, 90-degree rotation
     (clockwise, counter-clockwise, upside-down) -> 4,331 total images.
  3. Split into 3,566 train / 511 val / 254 test at 640x640, and emit a
     YOLO-format `data.yaml` consumable by perception_agent/train_yolov11n.py.

This script expects Roboflow exports (YOLO format: images/ + labels/) for
each source dataset, downloaded separately per each dataset's license
terms (see the References [41]-[43] in the paper / README for links).

Usage:
    python prepare_dataset.py \
        --sources roboflow_accident/ roboflow_fire/ previous_study/ \
        --output dataset_finetuning/compiled \
        --augment
"""
import argparse
import os
import random
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.utils import get_logger  # noqa: E402

logger = get_logger("dataset_finetuning.prepare")

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _list_image_label_pairs(source_dir: str):
    """Expects Roboflow's standard YOLO export layout:
        source_dir/images/*.jpg
        source_dir/labels/*.txt
    """
    images_dir = os.path.join(source_dir, "images")
    labels_dir = os.path.join(source_dir, "labels")
    pairs = []
    if not os.path.isdir(images_dir):
        logger.warning(f"No images/ folder found under {source_dir}; skipping.")
        return pairs
    for fname in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMG_EXTENSIONS:
            continue
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        if os.path.exists(label_path):
            pairs.append((os.path.join(images_dir, fname), label_path))
        else:
            logger.warning(f"No label for {fname}; skipping (background image).")
    return pairs


def merge_sources(source_dirs, output_dir):
    """Merges multiple Roboflow-exported source datasets into one pool,
    matching the 'Roboflow Compilation' step of Fig. 3."""
    merged_images_dir = os.path.join(output_dir, "raw", "images")
    merged_labels_dir = os.path.join(output_dir, "raw", "labels")
    os.makedirs(merged_images_dir, exist_ok=True)
    os.makedirs(merged_labels_dir, exist_ok=True)

    total = 0
    for src in source_dirs:
        pairs = _list_image_label_pairs(src)
        prefix = os.path.basename(os.path.normpath(src))
        for img_path, label_path in pairs:
            new_stem = f"{prefix}_{os.path.splitext(os.path.basename(img_path))[0]}"
            ext = os.path.splitext(img_path)[1]
            shutil.copy(img_path, os.path.join(merged_images_dir, new_stem + ext))
            shutil.copy(label_path, os.path.join(merged_labels_dir, new_stem + ".txt"))
            total += 1
        logger.info(f"Merged {len(pairs)} image/label pairs from {src}")

    logger.info(f"Total merged images (pre-augmentation): {total}")
    return merged_images_dir, merged_labels_dir


def augment_dataset(images_dir, labels_dir, output_dir):
    """Applies horizontal/vertical flip and 90-degree rotation
    (clockwise/counter-clockwise/upside-down) per Fig. 3's 'Data
    Augmentations' block. Bounding boxes are transformed alongside images
    since YOLO labels are normalized (cx, cy, w, h) — flips/rotations of
    multiples of 90 degrees keep boxes axis-aligned.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow not installed. `pip install Pillow`.")
        raise SystemExit(1)

    aug_images_dir = os.path.join(output_dir, "augmented", "images")
    aug_labels_dir = os.path.join(output_dir, "augmented", "labels")
    os.makedirs(aug_images_dir, exist_ok=True)
    os.makedirs(aug_labels_dir, exist_ok=True)

    transforms = {
        "orig": lambda im: im,
        "fliph": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
        "flipv": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
        "rot90cw": lambda im: im.transpose(Image.ROTATE_270),
        "rot90ccw": lambda im: im.transpose(Image.ROTATE_90),
        "rot180": lambda im: im.transpose(Image.ROTATE_180),
    }

    def _transform_box(cx, cy, w, h, kind):
        if kind == "orig":
            return cx, cy, w, h
        if kind == "fliph":
            return 1 - cx, cy, w, h
        if kind == "flipv":
            return cx, 1 - cy, w, h
        if kind == "rot90cw":
            return 1 - cy, cx, h, w
        if kind == "rot90ccw":
            return cy, 1 - cx, h, w
        if kind == "rot180":
            return 1 - cx, 1 - cy, w, h
        return cx, cy, w, h

    count = 0
    for fname in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(fname)
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        if not os.path.exists(label_path):
            continue
        with open(label_path, "r") as f:
            lines = [l.strip().split() for l in f if l.strip()]

        image = Image.open(os.path.join(images_dir, fname)).convert("RGB")
        for kind, transform_fn in transforms.items():
            out_image = transform_fn(image)
            out_stem = f"{stem}_{kind}"
            out_image.save(os.path.join(aug_images_dir, out_stem + ext))
            with open(os.path.join(aug_labels_dir, out_stem + ".txt"), "w") as out_f:
                for parts in lines:
                    cls, cx, cy, w, h = parts[0], *map(float, parts[1:5])
                    ncx, ncy, nw, nh = _transform_box(cx, cy, w, h, kind)
                    out_f.write(f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")
            count += 1

    logger.info(f"Augmentation complete: {count} images written "
                f"(orig + 5 augmentations per source image).")
    return aug_images_dir, aug_labels_dir


def split_dataset(images_dir, labels_dir, output_dir,
                   train_ratio=0.823, val_ratio=0.118):
    """Splits into train/val/test. Default ratios approximate the paper's
    3,566 / 511 / 254 split (~82.3% / 11.8% / 5.9%) out of 4,331 total."""
    stems = sorted({
        os.path.splitext(f)[0] for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    })
    random.seed(42)
    random.shuffle(stems)

    n = len(stems)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }

    for split_name, split_stems in splits.items():
        img_out = os.path.join(output_dir, split_name, "images")
        lbl_out = os.path.join(output_dir, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for stem in split_stems:
            matches = [f for f in os.listdir(images_dir) if f.startswith(stem + ".")]
            if not matches:
                continue
            img_file = matches[0]
            ext = os.path.splitext(img_file)[1]
            shutil.copy(os.path.join(images_dir, img_file), os.path.join(img_out, img_file))
            label_file = stem + ".txt"
            label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(lbl_out, label_file))
        logger.info(f"{split_name}: {len(split_stems)} images")

    return splits


def write_data_yaml(output_dir):
    yaml_path = os.path.join(output_dir, "data.yaml")
    content = f"""# Auto-generated by prepare_dataset.py — matches Section 4.1 dataset spec
train: {os.path.abspath(os.path.join(output_dir, 'train', 'images'))}
val: {os.path.abspath(os.path.join(output_dir, 'val', 'images'))}
test: {os.path.abspath(os.path.join(output_dir, 'test', 'images'))}

nc: 2
names: ['accident', 'fire']
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    logger.info(f"Wrote {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Compile & augment the accident/fire dataset (Section 4.1)")
    parser.add_argument("--sources", nargs="+", required=True,
                         help="Paths to Roboflow-exported source dataset dirs (each with images/, labels/)")
    parser.add_argument("--output", type=str, default="dataset_finetuning/compiled")
    parser.add_argument("--augment", action="store_true", help="Apply flip/rotation augmentations")
    args = parser.parse_args()

    images_dir, labels_dir = merge_sources(args.sources, args.output)

    if args.augment:
        images_dir, labels_dir = augment_dataset(images_dir, labels_dir, args.output)

    split_dataset(images_dir, labels_dir, args.output)
    write_data_yaml(args.output)
    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
