#!/usr/bin/env python3
"""
dataset_finetuning/prepare_dataset.py
=====================================

Reproduces the dataset preparation procedure described in Section 4.1:

1. Merge the two public Roboflow datasets and the previous-study dataset.
2. After manual review, obtain 2,548 base images.
3. Split the BASE images before augmentation:
       train = 1,783
       val   =   511
       test  =   254
4. Apply augmentation ONLY to the training partition.
5. Generate one additional augmented sample per training image:
       original train = 1,783
       augmented      = 1,783
       final train    = 3,566
6. Validation and test partitions remain unchanged.

Final dataset:
       train = 3,566
       val   =   511
       test  =   254
       total = 4,331

Usage:
    python dataset_finetuning/prepare_dataset.py \
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

BASE_TOTAL = 2548
TRAIN_COUNT = 1783
VAL_COUNT = 511
TEST_COUNT = 254

AUGMENTATION_KINDS = [
    "fliph",
    "flipv",
    "rot90cw",
    "rot90ccw",
    "rot180",
]


def _list_image_label_pairs(source_dir: str):
    """
    Expected YOLO export structure:

        source_dir/
            images/
            labels/
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

        image_path = os.path.join(images_dir, fname)
        label_path = os.path.join(labels_dir, f"{stem}.txt")

        if os.path.exists(label_path):
            pairs.append((image_path, label_path))
        else:
            logger.warning(f"No label found for {fname}; skipping.")

    return pairs


def _reset_directory(path: str):
    """Remove an existing directory and recreate it empty."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def merge_sources(source_dirs, output_dir):
    """
    Merge all source datasets into one base-image pool.

    No augmentation is performed here.
    """
    raw_root = os.path.join(output_dir, "raw")
    merged_images_dir = os.path.join(raw_root, "images")
    merged_labels_dir = os.path.join(raw_root, "labels")

    _reset_directory(merged_images_dir)
    _reset_directory(merged_labels_dir)

    total = 0

    for src in source_dirs:
        pairs = _list_image_label_pairs(src)
        prefix = os.path.basename(os.path.normpath(src))

        for img_path, label_path in pairs:
            original_stem = os.path.splitext(os.path.basename(img_path))[0]
            new_stem = f"{prefix}_{original_stem}"

            ext = os.path.splitext(img_path)[1].lower()

            shutil.copy2(
                img_path,
                os.path.join(merged_images_dir, new_stem + ext),
            )

            shutil.copy2(
                label_path,
                os.path.join(merged_labels_dir, new_stem + ".txt"),
            )

            total += 1

        logger.info(f"Merged {len(pairs)} samples from {src}")

    logger.info(f"Total base images after merge/manual curation: {total}")

    if total != BASE_TOTAL:
        logger.warning(
            f"The paper reports {BASE_TOTAL} base images, but this run contains "
            f"{total}. Check the source datasets/manual curation before using "
            f"this run to reproduce the reported experiment."
        )

    return merged_images_dir, merged_labels_dir


def _image_map(images_dir):
    """Return {stem: filename} for supported image files."""
    mapping = {}

    for fname in os.listdir(images_dir):
        stem, ext = os.path.splitext(fname)

        if ext.lower() in IMG_EXTENSIONS:
            mapping[stem] = fname

    return mapping


def split_base_dataset(
    images_dir,
    labels_dir,
    output_dir,
    seed=42,
):
    """
    Split ORIGINAL/base images before augmentation.

    For the paper dataset:
        1,783 train
          511 val
          254 test
    """

    image_files = _image_map(images_dir)
    stems = sorted(image_files.keys())

    if len(stems) != BASE_TOTAL:
        raise ValueError(
            f"Expected {BASE_TOTAL} base images to reproduce the paper, "
            f"but found {len(stems)}."
        )

    rng = random.Random(seed)
    rng.shuffle(stems)

    train_stems = stems[:TRAIN_COUNT]

    val_start = TRAIN_COUNT
    val_end = TRAIN_COUNT + VAL_COUNT
    val_stems = stems[val_start:val_end]

    test_stems = stems[val_end:val_end + TEST_COUNT]

    splits = {
        "train": train_stems,
        "val": val_stems,
        "test": test_stems,
    }

    for split_name, split_stems in splits.items():

        img_out = os.path.join(output_dir, split_name, "images")
        lbl_out = os.path.join(output_dir, split_name, "labels")

        _reset_directory(img_out)
        _reset_directory(lbl_out)

        for stem in split_stems:

            image_filename = image_files[stem]

            shutil.copy2(
                os.path.join(images_dir, image_filename),
                os.path.join(img_out, image_filename),
            )

            label_filename = stem + ".txt"
            source_label = os.path.join(labels_dir, label_filename)

            if not os.path.exists(source_label):
                raise FileNotFoundError(
                    f"Missing label for source image {image_filename}"
                )

            shutil.copy2(
                source_label,
                os.path.join(lbl_out, label_filename),
            )

        logger.info(
            f"{split_name}: {len(split_stems)} ORIGINAL images"
        )

    # Sanity check: source images must not occur across splits.
    train_set = set(train_stems)
    val_set = set(val_stems)
    test_set = set(test_stems)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    return splits


def _transform_box(cx, cy, w, h, kind):
    """
    Transform normalized YOLO bounding box coordinates.
    """

    if kind == "fliph":
        return 1.0 - cx, cy, w, h

    if kind == "flipv":
        return cx, 1.0 - cy, w, h

    if kind == "rot90cw":
        return 1.0 - cy, cx, h, w

    if kind == "rot90ccw":
        return cy, 1.0 - cx, h, w

    if kind == "rot180":
        return 1.0 - cx, 1.0 - cy, w, h

    raise ValueError(f"Unknown augmentation type: {kind}")


def augment_training_partition(output_dir, seed=42):
    """
    Augment TRAINING IMAGES ONLY.

    One additional transformed image is generated for every original
    training image.

    Therefore:

        1,783 originals
        + 1,783 augmented
        = 3,566 final training images

    Validation and test data are never modified.
    """

    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow is required. Install with: pip install Pillow")
        raise SystemExit(1)

    train_images_dir = os.path.join(output_dir, "train", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")

    image_files = sorted(
        fname
        for fname in os.listdir(train_images_dir)
        if os.path.splitext(fname)[1].lower() in IMG_EXTENSIONS
    )

    if len(image_files) != TRAIN_COUNT:
        raise ValueError(
            f"Expected {TRAIN_COUNT} original training images before "
            f"augmentation, but found {len(image_files)}."
        )

    # Pillow compatibility across versions.
    transpose = getattr(Image, "Transpose", Image)

    transforms = {
        "fliph": lambda im: im.transpose(transpose.FLIP_LEFT_RIGHT),
        "flipv": lambda im: im.transpose(transpose.FLIP_TOP_BOTTOM),
        "rot90cw": lambda im: im.transpose(transpose.ROTATE_270),
        "rot90ccw": lambda im: im.transpose(transpose.ROTATE_90),
        "rot180": lambda im: im.transpose(transpose.ROTATE_180),
    }

    rng = random.Random(seed)

    augmented_count = 0

    for fname in image_files:

        stem, ext = os.path.splitext(fname)

        image_path = os.path.join(train_images_dir, fname)
        label_path = os.path.join(train_labels_dir, stem + ".txt")

        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"Missing training label for {fname}"
            )

        # Deterministically select one of the augmentation types.
        kind = rng.choice(AUGMENTATION_KINDS)

        image = Image.open(image_path).convert("RGB")
        augmented_image = transforms[kind](image)

        augmented_stem = f"{stem}_aug_{kind}"

        augmented_image.save(
            os.path.join(
                train_images_dir,
                augmented_stem + ext,
            )
        )

        with open(label_path, "r", encoding="utf-8") as f:
            label_lines = [
                line.strip().split()
                for line in f
                if line.strip()
            ]

        augmented_label_path = os.path.join(
            train_labels_dir,
            augmented_stem + ".txt",
        )

        with open(
            augmented_label_path,
            "w",
            encoding="utf-8",
        ) as out_f:

            for parts in label_lines:

                if len(parts) < 5:
                    raise ValueError(
                        f"Invalid YOLO label in {label_path}: {parts}"
                    )

                cls = parts[0]

                cx, cy, w, h = map(
                    float,
                    parts[1:5],
                )

                ncx, ncy, nw, nh = _transform_box(
                    cx,
                    cy,
                    w,
                    h,
                    kind,
                )

                out_f.write(
                    f"{cls} "
                    f"{ncx:.6f} "
                    f"{ncy:.6f} "
                    f"{nw:.6f} "
                    f"{nh:.6f}\n"
                )

        augmented_count += 1

    final_train_count = len(
        [
            f
            for f in os.listdir(train_images_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ]
    )

    logger.info(
        f"Generated {augmented_count} augmented training images."
    )

    logger.info(
        f"Final training partition: {final_train_count} images."
    )

    if final_train_count != 3566:
        raise RuntimeError(
            f"Expected 3566 final training images, "
            f"but found {final_train_count}."
        )


def count_partition(output_dir, split):
    images_dir = os.path.join(output_dir, split, "images")

    return len(
        [
            f
            for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ]
    )


def write_data_yaml(output_dir):

    yaml_path = os.path.join(
        output_dir,
        "data.yaml",
    )

    content = f"""# Auto-generated by prepare_dataset.py
# Dataset split follows Section 4.1:
# split original images first; augment training partition only.

train: {os.path.abspath(os.path.join(output_dir, 'train', 'images'))}
val: {os.path.abspath(os.path.join(output_dir, 'val', 'images'))}
test: {os.path.abspath(os.path.join(output_dir, 'test', 'images'))}

nc: 2
names: ['accident', 'fire']
"""

    with open(
        yaml_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(content)

    logger.info(f"Wrote dataset configuration: {yaml_path}")

    return yaml_path


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compile and prepare the accident/fire dataset "
            "using split-before-augmentation."
        )
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help=(
            "Roboflow-format source directories "
            "(each containing images/ and labels/)."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset_finetuning/compiled",
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augment the training partition only.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    images_dir, labels_dir = merge_sources(
        args.sources,
        args.output,
    )

    split_base_dataset(
        images_dir,
        labels_dir,
        args.output,
        seed=args.seed,
    )

    if args.augment:
        augment_training_partition(
            args.output,
            seed=args.seed,
        )

    train_count = count_partition(
        args.output,
        "train",
    )

    val_count = count_partition(
        args.output,
        "val",
    )

    test_count = count_partition(
        args.output,
        "test",
    )

    logger.info(
        f"Final dataset counts: "
        f"train={train_count}, "
        f"val={val_count}, "
        f"test={test_count}, "
        f"total={train_count + val_count + test_count}"
    )

    if args.augment:

        assert train_count == 3566
        assert val_count == 511
        assert test_count == 254

    write_data_yaml(args.output)

    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
