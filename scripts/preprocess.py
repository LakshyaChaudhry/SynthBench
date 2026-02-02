#!/usr/bin/env python3
"""
HEIC to JPEG preprocessing for Mirage project.

Converts iPhone HEIC photos to 224x224 JPEG images with train/test splitting.

Usage:
    python scripts/preprocess.py --input data/raw --output data/real --split 0.2

Input:  data/raw/<class>/*.HEIC
Output: data/real/train/<class>/*.jpg + data/real/test/<class>/*.jpg
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image
import pillow_heif
from tqdm import tqdm

# Register HEIF opener so PIL can open .HEIC files directly
pillow_heif.register_heif_opener()

CLASSES = ["mouse", "pen", "phone", "laptop", "water_bottle", "rubiks_cube"]
IMAGE_EXTENSIONS = {".heic", ".jpg", ".jpeg", ".png"}


def resize_center_crop(img: Image.Image, target_size: int = 224) -> Image.Image:
    """Resize shortest edge to 256px, then center crop to target_size x target_size."""
    intermediate = int(target_size * 256 / 224)  # 256 for target_size=224
    w, h = img.size

    if w < h:
        new_w = intermediate
        new_h = int(h * intermediate / w)
    else:
        new_h = intermediate
        new_w = int(w * intermediate / h)

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    return img.crop((left, top, left + target_size, top + target_size))


def get_images(directory: Path) -> list[Path]:
    """Get all image files from a directory."""
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_class(
    class_name: str,
    input_dir: Path,
    train_dir: Path,
    test_dir: Path,
    test_ratio: float,
    target_size: int,
    quality: int,
    seed: int,
) -> tuple[int, int]:
    """Process all images for one class: convert, resize, split."""
    class_input = input_dir / class_name
    if not class_input.exists():
        print(f"  Skipping {class_name}: {class_input} not found")
        return 0, 0

    files = get_images(class_input)
    if not files:
        print(f"  Skipping {class_name}: no images found")
        return 0, 0

    # Seeded shuffle for reproducible split
    random.seed(seed)
    shuffled = files.copy()
    random.shuffle(shuffled)

    split_idx = max(1, int(len(shuffled) * (1 - test_ratio)))
    train_files = shuffled[:split_idx]
    test_files = shuffled[split_idx:]

    # Ensure at least 1 test image if we have >= 2 total
    if len(files) >= 2 and not test_files:
        test_files = [train_files.pop()]

    train_out = train_dir / class_name
    test_out = test_dir / class_name
    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    def save_batch(file_list, out_dir, label):
        for idx, path in enumerate(tqdm(file_list, desc=f"  {class_name} {label}", leave=False)):
            try:
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = resize_center_crop(img, target_size)
                img.save(out_dir / f"{idx:04d}.jpg", "JPEG", quality=quality)
            except Exception as e:
                print(f"  Error processing {path.name}: {e}")

    save_batch(train_files, train_out, "train")
    save_batch(test_files, test_out, "test")

    return len(train_files), len(test_files)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess images: HEIC→JPEG, resize to 224x224, train/test split"
    )
    parser.add_argument("--input", default="data/raw", help="Input directory (default: data/raw)")
    parser.add_argument("--output", default="data/real", help="Output directory (default: data/real)")
    parser.add_argument("--split", type=float, default=0.2, help="Test set ratio (default: 0.2)")
    parser.add_argument("--size", type=int, default=224, help="Target image size (default: 224)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--classes", nargs="+", default=CLASSES, help="Class names to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        return

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Split:  {1 - args.split:.0%} train / {args.split:.0%} test")
    print(f"Size:   {args.size}x{args.size}")
    print()

    total_train, total_test = 0, 0
    for cls in args.classes:
        n_train, n_test = process_class(
            cls, input_dir, train_dir, test_dir,
            args.split, args.size, args.quality, args.seed,
        )
        total_train += n_train
        total_test += n_test

    print()
    print(f"Done: {total_train} train + {total_test} test = {total_train + total_test} images")


if __name__ == "__main__":
    main()
