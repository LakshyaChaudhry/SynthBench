#!/usr/bin/env python3
"""
Programmatic augmentation using torchvision transforms.

Takes real training images as seeds and applies aggressive augmentations
to generate synthetic training data.

Usage:
    python scripts/generate_aug.py --num-per-class 300
    python scripts/generate_aug.py --num-per-class 10 --classes mouse  # test run
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

CLASSES = ["mouse", "pen", "phone", "laptop", "water_bottle", "rubiks_cube"]

AUGMENT = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
])


def load_seeds(class_dir: Path) -> list[Image.Image]:
    """Load all images from a class directory as PIL Images."""
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in exts)
    images = []
    for p in paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"  Skipping {p.name}: {e}")
    return images


def main():
    parser = argparse.ArgumentParser(description="Generate augmented images from real training data")
    parser.add_argument("--input", default="data/real/train", help="Seed image directory")
    parser.add_argument("--output", default="data/synthetic_aug", help="Output directory")
    parser.add_argument("--num-per-class", type=int, default=300, help="Images per class (default: 300)")
    parser.add_argument("--classes", nargs="+", default=CLASSES, help="Classes to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target: {args.num_per_class} images per class")
    print()

    for class_name in args.classes:
        class_input = input_dir / class_name
        if not class_input.exists():
            print(f"Skipping {class_name}: {class_input} not found")
            continue

        seeds = load_seeds(class_input)
        if not seeds:
            print(f"Skipping {class_name}: no seed images")
            continue

        class_output = output_dir / class_name
        class_output.mkdir(parents=True, exist_ok=True)

        print(f"{class_name}: {len(seeds)} seeds -> {args.num_per_class} augmented images")
        for idx in tqdm(range(args.num_per_class), desc=f"  {class_name}", leave=False):
            seed_img = seeds[idx % len(seeds)]
            aug_img = AUGMENT(seed_img)
            aug_img.save(class_output / f"{idx:04d}.jpg", "JPEG", quality=95)

    # Summary
    print()
    for class_name in args.classes:
        class_output = output_dir / class_name
        if class_output.exists():
            count = len(list(class_output.glob("*.jpg")))
            print(f"  {class_name}: {count} images")
    print("Done")


if __name__ == "__main__":
    main()
