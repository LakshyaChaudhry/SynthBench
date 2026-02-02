#!/usr/bin/env python3
"""
Generate comparison grid: real vs FLUX.2 Pro T2I vs programmatic augmentation.

Usage:
    python scripts/visualize.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

CLASSES = ["mouse", "pen", "phone", "laptop", "water_bottle", "rubiks_cube"]
DISPLAY_NAMES = {
    "mouse": "Mouse",
    "pen": "Pen",
    "phone": "Phone",
    "laptop": "Laptop",
    "water_bottle": "Water Bottle",
    "rubiks_cube": "Rubik's Cube",
}

SOURCES = [
    ("Real", Path("data/real/train")),
    ("FLUX.2 Pro", Path("data/synthetic_t2i")),
    ("Augmented", Path("data/synthetic_aug")),
]


def pick_image(directory: Path, index: int = 0) -> Image.Image | None:
    """Pick an image from a directory by index."""
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)
    if not files:
        return None
    return Image.open(files[index % len(files)]).convert("RGB")


def main():
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle("Data Sources: Real vs FLUX.2 Pro vs Augmentation", fontsize=16, y=0.98)

    for row, (source_name, source_dir) in enumerate(SOURCES):
        for col, class_name in enumerate(CLASSES):
            ax = axes[row][col]
            img = pick_image(source_dir / class_name)
            if img:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(DISPLAY_NAMES[class_name], fontsize=12)
            if col == 0:
                ax.set_ylabel(source_name, fontsize=13, fontweight="bold", rotation=90, labelpad=15)

    plt.tight_layout(rect=[0.03, 0, 1, 0.95])

    out_path = Path("figures/data_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
