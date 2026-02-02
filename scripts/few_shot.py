#!/usr/bin/env python3
"""
Few-shot experiments: train on synthetic + varying amounts of real data.

Runs a full experiment matrix across methods (T2I, Aug, Real-only),
real-shot counts (0, 5, 10, 25), and random seeds (3 runs each).
Generates a learning curve plot.

Usage:
    python scripts/few_shot.py
    python scripts/few_shot.py --real-shots 5 10 --seeds 42 43 --epochs 10  # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

CLASSES = ["laptop", "mouse", "pen", "phone", "rubiks_cube", "water_bottle"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SCRATCHPAD = Path("/private/tmp/claude-501/-Users-lakshyachaudhry-Downloads-SynthBench/1a13e6fe-6254-4857-b1ee-0d4ee000a30f/scratchpad")


def build_combined_dataset(
    synthetic_dir: Path | None,
    real_train_dir: Path,
    num_real: int,
    seed: int,
    tmp_dir: Path,
):
    """Create a temp dataset directory with symlinks to synthetic + sampled real images."""
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    rng = random.Random(seed)

    for cls in CLASSES:
        cls_dir = tmp_dir / cls
        cls_dir.mkdir(parents=True)
        idx = 0

        # Symlink synthetic images
        if synthetic_dir:
            syn_cls = synthetic_dir / cls
            if syn_cls.exists():
                for img in sorted(syn_cls.glob("*.jpg")):
                    os.symlink(img.resolve(), cls_dir / f"syn_{idx:04d}.jpg")
                    idx += 1

        # Sample and symlink real images
        if num_real > 0:
            real_cls = real_train_dir / cls
            if real_cls.exists():
                real_imgs = sorted(real_cls.glob("*.jpg"))
                sampled = rng.sample(real_imgs, min(num_real, len(real_imgs)))
                for img in sampled:
                    os.symlink(img.resolve(), cls_dir / f"real_{idx:04d}.jpg")
                    idx += 1


def train_and_eval(
    train_dir: Path,
    test_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict:
    """Train ResNet-18 and evaluate. Returns metrics dict."""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = len(train_dataset.classes)
    model = timm.create_model("resnet18", pretrained=True, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    train_class_to_idx = train_dataset.class_to_idx
    test_class_to_idx = test_dataset.class_to_idx

    # Map test labels to train label space
    test_to_train = {}
    for name, tidx in test_class_to_idx.items():
        if name in train_class_to_idx:
            test_to_train[tidx] = train_class_to_idx[name]

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            remapped = [test_to_train[l.item()] for l in labels]
            all_preds.extend(preds)
            all_labels.extend(remapped)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean() * 100

    # Per-class accuracy
    per_class = {}
    for i, cls in enumerate(train_dataset.classes):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class[cls] = round((all_preds[mask] == i).mean() * 100, 1)

    return {"accuracy": round(accuracy, 2), "per_class": per_class}


def main():
    parser = argparse.ArgumentParser(description="Run few-shot experiments")
    parser.add_argument("--real-shots", type=int, nargs="+", default=[5, 10, 25])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="results/few_shot_results.json")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    real_train = Path("data/real/train")
    real_test = Path("data/real/test")
    t2i_dir = Path("data/synthetic_t2i")
    aug_dir = Path("data/synthetic_aug")
    tmp_dir = SCRATCHPAD / "few_shot_run"

    # Load zero-shot results from Day 4
    zero_shot_path = Path("results/zero_shot_results.json")
    if zero_shot_path.exists():
        zero_shot = json.loads(zero_shot_path.read_text())
    else:
        zero_shot = {}

    # Experiment configs: (name, synthetic_dir_or_none, include in "real only")
    methods = [
        ("t2i", t2i_dir),
        ("aug", aug_dir),
        ("real_only", None),
    ]

    all_results = {}  # key: "method_numreal" -> list of accuracies across seeds

    total_runs = len(methods) * len(args.real_shots) * len(args.seeds)
    run_idx = 0

    for method_name, syn_dir in methods:
        for num_real in args.real_shots:
            key = f"{method_name}_{num_real}"
            all_results[key] = []

            for seed in args.seeds:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {method_name} + {num_real} real/class, seed={seed}")

                build_combined_dataset(syn_dir, real_train, num_real, seed, tmp_dir)

                # Count images
                total_imgs = sum(len(list((tmp_dir / c).iterdir())) for c in CLASSES)
                print(f"  Dataset: {total_imgs} images")

                result = train_and_eval(
                    tmp_dir, real_test, args.epochs, args.batch_size, args.lr, device
                )
                print(f"  Accuracy: {result['accuracy']}%")
                all_results[key].append(result)

    # Clean up
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    # Aggregate: compute mean/std per config
    summary = {"configs": {}, "zero_shot": {}}

    # Add zero-shot points
    if "t2i_zero_shot" in zero_shot:
        summary["zero_shot"]["t2i"] = zero_shot["t2i_zero_shot"]["accuracy"]
    if "aug_zero_shot" in zero_shot:
        summary["zero_shot"]["aug"] = zero_shot["aug_zero_shot"]["accuracy"]
    if "real_baseline" in zero_shot:
        summary["zero_shot"]["real_only"] = zero_shot["real_baseline"]["accuracy"]

    for key, runs in all_results.items():
        accs = [r["accuracy"] for r in runs]
        summary["configs"][key] = {
            "mean": round(np.mean(accs), 2),
            "std": round(np.std(accs), 2),
            "runs": runs,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")

    # Generate learning curve plot
    plot_learning_curves(summary, args.real_shots)


def plot_learning_curves(summary: dict, real_shots: list[int]):
    """Generate the few-shot learning curve figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [
        ("t2i", "FLUX.2 Pro (T2I) + Real", "#2196F3"),
        ("aug", "Augmented + Real", "#FF9800"),
        ("real_only", "Real Only", "#4CAF50"),
    ]

    x_vals = [0] + real_shots

    for method_key, label, color in methods:
        means = []
        stds = []

        # Zero-shot point
        zs = summary["zero_shot"].get(method_key, 0)
        means.append(zs)
        stds.append(0)  # single run, no std

        # Few-shot points
        for n in real_shots:
            key = f"{method_key}_{n}"
            if key in summary["configs"]:
                means.append(summary["configs"][key]["mean"])
                stds.append(summary["configs"][key]["std"])
            else:
                means.append(0)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(x_vals, means, "o-", label=label, color=color, linewidth=2, markersize=8)
        ax.fill_between(x_vals, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel("Real Examples per Class", fontsize=13)
    ax.set_ylabel("Accuracy on Real Test Set (%)", fontsize=13)
    ax.set_title("Few-Shot Learning Curves: Synthetic Data + Real Data", fontsize=14)
    ax.set_xticks(x_vals)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    fig_path = Path("figures/few_shot_curves.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
