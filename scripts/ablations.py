#!/usr/bin/env python3
"""
Ablation studies: prompt diversity and dataset size.

1. Prompt diversity: low-diversity (generic prompt) vs diverse prompts, size-matched
2. Dataset size: subsample T2I to 50, 100, 200, full — accuracy vs size

Usage:
    python scripts/ablations.py                    # full run (generates low-div images + all experiments)
    python scripts/ablations.py --skip-generate    # skip image generation, just run experiments
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import replicate
import requests
import timm
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

CLASSES = ["laptop", "mouse", "pen", "phone", "rubiks_cube", "water_bottle"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SCRATCHPAD = Path("/private/tmp/claude-501/-Users-lakshyachaudhry-Downloads-SynthBench/1a13e6fe-6254-4857-b1ee-0d4ee000a30f/scratchpad")

GENERIC_PROMPTS = {
    "laptop": "a photo of a laptop computer",
    "mouse": "a photo of a computer mouse",
    "pen": "a photo of a pen",
    "phone": "a photo of a smartphone",
    "rubiks_cube": "a photo of a Rubik's cube",
    "water_bottle": "a photo of a water bottle",
}


# --- Image generation (low-diversity) ---

def generate_image(prompt: str) -> bytes | None:
    """Call Replicate FLUX.2 Pro and return image bytes."""
    try:
        output = replicate.run(
            "black-forest-labs/flux-2-pro",
            input={
                "prompt": prompt,
                "aspect_ratio": "1:1",
                "output_format": "jpg",
                "output_quality": 95,
            },
        )
        if hasattr(output, "read"):
            return output.read()
        url = str(output)
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            return resp.content
        return None
    except Exception as e:
        print(f"    API error: {str(e)[:200]}")
        return None


def generate_low_diversity(output_dir: Path, num_per_class: int = 100):
    """Generate low-diversity T2I images using one generic prompt per class."""
    print(f"Generating {num_per_class} low-diversity images per class...")

    for cls in CLASSES:
        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(cls_dir.glob("*.jpg")))
        if existing >= num_per_class:
            print(f"  {cls}: already have {existing}, skipping")
            continue

        prompt = GENERIC_PROMPTS[cls]
        tasks = list(range(existing, num_per_class))

        failures = 0
        pbar = tqdm(tasks, desc=f"  {cls}")
        for idx in pbar:
            result = generate_image(prompt)
            if result:
                (cls_dir / f"{idx:04d}.jpg").write_bytes(result)
            else:
                failures += 1
            pbar.set_postfix(failures=failures)
            time.sleep(2)  # stay well under 6 req/min rate limit

        # Resize to 224x224
        for img_path in cls_dir.glob("*.jpg"):
            img = Image.open(img_path).convert("RGB")
            if img.size != (224, 224):
                intermediate = int(224 * 256 / 224)
                w, h = img.size
                if w < h:
                    nw, nh = intermediate, int(h * intermediate / w)
                else:
                    nh, nw = intermediate, int(w * intermediate / h)
                img = img.resize((nw, nh), Image.Resampling.LANCZOS)
                left = (nw - 224) // 2
                top = (nh - 224) // 2
                img = img.crop((left, top, left + 224, top + 224))
                img.save(img_path, "JPEG", quality=95)

    print("Low-diversity generation complete\n")


# --- Training/eval (reused from few_shot.py) ---

def train_and_eval(train_dir, test_dir, epochs, batch_size, lr, device):
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

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    train_c2i = train_dataset.class_to_idx
    test_c2i = test_dataset.class_to_idx
    t2t = {tidx: train_c2i[name] for name, tidx in test_c2i.items() if name in train_c2i}

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend([t2t[l.item()] for l in labels])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return round((all_preds == all_labels).mean() * 100, 2)


def build_subsampled_dataset(source_dir: Path, tmp_dir: Path, num_per_class: int, seed: int = 42):
    """Create a temp dataset with num_per_class images subsampled from source."""
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    rng = random.Random(seed)
    for cls in CLASSES:
        cls_out = tmp_dir / cls
        cls_out.mkdir(parents=True)
        src = source_dir / cls
        if not src.exists():
            continue
        imgs = sorted(src.glob("*.jpg"))
        sampled = rng.sample(imgs, min(num_per_class, len(imgs)))
        for i, img in enumerate(sampled):
            os.symlink(img.resolve(), cls_out / f"{i:04d}.jpg")


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--skip-generate", action="store_true", help="Skip low-diversity image generation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", default="results/ablation_results.json")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    # Load API token for generation
    env_path = Path(".env")
    if not os.environ.get("REPLICATE_API_TOKEN") and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("REPLICATE_API_TOKEN="):
                os.environ["REPLICATE_API_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")

    real_test = Path("data/real/test")
    t2i_dir = Path("data/synthetic_t2i")
    lowdiv_dir = Path("data/synthetic_t2i_lowdiv")
    tmp_dir = SCRATCHPAD / "ablation_run"

    results = {}

    # --- Ablation 1: Prompt Diversity (multi-seed) ---
    print("=" * 60)
    print("ABLATION 1: Prompt Diversity")
    print("=" * 60)

    if not args.skip_generate:
        generate_low_diversity(lowdiv_dir, num_per_class=100)

    lowdiv_accs, diverse100_accs, diverse_full_accs = [], [], []
    for seed in args.seeds:
        print(f"\n  Seed {seed}:")
        torch.manual_seed(seed)

        print("    Training on low-diversity T2I (100/class)...")
        acc = train_and_eval(lowdiv_dir, real_test, args.epochs, args.batch_size, args.lr, device)
        print(f"    Low-diversity accuracy: {acc}%")
        lowdiv_accs.append(acc)

        print("    Training on diverse T2I subsampled to 100/class...")
        build_subsampled_dataset(t2i_dir, tmp_dir, 100, seed=seed)
        acc = train_and_eval(tmp_dir, real_test, args.epochs, args.batch_size, args.lr, device)
        print(f"    Diverse (100/class) accuracy: {acc}%")
        diverse100_accs.append(acc)

        print("    Training on full diverse T2I...")
        acc = train_and_eval(t2i_dir, real_test, args.epochs, args.batch_size, args.lr, device)
        print(f"    Diverse (full) accuracy: {acc}%")
        diverse_full_accs.append(acc)

    results["prompt_diversity"] = {
        "low_diversity_100": {"mean": round(np.mean(lowdiv_accs), 2), "std": round(np.std(lowdiv_accs), 2)},
        "diverse_100": {"mean": round(np.mean(diverse100_accs), 2), "std": round(np.std(diverse100_accs), 2)},
        "diverse_full": {"mean": round(np.mean(diverse_full_accs), 2), "std": round(np.std(diverse_full_accs), 2)},
    }
    print(f"\n  Summary: low-div={results['prompt_diversity']['low_diversity_100']}, "
          f"diverse-100={results['prompt_diversity']['diverse_100']}, "
          f"diverse-full={results['prompt_diversity']['diverse_full']}")

    # --- Ablation 2: Dataset Size (multi-seed, nested subsets) ---
    print()
    print("=" * 60)
    print("ABLATION 2: Dataset Size")
    print("=" * 60)

    sizes = [100, 200]
    size_accs = {s: [] for s in sizes}
    size_accs["full"] = []

    for seed in args.seeds:
        print(f"\n  Seed {seed}:")
        torch.manual_seed(seed)

        # Shuffle once per seed, then take nested slices: 50 ⊂ 100 ⊂ 200 ⊂ full
        rng = random.Random(seed)
        shuffled_per_class = {}
        for cls in CLASSES:
            cls_imgs = sorted((t2i_dir / cls).glob("*.jpg"))
            rng.shuffle(cls_imgs)
            shuffled_per_class[cls] = cls_imgs

        for size in sizes:
            print(f"    Training on T2I subsampled to {size}/class...")
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            for cls in CLASSES:
                cls_out = tmp_dir / cls
                cls_out.mkdir(parents=True)
                for i, img in enumerate(shuffled_per_class[cls][:size]):
                    os.symlink(img.resolve(), cls_out / f"{i:04d}.jpg")
            acc = train_and_eval(tmp_dir, real_test, args.epochs, args.batch_size, args.lr, device)
            print(f"    {size}/class accuracy: {acc}%")
            size_accs[size].append(acc)

        print("    Training on full T2I dataset...")
        acc = train_and_eval(t2i_dir, real_test, args.epochs, args.batch_size, args.lr, device)
        print(f"    Full (~300/class) accuracy: {acc}%")
        size_accs["full"].append(acc)

    size_results = {}
    for key, accs in size_accs.items():
        size_results[key] = {"mean": round(np.mean(accs), 2), "std": round(np.std(accs), 2)}
    results["dataset_size"] = size_results

    # Clean up
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    # --- Generate figures ---
    plot_prompt_diversity(results["prompt_diversity"])
    plot_dataset_size(results["dataset_size"])


def plot_prompt_diversity(data: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Low-Diversity\n(100/class)", "Diverse\n(100/class)", "Diverse\n(full ~300/class)"]
    keys = ["low_diversity_100", "diverse_100", "diverse_full"]
    means = [data[k]["mean"] for k in keys]
    stds = [data[k]["std"] for k in keys]
    colors = ["#EF5350", "#42A5F5", "#2196F3"]

    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.5, error_kw={"linewidth": 1.5})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 1.5,
                f"{m:.1f}%", ha="center", fontsize=13, fontweight="bold")

    ax.set_ylabel("Accuracy on Real Test Set (%)", fontsize=12)
    ax.set_title("Ablation: Prompt Diversity Impact", fontsize=14)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = Path("figures/ablation_prompt_diversity.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Figure saved to {path}")
    plt.close()


def plot_dataset_size(data: dict):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Sort by size
    sizes = []
    means = []
    stds = []
    for k, v in data.items():
        sizes.append(int(k) if k != "full" else 300)
        means.append(v["mean"])
        stds.append(v["std"])
    order = np.argsort(sizes)
    sizes = [sizes[i] for i in order]
    means = np.array([means[i] for i in order])
    stds = np.array([stds[i] for i in order])

    ax.plot(sizes, means, "o-", color="#2196F3", linewidth=2, markersize=10)
    ax.fill_between(sizes, means - stds, means + stds, alpha=0.15, color="#2196F3")
    for x, m, s in zip(sizes, means, stds):
        ax.annotate(f"{m:.1f}%", (x, m), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Synthetic Images per Class", fontsize=12)
    ax.set_ylabel("Accuracy on Real Test Set (%)", fontsize=12)
    ax.set_title("Ablation: T2I Dataset Size Impact", fontsize=14)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path("figures/ablation_dataset_size.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Figure saved to {path}")
    plt.close()


if __name__ == "__main__":
    main()
