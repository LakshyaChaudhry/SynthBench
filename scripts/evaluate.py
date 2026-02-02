#!/usr/bin/env python3
"""
Evaluate a trained model on a test dataset.

Outputs accuracy, per-class F1, confusion matrix, and saves results.

Usage:
    python scripts/evaluate.py --model models/t2i_zero_shot.pth --output-json results/eval_t2i.json --output-fig figures/confusion_t2i.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on test data")
    parser.add_argument("--model", required=True, help="Path to saved checkpoint")
    parser.add_argument("--data", default="data/real/test", help="Test data directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-json", default=None, help="Save results to JSON")
    parser.add_argument("--output-fig", default=None, help="Save confusion matrix figure")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    train_classes = checkpoint["classes"]
    train_class_to_idx = checkpoint["class_to_idx"]
    num_classes = checkpoint["num_classes"]

    print(f"Model trained on: {checkpoint.get('data', 'unknown')}")
    print(f"Classes (train order): {train_classes}")

    # Load model
    model = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    test_dataset = datasets.ImageFolder(args.data, transform=transform)
    test_classes = test_dataset.classes
    print(f"Test set: {len(test_dataset)} images, classes: {test_classes}")

    # Build mapping from test indices to train indices
    # ImageFolder assigns indices alphabetically, which may differ between datasets
    test_to_train_idx = {}
    for cls_name, test_idx in test_dataset.class_to_idx.items():
        if cls_name in train_class_to_idx:
            test_to_train_idx[test_idx] = train_class_to_idx[cls_name]
        else:
            print(f"Warning: test class '{cls_name}' not in training classes")

    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            # Remap test labels to train label space
            remapped = [test_to_train_idx[l.item()] for l in labels]
            all_labels.extend(remapped)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    accuracy = (all_preds == all_labels).mean() * 100
    print(f"\nOverall accuracy: {accuracy:.1f}%\n")

    report = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=train_classes,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=train_classes,
        zero_division=0,
    )
    print(report_str)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # Save JSON
    if args.output_json:
        results = {
            "model": args.model,
            "test_data": args.data,
            "accuracy": round(accuracy, 2),
            "per_class": {
                name: {
                    "precision": round(report[name]["precision"], 3),
                    "recall": round(report[name]["recall"], 3),
                    "f1": round(report[name]["f1-score"], 3),
                    "support": report[name]["support"],
                }
                for name in train_classes
            },
            "confusion_matrix": cm.tolist(),
        }
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"Results saved to {out}")

    # Save confusion matrix figure
    if args.output_fig:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=train_classes, yticklabels=train_classes, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {Path(args.model).stem} ({accuracy:.1f}%)")
        plt.tight_layout()
        fig_path = Path(args.output_fig)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150)
        print(f"Figure saved to {fig_path}")
        plt.close()


if __name__ == "__main__":
    main()
