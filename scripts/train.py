#!/usr/bin/env python3
"""
Fine-tune pretrained ResNet-18 on a given image dataset.

Usage:
    python scripts/train.py --data data/synthetic_t2i --output models/t2i_zero_shot.pth
    python scripts/train.py --data data/real/train --output models/real_baseline.pth --epochs 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 on image classification")
    parser.add_argument("--data", required=True, help="Training data directory (ImageFolder format)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--output", default="models/model.pth", help="Output checkpoint path")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset = datasets.ImageFolder(args.data, transform=transform)
    num_classes = len(dataset.classes)
    print(f"Dataset: {len(dataset)} images, {num_classes} classes")
    print(f"Classes: {dataset.classes}")
    print(f"Class mapping: {dataset.class_to_idx}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Model
    model = timm.create_model("resnet18", pretrained=True, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    best_loss = float("inf")
    print(f"\nTraining for {args.epochs} epochs, lr={args.lr}, batch_size={args.batch_size}")
    print()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=loss.item(), acc=f"{100.*correct/total:.1f}%")

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{args.epochs} — loss: {epoch_loss:.4f}, acc: {epoch_acc:.1f}%")

        if epoch_loss < best_loss:
            best_loss = epoch_loss

    # Save checkpoint
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
        "num_classes": num_classes,
        "epochs": args.epochs,
        "lr": args.lr,
        "data": args.data,
    }, out_path)
    print(f"\nModel saved to {out_path}")


if __name__ == "__main__":
    main()
