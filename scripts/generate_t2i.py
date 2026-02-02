#!/usr/bin/env python3
"""
Text-to-image synthetic data generation using FLUX.2 Pro via Replicate.

Generates diverse synthetic images for each object class using systematically
varied prompts (object descriptions, backgrounds, lighting, angles, styles).

Usage:
    python scripts/generate_t2i.py --num-per-class 300
    python scripts/generate_t2i.py --num-per-class 2 --classes mouse pen  # test run

Requires REPLICATE_API_TOKEN in .env file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import replicate
import requests
from tqdm import tqdm

CLASSES = ["mouse", "pen", "phone", "laptop", "water_bottle", "rubiks_cube"]

# --- Prompt templates ---
# Each class gets specific object descriptions; backgrounds, lighting, angles, styles are shared.

OBJECT_DESCRIPTIONS = {
    "mouse": [
        "a white wireless computer mouse",
        "a black ergonomic mouse",
        "a gaming mouse with RGB lighting",
        "a simple office mouse",
        "a compact travel mouse",
        "a sleek modern wireless mouse",
        "a computer mouse with scroll wheel",
        "a Logitech-style wireless mouse",
    ],
    "pen": [
        "a blue ballpoint pen",
        "a black ink pen",
        "a fountain pen",
        "a mechanical pencil",
        "a felt-tip pen",
        "a red pen",
        "a clear plastic pen",
        "a fine-point writing pen",
    ],
    "phone": [
        "a modern smartphone",
        "a black iPhone",
        "a smartphone with a dark screen",
        "a phone in a clear case",
        "a smartphone face-down",
        "a slim modern phone",
        "a phone with a cracked screen protector",
        "a white smartphone",
    ],
    "laptop": [
        "a silver laptop computer",
        "a MacBook-style laptop",
        "a thin ultrabook laptop",
        "a laptop with the lid open",
        "a closed laptop computer",
        "a dark gray laptop",
        "a laptop showing a blank screen",
        "a modern slim laptop",
    ],
    "water_bottle": [
        "a plastic water bottle",
        "a Hydro Flask with stickers",
        "a clear reusable water bottle",
        "a stainless steel water bottle",
        "a sports water bottle",
        "a water bottle with a straw lid",
        "a colorful reusable bottle",
        "a tall plastic water bottle",
    ],
    "rubiks_cube": [
        "a classic Rubik's cube",
        "a scrambled Rubik's cube",
        "a solved Rubik's cube",
        "a 3x3 Rubik's cube",
        "a colorful puzzle cube",
        "a speed cube",
        "a Rubik's cube with bright colors",
        "a partially solved Rubik's cube",
    ],
}

BACKGROUNDS = [
    "on a wooden desk",
    "on a white table",
    "on a dark surface",
    "on a cluttered desk",
    "on a plain background",
    "on a marble countertop",
    "on a mousepad",
    "on a notebook",
    "on a bedside table",
    "on a kitchen counter",
    "on a concrete surface",
    "on a colorful tablecloth",
]

LIGHTING = [
    "in natural daylight",
    "in warm indoor lighting",
    "in bright fluorescent light",
    "in soft ambient light",
    "in overhead office lighting",
    "with dramatic side lighting",
    "in dim evening light",
    "with window light from the side",
]

ANGLES = [
    "photographed from above",
    "at eye level",
    "from a slight angle",
    "in a close-up shot",
    "from a 45-degree angle",
    "photographed straight on",
]

STYLES = [
    "realistic photo",
    "product photography style",
    "casual smartphone photo",
    "clean studio photograph",
    "natural candid photo",
    "sharp DSLR photograph",
]


def build_prompt(class_name: str, rng: random.Random) -> str:
    """Build a single diverse prompt by randomly combining template elements."""
    obj = rng.choice(OBJECT_DESCRIPTIONS[class_name])
    bg = rng.choice(BACKGROUNDS)
    light = rng.choice(LIGHTING)
    angle = rng.choice(ANGLES)
    style = rng.choice(STYLES)
    return f"{style} of {obj} {bg}, {light}, {angle}"


def generate_image(prompt: str) -> bytes | None:
    """Call Replicate FLUX.2 Pro and return image bytes, or None on failure."""
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
        # output is a FileOutput URL — download it
        if hasattr(output, "read"):
            return output.read()
        # or it might be a URL string
        url = str(output)
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            return resp.content
        print(f"    Download failed: {resp.status_code}")
        return None
    except Exception as e:
        msg = str(e)
        if "rate" in msg.lower() or "429" in msg:
            return "RATE_LIMITED"
        print(f"    API error: {msg[:200]}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic T2I images via FLUX.2 Pro")
    parser.add_argument("--num-per-class", type=int, default=300, help="Images per class (default: 300)")
    parser.add_argument("--output", default="data/synthetic_t2i", help="Output directory")
    parser.add_argument("--classes", nargs="+", default=CLASSES, help="Classes to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent requests (default: 5)")
    parser.add_argument("--resume", action="store_true", help="Skip existing images")
    args = parser.parse_args()

    # Load API token from .env if not in environment
    env_path = Path(".env")
    if not os.environ.get("REPLICATE_API_TOKEN") and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("REPLICATE_API_TOKEN="):
                os.environ["REPLICATE_API_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")

    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Error: Set REPLICATE_API_TOKEN in .env or environment")
        return

    output_dir = Path(args.output)
    rng = random.Random(args.seed)
    log = []  # Track all prompts for ablation experiments

    print(f"Generating {args.num_per_class} images per class via FLUX.2 Pro (Replicate)")
    print(f"Output: {output_dir}")
    print(f"Classes: {', '.join(args.classes)}")
    print(f"Workers: {args.workers} concurrent")
    print(f"Estimated cost: ~${args.num_per_class * len(args.classes) * 0.03:.2f}")
    print()

    def generate_one(task):
        """Generate a single image. Called from thread pool."""
        idx, class_name, prompt, out_path = task
        for attempt in range(3):
            result = generate_image(prompt)
            if result == "RATE_LIMITED":
                time.sleep(2 ** attempt * 5)
                continue
            break
        if result and result != "RATE_LIMITED":
            out_path.write_bytes(result)
            return {"class": class_name, "index": idx, "prompt": prompt, "file": str(out_path)}
        return {"class": class_name, "index": idx, "prompt": prompt, "file": None, "error": True}

    for class_name in args.classes:
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # Figure out starting index if resuming
        start_idx = 0
        if args.resume:
            existing = list(class_dir.glob("*.jpg"))
            start_idx = len(existing)
            if start_idx >= args.num_per_class:
                print(f"{class_name}: already have {start_idx} images, skipping")
                continue
            elif start_idx > 0:
                print(f"{class_name}: resuming from image {start_idx}")

        # Build all tasks for this class
        tasks = []
        for idx in range(start_idx, args.num_per_class):
            prompt = build_prompt(class_name, rng)
            out_path = class_dir / f"{idx:04d}.jpg"
            tasks.append((idx, class_name, prompt, out_path))

        # Run in parallel
        failures = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(generate_one, t): t for t in tasks}
            pbar = tqdm(as_completed(futures), total=len(tasks), desc=class_name)
            for future in pbar:
                entry = future.result()
                log.append(entry)
                if entry.get("error"):
                    failures += 1
                pbar.set_postfix(failures=failures)

        print(f"  {class_name}: {len(tasks) - failures} generated, {failures} failed")

    # Save generation log
    log_path = Path("results") / "t2i_generation_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\nPrompt log saved to {log_path}")


if __name__ == "__main__":
    main()
