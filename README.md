# SynthBench

Benchmarking synthetic data generation methods for visual classification. Measures how well models trained on AI-generated images transfer to real-world photos in zero-shot and few-shot settings.

**Dataset & Models:** [huggingface.co/datasets/LakshC/SynthBench](https://huggingface.co/datasets/LakshC/SynthBench)

## Approach

1. **Synthetic data generation** — create training images via text-to-image models and programmatic augmentation
2. **Zero-shot evaluation** — train on synthetic data only, test on real photos
3. **Few-shot evaluation** — mix synthetic data with 5/10/25/50 real examples per class
4. **Ablations** — prompt diversity and dataset size experiments

**Classes:** mouse, pen, phone, laptop, water bottle, Rubik's cube

**Model:** ResNet-18 (via `timm`)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Verify PyTorch MPS (Apple Silicon GPU):
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Data Pipeline

### 1. Collect real images

Take photos with iPhone of each object class. Organize into:
```
data/raw/
├── mouse/
├── pen/
├── phone/
├── laptop/
├── water_bottle/
└── rubiks_cube/
```

### 2. Preprocess

Convert HEIC → JPEG, resize to 224x224, split into train/test:
```bash
python scripts/preprocess.py --input data/raw --output data/real --split 0.2
```

Options: `--size` (default 224), `--quality` (default 95), `--seed` (default 42), `--classes`

## Project Structure

```
├── data/
│   ├── raw/              # Raw iPhone photos (HEIC)
│   ├── real/             # Preprocessed real images (train/test)
│   ├── synthetic_t2i/    # Text-to-image generated
│   └── synthetic_aug/    # Programmatic augmentation
├── scripts/
│   ├── preprocess.py     # HEIC conversion + resize
│   ├── generate_t2i.py   # Text-to-image generation
│   ├── generate_aug.py   # Programmatic augmentation
│   ├── train.py          # Model training
│   ├── evaluate.py       # Evaluation + metrics
│   ├── few_shot.py       # Few-shot experiments
│   ├── ablations.py      # Ablation studies
│   └── visualize.py      # Charts and figures
├── results/              # Saved metrics
├── figures/              # Generated charts
└── models/               # Saved checkpoints
```
