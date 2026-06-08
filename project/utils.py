"""Configuration, reproducibility, and plotting utilities."""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "PlantVillage"
SAMPLES_DIR = PROJECT_ROOT / "samples"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

DISPLAY_NAMES = {
    "Potato___Early_blight": "Early Blight",
    "Potato___Late_blight": "Late Blight",
    "Potato___healthy": "Healthy",
}

# Hyperparameters (documented for exam reproducibility)
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
NUM_WORKERS = 0
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pth"
BASELINE_MODEL_PATH = OUTPUT_DIR / "baseline_model.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"
TEST_RESULTS_PATH = OUTPUT_DIR / "test_results.json"
SPLIT_PATH = OUTPUT_DIR / "data_split.json"
CURVES_PATH = OUTPUT_DIR / "training_curves.png"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "error_analysis.png"
BASELINE_COMPARISON_PATH = OUTPUT_DIR / "baseline_comparison.png"
COMPARISON_PATH = OUTPUT_DIR / "baseline_comparison.json"
MISCLASSIFIED_PATH = OUTPUT_DIR / "misclassified_samples.json"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_history(history: dict, path: Path = HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def load_history(path: Path = HISTORY_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def load_test_results(path: Path = TEST_RESULTS_PATH) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_baseline_comparison(path: Path = COMPARISON_PATH) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_training_curves(history: dict, save_path: Path = CURVES_PATH) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["val_acc"], marker="o", color="green")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")
