"""Custom Dataset, Albumentations transforms, and DataLoader creation."""

from pathlib import Path

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from utils import (
    BATCH_SIZE,
    CLASS_NAMES,
    CLASS_TO_IDX,
    DATA_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    SEED,
    TEST_RATIO,
    VAL_RATIO,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


def get_train_transforms() -> A.Compose:
    """Training augmentations with Albumentations."""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms() -> A.Compose:
    """Validation/test transforms (no augmentation)."""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class PotatoDataset(Dataset):
    """Custom PyTorch Dataset for potato disease images."""

    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        """
        Args:
            samples: list of (image_path, label_idx) tuples
            transform: Albumentations transform pipeline
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


def load_samples(data_dir: Path, class_names: list[str]) -> list[tuple[Path, int]]:
    """Collect all (path, label) pairs from class subfolders."""
    samples = []
    for class_name in class_names:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        label = CLASS_TO_IDX[class_name]
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))
    return samples


def split_samples(
    samples: list[tuple[Path, int]],
    seed: int = SEED,
) -> tuple[list, list, list]:
    """Stratified 70/15/15 train/val/test split."""
    labels = [label for _, label in samples]

    train_samples, temp_samples, _, temp_labels = train_test_split(
        samples,
        labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=seed,
        stratify=labels,
    )

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
        random_state=seed,
        stratify=temp_labels,
    )

    return train_samples, val_samples, test_samples


def create_dataloaders(
    data_dir: Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders."""
    all_samples = load_samples(data_dir, CLASS_NAMES)
    train_samples, val_samples, test_samples = split_samples(all_samples, seed=seed)

    train_dataset = PotatoDataset(train_samples, transform=get_train_transforms())
    val_dataset = PotatoDataset(val_samples, transform=get_eval_transforms())
    test_dataset = PotatoDataset(test_samples, transform=get_eval_transforms())

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    print(f"Total images: {len(all_samples)}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
