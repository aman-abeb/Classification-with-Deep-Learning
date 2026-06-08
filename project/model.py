"""ResNet18 transfer learning model with optional frozen backbone."""

import torch.nn as nn
from torchvision import models


def build_model(
    num_classes: int,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build ResNet18 with ImageNet weights and a replaced classifier head.

    Args:
        num_classes: number of output classes
        freeze_backbone: if True, freeze all layers except the classifier head
        pretrained: if True, load ImageNet weights (training); if False, random init (inference)
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# Notebook alias
build_classifier = build_model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
