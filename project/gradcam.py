"""Grad-CAM explainability for ResNet18 potato disease classifier."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for ResNet18.

    Hooks the last convolutional block (layer4) to produce a class-discriminative
    heatmap showing which image regions influenced the prediction.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list = []

        self._handles.append(
            self.target_layer.register_forward_hook(self._forward_hook)
        )
        self._handles.append(
            self.target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given class index.

        Returns:
            Normalized heatmap as a float numpy array in [0, 1], shape (H, W).
        """
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        output = self.model(input_tensor)
        score = output[:, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze(0)
        cam = F.relu(cam)

        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def get_resnet18_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last residual block of ResNet18 (standard Grad-CAM target)."""
    return model.layer4[-1]


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on the original RGB image."""
    h, w = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_colored + (1 - alpha) * image_rgb).astype(np.uint8)
    return overlay


def save_gradcam_visualization(
    image_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    save_path: Path,
    title: str = "Grad-CAM Explanation",
) -> Path:
    """Save side-by-side original and Grad-CAM overlay visualization."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title("Grad-CAM Overlay")
    axes[1].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path


def generate_gradcam_explanation(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    image_tensor: torch.Tensor,
    target_class: int,
    class_name: str,
    save_path: Path,
) -> tuple[np.ndarray, Path]:
    """
    Full Grad-CAM pipeline: heatmap, overlay, and saved side-by-side figure.

    Args:
        model: trained ResNet18 model (eval mode)
        image_rgb: original image (H, W, 3) uint8
        image_tensor: preprocessed tensor (1, 3, H, W) on correct device
        target_class: class index to explain
        class_name: human-readable class label for the figure title
        save_path: where to save the visualization PNG

    Returns:
        overlay_rgb array and path to saved figure
    """
    device = next(model.parameters()).device
    target_layer = get_resnet18_target_layer(model)
    grad_cam = GradCAM(model, target_layer)

    try:
        was_training = model.training
        model.eval()
        with torch.enable_grad():
            tensor = image_tensor.clone().detach().to(device)
            tensor.requires_grad_(True)
            heatmap = grad_cam.generate(tensor, target_class)
        if was_training:
            model.train()
    finally:
        grad_cam.remove_hooks()

    overlay_rgb = overlay_heatmap(image_rgb, heatmap)
    saved_path = save_gradcam_visualization(
        image_rgb,
        overlay_rgb,
        save_path,
        title=f"Grad-CAM — {class_name}",
    )

    return overlay_rgb, saved_path
