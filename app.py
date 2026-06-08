"""Streamlit demo — Potato leaf disease classification."""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_DIR = PROJECT_ROOT / "project"
SAMPLES_DIR = PROJECT_ROOT / "samples"
sys.path.insert(0, str(PROJECT_DIR))

from dataset import get_eval_transforms  # noqa: E402
from gradcam import generate_gradcam_explanation  # noqa: E402
from model import build_classifier  # noqa: E402
from utils import (  # noqa: E402
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DATA_DIR,
    DISPLAY_NAMES,
    GRADCAM_DIR,
    IMAGE_EXTENSIONS,
    NUM_CLASSES,
    get_device,
)

ALLOWED_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]


@st.cache_resource
def load_model(model_mtime: float):
    """Load the ResNet18 checkpoint saved by the notebook."""
    device = get_device()
    if not BEST_MODEL_PATH.exists():
        return None, device

    model = build_classifier(NUM_CLASSES, freeze_backbone=True, pretrained=True)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, device


def _samples_dir_mtime() -> float:
    if not SAMPLES_DIR.exists():
        return 0.0
    mtimes = [p.stat().st_mtime for p in SAMPLES_DIR.rglob("*") if p.is_file()]
    return max(mtimes, default=0.0)


@st.cache_data
def get_sample_image_paths(samples_mtime: float) -> list[tuple[str, str]]:
    """All sample images per class from samples/ (preferred) or PlantVillage/."""
    samples: list[tuple[str, str]] = []
    for class_name in CLASS_NAMES:
        for root in (SAMPLES_DIR, DATA_DIR):
            class_dir = root / class_name
            if not class_dir.is_dir():
                continue
            class_images = sorted(
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if class_images:
                samples.extend((class_name, str(p)) for p in class_images)
                break
    return samples


def load_image_rgb(source) -> np.ndarray:
    return np.array(Image.open(source).convert("RGB"))


@torch.inference_mode()
def predict(model, device, image_rgb: np.ndarray):
    tensor = get_eval_transforms()(image=image_rgb)["image"].unsqueeze(0).to(device)
    probs = F.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


def render_prediction(pred_class: str, confidence: float, probs: dict[str, float]) -> None:
    label = DISPLAY_NAMES.get(pred_class, pred_class)
    st.success(f"**Prediction: {label}** — confidence {confidence * 100:.2f}%")

    st.markdown("**Class probabilities**")
    for class_name in sorted(probs, key=probs.get, reverse=True):
        name = DISPLAY_NAMES.get(class_name, class_name)
        st.progress(float(min(max(probs[class_name], 0.0), 1.0)), text=f"{name}: {probs[class_name] * 100:.1f}%")


def render_gradcam(model, device, image_rgb: np.ndarray, pred_class: str) -> None:
    st.subheader("Grad-CAM")
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    pred_idx = CLASS_NAMES.index(pred_class)
    label = DISPLAY_NAMES.get(pred_class, pred_class)
    tensor = get_eval_transforms()(image=image_rgb)["image"].unsqueeze(0).to(device)

    overlay, _ = generate_gradcam_explanation(
        model,
        image_rgb,
        tensor,
        pred_idx,
        label,
        GRADCAM_DIR / f"gradcam_{datetime.now().strftime('%H%M%S')}.png",
    )

    c1, c2 = st.columns(2)
    c1.image(image_rgb, caption="Original", use_container_width=True)
    c2.image(overlay, caption="Grad-CAM overlay", use_container_width=True)


def render_sample_gallery(sample_paths: list[tuple[str, str]]) -> None:
    st.subheader("Sample images")
    for class_name in CLASS_NAMES:
        class_samples = [path for cn, path in sample_paths if cn == class_name]
        if not class_samples:
            continue

        st.markdown(f"**{DISPLAY_NAMES[class_name]}**")
        cols = st.columns(min(len(class_samples), 4))
        for col, path in zip(cols, class_samples):
            with col:
                st.image(path, use_container_width=True)
                if st.button("Use this image", key=f"sample_{path}", use_container_width=True):
                    st.session_state["selected_sample_path"] = path
                    st.rerun()


def main():
    st.set_page_config(page_title="Potato Disease Classifier", page_icon="🥔", layout="wide")

    st.title("Potato Leaf Disease Classification")
    st.markdown(
        "**Advanced Computational Techniques for Big Imaging and Signal Data**  \n"
        "Upload a potato leaf image to classify **Early Blight**, **Late Blight**, or **Healthy** "
        "using ResNet18 transfer learning on the PlantVillage dataset."
    )

    if not BEST_MODEL_PATH.exists():
        st.error(f"Model not found at `{BEST_MODEL_PATH}`. Run the notebook first.")
        st.stop()

    if "selected_sample_path" not in st.session_state:
        st.session_state["selected_sample_path"] = None

    model, device = load_model(BEST_MODEL_PATH.stat().st_mtime)
    sample_paths = get_sample_image_paths(_samples_dir_mtime())

    uploaded = st.file_uploader("Upload a leaf image", type=ALLOWED_TYPES)
    show_gradcam = st.checkbox("Show Grad-CAM explanation", value=True)

    if sample_paths:
        sample_options = {path: f"{DISPLAY_NAMES[class_name]} — {Path(path).name}" for class_name, path in sample_paths}
        selectbox_options = ["— choose from gallery or upload —", *sample_options.keys()]
        default_index = 0
        if st.session_state["selected_sample_path"] in sample_options:
            default_index = selectbox_options.index(st.session_state["selected_sample_path"])

        selected_sample = st.selectbox(
            "Or pick a sample image",
            options=selectbox_options,
            index=default_index,
            format_func=lambda value: sample_options.get(value, value),
        )
        if selected_sample == "— choose from gallery or upload —":
            selected_sample = None
        else:
            st.session_state["selected_sample_path"] = selected_sample

        render_sample_gallery(sample_paths)

    image_rgb: np.ndarray | None = None
    if uploaded is not None:
        image_rgb = load_image_rgb(uploaded)
        st.session_state["selected_sample_path"] = None
    elif st.session_state["selected_sample_path"]:
        image_rgb = load_image_rgb(st.session_state["selected_sample_path"])

    if image_rgb is None:
        st.info("Upload an image, pick from the dropdown, or click **Use this image** on a sample above.")
        return

    try:
        pred_class, confidence, probs = predict(model, device, image_rgb)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    render_prediction(pred_class, confidence, probs)

    if show_gradcam:
        try:
            render_gradcam(model, device, image_rgb, pred_class)
        except Exception as exc:
            st.warning(f"Grad-CAM could not be generated: {exc}")


if __name__ == "__main__":
    main()
