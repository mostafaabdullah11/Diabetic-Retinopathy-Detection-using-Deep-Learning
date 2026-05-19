"""
visualize_gradcam.py — Grad-CAM heatmap for ResNet-18 (CPU-friendly).

What is Grad-CAM?
    Gradient-weighted Class Activation Mapping highlights the image regions
    that most influenced the model's prediction. For DR grading, this reveals
    whether the model is correctly attending to lesions (microaneurysms,
    haemorrhages, neovascularisation) rather than image artefacts.

How it works (in one paragraph):
    We hook into the last convolutional layer (layer4[-1] for ResNet-18)
    and record: (a) the feature maps A produced by that layer, and (b) the
    gradient of the target class score w.r.t. those feature maps (dScore/dA).
    We global-average-pool the gradients to get per-channel importance weights,
    then take a weighted sum of the feature maps. ReLU removes negative values
    (we only care about what *supports* the prediction). The result is upsampled
    to image size and overlaid as a colourmap.

    This implementation uses manual hooks — no external grad-cam library needed.

Usage (CLI):
    python visualize_gradcam.py image.png
    python visualize_gradcam.py image.png --weights best_model_fold0.pth --out heatmap.png

Usage (Python API):
    from visualize_gradcam import run_gradcam
    run_gradcam("image.png", "best_model_fold0.pth", "config/config.yaml", "out.png")
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on CPU servers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from src.data.dataset import get_val_transforms
from src.models.model_builder import get_model
from inference import GRADE_LABELS, GRADE_URGENCY


# ─────────────────────────────────────────────
# Hook storage
# ─────────────────────────────────────────────

class _GradCamHooks:
    """Stores forward feature maps and backward gradients from one layer."""

    def __init__(self):
        self.feature_maps: torch.Tensor = None   # (1, C, H, W)
        self.gradients:    torch.Tensor = None   # (1, C, H, W)
        self._handles = []

    def register(self, layer: torch.nn.Module):
        self._handles.append(
            layer.register_forward_hook(self._save_features)
        )
        self._handles.append(
            layer.register_full_backward_hook(self._save_grads)
        )

    def remove(self):
        for h in self._handles:
            h.remove()

    def _save_features(self, module, input, output):
        self.feature_maps = output.detach()

    def _save_grads(self, module, grad_input, grad_output):
        # grad_output is a tuple; first element is gradient w.r.t. layer output
        self.gradients = grad_output[0].detach()


# ─────────────────────────────────────────────
# Target layer selector
# ─────────────────────────────────────────────

def _get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """
    Returns the last convolutional block of the model.
    This is where spatial information is richest while semantics are deepest.
    """
    if "resnet" in model_name:
        return model.layer4[-1]              # last BasicBlock of ResNet-18/34
    elif "mobilenet_v3_small" in model_name:
        return model.features[-1]            # last InvertedResidual block
    elif "efficientnet" in model_name:
        return model.features[-1]            # last MBConv block
    else:
        # Generic fallback: last child module that contains Conv2d
        for layer in reversed(list(model.children())):
            if hasattr(layer, "weight"):
                return layer
        raise ValueError(f"Cannot auto-detect target layer for {model_name}. "
                         f"Pass target_layer explicitly.")


# ─────────────────────────────────────────────
# Grad-CAM computation
# ─────────────────────────────────────────────

def compute_gradcam(
    model:        torch.nn.Module,
    input_tensor: torch.Tensor,        # (1, C, H, W)
    target_class: int,
    target_layer: torch.nn.Module,
) -> np.ndarray:
    """
    Computes the Grad-CAM heatmap for target_class.

    Returns:
        heatmap : np.ndarray of shape (H_orig, W_orig) with values in [0, 1].
                  H_orig and W_orig match input_tensor's spatial dims.
    """
    hooks = _GradCamHooks()
    hooks.register(target_layer)

    # ── Forward ──
    model.zero_grad()
    logits = model(input_tensor)       # (1, num_classes)

    # ── Select target class score and backprop ──
    score = logits[0, target_class]
    score.backward()

    # ── Retrieve stored tensors ──
    A   = hooks.feature_maps[0]        # (C, h, w)
    dA  = hooks.gradients[0]           # (C, h, w)
    hooks.remove()

    # ── Global average pooling of gradients → importance weights ──
    weights = dA.mean(dim=(1, 2))      # (C,)  — one weight per channel

    # ── Weighted sum of feature maps ──
    cam = torch.zeros(A.shape[1:], device=A.device)   # (h, w)
    for c_idx, w in enumerate(weights):
        cam += w * A[c_idx]

    # ── ReLU: keep only activations that support this class ──
    cam = F.relu(cam)

    # ── Normalise to [0, 1] ──
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    # ── Upsample to input image size ──
    h_in = input_tensor.shape[2]
    w_in = input_tensor.shape[3]
    cam_upsampled = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),          # (1, 1, h, w)
        size=(h_in, w_in),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()                   # (H, W)

    return cam_upsampled


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

def _overlay_heatmap(
    original_img_np: np.ndarray,   # (H, W, 3) float32 in [0, 1]
    heatmap: np.ndarray,           # (H, W) float32 in [0, 1]
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> np.ndarray:
    """
    Blends a Grad-CAM heatmap onto the original image.

    Args:
        original_img_np : RGB image normalised to [0, 1]
        heatmap         : Grad-CAM map normalised to [0, 1]
        alpha           : heatmap opacity (0=invisible, 1=opaque)
        colormap_name   : matplotlib colormap (jet is standard for Grad-CAM)

    Returns:
        blended : np.ndarray (H, W, 3) uint8
    """
    colormap = cm.get_cmap(colormap_name)
    heatmap_rgb = colormap(heatmap)[:, :, :3]   # (H, W, 3) float64

    blended = (1 - alpha) * original_img_np + alpha * heatmap_rgb
    blended = np.clip(blended, 0.0, 1.0)
    return (blended * 255).astype(np.uint8)


def _save_figure(
    original_pil:  Image.Image,
    heatmap:       np.ndarray,
    overlaid:      np.ndarray,
    predicted_grade: int,
    all_probs:     np.ndarray,
    image_path:    str,
    out_path:      str,
):
    """Saves a 3-panel figure: original | heatmap | overlay."""
    label, description = GRADE_LABELS[predicted_grade]
    urgency = GRADE_URGENCY[predicted_grade]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{urgency}  Predicted: Grade {predicted_grade} — {label}\n{description}",
        fontsize=13, fontweight="bold", y=1.02
    )

    # Panel 1: original
    axes[0].imshow(original_pil)
    axes[0].set_title("Original Image", fontsize=11)
    axes[0].axis("off")

    # Panel 2: raw heatmap
    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap\n(red = high attention)", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: overlay
    axes[2].imshow(overlaid)
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")

    # Probability bar chart inset
    inset = axes[2].inset_axes([0.0, -0.35, 1.0, 0.28])
    colors = ["#2ecc71" if i == predicted_grade else "#bdc3c7" for i in range(5)]
    inset.barh(range(5), all_probs, color=colors, height=0.6)
    inset.set_yticks(range(5))
    inset.set_yticklabels([f"G{i}" for i in range(5)], fontsize=9)
    inset.set_xlim(0, 1)
    inset.set_xlabel("Probability", fontsize=9)
    inset.set_title("Class probabilities", fontsize=9)
    for i, p in enumerate(all_probs):
        inset.text(p + 0.01, i, f"{p*100:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅  Grad-CAM saved to: {out_path}")


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def run_gradcam(
    image_path:   str,
    weights_path: str  = "best_model_fold0.pth",
    config_path:  str  = "config/config.yaml",
    out_path:     str  = "gradcam_output.png",
    target_class: int  = None,    # None → use predicted class
    alpha:        float = 0.45,
) -> np.ndarray:
    """
    Full Grad-CAM pipeline: load → predict → compute heatmap → save figure.

    Args:
        image_path   : path to fundus image
        weights_path : trained model weights .pth
        config_path  : config.yaml path
        out_path     : where to save the output PNG
        target_class : class to visualise (None = predicted class)
        alpha        : heatmap overlay opacity

    Returns:
        heatmap : np.ndarray (H, W) with Grad-CAM activations in [0, 1]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found: {weights_path}. Train the model first."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    image_size  = config["data"]["image_size"]
    model_name  = config["model"]["name"]
    num_classes = config["model"]["num_classes"]

    device = torch.device("cpu")   # Grad-CAM hooks work perfectly on CPU

    # ── Load model ──
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # ── Preprocess ──
    transform    = get_val_transforms(image_size)
    image_pil    = Image.open(image_path).convert("RGB")
    image_resized = image_pil.resize((image_size, image_size), Image.BILINEAR)
    input_tensor = transform(image_pil).unsqueeze(0).to(device)   # (1, 3, H, W)
    input_tensor.requires_grad_(True)    # needed for backward pass

    # ── Predict ──
    with torch.no_grad():
        logits     = model(input_tensor)
        probs      = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_grade = int(probs.argmax())

    vis_class = target_class if target_class is not None else pred_grade
    print(f"🔍  Computing Grad-CAM for Grade {vis_class} "
          f"({'predicted' if target_class is None else 'specified'})...")

    # ── Grad-CAM ──
    target_layer = _get_target_layer(model, model_name)

    # Re-run forward with gradients enabled
    input_tensor = transform(image_pil).unsqueeze(0).to(device)
    heatmap = compute_gradcam(model, input_tensor, vis_class, target_layer)

    # ── Prepare display image ──
    orig_np = np.array(image_resized).astype(np.float32) / 255.0   # (H, W, 3)
    # Resize heatmap to match display image
    heatmap_display = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (image_size, image_size), Image.BILINEAR
        )
    ).astype(np.float32) / 255.0

    overlaid = _overlay_heatmap(orig_np, heatmap_display, alpha=alpha)

    # ── Save ──
    _save_figure(
        original_pil=image_resized,
        heatmap=heatmap_display,
        overlaid=overlaid,
        predicted_grade=pred_grade,
        all_probs=probs,
        image_path=image_path,
        out_path=out_path,
    )

    return heatmap


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM for DR fundus images.")
    parser.add_argument("image",    type=str,                              help="Fundus image path")
    parser.add_argument("--weights", type=str, default="best_model_fold0.pth")
    parser.add_argument("--config",  type=str, default="config/config.yaml")
    parser.add_argument("--out",     type=str, default="gradcam_output.png")
    parser.add_argument("--class",   type=int, default=None,  dest="target_class",
                        help="Class to visualise (default: predicted class)")
    parser.add_argument("--alpha",   type=float, default=0.45, help="Heatmap opacity 0-1")
    args = parser.parse_args()

    run_gradcam(args.image, args.weights, args.config, args.out, args.target_class, args.alpha)
