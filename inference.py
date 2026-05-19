"""
inference.py — Single-image DR grade prediction.

Usage (CLI):
    python inference.py path/to/image.png
    python inference.py path/to/image.png --weights best_model_fold0.pth
    python inference.py path/to/image.png --weights best_model_fold0.pth --config config/config.yaml

Usage (Python API):
    from inference import predict_image
    grade, probs = predict_image("image.png", "best_model_fold0.pth", "config/config.yaml")
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
import numpy as np
from PIL import Image

from src.data.dataset import get_val_transforms
from src.models.model_builder import get_model


# ─────────────────────────────────────────────
# Grade metadata
# ─────────────────────────────────────────────

GRADE_LABELS = {
    0: ("No DR",          "No diabetic retinopathy detected."),
    1: ("Mild DR",        "Mild non-proliferative diabetic retinopathy."),
    2: ("Moderate DR",    "Moderate non-proliferative diabetic retinopathy."),
    3: ("Severe DR",      "Severe non-proliferative diabetic retinopathy."),
    4: ("Proliferative",  "Proliferative diabetic retinopathy — immediate referral recommended."),
}

GRADE_URGENCY = {0: "✅", 1: "🟡", 2: "🟠", 3: "🔴", 4: "🆘"}


# ─────────────────────────────────────────────
# Test-Time Augmentation
# ─────────────────────────────────────────────

def predict_with_tta(
    model:        torch.nn.Module,
    image_pil:    "Image.Image",
    image_size:   int,
    device:       torch.device,
    n_augments:   int = 5,
    bias:         torch.Tensor = None,
) -> np.ndarray:
    """
    Test-Time Augmentation: run the same image through N random augmented
    versions and average the softmax probabilities.

    WHY THIS HELPS:
      The model sees a deterministic val-transform image at test time, but
      was trained on randomly augmented images. There is a distribution gap.
      TTA bridges it: by averaging over N augmented versions, we get a more
      stable probability estimate that is less sensitive to any single crop,
      flip, or colour jitter realisation.

      For fundus images specifically, TTA with horizontal flip + slight rotation
      is especially effective because:
        1. Fundus images have no canonical orientation — a flipped version is
           equally valid and the model may be more confident on one orientation.
        2. Slight colour jitter at test time tests whether the model's decision
           is based on lesion structure (good) or image-specific colour artefact (bad).

      Zero training cost. Expected gain: +0.01–0.02 QWK, stronger improvement
      on minority classes because their predictions are most volatile.

    Args:
        model       : loaded model in eval mode
        image_pil   : PIL.Image (original, un-normalised)
        image_size  : from config
        device      : cpu or cuda
        n_augments  : number of augmented passes (5 is fast + effective on CPU)
        bias        : optional per-class logit bias tensor (5,)

    Returns:
        averaged_probs : np.ndarray (5,) — averaged softmax probabilities
    """
    from src.data.dataset import get_val_transforms, get_train_transforms

    val_transform   = get_val_transforms(image_size)
    train_transform = get_train_transforms(image_size)

    all_probs = []

    with torch.no_grad():
        # Pass 1: clean val transform (always include — anchors the average)
        t = val_transform(image_pil).unsqueeze(0).to(device)
        logits = model(t)
        if bias is not None:
            logits = logits + bias.unsqueeze(0).to(device)
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])

        # Passes 2..N: random augmented versions
        for _ in range(n_augments - 1):
            t = train_transform(image_pil).unsqueeze(0).to(device)
            logits = model(t)
            if bias is not None:
                logits = logits + bias.unsqueeze(0).to(device)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])

    return np.mean(all_probs, axis=0)   # (5,) averaged probabilities


def predict_image(
    image_path: str,
    weights_path: str = "best_model_fold0.pth",
    config_path: str  = "config/config.yaml",
    verbose: bool = True,
) -> tuple:
    """
    Run inference on a single fundus image.

    Args:
        image_path   : path to PNG/JPG fundus image
        weights_path : path to trained model .pth file
        config_path  : path to config.yaml
        verbose      : whether to print a human-readable report

    Returns:
        (predicted_grade: int, class_probabilities: np.ndarray of shape [5])

    Raises:
        FileNotFoundError if image or weights file is missing.
    """
    # ── Validate paths ──
    if not os.path.exists(image_path):
        # Try appending common extensions
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = image_path + ext
            if os.path.exists(candidate):
                image_path = candidate
                break
        else:
            raise FileNotFoundError(f"Image not found: {image_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            f"Train the model first with:  python main.py  (option 1)"
        )

    # ── Load config ──
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    image_size  = config["data"]["image_size"]
    model_name  = config["model"]["name"]
    num_classes = config["model"]["num_classes"]

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,       # weights will be loaded from file
        freeze_backbone=False,
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ── Preprocess ──
    # Exact same pipeline as validation — no augmentation
    transform = get_val_transforms(image_size)
    image_pil  = Image.open(image_path).convert("RGB")
    input_tensor = transform(image_pil).unsqueeze(0).to(device)  # (1, C, H, W)

    # ── Load logit bias (QWK-optimal, from find_qwk_optimal_bias) ──
    bias = None
    bias_path = weights_path.replace(".pth", "_bias.pt")
    if os.path.exists(bias_path):
        bias = torch.load(bias_path, map_location=device)
        if verbose:
            print(f"    Loaded QWK-optimal bias from {os.path.basename(bias_path)}")

    # ── Forward pass with TTA ──
    # Averages N augmented versions for more stable minority-class predictions.
    # Falls back to single-pass if TTA is disabled in config.
    use_tta     = config.get("training", {}).get("use_tta", True)
    n_augments  = config.get("training", {}).get("tta_n_augments", 5)

    if use_tta:
        probs = predict_with_tta(
            model, image_pil, image_size, device,
            n_augments=n_augments, bias=bias
        )
    else:
        with torch.no_grad():
            logits = model(input_tensor)
            if bias is not None:
                logits = logits + bias.unsqueeze(0).to(device)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    predicted_grade = int(probs.argmax())

    # ── Print report ──
    if verbose:
        label, description = GRADE_LABELS[predicted_grade]
        urgency = GRADE_URGENCY[predicted_grade]

        print("\n" + "─" * 52)
        print(f"    DR Prediction Report")
        print("─" * 52)
        print(f"  Image  : {os.path.basename(image_path)}")
        print(f"  Result : {urgency}  Grade {predicted_grade} — {label}")
        print(f"  Info   : {description}")
        print()
        print("  Class probabilities:")
        for grade in range(num_classes):
            bar_len   = int(probs[grade] * 30)
            bar       = "" * bar_len + "░" * (30 - bar_len)
            marker    = "" if grade == predicted_grade else " "
            lbl, _    = GRADE_LABELS[grade]
            print(f"  Grade {grade} [{bar}] {probs[grade]*100:5.1f}%  {lbl} {marker}")
        print("─" * 52 + "\n")

    return predicted_grade, probs


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR grade prediction for a single fundus image.")
    parser.add_argument("image",   type=str,                         help="Path to fundus image")
    parser.add_argument("--weights", type=str, default="best_model_fold0.pth", help="Model weights .pth")
    parser.add_argument("--config",  type=str, default="config/config.yaml",   help="config.yaml path")
    args = parser.parse_args()

    predict_image(args.image, args.weights, args.config, verbose=True)
