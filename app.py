"""app.py — Professional Streamlit GUI for DR Detection.

Merged features from both app versions:
- Sidebar navigation (Home, Batch Inference, Educational)
- Single-image inference with confidence bars for all 5 grades
- Natural-language explanation (rule-based)
- Grad-CAM visualisation on demand
- Batch inference from CSV (image_path column)
- Export batch results to CSV
- Automatic bias correction loading if present
- Professional dashboard styling and model metadata display
- CPU-friendly and stateless per interaction

Run:
    streamlit run app.py
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="DR Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    .dr-hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(34,193,195,0.10), rgba(253,187,45,0.08));
        border: 1px solid rgba(128,128,128,0.18);
        margin-bottom: 1rem;
    }
    .dr-card {
        padding: 1rem 1rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(255,255,255,0.03);
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        margin-bottom: 0.75rem;
    }
    .dr-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.35rem;
    }
    .small-muted { opacity: 0.75; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CONFIG_PATH = "config/config.yaml"

GRADE_INFO: Dict[int, Tuple[str, str, str, str]] = {
    0: ("No DR", "#27ae60", "No diabetic retinopathy detected. Routine screening is usually enough."),
    1: ("Mild", "#f1c40f", "Mild NPDR. Microaneurysms may appear. Blood sugar control is important."),
    2: ("Moderate", "#e67e22", "Moderate NPDR. More lesions are visible and follow-up is more important."),
    3: ("Severe", "#e74c3c", "Severe NPDR. Referral to an ophthalmologist is recommended."),
    4: ("Proliferative", "#8e44ad", "Proliferative DR. Urgent referral is needed because of high vision-loss risk."),
}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _load_config() -> dict:
    import yaml

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class LoadedModel:
    model: object
    image_size: int
    config: dict
    bias: Optional[object]
    device: object


@st.cache_resource(show_spinner=False)
def load_model_cached(weights_path: str) -> LoadedModel:
    """Load model weights once per Streamlit session."""
    import torch

    from src.models.model_builder import get_model

    config = _load_config()
    model_name = config["model"]["name"]
    num_classes = int(config["model"].get("num_classes", 5))
    image_size = int(config["data"]["image_size"])

    device = torch.device("cpu")

    # Support both older/newer get_model signatures.
    try:
        model = get_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            freeze_backbone=False,
        ).to(device)
    except TypeError:
        model = get_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
        ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    bias_path = weights_path.replace(".pth", "_bias.pt")
    bias = torch.load(bias_path, map_location=device) if os.path.exists(bias_path) else None

    return LoadedModel(model=model, image_size=image_size, config=config, bias=bias, device=device)


def _get_available_weights() -> List[str]:
    files = sorted(glob.glob("*.pth"))
    return files or ["best_model_fold0.pth"]


def _try_get_val_transforms(image_size: int):
    """Use project transform if available; otherwise fall back to a safe default."""
    try:
        from src.data.dataset import get_val_transforms

        return get_val_transforms(image_size)
    except Exception:
        import torchvision.transforms as T

        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def explain_prediction(grade: int, probs: np.ndarray) -> str:
    name, _, icon, desc = GRADE_INFO[grade]
    confidence = float(probs[grade]) * 100

    if grade == 0:
        detail = "The retina looks healthy and does not show clear signs of diabetic retinopathy."
    elif grade == 1:
        detail = "Very early lesions may be present. This stage is often subtle and easy to miss."
    elif grade == 2:
        detail = "Visible abnormalities are more common here. Careful monitoring is important."
    elif grade == 3:
        detail = "This stage can be associated with severe vessel damage and higher risk of progression."
    else:
        detail = "This is the most advanced stage in this dataset and usually needs urgent specialist attention."

    return (
        f"{icon} **Grade {grade} — {name}**  \n\n"
        f"{desc}  \n\n"
        f"**Simple explanation:** {detail}  \n\n"
        f"**Confidence:** {confidence:.1f}%"
    )


def config_model_name() -> str:
    return _load_config()["model"]["name"]


def _model_info_banner(weights_path: str, loaded: LoadedModel):
    model_name = loaded.config["model"]["name"]
    num_classes = loaded.config["model"].get("num_classes", 5)
    image_size = loaded.image_size
    bias_state = "Loaded" if loaded.bias is not None else "Not found"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f"""
        <div class="dr-card">
            <div class="small-muted">Model</div>
            <div style="font-size:1.15rem; font-weight:700;">{model_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"""
        <div class="dr-card">
            <div class="small-muted">Image size</div>
            <div style="font-size:1.15rem; font-weight:700;">{image_size} × {image_size}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"""
        <div class="dr-card">
            <div class="small-muted">Classes</div>
            <div style="font-size:1.15rem; font-weight:700;">{num_classes}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c4.markdown(
        f"""
        <div class="dr-card">
            <div class="small-muted">Bias correction</div>
            <div style="font-size:1.15rem; font-weight:700;">{bias_state}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────


def predict_image(model, image_pil: Image.Image, image_size: int, bias, device) -> Tuple[int, np.ndarray]:
    import torch

    transform = _try_get_val_transforms(image_size)
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        if bias is not None:
            logits = logits + bias.unsqueeze(0).to(device)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    return int(np.argmax(probs)), probs


def _confidence_plot(probs: np.ndarray, predicted_grade: int):
    grade_names = [f"Grade {i} — {GRADE_INFO[i][0]}" for i in range(5)]
    colors = [GRADE_INFO[i][1] if i == predicted_grade else "#c7c7c7" for i in range(5)]

    fig = go.Figure(
        go.Bar(
            x=[p * 100 for p in probs],
            y=grade_names,
            orientation="h",
            marker_color=colors,
            text=[f"{p * 100:.1f}%" for p in probs],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Confidence distribution",
        xaxis_title="Probability (%)",
        yaxis_title="",
        height=340,
        margin=dict(t=50, l=10, r=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 110]),
    )
    return fig


def _render_result_card(predicted_grade: int, probs: np.ndarray):
    name, color, icon, desc = GRADE_INFO[predicted_grade]
    confidence = float(probs[predicted_grade]) * 100

    st.markdown(
        f"""
        <div class="dr-card" style="border-left: 7px solid {color};">
            <div style="font-size: 1.05rem; opacity: 0.8;">Prediction result</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: {color};">{icon} Grade {predicted_grade} — {name}</div>
            <div style="margin-top: 0.35rem;">{desc}</div>
            <div style="margin-top: 0.75rem; font-weight: 700;">Confidence: {confidence:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(float(confidence) / 100.0)
    st.plotly_chart(_confidence_plot(probs, predicted_grade), use_container_width=True)

    with st.expander(" What does this mean?"):
        st.markdown(explain_prediction(predicted_grade, probs))


# ─────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────


def _run_gradcam(image_pil: Image.Image, weights_path: str, predicted_grade: int):
    from visualize_gradcam import compute_gradcam, _get_target_layer, _overlay_heatmap, GRADE_LABELS

    loaded = load_model_cached(weights_path)
    model = loaded.model
    image_size = loaded.image_size
    model_name = loaded.config["model"]["name"]
    transform = _try_get_val_transforms(image_size)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_tensor = transform(image_pil).unsqueeze(0).to(loaded.device)
    target_layer = _get_target_layer(model, model_name)
    heatmap = compute_gradcam(model, input_tensor, predicted_grade, target_layer)

    img_resized = image_pil.resize((image_size, image_size), Image.BILINEAR)
    orig_np = np.array(img_resized).astype(np.float32) / 255.0
    overlay = _overlay_heatmap(orig_np, heatmap, alpha=0.45)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    label = GRADE_LABELS[predicted_grade][0] if isinstance(GRADE_LABELS, dict) else str(predicted_grade)
    fig.suptitle(f"Grad-CAM — Grade {predicted_grade}: {label}", fontsize=13, fontweight="bold")

    axes[0].imshow(img_resized)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Attention heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    fig.savefig(tmp_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return tmp_path


# ─────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────


def home_page():
    st.markdown(
        """
        <div class="dr-hero">
            <h1 style="margin-bottom:0.25rem;">🩺 Diabetic Retinopathy Detection</h1>
            <div class="small-muted">
                Upload a fundus image, get a severity grade, inspect confidence, and generate Grad-CAM.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_weights = _get_available_weights()
    weights_path = st.sidebar.selectbox(
        "Model weights",
        available_weights,
        key="weights_home",
        help="Choose the trained model weights file.",
    )

    loaded = load_model_cached(weights_path)
    _model_info_banner(weights_path, loaded)

    uploaded_file = st.file_uploader(
        " Upload a fundus image",
        type=["png", "jpg", "jpeg"],
        help="APTOS-style fundus photograph.",
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        _show_demo_instructions()
        return

    image_pil = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("📸 Uploaded image")
        st.image(image_pil, use_container_width=True)
        st.caption(f"{image_pil.width} × {image_pil.height} px")

    with right:
        st.subheader("Prediction")
        with st.spinner("Running inference..."):
            predicted_grade, probs = predict_image(
                loaded.model,
                image_pil,
                loaded.image_size,
                loaded.bias,
                loaded.device,
            )
        _render_result_card(predicted_grade, probs)

        st.markdown("### Model output summary")
        summary_cols = st.columns(5)
        for grade in range(5):
            with summary_cols[grade]:
                label, color, icon, _ = GRADE_INFO[grade]
                st.markdown(
                    f"""
                    <div class="dr-card" style="text-align:center; border-top: 4px solid {color};">
                        <div>{icon}</div>
                        <div style="font-weight:700;">{label}</div>
                        <div class="small-muted">{probs[grade]*100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.subheader(" Grad-CAM Explainability")
    st.markdown(
        "Grad-CAM highlights the retinal regions that influenced the prediction. This is useful for checking whether the model focuses on lesions rather than irrelevant background areas."
    )

    if st.button("Generate Grad-CAM", type="primary"):
        with st.spinner("Computing Grad-CAM..."):
            try:
                tmp_path = _run_gradcam(image_pil, weights_path, predicted_grade)
                st.image(tmp_path, caption="Grad-CAM visualisation", use_container_width=True)
                with open(tmp_path, "rb") as f:
                    st.download_button(
                        label=" Download Grad-CAM image",
                        data=f,
                        file_name="gradcam_output.png",
                        mime="image/png",
                    )
            finally:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)


def batch_inference_page():
    st.title("Batch Inference")
    st.markdown(
        "Upload a CSV file that contains an `image_path` column. The app will predict each image and export the results."
    )

    available_weights = _get_available_weights()
    weights_path = st.sidebar.selectbox(
        "Model weights",
        available_weights,
        key="weights_batch",
        help="Choose the trained model weights file.",
    )

    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_csv is None:
        st.info("Upload a CSV file to run batch inference.")
        return

    df = pd.read_csv(uploaded_csv)
    if "image_path" not in df.columns:
        st.error("CSV must contain an `image_path` column.")
        return

    loaded = load_model_cached(weights_path)
    results = []
    success_count = 0
    fail_count = 0

    progress = st.progress(0)
    status = st.empty()

    for i, row in df.iterrows():
        img_path = str(row["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
            pred, probs = predict_image(loaded.model, img, loaded.image_size, loaded.bias, loaded.device)
            result = {
                "image_path": img_path,
                "predicted_grade": pred,
                "predicted_label": GRADE_INFO[pred][0],
                "confidence": float(probs[pred]),
            }
            for g in range(5):
                result[f"prob_grade_{g}"] = float(probs[g])
            results.append(result)
            success_count += 1
        except Exception as exc:
            results.append({"image_path": img_path, "error": str(exc)})
            fail_count += 1

        progress.progress((i + 1) / len(df))
        status.write(f"Processed {i + 1}/{len(df)}")

    result_df = pd.DataFrame(results)

    st.markdown("### Batch summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows", len(df))
    c2.metric("Succeeded", success_count)
    c3.metric("Failed", fail_count)

    st.dataframe(result_df, use_container_width=True)
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download results CSV", csv_bytes, "batch_results.csv", "text/csv")


def educational_page():
    st.title("📖 Educational Overview")
    st.markdown(
        """
        **Diabetic Retinopathy (DR)** is a complication of diabetes that damages the small blood vessels in the retina.
        This can lead to vision loss if it is not detected early.
        """
    )

    st.markdown("### DR stages")
    for grade, (name, color, icon, desc) in GRADE_INFO.items():
        st.markdown(
            f"""
            <div class="dr-card" style="border-left: 7px solid {color};">
                <div style="font-weight:800; font-size:1.05rem;">{icon} Grade {grade}: {name}</div>
                <div style="margin-top:0.3rem;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### How the AI helps")
    st.markdown(
        "The model analyses fundus photographs, estimates the DR severity, and can highlight the image regions that influenced the decision using Grad-CAM."
    )

    st.markdown("### Project performance")
    st.info("Current validation performance: QWK ≈ 0.9057 after bias correction.")

    with st.expander("What the pipeline does"):
        st.markdown(
            """
            1. Load and preprocess a fundus image
            2. Send it through a pretrained deep learning model
            3. Predict the DR severity grade (0–4)
            4. Apply bias correction if available
            5. Show confidence and Grad-CAM for interpretability
            """
        )


# ─────────────────────────────────────────────
# Demo instructions
# ─────────────────────────────────────────────


def _show_demo_instructions():
    with st.expander(" How to use the app"):
        st.markdown(
            """
            **1. Train the model** if needed:
            ```bash
            python main.py
            ```

            **2. Launch the GUI**:
            ```bash
            streamlit run app.py
            ```

            **3. Upload a fundus image** and review the prediction.

            **4. Click Grad-CAM** to visualise the attended regions.
            """
        )


# ─────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Batch Inference", "Educational"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About the model")
    st.sidebar.markdown(
        "APTOS 2019 DR grading with a pretrained CNN, confidence display, bias correction, and explainability."
    )

    if page == " Home":
        home_page()
    elif page == " Batch Inference":
        batch_inference_page()
    else:
        educational_page()


if __name__ == "__main__":
    main()
