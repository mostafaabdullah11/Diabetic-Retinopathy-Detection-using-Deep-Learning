"""
main.py — Entry point for the DR project.

Provides a clean CLI menu to run:
  1. Training  (delegates to trainer.train_model)
  2. Evaluation (delegates to evaluate.evaluate_model)
  3. Inference  (delegates to inference.predict_image)
  4. Grad-CAM   (delegates to visualize_gradcam.run_gradcam)
  5. GUI        (launches app.py via streamlit)

No training logic lives here — every concern is in its own module.
This file is purely a router.
"""

import sys
import os

# ── Make sure project root is on the path so "src.*" imports work ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH  = "config/config.yaml"
WEIGHTS_PATH = "best_model_fold0.pth"


# ─────────────────────────────────────────────
# Menu
# ─────────────────────────────────────────────

def print_banner():
    print("\n" + "=" * 56)
    print("     Diabetic Retinopathy Detection System")
    print("=" * 56)
    print("  1.  Train model")
    print("  2.  Evaluate model  (confusion matrix + ROC)")
    print("  3.  Predict single image")
    print("  4.  Grad-CAM visualisation")
    print("  5.  Launch GUI  (Streamlit)")
    print("  6.  Exit")
    print("=" * 56)


def ask(prompt: str) -> str:
    return input(prompt).strip()


# ─────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────

def action_train():
    from src.training.trainer import train_model
    train_model(CONFIG_PATH)


def action_evaluate():
    weights = ask(f"Model weights path [{WEIGHTS_PATH}]: ") or WEIGHTS_PATH
    fold_str = ask("Fold number [0]: ") or "0"
    from evaluate import evaluate_model
    evaluate_model(CONFIG_PATH, weights, fold=int(fold_str))


def action_predict():
    img_path = ask("Image path: ")
    weights  = ask(f"Model weights path [{WEIGHTS_PATH}]: ") or WEIGHTS_PATH
    from inference import predict_image
    predict_image(img_path, weights, CONFIG_PATH)


def action_gradcam():
    img_path = ask("Image path: ")
    weights  = ask(f"Model weights path [{WEIGHTS_PATH}]: ") or WEIGHTS_PATH
    out_path = ask("Output image path [gradcam_output.png]: ") or "gradcam_output.png"
    from visualize_gradcam import run_gradcam
    run_gradcam(img_path, weights, CONFIG_PATH, out_path)


def action_gui():
    import subprocess
    print("\n Launching Streamlit GUI...")
    print("   Open http://localhost:8501 in your browser.\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

ACTIONS = {
    "1": action_train,
    "2": action_evaluate,
    "3": action_predict,
    "4": action_gradcam,
    "5": action_gui,
}

if __name__ == "__main__":
    while True:
        print_banner()
        choice = ask("Choice (1-6): ")
        if choice == "6":
            print("Bye!")
            break
        action = ACTIONS.get(choice)
        if action:
            try:
                action()
            except KeyboardInterrupt:
                print("\n[interrupted]")
            except Exception as exc:
                print(f"\n  Error: {exc}")
        else:
            print("Invalid choice — enter 1 to 6.")
