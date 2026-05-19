"""
evaluate.py — Model evaluation with full visualization
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm

from src.data.dataset import APTOSDataset, get_val_transforms
from src.data.splitter import get_folds
from src.models.model_builder import get_model
from src.utils.metrics import quadratic_weighted_kappa, per_class_report


def evaluate_model(config_path: str, model_weights_path: str, fold: int = 0):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = config['data']['image_size']

    df    = pd.read_csv(config['data']['train_csv'])
    folds = get_folds(df, n_splits=config['training']['n_splits'])
    _, val_df = folds[fold]

    val_dataset = APTOSDataset(val_df, config['data']['images_dir'],
                               transform=get_val_transforms(image_size))
    val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = get_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        freeze_backbone=False
    ).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    # Load bias correction if exists
    bias_path = model_weights_path.replace('.pth', '_bias.pt')
    bias = None
    if os.path.exists(bias_path):
        bias = torch.load(bias_path, map_location=device)
        print(f"   ✅ Loaded bias correction from {bias_path}")
    else:
        print(f"   ℹ️ No bias file found at {bias_path}, proceeding without bias")
    model.eval()

    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images  = images.to(device)
            logits  = model(images)

            # 🔧 APPLY BIAS (if available)
            if bias is not None:
                logits = logits + bias.unsqueeze(0).to(device)

            probs   = torch.softmax(logits, dim=1)
            preds   = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    # ── Text report ──
    per_class_report(all_targets, all_preds)

    # ── Confusion matrix ──
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('Predicted Grade')
    plt.ylabel('True Grade')
    plt.title(f'Confusion Matrix — Fold {fold}  (QWK={quadratic_weighted_kappa(all_targets, all_preds):.4f})')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_fold{fold}.png', dpi=150)
    plt.show()
    print(f"📸 Confusion matrix saved to confusion_matrix_fold{fold}.png")

    # ── ROC curves ──
    all_probs_np = np.array(all_probs)
    plt.figure(figsize=(8, 6))
    for cls in range(5):
        binary_targets = (np.array(all_targets) == cls).astype(int)
        fpr, tpr, _ = roc_curve(binary_targets, all_probs_np[:, cls])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Grade {cls} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'roc_curves_fold{fold}.png', dpi=150)
    plt.show()
    print(f"📸 ROC curves saved to roc_curves_fold{fold}.png")

    return all_preds, all_targets, all_probs


if __name__ == "__main__":
    evaluate_model('config/config.yaml', 'best_model_fold0.pth', fold=0)