"""
generate_report.py — Comprehensive evaluation report for the doctor.
Run after training to get QWK, per‑class F1, confusion matrix, ROC curves, bias impact.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from torch.utils.data import DataLoader

from src.data.dataset import APTOSDataset, get_val_transforms
from src.data.splitter import get_folds
from src.models.model_builder import get_model
from src.utils.metrics import quadratic_weighted_kappa

# ---------- Configuration ----------
CONFIG_PATH = "config/config.yaml"
WEIGHTS_PATH = "best_model_fold0.pth"
FOLD = 0

# ---------- Load config ----------
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = config['data']['image_size']

# ---------- Data ----------
df = pd.read_csv(config['data']['train_csv'])
folds = get_folds(df, n_splits=config['training']['n_splits'])
_, val_df = folds[FOLD]

val_dataset = APTOSDataset(val_df, config['data']['images_dir'],
                           transform=get_val_transforms(image_size))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# ---------- Load model ----------
model = get_model(
    model_name=config['model']['name'],
    num_classes=config['model']['num_classes'],
    pretrained=False,
    freeze_backbone=False
).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

# ---------- Load bias ----------
bias = None
bias_path = WEIGHTS_PATH.replace('.pth', '_bias.pt')
if os.path.exists(bias_path):
    bias = torch.load(bias_path, map_location=device)
    print(f"Loaded bias correction from {bias_path}\n")
else:
    print("No bias file – running without bias.\n")

# ---------- Evaluate with bias ----------
all_preds, all_targets, all_probs = [], [], []
with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        logits = model(images)
        if bias is not None:
            logits = logits + bias.unsqueeze(0).to(device)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())
        all_probs.extend(probs.cpu().numpy())

y_true = np.array(all_targets)
y_pred = np.array(all_preds)
y_proba = np.array(all_probs)

# ---------- Metrics ----------
qwk = quadratic_weighted_kappa(y_true, y_pred)
accuracy = (y_true == y_pred).mean()
target_names = ["Grade 0 (No DR)", "Grade 1 (Mild)", "Grade 2 (Moderate)",
                "Grade 3 (Severe)", "Grade 4 (Proliferative)"]
report_dict = classification_report(y_true, y_pred, target_names=target_names,
                                     output_dict=True, zero_division=0)

# ---------- Print report ----------
print("="*70)
print("🩺 DIABETIC RETINOPATHY CLASSIFICATION – FINAL REPORT")
print("="*70)
print(f"Evaluation set size: {len(y_true)} images")
print(f"Overall Accuracy      : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
print()
print("PER-CLASS PERFORMANCE")
print("-"*70)
print(f"{'Class':<22} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>8}")
for name in target_names:
    prec = report_dict[name]['precision']
    rec  = report_dict[name]['recall']
    f1   = report_dict[name]['f1-score']
    sup  = report_dict[name]['support']
    print(f"{name:<22} {prec:10.4f} {rec:10.4f} {f1:10.4f} {sup:8d}")
print("-"*70)
macro_avg = report_dict['macro avg']
weighted_avg = report_dict['weighted avg']
print(f"{'Macro average':<22} {macro_avg['precision']:10.4f} {macro_avg['recall']:10.4f} {macro_avg['f1-score']:10.4f} {len(y_true):8d}")
print(f"{'Weighted average':<22} {weighted_avg['precision']:10.4f} {weighted_avg['recall']:10.4f} {weighted_avg['f1-score']:10.4f} {len(y_true):8d}")
print("="*70)

# ---------- Confusion matrix (heatmap + CSV) ----------
cm = confusion_matrix(y_true, y_pred, labels=range(5))
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(5), yticklabels=range(5))
plt.xlabel('Predicted Grade')
plt.ylabel('True Grade')
plt.title(f'Confusion Matrix (QWK = {qwk:.4f})')
plt.tight_layout()
plt.savefig('confusion_matrix_report.png', dpi=150)
plt.show()
cm_df = pd.DataFrame(cm, index=[f'True_{i}' for i in range(5)],
                     columns=[f'Pred_{i}' for i in range(5)])
cm_df.to_csv('confusion_matrix.csv')
print("Confusion matrix saved as 'confusion_matrix_report.png' and 'confusion_matrix.csv'")

# ---------- ROC curves ----------
plt.figure(figsize=(8,6))
for cls in range(5):
    binary_targets = (y_true == cls).astype(int)
    fpr, tpr, _ = roc_curve(binary_targets, y_proba[:, cls])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Grade {cls} (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (One-vs-Rest)')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curves_report.png', dpi=150)
plt.show()
print("📸 ROC curves saved as 'roc_curves_report.png'")

# ---------- Bias impact ----------
if bias is not None:
    model_no_bias = get_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        freeze_backbone=False
    ).to(device)
    model_no_bias.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model_no_bias.eval()
    preds_no_bias = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            logits = model_no_bias(images)
            preds_no_bias.extend(torch.argmax(logits, dim=1).cpu().numpy())
    qwk_without = quadratic_weighted_kappa(y_true, preds_no_bias)
    print("\nBIAS CORRECTION IMPACT")
    print(f"  QWK without bias: {qwk_without:.4f}")
    print(f"  QWK with bias   : {qwk:.4f}")
    print(f"  Improvement     : +{qwk - qwk_without:.4f}")

# ---------- Save all metrics to text file ----------
with open('evaluation_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("DIABETIC RETINOPATHY CLASSIFICATION – FINAL REPORT\n")
    f.write("="*70 + "\n")
    f.write(f"Evaluation set size: {len(y_true)} images\n")
    f.write(f"Overall Accuracy      : {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}\n\n")
    f.write("PER-CLASS PERFORMANCE\n")
    f.write("-"*70 + "\n")
    f.write(f"{'Class':<22} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>8}\n")
    for name in target_names:
        prec = report_dict[name]['precision']
        rec  = report_dict[name]['recall']
        f1   = report_dict[name]['f1-score']
        sup  = report_dict[name]['support']
        f.write(f"{name:<22} {prec:10.4f} {rec:10.4f} {f1:10.4f} {sup:8d}\n")
    f.write("-"*70 + "\n")
    f.write(f"{'Macro average':<22} {macro_avg['precision']:10.4f} {macro_avg['recall']:10.4f} {macro_avg['f1-score']:10.4f} {len(y_true):8d}\n")
    f.write(f"{'Weighted average':<22} {weighted_avg['precision']:10.4f} {weighted_avg['recall']:10.4f} {weighted_avg['f1-score']:10.4f} {len(y_true):8d}\n")
    f.write("="*70 + "\n")
    if bias is not None:
        f.write(f"QWK without bias: {qwk_without:.4f}\n")
        f.write(f"QWK with bias   : {qwk:.4f}\n")
        f.write(f"Improvement     : +{qwk - qwk_without:.4f}\n")
        f.write("="*70 + "\n")

print("\nAll reports saved:")
print("   - evaluation_report.txt   (full text summary)")
print("   - confusion_matrix.csv    (raw numbers)")
print("   - confusion_matrix_report.png")
print("   - roc_curves_report.png")