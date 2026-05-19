"""
metrics.py — Evaluation metrics for DR grading
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
from typing import List


def quadratic_weighted_kappa(y_true: List[int], y_pred: List[int]) -> float:
    """
    Quadratic Weighted Kappa — the official APTOS competition metric.
    Penalizes large disagreements more than small ones (ordinal awareness).

    Range: -1 (worse than random) to 1 (perfect agreement)
    Target: > 0.90 for competitive performance
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def per_class_report(y_true: List[int], y_pred: List[int]):
    """
    Prints a clean per-class breakdown with:
      - Precision, Recall, F1 per DR grade
      - Confusion matrix
      - QWK per adjacent pair (to see where confusion happens)
    """
    grade_names = {
        0: "Grade 0 (No DR)",
        1: "Grade 1 (Mild)",
        2: "Grade 2 (Moderate)",
        3: "Grade 3 (Severe)",
        4: "Grade 4 (Proliferative)"
    }

    print("\n" + "─"*60)
    print("📊 Per-Class Performance Report")
    print("─"*60)
    print(classification_report(
        y_true, y_pred,
        target_names=[grade_names[i] for i in range(5)],
        digits=4,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    print("Confusion Matrix (rows=True, cols=Predicted):")
    header = "       " + "  ".join([f"Pred{i}" for i in range(5)])
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{v:5d}" for v in row])
        print(f"True{i}: {row_str}")

    print("\nQWK overall:", f"{quadratic_weighted_kappa(y_true, y_pred):.4f}")
    print("─"*60 + "\n")
