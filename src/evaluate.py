"""Evaluation metrics and baseline comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    classes: np.ndarray | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Compute Accuracy, Precision, Recall, F1-Score, and optional Top-K Accuracy."""
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_weighted": round(
            float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "recall_weighted": round(
            float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }

    if y_prob is not None and classes is not None:
        k = min(top_k, len(classes))
        top_k_preds = classes[np.argsort(y_prob, axis=1)[:, -k:]]
        hits = [y_t in top_k_preds[i] for i, y_t in enumerate(y_true)]
        metrics[f"top_{k}_accuracy"] = round(float(np.mean(hits)), 4)

    return metrics

