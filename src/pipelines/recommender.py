"""Top-K variety recommender and feature importance justification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def recommend_top_k(
    clf: RandomForestClassifier,
    X_sample: np.ndarray,
    label_encoder: LabelEncoder,
    top_k: int = 3,
) -> list[list[dict[str, Any]]]:
    """Return Top-K recommended varieties with suitability probabilities."""
    probs = clf.predict_proba(X_sample)
    classes = label_encoder.inverse_transform(clf.classes_)

    recommendations: list[list[dict[str, Any]]] = []
    for row_probs in probs:
        top_idx = np.argsort(row_probs)[::-1][:top_k]
        recs = [
            {
                "variety": str(classes[idx]),
                "score": round(float(row_probs[idx]) * 100.0, 2),
            }
            for idx in top_idx
        ]
        recommendations.append(recs)
    return recommendations


def get_feature_importance_summary(
    clf: RandomForestClassifier,
    feature_names: list[str],
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Return top-N most influential features for recommendation justification."""
    importances = clf.feature_importances_
    paired = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return [(name, round(float(imp), 4)) for name, imp in paired[:top_n]]

