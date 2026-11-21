"""Feature concatenation and Random Forest classifier for hybrid model."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def concatenate_features(temporal_feats: np.ndarray, static_feats: np.ndarray) -> np.ndarray:
    """Concatenate 32-dim LSTM temporal vectors with static soil/location features."""
    return np.hstack([temporal_feats, static_feats])


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int | None = 15,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str | None = "balanced",
    seed: int = 42,
) -> RandomForestClassifier:
    """Train Random Forest classifier on concatenated hybrid features."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf

