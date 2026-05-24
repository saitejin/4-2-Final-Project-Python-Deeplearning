"""
Preprocessing utilities for Smart Grid data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def winsorize(frame: pd.DataFrame, low: float = 0.005, high: float = 0.995) -> pd.DataFrame:
    """Clip extreme outliers column-wise using quantile cut-offs."""
    lower = frame.quantile(low)
    upper = frame.quantile(high)
    return frame.clip(lower=lower, upper=upper, axis=1)


def smooth_sequence(values: np.ndarray) -> np.ndarray:
    """Apply a lightweight 3-point moving average to preserve temporal structure."""
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, values)


def simple_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple:
    """Custom SMOTE-style oversampling to balance classes."""
    rng = np.random.default_rng(random_state)
    unique_classes, counts = np.unique(y, return_counts=True)
    target_count = counts.max()

    augmented_features = []
    augmented_labels = []

    for cls, count in zip(unique_classes, counts):
        cls_mask = y == cls
        X_cls = X[cls_mask]
        augmented_features.append(X_cls)
        augmented_labels.append(np.full(X_cls.shape[0], cls))

        deficit = target_count - count
        if deficit <= 0 or X_cls.shape[0] == 0:
            continue

        for _ in range(deficit):
            i = rng.integers(0, X_cls.shape[0])
            j = rng.integers(0, X_cls.shape[0]) if X_cls.shape[0] > 1 else i
            sample = X_cls[i]
            neighbor = X_cls[j]
            lam = rng.random()
            synthetic = sample + lam * (neighbor - sample)
            augmented_features.append(synthetic[np.newaxis, :])
            augmented_labels.append(np.array([cls]))

    X_balanced = np.vstack(augmented_features)
    y_balanced = np.concatenate(augmented_labels)

    shuffled_idx = rng.permutation(len(y_balanced))
    return X_balanced[shuffled_idx], y_balanced[shuffled_idx]
