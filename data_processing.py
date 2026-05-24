from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "models"
ARTIFACT_DIR.mkdir(exist_ok=True)
SCALER_PATH = ARTIFACT_DIR / "grid_scaler.pkl"
DATA_PATH = ARTIFACT_DIR / "grid_processed_data.pkl"


def _winsorize(frame: pd.DataFrame, low: float = 0.005, high: float = 0.995) -> pd.DataFrame:
    """Clip extreme outliers column-wise using quantile cut-offs."""
    lower = frame.quantile(low)
    upper = frame.quantile(high)
    return frame.clip(lower=lower, upper=upper, axis=1)


def _smooth_sequence(values: np.ndarray) -> np.ndarray:
    """Apply a lightweight 3-point moving average to preserve temporal structure."""
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, values)


def _simple_smote(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """Custom SMOTE-style oversampling to balance classes without external deps."""
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


def preprocess_grid_data(file_path: str = r"C:\My_projects\Smart_grid\data\datasetsmall.csv") -> None:
    print("Step 1: Loading dataset…")
    df = pd.read_csv(file_path)

    if not {"CONS_NO", "FLAG"}.issubset(df.columns):
        raise ValueError("Expected 'CONS_NO' and 'FLAG' columns are missing from the dataset.")

    feature_cols = [col for col in df.columns if col not in {"CONS_NO", "FLAG"}]
    print(f"Detected {len(feature_cols)} temporal features.")

    print("Step 2: Cleaning numeric readings and interpolating missing values…")
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = (
        X.interpolate(method="linear", axis=1, limit_direction="both")
        .bfill(axis=1)
        .ffill(axis=1)
    )
    X = X.fillna(X.median())

    print("Step 3: Removing extreme outliers (winsorization)…")
    X = _winsorize(X)

    print("Step 4: Applying temporal smoothing…")
    X_values = _smooth_sequence(X.values)

    print("Step 5: Standard scaling…")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_values)

    y = df["FLAG"].astype(int).to_numpy()

    print("Step 6: Balancing classes with custom SMOTE…")
    X_balanced, y_balanced = _simple_smote(X_scaled, y, random_state=RANDOM_STATE)

    print("Step 7: Train/Test split (80/20 stratified)…")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced,
        y_balanced,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_balanced,
    )

    time_steps = X_train.shape[1]
    X_train_dl = X_train.reshape(X_train.shape[0], time_steps, 1).astype(np.float32)
    X_test_dl = X_test.reshape(X_test.shape[0], time_steps, 1).astype(np.float32)

    print(f"Saving scaler to {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)

    print(f"Saving processed data to {DATA_PATH}")
    data = {"X_train": X_train_dl, "X_test": X_test_dl, "y_train": y_train, "y_test": y_test}
    joblib.dump(data, DATA_PATH)
    print("✅ Preprocessing Complete.")


if __name__ == "__main__":
    preprocess_grid_data()