"""Preprocessing: encoding kategori, sliding window temporal, scaling, dan chronological split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


CATEGORICAL_MAPPINGS = {
    "tekstur_tanah": {"Pasir": 0.0, "Lanau": 1.0, "Lempung": 2.0},
    "drainase": {"Rusak": 0.0, "Baik": 1.0},
    "risiko_bencana_hama": {"Rendah": 0.0, "Sedang": 1.0, "Tinggi": 2.0},
}


def encode_static_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Ubah kolom kategorikal tanah & lingkungan menjadi angka."""
    out = df.copy()
    for col, mapping in CATEGORICAL_MAPPINGS.items():
        if col in out.columns and not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(mapping).fillna(0.0).astype(float)
    return out


def build_sequences_and_static(
    climate_df: pd.DataFrame,
    static_df: pd.DataFrame,
    climate_cols: list[str],
    static_cols: list[str],
    target_col: str,
    seq_len: int = 12,
    train_ratio: float = 0.8,
) -> dict[str, np.ndarray | LabelEncoder | StandardScaler]:
    """Buat sliding window per lokasi dengan split kronologis (mencegah data leakage)."""
    static_encoded = encode_static_categories(static_df)
    merged = climate_df.merge(static_encoded, on="id_lokasi", how="left")
    merged = merged.sort_values(["id_lokasi", "tahun", "bulan"]).reset_index(drop=True)

    label_encoder = LabelEncoder()
    merged["target_encoded"] = label_encoder.fit_transform(merged[target_col].astype(str))

    X_seq_train, X_stat_train, y_train = [], [], []
    X_seq_test, X_stat_test, y_test = [], [], []

    for _, group in merged.groupby("id_lokasi"):
        group = group.reset_index(drop=True)
        if len(group) <= seq_len:
            continue

        c_vals = group[climate_cols].to_numpy(dtype=np.float32)
        s_vals = group[static_cols].to_numpy(dtype=np.float32)
        y_vals = group["target_encoded"].to_numpy(dtype=np.int64)

        n_windows = len(group) - seq_len
        split_idx = int(n_windows * train_ratio)

        for i in range(n_windows):
            seq = c_vals[i : i + seq_len]
            stat = s_vals[i + seq_len]
            label = y_vals[i + seq_len]
            if i < split_idx:
                X_seq_train.append(seq)
                X_stat_train.append(stat)
                y_train.append(label)
            else:
                X_seq_test.append(seq)
                X_stat_test.append(stat)
                y_test.append(label)

    X_seq_tr = np.array(X_seq_train, dtype=np.float32)
    X_seq_te = np.array(X_seq_test, dtype=np.float32)
    X_stat_tr = np.array(X_stat_train, dtype=np.float32)
    X_stat_te = np.array(X_stat_test, dtype=np.float32)
    y_tr = np.array(y_train, dtype=np.int64)
    y_te = np.array(y_test, dtype=np.int64)

    # Fit scaler HANYA pada data train untuk mencegah kebocoran data
    seq_scaler = StandardScaler()
    n_tr, t_len, n_c = X_seq_tr.shape
    X_seq_tr = seq_scaler.fit_transform(X_seq_tr.reshape(-1, n_c)).reshape(n_tr, t_len, n_c)
    X_seq_te = seq_scaler.transform(X_seq_te.reshape(-1, n_c)).reshape(X_seq_te.shape[0], t_len, n_c)

    stat_scaler = StandardScaler()
    X_stat_tr = stat_scaler.fit_transform(X_stat_tr)
    X_stat_te = stat_scaler.transform(X_stat_te)

    return {
        "X_seq_train": X_seq_tr,
        "X_seq_test": X_seq_te,
        "X_stat_train": X_stat_tr,
        "X_stat_test": X_stat_te,
        "y_train": y_tr,
        "y_test": y_te,
        "label_encoder": label_encoder,
        "seq_scaler": seq_scaler,
        "stat_scaler": stat_scaler,
    }
