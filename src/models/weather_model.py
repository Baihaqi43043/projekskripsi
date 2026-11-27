"""Model BiLSTM + Attention cuaca harian beserta preprocessing-nya (satu sumber untuk training & inference).

Preprocessing di sini identik dengan notebook `notebooks/training_lstm_cuaca.ipynb` (sel 4):
1. Winsorizing: batas p1-p99 dari data train (curah hujan: 0 - p98).
2. log1p pada curah_hujan.
3. Feature engineering: rentang suhu, sin/cos day-of-year, rolling mean 7 hari.
4. One-hot lokasi.
5. Robust scaling (median & IQR dari data train) untuk input dan target.

Semua statistik dari langkah 1 & 5 disimpan ke file JSON di samping bobot model
(`<nama_model>_preproc.json`), sehingga inference memakai angka yang sama persis dengan training.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

FEATURES = [
    "curah_hujan",
    "suhu_rata_rata",
    "suhu_maksimal",
    "suhu_minimum",
    "kelembaban_rata_rata",
    "lama_penyinaran",  # Catatan: NASA POWER ALLSKY_SFC_SW_DWN = radiasi (MJ/m2/hari), bukan jam/hari
]
ENGINEERED_COLS = [
    "rentang_suhu",
    "doy_sin",
    "doy_cos",
    "roll_hujan_7d",
    "roll_lembab_7d",
    "roll_sinar_7d",
]
SCALE_IN_COLS = FEATURES + ENGINEERED_COLS
WINDOW_SIZE = 21

DEFAULT_MODEL_PATH = "models/best_lstm_weather.pth"
DEFAULT_TRAIN_CSV = "data/raw/lstm/training/training.csv"
DEFAULT_TEST_CSV = "data/raw/lstm/testing/testing.csv"


class WeatherLSTM(nn.Module):
    """Bidirectional LSTM + LayerNorm + Temporal Attention -> embedding 32-dim -> regresi 6 parameter cuaca."""

    def __init__(
        self,
        input_dim: int,
        hidden_1: int = 128,
        hidden_2: int = 64,
        embedding_dim: int = 32,
        output_dim: int = 6,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_1, batch_first=True, bidirectional=True)
        self.norm1 = nn.LayerNorm(hidden_1 * 2)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(input_size=hidden_1 * 2, hidden_size=hidden_2, batch_first=True, bidirectional=True)
        self.norm2 = nn.LayerNorm(hidden_2 * 2)
        self.dropout2 = nn.Dropout(dropout)

        self.attn_fc = nn.Linear(hidden_2 * 2, 1)
        self.fc_embed = nn.Linear(hidden_2 * 4, embedding_dim)
        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(embedding_dim, output_dim)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout1(self.norm1(out))
        out, _ = self.lstm2(out)
        out = self.dropout2(self.norm2(out))

        attn_weights = torch.softmax(self.attn_fc(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        last_step = out[:, -1, :]

        combined = torch.cat([last_step, context], dim=1)
        return self.relu(self.fc_embed(combined))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.extract_embedding(x))


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------
def load_daily_csv(path: str) -> pd.DataFrame:
    """Baca CSV harian, urutkan, buang duplikat, dan isi missing value (sesuai notebook sel 4a)."""
    df = pd.read_csv(path)
    df = df.sort_values(["id_lokasi", "tanggal"]).drop_duplicates(subset=["id_lokasi", "tanggal"]).reset_index(drop=True)
    df[FEATURES] = df[FEATURES].interpolate(method="linear").bfill().ffill()
    return df


def add_engineered_features(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    out["rentang_suhu"] = out["suhu_maksimal"] - out["suhu_minimum"]
    doy = pd.to_datetime(out["tanggal"]).dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    out = out.sort_values(["id_lokasi", "tanggal"]).reset_index(drop=True)
    for src_col, dst_col in [
        ("curah_hujan", "roll_hujan_7d"),
        ("kelembaban_rata_rata", "roll_lembab_7d"),
        ("lama_penyinaran", "roll_sinar_7d"),
    ]:
        out[dst_col] = out.groupby("id_lokasi")[src_col].transform(lambda s: s.rolling(7, min_periods=1).mean())
    return out


def _robust_fit(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median & IQR per kolom (setara sklearn RobustScaler default; IQR 0 diganti 1)."""
    median = np.median(arr, axis=0)
    q75, q25 = np.percentile(arr, [75, 25], axis=0)
    iqr = np.where((q75 - q25) == 0, 1.0, q75 - q25)
    return median, iqr


def _clip_and_log(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, (lb, ub) in bounds.items():
        out[col] = out[col].clip(lower=lb, upper=ub)
    out["curah_hujan"] = np.log1p(out["curah_hujan"])
    return out


def fit_preprocessing(df_train: pd.DataFrame, window_size: int = WINDOW_SIZE) -> dict[str, Any]:
    """Hitung semua statistik preprocessing HANYA dari data train."""
    bounds: dict[str, tuple[float, float]] = {}
    for col in FEATURES:
        lb = 0.0 if col == "curah_hujan" else float(df_train[col].quantile(0.01))
        ub = float(df_train[col].quantile(0.98 if col == "curah_hujan" else 0.99))
        bounds[col] = (lb, ub)

    proc = add_engineered_features(_clip_and_log(df_train, bounds))
    in_med, in_iqr = _robust_fit(proc[SCALE_IN_COLS].to_numpy(dtype=np.float32))
    tgt_med, tgt_iqr = _robust_fit(proc[FEATURES].to_numpy(dtype=np.float32))

    return {
        "window_size": window_size,
        "features": list(FEATURES),
        "engineered_cols": list(ENGINEERED_COLS),
        "loc_ids": sorted(df_train["id_lokasi"].unique().tolist()),
        "bounds": bounds,
        "in_median": in_med.astype(np.float32),
        "in_iqr": in_iqr.astype(np.float32),
        "tgt_median": tgt_med.astype(np.float32),
        "tgt_iqr": tgt_iqr.astype(np.float32),
    }


def input_feature_names(prep: dict[str, Any]) -> list[str]:
    return SCALE_IN_COLS + [f"loc_{loc}" for loc in prep["loc_ids"]]


def transform_inputs(df_raw: pd.DataFrame, prep: dict[str, Any]) -> pd.DataFrame:
    """Terapkan preprocessing ke data mentah. Hasil diurutkan per (id_lokasi, tanggal)."""
    proc = add_engineered_features(_clip_and_log(df_raw, prep["bounds"]))
    for loc in prep["loc_ids"]:
        proc[f"loc_{loc}"] = (proc["id_lokasi"] == loc).astype(np.float32)
    proc[SCALE_IN_COLS] = (proc[SCALE_IN_COLS].to_numpy(dtype=np.float32) - prep["in_median"]) / prep["in_iqr"]
    return proc


def transform_targets(df_raw: pd.DataFrame, prep: dict[str, Any]) -> np.ndarray:
    """Target ter-scale (urutan baris sama dengan `transform_inputs`)."""
    proc = add_engineered_features(_clip_and_log(df_raw, prep["bounds"]))
    return (proc[FEATURES].to_numpy(dtype=np.float32) - prep["tgt_median"]) / prep["tgt_iqr"]


def inverse_targets(preds_scaled: np.ndarray, prep: dict[str, Any]) -> np.ndarray:
    """Kembalikan prediksi ke satuan asli (curah hujan: expm1, minimal 0)."""
    out = preds_scaled * prep["tgt_iqr"] + prep["tgt_median"]
    out[:, 0] = np.expm1(np.clip(out[:, 0], 0.0, None))
    return out


def make_sequences(
    df_scaled: pd.DataFrame, targets_scaled: np.ndarray, prep: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sliding window per lokasi: input hari [i, i+W) -> target hari i+W."""
    window = prep["window_size"]
    in_cols = input_feature_names(prep)
    x_list, y_list, loc_list, date_list = [], [], [], []
    tmp = df_scaled.copy()
    tmp["_t_idx"] = np.arange(len(tmp))
    for loc_id, group in tmp.groupby("id_lokasi"):
        group = group.sort_values("tanggal").reset_index(drop=True)
        in_arr = group[in_cols].to_numpy(dtype=np.float32)
        out_arr = targets_scaled[group["_t_idx"].to_numpy()].astype(np.float32)
        dates = group["tanggal"].to_numpy()
        for i in range(len(group) - window):
            x_list.append(in_arr[i : i + window])
            y_list.append(out_arr[i + window])
            loc_list.append(loc_id)
            date_list.append(dates[i + window])
    return (
        np.array(x_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(loc_list),
        np.array(date_list),
    )


# --------------------------------------------------------------------------------------
# Simpan / muat
# --------------------------------------------------------------------------------------
def preproc_path_for(model_path: str) -> str:
    return os.path.splitext(model_path)[0] + "_preproc.json"


def save_preprocessing(prep: dict[str, Any], path: str) -> None:
    serializable = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in prep.items()
    }
    serializable["bounds"] = {k: list(v) for k, v in prep["bounds"].items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def load_preprocessing(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["bounds"] = {k: (float(v[0]), float(v[1])) for k, v in raw["bounds"].items()}
    for key in ("in_median", "in_iqr", "tgt_median", "tgt_iqr"):
        raw[key] = np.asarray(raw[key], dtype=np.float32)
    return raw


def get_preprocessing(model_path: str = DEFAULT_MODEL_PATH, train_csv: str = DEFAULT_TRAIN_CSV) -> dict[str, Any]:
    """Muat statistik preprocessing yang disimpan saat training; fallback hitung ulang dari CSV train."""
    path = preproc_path_for(model_path)
    if os.path.exists(path):
        return load_preprocessing(path)
    warnings.warn(
        f"{path} tidak ditemukan; statistik preprocessing dihitung ulang dari {train_csv}. "
        "Pastikan CSV ini sama dengan data yang dipakai saat melatih model.",
        stacklevel=2,
    )
    return fit_preprocessing(load_daily_csv(train_csv))


def load_weather_model(
    model_path: str, prep: dict[str, Any], device: torch.device | str = "cpu"
) -> WeatherLSTM:
    model = WeatherLSTM(input_dim=len(input_feature_names(prep)), output_dim=len(FEATURES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
