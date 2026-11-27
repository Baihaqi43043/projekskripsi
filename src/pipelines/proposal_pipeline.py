"""Persiapan data pipeline Hybrid LSTM-RF yang mengikuti proposal (Bab III).

Pemetaan ke proposal:
- Tabel 3.1  : 6 variabel agroklimat BULANAN (curah hujan, suhu rata-rata/maks/min, kelembaban, penyinaran).
- Tabel 3.2  : 9 fitur statis tanah & lingkungan.
- Bag. 3.10  : preprocessing = interpolasi missing value, penanganan outlier IQR, normalisasi Z-Score,
               feature engineering indeks musim; sliding window sekuens untuk LSTM.
- Bag. 3.7   : 80% data pelatihan / 20% pengujian (kronologis).
- Bag. 3.11  : data pelatihan dipecah lagi menjadi training & validation.

Satu sampel = (lokasi, bulan target t). Input LSTM = iklim bulan t-L .. t-1 (L = sequence_length),
sehingga tidak ada informasi bulan target yang masuk ke input.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data.dataset_loader import beri_label_varietas_prototipe, filter_active_locations
from src.data.preprocessing import encode_static_categories


def _interpolate_per_location(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.sort_values(["id_lokasi", "tahun", "bulan"]).reset_index(drop=True)
    out[cols] = out.groupby("id_lokasi")[cols].transform(lambda s: s.interpolate(method="linear").bfill().ffill())
    return out


def prepare_proposal_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    raw_dir = cfg["data"]["raw_dir"]
    pcfg = cfg["proposal"]
    climate_cols: list[str] = cfg["data"]["climate_features"]
    static_cols: list[str] = cfg["data"]["static_features"]
    seq_len: int = cfg["data"]["sequence_length"]
    train_end, val_year, test_year = pcfg["train_end_year"], pcfg["val_year"], pcfg["test_year"]

    climate = pd.read_csv(os.path.join(raw_dir, "aceh_utara_agroklimat_bulanan.csv"))
    soil = pd.read_csv(os.path.join(raw_dir, "aceh_utara_tanah_statis.csv"))
    climate, soil = filter_active_locations(climate, soil, cfg)

    # 1) Cleaning: interpolasi missing value per lokasi
    climate = _interpolate_per_location(climate, climate_cols)

    # Label (prototipe) dihitung dari data iklim ASLI bulan target
    climate["varietas"] = beri_label_varietas_prototipe(climate, soil)["varietas"].to_numpy()

    # 2) Outlier IQR + 3) Z-Score -> statistik HANYA dari bulan-bulan periode train (<= train_end)
    fit_rows = climate["tahun"] <= train_end
    q1 = climate.loc[fit_rows, climate_cols].quantile(0.25)
    q3 = climate.loc[fit_rows, climate_cols].quantile(0.75)
    iqr = q3 - q1
    k = float(pcfg.get("iqr_factor", 1.5))
    lower, upper = q1 - k * iqr, q3 + k * iqr
    capped = climate[climate_cols].clip(lower=lower, upper=upper, axis=1)
    n_outliers = int(((climate[climate_cols] < lower) | (climate[climate_cols] > upper)).sum().sum())

    clim_scaler = StandardScaler().fit(capped.loc[fit_rows])
    scaled = pd.DataFrame(clim_scaler.transform(capped), columns=climate_cols, index=climate.index)

    # 4) Feature engineering: indeks musim (sin/cos bulan)
    input_cols = list(climate_cols)
    if pcfg.get("seasonal_features", True):
        scaled["bulan_sin"] = np.sin(2 * np.pi * climate["bulan"] / 12.0)
        scaled["bulan_cos"] = np.cos(2 * np.pi * climate["bulan"] / 12.0)
        input_cols += ["bulan_sin", "bulan_cos"]

    static_enc = encode_static_categories(soil).set_index("id_lokasi")

    X_seq, y_fc, X_stat, labels, years, locs, months = [], [], [], [], [], [], []
    for loc_id, grp in climate.groupby("id_lokasi", sort=True):
        idx = grp.index.to_numpy()
        in_arr = scaled.loc[idx, input_cols].to_numpy(dtype=np.float32)
        tgt_arr = scaled.loc[idx, climate_cols].to_numpy(dtype=np.float32)
        stat = static_enc.loc[loc_id, static_cols].to_numpy(dtype=np.float32)
        for t in range(seq_len, len(idx)):
            X_seq.append(in_arr[t - seq_len : t])
            y_fc.append(tgt_arr[t])  # target peramalan LSTM: iklim bulan t
            X_stat.append(stat)
            labels.append(grp.loc[idx[t], "varietas"])
            years.append(int(grp.loc[idx[t], "tahun"]))
            months.append(int(grp.loc[idx[t], "bulan"]))
            locs.append(loc_id)

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_fc = np.asarray(y_fc, dtype=np.float32)
    X_stat = np.asarray(X_stat, dtype=np.float32)
    years = np.asarray(years)
    label_encoder = LabelEncoder().fit(labels)
    y = label_encoder.transform(labels)

    train_mask = years <= train_end
    val_mask = (years > train_end) & (years <= val_year)
    test_mask = years == test_year

    stat_scaler = StandardScaler().fit(X_stat[train_mask])
    X_stat = stat_scaler.transform(X_stat).astype(np.float32)

    return {
        "X_seq": X_seq,
        "y_forecast": y_fc,
        "X_stat": X_stat,
        "y": y,
        "years": years,
        "months": np.asarray(months),
        "locs": np.asarray(locs),
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "label_encoder": label_encoder,
        "soil_raw": soil.set_index("id_lokasi").to_dict("index"),
        # objek preprocessing (dipakai untuk menyimpan bundle inference aplikasi)
        "iqr_lower": lower,
        "iqr_upper": upper,
        "clim_scaler": clim_scaler,
        "stat_scaler": stat_scaler,
        "climate_cols": climate_cols,
        "input_cols": input_cols,
        "static_cols": static_cols,
        "preprocessing": {
            "iqr_lower": lower.round(4).to_dict(),
            "iqr_upper": upper.round(4).to_dict(),
            "n_outliers_capped": n_outliers,
            "zscore_mean": dict(zip(climate_cols, np.round(clim_scaler.mean_, 4).tolist())),
            "zscore_std": dict(zip(climate_cols, np.round(clim_scaler.scale_, 4).tolist())),
            "sequence_length": seq_len,
            "input_cols": input_cols,
            "locations": sorted(set(locs)),
        },
    }


def temporal_cv_folds(years: np.ndarray, val_years: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fold expanding-window: latih pada tahun < y, validasi pada tahun == y."""
    folds = []
    for y in val_years:
        tr = np.where(years < y)[0]
        va = np.where(years == y)[0]
        if len(tr) and len(va):
            folds.append((tr, va))
    return folds
