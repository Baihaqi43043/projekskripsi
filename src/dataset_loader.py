"""Shared dataset loader & preprocessing wrapper for LSTM and Random Forest pipelines."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import yaml

from src.data_fetcher import collect_aceh_utara_sample
from src.preprocessing import build_sequences_and_static


def load_config(config_path: str = "configs/model_config.yaml") -> dict[str, Any]:
    """Load YAML model configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def beri_label_varietas_prototipe(
    climate_df: pd.DataFrame,
    soil_df: pd.DataFrame,
) -> pd.DataFrame:
    """Berikan label varietas prototipe berbasis aturan agroekologi.

    Catatan: Ganti fungsi ini dengan label varietas aktual dari Dinas Pertanian / Kementan
    setelah data lapangan tersedia.
    """
    merged = climate_df.merge(soil_df, on="id_lokasi", how="left")
    labels = []
    for _, row in merged.iterrows():
        hujan = row["curah_hujan"]
        suhu = row["suhu_rata_rata"]
        ph = row["ph_tanah"]

        if hujan >= 200 and ph >= 6.0:
            labels.append("Padi_Inpari_32")
        elif hujan >= 150 and suhu >= 26.5:
            labels.append("Padi_Ciherang")
        elif hujan >= 120 and ph < 6.0:
            labels.append("Padi_Inpari_42")
        elif 80 <= hujan < 150 and suhu >= 26.0:
            labels.append("Jagung_BISI_18")
        elif 60 <= rain_val(hujan) < 120:
            labels.append("Jagung_NK_212")
        else:
            labels.append("Kedelai_Anjasmoro")

    out = climate_df.copy()
    out["varietas"] = labels
    return out


def rain_val(val: float) -> float:
    """Pass-through helper for numeric rainfall comparison."""
    return val


def prepare_shared_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load raw CSVs (or fetch if missing) and build shared train/test splits."""
    raw_dir = cfg["data"]["raw_dir"]
    climate_path = os.path.join(raw_dir, "aceh_utara_agroklimat_bulanan.csv")
    soil_path = os.path.join(raw_dir, "aceh_utara_tanah_statis.csv")

    if os.path.exists(climate_path) and os.path.exists(soil_path):
        print("Memuat dataset Aceh Utara dari data/raw/...")
        climate_df = pd.read_csv(climate_path)
        soil_df = pd.read_csv(soil_path)
    else:
        print("Mengambil data agroklimat (NASA POWER) & tanah (SoilGrids) untuk Aceh Utara...")
        climate_df, soil_df = collect_aceh_utara_sample(cfg)

    climate_df = beri_label_varietas_prototipe(climate_df, soil_df)

    return build_sequences_and_static(
        climate_df=climate_df,
        static_df=soil_df,
        climate_cols=cfg["data"]["climate_features"],
        static_cols=cfg["data"]["static_features"],
        target_col="varietas",
        seq_len=cfg["data"]["sequence_length"],
        train_ratio=cfg["data"]["train_ratio"],
    )

