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
    """Berikan label varietas dummy berbasis kombinasi agroklimat & kondisi tanah.

    Catatan: Ganti fungsi ini dengan label varietas aktual dari Dinas Pertanian / Kementan
    setelah data lapangan tersedia.
    """
    merged = climate_df.merge(soil_df, on="id_lokasi", how="left")
    labels = []
    for _, row in merged.iterrows():
        hujan = row["curah_hujan"]
        suhu = row["suhu_rata_rata"]
        ph = row["ph_tanah"]
        drainase = row["drainase"]
        risiko = row["risiko_bencana_hama"]
        elev = row["ketinggian_tempat"]

        if hujan >= 210:
            if risiko == "Tinggi":
                labels.append("Padi_Inpari_33")
            elif drainase == "Baik" and ph >= 5.5:
                labels.append("Padi_Inpari_32")
            else:
                labels.append("Padi_Ciherang")
        elif 140 <= hujan < 210:
            if elev >= 20 or drainase == "Rusak":
                labels.append("Padi_Inpari_42")
            elif ph >= 5.5:
                labels.append("Padi_Mekongga")
            else:
                labels.append("Padi_IR64")
        elif 85 <= hujan < 140:
            if suhu >= 26.8:
                labels.append("Jagung_BISI_18")
            elif risiko == "Tinggi":
                labels.append("Jagung_Bima_20_URI")
            else:
                labels.append("Jagung_NK_212")
        else:
            if elev >= 20:
                labels.append("Jagung_Lamuru")
            elif drainase == "Baik":
                labels.append("Kedelai_Anjasmoro")
            else:
                labels.append("Kedelai_Grobogan")

    out = climate_df.copy()
    out["varietas"] = labels
    return out


def prepare_shared_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load raw CSVs (or fetch if missing/incomplete) and build shared train/test splits."""
    raw_dir = cfg["data"]["raw_dir"]
    climate_path = os.path.join(raw_dir, "aceh_utara_agroklimat_bulanan.csv")
    soil_path = os.path.join(raw_dir, "aceh_utara_tanah_statis.csv")
    expected_locs = len(cfg["data"]["locations"])

    if os.path.exists(climate_path) and os.path.exists(soil_path):
        climate_df = pd.read_csv(climate_path)
        soil_df = pd.read_csv(soil_path)
        if len(soil_df) == expected_locs:
            print(f"Memuat dataset Aceh Utara ({expected_locs} lokasi) dari data/raw/...")
        else:
            print(f"Jumlah lokasi berubah ({len(soil_df)} -> {expected_locs}), mengambil ulang data...")
            climate_df, soil_df = collect_aceh_utara_sample(cfg)
    else:
        print(f"Mengambil data agroklimat & tanah untuk {expected_locs} lokasi Aceh Utara...")
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
