"""Shared dataset loader & preprocessing wrapper for LSTM and Random Forest pipelines."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.preprocessing import build_sequences_and_static, encode_static_categories


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

    missing = [p for p in (climate_path, soil_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"File data belum ada: {', '.join(missing)}. "
            "Jalankan `python fetch_data.py` (atau `--only soil` untuk data tanah saja) terlebih dahulu."
        )

    climate_df = pd.read_csv(climate_path)
    soil_df = pd.read_csv(soil_path)
    if len(soil_df) != expected_locs:
        raise ValueError(
            f"Data tanah berisi {len(soil_df)} lokasi, sedangkan config berisi {expected_locs} lokasi. "
            "Jalankan ulang `python fetch_data.py` agar data sinkron dengan configs/model_config.yaml."
        )
    if "sumber_tanah" in soil_df.columns and (soil_df["sumber_tanah"] == "DUMMY").any():
        dummy_locs = soil_df.loc[soil_df["sumber_tanah"] == "DUMMY", "id_lokasi"].tolist()
        print(f"[PERINGATAN] Data tanah lokasi {dummy_locs} masih DUMMY penuh (bukan SoilGrids).")
    print(f"Memuat dataset Aceh Utara ({expected_locs} lokasi) dari data/raw/...")

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


def prepare_from_weather_embeddings(
    weather_emb_path: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Gabungkan `lstm_weather_embeddings.npz` (dari notebook LSTM cuaca) dengan data tanah statis & label."""
    raw_dir = cfg["data"]["raw_dir"]
    climate_df = pd.read_csv(os.path.join(raw_dir, "aceh_utara_agroklimat_bulanan.csv"))
    soil_df = pd.read_csv(os.path.join(raw_dir, "aceh_utara_tanah_statis.csv"))

    labeled_monthly = beri_label_varietas_prototipe(climate_df, soil_df)
    static_encoded = encode_static_categories(soil_df)

    bundle = np.load(weather_emb_path, allow_pickle=True)
    emb_train = bundle["emb_train"].astype(np.float32)
    emb_test = bundle["emb_test"].astype(np.float32)

    static_cols = cfg["data"]["static_features"]
    climate_cols = cfg["data"]["climate_features"]

    def _align(locs: np.ndarray, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = pd.to_datetime(dates)
        keys = pd.DataFrame({"id_lokasi": locs, "tahun": dt.year, "bulan": dt.month})
        stat_merged = keys.merge(static_encoded[["id_lokasi"] + static_cols], on="id_lokasi", how="left")
        clim_merged = keys.merge(
            labeled_monthly[["id_lokasi", "tahun", "bulan", "varietas"] + climate_cols],
            on=["id_lokasi", "tahun", "bulan"],
            how="left",
        )
        stat_arr = stat_merged[static_cols].fillna(0.0).to_numpy(dtype=np.float32)
        clim_arr = clim_merged[climate_cols].fillna(0.0).to_numpy(dtype=np.float32)[:, None, :]
        labels = clim_merged["varietas"].fillna("Padi_Ciherang").astype(str).to_numpy()
        return stat_arr, clim_arr, labels

    X_stat_tr, X_seq_tr, y_tr_raw = _align(bundle["locs_train"], bundle["dates_train"])
    X_stat_te, X_seq_te, y_te_raw = _align(bundle["locs_test"], bundle["dates_test"])

    stat_scaler = StandardScaler()
    X_stat_tr = stat_scaler.fit_transform(X_stat_tr)
    X_stat_te = stat_scaler.transform(X_stat_te)

    seq_scaler = StandardScaler()
    X_seq_tr = seq_scaler.fit_transform(X_seq_tr.reshape(-1, len(climate_cols))).reshape(X_seq_tr.shape)
    X_seq_te = seq_scaler.transform(X_seq_te.reshape(-1, len(climate_cols))).reshape(X_seq_te.shape)

    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_tr_raw, y_te_raw]))
    y_tr = label_encoder.transform(y_tr_raw)
    y_te = label_encoder.transform(y_te_raw)

    return {
        "emb_train": emb_train,
        "emb_test": emb_test,
        "X_seq_train": X_seq_tr,
        "X_seq_test": X_seq_te,
        "X_stat_train": X_stat_tr,
        "X_stat_test": X_stat_te,
        "y_train": y_tr,
        "y_test": y_te,
        "classes": label_encoder.classes_,
    }
