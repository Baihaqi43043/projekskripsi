"""Buat dataset training Random Forest dari data dummy many-to-many (lokasi × bulan tanam × varietas).

Fitur yang dipakai (hanya informasi yang tersedia SAAT keputusan tanam):
  - 32 vektor temporal LSTM  : dari iklim 12 bulan SEBELUM bulan tanam (encoder pipeline proposal)
  - 9 fitur tanah/lokasi     : tekstur, pH, N, P, K, C-organik, drainase, ketinggian, risiko hama
  - 8 karakteristik varietas : umur panen, potensi hasil, ketahanan hama, kebutuhan air,
                               ketinggian maks, komoditas (one-hot padi/jagung/kedelai)
  - 2 indeks musim bulan tanam (sin/cos) — opsional
Target: skor_kecocokan (0-100) dan hasil_dummy_ton_ha.

TIDAK dipakai sebagai fitur (bocor jawaban): hujan_masa_tanam, suhu_masa_tanam, s_air … s_ketinggian.

Prasyarat: models/proposal_bundle.joblib (dibuat oleh `python train_hybrid_proposal.py`).
Encoder LSTM di bundle dilatih dengan data <= 2023 dan validasi 2024, jadi tidak pernah melihat 2025.

Output (data/processed/rf_varietas/):
  rf_varietas_train.csv  (bulan tanam 2021-2024, 80%)
  rf_varietas_test.csv   (bulan tanam 2025, 20%)
  daftar_fitur.json      (nama kolom fitur, target, dan kunci)

Jalankan: python buat_data_rf_varietas.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from src.preprocessing import CATEGORICAL_MAPPINGS
from src.proposal_inference import ProposalRecommender

SUMBER = "data/raw/dummy_varietas/kecocokan_varietas_semua.csv"
OUT_DIR = "data/processed/rf_varietas"

KUNCI = ["id_lokasi", "nama_lokasi", "tahun", "bulan", "id_varietas", "nama_varietas", "komoditas", "split"]
FITUR_TANAH = ["tekstur_tanah", "ph_tanah", "nitrogen", "fosfor", "kalium", "karbon_organik",
               "drainase", "ketinggian_tempat", "risiko_bencana_hama"]
FITUR_VARIETAS = ["umur_panen_hari_num", "potensi_hasil_num", "ketahanan_hama_skala", "kebutuhan_air_kode",
                  "ketinggian_maks_mdpl", "komoditas_padi", "komoditas_jagung", "komoditas_kedelai"]
FITUR_MUSIM = ["bulan_tanam_sin", "bulan_tanam_cos"]
TARGET = ["skor_kecocokan", "hasil_dummy_ton_ha"]
EVALUASI = ["peringkat", "varietas_terbaik"]
KEBUTUHAN_AIR_KODE = {"Sedikit": 0, "Sedang": 1, "Banyak": 2}


@torch.no_grad()
def vektor_lstm(rec: ProposalRecommender, pasangan: pd.DataFrame) -> pd.DataFrame:
    """Hitung vektor 32 angka untuk tiap (lokasi, bulan tanam) dari iklim 12 bulan sebelumnya."""
    rows = []
    for r in pasangan.itertuples(index=False):
        periode = int(r.tahun) * 12 + int(r.bulan) - 1
        x = torch.from_numpy(rec._encode_window(rec._window(r.id_lokasi, periode))[None])
        emb = rec.model.encode(x).numpy()[0]
        rows.append({"id_lokasi": r.id_lokasi, "tahun": r.tahun, "bulan": r.bulan,
                     **{f"lstm_{i}": float(v) for i, v in enumerate(emb)}})
    return pd.DataFrame(rows)


def main() -> None:
    rec = ProposalRecommender()
    data = pd.read_csv(SUMBER)

    # 1) Vektor LSTM: dihitung sekali per (lokasi, bulan tanam), lalu dipakai bersama oleh 25 varietas
    pasangan = data[["id_lokasi", "tahun", "bulan"]].drop_duplicates()
    emb = vektor_lstm(rec, pasangan)
    lstm_cols = [c for c in emb.columns if c.startswith("lstm_")]

    # 2) Fitur tanah lengkap dari file tanah (kategori diubah ke angka; RF tidak perlu normalisasi)
    tanah = rec.soil.reset_index()[["id_lokasi"] + FITUR_TANAH].copy()
    for col, mapping in CATEGORICAL_MAPPINGS.items():
        tanah[col] = tanah[col].map(mapping).astype(float)

    df = data.drop(columns=[c for c in FITUR_TANAH if c in data.columns])
    df = df.merge(emb, on=["id_lokasi", "tahun", "bulan"], how="left").merge(tanah, on="id_lokasi", how="left")

    # 3) Karakteristik varietas sebagai angka
    df["kebutuhan_air_kode"] = df["kebutuhan_air"].map(KEBUTUHAN_AIR_KODE)
    for k in ("padi", "jagung", "kedelai"):
        df[f"komoditas_{k}"] = (df["komoditas"].str.lower() == k).astype(int)
    df["bulan_tanam_sin"] = np.sin(2 * np.pi * df["bulan"] / 12.0).round(6)
    df["bulan_tanam_cos"] = np.cos(2 * np.pi * df["bulan"] / 12.0).round(6)

    fitur = lstm_cols + FITUR_TANAH + FITUR_VARIETAS + FITUR_MUSIM
    df = df[KUNCI + fitur + TARGET + EVALUASI].sort_values(["tahun", "bulan", "id_lokasi", "id_varietas"])
    df[lstm_cols] = df[lstm_cols].round(6)
    assert not df[fitur + TARGET].isna().any().any(), "Ada nilai kosong di fitur/target"

    os.makedirs(OUT_DIR, exist_ok=True)
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    tr.to_csv(os.path.join(OUT_DIR, "rf_varietas_train.csv"), index=False)
    te.to_csv(os.path.join(OUT_DIR, "rf_varietas_test.csv"), index=False)
    with open(os.path.join(OUT_DIR, "daftar_fitur.json"), "w", encoding="utf-8") as f:
        json.dump({
            "kunci": KUNCI, "fitur_lstm": lstm_cols, "fitur_tanah": FITUR_TANAH,
            "fitur_varietas": FITUR_VARIETAS, "fitur_musim_opsional": FITUR_MUSIM,
            "target": TARGET, "kolom_evaluasi": EVALUASI,
            "tidak_boleh_jadi_fitur": ["hujan_masa_tanam", "suhu_masa_tanam", "s_air", "s_suhu", "s_ph",
                                       "s_drainase", "s_hama", "s_ketinggian"],
            "encoder_lstm": {"bundle": "models/proposal_bundle.joblib", "run_id": rec.b["run_id"],
                             "dilatih": rec.b["created_at"]},
            "kode_kategori": {**{k: v for k, v in CATEGORICAL_MAPPINGS.items()}, "kebutuhan_air": KEBUTUHAN_AIR_KODE},
        }, f, indent=2, ensure_ascii=False)

    print(f"Fitur: {len(fitur)} = {len(lstm_cols)} LSTM + {len(FITUR_TANAH)} tanah + "
          f"{len(FITUR_VARIETAS)} varietas + {len(FITUR_MUSIM)} musim (opsional)")
    print(f"Train: {len(tr)} baris ({tr['tahun'].min()}-{tr['tahun'].max()}) | "
          f"Test: {len(te)} baris ({te['tahun'].min()}-{te['tahun'].max()}) -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
