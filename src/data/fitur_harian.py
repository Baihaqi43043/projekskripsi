"""Fitur turunan dari data HARIAN yang diringkas per bulan (cara B) untuk pipeline eksperimen terpisah.

Dipakai oleh `train_hybrid_harian.py`. Modul ini TIDAK mengubah pipeline proposal utama; ia hanya
menambahkan kolom fitur ke sekuens bulanan hasil `prepare_proposal_dataset`.
Resolusi tetap bulanan (Tabel 3.1 proposal); fitur ini termasuk feature engineering (Bag. 3.10).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FITUR_HARIAN = [
    "hari_hujan",            # jumlah hari hujan >= 1 mm
    "hari_hujan_lebat",      # jumlah hari hujan >= 50 mm
    "hujan_harian_maks",     # curah hujan harian tertinggi (mm)
    "kering_beruntun_maks",  # deret hari kering (< 1 mm) terpanjang dalam bulan
    "hari_panas",            # jumlah hari suhu maksimal >= persentil 90 data train
]


def _deret_terpanjang(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def ringkas_harian_ke_bulanan(daily: pd.DataFrame, ambang_panas: float) -> pd.DataFrame:
    rows = []
    for (loc, th, bl), g in daily.sort_values(["id_lokasi", "tanggal"]).groupby(["id_lokasi", "tahun", "bulan"]):
        hujan = g["curah_hujan"].to_numpy(dtype=float)
        rows.append({
            "id_lokasi": loc, "tahun": th, "bulan": bl,
            "hari_hujan": int((hujan >= 1.0).sum()),
            "hari_hujan_lebat": int((hujan >= 50.0).sum()),
            "hujan_harian_maks": float(hujan.max()),
            "kering_beruntun_maks": _deret_terpanjang(hujan < 1.0),
            "hari_panas": int((g["suhu_maksimal"].to_numpy(dtype=float) >= ambang_panas).sum()),
        })
    return pd.DataFrame(rows)


def tambah_fitur_harian(data: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Tambahkan 5 fitur harian-per-bulan ke setiap langkah sekuens LSTM (X_seq) tanpa mengubah sampel/split."""
    raw_dir = cfg["data"]["raw_dir"]
    train_end = cfg["proposal"]["train_end_year"]
    seq_len = data["preprocessing"]["sequence_length"]
    locs = data["preprocessing"]["locations"]

    daily = pd.read_csv(os.path.join(raw_dir, "aceh_utara_agroklimat_harian.csv"))
    daily = daily[daily["id_lokasi"].isin(locs)]
    ambang_panas = float(daily.loc[daily["tahun"] <= train_end, "suhu_maksimal"].quantile(0.9))
    feats = ringkas_harian_ke_bulanan(daily, ambang_panas)
    feats["periode"] = feats["tahun"] * 12 + feats["bulan"] - 1

    # Z-Score dari bulan periode train saja (tanpa capping IQR: banyak fitur bernilai 0)
    scaler = StandardScaler().fit(feats.loc[feats["tahun"] <= train_end, FITUR_HARIAN])
    feats[FITUR_HARIAN] = scaler.transform(feats[FITUR_HARIAN])
    lookup = {(r.id_lokasi, r.periode): np.array([getattr(r, c) for c in FITUR_HARIAN], dtype=np.float32)
              for r in feats.itertuples()}

    extra = np.zeros((len(data["y"]), seq_len, len(FITUR_HARIAN)), dtype=np.float32)
    for i, (loc, th, bl) in enumerate(zip(data["locs"], data["years"], data["months"])):
        p = int(th) * 12 + int(bl) - 1
        for k, q in enumerate(range(p - seq_len, p)):
            extra[i, k] = lookup.get((loc, q), 0.0)

    out = dict(data)
    out["X_seq"] = np.concatenate([data["X_seq"], extra], axis=2)
    out["input_cols"] = list(data["input_cols"]) + FITUR_HARIAN
    out["preprocessing"] = {
        **data["preprocessing"],
        "input_cols": out["input_cols"],
        "fitur_harian": {"kolom": FITUR_HARIAN, "ambang_hari_panas_C": round(ambang_panas, 2),
                         "zscore_mean": dict(zip(FITUR_HARIAN, np.round(scaler.mean_, 3).tolist())),
                         "zscore_std": dict(zip(FITUR_HARIAN, np.round(scaler.scale_, 3).tolist()))},
    }
    return out
