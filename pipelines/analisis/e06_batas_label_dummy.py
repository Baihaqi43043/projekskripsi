"""E06 — Kenapa akurasi klasifikasi rendah: batas atas label dummy if-else.

Membandingkan beberapa cara menebak label pada test 2025 memakai model di models/proposal_bundle.joblib.
Catatan: angka di katalog berasal dari bundle 5 lokasi (E05). Bundle saat ini mengikuti run terakhir
train_hybrid_proposal (20 lokasi, E09), sehingga angkanya bisa sedikit berbeda.
Jalankan: python -m pipelines.analisis.e06_batas_label_dummy
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.analisis._util import simpan_keluaran
from src.app.proposal_inference import ProposalRecommender
from src.data.dataset_loader import beri_label_varietas_prototipe

KODE = "E06_batas_label_dummy"


def pita(h: float) -> int:
    return 0 if h < 85 else 1 if h < 140 else 2 if h < 210 else 3


def main() -> None:
    rec = ProposalRecommender()
    soil = pd.read_csv("data/raw/aceh_utara_tanah_statis.csv")
    rows = []
    for loc in rec.b["active_locations"]:
        for m in range(1, 13):
            p = 2025 * 12 + m - 1
            d = rec.recommend(loc, 2025, m)
            fc = pd.DataFrame([{"id_lokasi": loc, **d["prakiraan_lstm"]}])
            prev = rec.climate[(rec.climate["id_lokasi"] == loc) & (rec.climate["periode"] == p - 1)][
                ["id_lokasi", "curah_hujan", "suhu_rata_rata"]]
            label = d["label_prototipe"] or rec.climate[(rec.climate["id_lokasi"] == loc) & (rec.climate["periode"] == p)]["varietas"].iloc[0]
            rows.append({
                "label": label,
                "aturan_ramalan": beri_label_varietas_prototipe(fc, soil)["varietas"].iloc[0],
                "aturan_bulan_lalu": beri_label_varietas_prototipe(prev, soil)["varietas"].iloc[0],
                "hujan_aktual": d["aktual"]["curah_hujan"], "hujan_ramalan": d["prakiraan_lstm"]["curah_hujan"],
            })
    df = pd.DataFrame(rows)
    err = (df["hujan_ramalan"] - df["hujan_aktual"]).abs()
    with simpan_keluaran(KODE):
        print(f"Bundle: run {rec.b['run_id']} | {len(rec.b['active_locations'])} lokasi | {len(df)} lokasi-bulan uji (2025)")
        print("Catatan: Top-1/Top-3 model klasifikasi ada di laporan E05/E09; di sini hanya aturan label.\n")
        print(f"Aturan label diterapkan ke iklim AKTUAL bulan itu : 100% (label memang dibuat dari situ)")
        print(f"Aturan label diterapkan ke iklim BULAN SEBELUMNYA : {np.mean(df.aturan_bulan_lalu == df.label):.0%}")
        print(f"Aturan label diterapkan ke ramalan iklim LSTM     : {np.mean(df.aturan_ramalan == df.label):.0%}")
        print(f"\nError ramalan hujan bulanan: median {err.median():.0f} mm, rata-rata {err.mean():.0f} mm")
        print("Batas kelas pada aturan label: 85 / 140 / 210 mm (lebar kelompok 55-70 mm)")
        tepat = np.mean([pita(a) == pita(b) for a, b in zip(df.hujan_aktual, df.hujan_ramalan)])
        print(f"Pita hujan (4 kelompok) tertebak benar oleh ramalan LSTM: {tepat:.0%}")


if __name__ == "__main__":
    main()
