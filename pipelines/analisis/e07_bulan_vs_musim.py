"""E07 — Stabilitas curah hujan per bulan vs per musim tanam (MT I Okt-Mar, MT II Apr-Sep).

Default 5 lokasi seperti analisis asli; ganti dengan env LOKASI_AKTIF (kosong = semua).
Jalankan: python -m pipelines.analisis.e07_bulan_vs_musim
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pipelines.analisis._util import simpan_keluaran

KODE = "E07_bulan_vs_musim"


def main() -> None:
    lokasi = os.environ.get("LOKASI_AKTIF", "L001,L002,L003,L004,L005")
    c = pd.read_csv("data/raw/aceh_utara_agroklimat_bulanan.csv")
    if lokasi.strip():
        c = c[c["id_lokasi"].isin([x.strip() for x in lokasi.split(",")])]
    c["musim"] = np.where(c["bulan"].isin([10, 11, 12, 1, 2, 3]), "MT1 Okt-Mar", "MT2 Apr-Sep")
    c["tahun_musim"] = np.where(c["bulan"] >= 10, c["tahun"], np.where(c["bulan"] <= 3, c["tahun"] - 1, c["tahun"]))
    s = c.groupby(["id_lokasi", "musim", "tahun_musim"]).agg(n=("bulan", "size"), hujan=("curah_hujan", "sum")).reset_index()
    s = s[s["n"] == 6]
    cv_bulan = c.groupby(["id_lokasi", "bulan"])["curah_hujan"].agg(lambda x: x.std() / x.mean()).mean()
    cv_musim = s.groupby(["id_lokasi", "musim"])["hujan"].agg(lambda x: x.std() / x.mean()).mean()
    with simpan_keluaran(KODE):
        print(f"Lokasi: {sorted(c['id_lokasi'].unique())}")
        print("Rata-rata hujan per bulan (mm):",
              {int(b): int(v) for b, v in c.groupby("bulan")["curah_hujan"].mean().round(0).items()})
        print(f"\nVariasi antar-tahun (koefisien variasi): per BULAN {cv_bulan:.0%} | per MUSIM {cv_musim:.0%}")
        l = s[s["id_lokasi"] == "L001"].pivot(index="tahun_musim", columns="musim", values="hujan").round(0)
        print("\nLhoksukon (L001) - total hujan per musim (mm):\n" + l.to_string())
        jan = c[(c["id_lokasi"] == "L001") & (c["bulan"] == 1)].set_index("tahun")["curah_hujan"].round(0)
        print("\nLhoksukon - hujan bulan Januari saja (mm):", {int(k): float(v) for k, v in jan.items()})
        print(f"\nJumlah sampel per musim: {len(s)}")


if __name__ == "__main__":
    main()
