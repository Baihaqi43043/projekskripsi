"""E02 — Model cuaca harian (BiLSTM Rahmat) dibandingkan tebakan sederhana.

Data: data/raw/lstm/ (5 lokasi harian) dan results/cuaca_harian/prediksi_production_3bulan_2025.csv.
Nilai aktual memakai angka asli (tidak dipotong).
Jalankan: python -m pipelines.analisis.e02_cuaca_vs_tebakan
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.analisis._util import simpan_keluaran

KODE = "E02_cuaca_harian_vs_tebakan"
F = ["curah_hujan", "suhu_rata_rata", "suhu_maksimal", "suhu_minimum", "kelembaban_rata_rata", "lama_penyinaran"]


def ukur(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return np.mean(np.abs(a - p)), 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)


def main() -> None:
    tr = pd.read_csv("data/raw/lstm/training/training.csv")
    te = pd.read_csv("data/raw/lstm/testing/testing.csv")
    df = pd.concat([tr, te]).sort_values(["id_lokasi", "tanggal"]).reset_index(drop=True)
    df["doy"] = pd.to_datetime(df["tanggal"]).dt.dayofyear
    for c in F:
        df[f"pers_{c}"] = df.groupby("id_lokasi")[c].shift(1)
    clim = {}
    for loc, g in df[df["tahun"] <= 2024].groupby("id_lokasi"):
        tab = g.groupby("doy")[F].mean().reindex(range(1, 367)).interpolate().bfill().ffill()
        tab = pd.concat([tab.tail(15), tab, tab.head(15)]).rolling(15, center=True).mean().iloc[15:-15]
        tab.index = range(1, 367)
        clim[loc] = tab
    for c in F:
        df[f"clim_{c}"] = [clim[lo].loc[d, c] for lo, d in zip(df["id_lokasi"], df["doy"])]
    test = df[df["tahun"] == 2025]

    with simpan_keluaran(KODE):
        prod = pd.read_csv("results/cuaca_harian/prediksi_production_3bulan_2025.csv")
        q = prod.merge(test[["id_lokasi", "tanggal"] + [f"pers_{c}" for c in F] + [f"clim_{c}" for c in F]],
                       on=["id_lokasi", "tanggal"])
        print(f"1) Jan-Mar 2025 ({len(q)} baris), MAE (R²) terhadap nilai aktual asli")
        print(f"{'variabel':22s} | {'BiLSTM':>15s} | {'persistence':>15s} | {'klimatologi':>15s}")
        for c in F:
            a = q[f"aktual_{c}"]
            r = [ukur(a, q[k]) for k in (f"pred_{c}", f"pers_{c}", f"clim_{c}")]
            print(f"{c:22s} | " + " | ".join(f"{m:7.3f} ({r2:+.2f})" for m, r2 in r))
        print("\n2) 'Akurasi WAPE' bila hanya menebak rata-rata data train (konstan), test 2025")
        for c in F:
            a = test[c].to_numpy()
            print(f"   {c:22s} {100 - np.sum(np.abs(a - tr[c].mean())) / np.sum(np.abs(a)) * 100:6.2f}%")
        print("\n3) Tebak hujan / tidak hujan (>= 0,5 mm)")
        r = te.sort_values(["id_lokasi", "tanggal"])
        hujan = r["curah_hujan"] >= 0.5
        prev = r.groupby("id_lokasi")["curah_hujan"].shift(1) >= 0.5
        m = r.groupby("id_lokasi")["curah_hujan"].shift(1).notna()
        a, b = q["aktual_curah_hujan"] >= 0.5, q["pred_curah_hujan"] >= 0.5
        print(f"   proporsi hari hujan 2025           : {hujan.mean():.1%}")
        print(f"   BiLSTM (Jan-Mar 2025)              : {(a == b).mean():.1%}")
        print(f"   selalu tebak 'hujan' (Jan-Mar 2025): {a.mean():.1%}")
        print(f"   besok = hari ini (2025)            : {(prev[m] == hujan[m]).mean():.1%}")


if __name__ == "__main__":
    main()
