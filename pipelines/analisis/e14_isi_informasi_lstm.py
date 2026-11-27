"""E14 — Kenapa RF saja mengungguli Hybrid: berapa banyak informasi iklim masa tanam yang dibawa fitur.

Membandingkan rata-rata iklim 12 bulan (6 fitur) dengan vektor LSTM bulanan & harian (32 fitur)
dalam menebak hujan dan suhu SELAMA masa tanam (R² pada 2025).
Prasyarat: data/processed/rf_varietas dan data/processed/rf_varietas_harian1.
Jalankan: python -m pipelines.analisis.e14_isi_informasi_lstm
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from pipelines.analisis._util import simpan_keluaran
from pipelines.varietas.train_rf_varietas import tambah_rata_iklim

KODE = "E14_isi_informasi_lstm"
SUMBER = {"LSTM bulanan": "data/processed/rf_varietas", "LSTM harian 1 epoch": "data/processed/rf_varietas_harian1"}
K = ["id_lokasi", "tahun", "bulan"]


def main() -> None:
    d = pd.read_csv("data/raw/dummy_varietas/kecocokan_varietas_semua.csv")
    d = d[d["bulan_tumbuh"] == 4][K + ["hujan_masa_tanam", "suhu_masa_tanam"]].drop_duplicates()
    vek = {}
    for nama, path in SUMBER.items():
        if not os.path.exists(path):
            continue
        f = pd.concat([pd.read_csv(f"{path}/rf_varietas_train.csv"), pd.read_csv(f"{path}/rf_varietas_test.csv")])
        f = f.drop_duplicates(K)
        vek[nama] = (f[K + [c for c in f.columns if c.startswith("lstm_")]], [c for c in f.columns if c.startswith("lstm_")])
    rata, _ = tambah_rata_iklim(d[K], "data/raw")
    rcols = [c for c in rata.columns if c.startswith("rata12_")]
    base = d.merge(rata, on=K)
    tr, te = (base["tahun"] <= 2024).to_numpy(), (base["tahun"] == 2025).to_numpy()

    def skor(X, y):
        out = []
        for m in (RidgeCV(alphas=np.logspace(-3, 3, 13)), RandomForestRegressor(200, min_samples_leaf=5, random_state=0, n_jobs=-1)):
            m.fit(X[tr], y[tr])
            out.append(r2_score(y[te], m.predict(X[te])))
        return out

    with simpan_keluaran(KODE):
        print("R² menebak iklim MASA TANAM pada 2025 (Ridge | RF)")
        for tgt in ["hujan_masa_tanam", "suhu_masa_tanam"]:
            y = base[tgt].to_numpy()
            print(f"\n  {tgt}")
            print(f"    rata-rata iklim 12 bulan (6 fitur) : {np.round(skor(base[rcols].to_numpy(), y), 3)}")
            for nama, (f, L) in vek.items():
                b2 = base.merge(f, on=K)
                print(f"    {nama:22s} (32 fitur)   : {np.round(skor(b2[L].to_numpy(), y), 3)}")
            clim = base[tr].groupby("bulan")[tgt].mean()
            print(f"    rata-rata bulan kalender           : {r2_score(y[te], base['bulan'][te].map(clim)):.3f}")
        for nama, (f, L) in vek.items():
            X = f[L].to_numpy()
            print(f"\n{nama}: dimensi selalu 0 = {(X.max(0) == 0).sum()}/32 | "
                  f"variasi dijelaskan 3 komponen PCA = {PCA().fit(X).explained_variance_ratio_[:3].sum():.1%}")
        if vek:
            f, L = next(iter(vek.values()))
            print(f"Pola iklim unik: {f.drop_duplicates(subset=L).shape[0]} dari {len(f)} lokasi-bulan")


if __name__ == "__main__":
    main()
