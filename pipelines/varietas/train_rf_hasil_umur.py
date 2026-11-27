"""Training RF untuk memprediksi HASIL PANEN (ton/ha) dan UMUR PANEN AKTUAL (hari) per varietas.

Data dan fitur sama dengan train_rf_varietas.py (many-to-many: lokasi × bulan tanam × 25 varietas).
Untuk tiap target dibandingkan:
  - Tebakan tanpa model : hasil = potensi hasil × rasio rata-rata train; umur = umur deskripsi varietas
  - Hybrid              : 32 vektor LSTM + tanah + varietas + musim
  - RF saja             : rata-rata iklim 12 bulan + tanah + varietas + musim
  - Tanpa iklim         : tanah + varietas + musim
Tuning per tahun (validasi 2023, 2024), refit di train 2021-2024, uji sekali di 2025, 5 seed.

Jalankan:
  python -m pipelines.varietas.train_rf_hasil_umur                                       # vektor LSTM bulanan (dipakai web)
  python -m pipelines.varietas.train_rf_hasil_umur --data-dir data/processed/rf_varietas_harian --label Harian1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.data.dataset_loader import load_config
from src.utils.experiment_logger import export_git_changelog, log_experiment
from pipelines.varietas.train_rf_varietas import (DATA_DIR, N_ITER, PARAM_DIST, SEEDS, folds_per_tahun, tambah_rata_iklim,
                               tandai_masa_tanam_ekstrem)

TARGETS = {
    "hasil_dummy_ton_ha": {"nama": "Hasil panen", "satuan": "t/ha", "file": "rf_hasil"},
    "umur_panen_aktual_dummy": {"nama": "Umur panen aktual", "satuan": "hari", "file": "rf_umur"},
}


def tebakan_sederhana(target: str, tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    if target == "hasil_dummy_ton_ha":
        rasio = (tr[target] / tr["potensi_hasil_num"]).mean()
        return (te["potensi_hasil_num"] * rasio).to_numpy()
    return te["umur_panen_hari_num"].to_numpy(dtype=float)


def metrik(y, p, mask) -> dict:
    return {
        "MAE": float(mean_absolute_error(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "MAE_normal": float(mean_absolute_error(y[mask], p[mask])), "R2_normal": float(r2_score(y[mask], p[mask])),
    }


def main(data_dir: str, label: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    info = json.load(open(os.path.join(data_dir, "daftar_fitur.json"), encoding="utf-8"))
    tr = pd.read_csv(os.path.join(data_dir, "rf_varietas_train.csv"))
    te = pd.read_csv(os.path.join(data_dir, "rf_varietas_test.csv"))
    tr, rata_cols = tambah_rata_iklim(tr, cfg["data"]["raw_dir"])
    te, _ = tambah_rata_iklim(te, cfg["data"]["raw_dir"])
    normal = ~tandai_masa_tanam_ekstrem(te, cfg["data"]["raw_dir"])
    folds = folds_per_tahun(tr["tahun"].to_numpy())

    dasar = info["fitur_tanah"] + info["fitur_varietas"] + info["fitur_musim_opsional"]
    konfigurasi = {
        "Hybrid (LSTM + tanah + varietas)": info["fitur_lstm"] + dasar,
        "RF saja (rata-rata iklim + tanah + varietas)": rata_cols + dasar,
        "Tanpa iklim (tanah + varietas)": dasar,
    }
    hyb = "Hybrid (LSTM + tanah + varietas)"
    sfx = f"_{label}" if label else ""
    print(f"[RF-Hasil-Umur{sfx}] train={len(tr)} test={len(te)} | encoder LSTM: {info['encoder_lstm']}")

    for target, meta in TARGETS.items():
        y_te = te[target].to_numpy(dtype=float)
        rows = [{"model": "Tebakan tanpa model", "seed": 0, **metrik(y_te, tebakan_sederhana(target, tr, te), normal)}]
        best, imps = None, []
        t0 = time.perf_counter()
        for nama, fitur in konfigurasi.items():
            for seed in SEEDS:
                s = RandomizedSearchCV(RandomForestRegressor(random_state=seed, n_jobs=1), PARAM_DIST, n_iter=N_ITER,
                                       scoring="neg_mean_absolute_error", cv=folds, random_state=seed, refit=True, n_jobs=-1)
                s.fit(tr[fitur], tr[target])
                rows.append({"model": nama, "seed": seed, **metrik(y_te, s.best_estimator_.predict(te[fitur]), normal),
                             "best_params": json.dumps(s.best_params_)})
                if nama == hyb:
                    imps.append(pd.Series(s.best_estimator_.feature_importances_, index=fitur))
                    if seed == SEEDS[0]:
                        best = s.best_estimator_
        durasi = time.perf_counter() - t0
        df = pd.DataFrame(rows)
        ring = df.groupby("model", sort=False)[["MAE", "RMSE", "R2", "MAE_normal", "R2_normal"]].agg(["mean", "std"]).round(3)
        ring.columns = [f"{a}_{b}" for a, b in ring.columns]

        imp = pd.concat(imps, axis=1).mean(axis=1)
        kelompok = {
            "LSTM (32)": float(imp[info["fitur_lstm"]].sum()), "Tanah (9)": float(imp[info["fitur_tanah"]].sum()),
            "Varietas (8)": float(imp[info["fitur_varietas"]].sum()), "Musim (2)": float(imp[info["fitur_musim_opsional"]].sum()),
        }
        sub = df[df["model"] == hyb]
        run_dir = log_experiment(
            stage=f"{meta['file'].upper()}_Hybrid{sfx}", train_samples=len(tr), test_samples=len(te),
            num_classes=int(te["nama_varietas"].nunique()),
            hyperparameters={"target": target, "param_dist": PARAM_DIST, "n_iter": N_ITER, "seeds": SEEDS,
                             "encoder_lstm": info["encoder_lstm"]},
            duration_sec=durasi,
            catatan=(f"{meta['nama']} ({meta['satuan']}); MAE={sub.MAE.mean():.2f}±{sub.MAE.std():.2f}, "
                     f"R2={sub.R2.mean():.3f}, MAE_normal={sub.MAE_normal.mean():.2f}, R2_normal={sub.R2_normal.mean():.3f}"),
        )
        df.to_csv(os.path.join(run_dir, "metrik_per_seed.csv"), index=False)
        ring.to_csv(os.path.join(run_dir, "metrik_ringkasan.csv"))
        joblib.dump({
            "model": best, "fitur": konfigurasi[hyb], "target": target, "satuan": meta["satuan"],
            "encoder_lstm": info["encoder_lstm"], "run_id": os.path.basename(run_dir),
            "metrik_test": {k: round(float(sub[k].mean()), 4) for k in ["MAE", "RMSE", "R2", "MAE_normal", "R2_normal"]},
            "importance_kelompok": kelompok,
        }, f"models/{meta['file']}_hybrid{sfx.lower()}.joblib")

        print(f"\n=== {meta['nama']} ({meta['satuan']}) — test 2025, rata-rata 5 seed ===")
        print(ring[["MAE_mean", "R2_mean", "MAE_normal_mean", "R2_normal_mean"]].to_string())
        print("Kontribusi fitur Hybrid:", {k: round(v, 3) for k, v in kelompok.items()})
    export_git_changelog()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RF prediksi hasil panen & umur panen aktual.")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    main(a.data_dir, a.label)
