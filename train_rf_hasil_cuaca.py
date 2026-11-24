"""Prediksi HASIL PANEN (ton/ha) dengan variabel cuaca biasa (tanpa vektor LSTM).

Fitur = cuaca (lihat src/fitur_cuaca.py) + 9 tanah + 8 karakteristik varietas + 2 musim.
Tiga set cuaca (C1-C3) dibandingkan; set terbaik dipilih dari skor VALIDASI (2023-2024), bukan dari test.
Sebagai acuan ditampilkan juga Hybrid (vektor LSTM) dari models/rf_hasil_hybrid.joblib.

Jalankan: python train_rf_hasil_cuaca.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.dataset_loader import load_config
from src.experiment_logger import export_git_changelog, log_experiment
from src.fitur_cuaca import KUNCI, SET_FITUR, fitur_cuaca
from train_rf_varietas import DATA_DIR, N_ITER, PARAM_DIST, SEEDS, folds_per_tahun, tandai_masa_tanam_ekstrem

TARGET = "hasil_dummy_ton_ha"
MODEL_PATH = "models/rf_hasil_cuaca.joblib"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    raw = cfg["data"]["raw_dir"]
    info = json.load(open(os.path.join(DATA_DIR, "daftar_fitur.json"), encoding="utf-8"))
    climate = pd.read_csv(os.path.join(raw, "aceh_utara_agroklimat_bulanan.csv"))
    fc = fitur_cuaca(climate)
    tr = pd.read_csv(os.path.join(DATA_DIR, "rf_varietas_train.csv")).merge(fc, on=KUNCI, how="left")
    te = pd.read_csv(os.path.join(DATA_DIR, "rf_varietas_test.csv")).merge(fc, on=KUNCI, how="left")
    normal = ~tandai_masa_tanam_ekstrem(te, raw)
    folds = folds_per_tahun(tr["tahun"].to_numpy())
    dasar = info["fitur_tanah"] + info["fitur_varietas"] + info["fitur_musim_opsional"]
    y = te[TARGET].to_numpy(float)

    rows, simpan = [], {}
    t0 = time.perf_counter()
    for nama, cuaca in SET_FITUR.items():
        fitur = cuaca + dasar
        for seed in SEEDS:
            s = RandomizedSearchCV(RandomForestRegressor(random_state=seed, n_jobs=1), PARAM_DIST, n_iter=N_ITER,
                                   scoring="neg_mean_absolute_error", cv=folds, random_state=seed, refit=True, n_jobs=-1)
            s.fit(tr[fitur], tr[TARGET])
            p = s.best_estimator_.predict(te[fitur])
            rows.append({"model": nama, "seed": seed, "MAE_validasi": -s.best_score_,
                         "MAE": mean_absolute_error(y, p), "R2": r2_score(y, p),
                         "MAE_normal": mean_absolute_error(y[normal], p[normal]), "R2_normal": r2_score(y[normal], p[normal]),
                         "best_params": json.dumps(s.best_params_)})
            if seed == SEEDS[0]:
                simpan[nama] = (s.best_estimator_, fitur)
        print(f"  {nama:30s} selesai")
    df = pd.DataFrame(rows)
    ring = df.groupby("model", sort=False)[["MAE_validasi", "MAE", "R2", "MAE_normal", "R2_normal"]].agg(["mean", "std"]).round(3)
    ring.columns = [f"{a}_{b}" for a, b in ring.columns]

    # acuan: Hybrid (vektor LSTM) yang sudah dilatih
    acuan = joblib.load("models/rf_hasil_hybrid.joblib")["metrik_test"] if os.path.exists("models/rf_hasil_hybrid.joblib") else None
    terbaik = ring["MAE_validasi_mean"].idxmin()
    model, fitur = simpan[terbaik]
    sub = df[df["model"] == terbaik]
    run_dir = log_experiment(
        stage="RF_HASIL_Cuaca", train_samples=len(tr), test_samples=len(te), num_classes=int(te["nama_varietas"].nunique()),
        hyperparameters={"target": TARGET, "set_terpilih": terbaik, "fitur_cuaca": SET_FITUR[terbaik],
                         "param_dist": PARAM_DIST, "n_iter": N_ITER, "seeds": SEEDS},
        duration_sec=time.perf_counter() - t0,
        catatan=(f"Hasil panen dengan cuaca biasa ({terbaik}); MAE={sub.MAE.mean():.2f}, R2={sub.R2.mean():.3f}, "
                 f"MAE_normal={sub.MAE_normal.mean():.2f}, R2_normal={sub.R2_normal.mean():.3f}"),
    )
    df.to_csv(os.path.join(run_dir, "metrik_per_seed.csv"), index=False)
    ring.to_csv(os.path.join(run_dir, "metrik_ringkasan.csv"))
    imp = pd.Series(model.feature_importances_, index=fitur)
    kelompok = {"Cuaca": float(imp[SET_FITUR[terbaik]].sum()), "Tanah (9)": float(imp[info["fitur_tanah"]].sum()),
                "Varietas (8)": float(imp[info["fitur_varietas"]].sum()), "Musim (2)": float(imp[info["fitur_musim_opsional"]].sum())}
    joblib.dump({"model": model, "fitur": fitur, "fitur_cuaca": SET_FITUR[terbaik], "set": terbaik, "target": TARGET,
                 "satuan": "t/ha", "run_id": os.path.basename(run_dir), "importance_kelompok": kelompok,
                 "metrik_test": {k: round(float(sub[k].mean()), 4) for k in ["MAE", "R2", "MAE_normal", "R2_normal"]}},
                MODEL_PATH)
    export_git_changelog()

    print("\n=== Prediksi hasil panen (t/ha) — test 2025, rata-rata 5 seed ===")
    print(ring[["MAE_validasi_mean", "MAE_mean", "R2_mean", "MAE_normal_mean", "R2_normal_mean"]].to_string())
    if acuan:
        print(f"Acuan Hybrid (vektor LSTM): MAE={acuan['MAE']:.3f} R2={acuan['R2']:.3f} "
              f"MAE_normal={acuan['MAE_normal']:.3f} R2_normal={acuan['R2_normal']:.3f}")
    print(f"\nDipilih (MAE validasi terkecil): {terbaik} -> {MODEL_PATH}")
    print("Kontribusi fitur:", {k: round(v, 3) for k, v in kelompok.items()})
    print(f"[LOG] {run_dir}")


if __name__ == "__main__":
    main()
