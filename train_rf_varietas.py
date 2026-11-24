"""Training Random Forest pada data many-to-many (lokasi × bulan tanam × varietas).

RF regresi memprediksi skor_kecocokan tiap varietas; skor diurutkan per lokasi-bulan menjadi Top-K.
Tiga konfigurasi fitur dibandingkan dengan proses tuning yang sama:
  1. Hybrid      : 32 vektor LSTM + 9 tanah + 8 varietas + 2 musim
  2. RF saja     : rata-rata 6 iklim 12 bulan sebelum tanam (tanpa LSTM) + tanah + varietas + musim
  3. Tanpa iklim : tanah + varietas + musim

Tuning: RandomizedSearchCV, validasi per tahun di data train (latih < 2023 -> uji 2023; latih < 2024 -> uji 2024),
lalu refit di seluruh train 2021-2024. Data test 2025 hanya dipakai sekali. Diulang 5 seed.

Prasyarat: python buat_data_rf_varietas.py  (membuat data/processed/rf_varietas/)
Jalankan  : python train_rf_varietas.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.dataset_loader import load_config
from src.experiment_logger import export_git_changelog, log_experiment

DATA_DIR = "data/processed/rf_varietas"
SEEDS = [42, 43, 44, 45, 46]
N_ITER = 12
TARGET = "skor_kecocokan"
GROUP = ["id_lokasi", "tahun", "bulan"]
PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.3, 0.5],
}
IKLIM = ["curah_hujan", "suhu_rata_rata", "suhu_maksimal", "suhu_minimum", "kelembaban_rata_rata", "lama_penyinaran"]


def tambah_rata_iklim(df: pd.DataFrame, raw_dir: str) -> tuple[pd.DataFrame, list[str]]:
    """Rata-rata tiap variabel iklim pada 12 bulan SEBELUM bulan tanam (fitur pembanding tanpa LSTM)."""
    c = pd.read_csv(os.path.join(raw_dir, "aceh_utara_agroklimat_bulanan.csv")).sort_values(["id_lokasi", "tahun", "bulan"])
    out = []
    for loc, g in c.groupby("id_lokasi"):
        g = g.reset_index(drop=True)
        m = g[IKLIM].rolling(12).mean().shift(1)  # bulan t-12 .. t-1
        m.columns = [f"rata12_{k}" for k in IKLIM]
        out.append(pd.concat([g[["id_lokasi", "tahun", "bulan"]], m], axis=1))
    feats = pd.concat(out)
    cols = [f"rata12_{k}" for k in IKLIM]
    return df.merge(feats, on=GROUP, how="left"), cols


def tandai_masa_tanam_ekstrem(df: pd.DataFrame, raw_dir: str, n_bulan: int = 4) -> pd.Series:
    """True bila masa tanam (bulan t .. t+n-1) memuat hujan bulanan di atas maksimum data train (<= 2024)."""
    c = pd.read_csv(os.path.join(raw_dir, "aceh_utara_agroklimat_bulanan.csv")).sort_values(["id_lokasi", "tahun", "bulan"])
    batas = c.loc[c["tahun"] <= 2024, "curah_hujan"].max()
    flags = []
    for loc, g in c.groupby("id_lokasi"):
        g = g.reset_index(drop=True)
        maks_depan = g["curah_hujan"][::-1].rolling(n_bulan, min_periods=1).max()[::-1]
        flags.append(pd.DataFrame({"id_lokasi": loc, "tahun": g["tahun"], "bulan": g["bulan"], "ekstrem": maks_depan > batas}))
    f = pd.concat(flags)
    return df[GROUP].merge(f, on=GROUP, how="left")["ekstrem"].fillna(False).to_numpy()


def folds_per_tahun(years: np.ndarray, val_years=(2023, 2024)):
    return [(np.where(years < y)[0], np.where(years == y)[0]) for y in val_years]


def metrik_peringkat(df: pd.DataFrame, pred: np.ndarray, k: int = 3) -> dict:
    d = df[GROUP + ["nama_varietas", "komoditas", TARGET]].copy()
    d["pred"] = pred
    top1, hit3, overlap3, rho, kom1 = [], [], [], [], []
    for _, g in d.groupby(GROUP):
        true_rank = g.sort_values(TARGET, ascending=False)
        pred_rank = g.sort_values("pred", ascending=False)
        best = true_rank["nama_varietas"].iloc[0]
        p3 = list(pred_rank["nama_varietas"].iloc[:k])
        t3 = set(true_rank["nama_varietas"].iloc[:k])
        top1.append(p3[0] == best)
        hit3.append(best in p3)
        overlap3.append(len(t3 & set(p3)) / k)
        kom1.append(pred_rank["komoditas"].iloc[0] == true_rank["komoditas"].iloc[0])
        rho.append(spearmanr(g[TARGET], g["pred"]).statistic)
    return {
        "top1_tepat": float(np.mean(top1)),
        "top3_memuat_terbaik": float(np.mean(hit3)),
        "irisan_top3": float(np.mean(overlap3)),
        "komoditas_top1_tepat": float(np.mean(kom1)),
        "spearman_rata2": float(np.nanmean(rho)),
    }


def main(data_dir: str = DATA_DIR, label: str = "") -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    DATA_DIR = data_dir  # noqa: N806 - folder data bisa diganti lewat argumen
    cfg = load_config()
    info = json.load(open(os.path.join(DATA_DIR, "daftar_fitur.json"), encoding="utf-8"))
    tr = pd.read_csv(os.path.join(DATA_DIR, "rf_varietas_train.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, "rf_varietas_test.csv"))
    tr, rata_cols = tambah_rata_iklim(tr, cfg["data"]["raw_dir"])
    te, _ = tambah_rata_iklim(te, cfg["data"]["raw_dir"])

    dasar = info["fitur_tanah"] + info["fitur_varietas"] + info["fitur_musim_opsional"]
    konfigurasi = {
        "Hybrid (LSTM + tanah + varietas)": info["fitur_lstm"] + dasar,
        "RF saja (rata-rata iklim + tanah + varietas)": rata_cols + dasar,
        "Tanpa iklim (tanah + varietas)": dasar,
    }
    folds = folds_per_tahun(tr["tahun"].to_numpy())
    ekstrem = tandai_masa_tanam_ekstrem(te, cfg["data"]["raw_dir"])
    print(f"[RF-Varietas] lokasi={te['id_lokasi'].nunique()} | train={len(tr)} (2021-2024) | test={len(te)} (2025) | "
          f"{te.groupby(GROUP).ngroups} lokasi-bulan uji × {te['nama_varietas'].nunique()} varietas | "
          f"lokasi-bulan uji kena hujan ekstrem: {te[ekstrem].groupby(GROUP).ngroups}")

    rows, importances, best_models = [], {}, {}
    t0 = time.perf_counter()
    for nama, fitur in konfigurasi.items():
        for seed in SEEDS:
            search = RandomizedSearchCV(
                RandomForestRegressor(random_state=seed, n_jobs=1), PARAM_DIST, n_iter=N_ITER,
                scoring="neg_mean_absolute_error", cv=folds, random_state=seed, refit=True, n_jobs=-1,
            )
            search.fit(tr[fitur], tr[TARGET])
            model = search.best_estimator_
            pred = model.predict(te[fitur])
            m = {
                "MAE": float(mean_absolute_error(te[TARGET], pred)),
                "RMSE": float(np.sqrt(mean_squared_error(te[TARGET], pred))),
                "R2": float(r2_score(te[TARGET], pred)),
                **metrik_peringkat(te, pred),
            }
            # Evaluasi tambahan: tanpa bulan tanam yang masa tanamnya kena hujan ekstrem (di luar jangkauan data latih)
            n = ~ekstrem
            m.update({
                "MAE_normal": float(mean_absolute_error(te.loc[n, TARGET], pred[n])),
                "R2_normal": float(r2_score(te.loc[n, TARGET], pred[n])),
                **{f"{k}_normal": v for k, v in metrik_peringkat(te[n], pred[n]).items()},
            })
            rows.append({"model": nama, "seed": seed, "jumlah_fitur": len(fitur), **m,
                         "best_params": json.dumps(search.best_params_)})
            importances.setdefault(nama, []).append(pd.Series(model.feature_importances_, index=fitur))
            if seed == SEEDS[0]:
                best_models[nama] = model
        sub = pd.DataFrame([r for r in rows if r["model"] == nama])
        print(f"  {nama:46s} SEMUA: MAE={sub.MAE.mean():.2f} R2={sub.R2.mean():.3f} Top3={sub.top3_memuat_terbaik.mean():.3f} | "
              f"NORMAL: MAE={sub.MAE_normal.mean():.2f} R2={sub.R2_normal.mean():.3f} "
              f"Top3={sub.top3_memuat_terbaik_normal.mean():.3f} Spearman={sub.spearman_rata2_normal.mean():.3f}")
    durasi = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    kunci = ["MAE", "RMSE", "R2", "top1_tepat", "top3_memuat_terbaik", "irisan_top3", "komoditas_top1_tepat", "spearman_rata2",
             "MAE_normal", "R2_normal", "top1_tepat_normal", "top3_memuat_terbaik_normal", "irisan_top3_normal",
             "komoditas_top1_tepat_normal", "spearman_rata2_normal"]
    ringkas = df.groupby("model", sort=False)[kunci].agg(["mean", "std"]).round(4)
    ringkas.columns = [f"{a}_{b}" for a, b in ringkas.columns]

    # Kontribusi kelompok fitur (rata-rata 5 seed)
    kelompok = []
    for nama, imps in importances.items():
        imp = pd.concat(imps, axis=1).mean(axis=1)
        grup = {
            "LSTM (32)": imp[[c for c in imp.index if c.startswith("lstm_")]].sum(),
            "Rata-rata iklim (6)": imp[[c for c in imp.index if c.startswith("rata12_")]].sum(),
            "Tanah (9)": imp[[c for c in info["fitur_tanah"] if c in imp.index]].sum(),
            "Varietas (8)": imp[[c for c in info["fitur_varietas"] if c in imp.index]].sum(),
            "Musim (2)": imp[[c for c in info["fitur_musim_opsional"] if c in imp.index]].sum(),
        }
        kelompok.append({"model": nama, **{k: round(float(v), 4) for k, v in grup.items()}})
    kelompok = pd.DataFrame(kelompok)

    hyb = "Hybrid (LSTM + tanah + varietas)"
    sub = df[df["model"] == hyb]
    run_dir = log_experiment(
        stage=f"RF_Varietas_Hybrid{'_' + label if label else ''}", train_samples=len(tr), test_samples=len(te),
        num_classes=int(te["nama_varietas"].nunique()),
        hyperparameters={"target": TARGET, "param_dist": PARAM_DIST, "n_iter": N_ITER, "seeds": SEEDS,
                         "encoder_lstm": info["encoder_lstm"]},
        duration_sec=durasi,
        metrics={"accuracy": round(sub.top1_tepat.mean(), 4), "top_3_accuracy": round(sub.top3_memuat_terbaik.mean(), 4)},
        catatan=(f"RF regresi skor kecocokan many-to-many; MAE={sub.MAE.mean():.2f}±{sub.MAE.std():.2f}, "
                 f"R2={sub.R2.mean():.3f}, Top1={sub.top1_tepat.mean():.3f}, Top3={sub.top3_memuat_terbaik.mean():.3f}"),
    )
    df.to_csv(os.path.join(run_dir, "metrik_per_seed.csv"), index=False)
    ringkas.to_csv(os.path.join(run_dir, "metrik_ringkasan.csv"))
    kelompok.to_csv(os.path.join(run_dir, "importance_kelompok.csv"), index=False)

    # Contoh rekomendasi Top-3 model Hybrid (seed pertama) pada data uji
    model = best_models[hyb]
    d = te[GROUP + ["nama_lokasi", "nama_varietas", "komoditas", TARGET]].copy()
    d["skor_prediksi"] = model.predict(te[konfigurasi[hyb]]).round(2)
    rek = []
    for (loc, th, bl), g in d.groupby(GROUP):
        p = g.sort_values("skor_prediksi", ascending=False).head(3)
        t = g.sort_values(TARGET, ascending=False).head(3)
        rek.append({"id_lokasi": loc, "nama_lokasi": g["nama_lokasi"].iloc[0], "tahun": th, "bulan": bl,
                    **{f"prediksi_{i + 1}": f"{r.nama_varietas} ({r.skor_prediksi})" for i, r in enumerate(p.itertuples())},
                    **{f"sebenarnya_{i + 1}": f"{r.nama_varietas} ({r.skor_kecocokan})" for i, r in enumerate(t.itertuples())}})
    pd.DataFrame(rek).to_csv(os.path.join(run_dir, "rekomendasi_top3_test2025.csv"), index=False)

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "fitur": konfigurasi[hyb], "target": TARGET},
                f"models/rf_varietas_hybrid{'_' + label.lower() if label else ''}.joblib")
    export_git_changelog()

    print(f"\n=== HASIL TEST 2025 (rata-rata ± std, {len(SEEDS)} seed) ===")
    print(ringkas.to_string())
    print("\nKontribusi kelompok fitur:")
    print(kelompok.to_string(index=False))
    print(f"\n[LOG] {run_dir}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Training RF many-to-many varietas.")
    ap.add_argument("--data-dir", default=DATA_DIR, help="folder hasil buat_data_rf_varietas.py")
    ap.add_argument("--label", default="", help="akhiran nama stage di log, mis. Harian")
    a = ap.parse_args()
    main(a.data_dir, a.label)
