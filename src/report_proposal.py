"""Artefak & laporan hasil (draf Bab IV) untuk pipeline Hybrid sesuai proposal."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

NAMA_BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
SATUAN_STATIS = {"ketinggian_tempat": " mdpl", "karbon_organik": "%"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_loss_curve(history: list[dict[str, float]], best_epoch: int, path: str) -> None:
    plt = _plt()
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["epoch"], h["train_loss"], label="Train (2021–2023)")
    ax.plot(h["epoch"], h["val_loss"], label="Validasi (2024)")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Epoch terbaik ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (skala Z-Score)")
    ax.set_title("Kurva Loss Encoder LSTM (peramalan iklim bulan berikutnya)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, labels, class_names, title: str, path: str) -> None:
    plt = _plt()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_group_importance(group_df: pd.DataFrame, path: str) -> None:
    plt = _plt()
    d = group_df.sort_values("importance_total")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(d["kelompok"], d["importance_total"], color="#2b7bba")
    ax.set_xlabel("Gini importance (rata-rata seed)")
    ax.set_title("Kontribusi Fitur pada Hybrid LSTM + RF")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_class_table(y_true, preds: dict[str, np.ndarray], label_encoder) -> pd.DataFrame:
    labels = np.arange(len(label_encoder.classes_))
    names = list(label_encoder.classes_)
    out = pd.DataFrame({"varietas": names})
    for model, y_pred in preds.items():
        rep = classification_report(y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0)
        out[f"{model}_precision"] = [round(rep[n]["precision"], 3) for n in names]
        out[f"{model}_recall"] = [round(rep[n]["recall"], 3) for n in names]
        out[f"{model}_f1"] = [round(rep[n]["f1-score"], 3) for n in names]
        out["support_test"] = [int(rep[n]["support"]) for n in names]
    return out


def group_importance(importances: np.ndarray, emb_dim: int, static_cols: list[str]) -> pd.DataFrame:
    rows = [{
        "kelompok": f"Fitur temporal LSTM ({emb_dim} dimensi)",
        "jumlah_fitur": emb_dim,
        "importance_total": float(importances[:emb_dim].sum()),
    }]
    for i, col in enumerate(static_cols):
        rows.append({"kelompok": col, "jumlah_fitur": 1, "importance_total": float(importances[emb_dim + i])})
    df = pd.DataFrame(rows)
    df["importance_per_fitur"] = df["importance_total"] / df["jumlah_fitur"]
    return df.sort_values("importance_total", ascending=False).round(4).reset_index(drop=True)


def topk_recommendations(
    proba: np.ndarray, clf_classes: np.ndarray, data: dict[str, Any], test_idx: np.ndarray,
    group_df: pd.DataFrame, top_k: int = 3,
) -> pd.DataFrame:
    """Top-K varietas + skor kecocokan + justifikasi berbasis feature importance (Bag. 3.10 proposal)."""
    le = data["label_encoder"]
    names = le.inverse_transform(clf_classes)
    top_static = [g for g in group_df["kelompok"] if g in data["static_cols"]][:2]
    lstm_share = group_df.loc[group_df["kelompok"].str.startswith("Fitur temporal"), "importance_total"].iloc[0]
    imp_map = dict(zip(group_df["kelompok"], group_df["importance_total"]))
    rows = []
    for row_i, idx in enumerate(test_idx):
        p = proba[row_i]
        order = np.argsort(p)[::-1][:top_k]
        loc = data["locs"][idx]
        soil = data["soil_raw"][loc]
        alasan = [f"pola iklim 12 bulan terakhir (LSTM, {lstm_share:.0%})"]
        for col in top_static:
            alasan.append(f"{col}={soil[col]}{SATUAN_STATIS.get(col, '')} ({imp_map[col]:.0%})")
        actual = le.inverse_transform([data["y"][idx]])[0]
        rec = {
            "id_lokasi": loc,
            "nama_lokasi": soil.get("nama_lokasi", loc),
            "periode": f"{NAMA_BULAN[data['months'][idx] - 1]} {data['years'][idx]}",
            "varietas_aktual_label": actual,
        }
        for r, j in enumerate(order, 1):
            rec[f"rekomendasi_{r}"] = f"{names[j]} ({p[j] * 100:.1f}%)"
        rec["aktual_di_top_k"] = actual in names[order]
        rec["justifikasi"] = "; ".join(alasan)
        rows.append(rec)
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in df.astype(object).itertuples(index=False):  # astype(object): integer tidak berubah jadi 42.0
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def write_report(path: str, ctx: dict[str, Any]) -> None:
    d, df = ctx["data"], ctx["df"]
    tr, va, te = d["train_mask"], d["val_mask"], d["test_mask"]
    le = d["label_encoder"]
    pp = d["preprocessing"]
    models = ctx["model_names"]

    def ms(model: str, key: str) -> str:
        s = df[df["model"] == model][key]
        return f"{s.mean():.3f} ± {s.std():.3f}"

    # 4.1 distribusi kelas
    dist = pd.DataFrame({"varietas": le.classes_})
    for nm, mask in (("train", tr), ("validasi", va), ("test", te)):
        dist[nm] = np.bincount(d["y"][mask], minlength=len(le.classes_))

    # 4.2 preprocessing
    prep_tbl = pd.DataFrame({
        "variabel": d["climate_cols"],
        "batas_bawah_IQR": [pp["iqr_lower"][c] for c in d["climate_cols"]],
        "batas_atas_IQR": [pp["iqr_upper"][c] for c in d["climate_cols"]],
        "mean_Z": [pp["zscore_mean"][c] for c in d["climate_cols"]],
        "std_Z": [pp["zscore_std"][c] for c in d["climate_cols"]],
    })

    hyb = df[df["model"] == models["hybrid"]]
    lstm_tbl = hyb[["seed", "lstm_best_epoch", "lstm_best_val_mse", "mae_lstm", "mae_persistence", "mae_klimatologi"]].copy()
    lstm_tbl["lstm_best_epoch"] = lstm_tbl["lstm_best_epoch"].astype(int)
    lstm_tbl.columns = ["seed", "epoch terbaik", "MSE validasi", "MAE test LSTM", "MAE persistence", "MAE klimatologi"]

    params_tbl = hyb[["seed", "best_params"]].copy()

    main_rows = []
    for key, label in (("hybrid", "Hybrid LSTM + RF (usulan)"), ("rf", "Random Forest saja"), ("lstm", "LSTM saja")):
        m = models[key]
        main_rows.append({
            "Model": label,
            "Accuracy": ms(m, "accuracy"),
            "Precision (w)": ms(m, "precision_weighted"),
            "Recall (w)": ms(m, "recall_weighted"),
            "F1-Score (w)": ms(m, "f1_weighted"),
            "F1-Score (macro)": ms(m, "f1_macro"),
            "Top-3 Acc": ms(m, "top_3_accuracy"),
        })
    main_tbl = pd.DataFrame(main_rows)

    seed_tbl = df.pivot_table(index="seed", columns="model", values="f1_weighted").round(3).reset_index()
    wins_rf = int((ctx["diff_rf"] > 0).sum())
    wins_lstm = int((ctx["diff_lstm"] > 0).sum())
    n = len(ctx["diff_rf"])

    rec = ctx["rec_df"]
    rec_show = rec[rec["periode"].isin(["Jan 2025", "Jul 2025"])].drop(columns=["justifikasi"])
    hit_rate = rec["aktual_di_top_k"].mean()

    md = f"""# Hasil Eksperimen Model Hybrid LSTM–Random Forest (Draf Bab IV)

> **PERINGATAN — HASIL PROTOTIPE.** Label varietas dibuat dengan aturan dummy
> (`beri_label_varietas_prototipe`) dan 5 dari 9 fitur tanah (N, P, K, drainase, risiko hama) masih dummy.
> Angka di bawah membuktikan **pipeline** berjalan sesuai rancangan proposal, tetapi **belum dapat diklaim
> sebagai temuan penelitian**. Jalankan ulang `python train_hybrid_proposal.py` setelah data riil tersedia.

- Run: `{ctx['run_id']}` · commit `{ctx['commit']}` · {ctx['timestamp']}
- Lokasi aktif: {', '.join(pp['locations'])} · Seed: {', '.join(map(str, ctx['seeds']))}
- Script: `train_hybrid_proposal.py`

## 4.1 Dataset dan Pembagian Data

Satu sampel = satu lokasi pada satu bulan target. Input LSTM adalah sekuens iklim **{pp['sequence_length']} bulan**
sebelum bulan target, sehingga tidak ada informasi bulan target yang digunakan. Data dibagi kronologis
(Bagian 3.7 dan 3.11 proposal):

| Bagian | Tahun bulan target | Jumlah sampel | Porsi |
|---|---|---|---|
| Training | 2021–2023 | {tr.sum()} | {tr.sum() / len(d['y']):.0%} |
| Validation | 2024 | {va.sum()} | {va.sum() / len(d['y']):.0%} |
| Testing | 2025 | {te.sum()} | {te.sum() / len(d['y']):.0%} |

Porsi pelatihan (training + validation) = {(tr.sum() + va.sum()) / len(d['y']):.0%}, pengujian = {te.sum() / len(d['y']):.0%}.
Jumlah kelas varietas: {len(le.classes_)}.

**Tabel 4.1 Distribusi kelas varietas**

{_md_table(dist)}

## 4.2 Hasil Preprocessing

Missing value diinterpolasi linier per lokasi. Outlier ditangani dengan metode IQR (pagar Q1 − 1,5·IQR dan
Q3 + 1,5·IQR) lalu dinormalisasi Z-Score; seluruh statistik dihitung **hanya dari data periode training**.
Jumlah nilai yang dibatasi (capping): **{pp['n_outliers_capped']}**. Fitur turunan indeks musim: sin/cos bulan.

**Tabel 4.2 Parameter preprocessing iklim bulanan**

{_md_table(prep_tbl)}

## 4.3 Pelatihan Encoder LSTM

Arsitektur sesuai Bagian 3.10: LSTM 128 unit → LSTM 64 unit (dropout {ctx['lstm_cfg']['dropout']}) → Dense
{ctx['lstm_cfg']['embedding_dim']} unit (ReLU) sebagai vektor fitur temporal. Encoder dilatih meramal iklim bulan
berikutnya (MSE) sehingga tidak menggunakan label varietas; early stopping (patience
{ctx['patience']}) berdasarkan loss validasi 2024.

![Kurva loss LSTM](lstm_loss_curve.png)

**Tabel 4.3 Hasil pelatihan LSTM dan pembanding ramalan naif (MAE pada test 2025, skala Z-Score)**

{_md_table(lstm_tbl)}

Persistence = iklim bulan sebelumnya; klimatologi = rata-rata bulan yang sama pada periode training.

## 4.4 Hyperparameter Tuning Random Forest

RandomizedSearchCV ({ctx['n_iter']} kombinasi, skor F1-weighted) dengan cross-validation expanding-window per
tahun (validasi {', '.join(map(str, ctx['cv_years']))}), kemudian model di-*refit* pada training + validation.

**Tabel 4.4 Hyperparameter terbaik Hybrid LSTM + RF per seed**

{_md_table(params_tbl)}

## 4.5 Perbandingan Kinerja Model (Rumusan Masalah 3)

Evaluasi pada data testing 2025, rata-rata ± simpangan baku dari {n} seed. (w) = weighted average.

**Tabel 4.5 Perbandingan model hybrid dengan single model**

{_md_table(main_tbl)}

**Tabel 4.6 F1-Score (weighted) per seed**

{_md_table(seed_tbl)}

Hybrid lebih tinggi dari Random Forest saja pada **{wins_rf} dari {n}** seed (selisih rata-rata
{ctx['diff_rf'].mean():+.3f}) dan lebih tinggi dari LSTM saja pada **{wins_lstm} dari {n}** seed
(selisih rata-rata {ctx['diff_lstm'].mean():+.3f}).

## 4.6 Kinerja per Kelas dan Confusion Matrix

**Tabel 4.7 Precision, Recall, F1 per varietas (seed {ctx['seeds'][0]})**

{_md_table(ctx['per_class'])}

![Confusion matrix Hybrid](confusion_matrix_hybrid.png)
![Confusion matrix RF](confusion_matrix_rf.png)
![Confusion matrix LSTM](confusion_matrix_lstm.png)

## 4.7 Analisis Feature Importance

**Tabel 4.8 Kontribusi kelompok fitur (Gini importance, rata-rata seed)**

{_md_table(ctx['group_df'])}

Fitur temporal LSTM berjumlah {ctx['emb_dim']} dari {ctx['emb_dim'] + len(d['static_cols'])} fitur
({ctx['emb_dim'] / (ctx['emb_dim'] + len(d['static_cols'])):.0%}); bandingkan porsi jumlah ini dengan porsi
importance-nya sebelum menyimpulkan dominasi fitur temporal.

![Feature importance](feature_importance_group.png)

## 4.8 Contoh Rekomendasi Top-{ctx['top_k']} (Recommendation Engine, Bagian 3.10)

Skor kecocokan = probabilitas kelas dari Random Forest. Varietas label berada di Top-{ctx['top_k']} pada
**{hit_rate:.0%}** sampel test (seed {ctx['seeds'][0]}). Daftar lengkap beserta justifikasi:
`rekomendasi_top{ctx['top_k']}_test2025.csv`.

**Tabel 4.9 Contoh rekomendasi (Januari dan Juli 2025)**

{_md_table(rec_show)}

Contoh justifikasi: *{rec['justifikasi'].iloc[0]}*.

## 4.9 Penyesuaian terhadap Proposal dan Keterbatasan

| Aspek | Proposal | Implementasi | Alasan |
|---|---|---|---|
| Panjang sekuens | 60 bulan | {pp['sequence_length']} bulan | data hanya 72 bulan (2020–2025) |
| Sumber iklim | BMKG | NASA POWER | data BMKG belum tersedia |
| Lama penyinaran | jam/hari | radiasi MJ/m²/hari | variabel NASA POWER ALLSKY_SFC_SW_DWN |
| Objektif pelatihan LSTM | tidak ditentukan | peramalan iklim bulan berikutnya | menghindari kebocoran label ke tahap RF |
| Lokasi | 20 titik | {len(pp['locations'])} titik | tahap awal |
| Label varietas | data lapangan | aturan dummy | data lapangan belum tersedia |

Keterbatasan: jumlah data test kecil ({te.sum()} sampel), label dan sebagian fitur tanah masih dummy,
serta validasi pakar dan *user acceptance testing* (Bagian 3.11 Tahap 8) belum dilakukan.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
