"""Pipeline Hybrid LSTM + Random Forest yang mengikuti proposal disertasi (Bab III).

Alur (lihat juga src/proposal_pipeline.py):
1. Data iklim BULANAN 6 variabel + 9 fitur statis tanah/lingkungan (Tabel 3.1 & 3.2).
2. Preprocessing: interpolasi, outlier IQR, Z-Score, indeks musim; statistik dari data train saja (Bag. 3.10).
3. Split kronologis per tahun bulan target: train <= 2023, validasi 2024, test 2025 (Bag. 3.7 & 3.11).
4. LSTM 2 layer (128 -> 64, dropout) -> Dense 32 sebagai temporal feature extractor (Bag. 3.10).
   Encoder dilatih MERAMAL iklim bulan berikutnya (tanpa label varietas), early stopping di validasi.
5. Random Forest pada [vektor LSTM 32-dim | fitur statis], hyperparameter tuning dengan
   cross-validation expanding-window per tahun, lalu refit di train+validasi (Bag. 3.11 Tahap 5).
6. Evaluasi sekali di test 2025: Accuracy, Precision, Recall, F1, confusion matrix, feature importance,
   Top-K, dibandingkan dengan single model RF saja & LSTM saja (Bag. 3.11 Tahap 6).
   Laporan draf Bab IV ditulis otomatis ke results/runs/<id>/laporan_hasil_bab4.md.
7. Diulang untuk beberapa seed; dilaporkan rata-rata +- simpangan baku.

Jalankan:  python train_hybrid_proposal.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from src import report_proposal as report
from src.dataset_loader import load_config
from src.evaluate import compute_classification_metrics
from src.experiment_logger import export_git_changelog, log_experiment, save_rf_artifacts
from src.lstm_encoder import (
    extract_temporal_features,
    predict_forecast,
    predict_proba_lstm,
    train_forecast_encoder,
    train_lstm_classifier,
)
from src.proposal_inference import BUNDLE_PATH, save_bundle
from src.proposal_pipeline import prepare_proposal_dataset, temporal_cv_folds

RF_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": [None, "balanced", "balanced_subsample"],
}
METRIC_KEYS = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "top_3_accuracy",
    "top_5_accuracy",
]


def tune_and_fit_rf(X_dev, y_dev, folds, seed: int, n_iter: int) -> tuple[RandomForestClassifier, dict]:
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=seed, n_jobs=1),
        param_distributions=RF_PARAM_DIST,
        n_iter=n_iter,
        scoring="f1_weighted",
        cv=folds,
        random_state=seed,
        refit=True,  # refit dengan parameter terbaik pada seluruh data train+validasi
        n_jobs=-1,
    )
    search.fit(X_dev, y_dev)
    return search.best_estimator_, search.best_params_


def _with_top5(metrics: dict, y_true, y_pred, proba, classes) -> dict:
    extra = compute_classification_metrics(y_true, y_pred, proba, classes, top_k=5)
    metrics.update({k: v for k, v in extra.items() if k.startswith("top_")})
    return metrics


def evaluate(clf, X_test, y_test, top_k: int) -> dict:
    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    m = compute_classification_metrics(y_test, y_pred, proba, clf.classes_, top_k=top_k)
    return _with_top5(m, y_test, y_pred, proba, clf.classes_)


def forecast_check(model, data) -> dict:
    """Bandingkan ramalan LSTM (bulan berikutnya) dengan baseline naif pada data test (skala Z-Score)."""
    n_clim = len(data["climate_cols"])
    te = data["test_mask"]
    tr = data["train_mask"]
    y_true = data["y_forecast"][te]
    pred_lstm = predict_forecast(model, data["X_seq"][te])
    pred_persist = data["X_seq"][te][:, -1, :n_clim]  # iklim bulan sebelumnya
    month_mean = (
        pd.DataFrame(data["y_forecast"][tr]).assign(m=data["months"][tr]).groupby("m").mean()
    )
    pred_clim = month_mean.loc[data["months"][te]].to_numpy()  # rata-rata bulan yang sama (train)
    mae = lambda p: float(np.mean(np.abs(y_true - p)))  # noqa: E731
    return {
        "mae_lstm": round(mae(pred_lstm), 4),
        "mae_persistence": round(mae(pred_persist), 4),
        "mae_klimatologi": round(mae(pred_clim), 4),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # agar simbol ± tampil benar di console Windows
    cfg = load_config()
    pcfg, lcfg = cfg["proposal"], cfg["lstm"]
    top_k = cfg["recommendation"]["top_k"]
    data = prepare_proposal_dataset(cfg)

    tr, va, te = data["train_mask"], data["val_mask"], data["test_mask"]
    dev = tr | va
    le = data["label_encoder"]
    print(
        f"[Proposal] Sampel train={tr.sum()} | validasi={va.sum()} | test={te.sum()} | "
        f"lokasi={len(data['preprocessing']['locations'])} | kelas={len(le.classes_)} | "
        f"seq_len={data['preprocessing']['sequence_length']} bulan | outlier IQR di-cap={data['preprocessing']['n_outliers_capped']}"
    )
    print(f"[Proposal] Kelas di test yang tidak ada di train+validasi: "
          f"{sorted(set(le.inverse_transform(data['y'][te])) - set(le.inverse_transform(data['y'][dev])))}")

    folds = temporal_cv_folds(data["years"][dev], pcfg["cv_val_years"])
    base_feat = data["X_seq"][:, :, : len(data["climate_cols"])].mean(axis=1)
    X_base = np.hstack([base_feat, data["X_stat"]])
    emb_dim = lcfg["embedding_dim"]
    feature_names = [f"fitur_temporal_lstm_{i}" for i in range(emb_dim)] + data["static_cols"]

    names = {
        "rf": "Random Forest saja (statis + rata-rata iklim)",
        "lstm": "LSTM saja (sekuens iklim)",
        "hybrid": "Hybrid LSTM + RF",
    }
    rows, first, hyb_importances = [], {}, []
    all_cls = np.arange(len(le.classes_))
    t_start = time.perf_counter()
    for i, seed in enumerate(pcfg["seeds"]):
        torch.manual_seed(seed)
        np.random.seed(seed)
        lstm_kwargs = dict(
            hidden_1=lcfg["hidden_size_1"], hidden_2=lcfg["hidden_size_2"], embedding_dim=emb_dim,
            dropout=lcfg["dropout"], max_epochs=pcfg["max_epochs"], patience=pcfg["early_stopping_patience"],
            batch_size=lcfg["batch_size"], learning_rate=lcfg["learning_rate"], seed=seed,
        )
        # Encoder LSTM untuk hybrid: meramal iklim bulan berikutnya (tanpa label varietas)
        lstm = train_forecast_encoder(
            data["X_seq"][tr], data["y_forecast"][tr], data["X_seq"][va], data["y_forecast"][va], **lstm_kwargs
        )
        hist = lstm.training_history_
        fc = forecast_check(lstm, data)
        emb = extract_temporal_features(lstm, data["X_seq"])
        X_hyb = np.hstack([emb, data["X_stat"]])

        # Single model 1: Random Forest saja
        base_rf, base_params = tune_and_fit_rf(X_base[dev], data["y"][dev], folds, seed, pcfg["rf_search_iter"])
        m_base = evaluate(base_rf, X_base[te], data["y"][te], top_k)

        # Single model 2: LSTM saja (klasifikasi langsung, early stopping di validasi 2024)
        lstm_clf = train_lstm_classifier(
            data["X_seq"][tr], data["y"][tr], data["X_seq"][va], data["y"][va],
            num_classes=len(le.classes_), **lstm_kwargs,
        )
        proba_lstm = predict_proba_lstm(lstm_clf, data["X_seq"][te])
        pred_lstm = proba_lstm.argmax(axis=1)
        m_lstm = compute_classification_metrics(data["y"][te], pred_lstm, proba_lstm, all_cls, top_k=top_k)
        m_lstm = _with_top5(m_lstm, data["y"][te], pred_lstm, proba_lstm, all_cls)

        # Model usulan: Hybrid LSTM + RF
        hyb_rf, hyb_params = tune_and_fit_rf(X_hyb[dev], data["y"][dev], folds, seed, pcfg["rf_search_iter"])
        m_hyb = evaluate(hyb_rf, X_hyb[te], data["y"][te], top_k)

        imp = hyb_rf.feature_importances_
        hyb_importances.append(imp)
        rows.append({"seed": seed, "model": names["rf"], **m_base, "best_params": json.dumps(base_params)})
        rows.append({"seed": seed, "model": names["lstm"], **m_lstm, "best_params": ""})
        rows.append({"seed": seed, "model": names["hybrid"], **m_hyb, "best_params": json.dumps(hyb_params),
                     "importance_lstm": round(float(imp[:emb_dim].sum()), 4), "lstm_best_epoch": hist["best_epoch"],
                     "lstm_best_val_mse": round(hist["best_val_loss"], 4), **fc})
        print(
            f"  seed {seed}: LSTM epoch terbaik={hist['best_epoch']:3d} | F1w RF={m_base['f1_weighted']:.3f} "
            f"LSTM={m_lstm['f1_weighted']:.3f} Hybrid={m_hyb['f1_weighted']:.3f}"
        )
        if i == 0:
            first = {
                "lstm": lstm, "hyb_rf": hyb_rf, "X_hyb": X_hyb, "hist": hist,
                "preds": {
                    "hybrid": hyb_rf.predict(X_hyb[te]),
                    "rf": base_rf.predict(X_base[te]),
                    "lstm": pred_lstm,
                },
            }
    duration = time.perf_counter() - t_start

    df = pd.DataFrame(rows)
    summary = df.groupby("model")[METRIC_KEYS].agg(["mean", "std"]).round(4)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()

    def _mean_metrics(name: str) -> dict:
        sub = df[df["model"] == name]
        return {k: round(float(sub[k].mean()), 4) for k in METRIC_KEYS}

    def _std_note(name: str) -> str:
        sub = df[df["model"] == name]
        return ", ".join(f"{k}={sub[k].mean():.3f}±{sub[k].std():.3f}" for k in ("accuracy", "f1_weighted", "f1_macro", "top_3_accuracy"))

    n_seeds = len(pcfg["seeds"])
    common = dict(train_samples=int(dev.sum()), test_samples=int(te.sum()), num_classes=len(le.classes_))
    log_experiment(
        stage="Baseline_RF_Proposal", hyperparameters={"rf_search": RF_PARAM_DIST, "cv_val_years": pcfg["cv_val_years"]},
        duration_sec=duration / 3, metrics=_mean_metrics(names["rf"]),
        catatan=f"Pipeline proposal, RF saja, rata-rata {n_seeds} seed: {_std_note(names['rf'])}", **common,
    )
    log_experiment(
        stage="Baseline_LSTM_Proposal", hyperparameters={"lstm": lcfg},
        duration_sec=duration / 3, metrics=_mean_metrics(names["lstm"]),
        catatan=f"Pipeline proposal, LSTM saja, rata-rata {n_seeds} seed: {_std_note(names['lstm'])}", **common,
    )
    hyb_sub = df[df["model"] == names["hybrid"]]
    mean_imp = np.mean(hyb_importances, axis=0)
    top_feats = sorted(zip(feature_names, mean_imp), key=lambda x: x[1], reverse=True)[:8]
    run_dir = log_experiment(
        stage="Hybrid_LSTM_RF_Proposal",
        hyperparameters={"lstm": lcfg, "proposal": pcfg, "rf_search": RF_PARAM_DIST},
        duration_sec=duration / 3, metrics=_mean_metrics(names["hybrid"]),
        loss_akhir=float(hyb_sub["lstm_best_val_mse"].mean()),
        top_features=[(n, round(float(v), 4)) for n, v in top_feats],
        catatan=f"Pipeline proposal, rata-rata {n_seeds} seed: {_std_note(names['hybrid'])}", **common,
    )

    # ---------------- Artefak & laporan Bab IV ----------------
    df.to_csv(os.path.join(run_dir, "metrics_per_seed.csv"), index=False)
    summary.to_csv(os.path.join(run_dir, "metrics_summary.csv"), index=False)
    pd.DataFrame(first["hist"]["loss_per_epoch"]).to_csv(os.path.join(run_dir, "lstm_loss_history_seed0.csv"), index=False)
    with open(os.path.join(run_dir, "preprocessing.json"), "w", encoding="utf-8") as f:
        json.dump(data["preprocessing"], f, indent=2, ensure_ascii=False)

    hyb_rf = first["hyb_rf"]
    y_te = data["y"][te]
    save_rf_artifacts(
        run_dir=run_dir, y_true=y_te, y_pred=first["preds"]["hybrid"], classes=hyb_rf.classes_,
        class_names=list(le.inverse_transform(hyb_rf.classes_)), feature_names=feature_names,
        feature_importances=hyb_rf.feature_importances_,
    )
    for key, title in (("hybrid", "Hybrid LSTM + RF"), ("rf", "Random Forest saja"), ("lstm", "LSTM saja")):
        report.plot_confusion(
            y_te, first["preds"][key], all_cls, list(le.classes_),
            f"Confusion Matrix - {title} (test 2025)", os.path.join(run_dir, f"confusion_matrix_{key}.png"),
        )
    report.plot_loss_curve(
        first["hist"]["loss_per_epoch"], first["hist"]["best_epoch"], os.path.join(run_dir, "lstm_loss_curve.png")
    )
    per_class = report.per_class_table(
        y_te, {"Hybrid": first["preds"]["hybrid"], "RF": first["preds"]["rf"], "LSTM": first["preds"]["lstm"]}, le
    )
    per_class = per_class[["varietas", "support_test", "Hybrid_precision", "Hybrid_recall", "Hybrid_f1", "RF_f1", "LSTM_f1"]]
    per_class.to_csv(os.path.join(run_dir, "metrics_per_kelas_seed0.csv"), index=False)
    group_df = report.group_importance(mean_imp, emb_dim, data["static_cols"])
    group_df.to_csv(os.path.join(run_dir, "feature_importance_kelompok.csv"), index=False)
    report.plot_group_importance(group_df, os.path.join(run_dir, "feature_importance_group.png"))
    rec_df = report.topk_recommendations(
        hyb_rf.predict_proba(first["X_hyb"][te]), hyb_rf.classes_, data, np.where(te)[0], group_df, top_k=top_k
    )
    rec_df.to_csv(os.path.join(run_dir, f"rekomendasi_top{top_k}_test2025.csv"), index=False)

    f1 = lambda key: df[df["model"] == names[key]]["f1_weighted"].to_numpy()  # noqa: E731
    with open(os.path.join(run_dir, "summary.json"), "r", encoding="utf-8") as f:
        run_meta = json.load(f)
    report.write_report(os.path.join(run_dir, "laporan_hasil_bab4.md"), {
        "data": data, "df": df, "model_names": names, "run_id": run_meta["run_id"],
        "commit": run_meta["git"]["commit"], "timestamp": run_meta["timestamp"], "seeds": pcfg["seeds"],
        "lstm_cfg": lcfg, "patience": pcfg["early_stopping_patience"], "n_iter": pcfg["rf_search_iter"],
        "cv_years": pcfg["cv_val_years"], "diff_rf": f1("hybrid") - f1("rf"), "diff_lstm": f1("hybrid") - f1("lstm"),
        "per_class": per_class, "group_df": group_df, "emb_dim": emb_dim, "rec_df": rec_df, "top_k": top_k,
    })

    os.makedirs("models", exist_ok=True)
    torch.save(first["lstm"].state_dict(), "models/proposal_lstm_encoder.pt")
    joblib.dump(hyb_rf, "models/proposal_hybrid_rf.pkl")
    # Bundle lengkap untuk aplikasi (app_rekomendasi_proposal.py): model seed pertama + preprocessing
    save_bundle(BUNDLE_PATH, data, first["lstm"], hyb_rf, lcfg, group_df, run_meta["run_id"],
                _mean_metrics(names["hybrid"]))
    export_git_changelog()

    print(f"\n=== HASIL TEST 2025 (rata-rata ± std dari {n_seeds} seed) ===")
    for key in ("rf", "lstm", "hybrid"):
        sub = df[df["model"] == names[key]]
        print(f"{names[key]:45s} | " + " | ".join(
            f"{k}={sub[k].mean():.3f}±{sub[k].std():.3f}" for k in ("accuracy", "f1_weighted", "f1_macro", "top_3_accuracy")))
    print(f"Porsi importance fitur LSTM: {hyb_sub['importance_lstm'].mean():.1%} "
          f"(porsi jumlah fitur {emb_dim / len(feature_names):.1%})")
    print(f"\n[LOG] Laporan Bab IV: {os.path.join(run_dir, 'laporan_hasil_bab4.md')}")


if __name__ == "__main__":
    main()
