"""Script khusus Tim Random Forest: muat output LSTM, latih RF, evaluasi, dan catat log otomatis."""

from __future__ import annotations

import os
import time
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.dataset_loader import load_config
from src.evaluate import compute_classification_metrics
from src.experiment_logger import export_git_changelog, log_experiment, save_rf_artifacts
from src.recommender import get_feature_importance_summary, recommend_top_k
from src.rf_classifier import concatenate_features, train_random_forest


def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    emb_path = os.path.join(cfg["data"]["processed_dir"], "lstm_embeddings.npz")

    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"File {emb_path} belum ditemukan. Jalankan `python train_lstm.py` terlebih dahulu."
        )

    print(f"[RF] Memuat vektor temporal LSTM & fitur statis dari {emb_path}...")
    bundle = np.load(emb_path, allow_pickle=True)
    emb_train = bundle["emb_train"]
    emb_test = bundle["emb_test"]
    X_seq_train = bundle["X_seq_train"]
    X_seq_test = bundle["X_seq_test"]
    X_stat_train = bundle["X_stat_train"]
    X_stat_test = bundle["X_stat_test"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]
    classes = bundle["classes"]

    label_encoder = LabelEncoder()
    label_encoder.classes_ = classes

    X_hybrid_train = concatenate_features(emb_train, X_stat_train)
    X_hybrid_test = concatenate_features(emb_test, X_stat_test)
    print(f"[RF] Fitur Hybrid siap -> Train: {X_hybrid_train.shape}, Test: {X_hybrid_test.shape}")

    rf_cfg = cfg["random_forest"]
    print("\n[RF] Melatih Hybrid Random Forest Classifier...")
    t0 = time.perf_counter()
    hybrid_rf = train_random_forest(
        X_train=X_hybrid_train,
        y_train=y_train,
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_split=rf_cfg["min_samples_split"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        class_weight=rf_cfg["class_weight"],
        seed=seed,
    )
    hybrid_duration = time.perf_counter() - t0

    # Pembanding Baseline RF (tanpa LSTM: hanya fitur statis + rata-rata iklim)
    X_base_train = np.hstack([X_seq_train.mean(axis=1), X_stat_train])
    X_base_test = np.hstack([X_seq_test.mean(axis=1), X_stat_test])
    t1 = time.perf_counter()
    baseline_rf = train_random_forest(
        X_train=X_base_train,
        y_train=y_train,
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        class_weight=rf_cfg["class_weight"],
        seed=seed,
    )
    base_duration = time.perf_counter() - t1

    os.makedirs("models", exist_ok=True)
    rf_model_path = os.path.join("models", "hybrid_rf.pkl")
    joblib.dump(hybrid_rf, rf_model_path)
    print(f"[RF] Model Random Forest disimpan ke: {rf_model_path}")

    top_k = cfg["recommendation"]["top_k"]
    hybrid_pred = hybrid_rf.predict(X_hybrid_test)
    hybrid_metrics = compute_classification_metrics(
        y_test,
        hybrid_pred,
        hybrid_rf.predict_proba(X_hybrid_test),
        hybrid_rf.classes_,
        top_k=top_k,
    )
    base_metrics = compute_classification_metrics(
        y_test,
        baseline_rf.predict(X_base_test),
        baseline_rf.predict_proba(X_base_test),
        baseline_rf.classes_,
        top_k=top_k,
    )

    print("\n=== PERBANDINGAN EVALUASI ===")
    print("Baseline RF (Statis + Rata-rata Iklim):", base_metrics)
    print("Hybrid LSTM + RF                     :", hybrid_metrics)

    feature_names = [f"fitur_temporal_lstm_{i}" for i in range(emb_train.shape[1])] + cfg["data"]["static_features"]
    top_feats = get_feature_importance_summary(hybrid_rf, feature_names, top_n=8)
    print("\n=== FEATURE IMPORTANCE TERATAS (HYBRID) ===")
    for name, score in top_feats:
        print(f"  - {name}: {score}")

    sample_recs = recommend_top_k(hybrid_rf, X_hybrid_test[:3], label_encoder, top_k=top_k)
    print("\n=== CONTOH REKOMENDASI TOP-3 (DATA UJI) ===")
    for idx, recs in enumerate(sample_recs, 1):
        print(f"  Sampel {idx}: {recs}")

    log_experiment(
        stage="Baseline_RF",
        train_samples=len(y_train),
        test_samples=len(y_test),
        num_classes=len(classes),
        hyperparameters=rf_cfg,
        duration_sec=base_duration,
        metrics=base_metrics,
        catatan=f"Static + Mean Climate, seed={seed}",
    )
    run_dir = log_experiment(
        stage="Hybrid_LSTM_RF",
        train_samples=len(y_train),
        test_samples=len(y_test),
        num_classes=len(classes),
        hyperparameters=rf_cfg,
        duration_sec=hybrid_duration,
        metrics=hybrid_metrics,
        top_features=top_feats,
        catatan=f"LSTM 32-dim + Static (41 fitur), seed={seed}",
    )
    save_rf_artifacts(
        run_dir=run_dir,
        y_true=y_test,
        y_pred=hybrid_pred,
        classes=hybrid_rf.classes_,
        class_names=list(label_encoder.inverse_transform(hybrid_rf.classes_)),
        feature_names=feature_names,
        feature_importances=hybrid_rf.feature_importances_,
    )
    export_git_changelog()
    print(f"\n[LOG] Hasil evaluasi, report, & grafik disimpan ke results/experiment_history.csv & {run_dir}")


if __name__ == "__main__":
    main()
