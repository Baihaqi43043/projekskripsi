"""Script khusus Tim Random Forest: muat output LSTM, latih & tuning RF, evaluasi, dan catat log."""

from __future__ import annotations

import argparse
import os
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.dataset_loader import load_config
from src.evaluate import compute_classification_metrics
from src.experiment_logger import export_git_changelog, log_experiment, save_rf_artifacts
from src.recommender import get_feature_importance_summary, recommend_top_k
from src.rf_classifier import concatenate_features, train_random_forest


def run_rf_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    n_iter: int = 30,
) -> tuple[RandomForestClassifier, dict]:
    """Cari kombinasi hyperparameter terbaik menggunakan RandomizedSearchCV + StratifiedKFold."""
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=seed, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_weighted",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def main(tune: bool = False) -> None:
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
    print("\n[RF] Melatih Hybrid Random Forest Classifier (Default Config)...")
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
    print("Hybrid LSTM + RF (Default)           :", hybrid_metrics)

    feature_names = [f"fitur_temporal_lstm_{i}" for i in range(emb_train.shape[1])] + cfg["data"]["static_features"]
    top_feats = get_feature_importance_summary(hybrid_rf, feature_names, top_n=8)

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

    best_model = hybrid_rf
    if tune:
        print("\n[RF] Menjalankan Hyperparameter Tuning (RandomizedSearchCV 30 iterasi x 5-fold)...")
        t2 = time.perf_counter()
        tuned_rf, best_params = run_rf_tuning(X_hybrid_train, y_train, seed=seed, n_iter=30)
        tune_duration = time.perf_counter() - t2

        tuned_path = os.path.join("models", "hybrid_rf_tuned.pkl")
        joblib.dump(tuned_rf, tuned_path)
        tuned_pred = tuned_rf.predict(X_hybrid_test)
        tuned_metrics = compute_classification_metrics(
            y_test,
            tuned_pred,
            tuned_rf.predict_proba(X_hybrid_test),
            tuned_rf.classes_,
            top_k=top_k,
        )
        tuned_top_feats = get_feature_importance_summary(tuned_rf, feature_names, top_n=8)
        print("Hybrid LSTM + RF (Tuned)             :", tuned_metrics)
        print("Best Hyperparameters                 :", best_params)

        tuned_dir = log_experiment(
            stage="Hybrid_RF_Tuned",
            train_samples=len(y_train),
            test_samples=len(y_test),
            num_classes=len(classes),
            hyperparameters=best_params,
            duration_sec=tune_duration,
            metrics=tuned_metrics,
            top_features=tuned_top_feats,
            catatan=f"RandomizedSearchCV 30x5 lokal, seed={seed}",
        )
        save_rf_artifacts(
            run_dir=tuned_dir,
            y_true=y_test,
            y_pred=tuned_pred,
            classes=tuned_rf.classes_,
            class_names=list(label_encoder.inverse_transform(tuned_rf.classes_)),
            feature_names=feature_names,
            feature_importances=tuned_rf.feature_importances_,
        )
        best_model = tuned_rf
        top_feats = tuned_top_feats

    print("\n=== FEATURE IMPORTANCE TERATAS ===")
    for name, score in top_feats:
        print(f"  - {name}: {score}")

    sample_recs = recommend_top_k(best_model, X_hybrid_test[:3], label_encoder, top_k=top_k)
    print("\n=== CONTOH REKOMENDASI TOP-3 (DATA UJI) ===")
    for idx, recs in enumerate(sample_recs, 1):
        print(f"  Sampel {idx}: {recs}")

    export_git_changelog()
    print("\n[LOG] Semua metrik, parameter & changelog diperbarui di results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training & tuning Hybrid Random Forest.")
    parser.add_argument("--tune", action="store_true", help="Jalankan RandomizedSearchCV hyperparameter tuning.")
    args = parser.parse_args()
    main(tune=args.tune)
