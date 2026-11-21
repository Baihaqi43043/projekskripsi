"""Pipeline end-to-end Model Hybrid LSTM-Random Forest untuk Rekomendasi Varietas Tanaman."""

from __future__ import annotations

import os
import yaml
import numpy as np
import pandas as pd

from src.data_fetcher import collect_aceh_utara_sample
from src.evaluate import compute_classification_metrics
from src.lstm_encoder import extract_temporal_features, train_lstm_encoder
from src.preprocessing import build_sequences_and_static
from src.recommender import get_feature_importance_summary, recommend_top_k
from src.rf_classifier import concatenate_features, train_random_forest


def beri_label_varietas_prototipe(
    climate_df: pd.DataFrame,
    soil_df: pd.DataFrame,
) -> pd.DataFrame:
    """Berikan label varietas prototipe berbasis aturan agroekologi.

    Catatan: Ganti fungsi ini dengan label varietas aktual dari Dinas Pertanian / Kementan
    setelah data lapangan tersedia.
    """
    merged = climate_df.merge(soil_df, on="id_lokasi", how="left")
    labels = []
    for _, row in merged.iterrows():
        hujan = row["curah_hujan"]
        suhu = row["suhu_rata_rata"]
        ph = row["ph_tanah"]

        if hujan >= 200 and ph >= 6.0:
            labels.append("Padi_Inpari_32")
        elif hujan >= 150 and suhu >= 26.5:
            labels.append("Padi_Ciherang")
        elif hujan >= 120 and ph < 6.0:
            labels.append("Padi_Inpari_42")
        elif 80 <= hujan < 150 and suhu >= 26.0:
            labels.append("Jagung_BISI_18")
        elif 60 <= hujan < 120:
            labels.append("Jagung_NK_212")
        else:
            labels.append("Kedelai_Anjasmoro")

    out = climate_df.copy()
    out["varietas"] = labels
    return out


def main() -> None:
    with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["project"]["seed"]
    raw_dir = cfg["data"]["raw_dir"]
    climate_path = os.path.join(raw_dir, "aceh_utara_agroklimat_bulanan.csv")
    soil_path = os.path.join(raw_dir, "aceh_utara_tanah_statis.csv")

    if os.path.exists(climate_path) and os.path.exists(soil_path):
        print("Memuat dataset Aceh Utara dari data/raw/...")
        climate_df = pd.read_csv(climate_path)
        soil_df = pd.read_csv(soil_path)
    else:
        print("Mengambil data agroklimat (NASA POWER) & tanah (SoilGrids) untuk Aceh Utara...")
        climate_df, soil_df = collect_aceh_utara_sample(cfg)

    climate_df = beri_label_varietas_prototipe(climate_df, soil_df)

    data = build_sequences_and_static(
        climate_df=climate_df,
        static_df=soil_df,
        climate_cols=cfg["data"]["climate_features"],
        static_cols=cfg["data"]["static_features"],
        target_col="varietas",
        seq_len=cfg["data"]["sequence_length"],
        train_ratio=cfg["data"]["train_ratio"],
    )

    num_classes = len(data["label_encoder"].classes_)
    print(
        f"\nDataset siap -> Train: {data['X_seq_train'].shape}, "
        f"Test: {data['X_seq_test'].shape}, Jumlah Varietas: {num_classes}"
    )

    print("\n1. Melatih LSTM Temporal Encoder...")
    lstm_cfg = cfg["lstm"]
    lstm_model = train_lstm_encoder(
        X_seq_train=data["X_seq_train"],
        y_train=data["y_train"],
        num_classes=num_classes,
        hidden_1=lstm_cfg["hidden_size_1"],
        hidden_2=lstm_cfg["hidden_size_2"],
        embedding_dim=lstm_cfg["embedding_dim"],
        dropout=lstm_cfg["dropout"],
        epochs=lstm_cfg["epochs"],
        batch_size=lstm_cfg["batch_size"],
        learning_rate=lstm_cfg["learning_rate"],
        seed=seed,
    )

    emb_train = extract_temporal_features(lstm_model, data["X_seq_train"])
    emb_test = extract_temporal_features(lstm_model, data["X_seq_test"])

    X_hybrid_train = concatenate_features(emb_train, data["X_stat_train"])
    X_hybrid_test = concatenate_features(emb_test, data["X_stat_test"])

    print("\n2. Melatih Hybrid Random Forest Classifier...")
    rf_cfg = cfg["random_forest"]
    hybrid_rf = train_random_forest(
        X_train=X_hybrid_train,
        y_train=data["y_train"],
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_split=rf_cfg["min_samples_split"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        class_weight=rf_cfg["class_weight"],
        seed=seed,
    )

    # Baseline: Random Forest dengan fitur statis + rata-rata iklim saja
    X_base_train = np.hstack([data["X_seq_train"].mean(axis=1), data["X_stat_train"]])
    X_base_test = np.hstack([data["X_seq_test"].mean(axis=1), data["X_stat_test"]])
    baseline_rf = train_random_forest(
        X_train=X_base_train,
        y_train=data["y_train"],
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        class_weight=rf_cfg["class_weight"],
        seed=seed,
    )

    top_k = cfg["recommendation"]["top_k"]
    hybrid_metrics = compute_classification_metrics(
        data["y_test"],
        hybrid_rf.predict(X_hybrid_test),
        hybrid_rf.predict_proba(X_hybrid_test),
        hybrid_rf.classes_,
        top_k=top_k,
    )
    base_metrics = compute_classification_metrics(
        data["y_test"],
        baseline_rf.predict(X_base_test),
        baseline_rf.predict_proba(X_base_test),
        baseline_rf.classes_,
        top_k=top_k,
    )

    print("\n=== PERBANDINGAN EVALUASI ===")
    print("Baseline RF (Statis + Rata-rata Iklim):", base_metrics)
    print("Hybrid LSTM + RF                     :", hybrid_metrics)

    feature_names = [f"fitur_temporal_lstm_{i}" for i in range(lstm_cfg["embedding_dim"])] + cfg["data"]["static_features"]
    top_feats = get_feature_importance_summary(hybrid_rf, feature_names, top_n=8)
    print("\n=== FEATURE IMPORTANCE TERATAS (HYBRID) ===")
    for name, score in top_feats:
        print(f"  - {name}: {score}")

    sample_recs = recommend_top_k(hybrid_rf, X_hybrid_test[:3], data["label_encoder"], top_k=top_k)
    print("\n=== CONTOH REKOMENDASI TOP-3 (DATA UJI) ===")
    for idx, recs in enumerate(sample_recs, 1):
        print(f"  Sampel {idx}: {recs}")


if __name__ == "__main__":
    main()
