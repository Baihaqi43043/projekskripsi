"""End-to-end pipeline runner for Hybrid LSTM-Random Forest Crop Variety Recommender."""

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


def assign_prototype_variety_labels(
    climate_df: pd.DataFrame,
    soil_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assign agro-ecological prototype variety labels when official Dinas labels are not yet available.

    Replace this step with actual Dinas Pertanian / Kementan ground-truth variety labels
    once field data is collected.
    """
    merged = climate_df.merge(soil_df, on="location_id", how="left")
    labels = []
    for _, row in merged.iterrows():
        rain = row["PRECTOTCORR"]
        temp = row["T2M"]
        ph = row["phh2o"]

        if rain >= 200 and ph >= 6.0:
            labels.append("Padi_Inpari_32")
        elif rain >= 150 and temp >= 26.5:
            labels.append("Padi_Ciherang")
        elif rain >= 120 and ph < 6.0:
            labels.append("Padi_Inpari_42")
        elif 80 <= rain < 150 and temp >= 26.0:
            labels.append("Jagung_BISI_18")
        elif 60 <= rain < 120:
            labels.append("Jagung_NK_212")
        else:
            labels.append("Kedelai_Anjasmoro")

    climate_df = climate_df.copy()
    climate_df["variety"] = labels
    return climate_df


def main() -> None:
    with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["project"]["seed"]
    raw_dir = cfg["data"]["raw_dir"]
    climate_path = os.path.join(raw_dir, "aceh_utara_climate_monthly.csv")
    soil_path = os.path.join(raw_dir, "aceh_utara_soil_static.csv")

    if os.path.exists(climate_path) and os.path.exists(soil_path):
        print("Loading cached Aceh Utara dataset from data/raw/...")
        climate_df = pd.read_csv(climate_path)
        soil_df = pd.read_csv(soil_path)
    else:
        print("Fetching real agroclimate (NASA POWER) & soil (SoilGrids) for Aceh Utara...")
        climate_df, soil_df = collect_aceh_utara_sample(cfg)

    climate_df = assign_prototype_variety_labels(climate_df, soil_df)

    data = build_sequences_and_static(
        climate_df=climate_df,
        static_df=soil_df,
        climate_cols=cfg["data"]["climate_features"],
        static_cols=cfg["data"]["static_features"],
        target_col="variety",
        seq_len=cfg["data"]["sequence_length"],
        train_ratio=cfg["data"]["train_ratio"],
    )

    num_classes = len(data["label_encoder"].classes_)
    print(f"\nDataset ready -> Train sequences: {data['X_seq_train'].shape}, "
          f"Test sequences: {data['X_seq_test'].shape}, Classes: {num_classes}")

    print("\n1. Training LSTM Temporal Encoder...")
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

    print("\n2. Training Hybrid Random Forest Classifier...")
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

    # Baseline: Random Forest on static + mean-aggregated climate features only
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

    print("\n=== EVALUATION COMPARISON ===")
    print("Baseline RF (Static + Mean Climate):", base_metrics)
    print("Hybrid LSTM + RF                   :", hybrid_metrics)

    feature_names = [f"lstm_feat_{i}" for i in range(lstm_cfg["embedding_dim"])] + cfg["data"]["static_features"]
    top_feats = get_feature_importance_summary(hybrid_rf, feature_names, top_n=8)
    print("\n=== TOP FEATURE IMPORTANCES (HYBRID) ===")
    for name, score in top_feats:
        print(f"  - {name}: {score}")

    sample_recs = recommend_top_k(hybrid_rf, X_hybrid_test[:3], data["label_encoder"], top_k=top_k)
    print("\n=== SAMPLE TOP-3 RECOMMENDATIONS (TEST SET) ===")
    for idx, recs in enumerate(sample_recs, 1):
        print(f"  Sample {idx}: {recs}")


if __name__ == "__main__":
    main()

