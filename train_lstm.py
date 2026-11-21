"""Script khusus Tim LSTM: melatih LSTM encoder, ekspor vektor 32-dim, dan catat log otomatis."""

from __future__ import annotations

import os
import time
import numpy as np
import torch

from src.dataset_loader import load_config, prepare_shared_dataset
from src.experiment_logger import export_git_changelog, log_experiment
from src.lstm_encoder import extract_temporal_features, train_lstm_encoder


def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    data = prepare_shared_dataset(cfg)

    classes = data["label_encoder"].classes_
    num_classes = len(classes)
    print(
        f"\n[LSTM] Dataset siap -> Train: {data['X_seq_train'].shape}, "
        f"Test: {data['X_seq_test'].shape}, Kelas Varietas: {num_classes}"
    )

    lstm_cfg = cfg["lstm"]
    print("\n[LSTM] Melatih Temporal Encoder (128 -> 64 -> 32-dim)...")
    t0 = time.perf_counter()
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
    duration = time.perf_counter() - t0

    emb_train = extract_temporal_features(lstm_model, data["X_seq_train"])
    emb_test = extract_temporal_features(lstm_model, data["X_seq_test"])

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "lstm_encoder.pt")
    torch.save(lstm_model.state_dict(), model_path)

    processed_dir = cfg["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    emb_path = os.path.join(processed_dir, "lstm_embeddings.npz")
    np.savez(
        emb_path,
        emb_train=emb_train,
        emb_test=emb_test,
        X_seq_train=data["X_seq_train"],
        X_seq_test=data["X_seq_test"],
        X_stat_train=data["X_stat_train"],
        X_stat_test=data["X_stat_test"],
        y_train=data["y_train"],
        y_test=data["y_test"],
        classes=classes,
    )

    loss_akhir = getattr(lstm_model, "training_history_", {}).get("loss_akhir")
    run_dir = log_experiment(
        stage="LSTM",
        train_samples=len(data["y_train"]),
        test_samples=len(data["y_test"]),
        num_classes=num_classes,
        hyperparameters=lstm_cfg,
        duration_sec=duration,
        loss_akhir=loss_akhir,
        catatan=f"seq_len={cfg['data']['sequence_length']}, seed={seed}",
    )
    export_git_changelog()

    print(f"\n[LSTM] Selesai! Bobot disimpan ke: {model_path}")
    print(f"[LSTM] Vektor temporal 32-dim disimpan ke: {emb_path} (shape train={emb_train.shape}, test={emb_test.shape})")
    print(f"[LOG] Tercatat otomatis di results/experiment_history.csv & {run_dir}")


if __name__ == "__main__":
    main()
