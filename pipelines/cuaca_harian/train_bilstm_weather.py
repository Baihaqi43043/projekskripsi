"""Latih BiLSTM + Attention cuaca harian secara lokal (setara notebook `training_lstm_cuaca.ipynb`).

Output:
- models/best_lstm_weather.pth           (bobot model terbaik)
- models/best_lstm_weather_preproc.json  (batas winsorizing + median/IQR; dipakai inference & dashboard)
- data/processed/lstm_weather_embeddings.npz  (embedding 32-dim untuk tahap Random Forest)
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.experiment_logger import export_git_changelog, log_experiment
from src.models.weather_model import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    FEATURES,
    WINDOW_SIZE,
    WeatherLSTM,
    fit_preprocessing,
    input_feature_names,
    load_daily_csv,
    make_sequences,
    preproc_path_for,
    save_preprocessing,
    transform_inputs,
    transform_targets,
)


def main(
    epochs: int = 50,
    batch_size: int = 32,
    window_size: int = WINDOW_SIZE,
    train_csv: str = DEFAULT_TRAIN_CSV,
    test_csv: str = DEFAULT_TEST_CSV,
    model_path: str = DEFAULT_MODEL_PATH,
) -> None:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    df_train = load_daily_csv(train_csv)
    df_test = load_daily_csv(test_csv)

    prep = fit_preprocessing(df_train, window_size=window_size)
    x_train, y_train, locs_train, dates_train = make_sequences(
        transform_inputs(df_train, prep), transform_targets(df_train, prep), prep
    )
    x_test, y_test, locs_test, dates_test = make_sequences(
        transform_inputs(df_test, prep), transform_targets(df_test, prep), prep
    )
    print(f"[BiLSTM-Weather] Train: {x_train.shape}, Test: {x_test.shape}, Lokasi: {len(prep['loc_ids'])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = 0.25
    model = WeatherLSTM(input_dim=len(input_feature_names(prep)), output_dim=len(FEATURES), dropout=dropout).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), batch_size=batch_size, shuffle=False
    )

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    save_preprocessing(prep, preproc_path_for(model_path))

    best_test_loss = float("inf")
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item() * len(bx)
        tr_loss /= len(x_train)

        # Catatan: sama seperti notebook, data test dipakai untuk memilih model terbaik & scheduler.
        model.eval()
        te_loss = 0.0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                te_loss += criterion(model(bx), by).item() * len(bx)
        te_loss /= len(x_test)
        scheduler.step(te_loss)

        mark = ""
        if te_loss < best_test_loss:
            best_test_loss = te_loss
            torch.save(model.state_dict(), model_path)
            mark = " [Best]"
        print(f"  Epoch [{ep:02d}/{epochs}] | Train Loss: {tr_loss:.5f} | Test Loss: {te_loss:.5f}{mark}")

    duration = time.perf_counter() - t0
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    @torch.no_grad()
    def _extract_batched(arr: np.ndarray) -> np.ndarray:
        embs = []
        for i in range(0, len(arr), 512):
            bt = torch.from_numpy(arr[i : i + 512]).to(device)
            embs.append(model.extract_embedding(bt).cpu().numpy())
        return np.vstack(embs)

    emb_train = _extract_batched(x_train)
    emb_test = _extract_batched(x_test)

    os.makedirs("data/processed", exist_ok=True)
    export_path = "data/processed/lstm_weather_embeddings.npz"
    np.savez_compressed(
        export_path,
        emb_train=emb_train,
        emb_test=emb_test,
        dates_train=dates_train,
        dates_test=dates_test,
        locs_train=locs_train,
        locs_test=locs_test,
        features=np.array(FEATURES),
    )

    log_experiment(
        stage="LSTM_Weather",
        train_samples=len(x_train),
        test_samples=len(x_test),
        num_classes=len(FEATURES),
        hyperparameters={
            "hidden_1": 128,
            "hidden_2": 64,
            "embedding_dim": 32,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 0.001,
            "window_size": window_size,
            "locations": len(prep["loc_ids"]),
            "train_csv": train_csv,
        },
        duration_sec=duration,
        loss_akhir=best_test_loss,
        catatan=f"BiLSTM+Attention lokal ({len(prep['loc_ids'])} lokasi, {window_size} hari)",
    )
    export_git_changelog()
    print(f"[BiLSTM-Weather] Selesai! Model: {model_path} | Preprocessing: {preproc_path_for(model_path)}")
    print(f"[BiLSTM-Weather] Embedding disimpan ke: {export_path} ({emb_train.shape}, {emb_test.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih BiLSTM+Attention cuaca harian di lokal.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        window_size=args.window_size,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        model_path=args.model_path,
    )
