"""Script lokal untuk melatih BiLSTM + Attention cuaca (adaptasi notebook kawanmu) pada data harian 20 lokasi."""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

from src.experiment_logger import export_git_changelog, log_experiment

FEATURES = [
    "curah_hujan",
    "suhu_rata_rata",
    "suhu_maksimal",
    "suhu_minimum",
    "kelembaban_rata_rata",
    "lama_penyinaran",
]
ENGINEERED_COLS = [
    "rentang_suhu",
    "doy_sin",
    "doy_cos",
    "roll_hujan_7d",
    "roll_lembab_7d",
    "roll_sinar_7d",
]


class WeatherDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class WeatherLSTM(nn.Module):
    """Bidirectional LSTM + LayerNorm + Temporal Attention -> 32-dim embedding."""

    def __init__(
        self,
        input_dim: int,
        hidden_1: int = 128,
        hidden_2: int = 64,
        embedding_dim: int = 32,
        output_dim: int = 6,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_1, batch_first=True, bidirectional=True)
        self.norm1 = nn.LayerNorm(hidden_1 * 2)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(input_size=hidden_1 * 2, hidden_size=hidden_2, batch_first=True, bidirectional=True)
        self.norm2 = nn.LayerNorm(hidden_2 * 2)
        self.dropout2 = nn.Dropout(dropout)

        self.attn_fc = nn.Linear(hidden_2 * 2, 1)
        self.fc_embed = nn.Linear(hidden_2 * 4, embedding_dim)
        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(embedding_dim, output_dim)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout1(self.norm1(out))
        out, _ = self.lstm2(out)
        out = self.dropout2(self.norm2(out))

        attn_weights = torch.softmax(self.attn_fc(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        last_step = out[:, -1, :]

        combined = torch.cat([last_step, context], dim=1)
        return self.relu(self.fc_embed(combined))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.extract_embedding(x))


def add_engineered_features(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    out["rentang_suhu"] = out["suhu_maksimal"] - out["suhu_minimum"]
    dt = pd.to_datetime(out["tanggal"])
    doy = dt.dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    out = out.sort_values(["id_lokasi", "tanggal"]).reset_index(drop=True)
    out["roll_hujan_7d"] = out.groupby("id_lokasi")["curah_hujan"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    out["roll_lembab_7d"] = out.groupby("id_lokasi")["kelembaban_rata_rata"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    out["roll_sinar_7d"] = out.groupby("id_lokasi")["lama_penyinaran"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    return out


def create_sequences_by_location(
    df_in_scaled: pd.DataFrame,
    target_scaled_arr: np.ndarray,
    in_cols: list[str],
    window_size: int = 21,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_list, y_list, loc_list, dates_list = [], [], [], []
    df_tmp = df_in_scaled.copy()
    df_tmp["_t_idx"] = np.arange(len(df_tmp))
    for loc_id, group in df_tmp.groupby("id_lokasi"):
        group_sorted = group.sort_values("tanggal").reset_index(drop=True)
        in_arr = group_sorted[in_cols].to_numpy(dtype=np.float32)
        out_arr = target_scaled_arr[group_sorted["_t_idx"].to_numpy()].astype(np.float32)
        dates_arr = group_sorted["tanggal"].to_numpy()

        for i in range(len(group_sorted) - window_size):
            x_list.append(in_arr[i : i + window_size])
            y_list.append(out_arr[i + window_size])
            loc_list.append(loc_id)
            dates_list.append(dates_arr[i + window_size])

    return (
        np.array(x_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(loc_list),
        np.array(dates_list),
    )


def main(epochs: int = 12, batch_size: int = 128, window_size: int = 21) -> None:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    daily_path = "data/raw/aceh_utara_agroklimat_harian.csv"
    df = pd.read_csv(daily_path)
    df = df.sort_values(["id_lokasi", "tanggal"]).drop_duplicates(subset=["id_lokasi", "tanggal"]).reset_index(drop=True)

    df_train = df[(df["tahun"] >= 2020) & (df["tahun"] <= 2024)].copy().reset_index(drop=True)
    df_test = df[df["tahun"] == 2025].copy().reset_index(drop=True)
    df_train[FEATURES] = df_train[FEATURES].interpolate(method="linear").bfill().ffill()
    df_test[FEATURES] = df_test[FEATURES].interpolate(method="linear").bfill().ffill()

    rain_p99 = df_train["curah_hujan"].quantile(0.99)
    df_train_proc = df_train.copy()
    df_test_proc = df_test.copy()
    df_train_proc["curah_hujan"] = np.log1p(df_train_proc["curah_hujan"].clip(lower=0, upper=rain_p99))
    df_test_proc["curah_hujan"] = np.log1p(df_test_proc["curah_hujan"].clip(lower=0))

    df_train_proc = add_engineered_features(df_train_proc)
    df_test_proc = add_engineered_features(df_test_proc)

    loc_ids = sorted(df_train_proc["id_lokasi"].unique())
    loc_cols = [f"loc_{loc}" for loc in loc_ids]
    for loc in loc_ids:
        df_train_proc[f"loc_{loc}"] = (df_train_proc["id_lokasi"] == loc).astype(np.float32)
        df_test_proc[f"loc_{loc}"] = (df_test_proc["id_lokasi"] == loc).astype(np.float32)

    input_features = FEATURES + ENGINEERED_COLS + loc_cols
    scale_in_cols = FEATURES + ENGINEERED_COLS

    input_scaler = RobustScaler()
    target_scaler = RobustScaler()
    df_train_scaled = df_train_proc.copy()
    df_test_scaled = df_test_proc.copy()
    df_train_scaled[scale_in_cols] = input_scaler.fit_transform(df_train_proc[scale_in_cols])
    df_test_scaled[scale_in_cols] = input_scaler.transform(df_test_proc[scale_in_cols])

    y_train_scaled = target_scaler.fit_transform(df_train_proc[FEATURES])
    y_test_scaled = target_scaler.transform(df_test_proc[FEATURES])

    x_train, y_train, locs_train, dates_train = create_sequences_by_location(
        df_train_scaled, y_train_scaled, input_features, window_size
    )
    x_test, y_test, locs_test, dates_test = create_sequences_by_location(
        df_test_scaled, y_test_scaled, input_features, window_size
    )
    print(f"[BiLSTM-Weather] Train: {x_train.shape}, Test: {x_test.shape}, Lokasi: {len(loc_ids)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeatherLSTM(input_dim=len(input_features), output_dim=len(FEATURES)).to(device)

    train_loader = DataLoader(WeatherDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(WeatherDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_test_loss = float("inf")
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/best_lstm_weather.pth"

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
            torch.save(model.state_dict(), best_model_path)
            mark = " [Best]"
        print(f"  Epoch [{ep:02d}/{epochs}] | Train Loss: {tr_loss:.5f} | Test Loss: {te_loss:.5f}{mark}")

    duration = time.perf_counter() - t0
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    @torch.no_grad()
    def _extract_batched(arr: np.ndarray) -> np.ndarray:
        embs = []
        for i in range(0, len(arr), 512):
            bt = torch.tensor(arr[i : i + 512], dtype=torch.float32).to(device)
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
            "dropout": 0.25,
            "epochs": epochs,
            "batch_size": batch_size,
            "window_size": window_size,
            "locations": len(loc_ids),
        },
        duration_sec=duration,
        loss_akhir=best_test_loss,
        catatan=f"BiLSTM+Attention lokal ({len(loc_ids)} lokasi, {window_size} hari)",
    )
    export_git_changelog()
    print(f"[BiLSTM-Weather] Selesai! Embedding disimpan ke: {export_path} ({emb_train.shape}, {emb_test.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih BiLSTM+Attention cuaca harian di lokal.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=21)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, window_size=args.window_size)

