"""2-layer LSTM temporal feature encoder (128 -> 64 -> 32-dim embedding)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TemporalLSTMEncoder(nn.Module):
    """LSTM network matching proposal architecture: 128 -> 64 -> Dense(32)."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_1: int = 128,
        hidden_2: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_1, hidden_2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc_embed = nn.Linear(hidden_2, embedding_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        return self.relu(self.fc_embed(out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encode(x)
        return self.classifier(emb)


def train_lstm_encoder(
    X_seq_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    hidden_1: int = 128,
    hidden_2: int = 64,
    embedding_dim: int = 32,
    dropout: float = 0.2,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> TemporalLSTMEncoder:
    """Train LSTM encoder on agroclimate sequences."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalLSTMEncoder(
        input_dim=X_seq_train.shape[-1],
        num_classes=num_classes,
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(device)

    dataset = TensorDataset(
        torch.tensor(X_seq_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history: list[float] = []
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  [LSTM] Epoch {epoch:02d}/{epochs} - Loss: {avg_loss:.4f}")

    model.training_history_ = {
        "loss_awal": loss_history[0] if loss_history else None,
        "loss_akhir": loss_history[-1] if loss_history else None,
        "loss_per_epoch": loss_history,
    }
    return model


def train_forecast_encoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_1: int = 128,
    hidden_2: int = 64,
    embedding_dim: int = 32,
    dropout: float = 0.2,
    max_epochs: int = 300,
    patience: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> TemporalLSTMEncoder:
    """Latih encoder LSTM (arsitektur proposal 128 -> 64 -> 32) untuk MERAMAL iklim bulan berikutnya.

    Encoder tidak pernah melihat label varietas (self-supervised), sehingga embedding 32-dim
    yang diekstrak tidak bocor ke tahap Random Forest. Early stopping & ReduceLROnPlateau
    memakai data VALIDASI (bukan data test).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalLSTMEncoder(
        input_dim=X_train.shape[-1],
        num_classes=y_train.shape[-1],  # head regresi: jumlah variabel iklim yang diramal
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(device)

    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
    )
    x_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, patience // 3))

    history: list[dict[str, float]] = []
    best_val, best_state, best_epoch, wait = float("inf"), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(bx)
        train_loss = total / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val_t), y_val_t).item())
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-6:
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.training_history_ = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "loss_per_epoch": history,
    }
    return model


def train_lstm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    hidden_1: int = 128,
    hidden_2: int = 64,
    embedding_dim: int = 32,
    dropout: float = 0.2,
    max_epochs: int = 300,
    patience: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> TemporalLSTMEncoder:
    """Baseline single model 'LSTM saja': sekuens iklim -> softmax varietas (early stopping di validasi)."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalLSTMEncoder(
        input_dim=X_train.shape[-1], num_classes=num_classes, hidden_1=hidden_1,
        hidden_2=hidden_2, embedding_dim=embedding_dim, dropout=dropout,
    ).to(device)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True, generator=gen,
    )
    x_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val, best_state, wait = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val_t), y_val_t).item())
        if val_loss < best_val - 1e-6:
            best_val, wait = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_proba_lstm(model: TemporalLSTMEncoder, X_seq: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    logits = model(torch.tensor(X_seq, dtype=torch.float32).to(device))
    return torch.softmax(logits, dim=1).cpu().numpy()


@torch.no_grad()
def predict_forecast(model: TemporalLSTMEncoder, X_seq: np.ndarray) -> np.ndarray:
    """Prediksi iklim bulan berikutnya (skala ter-normalisasi) dari encoder peramal."""
    device = next(model.parameters()).device
    model.eval()
    return model(torch.tensor(X_seq, dtype=torch.float32).to(device)).cpu().numpy()


@torch.no_grad()
def extract_temporal_features(model: TemporalLSTMEncoder, X_seq: np.ndarray) -> np.ndarray:
    """Extract 32-dimensional temporal feature vectors from trained LSTM."""
    device = next(model.parameters()).device
    model.eval()
    x_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    return model.encode(x_tensor).cpu().numpy()

