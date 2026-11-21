"""Pipeline end-to-end Model Hybrid LSTM-Random Forest (menjalankan tahap LSTM lalu RF)."""

from __future__ import annotations

from train_lstm import main as run_lstm
from train_rf import main as run_rf


def main() -> None:
    run_lstm()
    print("\n" + "=" * 60 + "\n")
    run_rf()


if __name__ == "__main__":
    main()
