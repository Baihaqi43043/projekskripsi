"""Gabungkan log & hasil training dari Google Colab ke folder lokal `results/`."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import pandas as pd

from src.experiment_logger import HISTORY_COLUMNS, export_git_changelog, get_git_info


def import_from_folder(source_dir: str, results_dir: str = "results") -> None:
    """Import colab_experiment_history.csv & artefak pendukung ke `results/` lokal."""
    colab_csv = os.path.join(source_dir, "colab_experiment_history.csv")
    if not os.path.exists(colab_csv):
        raise FileNotFoundError(
            f"Tidak menemukan {colab_csv}. Pastikan file hasil download Colab ada di folder tersebut."
        )

    colab_df = pd.read_csv(colab_csv)
    if colab_df.empty:
        raise ValueError("File colab_experiment_history.csv kosong.")

    # Isi git_commit dari repo lokal saat mengimpor hasil Colab
    git_info = get_git_info()
    colab_df["git_commit"] = git_info["commit"]
    colab_df["git_dirty"] = git_info["dirty"]

    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(results_dir, "experiment_history.csv")

    if os.path.exists(history_path):
        local_df = pd.read_csv(history_path)
        existing_ids = set(local_df["run_id"].astype(str))
        new_rows = colab_df[~colab_df["run_id"].astype(str).isin(existing_ids)]
        if new_rows.empty:
            print("Semua run_id dari Colab sudah tercatat sebelumnya di results/experiment_history.csv.")
        else:
            combined = pd.concat([local_df, new_rows[HISTORY_COLUMNS]], ignore_index=True)
            combined.to_csv(history_path, index=False)
            print(f"Berhasil menambahkan {len(new_rows)} baris log Colab ke {history_path}")
    else:
        colab_df[HISTORY_COLUMNS].to_csv(history_path, index=False)
        print(f"Membuat {history_path} dengan {len(colab_df)} baris log Colab.")

    run_id = str(colab_df.iloc[-1]["run_id"])
    run_dir = os.path.join(results_dir, "runs", f"{run_id}_Hybrid_RF_Tuned_Colab")
    os.makedirs(run_dir, exist_ok=True)

    for fname in [
        "colab_summary.json",
        "rf_benchmark_metrics.csv",
        "rf_colab_plots.png",
        "hybrid_rf_tuned.pkl",
    ]:
        src_file = os.path.join(source_dir, fname)
        if os.path.exists(src_file):
            dest = os.path.join("models", fname) if fname.endswith(".pkl") else os.path.join(run_dir, fname)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src_file, dest)
            print(f"  Disalin -> {dest}")

    export_git_changelog(os.path.join(results_dir, "changelog_kode.txt"))
    print(f"Selesai! Arsip run Colab tersimpan di: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import log hasil training Google Colab ke lokal.")
    parser.add_argument(
        "--from-dir",
        default=os.path.expanduser("~/Downloads"),
        help="Folder tempat file download dari Google Colab berada (default: ~/Downloads)",
    )
    args = parser.parse_args()
    import_from_folder(args.from_dir)


if __name__ == "__main__":
    main()

