"""Gabungkan log & hasil training dari Google Colab (.zip atau folder) ke folder lokal `results/`."""

from __future__ import annotations

import argparse
from datetime import datetime
import glob
import os
import shutil
import tempfile
import zipfile

import pandas as pd

from src.utils.experiment_logger import HISTORY_COLUMNS, export_git_changelog, get_git_info


def _build_df_from_benchmark_csv(bench_csv: str, run_id: str) -> pd.DataFrame:
    """Bangun DataFrame format experiment_history dari rf_benchmark_metrics.csv saja."""
    bench_df = pd.read_csv(bench_csv)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for _, r in bench_df.iterrows():
        top3_col = [c for c in bench_df.columns if c.startswith("Top-")]
        top3_val = r[top3_col[0]] if top3_col else ""
        rows.append(
            {
                "run_id": f"{run_id}_{r['Model'].replace(' ', '_')}",
                "timestamp": now_str,
                "git_commit": "colab",
                "git_dirty": False,
                "stage": r["Model"],
                "environment": "google_colab",
                "train_samples": "",
                "test_samples": "",
                "num_classes": "",
                "hyperparameters": "{}",
                "loss_akhir": "",
                "accuracy": r.get("Accuracy", ""),
                "precision_weighted": r.get("Precision (Weighted)", ""),
                "recall_weighted": r.get("Recall (Weighted)", ""),
                "f1_weighted": r.get("F1 (Weighted)", ""),
                "top_3_accuracy": top3_val,
                "top_features": "",
                "duration_sec": "",
                "catatan": "Diimpor dari rf_benchmark_metrics.csv (Colab)",
            }
        )
    return pd.DataFrame(rows)


def import_from_folder(source_dir: str, results_dir: str = "results") -> None:
    """Import log Colab (dari zip terbaru, colab_experiment_history.csv, atau rf_benchmark_metrics.csv)."""
    zip_files = sorted(glob.glob(os.path.join(source_dir, "colab_rf_bundle*.zip")))
    if zip_files:
        latest_zip = zip_files[-1]
        print(f"Mengekstrak paket Colab terbaru: {latest_zip}")
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(latest_zip, "r") as zf:
                zf.extractall(tmpdir)
            import_from_folder(tmpdir, results_dir=results_dir)
        return

    colab_csv = os.path.join(source_dir, "colab_experiment_history.csv")
    bench_csv = os.path.join(source_dir, "rf_benchmark_metrics.csv")

    if os.path.exists(colab_csv):
        colab_df = pd.read_csv(colab_csv)
        run_id = str(colab_df.iloc[-1]["run_id"])
    elif os.path.exists(bench_csv):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        colab_df = _build_df_from_benchmark_csv(bench_csv, run_id)
    else:
        raise FileNotFoundError(
            f"Tidak menemukan colab_rf_bundle*.zip, colab_experiment_history.csv, atau rf_benchmark_metrics.csv di {source_dir}."
        )

    if colab_df.empty:
        raise ValueError("Data log Colab kosong.")

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
