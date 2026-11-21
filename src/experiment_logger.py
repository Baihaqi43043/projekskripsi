"""Logger otomatis perubahan kode (Git) dan hasil eksperimen training model untuk dokumentasi penelitian."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

HISTORY_COLUMNS = [
    "run_id",
    "timestamp",
    "git_commit",
    "git_dirty",
    "stage",
    "environment",
    "train_samples",
    "test_samples",
    "num_classes",
    "hyperparameters",
    "loss_akhir",
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "top_3_accuracy",
    "top_features",
    "duration_sec",
    "catatan",
]


def get_git_info() -> dict[str, Any]:
    """Ambil hash commit Git, branch aktif, dan status perubahan yang belum di-commit."""
    info = {"commit": "unknown", "branch": "unknown", "dirty": False}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["dirty"] = bool(status)
    except Exception:
        pass
    return info


def _safe_float(val: Any) -> float | None:
    """Konversi aman ke float (mengembalikan None jika kosong atau string non-angka)."""
    if val in ("", None):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _format_param_diff(prev_params: dict[str, Any], curr_params: dict[str, Any]) -> str:
    """Bandingkan hyperparameter run saat ini dengan run sebelumnya."""
    if not curr_params:
        return "(parameter tidak tercatat)"
    if not prev_params:
        return ", ".join(f"{k}={v}" for k, v in curr_params.items())
    diffs = []
    all_keys = list(dict.fromkeys(list(prev_params.keys()) + list(curr_params.keys())))
    for k in all_keys:
        v_old = prev_params.get(k, "-")
        v_new = curr_params.get(k, "-")
        if v_old != v_new:
            diffs.append(f"{k}: {v_old} -> {v_new}")
    return "; ".join(diffs) if diffs else "Tidak ada perubahan parameter (sama seperti run sebelumnya)"


def export_git_changelog(
    out_path: str = "results/changelog_kode.txt",
    limit: int = 50,
    history_path: str = "results/experiment_history.csv",
) -> None:
    """Ekspor log perubahan parameter & dampaknya ke akurasi + riwayat commit Git."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    param_lines: list[str] = [
        "================================================================================",
        "LOG PERUBAHAN PARAMETER & DAMPAK TERHADAP AKURASI MODEL",
        f"Diperbarui: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "================================================================================",
    ]

    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # Cari run klasifikasi terbaik berdasarkan F1 lalu Accuracy (abaikan run regresi cuaca)
        scored_rows = [
            r
            for r in rows
            if "LSTM_Weather" not in r.get("stage", "")
            and _safe_float(r.get("f1_weighted")) is not None
            and _safe_float(r.get("accuracy")) is not None
        ]
        best_key = None
        if scored_rows:
            best_row = max(
                scored_rows,
                key=lambda r: (
                    _safe_float(r.get("f1_weighted")) or 0.0,
                    _safe_float(r.get("accuracy")) or 0.0,
                ),
            )
            best_key = (best_row["run_id"], best_row["stage"])
            param_lines.append(
                f"MODEL TERBAIK SAAT INI : {best_row['stage']} ({best_row['environment']}) | "
                f"Acc={best_row['accuracy']} | F1={best_row['f1_weighted']} | Top-3={best_row['top_3_accuracy']}"
            )
            param_lines.append(f"Parameter Terbaik      : {best_row['hyperparameters']}")
            param_lines.append("-" * 80)

        prev_by_group: dict[str, dict[str, Any]] = {}
        for idx, r in enumerate(rows, 1):
            stage = r.get("stage", "")
            group = (
                "LSTM_Weather"
                if "LSTM_Weather" in stage
                else ("LSTM" if stage == "LSTM" else ("Baseline" if "Baseline" in stage else "Hybrid_RF"))
            )
            try:
                curr_params = json.loads(r.get("hyperparameters") or "{}")
            except json.JSONDecodeError:
                curr_params = {}

            prev_entry = prev_by_group.get(group, {})
            prev_params = prev_entry.get("params", {})
            param_diff = _format_param_diff(prev_params, curr_params)

            if stage == "LSTM":
                metric_str = f"Loss Akhir={r.get('loss_akhir', '-')}"
                prev_by_group[group] = {"params": curr_params}
            elif "LSTM_Weather" in stage:
                metric_str = (
                    f"Loss={r.get('loss_akhir', '-')}, {r.get('precision_weighted', '')}, "
                    f"{r.get('recall_weighted', '')}, {r.get('f1_weighted', '')}, {r.get('top_3_accuracy', '')}"
                )
                prev_by_group[group] = {"params": curr_params}
            else:
                acc = _safe_float(r.get("accuracy"))
                f1 = _safe_float(r.get("f1_weighted"))
                top3 = _safe_float(r.get("top_3_accuracy"))
                prev_acc = prev_entry.get("acc")
                prev_f1 = prev_entry.get("f1")
                delta_acc = f" ({acc - prev_acc:+.4f})" if (acc is not None and prev_acc is not None) else ""
                delta_f1 = f" ({f1 - prev_f1:+.4f})" if (f1 is not None and prev_f1 is not None) else ""
                metric_str = f"Acc={acc}{delta_acc}, F1={f1}{delta_f1}, Top-3={top3}"
                prev_by_group[group] = {"params": curr_params, "acc": acc, "f1": f1}

            tag = " [TERBAIK]" if (r.get("run_id"), stage) == best_key else ""
            param_lines.append(f"[{idx}] {r.get('timestamp')} | {stage} ({r.get('environment')}) | Commit: {r.get('git_commit')}{tag}")
            param_lines.append(f"    Perubahan Param : {param_diff}")
            param_lines.append(f"    Hasil           : {metric_str}")
            if r.get("catatan"):
                param_lines.append(f"    Catatan         : {r.get('catatan')}")
    else:
        param_lines.append("Belum ada data di results/experiment_history.csv.")

    try:
        log_output = subprocess.check_output(
            [
                "git",
                "log",
                f"-n{limit}",
                "--pretty=format:%h | %ad | %an | %s",
                "--date=format:%Y-%m-%d %H:%M:%S",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        log_output = f"Gagal membaca git log: {exc}"

    git_section = [
        "",
        "================================================================================",
        "RIWAYAT COMMIT GIT",
        "================================================================================",
        "Commit  | Tanggal             | Author       | Pesan Commit",
        "--------------------------------------------------------------------------------",
        log_output,
        "",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(param_lines + git_section))


def save_rf_artifacts(
    run_dir: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    class_names: list[str],
    feature_names: list[str],
    feature_importances: np.ndarray,
) -> None:
    """Simpan classification_report.txt, confusion_matrix.png, dan feature_importance.png."""
    os.makedirs(run_dir, exist_ok=True)

    report = classification_report(
        y_true, y_pred, labels=classes, target_names=class_names, zero_division=0
    )
    with open(os.path.join(run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred, labels=classes)
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix")
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=150)
        plt.close(fig)

        paired = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)[:15]
        names = [p[0] for p in paired][::-1]
        vals = [float(p[1]) for p in paired][::-1]
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.barh(names, vals, color="#2b7bba")
        ax2.set_xlabel("Importance")
        ax2.set_title("Top-15 Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "feature_importance.png"), dpi=150)
        plt.close(fig2)
    except ImportError:
        pass


def log_experiment(
    stage: str,
    train_samples: int,
    test_samples: int,
    num_classes: int,
    hyperparameters: dict[str, Any],
    duration_sec: float,
    metrics: dict[str, Any] | None = None,
    loss_akhir: float | None = None,
    top_features: list[tuple[str, float]] | None = None,
    catatan: str = "",
    environment: str = "local",
    config_path: str = "configs/model_config.yaml",
    results_dir: str = "results",
) -> str:
    """Catat satu run eksperimen ke results/experiment_history.csv dan folder run detail."""
    now = datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    git_info = get_git_info()
    metrics = metrics or {}

    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(results_dir, "experiment_history.csv")
    run_dir = os.path.join(results_dir, "runs", f"{run_id}_{stage}")
    os.makedirs(run_dir, exist_ok=True)

    top_feat_str = (
        "; ".join([f"{name}:{score}" for name, score in top_features[:5]])
        if top_features
        else ""
    )

    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "stage": stage,
        "environment": environment,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "num_classes": num_classes,
        "hyperparameters": json.dumps(hyperparameters, ensure_ascii=False),
        "loss_akhir": round(float(loss_akhir), 4) if loss_akhir is not None else "",
        "accuracy": metrics.get("accuracy", ""),
        "precision_weighted": metrics.get("precision_weighted", ""),
        "recall_weighted": metrics.get("recall_weighted", ""),
        "f1_weighted": metrics.get("f1_weighted", ""),
        "top_3_accuracy": metrics.get("top_3_accuracy", ""),
        "top_features": top_feat_str,
        "duration_sec": round(float(duration_sec), 2),
        "catatan": catatan,
    }

    write_header = not os.path.exists(history_path)
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    summary_payload = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git": git_info,
        "stage": stage,
        "environment": environment,
        "dataset": {
            "train_samples": train_samples,
            "test_samples": test_samples,
            "num_classes": num_classes,
        },
        "hyperparameters": hyperparameters,
        "loss_akhir": loss_akhir,
        "metrics": metrics,
        "top_features": top_features or [],
        "duration_sec": round(float(duration_sec), 2),
        "catatan": catatan,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    if os.path.exists(config_path):
        shutil.copy2(config_path, os.path.join(run_dir, "config_snapshot.yaml"))

    return run_dir

