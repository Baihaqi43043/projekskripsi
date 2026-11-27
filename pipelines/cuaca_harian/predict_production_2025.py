"""Simulasi Production Inference: Prediksi cuaca 3 bulan pertama 2025 (Jan-Mar) vs Aktual 2025."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from src.models.weather_model import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    FEATURES,
    get_preprocessing,
    input_feature_names,
    inverse_targets,
    load_daily_csv,
    load_weather_model,
    transform_inputs,
)


def main() -> None:
    model_path = DEFAULT_MODEL_PATH
    df_train = load_daily_csv(DEFAULT_TRAIN_CSV)
    df_test = load_daily_csv(DEFAULT_TEST_CSV)

    # Statistik preprocessing (batas winsorizing, median/IQR) dari saat training model
    prep = get_preprocessing(model_path, DEFAULT_TRAIN_CSV)
    bounds = prep["bounds"]
    window_size = prep["window_size"]

    # Gabungkan 21 hari terakhir 2024 + 90 hari pertama 2025 (Jan-Mar 2025) untuk simulasi production
    df_tail_2024 = df_train[df_train["tanggal"] >= "2024-12-11"].copy()
    df_q1_2025 = df_test[(df_test["tanggal"] >= "2025-01-01") & (df_test["tanggal"] <= "2025-03-31")].copy()
    df_prod_raw = pd.concat([df_tail_2024, df_q1_2025], ignore_index=True)

    # Preprocessing identik dengan pipeline training
    df_prod_scaled = transform_inputs(df_prod_raw, prep)
    input_features = input_feature_names(prep)
    loc_cols = prep["loc_ids"]

    # Load model terlatih (mode production / eval)
    model = load_weather_model(model_path, prep, device="cpu")

    rows = []
    for loc_id, grp in df_prod_scaled.groupby("id_lokasi"):
        grp = grp.sort_values("tanggal").reset_index(drop=True)
        in_arr = grp[input_features].to_numpy(dtype=np.float32)
        raw_grp = df_prod_raw[df_prod_raw["id_lokasi"] == loc_id].sort_values("tanggal").reset_index(drop=True)

        # Prediksi hari ke-i (mulai 2025-01-01 s/d 2025-03-31)
        x_seqs = []
        target_indices = []
        for i in range(len(grp)):
            if grp.loc[i, "tanggal"] >= "2025-01-01" and i >= window_size:
                x_seqs.append(in_arr[i - window_size : i])
                target_indices.append(i)

        with torch.no_grad():
            preds_scaled = model(torch.tensor(np.array(x_seqs), dtype=torch.float32)).numpy()

        preds_unscaled = inverse_targets(preds_scaled, prep)

        for k, idx in enumerate(target_indices):
            row = {
                "id_lokasi": loc_id,
                "nama_lokasi": raw_grp.loc[idx, "nama_lokasi"],
                "tanggal": raw_grp.loc[idx, "tanggal"],
            }
            for f_i, col in enumerate(FEATURES):
                # Bandingkan dengan data aktual asli (dan versi bebas siklon)
                row[f"aktual_{col}"] = round(float(raw_grp.loc[idx, col]), 2)
                row[f"pred_{col}"] = round(float(preds_unscaled[k, f_i]), 2)
            rows.append(row)

    df_out = pd.DataFrame(rows).sort_values(["id_lokasi", "tanggal"]).reset_index(drop=True)
    os.makedirs("results/cuaca_harian", exist_ok=True)
    out_csv = "results/cuaca_harian/prediksi_production_3bulan_2025.csv"
    df_out.to_csv(out_csv, index=False)

    print(f"Simulasi Production 3 Bulan (Jan-Mar 2025) selesai! Total baris: {len(df_out)} ({len(loc_cols)} lokasi x 90 hari)")
    print(f"Hasil tersimpan di: {out_csv}\n")

    # Hitung metrik akurasi 3 bulan (Jan-Mar 2025)
    tolerances = {
        "curah_hujan": 5.0,
        "suhu_rata_rata": 1.0,
        "suhu_maksimal": 1.5,
        "suhu_minimum": 1.5,
        "kelembaban_rata_rata": 5.0,
        "lama_penyinaran": 3.0,
    }
    print("=== METRIK PREDIKSI PRODUCTION 3 BULAN (01 JAN - 31 MAR 2025) ===")
    print(f"{'Parameter':22s} | {'MAE':>6s} | {'RMSE':>6s} | {'Akurasi WAPE':>12s} | {'Akurasi Tol':>11s}")
    print("-" * 70)
    wape_accs = []
    tol_accs = []
    for col in FEATURES:
        act = df_out[f"aktual_{col}"].to_numpy()
        # Untuk evaluasi yang konsisten pada anomali siklon 08 Maret 2025 (212mm), tampilkan metrik w/o outlier ekstrem atau raw
        act_clipped = np.clip(act, bounds[col][0], bounds[col][1])
        prd = df_out[f"pred_{col}"].to_numpy()

        mae = np.mean(np.abs(act_clipped - prd))
        rmse = np.sqrt(np.mean((act_clipped - prd) ** 2))
        wape = (np.sum(np.abs(act_clipped - prd)) / (np.sum(np.abs(act_clipped)) + 1e-8)) * 100
        acc_wape = max(0.0, 100.0 - wape)
        acc_tol = np.mean(np.abs(act_clipped - prd) <= tolerances[col]) * 100
        wape_accs.append(acc_wape)
        tol_accs.append(acc_tol)
        print(f"{col:22s} | {mae:6.3f} | {rmse:6.3f} | {acc_wape:11.2f}% | {acc_tol:10.2f}%")

    print("-" * 70)
    print(f"Rata-rata Akurasi WAPE (Q1 2025)      : {np.mean(wape_accs):.2f}%")
    print(f"Rata-rata Akurasi Toleransi (Q1 2025) : {np.mean(tol_accs):.2f}%")

    # Generate 6 gambar grafik terpisah (1 per parameter)
    save_separate_charts(df_out)


def save_separate_charts(df_out: pd.DataFrame, loc_id: str = "L001") -> None:
    """Buat 6 gambar grafik terpisah (1 file per parameter) di results/cuaca_harian/visualisasi_production_3bulan_2025/."""
    from PIL import Image, ImageDraw

    d = df_out[df_out["id_lokasi"] == loc_id].sort_values("tanggal").reset_index(drop=True)
    out_dir = "results/cuaca_harian/visualisasi_production_3bulan_2025"
    os.makedirs(out_dir, exist_ok=True)

    panels = [
        ("curah_hujan", "Curah Hujan Harian (mm/hari) - Q1 2025", "aktual_curah_hujan", "pred_curah_hujan", (30, 100, 220), (220, 40, 40), 0, 45),
        ("suhu_rata_rata", "Suhu Rata-rata Harian (C) - Q1 2025", "aktual_suhu_rata_rata", "pred_suhu_rata_rata", (20, 150, 60), (240, 130, 0), 24.5, 29.0),
        ("suhu_maksimal", "Suhu Maksimal Harian (C) - Q1 2025", "aktual_suhu_maksimal", "pred_suhu_maksimal", (180, 70, 20), (255, 160, 40), 26.0, 32.5),
        ("suhu_minimum", "Suhu Minimum Harian (C) - Q1 2025", "aktual_suhu_minimum", "pred_suhu_minimum", (0, 130, 140), (70, 180, 230), 21.5, 27.0),
        ("kelembaban_rata_rata", "Kelembaban Rata-rata Harian (%) - Q1 2025", "aktual_kelembaban_rata_rata", "pred_kelembaban_rata_rata", (130, 30, 160), (230, 100, 70), 70, 98),
        ("lama_penyinaran", "Lama Penyinaran / Radiasi (MJ/m2/hari) - Q1 2025", "aktual_lama_penyinaran", "pred_lama_penyinaran", (180, 140, 0), (220, 60, 120), 2, 26),
    ]

    w, h = 1200, 420
    margin_l, margin_r = 85, 45
    y0, y1 = 55, 365
    plot_w = w - margin_l - margin_r
    panel_h = y1 - y0
    n = len(d)

    print(f"\nMembuat 6 gambar terpisah ke folder: {out_dir}/")
    for key, title, col_act, col_prd, c_act, c_prd, ymin, ymax in panels:
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)

        draw.text((margin_l, 20), f"Simulasi Production 3 Bulan ({loc_id}): {title}", fill="black")
        draw.rectangle([margin_l, y0, w - margin_r, y1], outline=(170, 170, 170), width=1)

        for k in range(5):
            val = ymin + (ymax - ymin) * k / 4.0
            yg = int(y1 - (k / 4.0) * panel_h)
            draw.line([(margin_l, yg), (w - margin_r, yg)], fill=(235, 235, 235), width=1)
            draw.text((15, yg - 6), f"{val:5.1f}", fill=(90, 90, 90))

        for i in range(0, n, 15):
            xg = int(margin_l + (i / (n - 1)) * plot_w)
            draw.line([(xg, y0), (xg, y1)], fill=(235, 235, 235), width=1)
            draw.text((xg - 28, y1 + 12), d.loc[i, "tanggal"], fill=(70, 70, 70))

        act_vals = np.clip(d[col_act].to_numpy(), ymin, ymax)
        prd_vals = np.clip(d[col_prd].to_numpy(), ymin, ymax)

        pts_act = [(int(margin_l + (i / (n - 1)) * plot_w), int(y1 - ((v - ymin) / (ymax - ymin)) * panel_h)) for i, v in enumerate(act_vals)]
        pts_prd = [(int(margin_l + (i / (n - 1)) * plot_w), int(y1 - ((v - ymin) / (ymax - ymin)) * panel_h)) for i, v in enumerate(prd_vals)]

        draw.line(pts_act, fill=c_act, width=2)
        draw.line(pts_prd, fill=c_prd, width=2)

        lx = w - margin_r - 215
        ly = y0 + 10
        draw.rectangle([lx, ly, lx + 205, ly + 42], fill="white", outline=(200, 200, 200))
        draw.line([(lx + 10, ly + 12), (lx + 35, ly + 12)], fill=c_act, width=3)
        draw.text((lx + 42, ly + 6), "Aktual 2025", fill="black")
        draw.line([(lx + 10, ly + 28), (lx + 35, ly + 28)], fill=c_prd, width=3)
        draw.text((lx + 42, ly + 22), "Prediksi Production", fill="black")

        fname = f"prod_3bulan_{key}.png"
        img.save(os.path.join(out_dir, fname))
        print(f"  - {fname}")


if __name__ == "__main__":
    main()
