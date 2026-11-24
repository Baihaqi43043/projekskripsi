"""LSTM dilatih per HARI (maks 5 epoch), lalu keluarannya dirata-rata menjadi BULANAN.

Alur:
1. Data harian 20 lokasi (2020-2025). Preprocessing: hujan log1p -> outlier IQR -> Z-Score;
   indeks musim sin/cos hari-dalam-tahun. Statistik dari data train (tanggal <= 2023) saja.
2. Input LSTM = 30 hari sebelumnya (30 × 8). Arsitektur proposal: LSTM 128 -> LSTM 64 -> Dense 32.
   Tugas latihan: meramal 6 variabel iklim BESOK. Maks 5 epoch, dipilih epoch terbaik di validasi 2024.
3. Vektor 32 angka dihitung untuk setiap hari, lalu DIRATA-RATA per (lokasi, bulan).
   Untuk bulan tanam t, RF memakai rata-rata bulan t-1 (tidak bocor ke masa depan).
4. Ramalan harian juga dirata-rata per bulan dan dibandingkan dengan iklim aktual bulanan.

Output:
  data/processed/lstm_harian/embedding_bulanan.csv  (id_lokasi, tahun, bulan, lstm_0..31)
  data/processed/lstm_harian/evaluasi_ramalan.json
  models/lstm_harian.pt
Lalu jalankan:
  python buat_data_rf_varietas.py --lstm-harian data/processed/lstm_harian/embedding_bulanan.csv --out data/processed/rf_varietas_harian
  python train_rf_varietas.py --data-dir data/processed/rf_varietas_harian --label Harian
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from src.dataset_loader import filter_active_locations, load_config
from src.lstm_encoder import train_forecast_encoder

F = ["curah_hujan", "suhu_rata_rata", "suhu_maksimal", "suhu_minimum", "kelembaban_rata_rata", "lama_penyinaran"]
WINDOW = 30
MAX_EPOCH = 5
BATCH = 64
SEED = 42
TRAIN_END, VAL_YEAR, TEST_YEAR = 2023, 2024, 2025
OUT_DIR = "data/processed/lstm_harian"


def siapkan(cfg) -> tuple[pd.DataFrame, dict]:
    raw = cfg["data"]["raw_dir"]
    d = pd.read_csv(os.path.join(raw, "aceh_utara_agroklimat_harian.csv"))
    soil = pd.read_csv(os.path.join(raw, "aceh_utara_tanah_statis.csv"))
    d, _ = filter_active_locations(d, soil, cfg)
    d = d.sort_values(["id_lokasi", "tanggal"]).reset_index(drop=True)
    d[F] = d.groupby("id_lokasi")[F].transform(lambda s: s.interpolate().bfill().ffill())

    x = d[F].astype(float).copy()
    x["curah_hujan"] = np.log1p(x["curah_hujan"].clip(lower=0))
    fit = d["tahun"] <= TRAIN_END
    q1, q3 = x[fit].quantile(0.25), x[fit].quantile(0.75)
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    capped = x.clip(lower=lo, upper=hi, axis=1)
    mu, sd = capped[fit].mean(), capped[fit].std(ddof=0)
    z = (capped - mu) / sd
    doy = pd.to_datetime(d["tanggal"]).dt.dayofyear
    z["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    z["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d = pd.concat([d[["id_lokasi", "tanggal", "tahun", "bulan"] + F], z.add_prefix("z_")], axis=1)
    stat = {"iqr_bawah": lo.round(4).to_dict(), "iqr_atas": hi.round(4).to_dict(),
            "mean": mu.round(4).to_dict(), "std": sd.round(4).to_dict(),
            "n_dipotong": int(((x < lo) | (x > hi)).sum().sum()), "catatan": "curah_hujan dalam skala log1p"}
    return d, stat


def jendela(d: pd.DataFrame):
    in_cols = [f"z_{c}" for c in F] + ["z_doy_sin", "z_doy_cos"]
    tgt_cols = [f"z_{c}" for c in F]
    X, Y, meta = [], [], []
    for loc, g in d.groupby("id_lokasi", sort=True):
        a = g[in_cols].to_numpy(np.float32)
        t = g[tgt_cols].to_numpy(np.float32)
        info = g[["id_lokasi", "tanggal", "tahun", "bulan"]].to_numpy()
        for i in range(WINDOW, len(g)):
            X.append(a[i - WINDOW:i])
            Y.append(t[i])
            meta.append(info[i])
    meta = pd.DataFrame(meta, columns=["id_lokasi", "tanggal", "tahun", "bulan"])
    return np.stack(X), np.stack(Y), meta


@torch.no_grad()
def jalankan(model, X: np.ndarray, batch: int = 4096):
    emb, pred = [], []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch])
        emb.append(model.encode(x).numpy())
        pred.append(model(x).numpy())
    return np.vstack(emb), np.vstack(pred)


def balik_skala(z: np.ndarray, stat: dict) -> np.ndarray:
    out = z * np.array([stat["std"][c] for c in F]) + np.array([stat["mean"][c] for c in F])
    out[:, 0] = np.expm1(np.clip(out[:, 0], 0, None))
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    lc = cfg["lstm"]
    d, stat = siapkan(cfg)
    X, Y, meta = jendela(d)
    th = meta["tahun"].astype(int).to_numpy()
    tr, va, te = th <= TRAIN_END, th == VAL_YEAR, th == TEST_YEAR
    print(f"[LSTM-Harian] lokasi={d['id_lokasi'].nunique()} | jendela {WINDOW} hari | "
          f"train={tr.sum()} val={va.sum()} test={te.sum()} sekuens | maks {MAX_EPOCH} epoch, batch {BATCH}")

    t0 = time.perf_counter()
    model = train_forecast_encoder(
        X[tr], Y[tr], X[va], Y[va], hidden_1=lc["hidden_size_1"], hidden_2=lc["hidden_size_2"],
        embedding_dim=lc["embedding_dim"], dropout=lc["dropout"], max_epochs=MAX_EPOCH, patience=MAX_EPOCH,
        batch_size=BATCH, learning_rate=lc["learning_rate"], seed=SEED,
    )
    durasi = time.perf_counter() - t0
    h = model.training_history_
    for r in h["loss_per_epoch"]:
        print(f"  epoch {r['epoch']}: train MSE={r['train_loss']:.4f}  val MSE={r['val_loss']:.4f}")
    print(f"  epoch terpilih: {h['best_epoch']} | durasi {durasi:.0f} dtk")

    emb, pred_z = jalankan(model, X)

    # Rata-rata vektor harian -> bulanan
    E = pd.DataFrame(emb, columns=[f"lstm_{i}" for i in range(emb.shape[1])])
    E = pd.concat([meta[["id_lokasi", "tahun", "bulan"]], E], axis=1)
    bulanan = E.groupby(["id_lokasi", "tahun", "bulan"], as_index=False).mean()
    bulanan[[c for c in bulanan.columns if c.startswith("lstm_")]] = bulanan.filter(like="lstm_").round(6)

    # Evaluasi ramalan pada test 2025: harian dan rata-rata bulanan, dibanding persistence (besok = hari ini)
    akt = meta.copy()
    akt[F] = d.set_index(["id_lokasi", "tanggal"]).loc[list(zip(meta.id_lokasi, meta.tanggal)), F].to_numpy()
    prd = pd.DataFrame(balik_skala(pred_z, stat), columns=F)
    pers = akt.groupby("id_lokasi")[F].shift(1)
    m = te & pers.notna().all(axis=1).to_numpy()
    ev = {"epoch_terpilih": h["best_epoch"], "epoch_dijalankan": h["epochs_run"],
          "riwayat_loss": h["loss_per_epoch"], "durasi_detik": round(durasi, 1), "preprocessing": stat,
          "mae_harian_test": {}, "mae_bulanan_test": {}}
    gb = ["id_lokasi", "tahun", "bulan"]
    for c in F:
        ev["mae_harian_test"][c] = {"lstm": round(float(np.mean(np.abs(prd[c][m] - akt[c][m]))), 3),
                                    "persistence": round(float(np.mean(np.abs(pers[c][m] - akt[c][m]))), 3)}
        # Rata-rata bulanan dari ramalan 1-hari-ke-depan (bukan ramalan bulan depan): laporkan MAE & bias
        a_b = akt[m].groupby(gb)[c].mean()
        p_b = prd[m].assign(**akt[m][gb]).groupby(gb)[c].mean()
        ev["mae_bulanan_test"][c] = {"lstm": round(float(np.mean(np.abs(p_b - a_b))), 3),
                                     "bias": round(float(np.mean(p_b - a_b)), 3)}

    os.makedirs(OUT_DIR, exist_ok=True)
    bulanan.to_csv(os.path.join(OUT_DIR, "embedding_bulanan.csv"), index=False)
    with open(os.path.join(OUT_DIR, "evaluasi_ramalan.json"), "w", encoding="utf-8") as f:
        json.dump(ev, f, indent=2, ensure_ascii=False)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_harian.pt")

    print("\nTest 2025 (satuan asli). Harian: MAE LSTM vs persistence | Rata-rata bulanan: MAE dan bias LSTM")
    for c in F:
        a, b = ev["mae_harian_test"][c], ev["mae_bulanan_test"][c]
        print(f"  {c:22s} harian {a['lstm']:7.3f} vs {a['persistence']:7.3f} | bulanan MAE {b['lstm']:6.3f} bias {b['bias']:+7.3f}")
    print(f"\nEmbedding bulanan: {len(bulanan)} baris -> {OUT_DIR}/embedding_bulanan.csv")


if __name__ == "__main__":
    main()
