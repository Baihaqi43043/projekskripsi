"""Jalankan ulang semua eksperimen aktif sesuai urutan yang benar.

Contoh:
  python run_all.py                       # semua langkah (±30 menit di CPU)
  python run_all.py --daftar              # lihat daftar langkah
  python run_all.py --hanya E10 E13       # hanya langkah dengan kode tersebut (prasyaratnya harus sudah ada)
  python run_all.py --hasil-dir tmp/cek   # tulis log & folder eksperimen ke tempat lain (uji reproduksi)

Catatan urutan:
- E05 & E08 dijalankan dengan 5 lokasi (env LOKASI_AKTIF), lalu E09 dengan 20 lokasi.
  E09 menimpa models/proposal_bundle.joblib, sehingga bundle akhir = 20 lokasi (dipakai web & jalur varietas).
- Jalur varietas (E10-E13) bergantung pada bundle E09, jadi selalu dijalankan sesudahnya.
- E01, E03, E04 bersifat historis (jalur lama) dan tidak dijalankan ulang; lihat docs/03_katalog_eksperimen.md.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

LIMA = "L001,L002,L003,L004,L005"
SEMUA = ""  # kosong = semua lokasi di config

# (kode, keterangan, lokasi, modul, argumen)
LANGKAH = [
    ("E05", "Pipeline proposal, 5 lokasi", LIMA, "pipelines.klasifikasi_proposal.train_hybrid_proposal", []),
    ("E08", "Fitur turunan harian (cara B), 5 lokasi", LIMA, "pipelines.klasifikasi_proposal.train_hybrid_harian", []),
    ("E09", "Pipeline proposal, 20 lokasi (bundle web)", SEMUA, "pipelines.klasifikasi_proposal.train_hybrid_proposal", []),
    ("DATA", "Data dummy many-to-many", SEMUA, "pipelines.data.buat_data_dummy_varietas", []),
    ("DATA", "Dataset RF varietas (vektor LSTM bulanan)", SEMUA, "pipelines.data.buat_data_rf_varietas", []),
    ("E10", "RF skor kecocokan varietas", SEMUA, "pipelines.varietas.train_rf_varietas", []),
    ("E12", "RF hasil & umur panen (LSTM bulanan)", SEMUA, "pipelines.varietas.train_rf_hasil_umur", []),
    ("E11", "LSTM harian 5 epoch", SEMUA, "pipelines.varietas.train_lstm_harian",
     ["--epoch", "5", "--out", "data/processed/lstm_harian"]),
    ("E11", "Dataset RF dari LSTM harian 5 epoch", SEMUA, "pipelines.data.buat_data_rf_varietas",
     ["--lstm-harian", "data/processed/lstm_harian/embedding_bulanan.csv", "--out", "data/processed/rf_varietas_harian"]),
    ("E11", "RF skor dengan LSTM harian 5 epoch", SEMUA, "pipelines.varietas.train_rf_varietas",
     ["--data-dir", "data/processed/rf_varietas_harian", "--label", "Harian"]),
    ("E11", "LSTM harian 1 epoch", SEMUA, "pipelines.varietas.train_lstm_harian",
     ["--epoch", "1", "--out", "data/processed/lstm_harian_1epoch"]),
    ("E11", "Dataset RF dari LSTM harian 1 epoch", SEMUA, "pipelines.data.buat_data_rf_varietas",
     ["--lstm-harian", "data/processed/lstm_harian_1epoch/embedding_bulanan.csv", "--out", "data/processed/rf_varietas_harian1"]),
    ("E11", "RF skor dengan LSTM harian 1 epoch", SEMUA, "pipelines.varietas.train_rf_varietas",
     ["--data-dir", "data/processed/rf_varietas_harian1", "--label", "Harian1"]),
    ("E12", "RF hasil & umur panen (LSTM harian 1 epoch)", SEMUA, "pipelines.varietas.train_rf_hasil_umur",
     ["--data-dir", "data/processed/rf_varietas_harian1", "--label", "Harian1"]),
    ("E13", "RF hasil panen dengan cuaca biasa", SEMUA, "pipelines.varietas.train_rf_hasil_cuaca", []),
    ("E02", "Analisis: model cuaca harian vs tebakan", SEMUA, "pipelines.analisis.e02_cuaca_vs_tebakan", []),
    ("E06", "Analisis: batas label dummy", SEMUA, "pipelines.analisis.e06_batas_label_dummy", []),
    ("E07", "Analisis: per bulan vs per musim", LIMA, "pipelines.analisis.e07_bulan_vs_musim", []),
    ("E14", "Analisis: isi informasi vektor LSTM", SEMUA, "pipelines.analisis.e14_isi_informasi_lstm", []),
]

# nama folder di results/eksperimen/ untuk tiap kode
FOLDER = {
    "E02": "E02_cuaca_harian_vs_tebakan", "E05": "E05_proposal_5_lokasi", "E06": "E06_batas_label_dummy",
    "E07": "E07_bulan_vs_musim", "E08": "E08_fitur_harian_cara_b", "E09": "E09_proposal_20_lokasi",
    "E10": "E10_rf_kecocokan_varietas", "E11": "E11_lstm_harian_untuk_rf", "E12": "E12_prediksi_hasil_umur",
    "E13": "E13_hasil_panen_cuaca_biasa", "E14": "E14_isi_informasi_lstm", "DATA": "_data",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Jalankan ulang semua eksperimen aktif.")
    ap.add_argument("--daftar", action="store_true", help="tampilkan langkah lalu keluar")
    ap.add_argument("--hanya", nargs="*", help="kode langkah yang dijalankan, mis. E10 E13")
    ap.add_argument("--hasil-dir", default=None, help="folder hasil pengganti 'results' (uji reproduksi)")
    a = ap.parse_args()

    langkah = [l for l in LANGKAH if not a.hanya or l[0] in a.hanya]
    for i, (kode, ket, lok, mod, arg) in enumerate(langkah, 1):
        print(f"{i:2d}. [{kode}] {ket}  ({'5 lokasi' if lok else 'semua lokasi'})  python -m {mod} {' '.join(arg)}")
    if a.daftar:
        return

    total = time.perf_counter()
    for i, (kode, ket, lok, mod, arg) in enumerate(langkah, 1):
        env = {**os.environ, "EKSPERIMEN": FOLDER[kode], "LOKASI_AKTIF": lok, "PYTHONWARNINGS": "ignore"}
        if a.hasil_dir:
            env["HASIL_DIR"] = a.hasil_dir
        print(f"\n=== [{i}/{len(langkah)}] {kode}: {ket} ===", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-m", mod, *arg], env=env)
        if r.returncode != 0:
            sys.exit(f"Langkah {kode} ({mod}) gagal dengan kode {r.returncode}.")
        print(f"--- selesai dalam {time.perf_counter() - t0:.0f} detik", flush=True)
    print(f"\nSemua langkah selesai dalam {(time.perf_counter() - total) / 60:.1f} menit.")


if __name__ == "__main__":
    main()
