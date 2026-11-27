# Model Hybrid LSTM + Random Forest untuk Pemilihan Varietas Tanaman Unggulan

Implementasi proposal disertasi *Pengembangan Model Hybrid Berbasis Machine Learning untuk Pemilihan Varietas
Tanaman Unggulan Pertanian* (Kabupaten Aceh Utara). LSTM meringkas pola iklim 12 bulan menjadi vektor 32 angka;
Random Forest menggabungkannya dengan data tanah (dan karakteristik varietas) untuk merekomendasikan Top-3
varietas padi, jagung, atau kedelai.

> **Status: prototipe.** Label varietas dan sebagian data tanah masih dummy. Pipeline sudah lengkap dan bisa
> dijalankan ulang; angka hasilnya belum boleh diklaim sebagai temuan penelitian.
> Ringkasan status dan temuan: [docs/00_ringkasan_proyek.md](docs/00_ringkasan_proyek.md).

## Mulai dari sini

| Ingin… | Buka |
|---|---|
| Memahami proyek dalam 5 menit | [docs/00_ringkasan_proyek.md](docs/00_ringkasan_proyek.md) |
| Melihat semua percobaan & hasilnya | [docs/03_katalog_eksperimen.md](docs/03_katalog_eksperimen.md) |
| Mengetahui asal & keterbatasan data | [docs/01_data.md](docs/01_data.md) |
| Mencocokkan dengan Bab III proposal | [docs/02_metodologi.md](docs/02_metodologi.md) |
| Menjalankan aplikasi web | [docs/04_aplikasi.md](docs/04_aplikasi.md) |
| Mengetahui kenapa sebuah pilihan diambil | [docs/05_log_keputusan.md](docs/05_log_keputusan.md) |
| Arti Accuracy, F1, Top-3, MAE, R², dll. | [docs/istilah.md](docs/istilah.md) |
| Mencari folder hasil sebuah run | [results/INDEX.md](results/INDEX.md) |
| Laporan Word ringkasan eksperimen | `docs/laporan/Ringkasan_Eksperimen_Hybrid.docx` |

## Setup

Python 3.11. Dari folder repo:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Semua perintah dijalankan **dari folder root repo** dengan `python -m …`.

## Menjalankan

```bash
python run_all.py                 # jalankan ulang semua eksperimen aktif (±30 menit di CPU)
python run_all.py --daftar        # lihat langkah-langkahnya
python run_all.py --hanya E10     # satu eksperimen (prasyaratnya harus sudah ada)

python -m apps.app_rekomendasi_proposal   # web rekomendasi varietas -> http://127.0.0.1:8001
python -m apps.app_prediksi_cuaca         # dashboard cuaca harian   -> http://127.0.0.1:8000
```

Menjalankan satu script dengan kode eksperimen sendiri:
`EKSPERIMEN=E15_nama python -m pipelines.varietas.train_rf_varietas` (hasil masuk `results/eksperimen/E15_nama/`).
Mengulang eksperimen 5 lokasi tanpa mengubah config: `LOKASI_AKTIF=L001,L002,L003,L004,L005 python -m …`.

## Struktur folder

```text
projekskripsi/
├── README.md                 # file ini
├── run_all.py                # jalankan ulang semua eksperimen sesuai urutan
├── configs/model_config.yaml # lokasi, fitur, hyperparameter LSTM/RF, pengaturan pipeline proposal
├── docs/                     # dokumentasi, PDF proposal, laporan Word, catatan lama (arsip/)
├── data/
│   ├── raw/                  # data mentah (di Git): iklim, tanah, varietas, dummy_varietas/, lstm/
│   └── processed/            # data turunan (tidak di Git, dibuat ulang)
├── src/                      # modul yang dipakai bersama
│   ├── data/                 # pengambilan data, loader, preprocessing, fitur cuaca & harian
│   ├── models/               # arsitektur LSTM (encoder & BiLSTM cuaca), RF
│   ├── pipelines/            # persiapan data proposal, evaluasi, laporan Bab IV
│   ├── app/                  # logika prediksi untuk aplikasi web
│   └── utils/                # pencatat eksperimen
├── pipelines/                # script yang dijalankan (python -m pipelines.<jalur>.<script>)
│   ├── data/                 # fetch_data, buat_data_dummy_varietas, buat_data_rf_varietas
│   ├── klasifikasi_proposal/ # E05, E08, E09
│   ├── varietas/             # E10-E13 (many-to-many, dipakai web)
│   ├── cuaca_harian/         # BiLSTM cuaca harian (Rahmat)
│   ├── analisis/             # E02, E06, E07, E14
│   └── arsip_lama/           # pipeline lama berlabel if-else (E01, E03, E04)
├── apps/                     # dua aplikasi web
├── notebooks/                # notebook Google Colab
├── results/
│   ├── INDEX.md              # peta folder hasil
│   ├── eksperimen/<kode>/    # satu folder per eksperimen, berisi folder run
│   ├── cuaca_harian/         # keluaran jalur cuaca harian
│   ├── experiment_history.csv
│   └── changelog_kode.txt
└── models/                   # model terlatih (tidak di Git, dibuat ulang oleh run_all.py)
```

## Aturan kerja
1. Satu eksperimen = satu kode (E15, E16, …), dicatat di katalog sebelum hasilnya dipakai.
2. Commit kode **sebelum** menjalankan eksperimen agar log mencatat commit yang bersih.
3. Hasil yang tidak dipakai tidak dihapus; tandai "usang" di `results/INDEX.md`.
4. Data dummy selalu ditandai dan dicatat di `docs/01_data.md`.
5. Perubahan pada jalur cuaca harian (Rahmat) dibicarakan dulu dengan Rahmat.
6. Branch kerja: `pipeline-proposal` (tidak digabung ke `main`).
