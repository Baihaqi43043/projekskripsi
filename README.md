# Model Hybrid LSTM + Random Forest untuk Pemilihan Varietas Tanaman Unggulan

Repositori ini berisi implementasi pipeline *machine learning* hybrid yang menggabungkan **Long Short-Term Memory (LSTM)** untuk ekstraksi pola temporal data agroklimat (*time-series*) dan **Random Forest** untuk klasifikasi serta rekomendasi **Top-K varietas tanaman** (Padi, Jagung, Kedelai).

---

## 1. Struktur Projek

```text
projekskripsi/
├── configs/
│   └── model_config.yaml         # Konfigurasi lokasi, fitur, & hyperparameter LSTM/RF
├── data/
│   ├── raw/                      # Data mentah (DI-COMMIT ke Git)
│   │   ├── aceh_utara_agroklimat_harian.csv    # 20 lokasi x 2.192 hari (NASA POWER)
│   │   ├── aceh_utara_agroklimat_bulanan.csv   # 20 lokasi x 72 bulan (agregasi harian)
│   │   ├── aceh_utara_tanah_statis.csv         # 20 lokasi (SoilGrids + dummy) — BELUM ADA, lihat bagian 3
│   │   └── lstm/
│   │       ├── training/training.csv           # Harian 5 lokasi (L001-L005), 2020-2024
│   │       └── testing/testing.csv             # Harian 5 lokasi (L001-L005), 2025
│   └── processed/                # Turunan (di-ignore): embedding LSTM .npz
├── models/                       # Bobot model (di-ignore, bagikan lewat Drive)
├── notebooks/                    # Notebook Google Colab (LSTM cuaca & RF)
├── results/                      # Log eksperimen, metrik, grafik (di-commit)
├── src/
│   ├── data_fetcher.py           # Penarik data NASA POWER & SoilGrids
│   ├── dataset_loader.py         # Loader dataset & split bersama (pipeline varietas)
│   ├── preprocessing.py          # Encoding, scaling, sliding window, & chronological split
│   ├── lstm_encoder.py           # LSTM encoder varietas (128 -> 64 -> Dense 32-dim)
│   ├── weather_model.py          # BiLSTM+Attention cuaca + preprocessing (training & inference)
│   ├── rf_classifier.py          # Concatenation fitur & Random Forest classifier
│   ├── recommender.py            # Ranking Top-K varietas & feature importance
│   ├── evaluate.py               # Metrik Accuracy, Precision, Recall, F1, & Top-K
│   └── experiment_logger.py      # Log otomatis ke results/
├── fetch_data.py                 # Unduh ulang data mentah (NASA POWER / SoilGrids)
├── train_lstm.py                 # Training LSTM encoder varietas (Tim LSTM)
├── train_rf.py                   # Training & evaluasi Random Forest (Tim RF)
├── train_bilstm_weather.py       # Training BiLSTM cuaca harian (versi lokal notebook)
├── predict_production_2025.py    # Simulasi inference Jan-Mar 2025
├── app_prediksi_cuaca.py         # Dashboard web FastAPI
├── main.py                       # Pipeline lengkap (LSTM -> RF)
└── requirements.txt
```

---

## 2. Setup Environment (Pertama Kali)

Gunakan Python 3.11.

```bash
# Windows (PowerShell / CMD)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Semua script dijalankan dari folder root repo (`python train_rf.py`, dst.).

---

## 3. Data

Data iklim sudah ada di `data/raw/` (ikut di-commit), jadi **tidak perlu download ulang**.

> **Status 24 Nov 2026:** `aceh_utara_tanah_statis.csv` belum ada di repo (file lama tidak pernah
> di-commit, dan API SoilGrids sedang tidak merespons). Sebelum `train_lstm.py` bisa jalan, salin
> file aslinya ke `data/raw/` **atau** jalankan `python fetch_data.py --only soil` saat SoilGrids aktif.
Untuk memperbarui data atau setelah mengubah daftar lokasi di config:

```bash
python fetch_data.py                  # iklim (NASA POWER) + tanah (SoilGrids)
python fetch_data.py --only climate   # hanya iklim
python fetch_data.py --only soil      # hanya tanah
```

- Jika SoilGrids gagal, `fetch_data.py` **berhenti dengan error** (tidak lagi diam-diam memakai nilai acak).
  Untuk sekadar menguji pipeline saat SoilGrids down: `python fetch_data.py --only soil --allow-dummy-soil`
  — lokasi yang memakai dummy ditandai `DUMMY` di kolom `sumber_tanah`.
- Script training **tidak** mengakses internet; jika file belum ada, script berhenti dan menyuruh menjalankan `fetch_data.py`.
- **Satuan `lama_penyinaran`**: data NASA POWER `ALLSKY_SFC_SW_DWN` adalah **radiasi (MJ/m²/hari)**, bukan jam/hari.

Detail sumber & status data (riil vs dummy) ada di `catatan_sumber_data.txt`.

---

## 4. Pembagian Kerja & Cara Menjalankan

### A. Pipeline Varietas — Tim LSTM
```bash
python train_lstm.py
```
Output: `models/lstm_encoder.pt` dan `data/processed/lstm_embeddings.npz`
(vektor temporal 32-dim + fitur statis + label dengan split yang sudah terkunci).

### B. Pipeline Varietas — Tim Random Forest
*(butuh `data/processed/lstm_embeddings.npz` dari langkah A)*
```bash
python train_rf.py          # default
python train_rf.py --tune   # + RandomizedSearchCV
```
Output: `models/hybrid_rf.pkl` + metrik (Accuracy, Precision, Recall, F1, Top-3) di `results/`.

### C. Full pipeline varietas
```bash
python main.py
```

### D. BiLSTM Cuaca Harian (prediksi 1 hari ke depan)
```bash
python train_bilstm_weather.py              # setara notebook training_lstm_cuaca.ipynb
python predict_production_2025.py           # simulasi Jan-Mar 2025
python app_prediksi_cuaca.py                # dashboard di http://127.0.0.1:8000
```
`train_bilstm_weather.py` menyimpan **dua file** yang harus selalu dibawa bersama:
- `models/best_lstm_weather.pth` — bobot model
- `models/best_lstm_weather_preproc.json` — batas winsorizing + median/IQR scaler

`predict_production_2025.py` dan dashboard membaca JSON tersebut. Jika JSON tidak ada
(mis. model lama dari Colab), statistik dihitung ulang dari `data/raw/lstm/training/training.csv`
dengan peringatan — hasilnya sama selama CSV train tidak berubah.

---

## 5. Aturan Penting Kolaborasi
1. **Jangan ubah aturan split secara sepihak:** `src/preprocessing.py` dan `src/dataset_loader.py` memakai *chronological split* (80/20) dan `seed: 42`. Jika diubah, pastikan kedua tim memakai versi yang sama agar `lstm_embeddings.npz` tetap sinkron.
2. **Preprocessing BiLSTM cuaca hanya ada di `src/weather_model.py`.** Notebook Colab masih punya salinan sendiri; jika salah satunya diubah, samakan yang lain.
3. **Commit dulu sebelum training** supaya `git_commit` di `results/experiment_history.csv` bisa dilacak.
4. **Status data:** iklim riil (NASA POWER); tanah sebagian riil (SoilGrids: pH, tekstur, C-organik);
   `nitrogen`, `fosfor`, `kalium`, `drainase`, `risiko_bencana_hama`, serta label `varietas` masih
   *placeholder* sambil menunggu data resmi BMKG, Dinas Pertanian Aceh, dan Kementan RI.
