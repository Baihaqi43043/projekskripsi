# Model Hybrid LSTM + Random Forest untuk Pemilihan Varietas Tanaman Unggulan

Repositori ini berisi implementasi pipeline *machine learning* hybrid yang menggabungkan **Long Short-Term Memory (LSTM)** untuk ekstraksi pola temporal data agroklimat (*time-series*) dan **Random Forest** untuk klasifikasi serta rekomendasi **Top-K varietas tanaman** (Padi, Jagung, Kedelai).

---

## 1. Struktur Projek

```text
projekskripsi/
├── configs/
│   └── model_config.yaml       # Konfigurasi lokasi, fitur, & hyperparameter LSTM/RF
├── data/
│   ├── raw/                    # Data mentah agroklimat, tanah statis, & varietas
│   └── processed/              # Output vektor temporal LSTM (lstm_embeddings.npz)
├── models/                     # Bobot model terlatih (lstm_encoder.pt & hybrid_rf.pkl)
├── src/
│   ├── data_fetcher.py         # Penarik data NASA POWER, SoilGrids, & Hugging Face
│   ├── dataset_loader.py       # Loader dataset & split bersama (shared)
│   ├── preprocessing.py        # Encoding, scaling, sliding window, & chronological split
│   ├── lstm_encoder.py         # Arsitektur LSTM (128 -> 64 -> Dense 32-dim)
│   ├── rf_classifier.py        # Concatenation fitur & Random Forest classifier
│   ├── recommender.py          # Ranking Top-K varietas & feature importance
│   └── evaluate.py             # Metrik Accuracy, Precision, Recall, F1, & Top-K
├── train_lstm.py               # Script training khusus Tim LSTM
├── train_rf.py                 # Script training & evaluasi khusus Tim Random Forest
├── main.py                     # Menjalankan pipeline lengkap (LSTM -> RF)
└── requirements.txt            # Daftar library Python
```

---

## 2. Setup Environment (Pertama Kali)

Gunakan Python 3.11+ (bisa memakai `uv` atau `venv` biasa):

```bash
# Opsi 1: menggunakan uv
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Opsi 2: menggunakan venv standar
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Pembagian Kerja & Cara Menjalankan

Pipeline dibagi menjadi **2 tahap terpisah** yang terhubung melalui `data/processed/lstm_embeddings.npz`, sehingga masing-masing bisa bereksperimen tanpa saling menunggu training ulang.

### A. Bagian LSTM (Tim LSTM)
- **File utama:** `src/lstm_encoder.py` dan bagian `lstm:` di `configs/model_config.yaml`
- **Tugas:** Mengoptimasi arsitektur LSTM (`128 -> 64 -> 32-dim`, dropout, learning rate, epoch, atau strategi *early stopping*).
- **Cara jalankan:**
  ```bash
  python train_lstm.py
  ```
- **Output:**
  - `models/lstm_encoder.pt` (bobot model LSTM)
  - `data/processed/lstm_embeddings.npz` (vektor temporal 32 dimensi + fitur statis + label dengan split yang sudah terkunci)

### B. Bagian Random Forest (Tim Random Forest)
- **File utama:** `src/rf_classifier.py`, `src/recommender.py`, `src/evaluate.py`, dan bagian `random_forest:` di `configs/model_config.yaml`
- **Tugas:** Melakukan *hyperparameter tuning* Random Forest, evaluasi perbandingan terhadap *Baseline RF*, analisis *feature importance*, serta ranking **Top-K rekomendasi**.
- **Cara jalankan:**
  *(Pastikan `data/processed/lstm_embeddings.npz` sudah ada dari hasil `train_lstm.py`)*
  ```bash
  python train_rf.py
  ```
- **Output:**
  - `models/hybrid_rf.pkl`
  - Ringkasan metrik evaluasi (`Accuracy`, `Precision`, `Recall`, `F1-Score`, `Top-3 Accuracy`) & contoh rekomendasi.

### C. Menjalankan Full Pipeline Sekaligus
```bash
python main.py
```

---

## 4. Aturan Penting Kolaborasi
1. **Jangan ubah aturan split secara sepihak:** `src/preprocessing.py` dan `src/dataset_loader.py` memakai *chronological split* (80/20) dan `seed: 42`. Jika diubah, pastikan kedua tim memakai versi yang sama agar `lstm_embeddings.npz` tetap sinkron.
2. **Status data saat ini:**
   - Data agroklimat (harian & bulanan 2020–2025) sudah diambil dari **NASA POWER API**, dan sebagian data tanah dari **SoilGrids** untuk 5 titik sampel di Aceh Utara.
   - Kolom `nitrogen`, `fosfor`, `kalium`, `drainase`, `risiko_bencana_hama`, serta label target `varietas` masih berupa *placeholder*/prototipe sambil menunggu data resmi dari BMKG, Dinas Pertanian Aceh, dan Kementan RI (detail ada di `catatan_sumber_data.txt`).

