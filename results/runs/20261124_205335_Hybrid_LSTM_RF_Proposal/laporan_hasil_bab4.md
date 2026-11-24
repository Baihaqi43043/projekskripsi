# Hasil Eksperimen Model Hybrid LSTM–Random Forest (Draf Bab IV)

> **PERINGATAN — HASIL PROTOTIPE.** Label varietas dibuat dengan aturan dummy
> (`beri_label_varietas_prototipe`) dan 5 dari 9 fitur tanah (N, P, K, drainase, risiko hama) masih dummy.
> Angka di bawah membuktikan **pipeline** berjalan sesuai rancangan proposal, tetapi **belum dapat diklaim
> sebagai temuan penelitian**. Jalankan ulang `python train_hybrid_proposal.py` setelah data riil tersedia.

- Run: `20261124_205335` · commit `847d669` · 2026-11-24 20:53:35
- Lokasi aktif: L001, L002, L003, L004, L005 · Seed: 42, 43, 44, 45, 46
- Script: `train_hybrid_proposal.py`

## 4.1 Dataset dan Pembagian Data

Satu sampel = satu lokasi pada satu bulan target. Input LSTM adalah sekuens iklim **12 bulan**
sebelum bulan target, sehingga tidak ada informasi bulan target yang digunakan. Data dibagi kronologis
(Bagian 3.7 dan 3.11 proposal):

| Bagian | Tahun bulan target | Jumlah sampel | Porsi |
|---|---|---|---|
| Training | 2021–2023 | 180 | 60% |
| Validation | 2024 | 60 | 20% |
| Testing | 2025 | 60 | 20% |

Porsi pelatihan (training + validation) = 80%, pengujian = 20%.
Jumlah kelas varietas: 11.

**Tabel 4.1 Distribusi kelas varietas**

| varietas | train | validasi | test |
|---|---|---|---|
| Jagung_BISI_18 | 21 | 3 | 12 |
| Jagung_Bima_20_URI | 2 | 0 | 0 |
| Jagung_Lamuru | 4 | 3 | 3 |
| Jagung_NK_212 | 18 | 2 | 6 |
| Kedelai_Anjasmoro | 16 | 15 | 9 |
| Padi_Ciherang | 45 | 8 | 12 |
| Padi_IR64 | 18 | 10 | 6 |
| Padi_Inpari_32 | 17 | 4 | 6 |
| Padi_Inpari_33 | 14 | 2 | 3 |
| Padi_Inpari_42 | 8 | 4 | 0 |
| Padi_Mekongga | 17 | 9 | 3 |

## 4.2 Hasil Preprocessing

Missing value diinterpolasi linier per lokasi. Outlier ditangani dengan metode IQR (pagar Q1 − 1,5·IQR dan
Q3 + 1,5·IQR) lalu dinormalisasi Z-Score; seluruh statistik dihitung **hanya dari data periode training**.
Jumlah nilai yang dibatasi (capping): **14**. Fitur turunan indeks musim: sin/cos bulan.

**Tabel 4.2 Parameter preprocessing iklim bulanan**

| variabel | batas_bawah_IQR | batas_atas_IQR | mean_Z | std_Z |
|---|---|---|---|---|
| curah_hujan | -89.015 | 470.385 | 198.029 | 101.4415 |
| suhu_rata_rata | 20.1389 | 31.1569 | 25.8083 | 1.4483 |
| suhu_maksimal | 25.0598 | 32.0675 | 28.4792 | 1.1218 |
| suhu_minimum | 16.7503 | 30.365 | 23.7306 | 1.7408 |
| kelembaban_rata_rata | 73.0924 | 94.6736 | 84.3106 | 3.4322 |
| lama_penyinaran | 11.2877 | 22.1365 | 16.78 | 2.0619 |

## 4.3 Pelatihan Encoder LSTM

Arsitektur sesuai Bagian 3.10: LSTM 128 unit → LSTM 64 unit (dropout 0.2) → Dense
32 unit (ReLU) sebagai vektor fitur temporal. Encoder dilatih meramal iklim bulan
berikutnya (MSE) sehingga tidak menggunakan label varietas; early stopping (patience
20) berdasarkan loss validasi 2024.

![Kurva loss LSTM](lstm_loss_curve.png)

**Tabel 4.3 Hasil pelatihan LSTM dan pembanding ramalan naif (MAE pada test 2025, skala Z-Score)**

| seed | epoch terbaik | MSE validasi | MAE test LSTM | MAE persistence | MAE klimatologi |
|---|---|---|---|---|---|
| 42 | 14 | 0.385 | 0.4513 | 0.5951 | 0.8116 |
| 43 | 23 | 0.4301 | 0.4481 | 0.5951 | 0.8116 |
| 44 | 24 | 0.4354 | 0.4293 | 0.5951 | 0.8116 |
| 45 | 14 | 0.4273 | 0.4346 | 0.5951 | 0.8116 |
| 46 | 28 | 0.4033 | 0.4335 | 0.5951 | 0.8116 |

Persistence = iklim bulan sebelumnya; klimatologi = rata-rata bulan yang sama pada periode training.

## 4.4 Hyperparameter Tuning Random Forest

RandomizedSearchCV (20 kombinasi, skor F1-weighted) dengan cross-validation expanding-window per
tahun (validasi 2023, 2024), kemudian model di-*refit* pada training + validation.

**Tabel 4.4 Hyperparameter terbaik Hybrid LSTM + RF per seed**

| seed | best_params |
|---|---|
| 42 | {"n_estimators": 200, "min_samples_split": 5, "min_samples_leaf": 4, "max_features": "sqrt", "max_depth": 10, "class_weight": "balanced"} |
| 43 | {"n_estimators": 200, "min_samples_split": 10, "min_samples_leaf": 2, "max_features": "log2", "max_depth": 10, "class_weight": "balanced_subsample"} |
| 44 | {"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 4, "max_features": "sqrt", "max_depth": 15, "class_weight": "balanced_subsample"} |
| 45 | {"n_estimators": 100, "min_samples_split": 10, "min_samples_leaf": 2, "max_features": "sqrt", "max_depth": 10, "class_weight": "balanced_subsample"} |
| 46 | {"n_estimators": 500, "min_samples_split": 2, "min_samples_leaf": 4, "max_features": "sqrt", "max_depth": null, "class_weight": "balanced"} |

## 4.5 Perbandingan Kinerja Model (Rumusan Masalah 3)

Evaluasi pada data testing 2025, rata-rata ± simpangan baku dari 5 seed. (w) = weighted average.

**Tabel 4.5 Perbandingan model hybrid dengan single model**

| Model | Accuracy | Precision (w) | Recall (w) | F1-Score (w) | F1-Score (macro) | Top-3 Acc |
|---|---|---|---|---|---|---|
| Hybrid LSTM + RF (usulan) | 0.367 ± 0.039 | 0.476 ± 0.051 | 0.367 ± 0.039 | 0.405 ± 0.043 | 0.288 ± 0.033 | 0.843 ± 0.019 |
| Random Forest saja | 0.290 ± 0.022 | 0.156 ± 0.006 | 0.290 ± 0.022 | 0.194 ± 0.009 | 0.247 ± 0.009 | 0.713 ± 0.025 |
| LSTM saja | 0.370 ± 0.062 | 0.283 ± 0.067 | 0.370 ± 0.062 | 0.308 ± 0.063 | 0.206 ± 0.053 | 0.670 ± 0.036 |

**Tabel 4.6 F1-Score (weighted) per seed**

| seed | Hybrid LSTM + RF | LSTM saja (sekuens iklim) | Random Forest saja (statis + rata-rata iklim) |
|---|---|---|---|
| 42 | 0.369 | 0.386 | 0.2 |
| 43 | 0.418 | 0.366 | 0.186 |
| 44 | 0.475 | 0.265 | 0.204 |
| 45 | 0.377 | 0.253 | 0.183 |
| 46 | 0.386 | 0.267 | 0.199 |

Hybrid lebih tinggi dari Random Forest saja pada **5 dari 5** seed (selisih rata-rata
+0.211) dan lebih tinggi dari LSTM saja pada **4 dari 5** seed
(selisih rata-rata +0.097).

## 4.6 Kinerja per Kelas dan Confusion Matrix

**Tabel 4.7 Precision, Recall, F1 per varietas (seed 42)**

| varietas | support_test | Hybrid_precision | Hybrid_recall | Hybrid_f1 | RF_f1 | LSTM_f1 |
|---|---|---|---|---|---|---|
| Jagung_BISI_18 | 12 | 0.625 | 0.417 | 0.5 | 0.0 | 0.75 |
| Jagung_Bima_20_URI | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Jagung_Lamuru | 3 | 0.0 | 0.0 | 0.0 | 0.308 | 0.0 |
| Jagung_NK_212 | 6 | 0.333 | 0.167 | 0.222 | 0.0 | 0.667 |
| Kedelai_Anjasmoro | 9 | 0.5 | 0.222 | 0.308 | 0.182 | 0.353 |
| Padi_Ciherang | 12 | 0.556 | 0.417 | 0.476 | 0.0 | 0.5 |
| Padi_IR64 | 6 | 0.0 | 0.0 | 0.0 | 0.471 | 0.0 |
| Padi_Inpari_32 | 6 | 0.714 | 0.833 | 0.769 | 0.667 | 0.167 |
| Padi_Inpari_33 | 3 | 0.5 | 0.667 | 0.571 | 0.462 | 0.0 |
| Padi_Inpari_42 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Padi_Mekongga | 3 | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 |

![Confusion matrix Hybrid](confusion_matrix_hybrid.png)
![Confusion matrix RF](confusion_matrix_rf.png)
![Confusion matrix LSTM](confusion_matrix_lstm.png)

## 4.7 Analisis Feature Importance

**Tabel 4.8 Kontribusi kelompok fitur (Gini importance, rata-rata seed)**

| kelompok | jumlah_fitur | importance_total | importance_per_fitur |
|---|---|---|---|
| Fitur temporal LSTM (32 dimensi) | 32 | 0.7519 | 0.0235 |
| ketinggian_tempat | 1 | 0.0489 | 0.0489 |
| risiko_bencana_hama | 1 | 0.0411 | 0.0411 |
| nitrogen | 1 | 0.0404 | 0.0404 |
| ph_tanah | 1 | 0.0393 | 0.0393 |
| fosfor | 1 | 0.0364 | 0.0364 |
| karbon_organik | 1 | 0.0362 | 0.0362 |
| tekstur_tanah | 1 | 0.0033 | 0.0033 |
| kalium | 1 | 0.0027 | 0.0027 |
| drainase | 1 | 0.0 | 0.0 |

Fitur temporal LSTM berjumlah 32 dari 41 fitur
(78%); bandingkan porsi jumlah ini dengan porsi
importance-nya sebelum menyimpulkan dominasi fitur temporal.

![Feature importance](feature_importance_group.png)

## 4.8 Contoh Rekomendasi Top-3 (Recommendation Engine, Bagian 3.10)

Skor kecocokan = probabilitas kelas dari Random Forest. Varietas label berada di Top-3 pada
**85%** sampel test (seed 42). Daftar lengkap beserta justifikasi:
`rekomendasi_top3_test2025.csv`.

**Tabel 4.9 Contoh rekomendasi (Januari dan Juli 2025)**

| id_lokasi | nama_lokasi | periode | varietas_aktual_label | rekomendasi_1 | rekomendasi_2 | rekomendasi_3 | aktual_di_top_k |
|---|---|---|---|---|---|---|---|
| L001 | Lhoksukon | Jan 2025 | Padi_IR64 | Padi_Ciherang (29.2%) | Jagung_NK_212 (24.1%) | Kedelai_Anjasmoro (23.4%) | False |
| L001 | Lhoksukon | Jul 2025 | Kedelai_Anjasmoro | Padi_IR64 (38.1%) | Jagung_BISI_18 (34.9%) | Padi_Ciherang (17.4%) | False |
| L002 | Matangkuli | Jan 2025 | Padi_IR64 | Padi_Ciherang (28.4%) | Jagung_NK_212 (24.7%) | Kedelai_Anjasmoro (23.7%) | False |
| L002 | Matangkuli | Jul 2025 | Kedelai_Anjasmoro | Padi_IR64 (40.4%) | Jagung_BISI_18 (32.8%) | Padi_Ciherang (17.5%) | False |
| L003 | Syamtalira Aron | Jan 2025 | Padi_Mekongga | Padi_Inpari_33 (46.5%) | Jagung_Bima_20_URI (31.9%) | Kedelai_Anjasmoro (10.3%) | False |
| L003 | Syamtalira Aron | Jul 2025 | Kedelai_Anjasmoro | Padi_Mekongga (34.4%) | Padi_Inpari_33 (31.9%) | Jagung_BISI_18 (25.8%) | False |
| L004 | Dewantara | Jan 2025 | Padi_Inpari_32 | Padi_Inpari_32 (64.8%) | Jagung_NK_212 (16.6%) | Padi_Mekongga (12.0%) | True |
| L004 | Dewantara | Jul 2025 | Kedelai_Anjasmoro | Padi_Mekongga (37.5%) | Padi_Inpari_32 (30.6%) | Kedelai_Anjasmoro (21.6%) | True |
| L005 | Kuta Makmur | Jan 2025 | Padi_Ciherang | Padi_Ciherang (32.5%) | Jagung_Lamuru (23.8%) | Padi_Inpari_42 (23.4%) | True |
| L005 | Kuta Makmur | Jul 2025 | Jagung_Lamuru | Padi_Inpari_42 (47.7%) | Jagung_Lamuru (36.9%) | Padi_Ciherang (7.7%) | True |

Contoh justifikasi: *pola iklim 12 bulan terakhir (LSTM, 75%); ketinggian_tempat=8.0 mdpl (5%); risiko_bencana_hama=Rendah (4%)*.

## 4.9 Penyesuaian terhadap Proposal dan Keterbatasan

| Aspek | Proposal | Implementasi | Alasan |
|---|---|---|---|
| Panjang sekuens | 60 bulan | 12 bulan | data hanya 72 bulan (2020–2025) |
| Sumber iklim | BMKG | NASA POWER | data BMKG belum tersedia |
| Lama penyinaran | jam/hari | radiasi MJ/m²/hari | variabel NASA POWER ALLSKY_SFC_SW_DWN |
| Objektif pelatihan LSTM | tidak ditentukan | peramalan iklim bulan berikutnya | menghindari kebocoran label ke tahap RF |
| Lokasi | 20 titik | 5 titik | tahap awal |
| Label varietas | data lapangan | aturan dummy | data lapangan belum tersedia |

Keterbatasan: jumlah data test kecil (60 sampel), label dan sebagian fitur tanah masih dummy,
serta validasi pakar dan *user acceptance testing* (Bagian 3.11 Tahap 8) belum dilakukan.
