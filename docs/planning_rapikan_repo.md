# Rencana Merapikan Repo & Dokumentasi Eksperimen

Status: **dikerjakan 27 Nov 2026** (tahap 0–8 selesai; `data/raw copy/` dihapus, model lama di `models/` dipertahankan atas permintaan; tahap 9 tanpa merge ke `main`).
Tujuan: siapa pun (kamu, Rahmat, dosen pembimbing) bisa membuka repo ini, mengerti apa saja yang sudah
dicoba, menemukan hasilnya, dan menjalankan ulang setiap eksperimen dengan satu perintah.

---

## 1. Kondisi sekarang (hasil inventaris)

| Hal | Kondisi | Masalah |
|---|---|---|
| Script di root | 21 file `.py` (training, pembuat data, aplikasi, script lama) | Tercampur; tidak jelas mana yang dipakai, mana yang usang |
| Jalur pipeline | 4 jalur berbeda (lihat 1.1) | Tidak ada peta yang menjelaskan hubungannya |
| `src/` | 15 modul, semua sejajar | Modul jalur lama dan baru bercampur |
| Catatan | `planning_teknis_model.txt`, `planning_random_forest.txt`, `planning_logging_eksperimen.txt`, `catatan_sumber_data.txt`, `ringkasan_analisis_pdf.txt` | Sebagian sudah usang (mis. masih menyebut 5 lokasi / label if-else) |
| `results/runs/` | 32 folder run | Nama tidak konsisten; tidak ada indeks run → eksperimen; ada run duplikat (`20261124_204507_*` digantikan `205335`) |
| `results/experiment_history.csv` | Semua run tercampur (klasifikasi, regresi, cuaca) | Kolom metrik dipakai untuk arti berbeda-beda |
| `models/` (146 MB, tidak di Git) | 18 file | Banyak model lama 5 lokasi yang sudah tidak dipakai |
| `data/raw copy/` | Salinan data dari kamu (tidak di Git) | Duplikat; isinya sudah ada di `data/raw/` |
| `dokumen/Ringkasan_Eksperimen_Hybrid.docx` | Tidak di Git | Masih versi E1–E8 (5 lokasi); belum memuat percobaan setelahnya |
| Git | `main` tertinggal 20 commit dari `pipeline-proposal`; ada juga `rapikan-reproduksibilitas` dan `origin/develop_rahmat` | Pekerjaan belum digabung ke `main`; Rahmat belum tahu perubahan |

### 1.1 Empat jalur pipeline yang ada

| Jalur | Isi | File utama | Status |
|---|---|---|---|
| **A. Lama (label if-else)** | LSTM klasifikasi + RF, evaluasi bocor | `train_lstm.py`, `train_rf.py`, `main.py`, `notebooks/training_random_forest_colab.ipynb`, `import_colab_log.py` | Usang → arsip |
| **B. Cuaca harian (Rahmat)** | BiLSTM + attention, ramal cuaca besok, dashboard cuaca | `notebooks/training_lstm_cuaca.ipynb`, `train_bilstm_weather.py`, `predict_production_2025.py`, `app_prediksi_cuaca.py`, `src/weather_model.py`, `data/raw/lstm/` | Tetap, milik Rahmat |
| **C. Proposal klasifikasi** | Sesuai Bab III, label if-else, Hybrid vs RF vs LSTM | `train_hybrid_proposal.py`, `train_hybrid_harian.py`, `src/proposal_pipeline.py`, `src/fitur_harian.py`, `src/report_proposal.py` | Aktif (Bab IV draf) |
| **D. Many-to-many varietas** | Lokasi × bulan × 25 varietas: skor, hasil, umur | `buat_data_dummy_varietas.py`, `buat_data_rf_varietas.py`, `train_lstm_harian.py`, `train_rf_varietas.py`, `train_rf_hasil_umur.py`, `train_rf_hasil_cuaca.py`, `src/fitur_cuaca.py`, `app_rekomendasi_proposal.py`, `src/proposal_inference.py` | Aktif (dipakai web) |

---

## 2. Struktur folder yang diusulkan

```text
projekskripsi/
├── README.md                     # pintu masuk: tujuan, setup, peta folder, cara menjalankan
├── run_all.py                    # jalankan ulang semua eksperimen aktif sesuai urutan
├── configs/model_config.yaml
├── docs/
│   ├── 00_ringkasan_proyek.md    # 1 halaman: masalah, pendekatan, status, temuan utama
│   ├── 01_data.md                # sumber, riil vs dummy, keterbatasan (2 kotak grid, Nov 2025)
│   ├── 02_metodologi.md          # pemetaan ke proposal Bab III + daftar penyesuaian
│   ├── 03_katalog_eksperimen.md  # E01–E14: tujuan, cara, perintah, run_id, commit, hasil, kesimpulan
│   ├── 04_aplikasi.md            # dua aplikasi web: cara pakai, model yang dipakai
│   ├── 05_log_keputusan.md       # keputusan penting + alasannya (mis. 12 bulan, LSTM peramal)
│   ├── istilah.md                # Accuracy, F1, Top-K, MAE, R², Spearman, seed, dll.
│   ├── proposal/                 # PDF proposal + ringkasan_analisis_pdf
│   ├── laporan/                  # Ringkasan_Eksperimen_Hybrid.docx (diperbarui)
│   └── arsip/                    # planning_*.txt & catatan lama (tidak dihapus)
├── data/
│   ├── raw/                      # iklim, tanah, varietas, dummy_varietas/, lstm/ (Rahmat)
│   └── processed/                # turunan, di-ignore (dibuat ulang oleh script)
├── src/
│   ├── data/                     # data_fetcher, dataset_loader, preprocessing, fitur_cuaca, fitur_harian
│   ├── models/                   # lstm_encoder, weather_model, rf_classifier
│   ├── pipelines/                # proposal_pipeline, report_proposal, evaluate, recommender
│   ├── app/                      # proposal_inference
│   └── utils/                    # experiment_logger
├── pipelines/                    # script yang dijalankan, dikelompokkan per jalur
│   ├── data/                     # fetch_data, buat_data_dummy_varietas, buat_data_rf_varietas
│   ├── klasifikasi_proposal/     # train_hybrid_proposal, train_hybrid_harian
│   ├── varietas/                 # train_lstm_harian, train_rf_varietas, train_rf_hasil_umur, train_rf_hasil_cuaca
│   ├── cuaca_harian/             # train_bilstm_weather, predict_production_2025 (jalur Rahmat)
│   └── arsip_lama/               # train_lstm, train_rf, main, import_colab_log (jalur A)
├── apps/                         # app_rekomendasi_proposal, app_prediksi_cuaca
├── notebooks/                    # notebook Colab (Rahmat & RF lama)
├── results/
│   ├── INDEX.md                  # tabel run_id → kode eksperimen → status (dipakai/usang)
│   ├── experiment_history.csv    # tetap (tidak diubah agar riwayat utuh)
│   └── runs/                     # folder run tetap dengan nama aslinya
└── models/                       # di-ignore; dibuat ulang oleh run_all.py
```

Script dijalankan sebagai modul dari root, misalnya `python -m pipelines.varietas.train_rf_varietas`,
supaya import `src.*` tetap jalan tanpa trik path.

---

## 3. Katalog eksperimen (isi awal `docs/03_katalog_eksperimen.md`)

| Kode | Eksperimen | Jalur | Run / bukti | Commit | Kesimpulan singkat |
|---|---|---|---|---|---|
| E01 | Audit hasil lama | A, B | `20261121_*` | – | Angka lama terlalu optimis (evaluasi bocor) |
| E02 | BiLSTM cuaca vs tebakan sederhana | B | analisis (tanpa run) | – | Suhu ≈ rata-rata kalender; hujan tak bisa ditebak |
| E03 | Reproduksi pipeline lama | A | analisis | – | Baseline identik; Hybrid beda antar-komputer |
| E04 | Cek dimensi embedding (PCA) | A | analisis | – | 32 angka ≈ 4 informasi |
| E05 | Pipeline proposal, 5 lokasi | C | `20261124_205335_*` | `0414b2f` | Hybrid F1 0,405; Top-3 0,843 |
| E06 | Diagnosa batas label dummy | C | analisis | – | Plafon ±40–50% karena label if-else |
| E07 | Per bulan vs per musim | – | analisis | – | Variasi hujan 52% vs 17% |
| E08 | Fitur turunan harian (cara B) | C | `20261124_223249_*_Harian` | `64239e8` | Tidak membantu Hybrid |
| E09 | Pipeline proposal, 20 lokasi | C | `20261125_004642_*` | `9f95e9c` | Hybrid tetap terbaik; LSTM overfit cepat |
| E10 | RF skor kecocokan many-to-many | D | `20261125_005058_RF_Varietas_Hybrid` | `9f95e9c` | RF saja > Hybrid |
| E11 | LSTM harian (5 & 1 epoch) untuk RF | D | `20261125_011034_*`, `20261125_014954_*` | `af47cda`, `328e2fc` | Epoch terbaik = 1; tetap kalah RF saja |
| E12 | Prediksi hasil & umur panen | D | `20261125_0129*`–`0144*` | `328e2fc` | Umur MAE ±1,7 hari; hasil: Hybrid < tebakan |
| E13 | Hasil panen dengan cuaca biasa | D | `20261125_021218_RF_HASIL_Cuaca` | `685ed3b` | Cuaca biasa > Hybrid 32 dimensi |
| E14 | Kenapa RF saja unggul (isi informasi) | D | analisis | – | Bukan kurang informasi; overfit & interaksi |

Setiap entri di katalog memakai format yang sama: **Tujuan → Cara → Perintah untuk mengulang → Hasil
(tabel) → Kesimpulan → Status (dipakai di Bab IV / usang)**. Analisis tanpa run (E02, E03, E04, E06, E07, E14)
dijadikan script kecil di `pipelines/analisis/` supaya angkanya bisa dibuat ulang.

---

## 4. Tahapan pengerjaan

| Tahap | Pekerjaan | Hasil | Perkiraan |
|---|---|---|---|
| **0. Pengaman** | Buat tag Git `sebelum-rapikan`; gabungkan `pipeline-proposal` ke `main` (sesudah kamu setuju) | Titik kembali bila ada yang rusak | 15 menit |
| **1. Katalog & indeks** | Tulis `docs/03_katalog_eksperimen.md` dan `results/INDEX.md` dari run & commit yang ada | Semua eksperimen terdata | 1–2 jam |
| **2. Pindah struktur** | Pindahkan script & modul sesuai bagian 2 dengan `git mv` (riwayat tetap); perbaiki import | Folder rapi | 1–2 jam |
| **3. Script analisis** | Jadikan E02, E03, E04, E06, E07, E14 script yang bisa dijalankan | Semua angka di katalog bisa dibuat ulang | 1–2 jam |
| **4. `run_all.py`** | Jalankan berurutan: data → jalur C → jalur D → analisis; opsi `--cepat` (1 seed) | Satu perintah untuk semua | 1 jam |
| **5. Dokumentasi** | README baru + `docs/00`–`05` + `istilah.md`; pindahkan catatan lama ke `docs/arsip/` | Dokumentasi lengkap | 2–3 jam |
| **6. Verifikasi** | Jalankan `run_all.py` dari repo bersih; bandingkan metrik dengan katalog (selisih harus 0) | Bukti semuanya masih reproducible | ±30 menit mesin |
| **7. Bersih-bersih** | Hapus `data/raw copy/`, `__pycache__`, model lama di `models/` (sesudah konfirmasi) | Folder ringan | 15 menit |
| **8. Laporan** | Perbarui `docs/laporan/Ringkasan_Eksperimen_Hybrid.docx` dengan E09–E14 | Laporan terkini | 1 jam |
| **9. Serah** | Commit per tahap; push ke GitHub dan kabari Rahmat (sesudah kamu setuju) | Tim memakai versi yang sama | 15 menit |

Setiap tahap di-commit terpisah, sehingga bisa dibatalkan satu per satu.

---

## 5. Aturan supaya tetap rapi ke depan

1. **Satu eksperimen = satu kode (E15, E16, …)** yang dicatat di katalog sebelum hasilnya dipakai.
2. **Commit dulu, baru jalankan** eksperimen, supaya log mencatat commit yang bersih.
3. **Nama stage di log diawali kode eksperimen**, mis. `E15_RF_Varietas_Musim`.
4. **Hasil yang tidak dipakai tidak dihapus**, cukup ditandai "usang" di `results/INDEX.md`.
5. **Data dummy selalu diberi tanda** di nama file atau kolom `sumber`, dan dicatat di `docs/01_data.md`.
6. **Perubahan pada jalur Rahmat (B)** dibicarakan dulu dengan Rahmat.

---

## 6. Keputusan (disetujui 27 Nov 2026)

| # | Pertanyaan | Keputusan |
|---|---|---|
| 1 | Pindahkan script ke folder (`pipelines/`, `apps/`)? | **Ya, dipindah** |
| 2 | Jalur lama (A) diarsipkan atau dihapus? | **Diarsipkan** ke `pipelines/arsip_lama/` |
| 3 | Folder run dikelompokkan per eksperimen? | **Ya**: `results/eksperimen/<kode>/<run_id>/`; nama folder run asli dipertahankan di dalamnya |
| 4 | Gabungkan `pipeline-proposal` ke `main`? | **Tidak**. Semua kerja tetap di branch `pipeline-proposal` |
| 5 | File jalur Rahmat (B) ikut dipindah? | **Ya**, ke `pipelines/cuaca_harian/` dan `apps/` |
| 6 | Bahasa dokumentasi | Bahasa Indonesia, istilah teknis tetap bahasa Inggris |

Penyesuaian dari rencana awal: tahap 0 hanya membuat tag `sebelum-rapikan` (tanpa merge ke `main`);
tahap 9 tidak mendorong ke `main`.
