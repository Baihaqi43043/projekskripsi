# Katalog Eksperimen

Setiap eksperimen punya kode tetap (E01, E02, …). Folder hasilnya ada di `results/eksperimen/<kode>_<nama>/`.
Format tiap entri: **Tujuan → Cara → Perintah → Hasil → Kesimpulan → Status**.

> **Semua hasil masih prototipe.** Label varietas (E01–E09) dibuat dengan aturan if-else, data many-to-many
> (E10–E14) dibuat dengan rumus dummy, dan 5 dari 9 fitur tanah masih dummy. Angka-angka di bawah menguji
> apakah pipeline bekerja benar, belum boleh diklaim sebagai temuan penelitian. Lihat [01_data.md](01_data.md).

Singkatan: Acc = Accuracy, P/R = Precision/Recall, (w) = weighted, (m) = macro, MAE = rata-rata kesalahan
mutlak, "normal" = tanpa bulan tanam yang masa tanamnya kena hujan ekstrem November 2025. Penjelasan lengkap:
[istilah.md](istilah.md).

## Ringkasan

| Kode | Eksperimen | Jalur | Status | Kesimpulan satu kalimat |
|---|---|---|---|---|
| [E01](#e01) | Audit hasil lama | A, B | Historis | Angka lama terlalu optimis karena evaluasinya bocor |
| [E02](#e02) | Model cuaca harian vs tebakan sederhana | B | Aktif | Untuk suhu, BiLSTM setara rata-rata kalender; hujan harian tak bisa ditebak |
| [E03](#e03) | Reproduksi pipeline lama | A | Historis | Baseline identik; Hybrid berubah antar-komputer → perlu banyak seed |
| [E04](#e04) | Cek dimensi embedding (PCA) | A | Historis | 32 angka LSTM hanya setara ±4 informasi |
| [E05](#e05) | Pipeline sesuai proposal, 5 lokasi | C | **Bab IV** | Hybrid terbaik di F1 & Top-3; Accuracy setara LSTM saja |
| [E06](#e06) | Batas atas label dummy | C | Aktif | Akurasi rendah karena label if-else, bukan karena model |
| [E07](#e07) | Per bulan vs per musim tanam | – | Aktif | Hujan per musim jauh lebih stabil (17% vs 52%) |
| [E08](#e08) | Fitur turunan data harian (cara B) | C | Ablation | Tidak meningkatkan Hybrid |
| [E09](#e09) | Pipeline sesuai proposal, 20 lokasi | C | **Bab IV** | Hybrid tetap terbaik di semua metrik; LSTM overfit cepat |
| [E10](#e10) | RF skor kecocokan many-to-many | D | Aktif (web) | RF saja > Hybrid; komoditas tepat ±93% |
| [E11](#e11) | LSTM harian (5 & 1 epoch) untuk RF | D | Ablation | Epoch terbaik = 1; tetap kalah dari RF saja |
| [E12](#e12) | Prediksi hasil & umur panen | D | Aktif (web) | Umur: MAE ±1,7 hari; hasil: Hybrid belum mengalahkan tebakan |
| [E13](#e13) | Hasil panen dengan cuaca biasa | D | Aktif (web) | Cuaca biasa mengalahkan Hybrid 32 dimensi |
| [E14](#e14) | Kenapa RF saja unggul | D | Aktif | Bukan karena kurang informasi; LSTM overfit & sulit menangkap interaksi |

Jalur: **A** = pipeline lama (label if-else, `pipelines/arsip_lama/`), **B** = cuaca harian Rahmat
(`pipelines/cuaca_harian/`), **C** = klasifikasi sesuai proposal (`pipelines/klasifikasi_proposal/`),
**D** = many-to-many varietas (`pipelines/varietas/`).

---

<a id="e01"></a>
## E01 — Audit hasil lama

- **Tujuan:** memeriksa apakah angka di log sebelum perbaikan layak dipakai untuk Bab IV.
- **Cara:** membaca kode dan log run 21 Nov 2026 (jalur A dan notebook Colab jalur B).
- **Hasil di log lama:** Hybrid 20 lokasi Acc 0,425 / F1 0,462 / Top-3 0,625; BiLSTM cuaca "akurasi WAPE 81,27%".
- **Masalah ditemukan:** data uji 2025 dipakai memilih model (BiLSTM); nilai aktual hujan dipotong sebelum dinilai
  (60 hari > 37,8 mm, maks 212 mm); tuning RF memakai pembagian acak/per lokasi; LSTM dan RF belajar dari label
  yang sama; label varietas berupa aturan if-else.
- **Kesimpulan:** angka lama tidak dipakai. Semua eksperimen berikutnya memakai pembagian data kronologis.
- **Folder:** `results/eksperimen/E01_audit_hasil_lama/` (13 run asli, termasuk impor Colab).
- **Status:** historis.

<a id="e02"></a>
## E02 — Model cuaca harian (BiLSTM) vs tebakan sederhana

- **Tujuan:** mengecek apakah BiLSTM cuaca Rahmat lebih baik dari tebakan tanpa model.
- **Cara:** Jan–Mar 2025, 5 lokasi, nilai aktual asli. Pembanding: persistence (besok = hari ini) dan
  klimatologi (rata-rata tanggal yang sama 2020–2024).
- **Perintah:** `python -m pipelines.analisis.e02_cuaca_vs_tebakan`
- **Hasil (MAE):**

| Variabel | BiLSTM | Persistence | Klimatologi |
|---|---|---|---|
| Curah hujan (mm) | 7,08 (R² −0,04) | 8,58 | 7,95 |
| Suhu rata-rata (°C) | 0,353 | 0,364 | 0,354 |
| Suhu maksimal (°C) | 0,664 | 0,875 | **0,657** |
| Suhu minimum (°C) | **0,413** | 0,469 | 0,538 |
| Kelembaban (%) | **2,16** | 2,51 | 2,89 |
| Radiasi | **2,97** | 3,61 | 3,20 |

  "Akurasi WAPE" bila hanya menebak rata-rata konstan: suhu 93–96%, kelembaban 95%. Tebak hujan/tidak:
  BiLSTM 72,7% = selalu tebak "hujan" 72,7%; besok = hari ini 82,9%.
- **Kesimpulan:** angka "81%" menyesatkan. Untuk suhu, BiLSTM tidak lebih baik dari rata-rata kalender;
  hujan harian praktis tidak bisa ditebak.
- **Folder:** `results/eksperimen/E02_cuaca_harian_vs_tebakan/hasil_analisis.txt`
- **Status:** aktif (perlu dicantumkan sebagai pembanding bila model cuaca dibahas).

<a id="e03"></a>
## E03 — Reproduksi pipeline lama

- **Tujuan:** apakah angka log lama bisa didapat ulang.
- **Hasil:** Baseline RF 20 lokasi identik (0,258 / 0,227 / 0,675). Hybrid 20 lokasi 0,367 (log: 0,425) karena
  beda versi torch/hardware; konsisten antar-run di komputer yang sama. 5 lokasi (11 kelas): Baseline 0,183,
  Hybrid 0,317, Hybrid tuned 0,300.
- **Kesimpulan:** satu kali run tidak cukup → eksperimen berikutnya memakai 5 seed.
- **Perintah:** `python -m pipelines.arsip_lama.train_lstm` lalu `python -m pipelines.arsip_lama.train_rf`
- **Status:** historis.

<a id="e04"></a>
## E04 — Cek dimensi embedding (PCA)

- **Hasil (jalur A, 5 lokasi):** 3 dari 32 dimensi selalu 0; 4 komponen PCA menjelaskan 99,7% variasi; porsi
  importance LSTM 81,7% vs porsi jumlah fitur 78%.
- **Kesimpulan:** 32 angka ≈ 4 informasi; LSTM tampak dominan karena jumlah kolomnya. Versi terkini ada di E14.
- **Status:** historis.

<a id="e05"></a>
## E05 — Pipeline sesuai proposal, 5 lokasi

- **Tujuan:** menjalankan rancangan Bab III proposal dengan evaluasi yang bersih.
- **Cara:** iklim bulanan, outlier IQR, Z-Score, input 12 bulan; train 2021–2023 (180), validasi 2024 (60),
  uji 2025 (60); LSTM 128→64→32 dilatih meramal iklim bulan depan; RF di-tuning per tahun; 5 seed.
  Detail: [02_metodologi.md](02_metodologi.md).
- **Perintah:** `LOKASI_AKTIF=L001,L002,L003,L004,L005 python -m pipelines.klasifikasi_proposal.train_hybrid_proposal`
  (atau `python run_all.py --hanya E05`)
- **Hasil (uji 2025, rata-rata ± std 5 seed):**

| Model | Acc | P (w) | R (w) | F1 (w) | F1 (m) | Top-3 |
|---|---|---|---|---|---|---|
| **Hybrid LSTM + RF** | 0,367 ± 0,039 | **0,476** | 0,367 | **0,405 ± 0,043** | **0,288** | **0,843 ± 0,019** |
| RF saja | 0,290 ± 0,022 | 0,156 | 0,290 | 0,194 ± 0,009 | 0,247 | 0,713 ± 0,025 |
| LSTM saja | **0,370** ± 0,062 | 0,283 | **0,370** | 0,308 ± 0,063 | 0,206 | 0,670 ± 0,036 |

- **Kesimpulan:** Hybrid > RF saja di 5/5 seed dan > LSTM saja di 4/5 seed (F1 w). Accuracy setara LSTM saja.
  Laporan lengkap (draf Bab IV): `results/eksperimen/E05_proposal_5_lokasi/20261124_205335_Hybrid_LSTM_RF_Proposal/laporan_hasil_bab4.md`.
- **Commit:** `0414b2f` · **Status:** dipakai untuk Bab IV.

<a id="e06"></a>
## E06 — Batas atas label dummy

- **Tujuan:** menjelaskan kenapa akurasi klasifikasi hanya ±36%.
- **Cara:** membandingkan beberapa cara menebak label pada uji 2025.
- **Perintah:** `python -m pipelines.analisis.e06_batas_label_dummy`
- **Hasil:**

| Cara menebak | 5 lokasi (asli) | 20 lokasi (bundle sekarang) |
|---|---|---|
| Aturan label pada iklim aktual bulan itu | 100% | 100% |
| Aturan label pada iklim bulan sebelumnya | 43% | 41% |
| Aturan label pada ramalan LSTM | 25% | 31% |
| Hybrid Top-1 / Top-3 (seed 42) | 33% / 85% | lihat E09 |

  Error ramalan hujan bulanan: median 73 mm (5 lokasi) / 80 mm (20 lokasi), lebih lebar dari jarak batas kelas
  (55–70 mm).
- **Kesimpulan:** label dummy ditentukan hujan bulan target; batas atas akurasi ±40–50%. Tebakan "sama dengan
  bulan lalu" perlu dicantumkan sebagai pembanding naif di Bab IV.
- **Status:** aktif.

<a id="e07"></a>
## E07 — Per bulan vs per musim tanam

- **Perintah:** `python -m pipelines.analisis.e07_bulan_vs_musim`
- **Hasil (5 lokasi):** variasi antar-tahun hujan per bulan 52%, per musim 17%. Lhoksukon: hujan Januari saja
  46–210 mm; total hujan MT I 996–1.433 mm. Sampel per musim: 55 (5 lokasi), ±220 (20 lokasi).
- **Kesimpulan:** rekomendasi per musim tanam lebih stabil dan sesuai cara petani menanam; butuh 20 lokasi
  dan label asli per musim.
- **Status:** aktif (bahan diskusi desain).

<a id="e08"></a>
## E08 — Fitur turunan data harian (cara B)

- **Cara:** sama dengan E05, ditambah 5 fitur per bulan dari data harian (hari hujan, hari hujan lebat ≥50 mm,
  hujan harian maks, deret kering terpanjang, hari panas).
- **Perintah:** `LOKASI_AKTIF=L001,L002,L003,L004,L005 python -m pipelines.klasifikasi_proposal.train_hybrid_harian`
- **Hasil (uji 2025, 5 seed):**

| Model | Acc | F1 (w) | F1 (m) | Top-3 |
|---|---|---|---|---|
| Hybrid tanpa fitur harian (E05) | 0,367 | **0,405** | **0,288** | 0,843 |
| Hybrid + fitur harian | 0,357 ± 0,048 | 0,376 ± 0,053 | 0,264 | **0,863** |
| LSTM saja + fitur harian | 0,387 | 0,311 | 0,214 | 0,710 |

- **Kesimpulan:** tidak ada peningkatan konsisten (F1 w naik hanya di 1/5 seed); ramalan iklim LSTM memburuk
  (MAE 0,44 → 0,48).
- **Commit:** `64239e8` · **Status:** ablation.

<a id="e09"></a>
## E09 — Pipeline sesuai proposal, 20 lokasi

- **Perintah:** `python -m pipelines.klasifikasi_proposal.train_hybrid_proposal` (config default = 20 lokasi)
- **Hasil (uji 2025: 240 sampel, 12 kelas, 5 seed):**

| Model | Acc | P (w) | F1 (w) | F1 (m) | Top-3 |
|---|---|---|---|---|---|
| **Hybrid LSTM + RF** | **0,367** ± 0,033 | **0,461** | **0,388** ± 0,022 | **0,332** ± 0,017 | **0,788** ± 0,053 |
| RF saja | 0,313 ± 0,016 | 0,215 | 0,243 ± 0,016 | 0,241 | 0,676 ± 0,014 |
| LSTM saja | 0,345 ± 0,034 | 0,204 | 0,246 ± 0,028 | 0,119 | 0,545 ± 0,034 |

- **Kesimpulan:** Hybrid terbaik di semua metrik. Epoch terbaik LSTM hanya 3–5 (5 lokasi: 14–28) karena 20 lokasi
  hanya punya 2 seri iklim unik. Model ini dipakai web sebagai Model 1.
- **Commit:** `9f95e9c` · **Status:** dipakai untuk Bab IV dan web.

<a id="e10"></a>
## E10 — RF skor kecocokan many-to-many

- **Tujuan:** memakai karakteristik varietas (Tujuan 2 proposal) dan menilai ke-25 varietas.
- **Cara:** data dummy lokasi × bulan tanam × varietas (30.000 baris); fitur = vektor LSTM (E09) + 9 tanah +
  8 varietas + 2 musim; target skor kecocokan 0–100; RF regresi; 3 konfigurasi; 5 seed.
- **Perintah:** `python -m pipelines.data.buat_data_dummy_varietas`, `python -m pipelines.data.buat_data_rf_varietas`,
  `python -m pipelines.varietas.train_rf_varietas`
- **Hasil (uji 2025):**

| Model | MAE | R² | MAE normal | R² normal | Spearman normal | Top-3 memuat terbaik (normal) | Komoditas tepat (normal) |
|---|---|---|---|---|---|---|---|
| Hybrid (LSTM + tanah + varietas) | 25,6 | −0,18 | 17,5 | −0,16 | 0,30 | 0,19 | 0,94 |
| **RF saja (rata-rata iklim)** | **23,3** | **0,01** | **14,4** | **0,24** | **0,51** | **0,27** | 0,93 |
| Tanpa iklim | 25,0 | −0,08 | 16,9 | −0,04 | 0,37 | 0,18 | 0,94 |

- **Kesimpulan:** RF saja mengungguli Hybrid. Hujan ekstrem Nov 2025 (782–801 mm) membuat R² semua model negatif.
  Model Hybrid tetap dipakai web sebagai Model 2 (pembahasan lanjut di E14).
- **Commit:** `9f95e9c` · **Status:** aktif (web).

<a id="e11"></a>
## E11 — LSTM harian untuk RF (5 epoch & 1 epoch)

- **Cara:** LSTM dilatih per hari (jendela 30 hari, ramal besok), vektor 32 harian dirata-rata per bulan; RF
  memakai rata-rata bulan t−1.
- **Perintah:** `python -m pipelines.varietas.train_lstm_harian --epoch 5|1 --out ...` lalu
  `buat_data_rf_varietas --lstm-harian ...` dan `train_rf_varietas --data-dir ... --label Harian|Harian1`
  (lihat `run_all.py`).
- **Hasil:** epoch terbaik di validasi = 1 (loss validasi naik sejak epoch 2). RF skor dengan LSTM harian:
  MAE normal 17,3, R² normal −0,09, Spearman normal 0,28 — setara LSTM bulanan, tetap kalah RF saja (0,51).
- **Commit:** `af47cda`, `328e2fc` · **Status:** ablation.

<a id="e12"></a>
## E12 — Prediksi hasil panen & umur panen aktual

- **Cara:** target `hasil_dummy_ton_ha` dan `umur_panen_aktual_dummy` (thermal time); pembanding tebakan tanpa
  model (potensi × rasio rata-rata; umur deskripsi).
- **Perintah:** `python -m pipelines.varietas.train_rf_hasil_umur` (dan `--data-dir data/processed/rf_varietas_harian1 --label Harian1`)
- **Hasil hasil panen (t/ha):**

| Model | MAE | R² | MAE normal | R² normal |
|---|---|---|---|---|
| Tebakan tanpa model | 1,93 | 0,25 | 1,19 | 0,70 |
| Hybrid, LSTM bulanan | 1,98 | 0,20 | 1,35 | 0,58 |
| Hybrid, LSTM harian 1 epoch | 1,91 | 0,25 | 1,31 | 0,62 |
| **RF saja** | **1,79** | **0,33** | **1,11** | **0,72** |

- **Hasil umur panen (hari):** tebakan (umur deskripsi) MAE 4,07; Hybrid bulanan 1,76 (R² 0,98); Hybrid harian
  1,77; tanpa iklim **1,74**; RF saja 2,07.
- **Kesimpulan:** umur panen tertebak hingga batas noise dummy (±2 hari) dan musim sudah cukup; hasil panen:
  Hybrid belum mengalahkan tebakan sederhana.
- **Commit:** `328e2fc` · **Status:** aktif (web).

<a id="e13"></a>
## E13 — Hasil panen dengan variabel cuaca biasa

- **Cara:** fitur cuaca biasa menggantikan vektor LSTM: C1 rata-rata 12 bulan; C2 + rata-rata 3 bulan; C3 +
  normal iklim masa tanam (dari tahun lain). Set dipilih dari MAE validasi.
- **Perintah:** `python -m pipelines.varietas.train_rf_hasil_cuaca`
- **Hasil (t/ha):**

| Set | MAE validasi | MAE uji | R² uji | MAE normal | R² normal |
|---|---|---|---|---|---|
| C1 rata-rata 12 bulan | 1,07 | 1,79 | 0,33 | **1,11** | **0,72** |
| **C2 + rata-rata 3 bulan** (dipilih) | **0,99** | **1,74** | **0,39** | 1,30 | 0,61 |
| C3 + normal iklim masa tanam | 1,02 | 1,81 | 0,35 | 1,25 | 0,65 |
| Acuan Hybrid 32 dimensi (E12) | – | 1,98 | 0,20 | 1,35 | 0,58 |

- **Kesimpulan:** beberapa angka cuaca yang jelas mengalahkan 32 angka abstrak. Web menampilkan keduanya
  berdampingan.
- **Commit:** `685ed3b` · **Status:** aktif (web).

<a id="e14"></a>
## E14 — Kenapa RF saja mengungguli Hybrid

- **Perintah:** `python -m pipelines.analisis.e14_isi_informasi_lstm`
- **Hasil (R² menebak iklim masa tanam pada 2025, Ridge | RF):**

| Fitur | Hujan masa tanam | Suhu masa tanam |
|---|---|---|
| Rata-rata iklim 12 bulan (6 fitur) | −0,08 / −0,37 | 0,89 / 0,78 |
| Vektor LSTM bulanan (32) | −0,03 / −0,07 | 0,92 / 0,97 |
| Vektor LSTM harian 1 epoch (32) | **0,17 / 0,16** | **0,97** / 0,94 |
| Rata-rata bulan kalender | 0,12 | 0,10 |

  3 komponen PCA menjelaskan 92–95% variasi vektor; hanya 240 pola iklim unik dari 1.200 lokasi-bulan.
- **Kesimpulan:** vektor LSTM tidak kekurangan informasi. RF saja unggul karena (1) pola iklim unik sedikit
  sehingga 32 fitur mudah dihafal, (2) interaksi iklim × kebutuhan varietas lebih mudah ditangkap dari satu kolom
  yang jelas, (3) data dummy dibangun dari rata-rata. Usulan lanjutan: beri RF keduanya, mampatkan vektor (PCA 4–8),
  latih LSTM meramal iklim 3–4 bulan, data iklim lebih detail.
- **Status:** aktif.

## Status reproduksi (diuji 27 Nov 2026)

`python run_all.py` dijalankan penuh dari struktur repo baru (19 langkah, 44,9 menit, tanpa error) dan hasilnya
dibandingkan dengan folder run di `results/eksperimen/`:

| Eksperimen | Hasil jalan ulang |
|---|---|
| E02, E05, E07, E08, E09, E12 (4 run), E13, E14 | Identik (selisih 0) |
| E06 | Angka identik; hanya nomor run bundle yang berbeda |
| E11 (1 epoch) | Identik |
| E10, E11 (5 epoch) | MAE, R², Spearman identik; Top-1/Top-3/irisan berbeda ≤ 1,4 poin persen karena run asli dibuat sebelum pemecah skor seri dijadikan stabil (commit `ebda3d4`). Angka E10 di katalog sudah memakai versi stabil |

Model di `models/` saat ini berasal dari jalan ulang tersebut.

---

## Cara menambah eksperimen baru

1. Ambil kode berikutnya (E15, …) dan tambahkan baris di tabel Ringkasan.
2. Commit kode **sebelum** menjalankan, agar log mencatat commit yang bersih.
3. Jalankan dengan kode eksperimen: `EKSPERIMEN=E15_nama_singkat python -m pipelines....`
   (hasil otomatis masuk `results/eksperimen/E15_nama_singkat/`).
4. Tambahkan langkahnya ke `run_all.py` dan entri lengkap di katalog ini.
