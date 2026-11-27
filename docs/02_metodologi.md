# Metodologi dan Pemetaan ke Proposal

## Pemetaan ke Bab III proposal

| Komponen | Proposal | Implementasi | Lokasi kode |
|---|---|---|---|
| Data iklim | Tabel 3.1: bulanan, 6 variabel, BMKG 2020–2025 | Bulanan, 6 variabel, NASA POWER 2020–2025 | `data/raw/` |
| Data tanah | Tabel 3.2: 9 fitur | 9 fitur (sebagian dummy) | `data/raw/aceh_utara_tanah_statis.csv` |
| Preprocessing | 3.10: interpolasi, outlier IQR, Min-Max/Z-Score, fitur turunan | Interpolasi linier per lokasi, outlier IQR (Q1−1,5·IQR, Q3+1,5·IQR), Z-Score, sin/cos bulan; statistik dari data train saja | `src/pipelines/proposal_pipeline.py` |
| Pembagian data | 3.7: 80/20; 3.11: train/validation/test | Kronologis per tahun bulan target: train ≤2023, validasi 2024, uji 2025 (80% = train + validasi) | `configs/model_config.yaml` → `proposal:` |
| LSTM | 3.10: 2 layer 128 & 64 + dropout → vektor 32 | Sama (`TemporalLSTMEncoder`); dilatih meramal iklim bulan berikutnya (MSE), early stopping di validasi | `src/models/lstm_encoder.py` |
| Random Forest | 3.10: vektor 32 + fitur statis; 3.11: tuning + cross-validation | RandomizedSearchCV, CV expanding-window per tahun (validasi 2023, 2024), refit train+validasi | `pipelines/klasifikasi_proposal/` |
| Recommendation engine | 3.10: Top-K + skor + justifikasi feature importance | Top-3 + probabilitas/skor + alasan per kelompok fitur | `src/app/proposal_inference.py` |
| Evaluasi | 2.7, 3.11: Accuracy, Precision, Recall, F1, confusion matrix, feature importance, vs single model | Sama + F1 macro, Top-3/Top-5, 5 seed (rata-rata ± std), pembanding RF saja & LSTM saja | `src/pipelines/report_proposal.py` |
| Validasi pakar & UAT | 3.11 tahap 8 | Belum | – |

## Penyesuaian dari proposal (wajib ditulis di disertasi)

| Aspek | Proposal | Implementasi | Alasan |
|---|---|---|---|
| Panjang sekuens | 60 bulan | 12 bulan | Data hanya 72 bulan; 60 bulan menyisakan 12 sampel per lokasi |
| Sumber iklim | BMKG | NASA POWER | Data BMKG belum tersedia |
| Lama penyinaran | jam/hari | radiasi MJ/m²/hari | Variabel yang tersedia di NASA POWER |
| Tugas latihan LSTM | tidak ditentukan | peramalan iklim bulan berikutnya | Mencegah kebocoran label ke tahap RF (stacking leakage) |
| Outlier | IQR | IQR (jalur proposal); jalur cuaca harian Rahmat memakai winsorizing persentil + RobustScaler | – |
| Karakteristik varietas | Bab I & Tujuan 2: masuk fitur RF | Hanya pada jalur many-to-many (E10–E14) | Pada desain klasifikasi, karakteristik melekat pada label, bukan sampel |
| Lama data | "5 tahun (2020–2025)" (3.7) | 6 tahun | Koreksi teks proposal |

## Aturan anti-kebocoran yang dipakai
1. Statistik preprocessing (batas IQR, mean/std, scaler) dihitung **hanya dari data train**.
2. Input LSTM untuk bulan target *t* hanya memakai bulan *t−12 … t−1*.
3. Data uji 2025 **tidak** dipakai untuk memilih epoch, learning rate, atau hyperparameter.
4. Cross-validation RF memakai **expanding window per tahun**, bukan acak.
5. Encoder LSTM tidak melihat label varietas (dilatih meramal iklim).
6. Pada data many-to-many, kolom iklim **selama** masa tanam dan komponen skor (`hujan_masa_tanam`,
   `suhu_masa_tanam`, `s_*`) **tidak** dijadikan fitur.
7. Hasil dinilai terhadap nilai aktual asli (tidak dipotong).

## Jalur pipeline

| Jalur | Folder | Isi |
|---|---|---|
| A. Lama | `pipelines/arsip_lama/` | LSTM klasifikasi + RF dengan label if-else; evaluasi lama (bocor). Arsip |
| B. Cuaca harian | `pipelines/cuaca_harian/`, `apps/app_prediksi_cuaca.py`, `notebooks/training_lstm_cuaca.ipynb` | BiLSTM + attention, ramal cuaca besok (Rahmat) |
| C. Klasifikasi proposal | `pipelines/klasifikasi_proposal/` | Bab III proposal, label if-else; E05, E08, E09 |
| D. Many-to-many varietas | `pipelines/data/`, `pipelines/varietas/` | Skor, hasil, umur per varietas; E10–E13; dipakai web |
| Analisis | `pipelines/analisis/` | Script untuk E02, E06, E07, E14 |
