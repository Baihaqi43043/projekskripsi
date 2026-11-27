# Istilah

| Istilah | Arti |
|---|---|
| **Accuracy** | Persentase tebakan pertama yang tepat. Pada klasifikasi ini selalu sama dengan Recall (weighted). |
| **Precision** | Dari semua rekomendasi untuk suatu varietas, berapa persen yang benar: TP / (TP + FP). |
| **Recall** | Dari semua kasus yang memang varietas itu, berapa persen yang ditemukan: TP / (TP + FN). |
| **F1-Score** | Rata-rata harmonik Precision dan Recall: 2·P·R / (P + R). |
| **(w) weighted** | Rata-rata antar-kelas yang ditimbang jumlah sampel tiap kelas. |
| **(m) macro** | Rata-rata antar-kelas biasa; semua kelas berbobot sama (menunjukkan kinerja di kelas jarang). |
| **Top-3 / Top-5** | Persentase kasus ketika jawaban yang benar ada di 3 (5) rekomendasi teratas. Top-3 adalah metrik utama sistem rekomendasi; Top-5 kurang informatif pada data ini. |
| **MAE** | Rata-rata selisih mutlak antara prediksi dan nilai sebenarnya (satuan sama dengan target). |
| **RMSE** | Akar rata-rata kuadrat selisih; lebih menghukum kesalahan besar. |
| **R²** | Seberapa besar variasi target dijelaskan model. 1 = sempurna, 0 = setara menebak rata-rata, negatif = lebih buruk dari menebak rata-rata. |
| **Spearman** | Kesamaan urutan peringkat prediksi dengan urutan sebenarnya (−1 s/d 1). |
| **"Normal"** | Evaluasi tanpa bulan tanam yang masa tanamnya kena hujan di atas maksimum data latih (Nov 2025). |
| **Seed** | Angka acak awal. Model dilatih 5 kali dengan seed 42–46; dilaporkan rata-rata ± simpangan baku (std). |
| **Baseline / single model** | Model pembanding satu metode: RF saja, LSTM saja, atau tebakan tanpa model. |
| **Persistence** | Tebakan "nilai berikutnya = nilai terakhir" (besok = hari ini, bulan depan = bulan ini). |
| **Klimatologi** | Tebakan rata-rata tanggal/bulan kalender yang sama pada tahun-tahun sebelumnya. |
| **Data train / validasi / uji** | Train (2021–2023) untuk belajar; validasi (2024) untuk memilih epoch & hyperparameter; uji (2025) dipakai sekali di akhir. |
| **Embedding / vektor temporal** | 32 angka keluaran LSTM yang meringkas pola iklim 12 bulan. |
| **Hybrid** | LSTM sebagai pengekstrak fitur temporal + Random Forest sebagai pengambil keputusan. |
| **Many-to-many** | Satu baris = satu varietas di satu lokasi pada satu bulan tanam (25 baris per lokasi-bulan). |
| **Kebocoran data (data leakage)** | Informasi dari data uji/masa depan ikut terpakai saat training, sehingga nilai evaluasi terlalu bagus. |
| **Stacking leakage** | LSTM dan RF dilatih dengan label yang sama, sehingga fitur LSTM pada data train "terlalu bersih". |
| **Overfit** | Model menghafal data latih sehingga buruk di data baru (loss validasi naik saat loss train turun). |
| **Epoch** | Satu putaran penuh model melihat seluruh data latih. |
| **Early stopping** | Berhenti melatih bila loss validasi tidak membaik selama beberapa epoch; model terbaik yang disimpan. |
| **IQR** | Rentang antar-kuartil (Q3 − Q1); dipakai untuk membatasi nilai pencilan (outlier). |
| **Z-Score** | Normalisasi (nilai − rata-rata) / simpangan baku. |
| **Thermal time** | Konsep agronomi: tanaman matang lebih cepat bila suhu lebih tinggi; dipakai untuk umur panen dummy. |
| **MT I / MT II** | Musim tanam rendeng (Okt–Mar) dan gadu (Apr–Sep). |
