# Log Keputusan

Keputusan penting beserta alasannya, urut waktu. Tambahkan baris baru di bawah bila ada keputusan baru.

| Tanggal | Keputusan | Alasan | Terkait |
|---|---|---|---|
| 24 Nov 2026 | Data mentah (`data/raw/`) di-commit ke Git | Reproduksibilitas; file kecil | – |
| 24 Nov 2026 | Preprocessing BiLSTM cuaca dijadikan satu modul (`src/models/weather_model.py`) + JSON statistik disimpan bersama model | Mencegah beda preprocessing antara training dan inference | Jalur B |
| 24 Nov 2026 | Pipeline sementara memakai 5 lokasi (L001–L005) | Permintaan tim; sama dengan split harian Rahmat | E05, E08 |
| 24 Nov 2026 | Pembagian data kronologis train ≤2023 / validasi 2024 / uji 2025 | Bab 3.7 (80/20) dan 3.11 (train/validation/test); data uji hanya dipakai sekali | E05 dst. |
| 24 Nov 2026 | LSTM dilatih meramal iklim bulan depan, bukan label varietas | Proposal tidak menentukan; mencegah stacking leakage ke RF | E05 dst. |
| 24 Nov 2026 | Input 12 bulan, bukan 60 | Data hanya 72 bulan | E05 dst. |
| 24 Nov 2026 | Setiap hasil dilaporkan rata-rata ± std dari 5 seed | Hasil satu run berubah antar-komputer (E03) | E05 dst. |
| 24 Nov 2026 | Fitur harian dibuat sebagai pipeline terpisah, pipeline utama tidak diubah | Permintaan tim; hasilnya tidak membantu | E08 |
| 25 Nov 2026 | Data dummy many-to-many tanpa if-else, memakai karakteristik varietas | Tujuan 2 proposal; 25 varietas bisa direkomendasikan | E10 |
| 25 Nov 2026 | Config default 20 lokasi | Permintaan tim; keragaman tanah lebih besar | E09 dst. |
| 25 Nov 2026 | Evaluasi many-to-many juga dilaporkan tanpa bulan hujan ekstrem | Hujan Nov 2025 di luar jangkauan data latih | E10–E13 |
| 25 Nov 2026 | Web memakai LSTM bulanan (E09) untuk Model 1 & 2 | Satu encoder untuk semua model; LSTM harian tidak lebih baik | E11 |
| 25 Nov 2026 | Prediksi hasil 32 dimensi dan cuaca biasa ditampilkan berdampingan | Permintaan tim agar bisa dibandingkan | E13 |
| 27 Nov 2026 | Repo dirapikan: script per jalur, run dikelompokkan per eksperimen, katalog E01–E14 | Dokumentasi & reproduksi | – |
| 27 Nov 2026 | Branch `pipeline-proposal` tidak digabung ke `main` | Keputusan tim | – |
