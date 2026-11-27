# Aplikasi Web

## 1. Rekomendasi varietas (versi proposal)

```bash
python -m apps.app_rekomendasi_proposal      # buka http://127.0.0.1:8001
```
Prasyarat: model sudah dibuat (`python run_all.py`, atau minimal `--hanya E09 DATA E10 E12 E13`).

**Alur:** pilih lokasi (20 kecamatan) dan bulan tanam (2021 s/d 1 bulan setelah data terakhir), lalu:

| Bagian | Model | Isi |
|---|---|---|
| Model 1 · LSTM | `models/proposal_bundle.joblib` (E09) | Prakiraan 6 variabel iklim bulan tanam + grafik 12 bulan terakhir |
| Model 2 · RF kecocokan | `models/rf_varietas_hybrid.joblib` (E10) | Top-3 dari 25 varietas, skor 0–100, alasan per kelompok fitur |
| Prediksi hasil (Hybrid 32 dimensi) | `models/rf_hasil_hybrid.joblib` (E12) | t/ha |
| Prediksi hasil (cuaca biasa) | `models/rf_hasil_cuaca.joblib` (E13) | t/ha, untuk dibandingkan |
| Prediksi umur panen | `models/rf_umur_hybrid.joblib` (E12) | hari |
| Pembanding | `data/raw/dummy_varietas/` | Top-3 menurut skor dummy "sebenarnya" |

Kondisi lahan bisa diubah manual (modul input manual, Bagian 3.8 proposal). Bagian bawah halaman menampilkan
metrik uji tiap model.

**Pengaman konsistensi:** model RF menyimpan `run_id` encoder LSTM yang dipakai saat training. Bila
`proposal_bundle.joblib` dilatih ulang tanpa melatih ulang jalur varietas, aplikasi memberi peringatan dan
kembali ke RF klasifikasi lama. Selalu jalankan `run_all.py` (atau E09 → DATA → E10–E13 berurutan).

Kode: `apps/app_rekomendasi_proposal.py` (halaman & API), `src/app/proposal_inference.py` (logika prediksi).
API: `GET /api/meta`, `POST /api/rekomendasi {id_lokasi, periode, tanah?}` dengan `periode = tahun×12 + (bulan−1)`.

## 2. Dashboard cuaca harian (jalur Rahmat)

```bash
python -m apps.app_prediksi_cuaca            # buka http://127.0.0.1:8000
```
Prasyarat: `models/best_lstm_weather.pth` (dari notebook Colab Rahmat atau
`python -m pipelines.cuaca_harian.train_bilstm_weather`) dan `models/best_lstm_weather_preproc.json` bila ada.

Menampilkan prediksi 1 hari ke depan untuk tahun 2025 (5 lokasi) dibandingkan nilai aktual. Catatan: nilai aktual
di dashboard masih dipotong (winsorizing) sebelum dihitung metriknya; lihat E01 & E02.
