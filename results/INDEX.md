# Indeks Hasil

Semua hasil run ada di `results/eksperimen/<kode>/`. Folder run memakai nama asli `<tanggal>_<jam>_<stage>`.
Riwayat semua run (satu baris per run) ada di `experiment_history.csv`; ringkasan perubahan parameter di
`changelog_kode.txt`. Penjelasan tiap eksperimen: [docs/03_katalog_eksperimen.md](../docs/03_katalog_eksperimen.md).

| Folder | Run | Isi penting | Status |
|---|---|---|---|
| `E01_audit_hasil_lama/` | 13 run 21 Nov 2026 (`*_LSTM`, `*_Baseline_RF`, `*_Hybrid_LSTM_RF`, `*_Hybrid_RF_Tuned`, `*_Colab`) | summary.json, grafik Colab | Usang (evaluasi bocor) |
| `E02_cuaca_harian_vs_tebakan/` | analisis | `hasil_analisis.txt` | Aktif |
| `E05_proposal_5_lokasi/` | `20261124_205335_{Hybrid_LSTM_RF,Baseline_RF,Baseline_LSTM}_Proposal` | **`laporan_hasil_bab4.md`**, metrics_per_seed/summary/per_kelas, confusion matrix, kurva loss, rekomendasi Top-3 | **Bab IV** |
| `E05_proposal_5_lokasi/_usang_run_pertama/` | `20261124_204507_*` | run pertama sebelum baseline LSTM ditambahkan | Usang (digantikan 205335) |
| `E06_batas_label_dummy/` | analisis | `hasil_analisis.txt` | Aktif |
| `E07_bulan_vs_musim/` | analisis | `hasil_analisis.txt` | Aktif |
| `E08_fitur_harian_cara_b/` | `20261124_223249_*_Harian` | laporan + metrik (5 lokasi + fitur harian) | Ablation |
| `E09_proposal_20_lokasi/` | `20261125_004642_*_Proposal` | **`laporan_hasil_bab4.md`** + metrik 20 lokasi | **Bab IV**, model web |
| `E10_rf_kecocokan_varietas/` | `20261125_005058_RF_Varietas_Hybrid` | metrik_ringkasan, importance_kelompok, rekomendasi Top-3 | Aktif (web) |
| `E11_lstm_harian_untuk_rf/` | `20261125_011034_RF_Varietas_Hybrid_Harian` (5 epoch), `20261125_014954_RF_Varietas_Hybrid_Harian1` (1 epoch) | metrik RF skor | Ablation |
| `E12_prediksi_hasil_umur/` | `20261125_012952_RF_HASIL_Hybrid`, `20261125_013424_RF_UMUR_Hybrid`, `*_Harian1` | metrik hasil & umur | Aktif (web) |
| `E13_hasil_panen_cuaca_biasa/` | `20261125_021218_RF_HASIL_Cuaca` | metrik C1–C3 | Aktif (web) |
| `E14_isi_informasi_lstm/` | analisis | `hasil_analisis.txt` | Aktif |
| `cuaca_harian/` | – | prediksi Jan–Mar 2025 jalur Rahmat + grafik | Jalur B |

Run baru otomatis masuk ke folder sesuai variabel lingkungan `EKSPERIMEN` (diatur oleh `run_all.py`);
tanpa kode, masuk ke `_tanpa_kode/`.
