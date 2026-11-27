# Ringkasan Proyek

**Judul proposal:** Pengembangan Model Hybrid Berbasis Machine Learning untuk Pemilihan Varietas Tanaman
Unggulan Pertanian (Salahuddin, Program Doktor Ilmu Teknik USK, 2026). PDF: [proposal/](proposal/).

## Masalah
Rekomendasi varietas tanaman pangan (padi, jagung, kedelai) di Kabupaten Aceh Utara yang mempertimbangkan
pola iklim jangka panjang, bukan hanya kondisi sesaat.

## Pendekatan (sesuai Bab III proposal)
```
Iklim bulanan 12 bulan terakhir ──► LSTM 128 → 64 → Dense 32 ──► vektor temporal (32 angka)
                                                                      │
Tanah & lokasi (9 fitur) [+ karakteristik varietas] ─────────────────┤
                                                                      ▼
                                                     Random Forest ──► Top-3 varietas + skor + alasan
```

## Status per 27 Nov 2026

| Bagian | Status |
|---|---|
| Kode pipeline sesuai proposal | ✅ Selesai, dapat dijalankan ulang (`python run_all.py`) |
| Data iklim | ✅ NASA POWER harian & bulanan 2020–2025, 20 kecamatan (hanya 2 seri unik, lihat [01_data.md](01_data.md)) |
| Data tanah | ⚠️ pH, tekstur, C-organik dari SoilGrids; N, P, K, drainase, risiko hama masih dummy |
| Label varietas | ❌ Dummy (aturan if-else untuk E01–E09; rumus kecocokan untuk E10–E14) |
| Aplikasi web | ✅ Rekomendasi varietas (20 lokasi, 25 varietas) dan dashboard cuaca harian |
| Validasi pakar & UAT (Bab III 3.11 tahap 8) | ❌ Belum |

## Temuan utama sejauh ini
1. Evaluasi lama bocor; angka lama (42,5%, "akurasi cuaca 81%") tidak dipakai (E01, E02).
2. Pada klasifikasi sesuai proposal, **Hybrid mengungguli RF saja dan LSTM saja** pada F1 dan Top-3,
   baik 5 maupun 20 lokasi (E05, E09).
3. Akurasi top-1 rendah (±36%) karena **label dummy** ditentukan hujan bulan depan yang hampir tidak bisa
   diramal (E06).
4. Pada data many-to-many yang memakai karakteristik varietas, **RF dengan rata-rata cuaca biasa mengungguli
   Hybrid** (E10, E12, E13). Penyebabnya bukan kurang informasi, melainkan sedikitnya pola iklim unik dan
   tugas latihan LSTM yang tidak sejalan dengan target (E14).
5. Rekomendasi per **musim tanam** lebih realistis dan lebih stabil daripada per bulan (E07).

## Langkah berikutnya yang paling berdampak
1. **Label asli** per kecamatan per musim tanam dari Dinas Pertanian/BPS.
2. **Data tanah asli** (N, P, K, drainase, risiko hama).
3. **Data iklim lebih detail** (BMKG stasiun, CHIRPS, ERA5-Land) dan lebih panjang (1981–2025) untuk input
   60 bulan sesuai proposal.
4. Eksperimen lanjutan yang sudah diusulkan di E14: RF diberi vektor LSTM + cuaca biasa, PCA 4–8 dimensi,
   LSTM meramal iklim masa tanam.

Rincian semua percobaan: [03_katalog_eksperimen.md](03_katalog_eksperimen.md).
