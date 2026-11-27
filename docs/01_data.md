# Data

Semua data mentah ada di `data/raw/` dan ikut di-commit. Data turunan (`data/processed/`) dan model (`models/`)
tidak di-commit; keduanya dibuat ulang oleh `python run_all.py`.

## Daftar file

| File | Isi | Sumber | Riil / dummy |
|---|---|---|---|
| `aceh_utara_agroklimat_harian.csv` | 43.840 baris: 20 lokasi × 2.192 hari (2020–2025) | NASA POWER (community AG) | Riil (estimasi satelit/reanalisis) |
| `aceh_utara_agroklimat_bulanan.csv` | 1.440 baris: 20 lokasi × 72 bulan | agregasi dari harian | Riil |
| `aceh_utara_tanah_statis.csv` | 20 lokasi × 9 fitur tanah/lingkungan | SoilGrids + generator dummy + config | Campuran (lihat bawah) |
| `varietas_tanaman.csv` | 25 varietas padi/jagung/kedelai | 10 dari PDF Tabel 3.7, 15 referensi Kementan | Campuran |
| `dummy_varietas/` | 30.000 baris lokasi × bulan tanam × varietas: skor, hasil, umur | `pipelines.data.buat_data_dummy_varietas` | **Dummy** (lihat `KETERANGAN.txt` di folder itu) |
| `lstm/training/training.csv`, `lstm/testing/testing.csv` | harian 5 lokasi (L001–L005), 2020–2024 / 2025 | potongan file harian | Riil; dipakai jalur cuaca harian Rahmat |

Mengunduh ulang data iklim/tanah: `python -m pipelines.data.fetch_data` (opsi `--only climate|soil`).

## Variabel iklim

| Kolom | Satuan | Catatan |
|---|---|---|
| `curah_hujan` | mm/hari (bulanan: mm/bulan) | |
| `suhu_rata_rata`, `suhu_maksimal`, `suhu_minimum` | °C | |
| `kelembaban_rata_rata` | % | |
| `lama_penyinaran` | **MJ/m²/hari** | Sebenarnya **radiasi matahari** (ALLSKY_SFC_SW_DWN), bukan jam/hari seperti Tabel 3.1 proposal |

## Fitur tanah & lingkungan (9)

| Fitur | Sumber | Status |
|---|---|---|
| `tekstur_tanah`, `ph_tanah`, `karbon_organik` | SoilGrids (grid 250 m) | Riil (estimasi model) |
| `nitrogen`, `fosfor`, `kalium`, `drainase`, `risiko_bencana_hama` | hash ID lokasi (`src/data/data_fetcher.py`) | **Dummy**, konsisten antar-run |
| `ketinggian_tempat` | diisi manual di `configs/model_config.yaml` | Perkiraan |

Kode kategori: tekstur Pasir 0 / Lanau 1 / Lempung 2; drainase Rusak 0 / Baik 1; risiko Rendah 0 / Sedang 1 /
Tinggi 2; kebutuhan air Sedikit 0 / Sedang 1 / Banyak 2.

## Label / target

| Eksperimen | Target | Cara dibuat |
|---|---|---|
| E01–E09 (klasifikasi) | 1 varietas per lokasi-bulan (12 kelas) | aturan if-else `beri_label_varietas_prototipe()` di `src/data/dataset_loader.py`: pita hujan bulan target (85/140/210 mm) + suhu + tanah |
| E10–E14 (many-to-many) | skor kecocokan 0–100, hasil t/ha, umur panen hari | rumus kontinu dari karakteristik varietas vs iklim masa tanam (`pipelines/data/buat_data_dummy_varietas.py`) |

Label if-else hampir sepenuhnya ditentukan **lokasi + pita hujan bulan target** (97% label tertebak dari dua hal
itu), sehingga model pada E01–E09 pada dasarnya meramal kategori hujan bulan depan.

## Keterbatasan penting

1. **Hanya 2 seri iklim unik untuk 20 kecamatan.** Grid NASA POWER ±0,5°: L004–L006, L010, L011, L020 identik;
   14 lokasi lainnya identik. Perbedaan antar-lokasi dalam satu kelompok hanya dari data tanah.
2. **Bukan data BMKG** seperti di proposal (Tabel 3.1).
3. **Hujan ekstrem November 2025** (782–801 mm/bulan; 25–26 Nov 122 dan 156 mm/hari) jatuh di tahun uji.
   Kemungkinan kejadian nyata (banjir Aceh akhir Nov 2025); perlu dikonfirmasi ke BMKG.
4. **Hanya 6 tahun data**, belum cukup untuk input 60 bulan (Bagian 3.10 proposal).
5. Satu titik SoilGrids per kecamatan; C-organik L007 (Samudera) 0,35% jauh di bawah lokasi lain (3–6%), perlu dicek.

## Usulan sumber data pengganti

| Prioritas | Sumber | Mengatasi |
|---|---|---|
| 1 | Data stasiun BMKG (termasuk lama penyinaran jam/hari) | 1, 2, satuan penyinaran |
| 2 | CHIRPS (hujan, ±5 km, sejak 1981) | 1, 4 |
| 3 | ERA5-Land (suhu & kelembaban, ±9 km, sejak 1950) | 1, 4 |
| 4 | NASA POWER 1981–2025 (ubah `start_date` di config) | 4 |
| – | Data uji tanah lab/Dinas; label varietas per kecamatan per musim tanam dari Distan/BPS | data dummy |
