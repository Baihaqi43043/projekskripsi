# Laporan

| File | Isi |
|---|---|
| `Ringkasan_Eksperimen_Hybrid.docx` | Ringkasan E01–E14 dalam format Word (tujuan, cara, hasil, kesimpulan; E05 lengkap per metrik, per seed, per varietas) |
| `buat_laporan_word.js` | Script pembuat file Word di atas |

Membuat ulang (perlu Node.js):

```bash
npm install docx@9.5.1
# Pratinjau tanpa Microsoft Word: D:WordCraftwordcraft-cli.exe render <file.docx> hal1.png --page 1
node docs/laporan/buat_laporan_word.js docs/laporan/Ringkasan_Eksperimen_Hybrid.docx
```

Angka di script ini disalin dari `docs/03_katalog_eksperimen.md`; bila katalog berubah, perbarui script lalu jalankan ulang.
Laporan draf Bab IV yang dibuat otomatis per run ada di `results/eksperimen/E05_*/…/laporan_hasil_bab4.md` dan `E09_*`.
