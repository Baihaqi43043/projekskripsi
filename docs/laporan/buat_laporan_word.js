// Pembuat laporan Word ringkasan eksperimen (E01-E14).
// Setiap eksperimen memakai format yang sama:
//   1. Data input  2. Data yang diproses  3. Output  4. Akurasi model  5. Ringkasan
// Pemakaian: npm install docx@9.5.1 && node docs/laporan/buat_laporan_word.js docs/laporan/Ringkasan_Eksperimen_Hybrid.docx
// Angka disalin dari docs/03_katalog_eksperimen.md dan results/eksperimen/<kode>/.

const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, BorderStyle, Footer, PageNumber, PageBreak,
} = require("docx");

const FONT = "Calibri";
const ACCENT = "2E6B46";
const W = 9026; // lebar isi A4, margin 1 inci

// ------------------------------------------------------------------ helper
const runs = parts => (Array.isArray(parts) ? parts : [parts]).map(t => (typeof t === "string" ? new TextRun(t) : t));
const p = (text, opts = {}) => new Paragraph({ spacing: { after: 120 }, ...opts, children: runs(text) });
const b = t => new TextRun({ text: t, bold: true });
const i = t => new TextRun({ text: t, italics: true });
const h1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const h3 = t => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const bullet = (parts, ref = "bul") => new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60 }, children: runs(parts) });
const gap = () => new Paragraph({ spacing: { after: 80 }, children: [] });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

const kotak = (label, parts, fill = "EEF5F0", warna = ACCENT) => new Paragraph({
  spacing: { before: 120, after: 200 }, indent: { left: 200 },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: warna, space: 8 } },
  shading: { type: ShadingType.CLEAR, fill, color: "auto" },
  children: [b(label), ...runs(parts)],
});

const garis = { style: BorderStyle.SINGLE, size: 4, color: "C9D3CB" };
const borders = { top: garis, bottom: garis, left: garis, right: garis };
function table(rows, widths, { numCols = [], highlight = [], size = 19, header = true, labelCol = false } = {}) {
  return new Table({
    width: { size: widths.reduce((a, c) => a + c, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((row, r) => new TableRow({
      tableHeader: header && r === 0,
      children: row.map((cell, c) => {
        const isHead = header && r === 0;
        const isLabel = labelCol && c === 0;
        return new TableCell({
          borders,
          width: { size: widths[c], type: WidthType.DXA },
          margins: { top: 60, bottom: 60, left: 100, right: 100 },
          shading: isHead ? { type: ShadingType.CLEAR, fill: "DCEBE1", color: "auto" }
            : isLabel ? { type: ShadingType.CLEAR, fill: "F4F7F4", color: "auto" }
            : highlight.includes(r) ? { type: ShadingType.CLEAR, fill: "F1F7F3", color: "auto" } : undefined,
          children: String(cell).split("\n").map(line => new Paragraph({
            alignment: numCols.includes(c) && !isHead ? AlignmentType.RIGHT : AlignmentType.LEFT,
            children: [new TextRun({ text: line, bold: isHead || isLabel || highlight.includes(r), size })],
          })),
        });
      }),
    })),
  });
}
// Tabel 2 kolom "label | isi" untuk Data input, Data yang diproses, Output
const daftar = rows => table(rows, [2400, W - 2400], { header: false, labelCol: true });

function eksperimen(kode, judul, status, { input, proses, output, akurasi, ringkasan }) {
  return [
    h1(`${kode} · ${judul}`),
    p([i("Status: "), i(status)]),
    h2("1. Data input"), daftar(input),
    h2("2. Data yang diproses"), daftar(proses),
    h2("3. Output"), daftar(output),
    h2("4. Akurasi model"), ...akurasi,
    h2("5. Ringkasan"), kotak("Apa yang dilakukan dan hasilnya: ", ringkasan),
  ];
}

// ------------------------------------------------------------------ data bersama
const IKLIM6 = "6 variabel iklim: curah hujan, suhu rata-rata, suhu maksimal, suhu minimum, kelembaban, radiasi matahari";
const TANAH9 = "9 fitur tanah/lokasi: tekstur, pH, N*, P*, K*, C-organik, drainase*, ketinggian, risiko hama* (* = dummy)";
const VAR8 = "8 karakteristik varietas: umur panen, potensi hasil, ketahanan hama, kebutuhan air, ketinggian maksimal, komoditas (padi/jagung/kedelai)";
const SPLIT = "Kronologis menurut tahun bulan target: train 2021–2023, validasi 2024, uji 2025 (dipakai sekali di akhir)";
const LSTM_ARS = "LSTM 128 unit → LSTM 64 unit (dropout 0,2) → Dense 32 (ReLU) = vektor temporal 32 angka";
const RF_TUNING = "Random Forest, RandomizedSearchCV, validasi per tahun (2023, 2024), refit di train + validasi; 5 seed (42–46)";

// ------------------------------------------------------------------ isi
const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Ringkasan Eksperimen Model Hybrid LSTM–Random Forest")] }),
  p([i("Sistem rekomendasi varietas tanaman pangan · Kabupaten Aceh Utara · 27 November 2026")]),
  kotak("Status: prototipe. ", "Label varietas dibuat dengan aturan dummy (E01–E09) atau rumus dummy (E10–E14), dan 5 dari 9 fitur tanah (N, P, K, drainase, risiko hama) masih dummy. Angka di dokumen ini menguji apakah pipeline bekerja benar dan belum boleh diklaim sebagai temuan penelitian.", "FFF4DC", "C9A24A"),
  p("Setiap eksperimen disajikan dengan format yang sama: (1) data input, (2) data yang diproses, (3) output, (4) akurasi model, dan (5) ringkasan apa yang dilakukan dan hasilnya. Semua eksperimen aktif dapat diulang dengan perintah python run_all.py; rincian teknis ada di docs/03_katalog_eksperimen.md."),

  h1("Gambaran umum data"),
  table([
    ["Data", "Isi", "Sumber", "Status"],
    ["Iklim harian", "20 lokasi × 2.192 hari (2020–2025), 6 variabel", "NASA POWER", "Riil (estimasi satelit/reanalisis)"],
    ["Iklim bulanan", "20 lokasi × 72 bulan, agregasi dari harian", "NASA POWER", "Riil"],
    ["Tanah & lokasi", "20 lokasi × 9 fitur", "SoilGrids + dummy + config", "pH, tekstur, C-organik riil; N, P, K, drainase, risiko dummy"],
    ["Varietas", "25 varietas padi/jagung/kedelai, 8 karakteristik", "PDF Tabel 3.7 (10) + referensi Kementan (15)", "Sebagian dummy"],
    ["Label klasifikasi (E01–E09)", "1 varietas per lokasi-bulan, 11–12 kelas", "Aturan if-else (pita hujan bulan target + tanah)", "Dummy"],
    ["Data many-to-many (E10–E14)", "20 lokasi × 60 bulan tanam × 25 varietas = 30.000 baris; skor, hasil, umur", "Rumus kecocokan dari karakteristik varietas", "Dummy"],
  ], [1900, 2900, 2200, 2026]),
  gap(),
  p([b("Keterbatasan data: "), new TextRun("20 kecamatan hanya memiliki 2 seri iklim unik (grid NASA POWER ±55 km); hujan ekstrem November 2025 (782–801 mm/bulan) jatuh di tahun uji; data hanya 6 tahun; \"lama penyinaran\" sebenarnya radiasi matahari (MJ/m²/hari).")]),
  p([b("Alur model: "), new TextRun("iklim 12 bulan terakhir → LSTM → vektor 32 angka → digabung dengan data tanah (dan karakteristik varietas) → Random Forest → Top-3 varietas + skor + alasan.")]),

  h1("Ringkasan semua eksperimen"),
  table([
    ["Kode", "Eksperimen", "Input utama", "Output", "Akurasi utama", "Hasil"],
    ["E01", "Audit hasil lama", "Log & kode lama", "Daftar masalah", "Log lama: Acc 0,425", "Evaluasi lama bocor"],
    ["E02", "Cuaca harian vs tebakan", "Harian 5 lokasi, prediksi BiLSTM", "MAE per variabel", "Suhu: BiLSTM ≈ klimatologi", "BiLSTM tidak lebih baik untuk suhu"],
    ["E03", "Reproduksi pipeline lama", "Bulanan 20/5 lokasi", "Metrik ulang", "Hybrid 0,367 (log 0,425)", "Hasil berubah antar-komputer"],
    ["E04", "Cek embedding LSTM", "Vektor 32 jalur lama", "Analisis PCA", "4 komponen = 99,7%", "32 angka ≈ 4 informasi"],
    ["E05", "Proposal, 5 lokasi", "Bulanan 5 lokasi + tanah", "Top-3 varietas", "F1 0,405; Top-3 0,843", "Hybrid terbaik di F1 & Top-3"],
    ["E06", "Batas label dummy", "Model E05/E09", "Pembanding naif", "Bulan lalu 43%", "Akurasi dibatasi label"],
    ["E07", "Bulan vs musim", "Bulanan 5 lokasi", "Variasi hujan", "52% vs 17%", "Per musim lebih stabil"],
    ["E08", "Fitur harian", "E05 + 5 fitur harian", "Top-3 varietas", "F1 0,376", "Tidak membantu"],
    ["E09", "Proposal, 20 lokasi", "Bulanan 20 lokasi + tanah", "Top-3 + model web", "F1 0,388; F1 (m) 0,332", "Hybrid terbaik semua metrik"],
    ["E10", "Skor kecocokan 25 varietas", "Many-to-many + vektor LSTM", "Skor 0–100", "Spearman 0,30 vs RF 0,51", "RF saja > Hybrid"],
    ["E11", "LSTM harian untuk RF", "Harian 20 lokasi", "Vektor bulanan", "Spearman 0,28", "Tidak lebih baik"],
    ["E12", "Hasil & umur panen", "Many-to-many", "t/ha dan hari", "Umur MAE 1,76 hari", "Umur baik; hasil < tebakan"],
    ["E13", "Hasil dengan cuaca biasa", "Many-to-many + cuaca", "t/ha", "MAE 1,74 vs Hybrid 1,98", "Cuaca biasa > Hybrid"],
    ["E14", "Kenapa RF saja unggul", "Vektor LSTM & cuaca", "R² informasi iklim", "LSTM ≥ rata-rata", "Overfit, bukan kurang info"],
  ], [700, 1700, 1800, 1300, 1800, 1726], { size: 17 }),

  pageBreak(),

  // ---------------------------------------------------------------- E01
  ...eksperimen("E01", "Audit hasil lama", "historis, tidak dipakai", {
    input: [
      ["Data", "Log 21 Nov 2026: results/experiment_history.csv, 13 folder run (jalur lama dan notebook Colab)"],
      ["Kode", "train_lstm.py, train_rf.py (sekarang pipelines/arsip_lama/), notebook cuaca Rahmat"],
      ["Model yang diaudit", "LSTM klasifikasi + RF (label if-else, 5 dan 20 lokasi); BiLSTM cuaca harian"],
    ],
    proses: [
      ["Langkah", "Membaca kode training dan evaluasi; memeriksa cara data dibagi, cara model dipilih, dan cara nilai dihitung"],
    ],
    output: [
      ["Hasil", "Daftar masalah evaluasi (lihat akurasi di bawah)"],
      ["Folder", "results/eksperimen/E01_audit_hasil_lama/"],
    ],
    akurasi: [
      table([
        ["Model (log lama)", "Accuracy", "F1 (w)", "Top-3"],
        ["Baseline RF, 20 lokasi", "0,258", "0,227", "0,675"],
        ["Hybrid LSTM + RF, 20 lokasi", "0,425", "0,462", "0,625"],
        ["Hybrid tuned, 20 lokasi", "0,350", "0,406", "0,663"],
        ["BiLSTM cuaca harian", "\"akurasi WAPE 81,27%\"", "", ""],
      ], [4226, 1600, 1600, 1600], { numCols: [1, 2, 3] }),
      gap(),
      table([
        ["Masalah ditemukan", "Akibat"],
        ["Data uji 2025 dipakai memilih model terbaik (BiLSTM)", "Nilai uji terlalu optimis"],
        ["Nilai hujan aktual dipotong sebelum dinilai (60 hari > 37,8 mm, maks 212 mm)", "Hari hujan ekstrem hilang dari penilaian"],
        ["Tuning RF memakai pembagian acak / per lokasi", "Model tuned justru lebih buruk"],
        ["LSTM dan RF belajar dari label yang sama", "RF terlalu percaya fitur LSTM"],
        ["Label varietas berupa aturan if-else", "Akurasi hanya mengukur tebakan aturan"],
      ], [5026, 4000]),
    ],
    ringkasan: "Hasil lama diperiksa dan ditemukan lima masalah evaluasi. Angka lama tidak dipakai; semua eksperimen berikutnya memakai pembagian data kronologis dan data uji yang hanya dipakai sekali.",
  }),

  // ---------------------------------------------------------------- E02
  ...eksperimen("E02", "Model cuaca harian vs tebakan sederhana", "aktif", {
    input: [
      ["Data", "Iklim harian 5 lokasi (L001–L005): train 2020–2024, uji 2025 (data/raw/lstm/)"],
      ["Prediksi", "Hasil BiLSTM Rahmat Jan–Mar 2025: 450 baris (5 lokasi × 90 hari)"],
    ],
    proses: [
      ["Pembanding", "Persistence (besok = hari ini); klimatologi (rata-rata tanggal yang sama 2020–2024, dihaluskan 15 hari)"],
      ["Penilaian", "MAE dan R² terhadap nilai aktual asli (tidak dipotong)"],
      ["Script", "python -m pipelines.analisis.e02_cuaca_vs_tebakan"],
    ],
    output: [["File", "results/eksperimen/E02_cuaca_harian_vs_tebakan/hasil_analisis.txt"]],
    akurasi: [
      table([
        ["Variabel (MAE)", "BiLSTM", "Persistence", "Klimatologi"],
        ["Curah hujan (mm)", "7,08 (R² −0,04)", "8,58", "7,95"],
        ["Suhu rata-rata (°C)", "0,353", "0,364", "0,354"],
        ["Suhu maksimal (°C)", "0,664", "0,875", "0,657"],
        ["Suhu minimum (°C)", "0,413", "0,469", "0,538"],
        ["Kelembaban (%)", "2,16", "2,51", "2,89"],
        ["Radiasi (MJ/m²/hari)", "2,97", "3,61", "3,20"],
      ], [3026, 2000, 2000, 2000], { numCols: [1, 2, 3] }),
      gap(),
      table([
        ["Tebak hujan / tidak hujan", "Akurasi"],
        ["BiLSTM (Jan–Mar 2025)", "72,7%"],
        ["Selalu tebak \"hujan\"", "72,7%"],
        ["Besok = hari ini (2025)", "82,9%"],
      ], [6026, 3000], { numCols: [1], highlight: [3] }),
      p("\"Akurasi WAPE\" bila hanya menebak rata-rata konstan: suhu 93–96%, kelembaban 95%, curah hujan −9%."),
    ],
    ringkasan: "Model cuaca harian dibandingkan dengan tebakan tanpa model. Untuk suhu, BiLSTM setara rata-rata kalender; tebakan hujan/tidak hujan sama dengan selalu menebak \"hujan\". Angka \"81%\" menyesatkan karena tebakan konstan pun mendapat 95% untuk suhu.",
  }),

  // ---------------------------------------------------------------- E03
  ...eksperimen("E03", "Reproduksi pipeline lama", "historis", {
    input: [
      ["Data", "Iklim bulanan + tanah, 20 lokasi dan 5 lokasi; label if-else (12 dan 11 kelas)"],
      ["Kode", "pipelines/arsip_lama/train_lstm.py, train_rf.py"],
    ],
    proses: [["Langkah", "Menjalankan ulang pipeline lama di laptop proyek, dua kali, lalu dibandingkan dengan log"]],
    output: [["Hasil", "Metrik ulang Baseline RF, Hybrid, Hybrid tuned"]],
    akurasi: [
      table([
        ["Percobaan", "Accuracy", "F1 (w)", "Top-3", "Catatan"],
        ["Baseline RF, 20 lokasi", "0,258", "0,227", "0,675", "Sama persis dengan log"],
        ["Hybrid, 20 lokasi", "0,367", "0,374", "0,583", "Log: 0,425 (beda versi torch/hardware)"],
        ["Baseline RF, 5 lokasi", "0,183", "0,163", "0,633", ""],
        ["Hybrid, 5 lokasi", "0,317", "0,352", "0,900", ""],
        ["Hybrid tuned, 5 lokasi", "0,300", "0,334", "0,850", "Tuning lebih buruk"],
      ], [2626, 1100, 1100, 1100, 3100], { numCols: [1, 2, 3] }),
    ],
    ringkasan: "Baseline bisa diulang persis, tetapi Hybrid berubah antar-komputer. Karena itu semua eksperimen berikutnya diulang dengan 5 seed dan dilaporkan rata-rata ± simpangan baku.",
  }),

  // ---------------------------------------------------------------- E04
  ...eksperimen("E04", "Cek dimensi embedding LSTM", "historis (versi terkini di E14)", {
    input: [["Data", "Vektor 32 angka dari LSTM jalur lama (5 lokasi) dan model RF Hybrid-nya"]],
    proses: [["Langkah", "Analisis PCA pada vektor 32 angka; membandingkan porsi feature importance dengan porsi jumlah fitur"]],
    output: [["Hasil", "Jumlah informasi efektif di dalam vektor"]],
    akurasi: [
      table([
        ["Ukuran", "Nilai"],
        ["Dimensi yang selalu 0", "3 dari 32"],
        ["Variasi dijelaskan 4 komponen PCA", "99,7%"],
        ["Porsi jumlah fitur LSTM", "78%"],
        ["Porsi feature importance LSTM", "81,7%"],
      ], [6026, 3000], { numCols: [1] }),
    ],
    ringkasan: "Isi 32 angka LSTM hanya setara sekitar 4 angka. LSTM tampak dominan di Random Forest terutama karena jumlah kolomnya banyak.",
  }),

  // ---------------------------------------------------------------- E05
  ...eksperimen("E05", "Pipeline sesuai proposal, 5 lokasi", "dipakai untuk Bab IV", {
    input: [
      ["Iklim", `Bulanan, 5 lokasi (L001–L005), 2020–2025; ${IKLIM6}`],
      ["Tanah", TANAH9],
      ["Label", "Varietas dari aturan if-else, 11 kelas"],
      ["Jumlah sampel", "Train 180 · validasi 60 · uji 60 (1 sampel = 1 lokasi × 1 bulan target)"],
    ],
    proses: [
      ["Pembersihan", "Interpolasi missing value; outlier IQR (Q1 − 1,5·IQR s/d Q3 + 1,5·IQR, 14 nilai dibatasi); Z-Score; indeks musim sin/cos; statistik dari data train saja"],
      ["Sekuens", "Input LSTM = iklim 12 bulan sebelum bulan target (12 × 8)"],
      ["Pembagian data", SPLIT],
      ["LSTM", `${LSTM_ARS}; dilatih meramal iklim bulan berikutnya (tanpa label), early stopping di validasi`],
      ["Random Forest", `Input = vektor 32 + 9 fitur tanah (41 fitur); ${RF_TUNING}`],
      ["Pembanding", "RF saja (tanah + rata-rata iklim 12 bulan) dan LSTM saja (klasifikasi langsung)"],
    ],
    output: [
      ["Prediksi", "Top-3 varietas + skor kecocokan + alasan, untuk 60 lokasi-bulan uji"],
      ["File", "laporan_hasil_bab4.md, metrik per seed/per kelas, confusion matrix, kurva loss (results/eksperimen/E05_proposal_5_lokasi/)"],
    ],
    akurasi: [
      h3("Ringkasan metrik (uji 2025, rata-rata ± std 5 seed)"),
      table([
        ["Metrik", "Hybrid LSTM + RF", "RF saja", "LSTM saja", "Terbaik"],
        ["Accuracy", "0,367 ± 0,039", "0,290 ± 0,022", "0,370 ± 0,062", "LSTM ≈ Hybrid"],
        ["Precision (w)", "0,476 ± 0,051", "0,156 ± 0,006", "0,283 ± 0,067", "Hybrid"],
        ["Recall (w)", "0,367 ± 0,039", "0,290 ± 0,022", "0,370 ± 0,062", "LSTM ≈ Hybrid"],
        ["F1-Score (w)", "0,405 ± 0,043", "0,194 ± 0,009", "0,308 ± 0,063", "Hybrid"],
        ["Precision (m)", "0,324 ± 0,040", "0,201 ± 0,004", "0,192 ± 0,060", "Hybrid"],
        ["Recall (m)", "0,273 ± 0,027", "0,382 ± 0,028", "0,250 ± 0,047", "RF saja"],
        ["F1-Score (m)", "0,288 ± 0,033", "0,247 ± 0,009", "0,206 ± 0,053", "Hybrid"],
        ["Top-3", "0,843 ± 0,019", "0,713 ± 0,025", "0,670 ± 0,036", "Hybrid"],
        ["Top-5", "0,997 ± 0,007", "1,000 ± 0,000", "0,813 ± 0,018", "kurang informatif"],
      ], [1726, 1900, 1900, 1900, 1600], { numCols: [1, 2, 3] }),
      p([i("Accuracy selalu sama dengan Recall (w). Top-5 RF = 1,000 tidak berarti RF sangat baik: dengan 5 lokasi, 5 tebakan hampir pasti mencakup label yang benar.")]),
      h3("F1-Score (w) per seed"),
      table([
        ["Seed", "Hybrid", "LSTM saja", "RF saja"],
        ["42", "0,369", "0,386", "0,200"],
        ["43", "0,418", "0,366", "0,186"],
        ["44", "0,475", "0,265", "0,204"],
        ["45", "0,377", "0,253", "0,183"],
        ["46", "0,386", "0,267", "0,199"],
      ], [2026, 2333, 2333, 2334], { numCols: [1, 2, 3] }),
      h3("Kinerja per varietas (Hybrid, seed 42)"),
      table([
        ["Varietas", "Jumlah uji", "Precision", "Recall", "F1 Hybrid", "F1 RF", "F1 LSTM"],
        ["Padi Inpari 32", "6", "0,714", "0,833", "0,769", "0,667", "0,167"],
        ["Padi Inpari 33", "3", "0,500", "0,667", "0,571", "0,462", "0,000"],
        ["Jagung BISI 18", "12", "0,625", "0,417", "0,500", "0,000", "0,750"],
        ["Padi Ciherang", "12", "0,556", "0,417", "0,476", "0,000", "0,500"],
        ["Kedelai Anjasmoro", "9", "0,500", "0,222", "0,308", "0,182", "0,353"],
        ["Jagung NK 212", "6", "0,333", "0,167", "0,222", "0,000", "0,667"],
        ["Jagung Lamuru", "3", "0,000", "0,000", "0,000", "0,308", "0,000"],
        ["Padi IR64", "6", "0,000", "0,000", "0,000", "0,471", "0,000"],
        ["Padi Mekongga", "3", "0,000", "0,000", "0,000", "0,400", "0,000"],
      ], [2226, 1100, 1100, 1000, 1200, 1200, 1200], { numCols: [1, 2, 3, 4, 5, 6] }),
      h3("Pelatihan LSTM dan kualitas ramalan iklim"),
      table([
        ["Seed", "Epoch terbaik", "MSE validasi", "MAE uji LSTM", "MAE persistence", "MAE klimatologi"],
        ["42", "14", "0,385", "0,451", "0,595", "0,812"],
        ["43", "23", "0,430", "0,448", "0,595", "0,812"],
        ["44", "24", "0,435", "0,429", "0,595", "0,812"],
        ["45", "14", "0,427", "0,435", "0,595", "0,812"],
        ["46", "28", "0,403", "0,434", "0,595", "0,812"],
      ], [1026, 1400, 1400, 1600, 1800, 1800], { numCols: [1, 2, 3, 4, 5] }),
      h3("Kontribusi fitur (feature importance, rata-rata 5 seed)"),
      table([
        ["Kelompok fitur", "Kontribusi"],
        ["Fitur temporal LSTM (32 kolom)", "75,2%"],
        ["Ketinggian tempat", "4,9%"],
        ["Risiko bencana/hama (dummy)", "4,1%"],
        ["Nitrogen (dummy)", "4,0%"],
        ["pH tanah", "3,9%"],
        ["Fosfor (dummy)", "3,6%"],
        ["C-organik", "3,6%"],
        ["Tekstur, kalium, drainase", "≤ 0,3%"],
      ], [6026, 3000], { numCols: [1] }),
      h3("Hyperparameter Random Forest terbaik (Hybrid)"),
      table([
        ["Seed", "n_estimators", "max_depth", "min_split", "min_leaf", "max_features", "class_weight"],
        ["42", "200", "10", "5", "4", "sqrt", "balanced"],
        ["43", "200", "10", "10", "2", "log2", "balanced_subsample"],
        ["44", "200", "15", "2", "4", "sqrt", "balanced_subsample"],
        ["45", "100", "10", "10", "2", "sqrt", "balanced_subsample"],
        ["46", "500", "None", "2", "4", "sqrt", "balanced"],
      ], [826, 1300, 1150, 1100, 1050, 1300, 2300], { numCols: [1, 2, 3, 4], size: 18 }),
    ],
    ringkasan: "Rancangan Bab III proposal dijalankan dengan evaluasi bersih. Hybrid mengungguli RF saja di 5 dari 5 seed dan LSTM saja di 4 dari 5 seed (F1 w), dengan Top-3 84%. Accuracy Hybrid setara LSTM saja. LSTM benar-benar belajar pola iklim karena ramalannya mengalahkan persistence dan klimatologi.",
  }),

  // ---------------------------------------------------------------- E06
  ...eksperimen("E06", "Batas atas label dummy", "aktif", {
    input: [
      ["Model", "Bundle LSTM + RF (E05 untuk 5 lokasi; E09 untuk 20 lokasi)"],
      ["Data", "Uji 2025: 60 lokasi-bulan (5 lokasi) / 240 (20 lokasi)"],
    ],
    proses: [
      ["Langkah", "Menerapkan aturan label if-else ke beberapa sumber iklim (aktual, bulan lalu, ramalan LSTM) lalu dibandingkan dengan label sebenarnya"],
      ["Script", "python -m pipelines.analisis.e06_batas_label_dummy"],
    ],
    output: [["File", "results/eksperimen/E06_batas_label_dummy/hasil_analisis.txt"]],
    akurasi: [
      table([
        ["Cara menebak label", "5 lokasi", "20 lokasi"],
        ["Aturan pada iklim aktual bulan itu", "100%", "100%"],
        ["Aturan pada iklim bulan sebelumnya", "43%", "41%"],
        ["Aturan pada ramalan LSTM", "25%", "31%"],
        ["Hybrid Top-1 / Top-3 (seed 42)", "33% / 85%", "lihat E09"],
        ["Error ramalan hujan bulanan (median)", "73 mm", "80 mm"],
      ], [5026, 2000, 2000], { numCols: [1, 2] }),
    ],
    ringkasan: "Label dummy ditentukan oleh hujan bulan target dengan jarak antar-batas kelas 55–70 mm, sedangkan ramalan hujan meleset sekitar 75 mm. Batas atas akurasi sekitar 40–50%. Tebakan \"sama dengan bulan lalu\" mengalahkan Top-1 Hybrid dan perlu dicantumkan sebagai pembanding naif.",
  }),

  // ---------------------------------------------------------------- E07
  ...eksperimen("E07", "Rekomendasi per bulan vs per musim tanam", "aktif (bahan desain)", {
    input: [["Data", "Hujan bulanan 5 lokasi (L001–L005), 2020–2025"]],
    proses: [
      ["Langkah", "Menghitung koefisien variasi hujan antar-tahun per bulan dan per musim (MT I Okt–Mar, MT II Apr–Sep)"],
      ["Script", "python -m pipelines.analisis.e07_bulan_vs_musim"],
    ],
    output: [["File", "results/eksperimen/E07_bulan_vs_musim/hasil_analisis.txt"]],
    akurasi: [
      table([
        ["Ukuran", "Nilai"],
        ["Variasi antar-tahun hujan per bulan", "52%"],
        ["Variasi antar-tahun hujan per musim", "17%"],
        ["Lhoksukon: hujan Januari saja (2020–2025)", "46–210 mm"],
        ["Lhoksukon: total hujan MT I", "996–1.433 mm"],
        ["Jumlah sampel per musim (5 / 20 lokasi)", "55 / ±220"],
      ], [6026, 3000], { numCols: [1] }),
    ],
    ringkasan: "Hujan per musim jauh lebih stabil daripada per bulan, sesuai cara petani menanam satu varietas per musim. Pendekatan per musim membutuhkan 20 lokasi dan label asli per musim tanam.",
  }),

  // ---------------------------------------------------------------- E08
  ...eksperimen("E08", "Fitur turunan data harian (cara B)", "ablation", {
    input: [
      ["Data", "Sama dengan E05 (5 lokasi) + data harian 5 lokasi"],
      ["Fitur tambahan", "5 fitur per bulan: hari hujan, hari hujan lebat (≥ 50 mm), hujan harian maksimum, deret kering terpanjang, hari panas (≥ 30,3 °C)"],
    ],
    proses: [
      ["Langkah", "Fitur harian diringkas per bulan, dinormalisasi Z-Score, ditambahkan ke input LSTM (12 × 13); sisanya identik dengan E05"],
      ["Script", "python -m pipelines.klasifikasi_proposal.train_hybrid_harian"],
    ],
    output: [["File", "results/eksperimen/E08_fitur_harian_cara_b/"]],
    akurasi: [
      table([
        ["Model", "Accuracy", "F1 (w)", "F1 (m)", "Top-3"],
        ["Hybrid tanpa fitur harian (E05)", "0,367", "0,405", "0,288", "0,843"],
        ["Hybrid + fitur harian", "0,357 ± 0,048", "0,376 ± 0,053", "0,264", "0,863"],
        ["LSTM saja + fitur harian", "0,387", "0,311", "0,214", "0,710"],
        ["RF saja", "0,290", "0,194", "0,247", "0,713"],
      ], [3226, 1500, 1500, 1300, 1500], { numCols: [1, 2, 3, 4] }),
      p("F1 (w) Hybrid naik hanya di 1 dari 5 seed; MAE ramalan iklim LSTM memburuk dari 0,44 menjadi 0,48."),
    ],
    ringkasan: "Menambah rangkuman data harian tidak meningkatkan Hybrid secara konsisten. Dengan 180 sampel train, input yang lebih banyak membuat model lebih mudah overfit, dan informasi hujan bulan depan tetap tidak ada di data harian bulan sebelumnya.",
  }),

  // ---------------------------------------------------------------- E09
  ...eksperimen("E09", "Pipeline sesuai proposal, 20 lokasi", "dipakai untuk Bab IV dan web (Model 1)", {
    input: [
      ["Iklim", `Bulanan, 20 lokasi, 2020–2025; ${IKLIM6} (hanya 2 seri unik)`],
      ["Tanah", TANAH9],
      ["Label", "Varietas dari aturan if-else, 12 kelas"],
      ["Jumlah sampel", "Train 720 · validasi 240 · uji 240"],
    ],
    proses: [
      ["Pemrosesan", "Identik dengan E05 (IQR: 61 nilai dibatasi)"],
      ["Model", `${LSTM_ARS}; ${RF_TUNING}`],
      ["Script", "python -m pipelines.klasifikasi_proposal.train_hybrid_proposal"],
    ],
    output: [
      ["Prediksi", "Top-3 varietas untuk 240 lokasi-bulan uji"],
      ["Model", "models/proposal_bundle.joblib (dipakai aplikasi web sebagai Model 1)"],
      ["File", "results/eksperimen/E09_proposal_20_lokasi/"],
    ],
    akurasi: [
      table([
        ["Model", "Accuracy", "Precision (w)", "F1 (w)", "F1 (m)", "Top-3"],
        ["Hybrid LSTM + RF", "0,367 ± 0,033", "0,461", "0,388 ± 0,022", "0,332 ± 0,017", "0,788 ± 0,053"],
        ["RF saja", "0,313 ± 0,016", "0,215", "0,243 ± 0,016", "0,241", "0,676 ± 0,014"],
        ["LSTM saja", "0,345 ± 0,034", "0,204", "0,246 ± 0,028", "0,119", "0,545 ± 0,034"],
      ], [2026, 1400, 1300, 1500, 1300, 1500], { numCols: [1, 2, 3, 4, 5], highlight: [1] }),
    ],
    ringkasan: "Dengan 20 lokasi, Hybrid menjadi yang terbaik di semua metrik termasuk Accuracy, dan F1 (m) naik dari 0,288 ke 0,332. Epoch terbaik LSTM hanya 3–5 karena 20 kecamatan hanya memiliki 2 seri iklim, sehingga LSTM cepat overfit.",
  }),

  // ---------------------------------------------------------------- E10
  ...eksperimen("E10", "Skor kecocokan many-to-many (25 varietas)", "aktif (web, Model 2)", {
    input: [
      ["Data", "Many-to-many dummy: 20 lokasi × 60 bulan tanam (2021–2025) × 25 varietas = 30.000 baris; train 24.000 (2021–2024), uji 6.000 (2025)"],
      ["Fitur", `32 vektor LSTM (encoder E09) + ${TANAH9} + ${VAR8} + musim (sin/cos bulan tanam) = 51 fitur`],
      ["Target", "Skor kecocokan 0–100 (dari kebutuhan air, suhu, pH, drainase, ketahanan hama, ketinggian)"],
    ],
    proses: [
      ["Anti-bocor", "Kolom iklim selama masa tanam dan komponen skor tidak dijadikan fitur"],
      ["Model", `Random Forest regresi; ${RF_TUNING}`],
      ["Pembanding", "RF saja (rata-rata iklim 12 bulan menggantikan vektor LSTM) dan tanpa iklim"],
      ["Evaluasi", "Seluruh uji dan \"normal\" (tanpa 98 lokasi-bulan yang kena hujan ekstrem Nov 2025)"],
    ],
    output: [
      ["Prediksi", "Skor untuk 25 varietas per lokasi-bulan → Top-3"],
      ["Model", "models/rf_varietas_hybrid.joblib (web)"],
      ["File", "results/eksperimen/E10_rf_kecocokan_varietas/"],
    ],
    akurasi: [
      table([
        ["Model", "MAE", "R²", "MAE normal", "R² normal", "Spearman normal", "Top-3 normal", "Komoditas tepat"],
        ["Hybrid", "25,6", "−0,18", "17,5", "−0,16", "0,30", "0,19", "0,94"],
        ["RF saja", "23,3", "0,01", "14,4", "0,24", "0,51", "0,27", "0,93"],
        ["Tanpa iklim", "25,0", "−0,08", "16,9", "−0,04", "0,37", "0,18", "0,94"],
      ], [1526, 900, 900, 1100, 1050, 1250, 1150, 1150], { numCols: [1, 2, 3, 4, 5, 6, 7], highlight: [2], size: 18 }),
      p("Kontribusi fitur Hybrid: varietas 48%, LSTM 38%, musim 9%, tanah 5%."),
    ],
    ringkasan: "Model dilatih menilai ke-25 varietas memakai karakteristik varietas (Tujuan 2 proposal). Komoditas yang dipilih tepat sekitar 93–94%, tetapi peringkat varietas di dalam satu komoditas masih lemah. Random Forest dengan rata-rata iklim biasa mengungguli Hybrid.",
  }),

  // ---------------------------------------------------------------- E11
  ...eksperimen("E11", "LSTM harian (5 dan 1 epoch) sebagai sumber vektor", "ablation", {
    input: [
      ["Data", "Iklim harian 20 lokasi 2020–2025; sekuens 30 hari: train 28.620, validasi 7.320, uji 7.300"],
      ["Data RF", "Sama dengan E10, vektor LSTM diganti versi harian"],
    ],
    proses: [
      ["Pemrosesan harian", "Hujan log1p → outlier IQR → Z-Score; sin/cos hari-dalam-tahun"],
      ["LSTM", `${LSTM_ARS}; meramal cuaca besok; maks 5 epoch dan 1 epoch, dipilih di validasi 2024`],
      ["Agregasi", "Vektor harian dirata-rata per bulan; RF memakai rata-rata bulan sebelum bulan tanam"],
    ],
    output: [["File", "results/eksperimen/E11_lstm_harian_untuk_rf/; data/processed/lstm_harian*/embedding_bulanan.csv"]],
    akurasi: [
      table([
        ["Ukuran", "5 epoch", "1 epoch"],
        ["Epoch terpilih di validasi", "1", "1"],
        ["MSE validasi epoch 1 → 5", "0,439 → 0,496", "0,439"],
        ["MAE ramalan hujan harian (uji)", "6,19 mm (persistence 7,68)", "6,19 mm"],
        ["RF skor: MAE normal", "17,3", "17,3"],
        ["RF skor: Spearman normal", "0,28", "0,28"],
        ["Acuan RF saja: Spearman normal", "0,51", "0,51"],
      ], [4026, 2500, 2500], { numCols: [1, 2] }),
    ],
    ringkasan: "LSTM yang dilatih per hari langsung overfit setelah satu epoch karena data hanya memuat 2 seri iklim. Vektornya tidak memperbaiki Random Forest dan tetap kalah dari rata-rata iklim biasa.",
  }),

  // ---------------------------------------------------------------- E12
  ...eksperimen("E12", "Prediksi hasil panen dan umur panen", "aktif (web)", {
    input: [
      ["Data", "Sama dengan E10 (30.000 baris, 51 fitur)"],
      ["Target", "Hasil panen dummy (t/ha) dan umur panen aktual dummy (hari, thermal time: makin panas makin cepat matang, −11 s/d +20 hari dari deskripsi)"],
    ],
    proses: [
      ["Model", `Random Forest regresi per target; ${RF_TUNING}`],
      ["Pembanding", "Tebakan tanpa model: potensi hasil × rasio rata-rata train; umur menurut deskripsi varietas"],
      ["Script", "python -m pipelines.varietas.train_rf_hasil_umur"],
    ],
    output: [
      ["Model", "models/rf_hasil_hybrid.joblib, models/rf_umur_hybrid.joblib (web)"],
      ["File", "results/eksperimen/E12_prediksi_hasil_umur/"],
    ],
    akurasi: [
      table([
        ["Model", "Hasil MAE (t/ha)", "Hasil R²", "Hasil MAE normal", "Umur MAE (hari)", "Umur R²"],
        ["Tebakan tanpa model", "1,93", "0,25", "1,19", "4,07", "0,85"],
        ["Hybrid, LSTM bulanan", "1,98", "0,20", "1,35", "1,76", "0,98"],
        ["Hybrid, LSTM harian 1 epoch", "1,91", "0,25", "1,31", "1,77", "0,98"],
        ["RF saja", "1,79", "0,33", "1,11", "2,07", "0,97"],
        ["Tanpa iklim", "1,91", "0,27", "1,28", "1,74", "0,98"],
      ], [2626, 1300, 1000, 1500, 1500, 1100], { numCols: [1, 2, 3, 4, 5] }),
      p("Kontribusi fitur Hybrid: hasil panen → varietas 82%, LSTM 13%; umur panen → varietas 90%, LSTM 10%."),
    ],
    ringkasan: "Umur panen tertebak hingga batas noise data dummy (±2 hari), dan bulan tanam saja sudah cukup. Untuk hasil panen, Hybrid belum mengalahkan tebakan sederhana, sedangkan RF saja paling baik.",
  }),

  // ---------------------------------------------------------------- E13
  ...eksperimen("E13", "Hasil panen dengan variabel cuaca biasa", "aktif (web, pembanding)", {
    input: [
      ["Data", "Sama dengan E10–E12, target hasil panen (t/ha)"],
      ["Fitur cuaca", "C1: rata-rata 6 variabel 12 bulan terakhir · C2: C1 + rata-rata 3 bulan terakhir · C3: C2 + normal iklim masa tanam (dari tahun lain)"],
    ],
    proses: [
      ["Model", `Random Forest regresi; vektor LSTM diganti fitur cuaca; ${RF_TUNING}`],
      ["Pemilihan", "Set fitur dipilih dari MAE validasi, bukan dari data uji → C2"],
      ["Script", "python -m pipelines.varietas.train_rf_hasil_cuaca"],
    ],
    output: [
      ["Model", "models/rf_hasil_cuaca.joblib (web, ditampilkan berdampingan dengan Hybrid)"],
      ["File", "results/eksperimen/E13_hasil_panen_cuaca_biasa/"],
    ],
    akurasi: [
      table([
        ["Set fitur", "MAE validasi", "MAE uji", "R² uji", "MAE normal", "R² normal"],
        ["C1 rata-rata 12 bulan", "1,07", "1,79", "0,33", "1,11", "0,72"],
        ["C2 + rata-rata 3 bulan (dipilih)", "0,99", "1,74", "0,39", "1,30", "0,61"],
        ["C3 + normal iklim masa tanam", "1,02", "1,81", "0,35", "1,25", "0,65"],
        ["Acuan: Hybrid 32 dimensi (E12)", "–", "1,98", "0,20", "1,35", "0,58"],
      ], [3026, 1200, 1100, 1100, 1300, 1300], { numCols: [1, 2, 3, 4, 5], highlight: [2] }),
    ],
    ringkasan: "Beberapa angka cuaca yang mudah dibaca menghasilkan prediksi hasil panen lebih baik daripada 32 angka vektor LSTM (MAE 1,74 vs 1,98 t/ha). Tanpa bulan hujan ekstrem, C1 paling baik, sehingga keduanya perlu dilaporkan.",
  }),

  // ---------------------------------------------------------------- E14
  ...eksperimen("E14", "Kenapa Random Forest saja unggul", "aktif", {
    input: [["Data", "Vektor LSTM bulanan (E09) dan harian 1 epoch (E11), rata-rata iklim 12 bulan, iklim selama masa tanam (1.200 lokasi-bulan)"]],
    proses: [
      ["Langkah", "Menebak hujan dan suhu selama masa tanam dari tiap jenis fitur (Ridge dan Random Forest, R² pada 2025); PCA pada vektor"],
      ["Script", "python -m pipelines.analisis.e14_isi_informasi_lstm"],
    ],
    output: [["File", "results/eksperimen/E14_isi_informasi_lstm/hasil_analisis.txt"]],
    akurasi: [
      table([
        ["Fitur (R², Ridge / RF)", "Hujan masa tanam", "Suhu masa tanam"],
        ["Rata-rata iklim 12 bulan (6 fitur)", "−0,08 / −0,37", "0,89 / 0,78"],
        ["Vektor LSTM bulanan (32)", "−0,03 / −0,07", "0,92 / 0,97"],
        ["Vektor LSTM harian 1 epoch (32)", "0,17 / 0,16", "0,97 / 0,94"],
        ["Rata-rata bulan kalender", "0,12", "0,10"],
      ], [4026, 2500, 2500], { numCols: [1, 2] }),
      p("3 komponen PCA menjelaskan 92–95% variasi vektor; hanya 240 pola iklim unik dari 1.200 lokasi-bulan."),
    ],
    ringkasan: "Vektor LSTM tidak kekurangan informasi iklim. Random Forest saja unggul karena (1) pola iklim unik terlalu sedikit sehingga 32 fitur mudah dihafal, (2) hubungan iklim dengan kebutuhan varietas lebih mudah ditangkap dari satu kolom yang jelas, dan (3) data dummy memang dibangun dari rata-rata iklim.",
  }),

  pageBreak(),
  h1("Kesimpulan umum dan langkah berikutnya"),
  bullet(["Pipeline sesuai proposal sudah lengkap, bisa diulang, dan pada klasifikasi ", b("Hybrid mengungguli single model"), " (E05, E09)."], "num"),
  bullet(["Akurasi rendah disebabkan ", b("label dummy"), ", bukan model (E06)."], "num"),
  bullet(["Pada data yang memakai karakteristik varietas, ", b("cuaca biasa mengungguli vektor LSTM"), " karena data iklim terlalu seragam (E10–E14)."], "num"),
  h3("Langkah berikutnya"),
  bullet([b("Label asli per musim tanam "), new TextRun("dari Dinas Pertanian/BPS.")], "num2"),
  bullet([b("Data tanah asli "), new TextRun("(N, P, K, drainase, risiko hama).")], "num2"),
  bullet([b("Data iklim lebih detail dan lebih panjang "), new TextRun("(BMKG, CHIRPS, ERA5-Land; 1981–2025) untuk input 60 bulan sesuai proposal.")], "num2"),
  bullet([b("Eksperimen lanjutan Hybrid: "), new TextRun("RF diberi vektor LSTM + cuaca biasa, PCA 4–8 dimensi, LSTM meramal iklim masa tanam 3–4 bulan.")], "num2"),
  bullet([b("Validasi pakar dan UAT "), new TextRun("(Bab III 3.11 tahap 8).")], "num2"),

  h1("Istilah"),
  table([
    ["Istilah", "Arti"],
    ["Accuracy", "Persentase tebakan pertama yang tepat."],
    ["Precision / Recall", "Precision: dari yang direkomendasikan, berapa yang benar. Recall: dari yang benar, berapa yang ditemukan."],
    ["F1-Score", "Gabungan Precision dan Recall. (w) ditimbang jumlah sampel tiap kelas; (m) semua kelas sama rata."],
    ["Top-3", "Persentase kasus ketika jawaban yang benar ada di 3 rekomendasi teratas."],
    ["MAE", "Rata-rata selisih mutlak prediksi dan nilai sebenarnya (satuan sama dengan target)."],
    ["R²", "Variasi target yang dijelaskan model: 1 sempurna, 0 setara menebak rata-rata, negatif lebih buruk."],
    ["Spearman", "Kesamaan urutan peringkat prediksi dengan urutan sebenarnya (−1 s/d 1)."],
    ["Normal", "Evaluasi tanpa bulan tanam yang kena hujan ekstrem November 2025."],
    ["Seed", "Angka acak awal; model dilatih 5 kali (42–46) dan dilaporkan rata-rata ± std."],
    ["Persistence / klimatologi", "Tebakan \"sama dengan sebelumnya\" / rata-rata tanggal atau bulan yang sama di tahun lain."],
  ], [2526, 6500]),
  gap(),
  p([i("Sumber angka: results/eksperimen/<kode>/ dan docs/03_katalog_eksperimen.md (branch pipeline-proposal). Semua eksperimen aktif dapat diulang dengan python run_all.py.")]),
];

// ------------------------------------------------------------------ dokumen
const heading = (size, before, after, color, level) => ({
  run: { font: FONT, size, bold: true, color },
  paragraph: { spacing: { before, after }, keepNext: true, outlineLevel: level },
});
const doc = new Document({
  creator: "Tim Proyek Skripsi",
  title: "Ringkasan Eksperimen Model Hybrid LSTM-RF",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 40, bold: true, color: "1B2A20" }, paragraph: { spacing: { before: 0, after: 120 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(30, 360, 120, ACCENT, 0) },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(24, 220, 100, "1B2A20", 1) },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(22, 160, 80, ACCENT, 2) },
    ],
  },
  numbering: {
    config: ["bul", "num", "num2"].map(ref => ({
      reference: ref,
      levels: [{
        level: 0, format: ref === "bul" ? LevelFormat.BULLET : LevelFormat.DECIMAL, text: ref === "bul" ? "•" : "%1.",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } },
      }],
    })),
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: "Ringkasan Eksperimen Hybrid LSTM–RF · halaman ", size: 18, color: "5A6B5F" }),
      new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "5A6B5F" }),
    ] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(process.argv[2], buf); console.log("OK", process.argv[2], buf.length); });
