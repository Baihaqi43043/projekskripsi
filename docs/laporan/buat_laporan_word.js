const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, BorderStyle, Footer, PageNumber,
} = require("docx");

const FONT = "Calibri";
const ACCENT = "2E6B46";
const CONTENT_W = 9026; // A4 dengan margin 1 inci

const p = (text, opts = {}) => new Paragraph({
  spacing: { after: 120 }, ...opts,
  children: (Array.isArray(text) ? text : [text]).map(t => typeof t === "string" ? new TextRun(t) : t),
});
const b = t => new TextRun({ text: t, bold: true });
const i = t => new TextRun({ text: t, italics: true });
const h1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const h3 = t => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const bullet = (parts, ref = "bul") => new Paragraph({
  numbering: { reference: ref, level: 0 }, spacing: { after: 60 },
  children: (Array.isArray(parts) ? parts : [parts]).map(t => typeof t === "string" ? new TextRun(t) : t),
});
const kesimpulan = parts => new Paragraph({
  spacing: { before: 120, after: 200 }, indent: { left: 200 },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT, space: 8 } },
  shading: { type: ShadingType.CLEAR, fill: "EEF5F0", color: "auto" },
  children: [b("Kesimpulan: "), ...(Array.isArray(parts) ? parts : [parts]).map(t => typeof t === "string" ? new TextRun(t) : t)],
});

const border = { style: BorderStyle.SINGLE, size: 4, color: "C9D3CB" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(rows, widths, { numCols = [], highlight = [], size = 20 } = {}) {
  const total = widths.reduce((a, c) => a + c, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((row, r) => new TableRow({
      tableHeader: r === 0,
      children: row.map((cell, c) => new TableCell({
        borders,
        width: { size: widths[c], type: WidthType.DXA },
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        shading: r === 0 ? { type: ShadingType.CLEAR, fill: "DCEBE1", color: "auto" }
          : highlight.includes(r) ? { type: ShadingType.CLEAR, fill: "F1F7F3", color: "auto" } : undefined,
        children: [new Paragraph({
          alignment: numCols.includes(c) && r > 0 ? AlignmentType.RIGHT : AlignmentType.LEFT,
          children: [new TextRun({ text: String(cell), bold: r === 0 || highlight.includes(r), size })],
        })],
      })),
    })),
  });
}
const gap = () => new Paragraph({ spacing: { after: 60 }, children: [] });

const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Ringkasan Eksperimen Model Hybrid LSTM–Random Forest")] }),
  p([i("Sistem rekomendasi varietas tanaman pangan · Kabupaten Aceh Utara · 27 November 2026")]),
  p("Dokumen ini merangkum semua percobaan yang sudah dijalankan: apa yang diuji, bagaimana caranya, hasilnya, dan artinya."),
  new Paragraph({
    spacing: { before: 120, after: 240 },
    shading: { type: ShadingType.CLEAR, fill: "FFF4DC", color: "auto" },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: "C9A24A", space: 8 } },
    indent: { left: 200 },
    children: [b("Status: prototipe. "), new TextRun("Label varietas dibuat dengan aturan dummy dan 5 dari 9 fitur tanah (N, P, K, drainase, risiko hama) masih dummy. Angka di dokumen ini menguji apakah pipeline bekerja benar dan belum boleh diklaim sebagai temuan penelitian.")],
  }),

  h1("Lima hal terpenting"),
  bullet(["Hasil lama (akurasi 42,5% dan \"akurasi cuaca 81%\") ", b("terlalu optimis"), " karena cara evaluasinya bocor, sehingga tidak bisa dipakai."], "num"),
  bullet(["Pipeline baru yang mengikuti proposal sudah jalan. ", b("Hybrid lebih baik dari RF saja di 5 dari 5 seed"), " (F1 0,405 vs 0,194), dan Top-3 mencapai 84%."], "num"),
  bullet("Accuracy Hybrid (36,7%) setara dengan LSTM saja (37,0%). Keunggulan Hybrid ada di F1, Precision, dan Top-3, bukan di Accuracy.", "num"),
  bullet(["Akurasi rendah disebabkan oleh ", b("desain label dummy"), " (ditentukan hujan bulan depan), bukan oleh model. Batas atasnya sekitar 40–50%."], "num"),
  bullet(["Dengan 20 lokasi Hybrid tetap terbaik pada klasifikasi (E09). Namun pada data many-to-many yang memakai karakteristik varietas, ", b("Random Forest dengan cuaca biasa mengungguli Hybrid"), " (E10–E13) karena pola iklim unik terlalu sedikit (E14)."], "num"),

  h1("Daftar percobaan"),
  table([
    ["Kode", "Percobaan", "Hasil singkat"],
    ["E01", "Audit hasil lama", "Angka lama terlalu optimis karena evaluasinya bocor"],
    ["E02", "Model cuaca vs tebakan sederhana", "Untuk suhu, BiLSTM setara rata-rata kalender"],
    ["E03", "Reproduksi pipeline lama", "Baseline sama persis; Hybrid berubah antar-komputer"],
    ["E04", "Cek embedding LSTM", "32 angka setara sekitar 4 angka"],
    ["E05", "Pipeline sesuai proposal, 5 lokasi", "Hybrid F1 0,405, Top-3 84%, unggul dari RF di 5/5 seed"],
    ["E06", "Kenapa akurasi sekitar 36%", "Disebabkan label dummy; batas atas 40–50%"],
    ["E07", "Per bulan vs per musim", "Variasi hujan 52% per bulan vs 17% per musim"],
    ["E08", "Fitur data harian", "Tidak meningkatkan Hybrid"],
    ["E09", "Pipeline sesuai proposal, 20 lokasi", "Hybrid terbaik di semua metrik; F1 (m) 0,332"],
    ["E10", "RF skor kecocokan many-to-many", "RF saja > Hybrid; komoditas tepat ±93%"],
    ["E11", "LSTM harian (5 & 1 epoch) untuk RF", "Epoch terbaik 1; tetap kalah RF saja"],
    ["E12", "Prediksi hasil & umur panen", "Umur ±1,7 hari; hasil: Hybrid < tebakan sederhana"],
    ["E13", "Hasil panen dengan cuaca biasa", "Cuaca biasa mengalahkan Hybrid 32 dimensi"],
    ["E14", "Kenapa RF saja unggul", "Bukan kurang informasi; overfit & interaksi"],
  ], [900, 3100, 5026]),

  // E1
  h1("E01 · Audit hasil lama"),
  h3("Tujuan"),
  p("Memeriksa apakah angka di results/experiment_history.csv (sebelum perbaikan) layak dipakai untuk Bab IV."),
  h3("Hasil yang tercatat di log"),
  table([
    ["Model (log lama)", "Accuracy", "F1 (w)", "Top-3"],
    ["Baseline RF, 20 lokasi", "0,258", "0,227", "0,675"],
    ["Hybrid LSTM + RF, 20 lokasi (ditandai \"terbaik\")", "0,425", "0,462", "0,625"],
    ["Hybrid tuned, 20 lokasi", "0,350", "0,406", "0,663"],
    ["BiLSTM cuaca harian (Rahmat)", "\"Akurasi WAPE 81,27%\"", "", ""],
  ], [4826, 1400, 1400, 1400], { numCols: [1, 2, 3] }),
  h3("Masalah yang ditemukan"),
  table([
    ["Masalah", "Akibat"],
    ["Data uji 2025 dipakai untuk memilih model terbaik (BiLSTM)", "Nilai uji terlalu optimis"],
    ["Nilai aktual hujan dipotong sebelum dinilai (60 hari > 37,8 mm, maksimum 212 mm)", "Hari dengan hujan ekstrem hilang dari penilaian"],
    ["Tuning RF memakai pembagian data acak / per lokasi", "Model \"tuned\" justru lebih buruk dari default"],
    ["LSTM dan RF belajar dari label yang sama", "RF terlalu percaya pada fitur LSTM"],
    ["Label varietas = aturan if-else buatan sendiri", "Akurasi hanya mengukur seberapa baik model menebak aturan itu"],
  ], [5026, 4000]),
  kesimpulan("Angka lama tidak bisa dipakai. Semua percobaan berikutnya memakai pembagian data yang bersih."),

  // E2
  h1("E02 · Model cuaca harian vs tebakan sederhana"),
  h3("Tujuan"),
  p("Mengecek apakah BiLSTM cuaca harian lebih baik dari tebakan tanpa model. Data: Januari–Maret 2025, 5 lokasi, nilai aktual asli (tidak dipotong)."),
  h3("Cara"),
  p([b("Persistence"), new TextRun(" = \"besok sama dengan hari ini\". "), b("Klimatologi"), new TextRun(" = rata-rata tanggal yang sama pada 2020–2024. Angka di tabel adalah MAE (rata-rata kesalahan, makin kecil makin baik).")]),
  table([
    ["Variabel", "BiLSTM", "Persistence", "Klimatologi", "Pemenang"],
    ["Curah hujan (mm)", "7,08", "8,58", "7,95", "BiLSTM, tapi R² −0,04 (tidak bisa menebak)"],
    ["Suhu rata-rata (°C)", "0,353", "0,364", "0,354", "Seri"],
    ["Suhu maksimal (°C)", "0,664", "0,875", "0,657", "Klimatologi"],
    ["Suhu minimum (°C)", "0,413", "0,469", "0,538", "BiLSTM"],
    ["Kelembaban (%)", "2,16", "2,51", "2,89", "BiLSTM"],
    ["Radiasi matahari", "2,97", "3,61", "3,20", "BiLSTM"],
  ], [2026, 1100, 1300, 1300, 3300], { numCols: [1, 2, 3] }),
  gap(),
  h3("\"Akurasi WAPE\" jika hanya menebak rata-rata data train"),
  table([
    ["Variabel", "Akurasi WAPE"],
    ["Suhu rata-rata", "94,8%"], ["Suhu maksimal", "95,8%"], ["Kelembaban", "95,2%"], ["Curah hujan", "−9,3%"],
  ], [5026, 4000], { numCols: [1] }),
  h3("Tebak \"hujan / tidak hujan\""),
  table([
    ["Cara", "Akurasi"],
    ["BiLSTM (Januari–Maret 2025)", "72,7%"],
    ["Selalu tebak \"hujan\"", "72,7%"],
    ["Besok = hari ini", "82,9%"],
  ], [5026, 4000], { numCols: [1], highlight: [3] }),
  kesimpulan("Angka \"81%\" menyesatkan karena suhu Aceh hampir selalu sekitar 27 °C, sehingga tebakan konstan pun mendapat 95%. Untuk suhu, BiLSTM tidak lebih baik dari rata-rata kalender. Curah hujan harian praktis tidak bisa ditebak."),

  // E3
  h1("E03 · Menjalankan ulang pipeline lama"),
  h3("Tujuan"),
  p("Mengecek apakah angka di log bisa didapat ulang dengan kode dan data yang sama."),
  table([
    ["Percobaan", "Accuracy", "F1 (w)", "Top-3", "Catatan"],
    ["Baseline RF, 20 lokasi", "0,258", "0,227", "0,675", "Sama persis dengan log"],
    ["Hybrid, 20 lokasi", "0,367", "0,374", "0,583", "Log: 0,425 (beda versi torch/hardware)"],
    ["Baseline RF, 5 lokasi (11 kelas)", "0,183", "0,163", "0,633", ""],
    ["Hybrid, 5 lokasi", "0,317", "0,352", "0,900", ""],
    ["Hybrid tuned, 5 lokasi", "0,300", "0,334", "0,850", "Tuning lebih buruk dari default"],
  ], [2826, 1000, 1000, 1000, 3200], { numCols: [1, 2, 3] }),
  kesimpulan("Hasil di satu laptop konsisten antar-run, tetapi berubah antar-komputer (0,367 vs 0,425). Satu kali run tidak cukup, sehingga percobaan berikutnya memakai 5 seed dan melaporkan rata-rata ± simpangan baku."),

  // E4
  h1("E04 · Apakah 32 fitur LSTM berisi 32 informasi?"),
  h3("Cara"),
  p("Analisis PCA pada 32 angka keluaran LSTM, lalu membandingkan porsi feature importance dengan porsi jumlah fitur."),
  table([
    ["Ukuran", "Nilai"],
    ["Dimensi yang selalu bernilai 0 (mati)", "3 dari 32"],
    ["Variasi yang dijelaskan 2 komponen PCA", "90,7%"],
    ["Variasi yang dijelaskan 4 komponen PCA", "99,7%"],
    ["Porsi jumlah fitur LSTM (32 dari 41)", "78%"],
    ["Porsi feature importance LSTM", "81,7%"],
  ], [6026, 3000], { numCols: [1], highlight: [3] }),
  kesimpulan("Isi 32 angka itu setara sekitar 4 angka. LSTM tampak dominan terutama karena jumlah kolomnya banyak. Uji PCA 8/16/32 layak masuk Bab IV untuk menjustifikasi angka 32 di proposal."),

  // E5
  h1("E05 · Pipeline yang mengikuti proposal (dipakai untuk Bab IV)"),
  h3("Cara"),
  p("Script train_hybrid_proposal.py. Data iklim bulanan, outlier IQR, Z-Score, input 12 bulan, 5 lokasi. Data dibagi menurut waktu: train 2021–2023 (180 sampel), validasi 2024 (60 sampel), uji 2025 (60 sampel). LSTM 128 → 64 → 32 dilatih meramal iklim bulan depan tanpa melihat label. RF di-tuning dengan validasi per tahun. Diulang dengan 5 seed."),
  h2("E05.1 Pengaturan percobaan"),
  table([
    ["Komponen", "Pengaturan"],
    ["Data", "Iklim bulanan 6 variabel (NASA POWER) + 9 fitur tanah/lokasi; 5 lokasi (L001–L005); 11 kelas varietas"],
    ["Satu sampel", "1 lokasi × 1 bulan target; input LSTM = iklim 12 bulan sebelum bulan target"],
    ["Pembagian data", "Train: bulan target 2021–2023 (180 sampel) · Validasi: 2024 (60) · Uji: 2025 (60)"],
    ["Preprocessing", "Interpolasi linier; outlier IQR (Q1 − 1,5·IQR s/d Q3 + 1,5·IQR, 14 nilai dibatasi); Z-Score; sin/cos bulan. Statistik dari data train saja"],
    ["LSTM", "LSTM 128 unit → LSTM 64 unit → Dense 32 (ReLU); dropout 0,2; Adam lr 0,001; batch 16; maks 300 epoch, early stopping patience 20 pada loss validasi"],
    ["Tugas LSTM (Hybrid)", "Meramal 6 variabel iklim bulan target (MSE), tanpa label varietas; vektor 32 angka diambil sebagai fitur temporal"],
    ["LSTM saja", "Arsitektur sama, tetapi langsung mengklasifikasi varietas (softmax)"],
    ["Random Forest", "RandomizedSearchCV 20 kombinasi, skor F1 (w); validasi per tahun (2023, 2024); refit di train + validasi"],
    ["Ruang tuning RF", "n_estimators 100/200/300/500 · max_depth None/10/15/20/30 · min_samples_split 2/5/10 · min_samples_leaf 1/2/4 · max_features sqrt/log2/0,5 · class_weight None/balanced/balanced_subsample"],
    ["Pengulangan", "5 seed (42–46); dilaporkan rata-rata ± simpangan baku"],
  ], [2326, 6700]),

  h2("E05.2 Penjelasan tiap metrik"),
  table([
    ["Metrik", "Arti", "Cara membaca"],
    ["Accuracy", "Persentase sampel uji yang varietas tebakan pertamanya tepat.", "0,367 = 22 dari 60 sampel benar."],
    ["Precision", "Dari semua rekomendasi untuk suatu varietas, berapa persen yang benar. Rumus TP / (TP + FP).", "Tinggi = jarang salah merekomendasikan."],
    ["Recall", "Dari semua kasus yang memang varietas itu, berapa persen yang berhasil ditemukan. Rumus TP / (TP + FN).", "Tinggi = jarang melewatkan varietas yang benar."],
    ["F1-Score", "Rata-rata harmonik Precision dan Recall: 2·P·R / (P + R).", "Satu angka yang menyeimbangkan keduanya."],
    ["(w) weighted", "Rata-rata antar-kelas yang ditimbang menurut jumlah sampel tiap kelas.", "Kelas yang sering muncul berpengaruh lebih besar."],
    ["(m) macro", "Rata-rata antar-kelas biasa; semua kelas berbobot sama.", "Menunjukkan kinerja di kelas yang jarang muncul."],
    ["Top-3", "Persentase sampel ketika varietas yang benar ada di 3 rekomendasi teratas.", "Paling sesuai dengan keluaran sistem (Top-K, Bag. 3.10)."],
    ["Top-5", "Sama seperti Top-3, untuk 5 rekomendasi teratas.", "Lihat catatan: kurang informatif di data ini."],
    ["± std", "Simpangan baku dari 5 seed.", "Makin kecil = hasil makin stabil."],
  ], [1626, 4200, 3200], { size: 19 }),
  p([b("Catatan 1. "), new TextRun("Accuracy selalu sama dengan Recall (w). Secara matematis, recall yang ditimbang menurut jumlah sampel tiap kelas sama dengan proporsi tebakan benar.")]),
  p([b("Catatan 2. "), new TextRun("Top-5 Random Forest = 1,000 di semua seed tidak berarti RF sangat baik. Dengan 5 lokasi, tiap lokasi hanya punya sedikit kemungkinan label di data uji, sehingga 5 tebakan hampir pasti mencakup label yang benar. Gunakan Top-3 sebagai metrik rekomendasi utama.")]),
  p([b("Catatan 3. "), new TextRun("Precision (w) RF saja rendah (0,156) karena RF hanya menebak beberapa kelas yang sama berulang-ulang, sehingga banyak kelas mendapat precision 0.")]),

  h2("E05.3 Ringkasan semua metrik (data uji 2025, rata-rata ± std, 5 seed)"),
  table([
    ["Metrik", "Hybrid LSTM + RF", "Random Forest saja", "LSTM saja", "Terbaik"],
    ["Accuracy", "0,367 ± 0,039", "0,290 ± 0,022", "0,370 ± 0,062", "LSTM ≈ Hybrid"],
    ["Precision (w)", "0,476 ± 0,051", "0,156 ± 0,006", "0,283 ± 0,067", "Hybrid"],
    ["Recall (w)", "0,367 ± 0,039", "0,290 ± 0,022", "0,370 ± 0,062", "LSTM ≈ Hybrid"],
    ["F1-Score (w)", "0,405 ± 0,043", "0,194 ± 0,009", "0,308 ± 0,063", "Hybrid"],
    ["Precision (m)", "0,324 ± 0,040", "0,201 ± 0,004", "0,192 ± 0,060", "Hybrid"],
    ["Recall (m)", "0,273 ± 0,027", "0,382 ± 0,028", "0,250 ± 0,047", "RF saja"],
    ["F1-Score (m)", "0,288 ± 0,033", "0,247 ± 0,009", "0,206 ± 0,053", "Hybrid"],
    ["Top-3", "0,843 ± 0,019", "0,713 ± 0,025", "0,670 ± 0,036", "Hybrid"],
    ["Top-5", "0,997 ± 0,007", "1,000 ± 0,000", "0,813 ± 0,018", "(lihat Catatan 2)"],
  ], [1726, 1900, 1900, 1900, 1600], { numCols: [1, 2, 3] }),
  p("Hybrid terbaik di 6 dari 9 metrik (Precision w/m, F1 w/m, Top-3). LSTM saja setara di Accuracy dan Recall (w). RF saja unggul di Recall (m) karena class_weight membuatnya lebih sering menebak kelas yang jarang."),

  h2("E05.4 Hasil per seed"),
  p([i("Singkatan: Acc = Accuracy, P = Precision, R = Recall, (w) = weighted, (m) = macro.")]),
  h3("Hybrid LSTM + RF"),
  table([
    ["Seed", "Acc", "P (w)", "R (w)", "F1 (w)", "P (m)", "R (m)", "F1 (m)", "Top-3", "Top-5"],
    ["42", "0,333", "0,441", "0,333", "0,369", "0,294", "0,248", "0,259", "0,850", "1,000"],
    ["43", "0,367", "0,494", "0,367", "0,418", "0,372", "0,292", "0,322", "0,833", "1,000"],
    ["44", "0,433", "0,555", "0,433", "0,475", "0,362", "0,308", "0,324", "0,850", "1,000"],
    ["45", "0,350", "0,429", "0,350", "0,377", "0,286", "0,248", "0,258", "0,867", "0,983"],
    ["46", "0,350", "0,458", "0,350", "0,386", "0,305", "0,270", "0,276", "0,817", "1,000"],
    ["Rata-rata", "0,367", "0,476", "0,367", "0,405", "0,324", "0,273", "0,288", "0,843", "0,997"],
  ], [1026, 889, 889, 889, 889, 889, 889, 889, 889, 889], { numCols: [1, 2, 3, 4, 5, 6, 7, 8, 9], highlight: [6], size: 18 }),
  h3("Random Forest saja (statis + rata-rata iklim 12 bulan)"),
  table([
    ["Seed", "Acc", "P (w)", "R (w)", "F1 (w)", "P (m)", "R (m)", "F1 (m)", "Top-3", "Top-5"],
    ["42", "0,300", "0,159", "0,300", "0,200", "0,202", "0,389", "0,249", "0,683", "1,000"],
    ["43", "0,267", "0,152", "0,267", "0,186", "0,200", "0,356", "0,241", "0,717", "1,000"],
    ["44", "0,317", "0,162", "0,317", "0,204", "0,207", "0,422", "0,259", "0,750", "1,000"],
    ["45", "0,267", "0,149", "0,267", "0,183", "0,195", "0,356", "0,237", "0,717", "1,000"],
    ["46", "0,300", "0,159", "0,300", "0,199", "0,202", "0,389", "0,250", "0,700", "1,000"],
    ["Rata-rata", "0,290", "0,156", "0,290", "0,194", "0,201", "0,382", "0,247", "0,713", "1,000"],
  ], [1026, 889, 889, 889, 889, 889, 889, 889, 889, 889], { numCols: [1, 2, 3, 4, 5, 6, 7, 8, 9], highlight: [6], size: 18 }),
  h3("LSTM saja (sekuens iklim → varietas)"),
  table([
    ["Seed", "Acc", "P (w)", "R (w)", "F1 (w)", "P (m)", "R (m)", "F1 (m)", "Top-3", "Top-5"],
    ["42", "0,467", "0,340", "0,467", "0,386", "0,245", "0,315", "0,271", "0,617", "0,817"],
    ["43", "0,383", "0,366", "0,383", "0,366", "0,264", "0,259", "0,254", "0,667", "0,817"],
    ["44", "0,317", "0,252", "0,317", "0,265", "0,154", "0,185", "0,158", "0,717", "0,783"],
    ["45", "0,367", "0,203", "0,367", "0,253", "0,126", "0,259", "0,163", "0,683", "0,833"],
    ["46", "0,317", "0,257", "0,317", "0,267", "0,170", "0,232", "0,181", "0,667", "0,817"],
    ["Rata-rata", "0,370", "0,283", "0,370", "0,308", "0,192", "0,250", "0,206", "0,670", "0,813"],
  ], [1026, 889, 889, 889, 889, 889, 889, 889, 889, 889], { numCols: [1, 2, 3, 4, 5, 6, 7, 8, 9], highlight: [6], size: 18 }),
  p("Hybrid lebih tinggi dari RF saja di 5 dari 5 seed dan lebih tinggi dari LSTM saja di 4 dari 5 seed (F1 w). LSTM saja paling tidak stabil (std Accuracy 0,062)."),

  h2("E05.5 Kinerja per varietas (Hybrid, seed 42)"),
  table([
    ["Varietas", "Jumlah di uji", "Precision", "Recall", "F1 Hybrid", "F1 RF saja", "F1 LSTM saja"],
    ["Padi Inpari 32", "6", "0,714", "0,833", "0,769", "0,667", "0,167"],
    ["Padi Inpari 33", "3", "0,500", "0,667", "0,571", "0,462", "0,000"],
    ["Jagung BISI 18", "12", "0,625", "0,417", "0,500", "0,000", "0,750"],
    ["Padi Ciherang", "12", "0,556", "0,417", "0,476", "0,000", "0,500"],
    ["Kedelai Anjasmoro", "9", "0,500", "0,222", "0,308", "0,182", "0,353"],
    ["Jagung NK 212", "6", "0,333", "0,167", "0,222", "0,000", "0,667"],
    ["Jagung Lamuru", "3", "0,000", "0,000", "0,000", "0,308", "0,000"],
    ["Padi IR64", "6", "0,000", "0,000", "0,000", "0,471", "0,000"],
    ["Padi Mekongga", "3", "0,000", "0,000", "0,000", "0,400", "0,000"],
    ["Jagung Bima 20 URI", "0", "–", "–", "–", "–", "–"],
    ["Padi Inpari 42", "0", "–", "–", "–", "–", "–"],
  ], [2226, 1100, 1100, 1000, 1200, 1200, 1200], { numCols: [1, 2, 3, 4, 5, 6], size: 19 }),
  p("Hybrid paling baik untuk Padi Inpari 32 dan Inpari 33. Hybrid gagal total pada Jagung Lamuru, Padi IR64, dan Padi Mekongga, yang justru ditebak cukup baik oleh RF saja. Dua varietas tidak muncul di data uji 2025 sehingga tidak bisa dinilai."),

  h2("E05.6 Pelatihan LSTM dan tuning RF per seed"),
  table([
    ["Seed", "Epoch terbaik", "MSE validasi", "MAE uji LSTM", "MAE persistence", "MAE klimatologi", "Importance LSTM"],
    ["42", "14", "0,385", "0,451", "0,595", "0,812", "73,5%"],
    ["43", "23", "0,430", "0,448", "0,595", "0,812", "75,0%"],
    ["44", "24", "0,435", "0,429", "0,595", "0,812", "76,2%"],
    ["45", "14", "0,427", "0,435", "0,595", "0,812", "77,6%"],
    ["46", "28", "0,403", "0,434", "0,595", "0,812", "73,6%"],
  ], [926, 1250, 1250, 1350, 1450, 1450, 1350], { numCols: [1, 2, 3, 4, 5, 6], size: 19 }),
  gap(),
  table([
    ["Seed", "Model", "n_estimators", "max_depth", "min_split", "min_leaf", "max_features", "class_weight"],
    ["42", "Hybrid", "200", "10", "5", "4", "sqrt", "balanced"],
    ["43", "Hybrid", "200", "10", "10", "2", "log2", "balanced_subsample"],
    ["44", "Hybrid", "200", "15", "2", "4", "sqrt", "balanced_subsample"],
    ["45", "Hybrid", "100", "10", "10", "2", "sqrt", "balanced_subsample"],
    ["46", "Hybrid", "500", "None", "2", "4", "sqrt", "balanced"],
    ["42", "RF saja", "500", "None", "2", "4", "sqrt", "balanced"],
    ["43", "RF saja", "200", "10", "10", "2", "log2", "balanced_subsample"],
    ["44", "RF saja", "100", "30", "5", "4", "sqrt", "balanced"],
    ["45", "RF saja", "100", "10", "10", "2", "sqrt", "balanced_subsample"],
    ["46", "RF saja", "100", "30", "10", "2", "log2", "balanced"],
  ], [726, 1000, 1200, 1050, 1000, 950, 1200, 1900], { numCols: [2, 3, 4, 5], size: 18 }),
  p("Semua hasil tuning memakai class_weight seimbang (balanced atau balanced_subsample), sesuai data yang kelasnya tidak seimbang. Hyperparameter terbaik berbeda tiap seed, tanda bahwa data train terlalu sedikit untuk memilih satu konfigurasi yang jelas unggul."),
  h2("E05.7 Kontribusi fitur (feature importance, rata-rata 5 seed)"),
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
  ], [6026, 3000], { numCols: [1], highlight: [1] }),
  h2("E05.8 Kualitas ramalan iklim dari LSTM (MAE pada data uji, skala Z)"),
  table([
    ["Cara meramal", "MAE"],
    ["LSTM", "0,44"],
    ["Persistence (sama dengan bulan lalu)", "0,60"],
    ["Klimatologi (rata-rata bulan yang sama)", "0,81"],
  ], [6026, 3000], { numCols: [1], highlight: [1] }),
  kesimpulan("Pipeline bekerja sesuai rancangan. LSTM benar-benar belajar pola iklim karena mengalahkan kedua tebakan sederhana. Hybrid unggul di F1, Precision, dan Top-3, tetapi Accuracy-nya setara dengan LSTM saja. Laporan lengkap: results/runs/20261124_205335_Hybrid_LSTM_RF_Proposal/laporan_hasil_bab4.md."),

  // E6
  h1("E06 · Kenapa akurasi hanya sekitar 36%?"),
  h3("Cara"),
  p("Membandingkan beberapa cara menebak label pada data uji 2025 (5 lokasi × 12 bulan)."),
  table([
    ["Cara menebak", "Akurasi"],
    ["Aturan label diterapkan ke iklim aktual bulan itu", "100%"],
    ["Aturan label diterapkan ke iklim bulan sebelumnya", "43%"],
    ["Model Hybrid, Top-1", "33%"],
    ["Aturan label diterapkan ke ramalan LSTM", "25%"],
    ["Model Hybrid, hanya komoditas (Padi/Jagung/Kedelai)", "53%"],
    ["Model Hybrid, Top-3", "85%"],
  ], [6526, 2500], { numCols: [1], highlight: [6] }),
  p("Label dummy ditentukan oleh curah hujan bulan target dengan batas 85 / 140 / 210 mm, sedangkan model hanya melihat data sampai bulan sebelumnya. Kesalahan ramalan hujan bulanan sekitar 73 mm (median), lebih lebar dari jarak antar-batas kelas (55–70 mm)."),
  kesimpulan("Akurasi rendah berasal dari desain label dummy, bukan dari model, dengan batas atas sekitar 40–50%. Tebakan \"sama dengan bulan lalu\" (43%) mengalahkan Top-1 Hybrid (33%), sehingga perlu dicantumkan sebagai pembanding naif di Bab IV. Akurasi di atas 90% baru realistis dengan label asli per musim tanam."),

  // E7
  h1("E07 · Rekomendasi per bulan vs per musim tanam"),
  h3("Cara"),
  p("Mengukur seberapa besar curah hujan berubah dari tahun ke tahun, per bulan dibanding per musim. MT I = Oktober–Maret, MT II = April–September."),
  table([
    ["Ukuran", "Nilai"],
    ["Variasi antar-tahun hujan per bulan", "52%"],
    ["Variasi antar-tahun hujan per musim", "17%"],
    ["Lhoksukon: hujan bulan Januari saja (2020–2025)", "46–210 mm"],
    ["Lhoksukon: total hujan MT I (2020–2025)", "996–1.433 mm"],
  ], [6526, 2500], { numCols: [1], highlight: [2] }),
  kesimpulan("Per musim jauh lebih stabil, sesuai cara petani menanam (padi 107–125 hari = satu musim), dan cocok dengan format data Dinas (per MT). Konsekuensinya, sampel turun menjadi 55 untuk 5 lokasi atau sekitar 220 untuk 20 lokasi, sehingga pendekatan ini perlu 20 lokasi dan label asli."),

  // E8
  h1("E08 · Menambah fitur dari data harian (cara B)"),
  h3("Cara"),
  p("Pipeline terpisah train_hybrid_harian.py. Semua sama dengan E5, ditambah 5 fitur per bulan dari data harian: jumlah hari hujan, hari hujan lebat (≥ 50 mm), hujan harian tertinggi, deret hari kering terpanjang, dan jumlah hari panas (≥ 30,3 °C)."),
  table([
    ["Model", "Metrik", "Tanpa (E5)", "Dengan fitur harian", "Naik di"],
    ["Hybrid", "Accuracy", "0,367", "0,357", "2/5 seed"],
    ["Hybrid", "F1 (w)", "0,405", "0,376", "1/5 seed"],
    ["Hybrid", "F1 (macro)", "0,288", "0,264", "2/5 seed"],
    ["Hybrid", "Top-3", "0,843", "0,863", "3/5 seed"],
    ["LSTM saja", "Accuracy", "0,370", "0,387", "3/5 seed"],
    ["LSTM saja", "Top-3", "0,670", "0,710", "4/5 seed"],
    ["Ramalan iklim LSTM", "MAE (kecil = baik)", "0,439", "0,476", ""],
  ], [2026, 2000, 1500, 2000, 1500], { numCols: [2, 3, 4] }),
  kesimpulan("Tidak ada peningkatan yang konsisten. Hybrid sedikit turun dan ramalan iklim LSTM memburuk karena 180 sampel train terlalu sedikit untuk input yang lebih banyak. Informasi yang dibutuhkan label (hujan bulan depan) juga tidak ada di data harian bulan sebelumnya. Hasil ini layak dilaporkan sebagai ablation dan diuji ulang setelah ada label asli per musim."),

  // E09
  h1("E09 · Pipeline yang mengikuti proposal, 20 lokasi"),
  h3("Cara"),
  p("Sama dengan E05, tetapi memakai ke-20 kecamatan: train 720 sampel, validasi 240, uji 240, 12 kelas. Model ini dipakai aplikasi web sebagai Model 1."),
  table([
    ["Model", "Accuracy", "Precision (w)", "F1 (w)", "F1 (macro)", "Top-3"],
    ["Hybrid LSTM + RF", "0,367 ± 0,033", "0,461", "0,388 ± 0,022", "0,332 ± 0,017", "0,788 ± 0,053"],
    ["Random Forest saja", "0,313 ± 0,016", "0,215", "0,243 ± 0,016", "0,241", "0,676 ± 0,014"],
    ["LSTM saja", "0,345 ± 0,034", "0,204", "0,246 ± 0,028", "0,119", "0,545 ± 0,034"],
  ], [2026, 1400, 1300, 1500, 1300, 1500], { numCols: [1, 2, 3, 4, 5], highlight: [1] }),
  kesimpulan("Hybrid terbaik di semua metrik. Epoch terbaik LSTM hanya 3–5 (5 lokasi: 14–28) karena 20 kecamatan hanya punya 2 seri iklim unik di NASA POWER, sehingga LSTM cepat overfit."),

  // E10
  h1("E10 · Skor kecocokan many-to-many (25 varietas)"),
  h3("Cara"),
  p("Data dummy baru tanpa if-else: 20 lokasi × 60 bulan tanam × 25 varietas (30.000 baris). Skor kecocokan 0–100 dihitung dari karakteristik varietas (kebutuhan air, suhu, pH, drainase, ketahanan hama, ketinggian) dibandingkan kondisi lahan dan iklim selama masa tanam. Random Forest regresi memprediksi skor; tiga konfigurasi fitur dibandingkan. Uji 2025; \"normal\" = tanpa bulan tanam yang terkena hujan ekstrem November 2025 (782–801 mm)."),
  table([
    ["Model", "MAE normal", "R² normal", "Spearman normal", "Top-3 memuat terbaik", "Komoditas tepat"],
    ["Hybrid (LSTM + tanah + varietas)", "17,5", "−0,16", "0,30", "0,20", "0,94"],
    ["RF saja (rata-rata iklim)", "14,4", "0,24", "0,51", "0,27", "0,93"],
    ["Tanpa iklim (tanah + varietas)", "16,9", "−0,04", "0,37", "0,18", "0,94"],
  ], [2826, 1200, 1100, 1300, 1400, 1200], { numCols: [1, 2, 3, 4, 5], highlight: [2] }),
  kesimpulan("Random Forest dengan rata-rata iklim biasa mengungguli Hybrid. Pemilihan komoditas sudah tepat sekitar 93–94%, tetapi peringkat varietas di dalam satu komoditas masih lemah."),

  // E11
  h1("E11 · LSTM harian (5 dan 1 epoch) sebagai sumber vektor"),
  p("LSTM dilatih per hari (jendela 30 hari, meramal cuaca besok), lalu vektor 32 harian dirata-rata per bulan. Dengan maksimum 5 epoch, epoch terbaik di validasi adalah epoch 1 (loss validasi naik sejak epoch 2). Hasil RF skor kecocokan: MAE normal 17,3, R² normal −0,09, Spearman normal 0,28, setara LSTM bulanan dan tetap kalah dari RF saja (0,51)."),
  kesimpulan("Melatih LSTM per hari tidak memperbaiki Hybrid. Data hanya memuat 2 seri iklim unik sehingga model overfit setelah satu epoch."),

  // E12
  h1("E12 · Prediksi hasil panen dan umur panen"),
  p("Target tambahan pada data many-to-many: hasil panen (t/ha) dan umur panen aktual (hari, dibuat dengan konsep thermal time: makin panas, makin cepat matang). Pembanding tanpa model: potensi hasil × rasio rata-rata, dan umur menurut deskripsi varietas."),
  table([
    ["Model", "Hasil MAE (t/ha)", "Hasil R²", "Umur MAE (hari)", "Umur R²"],
    ["Tebakan tanpa model", "1,93", "0,25", "4,07", "0,85"],
    ["Hybrid, LSTM bulanan", "1,98", "0,20", "1,76", "0,98"],
    ["Hybrid, LSTM harian 1 epoch", "1,91", "0,25", "1,77", "0,98"],
    ["RF saja (rata-rata iklim)", "1,79", "0,33", "2,07", "0,97"],
    ["Tanpa iklim", "1,91", "0,27", "1,74", "0,98"],
  ], [3026, 1500, 1200, 1600, 1700], { numCols: [1, 2, 3, 4] }),
  kesimpulan("Umur panen tertebak hingga batas noise data dummy (±2 hari) dan bulan tanam saja sudah cukup. Untuk hasil panen, Hybrid belum mengalahkan tebakan sederhana, sedangkan RF saja paling baik."),

  // E13
  h1("E13 · Hasil panen dengan variabel cuaca biasa"),
  p("Vektor 32 angka LSTM diganti fitur cuaca yang mudah dibaca. C1: rata-rata 6 variabel 12 bulan terakhir. C2: C1 + rata-rata 3 bulan terakhir. C3: C2 + normal iklim masa tanam (dari tahun lain). Set dipilih dari MAE validasi, bukan dari data uji."),
  table([
    ["Set fitur", "MAE validasi", "MAE uji", "R² uji", "MAE normal", "R² normal"],
    ["C1 rata-rata 12 bulan", "1,07", "1,79", "0,33", "1,11", "0,72"],
    ["C2 + rata-rata 3 bulan (dipilih)", "0,99", "1,74", "0,39", "1,30", "0,61"],
    ["C3 + normal iklim masa tanam", "1,02", "1,81", "0,35", "1,25", "0,65"],
    ["Acuan: Hybrid 32 dimensi", "–", "1,98", "0,20", "1,35", "0,58"],
  ], [3026, 1200, 1100, 1100, 1300, 1300], { numCols: [1, 2, 3, 4, 5], highlight: [2] }),
  kesimpulan("Beberapa angka cuaca yang jelas mengalahkan 32 angka abstrak. Aplikasi web menampilkan prediksi hasil Hybrid 32 dimensi dan cuaca biasa berdampingan agar bisa dibandingkan."),

  // E14
  h1("E14 · Kenapa Random Forest saja unggul"),
  p("Diuji seberapa baik tiap jenis fitur menebak iklim selama masa tanam (R² pada 2025)."),
  table([
    ["Fitur", "Hujan masa tanam", "Suhu masa tanam"],
    ["Rata-rata iklim 12 bulan (6 fitur)", "−0,08", "0,89"],
    ["Vektor LSTM bulanan (32 fitur)", "−0,03", "0,92–0,97"],
    ["Vektor LSTM harian 1 epoch (32 fitur)", "0,17", "0,94–0,97"],
    ["Rata-rata bulan kalender", "0,12", "0,10"],
  ], [4026, 2500, 2500], { numCols: [1, 2] }),
  kesimpulan("Vektor LSTM tidak kekurangan informasi. Random Forest saja unggul karena (1) hanya ada 240 pola iklim unik sehingga 32 fitur mudah dihafal, (2) hubungan iklim dengan kebutuhan varietas lebih mudah ditangkap dari satu kolom yang jelas, dan (3) data dummy memang dibangun dari rata-rata iklim."),

  h1("Langkah berikutnya"),
  bullet([b("Label asli per musim tanam. "), new TextRun("Varietas dominan atau hasil tertinggi per kecamatan per MT dari Dinas Pertanian/BPS. Ini satu-satunya jalan menuju akurasi tinggi yang sah.")], "num2"),
  bullet([b("Data tanah asli. "), new TextRun("N, P, K, drainase, dan risiko hama dari uji lab atau Dinas. Nitrogen bisa langsung diambil dari SoilGrids.")], "num2"),
  bullet([b("Data iklim lebih detail dan lebih panjang "), new TextRun("(BMKG, CHIRPS, ERA5-Land; 1981–2025) agar 20 kecamatan benar-benar berbeda dan input 60 bulan sesuai proposal bisa diterapkan.")], "num2"),
  bullet([b("Eksperimen lanjutan Hybrid: "), new TextRun("beri Random Forest vektor LSTM sekaligus cuaca biasa, mampatkan vektor ke 4–8 dimensi (PCA), dan latih LSTM meramal iklim masa tanam 3–4 bulan.")], "num2"),
  bullet([b("Lengkapi Bab IV: "), new TextRun("tambahkan pembanding \"sama dengan bulan lalu\" dan tulis penyesuaian dari proposal (docs/02_metodologi.md).")], "num2"),

  h1("Istilah"),
  table([
    ["Istilah", "Arti"],
    ["Accuracy", "Persentase tebakan varietas yang tepat."],
    ["Precision / Recall", "Precision: dari yang direkomendasikan, berapa yang benar. Recall: dari yang benar, berapa yang berhasil ditemukan."],
    ["F1-Score", "Gabungan Precision dan Recall. \"(w)\" ditimbang menurut jumlah sampel tiap kelas; \"macro\" menimbang semua kelas sama rata."],
    ["Top-3", "Persentase kasus ketika varietas yang benar ada di 3 rekomendasi teratas."],
    ["Seed", "Angka acak awal. Model dilatih 5 kali dengan seed berbeda supaya hasilnya tidak bergantung pada keberuntungan."],
    ["Baseline / single model", "Model pembanding yang memakai satu metode saja (RF saja atau LSTM saja)."],
    ["Data validasi", "Data 2024 untuk memilih model terbaik. Data uji 2025 hanya dipakai sekali di akhir."],
    ["MAE", "Rata-rata selisih antara ramalan dan nilai sebenarnya."],
  ], [2526, 6500]),
  gap(),
  p([i("Sumber angka: results/eksperimen/<kode>/ dan docs/03_katalog_eksperimen.md di branch pipeline-proposal. Script pembuat setiap angka tercantum di katalog; semua eksperimen aktif bisa diulang dengan python run_all.py.")]),
];

const heading = (size, before, after, color) => ({
  run: { font: FONT, size, bold: true, color }, paragraph: { spacing: { before, after }, keepNext: true },
});
const doc = new Document({
  creator: "Tim Proyek Skripsi",
  title: "Ringkasan Eksperimen Model Hybrid LSTM-RF",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 40, bold: true, color: "1B2A20" }, paragraph: { spacing: { before: 0, after: 120 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(30, 360, 120, ACCENT), paragraph: { spacing: { before: 360, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(26, 240, 100, "1B2A20"), paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(22, 180, 80, ACCENT), paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: ["bul", "num", "num2"].map(ref => ({
      reference: ref,
      levels: [{
        level: 0,
        format: ref === "bul" ? LevelFormat.BULLET : LevelFormat.DECIMAL,
        text: ref === "bul" ? "•" : "%1.",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 360 } } },
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
