"""Aplikasi Rekomendasi Varietas — Hybrid LSTM + Random Forest (versi proposal).

Dua model bekerja berurutan:
  Model 1 (LSTM)          : membaca iklim 12 bulan terakhir -> prakiraan iklim bulan target + vektor temporal
  Model 2 (Random Forest) : vektor temporal + kondisi lahan -> Top-3 varietas + skor kecocokan + alasan

Persiapan (sekali):  python train_hybrid_proposal.py    -> membuat models/proposal_bundle.joblib
Jalankan          :  python app_rekomendasi_proposal.py -> buka http://127.0.0.1:8001
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.proposal_inference import ProposalRecommender

app = FastAPI(title="Rekomendasi Varietas - Hybrid LSTM + RF (Proposal)")
ENGINE = ProposalRecommender()


class RekomendasiRequest(BaseModel):
    id_lokasi: str
    periode: int  # tahun*12 + (bulan-1)
    tanah: dict[str, Any] | None = None


@app.get("/api/meta")
def meta():
    return ENGINE.meta()


@app.post("/api/rekomendasi")
def rekomendasi(req: RekomendasiRequest):
    try:
        return ENGINE.recommend(req.id_lokasi, req.periode // 12, req.periode % 12 + 1, req.tanah)
    except (ValueError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Rekomendasi Varietas Hybrid</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #f5f7f4; --card: #ffffff; --border: #dfe5dc; --text: #1d2a1f; --muted: #5d6b60;
    --accent: #2f7d4f; --accent-soft: #e3f1e8; --model1: #2563a8; --model1-soft: #e4eef9;
    --warn-bg: #fff6e0; --warn-text: #7a5a00; --bar: #e9eee8;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #121814; --card: #1b231e; --border: #2c3830; --text: #e6efe8; --muted: #9db0a3;
      --accent: #5cc18a; --accent-soft: #1f3528; --model1: #6fa8e6; --model1-soft: #1c2b3c;
      --warn-bg: #3a2f12; --warn-text: #f2d27a; --bar: #26302a;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 20px 16px 40px; }
  h1 { font-size: 1.45rem; margin: 0; }
  .sub { color: var(--muted); margin: 4px 0 14px; }
  .warn { background: var(--warn-bg); color: var(--warn-text); border-radius: 10px; padding: 10px 14px;
          font-size: .88rem; margin-bottom: 18px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 18px; }
  .form { display: grid; grid-template-columns: 2fr 2fr auto; gap: 12px; align-items: end; }
  label { display: block; font-size: .8rem; font-weight: 600; color: var(--muted); margin-bottom: 5px; }
  select, input { width: 100%; padding: 9px 10px; border-radius: 8px; border: 1px solid var(--border);
                  background: var(--bg); color: var(--text); font-size: .93rem; }
  button { padding: 10px 20px; border: 0; border-radius: 8px; background: var(--accent); color: #fff;
           font-weight: 600; font-size: .93rem; cursor: pointer; }
  button.ghost { background: transparent; color: var(--accent); border: 1px solid var(--accent); padding: 7px 14px; }
  details { margin-top: 14px; }
  summary { cursor: pointer; font-weight: 600; color: var(--accent); }
  .soil { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 12px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
  .tag { display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
         padding: 3px 9px; border-radius: 999px; margin-bottom: 8px; }
  .m1 { background: var(--model1-soft); color: var(--model1); }
  .m2 { background: var(--accent-soft); color: var(--accent); }
  h2 { font-size: 1.1rem; margin: 0 0 4px; }
  .note { color: var(--muted); font-size: .84rem; margin: 0 0 12px; }
  table { width: 100%; border-collapse: collapse; font-size: .88rem; }
  th, td { padding: 7px 6px; border-bottom: 1px solid var(--border); text-align: left; }
  th { color: var(--muted); font-weight: 600; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .chart { position: relative; height: 220px; margin-top: 14px; }
  .rec { border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; }
  .rec.top { border-color: var(--accent); }
  .rec-head { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }
  .rec-name { font-weight: 700; font-size: 1.02rem; }
  .rec-score { font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums; }
  .meter { height: 7px; background: var(--bar); border-radius: 99px; margin: 8px 0; overflow: hidden; }
  .meter > span { display: block; height: 100%; background: var(--accent); border-radius: 99px; }
  .meta { color: var(--muted); font-size: .82rem; }
  .why li { margin-bottom: 8px; }
  .why .meter > span { background: var(--model1); }
  .footer { margin-top: 18px; color: var(--muted); font-size: .8rem; }
  .err { color: #c0392b; font-weight: 600; }
  @media (max-width: 820px) { .form, .grid2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <h1>Rekomendasi Varietas Tanaman Pangan</h1>
  <p class="sub">Model Hybrid LSTM + Random Forest · Kabupaten Aceh Utara</p>
  <div class="warn"><b>Prototipe.</b> Label varietas dan sebagian data tanah (N, P, K, drainase, risiko hama) masih dummy,
    sehingga rekomendasi belum boleh dipakai untuk keputusan tanam nyata.</div>

  <div class="card">
    <div class="form">
      <div><label for="lokasi">Lokasi (kecamatan)</label><select id="lokasi"></select></div>
      <div><label for="periode">Bulan tanam yang direncanakan</label><select id="periode"></select></div>
      <button id="btn">Rekomendasikan</button>
    </div>
    <details>
      <summary>Kondisi lahan (opsional — ubah bila punya data sendiri)</summary>
      <div class="soil" id="soil"></div>
      <div style="margin-top:10px"><button class="ghost" id="reset" type="button">Kembalikan ke data lokasi</button></div>
    </details>
    <div id="status" class="note" style="margin:10px 0 0"></div>
  </div>

  <div class="grid2" id="hasil" hidden>
    <div class="card">
      <span class="tag m1">Model 1 · LSTM</span>
      <h2>Prakiraan iklim <span id="t1"></span></h2>
      <p class="note">Dibaca dari pola iklim <span id="win"></span>.</p>
      <table><thead><tr><th>Variabel</th><th class="num">Prakiraan</th><th class="num">Aktual</th><th>Satuan</th></tr></thead>
        <tbody id="tbl"></tbody></table>
      <div class="chart"><canvas id="chart"></canvas></div>
    </div>
    <div class="card">
      <span class="tag m2">Model 2 · Random Forest</span>
      <h2>Top-3 varietas untuk <span id="t2"></span></h2>
      <p class="note" id="m2note">Skor kecocokan dari Random Forest.</p>
      <div id="recs"></div>
      <h2 style="margin-top:16px">Alasan rekomendasi</h2>
      <p class="note">Faktor dengan kontribusi terbesar (feature importance).</p>
      <ul class="why" id="why" style="padding-left:18px;margin:0"></ul>
      <p class="note" id="label" style="margin-top:12px"></p>
    </div>
  </div>
  <div class="footer" id="footer"></div>
</div>

<script>
const NAMA_VAR = {curah_hujan:"Curah hujan", suhu_rata_rata:"Suhu rata-rata", suhu_maksimal:"Suhu maksimal",
  suhu_minimum:"Suhu minimum", kelembaban_rata_rata:"Kelembaban", lama_penyinaran:"Radiasi matahari",
  tekstur_tanah:"Tekstur tanah", ph_tanah:"pH tanah", nitrogen:"Nitrogen", fosfor:"Fosfor", kalium:"Kalium",
  karbon_organik:"C-organik (%)", drainase:"Drainase", ketinggian_tempat:"Ketinggian (mdpl)",
  risiko_bencana_hama:"Risiko bencana/hama"};
const BULAN = ["Januari","Februari","Maret","April","Mei","Juni","Juli","Agustus","September","Oktober","November","Desember"];
let META = null, chart = null;
const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function lokasiAktif() { return META.lokasi.find(l => l.id === $("lokasi").value); }

function isiPeriode() {
  const l = lokasiAktif(), sel = $("periode"), prev = sel.value;
  sel.innerHTML = "";
  for (let p = l.periode_max; p >= l.periode_min; p--) {
    const o = document.createElement("option");
    o.value = p;
    o.textContent = BULAN[p % 12] + " " + Math.floor(p / 12) + (p === l.periode_max ? "  (ke depan)" : "");
    sel.appendChild(o);
  }
  if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
}

function isiTanah() {
  const t = lokasiAktif().tanah, box = $("soil");
  box.innerHTML = "";
  META.static_cols.forEach(col => {
    const wrap = document.createElement("div");
    const lab = document.createElement("label");
    lab.textContent = NAMA_VAR[col] || col; lab.htmlFor = "s_" + col;
    let inp;
    if (META.opsi[col]) {
      inp = document.createElement("select");
      META.opsi[col].forEach(v => { const o = document.createElement("option"); o.value = o.textContent = v; inp.appendChild(o); });
    } else { inp = document.createElement("input"); inp.type = "number"; inp.step = "any"; }
    inp.id = "s_" + col; inp.value = t[col];
    wrap.append(lab, inp); box.appendChild(wrap);
  });
}

function bacaTanah() {
  const out = {};
  META.static_cols.forEach(col => { out[col] = $("s_" + col).value; });
  return out;
}

async function jalankan() {
  $("status").textContent = "Memproses…";
  const res = await fetch("/api/rekomendasi", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id_lokasi: $("lokasi").value, periode: Number($("periode").value), tanah: bacaTanah()})});
  const d = await res.json();
  if (!res.ok) { $("status").innerHTML = '<span class="err">' + d.error + "</span>"; return; }
  $("status").textContent = "";
  $("hasil").hidden = false;
  const judul = d.lokasi.nama + ", " + d.target;
  $("t1").textContent = judul; $("t2").textContent = judul; $("win").textContent = d.jendela_input;

  $("tbl").innerHTML = META.climate_cols.map(c =>
    `<tr><td>${NAMA_VAR[c]}</td><td class="num"><b>${d.prakiraan_lstm[c]}</b></td>` +
    `<td class="num">${d.aktual ? d.aktual[c] : "–"}</td><td>${META.satuan[c] || ""}</td></tr>`).join("");

  const labels = d.riwayat.map(r => r.periode).concat([d.target.slice(0, 3) + " " + d.target.split(" ")[1] + " *"]);
  const hujan = d.riwayat.map(r => r.curah_hujan);
  const ds = [
    {type: "bar", label: "Curah hujan (mm)", data: hujan.concat([null]), backgroundColor: css("--model1") + "88", yAxisID: "y"},
    {type: "bar", label: "Prakiraan LSTM (mm)", data: hujan.map(() => null).concat([d.prakiraan_lstm.curah_hujan]),
     backgroundColor: css("--accent"), yAxisID: "y"},
    {type: "line", label: "Suhu rata-rata (°C)", data: d.riwayat.map(r => r.suhu_rata_rata).concat([d.prakiraan_lstm.suhu_rata_rata]),
     borderColor: css("--warn-text"), backgroundColor: css("--warn-text"), yAxisID: "y1", tension: .3, pointRadius: 2},
  ];
  if (chart) chart.destroy();
  chart = new Chart($("chart"), {data: {labels, datasets: ds}, options: {responsive: true, maintainAspectRatio: false,
    plugins: {legend: {labels: {color: css("--muted"), boxWidth: 12, font: {size: 11}}}},
    scales: {x: {ticks: {color: css("--muted"), font: {size: 10}}, grid: {display: false}},
             y: {position: "left", ticks: {color: css("--muted")}, grid: {color: css("--border")}},
             y1: {position: "right", ticks: {color: css("--muted")}, grid: {display: false}}}}});

  $("recs").innerHTML = d.rekomendasi.map(r => `
    <div class="rec ${r.peringkat === 1 ? "top" : ""}">
      <div class="rec-head"><span class="rec-name">${r.peringkat}. ${r.varietas.replaceAll("_", " ")}</span>
        <span class="rec-score">${r.skor}%</span></div>
      <div class="meter"><span style="width:${r.skor}%"></span></div>
      <div class="meta">${r.komoditas || ""} · umur panen ${r.umur_panen_hari ?? "–"} hari · potensi ${r.potensi_hasil_ton_ha ?? "–"} t/ha · kebutuhan air ${r.kebutuhan_air ?? "–"}${r.estimasi_hasil_ton_ha != null ? ` · <b>perkiraan hasil ${r.estimasi_hasil_ton_ha} t/ha</b>` : ""}</div>
    </div>`).join("");
  $("m2note").textContent = d.model2.startsWith("RF kecocokan")
    ? "Skor kecocokan 0–100: Random Forest menilai ke-25 varietas lalu mengambil 3 tertinggi. Perkiraan hasil = potensi hasil × skor."
    : "Skor kecocokan = probabilitas kelas dari Random Forest.";

  $("why").innerHTML = d.alasan.map(a => `
    <li><b>${NAMA_VAR[a.faktor] || a.faktor}</b> — ${a.nilai}
      <div class="meter"><span style="width:${Math.round(a.kontribusi * 100)}%"></span></div>
      <span class="meta">kontribusi ${Math.round(a.kontribusi * 100)}%</span></li>`).join("");

  $("label").textContent = d.pembanding_dummy
    ? `Top-3 menurut skor dummy "sebenarnya" periode ini: ${d.pembanding_dummy.join(", ")}.`
    : d.label_prototipe
      ? `Label prototipe untuk periode ini: ${d.label_prototipe.replaceAll("_", " ")} — ${d.label_di_top_k ? "masuk" : "tidak masuk"} Top-3.`
      : "Periode ke depan: belum ada data aktual untuk dibandingkan.";
}

async function init() {
  META = await (await fetch("/api/meta")).json();
  $("lokasi").innerHTML = META.lokasi.map(l => `<option value="${l.id}">${l.id} · ${l.nama}</option>`).join("");
  isiPeriode(); isiTanah();
  $("lokasi").onchange = () => { isiPeriode(); isiTanah(); jalankan(); };
  $("periode").onchange = jalankan;
  $("btn").onclick = jalankan;
  $("reset").onclick = () => { isiTanah(); jalankan(); };
  const m2 = META.model2, mt = m2.metrik_test;
  $("footer").textContent = `Model 1 LSTM: run ${META.run_id} (${META.lokasi.length} lokasi, dilatih ${META.created_at}) · ` +
    `Model 2: ${m2.nama}` + (mt ? ` · test 2025 (tanpa bulan hujan ekstrem): MAE skor ${mt.MAE_normal.toFixed(1)}, ` +
    `Spearman ${mt.spearman_rata2_normal.toFixed(2)}, komoditas tepat ${(mt.komoditas_top1_tepat_normal * 100).toFixed(0)}%` :
    ` · Accuracy ${META.metrics.accuracy} · Top-3 ${META.metrics.top_3_accuracy}`);
  jalankan();
}
init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML_PAGE)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
