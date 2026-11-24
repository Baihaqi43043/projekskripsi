"""Web Dashboard Interaktif Prediksi Cuaca Agroklimat (FastAPI + BiLSTM Production Inference)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from src.weather_model import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    FEATURES,
    get_preprocessing,
    input_feature_names,
    inverse_targets,
    load_daily_csv,
    load_weather_model,
    transform_inputs,
)

app = FastAPI(title="Prediksi Cuaca Agroklimat Aceh Utara - BiLSTM")

TOLERANCES = {
    "curah_hujan": 5.0,
    "suhu_rata_rata": 1.0,
    "suhu_maksimal": 1.5,
    "suhu_minimum": 1.5,
    "kelembaban_rata_rata": 5.0,
    "lama_penyinaran": 3.0,
}

# Preload model dan dataset sekali saat server menyala agar inference cepat (< 50 ms)
def _init_engine():
    df_train = load_daily_csv(DEFAULT_TRAIN_CSV)
    df_test = load_daily_csv(DEFAULT_TEST_CSV)
    prep = get_preprocessing(DEFAULT_MODEL_PATH, DEFAULT_TRAIN_CSV)

    df_tail_2024 = df_train[df_train["tanggal"] >= "2024-12-01"].copy()
    df_full_raw = pd.concat([df_tail_2024, df_test], ignore_index=True).sort_values(["id_lokasi", "tanggal"]).reset_index(drop=True)
    df_full_scaled = transform_inputs(df_full_raw, prep)
    loc_names = df_test.groupby("id_lokasi")["nama_lokasi"].first().to_dict()

    return {
        "model": load_weather_model(DEFAULT_MODEL_PATH, prep, device="cpu"),
        "prep": prep,
        "bounds": prep["bounds"],
        "window_size": prep["window_size"],
        "df_full_raw": df_full_raw,
        "df_full_scaled": df_full_scaled,
        "input_features": input_feature_names(prep),
        "loc_names": loc_names,
    }


ENGINE = _init_engine()


@app.get("/api/predict")
def predict_weather(
    location: str = Query("L001", description="ID Lokasi (L001 - L005)"),
    start_date: str = Query("2025-01-01", description="Tanggal mulai YYYY-MM-DD"),
    end_date: str = Query("2025-03-31", description="Tanggal akhir YYYY-MM-DD"),
):
    window_size = ENGINE["window_size"]
    df_scaled = ENGINE["df_full_scaled"]
    df_raw = ENGINE["df_full_raw"]

    grp = df_scaled[df_scaled["id_lokasi"] == location].sort_values("tanggal").reset_index(drop=True)
    raw_grp = df_raw[df_raw["id_lokasi"] == location].sort_values("tanggal").reset_index(drop=True)

    if grp.empty:
        return JSONResponse({"error": f"Lokasi {location} tidak ditemukan."}, status_code=404)

    in_arr = grp[ENGINE["input_features"]].to_numpy(dtype=np.float32)
    x_seqs, target_indices = [], []
    for i in range(len(grp)):
        tgl = grp.loc[i, "tanggal"]
        if start_date <= tgl <= end_date and i >= window_size:
            x_seqs.append(in_arr[i - window_size : i])
            target_indices.append(i)

    if not x_seqs:
        return JSONResponse(
            {"error": "Rentang tanggal tidak menghasilkan data prediksi. Pilih rentang antara 2025-01-01 s/d 2025-12-31."},
            status_code=400,
        )

    with torch.no_grad():
        preds_scaled = ENGINE["model"](torch.tensor(np.array(x_seqs), dtype=torch.float32)).numpy()

    preds_unscaled = inverse_targets(preds_scaled, ENGINE["prep"])

    dates = []
    series = {col: {"aktual": [], "prediksi": []} for col in FEATURES}
    table_rows = []

    for k, idx in enumerate(target_indices):
        tgl = str(raw_grp.loc[idx, "tanggal"])
        dates.append(tgl)
        row_item = {"tanggal": tgl}
        for f_i, col in enumerate(FEATURES):
            lb, ub = ENGINE["bounds"][col]
            act_val = float(np.clip(raw_grp.loc[idx, col], lb, ub))
            prd_val = float(preds_unscaled[k, f_i])
            if col == "curah_hujan":
                prd_val = max(0.0, prd_val)
            series[col]["aktual"].append(round(act_val, 2))
            series[col]["prediksi"].append(round(prd_val, 2))
            row_item[f"aktual_{col}"] = round(act_val, 2)
            row_item[f"pred_{col}"] = round(prd_val, 2)
        table_rows.append(row_item)

    metrics = []
    wape_list, tol_list = [], []
    for col in FEATURES:
        act = np.array(series[col]["aktual"], dtype=float)
        prd = np.array(series[col]["prediksi"], dtype=float)
        mae = float(np.mean(np.abs(act - prd)))
        rmse = float(np.sqrt(np.mean((act - prd) ** 2)))
        wape = (np.sum(np.abs(act - prd)) / (np.sum(np.abs(act)) + 1e-8)) * 100.0
        acc_wape = max(0.0, 100.0 - float(wape))
        acc_tol = float(np.mean(np.abs(act - prd) <= TOLERANCES[col]) * 100.0)
        wape_list.append(acc_wape)
        tol_list.append(acc_tol)
        metrics.append({
            "parameter": col,
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "acc_wape": round(acc_wape, 2),
            "acc_tol": round(acc_tol, 2),
            "toleransi": f"±{TOLERANCES[col]}",
        })

    rain_act = np.array(series["curah_hujan"]["aktual"]) >= 0.5
    rain_prd = np.array(series["curah_hujan"]["prediksi"]) >= 0.5
    rain_cls_acc = round(float(np.mean(rain_act == rain_prd) * 100.0), 2)

    return {
        "location": location,
        "location_name": ENGINE["loc_names"].get(location, location),
        "start_date": dates[0],
        "end_date": dates[-1],
        "total_days": len(dates),
        "summary": {
            "mean_acc_wape": round(float(np.mean(wape_list)), 2),
            "mean_acc_tol": round(float(np.mean(tol_list)), 2),
            "rain_cls_acc": rain_cls_acc,
        },
        "metrics": metrics,
        "dates": dates,
        "series": series,
        "rows": table_rows,
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Prediksi Cuaca Agroklimat Aceh Utara (BiLSTM)</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent2: #22c55e;
    }
    * { box-sizing: border-box; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
    body { margin: 0; background: var(--bg); color: var(--text); padding: 24px; }
    .container { max-width: 1320px; margin: 0 auto; }
    header { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 24px; }
    h1 { margin: 0; font-size: 1.5rem; font-weight: 700; color: #fff; }
    .subtitle { margin: 4px 0 0; color: var(--muted); font-size: 0.9rem; }
    .controls {
      background: var(--card); border: 1px solid var(--border); border-radius: 12px;
      padding: 18px; display: flex; flex-wrap: wrap; gap: 16px; align-items: end; margin-bottom: 22px;
    }
    .field { display: flex; flex-direction: column; gap: 6px; min-width: 180px; flex: 1; }
    label { font-size: 0.82rem; color: var(--muted); font-weight: 600; }
    select, input[type="date"] {
      background: #0f172a; color: #fff; border: 1px solid var(--border);
      border-radius: 8px; padding: 10px 12px; font-size: 0.92rem;
    }
    button {
      background: linear-gradient(135deg, #0284c7, #2563eb); color: #fff;
      border: none; border-radius: 8px; padding: 11px 22px; font-size: 0.92rem;
      font-weight: 600; cursor: pointer; transition: opacity .2s;
    }
    button:hover { opacity: 0.9; }
    .btn-secondary { background: #334155; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 22px; }
    .card {
      background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px;
    }
    .card .label { font-size: 0.8rem; color: var(--muted); }
    .card .val { font-size: 1.6rem; font-weight: 700; margin-top: 6px; color: var(--accent); }
    .card .val.green { color: var(--accent2); }
    .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 18px; margin-bottom: 24px; }
    .chart-box {
      background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px;
    }
    .chart-box h3 { margin: 0 0 12px; font-size: 1rem; color: #e2e8f0; display: flex; justify-content: space-between; }
    .chart-box h3 span { font-size: 0.8rem; color: var(--muted); font-weight: 500; }
    .chart-wrap { position: relative; height: 260px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
    th, td { padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; }
    th { color: var(--muted); font-weight: 600; background: #162032; }
    .table-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; overflow-x: auto; margin-bottom: 24px; }
    .badge { padding: 3px 8px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; background: rgba(34,197,94,.18); color: #4ade80; }
    #status { font-size: 0.88rem; color: var(--accent); margin-left: auto; }
  </style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>Dashboard Interaktif Prediksi Cuaca Agroklimat (BiLSTM + Attention)</h1>
      <p class="subtitle">Simulasi Production Inference Kabupaten Aceh Utara — Tahun Uji 2025</p>
    </div>
    <div id="status">Memuat model...</div>
  </header>

  <section class="controls">
    <div class="field">
      <label for="location">Pilih Lokasi Kecamatan</label>
      <select id="location">
        <option value="L001">L001 - Lhoksukon</option>
        <option value="L002">L002 - Matangkuli</option>
        <option value="L003">L003 - Syamtalira Aron</option>
        <option value="L004">L004 - Dewantara</option>
        <option value="L005">L005 - Kuta Makmur</option>
      </select>
    </div>
    <div class="field">
      <label for="start_date">Tanggal Mulai (2025)</label>
      <input type="date" id="start_date" value="2025-01-01" min="2025-01-01" max="2025-12-31" />
    </div>
    <div class="field">
      <label for="end_date">Tanggal Akhir (2025)</label>
      <input type="date" id="end_date" value="2025-03-31" min="2025-01-01" max="2025-12-31" />
    </div>
    <button onclick="runPrediction()">Jalankan Prediksi</button>
    <button class="btn-secondary" onclick="exportCSV()">Export CSV</button>
  </section>

  <section class="cards">
    <div class="card"><div class="label">Lokasi Terpilih</div><div class="val" id="c_loc">-</div></div>
    <div class="card"><div class="label">Jumlah Hari Diprediksi</div><div class="val" id="c_days">-</div></div>
    <div class="card"><div class="label">Rata-rata Akurasi WAPE</div><div class="val green" id="c_wape">-</div></div>
    <div class="card"><div class="label">Rata-rata Akurasi Toleransi</div><div class="val green" id="c_tol">-</div></div>
    <div class="card"><div class="label">Akurasi Hujan vs Cerah</div><div class="val" id="c_rain">-</div></div>
  </section>

  <section class="charts-grid" id="charts_container"></section>

  <section class="table-card">
    <h3 style="margin-top:0;">Ringkasan Metrik Evaluasi per Parameter</h3>
    <table>
      <thead>
        <tr>
          <th>Parameter Cuaca</th>
          <th>MAE</th>
          <th>RMSE</th>
          <th>Akurasi WAPE (%)</th>
          <th>Akurasi Toleransi (%)</th>
          <th>Batas Toleransi</th>
        </tr>
      </thead>
      <tbody id="metrics_body"></tbody>
    </table>
  </section>
</div>

<script>
const PARAM_META = {
  curah_hujan: { title: "1. Curah Hujan (mm/hari)", actColor: "#38bdf8", prdColor: "#f43f5e" },
  suhu_rata_rata: { title: "2. Suhu Rata-rata (°C)", actColor: "#22c55e", prdColor: "#f59e0b" },
  suhu_maksimal: { title: "3. Suhu Maksimal (°C)", actColor: "#fb923c", prdColor: "#e879f9" },
  suhu_minimum: { title: "4. Suhu Minimum (°C)", actColor: "#2dd4bf", prdColor: "#60a5fa" },
  kelembaban_rata_rata: { title: "5. Kelembaban Rata-rata (%)", actColor: "#a78bfa", prdColor: "#fb7185" },
  lama_penyinaran: { title: "6. Lama Penyinaran / Radiasi (MJ/m²/hari)", actColor: "#facc15", prdColor: "#34d399" }
};

let chartInstances = {};
let lastPayload = null;

async function runPrediction() {
  const loc = document.getElementById("location").value;
  const start = document.getElementById("start_date").value;
  const end = document.getElementById("end_date").value;
  const statusEl = document.getElementById("status");
  statusEl.textContent = "Menjalankan inference model...";

  const res = await fetch(`/api/predict?location=${loc}&start_date=${start}&end_date=${end}`);
  const data = await res.json();
  if (!res.ok) {
    statusEl.textContent = "Error: " + (data.error || "Gagal memprediksi");
    alert(data.error || "Gagal memprediksi");
    return;
  }

  lastPayload = data;
  statusEl.textContent = `Selesai: ${data.location_name} (${data.start_date} s/d ${data.end_date})`;

  document.getElementById("c_loc").textContent = data.location_name;
  document.getElementById("c_days").textContent = data.total_days + " Hari";
  document.getElementById("c_wape").textContent = data.summary.mean_acc_wape + "%";
  document.getElementById("c_tol").textContent = data.summary.mean_acc_tol + "%";
  document.getElementById("c_rain").textContent = data.summary.rain_cls_acc + "%";

  const mBody = document.getElementById("metrics_body");
  mBody.innerHTML = data.metrics.map(m => `
    <tr>
      <td><strong>${m.parameter}</strong></td>
      <td>${m.mae}</td>
      <td>${m.rmse}</td>
      <td><span class="badge">${m.acc_wape}%</span></td>
      <td><span class="badge">${m.acc_tol}%</span></td>
      <td>${m.toleransi}</td>
    </tr>
  `).join("");

  const container = document.getElementById("charts_container");
  if (Object.keys(chartInstances).length === 0) {
    container.innerHTML = Object.keys(PARAM_META).map(key => `
      <div class="chart-box">
        <h3>${PARAM_META[key].title} <span id="sub_${key}"></span></h3>
        <div class="chart-wrap"><canvas id="chart_${key}"></canvas></div>
      </div>
    `).join("");
  }

  data.metrics.forEach(m => {
    const key = m.parameter;
    const meta = PARAM_META[key];
    document.getElementById(`sub_${key}`).textContent = `MAE: ${m.mae} | Akurasi: ${m.acc_wape}%`;
    const ctx = document.getElementById(`chart_${key}`).getContext("2d");

    if (chartInstances[key]) {
      chartInstances[key].destroy();
    }

    chartInstances[key] = new Chart(ctx, {
      type: "line",
      data: {
        labels: data.dates,
        datasets: [
          {
            label: "Aktual 2025",
            data: data.series[key].aktual,
            borderColor: meta.actColor,
            backgroundColor: meta.actColor + "22",
            borderWidth: 2,
            pointRadius: data.total_days > 120 ? 0 : 2,
            tension: 0.25
          },
          {
            label: "Prediksi BiLSTM",
            data: data.series[key].prediksi,
            borderColor: meta.prdColor,
            borderDash: [5, 4],
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.25
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: "#e2e8f0" } }
        },
        scales: {
          x: { ticks: { color: "#94a3b8", maxTicksLimit: 10 }, grid: { color: "#33415555" } },
          y: { ticks: { color: "#94a3b8" }, grid: { color: "#33415555" } }
        }
      }
    });
  });
}

function exportCSV() {
  if (!lastPayload || !lastPayload.rows.length) return;
  const cols = Object.keys(lastPayload.rows[0]);
  const csvLines = [cols.join(",")];
  lastPayload.rows.forEach(r => {
    csvLines.push(cols.map(c => r[c]).join(","));
  });
  const blob = new Blob([csvLines.join("\\n")], { type: "text/csv;charset=utf-8;" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `prediksi_${lastPayload.location}_${lastPayload.start_date}_sd_${lastPayload.end_date}.csv`;
  a.click();
}

window.onload = runPrediction;
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_PAGE)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
