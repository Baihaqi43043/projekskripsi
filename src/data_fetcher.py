"""Data fetcher untuk data agroklimat (NASA POWER), tanah (SoilGrids), dan dummy terstruktur."""

from __future__ import annotations

import hashlib
import os
import random
import time
from typing import Any

import pandas as pd
import requests


def _klasifikasi_tekstur(clay: float, sand: float, silt: float) -> str:
    """Konversi persentase clay/sand/silt menjadi kategori tekstur tanah (Pasir/Lanau/Lempung)."""
    if sand >= max(clay, silt):
        return "Pasir"
    if silt >= max(clay, sand):
        return "Lanau"
    return "Lempung"


def _dummy_soil_extras(id_lokasi: str) -> dict[str, Any]:
    """Hasilkan nilai dummy deterministik untuk N, P, K, drainase, dan risiko hama/bencana."""
    seed_int = int(hashlib.md5(id_lokasi.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed_int)
    return {
        "nitrogen": round(rng.uniform(0.10, 0.90), 2),
        "fosfor": round(rng.uniform(8.5, 16.5), 1),
        "kalium": round(rng.uniform(0.22, 0.60), 2),
        "drainase": rng.choices(["Baik", "Rusak"], weights=[0.75, 0.25])[0],
        "risiko_bencana_hama": rng.choices(["Rendah", "Sedang", "Tinggi"], weights=[0.5, 0.35, 0.15])[0],
    }


def fetch_nasa_power_daily(
    lat: float,
    lon: float,
    start_date: str = "20200101",
    end_date: str = "20251231",
    id_lokasi: str = "L001",
) -> pd.DataFrame:
    """Ambil data agroklimat harian dari NASA POWER API."""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M,ALLSKY_SFC_SW_DWN",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }
    resp = requests.get(url, params=params, timeout=45)
    resp.raise_for_status()
    payload = resp.json()

    props = payload["properties"]["parameter"]
    df = pd.DataFrame(props)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999.0, pd.NA).dropna()

    df = df.rename(
        columns={
            "PRECTOTCORR": "curah_hujan",
            "T2M": "suhu_rata_rata",
            "T2M_MAX": "suhu_maksimal",
            "T2M_MIN": "suhu_minimum",
            "RH2M": "kelembaban_rata_rata",
            "ALLSKY_SFC_SW_DWN": "lama_penyinaran",
        }
    )
    df["id_lokasi"] = id_lokasi
    df["tanggal"] = df.index.strftime("%Y-%m-%d")
    df["tahun"] = df.index.year
    df["bulan"] = df.index.month

    cols = [
        "id_lokasi",
        "tanggal",
        "tahun",
        "bulan",
        "curah_hujan",
        "suhu_rata_rata",
        "suhu_maksimal",
        "suhu_minimum",
        "kelembaban_rata_rata",
        "lama_penyinaran",
    ]
    return df[cols].reset_index(drop=True)


def fetch_nasa_power_monthly(
    lat: float,
    lon: float,
    start_date: str = "20200101",
    end_date: str = "20251231",
    id_lokasi: str = "L001",
) -> pd.DataFrame:
    """Agregasi data agroklimat harian NASA POWER menjadi bulanan."""
    daily_df = fetch_nasa_power_daily(
        lat=lat, lon=lon, start_date=start_date, end_date=end_date, id_lokasi=id_lokasi
    )
    return aggregate_daily_to_monthly(daily_df, id_lokasi=id_lokasi)


def aggregate_daily_to_monthly(daily_df: pd.DataFrame, id_lokasi: str) -> pd.DataFrame:
    """Agregasi DataFrame harian ke bulanan."""
    tmp = daily_df.copy()
    tmp["tanggal_dt"] = pd.to_datetime(tmp["tanggal"])
    monthly = tmp.set_index("tanggal_dt").resample("ME").agg(
        {
            "curah_hujan": "sum",
            "suhu_rata_rata": "mean",
            "suhu_maksimal": "mean",
            "suhu_minimum": "mean",
            "kelembaban_rata_rata": "mean",
            "lama_penyinaran": "mean",
        }
    )
    monthly["id_lokasi"] = id_lokasi
    monthly["tahun"] = monthly.index.year
    monthly["bulan"] = monthly.index.month

    cols = [
        "id_lokasi",
        "tahun",
        "bulan",
        "curah_hujan",
        "suhu_rata_rata",
        "suhu_maksimal",
        "suhu_minimum",
        "kelembaban_rata_rata",
        "lama_penyinaran",
    ]
    return monthly[cols].reset_index(drop=True)


def fetch_soilgrids_point(lat: float, lon: float, id_lokasi: str = "L001") -> dict[str, Any]:
    """Ambil properti tanah statis (0-15cm) dari ISRIC SoilGrids v2.0 + dummy N, P, K, drainase, risiko."""
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lat": lat,
        "lon": lon,
        "property": ["phh2o", "clay", "sand", "silt", "soc"],
        "depth": ["0-5cm", "5-15cm"],
        "value": ["mean"],
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    raw_vals: dict[str, float] = {}
    for layer in data.get("properties", {}).get("layers", []):
        name = layer["name"]
        vals = [
            d["values"]["mean"]
            for d in layer.get("depths", [])
            if d.get("values", {}).get("mean") is not None
        ]
        if vals:
            mean_val = sum(vals) / len(vals)
            raw_vals[name] = mean_val / 10.0 if name in {"phh2o", "soc"} else mean_val

    clay = raw_vals.get("clay", 300.0)
    sand = raw_vals.get("sand", 350.0)
    silt = raw_vals.get("silt", 350.0)

    extras = _dummy_soil_extras(id_lokasi)
    return {
        "id_lokasi": id_lokasi,
        "lintang": lat,
        "bujur": lon,
        "tekstur_tanah": _klasifikasi_tekstur(clay, sand, silt),
        "ph_tanah": round(raw_vals.get("phh2o", 5.6), 2),
        "nitrogen": extras["nitrogen"],
        "fosfor": extras["fosfor"],
        "kalium": extras["kalium"],
        "karbon_organik": round(raw_vals.get("soc", 3.5) * 0.1, 2),
        "drainase": extras["drainase"],
        "risiko_bencana_hama": extras["risiko_bencana_hama"],
    }


def fetch_hf_crop_recommendation(save_path: str = "data/raw/hf_crop_recommendation.csv") -> pd.DataFrame:
    """Muat dataset rekomendasi tanaman dari Hugging Face (`liaad/agricultural-data`)."""
    from datasets import load_dataset

    ds = load_dataset("liaad/agricultural-data", split="train")
    df = ds.to_pandas()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def collect_aceh_utara_sample(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ambil data agroklimat (NASA POWER) & tanah (SoilGrids + dummy N/P/K/drainase/risiko) untuk semua titik."""
    raw_dir = config["data"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    daily_frames = []
    climate_frames = []
    soil_records = []

    for loc in config["data"]["locations"]:
        loc_id = loc["id"]
        print(f"Mengambil data NASA POWER untuk {loc_id} ({loc['name']})...")
        d_df = fetch_nasa_power_daily(
            lat=loc["lat"],
            lon=loc["lon"],
            start_date=config["data"]["start_date"],
            end_date=config["data"]["end_date"],
            id_lokasi=loc_id,
        )
        daily_frames.append(d_df)
        climate_frames.append(aggregate_daily_to_monthly(d_df, id_lokasi=loc_id))

        s_rec = None
        for attempt in range(1, 4):
            try:
                print(f"Mengambil data SoilGrids untuk {loc_id} ({loc['name']}) [percobaan {attempt}]...")
                s_rec = fetch_soilgrids_point(lat=loc["lat"], lon=loc["lon"], id_lokasi=loc_id)
                break
            except Exception as exc:
                print(f"  [Info] SoilGrids gagal percobaan {attempt} untuk {loc_id}: {exc}")
                time.sleep(1)

        if s_rec is None:
            print(f"  [Info] Memakai fallback tanah dummy untuk {loc_id}.")
            seed_int = int(hashlib.md5(loc_id.encode("utf-8")).hexdigest()[:8], 16)
            rng = random.Random(seed_int + 7)
            extras = _dummy_soil_extras(loc_id)
            s_rec = {
                "id_lokasi": loc_id,
                "lintang": loc["lat"],
                "bujur": loc["lon"],
                "tekstur_tanah": rng.choice(["Lanau", "Lempung", "Pasir"]),
                "ph_tanah": round(rng.uniform(5.0, 6.6), 2),
                "nitrogen": extras["nitrogen"],
                "fosfor": extras["fosfor"],
                "kalium": extras["kalium"],
                "karbon_organik": round(rng.uniform(2.5, 5.2), 2),
                "drainase": extras["drainase"],
                "risiko_bencana_hama": extras["risiko_bencana_hama"],
            }
        time.sleep(1)

        s_rec["nama_lokasi"] = loc["name"]
        s_rec["ketinggian_tempat"] = loc.get("ketinggian_tempat", 5.0)
        soil_records.append(s_rec)

    daily_df = pd.concat(daily_frames, ignore_index=True)
    climate_df = pd.concat(climate_frames, ignore_index=True)
    soil_df = pd.DataFrame(soil_records)

    daily_df.to_csv(os.path.join(raw_dir, "aceh_utara_agroklimat_harian.csv"), index=False)
    climate_df.to_csv(os.path.join(raw_dir, "aceh_utara_agroklimat_bulanan.csv"), index=False)
    soil_df.to_csv(os.path.join(raw_dir, "aceh_utara_tanah_statis.csv"), index=False)
    return climate_df, soil_df
