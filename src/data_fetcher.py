"""Data fetcher untuk data agroklimat (NASA POWER), tanah (SoilGrids), dan Hugging Face."""

from __future__ import annotations

import os
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
    resp = requests.get(url, params=params, timeout=30)
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
    daily_df["tanggal_dt"] = pd.to_datetime(daily_df["tanggal"])
    monthly = daily_df.set_index("tanggal_dt").resample("ME").agg(
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
    """Ambil properti tanah statis (0-15cm) dari ISRIC SoilGrids v2.0 REST API."""
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

    return {
        "id_lokasi": id_lokasi,
        "lintang": lat,
        "bujur": lon,
        "tekstur_tanah": _klasifikasi_tekstur(clay, sand, silt),
        "ph_tanah": round(raw_vals.get("phh2o", 6.1), 2),
        # Nilai awal placeholder N, P, K (mg/kg) sebelum diisi data Dinas/lab tanah
        "nitrogen": 0.15,
        "fosfor": 12.0,
        "kalium": 0.35,
        # Konversi dg/kg -> % (1 dg/kg = 0.01%)
        "karbon_organik": round(raw_vals.get("soc", 15.0) * 0.1, 2),
        "drainase": "Baik",
        "risiko_bencana_hama": "Rendah",
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
    """Ambil data agroklimat & tanah untuk titik lokasi Aceh Utara."""
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

        d_tmp = d_df.copy()
        d_tmp["tanggal_dt"] = pd.to_datetime(d_tmp["tanggal"])
        c_df = d_tmp.set_index("tanggal_dt").resample("ME").agg(
            {
                "curah_hujan": "sum",
                "suhu_rata_rata": "mean",
                "suhu_maksimal": "mean",
                "suhu_minimum": "mean",
                "kelembaban_rata_rata": "mean",
                "lama_penyinaran": "mean",
            }
        )
        c_df["id_lokasi"] = loc_id
        c_df["tahun"] = c_df.index.year
        c_df["bulan"] = c_df.index.month
        c_df = c_df[
            [
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
        ].reset_index(drop=True)
        climate_frames.append(c_df)

        s_rec = None
        for attempt in range(1, 5):
            try:
                print(f"Mengambil data SoilGrids untuk {loc_id} ({loc['name']}) [percobaan {attempt}]...")
                s_rec = fetch_soilgrids_point(lat=loc["lat"], lon=loc["lon"], id_lokasi=loc_id)
                break
            except Exception as exc:
                print(f"  [Info] SoilGrids gagal percobaan {attempt} untuk {loc_id}: {exc}")
                time.sleep(1)

        if s_rec is None:
            print(f"  [Info] Memakai nilai default sementara untuk {loc_id}.")
            s_rec = {
                "id_lokasi": loc_id,
                "lintang": loc["lat"],
                "bujur": loc["lon"],
                "tekstur_tanah": "Lempung",
                "ph_tanah": 6.1,
                "nitrogen": 0.15,
                "fosfor": 12.0,
                "kalium": 0.35,
                "karbon_organik": 1.5,
                "drainase": "Baik",
                "risiko_bencana_hama": "Rendah",
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
