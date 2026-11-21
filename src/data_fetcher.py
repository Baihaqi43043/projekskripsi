"""Data fetcher for agroclimate (NASA POWER), soil (SoilGrids), and Hugging Face datasets."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests


def fetch_nasa_power_monthly(
    lat: float,
    lon: float,
    start_date: str = "20200101",
    end_date: str = "20251231",
    location_id: str = "L001",
) -> pd.DataFrame:
    """Fetch daily agroclimate data from NASA POWER API and aggregate to monthly."""
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

    monthly = df.resample("ME").agg(
        {
            "PRECTOTCORR": "sum",
            "T2M": "mean",
            "T2M_MAX": "mean",
            "T2M_MIN": "mean",
            "RH2M": "mean",
            "ALLSKY_SFC_SW_DWN": "mean",
        }
    )
    monthly["location_id"] = location_id
    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.month
    return monthly.reset_index(drop=True)


def fetch_soilgrids_point(lat: float, lon: float, location_id: str = "L001") -> dict[str, Any]:
    """Fetch static soil properties (0-15cm) from ISRIC SoilGrids v2.0 REST API."""
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

    result: dict[str, Any] = {"location_id": location_id, "lat": lat, "lon": lon}
    for layer in data.get("properties", {}).get("layers", []):
        name = layer["name"]
        vals = [
            d["values"]["mean"]
            for d in layer.get("depths", [])
            if d.get("values", {}).get("mean") is not None
        ]
        if vals:
            mean_val = sum(vals) / len(vals)
            # SoilGrids stores pH*10 and soc in dg/kg
            result[name] = mean_val / 10.0 if name in {"phh2o", "soc"} else mean_val
        else:
            result[name] = None
    return result


def fetch_hf_crop_recommendation(save_path: str = "data/raw/hf_crop_recommendation.csv") -> pd.DataFrame:
    """Load tabular crop recommendation dataset from Hugging Face (`liaad/agricultural-data`)."""
    from datasets import load_dataset

    ds = load_dataset("liaad/agricultural-data", split="train")
    df = ds.to_pandas()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def collect_aceh_utara_sample(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch agroclimate & soil data for configured Aceh Utara locations."""
    raw_dir = config["data"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    climate_frames = []
    soil_records = []

    for loc in config["data"]["locations"]:
        loc_id = loc["id"]
        print(f"Fetching NASA POWER for {loc_id} ({loc['name']})...")
        c_df = fetch_nasa_power_monthly(
            lat=loc["lat"],
            lon=loc["lon"],
            start_date=config["data"]["start_date"],
            end_date=config["data"]["end_date"],
            location_id=loc_id,
        )
        climate_frames.append(c_df)

        try:
            print(f"Fetching SoilGrids for {loc_id} ({loc['name']})...")
            s_rec = fetch_soilgrids_point(lat=loc["lat"], lon=loc["lon"], location_id=loc_id)
        except Exception as exc:
            print(f"  [Warning] SoilGrids unavailable for {loc_id}: {exc}. Using fallback defaults.")
            s_rec = {
                "location_id": loc_id,
                "lat": loc["lat"],
                "lon": loc["lon"],
                "phh2o": 6.1,
                "clay": 300.0,
                "sand": 350.0,
                "silt": 350.0,
                "soc": 15.0,
            }
        s_rec["elevation"] = loc.get("elevation", 5.0)
        soil_records.append(s_rec)

    climate_df = pd.concat(climate_frames, ignore_index=True)
    soil_df = pd.DataFrame(soil_records)

    climate_df.to_csv(os.path.join(raw_dir, "aceh_utara_climate_monthly.csv"), index=False)
    soil_df.to_csv(os.path.join(raw_dir, "aceh_utara_soil_static.csv"), index=False)
    return climate_df, soil_df

