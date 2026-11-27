"""Unduh ulang data mentah ke data/raw/ (NASA POWER untuk iklim, SoilGrids untuk tanah).

Data mentah sudah di-commit ke Git, jadi script ini hanya perlu dijalankan bila ingin
memperbarui data atau mengganti daftar lokasi di configs/model_config.yaml.

Contoh:
    python -m pipelines.data.fetch_data                    # iklim + tanah
    python -m pipelines.data.fetch_data --only climate     # hanya NASA POWER (harian & bulanan)
    python -m pipelines.data.fetch_data --only soil        # hanya SoilGrids
    python -m pipelines.data.fetch_data --only soil --allow-dummy-soil   # uji pipeline saat SoilGrids down
"""

from __future__ import annotations

import argparse

from src.data.data_fetcher import collect_climate, collect_soil
from src.data.dataset_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Unduh data agroklimat & tanah ke data/raw/.")
    parser.add_argument("--only", choices=["all", "climate", "soil"], default="all")
    parser.add_argument(
        "--allow-dummy-soil",
        action="store_true",
        help="Izinkan nilai tanah dummy bila SoilGrids gagal (ditandai di kolom sumber_tanah).",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.only in ("all", "climate"):
        daily_df, monthly_df = collect_climate(cfg)
        print(f"[OK] Iklim: harian {len(daily_df)} baris, bulanan {len(monthly_df)} baris.")
    if args.only in ("all", "soil"):
        soil_df = collect_soil(cfg, allow_dummy=args.allow_dummy_soil)
        print(f"[OK] Tanah: {len(soil_df)} lokasi -> {soil_df['sumber_tanah'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
