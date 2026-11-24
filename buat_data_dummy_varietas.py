"""Buat data DUMMY many-to-many: lokasi × bulan tanam × varietas -> skor kecocokan & hasil panen.

Tujuan: mengganti label if-else (1 varietas per lokasi-bulan) dengan data berpasangan yang
memakai karakteristik varietas (data/raw/varietas_tanaman.csv), sesuai Tujuan 2 proposal.

Setiap baris = satu varietas yang "ditanam" di satu lokasi pada satu bulan tanam.
Skor kecocokan dihitung dengan fungsi kontinu (tanpa if-else yang memilih varietas):

    skor = 100 × s_air^0.35 × s_suhu^0.15 × s_ph^0.20 × s_drainase^0.10 × s_hama^0.10 × s_ketinggian^0.10 + noise

  s_air        kebutuhan air varietas vs rata-rata hujan bulanan selama masa tanam (umur panen)
  s_suhu       suhu optimum komoditas vs rata-rata suhu selama masa tanam
  s_ph         pH optimum komoditas vs pH tanah lokasi
  s_drainase   kecocokan komoditas dengan kondisi drainase
  s_hama       ketahanan hama varietas vs risiko hama lokasi
  s_ketinggian ketinggian lokasi vs batas ketinggian optimal varietas

  hasil_dummy_ton_ha = potensi hasil varietas × skor/100 × noise
  umur_panen_aktual_dummy = umur deskripsi × ((27 − 10) / (suhu masa tanam − 10))^0,5 + noise (thermal time)

PERINGATAN: semua angka optimum di bawah adalah ASUMSI dari kisaran umum literatur agronomi,
bukan data lapangan. Wajib divalidasi pakar/penyuluh sebelum dipakai sebagai dasar klaim.

Output (folder data/raw/dummy_varietas/):
  kecocokan_varietas_semua.csv   seluruh data (2021-2025)
  kecocokan_varietas_train.csv   80% pertama secara kronologis (bulan tanam 2021-2024)
  kecocokan_varietas_test.csv    20% terakhir (bulan tanam 2025)

Jalankan: python buat_data_dummy_varietas.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.dataset_loader import filter_active_locations, load_config

SEED = 42
TAHUN_MULAI, TAHUN_AKHIR = 2021, 2025  # 5 tahun × 12 bulan = 60 bulan tanam
OUT_DIR = "data/raw/dummy_varietas"

# ---------------------------------------------------------------- asumsi agronomi (perlu validasi pakar)
# Kebutuhan air varietas -> hujan bulanan optimum (mm/bulan) dan lebar toleransi
AIR_OPT = {"Banyak": (230.0, 90.0), "Sedang": (160.0, 75.0), "Sedikit": (100.0, 65.0)}
# Suhu rata-rata optimum per komoditas (°C) dan lebar toleransi
SUHU_OPT = {"Padi": (26.5, 2.5), "Jagung": (27.0, 3.0), "Kedelai": (26.0, 2.5)}
# pH tanah optimum per komoditas dan lebar toleransi
PH_OPT = {"Padi": (5.8, 0.9), "Jagung": (6.2, 0.9), "Kedelai": (6.5, 0.8)}
# Kecocokan komoditas terhadap drainase (padi sawah toleran drainase buruk)
DRAINASE = {("Padi", "Baik"): 0.95, ("Padi", "Rusak"): 1.0,
            ("Jagung", "Baik"): 1.0, ("Jagung", "Rusak"): 0.6,
            ("Kedelai", "Baik"): 1.0, ("Kedelai", "Rusak"): 0.55}
RISIKO_LEVEL = {"Rendah": 0.0, "Sedang": 1.0, "Tinggi": 2.0}
# Umur panen aktual: suhu acuan deskripsi varietas dan suhu basis perkembangan tanaman (°C)
T_REF, T_BASIS = 27.0, 10.0
BOBOT = {"s_air": 0.35, "s_suhu": 0.15, "s_ph": 0.20, "s_drainase": 0.10, "s_hama": 0.10, "s_ketinggian": 0.10}


def _rentang_ke_angka(val, ambil: str = "tengah") -> float:
    """'114-119' -> 116.5 (tengah) atau 119 (maks); '7.0' -> 7.0."""
    parts = [float(x) for x in str(val).replace(",", ".").split("-") if x.strip()]
    return max(parts) if ambil == "maks" else float(np.mean(parts))


def gauss(x, opt, lebar):
    return np.exp(-(((x - opt) / lebar) ** 2))


def muat_varietas(raw_dir: str) -> pd.DataFrame:
    v = pd.read_csv(os.path.join(raw_dir, "varietas_tanaman.csv"))
    v["umur_panen_hari_num"] = v["umur_panen_hari"].map(_rentang_ke_angka)
    v["potensi_hasil_num"] = v["potensi_hasil_ton_ha"].map(_rentang_ke_angka)
    v["ketinggian_maks_mdpl"] = v["ketinggian_optimal_mdpl"].map(lambda s: _rentang_ke_angka(s, "maks"))
    v["bulan_tumbuh"] = np.clip(np.round(v["umur_panen_hari_num"] / 30.0), 2, 4).astype(int)
    return v


def iklim_masa_tanam(climate: pd.DataFrame, n_bulan: int) -> pd.DataFrame:
    """Rata-rata hujan & suhu bulanan dari bulan tanam t sampai t+n-1 (dipotong di akhir data)."""
    out = []
    for loc, g in climate.sort_values(["id_lokasi", "tahun", "bulan"]).groupby("id_lokasi"):
        g = g.reset_index(drop=True)
        hujan = g["curah_hujan"][::-1].rolling(n_bulan, min_periods=1).mean()[::-1]
        suhu = g["suhu_rata_rata"][::-1].rolling(n_bulan, min_periods=1).mean()[::-1]
        out.append(pd.DataFrame({"id_lokasi": loc, "tahun": g["tahun"], "bulan": g["bulan"],
                                 "bulan_tumbuh": n_bulan, "hujan_masa_tanam": hujan.values,
                                 "suhu_masa_tanam": suhu.values}))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    cfg = load_config()
    raw_dir = cfg["data"]["raw_dir"]
    rng = np.random.default_rng(SEED)

    climate = pd.read_csv(os.path.join(raw_dir, "aceh_utara_agroklimat_bulanan.csv"))
    soil = pd.read_csv(os.path.join(raw_dir, "aceh_utara_tanah_statis.csv"))
    climate, soil = filter_active_locations(climate, soil, cfg)
    var = muat_varietas(raw_dir)

    iklim = pd.concat([iklim_masa_tanam(climate, n) for n in sorted(var["bulan_tumbuh"].unique())])
    lokasi = soil[["id_lokasi", "nama_lokasi", "ph_tanah", "drainase", "ketinggian_tempat", "risiko_bencana_hama"]]
    periode = climate[(climate["tahun"] >= TAHUN_MULAI) & (climate["tahun"] <= TAHUN_AKHIR)][["id_lokasi", "tahun", "bulan"]]

    # many-to-many: setiap (lokasi, bulan tanam) dipasangkan dengan SETIAP varietas
    df = periode.merge(lokasi, on="id_lokasi").merge(var, how="cross")
    df = df.merge(iklim, on=["id_lokasi", "tahun", "bulan", "bulan_tumbuh"], how="left")

    air = np.array([AIR_OPT[k] for k in df["kebutuhan_air"]])
    suhu = np.array([SUHU_OPT[k] for k in df["komoditas"]])
    ph = np.array([PH_OPT[k] for k in df["komoditas"]])
    df["s_air"] = gauss(df["hujan_masa_tanam"], air[:, 0], air[:, 1])
    df["s_suhu"] = gauss(df["suhu_masa_tanam"], suhu[:, 0], suhu[:, 1])
    df["s_ph"] = gauss(df["ph_tanah"], ph[:, 0], ph[:, 1])
    df["s_drainase"] = [DRAINASE[(k, d)] for k, d in zip(df["komoditas"], df["drainase"])]
    risiko = df["risiko_bencana_hama"].map(RISIKO_LEVEL)
    df["s_hama"] = 1.0 - 0.12 * risiko * (5 - df["ketahanan_hama_skala"]) / 4.0
    df["s_ketinggian"] = np.minimum(1.0, np.exp(-(df["ketinggian_tempat"] - df["ketinggian_maks_mdpl"]) / 200.0))

    skor = 100.0 * np.prod([df[k] ** w for k, w in BOBOT.items()], axis=0)
    df["skor_kecocokan"] = np.clip(skor + rng.normal(0, 3.0, len(df)), 0, 100).round(2)
    df["hasil_dummy_ton_ha"] = (df["potensi_hasil_num"] * df["skor_kecocokan"] / 100.0
                                * rng.lognormal(0, 0.08, len(df))).round(2)
    # Umur panen aktual (thermal time): makin panas masa tanam, makin cepat matang.
    # umur = umur_deskripsi × ((T_REF − T_BASIS) / (suhu_masa_tanam − T_BASIS))^0,5 + noise (sd 2 hari)
    # Noise diambil SETELAH noise skor & hasil agar kolom lama tetap identik.
    faktor = ((T_REF - T_BASIS) / (df["suhu_masa_tanam"] - T_BASIS)) ** 0.5
    df["umur_panen_aktual_dummy"] = (df["umur_panen_hari_num"] * faktor + rng.normal(0, 2.0, len(df))).round(0)

    grp = df.groupby(["id_lokasi", "tahun", "bulan"])["skor_kecocokan"]
    df["peringkat"] = grp.rank(ascending=False, method="first").astype(int)
    df["varietas_terbaik"] = df["peringkat"] == 1

    n_periode = (TAHUN_AKHIR - TAHUN_MULAI + 1) * 12
    batas = TAHUN_MULAI * 12 + int(round(n_periode * 0.8))  # 80% bulan pertama -> train
    df["split"] = np.where(df["tahun"] * 12 + df["bulan"] - 1 < batas, "train", "test")

    kolom = ["id_lokasi", "nama_lokasi", "tahun", "bulan", "id_varietas", "nama_varietas", "komoditas",
             "umur_panen_hari_num", "potensi_hasil_num", "kebutuhan_air", "ketahanan_hama_skala", "ketinggian_maks_mdpl",
             "ph_tanah", "drainase", "ketinggian_tempat", "risiko_bencana_hama",
             "bulan_tumbuh", "hujan_masa_tanam", "suhu_masa_tanam",
             "s_air", "s_suhu", "s_ph", "s_drainase", "s_hama", "s_ketinggian",
             "skor_kecocokan", "hasil_dummy_ton_ha", "umur_panen_aktual_dummy", "peringkat", "varietas_terbaik", "split"]
    df = df[kolom].sort_values(["tahun", "bulan", "id_lokasi", "peringkat"]).reset_index(drop=True)
    for c in ["hujan_masa_tanam", "suhu_masa_tanam", "s_air", "s_suhu", "s_ph", "s_drainase", "s_hama", "s_ketinggian"]:
        df[c] = df[c].round(4)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "kecocokan_varietas_semua.csv"), index=False)
    df[df["split"] == "train"].to_csv(os.path.join(OUT_DIR, "kecocokan_varietas_train.csv"), index=False)
    df[df["split"] == "test"].to_csv(os.path.join(OUT_DIR, "kecocokan_varietas_test.csv"), index=False)

    tr, te = (df["split"] == "train").sum(), (df["split"] == "test").sum()
    print(f"Total {len(df)} baris = {df['id_lokasi'].nunique()} lokasi × {n_periode} bulan × {df['id_varietas'].nunique()} varietas")
    print(f"Train {tr} ({tr / len(df):.0%}) | Test {te} ({te / len(df):.0%}) -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
