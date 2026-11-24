"""Fitur cuaca 'biasa' (tanpa LSTM) untuk Random Forest: rata-rata iklim lampau dan normal iklim masa tanam.

Dipakai bersama oleh train_rf_hasil_cuaca.py dan aplikasi web agar perhitungannya identik.
"""

from __future__ import annotations

import pandas as pd

IKLIM = ["curah_hujan", "suhu_rata_rata", "suhu_maksimal", "suhu_minimum", "kelembaban_rata_rata", "lama_penyinaran"]
KUNCI = ["id_lokasi", "tahun", "bulan"]


def fitur_cuaca(climate: pd.DataFrame, tahun_akhir_normal: int = 2024, n_bulan_tanam: int = 4,
                tambah_bulan_depan: bool = False) -> pd.DataFrame:
    """Hitung fitur cuaca untuk setiap (lokasi, bulan tanam) yang bisa dibentuk dari data bulanan.

    rata12_<var>        rata-rata 12 bulan sebelum bulan tanam (t-12 .. t-1)
    rata3_<var>         rata-rata 3 bulan sebelum bulan tanam (t-3 .. t-1)
    normal_hujan_tanam  rata-rata hujan bulanan pada bulan kalender t .. t+n-1, dari tahun <= tahun_akhir_normal
    normal_suhu_tanam   idem untuk suhu rata-rata; tahun bulan tanam itu sendiri DIKECUALIKAN (tidak bocor)
    """
    c = climate[KUNCI + IKLIM].sort_values(KUNCI).reset_index(drop=True)
    out = []
    for loc, g in c.groupby("id_lokasi"):
        g = g.reset_index(drop=True)
        if tambah_bulan_depan:  # baris bulan tanam berikutnya setelah data terakhir (fitur hanya dari bulan lampau)
            p = int(g["tahun"].iloc[-1]) * 12 + int(g["bulan"].iloc[-1])
            g = pd.concat([g, pd.DataFrame([{"id_lokasi": loc, "tahun": p // 12, "bulan": p % 12 + 1}])], ignore_index=True)
        r12 = g[IKLIM].rolling(12).mean().shift(1).add_prefix("rata12_")
        r3 = g[IKLIM].rolling(3).mean().shift(1).add_prefix("rata3_")
        f = pd.concat([g[KUNCI], r12, r3], axis=1)

        hist = g[g["tahun"] <= tahun_akhir_normal]
        tabel_hujan = hist.pivot_table(index="tahun", columns="bulan", values="curah_hujan")
        tabel_suhu = hist.pivot_table(index="tahun", columns="bulan", values="suhu_rata_rata")
        nh, ns = [], []
        for th, bl in zip(f["tahun"], f["bulan"]):
            bulan_tanam = [((bl - 1 + k) % 12) + 1 for k in range(n_bulan_tanam)]
            tahun_lain = [y for y in tabel_hujan.index if y != th]
            nh.append(tabel_hujan.loc[tahun_lain, bulan_tanam].to_numpy().mean())
            ns.append(tabel_suhu.loc[tahun_lain, bulan_tanam].to_numpy().mean())
        f["normal_hujan_tanam"], f["normal_suhu_tanam"] = nh, ns
        out.append(f)
    return pd.concat(out, ignore_index=True)


SET_FITUR = {
    "C1 rata-rata 12 bulan": [f"rata12_{v}" for v in IKLIM],
    "C2 + rata-rata 3 bulan": [f"rata12_{v}" for v in IKLIM] + [f"rata3_{v}" for v in IKLIM],
    "C3 + normal iklim masa tanam": [f"rata12_{v}" for v in IKLIM] + [f"rata3_{v}" for v in IKLIM]
    + ["normal_hujan_tanam", "normal_suhu_tanam"],
}
