"""Inference pipeline Hybrid versi proposal: Model 1 (LSTM) + Model 2 (Random Forest).

Bundle (`models/proposal_bundle.joblib`) dibuat oleh `train_hybrid_proposal.py` dan berisi semua yang
dibutuhkan untuk prediksi: bobot LSTM, model RF, batas IQR, parameter Z-Score, scaler fitur statis,
dan daftar kelas. Preprocessing di sini identik dengan `src/proposal_pipeline.py`.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from src.dataset_loader import beri_label_varietas_prototipe, filter_active_locations, load_config
from src.lstm_encoder import TemporalLSTMEncoder
from src.preprocessing import CATEGORICAL_MAPPINGS, encode_static_categories
from src.proposal_pipeline import _interpolate_per_location

BUNDLE_PATH = "models/proposal_bundle.joblib"
RF_VARIETAS_PATH = "models/rf_varietas_hybrid.joblib"
KEBUTUHAN_AIR_KODE = {"Sedikit": 0, "Sedang": 1, "Banyak": 2}
NAMA_BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
              "September", "Oktober", "November", "Desember"]
SATUAN = {
    "curah_hujan": "mm/bulan",
    "suhu_rata_rata": "°C",
    "suhu_maksimal": "°C",
    "suhu_minimum": "°C",
    "kelembaban_rata_rata": "%",
    "lama_penyinaran": "MJ/m²/hari",
    "ph_tanah": "",
    "karbon_organik": "%",
    "ketinggian_tempat": "mdpl",
}


def save_bundle(path: str, data: dict[str, Any], lstm: TemporalLSTMEncoder, rf, lstm_cfg: dict,
                group_df: pd.DataFrame, run_id: str, metrics: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    first_lstm = lstm.lstm1
    bundle = {
        "lstm_state": {k: v.detach().cpu() for k, v in lstm.state_dict().items()},
        "lstm_cfg": {
            "input_dim": first_lstm.input_size,
            "num_classes": len(data["climate_cols"]),
            "hidden_1": first_lstm.hidden_size,
            "hidden_2": lstm.lstm2.hidden_size,
            "embedding_dim": lstm.fc_embed.out_features,
            "dropout": float(lstm_cfg["dropout"]),
        },
        "rf": rf,
        "classes": list(data["label_encoder"].classes_),
        "climate_cols": list(data["climate_cols"]),
        "static_cols": list(data["static_cols"]),
        "input_cols": list(data["input_cols"]),
        "seq_len": int(data["preprocessing"]["sequence_length"]),
        "iqr_lower": data["iqr_lower"].to_numpy(dtype=float),
        "iqr_upper": data["iqr_upper"].to_numpy(dtype=float),
        "clim_mean": data["clim_scaler"].mean_.astype(float),
        "clim_std": data["clim_scaler"].scale_.astype(float),
        "stat_mean": data["stat_scaler"].mean_.astype(float),
        "stat_scale": data["stat_scaler"].scale_.astype(float),
        "active_locations": list(data["preprocessing"]["locations"]),
        "group_importance": group_df.to_dict("records"),
        "run_id": run_id,
        "metrics": metrics,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(bundle, path)


def _norm(text: str) -> str:
    return str(text).replace("-", "_").replace(" ", "_").lower()


class ProposalRecommender:
    """Muat bundle + data mentah, lalu hasilkan prakiraan iklim (LSTM) dan rekomendasi varietas (RF)."""

    def __init__(self, bundle_path: str = BUNDLE_PATH, cfg: dict[str, Any] | None = None) -> None:
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(
                f"{bundle_path} belum ada. Jalankan `python train_hybrid_proposal.py` terlebih dahulu."
            )
        self.b = joblib.load(bundle_path)
        cfg = cfg or load_config()
        raw_dir = cfg["data"]["raw_dir"]
        b = self.b

        climate = pd.read_csv(os.path.join(raw_dir, "aceh_utara_agroklimat_bulanan.csv"))
        soil = pd.read_csv(os.path.join(raw_dir, "aceh_utara_tanah_statis.csv"))
        cfg_active = {**cfg, "data": {**cfg["data"], "active_location_ids": b["active_locations"]}}
        climate, soil = filter_active_locations(climate, soil, cfg_active)
        climate = _interpolate_per_location(climate, b["climate_cols"])
        climate["varietas"] = beri_label_varietas_prototipe(climate, soil)["varietas"].to_numpy()
        climate["periode"] = climate["tahun"] * 12 + climate["bulan"] - 1
        self.climate = climate
        self.soil = soil.set_index("id_lokasi")

        var_path = os.path.join(raw_dir, "varietas_tanaman.csv")
        self.varietas = {}
        if os.path.exists(var_path):
            for _, r in pd.read_csv(var_path).iterrows():
                self.varietas[_norm(f"{r['komoditas']}_{r['nama_varietas']}")] = r.to_dict()

        self.model = TemporalLSTMEncoder(**b["lstm_cfg"])
        self.model.load_state_dict(b["lstm_state"])
        self.model.eval()
        self.rf = b["rf"]
        self.importance = {g["kelompok"]: g["importance_total"] for g in b["group_importance"]}

        # Model 2 versi many-to-many (RF kecocokan 25 varietas), dipakai bila tersedia & cocok dengan encoder LSTM
        self.rf_var = None
        if os.path.exists(RF_VARIETAS_PATH):
            pv = joblib.load(RF_VARIETAS_PATH)
            if pv.get("encoder_lstm", {}).get("run_id") == b["run_id"]:
                from buat_data_dummy_varietas import muat_varietas

                self.rf_var = pv
                self.var_tabel = muat_varietas(raw_dir).reset_index(drop=True)
                dummy_path = os.path.join(raw_dir, "dummy_varietas", "kecocokan_varietas_semua.csv")
                self.var_dummy = pd.read_csv(dummy_path) if os.path.exists(dummy_path) else None
            else:
                warnings.warn(
                    f"{RF_VARIETAS_PATH} dibuat dari encoder LSTM lain "
                    f"({pv.get('encoder_lstm', {}).get('run_id')} ≠ {b['run_id']}); Model 2 memakai RF klasifikasi lama. "
                    "Jalankan ulang buat_data_rf_varietas.py dan train_rf_varietas.py.",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------ metadata
    def meta(self) -> dict[str, Any]:
        b = self.b
        locs = []
        for loc_id in b["active_locations"]:
            c = self.climate[self.climate["id_lokasi"] == loc_id]
            first, last = int(c["periode"].min()), int(c["periode"].max())
            locs.append({
                "id": loc_id,
                "nama": self.soil.loc[loc_id, "nama_lokasi"],
                "periode_min": first + b["seq_len"],
                "periode_max": last + 1,  # bisa meramal 1 bulan setelah data terakhir
                "tanah": self._soil_dict(loc_id),
            })
        return {
            "lokasi": locs,
            "static_cols": b["static_cols"],
            "climate_cols": b["climate_cols"],
            "satuan": SATUAN,
            "seq_len": b["seq_len"],
            "run_id": b["run_id"],
            "metrics": b["metrics"],
            "created_at": b["created_at"],
            "model2": ({"nama": "RF kecocokan varietas (25 varietas)", "run_id": self.rf_var["run_id"],
                        "metrik_test": self.rf_var["metrik_test"]} if self.rf_var is not None
                       else {"nama": "RF klasifikasi (12 label prototipe)", "run_id": b["run_id"], "metrik_test": None}),
            "opsi": {"tekstur_tanah": ["Pasir", "Lanau", "Lempung"], "drainase": ["Baik", "Rusak"],
                     "risiko_bencana_hama": ["Rendah", "Sedang", "Tinggi"]},
        }

    def _soil_dict(self, loc_id: str) -> dict[str, Any]:
        row = self.soil.loc[loc_id]
        out = {}
        for col in self.b["static_cols"]:
            v = row[col]
            out[col] = v.item() if hasattr(v, "item") else v
        return out

    # ------------------------------------------------------------------ prediksi
    def _window(self, loc_id: str, periode: int) -> pd.DataFrame:
        L = self.b["seq_len"]
        c = self.climate[(self.climate["id_lokasi"] == loc_id)
                         & (self.climate["periode"] >= periode - L) & (self.climate["periode"] < periode)]
        c = c.sort_values("periode")
        if len(c) != L:
            raise ValueError(f"Butuh {L} bulan data iklim sebelum periode target, tersedia {len(c)}.")
        return c

    def _encode_window(self, win: pd.DataFrame) -> np.ndarray:
        b = self.b
        vals = win[b["climate_cols"]].to_numpy(dtype=float)
        vals = np.clip(vals, b["iqr_lower"], b["iqr_upper"])
        scaled = (vals - b["clim_mean"]) / b["clim_std"]
        feats = [scaled]
        if "bulan_sin" in b["input_cols"]:
            m = win["bulan"].to_numpy(dtype=float)
            feats += [np.sin(2 * np.pi * m / 12.0)[:, None], np.cos(2 * np.pi * m / 12.0)[:, None]]
        return np.hstack(feats).astype(np.float32)

    def _encode_static(self, soil: dict[str, Any]) -> np.ndarray:
        b = self.b
        df = pd.DataFrame([soil])
        for col in b["static_cols"]:
            if col not in ("tekstur_tanah", "drainase", "risiko_bencana_hama"):
                df[col] = pd.to_numeric(df[col], errors="raise")
        enc = encode_static_categories(df)[b["static_cols"]].to_numpy(dtype=float)
        return ((enc - b["stat_mean"]) / b["stat_scale"]).astype(np.float32)

    def _info_varietas(self, label: str) -> dict[str, Any]:
        info = self.varietas.get(_norm(label), {})
        return {k: info.get(k) for k in ("komoditas", "nama_varietas", "umur_panen_hari", "potensi_hasil_ton_ha",
                                          "kebutuhan_air", "ketahanan_hama_skala", "ketinggian_optimal_mdpl")}

    @torch.no_grad()
    def recommend(self, loc_id: str, tahun: int, bulan: int, soil_override: dict[str, Any] | None = None,
                  top_k: int = 3) -> dict[str, Any]:
        b = self.b
        if loc_id not in b["active_locations"]:
            raise ValueError(f"Lokasi {loc_id} tidak tersedia.")
        periode = int(tahun) * 12 + int(bulan) - 1
        win = self._window(loc_id, periode)

        # Model 1: LSTM -> prakiraan iklim bulan target + vektor temporal 32-dim
        x = torch.from_numpy(self._encode_window(win)[None])
        fc_scaled = self.model(x).numpy()[0]
        emb = self.model.encode(x).numpy()
        forecast = fc_scaled * b["clim_std"] + b["clim_mean"]
        forecast[0] = max(0.0, forecast[0])

        # Model 2: Random Forest -> probabilitas varietas
        soil = self._soil_dict(loc_id)
        if soil_override:
            soil.update({k: v for k, v in soil_override.items() if k in b["static_cols"] and v not in (None, "")})
        x_rf = np.hstack([emb, self._encode_static(soil)])
        proba = self.rf.predict_proba(x_rf)[0]
        names = [b["classes"][c] for c in self.rf.classes_]
        order = np.argsort(proba)[::-1][:top_k]

        actual_row = self.climate[(self.climate["id_lokasi"] == loc_id) & (self.climate["periode"] == periode)]
        actual = None
        label = None
        if len(actual_row):
            actual = {c: round(float(actual_row.iloc[0][c]), 2) for c in b["climate_cols"]}
            label = actual_row.iloc[0]["varietas"]

        lstm_key = next(k for k in self.importance if k.startswith("Fitur temporal"))
        static_rank = sorted(
            [(k, v) for k, v in self.importance.items() if k in b["static_cols"]], key=lambda kv: kv[1], reverse=True
        )
        alasan = [
            {"faktor": "Pola iklim 12 bulan terakhir (Model 1 LSTM)",
             "nilai": f"prakiraan hujan {forecast[0]:.0f} mm, suhu {forecast[1]:.1f} °C",
             "kontribusi": round(self.importance[lstm_key], 3)},
        ]
        for col, imp in static_rank[:3]:
            sat = SATUAN.get(col, "")
            alasan.append({"faktor": col, "nilai": f"{soil[col]} {sat}".strip(), "kontribusi": round(imp, 3)})

        bulan_mulai = (periode - b["seq_len"]) % 12
        hasil = {
            "lokasi": {"id": loc_id, "nama": self.soil.loc[loc_id, "nama_lokasi"]},
            "target": f"{NAMA_BULAN[int(bulan) - 1]} {int(tahun)}",
            "jendela_input": f"{NAMA_BULAN[bulan_mulai]} {(periode - b['seq_len']) // 12} – "
                             f"{NAMA_BULAN[(periode - 1) % 12]} {(periode - 1) // 12}",
            "riwayat": [
                {"periode": f"{NAMA_BULAN[int(r.bulan) - 1][:3]} {int(r.tahun)}",
                 **{c: round(float(getattr(r, c)), 2) for c in b["climate_cols"]}}
                for r in win.itertuples()
            ],
            "prakiraan_lstm": {c: round(float(v), 2) for c, v in zip(b["climate_cols"], forecast)},
            "aktual": actual,
            "rekomendasi": [
                {"peringkat": i + 1, "varietas": names[j], "skor": round(float(proba[j]) * 100, 1),
                 **self._info_varietas(names[j])}
                for i, j in enumerate(order)
            ],
            "label_prototipe": label,
            "label_di_top_k": (label in [names[j] for j in order]) if label else None,
            "alasan": alasan,
            "tanah_dipakai": soil,
            "model2": "RF klasifikasi (12 label prototipe)",
        }
        if self.rf_var is not None:
            hasil.update(self._rekomendasi_varietas(loc_id, int(tahun), int(bulan), emb[0], soil, forecast, top_k))
        return hasil

    def _rekomendasi_varietas(self, loc_id: str, tahun: int, bulan: int, emb: np.ndarray, soil: dict[str, Any],
                              forecast: np.ndarray, top_k: int) -> dict[str, Any]:
        """Model 2 many-to-many: nilai kecocokan ke-25 varietas (fitur sama dengan buat_data_rf_varietas.py)."""
        v = self.var_tabel
        X = pd.DataFrame(np.repeat(emb[None], len(v), axis=0), columns=[f"lstm_{i}" for i in range(len(emb))])
        for col in self.b["static_cols"]:
            val = soil[col]
            if col in CATEGORICAL_MAPPINGS and isinstance(val, str):
                val = CATEGORICAL_MAPPINGS[col].get(val, 0.0)
            X[col] = float(val)
        X["umur_panen_hari_num"] = v["umur_panen_hari_num"].to_numpy()
        X["potensi_hasil_num"] = v["potensi_hasil_num"].to_numpy()
        X["ketahanan_hama_skala"] = v["ketahanan_hama_skala"].to_numpy()
        X["kebutuhan_air_kode"] = v["kebutuhan_air"].map(KEBUTUHAN_AIR_KODE).to_numpy()
        X["ketinggian_maks_mdpl"] = v["ketinggian_maks_mdpl"].to_numpy()
        for k in ("padi", "jagung", "kedelai"):
            X[f"komoditas_{k}"] = (v["komoditas"].str.lower() == k).astype(int).to_numpy()
        X["bulan_tanam_sin"] = round(float(np.sin(2 * np.pi * bulan / 12.0)), 6)
        X["bulan_tanam_cos"] = round(float(np.cos(2 * np.pi * bulan / 12.0)), 6)

        skor = self.rf_var["model"].predict(X[self.rf_var["fitur"]])
        order = np.argsort(-skor, kind="stable")[:top_k]  # skor seri -> urut id varietas
        rek = []
        for i, j in enumerate(order):
            r = v.iloc[j]
            rek.append({
                "peringkat": i + 1, "varietas": f"{r['komoditas']} {r['nama_varietas']}", "skor": round(float(skor[j]), 1),
                "komoditas": r["komoditas"], "nama_varietas": r["nama_varietas"],
                "umur_panen_hari": r["umur_panen_hari"], "potensi_hasil_ton_ha": r["potensi_hasil_ton_ha"],
                "kebutuhan_air": r["kebutuhan_air"], "ketahanan_hama_skala": int(r["ketahanan_hama_skala"]),
                "ketinggian_optimal_mdpl": r["ketinggian_optimal_mdpl"],
                "estimasi_hasil_ton_ha": round(float(r["potensi_hasil_num"]) * float(skor[j]) / 100.0, 2),
            })

        imp = self.rf_var["importance_kelompok"]
        top = v.iloc[order[0]]
        alasan = [
            {"faktor": "Pola iklim 12 bulan terakhir (Model 1 LSTM)",
             "nilai": f"prakiraan hujan {forecast[0]:.0f} mm, suhu {forecast[1]:.1f} °C", "kontribusi": round(imp["LSTM (32)"], 3)},
            {"faktor": "Karakteristik varietas",
             "nilai": f"{top['nama_varietas']}: kebutuhan air {top['kebutuhan_air']}, umur {top['umur_panen_hari']} hari",
             "kontribusi": round(imp["Varietas (8)"], 3)},
            {"faktor": "Musim tanam", "nilai": NAMA_BULAN[bulan - 1], "kontribusi": round(imp["Musim (2)"], 3)},
            {"faktor": "Kondisi tanah", "nilai": f"pH {soil['ph_tanah']}, drainase {soil['drainase']}, "
                                                 f"{soil['ketinggian_tempat']} mdpl", "kontribusi": round(imp["Tanah (9)"], 3)},
        ]

        pembanding = None
        if self.var_dummy is not None:
            g = self.var_dummy[(self.var_dummy["id_lokasi"] == loc_id) & (self.var_dummy["tahun"] == tahun)
                               & (self.var_dummy["bulan"] == bulan)]
            if len(g):
                g = g.sort_values("skor_kecocokan", ascending=False).head(top_k)
                pembanding = [f"{r.komoditas} {r.nama_varietas} ({r.skor_kecocokan:.1f})" for r in g.itertuples()]

        return {
            "rekomendasi": rek,
            "alasan": alasan,
            "label_prototipe": None,
            "label_di_top_k": None,
            "pembanding_dummy": pembanding,
            "model2": "RF kecocokan varietas (menilai 25 varietas)",
        }
