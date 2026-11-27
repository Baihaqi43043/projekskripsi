"""Utilitas kecil untuk script analisis: tampilkan ke layar sekaligus simpan ke results/eksperimen/<kode>/."""

from __future__ import annotations

import contextlib
import io
import os
import sys
from datetime import datetime


@contextlib.contextmanager
def simpan_keluaran(kode: str, nama_file: str = "hasil_analisis.txt"):
    """Semua print di dalam blok ditampilkan dan disimpan ke results/eksperimen/<kode>/<nama_file>."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    buf = io.StringIO()

    class _Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    with contextlib.redirect_stdout(_Tee()):
        print(f"# {kode} — dibuat {datetime.now():%Y-%m-%d %H:%M}\n")
        yield
    folder = os.path.join(os.environ.get("HASIL_DIR", "results"), "eksperimen", kode)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, nama_file), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    sys.__stdout__.write(f"\n[disimpan] {os.path.join(folder, nama_file)}\n")
