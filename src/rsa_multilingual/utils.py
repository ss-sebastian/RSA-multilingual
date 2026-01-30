from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_table(path: str | Path, sheet: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".csv", ".tsv"}:
        sep = "\t" if suf == ".tsv" else ","
        return pd.read_csv(p, sep=sep)
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(p, sheet_name=sheet if sheet is not None else 0)
    raise ValueError("Unsupported file type. Use csv/tsv/xlsx/xls.")
