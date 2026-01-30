from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import pandas as pd

from .utils import read_table


def norm_form(x: str, lowercase: bool = True, nfc: bool = True) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    if nfc:
        s = unicodedata.normalize("NFC", s)
    if lowercase:
        s = s.lower()
    s = " ".join(s.split())
    return s


def dedup_preserve(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def load_concept_map(path: str | Path) -> Dict[str, List[str]]:
    """
    JSON:
      { "HOUSE": ["house","casa"], "DOG": ["dog","perro"] }

    CSV/TSV:
      concept,form
      HOUSE,house
      HOUSE,casa
      DOG,dog
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"inputs not found: {p}")

    suf = p.suffix.lower()
    if suf == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("JSON inputs must be a dict: concept -> list[str]")
        out: Dict[str, List[str]] = {}
        for k, v in obj.items():
            if isinstance(v, str):
                out[str(k)] = [v]
            elif isinstance(v, list):
                out[str(k)] = [str(x) for x in v if x is not None]
            else:
                raise ValueError(f"Bad value type for concept {k}: {type(v)}")
        return out

    if suf in {".csv", ".tsv"}:
        df = read_table(p)
        cols = {c.lower(): c for c in df.columns}
        if "concept" not in cols or "form" not in cols:
            raise ValueError("CSV/TSV inputs must have columns: concept, form")
        c_col, f_col = cols["concept"], cols["form"]

        concept_to_forms: Dict[str, List[str]] = {}
        for _, row in df.iterrows():
            concept = str(row[c_col])
            form = row[f_col]
            if pd.isna(form):
                continue
            form = str(form)
            concept_to_forms.setdefault(concept, []).append(form)

        if not concept_to_forms:
            raise ValueError("No valid rows in inputs.")
        return concept_to_forms

    raise ValueError("Unsupported inputs format. Use .json, .csv, or .tsv.")


def load_target_groups(path: Optional[str | Path]) -> Optional[Dict[str, str]]:
    """
    CSV/TSV:
      concept,group
      HOUSE,A
      DOG,B
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"target_groups not found: {p}")
    if p.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("target_groups must be .csv or .tsv")
    df = read_table(p)
    cols = {c.lower(): c for c in df.columns}
    if "concept" not in cols or "group" not in cols:
        raise ValueError("target_groups file must have columns: concept, group")
    c_col, g_col = cols["concept"], cols["group"]
    return dict(zip(df[c_col].astype(str), df[g_col].astype(str)))


LexiconFormat = Literal["wide", "long", "pairs"]


@dataclass
class PrepareLexiconConfig:
    input_path: str
    fmt: LexiconFormat
    sheet: Optional[str] = None

    # wide
    concept_col: Optional[str] = None
    lang_cols: Optional[List[str]] = None

    # long
    long_concept_col: str = "concept"
    long_form_col: str = "form"

    # pairs
    src_lang: str = "src"
    tgt_lang: str = "tgt"
    src_col: str = "src_form"
    tgt_col: str = "tgt_form"

    # normalization
    lowercase: bool = True
    nfc: bool = True


def _from_wide(df: pd.DataFrame, cfg: PrepareLexiconConfig) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    cols_lower = {c.lower(): c for c in df.columns}

    ccol = None
    if cfg.concept_col is not None:
        if cfg.concept_col in df.columns:
            ccol = cfg.concept_col
        elif cfg.concept_col.lower() in cols_lower:
            ccol = cols_lower[cfg.concept_col.lower()]
        else:
            raise ValueError(f"concept_col '{cfg.concept_col}' not found in columns: {list(df.columns)}")
    else:
        if "concept" in cols_lower:
            ccol = cols_lower["concept"]

    if cfg.lang_cols is None or len(cfg.lang_cols) == 0:
        lang_cols_use = list(df.columns) if ccol is None else [c for c in df.columns if c != ccol]
    else:
        lang_cols_use = []
        for lc in cfg.lang_cols:
            if lc in df.columns:
                lang_cols_use.append(lc)
            elif lc.lower() in cols_lower:
                lang_cols_use.append(cols_lower[lc.lower()])
            else:
                raise ValueError(f"lang column '{lc}' not found in columns: {list(df.columns)}")

    if ccol is None:
        concept_ids = [f"C{i:05d}" for i in range(len(df))]
    else:
        concept_ids = [str(x) for x in df[ccol].tolist()]

    concept_to_forms: Dict[str, List[str]] = {}
    rows = []
    for idx, concept in enumerate(concept_ids):
        forms = []
        for lc in lang_cols_use:
            v = df.iloc[idx][lc]
            if pd.isna(v):
                continue
            s = norm_form(str(v), lowercase=cfg.lowercase, nfc=cfg.nfc)
            if s == "":
                continue
            forms.append(s)

        forms = dedup_preserve(forms)
        if len(forms) == 0:
            continue

        concept_to_forms[concept] = forms
        for f in forms:
            rows.append({"concept": concept, "form": f})

    out_df = pd.DataFrame(rows, columns=["concept", "form"])
    return out_df, concept_to_forms


def _from_long(df: pd.DataFrame, cfg: PrepareLexiconConfig) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    cols_lower = {c.lower(): c for c in df.columns}

    def resolve(colname: str) -> str:
        if colname in df.columns:
            return colname
        if colname.lower() in cols_lower:
            return cols_lower[colname.lower()]
        raise ValueError(f"Column '{colname}' not found in {list(df.columns)}")

    ccol = resolve(cfg.long_concept_col)
    fcol = resolve(cfg.long_form_col)

    rows = []
    concept_to_forms: Dict[str, List[str]] = {}

    for _, r in df.iterrows():
        concept = str(r[ccol])
        v = r[fcol]
        if pd.isna(v):
            continue
        form = norm_form(str(v), lowercase=cfg.lowercase, nfc=cfg.nfc)
        if form == "":
            continue
        rows.append({"concept": concept, "form": form})
        concept_to_forms.setdefault(concept, []).append(form)

    for k in list(concept_to_forms.keys()):
        concept_to_forms[k] = dedup_preserve(concept_to_forms[k])

    out_df = pd.DataFrame(rows, columns=["concept", "form"]).drop_duplicates()
    return out_df, concept_to_forms


def _from_pairs(df: pd.DataFrame, cfg: PrepareLexiconConfig) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    cols_lower = {c.lower(): c for c in df.columns}

    def resolve(colname: str) -> str:
        if colname in df.columns:
            return colname
        if colname.lower() in cols_lower:
            return cols_lower[colname.lower()]
        raise ValueError(f"Column '{colname}' not found in {list(df.columns)}")

    scol = resolve(cfg.src_col)
    tcol = resolve(cfg.tgt_col)

    rows = []
    concept_to_forms: Dict[str, List[str]] = {}

    for i, r in df.iterrows():
        a = r[scol]
        b = r[tcol]
        if pd.isna(a) and pd.isna(b):
            continue
        fa = "" if pd.isna(a) else norm_form(str(a), lowercase=cfg.lowercase, nfc=cfg.nfc)
        fb = "" if pd.isna(b) else norm_form(str(b), lowercase=cfg.lowercase, nfc=cfg.nfc)
        forms = [x for x in [fa, fb] if x != ""]
        forms = dedup_preserve(forms)
        if len(forms) == 0:
            continue

        concept = f"P{i:05d}_{cfg.src_lang}-{cfg.tgt_lang}"
        concept_to_forms[concept] = forms
        for f in forms:
            rows.append({"concept": concept, "form": f})

    out_df = pd.DataFrame(rows, columns=["concept", "form"])
    return out_df, concept_to_forms


def prepare_lexicon(cfg: PrepareLexiconConfig) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    df = read_table(cfg.input_path, sheet=cfg.sheet)

    if cfg.fmt == "wide":
        concepts_df, concept_to_forms = _from_wide(df, cfg)
    elif cfg.fmt == "long":
        concepts_df, concept_to_forms = _from_long(df, cfg)
    elif cfg.fmt == "pairs":
        concepts_df, concept_to_forms = _from_pairs(df, cfg)
    else:
        raise ValueError("fmt must be one of: wide|long|pairs")

    concepts_df["concept"] = concepts_df["concept"].astype(str)
    concepts_df["form"] = concepts_df["form"].astype(str)
    concepts_df = concepts_df[concepts_df["form"].str.len() > 0].drop_duplicates()

    return concepts_df, concept_to_forms
