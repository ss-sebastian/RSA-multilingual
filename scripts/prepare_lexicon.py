#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare.py

Prepare multilingual lexicon files for RSA pipeline.

Outputs (default in out_dir):
- concepts.csv        columns: concept,form
- concepts.json       mapping: concept -> list[forms]
- target_groups.csv   optional columns: concept,group (if provided / derivable)

Supported input formats:
1) wide  (recommended)
   CSV/TSV/Excel where each row is one concept, and each language is a column.

   Example columns:
     concept,en,es,zh
     DOG,dog,perro,狗
     HOUSE,house,casa,房子

   You can also omit "concept" column and auto-generate concept ids:
     en,es,zh
     dog,perro,狗

2) long
   CSV/TSV/Excel with columns:
     concept,lang,form
   Example:
     DOG,en,dog
     DOG,es,perro
     DOG,zh,狗

3) pairs
   A bilingual dictionary with columns:
     src_lang,tgt_lang,src_form,tgt_form
   or
     src_form,tgt_form   (langs can be set via CLI)
   Each row becomes one concept id (auto), forms include both sides.

Optional: grouping
- Provide a file with columns: concept,group via --groups
  This becomes target_groups.csv for categorical target RDM.

Normalization:
- Unicode NFC
- strip whitespace
- optional lowercase (default True)
- optional drop empty, de-duplicate forms

Usage examples:

A) wide CSV:
python prepare.py \
  --input data/lexicon_wide.csv \
  --format wide \
  --concept_col concept \
  --lang_cols en es zh \
  --out_dir artifacts/lexicon

B) wide Excel:
python prepare.py \
  --input data/lexicon.xlsx \
  --format wide \
  --sheet Sheet1 \
  --lang_cols en es zh \
  --out_dir artifacts/lexicon

C) long:
python prepare.py \
  --input data/lexicon_long.tsv \
  --format long \
  --out_dir artifacts/lexicon

D) pairs:
python prepare.py \
  --input data/dict_pairs.csv \
  --format pairs \
  --src_lang en --tgt_lang es \
  --src_col src_form --tgt_col tgt_form \
  --out_dir artifacts/lexicon

Then feed rsa.py with:
  --inputs artifacts/lexicon/concepts.csv
  --target_groups artifacts/lexicon/target_groups.csv   (if you created it)
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Read helpers
# ----------------------------
def read_table(path: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".csv", ".tsv"}:
        sep = "\t" if suf == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    raise ValueError("Unsupported input file type. Use csv/tsv/xlsx/xls.")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ----------------------------
# Normalization
# ----------------------------
def norm_form(x: str, lowercase: bool = True, nfc: bool = True) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    if nfc:
        s = unicodedata.normalize("NFC", s)
    if lowercase:
        s = s.lower()
    # collapse internal whitespace
    s = " ".join(s.split())
    return s


def dedup_preserve(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ----------------------------
# Converters
# ----------------------------
def from_wide(
    df: pd.DataFrame,
    concept_col: Optional[str],
    lang_cols: Optional[List[str]],
    lowercase: bool,
    nfc: bool,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    wide -> long concept,form
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # Determine concept column
    if concept_col is not None:
        if concept_col not in df.columns:
            # allow case-insensitive
            if concept_col.lower() in cols_lower:
                concept_col = cols_lower[concept_col.lower()]
            else:
                raise ValueError(f"concept_col '{concept_col}' not found in columns: {list(df.columns)}")
        ccol = concept_col
    else:
        # try default "concept"
        if "concept" in cols_lower:
            ccol = cols_lower["concept"]
        else:
            ccol = None

    # Determine language columns
    if lang_cols is None or len(lang_cols) == 0:
        # default: all columns except concept_col
        if ccol is None:
            lang_cols_use = list(df.columns)
        else:
            lang_cols_use = [c for c in df.columns if c != ccol]
    else:
        # allow case-insensitive matching
        lang_cols_use = []
        for lc in lang_cols:
            if lc in df.columns:
                lang_cols_use.append(lc)
            elif lc.lower() in cols_lower:
                lang_cols_use.append(cols_lower[lc.lower()])
            else:
                raise ValueError(f"lang column '{lc}' not found in columns: {list(df.columns)}")

    # Build concepts
    concept_ids: List[str] = []
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
            s = norm_form(str(v), lowercase=lowercase, nfc=nfc)
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


def from_long(
    df: pd.DataFrame,
    concept_col: str,
    form_col: str,
    lowercase: bool,
    nfc: bool,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    cols_lower = {c.lower(): c for c in df.columns}

    # map columns case-insensitively
    def resolve(colname: str) -> str:
        if colname in df.columns:
            return colname
        if colname.lower() in cols_lower:
            return cols_lower[colname.lower()]
        raise ValueError(f"Column '{colname}' not found in {list(df.columns)}")

    ccol = resolve(concept_col)
    fcol = resolve(form_col)

    rows = []
    concept_to_forms: Dict[str, List[str]] = {}

    for _, r in df.iterrows():
        concept = str(r[ccol])
        v = r[fcol]
        if pd.isna(v):
            continue
        form = norm_form(str(v), lowercase=lowercase, nfc=nfc)
        if form == "":
            continue
        rows.append({"concept": concept, "form": form})
        concept_to_forms.setdefault(concept, []).append(form)

    # de-dup forms per concept (preserve order)
    for k in list(concept_to_forms.keys()):
        concept_to_forms[k] = dedup_preserve(concept_to_forms[k])

    out_df = pd.DataFrame(rows, columns=["concept", "form"]).drop_duplicates()
    return out_df, concept_to_forms


def from_pairs(
    df: pd.DataFrame,
    src_col: str,
    tgt_col: str,
    src_lang: str,
    tgt_lang: str,
    lowercase: bool,
    nfc: bool,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    cols_lower = {c.lower(): c for c in df.columns}

    def resolve(colname: str) -> str:
        if colname in df.columns:
            return colname
        if colname.lower() in cols_lower:
            return cols_lower[colname.lower()]
        raise ValueError(f"Column '{colname}' not found in {list(df.columns)}")

    scol = resolve(src_col)
    tcol = resolve(tgt_col)

    rows = []
    concept_to_forms: Dict[str, List[str]] = {}

    for i, r in df.iterrows():
        a = r[scol]
        b = r[tcol]
        if pd.isna(a) and pd.isna(b):
            continue
        fa = "" if pd.isna(a) else norm_form(str(a), lowercase=lowercase, nfc=nfc)
        fb = "" if pd.isna(b) else norm_form(str(b), lowercase=lowercase, nfc=nfc)
        forms = [x for x in [fa, fb] if x != ""]
        forms = dedup_preserve(forms)
        if len(forms) == 0:
            continue

        concept = f"P{i:05d}_{src_lang}-{tgt_lang}"
        concept_to_forms[concept] = forms
        for f in forms:
            rows.append({"concept": concept, "form": f})

    out_df = pd.DataFrame(rows, columns=["concept", "form"])
    return out_df, concept_to_forms


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare lexicon for RSA pipeline.")

    ap.add_argument("--input", required=True, type=str, help="csv/tsv/xlsx/xls")
    ap.add_argument("--format", required=True, type=str, choices=["wide", "long", "pairs"])
    ap.add_argument("--sheet", default=None, type=str, help="Excel sheet name (optional)")

    # wide
    ap.add_argument("--concept_col", default=None, type=str, help="wide: concept column (optional)")
    ap.add_argument("--lang_cols", nargs="*", default=None, help="wide: language columns (optional)")

    # long
    ap.add_argument("--long_concept_col", default="concept", type=str, help="long: concept column")
    ap.add_argument("--long_form_col", default="form", type=str, help="long: form column")

    # pairs
    ap.add_argument("--src_lang", default="src", type=str, help="pairs: source language tag")
    ap.add_argument("--tgt_lang", default="tgt", type=str, help="pairs: target language tag")
    ap.add_argument("--src_col", default="src_form", type=str, help="pairs: source form column")
    ap.add_argument("--tgt_col", default="tgt_form", type=str, help="pairs: target form column")

    # grouping
    ap.add_argument("--groups", default=None, type=str, help="optional concept,group file (csv/tsv/xlsx)")
    ap.add_argument("--groups_sheet", default=None, type=str, help="optional groups excel sheet")
    ap.add_argument("--groups_concept_col", default="concept", type=str)
    ap.add_argument("--groups_group_col", default="group", type=str)

    # normalization
    ap.add_argument("--no_lowercase", action="store_true", help="do NOT lowercase forms")
    ap.add_argument("--no_nfc", action="store_true", help="do NOT apply Unicode NFC normalization")

    # output
    ap.add_argument("--out_dir", default="artifacts/lexicon", type=str)

    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lowercase = not args.no_lowercase
    nfc = not args.no_nfc

    df = read_table(in_path, sheet=args.sheet)

    if args.format == "wide":
        concepts_df, concept_to_forms = from_wide(
            df=df,
            concept_col=args.concept_col,
            lang_cols=args.lang_cols,
            lowercase=lowercase,
            nfc=nfc,
        )
    elif args.format == "long":
        concepts_df, concept_to_forms = from_long(
            df=df,
            concept_col=args.long_concept_col,
            form_col=args.long_form_col,
            lowercase=lowercase,
            nfc=nfc,
        )
    else:  # pairs
        concepts_df, concept_to_forms = from_pairs(
            df=df,
            src_col=args.src_col,
            tgt_col=args.tgt_col,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            lowercase=lowercase,
            nfc=nfc,
        )

    # Final cleaning: drop duplicates, drop empties
    concepts_df["concept"] = concepts_df["concept"].astype(str)
    concepts_df["form"] = concepts_df["form"].astype(str)
    concepts_df = concepts_df[concepts_df["form"].str.len() > 0].drop_duplicates()

    # Save outputs
    write_csv(concepts_df, out_dir / "concepts.csv")
    (out_dir / "concepts.json").write_text(
        json.dumps(concept_to_forms, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Optional groups
    if args.groups is not None:
        gpath = Path(args.groups)
        gdf = read_table(gpath, sheet=args.groups_sheet)
        cols_lower = {c.lower(): c for c in gdf.columns}

        def resolve(colname: str) -> str:
            if colname in gdf.columns:
                return colname
            if colname.lower() in cols_lower:
                return cols_lower[colname.lower()]
            raise ValueError(f"Groups column '{colname}' not found in {list(gdf.columns)}")

        ccol = resolve(args.groups_concept_col)
        gcol = resolve(args.groups_group_col)

        tg = gdf[[ccol, gcol]].copy()
        tg.columns = ["concept", "group"]
        tg["concept"] = tg["concept"].astype(str)
        tg["group"] = tg["group"].astype(str)

        # Keep only concepts that exist in concepts.csv (avoid stray rows)
        valid = set(concepts_df["concept"].unique().tolist())
        tg = tg[tg["concept"].isin(valid)].drop_duplicates()

        write_csv(tg, out_dir / "target_groups.csv")

    # Report
    n_concepts = concepts_df["concept"].nunique()
    n_forms = len(concepts_df)
    print(f"Saved to: {out_dir.resolve()}")
    print(f"Concepts: {n_concepts} | Forms (rows): {n_forms}")
    print("Files: concepts.csv, concepts.json" + (", target_groups.csv" if args.groups else ""))


if __name__ == "__main__":
    main()
