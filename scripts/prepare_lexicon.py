#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/prepare_lexicon.py

Thin CLI wrapper:
- calls rsa_multilingual.lexicon.prepare_lexicon(...)
- writes:
  - concepts.csv (concept,form)
  - concepts.json (concept -> list[forms])
  - target_groups.csv (optional; if --groups provided)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure repo_root/src is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rsa_multilingual.lexicon import PrepareLexiconConfig, prepare_lexicon  # noqa: E402
from rsa_multilingual.utils import read_table  # noqa: E402


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _resolve_col(df: pd.DataFrame, name: str) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    if name in df.columns:
        return name
    if name.lower() in cols_lower:
        return cols_lower[name.lower()]
    raise ValueError(f"Column '{name}' not found in {list(df.columns)}")


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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PrepareLexiconConfig(
        input_path=args.input,
        fmt=args.format,
        sheet=args.sheet,
        concept_col=args.concept_col,
        lang_cols=args.lang_cols,
        long_concept_col=args.long_concept_col,
        long_form_col=args.long_form_col,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        src_col=args.src_col,
        tgt_col=args.tgt_col,
        lowercase=not args.no_lowercase,
        nfc=not args.no_nfc,
    )

    concepts_df, concept_to_forms = prepare_lexicon(cfg)

    # Save concepts.csv / concepts.json
    _write_csv(concepts_df, out_dir / "concepts.csv")
    (out_dir / "concepts.json").write_text(
        json.dumps(concept_to_forms, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Optional groups -> target_groups.csv (filtered to concepts that exist)
    if args.groups is not None:
        gdf = read_table(args.groups, sheet=args.groups_sheet)
        ccol = _resolve_col(gdf, args.groups_concept_col)
        gcol = _resolve_col(gdf, args.groups_group_col)

        tg = gdf[[ccol, gcol]].copy()
        tg.columns = ["concept", "group"]
        tg["concept"] = tg["concept"].astype(str)
        tg["group"] = tg["group"].astype(str)

        valid = set(concepts_df["concept"].astype(str).unique().tolist())
        tg = tg[tg["concept"].isin(valid)].drop_duplicates()

        _write_csv(tg, out_dir / "target_groups.csv")

    n_concepts = concepts_df["concept"].nunique()
    n_forms = len(concepts_df)
    print(f"Saved to: {out_dir.resolve()}")
    print(f"Concepts: {n_concepts} | Forms (rows): {n_forms}")
    print("Files: concepts.csv, concepts.json" + (", target_groups.csv" if args.groups else ""))


if __name__ == "__main__":
    main()
