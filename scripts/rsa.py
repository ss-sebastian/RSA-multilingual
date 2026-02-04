#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/rsa.py

Two modes:
1) long: inputs is concept,form (your existing pipeline)
2) wide: inputs is concept + language columns (en/es/zh ...). We compute:
   - per-language RDMs (concept x concept) for each layer
   - second-order RSA across languages (RDM similarity) for each layer
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure repo_root/src is importable when running as a script
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rsa_multilingual.pipeline import run_rsa  # type: ignore


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_rdm_csv(path: Path) -> np.ndarray:
    # assumes square matrix saved as csv with header+index or pure matrix.
    df = pd.read_csv(path, index_col=0)
    mat = df.values
    # if index_col=0 was wrong (pure matrix), fallback:
    if mat.shape[0] != mat.shape[1]:
        df = pd.read_csv(path, header=None)
        mat = df.values
    return mat


def _upper_tri(mat: np.ndarray) -> np.ndarray:
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"RDM must be square, got {mat.shape}")
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]


def _corr(a: np.ndarray, b: np.ndarray, method: str = "spearman") -> float:
    # minimal dependency: implement spearman via rank -> pearson
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be pearson or spearman")
    a = a.astype(float)
    b = b.astype(float)

    if method == "spearman":
        a = pd.Series(a).rank(method="average").to_numpy()
        b = pd.Series(b).rank(method="average").to_numpy()

    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _collect_layer_rdms(lang_dir: Path) -> Dict[str, Path]:
    """
    Find per-layer RDM csvs in a language output directory.
    We assume your pipeline writes files like rdm_layer-3.csv (or similar).
    """
    rdms = {}
    for p in sorted(lang_dir.glob("**/rdm_layer*.csv")):
        # keep the stem as key, e.g., "rdm_layer-3"
        rdms[p.stem] = p
    if not rdms:
        raise FileNotFoundError(f"No rdm_layer*.csv found under {lang_dir}")
    return rdms


def _write_rdm_similarity(
    out_dir: Path,
    by_lang_dirs: Dict[str, Path],
    corr_method: str = "spearman",
) -> None:
    """
    For each layer available in all languages, compute lang×lang correlation
    between vectorized upper triangle of each RDM.
    """
    out_sim = out_dir / "rdm_similarity"
    out_sim.mkdir(parents=True, exist_ok=True)

    # collect rdms per lang
    lang_rdms: Dict[str, Dict[str, Path]] = {lg: _collect_layer_rdms(d) for lg, d in by_lang_dirs.items()}
    # intersection of layer keys across languages
    common_layers = set.intersection(*(set(m.keys()) for m in lang_rdms.values()))
    if not common_layers:
        raise RuntimeError("No common layers found across languages (rdm_layer*.csv filenames mismatch).")

    langs = list(by_lang_dirs.keys())
    records = []

    for layer_key in sorted(common_layers):
        vecs = {}
        for lg in langs:
            mat = _load_rdm_csv(lang_rdms[lg][layer_key])
            vecs[lg] = _upper_tri(mat)

        sim = np.zeros((len(langs), len(langs)), dtype=float)
        for i, li in enumerate(langs):
            for j, lj in enumerate(langs):
                sim[i, j] = _corr(vecs[li], vecs[lj], method=corr_method)

        df = pd.DataFrame(sim, index=langs, columns=langs)
        df.to_csv(out_sim / f"rdm_sim_{layer_key}_{corr_method}.csv")

        # long-form too (handy for later plots)
        for i, li in enumerate(langs):
            for j, lj in enumerate(langs):
                records.append(
                    {"layer": layer_key.replace("rdm_", ""), "lang_i": li, "lang_j": lj, "rdm_corr": sim[i, j], "method": corr_method}
                )

    pd.DataFrame(records).to_csv(out_sim / f"rdm_sim_long_{corr_method}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--inputs", type=str, required=True, help="CSV path. long: concept,form ; wide: concept + lang cols")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--metric", type=str, default="correlation")
    ap.add_argument("--word_pooling", type=str, default="mean")
    ap.add_argument("--concept_pooling", type=str, default="mean")

    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=64)

    ap.add_argument("--out_dir", type=str, required=True)

    # NEW
    ap.add_argument("--input_format", type=str, choices=["long", "wide"], default="long")
    ap.add_argument("--concept_col", type=str, default="concept")
    ap.add_argument("--lang_cols", type=str, nargs="*", default=None, help="For wide input, e.g. --lang_cols en es zh")
    ap.add_argument("--rdm_corr", type=str, choices=["spearman", "pearson"], default="spearman")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_format == "long":
        # Existing behavior
        run_rsa(
            model_name=args.model_name,
            inputs=Path(args.inputs),
            layers=args.layers,
            metric=args.metric,
            word_pooling=args.word_pooling,
            concept_pooling=args.concept_pooling,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            out_dir=out_dir,
        )
        return

    # wide mode
    if not args.lang_cols:
        raise ValueError("wide input requires --lang_cols, e.g. --lang_cols en es zh")

    wide = pd.read_csv(args.inputs)
    if args.concept_col not in wide.columns:
        raise ValueError(f"Missing concept_col={args.concept_col} in wide CSV columns: {list(wide.columns)}")

    for lg in args.lang_cols:
        if lg not in wide.columns:
            raise ValueError(f"Missing lang column '{lg}' in wide CSV columns: {list(wide.columns)}")

    by_lang_base = out_dir / "by_lang"
    by_lang_base.mkdir(parents=True, exist_ok=True)

    by_lang_dirs: Dict[str, Path] = {}

    # 1) per-language run (each language becomes a long concept,form)
    for lg in args.lang_cols:
        tmp = pd.DataFrame(
            {
                "concept": wide[args.concept_col].astype(str),
                "form": wide[lg].astype(str),
            }
        )
        # drop empty/missing forms (important for partially filled lexicons)
        tmp = tmp.replace({"": np.nan}).dropna()

        tmp_path = by_lang_base / f"_tmp_inputs_{lg}.csv"
        _write_df(tmp, tmp_path)

        lg_out = by_lang_base / lg
        lg_out.mkdir(parents=True, exist_ok=True)

        run_rsa(
            model_name=args.model_name,
            inputs=tmp_path,
            layers=args.layers,
            metric=args.metric,
            word_pooling=args.word_pooling,
            concept_pooling="none",  # IMPORTANT: per language we already have 1 form per concept
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            out_dir=lg_out,
        )
        by_lang_dirs[lg] = lg_out

    # 2) second-order RSA across languages (compare RDM geometry)
    _write_rdm_similarity(out_dir=out_dir, by_lang_dirs=by_lang_dirs, corr_method=args.rdm_corr)


if __name__ == "__main__":
    main()
