#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/rsa.py

Two modes:
1) long: inputs is concept,form (your existing pipeline)
2) wide: inputs is concept + language columns (en/es/zh ...). We compute:
   - per-language RDMs (concept x concept) for each layer
   - second-order RSA across languages (RDM similarity) for each layer

Important:
- rsa_multilingual.pipeline.run_rsa DOES NOT take out_dir. It returns RSAResult.
- This script is responsible for saving outputs to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo_root/src is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rsa_multilingual.pipeline import run_rsa  # type: ignore


# -------------------------
# IO helpers
# -------------------------
def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _jsonify(obj):
    """Convert common non-JSON types into JSON-serializable objects (safe, compact)."""
    import numpy as _np
    from pathlib import Path as _Path

    if obj is None:
        return None

    # numpy scalars
    if isinstance(obj, _np.generic):
        return obj.item()

    # numpy arrays: store metadata only (prevents huge json)
    if isinstance(obj, _np.ndarray):
        return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}

    # pathlib
    if isinstance(obj, _Path):
        return str(obj)

    # basic containers
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # JSON keys must be str
            out[str(k)] = _jsonify(v)
        return out

    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(x) for x in obj]

    # pandas / other objects: try __dict__
    if hasattr(obj, "__dict__"):
        try:
            return _jsonify(obj.__dict__)
        except Exception:
            return repr(obj)

    # fallback: try direct JSON types, else repr
    if isinstance(obj, (str, int, float, bool)):
        return obj

    return repr(obj)


def _safe_dump_result(res, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "rsa_result.json"

    # Prefer to_dict if exists, else __dict__
    meta_obj = None
    if hasattr(res, "to_dict"):
        try:
            meta_obj = res.to_dict()
        except Exception:
            meta_obj = None
    if meta_obj is None:
        try:
            meta_obj = res.__dict__
        except Exception:
            meta_obj = {"repr": repr(res)}

    meta_obj = _jsonify(meta_obj)
    meta_path.write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------
# RDM utilities
# -------------------------
def _load_rdm_csv(path: Path) -> np.ndarray:
    # assumes square matrix saved as csv with header+index or pure matrix.
    df = pd.read_csv(path, index_col=0)
    mat = df.values
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


# -------------------------
# Extract layer-RDMs from RSAResult (preferred) or from files (fallback)
# -------------------------
def _try_get_layer_rdms_from_result(res) -> Optional[Dict[str, np.ndarray]]:
    """
    Try to extract per-layer RDM matrices from RSAResult.

    We don't assume exact field names; we probe common patterns.
    Returns: dict[layer_key -> rdm_matrix] or None.
    """
    # common attribute candidates
    candidates = [
        "rdms_by_layer",
        "layer_rdms",
        "rdms",
        "rdm_by_layer",
        "rdm_mats_by_layer",
    ]
    for name in candidates:
        if hasattr(res, name):
            obj = getattr(res, name)
            if isinstance(obj, dict) and obj:
                # values could be np.ndarray or list-like
                out: Dict[str, np.ndarray] = {}
                for k, v in obj.items():
                    try:
                        out[str(k)] = np.asarray(v)
                    except Exception:
                        pass
                if out:
                    return out

    # maybe stored as paths
    path_candidates = [
        "rdm_paths_by_layer",
        "layer_rdm_paths",
        "rdm_paths",
    ]
    for name in path_candidates:
        if hasattr(res, name):
            obj = getattr(res, name)
            if isinstance(obj, dict) and obj:
                out: Dict[str, np.ndarray] = {}
                ok = False
                for k, v in obj.items():
                    try:
                        p = Path(v)
                        if p.exists():
                            out[str(k)] = _load_rdm_csv(p)
                            ok = True
                    except Exception:
                        continue
                if ok and out:
                    return out

    return None


def _collect_layer_rdm_paths_from_dir(lang_dir: Path) -> Dict[str, Path]:
    """
    Find per-layer RDM csvs in a language output directory.
    We assume files like rdm_layer-3.csv (or similar).
    """
    rdms: Dict[str, Path] = {}
    for p in sorted(lang_dir.glob("**/rdm_layer*.csv")):
        rdms[p.stem] = p
    return rdms


def _write_layer_rdms_to_dir(layer_rdms: Dict[str, np.ndarray], out_dir: Path) -> None:
    """
    Save RDM matrices for each layer into out_dir/rdm_mats as CSV.
    """
    outp = out_dir / "rdm_mats"
    outp.mkdir(parents=True, exist_ok=True)

    for layer_key, mat in layer_rdms.items():
        mat = np.asarray(mat)
        df = pd.DataFrame(mat)
        safe_key = str(layer_key).replace("/", "_")
        df.to_csv(outp / f"rdm_{safe_key}.csv", index=False)


def _write_rdm_similarity(
    out_dir: Path,
    by_lang_layer_rdms: Dict[str, Dict[str, np.ndarray]],
    corr_method: str = "spearman",
) -> None:
    """
    For each layer available in all languages, compute lang×lang correlation
    between vectorized upper triangle of each RDM.
    """
    out_sim = out_dir / "rdm_similarity"
    out_sim.mkdir(parents=True, exist_ok=True)

    langs = list(by_lang_layer_rdms.keys())
    common_layers = set.intersection(*(set(d.keys()) for d in by_lang_layer_rdms.values()))
    if not common_layers:
        raise RuntimeError("No common layer keys found across languages. Check layer naming / extraction.")

    records = []
    for layer_key in sorted(common_layers, key=lambda x: str(x)):
        vecs = {}
        for lg in langs:
            mat = by_lang_layer_rdms[lg][layer_key]
            vecs[lg] = _upper_tri(np.asarray(mat))

        sim = np.zeros((len(langs), len(langs)), dtype=float)
        for i, li in enumerate(langs):
            for j, lj in enumerate(langs):
                sim[i, j] = _corr(vecs[li], vecs[lj], method=corr_method)

        df = pd.DataFrame(sim, index=langs, columns=langs)
        safe_layer = str(layer_key).replace("/", "_")
        df.to_csv(out_sim / f"rdm_sim_layer_{safe_layer}_{corr_method}.csv")

        for i, li in enumerate(langs):
            for j, lj in enumerate(langs):
                records.append(
                    {
                        "layer": str(layer_key),
                        "lang_i": li,
                        "lang_j": lj,
                        "rdm_corr": sim[i, j],
                        "method": corr_method,
                    }
                )

    pd.DataFrame(records).to_csv(out_sim / f"rdm_sim_long_{corr_method}.csv", index=False)


# -------------------------
# Main
# -------------------------
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
    ap.add_argument("--torch_dtype", type=str, default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--out_dir", type=str, required=True)

    # modes
    ap.add_argument("--input_format", type=str, choices=["long", "wide"], default="long")
    ap.add_argument("--concept_col", type=str, default="concept")
    ap.add_argument("--lang_cols", type=str, nargs="*", default=None, help="For wide input, e.g. --lang_cols en es zh")
    ap.add_argument("--rdm_corr", type=str, choices=["spearman", "pearson"], default="spearman")


    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_format == "long":
        # Compute
        res = run_rsa(
            model_name=args.model_name,
            inputs=str(Path(args.inputs)),
            layers=list(args.layers),
            metric=args.metric,
            word_pooling=args.word_pooling,
            concept_pooling=args.concept_pooling,
            device=args.device,
            max_length=int(args.max_length),
            batch_size=int(args.batch_size),
            torch_dtype=args.torch_dtype,
            trust_remote_code=bool(args.trust_remote_code),
            
        )
        # Save
        _safe_dump_result(res, out_dir)

        # Try also saving per-layer RDM matrices if present
        layer_rdms = _try_get_layer_rdms_from_result(res)
        if layer_rdms is not None:
            _write_layer_rdms_to_dir(layer_rdms, out_dir)

        print(f"Saved RSAResult to: {out_dir}")
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

    by_lang_layer_rdms: Dict[str, Dict[str, np.ndarray]] = {}
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
        tmp = tmp.replace({"": np.nan, "nan": np.nan}).dropna()

        tmp_path = by_lang_base / f"_tmp_inputs_{lg}.csv"
        _write_df(tmp, tmp_path)

        lg_out = by_lang_base / lg
        lg_out.mkdir(parents=True, exist_ok=True)

        res = run_rsa(
            model_name=args.model_name,
            inputs=str(tmp_path),
            layers=list(args.layers),
            metric=args.metric,
            word_pooling=args.word_pooling,
            concept_pooling="mean",  # safe: 1 form per concept => mean is identity
            device=args.device,
            max_length=int(args.max_length),
            batch_size=int(args.batch_size),
            torch_dtype=args.torch_dtype,
            trust_remote_code=bool(args.trust_remote_code),
        )

        _safe_dump_result(res, lg_out)

        # Prefer extracting layer RDMs from result object
        layer_rdms = _try_get_layer_rdms_from_result(res)
        if layer_rdms is not None:
            _write_layer_rdms_to_dir(layer_rdms, lg_out)
            by_lang_layer_rdms[lg] = layer_rdms
        else:
            # Fallback: try find rdm_layer*.csv in lg_out
            paths = _collect_layer_rdm_paths_from_dir(lg_out)
            if not paths:
                raise RuntimeError(
                    f"Could not find per-layer RDMs for language '{lg}'. "
                    f"Neither RSAResult exposed them nor rdm_layer*.csv were found under {lg_out}."
                )
            mats = {k: _load_rdm_csv(p) for k, p in paths.items()}
            by_lang_layer_rdms[lg] = mats

        by_lang_dirs[lg] = lg_out
        print(f"[ok] computed language={lg}, saved under {lg_out}")

    # 2) second-order RSA across languages (compare RDM geometry)
    _write_rdm_similarity(out_dir=out_dir, by_lang_layer_rdms=by_lang_layer_rdms, corr_method=args.rdm_corr)
    print(f"[ok] saved second-order RDM similarity under {out_dir / 'rdm_similarity'}")


if __name__ == "__main__":
    main()
