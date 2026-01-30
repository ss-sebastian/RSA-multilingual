#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rsa.py

Compute RSA (Representational Similarity Analysis) artifacts per Transformer layer.

Assumptions:
- rsa.py and extract.py are in the same directory (e.g., /script).
- extract.py provides: extract_transformer_hidden_states (wrapper/constructor)
- extractor instance provides: get_concept_vectors(...) (recommended)
- extractor instance optionally provides: get_config_page() (nice-to-have)

Outputs:
- rdm_layer{L}.npy        (square RDM)
- rdm_layer{L}.csv        (square RDM with labels)
- rsa_long.csv            (upper triangle long-form table: layer, item_i, item_j, dissimilarity)
- layerwise_rdm_similarity.csv (optional: Spearman between layer RDMs)
- target_rdm.csv/.npy + layer_vs_target_rsa.csv (optional if you provide --target_groups)

Distance metrics:
- correlation (default): 1 - PearsonCorr(vec_i, vec_j)
- cosine: 1 - cosine similarity
- euclidean: L2 distance
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# -----------------------------
# Import extract.py robustly
# -----------------------------
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from extract import extract_transformer_hidden_states  # must exist in extract.py
except Exception as e:
    raise ImportError(
        "Cannot import extract_transformer_hidden_states from extract.py. "
        "Make sure rsa.py and extract.py are in the same folder and extract.py defines it."
    ) from e


# -----------------------------
# Utilities: signature-safe calls
# -----------------------------
def _filter_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs that appear in fn signature."""
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _safe_construct_extractor(**kwargs):
    """Construct extractor using only supported kwargs."""
    ctor = extract_transformer_hidden_states
    use = _filter_kwargs(ctor, kwargs)
    return ctor(**use)


def _safe_call(obj, method_name: str, **kwargs):
    """Call obj.method_name using only supported kwargs."""
    if not hasattr(obj, method_name):
        raise AttributeError(f"Extractor has no method '{method_name}'")
    fn = getattr(obj, method_name)
    use = _filter_kwargs(fn, kwargs)
    return fn(**use)


# -----------------------------
# Input loaders
# -----------------------------
def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def load_concept_map(path: str) -> Dict[str, List[str]]:
    """
    JSON:
      { "HOUSE": ["house","casa"], "DOG": ["dog","perro"] }

    CSV/TSV:
      concept,form
      HOUSE,house
      HOUSE,casa
      DOG,dog

    NOTE: preserves concept order by first appearance in file.
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
        df = _read_table(p)
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
            if concept not in concept_to_forms:
                concept_to_forms[concept] = []
            concept_to_forms[concept].append(form)

        if not concept_to_forms:
            raise ValueError("No valid rows in inputs.")
        return concept_to_forms

    raise ValueError("Unsupported inputs format. Use .json, .csv, or .tsv.")


def load_target_groups(path: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Optional CSV/TSV:
      concept,group
      HOUSE,semantic_A
      DOG,semantic_B
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"target_groups not found: {p}")
    if p.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("target_groups must be .csv or .tsv")
    df = _read_table(p)
    cols = {c.lower(): c for c in df.columns}
    if "concept" not in cols or "group" not in cols:
        raise ValueError("target_groups file must have columns: concept, group")
    c_col, g_col = cols["concept"], cols["group"]
    return dict(zip(df[c_col].astype(str), df[g_col].astype(str)))


def ensure_out_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------
# RDM / Distance
# -----------------------------
def compute_rdm(X: np.ndarray, metric: str = "correlation", eps: float = 1e-12) -> np.ndarray:
    """
    X: [N, H] concept vectors
    Returns: [N, N] dissimilarity matrix
    """
    metric = metric.lower()
    if X.ndim != 2:
        raise ValueError(f"X must be [N,H], got {X.shape}")

    if metric == "correlation":
        Xc = X - X.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(Xc, axis=1, keepdims=True)
        denom = np.maximum(denom, eps)
        Xn = Xc / denom
        corr = Xn @ Xn.T
        D = 1.0 - corr

    elif metric == "cosine":
        denom = np.linalg.norm(X, axis=1, keepdims=True)
        denom = np.maximum(denom, eps)
        Xn = X / denom
        sim = Xn @ Xn.T
        D = 1.0 - sim

    elif metric == "euclidean":
        sq = np.sum(X * X, axis=1, keepdims=True)
        D2 = sq + sq.T - 2.0 * (X @ X.T)
        D2 = np.maximum(D2, 0.0)
        D = np.sqrt(D2)

    else:
        raise ValueError("metric must be: correlation | cosine | euclidean")

    # enforce symmetry + zero diagonal
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def rdm_upper_triangle(D: np.ndarray) -> np.ndarray:
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("RDM must be square")
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Spearman inputs must have same shape")
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    if denom == 0:
        return float("nan")
    return float(np.dot(ra, rb) / denom)


def categorical_target_rdm(concept_ids: List[str], concept_to_group: Dict[str, str]) -> np.ndarray:
    """
    Simple target RDM: 0 if same group else 1.
    Missing concept -> its own unique group.
    """
    groups = [concept_to_group.get(cid, f"__MISSING__:{cid}") for cid in concept_ids]
    N = len(concept_ids)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = 0.0 if groups[i] == groups[j] else 1.0
            D[j, i] = D[i, j]
    return D


# -----------------------------
# Extraction (aligned via signatures)
# -----------------------------
def compute_layer_vectors(
    model_name: str,
    device: str,
    max_length: int,
    batch_size: int,
    torch_dtype: Optional[str],
    trust_remote_code: bool,
    concept_to_forms: Dict[str, List[str]],
    layers: List[int],
    word_pooling: str,
    concept_pooling: str,
) -> Tuple[Dict[int, np.ndarray], List[str], Dict[str, Any]]:
    """
    Returns:
      vecs_by_layer: {layer: np.ndarray [N,H]}
      concept_ids: list[str] order for RDM labels
      meta: dict for debugging / provenance
    """
    ext = _safe_construct_extractor(
        model_name=model_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        default_pooling=word_pooling,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    # Prefer get_concept_vectors; if your extract.py later renames it, you'll see a clear error.
    concept_vecs_by_layer, concept_meta = _safe_call(
        ext,
        "get_concept_vectors",
        concept_to_forms=concept_to_forms,
        layers=layers,
        word_pooling=word_pooling,
        concept_pooling=concept_pooling,
        return_meta=True,
    )

    # meta should include concept_ids; fallback: use input order
    concept_ids = concept_meta.get("concept_ids")
    if concept_ids is None:
        concept_ids = list(concept_to_forms.keys())

    vecs_by_layer: Dict[int, np.ndarray] = {}
    for li in layers:
        X = concept_vecs_by_layer[li]
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu()
        vecs_by_layer[li] = np.asarray(X, dtype=np.float64)

    # best-effort config page
    cfg = None
    if hasattr(ext, "get_config_page"):
        try:
            cfg = ext.get_config_page()
        except Exception:
            cfg = None

    meta_out = {
        "concept_meta": concept_meta,
        "extractor_config": cfg,
        "model_name": model_name,
        "device": device,
        "layers": layers,
        "word_pooling": word_pooling,
        "concept_pooling": concept_pooling,
        "max_length": max_length,
        "batch_size": batch_size,
        "metric_note": "RDM computed on concept-level vectors returned by extract.py",
    }
    return vecs_by_layer, list(concept_ids), meta_out


# -----------------------------
# Saving
# -----------------------------
def save_rdm_square(out_csv: Path, out_npy: Path, D: np.ndarray, labels: List[str]) -> None:
    pd.DataFrame(D, index=labels, columns=labels).to_csv(out_csv, index=True)
    np.save(out_npy, D)


def make_long_table(layer: int, D: np.ndarray, labels: List[str]) -> pd.DataFrame:
    iu = np.triu_indices(D.shape[0], k=1)
    i, j = iu[0], iu[1]
    return pd.DataFrame(
        {
            "layer": layer,
            "i": i,
            "j": j,
            "item_i": [labels[k] for k in i],
            "item_j": [labels[k] for k in j],
            "dissimilarity": D[i, j],
        }
    )


def parse_layers(xs: List[str]) -> List[int]:
    out: List[int] = []
    for x in xs:
        if "," in x:
            out.extend([int(t.strip()) for t in x.split(",") if t.strip() != ""])
        else:
            out.append(int(x))
    # de-dup preserve order
    seen = set()
    dedup = []
    for li in out:
        if li not in seen:
            dedup.append(li)
            seen.add(li)
    return dedup


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="RSA per layer (aligned with extract.py).")

    ap.add_argument("--model_name", type=str, required=True, help="HF model name or local path")
    ap.add_argument("--inputs", type=str, required=True, help=".json or .csv/.tsv with concept,form")
    ap.add_argument("--layers", nargs="+", default=["-1"], help="e.g. --layers -1 -6 -12 or --layers -1,-6,-12")
    ap.add_argument("--metric", type=str, default="correlation", choices=["correlation", "cosine", "euclidean"])
    ap.add_argument("--word_pooling", type=str, default="mean", help="passed to extract.py (if supported)")
    ap.add_argument("--concept_pooling", type=str, default="mean", help="passed to extract.py (if supported)")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    ap.add_argument("--max_length", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--torch_dtype", type=str, default=None, help="e.g. float16 (if supported)")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--out_dir", type=str, default="artifacts/rsa")
    ap.add_argument("--target_groups", type=str, default=None, help="optional .csv/.tsv concept,group")

    args = ap.parse_args()

    layers = parse_layers(args.layers)
    out_dir = ensure_out_dir(args.out_dir)

    concept_to_forms = load_concept_map(args.inputs)
    target_map = load_target_groups(args.target_groups)

    # Save inputs map for provenance
    (out_dir / "inputs_concept_map.json").write_text(
        json.dumps(concept_to_forms, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    vecs_by_layer, concept_ids, meta = compute_layer_vectors(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        concept_to_forms=concept_to_forms,
        layers=layers,
        word_pooling=args.word_pooling,
        concept_pooling=args.concept_pooling,
    )

    (out_dir / "rsa_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # Compute & save RDMs
    rdm_by_layer: Dict[int, np.ndarray] = {}
    long_tables: List[pd.DataFrame] = []

    for li in layers:
        X = vecs_by_layer[li]
        D = compute_rdm(X, metric=args.metric)
        rdm_by_layer[li] = D

        save_rdm_square(
            out_csv=out_dir / f"rdm_layer{li}.csv",
            out_npy=out_dir / f"rdm_layer{li}.npy",
            D=D,
            labels=concept_ids,
        )

        long_tables.append(make_long_table(li, D, concept_ids))

    pd.concat(long_tables, ignore_index=True).to_csv(out_dir / "rsa_long.csv", index=False)

    # Layerwise similarity (Spearman over upper-triangle RDM)
    if len(layers) >= 2:
        rows = []
        for a in range(len(layers)):
            for b in range(a + 1, len(layers)):
                la, lb = layers[a], layers[b]
                va = rdm_upper_triangle(rdm_by_layer[la])
                vb = rdm_upper_triangle(rdm_by_layer[lb])
                rho = spearman_corr(va, vb)
                rows.append({"layer_a": la, "layer_b": lb, "spearman_rho": rho})
        pd.DataFrame(rows).to_csv(out_dir / "layerwise_rdm_similarity.csv", index=False)

    # Optional: target RSA
    if target_map is not None:
        target_D = categorical_target_rdm(concept_ids, target_map)
        save_rdm_square(out_dir / "target_rdm.csv", out_dir / "target_rdm.npy", target_D, concept_ids)

        vt = rdm_upper_triangle(target_D)
        rows = []
        for li in layers:
            v = rdm_upper_triangle(rdm_by_layer[li])
            rho = spearman_corr(v, vt)
            rows.append({"layer": li, "spearman_rho_to_target": rho})
        pd.DataFrame(rows).to_csv(out_dir / "layer_vs_target_rsa.csv", index=False)

    print(f"Saved RSA outputs to: {out_dir.resolve()}")
    print(f"Concepts: {len(concept_ids)} | Layers: {layers} | Metric: {args.metric}")


if __name__ == "__main__":
    main()
