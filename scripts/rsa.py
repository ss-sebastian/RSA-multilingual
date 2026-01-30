#!/usr/bin/env python3
"""
rsa.py

Compute Representational Similarity Analysis (RSA) artifacts for Transformer layers.

Designed to pair with your extract.py.

Pipeline
1) Load a concept list (concept -> surface forms) from JSON/CSV/TSV.
2) Use extractor to get one vector per concept, per requested layer.
3) Compute an RDM (Representational Dissimilarity Matrix) per layer.
4) Export:
   - Per-layer RDMs as .npy and .csv
   - Long-form pairwise table (upper triangle) for easy stats / plotting
   - Optional: layer-by-layer RDM similarity (Spearman)
   - Optional: layer-vs-target RSA (Spearman) if you provide a target grouping file

Distance metrics
- correlation (default): 1 - Pearson correlation between vectors
- cosine: 1 - cosine similarity
- euclidean: Euclidean distance

Input formats
A) JSON concept map
   {
     "HOUSE": ["house", "casa", "房子"],
     "DOG":   ["dog", "perro", "狗"]
   }

B) CSV/TSV with columns:
   concept,form
   HOUSE,house
   HOUSE,casa
   DOG,dog
   ...

Optional target grouping file (CSV/TSV) with columns:
   concept,group
This creates a categorical target RDM where d=0 if same group else d=1.

Example
python rsa.py \
  --model_name xlm-roberta-base \
  --inputs concepts.csv \
  --layers -1 -6 -12 \
  --metric correlation \
  --word_pooling mean \
  --out_dir artifacts/rsa_xlmr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch

from extract import extract_transformer_hidden_states
# -----------------------------
# Import extractor (local file)
# -----------------------------

def _import_extractor():
    """
    Import from extract.py with a couple of fallbacks so running is not brittle.
    """
    try:
        from extract import extract_transformer_hidden_states  # type: ignore
        return extract_transformer_hidden_states
    except Exception:
        import sys
        here = Path(__file__).resolve().parent
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        from extract import extract_transformer_hidden_states  # type: ignore
        return extract_transformer_hidden_states


extract_transformer_hidden_states = _import_extractor()


# -----------------------------
# I/O helpers
# -----------------------------

def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def load_concept_map(path: str) -> Dict[str, List[str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"inputs not found: {p}")

    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("JSON inputs must be an object mapping concept -> list[forms].")
        out: Dict[str, List[str]] = {}
        for k, v in obj.items():
            if isinstance(v, str):
                out[str(k)] = [v]
            elif isinstance(v, list):
                out[str(k)] = [str(x) for x in v]
            else:
                raise ValueError(f"Invalid value for concept {k}: expected str or list, got {type(v)}")
        return out

    if p.suffix.lower() in {".csv", ".tsv"}:
        df = _read_table(p)
        cols = {c.lower(): c for c in df.columns}
        if "concept" not in cols or "form" not in cols:
            raise ValueError("CSV/TSV inputs must have columns: concept, form")
        c_col, f_col = cols["concept"], cols["form"]
        concept_to_forms: Dict[str, List[str]] = {}
        for concept, sub in df.groupby(c_col):
            forms = [str(x) for x in sub[f_col].dropna().tolist()]
            if len(forms) == 0:
                continue
            concept_to_forms[str(concept)] = forms
        if len(concept_to_forms) == 0:
            raise ValueError("No valid rows found in inputs file.")
        return concept_to_forms

    raise ValueError("Unsupported inputs format. Use .json, .csv, or .tsv.")


def load_target_groups(path: Optional[str]) -> Optional[Dict[str, str]]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"target file not found: {p}")
    if p.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("target file must be .csv or .tsv with columns: concept, group")

    df = _read_table(p)
    cols = {c.lower(): c for c in df.columns}
    if "concept" not in cols or "group" not in cols:
        raise ValueError("target file must have columns: concept, group")

    c_col, g_col = cols["concept"], cols["group"]
    mapping = dict(zip(df[c_col].astype(str), df[g_col].astype(str)))
    return mapping


def ensure_out_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------
# Distance / RDM
# -----------------------------

def compute_rdm(X: np.ndarray, metric: str = "correlation", eps: float = 1e-12) -> np.ndarray:
    """
    X: [N, H]
    returns: [N, N] distances
    """
    metric = metric.lower()
    if X.ndim != 2:
        raise ValueError(f"X must be 2D [N,H], got shape {X.shape}")

    if metric == "correlation":
        # Pearson correlation between rows, computed stably
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
        sq = np.sum(X * X, axis=1, keepdims=True)  # [N,1]
        D2 = sq + sq.T - 2.0 * (X @ X.T)
        D2 = np.maximum(D2, 0.0)
        D = np.sqrt(D2)

    else:
        raise ValueError("metric must be one of: correlation | cosine | euclidean")

    # clean numerical noise; enforce symmetry & zero diagonal
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def rdm_upper_triangle(D: np.ndarray) -> np.ndarray:
    """Vectorize the upper triangle (excluding diagonal) as 1D array."""
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be square")
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman correlation using average ranks (handles ties) via pandas.
    """
    if a.shape != b.shape:
        raise ValueError("Inputs to spearman must have same shape")
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (np.linalg.norm(ra) * np.linalg.norm(rb))
    if denom == 0:
        return float("nan")
    return float(np.dot(ra, rb) / denom)


def categorical_target_rdm(concept_ids: List[str], concept_to_group: Dict[str, str]) -> np.ndarray:
    """
    Create a simple target RDM: 0 if same group, 1 if different group.
    Concepts absent from the mapping are treated as their own unique group.
    """
    groups = []
    for cid in concept_ids:
        groups.append(concept_to_group.get(cid, f"__MISSING__:{cid}"))
    N = len(concept_ids)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = 0.0 if groups[i] == groups[j] else 1.0
            D[j, i] = D[i, j]
    return D


# -----------------------------
# Main RSA computation
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
      vecs_by_layer: dict layer -> np.ndarray [N,H] (CPU)
      concept_ids: fixed order of concepts
      meta: extraction meta (tokenization, owner indices, etc.)
    """
    ext = extract_transformer_hidden_states(
        model_name=model_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        default_pooling=word_pooling,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    concept_vecs_by_layer, meta = ext.get_concept_vectors(
        concept_to_forms=concept_to_forms,
        layers=layers,
        word_pooling=word_pooling,
        concept_pooling=concept_pooling,
        return_meta=True,
    )

    concept_ids = meta["concept_ids"]
    vecs_by_layer: Dict[int, np.ndarray] = {}
    for li in layers:
        X = concept_vecs_by_layer[li]
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu()
        vecs_by_layer[li] = np.asarray(X, dtype=np.float64)

    meta_out = {
        "concept_meta": meta,
        "extractor_config": ext.get_config_page(),
    }
    return vecs_by_layer, concept_ids, meta_out


def save_rdm_square(out_path_csv: Path, out_path_npy: Path, D: np.ndarray, labels: List[str]) -> None:
    df = pd.DataFrame(D, index=labels, columns=labels)
    df.to_csv(out_path_csv, index=True)
    np.save(out_path_npy, D)


def make_long_table(layer: int, D: np.ndarray, labels: List[str]) -> pd.DataFrame:
    iu = np.triu_indices(D.shape[0], k=1)
    i = iu[0]
    j = iu[1]
    df = pd.DataFrame(
        {
            "layer": layer,
            "i": i,
            "j": j,
            "item_i": [labels[k] for k in i],
            "item_j": [labels[k] for k in j],
            "dissimilarity": D[i, j],
        }
    )
    return df


def parse_layers(xs: List[str]) -> List[int]:
    out: List[int] = []
    for x in xs:
        if "," in x:
            out.extend([int(t.strip()) for t in x.split(",") if t.strip() != ""])
        else:
            out.append(int(x))
    seen = set()
    dedup = []
    for li in out:
        if li not in seen:
            dedup.append(li)
            seen.add(li)
    return dedup


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute RSA RDMs across Transformer layers.")
    ap.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or path")
    ap.add_argument("--inputs", type=str, required=True, help="concept list: .json or .csv/.tsv (concept,form)")
    ap.add_argument("--layers", nargs="+", default=["-1"], help="e.g. --layers -1 -6 -12 or --layers -1,-6,-12")
    ap.add_argument("--metric", type=str, default="correlation", choices=["correlation", "cosine", "euclidean"])
    ap.add_argument("--word_pooling", type=str, default="mean", choices=["mean", "cls", "last"])
    ap.add_argument("--concept_pooling", type=str, default="mean", choices=["mean"])
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda (if available)")
    ap.add_argument("--max_length", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--torch_dtype", type=str, default=None, help="e.g. float16 on GPU; leave None for default")
    ap.add_argument("--trust_remote_code", action="store_true", help="set True for some HF community models")
    ap.add_argument("--out_dir", type=str, default="artifacts/rsa", help="output directory")
    ap.add_argument("--target_groups", type=str, default=None, help="optional .csv/.tsv with columns concept,group")

    args = ap.parse_args()

    layers = parse_layers(args.layers)
    out_dir = ensure_out_dir(args.out_dir)

    concept_to_forms = load_concept_map(args.inputs)
    target_map = load_target_groups(args.target_groups)

    (out_dir / "inputs_concept_map.json").write_text(
        json.dumps(concept_to_forms, ensure_ascii=False, indent=2), encoding="utf-8"
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

    (out_dir / "extraction_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    long_tables: List[pd.DataFrame] = []
    rdm_by_layer: Dict[int, np.ndarray] = {}

    for li in layers:
        X = vecs_by_layer[li]
        D = compute_rdm(X, metric=args.metric)
        rdm_by_layer[li] = D

        save_rdm_square(
            out_path_csv=out_dir / f"rdm_layer{li}.csv",
            out_path_npy=out_dir / f"rdm_layer{li}.npy",
            D=D,
            labels=concept_ids,
        )
        long_tables.append(make_long_table(li, D, concept_ids))

    rsa_long = pd.concat(long_tables, ignore_index=True)
    rsa_long.to_csv(out_dir / "rsa_long.csv", index=False)

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

    if target_map is not None:
        target_D = categorical_target_rdm(concept_ids, target_map)
        vt = rdm_upper_triangle(target_D)
        rows = []
        for li in layers:
            v = rdm_upper_triangle(rdm_by_layer[li])
            rho = spearman_corr(v, vt)
            rows.append({"layer": li, "spearman_rho_to_target": rho})
        pd.DataFrame(rows).to_csv(out_dir / "layer_vs_target_rsa.csv", index=False)

        save_rdm_square(
            out_path_csv=out_dir / "target_rdm.csv",
            out_path_npy=out_dir / "target_rdm.npy",
            D=target_D,
            labels=concept_ids,
        )

    print(f"Saved outputs to: {out_dir.resolve()}")
    print(f"#concepts: {len(concept_ids)} | layers: {layers} | metric: {args.metric}")
    print("Files: rsa_long.csv + per-layer rdm_layer*.csv/.npy")


if __name__ == "__main__":
    main()
