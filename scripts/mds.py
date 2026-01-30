#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/mds.py

Thin CLI wrapper:
- loads RDM from rdm_layer*.csv OR reconstructs from rsa_long.csv
- calls rsa_multilingual.viz.viz_rdm_and_mds(...) which uses rsatoolbox
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Ensure repo_root/src is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rsa_multilingual.viz import viz_rdm_and_mds  # noqa: E402


def load_rdm_from_csv(path: Path) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path, index_col=0)
    labels = df.index.astype(str).tolist()
    D = df.to_numpy(dtype=float)
    return D, labels


def load_rdm_from_long(path: Path, layer: int) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path)
    required = {"layer", "item_i", "item_j", "dissimilarity"}
    if not required.issubset(df.columns):
        raise ValueError(f"rsa_long.csv must contain columns {required}, got {set(df.columns)}")

    sub = df[df["layer"] == layer].copy()
    if len(sub) == 0:
        raise ValueError(f"No rows found for layer={layer} in {path}")

    labels = sorted(set(sub["item_i"].astype(str)).union(set(sub["item_j"].astype(str))))
    idx = {lab: i for i, lab in enumerate(labels)}
    N = len(labels)
    D = np.zeros((N, N), dtype=float)

    for _, r in sub.iterrows():
        a = str(r["item_i"])
        b = str(r["item_j"])
        d = float(r["dissimilarity"])
        i, j = idx[a], idx[b]
        D[i, j] = d
        D[j, i] = d

    np.fill_diagonal(D, 0.0)
    return D, labels


def infer_layer_from_name(name: str) -> int:
    import re
    m = re.search(r"layer(-?\d+)", name)
    return int(m.group(1)) if m else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize RSA RDMs using rsatoolbox.")
    ap.add_argument("--input", required=True, type=str, help="rdm_layer*.csv OR rsa_long.csv")
    ap.add_argument("--input_type", required=True, choices=["rdm_csv", "rsa_long"])
    ap.add_argument("--layer", type=int, default=None, help="required for rsa_long; optional for rdm_csv")
    ap.add_argument("--out_dir", type=str, default="artifacts/mds_rsatoolbox")
    ap.add_argument("--measure", type=str, default="correlation", help="label only, e.g. '1-r' or '1-rho'")
    ap.add_argument("--pattern_descriptor", type=str, default="label", help="label field to display on axes")
    ap.add_argument("--tag", type=str, default=None)

    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    if args.input_type == "rdm_csv":
        D, labels = load_rdm_from_csv(in_path)
        layer = args.layer if args.layer is not None else infer_layer_from_name(in_path.name)
    else:
        if args.layer is None:
            raise ValueError("--layer is required when input_type=rsa_long")
        layer = int(args.layer)
        D, labels = load_rdm_from_long(in_path, layer=layer)

    viz_rdm_and_mds(
        D=D,
        labels=labels,
        layer=layer,
        out_dir=args.out_dir,
        tag=args.tag,
        measure=args.measure,
        pattern_descriptor=args.pattern_descriptor,
    )

    print(f"Saved visualizations to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
