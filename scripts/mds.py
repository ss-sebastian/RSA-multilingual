#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MDS.py (rsatoolbox version)

Visualize RDMs using rsatoolbox:
- RDM heatmap: rsatoolbox.vis.show_rdm
- MDS scatter: rsatoolbox.vis.show_MDS

Inputs:
1) --input_type rdm_csv: rdm_layer{L}.csv (square, with labels as row/col names)
2) --input_type rsa_long: rsa_long.csv + --layer (reconstruct square RDM)

Outputs (in --out_dir):
- rdm_layer{L}_rsatoolbox.png
- mds_layer{L}_rsatoolbox.png
- mds_layer{L}_coords.csv

Notes:
- rsatoolbox expects RDMs in shape [n_rdm, n_cond, n_cond]; we wrap with leading dim.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from rsatoolbox import vis, rdm


def load_rdm_from_csv(path: Path) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path, index_col=0)
    labels = df.index.astype(str).tolist()
    D = df.to_numpy(dtype=float)
    _sanity_square(D, labels)
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
    _sanity_square(D, labels)
    return D, labels


def _sanity_square(D: np.ndarray, labels: List[str]) -> None:
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"RDM must be square, got {D.shape}")
    if len(labels) != D.shape[0]:
        raise ValueError(f"labels length {len(labels)} != RDM size {D.shape[0]}")
    # enforce symmetry + diagonal
    D[:] = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)


def build_rsatoolbox_rdms(D: np.ndarray, labels: List[str], layer: int, measure: str) -> rdm.RDMs:
    # rsatoolbox expects [n_rdm, n_cond, n_cond]
    Ds = D[None, :, :]
    rdms = rdm.RDMs(
        dissimilarities=Ds,
        dissimilarity_measure=measure,
        rdm_descriptors={"name": f"layer {layer}", "layer": layer},
        pattern_descriptors={"label": labels, "index": list(range(len(labels)))},
    )
    return rdms


def save_show_rdm(rdms: rdm.RDMs, out_png: Path, pattern_descriptor: str = "label") -> None:
    fig, ax, _ = vis.show_rdm(
        rdms,
        rdm_descriptor="name",
        pattern_descriptor=pattern_descriptor,
        show_colorbar="figure",
    )
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_show_mds(rdms: rdm.RDMs, out_png: Path, pattern_descriptor: str = "label") -> None:
    # show_MDS returns a matplotlib figure/axes in recent versions; we handle both cases.
    ret = vis.show_MDS(
        rdms,
        rdm_descriptor="name",
        pattern_descriptor=pattern_descriptor,
    )
    if isinstance(ret, tuple) and len(ret) >= 1 and hasattr(ret[0], "savefig"):
        fig = ret[0]
    else:
        fig = plt.gcf()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def export_mds_coords(D: np.ndarray, labels: List[str], out_csv: Path) -> None:
    # Classical MDS coords from double-centering (simple + no sklearn dependency)
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (J @ D2 @ J)
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    pos = evals > 1e-12
    evals = evals[pos]
    evecs = evecs[:, pos]
    k = min(2, len(evals))
    X = evecs[:, :k] @ np.diag(np.sqrt(evals[:k]))
    pd.DataFrame({"label": labels, "x": X[:, 0], "y": X[:, 1]}).to_csv(out_csv, index=False)


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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_type == "rdm_csv":
        D, labels = load_rdm_from_csv(in_path)
        layer = args.layer if args.layer is not None else infer_layer_from_name(in_path.name)
    else:
        if args.layer is None:
            raise ValueError("--layer is required when input_type=rsa_long")
        layer = int(args.layer)
        D, labels = load_rdm_from_long(in_path, layer=layer)

    rdms = build_rsatoolbox_rdms(D, labels, layer=layer, measure=args.measure)

    tag = f"_{args.tag}" if args.tag else ""
    out_rdm = out_dir / f"rdm_layer{layer}_rsatoolbox{tag}.png"
    out_mds = out_dir / f"mds_layer{layer}_rsatoolbox{tag}.png"
    out_xy = out_dir / f"mds_layer{layer}_coords{tag}.csv"

    save_show_rdm(rdms, out_rdm, pattern_descriptor=args.pattern_descriptor)
    save_show_mds(rdms, out_mds, pattern_descriptor=args.pattern_descriptor)
    export_mds_coords(D, labels, out_xy)

    print(f"Saved: {out_rdm.resolve()}")
    print(f"Saved: {out_mds.resolve()}")
    print(f"Saved: {out_xy.resolve()}")


if __name__ == "__main__":
    main()
