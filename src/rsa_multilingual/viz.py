from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rsatoolbox import rdm, vis

from .utils import ensure_dir


def _sanity_square(D: np.ndarray, labels: List[str]) -> np.ndarray:
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"RDM must be square, got {D.shape}")
    if len(labels) != D.shape[0]:
        raise ValueError(f"labels length {len(labels)} != RDM size {D.shape[0]}")
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def _build_rdms(D: np.ndarray, labels: List[str], layer: int, measure: str = "correlation") -> rdm.RDMs:
    Ds = D[None, :, :]
    return rdm.RDMs(
        dissimilarities=Ds,
        dissimilarity_measure=measure,
        rdm_descriptors={"name": f"layer {layer}", "layer": layer},
        pattern_descriptors={"label": labels, "index": list(range(len(labels)))},
    )


def _classical_mds_coords(D: np.ndarray, labels: List[str]) -> pd.DataFrame:
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
    return pd.DataFrame({"label": labels, "x": X[:, 0], "y": X[:, 1]})


def viz_rdm_and_mds(
    D: np.ndarray,
    labels: List[str],
    layer: int,
    out_dir: str,
    tag: Optional[str] = None,
    measure: str = "correlation",
    pattern_descriptor: str = "label",
) -> Path:
    out = ensure_dir(out_dir)
    D = _sanity_square(D, labels)
    rdms = _build_rdms(D, labels, layer=layer, measure=measure)

    suffix = f"_{tag}" if tag else ""
    out_rdm = out / f"rdm_layer{layer}_rsatoolbox{suffix}.png"
    out_mds = out / f"mds_layer{layer}_rsatoolbox{suffix}.png"
    out_xy = out / f"mds_layer{layer}_coords{suffix}.csv"

    fig1, _, _ = vis.show_rdm(
        rdms,
        rdm_descriptor="name",
        pattern_descriptor=pattern_descriptor,
        show_colorbar="figure",
    )
    fig1.savefig(out_rdm, bbox_inches="tight", dpi=300)
    plt.close(fig1)

    ret = vis.show_MDS(
        rdms,
        rdm_descriptor="name",
        pattern_descriptor=pattern_descriptor,
    )
    if isinstance(ret, tuple) and len(ret) >= 1 and hasattr(ret[0], "savefig"):
        fig2 = ret[0]
    else:
        fig2 = plt.gcf()
    fig2.savefig(out_mds, bbox_inches="tight", dpi=300)
    plt.close(fig2)

    _classical_mds_coords(D, labels).to_csv(out_xy, index=False)
    return out
