from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def compute_rdm(X: np.ndarray, metric: str = "correlation", eps: float = 1e-12) -> np.ndarray:
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
    groups = [concept_to_group.get(cid, f"__MISSING__:{cid}") for cid in concept_ids]
    N = len(concept_ids)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = 0.0 if groups[i] == groups[j] else 1.0
            D[j, i] = D[i, j]
    return D
