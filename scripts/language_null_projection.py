#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Language-ID subspace projection (single-shot).

Given embeddings per lang (n_concepts x D), train a linear classifier to predict language,
then project out the classifier row-space from all embeddings.

Outputs:
  out_dir/
    projected/
      layer_-6/en.npy ...
    probe_report.csv (layer, acc_cv, ...)
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def load_layer_embs(emb_root: Path, layer: int, langs: list[str]) -> dict[str, np.ndarray]:
    d = {}
    for lang in langs:
        p = emb_root / f"layer_{layer}" / f"{lang}.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        d[lang] = np.load(p)
    return d

def stack_for_probe(layer_embs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    langs = list(layer_embs.keys())
    for i, lang in enumerate(langs):
        X = layer_embs[lang]
        y = np.full((X.shape[0],), i, dtype=int)
        Xs.append(X); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)

def projection_matrix_from_W(W: np.ndarray) -> np.ndarray:
    # W: (C, D) classifier coefficients
    # P = W^T (W W^T)^-1 W
    WWt = W @ W.T
    inv = np.linalg.pinv(WWt)
    P = W.T @ inv @ W
    return P

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help=".../causal_language_null (contains embeddings/ and meta.json)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--cv_folds", type=int, default=5)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    emb_root = in_dir / "embeddings"
    meta = json.loads((in_dir / "meta.json").read_text(encoding="utf-8"))
    langs = meta["langs"]
    layers = meta["layers"]

    out_dir = Path(args.out_dir)
    proj_root = out_dir / "projected"
    proj_root.mkdir(parents=True, exist_ok=True)

    reports = []
    for layer in layers:
        layer_embs = load_layer_embs(emb_root, layer, langs)
        X, y = stack_for_probe(layer_embs)

        # standardize for probe only (projection computed in standardized space -> apply same transform)
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = scaler.fit_transform(X)

        # CV accuracy just for reporting (not used for projection)
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=0)
        accs = []
        for tr, te in skf.split(Xs, y):
            clf = LogisticRegression(
                penalty="l2", C=args.C, solver="lbfgs", multi_class="multinomial",
                max_iter=2000, n_jobs=1
            )
            clf.fit(Xs[tr], y[tr])
            accs.append(clf.score(Xs[te], y[te]))
        acc_cv = float(np.mean(accs))

        # fit on all data to get W for projection
        clf = LogisticRegression(
            penalty="l2", C=args.C, solver="lbfgs", multi_class="multinomial",
            max_iter=4000, n_jobs=1
        )
        clf.fit(Xs, y)
        W = clf.coef_  # (n_langs, D)
        P = projection_matrix_from_W(W)  # (D,D)

        # apply projection in standardized space, then inverse-transform back to original scale
        # h' = h - P h   (in standardized coordinates)
        offset = 0
        layer_out = proj_root / f"layer_{layer}"
        layer_out.mkdir(parents=True, exist_ok=True)

        for li, lang in enumerate(langs):
            X_lang = layer_embs[lang]
            n = X_lang.shape[0]
            Xs_lang = scaler.transform(X_lang)
            Xs_proj = Xs_lang - (Xs_lang @ P.T)
            # bring back (approximately) to original scale for downstream distances
            X_proj = scaler.inverse_transform(Xs_proj)
            np.save(layer_out / f"{lang}.npy", X_proj)
            offset += n

        reports.append({"layer": layer, "probe_acc_cv": acc_cv, "D": X.shape[1], "N": X.shape[0]})
        print(f"[ok] layer {layer} probe_acc_cv={acc_cv:.3f} projected saved")

    pd.DataFrame(reports).sort_values("layer").to_csv(out_dir / "probe_report.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[ok] wrote probe_report.csv")

if __name__ == "__main__":
    main()
