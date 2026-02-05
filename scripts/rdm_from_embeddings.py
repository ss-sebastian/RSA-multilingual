#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def rdm_corr_distance(X: np.ndarray) -> np.ndarray:
    # correlation distance = 1 - corr
    Xc = X - X.mean(axis=1, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    C = Xn @ Xn.T
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    return D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help=".../causal_language_null_proj (contains projected/ and meta.json)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    proj_root = in_dir / "projected"
    meta = json.loads((in_dir / "meta.json").read_text(encoding="utf-8"))
    langs = meta["langs"]
    layers = meta["layers"]
    concepts = meta["concepts"]

    out_dir = Path(args.out_dir)
    for lang in langs:
        (out_dir / "by_lang" / lang / "rdm_mats").mkdir(parents=True, exist_ok=True)

    for layer in layers:
        layer_dir = proj_root / f"layer_{layer}"
        for lang in langs:
            X = np.load(layer_dir / f"{lang}.npy")
            D = rdm_corr_distance(X)
            df = pd.DataFrame(D, index=concepts, columns=concepts)
            out = out_dir / "by_lang" / lang / "rdm_mats" / f"rdm_layer_{layer}.csv"
            df.to_csv(out)
        print(f"[ok] wrote RDMs for layer {layer}")

    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[ok] wrote meta.json")

if __name__ == "__main__":
    main()
