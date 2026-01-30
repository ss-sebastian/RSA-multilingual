#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/rsa.py

Thin CLI wrapper over src/rsa_multilingual/pipeline.py:
- run_rsa(...)
- save_rsa_outputs(...)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Ensure repo_root/src is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rsa_multilingual.pipeline import run_rsa, save_rsa_outputs  # noqa: E402


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

    res = run_rsa(
        model_name=args.model_name,
        inputs=args.inputs,
        layers=layers,
        metric=args.metric,
        word_pooling=args.word_pooling,
        concept_pooling=args.concept_pooling,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        target_groups=args.target_groups,
    )

    out = save_rsa_outputs(res, args.out_dir)
    print(f"Saved RSA outputs to: {out.resolve()}")
    print(f"Concepts: {len(res.concept_ids)} | Layers: {res.layers} | Metric: {res.metric}")


if __name__ == "__main__":
    main()
