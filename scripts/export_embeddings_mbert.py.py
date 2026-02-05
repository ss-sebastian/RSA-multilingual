#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export concept embeddings per language and per layer for multilingual BERT.

Input:
  - lexicon CSV (wide): concept_id, en, es, zh, it, ru, de (or your lang cols)
Output:
  out_dir/
    embeddings/
      layer_-12/
        en.npy, es.npy, ...
      ...
    meta.json  (concept order, langs, layers, model, pooling)
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

def pool_hidden(hidden: torch.Tensor, attn: torch.Tensor, mode: str) -> torch.Tensor:
    # hidden: (B,T,D), attn: (B,T)
    if mode == "cls":
        return hidden[:, 0, :]
    # exclude special tokens by attention mask (keeps [CLS]/[SEP] unless you pre-mask)
    mask = attn.unsqueeze(-1).float()  # (B,T,1)
    if mode == "mean":
        s = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return s / denom
    if mode == "sum":
        return (hidden * mask).sum(dim=1)
    raise ValueError(f"Unknown pooling: {mode}")

@torch.no_grad()
def encode_words(model, tok, words, layer_idx: int, pooling: str, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc, output_hidden_states=True, return_dict=True)
        hs = outputs.hidden_states  # tuple: (embeddings, layer1, ..., layerN)
        # layer_idx uses transformer convention: 1..N; but our file naming uses negative indices like -1..-12
        # We'll pass "true layer index" below.
        h = hs[layer_idx]  # (B,T,D)
        vec = pool_hidden(h, enc["attention_mask"], pooling)  # (B,D)
        out.append(vec.detach().cpu().numpy())
    return np.concatenate(out, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="bert-base-multilingual-cased")
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--concept_col", default="concept_id")
    ap.add_argument("--lang_cols", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", required=True, help="e.g. -12 -11 ... -1")
    ap.add_argument("--pooling", default="sum", choices=["sum","mean","cls"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    emb_root = out_dir / "embeddings"
    emb_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.lexicon)
    df = df.dropna(subset=[args.concept_col])
    concepts = df[args.concept_col].astype(str).tolist()

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(args.device)

    # mBERT has N transformer layers; hidden_states tuple length is N+1 (includes embedding layer at index 0)
    n_layers = model.config.num_hidden_layers
    # map negative layer like -1..-12 to hidden_states index:
    # hidden_states index: 0=emb, 1=layer1 ... N=layerN
    def neg_to_hs_index(layer_neg: int) -> int:
        # -1 -> N, -12 -> N-11
        return n_layers + layer_neg + 1

    layers_neg = [int(x) for x in args.layers]
    for ln in layers_neg:
        hs_idx = neg_to_hs_index(ln)
        layer_dir = emb_root / f"layer_{ln}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        for lang in args.lang_cols:
            words = df[lang].astype(str).fillna("").tolist()
            # if some concepts have multiple forms separated by ; you can split/choose here (kept simple)
            X = encode_words(model, tok, words, hs_idx, args.pooling, args.device, args.batch_size)
            np.save(layer_dir / f"{lang}.npy", X)
            print(f"[ok] layer {ln} lang {lang} -> {X.shape} saved")

    meta = {
        "model_name": args.model_name,
        "pooling": args.pooling,
        "layers": layers_neg,
        "langs": args.lang_cols,
        "concepts": concepts,
        "n_layers": n_layers,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[ok] wrote meta.json")

if __name__ == "__main__":
    main()
