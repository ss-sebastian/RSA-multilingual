#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
one_shot_form_controls_rsa_v3.py

Goal: make your pipeline STOP “卡死” by removing the real killers:

A) Shuffle permutations should NOT recompute Xn@Xn.T.
   If X' = P X then S' = X'n X'n^T = P S P^T.
   So for each perm we only do indexing: v_perm = S[p[iu0], p[iu1]].

B) The *other* killer is Spearman over ~2,000,000 entries.
   With 21 langs, 210 pairs, 12 layers, 50 perms:
   you’d be ranking millions of floats *hundreds of thousands* of times.
   That is not going to finish on a Mac.

   So this script defaults to RSA-on-a-fixed-subsample of the upper triangle.
   This is standard for permutation nulls: keep the same sampled indices across
   baseline / perms / perturb, and you get a stable estimate.
   You can set --rsa_sample_k 0 for full exact, but expect it to be extremely slow.

What it does:
- Baseline: compute embeddings (cached), then per layer:
  build S(lang,layer) once on CPU (numpy), store as float16 in RAM,
  extract sampled upper-triangle vector v, compute RSA across language pairs.
- Shuffle perms: for each perm, build perm indices within form strata per lang,
  then extract v_perm from the SAME S via indexing, compute RSA.
- Perturb: optional (scramble_inner), recompute embeddings (cached), rebuild S once.

Outputs:
out_dir/
  run_config.json
  analysis/
    form_sim_pairs.csv
    quick_stats.txt
  baseline/
    embeddings/  (emb_layer_-1_en.npy etc)
    rsa_layer_-1.csv
    rsa_pairs_layer_-1.csv
    _DONE.json
  shuffle/
    perm_000/
      rsa_pairs_layer_-1.csv
      _DONE.json
    null_summary_by_pair_layer.csv
  perturb/ (if enabled)
    embeddings/
    rsa_pairs_layer_-1.csv
    _DONE.json

Typical run (fast-ish, recommended):
python -u scripts/one_shot_form_controls_rsa_v3.py \
  --model_name bert-base-multilingual-cased \
  --inputs tmp_lexicon/lexicon_k20_anypos.csv \
  --concept_col concept_id \
  --lang_cols en fi de fr es ru pt it pl nl sv cs bg ja hu uk el ro da ko sk \
  --layers -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 \
  --device mps \
  --batch_size 32 \
  --max_length 32 \
  --out_dir results/oneshot_form_controls_21lang_mbert_v3 \
  --n_perm 50 \
  --shuffle_mode strat_len_sw \
  --len_bucket_size 2 \
  --sw_bucket_size 1 \
  --perturb_mode scramble_inner \
  --rsa_method spearman \
  --rsa_sample_k 200000 \
  --seed 0 \
  --resume

Notes:
- RDM / S is computed on CPU with numpy (BLAS). No MPS for S.
- S stored float16 to save RAM; extracted vectors float32 for RSA.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -------------------------
# IO utils
# -------------------------
def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".tsv", ".tab"]:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def done_marker(path: Path) -> Path:
    return path / "_DONE.json"


def is_done(path: Path) -> bool:
    return done_marker(path).exists()


def write_done(path: Path, payload: dict) -> None:
    with open(done_marker(path), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# -------------------------
# Fast Spearman / Pearson
# -------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float32)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float32)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = np.float32(avg)
        i = j + 1
    return ranks


def pearsonr_fast(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    if x.size != y.size or x.size < 3:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom == 0:
        return np.nan
    return float((x * y).sum() / denom)


def spearmanr_fast(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size or x.size < 3:
        return np.nan
    rx = _rankdata(x)
    ry = _rankdata(y)
    return pearsonr_fast(rx, ry)


# -------------------------
# Layers mapping
# -------------------------
def parse_layers(layer_args: List[int], num_hidden_layers: int) -> List[Tuple[int, int]]:
    """
    Accept negative layers -1..-N. Also supports positive explicit indices.
    hidden_states indexing: [0] is embeddings, [1..num_hidden_layers] are layers.
    """
    out = []
    for l in layer_args:
        if l == 0:
            raise ValueError("Use negative layers like -1..-N. 0 is ambiguous here.")
        if l > 0:
            idx = l
        else:
            idx = (num_hidden_layers + 1) + l  # -1 -> num_hidden_layers
        if idx < 0 or idx > num_hidden_layers:
            raise ValueError(f"Layer {l} out of range for num_hidden_layers={num_hidden_layers}")
        out.append((l, idx))
    return out


# -------------------------
# Pooling + embeddings
# -------------------------
def mean_pool_hidden(h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    B, T, D = h.shape
    mask = attn_mask.clone().bool()
    if T > 2:
        mask[:, 0] = False
        mask[:, -1] = False
    denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    x = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
    return x


@torch.no_grad()
def embed_words_by_layer(
    words: List[str],
    model,
    tokenizer,
    device: str,
    batch_size: int,
    layer_indices: List[Tuple[int, int]],
    max_length: int,
    pbar=None,
) -> Dict[int, np.ndarray]:
    model.eval()
    outs: Dict[int, List[np.ndarray]] = {ln: [] for (ln, _) in layer_indices}

    for i in range(0, len(words), batch_size):
        batch = ["" if (w is None or str(w).lower() == "nan") else str(w) for w in words[i : i + batch_size]]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, return_dict=True)
        hiddens = out.hidden_states
        attn = enc["attention_mask"]

        for layer_neg, hs_idx in layer_indices:
            h = hiddens[hs_idx]
            pooled = mean_pool_hidden(h, attn).detach().to("cpu").float().numpy()
            outs[layer_neg].append(pooled)

        if pbar is not None:
            pbar.update(1)

    return {k: np.vstack(v).astype(np.float32, copy=False) for k, v in outs.items()}


def save_embeddings_for_lang(emb_dir: Path, lang: str, emb_by_layer: Dict[int, np.ndarray]) -> None:
    ensure_dir(emb_dir)
    for layer, X in emb_by_layer.items():
        np.save(emb_dir / f"emb_layer_{layer}_{lang}.npy", X.astype(np.float32, copy=False))


def load_embeddings_for_lang(emb_dir: Path, lang: str, layers_neg: List[int]) -> Dict[int, np.ndarray]:
    out = {}
    for layer in layers_neg:
        p = emb_dir / f"emb_layer_{layer}_{lang}.npy"
        out[layer] = np.load(p, mmap_mode=None).astype(np.float32, copy=False)
    return out


def compute_embeddings_cached(
    df: pd.DataFrame,
    lang_cols: List[str],
    model,
    tokenizer,
    device: str,
    batch_size: int,
    layer_map: List[Tuple[int, int]],
    layers_neg: List[int],
    max_length: int,
    emb_dir: Path,
    pbar=None,
    resume: bool = False,
) -> Dict[str, Dict[int, np.ndarray]]:
    ensure_dir(emb_dir)
    emb_by_lang: Dict[str, Dict[int, np.ndarray]] = {}

    for lang in lang_cols:
        have_all = True
        for layer in layers_neg:
            if not (emb_dir / f"emb_layer_{layer}_{lang}.npy").exists():
                have_all = False
                break
        if resume and have_all:
            emb_by_lang[lang] = load_embeddings_for_lang(emb_dir, lang, layers_neg)
            continue

        words = df[lang].astype("string").fillna("").tolist()
        emb_by_layer = embed_words_by_layer(
            words=words,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            layer_indices=layer_map,
            max_length=max_length,
            pbar=pbar,
        )
        save_embeddings_for_lang(emb_dir, lang, emb_by_layer)
        emb_by_lang[lang] = {layer: emb_by_layer[layer] for layer in layers_neg}

    return emb_by_lang


# -------------------------
# Form controls (shuffle strata)
# -------------------------
def _len_bucket(series: pd.Series, bucket_size: int = 2, min_len: int = 1) -> np.ndarray:
    lens = series.fillna("").astype(str).str.len().clip(lower=min_len).to_numpy()
    return (lens // bucket_size).astype(np.int32)


def _subword_bucket(series: pd.Series, tokenizer, bucket_size: int = 1) -> np.ndarray:
    def _n_subwords(x):
        if x is None:
            return 0
        x = str(x)
        if x == "" or x.lower() == "nan":
            return 0
        return len(tokenizer.tokenize(x))

    nsw = series.astype("string").map(_n_subwords).astype(int).to_numpy()
    return (nsw // bucket_size).astype(np.int32)


def build_perm_index_for_language(
    words: pd.Series,
    tokenizer,
    shuffle_mode: str,
    len_bucket_size: int,
    sw_bucket_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(words)
    if shuffle_mode == "global":
        perm = np.arange(n, dtype=np.int32)
        rng.shuffle(perm)
        return perm

    lb = _len_bucket(words, bucket_size=len_bucket_size)
    if shuffle_mode == "strat_len":
        key = lb
    elif shuffle_mode == "strat_len_sw":
        sb = _subword_bucket(words, tokenizer, bucket_size=sw_bucket_size)
        key = lb.astype(np.int64) * 100000 + sb.astype(np.int64)
    else:
        raise ValueError(f"Unknown shuffle_mode: {shuffle_mode}")

    perm = np.arange(n, dtype=np.int32)
    order = np.argsort(key, kind="mergesort")
    key_sorted = key[order]
    start = 0
    while start < n:
        end = start + 1
        while end < n and key_sorted[end] == key_sorted[start]:
            end += 1
        if end - start > 1:
            block = order[start:end].copy()
            rng.shuffle(block)
            perm[order[start:end]] = block
        start = end
    return perm


# -------------------------
# Perturbation (scramble inner)
# -------------------------
_LATIN_RE = re.compile(r"[A-Za-z]")


def scramble_inner(word: str, rng: np.random.Generator) -> str:
    if word is None:
        return word
    w = str(word)
    if len(w) <= 3:
        return w
    latin_count = len(_LATIN_RE.findall(w))
    if latin_count / max(1, len(w)) < 0.6:
        return w
    inner = list(w[1:-1])
    rng.shuffle(inner)
    return w[0] + "".join(inner) + w[-1]


def perturb_df(df: pd.DataFrame, lang_cols: List[str], mode: str, seed: int) -> pd.DataFrame:
    if mode == "none":
        return df.copy()
    if mode != "scramble_inner":
        raise ValueError(f"Unknown perturb_mode: {mode}")
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in lang_cols:
        out[c] = out[c].astype("string").map(lambda x: scramble_inner(x, rng))
    return out


# -------------------------
# RDM via cosine Gram S (CPU numpy)
# -------------------------
def cosine_gram_cpu(X: np.ndarray) -> np.ndarray:
    """
    Compute S = Xn @ Xn.T using numpy BLAS on CPU.
    Returns float16 matrix to save RAM.
    """
    X = X.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32, copy=False)
    norms[norms == 0] = 1.0
    Xn = X / norms
    S = Xn @ Xn.T  # float32
    return S.astype(np.float16, copy=False)


def precompute_triu(n: int) -> Tuple[np.ndarray, np.ndarray]:
    iu = np.triu_indices(n, k=1)
    return iu[0].astype(np.int32, copy=False), iu[1].astype(np.int32, copy=False)


def make_sample_indices(
    m: int,
    k: int,
    seed: int,
) -> Optional[np.ndarray]:
    """
    Choose k indices from [0..m-1] without replacement.
    If k<=0 or k>=m -> None (meaning use full vector).
    """
    if k is None:
        return None
    if k <= 0 or k >= m:
        return None
    rng = np.random.default_rng(seed)
    return rng.choice(m, size=k, replace=False).astype(np.int64, copy=False)


def upper_vec_from_S(
    S: np.ndarray,
    iu0: np.ndarray,
    iu1: np.ndarray,
    perm: Optional[np.ndarray] = None,
    take_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return sampled upper triangle vector of S (float32).
    If perm is provided: return values from P S P^T without forming it:
      v = S[perm[iu0], perm[iu1]]
    If take_idx is provided: subselect those entries.
    """
    if perm is None:
        a = iu0
        b = iu1
    else:
        a = perm[iu0]
        b = perm[iu1]
    v = S[a, b]  # float16 view
    if take_idx is not None:
        v = v[take_idx]
    return v.astype(np.float32, copy=False)


# -------------------------
# RSA matrices
# -------------------------
def rsa_matrix_from_vecs(
    vecs_by_lang: Dict[str, np.ndarray],
    lang_cols: List[str],
    rsa_method: str,
) -> pd.DataFrame:
    mat = pd.DataFrame(index=lang_cols, columns=lang_cols, dtype=float)
    for i, la in enumerate(lang_cols):
        mat.loc[la, la] = 1.0
        va = vecs_by_lang[la]
        for j in range(i + 1, len(lang_cols)):
            lb = lang_cols[j]
            vb = vecs_by_lang[lb]
            if rsa_method == "pearson":
                rho = pearsonr_fast(va, vb)
            else:
                rho = spearmanr_fast(va, vb)
            mat.loc[la, lb] = rho
            mat.loc[lb, la] = rho
    return mat


def rsa_pairs_long(mat: pd.DataFrame) -> pd.DataFrame:
    cols = list(mat.columns)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append({"lang_a": cols[i], "lang_b": cols[j], "rho": float(mat.iloc[i, j])})
    return pd.DataFrame(rows)


# -------------------------
# Form similarity (analysis)
# -------------------------
def levenshtein(a: str, b: str) -> int:
    a, b = str(a), str(b)
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[-1]


def norm_lev_sim(a: str, b: str) -> float:
    a, b = str(a), str(b)
    L = max(len(a), len(b), 1)
    return 1.0 - (levenshtein(a, b) / L)


def subword_jaccard(a: str, b: str, tokenizer) -> float:
    ta = tokenizer.tokenize(str(a))
    tb = tokenizer.tokenize(str(b))
    sa, sb = set(ta), set(tb)
    if len(sa) == 0 and len(sb) == 0:
        return 1.0
    u = sa | sb
    if len(u) == 0:
        return 0.0
    return len(sa & sb) / len(u)


def all_pairs(lang_cols: List[str]) -> List[Tuple[str, str]]:
    pairs = []
    for i in range(len(lang_cols)):
        for j in range(i + 1, len(lang_cols)):
            pairs.append((lang_cols[i], lang_cols[j]))
    return pairs


def compute_form_sim_table(df: pd.DataFrame, lang_cols: List[str], tokenizer) -> pd.DataFrame:
    pairs = all_pairs(lang_cols)
    rows = []
    for a, b in pairs:
        s1 = df[a].astype("string").fillna("").tolist()
        s2 = df[b].astype("string").fillna("").tolist()
        lev = float(np.mean([norm_lev_sim(x, y) for x, y in zip(s1, s2)]))
        sw = float(np.mean([subword_jaccard(x, y, tokenizer) for x, y in zip(s1, s2)]))
        rows.append({"lang_a": a, "lang_b": b, "lev_sim": lev, "sw_jacc": sw})
    return pd.DataFrame(rows)


# -------------------------
# CLI
# -------------------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--inputs", type=str, required=True)
    ap.add_argument("--concept_col", type=str, default="concept_id")
    ap.add_argument("--lang_cols", type=str, nargs="+", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)

    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=32)

    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--n_perm", type=int, default=1)
    ap.add_argument("--shuffle_mode", type=str, default="strat_len_sw", choices=["global", "strat_len", "strat_len_sw"])
    ap.add_argument("--len_bucket_size", type=int, default=2)
    ap.add_argument("--sw_bucket_size", type=int, default=1)

    ap.add_argument("--perturb_mode", type=str, default="scramble_inner", choices=["none", "scramble_inner"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="Skip folders with _DONE.json marker and reuse cached embeddings")

    # RSA controls
    ap.add_argument("--rsa_method", type=str, default="spearman", choices=["spearman", "pearson"])
    ap.add_argument(
        "--rsa_sample_k",
        type=int,
        default=200000,
        help="Sample K entries from the upper triangle for RSA. 0 => use full (VERY slow for spearman).",
    )

    # Debug / memory
    ap.add_argument("--no_tqdm", action="store_true")
    return ap


def main():
    args = build_argparser().parse_args()
    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    # Save run config early
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # Load data
    df = read_table(Path(args.inputs))
    missing = [c for c in [args.concept_col] + args.lang_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    df = df[[args.concept_col] + args.lang_cols].copy()
    df = df.dropna(subset=args.lang_cols).reset_index(drop=True)
    n_rows = len(df)
    if n_rows < 50:
        raise ValueError(f"Too few rows after dropna: {n_rows}")

    # Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(args.device)

    cfg = model.config
    n_layers = int(getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) or 0)
    if n_layers <= 0:
        raise ValueError("Cannot infer num_hidden_layers from model config.")
    layer_map = parse_layers(args.layers, n_layers)
    layers_neg = [ln for (ln, _) in layer_map]

    # Progress bar: forward pass units only (embeddings)
    n_batches = int(math.ceil(n_rows / args.batch_size))
    forward_jobs = 1 + (1 if args.perturb_mode != "none" else 0)
    total_forward_units = forward_jobs * len(args.lang_cols) * n_batches

    use_tqdm = (tqdm is not None) and (not args.no_tqdm)
    pbar = tqdm(total=total_forward_units, desc="FORWARD", unit="batch", dynamic_ncols=True) if use_tqdm else None
    if pbar is None:
        print("NOTE: tqdm disabled or unavailable; no progress bar for forward pass.")

    # -------------------------
    # Baseline embeddings (cached)
    # -------------------------
    baseline_dir = out_root / "baseline"
    baseline_emb_dir = baseline_dir / "embeddings"
    ensure_dir(baseline_dir)

    emb_base = compute_embeddings_cached(
        df=df,
        lang_cols=args.lang_cols,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        layer_map=layer_map,
        layers_neg=layers_neg,
        max_length=args.max_length,
        emb_dir=baseline_emb_dir,
        pbar=pbar,
        resume=args.resume,
    )

    # -------------------------
    # Perturb embeddings (optional, cached)
    # -------------------------
    need_perturb = (args.perturb_mode != "none")
    if need_perturb:
        perturb_dir = out_root / "perturb"
        perturb_emb_dir = perturb_dir / "embeddings"
        ensure_dir(perturb_dir)

        df_pert = perturb_df(df, args.lang_cols, args.perturb_mode, seed=args.seed + 2000)
        emb_pert = compute_embeddings_cached(
            df=df_pert,
            lang_cols=args.lang_cols,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            batch_size=args.batch_size,
            layer_map=layer_map,
            layers_neg=layers_neg,
            max_length=args.max_length,
            emb_dir=perturb_emb_dir,
            pbar=pbar,
            resume=args.resume,
        )
    else:
        emb_pert = None

    if pbar is not None:
        pbar.close()

    # Precompute words series (for strata) once
    words_by_lang = {lang: df[lang].astype("string").fillna("") for lang in args.lang_cols}

    # Prepare analysis folder
    analysis_dir = out_root / "analysis"
    ensure_dir(analysis_dir)
    form_sim = compute_form_sim_table(df, args.lang_cols, tokenizer)
    form_sim.to_csv(analysis_dir / "form_sim_pairs.csv", index=False)

    # -------------------------
    # LAYER-WISE RUN (memory safe)
    # -------------------------
    shuffle_root = out_root / "shuffle"
    ensure_dir(shuffle_root)

    # Will accumulate shuffle outputs for null summary
    null_rows = []

    # DONE markers for whole conditions
    if args.resume and is_done(baseline_dir):
        baseline_done = True
    else:
        baseline_done = False

    if need_perturb:
        perturb_dir = out_root / "perturb"
        if args.resume and is_done(perturb_dir):
            perturb_done = True
        else:
            perturb_done = False
    else:
        perturb_done = True

    # For baseline/perturb: we write per-layer files; mark DONE at end.
    for layer in layers_neg:
        # -------------------------
        # Build S for baseline (this layer only), CPU numpy
        # -------------------------
        # if baseline already done and layer files exist, we can skip, but keep simple:
        layer_base_pairs_path = baseline_dir / f"rsa_pairs_layer_{layer}.csv"
        layer_base_mat_path = baseline_dir / f"rsa_layer_{layer}.csv"
        if args.resume and layer_base_pairs_path.exists() and layer_base_mat_path.exists():
            base_already = True
        else:
            base_already = False

        # Construct S_by_lang for this layer (baseline)
        # Only if needed for baseline OR needed for shuffle (shuffle always needs baseline S)
        # We always need baseline S for shuffle perms.
        S_base_layer: Dict[str, np.ndarray] = {}
        for lang in args.lang_cols:
            S_base_layer[lang] = cosine_gram_cpu(emb_base[lang][layer])

        n = next(iter(S_base_layer.values())).shape[0]
        iu0, iu1 = precompute_triu(n)
        m = iu0.shape[0]
        take_idx = make_sample_indices(m=m, k=int(args.rsa_sample_k), seed=args.seed + 12345)

        # -------------------------
        # Baseline RSA for this layer
        # -------------------------
        if not base_already:
            vecs = {lang: upper_vec_from_S(S_base_layer[lang], iu0, iu1, perm=None, take_idx=take_idx)
                    for lang in args.lang_cols}
            mat = rsa_matrix_from_vecs(vecs, args.lang_cols, rsa_method=args.rsa_method)
            mat.to_csv(layer_base_mat_path)
            rsa_pairs_long(mat).to_csv(layer_base_pairs_path, index=False)

        # -------------------------
        # Shuffle perms for this layer (FAST indexing from S)
        # -------------------------
        for pidx in range(args.n_perm):
            perm_dir = shuffle_root / f"perm_{pidx:03d}"
            ensure_dir(perm_dir)
            layer_perm_pairs_path = perm_dir / f"rsa_pairs_layer_{layer}.csv"

            if args.resume and layer_perm_pairs_path.exists() and is_done(perm_dir):
                # if perm folder done, skip
                continue

            rng = np.random.default_rng(args.seed + 1000 + pidx)

            perm_idx_by_lang = {}
            for lang in args.lang_cols:
                perm_idx_by_lang[lang] = build_perm_index_for_language(
                    words=words_by_lang[lang],
                    tokenizer=tokenizer,
                    shuffle_mode=args.shuffle_mode,
                    len_bucket_size=args.len_bucket_size,
                    sw_bucket_size=args.sw_bucket_size,
                    rng=rng,
                )

            vecs_perm = {
                lang: upper_vec_from_S(S_base_layer[lang], iu0, iu1, perm=perm_idx_by_lang[lang], take_idx=take_idx)
                for lang in args.lang_cols
            }
            matp = rsa_matrix_from_vecs(vecs_perm, args.lang_cols, rsa_method=args.rsa_method)
            pairs = rsa_pairs_long(matp)
            pairs.to_csv(layer_perm_pairs_path, index=False)

            # Collect for null summary
            pairs2 = pairs.copy()
            pairs2["layer"] = layer
            pairs2["perm"] = pidx
            null_rows.append(pairs2)

            # Mark perm folder done only after all layers finished.
            # But we want resume robust: mark done at end of each perm across layers.
            # We'll write a lightweight marker per-layer; final done written after loop.
            # Here: do nothing.

        # -------------------------
        # Perturb RSA for this layer (optional)
        # -------------------------
        if need_perturb:
            perturb_dir = out_root / "perturb"
            layer_pert_pairs_path = perturb_dir / f"rsa_pairs_layer_{layer}.csv"
            layer_pert_mat_path = perturb_dir / f"rsa_layer_{layer}.csv"

            if not (args.resume and layer_pert_pairs_path.exists() and layer_pert_mat_path.exists()):
                # Build S for perturb at this layer (CPU)
                S_pert_layer: Dict[str, np.ndarray] = {}
                assert emb_pert is not None
                for lang in args.lang_cols:
                    S_pert_layer[lang] = cosine_gram_cpu(emb_pert[lang][layer])

                vecs_pert = {lang: upper_vec_from_S(S_pert_layer[lang], iu0, iu1, perm=None, take_idx=take_idx)
                             for lang in args.lang_cols}
                matq = rsa_matrix_from_vecs(vecs_pert, args.lang_cols, rsa_method=args.rsa_method)
                matq.to_csv(layer_pert_mat_path)
                rsa_pairs_long(matq).to_csv(layer_pert_pairs_path, index=False)

                # free perturb S
                del S_pert_layer
                gc.collect()

        # free baseline S for this layer
        del S_base_layer
        gc.collect()

    # -------------------------
    # Finalize DONE markers
    # -------------------------
    if not (args.resume and is_done(baseline_dir)):
        write_done(baseline_dir, {"status": "ok", "tag": "baseline", "rsa_method": args.rsa_method, "rsa_sample_k": int(args.rsa_sample_k)})

    if need_perturb:
        perturb_dir = out_root / "perturb"
        if not (args.resume and is_done(perturb_dir)):
            write_done(perturb_dir, {"status": "ok", "tag": "perturb", "rsa_method": args.rsa_method, "rsa_sample_k": int(args.rsa_sample_k)})

    # Mark each perm folder done (after all layers are present)
    for pidx in range(args.n_perm):
        perm_dir = shuffle_root / f"perm_{pidx:03d}"
        if perm_dir.exists() and not (args.resume and is_done(perm_dir)):
            write_done(perm_dir, {"status": "ok", "tag": f"shuffle_perm_{pidx:03d}", "rsa_method": args.rsa_method, "rsa_sample_k": int(args.rsa_sample_k)})

    # -------------------------
    # Null summary
    # -------------------------
    if null_rows:
        null_all = pd.concat(null_rows, ignore_index=True)
        null_summary = (
            null_all.groupby(["lang_a", "lang_b", "layer"])["rho"]
            .agg(null_mean="mean", null_std="std")
            .reset_index()
        )
        null_summary.to_csv(shuffle_root / "null_summary_by_pair_layer.csv", index=False)
    else:
        # If resume skipped everything, try to rebuild from existing perm files
        rows = []
        for pidx in range(args.n_perm):
            perm_dir = shuffle_root / f"perm_{pidx:03d}"
            for layer in layers_neg:
                fp = perm_dir / f"rsa_pairs_layer_{layer}.csv"
                if fp.exists():
                    d = pd.read_csv(fp)
                    d["layer"] = layer
                    d["perm"] = pidx
                    rows.append(d)
        if rows:
            null_all = pd.concat(rows, ignore_index=True)
            null_summary = (
                null_all.groupby(["lang_a", "lang_b", "layer"])["rho"]
                .agg(null_mean="mean", null_std="std")
                .reset_index()
            )
            null_summary.to_csv(shuffle_root / "null_summary_by_pair_layer.csv", index=False)

    # -------------------------
    # Quick stats (drop vs form sim) if perturb enabled
    # -------------------------
    def load_pairs_for_dir(cond_dir: Path) -> pd.DataFrame:
        rows = []
        for layer in layers_neg:
            p = cond_dir / f"rsa_pairs_layer_{layer}.csv"
            if p.exists():
                d = pd.read_csv(p)
                d["layer"] = layer
                rows.append(d)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    base_pairs = load_pairs_for_dir(baseline_dir).rename(columns={"rho": "rho_base"})

    with open(analysis_dir / "quick_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"rsa_method={args.rsa_method}\n")
        f.write(f"rsa_sample_k={int(args.rsa_sample_k)} (0 means full upper triangle)\n")
        f.write("\n")

        if need_perturb:
            pert_pairs = load_pairs_for_dir(out_root / "perturb").rename(columns={"rho": "rho_pert"})
            merged = base_pairs.merge(pert_pairs, on=["lang_a", "lang_b", "layer"], how="inner")
            merged["drop"] = merged["rho_base"] - merged["rho_pert"]
            merged = merged.merge(form_sim, on=["lang_a", "lang_b"], how="left")

            # Spearman correlation helper (over pairs×layers)
            def corr_spearman(colx, coly, df_):
                x = df_[colx].to_numpy()
                y = df_[coly].to_numpy()
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 10:
                    return np.nan
                return spearmanr_fast(x[m], y[m])

            f.write("Spearman correlations (pair×layer):\n")
            f.write(f"  corr(drop, lev_sim) = {corr_spearman('drop','lev_sim', merged):.4f}\n")
            f.write(f"  corr(drop, sw_jacc) = {corr_spearman('drop','sw_jacc', merged):.4f}\n")
        else:
            f.write("Perturb disabled; only baseline + shuffle null computed.\n")

    print("DONE ✅")
    print(f"Outputs in: {out_root.resolve()}")
    if args.rsa_method == "spearman" and int(args.rsa_sample_k) == 0:
        print("WARNING: You used full Spearman. If it's slow, set --rsa_sample_k 200000 (or 100000).")


if __name__ == "__main__":
    main()
