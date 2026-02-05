#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/alignment_stats.py

End-to-end:
- load lexicon table (concept, lang, form)
- extract form embeddings per layer using HiddenStateExtractor (from extract.py)
- concept pooling within each language
- compute cross-language RDM similarity (Spearman) per layer
- permutation null per layer/pair -> p, z
- split-half reliability within-language per layer (requires >=2 forms per concept)
- write results to out_dir

Example:
python scripts/alignment_stats.py \
  --lexicon_csv data/concepts_forms.csv \
  --model_name bert-base-multilingual-cased \
  --langs en es it de zh ru \
  --layers 1 2 3 4 5 6 7 8 9 10 11 \
  --device cpu \
  --batch_size 64 \
  --metric cosine \
  --n_perm 2000 \
  --n_repeats 200 \
  --out_dir results/bert_alignment \
  --plot
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

# ---- import your extractor ----
# Assumes extract.py is in repo root or PYTHONPATH
from extract import ExtractionConfig, HiddenStateExtractor  # type: ignore


# -------------------------
# RDM helpers
# -------------------------
def rdm_from_embeddings(X: np.ndarray, metric: str = "cosine") -> np.ndarray:
    # X: (n_concepts, d)
    return squareform(pdist(X, metric=metric))

def upper_tri_vec(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]

def rdm_spearman(A: np.ndarray, B: np.ndarray) -> float:
    a = upper_tri_vec(A)
    b = upper_tri_vec(B)
    rho, _ = spearmanr(a, b)
    return float(rho)

def perm_null_rdm_similarity(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    n_perm: int = 2000,
    seed: int = 0,
) -> Tuple[float, np.ndarray, float, float]:
    """
    returns: (obs, null, p_two_sided, z)
    """
    assert rdm_a.shape == rdm_b.shape
    n = rdm_a.shape[0]
    rng = np.random.default_rng(seed)

    obs = rdm_spearman(rdm_a, rdm_b)

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(n)
        rb = rdm_b[np.ix_(perm, perm)]
        null[i] = rdm_spearman(rdm_a, rb)

    # two-sided p, +1 smoothing
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)

    mu = null.mean()
    sd = null.std(ddof=1) + 1e-12
    z = (obs - mu) / sd
    return obs, null, float(p), float(z)

def split_half_reliability_rdm(
    forms_embeds: List[List[np.ndarray]],
    metric: str = "cosine",
    n_repeats: int = 200,
    min_forms: int = 2,
    seed: int = 0,
) -> Optional[Tuple[np.ndarray, float, Tuple[float, float]]]:
    """
    forms_embeds: len=n_concepts, each is list of vectors (d,)
    returns: (rhos, mean, (ci_lo, ci_hi)) or None if insufficient data
    """
    kept = [i for i, fs in enumerate(forms_embeds) if len(fs) >= min_forms]
    if len(kept) < 4:
        return None

    rng = np.random.default_rng(seed)
    rhos = []

    for _ in range(n_repeats):
        X1, X2 = [], []
        for i in kept:
            fs = forms_embeds[i]
            idx = rng.permutation(len(fs))
            mid = len(fs) // 2
            h1, h2 = idx[:mid], idx[mid:]
            if len(h1) == 0 or len(h2) == 0:
                continue
            v1 = np.mean([fs[j] for j in h1], axis=0)
            v2 = np.mean([fs[j] for j in h2], axis=0)
            X1.append(v1)
            X2.append(v2)

        if len(X1) < 4:
            continue

        X1 = np.stack(X1, axis=0)
        X2 = np.stack(X2, axis=0)
        R1 = rdm_from_embeddings(X1, metric=metric)
        R2 = rdm_from_embeddings(X2, metric=metric)
        rhos.append(rdm_spearman(R1, R2))

    if len(rhos) == 0:
        return None

    rhos = np.asarray(rhos, dtype=float)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return rhos, float(rhos.mean()), (float(lo), float(hi))


# -------------------------
# Data loading & extraction
# -------------------------
def load_lexicon_csv(path: Path, langs: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"concept", "lang", "form"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"lexicon_csv missing columns: {sorted(missing)}; need concept, lang, form")
    df["lang"] = df["lang"].astype(str)
    df["concept"] = df["concept"].astype(str)
    df["form"] = df["form"].astype(str)
    df = df[df["lang"].isin(langs)].copy()
    if df.empty:
        raise ValueError("No rows left after filtering langs. Check --langs vs lexicon_csv.")
    return df

def build_concept_index(df: pd.DataFrame, langs: List[str]) -> Tuple[List[str], Dict[str, int]]:
    # concepts present in ALL selected langs (strict intersection)
    by_lang = {L: set(df.loc[df["lang"] == L, "concept"].unique()) for L in langs}
    common = set.intersection(*by_lang.values())
    concepts = sorted(common)
    if len(concepts) < 4:
        raise ValueError(f"Too few common concepts across langs ({len(concepts)}).")
    idx = {c: i for i, c in enumerate(concepts)}
    return concepts, idx

def extract_form_vectors(
    extractor: HiddenStateExtractor,
    df: pd.DataFrame,
    concepts: List[str],
    concept_to_i: Dict[str, int],
    langs: List[str],
    layers: List[int],
    pooling: Optional[str],
) -> Dict[str, Dict[int, List[List[np.ndarray]]]]:
    """
    returns:
      forms_by_lang_layer[lang][layer] = list over concepts (len=n_concepts):
           each item is list of vectors for that concept's forms
    """
    n_concepts = len(concepts)
    forms_by_lang_layer: Dict[str, Dict[int, List[List[np.ndarray]]]] = {
        L: {ly: [[] for _ in range(n_concepts)] for ly in layers} for L in langs
    }

    # group by lang to batch efficiently
    for L in langs:
        sub = df[df["lang"] == L].copy()
        sub = sub[sub["concept"].isin(concepts)]
        if sub.empty:
            continue

        # process in batches of forms
        forms = sub["form"].tolist()
        concept_ids = [concept_to_i[c] for c in sub["concept"].tolist()]

        bs = extractor.cfg.batch_size
        for start in range(0, len(forms), bs):
            batch_forms = forms[start : start + bs]
            batch_cids = concept_ids[start : start + bs]

            pooled_by_layer = extractor.get_hidden_states(
                batch_forms,
                layers=layers,
                pooling=pooling,
                return_token_level=False,
                return_tokenization_meta=False,
            )  # dict layer -> [B, H]

            # move to cpu numpy
            for ly, mat in pooled_by_layer.items():
                mat_np = mat.detach().cpu().numpy()
                for j in range(mat_np.shape[0]):
                    cidx = batch_cids[j]
                    forms_by_lang_layer[L][ly][cidx].append(mat_np[j])

    return forms_by_lang_layer

def concept_pool(forms_list: List[List[np.ndarray]]) -> np.ndarray:
    """
    forms_list: list over concepts; each is list of vectors
    returns X: (n_concepts, d)
    """
    X = []
    for fs in forms_list:
        if len(fs) == 0:
            # should not happen if concepts are common, but be defensive
            X.append(np.full((1,), np.nan))
        else:
            X.append(np.mean(np.stack(fs, axis=0), axis=0))
    X = np.stack(X, axis=0)
    if np.isnan(X).any():
        raise ValueError("Found missing concept vectors (nan). Check lexicon completeness after intersection.")
    return X


# -------------------------
# Plotting
# -------------------------
def plot_summary(out_dir: Path, df_pairs: pd.DataFrame, df_rel: pd.DataFrame, title: str) -> None:
    import matplotlib.pyplot as plt

    # mean across pairs per layer with 95% CI from bootstrap not included here; we plot mean±sem of obs
    g = df_pairs.groupby("layer")["rho_obs"]
    layers = np.array(sorted(g.groups.keys()))
    mean = np.array([g.get_group(l).mean() for l in layers])
    sem = np.array([g.get_group(l).std(ddof=1) / np.sqrt(len(g.get_group(l))) for l in layers])

    plt.figure()
    plt.plot(layers, mean)
    plt.fill_between(layers, mean - 1.96 * sem, mean + 1.96 * sem, alpha=0.2)
    plt.xlabel("layer")
    plt.ylabel("mean rho_obs across language pairs")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / "mean_rho_across_pairs.png", dpi=200)
    plt.close()

    # reliability per language: mean across repeats
    if not df_rel.empty:
        plt.figure()
        for L in sorted(df_rel["lang"].unique()):
            sub = df_rel[df_rel["lang"] == L].sort_values("layer")
            plt.plot(sub["layer"].values, sub["rho_mean"].values, label=L)
        plt.xlabel("layer")
        plt.ylabel("split-half reliability (rho)")
        plt.title(f"{title} | split-half reliability")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "split_half_reliability.png", dpi=200)
        plt.close()


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon_csv", type=str, required=True, help="CSV with columns: concept, lang, form")
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", type=int, required=True, help="HF hidden_states indices you want (e.g., 1..N)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=32)
    ap.add_argument("--pooling", type=str, default=None, help='mean|cls|last; default from extractor')
    ap.add_argument("--metric", type=str, default="cosine", help="pdist metric for RDM (cosine recommended)")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--perm_seed", type=int, default=0)
    ap.add_argument("--save_null", action="store_true", help="also save null arrays (can be large)")
    ap.add_argument("--n_repeats", type=int, default=200)
    ap.add_argument("--rel_seed", type=int, default=0)
    ap.add_argument("--min_forms", type=int, default=2)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load lexicon
    df = load_lexicon_csv(Path(args.lexicon_csv), args.langs)
    concepts, concept_to_i = build_concept_index(df, args.langs)

    # extractor
    cfg = ExtractionConfig(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    extractor = HiddenStateExtractor(cfg)

    # save config snapshot
    snap = {
        "args": vars(args),
        "extraction_config": asdict(cfg),
        "model_config": extractor.model_config_dict,
        "tokenizer_config": extractor.tokenizer_config_dict,
        "n_concepts_common": len(concepts),
    }
    (out_dir / "run_config.json").write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "concepts_common.json").write_text(json.dumps(concepts, ensure_ascii=False, indent=2), encoding="utf-8")

    # extract form vectors for all langs & layers
    forms_by_lang_layer = extract_form_vectors(
        extractor=extractor,
        df=df,
        concepts=concepts,
        concept_to_i=concept_to_i,
        langs=args.langs,
        layers=args.layers,
        pooling=args.pooling,
    )

    # concept pooling -> X[lang][layer] and RDM[lang][layer]
    X: Dict[str, Dict[int, np.ndarray]] = {L: {} for L in args.langs}
    R: Dict[str, Dict[int, np.ndarray]] = {L: {} for L in args.langs}
    for L in args.langs:
        for ly in args.layers:
            X[L][ly] = concept_pool(forms_by_lang_layer[L][ly])
            R[L][ly] = rdm_from_embeddings(X[L][ly], metric=args.metric)

    # cross-language pair stats per layer
    pair_rows = []
    null_store = {}  # key -> null array
    for L1, L2 in itertools.combinations(args.langs, 2):
        for ly in args.layers:
            obs, null, p, z = perm_null_rdm_similarity(
                R[L1][ly], R[L2][ly], n_perm=args.n_perm, seed=args.perm_seed
            )
            pair_rows.append(
                {
                    "model": args.model_name,
                    "lang1": L1,
                    "lang2": L2,
                    "layer": ly,
                    "rho_obs": obs,
                    "p_two_sided": p,
                    "z_vs_null": z,
                    "n_concepts": len(concepts),
                    "n_perm": args.n_perm,
                }
            )
            if args.save_null:
                null_store[f"{L1}-{L2}-L{ly}"] = null

    df_pairs = pd.DataFrame(pair_rows)
    df_pairs.to_csv(out_dir / "pairwise_perm_stats.csv", index=False)

    if args.save_null:
        np.savez_compressed(out_dir / "pairwise_null_rhos.npz", **null_store)

    # within-language split-half reliability
    rel_rows = []
    for L in args.langs:
        for ly in args.layers:
            res = split_half_reliability_rdm(
                forms_embeds=forms_by_lang_layer[L][ly],
                metric=args.metric,
                n_repeats=args.n_repeats,
                min_forms=args.min_forms,
                seed=args.rel_seed,
            )
            if res is None:
                # no enough multi-form concepts; record skip
                rel_rows.append(
                    {
                        "model": args.model_name,
                        "lang": L,
                        "layer": ly,
                        "rho_mean": np.nan,
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                        "n_repeats": args.n_repeats,
                        "min_forms": args.min_forms,
                        "note": "skipped (insufficient concepts with >=min_forms forms)",
                    }
                )
            else:
                rhos, mean, (lo, hi) = res
                rel_rows.append(
                    {
                        "model": args.model_name,
                        "lang": L,
                        "layer": ly,
                        "rho_mean": mean,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "n_repeats": args.n_repeats,
                        "min_forms": args.min_forms,
                        "note": "",
                    }
                )

    df_rel = pd.DataFrame(rel_rows)
    df_rel.to_csv(out_dir / "split_half_reliability.csv", index=False)

    # optional plots
    if args.plot:
        title = f"{args.model_name} | RDM Spearman | metric={args.metric}"
        plot_summary(out_dir, df_pairs, df_rel.dropna(subset=["rho_mean"]), title)

    # small console summary
    # (avoid pandas heavy printing; keep it readable)
    best = df_pairs.sort_values("rho_obs", ascending=False).head(10)
    (out_dir / "top10_pairs_layers.json").write_text(best.to_json(orient="records", indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
