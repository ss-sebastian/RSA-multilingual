from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .lexicon import load_concept_map, load_target_groups
from .rdm import compute_rdm, rdm_upper_triangle, spearman_corr, categorical_target_rdm
from .utils import ensure_dir
from .extraction import extract_transformer_hidden_states


@dataclass
class RSAResult:
    concept_ids: List[str]
    layers: List[int]
    metric: str
    rdms: Dict[int, np.ndarray]                 # layer -> [N,N]
    rsa_long: pd.DataFrame                      # long form
    layerwise_rdm_similarity: Optional[pd.DataFrame]
    target_rdm: Optional[np.ndarray]
    layer_vs_target: Optional[pd.DataFrame]
    meta: Dict[str, Any]


def _make_long_table(layer: int, D: np.ndarray, labels: List[str]) -> pd.DataFrame:
    iu = np.triu_indices(D.shape[0], k=1)
    i, j = iu[0], iu[1]
    return pd.DataFrame(
        {
            "layer": layer,
            "i": i,
            "j": j,
            "item_i": [labels[k] for k in i],
            "item_j": [labels[k] for k in j],
            "dissimilarity": D[i, j],
        }
    )


def run_rsa(
    model_name: str,
    inputs: str,
    layers: List[int],
    metric: str = "correlation",
    word_pooling: str = "mean",
    concept_pooling: str = "mean",
    device: str = "cpu",
    max_length: int = 32,
    batch_size: int = 32,
    torch_dtype: Optional[str] = None,
    trust_remote_code: bool = False,
    target_groups: Optional[str] = None,
) -> RSAResult:
    concept_to_forms = load_concept_map(inputs)
    target_map = load_target_groups(target_groups)

    ext = extract_transformer_hidden_states(
        model_name=model_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        default_pooling=word_pooling,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    concept_vecs_by_layer, concept_meta = ext.get_concept_vectors(
        concept_to_forms=concept_to_forms,
        layers=layers,
        word_pooling=word_pooling,
        concept_pooling=concept_pooling,
        return_meta=True,
    )

    concept_ids = concept_meta.get("concept_ids") or list(concept_to_forms.keys())

    rdms: Dict[int, np.ndarray] = {}
    long_tables: List[pd.DataFrame] = []

    for li in layers:
        X = concept_vecs_by_layer[li]
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu()
        Xn = np.asarray(X, dtype=np.float64)

        D = compute_rdm(Xn, metric=metric)
        rdms[li] = D
        long_tables.append(_make_long_table(li, D, concept_ids))

    rsa_long = pd.concat(long_tables, ignore_index=True)

    layerwise_df = None
    if len(layers) >= 2:
        rows = []
        for a in range(len(layers)):
            for b in range(a + 1, len(layers)):
                la, lb = layers[a], layers[b]
                rho = spearman_corr(rdm_upper_triangle(rdms[la]), rdm_upper_triangle(rdms[lb]))
                rows.append({"layer_a": la, "layer_b": lb, "spearman_rho": rho})
        layerwise_df = pd.DataFrame(rows)

    target_rdm = None
    layer_vs_target = None
    if target_map is not None:
        target_rdm = categorical_target_rdm(concept_ids, target_map)
        vt = rdm_upper_triangle(target_rdm)
        rows = []
        for li in layers:
            rows.append({"layer": li, "spearman_rho_to_target": spearman_corr(rdm_upper_triangle(rdms[li]), vt)})
        layer_vs_target = pd.DataFrame(rows)

    meta = {
        "model_name": model_name,
        "device": device,
        "layers": layers,
        "metric": metric,
        "word_pooling": word_pooling,
        "concept_pooling": concept_pooling,
        "max_length": max_length,
        "batch_size": batch_size,
        "extractor_config": ext.get_config_page(),
        "concept_meta": concept_meta,
    }

    return RSAResult(
        concept_ids=concept_ids,
        layers=layers,
        metric=metric,
        rdms=rdms,
        rsa_long=rsa_long,
        layerwise_rdm_similarity=layerwise_df,
        target_rdm=target_rdm,
        layer_vs_target=layer_vs_target,
        meta=meta,
    )


def save_rsa_outputs(res: RSAResult, out_dir: str) -> Path:
    out = ensure_dir(out_dir)

    # provenance
    (out / "rsa_meta.json").write_text(json.dumps(res.meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # RDMs
    for li, D in res.rdms.items():
        pd.DataFrame(D, index=res.concept_ids, columns=res.concept_ids).to_csv(out / f"rdm_layer{li}.csv")
        np.save(out / f"rdm_layer{li}.npy", D)

    # long
    res.rsa_long.to_csv(out / "rsa_long.csv", index=False)

    # layerwise
    if res.layerwise_rdm_similarity is not None:
        res.layerwise_rdm_similarity.to_csv(out / "layerwise_rdm_similarity.csv", index=False)

    # target
    if res.target_rdm is not None:
        pd.DataFrame(res.target_rdm, index=res.concept_ids, columns=res.concept_ids).to_csv(out / "target_rdm.csv")
        np.save(out / "target_rdm.npy", res.target_rdm)

    if res.layer_vs_target is not None:
        res.layer_vs_target.to_csv(out / "layer_vs_target_rsa.csv", index=False)

    return out
