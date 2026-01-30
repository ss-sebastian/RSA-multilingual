from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class ExtractionConfig:
    model_name: str
    device: str = "cpu"
    max_length: int = 32
    batch_size: int = 32
    default_pooling: str = "mean"   # mean|cls|last
    torch_dtype: Optional[str] = None
    trust_remote_code: bool = False


def _resolve_layer_index(hidden_states_len: int, layer: int) -> int:
    """
    hidden_states[0] = embeddings
    hidden_states[1..N] = blocks
    hidden_states[-1] = last block
    We accept negative indices like python.
    """
    if -hidden_states_len <= layer < hidden_states_len:
        return layer
    raise ValueError(f"Layer index out of range: {layer} for hidden_states_len={hidden_states_len}")


def _token_pooling(
    hs: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: set[int],
    mode: str,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    hs: [B, T, H]
    returns: [B, H]
    """
    mode = mode.lower()
    B, T, H = hs.shape

    # mask non-pad first
    mask = attention_mask.bool()  # [B,T]

    # exclude special tokens from mean pooling
    if mode == "mean":
        # special mask
        is_special = torch.zeros_like(mask)
        for sid in special_ids:
            is_special |= (input_ids == sid)
        use = mask & (~is_special)
        # fallback: if everything is special (rare), fall back to attention mask only
        denom = use.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (hs * use.unsqueeze(-1)).sum(dim=1) / denom
        return pooled

    if mode == "cls":
        return hs[:, 0, :]

    if mode == "last":
        # last non-pad token
        idx = mask.long().sum(dim=1) - 1
        idx = idx.clamp_min(0)
        out = hs[torch.arange(B, device=hs.device), idx, :]
        return out

    raise ValueError("word_pooling must be one of: mean|cls|last")


class HiddenStateExtractor:
    """
    HuggingFace Transformer hidden-state extractor.
    """

    def __init__(self, cfg: ExtractionConfig):
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, use_fast=True, trust_remote_code=cfg.trust_remote_code
        )

        if self.tokenizer.pad_token is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        torch_dtype = None
        if cfg.torch_dtype is not None:
            if not hasattr(torch, cfg.torch_dtype):
                raise ValueError(f"Unknown torch dtype: {cfg.torch_dtype}")
            torch_dtype = getattr(torch, cfg.torch_dtype)

        self.model = AutoModel.from_pretrained(
            cfg.model_name,
            output_hidden_states=True,
            torch_dtype=torch_dtype,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.model.to(cfg.device)
        self.model.eval()

        if len(self.tokenizer) > self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self._special_ids = set(getattr(self.tokenizer, "all_special_ids", []))

    @torch.no_grad()
    def get_word_vectors(
        self,
        words: List[str],
        layers: List[int],
        word_pooling: Optional[str] = None,
        return_meta: bool = False,
    ) -> Tuple[Dict[int, np.ndarray], Optional[dict]]:
        """
        Return per-layer vectors for each input word.
        Output: {layer: [N,H]}  (numpy float64 on CPU)
        """
        pooling = (word_pooling or self.cfg.default_pooling).lower()
        bs = int(self.cfg.batch_size)
        max_len = int(self.cfg.max_length)

        out: Dict[int, List[np.ndarray]] = {li: [] for li in layers}
        meta = {"words": words, "layers": layers, "word_pooling": pooling, "max_length": max_len} if return_meta else None

        for start in range(0, len(words), bs):
            batch = words[start : start + bs]
            toks = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            toks = {k: v.to(self.cfg.device) for k, v in toks.items()}

            outputs = self.model(**toks)
            hs_all = outputs.hidden_states  # tuple length L+1; each [B,T,H]
            hs_len = len(hs_all)

            for li in layers:
                li2 = _resolve_layer_index(hs_len, li)
                hs = hs_all[li2]
                pooled = _token_pooling(
                    hs=hs,
                    input_ids=toks["input_ids"],
                    attention_mask=toks["attention_mask"],
                    special_ids=self._special_ids,
                    mode=pooling,
                )
                out[li].append(pooled.detach().cpu().numpy().astype(np.float64))

        out2 = {li: np.concatenate(chunks, axis=0) for li, chunks in out.items()}
        return out2, meta

    @torch.no_grad()
    def get_concept_vectors(
        self,
        concept_to_forms: Dict[str, List[str]],
        layers: List[int],
        word_pooling: Optional[str] = None,
        concept_pooling: str = "mean",
        return_meta: bool = False,
    ) -> Tuple[Dict[int, np.ndarray], Optional[dict]]:
        """
        For each concept with multiple surface forms, compute form vectors then pool across forms.
        Output: {layer: [N,H]} (concept order = insertion order of dict)
        """
        concept_pooling = concept_pooling.lower()
        if concept_pooling != "mean":
            raise ValueError("concept_pooling currently supports only 'mean'")

        concept_ids = list(concept_to_forms.keys())
        # flatten all forms with ownership indices
        all_forms: List[str] = []
        owners: List[int] = []
        for ci, cid in enumerate(concept_ids):
            forms = concept_to_forms[cid]
            for f in forms:
                all_forms.append(str(f))
                owners.append(ci)

        word_vecs_by_layer, word_meta = self.get_word_vectors(
            all_forms, layers=layers, word_pooling=word_pooling, return_meta=True
        )

        N = len(concept_ids)
        out: Dict[int, np.ndarray] = {}

        owners_np = np.asarray(owners, dtype=int)
        for li in layers:
            W = word_vecs_by_layer[li]  # [n_forms, H]
            H = W.shape[1]
            C = np.zeros((N, H), dtype=np.float64)
            counts = np.zeros((N, 1), dtype=np.float64)

            for k in range(W.shape[0]):
                i = owners_np[k]
                C[i] += W[k]
                counts[i] += 1.0

            counts[counts == 0] = 1.0
            C = C / counts
            out[li] = C

        meta = None
        if return_meta:
            meta = {
                "concept_ids": concept_ids,
                "concept_to_forms": concept_to_forms,
                "owners": owners,
                "word_meta": word_meta,
            }
        return out, meta

    def get_config_page(self) -> dict:
        return asdict(self.cfg)

    def save_config_page(self, out_path: Union[str, Path]) -> None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.get_config_page(), ensure_ascii=False, indent=2), encoding="utf-8")


class extract_transformer_hidden_states:
    """
    Compatibility wrapper: keep your current scripts' usage pattern.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 32,
        batch_size: int = 32,
        default_pooling: str = "mean",
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        cfg = ExtractionConfig(
            model_name=model_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            default_pooling=default_pooling,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.extractor = HiddenStateExtractor(cfg)

    def get_word_vectors(self, *args, **kwargs):
        return self.extractor.get_word_vectors(*args, **kwargs)

    def get_concept_vectors(self, *args, **kwargs):
        return self.extractor.get_concept_vectors(*args, **kwargs)

    def get_config_page(self) -> dict:
        return self.extractor.get_config_page()

    def save_config_page(self, out_path: Union[str, Path]) -> None:
        self.extractor.save_config_page(out_path)
