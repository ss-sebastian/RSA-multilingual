from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from transformers import AutoModel, AutoTokenizer

Tensor = torch.Tensor


@dataclass
class ExtractionConfig:
    model_name: str
    device: str = "cpu"
    max_length: int = 32
    default_pooling: str = "mean"  # "mean" | "cls" | "last"
    batch_size: int = 32
    torch_dtype: Optional[str] = None  # e.g. "float16" if on GPU
    trust_remote_code: bool = False


class HiddenStateExtractor:
    """
    Unified hidden-state extractor for HuggingFace Transformers.

    Designed for your setting:
      - single-word (or single surface form) inputs
      - word-level pooling excluding special/pad tokens
      - concept pooling across multiple surface forms (e.g., cross-lingual synonyms)

    HF hidden_states indexing:
      hidden_states[0] = embeddings
      hidden_states[1..N] = transformer blocks
      hidden_states[-1] = last block
    """

    def __init__(self, cfg: ExtractionConfig):
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, use_fast=True, trust_remote_code=cfg.trust_remote_code
        )

        # Ensure pad_token exists for batching/padding
        if self.tokenizer.pad_token is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # dtype handling
        torch_dtype = None
        if cfg.torch_dtype is not None:
            torch_dtype = getattr(torch, cfg.torch_dtype)

        self.model = AutoModel.from_pretrained(
            cfg.model_name,
            output_hidden_states=True,
            torch_dtype=torch_dtype,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.model.to(cfg.device)
        self.model.eval()

        # If we added a special token, tokenizer size may exceed embedding size
        if len(self.tokenizer) > self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Robust decoder-only detection
        mcfg = getattr(self.model, "config", None)
        is_encoder_decoder = bool(getattr(mcfg, "is_encoder_decoder", False))
        is_decoder_flag = bool(getattr(mcfg, "is_decoder", False))
        # For decoder-only LMs, is_decoder True and is_encoder_decoder False
        self.is_decoder_only = (is_decoder_flag and not is_encoder_decoder)

        # cached config
        self.model_config_dict = self.model.config.to_dict() if hasattr(self.model, "config") else {}
        self.tokenizer_config_dict = {
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "is_fast": getattr(self.tokenizer, "is_fast", None),
            "vocab_size": getattr(self.tokenizer, "vocab_size", None),
            "model_max_length": getattr(self.tokenizer, "model_max_length", None),
            "pad_token": self.tokenizer.pad_token,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token": getattr(self.tokenizer, "eos_token", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
            "bos_token": getattr(self.tokenizer, "bos_token", None),
            "bos_token_id": getattr(self.tokenizer, "bos_token_id", None),
            "unk_token": getattr(self.tokenizer, "unk_token", None),
            "unk_token_id": getattr(self.tokenizer, "unk_token_id", None),
            "cls_token": getattr(self.tokenizer, "cls_token", None),
            "cls_token_id": getattr(self.tokenizer, "cls_token_id", None),
            "sep_token": getattr(self.tokenizer, "sep_token", None),
            "sep_token_id": getattr(self.tokenizer, "sep_token_id", None),
            "special_tokens_map": getattr(self.tokenizer, "special_tokens_map", None),
        }

    # -------------------------
    # Core: token->pooled vector
    # -------------------------

    @torch.no_grad()
    def get_hidden_states(
        self,
        text: Union[str, List[str]],
        layers: Optional[List[int]] = None,
        pooling: Optional[str] = None,
        return_token_level: bool = False,
        return_tokenization_meta: bool = False,
    ) -> Union[
        Dict[int, Tensor],
        Tuple[Dict[int, Tensor], Tuple[Tensor, ...], Dict[str, Tensor]],
        Tuple[Dict[int, Tensor], Dict[str, Any]],
        Tuple[Dict[int, Tensor], Tuple[Tensor, ...], Dict[str, Tensor], Dict[str, Any]],
    ]:
        """
        Extract pooled representations for selected layers.

        Pooling here is word-level pooling for single-form inputs:
        we exclude special tokens and padding.

        Args:
            text: string or list of strings
            layers: layer indices (default [-1])
            pooling: "mean" | "cls" | "last"
                - mean: mean over valid content tokens (non-pad & non-special)
                - cls: first token vector (for BERT-like; usually NOT "word vector")
                - last: last valid content token (non-pad & non-special), safer for decoder-only
            return_token_level: also return hidden_states tuple and inputs dict
            return_tokenization_meta: return a dict with tokens, token_ids, masks, token counts

        Returns:
            pooled_by_layer: dict layer_idx -> [B, H]
            optionally also token-level data and/or tokenization meta
        """
        texts = [text] if isinstance(text, str) else text
        if layers is None:
            layers = [-1]

        pooling = pooling or (("last" if self.is_decoder_only else self.cfg.default_pooling))
        pooling = pooling.lower()

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_special_tokens_mask=True,
            return_attention_mask=True,
        )
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states  # tuple of [B, T, H]

        attn_mask = inputs.get("attention_mask", None)
        special_mask = inputs.get("special_tokens_mask", None)

        pooled_by_layer: Dict[int, Tensor] = {}
        for li in layers:
            x = hidden_states[li]  # [B, T, H]
            pooled_by_layer[li] = self._pool(
                x=x,
                attention_mask=attn_mask,
                special_tokens_mask=special_mask,
                pooling=pooling,
            )

        meta: Optional[Dict[str, Any]] = None
        if return_tokenization_meta:
            meta = self._build_tokenization_meta(inputs)

        if return_token_level and return_tokenization_meta:
            return pooled_by_layer, hidden_states, inputs, meta
        if return_token_level:
            return pooled_by_layer, hidden_states, inputs
        if return_tokenization_meta:
            return pooled_by_layer, meta  # type: ignore
        return pooled_by_layer

    def _pool(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor],
        special_tokens_mask: Optional[Tensor],
        pooling: str,
    ) -> Tensor:
        """
        Pool token-level hidden states into a single vector per sequence.

        x: [B, T, H]
        attention_mask: [B, T] (1 real, 0 pad)
        special_tokens_mask: [B, T] (1 special, 0 normal)
        returns: [B, H]
        """
        B, T, H = x.shape

        # valid = non-pad & non-special
        if attention_mask is None:
            valid = torch.ones((B, T), dtype=torch.bool, device=x.device)
        else:
            valid = attention_mask.bool()

        if special_tokens_mask is not None:
            valid = valid & (~special_tokens_mask.bool())

        # Edge case: if a sequence somehow has zero valid tokens,
        # fallback to using attention_mask-only (non-pad) to avoid NaNs
        if valid.sum(dim=1).min().item() == 0:
            if attention_mask is not None:
                valid = attention_mask.bool()
            else:
                valid = torch.ones((B, T), dtype=torch.bool, device=x.device)

        if pooling == "mean":
            m = valid.unsqueeze(-1).float()  # [B,T,1]
            denom = m.sum(dim=1).clamp_min(1.0)  # [B,1]
            return (x * m).sum(dim=1) / denom  # [B,H]

        if pooling == "cls":
            # Note: for BERT-like models, x[:,0,:] corresponds to [CLS], not "word".
            return x[:, 0, :]

        if pooling in ("last", "last_token"):
            # Find last valid content token per sequence
            pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # [B,T]
            pos = pos.masked_fill(~valid, -1)
            idx = pos.max(dim=1).values.clamp_min(0)  # [B]
            return x[torch.arange(B, device=x.device), idx, :]  # [B,H]

        raise ValueError("Unknown pooling. Use one of: mean | cls | last")

    def _build_tokenization_meta(self, inputs: Dict[str, Tensor]) -> Dict[str, Any]:
        """
        Build tokenization metadata useful for later controls (token count, etc.).
        """
        input_ids = inputs["input_ids"].detach().cpu()
        attn = inputs.get("attention_mask", None)
        special = inputs.get("special_tokens_mask", None)

        tokens_per_item: List[List[str]] = []
        for row in input_ids:
            tokens_per_item.append(self.tokenizer.convert_ids_to_tokens(row.tolist()))

        meta: Dict[str, Any] = {
            "input_ids": input_ids,
            "tokens": tokens_per_item,
        }

        if attn is not None:
            meta["attention_mask"] = attn.detach().cpu()
        if special is not None:
            meta["special_tokens_mask"] = special.detach().cpu()

        # content token count = non-pad & non-special
        if attn is not None and special is not None:
            valid = attn.bool() & (~special.bool())
            meta["content_token_count"] = valid.sum(dim=1).detach().cpu()
        elif attn is not None:
            meta["content_token_count"] = attn.sum(dim=1).detach().cpu()

        return meta

    # -------------------------
    # Word vectors & concept pooling
    # -------------------------

    @torch.no_grad()
    def get_word_vectors(
        self,
        surface_forms: List[str],
        layers: Optional[List[int]] = None,
        pooling: str = "mean",
        return_meta: bool = False,
    ) -> Union[Dict[int, Tensor], Tuple[Dict[int, Tensor], List[Dict[str, Any]]]]:
        """
        Compute word vectors for multiple surface forms (batching internally).
        Returns dict: layer -> [N, H]
        """
        if layers is None:
            layers = [-1]

        pooling = pooling.lower()
        bs = max(1, self.cfg.batch_size)

        chunks = [surface_forms[i : i + bs] for i in range(0, len(surface_forms), bs)]
        out_by_layer: Dict[int, List[Tensor]] = {li: [] for li in layers}
        metas: List[Dict[str, Any]] = []

        for chunk in chunks:
            if return_meta:
                pooled, meta = self.get_hidden_states(
                    text=chunk,
                    layers=layers,
                    pooling=pooling,
                    return_token_level=False,
                    return_tokenization_meta=True,
                )
                metas.extend(self._split_meta_per_item(meta))
            else:
                pooled = self.get_hidden_states(
                    text=chunk,
                    layers=layers,
                    pooling=pooling,
                    return_token_level=False,
                    return_tokenization_meta=False,
                )

            for li in layers:
                out_by_layer[li].append(pooled[li].detach().cpu())

        stacked = {li: torch.cat(out_by_layer[li], dim=0) for li in layers}
        if return_meta:
            return stacked, metas
        return stacked

    def _split_meta_per_item(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert batch meta dict into a list of per-item meta dicts.
        """
        n = meta["input_ids"].shape[0]
        per: List[Dict[str, Any]] = []
        for i in range(n):
            item: Dict[str, Any] = {
                "input_ids": meta["input_ids"][i].tolist(),
                "tokens": meta["tokens"][i],
            }
            if "attention_mask" in meta:
                item["attention_mask"] = meta["attention_mask"][i].tolist()
            if "special_tokens_mask" in meta:
                item["special_tokens_mask"] = meta["special_tokens_mask"][i].tolist()
            if "content_token_count" in meta:
                item["content_token_count"] = int(meta["content_token_count"][i].item())
            per.append(item)
        return per

    @torch.no_grad()
    def get_concept_vectors(
        self,
        concept_to_forms: Dict[str, List[str]],
        layers: Optional[List[int]] = None,
        word_pooling: str = "mean",
        concept_pooling: str = "mean",  # for now: mean only
        return_meta: bool = False,
    ) -> Union[
        Dict[int, Tensor],
        Tuple[Dict[int, Tensor], Dict[str, Any]],
    ]:
        """
        Concept pooling across multiple surface forms (cross-lingual synonyms).

        Steps:
          1) for each surface form: word-level pooling over content tokens -> v_form
          2) for each concept: aggregate its forms -> v_concept
          3) returns dict layer -> [num_concepts, H] (concepts in fixed order)
        """
        if layers is None:
            layers = [-1]
        if concept_pooling.lower() != "mean":
            raise ValueError("Currently only concept_pooling='mean' is implemented (robust baseline).")

        concept_ids = list(concept_to_forms.keys())
        all_forms: List[str] = []
        form_owner: List[str] = []
        for cid in concept_ids:
            forms = concept_to_forms[cid]
            if len(forms) == 0:
                raise ValueError(f"Concept '{cid}' has 0 surface forms.")
            all_forms.extend(forms)
            form_owner.extend([cid] * len(forms))

        if return_meta:
            form_vecs_by_layer, metas = self.get_word_vectors(
                all_forms, layers=layers, pooling=word_pooling, return_meta=True
            )
        else:
            form_vecs_by_layer = self.get_word_vectors(
                all_forms, layers=layers, pooling=word_pooling, return_meta=False
            )
            metas = []

        # map concept -> indices in all_forms
        owner_to_indices: Dict[str, List[int]] = {cid: [] for cid in concept_ids}
        for i, cid in enumerate(form_owner):
            owner_to_indices[cid].append(i)

        # aggregate
        concept_vecs_by_layer: Dict[int, Tensor] = {}
        for li in layers:
            M = form_vecs_by_layer[li]  # [num_forms, H] on CPU
            concept_vecs: List[Tensor] = []
            for cid in concept_ids:
                idxs = owner_to_indices[cid]
                X = M[idxs, :]  # [k, H]
                concept_vecs.append(X.mean(dim=0, keepdim=True))  # [1, H]
            concept_vecs_by_layer[li] = torch.cat(concept_vecs, dim=0)  # [num_concepts, H]

        if return_meta:
            meta_out = {
                "concept_ids": concept_ids,
                "all_forms": all_forms,
                "form_owner": form_owner,
                "form_tokenization": metas,
                "owner_to_indices": owner_to_indices,
                "word_pooling": word_pooling,
                "concept_pooling": concept_pooling,
            }
            return concept_vecs_by_layer, meta_out

        return concept_vecs_by_layer

    # -------------------------
    # Config page helpers
    # -------------------------

    def get_config_page(self) -> dict:
        return {
            "model_name": self.cfg.model_name,
            "device": self.cfg.device,
            "max_length": self.cfg.max_length,
            "default_pooling": self.cfg.default_pooling,
            "batch_size": self.cfg.batch_size,
            "torch_dtype": self.cfg.torch_dtype,
            "trust_remote_code": self.cfg.trust_remote_code,
            "is_decoder_only": self.is_decoder_only,
            "model_config": self.model_config_dict,
            "tokenizer_config": self.tokenizer_config_dict,
        }

    def save_config_page(self, out_path: Union[str, Path]) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.get_config_page(), ensure_ascii=False, indent=2))


class extract_transformer_hidden_states:
    """
    Wrapper keeping your original usage pattern, plus concept pooling.

    Example:
        ext = extract_transformer_hidden_states("xlm-roberta-base", device="cpu")
        # word vectors
        word_vecs = ext.get_word_vectors(["casa", "house", "房子"], layers=[-1, -6], word_pooling="mean")
        # concept vectors
        concept_to_forms = {
            "HOUSE": ["house", "casa", "房子", "住宅"],
            "DOG": ["dog", "perro", "狗"],
        }
        concept_vecs, meta = ext.get_concept_vectors(concept_to_forms, layers=[-1], return_meta=True)
        ext.save_config_page("artifacts/xlm-roberta-base_config.json")
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

    def get_hidden_states(self, *args, **kwargs):
        return self.extractor.get_hidden_states(*args, **kwargs)

    def get_word_vectors(
        self,
        surface_forms: List[str],
        layers: Optional[List[int]] = None,
        word_pooling: str = "mean",
        return_meta: bool = False,
    ):
        return self.extractor.get_word_vectors(
            surface_forms=surface_forms,
            layers=layers,
            pooling=word_pooling,
            return_meta=return_meta,
        )

    def get_concept_vectors(
        self,
        concept_to_forms: Dict[str, List[str]],
        layers: Optional[List[int]] = None,
        word_pooling: str = "mean",
        concept_pooling: str = "mean",
        return_meta: bool = False,
    ):
        return self.extractor.get_concept_vectors(
            concept_to_forms=concept_to_forms,
            layers=layers,
            word_pooling=word_pooling,
            concept_pooling=concept_pooling,
            return_meta=return_meta,
        )

    def get_config_page(self) -> dict:
        return self.extractor.get_config_page()

    def save_config_page(self, out_path: Union[str, Path]) -> None:
        self.extractor.save_config_page(out_path)
