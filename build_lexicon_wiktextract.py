#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a multilingual aligned lexicon (concept -> word per language) from
kaikki.org "raw-wiktextract-data.jsonl.gz" (English Wiktionary edition).

Key idea:
- Use English entries as pivot.
- Use sense-level translations to define a "concept key" (en_word, pos, gloss).
- Auto-select K languages among a candidate list (taken from the Kaikki rawdata page language codes)
  to maximize K-language intersection size.
- Filter to clean single-word noun translations.
- Output CSV/JSON with exactly K languages (+ English pivot).

Citations (for your paper):
- Wiktextract (LREC 2022) and raw data from kaikki.org.

This script does not download data for you (no internet needed); you pass --in_gz.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# -------------------------
# Candidate languages
# -------------------------
# These codes mirror the "Raw downloads for other Wiktionary editions" visible on
# https://kaikki.org/dictionary/rawdata.html (not exhaustive of the English dump; it's a candidate set).
# You can edit/extend this list.
DEFAULT_CANDIDATE_LANGS = [
    "zh", "cs", "nl", "fr", "de", "el", "id", "it", "ja", "ko", "ku", "ms",
    "pl", "pt", "ru", "es", "th", "tr", "vi", "simple"
]


# -------------------------
# Filtering utilities
# -------------------------
_BAD_CHARS = set("()[]{};,:/\\|<>\"")
_SPACE_RE = re.compile(r"\s+")
_PAREN_RE = re.compile(r"[\(\)\[\]\{\}]")
_DIGIT_RE = re.compile(r"\d")


def is_clean_single_word(s: str, allow_hyphen: bool = False) -> bool:
    """
    Conservative single-word filter:
    - no whitespace
    - no bracket/paren/annotation punctuation
    - no commas/semicolons/slashes etc
    - optionally disallow hyphens
    """
    if not s:
        return False
    s = s.strip()
    if not s:
        return False
    if _SPACE_RE.search(s):
        return False
    if any(ch in _BAD_CHARS for ch in s):
        return False
    # Many Wiktionary translations include qualifiers like "犬 (いぬ)" etc; reject parentheses.
    if _PAREN_RE.search(s):
        return False
    if not allow_hyphen and "-" in s:
        return False
    # Remove obvious "alt-form" markers, pipes, etc
    if "|" in s or "=" in s:
        return False
    return True


def lemma_score(s: str) -> Tuple[int, int, int]:
    """
    Lower is better.
    Heuristic:
    - penalize digits
    - penalize punctuation-ish characters
    - shorter is better
    """
    digit_pen = 1 if _DIGIT_RE.search(s) else 0
    punct_pen = sum(1 for ch in s if not ch.isalnum() and ch not in ("'", "’", "·", "・"))
    return (digit_pen, punct_pen, len(s))


def choose_best_lemma(cands: List[str]) -> Optional[str]:
    cands2 = [c.strip() for c in cands if c and c.strip()]
    if not cands2:
        return None
    cands2.sort(key=lemma_score)
    return cands2[0]


# -------------------------
# Parsing wiktextract raw JSON
# -------------------------
@dataclass(frozen=True)
class Concept:
    concept_id: str
    en: str
    pos: str
    gloss: str


def iter_jsonl_gz(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def get_english_entry_fields(obj: dict) -> Optional[Tuple[str, str, List[dict]]]:
    """
    Return (en_word, pos, senses) or None.
    Raw format varies; we try robustly:
    - obj["lang"] may be "English"
    - obj["word"] is lemma
    - obj["pos"] is part-of-speech
    - obj["senses"] is list
    """
    lang = obj.get("lang") or obj.get("language")
    if lang != "English":
        return None
    en_word = obj.get("word") or obj.get("title") or obj.get("lemma")
    if not en_word:
        return None
    pos = obj.get("pos") or obj.get("part_of_speech") or ""
    senses = obj.get("senses")
    if not isinstance(senses, list) or not senses:
        return None
    return en_word, pos, senses


def sense_gloss(sense: dict) -> str:
    g = sense.get("glosses") or sense.get("gloss") or sense.get("gloss_text")
    if isinstance(g, list) and g:
        return str(g[0])
    if isinstance(g, str):
        return g
    # fallback: try "raw_glosses" or "sense" id
    g2 = sense.get("raw_glosses")
    if isinstance(g2, list) and g2:
        return str(g2[0])
    sid = sense.get("id") or sense.get("senseid") or ""
    return str(sid) if sid else "NO_GLOSS"


def extract_translations(sense: dict) -> List[dict]:
    """
    Wiktextract raw translations often appear as:
    - sense["translations"] : list of dicts with keys like "lang", "lang_code", "word"
    Sometimes nested. We handle common cases.
    """
    tr = sense.get("translations")
    if isinstance(tr, list):
        return tr
    # some versions store under "translations" inside other dicts
    if isinstance(tr, dict):
        # flatten possible dict-of-lists
        out = []
        for v in tr.values():
            if isinstance(v, list):
                out.extend(v)
        return out
    return []

def extract_entry_translations(obj: dict) -> List[dict]:
    tr = obj.get("translations")
    return tr if isinstance(tr, list) else []


def tr_lang_code(t: dict) -> Optional[str]:
    # try likely keys
    code = t.get("lang_code") or t.get("langcode") or t.get("code")
    if isinstance(code, str) and code:
        return code
    # sometimes only language name is present (rare); we ignore those
    return None


def tr_word(t: dict) -> Optional[str]:
    w = t.get("word") or t.get("translation") or t.get("term")
    if isinstance(w, str) and w:
        return w
    return None


def build_concept_id(en_word: str, pos: str, gloss: str) -> str:
    # A stable, readable key; you can later hash if you prefer.
    # Keep it short-ish to avoid insane CSV width.
    gloss_norm = " ".join(gloss.split())[:80]
    return f"{en_word}||{pos}||{gloss_norm}"


# -------------------------
# Mask counting and subset selection
# -------------------------
def subset_intersection_size(mask_counts: Dict[int, int], subset_mask: int) -> int:
    """
    Intersection size for a subset S:
    sum over all masks M that cover S (M & S == S) of count[M].
    """
    total = 0
    for m, c in mask_counts.items():
        if (m & subset_mask) == subset_mask:
            total += c
    return total


def greedy_select_languages(
    mask_counts: Dict[int, int],
    lang_to_bit: Dict[str, int],
    k: int,
    min_intersection: int,
    prefer: Optional[List[str]] = None,
) -> List[str]:
    """
    Greedy: start with preferred languages if possible, then iteratively add
    the language that maximizes the resulting intersection size.
    """
    bits = list(lang_to_bit.items())

    chosen: List[str] = []
    chosen_mask = 0

    # Try to include preferred languages first (in order) if they exist
    if prefer:
        for l in prefer:
            if l in lang_to_bit and l not in chosen:
                candidate_mask = chosen_mask | (1 << lang_to_bit[l])
                # Only accept if intersection doesn't collapse below min_intersection too early
                inter = subset_intersection_size(mask_counts, candidate_mask)
                if inter >= max(1, min_intersection // 10):  # mild guard, not too strict
                    chosen.append(l)
                    chosen_mask = candidate_mask
                    if len(chosen) == k:
                        return chosen

    # Fill remaining slots
    remaining = [l for l, _ in bits if l not in chosen]
    while len(chosen) < k and remaining:
        best_l = None
        best_inter = -1

        for l in remaining:
            cand_mask = chosen_mask | (1 << lang_to_bit[l])
            inter = subset_intersection_size(mask_counts, cand_mask)
            if inter > best_inter:
                best_inter = inter
                best_l = l

        if best_l is None:
            break

        # stop if adding any language kills the intersection too much
        if len(chosen) >= 1 and best_inter < 1:
            break

        chosen.append(best_l)
        chosen_mask |= (1 << lang_to_bit[best_l])
        remaining.remove(best_l)

    # Final guard: if chosen k but intersection too small, we still return chosen
    # and later stage will tell you actual intersection and you can relax filters or lower k.
    return chosen


# -------------------------
# Reservoir sampling
# -------------------------
class Reservoir:
    def __init__(self, k: int, seed: int):
        self.k = k
        self.rng = random.Random(seed)
        self.items: List[dict] = []
        self.n_seen = 0

    def add(self, item: dict):
        self.n_seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
            return
        j = self.rng.randint(1, self.n_seen)
        if j <= self.k:
            self.items[j - 1] = item


# -------------------------
# Main pipeline
# -------------------------
def extract_entry_translations(obj: dict) -> List[dict]:
    """
    In your Kaikki enwiktionary raw JSONL, translations live at the ENTRY level:
      obj["translations"] : list
    """
    tr = obj.get("translations")
    return tr if isinstance(tr, list) else []


def pass1_mask_count(
    in_gz: Path,
    candidate_langs: List[str],
    pos_keep: str,
    allow_hyphen: bool,
) -> Tuple[Dict[int, int], Dict[str, int], Dict[str, int]]:
    """
    First pass (ENTRY-level translations):
    - For each English entry (filtered by POS), compute which candidate languages have a clean single-word translation.
    - Increment mask_counts[mask] += 1 where mask denotes the set of languages present.
    - Also track per-language coverage counts.
    """
    candidate_langs = [l for l in candidate_langs if l != "en"]
    lang_to_bit = {l: i for i, l in enumerate(candidate_langs)}

    mask_counts: Dict[int, int] = {}
    coverage: Dict[str, int] = {l: 0 for l in candidate_langs}

    for obj in iter_jsonl_gz(in_gz):
        parsed = get_english_entry_fields(obj)
        if not parsed:
            continue
        en_word, pos, senses = parsed

        if pos_keep:
            if (pos or "").strip().lower() != pos_keep.strip().lower():
                continue

        trs = extract_entry_translations(obj)
        if not trs:
            continue

        # collect candidate translations per language
        per_lang: Dict[str, List[str]] = {}
        for t in trs:
            code = tr_lang_code(t)
            if not code or code not in lang_to_bit:
                continue
            w = tr_word(t)
            if not w:
                continue
            if not is_clean_single_word(w, allow_hyphen=allow_hyphen):
                continue
            per_lang.setdefault(code, []).append(w)

        mask = 0
        for l, cands in per_lang.items():
            best = choose_best_lemma(cands)
            if best is None:
                continue
            mask |= (1 << lang_to_bit[l])

        if mask == 0:
            continue

        mask_counts[mask] = mask_counts.get(mask, 0) + 1

        # per-language coverage: count a hit if language present in this entry
        for l in candidate_langs:
            if mask & (1 << lang_to_bit[l]):
                coverage[l] += 1

    return mask_counts, lang_to_bit, coverage


def pass2_extract_lexicon(
    in_gz: Path,
    selected_langs: List[str],
    lang_to_bit: Dict[str, int],
    target: int,
    seed: int,
    pos_keep: str,
    allow_hyphen: bool,
) -> List[dict]:
    """
    Second pass (ENTRY-level translations):
    - Keep only entries where ALL selected languages have clean translations.
    - Use reservoir sampling to cap at target.
    """
    reservoir = Reservoir(k=target, seed=seed)

    for obj in iter_jsonl_gz(in_gz):
        parsed = get_english_entry_fields(obj)
        if not parsed:
            continue
        en_word, pos, senses = parsed

        if pos_keep:
            if (pos or "").strip().lower() != pos_keep.strip().lower():
                continue

        # English lemma filter (optional): keep it clean-ish too
        if not is_clean_single_word(en_word, allow_hyphen=True):
            continue

        trs = extract_entry_translations(obj)
        if not trs:
            continue

        per_lang: Dict[str, List[str]] = {}
        for t in trs:
            code = tr_lang_code(t)
            if not code or code not in selected_langs:
                continue
            w = tr_word(t)
            if not w:
                continue
            if not is_clean_single_word(w, allow_hyphen=allow_hyphen):
                continue
            per_lang.setdefault(code, []).append(w)

        # ensure full intersection
        if any(l not in per_lang for l in selected_langs):
            continue

        # use the first sense gloss if available; otherwise NO_GLOSS
        gloss = sense_gloss(senses[0]) if senses else "NO_GLOSS"

        row = {
            "concept_id": build_concept_id(en_word, pos, gloss),
            "en": en_word,
            "pos": pos,
            "gloss": gloss,
        }

        ok = True
        for l in selected_langs:
            best = choose_best_lemma(per_lang[l])
            if best is None:
                ok = False
                break
            row[l] = best
        if not ok:
            continue

        reservoir.add(row)

    return reservoir.items

def write_outputs(rows: List[dict], selected_langs: List[str], out_prefix: Path):
    out_csv = out_prefix.with_suffix(".csv")
    out_json = out_prefix.with_suffix(".json")

    fieldnames = ["concept_id", "en"] + selected_langs
    # CSV
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # JSON (your preferred lexicon format)
    out = {
        "concept": [r["concept_id"] for r in rows],
        "en": [r["en"] for r in rows],
    }
    for l in selected_langs:
        out[l] = [r[l] for r in rows]

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_gz", type=str, required=True, help="Path to raw-wiktextract-data.jsonl.gz")
    ap.add_argument("--k", type=int, default=6, help="Number of non-English languages to select")
    ap.add_argument("--target", type=int, default=1000, help="Number of concepts to export (reservoir sample)")
    ap.add_argument("--pos", type=str, default="noun", help="Keep only this POS (e.g., noun). Use '' to disable.")
    ap.add_argument("--seed", type=int, default=13, help="Random seed (for sampling)")
    ap.add_argument("--allow_hyphen", action="store_true", help="Allow hyphens in translations (default: False)")
    ap.add_argument("--min_intersection", type=int, default=1000, help="Preferred minimum K-lang intersection size")
    ap.add_argument("--candidate_langs", type=str, default="",
                    help="Comma-separated candidate lang codes. Default: languages listed on the Kaikki rawdata page.")
    ap.add_argument("--prefer", type=str, default="",
                    help="Comma-separated preferred langs to try include early (e.g. 'es,fr,de,zh,ja').")
    ap.add_argument("--out_prefix", type=str, default="lexicon_6lang_1000", help="Output prefix (no extension)")
    args = ap.parse_args()

    in_gz = Path(args.in_gz)
    if not in_gz.exists():
        raise FileNotFoundError(f"Missing file: {in_gz}")

    candidate_langs = DEFAULT_CANDIDATE_LANGS[:]
    if args.candidate_langs.strip():
        candidate_langs = [x.strip() for x in args.candidate_langs.split(",") if x.strip()]

    # The English raw dump contains hundreds of languages; we *restrict* candidate set to avoid silly selection.
    # If you really want 'any language in data', set --candidate_langs '' and then edit DEFAULT list to huge,
    # or implement a dynamic discovery stage.
    print(f"[info] candidate langs ({len(candidate_langs)}): {candidate_langs}")

    prefer = [x.strip() for x in args.prefer.split(",") if x.strip()] if args.prefer.strip() else None

    print("[pass1] scanning for language coverage + mask counts...")
    mask_counts, lang_to_bit, coverage = pass1_mask_count(
        in_gz=in_gz,
        candidate_langs=candidate_langs,
        pos_keep=args.pos,
        allow_hyphen=args.allow_hyphen,
    )
    print(f"[pass1] observed masks: {len(mask_counts)}")
    # report top coverage
    top_cov = sorted(coverage.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print("[pass1] top language coverage (approx, after filters):")
    for l, c in top_cov:
        print(f"  {l:>5s}: {c}")

    print(f"[select] selecting k={args.k} languages to maximize K-lang intersection...")
    selected_langs = greedy_select_languages(
        mask_counts=mask_counts,
        lang_to_bit=lang_to_bit,
        k=args.k,
        min_intersection=args.min_intersection,
        prefer=prefer,
    )
    selected_mask = 0
    for l in selected_langs:
        selected_mask |= (1 << lang_to_bit[l])
    inter = subset_intersection_size(mask_counts, selected_mask)
    print(f"[select] selected langs: {selected_langs}")
    print(f"[select] estimated K-lang intersection size: {inter}")

    print("[pass2] extracting aligned lexicon rows...")
    rows = pass2_extract_lexicon(
        in_gz=in_gz,
        selected_langs=selected_langs,
        lang_to_bit=lang_to_bit,
        target=args.target,
        seed=args.seed,
        pos_keep=args.pos,
        allow_hyphen=args.allow_hyphen,
    )
    print(f"[pass2] extracted rows (after sampling cap={args.target}): {len(rows)}")
    if len(rows) < min(args.target, 200):
        print("[warn] got surprisingly few rows. You may need to relax filters or lower k.")
        print("       try: --allow_hyphen, or --pos '' (disable POS filter), or lower --k.")

    out_prefix = Path(args.out_prefix)
    write_outputs(rows, selected_langs, out_prefix)


if __name__ == "__main__":
    main()
