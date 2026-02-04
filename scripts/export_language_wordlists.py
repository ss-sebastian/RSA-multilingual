#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

SPACE_RE = re.compile(r"\s+")


def normalize_token(s: str, lowercase: bool = True, nfc: bool = True) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    if nfc:
        s = unicodedata.normalize("NFC", s)
    if lowercase:
        s = s.lower()
    s = SPACE_RE.sub(" ", s).strip()
    return s


def main():
    ap = argparse.ArgumentParser(description="Export per-language wordlists from a lexicon JSON.")
    ap.add_argument("--in_json", required=True, type=str, help="Input lexicon JSON (e.g., lexicon_6lang_1000.json)")
    ap.add_argument("--out_dir", default="artifacts/wordlists", type=str, help="Output directory")
    ap.add_argument("--no_lowercase", action="store_true", help="Do NOT lowercase")
    ap.add_argument("--no_nfc", action="store_true", help="Do NOT apply Unicode NFC")
    ap.add_argument("--min_len", default=1, type=int, help="Minimum token length")
    ap.add_argument("--drop_empty", action="store_true", help="Drop empty tokens after normalization (recommended)")
    ap.add_argument("--exclude_keys", default="concept,concept_id,pos,gloss", type=str,
                    help="Comma-separated keys to ignore if present")
    args = ap.parse_args()

    in_path = Path(args.in_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text(encoding="utf-8"))

    exclude = {k.strip() for k in args.exclude_keys.split(",") if k.strip()}

    # Expect: dict of arrays. We'll treat any key whose value is a list of same length as a language column.
    # We'll ignore excluded keys and any non-list values.
    lang_to_words: Dict[str, List[str]] = {}

    for k, v in data.items():
        if k in exclude:
            continue
        if not isinstance(v, list):
            continue
        lang_to_words[k] = v

    if not lang_to_words:
        raise RuntimeError(
            f"No language columns found in JSON. Keys present: {list(data.keys())[:50]}"
        )

    lowercase = not args.no_lowercase
    nfc = not args.no_nfc

    summary = []
    for lang, words in sorted(lang_to_words.items(), key=lambda kv: kv[0]):
        seen = set()
        out = []
        for w in words:
            t = normalize_token(w, lowercase=lowercase, nfc=nfc)
            if args.drop_empty and not t:
                continue
            if len(t) < args.min_len:
                continue
            if t not in seen:
                seen.add(t)
                out.append(t)

        # write txt: one token per line
        out_path = out_dir / f"wordlist_{lang}.txt"
        out_path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")

        summary.append((lang, len(words), len(out)))

    # write a small summary
    summary_path = out_dir / "summary.tsv"
    summary_lines = ["lang\trows_in\tunique_out"] + [f"{l}\t{a}\t{b}" for l, a, b in summary]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved to: {out_dir.resolve()}")
    for l, a, b in summary:
        print(f"{l}: {b} unique words (from {a} rows)")
    print("Files: wordlist_<lang>.txt + summary.tsv")


if __name__ == "__main__":
    main()
