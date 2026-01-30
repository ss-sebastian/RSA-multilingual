#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/extract.py

Thin wrapper that re-exports extraction utilities from src/rsa_multilingual.
Keeps your old import paths working if you were doing:
  from scripts.extract import extract_transformer_hidden_states
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo_root/src is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rsa_multilingual.extraction import (  # noqa: F401
    ExtractionConfig,
    HiddenStateExtractor,
    extract_transformer_hidden_states,
)

__all__ = [
    "ExtractionConfig",
    "HiddenStateExtractor",
    "extract_transformer_hidden_states",
]
