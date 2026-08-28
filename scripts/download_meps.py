#!/usr/bin/env python3
"""Thin CLI entrypoint for MEPS artifact ingestion."""

from __future__ import annotations

import pathlib
import sys

# Deterministically add repository src directory derived from resolved script path
_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from meps_fairness.data.download import main

if __name__ == "__main__":
    sys.exit(main())
