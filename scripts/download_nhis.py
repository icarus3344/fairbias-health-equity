#!/usr/bin/env python3
"""CLI entrypoint for the official NHIS Sample Adult ZIP downloader."""

from __future__ import annotations

import pathlib
import sys

_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.download import main


if __name__ == "__main__":
    sys.exit(main())
