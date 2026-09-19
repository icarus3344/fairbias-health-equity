"""Benchmark V1 Architecture for NHIS Fairness Evaluation.

Provides:
- Data contracts and synthetic multi-arm cohort generator with complex survey design.
- 3-tier semantic, geometric, and one-hot preprocessing with reserved tokens.
- Survey-weighted metrics (Balanced Accuracy, Demographic Parity Gap, Equalized Odds Gap).
- Rescaled PSU bootstrap engine for survey design inference.
- Standardized fairness method adapters wrapping Fairlearn, AIF360, and FairBias.
- Candidate configuration selection on Set S and frozen evaluation on Set T.
"""

from __future__ import annotations

__version__ = "1.0.0"
