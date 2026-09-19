"""Benchmark method adapters wrapping standard open-source libraries.

Provides:
- UnmitigatedAdapter: scikit-learn standard baseline
- FairBiasAdapter: Tang et al. (2024) application-v1
- ReweighingAdapter: Kamiran & Calders (2012) via AIF360
- LFRAdapter: Zemel et al. (ICML 2013) via AIF360
- ExponentiatedGradientAdapter: Agarwal et al. (ICML 2018) via Fairlearn (DP & EO)
- ThresholdOptimizerAdapter: Hardt et al. (NeurIPS 2016) via Fairlearn (EO)
"""

from __future__ import annotations

from .adapter_fairbias import FairBiasAdapter
from .adapter_lfr import LFRAdapter
from .adapter_reductions import ExponentiatedGradientAdapter
from .adapter_reweighing import ReweighingAdapter
from .adapter_threshold_optimizer import ThresholdOptimizerAdapter
from .adapter_unmitigated import UnmitigatedAdapter
from .base import BaseMethodAdapter, NotSupportedError

__all__ = [
    "BaseMethodAdapter",
    "NotSupportedError",
    "UnmitigatedAdapter",
    "FairBiasAdapter",
    "ReweighingAdapter",
    "LFRAdapter",
    "ExponentiatedGradientAdapter",
    "ThresholdOptimizerAdapter",
]
