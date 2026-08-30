"""FairBias: A mathematically sound, leakage-free, and reproducible algorithmic fairness benchmarking framework."""

from fairbias.config import FairBiasConfig
from fairbias.data import FairDataLoader
from fairbias.evaluator import FairEvaluator
from fairbias.pipeline import run_fairbias_pipeline

__version__ = "1.0.0"
__all__ = [
    "FairBiasConfig",
    "FairDataLoader",
    "FairEvaluator",
    "run_fairbias_pipeline",
]
