"""NHIS FairBias application-study data foundation and temporal interface."""

from .adapter import DISABILITY_COMPONENTS, NHISStudyAdapter
from .preprocessing import (
    NHISLeakageError,
    NHISPreprocessingError,
    NHISPreprocessor,
    construct_empwrkft_series,
)

__all__ = [
    "__version__",
    "NHISStudyAdapter",
    "NHISPreprocessor",
    "construct_empwrkft_series",
    "DISABILITY_COMPONENTS",
    "NHISLeakageError",
    "NHISPreprocessingError",
]
__version__ = "0.2.0"
