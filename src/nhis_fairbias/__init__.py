"""NHIS FairBias application-study data foundation, temporal interface, and pooled baseline."""

from .adapter import DISABILITY_COMPONENTS, NHISStudyAdapter
from .pooled import (
    DEFAULT_POOLED_SEED,
    EXPECTED_TEST_ROWS,
    EXPECTED_TOTAL_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VAL_ROWS,
    NHISPooledAdapter,
    audit_pooled_splits,
    generate_pooled_splits,
)
from .preprocessing import (
    NHISLeakageError,
    NHISPreprocessingError,
    NHISPreprocessor,
    construct_empwrkft_series,
)

__all__ = [
    "__version__",
    "NHISStudyAdapter",
    "NHISPooledAdapter",
    "NHISPreprocessor",
    "construct_empwrkft_series",
    "generate_pooled_splits",
    "audit_pooled_splits",
    "DISABILITY_COMPONENTS",
    "NHISLeakageError",
    "NHISPreprocessingError",
    "DEFAULT_POOLED_SEED",
    "EXPECTED_TOTAL_ROWS",
    "EXPECTED_TRAIN_ROWS",
    "EXPECTED_VAL_ROWS",
    "EXPECTED_TEST_ROWS",
]
__version__ = "0.3.0"
