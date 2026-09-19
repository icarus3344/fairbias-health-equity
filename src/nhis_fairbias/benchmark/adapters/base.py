"""Base interface and capability contract for Benchmark V1 Method Adapters.

Every adapter must explicitly declare:
1. Literature reference & open-source provenance
2. Arm 002 (7-group) support capability
3. Sensitive attribute dependency at inference time
4. Output contract: event probability p vs decision probability q
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import numpy as np


class NotSupportedError(NotImplementedError):
    """Raised when an algorithm cannot theoretically or architecturally support a condition."""


class BaseMethodAdapter(abc.ABC):
    """Abstract base class for benchmark method adapters."""

    name: str = "base"
    literature_reference: str = "N/A"
    upstream_implementation: str = "N/A"
    supports_arm2: bool = True
    requires_sensitive_at_predict: bool = False
    output_type: str = "event_probability_p"  # 'event_probability_p' or 'decision_probability_q'

    @abc.abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        A: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> BaseMethodAdapter:
        """Fit model on training partition F."""

    @abc.abstractmethod
    def predict(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict binary class labels in {0, 1}."""

    @abc.abstractmethod
    def predict_decision_proba(
        self, X: np.ndarray, A: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict decision probabilities q in [0, 1] or positive class probabilities p."""

    def get_capabilities(self) -> Dict[str, Any]:
        """Return declared method capabilities dictionary."""
        return {
            "name": self.name,
            "literature_reference": self.literature_reference,
            "upstream_implementation": self.upstream_implementation,
            "supports_arm2": self.supports_arm2,
            "requires_sensitive_at_predict": self.requires_sensitive_at_predict,
            "output_type": self.output_type,
        }
