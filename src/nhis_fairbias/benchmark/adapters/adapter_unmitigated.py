"""Unmitigated standard empirical risk minimization baseline adapter.

Literature Reference:
- Standard statistical classification baseline without fairness intervention.
Upstream Implementation:
- scikit-learn LogisticRegression / GradientBoostingClassifier.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from .base import BaseMethodAdapter
from .estimators import make_estimator
from fairbias.prediction_contracts import validate_and_extract_positive_probabilities


class UnmitigatedAdapter(BaseMethodAdapter):
    """Adapter for unmitigated base classifier."""

    name: str = "UNMITIGATED"
    literature_reference: str = "Standard Empirical Risk Minimization Baseline"
    upstream_implementation: str = "scikit-learn.linear_model.LogisticRegression"
    supports_arm2: bool = True
    requires_sensitive_at_predict: bool = False
    output_type: str = "event_probability_p"

    def __init__(
        self,
        C: float = 1.0,
        random_state: int = 42,
        backbone: str = "LR",
        estimator_params: Optional[Dict[str, Any]] = None,
    ):
        self.C = float(C)
        self.random_state = int(random_state)
        self.backbone = str(backbone).upper()
        self.estimator_params = dict(estimator_params or {})
        self.clf = make_estimator(
            self.backbone,
            C=self.C,
            random_state=self.random_state,
            estimator_params=self.estimator_params,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        A: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> UnmitigatedAdapter:
        self.clf.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        return self.clf.predict_proba(X)

    def predict_decision_proba(
        self, X: np.ndarray, A: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.predict_event_probability(X, A=A)

    def predict_event_probability(
        self, X: np.ndarray, A: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return validate_and_extract_positive_probabilities(self.clf, X, pos_label=1)
