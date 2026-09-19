"""Reweighing method adapter (Kamiran & Calders, 2012).

Literature Reference:
- Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination.
  Knowledge and Information Systems, 33(1), 1-33.
Upstream Implementation:
- IBM AI Fairness 360 (aif360.sklearn.preprocessing.Reweighing).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from aif360.sklearn.preprocessing import Reweighing
from .base import BaseMethodAdapter, NotSupportedError
from .estimators import make_estimator
from fairbias.prediction_contracts import validate_and_extract_positive_probabilities


class ReweighingAdapter(BaseMethodAdapter):
    """Adapter for Kamiran & Calders (2012) multi-group Reweighing."""

    name: str = "REWEIGHING"
    literature_reference: str = "Kamiran & Calders (2012) KAIS 33(1):1-33"
    upstream_implementation: str = "aif360.sklearn.preprocessing.Reweighing"
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
        self.reweigher = Reweighing()
        self.clf = make_estimator(
            self.backbone,
            C=self.C,
            random_state=self.random_state,
            estimator_params=self.estimator_params,
        )
        self.weights_fair_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        A: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> ReweighingAdapter:
        X = np.asarray(X)
        y = np.asarray(y)
        A = np.asarray(A)
        if len(X) != len(y) or len(y) != len(A):
            raise ValueError("X, y, and A must have equal length")
        if not np.all(np.isfinite(y)) or not np.all((y == 0) | (y == 1)):
            raise ValueError("y must contain finite binary labels")
        if sample_weight is None:
            base_weight = np.ones(len(y), dtype=float)
        else:
            base_weight = np.asarray(sample_weight, dtype=float)
            if base_weight.ndim != 1 or len(base_weight) != len(y):
                raise ValueError("sample_weight must be a vector matching y")
            if not np.all(np.isfinite(base_weight)) or np.any(base_weight < 0.0):
                raise ValueError("sample_weight must be finite and non-negative")
        if np.any(base_weight <= 0.0):
            raise NotSupportedError("Reweighing requires positive support in every observed A×Y cell")
        if any(np.sum((A == group) & (y == label)) == 0 for group in np.unique(A) for label in (0, 1)):
            raise NotSupportedError("Reweighing requires positive support in every A×Y cell")

        # Construct a DataFrame with sensitive attribute in the index.  The
        # weight argument is routed into AIF360 and then to the classifier.
        X_df = pd.DataFrame(X, index=pd.Index(A, name="sensitive_attr"))
        y_s = pd.Series(y, index=X_df.index)

        # Reweighing calculates W_gc = P(A=g)*P(Y=c) / P(A=g, Y=c)
        _, fair_weights = self.reweigher.fit_transform(X_df, y_s, sample_weight=base_weight)
        self.weights_fair_ = np.asarray(fair_weights, dtype=float)
        if (
            self.weights_fair_.ndim != 1
            or len(self.weights_fair_) != len(y)
            or not np.all(np.isfinite(self.weights_fair_))
            or np.any(self.weights_fair_ < 0.0)
        ):
            raise ValueError("Reweighing produced invalid fairness weights")

        # Train downstream classifier with fairness sample weights
        self.clf.fit(X, y, sample_weight=self.weights_fair_)
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
