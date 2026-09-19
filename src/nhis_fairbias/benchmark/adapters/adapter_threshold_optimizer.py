"""Fairlearn ThresholdOptimizer postprocessing adapter (Hardt et al., NeurIPS 2016).

Literature Reference:
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in Supervised Learning.
  NeurIPS 2016, pp. 3315-3323.
Upstream Implementation:
- Microsoft Fairlearn (fairlearn.postprocessing.ThresholdOptimizer).

Execution Contracts:
1. Two-stage Calibration: Base estimator fit on F, threshold optimizer calibrated on Set C (prefit=True).
2. Requires sensitive attribute A at inference time: Group-specific thresholds depend on A.
3. Decision probability q: Evaluated via interpolated thresholder PMF.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer

from .base import BaseMethodAdapter, NotSupportedError
from .estimators import make_estimator
from fairbias.prediction_contracts import validate_and_extract_positive_probabilities


class ThresholdOptimizerAdapter(BaseMethodAdapter):
    """Adapter for Hardt et al. (NeurIPS 2016) ThresholdOptimizer."""

    name: str = "TO_EO"
    literature_reference: str = "Hardt et al. (NeurIPS 2016) pp. 3315-3323"
    upstream_implementation: str = "fairlearn.postprocessing.ThresholdOptimizer"
    supports_arm2: bool = True
    requires_sensitive_at_predict: bool = True
    output_type: str = "decision_probability_q"

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
        self.base_estimator = make_estimator(
            self.backbone,
            C=self.C,
            random_state=self.random_state,
            estimator_params=self.estimator_params,
        )
        self.postprocessor: Optional[ThresholdOptimizer] = None

    def fit_base(
        self,
        X_F: np.ndarray,
        y_F: np.ndarray,
        sample_weight_F: Optional[np.ndarray] = None,
    ) -> ThresholdOptimizerAdapter:
        """Stage 1: Fit base predictive estimator on Partition F."""
        self.base_estimator.fit(X_F, y_F, sample_weight=sample_weight_F)
        return self

    def calibrate(
        self,
        X_C: np.ndarray,
        y_C: np.ndarray,
        A_C: np.ndarray,
    ) -> ThresholdOptimizerAdapter:
        """Stage 2: Calibrate group-specific thresholds on Partition C."""
        self.postprocessor = ThresholdOptimizer(
            estimator=self.base_estimator,
            constraints="equalized_odds",
            objective="balanced_accuracy_score",
            prefit=True,
            predict_method="predict_proba",
        )
        try:
            self.postprocessor.fit(X_C, y_C, sensitive_features=A_C)
        except ValueError as e:
            raise NotSupportedError(
                f"ThresholdOptimizer requires positive and negative support in all groups on Set C: {e}"
            ) from e
        return self

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        A: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> ThresholdOptimizerAdapter:
        """Reject ambiguous one-partition fitting; F and C must stay separate."""
        raise NotSupportedError(
            "ThresholdOptimizerAdapter requires explicit two-stage fit_base(F) then calibrate(C); "
            "unified fit would reuse F as C"
        )

    def predict(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        if self.postprocessor is None:
            raise RuntimeError("ThresholdOptimizerAdapter must be calibrated before predicting.")
        if A is None:
            raise ValueError("ThresholdOptimizer requires sensitive attribute A at prediction time.")
        return self.postprocessor.predict(X, sensitive_features=A)

    def predict_decision_proba(
        self, X: np.ndarray, A: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self.postprocessor is None:
            raise RuntimeError("ThresholdOptimizerAdapter must be calibrated before predicting.")
        if A is None:
            raise ValueError("ThresholdOptimizer requires sensitive attribute A at prediction time.")
        pmf = self.postprocessor._pmf_predict(X, sensitive_features=A)
        return np.asarray(pmf)[:, 1]

    def predict_event_probability(
        self, X: np.ndarray, A: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Expose the frozen base estimator's p separately from final q."""
        return validate_and_extract_positive_probabilities(self.base_estimator, X, pos_label=1)
