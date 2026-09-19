"""Fairlearn Exponentiated Gradient (Reductions) method adapter (Agarwal et al., ICML 2018).

Literature Reference:
- Agarwal, A., Beygelzimer, A., Dudik, M., Langford, J., & Wallach, H. (2018).
  A Reductions Approach to Fair Classification. ICML 2018, pp. 60-69.
Upstream Implementation:
- Microsoft Fairlearn (fairlearn.reductions.ExponentiatedGradient).

Execution Contracts:
1. Decision Probability q: Evaluates stochastic randomized decision mixture q_i = sum_m beta_m h_m(x_i).
2. Base estimator oracle sample weights: Oracle costs routed directly into LogisticRegression(sample_weight).
3. Inference independence from A: Base estimators h_m operate solely on features X.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from fairlearn.reductions import (
    DemographicParity,
    EqualizedOdds,
    ExponentiatedGradient,
)

from .base import BaseMethodAdapter, NotSupportedError
from .estimators import make_estimator


class ExponentiatedGradientAdapter(BaseMethodAdapter):
    """Adapter for Fairlearn Exponentiated Gradient reductions (DP or EO)."""

    def __init__(
        self,
        constraint_type: str = "equalized_odds",  # 'demographic_parity' or 'equalized_odds'
        eps: float = 0.01,
        max_iter: int = 50,
        C: float = 1.0,
        random_state: int = 42,
        difference_bound: Optional[float] = None,
        backbone: str = "LR",
        estimator_params: Optional[Dict[str, Any]] = None,
    ):
        self.constraint_type = constraint_type.lower()
        self.difference_bound = difference_bound
        if self.constraint_type in ("demographic_parity", "dp"):
            self.name = "EG_DP"
            self.constraints = DemographicParity(difference_bound=difference_bound)
        elif self.constraint_type in ("equalized_odds", "eo"):
            self.name = "EG_EO"
            self.constraints = EqualizedOdds(difference_bound=difference_bound)
        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type}")

        self.literature_reference = "Agarwal et al. (ICML 2018) pp. 60-69"
        self.upstream_implementation = "fairlearn.reductions.ExponentiatedGradient"
        self.supports_arm2 = True
        self.requires_sensitive_at_predict = False
        self.output_type = "decision_probability_q"

        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.C = float(C)
        self.random_state = int(random_state)
        self.backbone = str(backbone).upper()
        self.estimator_params = dict(estimator_params or {})

        base_estimator = make_estimator(
            self.backbone,
            C=self.C,
            random_state=self.random_state,
            estimator_params=self.estimator_params,
        )
        self.model = ExponentiatedGradient(
            estimator=base_estimator,
            constraints=self.constraints,
            eps=self.eps,
            max_iter=self.max_iter,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        A: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> ExponentiatedGradientAdapter:
        # Note: Fairlearn ExponentiatedGradient takes sensitive_features
        fit_kwargs: Dict[str, Any] = {"sensitive_features": A}
        if sample_weight is not None:
            raise NotSupportedError("Survey-weighted EG moments are not implemented; oracle costs are not survey weights")
        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        return self.model.predict(X)

    def predict_decision_proba(
        self, X: np.ndarray, A: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # Decision probability q is the expectation of the randomized policy
        pmf = self.model._pmf_predict(X)
        if isinstance(pmf, np.ndarray):
            return pmf[:, 1]
        import pandas as pd
        if isinstance(pmf, pd.DataFrame):
            return pmf.iloc[:, 1].values
        return np.asarray(pmf)[:, 1]
