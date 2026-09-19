"""Learning Fair Representations (LFR) method adapter (Zemel et al., ICML 2013).

Literature Reference:
- Zemel, R., Wu, Y., Swersky, K., Pitassi, T., & Dwork, C. (2013). Learning Fair Representations.
  International Conference on Machine Learning (ICML 2013), pp. 325-333.
Upstream Implementation:
- IBM AI Fairness 360 (aif360.algorithms.preprocessing.LFR).

Execution Contracts:
1. Binary-only support: Arm 002 (7 groups) raises NotSupportedError.
2. Label isolation: LFR reconstructed prototype features X_hat are used for downstream training,
   while original ground-truth labels y are preserved and never overwritten.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import LFR
from aif360.datasets import BinaryLabelDataset
from .base import BaseMethodAdapter, NotSupportedError
from .estimators import make_estimator
from fairbias.prediction_contracts import validate_and_extract_positive_probabilities


class LFRAdapter(BaseMethodAdapter):
    """Adapter for Zemel et al. (ICML 2013) Learning Fair Representations."""

    name: str = "LFR_RECONSTRUCTED"
    literature_reference: str = "Zemel et al. (ICML 2013) pp. 325-333"
    upstream_implementation: str = "aif360.algorithms.preprocessing.LFR"
    supports_arm2: bool = False
    requires_sensitive_at_predict: bool = True
    output_type: str = "event_probability_p"

    def __init__(
        self,
        k: int = 5,
        Ax: float = 0.01,
        Ay: float = 1.0,
        Az: float = 50.0,
        C: float = 1.0,
        random_state: int = 42,
        backbone: str = "LR",
        estimator_params: Optional[Dict[str, Any]] = None,
        maxiter: int = 5000,
        maxfun: int = 5000,
    ):
        self.k = int(k)
        self.Ax = float(Ax)
        self.Ay = float(Ay)
        self.Az = float(Az)
        self.C = float(C)
        self.random_state = int(random_state)
        self.backbone = str(backbone).upper()
        self.estimator_params = dict(estimator_params or {})
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in (maxiter, maxfun)):
            raise ValueError("LFR maxiter/maxfun must be positive integers")
        self.maxiter, self.maxfun = int(maxiter), int(maxfun)
        self.optimization_result_ = {}
        self.lfr: Optional[LFR] = None
        self.clf = make_estimator(
            self.backbone,
            C=self.C,
            random_state=self.random_state,
            estimator_params=self.estimator_params,
        )
        self.privileged_val_: Optional[int] = None
        self.unprivileged_val_: Optional[int] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        A: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> LFRAdapter:
        self.lfr = None
        self.optimization_result_ = {}
        if sample_weight is not None:
            raise NotSupportedError("Survey-weighted LFR representation objective is not implemented")
        unique_groups = sorted(np.unique(A))
        if len(unique_groups) != 2:
            raise NotSupportedError(
                f"LFR algorithm does not support multi-group protected attributes "
                f"(expected exactly 2, found {len(unique_groups)}). Arm 002 is NOT_SUPPORTED."
            )

        # Standard binary group assignment (1=unprivileged, 2=privileged, or 0/1)
        self.privileged_val_ = int(unique_groups[-1])
        self.unprivileged_val_ = int(unique_groups[0])

        unprivileged_groups = [{"prot_attr": self.unprivileged_val_}]
        privileged_groups = [{"prot_attr": self.privileged_val_}]

        # Construct AIF360 BinaryLabelDataset
        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        df["prot_attr"] = A
        df["target"] = y

        bld = BinaryLabelDataset(
            df=df,
            label_names=["target"],
            protected_attribute_names=["prot_attr"],
            favorable_label=1,
            unfavorable_label=0,
        )
        # AIF360's tabular constructor includes protected columns in
        # ``features`` by default.  Keep A solely in protected_attributes.
        bld.features = np.asarray(X, dtype=float).copy()
        bld.feature_names = [f"feat_{i}" for i in range(X.shape[1])]

        self.lfr = LFR(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            k=self.k,
            Ax=self.Ax,
            Ay=self.Ay,
            Az=self.Az,
            print_interval=0,
            verbose=0,
            seed=self.random_state,
        )

        # AIF360 discards L-BFGS diagnostics. Observe the unmodified optimizer
        # result and restore its legacy RNG side effect. Formal fits run in
        # separate processes; no concurrent fitting shares this module patch.
        import aif360.algorithms.preprocessing.lfr as upstream
        original = upstream.optim.fmin_l_bfgs_b
        def observed_optimizer(*args, **kwargs):
            result = original(*args, **kwargs)
            self.optimization_result_ = {key: value for key, value in result[2].items() if key != "grad"}
            self.optimization_result_["objective"] = float(result[1])
            self.optimization_result_["training_n"] = len(bld.features)
            return result
        rng_state = np.random.get_state()
        try:
            with patch.object(upstream, "optim", SimpleNamespace(fmin_l_bfgs_b=observed_optimizer)):
                self.lfr.fit(bld, maxiter=self.maxiter, maxfun=self.maxfun)
        finally:
            np.random.set_state(rng_state)
        self.converged_ = self.optimization_result_.get("warnflag") == 0

        # Transform training data to obtain reconstructed representations X_hat
        rng_state = np.random.get_state()
        try:
            bld_trans = self.lfr.transform(bld)
        finally:
            np.random.set_state(rng_state)
        X_hat = np.asarray(bld_trans.features, dtype=float)

        # Strict contract: isolate original labels y from LFR mutated labels
        self.clf.fit(X_hat, y, sample_weight=sample_weight)
        return self

    def _transform_X(self, X: np.ndarray, A: Optional[np.ndarray]) -> np.ndarray:
        if self.lfr is None:
            raise RuntimeError("LFRAdapter must be fitted before transforming.")
        if A is None:
            raise ValueError("LFR requires sensitive attribute A at transform/predict time.")
        if not set(np.asarray(A).tolist()).issubset({self.privileged_val_, self.unprivileged_val_}):
            raise ValueError("LFR prediction contains unseen protected groups")

        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        df["prot_attr"] = A
        # Placeholder labels (proven to not affect prototype reconstruction features)
        df["target"] = np.zeros(len(X), dtype=int)

        bld = BinaryLabelDataset(
            df=df,
            label_names=["target"],
            protected_attribute_names=["prot_attr"],
            favorable_label=1,
            unfavorable_label=0,
        )
        bld.features = np.asarray(X, dtype=float).copy()
        bld.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        rng_state = np.random.get_state()
        try:
            bld_trans = self.lfr.transform(bld)
        finally:
            np.random.set_state(rng_state)
        return np.asarray(bld_trans.features, dtype=float)

    def predict(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        X_hat = self._transform_X(X, A)
        return self.clf.predict(X_hat)

    def predict_proba(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        X_hat = self._transform_X(X, A)
        return self.clf.predict_proba(X_hat)

    def predict_decision_proba(
        self, X: np.ndarray, A: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.predict_event_probability(X, A=A)

    def predict_event_probability(
        self, X: np.ndarray, A: Optional[np.ndarray] = None
    ) -> np.ndarray:
        X_hat = self._transform_X(X, A)
        return validate_and_extract_positive_probabilities(self.clf, X_hat, pos_label=1)
