"""Pinned OxonFair 0.3 post-processing adapter.

This wrapper intentionally exposes OxonFair's adjusted hard decision as q.  The
library documents ``FairPredictor.predict_proba`` as an adjusted score rather
than a probability, so it is never exported as the benchmark event p.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import BaseMethodAdapter, NotSupportedError
from .estimators import make_estimator
from fairbias.prediction_contracts import validate_and_extract_positive_probabilities


class OxonFairAdapter(BaseMethodAdapter):
    name = "OXONFAIR_EO"
    literature_reference = "OxonFair, NeurIPS 2024, arXiv:2407.13710"
    upstream_implementation = "oxonfair==0.3; PyPI wheel SHA256 b8667c73a78ce9e6199244fbebd84fbbb855b40424b290120fbbeedbce5103ab"
    supports_arm2 = True
    requires_sensitive_at_predict = True
    output_type = "decision_probability_q"

    def __getstate__(self):
        # OxonFair stores objective lambdas in its fitted frontier. The bundled
        # cloudpickle handles these without refitting or dropping policy state.
        from joblib.externals import cloudpickle
        state = dict(self.__dict__)
        state["fair_predictor"] = cloudpickle.dumps(self.fair_predictor)
        return state

    def __setstate__(self, state):
        from joblib.externals import cloudpickle
        state["fair_predictor"] = cloudpickle.loads(state["fair_predictor"])
        self.__dict__.update(state)

    def __init__(self, *, bound: float = 0.10, backbone: str = "LR", C: float = 1.0,
                 random_state: int = 42, estimator_params: Optional[Dict[str, Any]] = None,
                 grid_width: int = 6):
        self.bound = float(bound)
        if not np.isfinite(self.bound) or not 0 <= self.bound <= 1:
            raise ValueError("bound must lie in [0, 1]")
        self.backbone = str(backbone).upper()
        self.C = float(C)
        self.random_state = int(random_state)
        self.estimator_params = dict(estimator_params or {})
        if isinstance(grid_width, bool) or not isinstance(grid_width, (int, np.integer)) or int(grid_width) < 1:
            raise ValueError("grid_width must be a positive integer")
        self.grid_width = int(grid_width)
        self.base_estimator = make_estimator(self.backbone, C=self.C, random_state=self.random_state,
                                             estimator_params=self.estimator_params)
        self.fair_predictor = None
        self.calibration_metadata: Dict[str, Any] = {}

    @staticmethod
    def _validate_partition(X: np.ndarray, y: np.ndarray, A: np.ndarray, *, expected_groups=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, y, A = np.asarray(X), np.asarray(y), np.asarray(A)
        if X.ndim != 2 or y.ndim != 1 or A.ndim != 1 or not (len(X) == len(y) == len(A)):
            raise ValueError("X, y, and A must be aligned (X 2D; y/A 1D)")
        if not np.all(np.isfinite(y)) or not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must be finite binary values")
        if not np.all(np.isfinite(A)) or not np.all(np.equal(A, np.floor(A))):
            raise ValueError("A must contain finite integer group labels")
        if expected_groups is not None and set(A.astype(int)) != set(expected_groups):
            raise ValueError("calibration groups must exactly match expected groups")
        return X, y.astype(int), A.astype(int)

    def fit_base(self, X_F: np.ndarray, y_F: np.ndarray, sample_weight_F: Optional[np.ndarray] = None) -> "OxonFairAdapter":
        X_F, y_F, _ = self._validate_partition(X_F, y_F, np.zeros(len(y_F)))
        self.fair_predictor = None
        self.calibration_metadata = {}
        if sample_weight_F is None:
            self.base_estimator.fit(X_F, y_F)
        else:
            sw = np.asarray(sample_weight_F, dtype=float)
            if sw.ndim != 1 or len(sw) != len(y_F) or not np.all(np.isfinite(sw)) or np.any(sw < 0):
                raise ValueError("sample_weight_F must be finite and non-negative")
            self.base_estimator.fit(X_F, y_F, sample_weight=sw)
        return self

    def calibrate(self, X_C: np.ndarray, y_C: np.ndarray, A_C: np.ndarray, *, expected_groups=None) -> "OxonFairAdapter":
        if not hasattr(self.base_estimator, "classes_"):
            raise RuntimeError("fit_base must be called before calibrate")
        X_C, y_C, A_C = self._validate_partition(X_C, y_C, A_C, expected_groups=expected_groups)
        groups = sorted(np.unique(A_C).tolist())
        for g in groups:
            if not np.any((A_C == g) & (y_C == 0)) or not np.any((A_C == g) & (y_C == 1)):
                raise NotSupportedError("OxonFair equalized-odds calibration requires both outcomes in every group")
        from oxonfair import DataDict, FairPredictor, group_metrics as gm
        validation = DataDict(y_C, X_C, A_C)
        # Keep groups in the validation DataDict rather than binding the C
        # vector through the constructor; this makes prediction consume the
        # current query's groups payload.
        self.fair_predictor = FairPredictor(self.base_estimator, validation, groups=None, use_fast=True)
        # Version 0.3's call_fast accepts grid_width but passes a hard-coded
        # default to efficient_compute.grid_search. Honor the registered grid
        # without changing its frontier algorithm. One fit per process; always
        # restore the upstream binding, including on optimization failure.
        from oxonfair.learners import efficient_compute
        original_search = efficient_compute.grid_search
        seen_steps = []
        def registered_search(*args, **kwargs):
            kwargs["steps"] = self.grid_width
            seen_steps.append(kwargs["steps"])
            return original_search(*args, **kwargs)
        try:
            efficient_compute.grid_search = registered_search
            self.fair_predictor.fit(gm.balanced_accuracy, gm.equalized_odds_max, self.bound,
                                    grid_width=self.grid_width)
        finally:
            efficient_compute.grid_search = original_search
        if not seen_steps:
            raise RuntimeError("Pinned OxonFair fast search did not consume the registered grid")
        q_C = np.asarray(self.fair_predictor.predict({"data": X_C, "groups": A_C, "target": np.zeros(len(X_C), dtype=int)}), dtype=int)
        # External check is descriptive: OxonFair may choose its closest point
        # when the native frontier has no feasible solution.
        tpr, fpr = [], []
        for g in groups:
            mask = A_C == g
            tpr.append(np.sum(y_C[mask] * q_C[mask]) / np.sum(y_C[mask]))
            fpr.append(np.sum((1 - y_C[mask]) * q_C[mask]) / np.sum(1 - y_C[mask]))
        eo = max(max(tpr) - min(tpr), max(fpr) - min(fpr))
        self.calibration_metadata = {"bound": self.bound, "native_metric": "equalized_odds_max",
                                     "external_unweighted_c_eo_gap": float(eo),
                                     "external_unweighted_c_feasible": bool(eo <= self.bound),
                                     "groups": groups, "grid_width": self.grid_width}
        self.calibration_metadata.update(effective_grid_steps=seen_steps,
            compatibility_patch="OxonFair 0.3 call_fast grid_width forwarding; scoped and restored")
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, A: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "OxonFairAdapter":
        raise NotSupportedError("OxonFair requires explicit fit_base(F) then calibrate(C); unified fit would reuse F as C")

    def _require(self, X: np.ndarray, A: Optional[np.ndarray]) -> np.ndarray:
        if self.fair_predictor is None:
            raise RuntimeError("OxonFairAdapter must be calibrated before prediction")
        if A is None:
            raise ValueError("OxonFair requires sensitive attribute A at prediction time")
        X, _, A = self._validate_partition(X, np.zeros(len(X)), A)
        groups = set(self.calibration_metadata["groups"])
        if not set(A.tolist()).issubset(groups):
            raise ValueError("prediction A contains an unseen group label")
        return np.asarray(self.fair_predictor.predict({"data": X, "groups": A, "target": np.zeros(len(X), dtype=int)}), dtype=int)

    def predict(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        return self._require(X, A)

    def predict_decision_proba(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        return self._require(X, A).astype(float)

    def predict_event_probability(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the untouched frozen base estimator's positive-class p."""
        X = np.asarray(X)
        return validate_and_extract_positive_probabilities(self.base_estimator, X, pos_label=1)

    def get_capabilities(self) -> Dict[str, Any]:
        out = super().get_capabilities()
        out.update({"native_objective": "balanced_accuracy subject to equalized_odds_max",
                    "requires_A_fit": True, "requires_A_predict": True,
                    "supports_survey_training_weight": False,
                    "calibration_metadata": dict(self.calibration_metadata)})
        return out
