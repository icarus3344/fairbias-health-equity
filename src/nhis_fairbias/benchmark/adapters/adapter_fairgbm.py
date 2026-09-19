"""Fail-closed FairGBM 0.9.14 adapter.

The upstream package is a native LightGBM fork.  This adapter keeps the
documented sklearn contract explicit and reports a platform dependency error
when the native extension is unavailable; it never silently substitutes an
ordinary LightGBM model.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
import sys

import numpy as np

from .base import BaseMethodAdapter, NotSupportedError

_NATIVE_SITE = Path(__file__).resolve().parents[4] / "artifacts/nhis/benchmark_dependencies_20260916/fairgbm/local_site/lib/python3.11/site-packages"
if str(_NATIVE_SITE) not in sys.path:
    sys.path.insert(0, str(_NATIVE_SITE))


class FairGBMAdapter(BaseMethodAdapter):
    name = "FAIRGBM_EO"
    literature_reference = "FairGBM, ICLR 2023, OpenReview x-mXzBgCX3a"
    upstream_implementation = "fairgbm==0.9.14; PyPI sdist SHA256 b9d69aef01740fea5956157f097aba6daeb6bdc35dba4449a41d4f9e3d59dc7a"
    supports_arm2 = True
    requires_sensitive_at_predict = False
    output_type = "event_probability_p"

    def __init__(self, *, constraint_type: str = "FPR,FNR", n_estimators: int = 100,
                 num_leaves: int = 31, learning_rate: float = 0.1,
                 random_state: int = 42, n_jobs: int = 1, **kwargs: Any):
        tokens = tuple(t.strip().upper() for t in str(constraint_type).split(",") if t.strip())
        if not tokens or any(t not in {"FPR", "FNR"} for t in tokens) or len(set(tokens)) != len(tokens):
            raise ValueError("constraint_type must contain unique FPR/FNR tokens")
        if isinstance(n_estimators, bool) or int(n_estimators) < 1:
            raise ValueError("n_estimators must be a positive integer")
        self.constraint_type = ",".join(tokens)
        self.params = dict(kwargs)
        self.params.update({"constraint_type": self.constraint_type, "n_estimators": int(n_estimators),
                            "num_leaves": int(num_leaves), "learning_rate": float(learning_rate),
                            "random_state": int(random_state), "n_jobs": int(n_jobs)})
        self.model = None

    @staticmethod
    def _validate(X, y, A):
        X, y, A = np.asarray(X), np.asarray(y), np.asarray(A)
        if X.ndim != 2 or y.ndim != 1 or A.ndim != 1 or not (len(X) == len(y) == len(A)):
            raise ValueError("X, y, and A must be aligned (X 2D; y/A 1D)")
        if np.iscomplexobj(X) or not np.all(np.isfinite(X)):
            raise ValueError("X must be finite and real")
        if np.iscomplexobj(y) or not np.all(np.isfinite(y)) or not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must be finite binary values")
        if np.iscomplexobj(A) or not np.all(np.isfinite(A)) or not np.all(np.equal(A, np.floor(A))):
            raise ValueError("A must contain finite integer group labels")
        return X, y.astype(int), A.astype(int)

    def _load_classes(self):
        site = _NATIVE_SITE
        if str(site) not in sys.path:
            sys.path.insert(0, str(site))
        try:
            import inspect
            import fairgbm
            if fairgbm.__version__ != "0.9.14" or Path(fairgbm.__file__).resolve().parent != (site / "fairgbm").resolve():
                raise ImportError("Registered native FairGBM version/path mismatch")
            from fairgbm import FairGBMClassifier, LGBMClassifier
        except Exception as exc:
            raise NotSupportedError(
                "FairGBM native extension is unavailable on this platform; no ordinary LightGBM fallback is permitted"
            ) from exc
        self._patch_sklearn_compat()
        return FairGBMClassifier, LGBMClassifier

    @staticmethod
    def _patch_sklearn_compat():
        # FairGBM 0.9.14 calls the removed sklearn 1.9 keyword
        # ``force_all_finite``. Adapt only that compatibility spelling when
        # the installed sklearn validator no longer accepts it.
        import inspect
        try:
            import fairgbm.sklearn as fair_sklearn
            from sklearn.utils.validation import check_array, check_X_y
            if "force_all_finite" not in inspect.signature(fair_sklearn._LGBMCheckXY).parameters:
                def check_xy(*args, **kwargs):
                    if "force_all_finite" in kwargs:
                        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
                    return check_X_y(*args, **kwargs)
                fair_sklearn._LGBMCheckXY = check_xy
            if "force_all_finite" not in inspect.signature(fair_sklearn._LGBMCheckArray).parameters:
                def check_arr(*args, **kwargs):
                    if "force_all_finite" in kwargs:
                        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
                    return check_array(*args, **kwargs)
                fair_sklearn._LGBMCheckArray = check_arr
        except (ImportError, TypeError, ValueError):
            pass

    def _load_class(self):
        return self._load_classes()[0]

    @property
    def classes_(self):
        if self.model is None:
            raise AttributeError("FairGBMAdapter has not been fitted")
        return np.asarray(self.model.classes_)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(self.params)

    def set_params(self, **params: Any) -> "FairGBMAdapter":
        if "constraint_type" in params:
            tokens = tuple(t.strip().upper() for t in str(params["constraint_type"]).split(",") if t.strip())
            if not tokens or any(t not in {"FPR", "FNR"} for t in tokens) or len(set(tokens)) != len(tokens):
                raise ValueError("constraint_type must contain unique FPR/FNR tokens")
            params["constraint_type"] = ",".join(tokens)
            self.constraint_type = params["constraint_type"]
        self.params.update(params)
        self.model = None
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, A: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> "FairGBMAdapter":
        X, y, A = self._validate(X, y, A)
        if len(np.unique(A)) < 2:
            raise NotSupportedError("FairGBM requires at least two constraint groups")
        sw = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            if sw.ndim != 1 or len(sw) != len(y) or not np.all(np.isfinite(sw)) or np.any(sw < 0):
                raise ValueError("sample_weight must be finite and non-negative")
        cls = self._load_class()
        self.model = cls(**self.params)
        self.model.fit(X, y, constraint_group=A, sample_weight=sw)
        return self

    def predict_event_probability(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("FairGBMAdapter must be fitted before prediction")
        self._patch_sklearn_compat()
        X = np.asarray(X)
        if X.ndim != 2 or np.iscomplexobj(X) or not np.all(np.isfinite(X)):
            raise ValueError("X must be a finite real 2D array")
        proba = np.asarray(self.model.predict_proba(X))
        if proba.ndim != 2 or proba.shape[1] != 2:
            raise ValueError("FairGBM output is not binary class probability")
        return proba[:, 1].astype(float)

    def predict_decision_proba(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        return self.predict_event_probability(X, A)

    def predict(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        p = self.predict_event_probability(X, A)
        return (p >= 0.5).astype(int)

    def get_capabilities(self) -> Dict[str, Any]:
        out = super().get_capabilities()
        out.update({"native_objective": "constrained_cross_entropy",
                    "constraint_type": self.constraint_type,
                    "requires_A_fit": True, "requires_A_predict": False,
                    "supports_survey_training_weight": "unverified",
                    "platform_status": "native build required; Linux officially supported; macOS ARM build attempted"})
        return out


class OrdinaryFairGBMClassifier(BaseMethodAdapter):
    """Matched ordinary classifier from the pinned FairGBM LightGBM fork."""

    name = "FAIRGBM_BASE"
    literature_reference = "Matched ordinary LightGBM classifier from fairgbm 0.9.14 fork"
    upstream_implementation = FairGBMAdapter.upstream_implementation
    supports_arm2 = True
    output_type = "event_probability_p"

    def __init__(self, *, n_estimators: int = 100, num_leaves: int = 31,
                 learning_rate: float = 0.1, random_state: int = 42,
                 n_jobs: int = 1, **kwargs: Any):
        self.params = dict(kwargs)
        self.params.update({"objective": "binary", "n_estimators": int(n_estimators),
                            "num_leaves": int(num_leaves), "learning_rate": float(learning_rate),
                            "random_state": int(random_state), "n_jobs": int(n_jobs)})
        self.model = None

    @property
    def classes_(self):
        if self.model is None:
            raise AttributeError("OrdinaryFairGBMClassifier has not been fitted")
        return np.asarray(self.model.classes_)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(self.params)

    def set_params(self, **params: Any) -> "OrdinaryFairGBMClassifier":
        self.params.update(params)
        self.model = None
        return self

    def fit(self, X, y, A=None, sample_weight=None):
        X, y, A = FairGBMAdapter._validate(X, y, np.zeros(len(y)) if A is None else A)
        _, lgbm_cls = FairGBMAdapter()._load_classes()
        self.model = lgbm_cls(**self.params)
        if sample_weight is None:
            self.model.fit(X, y)
        else:
            sw = np.asarray(sample_weight, dtype=float)
            if sw.ndim != 1 or len(sw) != len(y) or not np.all(np.isfinite(sw)) or np.any(sw < 0):
                raise ValueError("sample_weight must be finite and non-negative")
            self.model.fit(X, y, sample_weight=sw)
        return self

    def predict_event_probability(self, X, A=None):
        if self.model is None:
            raise RuntimeError("OrdinaryFairGBMClassifier must be fitted before prediction")
        FairGBMAdapter._patch_sklearn_compat()
        proba = np.asarray(self.model.predict_proba(np.asarray(X)))
        if proba.ndim != 2 or proba.shape[1] != 2:
            raise ValueError("ordinary FairGBM output is not binary class probability")
        return proba[:, 1].astype(float)

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("OrdinaryFairGBMClassifier must be fitted before prediction")
        FairGBMAdapter._patch_sklearn_compat()
        return self.model.predict_proba(np.asarray(X))

    def predict_decision_proba(self, X, A=None):
        return self.predict_event_probability(X, A)

    def predict(self, X, A=None):
        return (self.predict_event_probability(X, A) >= .5).astype(int)

    def get_capabilities(self):
        out = super().get_capabilities()
        out.update({"native_objective": "binary", "constraint_type": "NONE",
                    "requires_A_fit": False, "requires_A_predict": False,
                    "supports_survey_training_weight": True})
        return out
