"""Frozen benchmark backbone factory.

The factory keeps the registered LR/GBDT parameter defaults in one place so
all baseline adapters use the same estimator identity and can still accept a
small, explicit smoke-test override.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def make_estimator(
    backbone: str = "LR",
    *,
    C: float = 1.0,
    random_state: int = 42,
    estimator_params: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Build a registered LR or sklearn GBDT estimator.

    ``estimator_params`` is deliberately applied last for bounded smoke
    overrides (for example ``n_estimators=2``); production grids must still
    be registered by the caller.
    """
    key = str(backbone).upper()
    params = dict(estimator_params or {})
    if key in ("LR", "LOGISTICREGRESSION", "LOGISTIC_REGRESSION"):
        defaults = {"C": float(C), "max_iter": 1000, "random_state": int(random_state)}
        defaults.update(params)
        return LogisticRegression(**defaults)
    if key in ("GBDT", "GRADIENTBOOSTING", "GRADIENT_BOOSTING"):
        defaults = {
            "n_estimators": 100,
            "max_depth": 2,
            "learning_rate": 0.05,
            "subsample": 1.0,
            "random_state": int(random_state),
        }
        defaults.update(params)
        return GradientBoostingClassifier(**defaults)
    if key in ("MLP", "TORCHMLP", "TORCH_MLP"):
        # Lazy import keeps LR/GBDT factory use independent of the optional
        # Fairret artifact and its torch dependency.
        from .adapter_fairret import TorchMLPClassifier

        defaults = {
            "input_dim": None,
            "hidden_layers": (64, 32),
            "learning_rate": 0.001,
            "epochs": 100,
            "random_state": int(random_state),
        }
        defaults.update(params)
        return TorchMLPClassifier(**defaults)
    if key == "TABM":
        from .adapter_tabm import TabMClassifier
        return TabMClassifier(random_state=random_state, **params)
    if key == "FAIRGBM_BASE":
        from .adapter_fairgbm import OrdinaryFairGBMClassifier
        defaults = {"learning_rate": .05, "n_jobs": 1, "random_state": random_state}
        defaults.update(params)
        return OrdinaryFairGBMClassifier(**defaults)
    raise ValueError(f"Unsupported benchmark backbone: {backbone!r}; expected 'LR' or 'GBDT'")
