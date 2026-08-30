"""Classifier factory with strict lazy imports for optional deep learning and boosting dependencies."""

from __future__ import annotations

from typing import Any, Optional

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def get_classifier(classifier_name: str = "LR", random_state: int = 0, **kwargs: Any) -> BaseEstimator:
    """
    Factory creating scikit-learn compatible classifiers with lazy loading for optional dependencies.
    """
    name = classifier_name.upper().strip()

    if name in ("LR", "LOGISTIC", "LOGISTICREGRESSION"):
        params = {"random_state": random_state, "max_iter": 1000, "solver": "lbfgs"}
        params.update(kwargs)
        return LogisticRegression(**params)

    elif name in ("DT", "DECISIONTREE"):
        params = {"random_state": random_state, "max_depth": 6}
        params.update(kwargs)
        return DecisionTreeClassifier(**params)

    elif name in ("RF", "RANDOMFOREST"):
        params = {"random_state": random_state, "n_estimators": 100, "max_depth": 8, "n_jobs": -1}
        params.update(kwargs)
        return RandomForestClassifier(**params)

    elif name in ("GB", "GBDT", "GRADIENTBOOSTING"):
        params = {"random_state": random_state, "n_estimators": 100, "max_depth": 4}
        params.update(kwargs)
        return GradientBoostingClassifier(**params)

    elif name in ("XGB", "XGBOOST"):
        try:
            import xgboost as xgb
        except ImportError as err:
            raise ImportError(
                "xgboost is required for classifier='XGB'. Install it via: pip install xgboost"
            ) from err
        params = {"random_state": random_state, "n_estimators": 100, "max_depth": 4, "eval_metric": "logloss"}
        params.update(kwargs)
        return xgb.XGBClassifier(**params)

    elif name in ("LGBM", "LIGHTGBM"):
        try:
            import lightgbm as lgb
        except ImportError as err:
            raise ImportError(
                "lightgbm is required for classifier='LGBM'. Install it via: pip install lightgbm"
            ) from err
        params = {"random_state": random_state, "n_estimators": 100, "max_depth": 4, "verbose": -1}
        params.update(kwargs)
        return lgb.LGBMClassifier(**params)

    elif name in ("CATBOOST", "CB"):
        try:
            import catboost as cb
        except ImportError as err:
            raise ImportError(
                "catboost is required for classifier='CatBoost'. Install it via: pip install catboost"
            ) from err
        params = {"random_state": random_state, "iterations": 100, "depth": 4, "verbose": False}
        params.update(kwargs)
        return cb.CatBoostClassifier(**params)

    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}. Supported: LR, DT, RF, GBDT, XGB, LGBM, CatBoost.")
