"""Configuration dataclasses and validation for the FairBias framework."""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence


@dataclasses.dataclass(frozen=True)
class FairBiasConfig:
    """Immutable configuration container for FairBias benchmarking runs."""

    dataset_name: str = "compas"
    dataset_path: str = "data_COMPAS.csv"
    label_Y: str = "two_year_recid"
    label_O: tuple[str, ...] = ("sex",)
    
    test_size: float = 0.3
    random_seed: int = 0
    stratify_split: bool = True
    
    classifier: str = "LR"  # LR, DT, RF, GBDT, XGB, LGBM, CatBoost
    eval_norm: str = "min-max"  # min-max, z-score, none
    
    max_iterations: int = 5
    threshold_epsilon: float = 0.5
    threshold_accuracy: float = 0.01
    accuracy_tolerance_tau: float = 0.01  # Pareto constraint: ACC >= initial_ACC - tau
    selection_metric: str = "EO"  # EO or SP for best-iteration selection
    
    use_bias_mitigation: bool = True
    use_accuracy_enhancement: bool = False
    
    # Feature transform bounds
    transform_n_bins: int = 10
    transform_log_epsilon: float = 1e-5
    transform_x_max: float = 1e9
    
    adaptive_threshold_method: str = "kmeans_125"  # "kmeans_125" or "ratio"
    h_order: int = 1  # Shapley truncation order (low-order combinations: k <= h_order)

    # Paper-level d_phi computation (bias_metric)
    mds_max_components: int = 15
    mds_slope_threshold: float = 0.01
    eval_divergence_num: str = "num-a"  # baseline default: group mean difference on normalized values
    eval_divergence_cat: str = "cat-a"  # baseline default: normalized proportion L1 / K
    eval_divergence_scale: str = "mean"  # baseline default scaling family

    # Mitigation acceptance criteria
    phi_threshold: float = 100.0  # NMI information-loss gate (baseline PARAMS_MAIN_THRESHOLD_PHI)
    transform_poly_exponents: tuple = (1 / 3, 1 / 2, 2 / 3, 3.0, 5.0)
    
    verbose: bool = False
    output_dir: str = "runs"

    @classmethod
    def compas_default(cls, **overrides: Any) -> FairBiasConfig:
        """Standard preset for COMPAS dataset."""
        params = {
            "dataset_name": "compas",
            "dataset_path": "data_COMPAS.csv",
            "label_Y": "two_year_recid",
            "label_O": ("sex",),
            "test_size": 0.3,
            "random_seed": 0,
            "classifier": "LR",
            "eval_norm": "min-max",
            "max_iterations": 5,
            "use_bias_mitigation": True,
            "use_accuracy_enhancement": False,
        }
        params.update(overrides)
        if isinstance(params.get("label_O"), (list, set)):
            params["label_O"] = tuple(params["label_O"])
        return cls(**params)

    @classmethod
    def credit_default(cls, **overrides: Any) -> FairBiasConfig:
        """Standard preset for Taiwan Credit Card dataset."""
        params = {
            "dataset_name": "credit",
            "dataset_path": "data_Credit_Card.csv",
            "label_Y": "default payment next month",
            "label_O": ("SEX",),
            "test_size": 0.3,
            "random_seed": 0,
            "classifier": "LR",
            "eval_norm": "min-max",
            "max_iterations": 5,
            "use_bias_mitigation": True,
            "use_accuracy_enhancement": False,
        }
        params.update(overrides)
        if isinstance(params.get("label_O"), (list, set)):
            params["label_O"] = tuple(params["label_O"])
        return cls(**params)
