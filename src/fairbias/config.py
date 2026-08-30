"""Configuration dataclasses and validation for the FairBias framework."""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence


@dataclasses.dataclass(frozen=True)
class FairBiasConfig:
    """Immutable configuration container for FairBias benchmarking runs."""

    dataset_name: str = "compas"
    dataset_path: str = "data_COMPAS.csv"
    label_Y: str = "two_year_recid"
    label_O: tuple[str, ...] = ("sex",)
    
    # Paper split: training 64% / validation 16% / test 20% (Eq. "Implementation details")
    test_size: float = 0.20
    val_size: float = 0.16
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
    # None = strict paper mode: the only magnitude bound on power transforms
    # is the paper's numpy.float32 overflow rule (≈3.4e38 -> attribute dropped).
    # Setting a numeric value re-enables a NON-PAPER engineering guard and
    # must be reported as such.
    transform_x_max: Optional[float] = None
    
    adaptive_threshold_method: str = "kmeans_125"  # "kmeans_125" or "ratio"
    h_order: int = 1  # Level-H exclusion order (Eq. 4/5): near-full companion contexts

    # Paper-level d_phi computation (bias_metric, Eqs. 1-6)
    mds_max_components: int = 15
    mds_slope_threshold: float = 0.01
    eval_divergence_num: str = "num-a"  # Eq. 2 numerical: centroid distance after min-max normalization
    eval_divergence_cat: str = "cat-a"  # Eq. 2 categorical: mean absolute frequency gap over K categories

    # Mitigation acceptance criteria
    phi_threshold: float = 100.0  # NMI information-loss gate (baseline PARAMS_MAIN_THRESHOLD_PHI)
    # CONFIGURED candidate grid (finite implementation budget).  The
    # paper's main text lists odd integers (3, 5, 7) and odd fractions
    # (1/3, 1/5, 1/7) as EXAMPLES ("e.g.") and prescribes an
    # increasing-order search with no stated finite upper bound; the
    # official code repository (zftang/MachineClassifer_BiasMitigation_beta)
    # uses an interleaved stream [3, 1/3, 5, 1/5, ..., 1999, 1/1999].
    # The Supplementary Materials (Algorithm 1) were not accessible for a
    # definitive bound, so this six-value grid is an implementation
    # budget: exhausting it is recorded as "candidate_grid_exhausted",
    # NOT as paper-level algorithmic non-convergence.
    transform_poly_exponents: tuple = (1 / 7, 1 / 5, 1 / 3, 3.0, 5.0, 7.0)

    # Failure semantics for the greedy mitigation search:
    #   "stop" (default, strict paper): when the CURRENT highest-d_phi
    #       attribute's CONFIGURED candidate-grid search cannot reach the
    #       epsilon ball, the run records the failure (search_scope=
    #       "configured_grid") and terminates (the paper keeps operating
    #       on the highest attribute).  Note: this is "configured grid
    #       exhausted", not a paper-level non-convergence claim.
    #   "next" (explicitly named ENGINEERING extension): record the failure
    #       keyed by (protected attribute, feature) and try the next-ranked
    #       attribute instead.
    failed_attribute_mode: str = "stop"

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
            "test_size": 0.20,
            "val_size": 0.16,
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
            "test_size": 0.20,
            "val_size": 0.16,
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
