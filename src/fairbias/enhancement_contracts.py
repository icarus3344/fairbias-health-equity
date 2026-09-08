"""Candidate evaluation contracts, partition validation, audit schemas, and oracle utility evaluation.

Defines the single immutable evaluation contract for Accuracy Enhancement:
- fit and selection partitions receive the EXACT same feature transformations
- scalers are fit strictly on the fit partition (zero leakage from selection)
- AUROC evaluation with explicit invalid states (never swallowed into 0.0)
- audit event dataclass capturing all 25 mandated fields.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from fairbias.evaluator import FairEvaluator
from fairbias.transform import FairTransform


class EnhancementStatus:
    """Canonical status vocabulary for accuracy enhancement evaluations."""

    VALID = "VALID"
    CANDIDATE_INVALID = "CANDIDATE_INVALID"
    NON_NUMERIC_FEATURE = "NON_NUMERIC_FEATURE"
    SINGLE_CLASS_SELECTION = "SINGLE_CLASS_SELECTION"
    SINGLE_CLASS_SELECTION_TARGET = "SINGLE_CLASS_SELECTION_TARGET"
    SINGLE_CLASS_FIT_TARGET = "SINGLE_CLASS_FIT_TARGET"
    ALL_FEATURES_DROPPED = "ALL_FEATURES_DROPPED"
    COLUMN_MISMATCH = "COLUMN_MISMATCH"
    NON_FINITE_OUTPUT = "NON_FINITE_OUTPUT"
    MODEL_FIT_FAILED = "MODEL_FIT_FAILED"
    MISSING_PROBABILITIES = "MISSING_PROBABILITIES"
    CYCLE_DETECTED = "CYCLE_DETECTED"
    FAIRNESS_CAP_EXCEEDED = "FAIRNESS_CAP_EXCEEDED"
    EXCEEDS_FAIRNESS_CAP = "EXCEEDS_FAIRNESS_CAP"
    NO_UTILITY_GAIN = "NO_UTILITY_GAIN"
    ELIGIBLE_NOT_COMMITTED = "ELIGIBLE_NOT_COMMITTED"
    CATEGORY_MAPPING_INVALID = "CATEGORY_MAPPING_INVALID"
    NOT_EVALUATED = "NOT_EVALUATED"


MISSING_PROBABILITY_PREDICTOR = EnhancementStatus.MISSING_PROBABILITIES


@dataclasses.dataclass(frozen=True)
class EvaluationPartition:
    """Immutable pair of fit and selection data partitions with partition validation."""

    fit_X: pd.DataFrame
    fit_y: pd.Series
    selection_X: pd.DataFrame
    selection_y: pd.Series
    protected_fit: Optional[pd.DataFrame] = None
    protected_selection: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        # Defensive snapshot copies prevent external mutation from polluting partition state
        object.__setattr__(self, "fit_X", self.fit_X.copy(deep=True))
        object.__setattr__(self, "fit_y", self.fit_y.copy(deep=True))
        object.__setattr__(self, "selection_X", self.selection_X.copy(deep=True))
        object.__setattr__(self, "selection_y", self.selection_y.copy(deep=True))
        if self.protected_fit is not None:
            object.__setattr__(self, "protected_fit", self.protected_fit.copy(deep=True))
        if self.protected_selection is not None:
            object.__setattr__(self, "protected_selection", self.protected_selection.copy(deep=True))

        self.validate()

        # Bind initial content fingerprints at construction time
        init_fit_fp = self._calc_fit_fingerprint()
        init_sel_fp = self._calc_selection_fingerprint()
        object.__setattr__(self, "_initial_fit_fingerprint", init_fit_fp)
        object.__setattr__(self, "_initial_selection_fingerprint", init_sel_fp)

    def validate(self) -> None:
        """Validate partition dimensions, column alignment, and indices."""
        if len(self.fit_X) != len(self.fit_y):
            raise ValueError(
                f"fit_X length ({len(self.fit_X)}) != fit_y length ({len(self.fit_y)})"
            )
        if len(self.selection_X) != len(self.selection_y):
            raise ValueError(
                f"selection_X length ({len(self.selection_X)}) != selection_y length ({len(self.selection_y)})"
            )
        if list(self.fit_X.columns) != list(self.selection_X.columns):
            raise ValueError(
                f"fit_X columns ({list(self.fit_X.columns)}) do not match "
                f"selection_X columns ({list(self.selection_X.columns)}) in name or order"
            )
        if not self.fit_X.index.equals(self.fit_y.index):
            raise ValueError("fit_X index does not match fit_y index")
        if not self.selection_X.index.equals(self.selection_y.index):
            raise ValueError("selection_X index does not match selection_y index")
        if self.protected_fit is not None:
            if len(self.protected_fit) != len(self.fit_X):
                raise ValueError("protected_fit length does not match fit_X length")
            if not self.protected_fit.index.equals(self.fit_X.index):
                raise ValueError("protected_fit index does not match fit_X index")
        if self.protected_selection is not None:
            if len(self.protected_selection) != len(self.selection_X):
                raise ValueError("protected_selection length does not match selection_X length")
            if not self.protected_selection.index.equals(self.selection_X.index):
                raise ValueError("protected_selection index does not match selection_X index")

    def _calc_fit_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(f"shape={self.fit_X.shape}_cols={list(self.fit_X.columns)}".encode("utf-8"))
        h.update(pd.util.hash_pandas_object(self.fit_X, index=True).values.tobytes())
        h.update(pd.util.hash_pandas_object(self.fit_y, index=True).values.tobytes())
        if self.protected_fit is not None:
            h.update(pd.util.hash_pandas_object(self.protected_fit, index=True).values.tobytes())
        return h.hexdigest()[:16]

    def _calc_selection_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(f"shape={self.selection_X.shape}_cols={list(self.selection_X.columns)}".encode("utf-8"))
        h.update(pd.util.hash_pandas_object(self.selection_X, index=True).values.tobytes())
        h.update(pd.util.hash_pandas_object(self.selection_y, index=True).values.tobytes())
        if self.protected_selection is not None:
            h.update(pd.util.hash_pandas_object(self.protected_selection, index=True).values.tobytes())
        return h.hexdigest()[:16]

    def fit_fingerprint(self) -> str:
        """Deterministic fingerprint of fit partition shape, columns, index, and data content."""
        self.verify_not_mutated()
        return self._calc_fit_fingerprint()

    def selection_fingerprint(self) -> str:
        """Deterministic fingerprint of selection partition shape, columns, index, and data content."""
        self.verify_not_mutated()
        return self._calc_selection_fingerprint()

    def verify_not_mutated(self) -> None:
        """Verify partition content has not mutated since construction."""
        init_fit_fp = getattr(self, "_initial_fit_fingerprint", None)
        if init_fit_fp is not None:
            cur_fit = self._calc_fit_fingerprint()
            if cur_fit != init_fit_fp:
                raise ValueError(
                    f"EvaluationPartition fit content mutated post-construction: "
                    f"initial={init_fit_fp}, current={cur_fit}"
                )
        init_sel_fp = getattr(self, "_initial_selection_fingerprint", None)
        if init_sel_fp is not None:
            cur_sel = self._calc_selection_fingerprint()
            if cur_sel != init_sel_fp:
                raise ValueError(
                    f"EvaluationPartition selection content mutated post-construction: "
                    f"initial={init_sel_fp}, current={cur_sel}"
                )


def compute_configuration_fingerprint(
    config: Optional[Any] = None,
    max_fairness_degradation: float = 0.02,
    min_utility_gain: float = 0.0,
    poly_exponents: Sequence[float] = (),
    label_Y: str = "target",
    label_O: Sequence[str] = (),
    cate_attrs: Sequence[str] = (),
    num_attrs: Sequence[str] = (),
    transformer: Optional[Any] = None,
    algorithm_mode: Optional[str] = None,
    random_seed: Optional[int] = None,
    classifier: Optional[str] = None,
    eval_norm: Optional[str] = None,
    transform_n_bins: Optional[int] = None,
    transform_log_epsilon: Optional[float] = None,
    transform_x_max: Optional[float] = None,
) -> str:
    """Canonical JSON configuration fingerprint covering all settings materially affecting candidate evaluation."""
    from fairbias.enhancement_state import canonical_json_dump

    alg_mode = algorithm_mode if algorithm_mode is not None else getattr(config, "algorithm_mode", "unknown")
    seed = random_seed if random_seed is not None else getattr(config, "random_seed", 42)
    cls_name = classifier if classifier is not None else getattr(config, "classifier", "LR")
    norm_name = eval_norm if eval_norm is not None else getattr(config, "eval_norm", "min-max")

    t_n_bins = (
        transform_n_bins
        if transform_n_bins is not None
        else getattr(transformer, "n_bins", getattr(config, "transform_n_bins", 10))
    )
    t_log_eps = (
        transform_log_epsilon
        if transform_log_epsilon is not None
        else getattr(transformer, "log_epsilon", getattr(config, "transform_log_epsilon", 1e-6))
    )
    t_x_max = (
        transform_x_max
        if transform_x_max is not None
        else getattr(transformer, "x_max", getattr(config, "transform_x_max", None))
    )

    payload = {
        "algorithm_mode": str(alg_mode),
        "random_seed": int(seed),
        "classifier": str(cls_name),
        "eval_norm": str(norm_name),
        "poly_exponents": [round(float(p), 6) for p in poly_exponents],
        "min_utility_gain": round(float(min_utility_gain), 6),
        "max_fairness_degradation": round(float(max_fairness_degradation), 6),
        "label_Y": str(label_Y),
        "label_O": sorted([str(o) for o in label_O]),
        "cate_attrs": sorted([str(c) for c in cate_attrs]),
        "num_attrs": sorted([str(n) for n in num_attrs]),
        "transform_n_bins": int(t_n_bins),
        "transform_log_epsilon": float(t_log_eps),
        "transform_x_max": float(t_x_max) if t_x_max is not None else None,
    }
    canon = canonical_json_dump(payload)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


@dataclasses.dataclass
class CandidateEvaluationResult:
    """Outcome of evaluating utility on a candidate transformed representation."""

    validity_status: str  # VALID, CANDIDATE_INVALID, SINGLE_CLASS_SELECTION, ALL_FEATURES_DROPPED, MODEL_FIT_FAILED, NON_FINITE_OUTPUT
    utility_score: Optional[float]
    utility_metric: str = "AUROC"
    error_message: Optional[str] = None
    model_fit_count: int = 0

    @property
    def is_valid(self) -> bool:
        return self.validity_status == EnhancementStatus.VALID and self.utility_score is not None


@dataclasses.dataclass
class FairnessEvaluationResult:
    """Outcome of evaluating fairness bounding constraint on candidate transformed data."""

    is_acceptable: bool
    candidate_max_dphi: Optional[float]
    cap_applied: Optional[float]
    rejection_reason: Optional[str] = None
    geometry_eval_count: int = 0
    evaluation_status: str = "EVALUATED"


@dataclasses.dataclass
class CandidateAuditEvent:
    """Complete audit record for each evaluated candidate transform step."""

    run_id: str
    arm_id: str
    condition: str
    iteration: int
    engine: str  # 'AE' or 'BM'
    parent_state_hash: str
    candidate_state_hash: str
    selected_feature: str
    proposed_transform: Any
    fit_partition_fingerprint: str
    selection_partition_fingerprint: str
    utility_metric: str
    utility_before: Optional[float]
    utility_candidate: Optional[float]
    utility_gain: Optional[float]
    train_max_dphi_before: Optional[float]
    train_max_dphi_candidate: Optional[float]
    final_epsilon: Optional[float]
    effective_candidate_cap: Optional[float]
    step_rebound: Optional[float]
    accepted: bool
    rejection_reason: Optional[str]
    validity_status: str
    model_fit_count: int
    geometry_eval_count: int
    config_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary with zero microdata content."""
        def _clean_val(v: Any) -> Any:
            if v is None:
                return None
            if isinstance(v, (np.floating, float)):
                if np.isnan(v) or np.isinf(v):
                    return None
                return float(v)
            if isinstance(v, (np.integer, int)):
                return int(v)
            if isinstance(v, dict):
                return {str(k): _clean_val(val) for k, val in v.items()}
            return v

        return {k: _clean_val(v) for k, v in dataclasses.asdict(self).items()}


def evaluate_candidate_utility(
    partition: EvaluationPartition,
    changed_dict: Dict[str, Any],
    num_attrs: List[str],
    cate_attrs: List[str],
    transformer: FairTransform,
    evaluator: FairEvaluator,
    scaler_factory: Optional[Callable[[], Any]] = None,
    model_factory: Optional[Callable[[], Any]] = None,
) -> CandidateEvaluationResult:
    """
    Fit model strictly on transformed fit partition and evaluate AUROC on transformed selection partition.

    Enforces the candidate evaluation contract:
    - Exactly identical feature transformation applied to fit_X and selection_X
    - Model-ready features must be numeric (explicit failure on nonnumeric categories)
    - Scaler fit ONLY on transformed fit_X
    - Classifier fit ONLY on scaled transformed fit_X
    - AUROC evaluated on scaled transformed selection_X
    - Explicit invalid status (never returns 0.0 silently on errors).
    """
    # Verify partition immutability before candidate evaluation
    try:
        partition.verify_not_mutated()
    except Exception as exc:
        return CandidateEvaluationResult(
            validity_status=EnhancementStatus.CANDIDATE_INVALID,
            utility_score=None,
            error_message=f"Partition immutability violation: {exc}",
            model_fit_count=0,
        )

    try:
        t_fit = transformer.transform_data(partition.fit_X, changed_dict, num_attrs, cate_attrs)
        t_sel = transformer.transform_data(partition.selection_X, changed_dict, num_attrs, cate_attrs)
    except Exception as exc:
        return CandidateEvaluationResult(
            validity_status=EnhancementStatus.CANDIDATE_INVALID,
            utility_score=None,
            error_message=f"Transform application failed: {type(exc).__name__}: {exc}",
            model_fit_count=0,
        )

    if t_fit.empty or t_fit.shape[1] == 0:
        return CandidateEvaluationResult(
            validity_status=EnhancementStatus.ALL_FEATURES_DROPPED,
            utility_score=None,
            error_message="All features dropped; classification impossible",
            model_fit_count=0,
        )

    if list(t_fit.columns) != list(t_sel.columns):
        return CandidateEvaluationResult(
            validity_status=EnhancementStatus.COLUMN_MISMATCH,
            utility_score=None,
            error_message=f"Columns mismatch: fit has {list(t_fit.columns)}, selection has {list(t_sel.columns)}",
            model_fit_count=0,
        )

    # Enforce model-ready numeric representation contract (no ad-hoc alphabetical encoding)
    non_num_fit = [c for c in t_fit.columns if not pd.api.types.is_numeric_dtype(t_fit[c])]
    if non_num_fit:
        return CandidateEvaluationResult(
            validity_status=EnhancementStatus.NON_NUMERIC_FEATURE,
            utility_score=None,
            error_message=f"Model-ready features must be numeric; found non-numeric column: {non_num_fit[0]!r}",
            model_fit_count=0,
        )
    non_num_sel = [c for c in t_sel.columns if not pd.api.types.is_numeric_dtype(t_sel[c])]
    if non_num_sel:
        return CandidateEvaluationResult(
            validity_status=EnhancementStatus.NON_NUMERIC_FEATURE,
            utility_score=None,
            error_message=f"Model-ready selection features must be numeric; found non-numeric column: {non_num_sel[0]!r}",
            model_fit_count=0,
        )

    # Verify column ordering consistency
    col_order = list(t_fit.columns)
    t_fit_num = t_fit[col_order].astype(float)
    t_sel_num = t_sel[col_order].astype(float)

    fit_vals = t_fit_num.to_numpy(dtype=float)
    sel_vals = t_sel_num.to_numpy(dtype=float)
    if np.any(np.isnan(fit_vals)) or np.any(np.isinf(fit_vals)):
        return CandidateEvaluationResult(
            validity_status=EnhancementStatus.NON_FINITE_OUTPUT,
            utility_score=None,
            error_message="Transformed fit data contains NaN or Inf",
            model_fit_count=0,
        )
    if np.any(np.isnan(sel_vals)) or np.any(np.isinf(sel_vals)):
        return CandidateEvaluationResult(
            validity_status=EnhancementStatus.NON_FINITE_OUTPUT,
            utility_score=None,
            error_message="Transformed selection data contains NaN or Inf",
            model_fit_count=0,
        )

    # Check selection target labels for class diversity
    sel_y_arr = np.asarray(partition.selection_y)
    unique_classes = np.unique(sel_y_arr)
    if len(unique_classes) < 2:
        return CandidateEvaluationResult(
            validity_status=EnhancementStatus.SINGLE_CLASS_SELECTION,
            utility_score=None,
            error_message=f"Selection partition has only 1 unique class: {unique_classes}",
            model_fit_count=0,
        )

    # Scaling fit strictly on fit partition
    if scaler_factory is not None:
        scaler = scaler_factory()
    elif hasattr(evaluator, "_get_scaler"):
        scaler = evaluator._get_scaler()
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))

    if scaler is not None:
        try:
            scaled_fit = scaler.fit_transform(t_fit_num)
            scaled_sel = scaler.transform(t_sel_num)
        except Exception as exc:
            return CandidateEvaluationResult(
                validity_status=EnhancementStatus.MODEL_FIT_FAILED,
                utility_score=None,
                error_message=f"Scaler fit/transform failed: {type(exc).__name__}: {exc}",
                model_fit_count=0,
            )
    else:
        scaled_fit = t_fit_num.to_numpy(dtype=float)
        scaled_sel = t_sel_num.to_numpy(dtype=float)

    # Model fit strictly on fit partition
    from sklearn.base import clone
    if model_factory is not None:
        model = model_factory()
    elif hasattr(evaluator, "model") and evaluator.model is not None:
        model = clone(evaluator.model)
    else:
        seed = getattr(evaluator.config, "random_seed", 42)
        model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)

    try:
        model.fit(scaled_fit, partition.fit_y.values)
    except Exception as exc:
        return CandidateEvaluationResult(
            validity_status=EnhancementStatus.MODEL_FIT_FAILED,
            utility_score=None,
            error_message=f"Model fit failed: {type(exc).__name__}: {exc}",
            model_fit_count=1,
        )

    # Require probabilistic predictor
    if not hasattr(model, "predict_proba"):
        return CandidateEvaluationResult(
            validity_status=EnhancementStatus.MISSING_PROBABILITIES,
            utility_score=None,
            error_message=f"{EnhancementStatus.MISSING_PROBABILITIES}: Model {type(model).__name__} does not provide predict_proba; probabilistic predictor required",
            model_fit_count=1,
        )

    # Predict probabilities on selection partition
    try:
        proba = model.predict_proba(scaled_sel)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            probs = proba[:, 1]
        else:
            probs = proba.ravel()
    except Exception as exc:
        return CandidateEvaluationResult(
            validity_status=EnhancementStatus.MODEL_FIT_FAILED,
            utility_score=None,
            error_message=f"Model prediction failed: {type(exc).__name__}: {exc}",
            model_fit_count=1,
        )

    if probs is None or np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        return CandidateEvaluationResult(
            validity_status=EnhancementStatus.NON_FINITE_OUTPUT,
            utility_score=None,
            error_message="Predicted probabilities contain NaN or Inf",
            model_fit_count=1,
        )

    try:
        score = float(roc_auc_score(sel_y_arr, probs))
    except Exception as exc:
        return CandidateEvaluationResult(
            validity_status=EnhancementStatus.MODEL_FIT_FAILED,
            utility_score=None,
            error_message=f"roc_auc_score computation failed: {type(exc).__name__}: {exc}",
            model_fit_count=1,
        )

    return CandidateEvaluationResult(
        validity_status=EnhancementStatus.VALID,
        utility_score=score,
        utility_metric="AUROC",
        error_message=None,
        model_fit_count=1,
    )
