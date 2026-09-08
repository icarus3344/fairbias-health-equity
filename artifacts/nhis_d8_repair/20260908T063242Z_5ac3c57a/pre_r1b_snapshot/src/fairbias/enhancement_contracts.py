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
        self.validate()

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
        if self.protected_fit is not None and len(self.protected_fit) != len(self.fit_X):
            raise ValueError("protected_fit length does not match fit_X length")
        if self.protected_selection is not None and len(self.protected_selection) != len(self.selection_X):
            raise ValueError("protected_selection length does not match selection_X length")

    def fit_fingerprint(self) -> str:
        """Deterministic fingerprint of fit partition shape, columns, and index."""
        meta = f"{self.fit_X.shape}_{list(self.fit_X.columns)}_{list(self.fit_X.index[:5])}_{list(self.fit_X.index[-5:])}"
        return hashlib.sha256(meta.encode("utf-8")).hexdigest()[:16]

    def selection_fingerprint(self) -> str:
        """Deterministic fingerprint of selection partition shape, columns, and index."""
        meta = f"{self.selection_X.shape}_{list(self.selection_X.columns)}_{list(self.selection_X.index[:5])}_{list(self.selection_X.index[-5:])}"
        return hashlib.sha256(meta.encode("utf-8")).hexdigest()[:16]


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
        return self.validity_status == "VALID" and self.utility_score is not None


@dataclasses.dataclass
class FairnessEvaluationResult:
    """Outcome of evaluating fairness bounding constraint on candidate transformed data."""

    is_acceptable: bool
    candidate_max_dphi: float
    cap_applied: float
    rejection_reason: Optional[str] = None
    geometry_eval_count: int = 0


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
    final_epsilon: float
    effective_candidate_cap: float
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
    - Scaler fit ONLY on transformed fit_X
    - Classifier fit ONLY on scaled transformed fit_X
    - AUROC evaluated on scaled transformed selection_X
    - Explicit invalid status (never returns 0.0 silently on errors).
    """
    try:
        t_fit = transformer.transform_data(partition.fit_X, changed_dict, num_attrs, cate_attrs)
        t_sel = transformer.transform_data(partition.selection_X, changed_dict, num_attrs, cate_attrs)
    except Exception as exc:
        return CandidateEvaluationResult(
            validity_status="CANDIDATE_INVALID",
            utility_score=None,
            error_message=f"Transform application failed: {type(exc).__name__}: {exc}",
            model_fit_count=0,
        )

    if t_fit.empty or t_fit.shape[1] == 0:
        return CandidateEvaluationResult(
            validity_status="ALL_FEATURES_DROPPED",
            utility_score=None,
            error_message="All features dropped; classification impossible",
            model_fit_count=0,
        )

    if list(t_fit.columns) != list(t_sel.columns):
        return CandidateEvaluationResult(
            validity_status="COLUMN_MISMATCH",
            utility_score=None,
            error_message=f"Columns mismatch: fit has {list(t_fit.columns)}, selection has {list(t_sel.columns)}",
            model_fit_count=0,
        )

    # Ensure numerical representation for modeling (handling string categories if present)
    t_fit_num = t_fit.copy()
    t_sel_num = t_sel.copy()
    for col in t_fit.columns:
        if not pd.api.types.is_numeric_dtype(t_fit[col]):
            unique_cats = list(pd.Series(t_fit[col].dropna().unique()).sort_values())
            cat_map = {c: float(idx) for idx, c in enumerate(unique_cats)}
            t_fit_num[col] = t_fit[col].map(cat_map).fillna(-1.0).astype(float)
            t_sel_num[col] = t_sel[col].map(cat_map).fillna(-1.0).astype(float)
        else:
            t_fit_num[col] = pd.to_numeric(t_fit[col], errors="coerce").astype(float)
            t_sel_num[col] = pd.to_numeric(t_sel[col], errors="coerce").astype(float)

    fit_vals = t_fit_num.to_numpy(dtype=float)
    sel_vals = t_sel_num.to_numpy(dtype=float)
    if np.any(np.isnan(fit_vals)) or np.any(np.isinf(fit_vals)):
        return CandidateEvaluationResult(
            validity_status="NON_FINITE_OUTPUT",
            utility_score=None,
            error_message="Transformed fit data contains NaN or Inf",
            model_fit_count=0,
        )
    if np.any(np.isnan(sel_vals)) or np.any(np.isinf(sel_vals)):
        return CandidateEvaluationResult(
            validity_status="NON_FINITE_OUTPUT",
            utility_score=None,
            error_message="Transformed selection data contains NaN or Inf",
            model_fit_count=0,
        )

    # Check selection target labels for class diversity
    sel_y_arr = np.asarray(partition.selection_y)
    unique_classes = np.unique(sel_y_arr)
    if len(unique_classes) < 2:
        return CandidateEvaluationResult(
            validity_status="SINGLE_CLASS_SELECTION",
            utility_score=None,
            error_message=f"Selection partition has only 1 unique class: {unique_classes}",
            model_fit_count=0,
        )

    # Scaling fit strictly on fit partition
    if scaler_factory is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler = scaler_factory()

    try:
        scaled_fit = scaler.fit_transform(t_fit_num)
        scaled_sel = scaler.transform(t_sel_num)
    except Exception as exc:
        return CandidateEvaluationResult(
            validity_status="MODEL_FIT_FAILED",
            utility_score=None,
            error_message=f"Scaler fit/transform failed: {type(exc).__name__}: {exc}",
            model_fit_count=0,
        )

    # Model fit strictly on fit partition
    if model_factory is None:
        seed = getattr(evaluator.config, "random_seed", 42)
        model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    else:
        model = model_factory()

    try:
        model.fit(scaled_fit, partition.fit_y.values)
    except Exception as exc:
        return CandidateEvaluationResult(
            validity_status="MODEL_FIT_FAILED",
            utility_score=None,
            error_message=f"Model fit failed: {type(exc).__name__}: {exc}",
            model_fit_count=1,
        )

    # Predict probabilities on selection partition
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled_sel)[:, 1]
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function(scaled_sel)
            probs = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            probs = model.predict(scaled_sel).astype(float)
    except Exception as exc:
        return CandidateEvaluationResult(
            validity_status="MODEL_FIT_FAILED",
            utility_score=None,
            error_message=f"Model prediction failed: {type(exc).__name__}: {exc}",
            model_fit_count=1,
        )

    if probs is None or np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        return CandidateEvaluationResult(
            validity_status="NON_FINITE_OUTPUT",
            utility_score=None,
            error_message="Predicted probabilities contain NaN or Inf",
            model_fit_count=1,
        )

    try:
        score = float(roc_auc_score(sel_y_arr, probs))
    except Exception as exc:
        return CandidateEvaluationResult(
            validity_status="MODEL_FIT_FAILED",
            utility_score=None,
            error_message=f"roc_auc_score computation failed: {type(exc).__name__}: {exc}",
            model_fit_count=1,
        )

    return CandidateEvaluationResult(
        validity_status="VALID",
        utility_score=score,
        utility_metric="AUROC",
        error_message=None,
        model_fit_count=1,
    )
