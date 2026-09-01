"""Audit logging and structured trace records for FairBias transformations (Gate D3).

Captures:
1. Step-by-step mitigation traces:
   - iteration
   - selected feature
   - feature semantic type
   - d_phi_before
   - epsilon
   - proposed transformation
   - accepted transformation
   - numerical exponent (if applicable)
   - categorical merge mapping (if applicable)
   - d_phi_after
   - whether feature was dropped
   - reason for stopping
2. Numerical transform audit schema and calculation:
   - Spearman rank correlation between original and transformed feature
   - Protected-group predictability before / after
   - Downstream metric changes
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats


@dataclasses.dataclass
class FairBiasTransformStep:
    """Audit record for a single mitigation transform attempt or accepted step."""

    iteration: int
    selected_feature: str
    feature_semantic_type: str
    d_phi_before: float
    epsilon: float
    proposed_transformation: Any
    accepted_transformation: Any
    numerical_exponent: Optional[float] = None
    categorical_merge_mapping: Optional[Dict[str, str]] = None
    d_phi_after: Optional[float] = None
    dropped: bool = False
    stopped_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": int(self.iteration),
            "selected_feature": str(self.selected_feature),
            "feature_semantic_type": str(self.feature_semantic_type),
            "d_phi_before": float(self.d_phi_before),
            "epsilon": float(self.epsilon),
            "proposed_transformation": (
                str(self.proposed_transformation)
                if not isinstance(self.proposed_transformation, (dict, list, int, float, bool, type(None)))
                else self.proposed_transformation
            ),
            "accepted_transformation": (
                str(self.accepted_transformation)
                if not isinstance(self.accepted_transformation, (dict, list, int, float, bool, type(None)))
                else self.accepted_transformation
            ),
            "numerical_exponent": (
                float(self.numerical_exponent)
                if self.numerical_exponent is not None
                else None
            ),
            "categorical_merge_mapping": (
                {str(k): str(v) for k, v in self.categorical_merge_mapping.items()}
                if self.categorical_merge_mapping is not None
                else None
            ),
            "d_phi_after": (
                float(self.d_phi_after) if self.d_phi_after is not None else None
            ),
            "dropped": bool(self.dropped),
            "stopped_reason": (
                str(self.stopped_reason) if self.stopped_reason is not None else None
            ),
        }


@dataclasses.dataclass
class FairBiasTransformTrace:
    """Full execution trace of FairBias mitigation steps."""

    algorithm_mode: str
    protected_attribute: str
    epsilon_threshold: float
    steps: List[FairBiasTransformStep] = dataclasses.field(default_factory=list)
    final_status: str = "IN_PROGRESS"
    final_max_dphi: float = 0.0

    def add_step(self, step: FairBiasTransformStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_mode": self.algorithm_mode,
            "protected_attribute": self.protected_attribute,
            "epsilon_threshold": float(self.epsilon_threshold),
            "final_status": self.final_status,
            "final_max_dphi": float(self.final_max_dphi),
            "total_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }


def compute_numerical_transform_audit(
    original_series: pd.Series,
    transformed_series: pd.Series,
    protected_series: pd.Series,
    feature_name: str,
) -> Dict[str, Any]:
    """
    Compute informational audit metrics for a transformed numerical feature.

    Notice: This audit is for logging and downstream diagnostics ONLY.
    It MUST NOT influence transform selection.
    """
    orig_clean = pd.to_numeric(original_series, errors="coerce")
    trans_clean = pd.to_numeric(transformed_series, errors="coerce")
    prot_clean = protected_series.copy()

    valid_mask = orig_clean.notna() & trans_clean.notna() & prot_clean.notna()
    if not valid_mask.any():
        return {
            "feature_name": feature_name,
            "spearman_rank_correlation": None,
            "mean_diff_before": None,
            "mean_diff_after": None,
            "audit_status": "NO_VALID_DATA",
        }

    x_orig = orig_clean[valid_mask].to_numpy(dtype=float)
    x_trans = trans_clean[valid_mask].to_numpy(dtype=float)
    o_vals = prot_clean[valid_mask].to_numpy()

    # Spearman rank correlation
    if np.all(x_orig == x_orig[0]) or np.all(x_trans == x_trans[0]):
        spearman_corr = 0.0
    else:
        res = stats.spearmanr(x_orig, x_trans)
        spearman_corr = float(res.statistic) if hasattr(res, "statistic") else float(res[0])

    # Protected group means before & after
    groups = np.unique(o_vals)
    mean_diff_before = None
    mean_diff_after = None
    if len(groups) >= 2:
        g0 = groups[0]
        g1 = groups[1]
        m0_b = float(np.mean(x_orig[o_vals == g0]))
        m1_b = float(np.mean(x_orig[o_vals == g1]))
        mean_diff_before = float(abs(m0_b - m1_b))

        m0_a = float(np.mean(x_trans[o_vals == g0]))
        m1_a = float(np.mean(x_trans[o_vals == g1]))
        mean_diff_after = float(abs(m0_a - m1_a))

    return {
        "feature_name": feature_name,
        "spearman_rank_correlation": spearman_corr,
        "mean_diff_before": mean_diff_before,
        "mean_diff_after": mean_diff_after,
        "audit_status": "PASS",
    }
