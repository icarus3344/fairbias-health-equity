"""Model configuration selection engine for Set S (2023) in NHIS Benchmark V1.

Implements:
1. Pre-registered fairness operating points:
   - Primary: tau_EO <= 0.10
   - Secondary: tau_EO <= 0.05, tau_EO <= 0.20
2. Constrained Optimization on Set S:
   - Maximize survey-weighted Balanced Accuracy s.t. EO_gap_S <= tau
3. Ties: smaller EO, registered complexity, then candidate identifier.
4. If none meets tau, retain a separately labelled boundary candidate and no
   feasible winner. An algorithm budget failure is a different condition.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import compute_survey_fairness_metrics


@dataclasses.dataclass(frozen=True)
class SelectionResult:
    """Result of Set S model selection."""

    budget_tau: float
    selected_candidate_id: Optional[str]
    status: str  # FEASIBLE, NO_FEASIBLE_CONFIGURATION, NOT_ESTIMABLE
    balanced_accuracy_S: float
    eo_gap_S: float
    dp_gap_S: float
    all_evaluated_candidates: List[Dict[str, Any]]
    boundary_candidate_id: Optional[str] = None


def select_best_configuration_on_S(
    candidates_predictions: Dict[str, Any],  # candidate_id -> array of predictions on S
    y_S: Any,
    A_S: Any,
    weights_S: Any,
    expected_groups: Sequence[int],
    *,
    budget_tau: float = 0.10,
    candidate_complexities: Optional[Dict[str, float]] = None,
) -> SelectionResult:
    """Select the best candidate configuration on Selection Set S (2023)."""
    if not np.isfinite(budget_tau) or not 0 <= budget_tau <= 1:
        raise ValueError("budget_tau must be finite in [0, 1]")
    evaluated = []
    complexities = candidate_complexities or {}

    for cid, preds in sorted(candidates_predictions.items()):
        metrics = compute_survey_fairness_metrics(
            y_S, preds, A_S, weights_S, expected_groups=expected_groups
        )
        ba = metrics["balanced_accuracy"]
        eo = metrics["eo_gap"]
        dp = metrics["dp_gap"]
        complexity = float(complexities.get(cid, 0.0))
        if not np.isfinite(complexity) or complexity < 0:
            raise ValueError("Candidate complexity must be finite and nonnegative")
        is_feasible = bool(np.isfinite(ba) and np.isfinite(eo) and eo <= budget_tau)

        evaluated.append({
            "candidate_id": cid,
            "balanced_accuracy": ba,
            "eo_gap": eo,
            "dp_gap": dp,
            "is_feasible": is_feasible,
            "complexity": complexity,
        })

    # Sort feasible candidates: max BA, then min EO, min DP, then cid
    feasible_cands = [c for c in evaluated if c["is_feasible"]]

    if feasible_cands:
        feasible_cands.sort(
            key=lambda c: (-c["balanced_accuracy"], c["eo_gap"], c["complexity"], c["candidate_id"])
        )
        winner = feasible_cands[0]
        status = "FEASIBLE"
    else:
        # Fallback: minimize EO gap, then maximize BA
        evaluated_valid = [c for c in evaluated if np.isfinite(c["eo_gap"]) and np.isfinite(c["balanced_accuracy"])]
        if evaluated_valid:
            evaluated_valid.sort(
                key=lambda c: (c["eo_gap"], -c["balanced_accuracy"], c["complexity"], c["candidate_id"])
            )
            winner = evaluated_valid[0]
        else:
            winner = {"candidate_id": None, "balanced_accuracy": np.nan, "eo_gap": np.nan, "dp_gap": np.nan}
        status = "NO_FEASIBLE_CONFIGURATION" if evaluated_valid else "NOT_ESTIMABLE"

    return SelectionResult(
        budget_tau=float(budget_tau),
        selected_candidate_id=winner["candidate_id"] if status == "FEASIBLE" else None,
        status=status,
        balanced_accuracy_S=float(winner["balanced_accuracy"]),
        eo_gap_S=float(winner["eo_gap"]),
        dp_gap_S=float(winner["dp_gap"]),
        all_evaluated_candidates=evaluated,
        boundary_candidate_id=winner["candidate_id"] if status == "NO_FEASIBLE_CONFIGURATION" else None,
    )
