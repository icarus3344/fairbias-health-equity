"""Unit tests for Set S candidate configuration selection."""

import numpy as np
import pytest

from nhis_fairbias.benchmark.selection import select_best_configuration_on_S


def test_selection_feasible_winner():
    # Two candidates:
    # Cand A: high BA, but EO gap > tau (infeasible)
    # Cand B: moderate BA, EO gap <= tau (feasible)
    y_S = np.array([1, 0, 1, 0])
    A_S = np.array([1, 1, 2, 2])
    w_S = np.array([1.0, 1.0, 1.0, 1.0])

    # Cand A: perfect predictions on group 1, zero on group 2 -> high EO gap
    preds_A = np.array([1.0, 0.0, 0.0, 0.0])  # TPR(1)=1, TPR(2)=0 -> EO gap = 1.0
    # Cand B: balanced predictions across groups -> low EO gap
    preds_B = np.array([0.8, 0.2, 0.8, 0.2])  # TPR(1)=0.8, TPR(2)=0.8 -> EO gap = 0.0

    candidates = {"cand_A": preds_A, "cand_B": preds_B}

    res = select_best_configuration_on_S(
        candidates, y_S, A_S, w_S, expected_groups=[1, 2], budget_tau=0.20
    )
    assert res.status == "FEASIBLE"
    assert res.selected_candidate_id == "cand_B"


def test_selection_budget_exhausted_fallback():
    # Neither candidate meets tau=0.05
    y_S = np.array([1, 0, 1, 0])
    A_S = np.array([1, 1, 2, 2])
    w_S = np.array([1.0, 1.0, 1.0, 1.0])

    preds_1 = np.array([1.0, 0.0, 0.5, 0.5])  # EO gap = 0.5
    preds_2 = np.array([1.0, 0.0, 0.0, 0.0])  # EO gap = 1.0

    candidates = {"cand_1": preds_1, "cand_2": preds_2}

    res = select_best_configuration_on_S(
        candidates, y_S, A_S, w_S, expected_groups=[1, 2], budget_tau=0.05
    )
    assert res.status == "NO_FEASIBLE_CONFIGURATION"
    # Fallback selects minimal EO gap -> cand_1
    assert res.selected_candidate_id is None
    assert res.boundary_candidate_id == "cand_1"


def test_no_estimable_candidate_does_not_select_arbitrary_id():
    for candidates in ({}, {"invalid": [0, 1]}):
        result = select_best_configuration_on_S(candidates, [0, 1], [1, 1], [1., 1.], [1, 2])
        assert result.status == "NOT_ESTIMABLE"
        assert result.selected_candidate_id is None
        assert result.boundary_candidate_id is None


def test_complexity_breaks_equal_performance_ties():
    result = select_best_configuration_on_S(
        {'alphabetical_first_expensive': [1, 0, 1, 0], 'z_cheaper': [1, 0, 1, 0]},
        [1, 0, 1, 0], [1, 1, 2, 2], [1., 1., 1., 1.], [1, 2],
        candidate_complexities={'alphabetical_first_expensive': 10, 'z_cheaper': 1})
    assert result.selected_candidate_id == 'z_cheaper'
