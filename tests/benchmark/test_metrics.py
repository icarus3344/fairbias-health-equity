"""Unit tests for survey-weighted metrics and rescaled PSU bootstrap inference."""

import numpy as np
import pytest

from nhis_fairbias.benchmark.metrics import (
    compute_survey_fairness_metrics,
    compute_weighted_confusion_metrics,
)
from nhis_fairbias.benchmark.survey_inference import (
    RescaledPSUBootstrapEngine,
    evaluate_with_survey_bootstrap,
)


def test_hand_calculated_survey_metrics():
    """Verify analytical weighted calculations against hand-derived values."""
    # 4 respondents:
    # 0: y=1, q=1, w=10  (TP)
    # 1: y=1, q=0, w=10  (FN) -> TPR = 10 / 20 = 0.50
    # 2: y=0, q=0, w=30  (TN)
    # 3: y=0, q=1, w=10  (FP) -> FPR = 10 / 40 = 0.25 -> TNR = 0.75
    # Balanced Accuracy = 0.5 * (0.50 + 0.75) = 0.625
    # Selection Rate = (10 + 10) / (10 + 10 + 30 + 10) = 20 / 60 = 0.3333333333333333
    y_true = np.array([1, 1, 0, 0])
    q = np.array([1.0, 0.0, 0.0, 1.0])
    w = np.array([10.0, 10.0, 30.0, 10.0])

    m = compute_weighted_confusion_metrics(y_true, q, w)
    assert np.isclose(m["tpr"], 0.50, atol=1e-6)
    assert np.isclose(m["fpr"], 0.25, atol=1e-6)
    assert np.isclose(m["tnr"], 0.75, atol=1e-6)
    assert np.isclose(m["balanced_accuracy"], 0.625, atol=1e-6)
    assert np.isclose(m["selection_rate"], 20.0 / 60.0, atol=1e-6)


def test_survey_fairness_gaps():
    # 2 groups: Group 1 (indices 0,1), Group 2 (indices 2,3)
    y_true = np.array([1, 0, 1, 0])
    q = np.array([1.0, 0.0, 0.5, 0.5])
    A = np.array([1, 1, 2, 2])
    w = np.array([1.0, 1.0, 1.0, 1.0])

    # Group 1:
    # y=1, q=1 -> TPR(1) = 1.0
    # y=0, q=0 -> FPR(1) = 0.0
    # SR(1) = (1+0)/2 = 0.5
    # Group 2:
    # y=1, q=0.5 -> TPR(2) = 0.5
    # y=0, q=0.5 -> FPR(2) = 0.5
    # SR(2) = (0.5+0.5)/2 = 0.5
    # DP Gap = |0.5 - 0.5| = 0.0
    # EO Gap = max(|1.0 - 0.5|, |0.0 - 0.5|) = 0.5
    m = compute_survey_fairness_metrics(y_true, q, A, w, expected_groups=[1, 2])
    assert np.isclose(m["dp_gap"], 0.0, atol=1e-6)
    assert np.isclose(m["eo_gap"], 0.5, atol=1e-6)


def test_rescaled_psu_bootstrap_engine():
    # 2 strata, 2 PSUs per stratum
    # Stratum 101: PSU 1 (rows 0, 1), PSU 2 (rows 2, 3)
    # Stratum 102: PSU 1 (rows 4, 5), PSU 2 (rows 6, 7)
    strata = np.array([101, 101, 101, 101, 102, 102, 102, 102])
    psus = np.array([1, 1, 2, 2, 1, 1, 2, 2])
    w = np.array([10.0, 10.0, 20.0, 20.0, 15.0, 15.0, 25.0, 25.0])

    engine = RescaledPSUBootstrapEngine(strata, psus, w, seed=123)
    assert engine.df == 2  # 4 PSUs - 2 strata = 2 df

    rep_w = engine.generate_replicate_weights(B=10)
    assert rep_w.shape == (10, 8)
    assert np.all(rep_w >= 0.0)
    # In rescaled bootstrap, weight scaling preserves non-negative weights


def test_evaluate_with_survey_bootstrap():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    q_fairbias = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.8, 0.2])
    q_unmitigated = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    A = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    strata = np.array([101, 101, 101, 101, 102, 102, 102, 102])
    psus = np.array([1, 1, 2, 2, 1, 1, 2, 2])
    w = np.array([10.0, 10.0, 20.0, 20.0, 15.0, 15.0, 25.0, 25.0])

    preds = {
        "FAIRBIAS_BM": q_fairbias,
        "UNMITIGATED": q_unmitigated,
    }

    res = evaluate_with_survey_bootstrap(
        y_true, preds, A, strata, psus, w, expected_groups=[1, 2], B=20
    )
    assert "FAIRBIAS_BM" in res["method_inference"]
    assert "UNMITIGATED" in res["method_inference"]
    assert "UNMITIGATED" in res["paired_contrasts"]
    assert "balanced_accuracy" in res["paired_contrasts"]["UNMITIGATED"]
    assert "eo_gap" in res["paired_contrasts"]["UNMITIGATED"]
