import numpy as np
import pytest
from nhis_fairbias.benchmark.predictions import PredictionBundle
from nhis_fairbias.benchmark.risk_metrics import compute_risk_metrics
from nhis_fairbias.benchmark.metrics import compute_weighted_confusion_metrics


def test_risk_and_decisions_have_distinct_valid_values():
    bundle = PredictionBundle([.9, .1], [1., 0.], [1., 0.])
    risk = compute_risk_metrics([1, 0], bundle, [2., 1.])
    policy = compute_weighted_confusion_metrics([1, 0], bundle.q_decision, [2., 1.])
    assert policy['balanced_accuracy'] == 1
    assert risk['brier'] == pytest.approx(.01)
    assert risk['auroc'] == risk['average_precision'] == 1
    assert sum(x['n'] for x in risk['calibration']) == 2
    changed_decision = compute_risk_metrics([1, 0], PredictionBundle([.9, .1], [.2, .2]), [2., 1.])
    assert changed_decision == risk


def test_q_never_receives_risk_metrics():
    result = compute_risk_metrics([1, 0], PredictionBundle(None, [.9, .1]), [1., 1.])
    assert result['status'] == 'NOT_AVAILABLE'
    assert result['auroc'] is result['brier'] is result['average_precision'] is None


def test_missing_group_and_single_class_ranking_are_explicit():
    result = compute_risk_metrics([1, 0], PredictionBundle([1., 0.], [1., 0.]), [1., 1.], [1, 2], [1, 2, 3])
    assert result['group_metrics']['1']['status'] == 'RANKING_NOT_ESTIMABLE'
    assert result['group_metrics']['3']['status'] == 'NOT_ESTIMABLE'
    assert result['calibration'][-1]['n'] == 1
