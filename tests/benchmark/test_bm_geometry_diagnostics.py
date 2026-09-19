import numpy as np
import pandas as pd
import json
import pytest

from nhis_fairbias.benchmark.bm_geometry_diagnostics import diagnose_bm_geometry
from nhis_fairbias.benchmark.bm_geometry_diagnostics import replay_fairbias_fit


def test_valid_geometry_receipt_is_aggregate_only():
    X = pd.DataFrame({"num": [0.0, 1.0, 2.0, 3.0], "cat": ["a", "a", "b", "b"]})
    A = np.array([0, 0, 1, 1])
    result = diagnose_bm_geometry(X, A, cate_attrs=["cat"], num_attrs=["num"])
    assert result["status"] == "VALID"
    assert result["partitions"]["F"]["rows"] == 4
    assert "values" not in result and "records" not in result


def test_missing_group_feature_is_explicitly_unestimable():
    X = pd.DataFrame({"num": [np.nan, np.nan, 2.0, 3.0], "cat": ["a", "a", "b", "b"]})
    A = np.array([0, 0, 1, 1])
    result = diagnose_bm_geometry(X, A, cate_attrs=["cat"], num_attrs=["num"])
    assert result["status"] == "PAIRWISE_FAILURE"
    assert result["failure_class"] == "EMPTY_OR_UNESTIMABLE_GEOMETRY"
    assert "Numeric feature 'num' has no valid observations" in result["error"]


def test_c_is_schema_only_and_does_not_change_f_geometry():
    X = pd.DataFrame({"num": [0.0, 1.0, 2.0, 3.0], "cat": ["a", "a", "b", "b"]})
    A = np.array([0, 0, 1, 1])
    C = X.copy()
    result = diagnose_bm_geometry(X, A, X_C=C, A_C=A, cate_attrs=["cat"], num_attrs=["num"])
    assert result["status"] == "VALID"
    assert result["C_schema_match_F"] is True
    assert result["fit_operation"] == "geometry_only"


def test_registered_f_replay_records_each_geometry_call_without_values():
    X = pd.DataFrame({"num": [0.0, 1.0, 2.0, 3.0], "cat": ["a", "a", "b", "b"]})
    A = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])
    config = {"candidate_id": "synthetic", "arm_id": "arm_001", "method": "FAIRBIAS_BM",
              "backbone": "LR", "training_weighted": False,
              "params": {"C": 1.0, "epsilon_ratio": 1.0, "max_iterations": 1,
                         "max_geometry_evaluations": 20}}
    result = replay_fairbias_fit(X, A, y_F=y, config=config, seed=0)
    assert result["geometry_call_count"] >= 1
    # A diagnostic replay may stop at the registered iteration budget. It
    # must preserve that failure while still receipt-ing every geometry call.
    assert result["status"] in {"FIT_REPRESENTATION_RETURNED", "FIT_REPRESENTATION_RAISED"}
    assert result.get("failure_class") != "VALID"
    assert all("min" in call and "max" in call for call in result["geometry_calls"] if call["status"] == "RETURNED")
    assert all("values" not in call for call in result["geometry_calls"])


@pytest.mark.parametrize('terminal_empty', [False, True])
def test_empty_trial_is_not_mistaken_for_terminal_exhaustion(monkeypatch, terminal_empty):
    from fairbias.evaluator import FairEvaluator
    from nhis_fairbias.benchmark import experiment_registry
    X = pd.DataFrame({'num': [0., 1., 2., 3.]})
    A = np.array([0, 0, 1, 1])
    monkeypatch.setattr(FairEvaluator, 'calculate_epsilon',
        lambda self, frame, groups: {'protected': {} if not len(frame.columns) else {'num': float('nan')}})

    class FakeAdapter:
        def fit_representation(self, X, y, A):
            FairEvaluator.calculate_epsilon(None, X.iloc[:, :0], None)
            if terminal_empty:
                raise ValueError('empty geometry')
            FairEvaluator.calculate_epsilon(None, X, None)
            raise ValueError('nonfinite geometry')

    monkeypatch.setattr(experiment_registry, 'make_adapter', lambda config, seed: FakeAdapter())
    result = replay_fairbias_fit(X, A, y_F=np.array([0, 1, 0, 1]),
                                config={'method': 'FAIRBIAS_BM'}, seed=0)
    assert (result.get('terminal_diagnosis') == 'SEARCH_FEATURE_EXHAUSTED') == terminal_empty
    json.dumps(result, allow_nan=False)
    if not terminal_empty:
        assert result['geometry_calls'][-1]['nonfinite_count'] == 1
        assert result['geometry_calls'][-1]['min'] is None
