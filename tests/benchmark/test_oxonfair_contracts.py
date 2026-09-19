import numpy as np
import pytest

from nhis_fairbias.benchmark.adapters.adapter_oxonfair import OxonFairAdapter
from nhis_fairbias.benchmark.adapters.base import NotSupportedError


def _data(groups):
    groups = np.asarray(groups, dtype=int)
    # two positives and two negatives per group, with a group signal and a task signal
    rows = []
    y = []
    for g in groups:
        rows.extend([[float(g), 0.0], [float(g), 1.0], [float(g), 2.0], [float(g), 3.0]])
        y.extend([0, 0, 1, 1])
    return np.asarray(rows), np.asarray(y), np.repeat(groups, 4)


@pytest.mark.parametrize("n_groups", [2, 7])
def test_oxonfair_real_algorithm_binary_and_multigroup(n_groups):
    X, y, A = _data(np.arange(n_groups))
    X_f, y_f, A_f = X[::2], y[::2], A[::2]
    X_c, y_c, A_c = X[1::2], y[1::2], A[1::2]
    # Keep both outcomes in each C group after the deterministic split.
    X_f, y_f, A_f = X, y, A
    X_c, y_c, A_c = X + 0.125, y.copy(), A.copy()
    adapter = OxonFairAdapter(bound=0.5, random_state=7, grid_width=3)
    adapter.fit_base(X_f, y_f)
    adapter.calibrate(X_c, y_c, A_c, expected_groups=list(range(n_groups)))
    # Query has a different row count and order than C; groups must come from
    # this query payload, not the cached calibration group array.
    order = np.arange(len(X))[::-1][: max(1, n_groups)]
    q = adapter.predict_decision_proba(X[order], A[order])
    q_one = np.array([adapter.predict_decision_proba(X[i:i + 1], A[i:i + 1])[0] for i in order])
    p = adapter.predict_event_probability(X[order])
    assert set(np.unique(q)).issubset({0.0, 1.0})
    assert np.array_equal(q, q_one)
    assert np.all((p >= 0) & (p <= 1))
    assert adapter.calibration_metadata["native_metric"] == "equalized_odds_max"
    assert adapter.get_capabilities()["requires_A_predict"] is True


def test_oxonfair_rejects_unified_fit_and_bad_prediction_groups():
    X, y, A = _data([0, 1])
    adapter = OxonFairAdapter(bound=0.5, grid_width=3)
    with pytest.raises(NotSupportedError):
        adapter.fit(X, y, A)
    adapter.fit_base(X, y)
    adapter.calibrate(X, y, A, expected_groups=[0, 1])
    with pytest.raises(ValueError):
        adapter.predict(X[:1], np.array([2], dtype=int))
    with pytest.raises(ValueError):
        adapter.predict(X, None)


def test_oxonfair_rejects_calibration_missing_outcome_support():
    X, y, A = _data([0, 1])
    y[A == 1] = 1
    adapter = OxonFairAdapter(bound=0.5, grid_width=3)
    adapter.fit_base(X, y)
    with pytest.raises(NotSupportedError):
        adapter.calibrate(X, y, A, expected_groups=[0, 1])


def test_frozen_oxonfair_frontier_reloads_in_fresh_process(tmp_path):
    import joblib
    import subprocess
    import sys
    X, y, A = _data([1, 2])
    adapter = OxonFairAdapter(bound=.1)
    adapter.fit_base(X, y)
    adapter.calibrate(X + .1, y, A)
    joblib.dump(adapter, tmp_path / 'model.joblib')
    np.savez(tmp_path / 'query.npz', X=X, A=A)
    code = "import joblib,numpy as np,sys; from pathlib import Path; p=Path(sys.argv[1]); m=joblib.load(p/'model.joblib'); a=np.load(p/'query.npz'); np.save(p/'q.npy',m.predict_decision_proba(a['X'],a['A']))"
    subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path)], check=True, capture_output=True)
    np.testing.assert_array_equal(np.load(tmp_path / 'q.npy'), adapter.predict_decision_proba(X, A))


def test_refit_clears_old_calibration_and_partial_batches_are_allowed():
    X, y, A = _data([0, 1])
    adapter = OxonFairAdapter(bound=0.5, grid_width=3)
    adapter.fit_base(X, y)
    adapter.calibrate(X, y, A, expected_groups=[0, 1])
    assert adapter.fair_predictor is not None
    adapter.fit_base(X + 0.1, y)
    assert adapter.fair_predictor is None
    with pytest.raises(RuntimeError):
        adapter.predict(X[:1], A[:1])
    adapter.calibrate(X, y, A, expected_groups=[0, 1])
    assert adapter.predict(X[:1], A[:1]).shape == (1,)


@pytest.mark.parametrize("bad_grid", [0, -1, 1.5, float("nan")])
def test_grid_width_must_be_positive_integer(bad_grid):
    with pytest.raises(ValueError):
        OxonFairAdapter(grid_width=bad_grid)
