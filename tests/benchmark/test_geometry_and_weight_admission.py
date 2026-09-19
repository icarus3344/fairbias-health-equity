import numpy as np
import pandas as pd
import pytest
from fairbias import bias_metric
from nhis_fairbias.benchmark.adapters.geometry_audit import audited_mds
from nhis_fairbias.benchmark.metrics import compute_weighted_confusion_metrics
from nhis_fairbias.benchmark.adapters.adapter_fairbias import FairBiasAdapter


def test_finite_huge_weights_keep_ratios_and_ppv():
    y, q = np.array([0, 1]), np.array([0., 1.])
    a = compute_weighted_confusion_metrics(y, q, np.array([1., 1.]))
    b = compute_weighted_confusion_metrics(y, q, np.array([1e308, 1e308]))
    for key in a:
        assert b[key] == pytest.approx(a[key])
    assert b['ppv'] == 1.


def test_fractional_median_in_nullable_integer_feature():
    adapter = FairBiasAdapter()
    X = pd.DataFrame({'agep_a': pd.Series([20, 21, None], dtype='Int64')})
    F = adapter._prepare_fit_semantic(X)
    C = adapter._prepare_predict_semantic(X)
    assert F.agep_a.iloc[-1] == C.agep_a.iloc[-1] == 20.5


def test_mds_cap_rejects_and_restores_binding():
    original = bias_metric.MDS
    records = []
    with pytest.raises(RuntimeError, match='MDS_ITERATION_CAP'):
        with audited_mds(records, lambda: 7):
            model = bias_metric.MDS(n_components=1, dissimilarity='precomputed', random_state=0,
                                   n_init=1, max_iter=1, normalized_stress=False)
            model.fit_transform(np.array([[0., 1., 2.], [1., 0., 1.5], [2., 1.5, 0.]]))
    assert bias_metric.MDS is original
    assert records[0]['geometry_call'] == 7
    assert records[0]['status'] == 'MDS_ITERATION_CAP'
