import numpy as np
from nhis_fairbias.benchmark.survey_batch import bootstrap_metric_arrays
from nhis_fairbias.benchmark.survey_inference import RescaledPSUBootstrapEngine
from nhis_fairbias.benchmark.metrics import compute_survey_fairness_metrics


def test_psu_totals_match_independent_row_level_bootstrap_with_absent_domain_psu():
    rng = np.random.default_rng(43)
    n = 96
    s = np.repeat(np.arange(4), 24)
    psu = np.tile(np.repeat([1, 2, 3], 8), 4)
    y = np.tile([0, 1], n//2)
    a = np.tile([1, 1, 2, 2], n//4)
    w = rng.uniform(1, 10, n)
    domain = ~((s == 0) & (psu == 1))
    a[~domain] = 0
    predictions = {'A': rng.uniform(0, 1, n), 'B': rng.integers(0, 2, n).astype(float)}
    actual = bootstrap_metric_arrays(y, predictions, a, s, psu, w, [1, 2], domain_mask=domain, B=30, seed=98)
    expected = {name: [] for name in predictions}
    for rep in RescaledPSUBootstrapEngine(s, psu, w, seed=98).iter_replicate_weights(30):
        for name, q in predictions.items():
            metrics = compute_survey_fairness_metrics(y, q, a, rep*domain, [1, 2])
            expected[name].append([metrics[k] for k in ('balanced_accuracy', 'dp_gap', 'eo_gap')])
    for name in predictions:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-14, rtol=0, equal_nan=True)
