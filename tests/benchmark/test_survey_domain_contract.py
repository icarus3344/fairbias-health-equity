import numpy as np
import pytest

from nhis_fairbias.benchmark.metrics import compute_survey_fairness_metrics, kish_effective_sample_size
from nhis_fairbias.benchmark.survey_inference import (
    DesignNotEstimableError,
    RescaledPSUBootstrapEngine,
    evaluate_with_survey_bootstrap,
)


def _design():
    # Two PSUs per stratum; rows 1 and 5 are outside the analysis domain.
    strata = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    psus = np.array([10, 10, 11, 11, 20, 20, 21, 21])
    w = np.ones(8)
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    a = np.array([1, 1, 2, 2, 1, 1, 2, 2])
    domain = np.array([1, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    return y, a, strata, psus, w, domain


def test_domain_mask_preserves_full_design_and_psu_factor():
    y, a, strata, psus, w, domain = _design()
    engine = RescaledPSUBootstrapEngine(strata, psus, w, seed=20260914)
    reps = engine.generate_replicate_weights(1)
    # n_h=2, m_h=1, so the selected PSU receives factor 2.
    assert set(np.unique(reps[0][strata == 1])).issubset({0.0, 2.0})
    result = evaluate_with_survey_bootstrap(
        y, {"FAIRBIAS_BM": y.astype(float), "UNMITIGATED": y.astype(float)}, a,
        strata, psus, w, [1, 2], B=20, domain_mask=domain,
    )
    assert result["status"] == "VALID"
    assert result["design"]["psus"] == 4
    assert result["design"]["domain_n"] == 6


def test_singleton_design_is_not_estimable():
    with pytest.raises(DesignNotEstimableError):
        RescaledPSUBootstrapEngine(np.array([1, 1, 2]), np.array([1, 2, 1]), np.ones(3))
    result = evaluate_with_survey_bootstrap(
        np.array([1, 0, 1]), {"FAIRBIAS_BM": np.ones(3)}, np.array([1, 1, 1]),
        np.array([1, 1, 2]), np.array([1, 2, 1]), np.ones(3), [1], B=10,
    )
    assert result["status"] == "DESIGN_NOT_ESTIMABLE"


@pytest.mark.parametrize("weights", [np.array([1.0, np.nan]), np.array([1.0, -1.0])])
def test_invalid_base_weights_rejected(weights):
    with pytest.raises(ValueError):
        RescaledPSUBootstrapEngine(np.array([1, 1]), np.array([1, 2]), weights)


def test_missing_expected_group_makes_gap_non_estimable():
    m = compute_survey_fairness_metrics(
        np.array([1, 0]), np.array([1.0, 0.0]), np.array([1, 1]), np.ones(2), expected_groups=[1, 2]
    )
    assert np.isnan(m["dp_gap"])
    assert np.isnan(m["eo_gap"])


def test_kish_diagnostic_and_paired_intersection():
    y, a, strata, psus, w, domain = _design()
    assert kish_effective_sample_size(np.array([1.0, 1.0, 2.0])) == pytest.approx(2.6666667)
    pred = y.astype(float)
    result = evaluate_with_survey_bootstrap(
        y, {"FAIRBIAS_BM": pred, "UNMITIGATED": pred}, a, strata, psus, w, [1, 2], B=20, domain_mask=domain,
    )
    contrast = result["paired_contrasts"]["UNMITIGATED"]["balanced_accuracy"]
    assert contrast.replicates_valid <= 20
    assert contrast.point_contrast == pytest.approx(0.0)


def test_below_95_percent_valid_replicates_is_not_estimable(monkeypatch):
    y, a, strata, psus, w, domain = _design()
    original = compute_survey_fairness_metrics
    calls = {"n": 0}

    def invalid_after_first(*args, **kwargs):
        calls["n"] += 1
        out = original(*args, **kwargs)
        if calls["n"] > 2:
            out["eo_gap"] = np.nan
        return out

    monkeypatch.setattr("nhis_fairbias.benchmark.survey_inference.compute_survey_fairness_metrics", invalid_after_first)
    result = evaluate_with_survey_bootstrap(
        y, {"FAIRBIAS_BM": y.astype(float)}, a, strata, psus, w, [1, 2], B=20, domain_mask=domain,
    )
    assert result["method_inference"]["FAIRBIAS_BM"]["eo_gap"].status == "NOT_ESTIMABLE"


@pytest.mark.parametrize("bad_y", [np.array([0.25, 1.0]), np.array([0, 2])])
def test_metrics_reject_fractional_or_nonbinary_outcomes(bad_y):
    with pytest.raises(ValueError):
        compute_survey_fairness_metrics(bad_y, np.array([0.0, 1.0]), np.array([1, 2]), np.ones(2), [1, 2])


def test_domain_mask_rejects_nan_and_fractional_values():
    y, a, strata, psus, w, _ = _design()
    for mask in (np.array([1, 0, np.nan, 1, 1, 0, 1, 1]), np.array([1, 0.5, 1, 1, 1, 0, 1, 1])):
        with pytest.raises(ValueError):
            evaluate_with_survey_bootstrap(y, {"FAIRBIAS_BM": y.astype(float)}, a, strata, psus, w, [1, 2], B=10, domain_mask=mask)


def test_identical_reference_policy_has_pvalue_one():
    y, a, strata, psus, w, domain = _design()
    pred = y.astype(float)
    result = evaluate_with_survey_bootstrap(y, {"FAIRBIAS_BM": pred, "UNMITIGATED": pred}, a, strata, psus, w, [1, 2], B=20, domain_mask=domain)
    contrast = result["paired_contrasts"]["UNMITIGATED"]["balanced_accuracy"]
    assert contrast.point_contrast == pytest.approx(0.0)
    if contrast.status == "VALID":
        assert contrast.p_value == pytest.approx(1.0)
