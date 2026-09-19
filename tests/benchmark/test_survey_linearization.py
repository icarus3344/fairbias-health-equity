import numpy as np
import pytest

from nhis_fairbias.benchmark.survey_linearization import linearized_survey_inference


def _design():
    # Two strata, two PSUs each, two rows per PSU.  Every row is in the domain.
    strata = np.repeat([1, 2], 4)
    psus = np.array([10, 10, 11, 11, 20, 20, 21, 21])
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    A = np.zeros(8, dtype=int)
    w = np.ones(8)
    return y, A, strata, psus, w


def test_exact_hand_taylor_covariance_and_psu_centering():
    y, A, strata, psus, w = _design()
    # Positive-outcome q values by PSU: .2, .6, .4, .8; overall TPR=.5.
    q = np.array([.2, .0, .6, .0, .4, .0, .8, .0])
    result = linearized_survey_inference(y, {"M": q}, A, strata, psus, w, [0])
    assert result["status"] == "VALID"
    assert result["degrees_of_freedom"] == 2
    # Each stratum contributes 2*(.05^2 + .05^2)=.01; total=.02.
    assert result["joint_covariance"]["M"]["covariance"][0, 0] == pytest.approx(.02)
    assert result["methods"]["M"]["rates"]["tpr"].point_estimate == pytest.approx(.5)


def test_identical_methods_have_zero_paired_ba_and_eo_gap_contrast():
    y, A, strata, psus, w = _design()
    q = np.array([.2, .0, .6, .0, .4, .0, .8, .0])
    result = linearized_survey_inference(
        y, {"FAIRBIAS_BM": q, "BASE": q.copy()}, A, strata, psus, w, [0]
    )
    paired = result["paired"]["BASE"]
    assert paired["balanced_accuracy"].point_estimate == pytest.approx(0.0)
    assert paired["balanced_accuracy"].std_error == pytest.approx(0.0)
    assert paired["eo_gap_interval"][0] == pytest.approx(0.0)
    assert paired["eo_gap_interval"][1] == pytest.approx(0.0)


def test_missing_group_outcome_is_not_estimable():
    y, A, strata, psus, w = _design()
    q = np.array([.2, .0, .6, .0, .4, .0, .8, .0])
    result = linearized_survey_inference(y, {"M": q}, A, strata, psus, w, [0, 1])
    assert result["status"] == "VALID"
    assert result["methods"]["M"]["status"] == "NOT_ESTIMABLE"
    assert result["methods"]["M"]["rates"]["selection_g1"].status == "NOT_ESTIMABLE"


def test_singleton_and_shape_validation():
    y, A, strata, psus, w = _design()
    q = np.array([.2, .0, .6, .0, .4, .0, .8, .0])
    singleton = psus.copy(); singleton[4:] = 20
    result = linearized_survey_inference(y, {"M": q}, A, strata, singleton, w, [0])
    assert result["status"] == "DESIGN_NOT_ESTIMABLE"
    with pytest.raises(ValueError):
        linearized_survey_inference(y, {"M": q[:-1]}, A, strata, psus, w, [0])
    with pytest.raises(ValueError):
        linearized_survey_inference(y, {"M": q}, A, strata, psus, w, [0, 0])


def test_out_of_domain_group_placeholders_are_ignored():
    y, A, strata, psus, w = _design()
    q = np.array([.2, .0, .6, .0, .4, .0, .8, .0])
    A = np.where(np.arange(len(A)) < 4, 1, 0)
    domain = np.arange(len(A)) < 4
    result = linearized_survey_inference(y, {"M": q}, A, strata, psus, w, [1], domain_mask=domain)
    assert result["status"] == "VALID"
    assert result["domain_n"] == 4


def test_joint_rate_critical_scales_with_method_count_but_ba_t_is_individual():
    y, A, strata, psus, w = _design()
    q = np.array([.2, .0, .6, .0, .4, .0, .8, .0])
    one = linearized_survey_inference(y, {"M": q}, A, strata, psus, w, [0])
    many = linearized_survey_inference(y, {"M": q, "N": q.copy()}, A, strata, psus, w, [0])
    assert many["methods"]["M"]["t_critical"] > one["methods"]["M"]["t_critical"]
    assert many["methods"]["M"]["balanced_accuracy"].ci_lower == pytest.approx(
        one["methods"]["M"]["balanced_accuracy"].ci_lower
    )
