import numpy as np
import pandas as pd
from types import SimpleNamespace

from nhis_fairbias.benchmark.cohort_reporting import summarize_cohort, summarize_partition, summarize_eligibility
from nhis_fairbias.benchmark.data_contracts import AnnualSurveyDesign


def _partition(metadata=None):
    n = 4
    design = AnnualSurveyDesign(
        2024, np.array(["r1", "r2", "r3", "r4", "zero_psu"]),
        np.array([1, 1, 2, 2, 3]), np.array([10, 10, 20, 21, 30]),
        np.ones(5), np.array([True, True, True, True, False]),
    )
    return SimpleNamespace(
        role="evaluation_T", year=2024, arm_id="arm_001", record_keys=np.array(["r1", "r2", "r3", "r4"]),
        X_semantic=pd.DataFrame({"x": [1, None, 3, 4]}),
        y=np.array([1, 0, 1, 0]), A=np.array([1, 1, 2, 2]), WTFA_A=np.array([1., 2., 3., 4.]),
        PSTRAT=np.array([1, 1, 2, 2]), PPSU=np.array([10, 10, 20, 21]), feature_names=("x",),
        metadata=metadata or {}, annual_design=design,
    )


def test_reports_weighted_overall_groups_kish_and_full_design_coverage():
    result = summarize_partition(_partition(), expected_groups=[1, 2])
    assert result["overall"] == {
        "n": 4, "event_count": 2, "weight_sum": 10.0,
        "weighted_event_rate": 0.4, "kish_ess": 10.0**2 / 30.0, "status": "VALID",
    }
    assert result["groups"]["1"]["weight_sum"] == 3.0
    assert result["groups"]["2"]["weighted_event_rate"] == 3.0 / 7.0
    assert result["design"]["design_rows"] == 5
    assert result["design"]["strata_count"] == 3
    assert result["design"]["psu_count"] == 4
    assert {x["stratum"] for x in result["design"]["strata_psu_coverage"]} == {1, 2, 3}
    assert result["feature_missingness"]["status"] == "SEMANTIC_AVAILABLE"
    assert result["feature_missingness"]["semantic_layer"]["x"]["semantic_missing_count"] == 1
    assert result["feature_missingness"]["raw_cdc"]["status"] == "UNAVAILABLE"
    assert result["design"]["domain_psu_count"] == 3
    assert "record_keys" not in result


def test_raw_missingness_only_reported_when_explicit_aggregate_is_attached():
    p = _partition({"raw_feature_missingness": {"x": {"missing_count": 1, "denominator": 4}}})
    result = summarize_partition(p, expected_groups=[1, 2])
    assert result["feature_missingness"]["raw_cdc"]["features"]["x"]["missing_fraction"] == 0.25


def test_unknown_uses_existing_fitted_vocabulary_without_refitting():
    class Fitted:
        cat_vocabularies_ = {"x": ["1.0", "3.0", "MISSING", "UNKNOWN"]}
    result = summarize_partition(_partition(), expected_groups=[1, 2], fitted_preprocessor=Fitted())
    assert result["feature_missingness"]["semantic_layer"]["x"]["unknown_count"] == 1


def test_cohort_roles_and_invalid_weights_are_rejected():
    p = _partition()
    summary = summarize_cohort({"fitting_F": p, "calibration_C": p}, expected_groups=[1, 2])
    assert set(summary["partitions"]) == {"fitting_F", "calibration_C"}
    p.WTFA_A = np.array([1., 0., 1., 1.])
    try:
        summarize_partition(p)
    except ValueError as exc:
        assert "strictly positive" in str(exc)
    else:
        raise AssertionError("zero survey weight must be rejected")


def test_eligibility_reports_annual_item_reasons_and_weight_nonoverlap():
    cohort = pd.DataFrame({
        "year": [2024] * 5 + [2023] * 2,
        "WTFA_A": [1., 2., 3., 4., 5., 2., 3.],
        "MEDDL12M_A": [1, 0, np.nan, 1, 1, 0, 1],
        "SEX_A": [1, 2, 1, np.nan, 1, 2, 1],
    })
    specs = {"arm_001": {"protected_attribute": "SEX_A", "expected_categories": [1, 2]}}
    report = summarize_eligibility(cohort, specs)
    row = report["years"]["2024"]["arm_001"]
    assert row["full_annual"]["n"] == 5
    assert row["outcome_Y"]["valid"]["n"] == 4
    assert row["protected_A"]["valid"]["n"] == 4
    assert row["joint"]["eligible"]["n"] == 3
    reasons = row["exclusion_reasons"]
    assert reasons["both"]["n"] == 0
    assert reasons["onlyY"]["n"] == 1
    assert reasons["onlyA"]["n"] == 1
    assert reasons["both"]["n"] + reasons["onlyY"]["n"] + reasons["onlyA"]["n"] == row["joint"]["excluded"]["n"]
    assert sum(reasons[k]["weight_sum"] for k in reasons) == row["joint"]["excluded"]["weight_sum"]
    assert report["eligibility_semantics"].startswith("annual item eligibility")
