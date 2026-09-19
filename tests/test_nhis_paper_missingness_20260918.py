"""Synthetic boundary tests for the aggregate-only paper missingness exporter."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_nhis_predictor_missingness_20260918.py"
sys.path.insert(0, str(SCRIPT.parents[1] / "src"))
SPEC = importlib.util.spec_from_file_location("nhis_paper_missingness", SCRIPT)
assert SPEC and SPEC.loader
missingness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(missingness)


def test_employment_routing_distinguishes_structural_item_null_and_unknown() -> None:
    frame = pd.DataFrame(
        {
            "EMPWRKLSW1_A": [2, 1, 1, 1, 1, 7, 5, np.nan],
            "EMPWRKFT1_A": [np.nan, 7, np.nan, 1, 3, np.nan, 1, np.nan],
        }
    )
    spec = {
        "official_name": "EMPWRKFT1_A",
        "substantive_codes": [1, 2],
        "missing_codes": [7, 8, 9],
    }
    result = missingness.classify_variable(frame, "EMPWRKFT1_A", spec, pd.Series([1.0] * len(frame)))

    assert result["denominator_n"] == 8
    assert result["observed"]["n"] == 1
    assert result["structural_niu"]["n"] == 1, "EMPWRKLSW1_A=2 routes even when intensity is raw null"
    assert result["item_nonresponse"]["n"] == 2, "official refusal/don't-know codes remain item nonresponse"
    assert result["raw_null"]["n"] == 2, "blank intensity or blank routing remains raw-null"
    assert result["unknown_unmapped"]["n"] == 2, "unexpected intensity/routing codes remain unknown"
    assert result["preprocessing_missing"]["n"] == 7
    assert sum(result[state]["n"] for state in ("observed", *missingness.CAT_MISSINGNESS)) == 8


def test_weighted_fraction_changes_under_nonuniform_survey_weights() -> None:
    frame = pd.DataFrame({"RATCAT_A": [1, 98, 1]})
    spec = {
        "official_name": "RATCAT_A",
        "substantive_codes": list(range(1, 15)),
        "missing_codes": [98],
    }
    result = missingness.classify_variable(frame, "RATCAT_A", spec, pd.Series([1.0, 9.0, 1.0]))

    assert result["item_nonresponse"]["fraction"] == 1 / 3
    assert result["item_nonresponse"]["weighted_fraction"] == 9 / 11
    assert result["item_nonresponse"]["weighted_fraction"] != result["item_nonresponse"]["fraction"]


def test_ratcat_item_nonresponse_raw_null_and_unknown_are_separate() -> None:
    frame = pd.DataFrame({"RATCAT_A": [1, 7, 98, np.nan, 99]})
    spec = {
        "official_name": "RATCAT_A",
        "substantive_codes": list(range(1, 15)),
        "missing_codes": [98],
    }
    result = missingness.classify_variable(frame, "RATCAT_A", spec, pd.Series([2.0] * len(frame)))

    assert result["observed"]["n"] == 2
    assert result["item_nonresponse"]["n"] == 1
    assert result["raw_null"]["n"] == 1
    assert result["unknown_unmapped"]["n"] == 1
    assert result["item_nonresponse"]["weighted_fraction"] == 0.2
    assert result["preprocessing_missing"]["n"] == 3


def _synthetic_raw(year: int, n: int = 4) -> pd.DataFrame:
    registry, specs = missingness.load_feature_specs()
    frame: dict[str, list[object]] = {
        "HHX": [f"H{year}{i:04d}" for i in range(n)],
        "WTFA_A": [1.0 + i for i in range(n)],
        "PSTRAT": [10, 10, 11, 11],
        "PPSU": [1, 2, 1, 2],
        "MEDDL12M_A": [1, 2, 1, 2],
        "MEDNG12M_A": [1, 2, 1, 2],
        "SEX_A": [1, 2, 1, 2],
        "HISPALLP_A": [1, 2, 3, 4],
        "DISAB3_A": [1, 2, 1, 2],
    }
    for harm_name in registry["feature_lists"]["primary_core_features"]:
        spec = specs[harm_name]
        raw_name = spec["official_name"]
        valid = int(spec["substantive_codes"][0])
        frame[raw_name] = [valid] * n
    # Required for EMPWRKFT1_A routing and identity-bound domain membership.
    frame["EMPWRKLSW1_A"] = [1, 2, 1, 1]
    frame["EMPWRKFT1_A"] = [1, np.nan, 2, 1]
    raw = pd.DataFrame(frame)
    raw["_record_key"] = "nhis:" + str(year) + ":" + raw["HHX"].astype(str)
    return raw


def test_build_aggregates_emits_arm004_exclusions_as_na() -> None:
    registry, specs = missingness.load_feature_specs()
    raw_frames = {2022: _synthetic_raw(2022), 2023: _synthetic_raw(2023), 2024: _synthetic_raw(2024)}
    all_keys = {year: set(frame["_record_key"]) for year, frame in raw_frames.items()}
    domains = {
        arm: {
            "fitting_F": all_keys[2022],
            "calibration_C": all_keys[2022],
            "selection_S": all_keys[2023],
        }
        for arm in missingness.ARM_ORDER
    }

    rows, eligibility = missingness.build_aggregates(raw_frames, domains, registry, specs)

    assert len(rows) == 4 * 4 * 21
    assert len(eligibility) == 4 * 4
    excluded = [
        row for row in rows
        if row["arm_id"] == "arm_004" and row["variable"] == "diff_a"
    ]
    assert len(excluded) == 4
    assert all(row["status"] == "NA_EXCLUDED_ARM004" for row in excluded)
    assert all(row["denominator_n"] == "NA" for row in excluded)
    assert all(row["observed_n"] == "NA" and row["item_nonresponse_n"] == "NA" for row in excluded)

    included = next(
        row for row in rows
        if row["arm_id"] == "arm_001" and row["partition"] == "F" and row["variable"] == "empwrkft1_a"
    )
    assert included["category_sum_n"] == included["denominator_n"] == 4
    assert included["structural_niu_n"] == 1
    assert included["observed_n"] == 3


def test_eligibility_exclusions_keep_full_year_denominators() -> None:
    raw = _synthetic_raw(2024)
    raw.loc[0, "MEDDL12M_A"] = 7
    raw.loc[1, "DISAB3_A"] = 9
    domain = pd.Series([True, True, False, False], index=raw.index)
    row = missingness.eligibility_rows(raw, "arm_003", "evaluation_T", domain, "DISAB3_A")

    assert row["full_year_n"] == 4
    assert row["full_year_outcome_invalid_n"] == 1
    assert row["full_year_protected_invalid_n"] == 1
    assert row["full_year_joint_eligible_n"] == 2
    assert row["full_year_joint_excluded_n"] == 2
    assert row["analysis_domain_n"] == 2
    assert row["analysis_domain_joint_eligible_n"] == 0
    assert row["analysis_domain_joint_excluded_n"] == 2
