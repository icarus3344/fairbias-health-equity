"""Regression coverage for annual survey-domain mask normalization."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from nhis_fairbias.benchmark import catalog_evaluation as ev
from nhis_fairbias.benchmark.data_contracts import (
    generate_synthetic_nhis_cohort,
    load_arm_partitions,
)
from test_catalog_evaluation import _annual, _evaluate, _loader, study_case


def test_real_loader_nullable_columns_object_domain_preserves_annual_design():
    cohort = generate_synthetic_nhis_cohort(
        n_records_per_year=64,
        years=(2024,),
        n_strata=4,
        psus_per_stratum=2,
        seed=20260917,
    )
    cohort["SEX_A"] = cohort["SEX_A"].astype("Int64")
    cohort["MEDDL12M_A"] = cohort["MEDDL12M_A"].astype("Int64")

    # Make one complete sampled PSU contribute no domain rows, and add an
    # independent nullable outcome so this exercises both nullable inputs.
    first_group = cohort.groupby(["PSTRAT", "PPSU"], sort=False).size().index[0]
    zero_psu = (cohort["PSTRAT"] == first_group[0]) & (cohort["PPSU"] == first_group[1])
    cohort.loc[zero_psu, "SEX_A"] = pd.NA
    missing_outcome_index = cohort.index[~zero_psu][0]
    cohort.loc[missing_outcome_index, "MEDDL12M_A"] = pd.NA

    assert str(cohort["SEX_A"].dtype) == "Int64"
    assert str(cohort["MEDDL12M_A"].dtype) == "Int64"
    assert cohort["SEX_A"].isna().any() and cohort["MEDDL12M_A"].isna().any()

    T = load_arm_partitions(cohort, "arm_001")["evaluation_T"]
    original = T.annual_design
    original_mask = original.domain_mask.copy()
    original_fields = {
        field: np.asarray(getattr(original, field)).copy()
        for field in ("record_keys", "strata", "psus", "weights")
    }

    # Exercise the real loader output directly as well; pandas versions may
    # expose the loader mask as bool or as object, but normalization is stable.
    real_normalized = ev._normalized_annual_design(T)
    assert real_normalized.domain_mask.dtype == bool
    assert np.array_equal(real_normalized.domain_mask, original_mask)

    # This is the storage shape that triggered the regression after a pandas
    # nullable boolean mask was converted through to_numpy(dtype=object).
    object_mask = np.asarray(original.domain_mask, dtype=object)
    assert object_mask.dtype == object
    object_mask_before = object_mask.copy()
    object_design = dataclasses.replace(original, domain_mask=object_mask)
    object_T = dataclasses.replace(T, annual_design=object_design)
    normalized = ev._normalized_annual_design(object_T)

    assert normalized is not original
    assert normalized.domain_mask.dtype == bool
    assert normalized.domain_mask.sum() == len(T)
    assert np.array_equal(normalized.domain_mask, real_normalized.domain_mask)
    assert np.array_equal(object_design.domain_mask, object_mask_before)
    assert np.array_equal(original.domain_mask, original_mask)
    for field, before in original_fields.items():
        assert np.array_equal(getattr(original, field), before)

    # Every selected annual row must retain T's order and the exact design
    # values used by the estimator.
    for annual_field, partition_field in (
        ("record_keys", "record_keys"),
        ("strata", "PSTRAT"),
        ("psus", "PPSU"),
        ("weights", "WTFA_A"),
    ):
        assert np.array_equal(
            getattr(normalized, annual_field)[normalized.domain_mask],
            getattr(T, partition_field),
        )

    annual_zero_psu = (normalized.strata == first_group[0]) & (normalized.psus == first_group[1])
    assert annual_zero_psu.any()
    assert not normalized.domain_mask[annual_zero_psu].any()
    assert (first_group[0], first_group[1]) not in set(zip(T.PSTRAT, T.PPSU))


@pytest.mark.parametrize(
    "kind",
    ["na", "integer", "string", "two_dimensional"],
)
def test_normalized_annual_design_rejects_invalid_domain_values(kind):
    T = _annual()
    if kind == "na":
        mask = np.array([True] * 7 + [pd.NA], dtype=object)
    elif kind == "integer":
        mask = np.array([True] * 8, dtype=object)
        mask[0] = 1
    elif kind == "string":
        mask = np.array([True] * 8, dtype=object)
        mask[0] = "True"
    else:
        mask = np.ones((8, 1), dtype=bool)
    design = dataclasses.replace(T.annual_design, domain_mask=mask)
    with pytest.raises(ValueError, match="Annual domain"):
        ev._normalized_annual_design(_with_design(T, design))


@pytest.mark.parametrize("field", ["record_keys", "strata", "psus", "weights"])
def test_normalized_annual_design_rejects_misaligned_design_field(field):
    T = _annual()
    values = np.asarray(getattr(T.annual_design, field)).copy()
    values[0] = values[0] + 999 if field != "record_keys" else 999
    design = dataclasses.replace(T.annual_design, **{field: values})
    with pytest.raises(ValueError, match="Annual domain row alignment mismatch"):
        ev._normalized_annual_design(_with_design(T, design))


def _with_design(T, design):
    """Copy the SimpleNamespace fixture while replacing only its design."""
    return type(T)(**{**vars(T), "annual_design": design})


def _numeric_results(summary):
    ignored = {
        "model_id",
        "model_ids",
        "frozen_seed_models",
        "selection_id",
        "study_freeze_sha256",
        "selection_sha256",
        "evaluation_manifest_sha256",
        "reference_selection_id",
        "comparison_selection_id",
        "reference_model_ids",
        "comparison_model_ids",
    }

    def strip(value):
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if key not in ignored}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return strip({"selections": summary["selections"], "paired_contrasts": summary["paired_contrasts"]})


def test_complete_catalog_evaluation_object_and_bool_masks_have_identical_statistics(
    study_case, monkeypatch
):
    case = study_case

    bool_T = _annual()
    _loader(case, monkeypatch, bool_T)
    bool_summary = _evaluate(case, case["root"] / "bool_mask")

    object_design = dataclasses.replace(
        bool_T.annual_design,
        domain_mask=np.asarray(bool_T.annual_design.domain_mask, dtype=object),
    )
    object_T = _with_design(bool_T, object_design)
    _loader(case, monkeypatch, object_T)
    object_summary = _evaluate(case, case["root"] / "object_mask")

    assert _numeric_results(object_summary) == _numeric_results(bool_summary)
    for name in ("arm_001_individual_model_metrics.json",):
        left = json.loads((case["root"] / "bool_mask" / name).read_text())
        right = json.loads((case["root"] / "object_mask" / name).read_text())
        assert right == left

    left_arrays = np.load(case["root"] / "bool_mask" / "arm_001_replicate_metric_arrays.npz")
    right_arrays = np.load(case["root"] / "object_mask" / "arm_001_replicate_metric_arrays.npz")
    assert left_arrays.files == right_arrays.files
    for key in left_arrays.files:
        assert np.array_equal(right_arrays[key], left_arrays[key], equal_nan=True)
