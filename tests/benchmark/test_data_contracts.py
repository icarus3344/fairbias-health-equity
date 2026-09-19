"""Unit tests for benchmark data contracts, synthetic cohorts, and partitioning."""

import numpy as np
import pytest

from nhis_fairbias.benchmark.data_contracts import (
    ARM_SPECS,
    generate_synthetic_nhis_cohort,
    load_arm_partitions,
    make_record_key,
    partition_master_psus,
)


def test_make_record_key():
    k = make_record_key("nhis", 2022, "1001")
    assert k == "nhis:2022:1001"
    # Handles float representation canonicalization
    k2 = make_record_key("nhis", 2023.0, "1002")
    assert k2 == "nhis:2023:1002"


def test_master_psu_partitioning_atomic():
    strata = np.array([101, 101, 101, 102, 102, 102])
    psus = np.array([1, 1, 2, 1, 2, 2])
    mapping = partition_master_psus(strata, psus, seed=20260913)

    # Every unique cluster (s, p) is assigned atomically
    for cluster in {(101, 1), (101, 2), (102, 1), (102, 2)}:
        assert cluster in mapping
        assert mapping[cluster] in ("fitting_F", "calibration_C")


def test_generate_synthetic_cohort_invariants():
    df = generate_synthetic_nhis_cohort(
        n_records_per_year=100,
        years=(2022, 2023, 2024),
        seed=42,
    )
    assert len(df) == 300
    assert set(df["year"].unique()) == {2022, 2023, 2024}
    # Weights must be finite and strictly positive
    assert np.all(df["WTFA_A"] > 0)
    assert np.all(np.isfinite(df["WTFA_A"]))
    # Binary outcome MEDDL12M_A
    assert set(df["MEDDL12M_A"].unique()).issubset({0, 1})


def test_arm3_arm4_cohort_parity():
    """Arm 003 and Arm 004 must have identical respondents and weights, differing only by features."""
    df = generate_synthetic_nhis_cohort(
        n_records_per_year=200,
        years=(2022, 2023, 2024),
        seed=100,
    )
    p3 = load_arm_partitions(df, "arm_003")
    p4 = load_arm_partitions(df, "arm_004")

    for role in ("fitting_F", "calibration_C", "selection_S", "evaluation_T"):
        part3 = p3[role]
        part4 = p4[role]
        # Identical row count
        assert len(part3) == len(part4)
        # Identical record keys
        assert np.array_equal(part3.record_keys, part4.record_keys)
        # Identical weights
        assert np.allclose(part3.WTFA_A, part4.WTFA_A)
        # Identical outcomes
        assert np.array_equal(part3.y, part4.y)
        # Identical sensitive attributes
        assert np.array_equal(part3.A, part4.A)
        # Feature difference: Arm 003 has 21 features, Arm 004 has 15 features
        assert len(part3.feature_names) == 21
        assert len(part4.feature_names) == 15
        assert set(part3.feature_names) - set(part4.feature_names) == {
            "visiondf_a",
            "hearingdf_a",
            "diff_a",
            "comdiff_a",
            "uppslfcr_a",
            "cogmemdff_a",
        }
