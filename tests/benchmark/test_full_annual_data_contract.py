"""Synthetic witnesses for annual domains, source identity and split isolation."""
import numpy as np
import pandas as pd
import pytest

from nhis_fairbias.benchmark.data_contracts import (
    generate_synthetic_nhis_cohort, load_arm_partitions,
    load_processed_nhis_cohort, make_record_key,
)


def cohort():
    frame = generate_synthetic_nhis_cohort(n_records_per_year=24, n_strata=3, seed=4)
    for year in (2022, 2023, 2024):
        mask = frame.year == year
        frame.loc[mask, 'PSTRAT'] = np.repeat([101, 102, 103], 8)
        frame.loc[mask, 'PPSU'] = np.tile(np.repeat([1, 2], 4), 3)
    return frame


def test_domain_exclusion_preserves_master_split_and_annual_psus():
    data = cohort()
    before = load_arm_partitions(data, 'arm_001')
    outside = (data.PSTRAT == 101) & (data.PPSU == 1)
    data.loc[outside, 'SEX_A'] = np.nan
    data.loc[outside, 'MEDDL12M_A'] = np.nan
    after = load_arm_partitions(data, 'arm_001')
    for role in before:
        assert before[role].metadata['psu_assignment_sha256'] == after[role].metadata['psu_assignment_sha256']
        assert np.array_equal(before[role].annual_design.record_keys, after[role].annual_design.record_keys)
    part = after['evaluation_T']
    design = part.annual_design
    assert len(design.record_keys) == 24
    assert len(part) == 20
    expanded = design.expand(np.ones(20))
    assert np.all(expanded[(design.strata == 101) & (design.psus == 1)] == 0)
    assert np.array_equal(design.record_keys[design.domain_mask], part.record_keys)
    with pytest.raises(ValueError, match='align'):
        design.expand(np.ones(19))


def test_disability_arms_share_master_and_respondents_with_missing_outcome():
    data = cohort()
    data.loc[0, 'MEDDL12M_A'] = np.nan
    a = load_arm_partitions(data, 'arm_003')
    b = load_arm_partitions(data, 'arm_004')
    for role in a:
        assert np.array_equal(a[role].record_keys, b[role].record_keys)
        assert np.array_equal(a[role].annual_design.domain_mask, b[role].annual_design.domain_mask)
        assert np.array_equal(a[role].annual_design.weights, b[role].annual_design.weights)
    assert sum(len(a[k]) for k in ('fitting_F', 'calibration_C')) == 23


@pytest.mark.parametrize('column,value', [('year', 2022.5), ('PPSU', 1.2), ('WTFA_A', -1), ('MEDDL12M_A', 2)])
def test_reject_malformed_identifiers_design_and_unrecoded_outcome(column, value):
    data = cohort()
    data[column] = data[column].astype(float)
    data.loc[0, column] = value
    with pytest.raises(ValueError):
        load_arm_partitions(data, 'arm_001')


def test_source_identity_is_required_and_never_synthesized(monkeypatch):
    data = cohort()
    data.loc[1, 'HHX'] = data.loc[0, 'HHX']
    with pytest.raises(ValueError, match='Duplicate'):
        load_arm_partitions(data, 'arm_001')
    monkeypatch.setattr(pd, 'read_parquet', lambda path: cohort().drop(columns='HHX'))
    with pytest.raises(ValueError, match='positional'):
        load_processed_nhis_cohort('synthetic-input.parquet')
    with pytest.raises(ValueError, match='finite integer'):
        make_record_key('nhis', 2022.5, '001')
    assert make_record_key('nhis', 2022, '001') != make_record_key('nhis', 2022, '1')
