"""Synthetic verification suite for REGISTRY-1/2/3 and WEIGHT-1/2/3/4/5 contracts."""

import copy
import hashlib
import json
import pathlib
import numpy as np
import pandas as pd
import pytest

from nhis_fairbias.adapter import NHISStudyAdapter, load_feature_registry
from nhis_fairbias.preprocessing import NHISPreprocessor, PreprocessingFitRecord
from nhis_fairbias.survey import (
    validate_survey_weights,
    weighted_binary_proportion,
    weighted_category_proportion,
)


def _make_synthetic_raw_df() -> pd.DataFrame:
    """Create minimal synthetic dataframe (24 rows total, 8 per partition) conforming to registry schema."""
    reg = load_feature_registry("configs/nhis/features.json")
    prep = NHISPreprocessor(feature_registry=reg)

    roles = {2022: "development_train", 2023: "development_validation", 2024: "frozen_test"}
    rows = []
    for yr in (2022, 2023, 2024):
        cnt = 8
        sub_data = {
            "survey_year": [yr] * cnt,
            "study_role": [roles[yr]] * cnt,
            "agep_a": [25, 30, 35, 40, 45, 50, 55, 60],
            "pcnt18uptc": [1] * cnt,
            "pcntlt18tc": [0] * cnt,
            "emerg12mtc_a": [0] * cnt,
            "wtfa_a": [1.0] * cnt,
            "pstrat": [100] * cnt,
            "ppsu": [1] * cnt,
        }
        for col in prep.categorical_features:
            sub_data[col] = [1] * cnt
        rows.append(pd.DataFrame(sub_data))
    return pd.concat(rows, ignore_index=True)


def _make_dummy_fit_record(fit_year: int = 2022, fit_study_role: str = "development_train") -> PreprocessingFitRecord:
    return PreprocessingFitRecord(
        fit_year=fit_year,
        fit_study_role=fit_study_role,
        row_count=8,
        numerical_medians={},
        numerical_stats={},
        categorical_categories={},
        empwrkft_distribution={},
        rules_manifest={},
    )


def test_registry_1_matching_preprocessor_accepted(monkeypatch):
    """REGISTRY-1: Synthetic fitted preprocessor with matching schema and valid provenance is accepted."""
    reg = load_feature_registry("configs/nhis/features.json")
    prep = NHISPreprocessor(feature_registry=reg)
    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()
    prep.fit(train_df)
    assert prep.is_fitted
    assert prep.fitted_record.fit_year == 2022
    assert prep.fitted_record.row_count == 8

    orig_is_file = pathlib.Path.is_file
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: True if "synthetic_dummy" in str(self) else orig_is_file(self))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: raw_df)
    monkeypatch.setattr(NHISStudyAdapter, "_validate_partitions", lambda self: None)

    adapter = NHISStudyAdapter(
        features_parquet_path="synthetic_dummy.parquet",
        preprocessor=prep,
    )
    assert adapter.preprocessor is prep


def test_registry_2_modified_feature_schema_rejected(monkeypatch):
    """REGISTRY-2: Preprocessor with modified AGEP_A valid range or encoding is rejected (not just key set)."""
    reg = load_feature_registry("configs/nhis/features.json")
    corrupt_reg = copy.deepcopy(reg)
    # Modify valid_range of AGEP_A from None to [0, 100]
    corrupt_reg["primary_core"]["AGEP_A"]["valid_range"] = [0, 100]

    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()
    prep = NHISPreprocessor(feature_registry=corrupt_reg)
    prep.fit(train_df)

    orig_is_file = pathlib.Path.is_file
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: True if "synthetic_dummy" in str(self) else orig_is_file(self))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: raw_df)
    monkeypatch.setattr(NHISStudyAdapter, "_validate_partitions", lambda self: None)

    with pytest.raises(ValueError, match="feature registry mismatch on feature 'agep_a' field 'valid_range'"):
        NHISStudyAdapter(
            features_parquet_path="synthetic_dummy.parquet",
            preprocessor=prep,
        )


def test_registry_3_missing_metadata_and_provenance_rejected(monkeypatch):
    """REGISTRY-3: Missing registry or invalid provenance rejected."""
    reg = load_feature_registry("configs/nhis/features.json")
    raw_df = _make_synthetic_raw_df()
    prep = NHISPreprocessor(feature_registry=reg)
    # Provenance fitted on 2023 instead of 2022
    prep._fitted_record = _make_dummy_fit_record(2023, "development_validation")

    orig_is_file = pathlib.Path.is_file
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: True if "synthetic_dummy" in str(self) else orig_is_file(self))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: raw_df)
    monkeypatch.setattr(NHISStudyAdapter, "_validate_partitions", lambda self: None)

    with pytest.raises(ValueError, match="invalid provenance"):
        NHISStudyAdapter(
            features_parquet_path="synthetic_dummy.parquet",
            preprocessor=prep,
        )


def test_weight_1_extreme_weight_proportion():
    """WEIGHT-1: Two records with weights 1e308 and 1e308 yield exactly 0.5 without overflow."""
    codes = pd.Series([1, 2])
    weights = pd.Series([1e308, 1e308])

    prop = weighted_binary_proportion(codes, weights, code=1, valid_codes=(1, 2))
    assert prop == 0.5, f"Expected 0.5 for two equal 1e308 weights, got {prop}"

    cat_prop = weighted_category_proportion(codes, weights, code=1, valid_codes=(1, 2))
    assert cat_prop == 0.5, f"Expected 0.5 for category proportion, got {cat_prop}"


def test_weight_2_scale_invariance():
    """WEIGHT-2: Scalar multiplication of weights preserves proportions exactly."""
    codes = pd.Series([1, 1, 2, 2])
    weights = pd.Series([10.0, 20.0, 30.0, 40.0])
    scaled_weights = weights * 1e12

    p1 = weighted_binary_proportion(codes, weights, code=1, valid_codes=(1, 2))
    p2 = weighted_binary_proportion(codes, scaled_weights, code=1, valid_codes=(1, 2))

    assert abs(p1 - p2) < 1e-12, "Proportion must be invariant to positive scalar multiplication"


def test_weight_3_invalid_weight_rejection():
    """WEIGHT-3: Negative, NaN, Inf, all-zero, or mismatched index rejected by validate_survey_weights."""
    # Negative weight
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_survey_weights(pd.Series([1.0, -0.5]))

    # NaN weight
    with pytest.raises(ValueError, match="NaN, Inf, or non-finite"):
        validate_survey_weights(pd.Series([1.0, float("nan")]))

    # Inf weight
    with pytest.raises(ValueError, match="NaN, Inf, or non-finite"):
        validate_survey_weights(pd.Series([1.0, float("inf")]))

    # All zero
    with pytest.raises(ValueError, match="cannot be all-zero"):
        validate_survey_weights(pd.Series([0.0, 0.0]))

    # Index mismatch
    with pytest.raises(ValueError, match="Index alignment mismatch"):
        validate_survey_weights(pd.Series([1.0, 2.0], index=[0, 1]), expected_index=pd.Index([1, 2]))

    # Sum overflow to infinity rejected
    with pytest.raises(ValueError, match="overflows to infinity"):
        validate_survey_weights(pd.Series([1e308, 1e308]))


def test_weight_4_insufficient_group_weight_stability():
    """WEIGHT-4: Zero or missing positive weights handled with stable status (None, not NaN or crash)."""
    codes = pd.Series([1, 2])
    # Weights all zero or missing
    weights = pd.Series([0.0, 0.0])
    prop = weighted_binary_proportion(codes, weights, code=1, valid_codes=(1, 2))
    assert prop is None, "When no positive weights exist, proportion must be None"


def test_weight_5_proportion_alignment_and_uniqueness():
    """WEIGHT-5: Direct proportion functions strictly reject index alignment and uniqueness mismatches."""
    # Partial index mismatch
    codes = pd.Series([1, 2], index=["a", "b"])
    weights = pd.Series([1.0, 1.0], index=["b", "c"])
    with pytest.raises(ValueError, match="Index alignment mismatch"):
        weighted_binary_proportion(codes, weights, code=1)
    with pytest.raises(ValueError, match="Index alignment mismatch"):
        weighted_category_proportion(codes, weights, code=1, valid_codes=(1, 2))

    # Length mismatch
    with pytest.raises(ValueError, match="Input length mismatch"):
        weighted_binary_proportion(pd.Series([1, 2]), pd.Series([1.0]), code=1)

    # Non-unique indices
    with pytest.raises(ValueError, match="raw_codes index must be unique"):
        weighted_binary_proportion(pd.Series([1, 2], index=["a", "a"]), pd.Series([1.0, 1.0], index=["a", "a"]), code=1)
    with pytest.raises(ValueError, match="weights index must be unique"):
        weighted_binary_proportion(pd.Series([1, 2], index=["a", "b"]), pd.Series([1.0, 1.0], index=["a", "a"]), code=1)


def test_registry_4_reversed_feature_lists_order_rejected(monkeypatch):
    """REGISTRY-4: Adapter rejects pre-fitted preprocessor if feature_lists order does not match."""
    reg = load_feature_registry("configs/nhis/features.json")
    reordered_reg = copy.deepcopy(reg)
    reordered_reg["feature_lists"]["primary_core_features"].reverse()

    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()
    prep = NHISPreprocessor(feature_registry=reordered_reg)
    prep.fit(train_df)

    orig_is_file = pathlib.Path.is_file
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: True if "synthetic_dummy" in str(self) else orig_is_file(self))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: raw_df)
    monkeypatch.setattr(NHISStudyAdapter, "_validate_partitions", lambda self: None)

    with pytest.raises(ValueError, match="feature_lists 'primary_core_features' mismatch"):
        NHISStudyAdapter(
            features_parquet_path="synthetic_dummy.parquet",
            preprocessor=prep,
        )


def test_registry_5_caller_dict_mutation_after_fit_invariant(monkeypatch):
    """REGISTRY-5: External caller mutating its feature_registry dictionary after fit does NOT alter preprocessor cleaning rules."""
    reg = load_feature_registry("configs/nhis/features.json")
    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()
    prep = NHISPreprocessor(feature_registry=reg)
    prep.fit(train_df)

    # Test frame with out-of-range age (90 is outside substantive [18, 85])
    test_df = train_df.iloc[:1].copy()
    test_df["agep_a"] = 90

    # Before mutating caller's dict
    out_before = prep.transform(test_df)
    train_median_age = prep.fitted_record.numerical_medians["agep_a"]
    assert float(out_before["agep_a"].iloc[0]) == train_median_age, (
        f"Age 90 must be imputed to train median {train_median_age}"
    )

    # External caller mutates their dictionary to accept [0, 100]
    reg["primary_core"]["AGEP_A"]["substantive_codes"] = [0, 100]

    # After mutating caller's dict: preprocessor must be unaffected because __init__ made a deep copy
    out_after = prep.transform(test_df)
    assert float(out_after["agep_a"].iloc[0]) == train_median_age, (
        "Preprocessor must be isolated from external dict mutation"
    )


def test_registry_6_internal_mutation_raises_error():
    """REGISTRY-6: Direct mutation of preprocessor's specs after fit causes transform to raise ValueError."""
    reg = load_feature_registry("configs/nhis/features.json")
    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()
    prep = NHISPreprocessor(feature_registry=reg)
    prep.fit(train_df)

    test_df = train_df.iloc[:1].copy()

    # Direct mutation of preprocessor's specs
    prep.specs["agep_a"]["substantive_codes"] = [0, 100]

    with pytest.raises(ValueError, match="Registry rules were mutated after fit"):
        prep.transform(test_df)


def test_t_frozen_preprocessor_immutability():
    """T-FROZEN: NHISPreprocessor.fit freezes output column order and fitted statistics against mutation."""
    reg = load_feature_registry("configs/nhis/features.json")
    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()
    prep = NHISPreprocessor(feature_registry=reg).fit(train_df)

    test_df = train_df.iloc[:1].copy()
    test_df["agep_a"] = 90
    cols_before = list(prep.transform(test_df, preserve_metadata=False).columns)

    # 1. Reverse primary_core_features on the instance: transform output columns must NOT change
    prep.primary_core_features.reverse()
    cols_after = list(prep.transform(test_df, preserve_metadata=False).columns)
    assert cols_before == cols_after, "Reversing primary_core_features must not affect transform column ordering"

    # 2. Mutating fitted_record numerical median must not change subsequent transform behavior
    med_before = float(prep.transform(test_df)["agep_a"].iloc[0])
    prep.fitted_record.numerical_medians["agep_a"] = 77.0
    med_after = float(prep.transform(test_df)["agep_a"].iloc[0])
    assert med_before == med_after, "Mutating fitted_record dictionary must not change frozen transform median"


def test_t_load_versioned_fit_and_rule_mismatch(tmp_path, monkeypatch):
    """T-LOAD: Fit with age 0-100 exported; official preprocessor load must reject mismatch and adapter rejects."""
    reg = load_feature_registry("configs/nhis/features.json")
    altered_reg = copy.deepcopy(reg)
    altered_reg["primary_core"]["AGEP_A"]["substantive_codes"] = [0, 100]

    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()
    train_df["agep_a"] = 90

    donor = NHISPreprocessor(feature_registry=altered_reg).fit(train_df)
    fit_path = tmp_path / "donor_fit.json"
    donor.export_fit_json(fit_path)

    # Official preprocessor loading altered rules artifact must reject mismatch
    official_prep = NHISPreprocessor(feature_registry=reg)
    with pytest.raises(ValueError, match="rule identity mismatch"):
        official_prep.load_fit_json(fit_path)

    # Valid same-schema roundtrip matches exactly
    train_df_good = train_df.copy()
    train_df_good["agep_a"] = 45
    good_donor = NHISPreprocessor(feature_registry=reg).fit(train_df_good)
    good_path = tmp_path / "official_fit.json"
    good_donor.export_fit_json(good_path)

    restored = NHISPreprocessor(feature_registry=reg).load_fit_json(good_path)
    assert restored.is_fitted
    assert restored.transform(train_df_good)["agep_a"].tolist() == good_donor.transform(train_df_good)["agep_a"].tolist()


def test_r4_01_lifecycle_legacy_unverified_export_and_adapter_denial(tmp_path, monkeypatch):
    """R4-01: Legacy unverified artifact must not gain verified status across export/load; adapter must reject."""
    reg = load_feature_registry("configs/nhis/features.json")
    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()

    # Fit a valid donor to get realistic statistics
    donor = NHISPreprocessor(feature_registry=reg).fit(train_df)
    fit_path = tmp_path / "valid_fit.json"
    donor.export_fit_json(fit_path)

    # Convert to legacy format (no rule_identity or format_version)
    import json
    with open(fit_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.pop("rule_identity", None)
    data.pop("format_version", None)
    legacy_file = tmp_path / "synthetic_legacy.json"
    with open(legacy_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # 1. Load legacy artifact into preprocessor
    legacy_prep = NHISPreprocessor(feature_registry=reg).load_fit_json(legacy_file)
    assert legacy_prep.is_legacy_unverified is True, "Legacy artifact without rule_identity must be flagged unverified"

    # Adapter must reject legacy unverified artifact
    orig_is_file = pathlib.Path.is_file
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: True if "synthetic_dummy" in str(self) else orig_is_file(self))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: raw_df)
    monkeypatch.setattr(NHISStudyAdapter, "_validate_partitions", lambda self: None)

    with pytest.raises(ValueError, match="rejected unverified legacy preprocessor"):
        NHISStudyAdapter(features_parquet_path="synthetic_dummy.parquet", preprocessor=legacy_prep)

    # 2. Export of unverified legacy object MUST be rejected explicitly
    roundtrip_path = tmp_path / "legacy_reexported.json"
    with pytest.raises(ValueError, match="Cannot export unverified legacy"):
        legacy_prep.export_fit_json(roundtrip_path)
    assert not roundtrip_path.exists(), "Export file must not be created when export is denied"


def test_r5_03_lifecycle_export_sensitivity_mutant_detection(tmp_path):
    """R5-03: Verify test sensitivity by injecting known bad export mutant into an isolated preprocessor;
    demonstrate that mutant improperly permits export and clears legacy flag upon reload, proving fault detection."""
    reg = load_feature_registry("configs/nhis/features.json")
    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()

    donor = NHISPreprocessor(feature_registry=reg).fit(train_df)
    fit_path = tmp_path / "temp_donor.json"
    donor.export_fit_json(fit_path)

    with open(fit_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.pop("rule_identity", None)
    data.pop("format_version", None)
    legacy_file = tmp_path / "synthetic_legacy_probe.json"
    with open(legacy_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    legacy_prep = NHISPreprocessor(feature_registry=reg).load_fit_json(legacy_file)
    assert legacy_prep.is_legacy_unverified is True

    # 1. Inject known bad export mutant in-process:
    # The bad implementation fails to check is_legacy_unverified and attaches rule_identity
    def mutant_export_fit_json(self, path):
        rec_dict = self._fitted_record.to_dict()
        specs_json = json.dumps(self.specs, sort_keys=True, default=str)
        rec_dict["format_version"] = "nhis_fairbias_fit_v2"
        rec_dict["rule_identity"] = {
            "specs_hash": hashlib.sha256(specs_json.encode("utf-8")).hexdigest(),
            "specs": copy.deepcopy(self.specs),
            "primary_core_features": list(self.primary_core_features),
            "expanded_features": list(self.expanded_features),
            "categorical_features": list(self.categorical_features),
            "numerical_features": list(self.numerical_features),
        }
        with pathlib.Path(path).open("w", encoding="utf-8") as h:
            json.dump(rec_dict, h, indent=2)

    mutant_out = tmp_path / "mutant_exported.json"
    # Mutant succeeds when it should fail
    mutant_export_fit_json(legacy_prep, mutant_out)
    assert mutant_out.exists()

    # When reloaded, mutant artifact improperly clears is_legacy_unverified:
    reloaded_from_mutant = NHISPreprocessor(feature_registry=reg).load_fit_json(mutant_out)
    assert reloaded_from_mutant.is_legacy_unverified is False, "Mutant demonstrates improper elevation of legacy object"

    # 2. In contrast, under current production implementation, export_fit_json STRICTLY raises ValueError
    production_out = tmp_path / "production_exported.json"
    with pytest.raises(ValueError, match="Cannot export unverified legacy"):
        legacy_prep.export_fit_json(production_out)
    assert not production_out.exists()


def test_r5_01_load_fit_atomicity_and_schema_validation(tmp_path, monkeypatch):
    """R5-01: Consecutive operations on preprocessor enforce load/fit atomicity and rigorous schema/provenance validation."""
    reg = load_feature_registry("configs/nhis/features.json")

    # 1. Legitimate legacy object: fit with age substantive_codes [18, 85], 8 rows of age 45
    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].iloc[:8].copy()
    train_df["agep_a"] = 45

    donor = NHISPreprocessor(feature_registry=reg).fit(train_df)
    assert donor.fitted_record.numerical_medians["agep_a"] == 45.0

    donor_fit_path = tmp_path / "donor_fit.json"
    donor.export_fit_json(donor_fit_path)
    with open(donor_fit_path, "r", encoding="utf-8") as f:
        donor_data = json.load(f)
    donor_data.pop("rule_identity", None)
    donor_data.pop("format_version", None)

    legacy_file = tmp_path / "synthetic_legacy_age45.json"
    with open(legacy_file, "w", encoding="utf-8") as f:
        json.dump(donor_data, f)

    # Load into official preprocessor instance
    prep = NHISPreprocessor(feature_registry=reg).load_fit_json(legacy_file)
    assert prep.is_legacy_unverified is True
    assert prep.fitted_record.numerical_medians["agep_a"] == 45.0

    # Adapter setup mocks
    orig_is_file = pathlib.Path.is_file
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: True if "synthetic_dummy" in str(self) else orig_is_file(self))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: raw_df)
    monkeypatch.setattr(NHISStudyAdapter, "_validate_partitions", lambda self: None)

    # Adapter initially rejects legacy unverified
    with pytest.raises(ValueError, match="rejected unverified legacy preprocessor"):
        NHISStudyAdapter(features_parquet_path="synthetic_dummy.parquet", preprocessor=prep)

    # Construct an artifact that has valid rule_identity matching official specs, but is MISSING row_count
    valid_prep = NHISPreprocessor(feature_registry=reg).fit(train_df)
    valid_path = tmp_path / "valid_v2.json"
    valid_prep.export_fit_json(valid_path)
    with open(valid_path, "r", encoding="utf-8") as f:
        missing_row_cnt_data = json.load(f)
    missing_row_cnt_data.pop("row_count", None)
    missing_row_cnt_file = tmp_path / "missing_row_count.json"
    with open(missing_row_cnt_file, "w", encoding="utf-8") as f:
        json.dump(missing_row_cnt_data, f)

    # Load missing_row_count into prep: MUST RAISE ValueError
    with pytest.raises(ValueError, match="row_count"):
        prep.load_fit_json(missing_row_cnt_file)

    # AFTER EXCEPTION: prep MUST STILL have is_legacy_unverified == True, export still denied, adapter still denied!
    assert prep.is_legacy_unverified is True, "Failed load must not clear is_legacy_unverified"
    assert prep.fitted_record.numerical_medians["agep_a"] == 45.0, "Prior statistics must remain untouched"
    with pytest.raises(ValueError, match="Cannot export unverified legacy"):
        prep.export_fit_json(tmp_path / "should_fail.json")
    with pytest.raises(ValueError, match="rejected unverified legacy preprocessor"):
        NHISStudyAdapter(features_parquet_path="synthetic_dummy.parquet", preprocessor=prep)

    # 2. Verified object -> failed load atomicity
    verified_prep = NHISPreprocessor(feature_registry=reg).load_fit_json(valid_path)
    assert verified_prep.is_legacy_unverified is False
    assert verified_prep.is_fitted is True
    orig_medians = copy.deepcopy(verified_prep.fitted_record.numerical_medians)
    orig_record_hash = hashlib.sha256(json.dumps(verified_prep.fitted_record.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()

    # Test cases that must be rejected:
    # a) NaN numerical median
    with open(valid_path, "r", encoding="utf-8") as f:
        nan_data = json.load(f)
    nan_data["numerical_medians"]["agep_a"] = float("nan")
    nan_file = tmp_path / "nan_median.json"
    with open(nan_file, "w", encoding="utf-8") as f:
        json.dump(nan_data, f)
    with pytest.raises(ValueError, match="Non-finite"):
        verified_prep.load_fit_json(nan_file)

    # b) Out of substantive range median (90 for age in official [18, 85])
    with open(valid_path, "r", encoding="utf-8") as f:
        out_of_range_data = json.load(f)
    out_of_range_data["numerical_medians"]["agep_a"] = 90.0
    oor_file = tmp_path / "oor_median.json"
    with open(oor_file, "w", encoding="utf-8") as f:
        json.dump(out_of_range_data, f)
    with pytest.raises(ValueError, match="outside substantive range"):
        verified_prep.load_fit_json(oor_file)

    # c) Rule identity missing primary_core_features
    with open(valid_path, "r", encoding="utf-8") as f:
        missing_core_data = json.load(f)
    missing_core_data["rule_identity"].pop("primary_core_features", None)
    mc_file = tmp_path / "missing_core.json"
    with open(mc_file, "w", encoding="utf-8") as f:
        json.dump(missing_core_data, f)
    with pytest.raises(ValueError, match="primary_core_features"):
        verified_prep.load_fit_json(mc_file)

    # d) Mismatched categorical/numerical attribution
    with open(valid_path, "r", encoding="utf-8") as f:
        cat_num_data = json.load(f)
    cat_num_data["rule_identity"]["categorical_features"].append("agep_a")
    cn_file = tmp_path / "cat_num_mismatch.json"
    with open(cn_file, "w", encoding="utf-8") as f:
        json.dump(cat_num_data, f)
    with pytest.raises(ValueError, match="Categorical features mismatch"):
        verified_prep.load_fit_json(cn_file)

    # e) Invalid fit_year (e.g. 2023)
    with open(valid_path, "r", encoding="utf-8") as f:
        bad_yr_data = json.load(f)
    bad_yr_data["fit_year"] = 2023
    by_file = tmp_path / "bad_year.json"
    with open(by_file, "w", encoding="utf-8") as f:
        json.dump(bad_yr_data, f)
    with pytest.raises(ValueError, match="Invalid fit_year"):
        verified_prep.load_fit_json(by_file)

    # f) Invalid fit_study_role
    with open(valid_path, "r", encoding="utf-8") as f:
        bad_role_data = json.load(f)
    bad_role_data["fit_study_role"] = "test_evaluation"
    br_file = tmp_path / "bad_role.json"
    with open(br_file, "w", encoding="utf-8") as f:
        json.dump(bad_role_data, f)
    with pytest.raises(ValueError, match="Invalid fit_study_role"):
        verified_prep.load_fit_json(br_file)

    # Verify verified_prep status and content are completely unchanged after all these failed loads!
    assert verified_prep.is_legacy_unverified is False
    assert verified_prep.is_fitted is True
    curr_record_hash = hashlib.sha256(json.dumps(verified_prep.fitted_record.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()
    assert curr_record_hash == orig_record_hash, "Fitted record must remain perfectly intact after failed loads"
    assert verified_prep.fitted_record.numerical_medians == orig_medians

    # 3. Failed fit preserves prior state
    bad_df = pd.DataFrame({"dummy": [1, 2]})
    with pytest.raises(ValueError):
        verified_prep.fit(bad_df)
    assert verified_prep.is_legacy_unverified is False
    assert verified_prep.is_fitted is True
    post_bad_fit_hash = hashlib.sha256(json.dumps(verified_prep.fitted_record.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()
    assert post_bad_fit_hash == orig_record_hash, "Fitted record must remain intact after failed fit"

    # 4. Legitimate roundtrip preserves identity and stats
    reloaded_valid = NHISPreprocessor(feature_registry=reg).load_fit_json(valid_path)
    assert reloaded_valid.is_legacy_unverified is False
    assert reloaded_valid.is_fitted is True
    assert reloaded_valid.fitted_record.numerical_medians == orig_medians
    cats_v, nums_v = reloaded_valid.get_feature_family_lists("primary_core")
    assert len(cats_v) > 0 and len(nums_v) > 0


def test_r4_01_lifecycle_mutated_specs_before_export_rejected(tmp_path):
    """R4-01: Public specs mutated after fit must cause export_fit_json to reject, preventing binding old stats to current specs."""
    reg = load_feature_registry("configs/nhis/features.json")
    altered_reg = copy.deepcopy(reg)
    altered_reg["primary_core"]["AGEP_A"]["substantive_codes"] = [0, 100]

    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()
    train_df["agep_a"] = 90

    donor = NHISPreprocessor(feature_registry=altered_reg).fit(train_df)
    # Mutate specs back to official [18, 85]
    donor.specs["agep_a"]["substantive_codes"] = [18, 85]

    # Transform already rejects mutated specs
    with pytest.raises(ValueError, match="mutated"):
        donor.transform(train_df)

    # Export must ALSO reject mutated specs consistently
    rebound_path = tmp_path / "mutated_export.json"
    with pytest.raises(ValueError, match="mutated"):
        donor.export_fit_json(rebound_path)


def test_r4_01_version_and_order_validation(tmp_path, monkeypatch):
    """R4-01: Unknown format_version or mismatched column order in artifact must be rejected."""
    reg = load_feature_registry("configs/nhis/features.json")
    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()

    good = NHISPreprocessor(feature_registry=reg).fit(train_df)
    f = tmp_path / "good_v2.json"
    good.export_fit_json(f)

    # Case A: Unknown format_version
    import json
    with open(f, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["format_version"] = "unsupported_future_version"
    bad_ver = tmp_path / "bad_ver.json"
    with open(bad_ver, "w", encoding="utf-8") as handle:
        json.dump(data, handle)

    with pytest.raises(ValueError, match="format_version"):
        NHISPreprocessor(feature_registry=reg).load_fit_json(bad_ver)

    # Case B: Mismatched column ordering in rule_identity
    with open(f, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["rule_identity"]["primary_core_features"].reverse()
    bad_order = tmp_path / "bad_order.json"
    with open(bad_order, "w", encoding="utf-8") as handle:
        json.dump(data, handle)

    with pytest.raises(ValueError, match="mismatch"):
        NHISPreprocessor(feature_registry=reg).load_fit_json(bad_order)


def test_r4_01_legitimate_refit_clears_legacy_unverified(tmp_path, monkeypatch):
    """R4-01: A legacy unverified preprocessor that is legitimately refitted on valid 2022 data clears legacy status."""
    reg = load_feature_registry("configs/nhis/features.json")
    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()

    # Create legacy unverified artifact
    donor = NHISPreprocessor(feature_registry=reg).fit(train_df)
    fit_path = tmp_path / "temp.json"
    donor.export_fit_json(fit_path)
    import json
    with open(fit_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data.pop("rule_identity", None)
    data.pop("format_version", None)
    legacy_file = tmp_path / "legacy_for_refit.json"
    with open(legacy_file, "w", encoding="utf-8") as handle:
        json.dump(data, handle)

    prep = NHISPreprocessor(feature_registry=reg).load_fit_json(legacy_file)
    assert prep.is_legacy_unverified is True

    # Legitimate refit on valid 2022 development_train data
    prep.fit(train_df)
    assert prep.is_fitted is True
    assert prep.is_legacy_unverified is False, "Legitimate refit must clear unverified legacy flag"

    # Adapter accepts newly refitted preprocessor
    orig_is_file = pathlib.Path.is_file
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: True if "synthetic_dummy" in str(self) else orig_is_file(self))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: raw_df)
    monkeypatch.setattr(NHISStudyAdapter, "_validate_partitions", lambda self: None)

    adapter = NHISStudyAdapter(features_parquet_path="synthetic_dummy.parquet", preprocessor=prep)
    assert adapter.preprocessor is prep


def test_r4_02_adapter_get_cohort_invariant_to_public_feature_lists_mutation(monkeypatch):
    """R4-02: adapter.get_cohort uses frozen feature definitions; mutating public lists does not alter cohort matrix."""
    reg = load_feature_registry("configs/nhis/features.json")
    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()
    prep = NHISPreprocessor(feature_registry=reg).fit(train_df)

    # Setup adapter with synthetic data containing outcome and protected attribute
    raw_df["meddl12m"] = [0, 1] * 12
    raw_df["sex_a"] = [1, 2] * 12
    raw_df["WTFA_A"] = 1.0
    raw_df["PSTRAT"] = 100
    raw_df["PPSU"] = 1

    orig_is_file = pathlib.Path.is_file
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: True if "synthetic_dummy" in str(self) else orig_is_file(self))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: raw_df)
    monkeypatch.setattr(NHISStudyAdapter, "_validate_partitions", lambda self: None)

    adapter = NHISStudyAdapter(features_parquet_path="synthetic_dummy.parquet", preprocessor=prep)

    # 1. Before mutation: get_cohort returns 21 columns and strictly binary y
    cohort_before, y_before, _, _, _ = adapter.get_cohort(2022, outcome="MEDDL12M_A")
    assert cohort_before.shape[1] == 21, f"Expected 21 features, got {cohort_before.shape[1]}"
    assert "agep_a" in cohort_before.columns
    assert set(y_before.unique()).issubset({0, 1}), "y labels must be binary {0, 1}"

    # 2. Mutate public primary_core_features on preprocessor instance
    prep.primary_core_features.remove("agep_a")
    assert len(prep.primary_core_features) == 20

    # 3. get_cohort MUST remain invariant (still 21 columns) or explicitly reject mutation; NEVER silently 20 columns
    cohort_after, y_after, _, _, _ = adapter.get_cohort(2022, outcome="MEDDL12M_A")
    assert cohort_after.shape[1] == 21, (
        f"adapter.get_cohort must not silently drop agep_a from 21 to {cohort_after.shape[1]} columns"
    )
    assert "agep_a" in cohort_after.columns
    assert list(cohort_after.columns) == list(cohort_before.columns)
    assert set(y_after.unique()).issubset({0, 1})

    # 4. Modifying returned family lists does not affect internal preprocessor state
    cats, nums = prep.get_feature_family_lists("primary_core")
    cats_len = len(cats)
    cats.pop()
    cats_again, _ = prep.get_feature_family_lists("primary_core")
    assert len(cats_again) == cats_len, "get_feature_family_lists must return independent copies"


def test_r5_02_adapter_outcome_contract(monkeypatch):
    """R5-02: Adapter enforces strict harmonized outcome contract, binary {0, 1} targets, and rejects raw codes or unharmonized fields."""
    reg = load_feature_registry("configs/nhis/features.json")
    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()
    prep = NHISPreprocessor(feature_registry=reg).fit(train_df)

    # 1. Valid harmonized outcome: MEDDL12M_A -> column 'meddl12m' with binary {0, 1}
    valid_df = raw_df.copy()
    valid_df["meddl12m"] = [0, 1] * 12
    valid_df["sex_a"] = [1, 2] * 12
    valid_df["WTFA_A"] = 1.0
    valid_df["PSTRAT"] = 100
    valid_df["PPSU"] = 1

    orig_is_file = pathlib.Path.is_file
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: True if "synthetic_dummy" in str(self) else orig_is_file(self))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: valid_df)
    monkeypatch.setattr(NHISStudyAdapter, "_validate_partitions", lambda self: None)

    adapter = NHISStudyAdapter(features_parquet_path="synthetic_dummy.parquet", preprocessor=prep)
    X, y, o, w, psu = adapter.get_cohort(2022, outcome="MEDDL12M_A")
    assert set(y.unique()).issubset({0, 1})
    assert len(y) == 8

    # 2. Unknown outcome (e.g. agep_a) must be rejected
    with pytest.raises(ValueError, match="Unsupported outcome"):
        adapter.get_cohort(2022, outcome="agep_a")

    # 3. Only raw survey outcome column meddl12m_a exists with values {1, 2, 7, 8, 9}, but harmonized meddl12m is missing
    raw_only_df = valid_df.drop(columns=["meddl12m"]).copy()
    raw_only_df["meddl12m_a"] = [1, 2, 7, 8, 9, 1] * 4
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: raw_only_df)
    raw_adapter = NHISStudyAdapter(features_parquet_path="synthetic_dummy.parquet", preprocessor=prep)
    with pytest.raises(ValueError, match="Harmonized outcome column 'meddl12m' not found"):
        raw_adapter.get_cohort(2022, outcome="MEDDL12M_A")

    # 4. Harmonized outcome with non-binary values (e.g. {0, 1, 2}) must be rejected
    non_binary_df = valid_df.copy()
    non_binary_df["meddl12m"] = [0, 1, 2] * 8
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: non_binary_df)
    nb_adapter = NHISStudyAdapter(features_parquet_path="synthetic_dummy.parquet", preprocessor=prep)
    with pytest.raises(ValueError, match="non-binary"):
        nb_adapter.get_cohort(2022, outcome="MEDDL12M_A")


def test_r6_02_artifact_validation_and_pooled_roundtrip(tmp_path, monkeypatch):
    """R6-02: Strict artifact schema validation, pooled fit roundtrip equivalence, and adapter denial."""
    reg = load_feature_registry("configs/nhis/features.json")
    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()

    # 1. Temporal roundtrip: fit -> export -> load -> transform equivalence
    prep_temp = NHISPreprocessor(feature_registry=reg).fit(train_df)
    temp_path = tmp_path / "temp_valid.json"
    prep_temp.export_fit_json(temp_path)

    prep_temp_loaded = NHISPreprocessor(feature_registry=reg).load_fit_json(temp_path)
    assert prep_temp_loaded.fitted_record.fit_year == 2022
    assert prep_temp_loaded.fitted_record.fit_study_role == "development_train"
    assert prep_temp_loaded.is_legacy_unverified is False

    # Primary and expanded transforms match exactly
    t1_pri = prep_temp.transform(raw_df, feature_set="primary_core", preserve_metadata=False)
    t2_pri = prep_temp_loaded.transform(raw_df, feature_set="primary_core", preserve_metadata=False)
    pd.testing.assert_frame_equal(t1_pri, t2_pri)

    t1_exp = prep_temp.transform(raw_df, feature_set="expanded", preserve_metadata=False)
    t2_exp = prep_temp_loaded.transform(raw_df, feature_set="expanded", preserve_metadata=False)
    pd.testing.assert_frame_equal(t1_exp, t2_exp)

    # 2. Pooled roundtrip: fit -> export -> load equivalence (strictly <= 8 rows partition)
    pooled_df = raw_df.iloc[:8].copy()
    pooled_df["split_role"] = "train"
    prep_pooled = NHISPreprocessor(feature_registry=reg).fit(pooled_df, regime="pooled")
    pooled_path = tmp_path / "pooled_valid.json"
    prep_pooled.export_fit_json(pooled_path)

    prep_pooled_loaded = NHISPreprocessor(feature_registry=reg).load_fit_json(pooled_path)
    assert prep_pooled_loaded.fitted_record.fit_year == "2022-2024_pooled"
    assert prep_pooled_loaded.fitted_record.fit_study_role == "pooled_train"
    assert prep_pooled_loaded.fitted_record.rules_manifest["regime"] == "pooled"
    assert prep_pooled_loaded.is_legacy_unverified is False

    t_pool_1 = prep_pooled.transform(pooled_df, feature_set="primary_core", preserve_metadata=False)
    t_pool_2 = prep_pooled_loaded.transform(pooled_df, feature_set="primary_core", preserve_metadata=False)
    pd.testing.assert_frame_equal(t_pool_1, t_pool_2)

    # 3. Temporal adapter strictly rejects pooled preprocessor
    orig_is_file = pathlib.Path.is_file
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: True if "synthetic_dummy" in str(self) else orig_is_file(self))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: raw_df)
    monkeypatch.setattr(NHISStudyAdapter, "_validate_partitions", lambda self: None)

    with pytest.raises(ValueError, match="rejected non-temporal preprocessor|invalid provenance"):
        NHISStudyAdapter(features_parquet_path="synthetic_dummy.parquet", preprocessor=prep_pooled_loaded)

    # 4. Strict validation of artifact content and types against counterexamples
    valid_dict = json.loads(temp_path.read_text(encoding="utf-8"))

    # 4a. Missing row_count rejected
    bad_dict = copy.deepcopy(valid_dict)
    bad_dict.pop("row_count")
    p = tmp_path / "missing_row_count.json"
    p.write_text(json.dumps(bad_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field: 'row_count'"):
        NHISPreprocessor(feature_registry=reg).load_fit_json(p)

    # 4b. Fractional / bool row_count rejected
    for bad_rc in (1.9, True, False, -10):
        bad_dict = copy.deepcopy(valid_dict)
        bad_dict["row_count"] = bad_rc
        p = tmp_path / f"bad_row_count_{bad_rc}.json"
        p.write_text(json.dumps(bad_dict), encoding="utf-8")
        with pytest.raises(ValueError, match="row_count must be a positive integer"):
            NHISPreprocessor(feature_registry=reg).load_fit_json(p)

    # 4c. Non-finite median rejected
    bad_dict = copy.deepcopy(valid_dict)
    bad_dict["numerical_medians"]["agep_a"] = float("nan")
    p = tmp_path / "nan_median.json"
    p.write_text(json.dumps(bad_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite median"):
        NHISPreprocessor(feature_registry=reg).load_fit_json(p)

    # 4d. Invalid numerical stats body: mean=NaN, count_valid=-8
    bad_dict = copy.deepcopy(valid_dict)
    bad_dict["numerical_stats"]["agep_a"] = {"median": 40, "mean": float("nan"), "min": 25, "max": 60, "count_valid": -8, "count_missing": 0}
    p = tmp_path / "bad_stats.json"
    p.write_text(json.dumps(bad_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite mean|Invalid count_valid"):
        NHISPreprocessor(feature_registry=reg).load_fit_json(p)

    # 4d-1. Supervisor counterexample 1: Missing count_missing rejected (R7-05)
    bad_dict = copy.deepcopy(valid_dict)
    bad_dict["numerical_stats"]["agep_a"].pop("count_missing", None)
    p = tmp_path / "missing_count_missing.json"
    p.write_text(json.dumps(bad_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field.*count_missing"):
        NHISPreprocessor(feature_registry=reg).load_fit_json(p)

    # 4d-2. Supervisor counterexample 2: count_valid + count_missing != row_count rejected (R7-05)
    bad_dict = copy.deepcopy(valid_dict)
    bad_dict["numerical_stats"]["agep_a"]["count_missing"] = 1  # 8 + 1 = 9 != row_count (8)
    p = tmp_path / "count_sum_mismatch.json"
    p.write_text(json.dumps(bad_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="Count mismatch in numerical_stats"):
        NHISPreprocessor(feature_registry=reg).load_fit_json(p)

    # 4d-3. Supervisor counterexample 3: min > mean or mean > max rejected (R7-05)
    bad_dict = copy.deepcopy(valid_dict)
    bad_dict["numerical_stats"]["agep_a"]["mean"] = 10.0  # < min (25.0)
    p = tmp_path / "mean_below_min.json"
    p.write_text(json.dumps(bad_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="numerical_stats bounds invalid.*min .* <= mean .* <= max"):
        NHISPreprocessor(feature_registry=reg).load_fit_json(p)

    bad_dict = copy.deepcopy(valid_dict)
    bad_dict["numerical_stats"]["agep_a"]["mean"] = 80.0  # > max (60.0)
    p = tmp_path / "mean_above_max.json"
    p.write_text(json.dumps(bad_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="numerical_stats bounds invalid.*min .* <= mean .* <= max"):
        NHISPreprocessor(feature_registry=reg).load_fit_json(p)

    # 4d-4. Supervisor counterexample 4: min or max outside substantive range rejected (R7-05)
    bad_dict = copy.deepcopy(valid_dict)
    bad_dict["numerical_stats"]["agep_a"]["min"] = 10.0  # AGEP_A valid_range is [18, 85]
    bad_dict["numerical_stats"]["agep_a"]["mean"] = 35.0
    p = tmp_path / "min_out_of_bounds.json"
    p.write_text(json.dumps(bad_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="numerical_stats min/max for agep_a.*outside substantive range"):
        NHISPreprocessor(feature_registry=reg).load_fit_json(p)

    bad_dict = copy.deepcopy(valid_dict)
    bad_dict["numerical_stats"]["agep_a"]["max"] = 999.0
    bad_dict["numerical_stats"]["agep_a"]["mean"] = 35.0
    p = tmp_path / "max_out_of_bounds.json"
    p.write_text(json.dumps(bad_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="numerical_stats min/max for agep_a.*outside substantive range"):
        NHISPreprocessor(feature_registry=reg).load_fit_json(p)

    # 4d-5. Supervisor counterexample 5: allow_unspecified_source fit cannot be exported (R7-05)
    drop_cols = [c for c in ("survey_year", "study_role") if c in train_df.columns]
    raw_unspecified = train_df.drop(columns=drop_cols).copy()
    prep_unspec = NHISPreprocessor(feature_registry=reg).fit(raw_unspecified, allow_unspecified_source=True)
    assert prep_unspec.is_legacy_unverified is True
    assert prep_unspec.fitted_record.rules_manifest["regime"] == "unspecified"
    with pytest.raises(ValueError, match="Cannot export unverified legacy"):
        prep_unspec.export_fit_json(tmp_path / "unspecified_export.json")

    # 4e. Invalid categorical categories outside substantive schema
    bad_dict = copy.deepcopy(valid_dict)
    bad_dict["categorical_categories"][prep_temp.categorical_features[0]] = [999]
    p = tmp_path / "bad_cats.json"
    p.write_text(json.dumps(bad_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="Categorical categories.*contains invalid codes"):
        NHISPreprocessor(feature_registry=reg).load_fit_json(p)

    # 4f. Inconsistent rules manifest regime vs fit_year/role
    bad_dict = copy.deepcopy(valid_dict)
    bad_dict["rules_manifest"] = {"regime": "pooled", "schema_version": "wrong"}
    p = tmp_path / "bad_manifest.json"
    p.write_text(json.dumps(bad_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid fit_year for pooled regime|Inconsistent"):
        NHISPreprocessor(feature_registry=reg).load_fit_json(p)

    # 4g. Legacy format version with v2 rule_identity rejected as contradictory
    bad_dict = copy.deepcopy(valid_dict)
    bad_dict["format_version"] = "nhis_fairbias_fit_v1_legacy"
    p = tmp_path / "v1_with_v2_identity.json"
    p.write_text(json.dumps(bad_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="Contradictory artifact"):
        NHISPreprocessor(feature_registry=reg).load_fit_json(p)

    # 4h. Load atomicity: failed load preserves old state
    legacy_dict = copy.deepcopy(valid_dict)
    legacy_dict.pop("rule_identity")
    legacy_dict.pop("format_version")
    legacy_p = tmp_path / "legacy_donor.json"
    legacy_p.write_text(json.dumps(legacy_dict), encoding="utf-8")

    prep_obj = NHISPreprocessor(feature_registry=reg).load_fit_json(legacy_p)
    assert prep_obj.is_legacy_unverified is True
    prior_rec = prep_obj.fitted_record.to_dict()

    with pytest.raises(ValueError, match="missing required field: 'row_count'"):
        prep_obj.load_fit_json(tmp_path / "missing_row_count.json")
    assert prep_obj.is_legacy_unverified is True
    assert prep_obj.fitted_record.to_dict() == prior_rec
    with pytest.raises(ValueError, match="Cannot export unverified legacy"):
        prep_obj.export_fit_json(tmp_path / "illegal_export.json")


def test_r6_03_fit_numeric_cleaning_and_atomicity():
    """R6-03: Unified numerical cleaning in fit and transform; out-of-bounds non-response codes cleaned before median estimation."""
    reg = load_feature_registry("configs/nhis/features.json")
    raw_df = _make_synthetic_raw_df()
    train_df = raw_df[raw_df["survey_year"] == 2022].copy()

    # 8 rows age: [20, 30, 40, 50, 80, 997, 997, 997]
    # Substantive range is [18, 85]. 997 is out-of-bounds non-response.
    # Cleaning must mask 997 as NaN before estimating median.
    # The 5 substantive valid values are [20, 30, 40, 50, 80], whose median is 40.0 (not 65.0).
    train_df["agep_a"] = [20.0, 30.0, 40.0, 50.0, 80.0, 997.0, 997.0, 997.0]

    prep = NHISPreprocessor(feature_registry=reg).fit(train_df)
    assert prep.fitted_record.numerical_medians["agep_a"] == 40.0
    assert prep.fitted_record.numerical_stats["agep_a"]["median"] == 40.0
    assert prep.fitted_record.numerical_stats["agep_a"]["max"] == 80.0
    assert prep.fitted_record.numerical_stats["agep_a"]["min"] == 20.0
    assert prep.fitted_record.numerical_stats["agep_a"]["count_valid"] == 5
    assert prep.fitted_record.numerical_stats["agep_a"]["count_missing"] == 3

    # Transform on same data produces median 40.0 imputation for the 997s
    t_df = prep.transform(train_df, preserve_metadata=False)
    imputed_age = t_df["agep_a"].tolist()
    assert imputed_age == [20.0, 30.0, 40.0, 50.0, 80.0, 40.0, 40.0, 40.0]

    # Rejection of all-invalid inputs preserves unfitted state
    all_invalid_df = train_df.copy()
    all_invalid_df["agep_a"] = [997.0] * 8
    prep_unfitted = NHISPreprocessor(feature_registry=reg)
    with pytest.raises(Exception, match="has no valid training records"):
        prep_unfitted.fit(all_invalid_df)
    assert prep_unfitted.is_fitted is False




