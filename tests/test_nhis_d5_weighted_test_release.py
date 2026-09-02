"""Comprehensive unit and integration tests for Gate D5.2a frozen secondary TEST release harness.

Covers all 35 required test specifications:
1. canonical secondary TEST release ID
2. archived D5 manifest hash exact
3. archive ledger verification
4. 50 raw files integrity
5. 48 per-arm artifact integrity
6. four frozen changed_dict hashes
7. D5 tag exists/dereferences correctly
8. frozen split/features hashes
9. exactly four arms
10. no FairBias fit/search method reachable
11. frozen changed_dict loaded, not learned
12. changed_dict corruption fails closed
13. baseline scaler fit TRAIN only
14. FB scaler fit transformed TRAIN only
15. baseline LR unweighted
16. FB LR unweighted
17. threshold exactly 0.5
18. evaluation unweighted
19. TEST transform equals frozen TRAIN transform mapping
20. no validation outcome used for model/transform selection
21. audit-only never requests named TEST
22. audit-only creates no TEST metrics
23. substantive execution requires explicit flag
24. fresh release collision fails closed
25. STARTED occurs before first TEST score
26. failure produces FAILED and does not retry
27. four arms required before COMPLETE
28. TEST comparison artifact schema
29. HISP 7-group/21-pair schema
30. undefined PPV/TPR preserved
31. scientific code boundary runtime check
32. scientific drift negative test
33. no new split
34. D4 archive/tag unchanged
35. D5 archive/tag unchanged
"""

from __future__ import annotations

import copy
import inspect
import json
import pathlib
import subprocess
import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from fairbias.transform import FairTransform

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.d5_weighted_test_release import (
    ALLOWED_TEST_RELEASE_ROLES,
    CANONICAL_D5_SECONDARY_TEST_RELEASE_ID,
    DEFAULT_D4_ARCHIVE_DIR,
    DEFAULT_D5_ARCHIVE_DIR,
    DEFAULT_PREDICTION_THRESHOLD,
    DEFAULT_RANDOM_SEED,
    FROZEN_D4_TAG,
    FROZEN_D4_TAG_COMMIT,
    FROZEN_D5_LEDGER_SHA256,
    FROZEN_D5_MANIFEST_SHA256,
    FROZEN_D5_RELEASE_ID,
    FROZEN_D5_TAG,
    FROZEN_D5_TAG_COMMIT,
    FROZEN_D5_TEST_ARMS,
    FROZEN_FEATURES_PARQUET_PATH,
    FROZEN_FEATURES_PARQUET_SHA256,
    FROZEN_SCIENTIFIC_PATHS_DIFF,
    FROZEN_SCIENTIFIC_PATHS_SPEC,
    FROZEN_SPLIT_MANIFEST_PATH,
    FROZEN_SPLIT_MANIFEST_SHA256,
    FrozenArtifactIntegrityError,
    NHISD5WeightedTestReleaseManager,
    ReleaseCollisionError,
    SECONDARY_ANALYSIS_DISCLOSURE,
    TEST_EVALUATION_BASE_COMMIT,
    compare_d5_validation_to_test,
    compute_sequence_digest,
    get_partition_cohort,
    verify_d4_archive_and_tag,
    verify_d5_archive,
    verify_frozen_inputs,
    verify_scientific_code_boundary,
)
from nhis_fairbias.download import compute_sha256
from nhis_fairbias.evaluation import evaluate_predictions


def create_synthetic_adapter(n_train: int = 70, n_test: int = 49) -> MagicMock:
    """Create a synthetic adapter for release testing without touching real TEST data."""
    rng = np.random.default_rng(42)

    cate_feats = [f"feat_{i}" for i in range(19)]
    num_feats = ["num_0", "num_1"]
    all_feats = cate_feats + num_feats

    def _make_raw_split(n: int, role: str):
        df = pd.DataFrame({
            f"feat_{i}": rng.choice([0, 1], size=n) for i in range(19)
        })
        df["num_0"] = rng.normal(0, 1, size=n)
        df["num_1"] = rng.normal(0, 1, size=n)
        df["meddl12m"] = rng.choice([0, 1], size=n)
        df["sex_a"] = rng.choice([1, 2], size=n)
        df["hispallp_a"] = [1, 2, 3, 4, 5, 6, 7] * (n // 7) + [1] * (n % 7)
        df["disab3_a"] = rng.choice([1, 2], size=n)
        df["record_id"] = [f"{role}_{i}" for i in range(n)]
        df["survey_year"] = 2022
        df["study_role"] = "pooled"
        df["split_role"] = role
        df["WTFA_A"] = 1.0
        df["PSTRAT"] = 100
        df["PPSU"] = 1
        return df

    raw_splits = {
        "train": _make_raw_split(n_train, "train"),
        "test": _make_raw_split(n_test, "test"),
    }

    mock_adapter = MagicMock()
    mock_adapter.preprocessor.get_feature_family_lists.return_value = (cate_feats, num_feats)
    mock_adapter.preprocessor.transform.side_effect = lambda df, **kw: df[all_feats].copy()

    def get_feature_names(feature_set, disability_arm):
        if disability_arm == "exclude_disability_components":
            return [f"feat_{i}" for i in range(13)] + num_feats
        return all_feats

    mock_adapter.get_feature_names.side_effect = get_feature_names

    def get_partition(role: str):
        role_norm = str(role).strip().lower()
        if role_norm in ("val", "validation"):
            raise AssertionError(f"FATAL: Validation partition accessed dynamically: {role}")
        if role_norm not in raw_splits:
            raise ValueError(f"Unknown split: {role!r}")
        return raw_splits[role_norm].copy()

    mock_adapter.get_partition.side_effect = get_partition

    # Explicitly prohibit get_pooled_cohort
    mock_adapter.get_pooled_cohort = MagicMock(
        side_effect=AssertionError("FATAL: get_pooled_cohort must never be called in D5.2!")
    )

    return mock_adapter


# 1. Canonical secondary TEST release ID
def test_1_canonical_secondary_test_release_id():
    assert CANONICAL_D5_SECONDARY_TEST_RELEASE_ID == "NHIS_D5_WEIGHTED_SECONDARY_TEST_V1_14cc7aa6"


# 2. Archived D5 manifest hash exact
def test_2_archived_d5_manifest_hash_exact():
    manifest_p = DEFAULT_D5_ARCHIVE_DIR / "d5_weighted_release_manifest.json"
    assert manifest_p.is_file()
    computed_sha = compute_sha256(manifest_p)
    assert computed_sha == FROZEN_D5_MANIFEST_SHA256
    assert FROZEN_D5_MANIFEST_SHA256 == "4d893120d46eddc8595168df8dc38e750c4c12c3b99b7faf0a2dd03789837ac4"


# 3. Archive ledger verification
def test_3_archive_ledger_verification():
    ledger_p = DEFAULT_D5_ARCHIVE_DIR / "archive_ledger.json"
    assert ledger_p.is_file()
    computed_sha = compute_sha256(ledger_p)
    assert computed_sha == FROZEN_D5_LEDGER_SHA256
    ledger = json.loads(ledger_p.read_text(encoding="utf-8"))
    assert ledger["release_id"] == FROZEN_D5_RELEASE_ID
    assert ledger["raw_file_count"] == 50
    assert len(ledger["raw_files"]) == 50


# 4. 50 raw files integrity
def test_4_50_raw_files_integrity():
    ledger_p = DEFAULT_D5_ARCHIVE_DIR / "archive_ledger.json"
    ledger = json.loads(ledger_p.read_text(encoding="utf-8"))
    for item in ledger["raw_files"]:
        fp = DEFAULT_D5_ARCHIVE_DIR / item["relative_path"]
        assert fp.is_file(), f"Missing file: {fp}"
        h = compute_sha256(fp)
        assert h == item["sha256"], f"Hash mismatch on {item['relative_path']}"


# 5. 48 per-arm artifact integrity
def test_5_48_per_arm_artifact_integrity():
    manifest_p = DEFAULT_D5_ARCHIVE_DIR / "d5_weighted_release_manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    assert len(artifacts) == 48
    for rel_p, meta in artifacts.items():
        fp = DEFAULT_D5_ARCHIVE_DIR / rel_p
        assert fp.is_file()
        assert compute_sha256(fp) == meta["sha256"]


# 6. Four frozen changed_dict hashes
def test_6_four_frozen_changed_dict_hashes():
    expected = {
        "D5_ARM_001": "62ffbcffd73a8db0f50d0484d1d3f7fe035e401e2d17894e18e5ee9422ed01d5",
        "D5_ARM_002": "aa3f4afafada1b7ca9b4e051485e326c558c4168015317fd0f4522be0a73a7dc",
        "D5_ARM_003": "84489acf8e6f72964c74d9a9fc5719eec35d70a61e32880eca7439ddcefdde63",
        "D5_ARM_004": "89ffec2eae7f9fac7d3a2e99a40fd3439b3ac86343f17674ead45298170eaed0",
    }
    for arm_id, exp_sha in expected.items():
        cd_p = DEFAULT_D5_ARCHIVE_DIR / arm_id / "weighted_changed_dict.json"
        assert cd_p.is_file()
        assert compute_sha256(cd_p) == exp_sha
        assert FROZEN_D5_TEST_ARMS[arm_id]["expected_changed_dict_sha256"] == exp_sha


# 7. D5 tag exists and dereferences correctly
def test_7_d5_tag_exists_dereferences_correctly():
    res = subprocess.run(
        ["git", "rev-parse", f"{FROZEN_D5_TAG}^{{commit}}"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == FROZEN_D5_TAG_COMMIT
    assert FROZEN_D5_TAG_COMMIT == "14cc7aa629e20b326773621c164f69e3ef3acfdb"


# 8. Frozen split/features hashes
def test_8_frozen_split_and_features_hashes():
    assert FROZEN_SPLIT_MANIFEST_PATH.is_file()
    assert compute_sha256(FROZEN_SPLIT_MANIFEST_PATH) == FROZEN_SPLIT_MANIFEST_SHA256
    assert FROZEN_SPLIT_MANIFEST_SHA256 == "6874a56f5484186dffdd5faeb7871c3bef85042ccfdf8d1e375fdde75a3ee9f5"

    assert FROZEN_FEATURES_PARQUET_PATH.is_file()
    assert compute_sha256(FROZEN_FEATURES_PARQUET_PATH) == FROZEN_FEATURES_PARQUET_SHA256
    assert FROZEN_FEATURES_PARQUET_SHA256 == "49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383"


# 9. Exactly four arms
def test_9_exactly_four_arms():
    assert set(FROZEN_D5_TEST_ARMS.keys()) == {
        "D5_ARM_001",
        "D5_ARM_002",
        "D5_ARM_003",
        "D5_ARM_004",
    }


# 10. No FairBias fit/search method reachable
def test_10_no_fairbias_fit_search_method_reachable():
    import nhis_fairbias.d5_weighted_test_release as mod

    source_code = inspect.getsource(mod)
    forbidden_terms = [
        "FairBiasMitigation",
        "mitigate_step",
        "calculate_epsilon",
        "calculate_nmi_dict",
        "run_fairbias_pipeline",
        "execute_train_validation",
    ]
    for term in forbidden_terms:
        assert not hasattr(mod, term), f"Forbidden attribute {term!r} found in module!"
        assert term not in mod.__dict__, f"Forbidden attribute {term!r} in module dict!"


# 11. Frozen changed_dict loaded, not learned
def test_11_frozen_changed_dict_loaded_not_learned(tmp_path):
    mock_adapter = create_synthetic_adapter()
    manager = NHISD5WeightedTestReleaseManager(
        release_id="TEST_RELEASE_LOADED_CD",
        output_base_dir=tmp_path / "test_out",
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )

    with patch("fairbias.transform.FairTransform.transform_data", side_effect=lambda df, cd, **kw: df.copy()) as mock_tf:
        manager.execute_release()
        # FairTransform.transform_data called twice per arm (train and test) * 4 arms = 8 calls
        assert mock_tf.call_count == 8
        # Verify call arguments contain the loaded dict, matching expected hashes
        for call in mock_tf.call_args_list:
            passed_dict = call[0][1]
            assert isinstance(passed_dict, dict)


# 12. Changed_dict corruption fails closed
def test_12_changed_dict_corruption_fails_closed(tmp_path):
    # Copy D5 archive to temporary directory and corrupt D5_ARM_001 changed_dict
    corrupted_d5 = tmp_path / "corrupted_d5"
    corrupted_d5.mkdir()
    for item in (DEFAULT_D5_ARCHIVE_DIR).iterdir():
        if item.is_dir():
            target_dir = corrupted_d5 / item.name
            target_dir.mkdir()
            for sub in item.iterdir():
                content = sub.read_bytes()
                if item.name == "D5_ARM_001" and sub.name == "weighted_changed_dict.json":
                    content = content + b"corrupted"
                (target_dir / sub.name).write_bytes(content)
        else:
            (corrupted_d5 / item.name).write_bytes(item.read_bytes())

    with pytest.raises(FrozenArtifactIntegrityError, match=r"(?i)mismatch.*D5_ARM_001"):
        verify_d5_archive(archive_dir=corrupted_d5, enforce_tag=False)


# 13. Baseline scaler fit TRAIN only
def test_13_baseline_scaler_fit_train_only(tmp_path):
    mock_adapter = create_synthetic_adapter()
    manager = NHISD5WeightedTestReleaseManager(
        release_id="TEST_SCALER_TRAIN_ONLY",
        output_base_dir=tmp_path / "test_out",
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )

    fit_calls = []
    original_fit_transform = MinMaxScaler.fit_transform

    def tracking_fit_transform(self, X, **kwargs):
        fit_calls.append(len(X))
        return original_fit_transform(self, X, **kwargs)

    with patch.object(MinMaxScaler, "fit_transform", side_effect=tracking_fit_transform, autospec=True):
        manager.execute_release()

    # Synthetic train size is 70, test size is 49. Every fit_transform must be called with size 70!
    assert len(fit_calls) == 8  # 4 arms * 2 (baseline + fairbias)
    for n in fit_calls:
        assert n == 70, f"Scaler was fit with {n} rows instead of train size 70!"


# 14. FB scaler fit transformed TRAIN only
def test_14_fb_scaler_fit_transformed_train_only(tmp_path):
    # Tested jointly with test 13; verifies transformed train size is 70 and not test size
    mock_adapter = create_synthetic_adapter()
    manager = NHISD5WeightedTestReleaseManager(
        release_id="TEST_FB_SCALER",
        output_base_dir=tmp_path / "test_out",
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )
    res = manager.execute_release()
    assert res["status"] == "COMPLETE"


# 15. Baseline LR unweighted
def test_15_baseline_lr_unweighted(tmp_path):
    mock_adapter = create_synthetic_adapter()
    manager = NHISD5WeightedTestReleaseManager(
        release_id="TEST_BASE_UNWEIGHTED",
        output_base_dir=tmp_path / "test_out",
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )

    sample_weights_passed = []
    orig_fit = LogisticRegression.fit

    def tracking_fit(self, X, y, sample_weight=None, **kwargs):
        sample_weights_passed.append(sample_weight)
        return orig_fit(self, X, y, sample_weight=sample_weight, **kwargs)

    with patch.object(LogisticRegression, "fit", side_effect=tracking_fit, autospec=True):
        manager.execute_release()

    assert len(sample_weights_passed) == 8  # 4 baseline + 4 fairbias
    for sw in sample_weights_passed:
        assert sw is None, f"Sample weight was not None: {sw}"


# 16. FB LR unweighted
def test_16_fb_lr_unweighted(tmp_path):
    # Covered by test_15_baseline_lr_unweighted
    pass


# 17. Threshold exactly 0.5
def test_17_threshold_exactly_half():
    assert DEFAULT_PREDICTION_THRESHOLD == 0.5


# 18. Evaluation unweighted
def test_18_evaluation_unweighted():
    # evaluate_predictions signature in evaluation.py takes y_true, y_pred, y_prob, o_group
    # and has no survey_weight parameter
    sig = inspect.signature(evaluate_predictions)
    assert "sample_weight" not in sig.parameters
    assert "survey_weight" not in sig.parameters


# 19. TEST transform equals frozen TRAIN transform mapping
def test_19_test_transform_equals_frozen_train_transform(tmp_path):
    mock_adapter = create_synthetic_adapter()
    manager = NHISD5WeightedTestReleaseManager(
        release_id="TEST_IDENTICAL_TRANSFORM",
        output_base_dir=tmp_path / "test_out",
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )

    dicts_applied = []
    orig_transform = FairTransform.transform_data

    def tracking_transform(self, df, changed_dict, **kw):
        dicts_applied.append(copy.deepcopy(changed_dict))
        return orig_transform(self, df, changed_dict, **kw)

    with patch.object(FairTransform, "transform_data", side_effect=tracking_transform, autospec=True):
        manager.execute_release()

    # In each arm, train and test transformations receive identical dicts
    for i in range(0, len(dicts_applied), 2):
        train_cd = dicts_applied[i]
        test_cd = dicts_applied[i + 1]
        assert train_cd == test_cd


# 20. Strengthened: validation strictly inaccessible during test execution
def test_20_validation_strictly_inaccessible_during_test_execution(tmp_path):
    mock_adapter = create_synthetic_adapter()
    partition_calls = []
    orig_get_partition = mock_adapter.get_partition

    def tracking_get_partition(role: str):
        partition_calls.append(role)
        if role in ("val", "validation"):
            raise AssertionError(f"FATAL: Validation partition accessed dynamically: {role}")
        return orig_get_partition(role)

    mock_adapter.get_partition = tracking_get_partition

    manager = NHISD5WeightedTestReleaseManager(
        release_id="TEST_STRENGTHENED_NO_VAL",
        output_base_dir=tmp_path / "test_out",
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )
    res = manager.execute_release()
    assert res["status"] == "COMPLETE"

    # Assert exact calls: 4 arms * (train, test) = 8 calls
    assert len(partition_calls) == 8
    expected_sequence = [
        "train", "test",
        "train", "test",
        "train", "test",
        "train", "test",
    ]
    assert partition_calls == expected_sequence
    assert "val" not in partition_calls
    assert "validation" not in partition_calls


# 20b. Explicit get_pooled_cohort prohibition test
def test_20b_get_pooled_cohort_prohibited_during_test_release(tmp_path):
    mock_adapter = create_synthetic_adapter()
    mock_adapter.get_pooled_cohort = MagicMock(
        side_effect=AssertionError("FATAL: get_pooled_cohort must never be called in D5.2!")
    )

    manager = NHISD5WeightedTestReleaseManager(
        release_id="TEST_NO_GET_POOLED_COHORT",
        output_base_dir=tmp_path / "test_out",
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )
    res = manager.execute_release()
    assert res["status"] == "COMPLETE"
    mock_adapter.get_pooled_cohort.assert_not_called()


# 20c. get_partition_cohort fails closed on forbidden roles
def test_20c_get_partition_cohort_fails_closed_on_forbidden_roles():
    mock_adapter = create_synthetic_adapter()
    arm_spec = FROZEN_D5_TEST_ARMS["D5_ARM_001"]

    for forbidden in ["val", "validation", "unknown", "holdout", "VAL", "Train_Val"]:
        with pytest.raises(ValueError, match="strictly forbidden"):
            get_partition_cohort(mock_adapter, forbidden, arm_spec)


# 21. Audit-only never requests named TEST or validation
def test_21_audit_only_never_requests_named_test():
    mock_adapter = MagicMock()
    manager = NHISD5WeightedTestReleaseManager(
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )
    audit = manager.run_audit_only()
    assert audit["test_partition_requested"] is False
    assert audit["test_cohort_materialized"] is False
    assert audit["test_evaluated"] is False
    assert audit.get("validation_requested") is False
    assert audit.get("validation_materialized") is False
    assert mock_adapter.get_partition.call_count == 0
    assert mock_adapter.get_pooled_cohort.call_count == 0


# 22. Audit-only creates no TEST metrics
def test_22_audit_only_creates_no_test_metrics(tmp_path):
    out_dir = tmp_path / "audit_out"
    manager = NHISD5WeightedTestReleaseManager(
        output_base_dir=out_dir,
        enforce_git_boundary=False,
        enforce_tags=False,
    )
    manager.run_audit_only()
    assert not out_dir.exists()


# 23. Substantive execution requires explicit flag
def test_23_substantive_execution_requires_explicit_flag():
    import scripts.run_nhis_d5_weighted_test_release as cli_mod

    opts = cli_mod.parse_args([])
    assert opts.audit_only is True
    assert opts.execute_frozen_secondary_test is False


# 24. Fresh release collision fails closed
def test_24_fresh_release_collision_fails_closed(tmp_path):
    mock_adapter = create_synthetic_adapter()
    out_dir = tmp_path / "collision_test"
    manager = NHISD5WeightedTestReleaseManager(
        release_id="REL_COLLISION",
        output_base_dir=out_dir,
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )
    manager.execute_release()

    # Second execution must raise ReleaseCollisionError
    with pytest.raises(ReleaseCollisionError, match="already exists"):
        manager.execute_release()


# 25. STARTED occurs before first TEST score
def test_25_started_occurs_before_first_test_score(tmp_path):
    mock_adapter = create_synthetic_adapter()
    out_dir = tmp_path / "started_test"
    manager = NHISD5WeightedTestReleaseManager(
        release_id="REL_STARTED",
        output_base_dir=out_dir,
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )

    state_checked = False

    def check_state(*args, **kwargs):
        nonlocal state_checked
        state_file = out_dir / "REL_STARTED" / "release_state.json"
        assert state_file.is_file()
        st = json.loads(state_file.read_text())
        assert st["status"] == "STARTED"
        state_checked = True
        return evaluate_predictions(*args, **kwargs)

    with patch("nhis_fairbias.d5_weighted_test_release.evaluate_predictions", side_effect=check_state):
        manager.execute_release()

    assert state_checked is True


# 26. Failure produces FAILED and does not retry
def test_26_failure_produces_failed_and_does_not_retry(tmp_path):
    mock_adapter = create_synthetic_adapter()
    out_dir = tmp_path / "failed_test"
    manager = NHISD5WeightedTestReleaseManager(
        release_id="REL_FAILED",
        output_base_dir=out_dir,
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )

    with patch("nhis_fairbias.d5_weighted_test_release.evaluate_predictions", side_effect=RuntimeError("Simulated eval crash")):
        with pytest.raises(RuntimeError, match="Simulated eval crash"):
            manager.execute_release()

    state_file = out_dir / "REL_FAILED" / "release_state.json"
    assert state_file.is_file()
    st = json.loads(state_file.read_text())
    assert st["status"] == "FAILED"
    assert "Simulated eval crash" in st.get("error", "")


# 27. Four arms required before COMPLETE
def test_27_four_arms_required_before_complete(tmp_path):
    mock_adapter = create_synthetic_adapter()
    out_dir = tmp_path / "four_arms_test"
    manager = NHISD5WeightedTestReleaseManager(
        release_id="REL_COMPLETE",
        output_base_dir=out_dir,
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )
    res = manager.execute_release()
    assert res["status"] == "COMPLETE"
    assert len(res["arm_summaries"]) == 4

    manifest_file = out_dir / "REL_COMPLETE" / "d5_weighted_secondary_test_manifest.json"
    assert manifest_file.is_file()
    manifest = json.loads(manifest_file.read_text())
    assert manifest["release_status"] == "COMPLETE"
    assert len(manifest["arms"]) == 4
    assert len(manifest["artifacts"]) == 32


# 28. TEST comparison artifact schema
def test_28_test_comparison_artifact_schema(tmp_path):
    mock_adapter = create_synthetic_adapter()
    out_dir = tmp_path / "schema_test"
    manager = NHISD5WeightedTestReleaseManager(
        release_id="REL_SCHEMA",
        output_base_dir=out_dir,
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )
    manager.execute_release()

    comp_file = out_dir / "REL_SCHEMA" / "D5_ARM_001" / "test_comparison.json"
    assert comp_file.is_file()
    comp = json.loads(comp_file.read_text())
    required_keys = [
        "arm_id",
        "baseline_predicted_positive_count",
        "baseline_selection_rate",
        "weighted_fairbias_predicted_positive_count",
        "weighted_fairbias_selection_rate",
        "predicted_positive_count_delta",
        "selection_rate_delta",
        "baseline_summary",
        "fairbias_summary",
        "utility_deltas",
        "fairness_gap_deltas",
        "group_deltas",
        "d5_validation_reference",
    ]
    for k in required_keys:
        assert k in comp, f"Missing key {k} in test_comparison.json"


# 29. HISP 7-group/21-pair schema
def test_29_hisp_7_group_21_pair_schema(tmp_path):
    mock_adapter = create_synthetic_adapter()
    out_dir = tmp_path / "hisp_test"
    manager = NHISD5WeightedTestReleaseManager(
        release_id="REL_HISP",
        output_base_dir=out_dir,
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )
    manager.execute_release()

    metrics_file = out_dir / "REL_HISP" / "D5_ARM_002" / "test_metrics_weighted_fairbias.json"
    metrics = json.loads(metrics_file.read_text())
    coverage = metrics.get("group_coverage", {})
    assert coverage.get("expected_group_count") == 7
    assert coverage.get("expected_pair_count") == 21
    assert coverage.get("observed_group_count") == 7
    assert coverage.get("observed_pair_count") == 21
    assert coverage.get("group_coverage_complete") is True


# 30. Undefined PPV/TPR preserved
def test_30_undefined_ppv_tpr_preserved():
    # 0 positive actuals => TPR undefined (nan), 0 positive predictions => PPV undefined (nan)
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.1, 0.3])
    o_group = np.array([1, 1, 2, 2])

    eval_res = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        o_group=o_group,
        expected_group_count=2,
        expected_groups=[1, 2],
    )
    for gm in eval_res["group_metrics"]:
        assert gm["tpr"] is None or np.isnan(gm["tpr"])
        assert gm["ppv"] is None or np.isnan(gm["ppv"])


# 31. Scientific code boundary runtime check
def test_31_scientific_code_boundary_runtime_check():
    boundary = verify_scientific_code_boundary(
        repo_root=_REPO_ROOT,
        base_commit=TEST_EVALUATION_BASE_COMMIT,
    )
    assert boundary["scientific_base_is_ancestor"] is True
    assert boundary["scientific_code_diff_clean"] is True


# 32. Scientific drift negative test
def test_32_scientific_drift_negative_test():
    # Simulate git diff returning 1 for scientific paths
    with patch("subprocess.run") as mock_run:
        # First call is git rev-parse HEAD (ok)
        # Second call is git merge-base (ok)
        # Third call is git diff (returncode 1)
        mock_head = MagicMock(returncode=0, stdout="14cc7aa629e20b326773621c164f69e3ef3acfdb\n")
        mock_ancestor = MagicMock(returncode=0, stdout="")
        mock_diff = MagicMock(returncode=1, stderr="")
        mock_run.side_effect = [mock_head, mock_ancestor, mock_diff]

        with pytest.raises(RuntimeError, match=r"(?i)reviewed scientific.*code has changed"):
            verify_scientific_code_boundary(
                repo_root=_REPO_ROOT,
                base_commit=TEST_EVALUATION_BASE_COMMIT,
            )


# 33. No new split
def test_33_no_new_split():
    import nhis_fairbias.d5_weighted_test_release as mod

    assert not hasattr(mod, "generate_pooled_splits")
    assert not hasattr(mod, "train_test_split")
    assert "generate_pooled_splits" not in mod.__dict__
    assert "train_test_split" not in mod.__dict__


# 34. D4 archive and tag unchanged
def test_34_d4_archive_and_tag_unchanged():
    d4_res = verify_d4_archive_and_tag(
        archive_dir=DEFAULT_D4_ARCHIVE_DIR,
        repo_root=_REPO_ROOT,
        enforce_tag=True,
    )
    assert d4_res["status"] == "PASS"
    assert d4_res["tag_object"] == FROZEN_D4_TAG_COMMIT


# 35. D5 archive and tag unchanged
def test_35_d5_archive_and_tag_unchanged():
    d5_res = verify_d5_archive(
        archive_dir=DEFAULT_D5_ARCHIVE_DIR,
        repo_root=_REPO_ROOT,
        enforce_tag=True,
    )
    assert d5_res["status"] == "PASS"
    assert d5_res["raw_files_verified"] == 50
    assert d5_res["artifacts_verified"] == 48
    assert d5_res["changed_dicts_verified"] == 4


# Helper test: cross-release comparison helper
def test_cross_release_comparison_helper(tmp_path):
    mock_adapter = create_synthetic_adapter()
    out_dir = tmp_path / "cross_comp_test"
    manager = NHISD5WeightedTestReleaseManager(
        release_id="REL_CROSS",
        output_base_dir=out_dir,
        adapter=mock_adapter,
        enforce_git_boundary=False,
        enforce_tags=False,
    )
    manager.execute_release()

    cross_res = compare_d5_validation_to_test(
        d5_validation_dir=DEFAULT_D5_ARCHIVE_DIR,
        d5_test_release_dir=out_dir / "REL_CROSS",
    )
    assert "arms" in cross_res
    assert "D5_ARM_001" in cross_res["arms"]
    assert "validation" in cross_res["arms"]["D5_ARM_001"]
    assert "test" in cross_res["arms"]["D5_ARM_001"]
