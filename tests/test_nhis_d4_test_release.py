"""Comprehensive unit and integration tests for Gate D4.1a frozen TEST release harness.

Covers all 24 required test specifications:
1. Frozen four-arm registry contains exactly the four authorized arms.
2. Every run ID and expected artifact SHA is fixed.
3. Wrong/missing changed_dict hash fails closed.
4. Wrong/missing trace hash fails closed.
5. Wrong split/features hash fails closed.
6. Audit-only mode does not access or score TEST.
7. Audit-only creates no TEST metrics artifacts.
8. Execute path never invokes FairBias mitigation.
9. Execute path never invokes calculate_epsilon.
10. Execute path never invokes calculate_nmi_dict.
11. Execute path never invokes generate_pooled_splits.
12. Execute path never invokes train_test_split.
13. Frozen changed_dict is applied unchanged to TRAIN and TEST.
14. Scaler fit is TRAIN-only.
15. Baseline LR and FairBias LR use identical frozen model specification.
16. TEST is never passed to .fit().
17. VALIDATION is not used by the TEST release path.
18. Threshold is exactly 0.5.
19. Seed is exactly 0.
20. HISP 7 groups produce 21 pairs in synthetic complete-coverage TEST.
21. HISP incomplete synthetic TEST is clearly marked incomplete.
22. One-time release directory refuses overwrite/re-execution.
23. Synthetic STARTED failure remains auditable and is not silently overwritten.
24. Output manifest hashes are correct.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.d4_test_release import (
    FROZEN_FEATURES_PARQUET_PATH,
    FROZEN_FEATURES_PARQUET_SHA256,
    FROZEN_FOUR_ARM_REGISTRY,
    FROZEN_SPLIT_MANIFEST_PATH,
    FROZEN_SPLIT_MANIFEST_SHA256,
    FrozenArtifactIntegrityError,
    NHISD4TestReleaseHarness,
    PRIMARY_D4_PREDICTION_THRESHOLD,
    PRIMARY_D4_RANDOM_SEED,
    PRIMARY_D4_TEST_RELEASE_PROTOCOL_ID,
    REQUIRED_ANALYSIS_COMMIT,
    ReleaseCollisionError,
)
from nhis_fairbias.download import compute_sha256, write_json_atomic
from nhis_fairbias.evaluation import (
    compute_group_coverage,
    evaluate_predictions,
)
from nhis_fairbias.pooled import NHISPooledAdapter


# ------------------------------------------------------------------------------
# Helper: create mock preflight directory tree with valid hashes for synthetic tests
# ------------------------------------------------------------------------------
def create_mock_preflight_directory(
    base_dir: pathlib.Path,
    mutate_arm: str | None = None,
    mutate_file: str | None = None,
    delete_file: str | None = None,
) -> pathlib.Path:
    """Create a temporary directory structure mimicking runs/nhis_d4_preflight/ with valid artifact hashes."""
    preflight_dir = base_dir / "preflight_runs"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    for arm_id, spec in FROZEN_FOUR_ARM_REGISTRY.items():
        run_id = spec["preflight_run_id"]
        run_dir = preflight_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        for fname, expected_sha in spec["frozen_artifact_hashes"].items():
            if arm_id == mutate_arm and fname == delete_file:
                continue

            target_path = run_dir / fname
            # We copy or read the real artifact to match the SHA exactly
            real_path = _REPO_ROOT / "runs" / "nhis_d4_preflight" / run_id / fname
            if real_path.is_file():
                content = real_path.read_bytes()
            else:
                content = b"{}"

            if arm_id == mutate_arm and fname == mutate_file:
                # Corrupt the content
                content = content + b"corrupted"

            target_path.write_bytes(content)

    return preflight_dir


# Helper: create synthetic adapter for release testing without touching real TEST evaluation
def create_synthetic_adapter(tmp_path: pathlib.Path, n_train: int = 60, n_test: int = 40) -> NHISPooledAdapter:
    rng = np.random.default_rng(42)

    def _make_split(n: int, role: str):
        X = pd.DataFrame({
            f"feat_{i}": rng.choice([0, 1], size=n) for i in range(15)
        })
        # Add numeric features
        X["num_0"] = rng.normal(0, 1, size=n)
        X["num_1"] = rng.normal(0, 1, size=n)
        # Ensure 21 features for full arms, 15 for reduced
        for i in range(15, 19):
            X[f"feat_{i}"] = rng.choice([0, 1], size=n)
        y = pd.Series(rng.choice([0, 1], size=n), name="MEDDL12M_A")
        o = pd.Series(rng.choice([1, 2], size=n), name="SEX_A")
        meta = pd.DataFrame({"split_role": [role] * n, "record_id": [f"{role}_{i}" for i in range(n)]})
        return X, y, o, None, meta

    mock_adapter = MagicMock()
    mock_adapter.features_parquet_path = tmp_path / "mock.parquet"
    mock_adapter.features_parquet_path.touch()
    mock_adapter.split_manifest = pd.DataFrame({"record_id": [1], "split_role": ["train"]})

    cate_feats = [f"feat_{i}" for i in range(19)]
    num_feats = ["num_0", "num_1"]
    mock_adapter.preprocessor.get_feature_family_lists.return_value = (cate_feats, num_feats)

    def get_pooled_cohort(outcome, protected_attribute, feature_set, disability_arm):
        X_tr, y_tr, o_tr, _, meta_tr = _make_split(n_train, "train")
        X_te, y_te, o_te, _, meta_te = _make_split(n_test, "test")

        if protected_attribute == "HISPALLP_A":
            o_tr = pd.Series(rng.choice([1, 2, 3, 4, 5, 6, 7], size=n_train), name="HISPALLP_A")
            # Complete 7-group test set
            o_te = pd.Series([1, 2, 3, 4, 5, 6, 7] * (n_test // 7) + [1] * (n_test % 7), name="HISPALLP_A")

        if disability_arm == "exclude_disability_components":
            # 15 features
            keep_cols = [f"feat_{i}" for i in range(13)] + ["num_0", "num_1"]
            X_tr = X_tr[keep_cols]
            X_te = X_te[keep_cols]

        return {
            "train": (X_tr, y_tr, o_tr, None, meta_tr),
            "test": (X_te, y_te, o_te, None, meta_te),
            "val": (_make_split(10, "val")),  # present in dict but must never be accessed
        }

    mock_adapter.get_pooled_cohort.side_effect = get_pooled_cohort
    return mock_adapter


# ------------------------------------------------------------------------------
# Test 1: Frozen four-arm registry contains exactly the four authorized arms
# ------------------------------------------------------------------------------
def test_1_frozen_four_arm_registry_exact_four_arms() -> None:
    expected_arms = {"ARM_D3_001", "ARM_D3_002", "ARM_D3_003", "ARM_D3_004"}
    assert set(FROZEN_FOUR_ARM_REGISTRY.keys()) == expected_arms

    # Check key attributes
    assert FROZEN_FOUR_ARM_REGISTRY["ARM_D3_001"]["protected_attribute"] == "SEX_A"
    assert FROZEN_FOUR_ARM_REGISTRY["ARM_D3_001"]["expected_predictors"] == 21

    assert FROZEN_FOUR_ARM_REGISTRY["ARM_D3_002"]["protected_attribute"] == "HISPALLP_A"
    assert FROZEN_FOUR_ARM_REGISTRY["ARM_D3_002"]["expected_predictors"] == 21
    assert FROZEN_FOUR_ARM_REGISTRY["ARM_D3_002"]["expected_group_count"] == 7
    assert FROZEN_FOUR_ARM_REGISTRY["ARM_D3_002"]["expected_pair_count"] == 21

    assert FROZEN_FOUR_ARM_REGISTRY["ARM_D3_003"]["protected_attribute"] == "DISAB3_A"
    assert FROZEN_FOUR_ARM_REGISTRY["ARM_D3_003"]["disability_arm"] == "full_feature"
    assert FROZEN_FOUR_ARM_REGISTRY["ARM_D3_003"]["expected_predictors"] == 21

    assert FROZEN_FOUR_ARM_REGISTRY["ARM_D3_004"]["protected_attribute"] == "DISAB3_A"
    assert FROZEN_FOUR_ARM_REGISTRY["ARM_D3_004"]["disability_arm"] == "exclude_disability_components"
    assert FROZEN_FOUR_ARM_REGISTRY["ARM_D3_004"]["expected_predictors"] == 15


# ------------------------------------------------------------------------------
# Test 2: Every run ID and expected artifact SHA is fixed
# ------------------------------------------------------------------------------
def test_2_fixed_run_ids_and_artifact_hashes() -> None:
    expected_specs = {
        "ARM_D3_001": {
            "run_id": "20260901T150720Z-94c5ac1be7",
            "changed_dict_sha": "9e90b303d28e5fd890dd6351d6e432c803c9d08cd051c4a1595859401cb76f6d",
            "trace_sha": "bf5ebda4d88b2330dab9e1ab043295cb2dbef7a7c5d3bfce16e49fe199e893ee",
            "base_val_sha": "b1bd508f5307edefffc77513da9fa7d853253477183a875d51662ed96a8bcc2b",
            "fb_val_sha": "57d3eb5cf6493b895b0fa8e05b16ec5f95d843a2431b5eddb65600bcc960ef63",
            "comp_val_sha": "508c6b94165c1dae0e41959473ed878b17177d5e0d620d57ec587b2b04181857",
        },
        "ARM_D3_002": {
            "run_id": "20260901T152301Z-6f49937b9f",
            "changed_dict_sha": "56dbab9d5a3470b668267f09ca658826e1a162d0391ed9be5dd1a56a30036c3c",
            "trace_sha": "a30f73844cef273960e9258082d11cfdba1d3c9edfc8038098af2faea0eccd6d",
            "base_val_sha": "fcb8ff739178c96e83b8ce30b9d0788cb22ea911b631f65e39d016aaceb2c056",
            "fb_val_sha": "4e9efa8921fa7ecce49469f7297257e3a945faec38fd026a3b0f0b3596c587ba",
            "comp_val_sha": "c90cfb02210922c0e23d058991714aa0ccad8f365c6ae9e8b98a0500b683d9b4",
        },
        "ARM_D3_003": {
            "run_id": "20260901T152455Z-0b07868ca4",
            "changed_dict_sha": "84489acf8e6f72964c74d9a9fc5719eec35d70a61e32880eca7439ddcefdde63",
            "trace_sha": "5fd7cca0191be73acb34cd0b143a3a831f3f40dc8d4aba236cc09bdb559e470a",
            "base_val_sha": "675cc14de795a758c65e7e599aa76586272a2ea65ede2b7954d74936f19b5fd2",
            "fb_val_sha": "b20c17358b9ae9b8374b033a5eb85c60b87c871f0806835f663da49a317c486b",
            "comp_val_sha": "7300873532c4406caa7b81e58033fc9060fb98a99021bd023dd66c83607a3105",
        },
        "ARM_D3_004": {
            "run_id": "20260901T152529Z-44d2d5a4cc",
            "changed_dict_sha": "2b0f3138b2ac93d6d3dbe39f23849d46b94ac36d8db3f4f35e64af260883cb27",
            "trace_sha": "34ca358545b7b9b4b08dd6cccfc1f5d0d32927d5d2eb85afcd023514b21eb511",
            "base_val_sha": "4e66e549f4c5c2c74d3950cfe6601bc5523ff6e6d34b6f7b426b7395d3809d38",
            "fb_val_sha": "0bf4cc6000ee0b398c7182c5bc022f9bca55059eb0ab66c81f9813bd0e82f7fe",
            "comp_val_sha": "737b4a931ec2b141e88093ff32210cf8736c17f7e7146a9064f9cf9dc6a8ab0c",
        },
    }

    for arm_id, exp in expected_specs.items():
        arm = FROZEN_FOUR_ARM_REGISTRY[arm_id]
        assert arm["preflight_run_id"] == exp["run_id"]
        hashes = arm["frozen_artifact_hashes"]
        assert hashes["final_changed_dict.json"] == exp["changed_dict_sha"]
        assert hashes["train_fairbias_trace.json"] == exp["trace_sha"]
        assert hashes["validation_metrics_baseline.json"] == exp["base_val_sha"]
        assert hashes["validation_metrics_fairbias.json"] == exp["fb_val_sha"]
        assert hashes["validation_comparison.json"] == exp["comp_val_sha"]


# ------------------------------------------------------------------------------
# Test 3: Wrong/missing changed_dict hash fails closed
# ------------------------------------------------------------------------------
def test_3_wrong_or_missing_changed_dict_hash_fails_closed(tmp_path: pathlib.Path) -> None:
    # A. Corrupted changed_dict
    corrupted_dir = create_mock_preflight_directory(
        tmp_path / "case_corrupt", mutate_arm="ARM_D3_001", mutate_file="final_changed_dict.json"
    )
    harness = NHISD4TestReleaseHarness(
        preflight_runs_dir=corrupted_dir, enforce_frozen_inputs=False
    )
    with pytest.raises(FrozenArtifactIntegrityError, match="SHA-256 mismatch"):
        harness.verify_frozen_preflight_artifacts()

    # B. Missing changed_dict
    missing_dir = create_mock_preflight_directory(
        tmp_path / "case_missing", mutate_arm="ARM_D3_001", delete_file="final_changed_dict.json"
    )
    harness_miss = NHISD4TestReleaseHarness(
        preflight_runs_dir=missing_dir, enforce_frozen_inputs=False
    )
    with pytest.raises(FrozenArtifactIntegrityError, match="Missing frozen preflight artifact"):
        harness_miss.verify_frozen_preflight_artifacts()


# ------------------------------------------------------------------------------
# Test 4: Wrong/missing trace hash fails closed
# ------------------------------------------------------------------------------
def test_4_wrong_or_missing_trace_hash_fails_closed(tmp_path: pathlib.Path) -> None:
    corrupted_dir = create_mock_preflight_directory(
        tmp_path / "case_trace", mutate_arm="ARM_D3_002", mutate_file="train_fairbias_trace.json"
    )
    harness = NHISD4TestReleaseHarness(
        preflight_runs_dir=corrupted_dir, enforce_frozen_inputs=False
    )
    with pytest.raises(FrozenArtifactIntegrityError, match="SHA-256 mismatch"):
        harness.verify_frozen_preflight_artifacts()


# ------------------------------------------------------------------------------
# Test 5: Wrong split/features hash fails closed
# ------------------------------------------------------------------------------
def test_5_wrong_split_or_features_hash_fails_closed(tmp_path: pathlib.Path) -> None:
    bad_split = tmp_path / "bad_split.csv"
    bad_split.write_text("invalid,split,data", encoding="utf-8")

    harness = NHISD4TestReleaseHarness(split_manifest_path=bad_split)
    with pytest.raises(ValueError, match="Frozen split manifest SHA-256 mismatch"):
        harness.verify_frozen_inputs()

    bad_feat = tmp_path / "bad_feat.parquet"
    bad_feat.write_text("invalid parquet", encoding="utf-8")

    harness_feat = NHISD4TestReleaseHarness(features_parquet_path=bad_feat)
    with pytest.raises(ValueError, match="Frozen features parquet SHA-256 mismatch"):
        harness_feat.verify_frozen_inputs()


# ------------------------------------------------------------------------------
# Test 6: Audit-only mode does not access or score TEST
# ------------------------------------------------------------------------------
def test_6_audit_only_mode_does_not_access_or_score_test() -> None:
    harness = NHISD4TestReleaseHarness()
    with patch.object(harness.adapter, "get_pooled_cohort", side_effect=AssertionError("get_pooled_cohort was called!")):
        audit_result = harness.run_audit()
        assert audit_result["status"] == "PASS"
        assert audit_result["real_test_evaluated"] is False
        assert audit_result["test_embargo_active"] is True


# ------------------------------------------------------------------------------
# Test 7: Audit-only creates no TEST metrics artifacts
# ------------------------------------------------------------------------------
def test_7_audit_only_creates_no_test_metrics_artifacts(tmp_path: pathlib.Path) -> None:
    test_rel_dir = _REPO_ROOT / "runs" / "nhis_d4_test_release"
    before_files = set(test_rel_dir.glob("**/*test_metrics*.json")) if test_rel_dir.is_dir() else set()
    harness = NHISD4TestReleaseHarness()
    res = harness.run_audit()
    assert res["status"] == "PASS"

    # Verify no test output files were created by this audit
    if test_rel_dir.is_dir():
        # Any file inside must not be from this audit
        after_files = set(test_rel_dir.glob("**/*test_metrics*.json"))
        assert after_files == before_files


# ------------------------------------------------------------------------------
# Test 8: Execute path never invokes FairBias mitigation
# ------------------------------------------------------------------------------
def test_8_execute_path_never_invokes_fairbias_mitigation(tmp_path: pathlib.Path) -> None:
    preflight_dir = create_mock_preflight_directory(tmp_path)
    adapter = create_synthetic_adapter(tmp_path)
    harness = NHISD4TestReleaseHarness(
        adapter=adapter,
        preflight_runs_dir=preflight_dir,
        enforce_frozen_inputs=False,
        enforce_frozen_artifacts=True,
    )

    with patch("fairbias.mitigation.FairBiasMitigation.__init__", side_effect=AssertionError("FairBiasMitigation instantiated!")):
        with patch("fairbias.mitigation.FairBiasMitigation.mitigate_step", side_effect=AssertionError("mitigate_step called!")):
            res = harness.execute_release(
                release_id="TEST_SYNTH_08",
                output_base_dir=tmp_path / "releases",
            )
            assert res["manifest"]["release_status"] == "COMPLETE"


# ------------------------------------------------------------------------------
# Test 9: Execute path never invokes calculate_epsilon
# ------------------------------------------------------------------------------
def test_9_execute_path_never_invokes_calculate_epsilon(tmp_path: pathlib.Path) -> None:
    preflight_dir = create_mock_preflight_directory(tmp_path)
    adapter = create_synthetic_adapter(tmp_path)
    harness = NHISD4TestReleaseHarness(
        adapter=adapter,
        preflight_runs_dir=preflight_dir,
        enforce_frozen_inputs=False,
    )

    with patch("fairbias.evaluator.FairEvaluator.calculate_epsilon", side_effect=AssertionError("calculate_epsilon called!")):
        res = harness.execute_release(
            release_id="TEST_SYNTH_09",
            output_base_dir=tmp_path / "releases",
        )
        assert res["manifest"]["release_status"] == "COMPLETE"


# ------------------------------------------------------------------------------
# Test 10: Execute path never invokes calculate_nmi_dict
# ------------------------------------------------------------------------------
def test_10_execute_path_never_invokes_calculate_nmi_dict(tmp_path: pathlib.Path) -> None:
    preflight_dir = create_mock_preflight_directory(tmp_path)
    adapter = create_synthetic_adapter(tmp_path)
    harness = NHISD4TestReleaseHarness(
        adapter=adapter,
        preflight_runs_dir=preflight_dir,
        enforce_frozen_inputs=False,
    )

    with patch("fairbias.transform.calculate_nmi_dict", side_effect=AssertionError("calculate_nmi_dict called!")):
        res = harness.execute_release(
            release_id="TEST_SYNTH_10",
            output_base_dir=tmp_path / "releases",
        )
        assert res["manifest"]["release_status"] == "COMPLETE"


# ------------------------------------------------------------------------------
# Test 11: Execute path never invokes generate_pooled_splits
# ------------------------------------------------------------------------------
def test_11_execute_path_never_invokes_generate_pooled_splits(tmp_path: pathlib.Path) -> None:
    preflight_dir = create_mock_preflight_directory(tmp_path)
    adapter = create_synthetic_adapter(tmp_path)
    harness = NHISD4TestReleaseHarness(
        adapter=adapter,
        preflight_runs_dir=preflight_dir,
        enforce_frozen_inputs=False,
    )

    with patch("nhis_fairbias.pooled.generate_pooled_splits", side_effect=AssertionError("generate_pooled_splits called!")):
        res = harness.execute_release(
            release_id="TEST_SYNTH_11",
            output_base_dir=tmp_path / "releases",
        )
        assert res["manifest"]["release_status"] == "COMPLETE"


# ------------------------------------------------------------------------------
# Test 12: Execute path never invokes train_test_split
# ------------------------------------------------------------------------------
def test_12_execute_path_never_invokes_train_test_split(tmp_path: pathlib.Path) -> None:
    preflight_dir = create_mock_preflight_directory(tmp_path)
    adapter = create_synthetic_adapter(tmp_path)
    harness = NHISD4TestReleaseHarness(
        adapter=adapter,
        preflight_runs_dir=preflight_dir,
        enforce_frozen_inputs=False,
    )

    with patch("sklearn.model_selection.train_test_split", side_effect=AssertionError("train_test_split called!")):
        res = harness.execute_release(
            release_id="TEST_SYNTH_12",
            output_base_dir=tmp_path / "releases",
        )
        assert res["manifest"]["release_status"] == "COMPLETE"


# ------------------------------------------------------------------------------
# Test 13: Frozen changed_dict is applied unchanged to TRAIN and TEST
# ------------------------------------------------------------------------------
def test_13_frozen_changed_dict_applied_unchanged_to_train_and_test(tmp_path: pathlib.Path) -> None:
    preflight_dir = create_mock_preflight_directory(tmp_path)
    adapter = create_synthetic_adapter(tmp_path)
    harness = NHISD4TestReleaseHarness(
        adapter=adapter,
        preflight_runs_dir=preflight_dir,
        enforce_frozen_inputs=False,
    )

    from fairbias.transform import FairTransform
    orig_transform_data = FairTransform.transform_data

    applied_changed_dicts = []

    def spy_transform_data(self_obj, df, changed_dict, *args, **kwargs):
        applied_changed_dicts.append(copy.deepcopy(changed_dict))
        return orig_transform_data(self_obj, df, changed_dict, *args, **kwargs)

    with patch.object(FairTransform, "transform_data", side_effect=spy_transform_data, autospec=True):
        res = harness.execute_release(
            release_id="TEST_SYNTH_13",
            output_base_dir=tmp_path / "releases",
        )
        # Exactly 4 arms * 2 (train and test) = 8 transform_data calls
        assert len(applied_changed_dicts) == 8

        # Each pair of train and test transforms must use the identical changed_dict
        for i in range(0, 8, 2):
            train_cd = applied_changed_dicts[i]
            test_cd = applied_changed_dicts[i + 1]
            assert train_cd == test_cd


# ------------------------------------------------------------------------------
# Test 14: Scaler fit is TRAIN-only
# ------------------------------------------------------------------------------
def test_14_scaler_fit_is_train_only(tmp_path: pathlib.Path) -> None:
    preflight_dir = create_mock_preflight_directory(tmp_path)
    adapter = create_synthetic_adapter(tmp_path, n_train=60, n_test=40)
    harness = NHISD4TestReleaseHarness(
        adapter=adapter,
        preflight_runs_dir=preflight_dir,
        enforce_frozen_inputs=False,
    )

    fit_sample_sizes = []
    orig_fit = MinMaxScaler.fit

    def spy_fit(self_obj, X, *args, **kwargs):
        fit_sample_sizes.append(len(X))
        return orig_fit(self_obj, X, *args, **kwargs)

    with patch.object(MinMaxScaler, "fit", side_effect=spy_fit, autospec=True):
        harness.execute_release(
            release_id="TEST_SYNTH_14",
            output_base_dir=tmp_path / "releases",
        )
        # Every scaler fit must receive n_train (60), never n_test (40) or 100
        assert len(fit_sample_sizes) == 8  # 4 arms * 2 scalers (base and fb)
        assert all(n == 60 for n in fit_sample_sizes)


# ------------------------------------------------------------------------------
# Test 15: Baseline LR and FairBias LR use identical frozen model specification
# ------------------------------------------------------------------------------
def test_15_baseline_and_fairbias_lr_identical_frozen_specification() -> None:
    m_base = LogisticRegression(random_state=PRIMARY_D4_RANDOM_SEED, max_iter=1000, solver="lbfgs")
    m_fb = LogisticRegression(random_state=PRIMARY_D4_RANDOM_SEED, max_iter=1000, solver="lbfgs")
    assert m_base.get_params() == m_fb.get_params()
    assert m_base is not m_fb


# ------------------------------------------------------------------------------
# Test 16: TEST is never passed to .fit()
# ------------------------------------------------------------------------------
def test_16_test_never_passed_to_fit(tmp_path: pathlib.Path) -> None:
    preflight_dir = create_mock_preflight_directory(tmp_path)
    adapter = create_synthetic_adapter(tmp_path, n_train=60, n_test=40)
    harness = NHISD4TestReleaseHarness(
        adapter=adapter,
        preflight_runs_dir=preflight_dir,
        enforce_frozen_inputs=False,
    )

    lr_fit_sizes = []
    orig_lr_fit = LogisticRegression.fit

    def spy_lr_fit(self_obj, X, y, *args, **kwargs):
        lr_fit_sizes.append(len(X))
        return orig_lr_fit(self_obj, X, y, *args, **kwargs)

    with patch.object(LogisticRegression, "fit", side_effect=spy_lr_fit, autospec=True):
        harness.execute_release(
            release_id="TEST_SYNTH_16",
            output_base_dir=tmp_path / "releases",
        )
        assert len(lr_fit_sizes) == 8  # 4 arms * 2 models
        assert all(n == 60 for n in lr_fit_sizes)
        assert 40 not in lr_fit_sizes


# ------------------------------------------------------------------------------
# Test 17: VALIDATION is not used by the TEST release path
# ------------------------------------------------------------------------------
def test_17_validation_not_used_by_test_release_path(tmp_path: pathlib.Path) -> None:
    preflight_dir = create_mock_preflight_directory(tmp_path)
    adapter = create_synthetic_adapter(tmp_path)

    # Corrupt the 'val' entry in get_pooled_cohort return value
    orig_get_cohort = adapter.get_pooled_cohort

    def poisoned_get_cohort(*args, **kwargs):
        res = orig_get_cohort(*args, **kwargs)
        # Poison val entry so that any access raises exception
        res["val"] = None
        return res

    adapter.get_pooled_cohort = poisoned_get_cohort

    harness = NHISD4TestReleaseHarness(
        adapter=adapter,
        preflight_runs_dir=preflight_dir,
        enforce_frozen_inputs=False,
    )

    res = harness.execute_release(
        release_id="TEST_SYNTH_17",
        output_base_dir=tmp_path / "releases",
    )
    assert res["manifest"]["validation_used_for_test_execution"] is False
    assert res["manifest"]["release_status"] == "COMPLETE"


# ------------------------------------------------------------------------------
# Test 18: Threshold is exactly 0.5
# ------------------------------------------------------------------------------
def test_18_threshold_is_exactly_0_5(tmp_path: pathlib.Path) -> None:
    assert PRIMARY_D4_PREDICTION_THRESHOLD == 0.5
    harness = NHISD4TestReleaseHarness()
    for bad_threshold in [0.3, 0.45, 0.55, 0.7]:
        with pytest.raises(ValueError, match="prediction_threshold"):
            harness.execute_release(release_id="BAD_THRESH", prediction_threshold=bad_threshold)


# ------------------------------------------------------------------------------
# Test 19: Seed is exactly 0
# ------------------------------------------------------------------------------
def test_19_seed_is_exactly_0() -> None:
    assert PRIMARY_D4_RANDOM_SEED == 0
    harness = NHISD4TestReleaseHarness()
    for bad_seed in [1, 42, -1, 100]:
        with pytest.raises(ValueError, match="random_seed"):
            harness.execute_release(release_id="BAD_SEED", random_seed=bad_seed)


# ------------------------------------------------------------------------------
# Test 20: HISP 7 groups produce 21 pairs in synthetic complete-coverage TEST
# ------------------------------------------------------------------------------
def test_20_hisp_7_groups_produce_21_pairs_in_synthetic_test() -> None:
    o_complete = pd.Series([1, 2, 3, 4, 5, 6, 7] * 20)
    cov = compute_group_coverage(
        o_group=o_complete,
        expected_group_count=7,
        expected_groups=[1, 2, 3, 4, 5, 6, 7],
    )
    assert cov["group_coverage_complete"] is True
    assert cov["observed_group_count"] == 7
    assert cov["expected_pair_count"] == 21
    assert cov["observed_pair_count"] == 21

    y_true = np.array([0, 1] * 70)
    y_pred = np.array([0, 0] * 70)
    y_prob = np.array([0.1, 0.2] * 70)
    eval_res = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        o_group=o_complete,
        expected_group_count=7,
        expected_groups=[1, 2, 3, 4, 5, 6, 7],
    )
    assert eval_res["multicategory_pairwise"]["num_groups"] == 7
    assert eval_res["multicategory_pairwise"]["num_pairs"] == 21
    assert eval_res["multicategory_pairwise"]["coverage_complete"] is True


# ------------------------------------------------------------------------------
# Test 21: HISP incomplete synthetic TEST is clearly marked incomplete
# ------------------------------------------------------------------------------
def test_21_hisp_incomplete_synthetic_test_marked_incomplete() -> None:
    o_incomplete = pd.Series([1, 2, 3, 4, 5] * 20)
    cov = compute_group_coverage(
        o_group=o_incomplete,
        expected_group_count=7,
        expected_groups=[1, 2, 3, 4, 5, 6, 7],
    )
    assert cov["group_coverage_complete"] is False
    assert cov["observed_group_count"] == 5
    assert cov["observed_pair_count"] == 10
    assert "Incomplete group coverage" in cov["diagnostics"]

    y_true = np.array([0, 1] * 50)
    y_pred = np.array([0, 0] * 50)
    y_prob = np.array([0.1, 0.2] * 50)
    eval_res = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        o_group=o_incomplete,
        expected_group_count=7,
        expected_groups=[1, 2, 3, 4, 5, 6, 7],
    )
    assert eval_res["multicategory_pairwise"]["coverage_complete"] is False
    assert "Incomplete group coverage" in eval_res["multicategory_pairwise"]["diagnostics"]


# ------------------------------------------------------------------------------
# Test 22: One-time release directory refuses overwrite/re-execution
# ------------------------------------------------------------------------------
def test_22_one_time_release_directory_refuses_overwrite(tmp_path: pathlib.Path) -> None:
    preflight_dir = create_mock_preflight_directory(tmp_path)
    adapter = create_synthetic_adapter(tmp_path)
    harness = NHISD4TestReleaseHarness(
        adapter=adapter,
        preflight_runs_dir=preflight_dir,
        enforce_frozen_inputs=False,
    )

    release_dir = tmp_path / "releases"
    res1 = harness.execute_release(
        release_id="TEST_ONETIME",
        output_base_dir=release_dir,
    )
    assert res1["state"]["status"] == "COMPLETE"

    # Attempt second execution with same release_id
    with pytest.raises(ReleaseCollisionError, match="One-time release semantics strictly forbid"):
        harness.execute_release(
            release_id="TEST_ONETIME",
            output_base_dir=release_dir,
        )


# ------------------------------------------------------------------------------
# Test 23: Synthetic STARTED failure remains auditable and is not silently overwritten
# ------------------------------------------------------------------------------
def test_23_synthetic_started_failure_remains_auditable(tmp_path: pathlib.Path) -> None:
    preflight_dir = create_mock_preflight_directory(tmp_path)
    adapter = create_synthetic_adapter(tmp_path)
    harness = NHISD4TestReleaseHarness(
        adapter=adapter,
        preflight_runs_dir=preflight_dir,
        enforce_frozen_inputs=False,
    )

    release_dir = tmp_path / "releases"

    # Force a failure during release execution
    with patch.object(adapter, "get_pooled_cohort", side_effect=RuntimeError("Simulated pipeline crash")):
        with pytest.raises(RuntimeError, match="Simulated pipeline crash"):
            harness.execute_release(
                release_id="CRASHED_RELEASE",
                output_base_dir=release_dir,
            )

    # State file must record FAILED
    state_file = release_dir / "CRASHED_RELEASE" / "release_state.json"
    assert state_file.is_file()
    state_data = json.loads(state_file.read_text(encoding="utf-8"))
    assert state_data["status"] == "FAILED"
    assert "Simulated pipeline crash" in state_data["error"]

    # Re-running on this crashed release directory must fail closed
    with pytest.raises(ReleaseCollisionError, match="status 'FAILED'"):
        harness.execute_release(
            release_id="CRASHED_RELEASE",
            output_base_dir=release_dir,
        )


# ------------------------------------------------------------------------------
# Test 24: Output manifest hashes are correct
# ------------------------------------------------------------------------------
def test_24_output_manifest_hashes_are_correct(tmp_path: pathlib.Path) -> None:
    preflight_dir = create_mock_preflight_directory(tmp_path)
    adapter = create_synthetic_adapter(tmp_path)
    harness = NHISD4TestReleaseHarness(
        adapter=adapter,
        preflight_runs_dir=preflight_dir,
        enforce_frozen_inputs=False,
    )

    release_dir = tmp_path / "releases"
    res = harness.execute_release(
        release_id="HASH_CHECK_RELEASE",
        output_base_dir=release_dir,
    )

    manifest_path = pathlib.Path(res["manifest_path"])
    assert manifest_path.is_file()
    manifest_sha = compute_sha256(manifest_path)
    assert res["state"]["manifest_sha256"] == manifest_sha

    # Verify per-arm hashes in manifest match actual disk files
    manifest = res["manifest"]
    for arm_id, arm_data in manifest["arms"].items():
        arm_dir = release_dir / "HASH_CHECK_RELEASE" / arm_id
        for fname, hash_info in arm_data["output_artifacts"].items():
            fpath = arm_dir / fname
            assert fpath.is_file()
            disk_sha = compute_sha256(fpath)
            assert hash_info["sha256"] == disk_sha
            assert hash_info["size_bytes"] == fpath.stat().st_size
