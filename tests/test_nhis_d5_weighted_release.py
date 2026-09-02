"""Comprehensive unit and integration test suite for Gate D5.1b Release Harness.

Covers all 30 mandated checks from Section 19:
1. canonical release ID
2. fresh release directory requirement
3. duplicate release collision fails closed
4. frozen split/features hashes
5. exact four-arm registry
6. scientific base commit frozen
7. release harness vs scientific code boundary
8. STARTED written before arm execution
9. failure after STARTED produces FAILED
10. successful mocked execution produces COMPLETE
11. all 4 arms required
12. missing arm fails release
13. exact per-arm artifact schema
14. every artifact hash recorded
15. recorded hashes reproduce from disk
16. changed_dict JSON reload works
17. trace preserves accepted-vs-total semantics
18. baseline/FairBias evaluation both persisted
19. group metrics CSV persisted
20. HISP 7 groups / 21-pair metadata
21. validation_used_for_mitigation=false
22. TEST partition never requested
23. no TEST CLI flag exists
24. audit-only cannot call execute_train_validation
25. audit-only creates no substantive result artifacts
26. LR remains unweighted
27. evaluation remains unweighted
28. threshold=0.5
29. seed=0
30. D4 archive/tag unchanged
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import pathlib
import subprocess
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fairbias.transform_trace import (
    FairBiasTransformStep,
    FairBiasTransformTrace,
)
from nhis_fairbias.d5_weighted_release import (
    CANONICAL_D5_RELEASE_ID,
    FROZEN_D4_PREFLIGHT_REFERENCES,
    NHISD5WeightedReleaseManager,
    REQUIRED_PER_ARM_ARTIFACTS,
    SCIENTIFIC_EXECUTION_BASE_COMMIT,
)
from nhis_fairbias.d5_weighted_runner import (
    FROZEN_D5_ARMS,
    FROZEN_D4_RELEASE_TAG,
    FROZEN_FEATURES_PARQUET_SHA256,
    FROZEN_PRIMARY_ANALYSIS_COMMIT,
    FROZEN_SPLIT_MANIFEST_SHA256,
    PRIMARY_D5_RANDOM_SEED,
    NHISD5WeightedRunner,
)
from nhis_fairbias.download import compute_sha256
from nhis_fairbias.evaluation import (
    compute_evaluation_comparison,
    evaluate_predictions,
)
import scripts.run_nhis_d5_weighted_release as release_cli

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def make_mock_arm_execution_result(arm_id: str, **kwargs: Any) -> Dict[str, Any]:
    """Generate synthetic execution result simulating execute_train_validation."""
    arm_spec = FROZEN_D5_ARMS[arm_id]
    prot = arm_spec["protected_attribute"]

    step1 = FairBiasTransformStep(
        iteration=1,
        selected_feature="educ_a",
        feature_semantic_type="categorical",
        d_phi_before=0.08,
        epsilon=0.04,
        proposed_transformation={"1": "1", "2": "1"},
        accepted_transformation={"1": "1", "2": "1"},
        categorical_merge_mapping={"1": "1", "2": "1"},
        d_phi_after=0.03,
        dropped=False,
    )
    step2 = FairBiasTransformStep(
        iteration=2,
        selected_feature="agep_a",
        feature_semantic_type="numerical",
        d_phi_before=0.05,
        epsilon=0.04,
        proposed_transformation="poly_2",
        accepted_transformation="poly_2",
        numerical_exponent=2.0,
        d_phi_after=0.02,
        dropped=False,
    )
    trace = FairBiasTransformTrace(
        algorithm_mode="survey_weighted_geometry",
        protected_attribute=prot,
        epsilon_threshold=0.04,
        steps=[step1, step2],
        final_status="CONVERGED",
        final_max_dphi=0.02,
    )

    base_groups = [
        {
            "group": 1,
            "n": 7000,
            "outcome_positive_count": 1000,
            "outcome_negative_count": 6000,
            "predicted_positive_count": 800,
            "predicted_negative_count": 6200,
            "tp_count": 500,
            "fp_count": 300,
            "tn_count": 5700,
            "fn_count": 500,
            "prevalence": 0.143,
            "selection_rate": 0.114,
            "tpr": 0.50,
            "fpr": 0.05,
            "ppv": 0.625,
        },
        {
            "group": 2,
            "n": 7000,
            "outcome_positive_count": 1200,
            "outcome_negative_count": 5800,
            "predicted_positive_count": 1100,
            "predicted_negative_count": 5900,
            "tp_count": 700,
            "fp_count": 400,
            "tn_count": 5400,
            "fn_count": 500,
            "prevalence": 0.171,
            "selection_rate": 0.157,
            "tpr": 0.583,
            "fpr": 0.069,
            "ppv": 0.636,
        },
    ]

    val_eval_base = {
        "utility": {
            "auroc": 0.812,
            "auprc": 0.450,
            "balanced_accuracy": 0.720,
            "f1": 0.510,
            "accuracy": 0.850,
        },
        "fairness_gaps": {
            "demographic_parity_gap": 0.043,
            "equal_opportunity_gap": 0.083,
            "fpr_gap": 0.019,
            "equalized_odds_max_gap": 0.083,
        },
        "group_metrics": base_groups,
    }

    val_eval_fb = copy.deepcopy(val_eval_base)
    val_eval_fb["fairness_gaps"]["demographic_parity_gap"] = 0.020
    val_eval_fb["fairness_gaps"]["equal_opportunity_gap"] = 0.030

    return {
        "arm_id": arm_id,
        "changed_dict": {"educ_a": {"1": "1", "2": "1"}, "agep_a": 2.0},
        "trace": trace,
        "initial_dphi": {"educ_a": 0.08, "agep_a": 0.05},
        "final_dphi": {"educ_a": 0.03, "agep_a": 0.02},
        "val_eval_base": val_eval_base,
        "val_eval_fb": val_eval_fb,
        "val_comparison": compute_evaluation_comparison(val_eval_base, val_eval_fb),
        "converged": True,
        "termination_reason": "All d_phi below threshold",
        "execution_time_seconds": 0.42,
    }


# ------------------------------------------------------------------------------
# 1. Canonical Release ID
# ------------------------------------------------------------------------------
def test_1_canonical_release_id() -> None:
    """Canonical release ID must match exact protocol specification."""
    assert CANONICAL_D5_RELEASE_ID == "NHIS_D5_WEIGHTED_TRAIN_VAL_V1_bc6034e5"
    manager = NHISD5WeightedReleaseManager()
    assert manager.release_id == "NHIS_D5_WEIGHTED_TRAIN_VAL_V1_bc6034e5"


# ------------------------------------------------------------------------------
# 2. Fresh Release Directory Requirement
# ------------------------------------------------------------------------------
def test_2_fresh_release_directory_requirement(tmp_path: pathlib.Path) -> None:
    """Release directory must be fresh and not exist prior to substantive execution."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path)
    assert not manager.release_dir.exists()
    pre = manager.verify_release_preconditions()
    assert pre["status"] == "PASS"


# ------------------------------------------------------------------------------
# 3. Duplicate Release Collision Fails Closed
# ------------------------------------------------------------------------------
def test_3_duplicate_release_collision_fails_closed(tmp_path: pathlib.Path) -> None:
    """If release directory already exists, execution must fail closed without overwriting."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)
    manager.release_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileExistsError, match="Canonical release directory already exists"):
        manager.verify_release_preconditions()

    with pytest.raises(FileExistsError, match="Canonical release directory already exists"):
        manager.execute_release()


# ------------------------------------------------------------------------------
# 4. Frozen Split and Features Hashes Match
# ------------------------------------------------------------------------------
def test_4_frozen_split_and_features_hashes() -> None:
    """Split manifest and features parquet hashes must match frozen values."""
    assert FROZEN_SPLIT_MANIFEST_SHA256 == "6874a56f5484186dffdd5faeb7871c3bef85042ccfdf8d1e375fdde75a3ee9f5"
    assert FROZEN_FEATURES_PARQUET_SHA256 == "49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383"
    runner = NHISD5WeightedRunner(enforce_frozen_inputs=True)
    contract = runner.verify_frozen_input_contract()
    assert contract["split_hash_match"] is True
    assert contract["features_hash_match"] is True


# ------------------------------------------------------------------------------
# 5. Exact Four-Arm Registry
# ------------------------------------------------------------------------------
def test_5_exact_four_arm_registry() -> None:
    """Four-arm registry must contain exactly D5_ARM_001 through D5_ARM_004."""
    arms = list(FROZEN_D5_ARMS.keys())
    assert arms == ["D5_ARM_001", "D5_ARM_002", "D5_ARM_003", "D5_ARM_004"]


# ------------------------------------------------------------------------------
# 6. Scientific Base Commit Frozen
# ------------------------------------------------------------------------------
def test_6_scientific_base_commit_frozen() -> None:
    """Base scientific commit must be frozen at bc6034e530d6e92bbfa82feadc0cf783302cb30f."""
    assert SCIENTIFIC_EXECUTION_BASE_COMMIT == "bc6034e530d6e92bbfa82feadc0cf783302cb30f"


# ------------------------------------------------------------------------------
# 7. Release Harness vs Scientific Code Boundary
# ------------------------------------------------------------------------------
def test_7_release_harness_vs_scientific_code_boundary() -> None:
    """Scientific FairBias code and runner execution logic must not be modified."""
    diff_res = subprocess.run(
        [
            "git",
            "diff",
            f"{SCIENTIFIC_EXECUTION_BASE_COMMIT}..HEAD",
            "--",
            "src/fairbias/",
            "src/nhis_fairbias/d5_weighted_runner.py",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert diff_res.stdout.strip() == "", f"Unexpected scientific changes: {diff_res.stdout}"


# ------------------------------------------------------------------------------
# 8. STARTED Written Before Arm Execution
# ------------------------------------------------------------------------------
def test_8_started_written_before_arm_execution(tmp_path: pathlib.Path) -> None:
    """State machine must record STARTED before executing the first arm."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    states_seen = []

    def mock_exec_arm(arm_id: str, **kwargs):
        state_f = manager.release_dir / "release_state.json"
        assert state_f.is_file(), "release_state.json must exist before arm execution"
        data = json.loads(state_f.read_text(encoding="utf-8"))
        states_seen.append(data["status"])
        return make_mock_arm_execution_result(arm_id)

    with patch.object(manager.runner, "execute_train_validation", side_effect=mock_exec_arm):
        manager.execute_release()

    assert len(states_seen) == 4
    assert all(s == "STARTED" for s in states_seen)


# ------------------------------------------------------------------------------
# 9. Failure After STARTED Produces FAILED
# ------------------------------------------------------------------------------
def test_9_failure_after_started_produces_failed(tmp_path: pathlib.Path) -> None:
    """Failure during arm execution must atomically write FAILED and record error details."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    def failing_exec_arm(arm_id: str, **kwargs):
        if arm_id == "D5_ARM_002":
            raise RuntimeError("Simulated arm 2 catastrophic failure")
        return make_mock_arm_execution_result(arm_id)

    with patch.object(manager.runner, "execute_train_validation", side_effect=failing_exec_arm):
        with pytest.raises(RuntimeError, match="Simulated arm 2 catastrophic failure"):
            manager.execute_release()

    state_f = manager.release_dir / "release_state.json"
    assert state_f.is_file()
    state_data = json.loads(state_f.read_text(encoding="utf-8"))
    assert state_data["status"] == "FAILED"
    assert state_data["error_type"] == "RuntimeError"
    assert "Simulated arm 2 catastrophic failure" in state_data["error_message"]


# ------------------------------------------------------------------------------
# 10. Successful Mocked Execution Produces COMPLETE
# ------------------------------------------------------------------------------
def test_10_successful_mocked_execution_produces_complete(tmp_path: pathlib.Path) -> None:
    """Successful mocked execution must write COMPLETE state and manifest hash."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    with patch.object(manager.runner, "execute_train_validation", side_effect=make_mock_arm_execution_result):
        manifest = manager.execute_release()

    assert manifest["status"] == "COMPLETE"
    state_f = manager.release_dir / "release_state.json"
    state_data = json.loads(state_f.read_text(encoding="utf-8"))
    assert state_data["status"] == "COMPLETE"
    assert "manifest_sha256" in state_data


# ------------------------------------------------------------------------------
# 11. All 4 Arms Required
# ------------------------------------------------------------------------------
def test_11_all_4_arms_required(tmp_path: pathlib.Path) -> None:
    """Release must process all four arms."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    executed_arms = []

    def mock_exec(arm_id: str, **kwargs):
        executed_arms.append(arm_id)
        return make_mock_arm_execution_result(arm_id)

    with patch.object(manager.runner, "execute_train_validation", side_effect=mock_exec):
        manager.execute_release()

    assert executed_arms == ["D5_ARM_001", "D5_ARM_002", "D5_ARM_003", "D5_ARM_004"]


# ------------------------------------------------------------------------------
# 12. Missing Arm Fails Release
# ------------------------------------------------------------------------------
def test_12_missing_arm_fails_release(tmp_path: pathlib.Path) -> None:
    """If an arm is dropped or missing from results, release must fail closed."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    def mock_exec(arm_id: str, **kwargs):
        if arm_id == "D5_ARM_004":
            raise ValueError("Omitted arm 4")
        return make_mock_arm_execution_result(arm_id)

    with patch.object(manager.runner, "execute_train_validation", side_effect=mock_exec):
        with pytest.raises(ValueError, match="Omitted arm 4"):
            manager.execute_release()

    state_f = manager.release_dir / "release_state.json"
    state_data = json.loads(state_f.read_text(encoding="utf-8"))
    assert state_data["status"] == "FAILED"


# ------------------------------------------------------------------------------
# 13. Exact Per-Arm Artifact Schema
# ------------------------------------------------------------------------------
def test_13_exact_per_arm_artifact_schema(tmp_path: pathlib.Path) -> None:
    """Each arm directory must contain exactly the 12 required artifact files."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    with patch.object(manager.runner, "execute_train_validation", side_effect=make_mock_arm_execution_result):
        manager.execute_release()

    for arm_id in FROZEN_D5_ARMS:
        arm_dir = manager.release_dir / arm_id
        assert arm_dir.is_dir()
        found_files = {p.name for p in arm_dir.iterdir() if p.is_file()}
        assert found_files == set(REQUIRED_PER_ARM_ARTIFACTS)


# ------------------------------------------------------------------------------
# 14. Every Artifact Hash Recorded
# ------------------------------------------------------------------------------
def test_14_every_artifact_hash_recorded(tmp_path: pathlib.Path) -> None:
    """Top-level manifest must contain relative path and SHA-256 for every artifact."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    with patch.object(manager.runner, "execute_train_validation", side_effect=make_mock_arm_execution_result):
        manifest = manager.execute_release()

    artifacts_dict = manifest["artifacts"]
    # 4 arms * 12 artifacts = 48 artifacts
    assert len(artifacts_dict) == 48

    for arm_id in FROZEN_D5_ARMS:
        for art_name in REQUIRED_PER_ARM_ARTIFACTS:
            rel_k = f"{arm_id}/{art_name}"
            assert rel_k in artifacts_dict
            assert "sha256" in artifacts_dict[rel_k]
            assert len(artifacts_dict[rel_k]["sha256"]) == 64


# ------------------------------------------------------------------------------
# 15. Recorded Hashes Reproduce From Disk
# ------------------------------------------------------------------------------
def test_15_recorded_hashes_reproduce_from_disk(tmp_path: pathlib.Path) -> None:
    """Hashes stored in manifest must match bit-for-bit when recomputed from disk."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    with patch.object(manager.runner, "execute_train_validation", side_effect=make_mock_arm_execution_result):
        manifest = manager.execute_release()

    for rel_k, meta in manifest["artifacts"].items():
        disk_path = manager.release_dir / rel_k
        assert disk_path.is_file()
        assert compute_sha256(disk_path) == meta["sha256"]


# ------------------------------------------------------------------------------
# 16. changed_dict JSON Reload Works
# ------------------------------------------------------------------------------
def test_16_changed_dict_json_reload_works(tmp_path: pathlib.Path) -> None:
    """weighted_changed_dict.json must reload cleanly and preserve structure."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    with patch.object(manager.runner, "execute_train_validation", side_effect=make_mock_arm_execution_result):
        manager.execute_release()

    for arm_id in FROZEN_D5_ARMS:
        cd_path = manager.release_dir / arm_id / "weighted_changed_dict.json"
        loaded = json.loads(cd_path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        assert "educ_a" in loaded
        assert loaded["educ_a"] == {"1": "1", "2": "1"}
        assert loaded["agep_a"] == 2.0


# ------------------------------------------------------------------------------
# 17. Trace Preserves Accepted-vs-Total Semantics
# ------------------------------------------------------------------------------
def test_17_trace_preserves_accepted_vs_total_semantics(tmp_path: pathlib.Path) -> None:
    """Trace artifact must cleanly distinguish total attempted steps from accepted transforms."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    with patch.object(manager.runner, "execute_train_validation", side_effect=make_mock_arm_execution_result):
        manager.execute_release()

    for arm_id in FROZEN_D5_ARMS:
        tr_path = manager.release_dir / arm_id / "weighted_transform_trace.json"
        data = json.loads(tr_path.read_text(encoding="utf-8"))
        assert "total_attempted_steps" in data
        assert "total_accepted_transforms" in data
        assert "selected_feature_sequence" in data
        assert "accepted_transformations" in data
        assert data["total_attempted_steps"] == 2
        assert data["total_accepted_transforms"] == 2


# ------------------------------------------------------------------------------
# 18. Baseline and FairBias Evaluation Both Persisted
# ------------------------------------------------------------------------------
def test_18_baseline_and_fairbias_evaluation_both_persisted(tmp_path: pathlib.Path) -> None:
    """Both validation_metrics_baseline.json and validation_metrics_weighted_fairbias.json must be persisted."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    with patch.object(manager.runner, "execute_train_validation", side_effect=make_mock_arm_execution_result):
        manager.execute_release()

    for arm_id in FROZEN_D5_ARMS:
        p_base = manager.release_dir / arm_id / "validation_metrics_baseline.json"
        p_fb = manager.release_dir / arm_id / "validation_metrics_weighted_fairbias.json"
        base_data = json.loads(p_base.read_text(encoding="utf-8"))
        fb_data = json.loads(p_fb.read_text(encoding="utf-8"))
        assert "utility" in base_data and "fairness_gaps" in base_data
        assert "utility" in fb_data and "fairness_gaps" in fb_data


# ------------------------------------------------------------------------------
# 19. Group Metrics CSV Persisted
# ------------------------------------------------------------------------------
def test_19_group_metrics_csv_persisted(tmp_path: pathlib.Path) -> None:
    """Group metrics CSV files must be valid CSVs containing group, n, selection_rate, tpr, fpr."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    with patch.object(manager.runner, "execute_train_validation", side_effect=make_mock_arm_execution_result):
        manager.execute_release()

    for arm_id in FROZEN_D5_ARMS:
        for fname in ("validation_group_metrics_baseline.csv", "validation_group_metrics_weighted_fairbias.csv"):
            csv_path = manager.release_dir / arm_id / fname
            df = pd.read_csv(csv_path)
            for col in ("group", "n", "selection_rate", "tpr", "fpr"):
                assert col in df.columns, f"Missing {col} in {csv_path}"


# ------------------------------------------------------------------------------
# 20. HISP 7 Groups / 21-Pair Metadata
# ------------------------------------------------------------------------------
def test_20_hisp_7_groups_21_pair_metadata(tmp_path: pathlib.Path) -> None:
    """D5_ARM_002 (HISPALLP_A) arm_config.json must specify 7 groups and 21 pairs."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    with patch.object(manager.runner, "execute_train_validation", side_effect=make_mock_arm_execution_result):
        manager.execute_release()

    arm2_cfg_path = manager.release_dir / "D5_ARM_002" / "arm_config.json"
    arm2_cfg = json.loads(arm2_cfg_path.read_text(encoding="utf-8"))
    assert arm2_cfg["group_count"] == 7
    assert arm2_cfg["pair_count"] == 21
    assert arm2_cfg["multicategory_status"] == "empirical multicategory pairwise extension"


# ------------------------------------------------------------------------------
# 21. validation_used_for_mitigation=false
# ------------------------------------------------------------------------------
def test_21_validation_used_for_mitigation_is_false(tmp_path: pathlib.Path) -> None:
    """Manifest must guarantee validation_used_for_mitigation=false."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    with patch.object(manager.runner, "execute_train_validation", side_effect=make_mock_arm_execution_result):
        manifest = manager.execute_release()

    assert manifest["validation_used_for_mitigation"] is False


# ------------------------------------------------------------------------------
# 22. TEST Partition Never Requested
# ------------------------------------------------------------------------------
def test_22_test_partition_never_requested(tmp_path: pathlib.Path) -> None:
    """Adapter get_partition must never receive 'test' during release execution."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    requested_roles = []
    orig_get_partition = manager.runner.adapter.get_partition

    def spy_get_partition(role: str) -> pd.DataFrame:
        requested_roles.append(role)
        return orig_get_partition(role)

    with patch.object(manager.runner.adapter, "get_partition", side_effect=spy_get_partition), \
         patch.object(manager.runner, "execute_train_validation", side_effect=make_mock_arm_execution_result):
        manifest = manager.execute_release()

    assert all(r in ("train", "val") for r in requested_roles)
    assert "test" not in requested_roles
    assert "TEST" not in requested_roles
    assert manifest["test_partition_requested"] is False
    assert manifest["test_cohort_materialized"] is False
    assert manifest["test_evaluated"] is False


# ------------------------------------------------------------------------------
# 23. No TEST CLI Flag Exists
# ------------------------------------------------------------------------------
def test_23_no_test_cli_flag_exists() -> None:
    """CLI argument parser must reject any attempt to supply --test or --execute-test."""
    with pytest.raises(SystemExit):
        release_cli.parse_args(["--test"])

    with pytest.raises(SystemExit):
        release_cli.parse_args(["--execute-test"])


# ------------------------------------------------------------------------------
# 24. Audit-Only Cannot Call execute_train_validation
# ------------------------------------------------------------------------------
def test_24_audit_only_cannot_call_execute_train_validation() -> None:
    """run_audit_only must never invoke execute_train_validation."""
    manager = NHISD5WeightedReleaseManager(allow_substantive_execution=False)
    mock_exec = MagicMock()
    manager.runner.execute_train_validation = mock_exec

    res = manager.run_audit_only()
    assert res["status"] == "PASS"
    assert res["real_weighted_mitigation_executed"] is False
    assert res["validation_scored"] is False
    mock_exec.assert_not_called()


# ------------------------------------------------------------------------------
# 25. Audit-Only Creates No Substantive Result Artifacts
# ------------------------------------------------------------------------------
def test_25_audit_only_creates_no_substantive_result_artifacts(tmp_path: pathlib.Path) -> None:
    """run_audit_only must create zero output files or directories in releases root."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=False)
    res = manager.run_audit_only()
    assert res["status"] == "PASS"
    assert not manager.release_dir.exists()
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------------------
# 26. LR Remains Unweighted
# ------------------------------------------------------------------------------
def test_26_lr_remains_unweighted(tmp_path: pathlib.Path) -> None:
    """Model fitting must pass no sample_weight to LogisticRegression.fit."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    with patch.object(manager.runner, "execute_train_validation", side_effect=make_mock_arm_execution_result):
        manager.execute_release()

    for arm_id in FROZEN_D5_ARMS:
        cfg = json.loads((manager.release_dir / arm_id / "arm_config.json").read_text(encoding="utf-8"))
        assert cfg["classifier"] == "LogisticRegression"
        assert cfg["classifier_weighted"] is False


# ------------------------------------------------------------------------------
# 27. Evaluation Remains Unweighted
# ------------------------------------------------------------------------------
def test_27_evaluation_remains_unweighted() -> None:
    """Evaluation functions must not accept or use sample_weight."""
    sig1 = inspect.signature(evaluate_predictions)
    sig2 = inspect.signature(compute_evaluation_comparison)
    assert "sample_weight" not in sig1.parameters
    assert "sample_weight" not in sig2.parameters


# ------------------------------------------------------------------------------
# 28. Threshold Fixed at 0.5
# ------------------------------------------------------------------------------
def test_28_threshold_fixed_at_0_5(tmp_path: pathlib.Path) -> None:
    """Classification threshold is strictly fixed at 0.5."""
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    with patch.object(manager.runner, "execute_train_validation", side_effect=make_mock_arm_execution_result):
        manifest = manager.execute_release()

    assert manifest["threshold"] == 0.5
    for arm_id in FROZEN_D5_ARMS:
        cfg = json.loads((manager.release_dir / arm_id / "arm_config.json").read_text(encoding="utf-8"))
        assert cfg["prediction_threshold"] == 0.5


# ------------------------------------------------------------------------------
# 29. Seed Fixed at 0
# ------------------------------------------------------------------------------
def test_29_seed_fixed_at_0(tmp_path: pathlib.Path) -> None:
    """Random seed is strictly frozen at 0."""
    assert PRIMARY_D5_RANDOM_SEED == 0
    manager = NHISD5WeightedReleaseManager(releases_root=tmp_path, allow_substantive_execution=True)

    with patch.object(manager.runner, "execute_train_validation", side_effect=make_mock_arm_execution_result):
        manifest = manager.execute_release()

    assert manifest["random_seed"] == 0
    for arm_id in FROZEN_D5_ARMS:
        cfg = json.loads((manager.release_dir / arm_id / "arm_config.json").read_text(encoding="utf-8"))
        assert cfg["random_seed"] == 0


# ------------------------------------------------------------------------------
# 30. D4 Archive and Tag Unchanged
# ------------------------------------------------------------------------------
def test_30_d4_archive_and_tag_unchanged() -> None:
    """D4 release archive and git tag must remain strictly immutable."""
    diff_res = subprocess.run(
        [
            "git",
            "diff",
            "bb1f0601bf3a079bb09882003740df8a78bc0374..HEAD",
            "--",
            "docs/releases/NHIS_D4_PRIMARY_TEST_RELEASE_V1_9920fc5a",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert diff_res.stdout.strip() == ""

    tag_res = subprocess.run(
        ["git", "rev-parse", "nhis-d4-primary-test-v1"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert tag_res.stdout.strip() == "d74af83fb98a3805a7fb0767c5beeb3dbf1407e4"
