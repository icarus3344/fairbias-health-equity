"""Dynamic Gate D6.1a.1 tests.

The tests use the real immutable D6.0c JSON/CSV archive for static verification,
but every cohort request is handled by an in-memory synthetic adapter.  No real
NHIS cohort, real parquet materialization, or canonical D6.1b release is used.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import pathlib
import shutil
import subprocess
import sys
import types
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import pytest

import nhis_fairbias.d6_temporal_test_release as harness
from nhis_fairbias.d6_temporal_test_release import (
    AnchorMismatchError,
    D6TemporalTestEvaluator,
    D6TrainingStateReproducer,
    git_tracked_worktree_clean,
    D6_TRAIN_VAL_ARCHIVE_COMMIT,
    D6_TRAIN_VAL_ARCHIVE_DIR,
    D6_TRAIN_VAL_LEDGER_SHA256,
    D6_TRAIN_VAL_MANIFEST_SHA256,
    D6_TRAIN_VAL_RELEASE_ID,
    D6_TRAIN_VAL_TAG,
    D6_TRAIN_VAL_TAG_OBJECT,
    DEFAULT_RANDOM_SEED,
    EXCLUDED_DISABILITY_COMPONENTS,
    EXPECTED_TRAINING_STATE_ANCHORS,
    EXPECTED_TRAIN_SOURCE_ROW_DIGESTS,
    FAIRBIAS_ALGORITHM_MODE,
    FROZEN_D6_ARMS,
    FROZEN_D6_ARM_IDS,
    FROZEN_D6_SCIENTIFIC_PATHS,
    FUTURE_COMPLETE_FILE_COUNT,
    FUTURE_MANIFEST_TRACKED_ARTIFACT_COUNT,
    FUTURE_PER_ARM_ARTIFACT_COUNT,
    FUTURE_PER_ARM_TEST_ARTIFACTS,
    FrozenArchiveIntegrityError,
    FrozenD6ArchiveLoader,
    GlobalBarrierError,
    NHISD6TemporalTestReleaseManager,
    PREDICTION_THRESHOLD,
    PREPROCESSING_STATE_SHA256,
    ProductionD6TemporalArtifactWriter,
    ProductionD6TemporalRuntime,
    ReproducedArmState,
    ReleaseCollisionError,
    TEMPORAL_2024_DISCLOSURE,
    TEMPORAL_TEST_YEAR,
    TEMPORAL_TRAIN_YEAR,
    TEMPORAL_VALIDATION_YEAR,
    compute_canonical_json_sha256,
    compute_file_sha256,
)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CLI = _REPO_ROOT / "scripts" / "run_nhis_d6_temporal_test.py"


class SyntheticAdapter:
    """In-memory adapter that records requests and contains poison sentinels."""

    def __init__(self, *, summary_path: pathlib.Path | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.summary_path = summary_path

    def get_cohort(self, year: int, **kwargs: Any) -> Mapping[str, Any]:
        call = {"year": int(year), **kwargs}
        self.calls.append(call)
        if int(year) == TEMPORAL_VALIDATION_YEAR:
            raise AssertionError(
                "FATAL: D6.1 attempted to access frozen 2023 validation cohort"
            )
        if int(year) == TEMPORAL_TEST_YEAR and self.summary_path is not None:
            if not self.summary_path.is_file():
                raise AssertionError(
                    "FATAL: 2024 request occurred before training summary was frozen"
                )
        if int(year) not in (TEMPORAL_TRAIN_YEAR, TEMPORAL_TEST_YEAR):
            raise AssertionError(f"FATAL: unexpected synthetic year {year}")
        return {"synthetic": True, "year": int(year), "kwargs": dict(kwargs)}


def _good_reproduction(**kwargs: Any) -> Dict[str, Any]:
    loader = kwargs["archive_loader"]
    arm_id = kwargs["arm_id"]
    return {
        "observed_train_source_row_digest": loader.expected_train_digests()[arm_id],
        "preprocessing_state": loader.load_preprocessing_state(),
        "state_hashes": loader.expected_state_anchors()[arm_id],
    }


def _good_score(**kwargs: Any) -> Dict[str, Any]:
    return {
        "arm_id": kwargs["arm_id"],
        "test_year": TEMPORAL_TEST_YEAR,
        "diagnostic_only": True,
        "prediction_threshold": PREDICTION_THRESHOLD,
    }


def _copy_archive(tmp_path: pathlib.Path) -> pathlib.Path:
    target = tmp_path / D6_TRAIN_VAL_RELEASE_ID
    shutil.copytree(D6_TRAIN_VAL_ARCHIVE_DIR, target)
    return target


def _synthetic_summary(loader: FrozenD6ArchiveLoader) -> Dict[str, Any]:
    arms: Dict[str, Any] = {}
    for arm_id in FROZEN_D6_ARM_IDS:
        digest = loader.expected_train_digests()[arm_id]
        arm_anchors = loader.expected_state_anchors()[arm_id]
        payload: Dict[str, Any] = {
            "arm_id": arm_id,
            "expected_2022_source_row_digest": digest,
            "observed_2022_source_row_digest": digest,
            "digest_match": True,
            "preprocessing_state_expected_hash": PREPROCESSING_STATE_SHA256,
            "preprocessing_state_observed_hash": PREPROCESSING_STATE_SHA256,
            "preprocessing_state_match": True,
            "all_five_state_hashes_match": True,
        }
        for key, value in arm_anchors.items():
            payload[key] = {
                "expected_sha256": value,
                "observed_sha256": value,
                "match": True,
            }
        arms[arm_id] = payload
    return {
        "arms": arms,
        "global": {
            "cohort_digest_match_count": 4,
            "cohort_digest_expected_count": 4,
            "cohort_digest_matches": "4/4",
            "training_state_anchor_match_count": 20,
            "training_state_anchor_expected_count": 20,
            "training_state_anchor_matches": "20/20",
            "all_training_state_anchors_verified": True,
        },
    }


def _make_manager(
    tmp_path: pathlib.Path,
    *,
    loader: FrozenD6ArchiveLoader,
    reproduction_fn: Any = _good_reproduction,
    score_fn: Any = _good_score,
) -> tuple[NHISD6TemporalTestReleaseManager, SyntheticAdapter, pathlib.Path]:
    release_dir = tmp_path / "synthetic_d6_test"
    adapter = SyntheticAdapter(
        summary_path=release_dir / "training_state_reproduction_summary.json"
    )
    manager = NHISD6TemporalTestReleaseManager(
        archive_loader=loader,
        adapter=adapter,
        reproduction_fn=reproduction_fn,
        score_fn=score_fn,
    )
    return manager, adapter, release_dir


@pytest.fixture(scope="module")
def archive_loader() -> FrozenD6ArchiveLoader:
    return FrozenD6ArchiveLoader()


def test_01_exact_d6_tag_constants() -> None:
    assert D6_TRAIN_VAL_TAG == "nhis-d6-temporal-train-val-v1"
    assert D6_TRAIN_VAL_TAG_OBJECT == "e23941edf2085c84290868f69289d30af62b7e95"
    assert D6_TRAIN_VAL_ARCHIVE_COMMIT == "bdf154c541d365b216a732bc3a58415af41cdcc7"


def test_02_archive_loader_has_no_cohort_api(archive_loader: FrozenD6ArchiveLoader) -> None:
    assert not hasattr(archive_loader, "get_cohort")
    assert not hasattr(archive_loader, "adapter")


def test_03_archive_audit_passes(archive_loader: FrozenD6ArchiveLoader) -> None:
    result = archive_loader.verify()
    assert result["status"] == "PASS"
    assert result["archive_canonical_file_count"] == 47
    assert result["archive_total_file_count"] == 48


def test_04_archive_manifest_sha_exact(archive_loader: FrozenD6ArchiveLoader) -> None:
    result = archive_loader.verify()
    assert result["archive_manifest_sha256"] == D6_TRAIN_VAL_MANIFEST_SHA256


def test_05_archive_ledger_sha_exact(archive_loader: FrozenD6ArchiveLoader) -> None:
    result = archive_loader.verify()
    assert result["archive_ledger_sha256"] == D6_TRAIN_VAL_LEDGER_SHA256


def test_05a_archive_manifest_mutation_fails_closed(tmp_path: pathlib.Path) -> None:
    copied_archive = _copy_archive(tmp_path)
    manifest_path = copied_archive / "d6_temporal_train_val_manifest.json"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            '"status": "COMPLETE"', '"status": "MUTATED"', 1
        ),
        encoding="utf-8",
    )
    loader = FrozenD6ArchiveLoader(
        archive_dir=copied_archive,
        enforce_git_identity=False,
    )
    with pytest.raises(FrozenArchiveIntegrityError, match="manifest SHA mismatch"):
        loader.verify()


def test_05b_archive_ledger_anchor_mutation_fails_closed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied_archive = _copy_archive(tmp_path)
    ledger_path = copied_archive / "archive_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["model_state_anchors"]["D6_ARM_001"]["changed_dict"] = "0" * 64
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(harness, "D6_TRAIN_VAL_LEDGER_SHA256", compute_file_sha256(ledger_path))
    loader = FrozenD6ArchiveLoader(
        archive_dir=copied_archive,
        enforce_git_identity=False,
    )
    with pytest.raises(FrozenArchiveIntegrityError, match="model-state anchors"):
        loader.verify()


def test_06_four_train_digests_loaded(archive_loader: FrozenD6ArchiveLoader) -> None:
    assert len(archive_loader.expected_train_digests()) == 4
    assert set(archive_loader.expected_train_digests()) == set(FROZEN_D6_ARM_IDS)


def test_07_twenty_state_anchors_loaded(archive_loader: FrozenD6ArchiveLoader) -> None:
    anchors = archive_loader.expected_state_anchors()
    assert sum(len(values) for values in anchors.values()) == 20
    assert anchors == EXPECTED_TRAINING_STATE_ANCHORS


def test_08_preprocessing_hash_exact(archive_loader: FrozenD6ArchiveLoader) -> None:
    result = archive_loader.verify()
    assert result["preprocessing_state_expected_hash"] == PREPROCESSING_STATE_SHA256
    assert result["preprocessing_state_observed_hash"] == PREPROCESSING_STATE_SHA256
    assert result["preprocessing_state_match"] is True


def test_09_archive_release_state_complete() -> None:
    state = json.loads(
        (D6_TRAIN_VAL_ARCHIVE_DIR / "release_state.json").read_text(encoding="utf-8")
    )
    assert state == {
        "created_at": state["created_at"],
        "error": None,
        "manifest_sha256": D6_TRAIN_VAL_MANIFEST_SHA256,
        "release_id": D6_TRAIN_VAL_RELEASE_ID,
        "status": "COMPLETE",
        "updated_at": state["updated_at"],
    }


def test_10_archive_ledger_gate_is_historical_d60c() -> None:
    ledger = json.loads(
        (D6_TRAIN_VAL_ARCHIVE_DIR / "archive_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["archive_gate"] == "D6.0c"
    assert ledger["execution_commit"] == "fa8eb6096de995463b437c4e74f0e3da6a2c196d"


def test_11_scientific_boundary_includes_frozen_d6_runner() -> None:
    assert "src/nhis_fairbias/d6_temporal_runner.py" in FROZEN_D6_SCIENTIFIC_PATHS


def test_12_exact_four_arms() -> None:
    assert FROZEN_D6_ARM_IDS == (
        "D6_ARM_001", "D6_ARM_002", "D6_ARM_003", "D6_ARM_004"
    )
    assert set(FROZEN_D6_ARMS) == set(FROZEN_D6_ARM_IDS)


def test_13_arm_semantics_are_frozen() -> None:
    assert [FROZEN_D6_ARMS[arm]["expected_predictors"] for arm in FROZEN_D6_ARM_IDS] == [21, 21, 21, 15]
    assert FROZEN_D6_ARMS["D6_ARM_001"]["protected_attribute"] == "SEX_A"
    assert FROZEN_D6_ARMS["D6_ARM_002"]["protected_attribute"] == "HISPALLP_A"
    assert FROZEN_D6_ARMS["D6_ARM_003"]["protected_attribute"] == "DISAB3_A"
    assert FROZEN_D6_ARMS["D6_ARM_004"]["disability_arm"] == "exclude_disability_components"


def test_14_strict_unweighted_semantics_constants() -> None:
    assert DEFAULT_RANDOM_SEED == 0
    assert PREDICTION_THRESHOLD == 0.5
    assert all(FROZEN_D6_ARMS[arm]["feature_set"] == "PRIMARY_CORE" for arm in FROZEN_D6_ARM_IDS)


def test_15_fairbias_semantics_constants() -> None:
    assert FAIRBIAS_ALGORITHM_MODE == "tang2024_paper_faithful"
    assert harness.POWER_SEQUENCE_POLICY == "official_stream"
    assert harness.POWER_REVISIT_POLICY == "restart"
    assert harness.FAILED_ATTRIBUTE_MODE == "stop"


def test_16_disability_exclusion_contract() -> None:
    assert EXCLUDED_DISABILITY_COMPONENTS == (
        "visiondf_a", "hearingdf_a", "diff_a", "comdiff_a", "uppslfcr_a", "cogmemdff_a"
    )
    assert len(EXCLUDED_DISABILITY_COMPONENTS) == 6


def test_17_hisp_7_group_21_pair_contract() -> None:
    arm = FROZEN_D6_ARMS["D6_ARM_002"]
    assert arm["expected_group_count"] == 7
    assert arm["expected_pair_count"] == 21
    assert arm["expected_groups"] == [1, 2, 3, 4, 5, 6, 7]


def test_18_future_file_count_contract() -> None:
    assert len(FUTURE_PER_ARM_TEST_ARTIFACTS) == FUTURE_PER_ARM_ARTIFACT_COUNT == 10
    assert FUTURE_MANIFEST_TRACKED_ARTIFACT_COUNT == 41
    assert FUTURE_COMPLETE_FILE_COUNT == 43


def test_19_disclosure_rejects_global_holdout_language() -> None:
    assert "globally untouched primary holdout" in TEMPORAL_2024_DISCLOSURE
    assert "were previously included" in TEMPORAL_2024_DISCLOSURE


def test_20_audit_only_does_not_retain_adapter(archive_loader: FrozenD6ArchiveLoader) -> None:
    manager = NHISD6TemporalTestReleaseManager(archive_loader=archive_loader)
    result = manager.run_audit_only()
    assert manager.adapter is None
    assert result["real_2022_state_reproduction_executed"] is False


def test_21_audit_only_requests_zero_cohorts(archive_loader: FrozenD6ArchiveLoader) -> None:
    manager = NHISD6TemporalTestReleaseManager(archive_loader=archive_loader)
    result = manager.run_audit_only()
    assert result["2023_cohort_requested"] is False
    assert result["2024_test_cohort_requested"] is False
    assert result["2024_test_evaluated"] is False


def test_22_cli_audit_only_exact_semantic_markers() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(_CLI), "--audit-only"],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for marker in (
        "D6 TEMPORAL TEST HARNESS AUDIT",
        "VERIFIED",
        "4 / 4 LOADED",
        "20 / 20 LOADED",
        "2023 — ACCESS PROHIBITED",
        "PRODUCTION 2022 REPRODUCER:\nIMPLEMENTED",
        "PRODUCTION 2024 SCORER:\nIMPLEMENTED",
        "PRODUCTION ARTIFACT WRITER:\nIMPLEMENTED",
        "GLOBAL BARRIER:\nENFORCED",
        "REAL 2022 STATE REPRODUCTION EXECUTED:\nFALSE",
        "2024 TEST EVALUATED:\nFALSE",
        "D6.1a.1 AUDIT:\nPASS",
    ):
        assert marker in result.stdout


def test_23_cli_substantive_flag_fails_closed() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(_CLI), "--execute-frozen-temporal-test"],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires both --release-id and --expected-execution-head" in result.stderr


def test_24_cli_rejects_validation_flag() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(_CLI), "--year-2023"],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


@pytest.mark.parametrize("forbidden_flag", ("--validation", "--retrain-with-validation", "--tune-threshold"))
def test_24a_cli_rejects_validation_or_tuning_flags(forbidden_flag: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(_CLI), forbidden_flag],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_25_canonical_json_hash_is_deterministic() -> None:
    assert compute_canonical_json_sha256({"b": 2, "a": 1}) == compute_canonical_json_sha256({"a": 1, "b": 2})


def test_26_canonical_json_hash_rejects_nonfinite() -> None:
    with pytest.raises(FrozenArchiveIntegrityError):
        compute_canonical_json_sha256({"bad": float("inf")})


def test_27_training_reproducer_requests_only_2022(archive_loader: FrozenD6ArchiveLoader) -> None:
    adapter = SyntheticAdapter()
    reproducer = D6TrainingStateReproducer(
        archive_loader,
        adapter,
        reproduction_fn=_good_reproduction,
    )
    result = reproducer.reproduce_all()
    assert result["global"]["cohort_digest_matches"] == "4/4"
    assert [call["year"] for call in adapter.calls] == [2022, 2022, 2022, 2022]


def test_28_training_reproducer_checks_preprocessing_state(archive_loader: FrozenD6ArchiveLoader) -> None:
    adapter = SyntheticAdapter()

    def wrong_preprocessing(**kwargs: Any) -> Dict[str, Any]:
        payload = _good_reproduction(**kwargs)
        payload["preprocessing_state"] = {"fit_year": 2023}
        return payload

    reproducer = D6TrainingStateReproducer(archive_loader, adapter, reproduction_fn=wrong_preprocessing)
    with pytest.raises(AnchorMismatchError, match="preprocessing state"):
        reproducer.reproduce_arm("D6_ARM_001")


def test_29_training_reproducer_checks_changed_dict_hash(archive_loader: FrozenD6ArchiveLoader) -> None:
    adapter = SyntheticAdapter()

    def wrong_changed_dict(**kwargs: Any) -> Dict[str, Any]:
        payload = _good_reproduction(**kwargs)
        payload["state_hashes"] = dict(payload["state_hashes"])
        payload["state_hashes"]["changed_dict"] = "0" * 64
        return payload

    reproducer = D6TrainingStateReproducer(archive_loader, adapter, reproduction_fn=wrong_changed_dict)
    with pytest.raises(AnchorMismatchError, match="five training-state hashes"):
        reproducer.reproduce_arm("D6_ARM_001")


def test_30_training_reproducer_checks_all_five_state_hashes(archive_loader: FrozenD6ArchiveLoader) -> None:
    adapter = SyntheticAdapter()
    reproducer = D6TrainingStateReproducer(archive_loader, adapter, reproduction_fn=_good_reproduction)
    observation = reproducer.reproduce_arm("D6_ARM_002")
    assert observation.all_five_state_hashes_match is True
    assert len(observation.state_hashes) == 5


def test_31_training_summary_contains_global_4_of_4_and_20_of_20(archive_loader: FrozenD6ArchiveLoader) -> None:
    reproducer = D6TrainingStateReproducer(archive_loader, SyntheticAdapter(), reproduction_fn=_good_reproduction)
    summary = reproducer.reproduce_all()
    assert summary["global"]["cohort_digest_matches"] == "4/4"
    assert summary["global"]["training_state_anchor_matches"] == "20/20"
    assert summary["global"]["all_training_state_anchors_verified"] is True


def test_32_training_summary_has_expected_archive_provenance(archive_loader: FrozenD6ArchiveLoader) -> None:
    reproducer = D6TrainingStateReproducer(archive_loader, SyntheticAdapter(), reproduction_fn=_good_reproduction)
    summary = reproducer.reproduce_all()
    assert summary["d6_train_val_archive_tag"] == D6_TRAIN_VAL_TAG
    assert summary["d6_archive_commit"] == D6_TRAIN_VAL_ARCHIVE_COMMIT
    assert summary["d6_archive_manifest_sha"] == D6_TRAIN_VAL_MANIFEST_SHA256


def test_33_successful_manager_requests_all_four_2022_then_all_four_2024(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, adapter, release_dir = _make_manager(tmp_path, loader=archive_loader)
    result = manager.execute_release(release_dir, allow_substantive_execution=True)
    assert result["status"] == "COMPLETE"
    assert [call["year"] for call in adapter.calls] == [2022, 2022, 2022, 2022, 2024, 2024, 2024, 2024]


def test_34_successful_manager_has_no_2023_request(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, adapter, release_dir = _make_manager(tmp_path, loader=archive_loader)
    manager.execute_release(release_dir, allow_substantive_execution=True)
    assert [call["year"] for call in adapter.calls].count(2023) == 0


def test_35_global_barrier_event_precedes_first_2024_request(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _, release_dir = _make_manager(tmp_path, loader=archive_loader)
    result = manager.execute_release(release_dir, allow_substantive_execution=True)
    events = result["events"]
    barrier = events.index("ALL_D6_TRAINING_STATE_ANCHORS_VERIFIED")
    first_test = next(index for index, event in enumerate(events) if event.startswith("request_2024_"))
    assert barrier < first_test
    assert all(event.startswith("request_2022_") or event.startswith("reproduce_") for event in events[:barrier])


def test_35a_successful_event_sequence_is_two_phase_global_barrier(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _, release_dir = _make_manager(tmp_path, loader=archive_loader)
    result = manager.execute_release(release_dir, allow_substantive_execution=True)
    expected = [
        "request_2022_D6_ARM_001",
        "reproduce_D6_ARM_001",
        "request_2022_D6_ARM_002",
        "reproduce_D6_ARM_002",
        "request_2022_D6_ARM_003",
        "reproduce_D6_ARM_003",
        "request_2022_D6_ARM_004",
        "reproduce_D6_ARM_004",
        "ALL_D6_TRAINING_STATE_ANCHORS_VERIFIED",
        "request_2024_D6_ARM_001",
        "evaluate_2024_D6_ARM_001",
        "request_2024_D6_ARM_002",
        "evaluate_2024_D6_ARM_002",
        "request_2024_D6_ARM_003",
        "evaluate_2024_D6_ARM_003",
        "request_2024_D6_ARM_004",
        "evaluate_2024_D6_ARM_004",
    ]
    assert result["events"] == expected


def test_36_training_summary_is_frozen_before_first_2024_request(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, adapter, release_dir = _make_manager(tmp_path, loader=archive_loader)
    manager.execute_release(release_dir, allow_substantive_execution=True)
    assert (release_dir / "training_state_reproduction_summary.json").is_file()
    assert adapter.calls[4]["year"] == 2024


def test_37_successful_release_state_is_complete(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _, release_dir = _make_manager(tmp_path, loader=archive_loader)
    manager.execute_release(release_dir, allow_substantive_execution=True)
    state = json.loads((release_dir / "release_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "COMPLETE"
    assert state["training_state_anchor_matches"] == "20/20"
    assert state["cohort_digest_matches"] == "4/4"


def test_38_evaluator_2023_poison_is_exact(archive_loader: FrozenD6ArchiveLoader) -> None:
    evaluator = D6TemporalTestEvaluator(
        archive_loader,
        SyntheticAdapter(),
        _synthetic_summary(archive_loader),
        barrier_passed=True,
    )
    with pytest.raises(AssertionError, match="frozen 2023 validation cohort"):
        evaluator.get_cohort(2023)


def test_39_evaluator_2024_prebarrier_poison_is_exact(archive_loader: FrozenD6ArchiveLoader) -> None:
    evaluator = D6TemporalTestEvaluator(
        archive_loader,
        SyntheticAdapter(),
        _synthetic_summary(archive_loader),
        barrier_passed=False,
    )
    with pytest.raises(AssertionError, match="before all frozen training states were reproduced"):
        evaluator.get_cohort(2024)


def test_39a_explicit_barrier_flag_cannot_bypass_incomplete_summary(
    archive_loader: FrozenD6ArchiveLoader,
) -> None:
    incomplete = {
        "global": {
            "all_training_state_anchors_verified": True,
            "cohort_digest_matches": "4/4",
            "training_state_anchor_matches": "20/20",
        },
        "arms": {},
    }
    evaluator = D6TemporalTestEvaluator(
        archive_loader,
        SyntheticAdapter(),
        incomplete,
        barrier_passed=True,
    )
    with pytest.raises(AssertionError, match="before all frozen training states were reproduced"):
        evaluator.get_cohort(2024, arm_id="D6_ARM_001")


def test_40_evaluator_allows_2024_only_after_barrier(archive_loader: FrozenD6ArchiveLoader) -> None:
    adapter = SyntheticAdapter()
    evaluator = D6TemporalTestEvaluator(
        archive_loader,
        adapter,
        _synthetic_summary(archive_loader),
        barrier_passed=True,
        score_fn=_good_score,
    )
    result = evaluator.evaluate_arm("D6_ARM_001")
    assert result["test_year"] == 2024
    assert [call["year"] for call in adapter.calls] == [2024]


def test_40a_evaluator_get_cohort_requires_arm_for_2024(
    archive_loader: FrozenD6ArchiveLoader,
) -> None:
    evaluator = D6TemporalTestEvaluator(
        archive_loader,
        SyntheticAdapter(),
        _synthetic_summary(archive_loader),
        barrier_passed=True,
    )
    with pytest.raises(ValueError, match="arm_id is required"):
        evaluator.get_cohort(2024)
    cohort = evaluator.get_cohort(2024, arm_id="D6_ARM_001")
    assert cohort["year"] == 2024


def test_41_evaluator_rejects_threshold_change(archive_loader: FrozenD6ArchiveLoader) -> None:
    def bad_score(**kwargs: Any) -> Dict[str, Any]:
        return {"prediction_threshold": 0.6}

    evaluator = D6TemporalTestEvaluator(
        archive_loader,
        SyntheticAdapter(),
        _synthetic_summary(archive_loader),
        barrier_passed=True,
        score_fn=bad_score,
    )
    with pytest.raises(harness.D6HarnessError, match="change threshold"):
        evaluator.evaluate_arm("D6_ARM_001")


def test_42_evaluator_rejects_frozen_state_mutation(archive_loader: FrozenD6ArchiveLoader) -> None:
    def bad_score(**kwargs: Any) -> Dict[str, Any]:
        mutated = kwargs["frozen_state"]
        mutated["changed_dict_observed_sha256"] = "0" * 64
        return {"frozen_state": mutated}

    evaluator = D6TemporalTestEvaluator(
        archive_loader,
        SyntheticAdapter(),
        _synthetic_summary(archive_loader),
        barrier_passed=True,
        score_fn=bad_score,
    )
    with pytest.raises(harness.D6HarnessError, match="change frozen training state"):
        evaluator.evaluate_arm("D6_ARM_001")


def test_42a_evaluator_rejects_in_place_frozen_state_mutation(
    archive_loader: FrozenD6ArchiveLoader,
) -> None:
    def bad_score(**kwargs: Any) -> Dict[str, Any]:
        kwargs["frozen_state"]["changed_dict"]["observed_sha256"] = "0" * 64
        return {"test_year": TEMPORAL_TEST_YEAR}

    evaluator = D6TemporalTestEvaluator(
        archive_loader,
        SyntheticAdapter(),
        _synthetic_summary(archive_loader),
        barrier_passed=True,
        score_fn=bad_score,
    )
    with pytest.raises(harness.D6HarnessError, match="mutated frozen training state"):
        evaluator.evaluate_arm("D6_ARM_001")


def test_43_evaluator_all_four_results_are_test_only(archive_loader: FrozenD6ArchiveLoader) -> None:
    evaluator = D6TemporalTestEvaluator(
        archive_loader,
        SyntheticAdapter(),
        _synthetic_summary(archive_loader),
        barrier_passed=True,
        score_fn=_good_score,
    )
    results = evaluator.evaluate_all()
    assert list(results) == list(FROZEN_D6_ARM_IDS)
    assert all(item["test_year"] == 2024 for item in results.values())


@pytest.mark.parametrize("mismatched_arm", FROZEN_D6_ARM_IDS)
def test_44_any_arm_mismatch_blocks_all_2024_requests(
    mismatched_arm: str, tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    def mismatch(**kwargs: Any) -> Dict[str, Any]:
        payload = _good_reproduction(**kwargs)
        if kwargs["arm_id"] == mismatched_arm:
            payload["state_hashes"] = dict(payload["state_hashes"])
            payload["state_hashes"]["FairBias_LR"] = "1" * 64
        return payload

    manager, adapter, release_dir = _make_manager(
        tmp_path, loader=archive_loader, reproduction_fn=mismatch
    )
    with pytest.raises(GlobalBarrierError):
        manager.execute_release(release_dir, allow_substantive_execution=True)
    assert [call["year"] for call in adapter.calls] == [2022, 2022, 2022, 2022]
    state = json.loads((release_dir / "release_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert state["2024_request_count"] == 0


def test_49_late_arm004_mismatch_is_global_poison(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    def late_mismatch(**kwargs: Any) -> Dict[str, Any]:
        payload = _good_reproduction(**kwargs)
        if kwargs["arm_id"] == "D6_ARM_004":
            payload["observed_train_source_row_digest"] = "0" * 64
        return payload

    manager, adapter, release_dir = _make_manager(
        tmp_path, loader=archive_loader, reproduction_fn=late_mismatch
    )
    with pytest.raises(GlobalBarrierError):
        manager.execute_release(release_dir, allow_substantive_execution=True)
    assert len(adapter.calls) == 4
    assert all(call["year"] == 2022 for call in adapter.calls)


def test_50_preprocessing_mismatch_blocks_2024(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    def bad_preprocessing(**kwargs: Any) -> Dict[str, Any]:
        payload = _good_reproduction(**kwargs)
        payload["preprocessing_state"] = {"fit_year": 2023}
        return payload

    manager, adapter, release_dir = _make_manager(
        tmp_path, loader=archive_loader, reproduction_fn=bad_preprocessing
    )
    with pytest.raises(GlobalBarrierError):
        manager.execute_release(release_dir, allow_substantive_execution=True)
    assert len(adapter.calls) == 4
    assert all(call["year"] == 2022 for call in adapter.calls)


def test_51_train_digest_mismatch_blocks_2024(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    def bad_digest(**kwargs: Any) -> Dict[str, Any]:
        payload = _good_reproduction(**kwargs)
        if kwargs["arm_id"] == "D6_ARM_002":
            payload["observed_train_source_row_digest"] = "2" * 64
        return payload

    manager, adapter, release_dir = _make_manager(
        tmp_path, loader=archive_loader, reproduction_fn=bad_digest
    )
    with pytest.raises(GlobalBarrierError):
        manager.execute_release(release_dir, allow_substantive_execution=True)
    assert len(adapter.calls) == 4
    assert not any(call["year"] == 2024 for call in adapter.calls)


def test_52_failed_release_records_zero_2023_and_2024(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    def fail_all(**kwargs: Any) -> Dict[str, Any]:
        payload = _good_reproduction(**kwargs)
        payload["state_hashes"] = dict(payload["state_hashes"])
        payload["state_hashes"]["changed_dict"] = "3" * 64
        return payload

    manager, _, release_dir = _make_manager(
        tmp_path, loader=archive_loader, reproduction_fn=fail_all
    )
    with pytest.raises(GlobalBarrierError):
        manager.execute_release(release_dir, allow_substantive_execution=True)
    state = json.loads((release_dir / "release_state.json").read_text(encoding="utf-8"))
    assert state["2023_request_count"] == 0
    assert state["2024_request_count"] == 0


def test_53_failed_release_cannot_silently_retry(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _, release_dir = _make_manager(tmp_path, loader=archive_loader)
    with pytest.raises(GlobalBarrierError):
        manager.execute_release(
            release_dir,
            allow_substantive_execution=True,
            reproduction_fn=lambda **kwargs: {
                **_good_reproduction(**kwargs),
                "observed_train_source_row_digest": "4" * 64,
            },
        )
    with pytest.raises(ReleaseCollisionError):
        manager.execute_release(release_dir, allow_substantive_execution=True)


def test_54_collision_fails_before_any_new_request(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, adapter, release_dir = _make_manager(tmp_path, loader=archive_loader)
    release_dir.mkdir()
    with pytest.raises(ReleaseCollisionError):
        manager.execute_release(release_dir, allow_substantive_execution=True)
    assert adapter.calls == []


def test_55_precondition_failure_creates_no_release_directory(tmp_path: pathlib.Path) -> None:
    class FailingLoader:
        def verify(self) -> Dict[str, Any]:
            raise FrozenArchiveIntegrityError("synthetic archive mismatch")

    adapter = SyntheticAdapter()
    manager = NHISD6TemporalTestReleaseManager(archive_loader=FailingLoader(), adapter=adapter)
    target = tmp_path / "should_not_exist"
    with pytest.raises(FrozenArchiveIntegrityError, match="synthetic archive mismatch"):
        manager.execute_release(
            target,
            allow_substantive_execution=True,
            reproduction_fn=_good_reproduction,
        )
    assert not target.exists()
    assert adapter.calls == []


def test_56_execution_requires_explicit_opt_in(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _, release_dir = _make_manager(tmp_path, loader=archive_loader)
    with pytest.raises(harness.D6HarnessError, match="not authorized"):
        manager.execute_release(release_dir)
    assert not release_dir.exists()


def test_57_synthetic_success_writes_no_canonical_test_manifest(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _, release_dir = _make_manager(tmp_path, loader=archive_loader)
    manager.execute_release(release_dir, allow_substantive_execution=True)
    assert not (release_dir / "d6_temporal_test_manifest.json").exists()
    assert release_dir != D6_TRAIN_VAL_ARCHIVE_DIR


def test_58_global_event_sequence_has_no_2024_interleaving(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _, release_dir = _make_manager(tmp_path, loader=archive_loader)
    result = manager.execute_release(release_dir, allow_substantive_execution=True)
    events = result["events"]
    first_2024 = next(i for i, event in enumerate(events) if event.startswith("request_2024_"))
    assert sum(event.startswith("request_2022_") for event in events[:first_2024]) == 4
    assert "ALL_D6_TRAINING_STATE_ANCHORS_VERIFIED" in events[:first_2024]


def test_59_hisp_arm_is_not_binarized_in_registry() -> None:
    assert FROZEN_D6_ARMS["D6_ARM_002"]["expected_groups"] == [1, 2, 3, 4, 5, 6, 7]


def test_60_no_public_validation_accessor_is_defined() -> None:
    public_methods = {
        name for name, value in inspect.getmembers(D6TemporalTestEvaluator, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    assert "get_validation_cohort" not in public_methods
    assert "request_validation_cohort" not in public_methods


def test_61_source_module_has_no_real_adapter_import() -> None:
    source = inspect.getsource(harness)
    assert "from .adapter import" not in source
    assert "NHISStudyAdapter" not in harness.__dict__


def test_62_future_arm_artifact_names_are_exact() -> None:
    assert FUTURE_PER_ARM_TEST_ARTIFACTS == (
        "arm_config.json",
        "input_provenance.json",
        "training_state_reproduction.json",
        "frozen_changed_dict.json",
        "test_dphi_before_after.json",
        "test_metrics_baseline.json",
        "test_metrics_fairbias.json",
        "test_group_metrics_baseline.csv",
        "test_group_metrics_fairbias.csv",
        "test_comparison.json",
    )


def test_63_expected_digests_are_exact_constants() -> None:
    assert EXPECTED_TRAIN_SOURCE_ROW_DIGESTS["D6_ARM_001"].startswith("fbf4c5b0")
    assert EXPECTED_TRAIN_SOURCE_ROW_DIGESTS["D6_ARM_004"] == EXPECTED_TRAIN_SOURCE_ROW_DIGESTS["D6_ARM_003"]


def test_64_archive_loader_returns_copies(archive_loader: FrozenD6ArchiveLoader) -> None:
    anchors = archive_loader.expected_state_anchors()
    anchors["D6_ARM_001"]["changed_dict"] = "0" * 64
    assert archive_loader.expected_state_anchors()["D6_ARM_001"]["changed_dict"] != "0" * 64


def test_65_successful_manager_reports_exact_global_counts(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _, release_dir = _make_manager(tmp_path, loader=archive_loader)
    result = manager.execute_release(release_dir, allow_substantive_execution=True)
    summary = result["training_state_reproduction_summary"]["global"]
    assert summary["cohort_digest_matches"] == "4/4"
    assert summary["training_state_anchor_matches"] == "20/20"


def test_66_archive_constants_are_not_repurposed_for_d61() -> None:
    assert D6_TRAIN_VAL_RELEASE_ID.endswith("fa8eb609")
    assert "2024" in TEMPORAL_2024_DISCLOSURE
    assert TEMPORAL_TEST_YEAR == 2024


class _LiveObject:
    def __init__(self, state: Mapping[str, Any]) -> None:
        self.state = copy.deepcopy(dict(state))


class _StateOnlyRunner:
    compute_canonical_json_sha256 = staticmethod(compute_canonical_json_sha256)

    @staticmethod
    def extract_minmax_scaler_state(obj: _LiveObject, feature_order: list[str]) -> Dict[str, Any]:
        return {"kind": "scaler", "features": list(feature_order), "state": copy.deepcopy(obj.state)}

    @staticmethod
    def extract_logistic_regression_state(obj: _LiveObject, feature_order: list[str]) -> Dict[str, Any]:
        return {"kind": "lr", "features": list(feature_order), "state": copy.deepcopy(obj.state)}


def _production_digest(arm_id: str) -> str:
    return compute_canonical_json_sha256({"synthetic_production_train": arm_id})


def _production_hashes(arm_id: str) -> Dict[str, str]:
    return {
        logical_name: compute_canonical_json_sha256(
            {"synthetic_production_state": logical_name, "arm_id": arm_id}
        )
        for logical_name in EXPECTED_TRAINING_STATE_ANCHORS[arm_id]
    }


class _ProductionArchiveLoader:
    """Synthetic expected-value provider; production runtime never receives it."""

    def __init__(self) -> None:
        self.digests = {arm_id: _production_digest(arm_id) for arm_id in FROZEN_D6_ARM_IDS}
        self.anchors = {arm_id: _production_hashes(arm_id) for arm_id in FROZEN_D6_ARM_IDS}

    def verify(self) -> Dict[str, Any]:
        return {"status": "PASS", "synthetic": True}

    def expected_train_digests(self) -> Dict[str, str]:
        return copy.deepcopy(self.digests)

    def expected_state_anchors(self) -> Dict[str, Dict[str, str]]:
        return copy.deepcopy(self.anchors)


class _SyntheticProductionRuntime(ProductionD6TemporalRuntime):
    """Exercises the production manager/writer route without real data or parquet."""

    def __init__(self, preprocessing_state: Mapping[str, Any], summary_path: pathlib.Path | None = None) -> None:
        self.preprocessing_state = copy.deepcopy(dict(preprocessing_state))
        self.summary_path = summary_path
        self.calls: list[dict[str, Any]] = []
        self.reproduction_calls: list[str] = []
        self.score_calls: list[str] = []
        self.reproduced_states: Dict[str, ReproducedArmState] = {}
        self.mutated_arm: str | None = None
        self.authorize_error: Exception | None = None
        self.events: list[str] = []
        self._authorized = False
        self._test_authorized = False

    def authorize(self, expected_execution_head: str) -> Dict[str, Any]:
        if self.authorize_error is not None:
            raise self.authorize_error
        assert len(expected_execution_head) == 40
        self._authorized = True
        return {"expected_execution_head": expected_execution_head, "synthetic": True, "tracked_worktree_clean": True}

    def get_cohort(self, year: int, **kwargs: Any) -> Mapping[str, Any]:
        year_int = int(year)
        if year_int == 2023:
            raise AssertionError("FATAL: D6.1 attempted to access frozen 2023 validation cohort")
        if year_int == 2024 and not self._test_authorized:
            raise GlobalBarrierError("FATAL: 2024 request before runtime TEST authorization")
        if year_int == 2024 and self.summary_path is not None and not self.summary_path.is_file():
            raise AssertionError("FATAL: 2024 request occurred before training summary was frozen")
        self.calls.append({"year": year_int, **kwargs})
        return {"synthetic": True, "year": year_int}

    def reproduce_training_state(
        self, *, arm_id: str, arm_config: Mapping[str, Any], cohort: Any, **_unused: Any
    ) -> Dict[str, Any]:
        assert cohort["year"] == 2022
        self.reproduction_calls.append(arm_id)
        observed = _production_hashes(arm_id)
        if arm_id == self.mutated_arm:
            observed = dict(observed)
            observed["baseline_LR"] = compute_canonical_json_sha256({"mutated": arm_id})
        changed_dict = {"synthetic": arm_id}
        state = ReproducedArmState(
            arm_id=arm_id,
            arm_config=copy.deepcopy(dict(arm_config)),
            train_source_row_digest=_production_digest(arm_id),
            preprocessing_state=copy.deepcopy(self.preprocessing_state),
            preprocessing_state_hash=compute_canonical_json_sha256(self.preprocessing_state),
            frozen_changed_dict=changed_dict,
            transformer=_LiveObject({"transform": arm_id}),
            evaluator=_LiveObject({"evaluator": arm_id}),
            categorical_features=["cat"],
            numerical_features=["num"],
            epsilon_threshold=0.1,
            baseline_scaler=_LiveObject({"baseline_scaler": arm_id}),
            baseline_model=_LiveObject({"baseline_lr": arm_id}),
            fairbias_scaler=_LiveObject({"fairbias_scaler": arm_id}),
            fairbias_model=_LiveObject({"fairbias_lr": arm_id}),
            baseline_feature_order=["cat", "num"],
            fairbias_feature_order=["cat", "num"],
            train_dphi_before_after={"diagnostic_only": True},
            observed_state_hashes=observed,
        )
        self.reproduced_states[arm_id] = state
        return {
            "observed_train_source_row_digest": _production_digest(arm_id),
            "preprocessing_state": copy.deepcopy(self.preprocessing_state),
            "state_hashes": observed,
        }

    def score_temporal_test(
        self, *, arm_id: str, cohort: Any, prediction_threshold: float, **_unused: Any
    ) -> Dict[str, Any]:
        assert cohort["year"] == 2024
        assert prediction_threshold == 0.5
        assert arm_id in self.reproduced_states
        self.score_calls.append(arm_id)
        hashes = copy.deepcopy(self.reproduced_states[arm_id].observed_state_hashes)
        metrics = {
            "utility": {"accuracy": 0.5, "auroc": 0.5, "auprc": 0.5, "balanced_accuracy": 0.5, "f1": 0.5},
            "fairness_gaps": {"demographic_parity_gap": 0.0, "equal_opportunity_gap": 0.0, "fpr_gap": 0.0, "equalized_odds_max_gap": 0.0},
            "group_metrics": [{"group": 1, "n": 2, "selection_rate": 0.5, "tpr": 0.5, "fpr": 0.5, "ppv": 0.5}],
        }
        return {
            "arm_id": arm_id,
            "test_year": 2024,
            "prediction_threshold": 0.5,
            "diagnostic_only": True,
            "test_source_row_digest": compute_canonical_json_sha256({"synthetic_test": arm_id}),
            "test_n": 2,
            "test_outcome_positive_count": 1,
            "test_prevalence": 0.5,
            "test_group_counts": {"1": 2},
            "test_dphi_before_after": {"diagnostic_only": True, "dphi_relearned": False},
            "test_metrics_baseline": copy.deepcopy(metrics),
            "test_metrics_fairbias": copy.deepcopy(metrics),
            "test_comparison": {"diagnostic_only": True},
            "test_group_metrics_baseline": copy.deepcopy(metrics["group_metrics"]),
            "test_group_metrics_fairbias": copy.deepcopy(metrics["group_metrics"]),
            "pre_score_state_hashes": hashes,
            "post_score_state_hashes": hashes,
        }


def _production_manager(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> tuple[NHISD6TemporalTestReleaseManager, _ProductionArchiveLoader, _SyntheticProductionRuntime, pathlib.Path]:
    release_root = tmp_path / "production_releases"
    release_id = "SYNTHETIC_D6_TEMPORAL_TEST"
    target = release_root / release_id
    fake_loader = _ProductionArchiveLoader()
    runtime = _SyntheticProductionRuntime(
        archive_loader.load_preprocessing_state(),
        summary_path=target / "training_state_reproduction_summary.json",
    )
    manager = NHISD6TemporalTestReleaseManager(archive_loader=fake_loader)
    return manager, fake_loader, runtime, release_root


def test_67_production_runtime_is_lazy_and_2023_is_poison() -> None:
    constructed: list[bool] = []

    def poison_adapter(**_kwargs: Any) -> Any:
        constructed.append(True)
        raise AssertionError("FATAL: real NHIS adapter construction is forbidden in D6.1a.1 tests")

    runtime = ProductionD6TemporalRuntime(adapter_factory=poison_adapter)
    with pytest.raises(AssertionError, match="frozen 2023 validation cohort"):
        runtime.get_cohort(2023)
    assert constructed == []


def test_68_production_runtime_head_mismatch_precedes_adapter_or_parquet_access(tmp_path: pathlib.Path) -> None:
    constructed: list[bool] = []

    def poison_adapter(**_kwargs: Any) -> Any:
        constructed.append(True)
        raise AssertionError("adapter must not be constructed")

    runtime = ProductionD6TemporalRuntime(
        features_parquet_path=tmp_path / "never-read.parquet",
        adapter_factory=poison_adapter,
    )
    with pytest.raises(harness.D6HarnessError, match="HEAD mismatch"):
        runtime.authorize("0" * 40)
    assert constructed == []
    assert not (tmp_path / "never-read.parquet").exists()


def test_69_live_object_hashes_are_independent_of_expected_archive_values() -> None:
    state = ReproducedArmState(
        arm_id="D6_ARM_001",
        arm_config=copy.deepcopy(FROZEN_D6_ARMS["D6_ARM_001"]),
        train_source_row_digest="x",
        preprocessing_state={},
        preprocessing_state_hash="x",
        frozen_changed_dict={"a": 1},
        transformer=None,
        evaluator=None,
        categorical_features=["cat"],
        numerical_features=["num"],
        epsilon_threshold=0.1,
        baseline_scaler=_LiveObject({"v": 1}),
        baseline_model=_LiveObject({"v": 1}),
        fairbias_scaler=_LiveObject({"v": 1}),
        fairbias_model=_LiveObject({"v": 1}),
        baseline_feature_order=["cat", "num"],
        fairbias_feature_order=["cat", "num"],
        train_dphi_before_after={},
        observed_state_hashes={},
    )
    before = state.recompute_state_hashes(_StateOnlyRunner)
    state.baseline_model.state["v"] = 2
    after = state.recompute_state_hashes(_StateOnlyRunner)
    assert before["baseline_LR"] != after["baseline_LR"]
    assert EXPECTED_TRAINING_STATE_ANCHORS["D6_ARM_001"]["baseline_LR"] != after["baseline_LR"]


def test_70_production_runtime_reproducer_does_not_read_expected_anchors() -> None:
    source = inspect.getsource(ProductionD6TemporalRuntime.reproduce_training_state)
    assert "expected_state_anchors" not in source
    assert "expected_train_source_row_digest" not in source
    assert "EXPECTED_TRAINING_STATE_ANCHORS" not in source


def test_70a_actual_production_reproducer_and_scorer_run_on_only_synthetic_state(
    archive_loader: FrozenD6ArchiveLoader,
) -> None:
    """Exercise the concrete A--M/2024 methods with fake FairBias and synthetic frames."""
    from nhis_fairbias import d6_temporal_runner as frozen_runner

    constructed: list[Any] = []
    preprocessing_state = archive_loader.load_preprocessing_state()
    columns = [f"cat_{index}" for index in range(18)] + ["num_0", "num_1", "num_2"]
    index = pd.Index(range(100, 108), name="synthetic_row")
    X_train = pd.DataFrame(
        {
            column: np.array([(row + column_index) % 3 for row in range(8)], dtype=float)
            for column_index, column in enumerate(columns)
        },
        index=index,
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], index=index)
    o_train = pd.Series([1, 2, 1, 2, 1, 2, 1, 2], index=index)

    class FakeFitRecord:
        def to_dict(self) -> Dict[str, Any]:
            return copy.deepcopy(preprocessing_state)

    class FakePreprocessor:
        fitted_record = FakeFitRecord()

        @staticmethod
        def get_feature_family_lists(_feature_set: str) -> tuple[list[str], list[str]]:
            return columns[:18], columns[18:]

    class FakeAdapter:
        def __init__(self, **_kwargs: Any) -> None:
            constructed.append(True)
            self.preprocessor = FakePreprocessor()

    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)
            self.transform_n_bins = 10
            self.transform_log_epsilon = 1e-12
            self.transform_x_max = 10.0
            self.phi_threshold = 0.0
            self.transform_poly_exponents = [1]

        def resolved(self) -> "FakeConfig":
            return self

    class FakeEvaluator:
        def __init__(self, *, label_O: list[str], **_kwargs: Any) -> None:
            self.protected = label_O[0]

        def calculate_epsilon(self, X: pd.DataFrame, _o: pd.DataFrame, *, sample_weight: Any, **_kwargs: Any) -> Dict[str, Any]:
            assert sample_weight is None
            return {self.protected: {str(X.columns[0]): 0.0}}

        @staticmethod
        def compute_threshold(_epsilon: Mapping[str, Any]) -> float:
            return 0.0

    class FakeTransform:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        @staticmethod
        def transform_data(X: pd.DataFrame, _changed: Mapping[str, Any], *_args: Any) -> pd.DataFrame:
            return X.copy()

    class FakeMitigation:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    fake_runner = types.SimpleNamespace(
        NHISStudyAdapter=FakeAdapter,
        FairBiasConfig=FakeConfig,
        ALGORITHM_MODE_PAPER_FAITHFUL="tang2024_paper_faithful",
        FairEvaluator=FakeEvaluator,
        FairTransform=FakeTransform,
        FairBiasMitigation=FakeMitigation,
        calculate_nmi_dict=lambda *_args: {},
        MinMaxScaler=frozen_runner.MinMaxScaler,
        get_classifier=frozen_runner.get_classifier,
        pd=pd,
        np=np,
        verify_cohort_alignment_and_uniqueness=frozen_runner.verify_cohort_alignment_and_uniqueness,
        compute_cohort_source_row_digest=frozen_runner.compute_cohort_source_row_digest,
        compute_canonical_json_sha256=frozen_runner.compute_canonical_json_sha256,
        extract_minmax_scaler_state=frozen_runner.extract_minmax_scaler_state,
        extract_logistic_regression_state=frozen_runner.extract_logistic_regression_state,
        evaluate_predictions=frozen_runner.evaluate_predictions,
        compute_evaluation_comparison=frozen_runner.compute_evaluation_comparison,
    )
    runtime = ProductionD6TemporalRuntime(adapter_factory=FakeAdapter, runner_module=fake_runner)
    runtime._authorized = True  # Synthetic-only unit wiring; no real adapter or parquet is available.
    observed = runtime.reproduce_training_state(
        arm_id="D6_ARM_001",
        arm_config=FROZEN_D6_ARMS["D6_ARM_001"],
        cohort=(X_train, y_train, o_train, pd.Series(1.0, index=index), pd.DataFrame(index=index)),
    )
    assert constructed == [True]
    assert observed["preprocessing_state"] == preprocessing_state
    state = runtime.reproduced_states["D6_ARM_001"]
    pre_score_hashes = copy.deepcopy(state.observed_state_hashes)
    score = runtime.score_temporal_test(
        arm_id="D6_ARM_001",
        cohort=(X_train.copy(), y_train.copy(), o_train.copy(), pd.Series(1.0, index=index), pd.DataFrame(index=index)),
        prediction_threshold=0.5,
    )
    assert score["diagnostic_only"] is True
    assert score["pre_score_state_hashes"] == score["post_score_state_hashes"] == pre_score_hashes
    state.baseline_model.coef_[0, 0] += 0.01
    assert state.recompute_state_hashes(fake_runner)["baseline_LR"] != pre_score_hashes["baseline_LR"]


def test_71_production_route_successfully_writes_40_41_43_schema(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _expected, runtime, root = _production_manager(tmp_path, archive_loader)
    result = manager.execute_production_release(
        root,
        release_id="SYNTHETIC_D6_TEMPORAL_TEST",
        expected_execution_head="a" * 40,
        runtime=runtime,
    )
    target = pathlib.Path(result["release_dir"])
    assert result["status"] == "COMPLETE"
    assert [call["year"] for call in runtime.calls] == [2022, 2022, 2022, 2022, 2024, 2024, 2024, 2024]
    assert runtime.reproduction_calls == list(FROZEN_D6_ARM_IDS)
    assert runtime.score_calls == list(FROZEN_D6_ARM_IDS)
    assert len([path for path in target.rglob("*") if path.is_file()]) == 43
    manifest = json.loads((target / "d6_temporal_test_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["artifacts"]) == 41
    for rel_path, metadata in manifest["artifacts"].items():
        artifact = target / rel_path
        assert metadata["sha256"] == compute_file_sha256(artifact)
        assert metadata["size_bytes"] == artifact.stat().st_size
    state = json.loads((target / "release_state.json").read_text(encoding="utf-8"))
    assert state["manifest_sha256"] == compute_file_sha256(target / "d6_temporal_test_manifest.json")
    assert manifest["disclosure_2024"] == TEMPORAL_2024_DISCLOSURE


def test_72_production_route_global_event_order_and_summary_freeze(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _expected, runtime, root = _production_manager(tmp_path, archive_loader)
    result = manager.execute_production_release(
        root,
        release_id="SYNTHETIC_D6_TEMPORAL_TEST",
        expected_execution_head="b" * 40,
        runtime=runtime,
    )
    events = result["events"]
    barrier = events.index("ALL_D6_TRAINING_STATE_ANCHORS_VERIFIED")
    first_2024 = next(index for index, event in enumerate(events) if event.startswith("request_2024_"))
    assert barrier < first_2024
    assert sum(event.startswith("request_2022_") for event in events[:barrier]) == 4
    assert all(call["year"] != 2023 for call in runtime.calls)


def test_73_expected_anchor_mutation_blocks_production_2024(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, expected, runtime, root = _production_manager(tmp_path, archive_loader)
    expected.anchors["D6_ARM_004"]["FairBias_LR"] = "0" * 64
    with pytest.raises(GlobalBarrierError):
        manager.execute_production_release(
            root,
            release_id="SYNTHETIC_D6_TEMPORAL_TEST",
            expected_execution_head="c" * 40,
            runtime=runtime,
        )
    assert [call["year"] for call in runtime.calls] == [2022, 2022, 2022, 2022]
    assert runtime.reproduced_states["D6_ARM_004"].observed_state_hashes["FairBias_LR"] != "0" * 64


def test_74_actual_object_hash_mutation_blocks_production_2024(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _expected, runtime, root = _production_manager(tmp_path, archive_loader)
    runtime.mutated_arm = "D6_ARM_004"
    with pytest.raises(GlobalBarrierError):
        manager.execute_production_release(
            root,
            release_id="SYNTHETIC_D6_TEMPORAL_TEST",
            expected_execution_head="d" * 40,
            runtime=runtime,
        )
    assert [call["year"] for call in runtime.calls] == [2022, 2022, 2022, 2022]


def test_75_production_precondition_failure_creates_no_release(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _expected, runtime, root = _production_manager(tmp_path, archive_loader)
    runtime.authorize_error = harness.D6HarnessError("synthetic reviewed HEAD mismatch")
    with pytest.raises(harness.D6HarnessError, match="reviewed HEAD mismatch"):
        manager.execute_production_release(
            root,
            release_id="SYNTHETIC_D6_TEMPORAL_TEST",
            expected_execution_head="e" * 40,
            runtime=runtime,
        )
    assert not root.exists()
    assert runtime.calls == []


def test_76_production_collision_fails_before_adapter_access(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _expected, runtime, root = _production_manager(tmp_path, archive_loader)
    (root / "SYNTHETIC_D6_TEMPORAL_TEST").mkdir(parents=True)
    with pytest.raises(ReleaseCollisionError):
        manager.execute_production_release(
            root,
            release_id="SYNTHETIC_D6_TEMPORAL_TEST",
            expected_execution_head="f" * 40,
            runtime=runtime,
        )
    assert runtime.calls == []


def test_77_production_release_state_preserves_failure_without_2024(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _expected, runtime, root = _production_manager(tmp_path, archive_loader)
    runtime.mutated_arm = "D6_ARM_002"
    with pytest.raises(GlobalBarrierError):
        manager.execute_production_release(
            root,
            release_id="SYNTHETIC_D6_TEMPORAL_TEST",
            expected_execution_head="1" * 40,
            runtime=runtime,
        )
    state = json.loads((root / "SYNTHETIC_D6_TEMPORAL_TEST" / "release_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert state["2023_request_count"] == 0
    assert state["2024_request_count"] == 0


def _setup_synthetic_git_repo(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str, pathlib.Path, str]:
    """Create a fully valid, self-contained git repository with the 3 target tracked files and parquet."""
    import hashlib

    repo = tmp_path / "synthetic_git_repo"
    repo.mkdir(parents=True)
    (repo / "src" / "nhis_fairbias").mkdir(parents=True)
    (repo / "scripts").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "data" / "processed" / "nhis").mkdir(parents=True)

    release_harness = repo / "src" / "nhis_fairbias" / "d6_temporal_test_release.py"
    cli_script = repo / "scripts" / "run_nhis_d6_temporal_test.py"
    test_file = repo / "tests" / "test_nhis_d6_temporal_test_release.py"
    parquet_file = repo / "data" / "processed" / "nhis" / "nhis_2022_2024_features.parquet"

    release_harness.write_text("# release harness\n", encoding="utf-8")
    cli_script.write_text("# cli script\n", encoding="utf-8")
    test_file.write_text("# test file\n", encoding="utf-8")
    parquet_bytes = b"synthetic_parquet_bytes_for_testing"
    parquet_file.write_bytes(parquet_bytes)
    parquet_sha = hashlib.sha256(parquet_bytes).hexdigest()

    subprocess.run(["git", "init", "-b", "research/nhis-fairbias"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "update-ref", "refs/remotes/origin/research/nhis-fairbias", "HEAD"], cwd=str(repo), check=True, capture_output=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), check=True, capture_output=True, text=True).stdout.strip()
    return repo, head, parquet_file, parquet_sha


def test_78_tracked_worktree_clean_behavior_and_untracked_tolerance(tmp_path: pathlib.Path) -> None:
    # 1. Direct proof on real repository: archive/baseline_v0.3/ is untracked and tolerated
    status_all = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", "archive/baseline_v0.3/"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "archive/baseline_v0.3/" in status_all.stdout
    status_no_untracked = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no", "--", "archive/baseline_v0.3/"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert status_no_untracked.stdout.strip() == ""

    # 2. In an isolated synthetic git repository, verify clean tree, untracked tolerance, and authorization
    repo, head, parquet_path, parquet_sha = _setup_synthetic_git_repo(tmp_path)
    assert git_tracked_worktree_clean(repo) is True

    # Untracked file does NOT cause failure
    untracked = repo / "some_untracked_file.csv"
    untracked.write_text("a,b,c\n", encoding="utf-8")
    assert git_tracked_worktree_clean(repo) is True

    # Substantive authorization succeeds on clean tree with untracked files present
    runtime = ProductionD6TemporalRuntime(
        repo_root=repo,
        features_parquet_path=parquet_path,
        expected_features_sha256=parquet_sha,
    )
    binding = runtime.authorize(head)
    assert binding["tracked_worktree_clean"] is True


def test_79_unstaged_modification_to_release_harness_fails_authorization(tmp_path: pathlib.Path) -> None:
    repo, head, parquet_path, parquet_sha = _setup_synthetic_git_repo(tmp_path)
    target = repo / "src" / "nhis_fairbias" / "d6_temporal_test_release.py"
    target.write_text("# unstaged modification\n", encoding="utf-8")
    assert git_tracked_worktree_clean(repo) is False
    runtime = ProductionD6TemporalRuntime(
        repo_root=repo,
        features_parquet_path=parquet_path,
        expected_features_sha256=parquet_sha,
    )
    with pytest.raises(harness.D6HarnessError, match="clean tracked working tree"):
        runtime.authorize(head)


def test_80_staged_modification_to_release_harness_fails_authorization(tmp_path: pathlib.Path) -> None:
    repo, head, parquet_path, parquet_sha = _setup_synthetic_git_repo(tmp_path)
    target = repo / "src" / "nhis_fairbias" / "d6_temporal_test_release.py"
    target.write_text("# staged modification\n", encoding="utf-8")
    subprocess.run(["git", "add", str(target)], cwd=str(repo), check=True, capture_output=True)
    assert git_tracked_worktree_clean(repo) is False
    runtime = ProductionD6TemporalRuntime(
        repo_root=repo,
        features_parquet_path=parquet_path,
        expected_features_sha256=parquet_sha,
    )
    with pytest.raises(harness.D6HarnessError, match="clean tracked working tree"):
        runtime.authorize(head)


def test_81_unstaged_modification_to_cli_script_fails_authorization(tmp_path: pathlib.Path) -> None:
    repo, head, parquet_path, parquet_sha = _setup_synthetic_git_repo(tmp_path)
    target = repo / "scripts" / "run_nhis_d6_temporal_test.py"
    target.write_text("# unstaged cli modification\n", encoding="utf-8")
    assert git_tracked_worktree_clean(repo) is False
    runtime = ProductionD6TemporalRuntime(
        repo_root=repo,
        features_parquet_path=parquet_path,
        expected_features_sha256=parquet_sha,
    )
    with pytest.raises(harness.D6HarnessError, match="clean tracked working tree"):
        runtime.authorize(head)


def test_82_staged_modification_to_cli_script_fails_authorization(tmp_path: pathlib.Path) -> None:
    repo, head, parquet_path, parquet_sha = _setup_synthetic_git_repo(tmp_path)
    target = repo / "scripts" / "run_nhis_d6_temporal_test.py"
    target.write_text("# staged cli modification\n", encoding="utf-8")
    subprocess.run(["git", "add", str(target)], cwd=str(repo), check=True, capture_output=True)
    assert git_tracked_worktree_clean(repo) is False
    runtime = ProductionD6TemporalRuntime(
        repo_root=repo,
        features_parquet_path=parquet_path,
        expected_features_sha256=parquet_sha,
    )
    with pytest.raises(harness.D6HarnessError, match="clean tracked working tree"):
        runtime.authorize(head)


def test_83_unstaged_modification_to_release_tests_fails_authorization(tmp_path: pathlib.Path) -> None:
    repo, head, parquet_path, parquet_sha = _setup_synthetic_git_repo(tmp_path)
    target = repo / "tests" / "test_nhis_d6_temporal_test_release.py"
    target.write_text("# unstaged test modification\n", encoding="utf-8")
    assert git_tracked_worktree_clean(repo) is False
    runtime = ProductionD6TemporalRuntime(
        repo_root=repo,
        features_parquet_path=parquet_path,
        expected_features_sha256=parquet_sha,
    )
    with pytest.raises(harness.D6HarnessError, match="clean tracked working tree"):
        runtime.authorize(head)


def test_84_staged_modification_to_release_tests_fails_authorization(tmp_path: pathlib.Path) -> None:
    repo, head, parquet_path, parquet_sha = _setup_synthetic_git_repo(tmp_path)
    target = repo / "tests" / "test_nhis_d6_temporal_test_release.py"
    target.write_text("# staged test modification\n", encoding="utf-8")
    subprocess.run(["git", "add", str(target)], cwd=str(repo), check=True, capture_output=True)
    assert git_tracked_worktree_clean(repo) is False
    runtime = ProductionD6TemporalRuntime(
        repo_root=repo,
        features_parquet_path=parquet_path,
        expected_features_sha256=parquet_sha,
    )
    with pytest.raises(harness.D6HarnessError, match="clean tracked working tree"):
        runtime.authorize(head)


def test_85_tracked_modification_fails_closed_before_release_dir_adapter_or_cohorts(
    tmp_path: pathlib.Path,
) -> None:
    repo, head, parquet_path, parquet_sha = _setup_synthetic_git_repo(tmp_path)
    constructed: list[bool] = []

    def poison_adapter(**_kwargs: Any) -> Any:
        constructed.append(True)
        raise AssertionError("Adapter must not be constructed when tracked worktree is dirty")

    runtime = ProductionD6TemporalRuntime(
        repo_root=repo,
        features_parquet_path=parquet_path,
        expected_features_sha256=parquet_sha,
        adapter_factory=poison_adapter,
    )
    fake_loader = _ProductionArchiveLoader()
    manager = NHISD6TemporalTestReleaseManager(archive_loader=fake_loader, repo_root=repo)
    release_root = tmp_path / "substantive_releases"
    target_file = repo / "scripts" / "run_nhis_d6_temporal_test.py"
    target_file.write_text("# mutation\n", encoding="utf-8")

    with pytest.raises(harness.D6HarnessError, match="clean tracked working tree"):
        manager.execute_production_release(
            release_root,
            release_id="SHOULD_NOT_BE_CREATED",
            expected_execution_head=head,
            runtime=runtime,
        )
    assert not release_root.exists()
    assert not (release_root / "SHOULD_NOT_BE_CREATED").exists()
    assert constructed == []
    assert runtime.reproduced_states == {}


def test_86_runtime_get_cohort_2024_fails_when_authorized_head_but_no_global_barrier() -> None:
    calls: list[int] = []

    class FakeAdapter:
        def get_cohort(self, year: int, **_kwargs: Any) -> Any:
            calls.append(year)
            return {"synthetic": True, "year": year}

    runtime = ProductionD6TemporalRuntime(repo_root=_REPO_ROOT, adapter_factory=FakeAdapter)
    runtime._authorized = True
    assert runtime.is_test_authorized is False
    with pytest.raises(GlobalBarrierError, match="runtime TEST gate is closed"):
        runtime.get_cohort(2024)
    assert calls == []


def test_87_runtime_authorize_test_access_fails_on_incomplete_3_of_4_summary(
    archive_loader: FrozenD6ArchiveLoader,
) -> None:
    summary = _synthetic_summary(archive_loader)
    summary["global"]["cohort_digest_match_count"] = 3
    summary["global"]["cohort_digest_matches"] = "3/4"
    summary["arms"]["D6_ARM_004"]["digest_match"] = False
    runtime = ProductionD6TemporalRuntime(repo_root=_REPO_ROOT)
    runtime._authorized = True
    assert runtime.is_test_authorized is False
    with pytest.raises(GlobalBarrierError, match="Global 2022 training-state reproduction barrier not satisfied"):
        runtime.authorize_test_access(summary, archive_loader.expected_state_anchors())
    assert runtime.is_test_authorized is False
    with pytest.raises(GlobalBarrierError, match="runtime TEST gate is closed"):
        runtime.get_cohort(2024)


def test_88_runtime_authorize_test_access_fails_on_incomplete_19_of_20_summary(
    archive_loader: FrozenD6ArchiveLoader,
) -> None:
    summary = _synthetic_summary(archive_loader)
    summary["global"]["training_state_anchor_match_count"] = 19
    summary["global"]["training_state_anchor_matches"] = "19/20"
    summary["arms"]["D6_ARM_004"]["FairBias_LR"]["match"] = False
    runtime = ProductionD6TemporalRuntime(repo_root=_REPO_ROOT)
    runtime._authorized = True
    assert runtime.is_test_authorized is False
    with pytest.raises(GlobalBarrierError, match="Global 2022 training-state reproduction barrier not satisfied"):
        runtime.authorize_test_access(summary, archive_loader.expected_state_anchors())
    assert runtime.is_test_authorized is False
    with pytest.raises(GlobalBarrierError, match="runtime TEST gate is closed"):
        runtime.get_cohort(2024)


def test_89_runtime_authorize_test_access_fails_on_forged_global_boolean(
    archive_loader: FrozenD6ArchiveLoader,
) -> None:
    summary = _synthetic_summary(archive_loader)
    # Global counts look verified, but per-arm observed hash is forged/mismatched
    summary["arms"]["D6_ARM_003"]["FairBias_LR"]["observed_sha256"] = "0" * 64
    runtime = ProductionD6TemporalRuntime(repo_root=_REPO_ROOT)
    runtime._authorized = True
    assert runtime.is_test_authorized is False
    with pytest.raises(GlobalBarrierError, match="Global 2022 training-state reproduction barrier not satisfied"):
        runtime.authorize_test_access(summary, archive_loader.expected_state_anchors())
    assert runtime.is_test_authorized is False
    with pytest.raises(GlobalBarrierError, match="runtime TEST gate is closed"):
        runtime.get_cohort(2024)


def test_90_runtime_authorize_test_access_opens_test_gate_on_valid_full_summary(
    archive_loader: FrozenD6ArchiveLoader,
) -> None:
    summary = _synthetic_summary(archive_loader)
    calls: list[int] = []

    class FakeAdapter:
        def get_cohort(self, year: int, **_kwargs: Any) -> Any:
            calls.append(year)
            return {"synthetic_cohort": True, "year": year}

    runtime = ProductionD6TemporalRuntime(repo_root=_REPO_ROOT, adapter_factory=FakeAdapter)
    runtime._authorized = True
    assert runtime.is_test_authorized is False
    opened = runtime.authorize_test_access(summary, archive_loader.expected_state_anchors())
    assert opened is True
    assert runtime.is_test_authorized is True
    assert "open_production_test_gate" in runtime.events
    cohort = runtime.get_cohort(2024)
    assert cohort == {"synthetic_cohort": True, "year": 2024}
    assert calls == [2024]
    assert "production_get_cohort_2024" in runtime.events


def test_91_canonical_production_manager_produces_exact_two_phase_sequence(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _expected, runtime, root = _production_manager(tmp_path, archive_loader)
    result = manager.execute_production_release(
        root,
        release_id="SYNTHETIC_D6_TEMPORAL_TEST",
        expected_execution_head="a" * 40,
        runtime=runtime,
    )
    events = result["events"]
    barrier = events.index("ALL_D6_TRAINING_STATE_ANCHORS_VERIFIED")
    first_2024 = next(index for index, event in enumerate(events) if event.startswith("request_2024_"))
    assert barrier < first_2024
    assert sum(event.startswith("request_2022_") for event in events[:barrier]) == 4
    assert sum(event.startswith("request_2024_") for event in events[barrier:]) == 4
    assert runtime.is_test_authorized is True
    assert [call["year"] for call in runtime.calls] == [2022, 2022, 2022, 2022, 2024, 2024, 2024, 2024]
    assert all(call["year"] != 2023 for call in runtime.calls)
    assert result["status"] == "COMPLETE"


def test_92_late_arm004_mismatch_prevents_runtime_test_gate_opening_and_zero_2024_requests(
    tmp_path: pathlib.Path, archive_loader: FrozenD6ArchiveLoader
) -> None:
    manager, _expected, runtime, root = _production_manager(tmp_path, archive_loader)
    runtime.mutated_arm = "D6_ARM_004"
    with pytest.raises(GlobalBarrierError):
        manager.execute_production_release(
            root,
            release_id="SYNTHETIC_D6_TEMPORAL_TEST",
            expected_execution_head="d" * 40,
            runtime=runtime,
        )
    assert runtime.is_test_authorized is False
    assert [call["year"] for call in runtime.calls] == [2022, 2022, 2022, 2022]
    state = json.loads((root / "SYNTHETIC_D6_TEMPORAL_TEST" / "release_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert state["2023_request_count"] == 0
    assert state["2024_request_count"] == 0
    assert state["test_year_requested"] is False
    assert state["test_year_evaluated"] is False
