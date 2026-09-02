"""Dynamic Gate D6.1a tests.

The tests use the real immutable D6.0c JSON/CSV archive for static verification,
but every cohort request is handled by an in-memory synthetic adapter.  No real
NHIS cohort, model, FairBias implementation, or canonical TEST release is used.
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
from typing import Any, Dict, Mapping

import pytest

import nhis_fairbias.d6_temporal_test_release as harness
from nhis_fairbias.d6_temporal_test_release import (
    AnchorMismatchError,
    D6TemporalTestEvaluator,
    D6TrainingStateReproducer,
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
        "GLOBAL TRAINING-STATE REPRODUCTION BARRIER:\nENFORCED",
        "REAL 2022 STATE REPRODUCTION EXECUTED:\nFALSE",
        "2024 TEST EVALUATED:\nFALSE",
        "D6.1a AUDIT:\nPASS",
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
    assert "reserved for D6.1b" in result.stderr


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
