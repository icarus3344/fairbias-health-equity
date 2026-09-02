"""Gate D6.1a: isolated temporal TEST release harness.

This module deliberately separates archive inspection, future 2022 training-state
reproduction, and future 2024 evaluation.  The default audit path is standard
library only: it reads the immutable D6 TRAIN/VALIDATION archive and Git metadata,
but never constructs an NHIS adapter, requests a cohort, fits a model, or creates
a TEST release.

The substantive methods are dependency-injected so synthetic tests can exercise the
global barrier without touching real NHIS data.  A production D6.1b implementation
must provide the reproduction and scoring callables explicitly; this module does
not silently fall back to a real-data execution path.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import pathlib
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

D6_GATE = "D6.1a"
D6_ARCHIVE_GATE = "D6.0c"
D6_TRAIN_VAL_RELEASE_ID = "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
D6_TRAIN_VAL_TAG = "nhis-d6-temporal-train-val-v1"
D6_TRAIN_VAL_TAG_OBJECT = "e23941edf2085c84290868f69289d30af62b7e95"
D6_TRAIN_VAL_ARCHIVE_COMMIT = "bdf154c541d365b216a732bc3a58415af41cdcc7"
D6_SCIENTIFIC_EXECUTION_COMMIT = "fa8eb6096de995463b437c4e74f0e3da6a2c196d"
D6_TRAIN_VAL_MANIFEST_SHA256 = (
    "773d43b2d893232eb4cbba9dfcdcc639ecdc2fbc8efb2e353f7d2a68514159c4"
)
D6_TRAIN_VAL_LEDGER_SHA256 = (
    "d40a8dce1f117961ee866cdfc2a12879ba7ce3b939695797be80b4a2acc47ca6"
)
D6_TRAIN_VAL_ARCHIVE_DIR = (
    _REPO_ROOT / "docs" / "releases" / D6_TRAIN_VAL_RELEASE_ID
)

TEMPORAL_TRAIN_YEAR = 2022
TEMPORAL_VALIDATION_YEAR = 2023
TEMPORAL_TEST_YEAR = 2024
TEMPORAL_ROLES = {
    TEMPORAL_TRAIN_YEAR: "development_train",
    TEMPORAL_VALIDATION_YEAR: "development_validation",
    TEMPORAL_TEST_YEAR: "frozen_test",
}

PREDICTION_THRESHOLD = 0.5
DEFAULT_RANDOM_SEED = 0
CLASSIFIER_TYPE = "LogisticRegression"
CLASSIFIER_SOLVER = "lbfgs"
CLASSIFIER_MAX_ITER = 1000
FAIRBIAS_ALGORITHM_MODE = "tang2024_paper_faithful"
POWER_SEQUENCE_POLICY = "official_stream"
POWER_REVISIT_POLICY = "restart"
FAILED_ATTRIBUTE_MODE = "stop"

PREPROCESSING_STATE_SHA256 = (
    "f106967a8ed9bf46ff1c3ff009e40763280fa1dbc78983d3a479a1508fd83db9"
)

TEMPORAL_2024_DISCLOSURE = (
    "D6 is a predeclared temporal robustness analysis using 2022 for development training, "
    "2023 for temporal validation, and 2024 for forward-time evaluation. However, 2024 respondents "
    "were previously included in the pooled D4/D5 analyses, so the 2024 partition is not a "
    "globally untouched primary holdout. D6-specific transformations and models were nevertheless "
    "frozen before D6-specific 2024 evaluation."
)

EXCLUDED_DISABILITY_COMPONENTS: Tuple[str, ...] = (
    "visiondf_a",
    "hearingdf_a",
    "diff_a",
    "comdiff_a",
    "uppslfcr_a",
    "cogmemdff_a",
)

FROZEN_D6_ARMS: Dict[str, Dict[str, Any]] = {
    "D6_ARM_001": {
        "arm_id": "D6_ARM_001",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "SEX_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_groups": [1, 2],
        "expected_group_count": 2,
        "expected_pair_count": 1,
    },
    "D6_ARM_002": {
        "arm_id": "D6_ARM_002",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "HISPALLP_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_groups": [1, 2, 3, 4, 5, 6, 7],
        "expected_group_count": 7,
        "expected_pair_count": 21,
    },
    "D6_ARM_003": {
        "arm_id": "D6_ARM_003",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "DISAB3_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_groups": [1, 2],
        "expected_group_count": 2,
        "expected_pair_count": 1,
    },
    "D6_ARM_004": {
        "arm_id": "D6_ARM_004",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "DISAB3_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "exclude_disability_components",
        "expected_predictors": 15,
        "expected_groups": [1, 2],
        "expected_group_count": 2,
        "expected_pair_count": 1,
    },
}
FROZEN_D6_ARM_IDS: Tuple[str, ...] = tuple(FROZEN_D6_ARMS)

EXPECTED_TRAIN_SOURCE_ROW_DIGESTS: Dict[str, str] = {
    "D6_ARM_001": "fbf4c5b0cadd74b7dfa565082e5d6577c8ec75fbdbf5abbdc8e48d712896ff6e",
    "D6_ARM_002": "30a71c454f54871cde9c03e34eabe936ce0c1a55a789e31017d9b31e81f23aa4",
    "D6_ARM_003": "e54056b34627b15440af95bcdb8e3f6ca1f282a909d3a59a3f47d62e7959185d",
    "D6_ARM_004": "e54056b34627b15440af95bcdb8e3f6ca1f282a909d3a59a3f47d62e7959185d",
}

# Names here are the stable logical labels exposed in the D6.1 reproduction summary.
# The archive ledger uses the corresponding snake-case names for the LR entries.
EXPECTED_TRAINING_STATE_ANCHORS: Dict[str, Dict[str, str]] = {
    "D6_ARM_001": {
        "changed_dict": "40511e6c0d55b0ffdcef534f6b0eb120a16cdfe24ca75bf135ce1ce803f85f79",
        "baseline_scaler": "75bfcaae59a34c0f43127cb73473730971fb3e37c660ebc298044a286e50f0ad",
        "baseline_LR": "3a260ed45d5232ff14d2b2070de2196d91380e0b7ea5005972436a28138f46df",
        "FairBias_scaler": "1fc71b9805939e5627fcbe081c1b23f3fc14373c919cdbe6de557e188b26cfb6",
        "FairBias_LR": "cdfe73f72d0046f0e0068d0047b1b6ef1d934b334d287e208690c176ce95764e",
    },
    "D6_ARM_002": {
        "changed_dict": "4c0bbba5d40022d61119f5ef16db6aa46f0b40bc89132777a3e7b877e42bbe6e",
        "baseline_scaler": "75bfcaae59a34c0f43127cb73473730971fb3e37c660ebc298044a286e50f0ad",
        "baseline_LR": "adca7c506795db6a15d7293be0363e546c79f5c85523cebd20f979c2287ed2f8",
        "FairBias_scaler": "c89d08e57c67bc1fbd9d5c70671d1eb71ca6b51798e7173d08413a69c1aac5dc",
        "FairBias_LR": "6d03134ce077895b8ffb4914bd57f7d73b1b4c94ed09f99456e7cc266bab05c3",
    },
    "D6_ARM_003": {
        "changed_dict": "95ce9da442e66adbd8e1d5e0deab32d831a7a171922f5d6a8cda21a43e9c16af",
        "baseline_scaler": "75bfcaae59a34c0f43127cb73473730971fb3e37c660ebc298044a286e50f0ad",
        "baseline_LR": "ba6914f12b8c1e653c161791199e13392597911de31d19ff2e42f50c26d3f58d",
        "FairBias_scaler": "3fad8127699fa2da2b6fad79d883dc02f45ba1e6494ee896cdcbd93dcc3a3b95",
        "FairBias_LR": "48c0dced8358bd123bd41aba265bf728eae7d2fa32aaa0005ecedc450859f6f5",
    },
    "D6_ARM_004": {
        "changed_dict": "ff0a2fb81596b598b13331803b03a8a1572b88c1a96d53d352807cab93eeb568",
        "baseline_scaler": "f59d3f21087ae95c2accbc70e0edf9a5e055abe97e2a69d5279aa691688f07a1",
        "baseline_LR": "d050dc2a3ba59b2ea1f1767342dbb468680175cc182e94d2143d0ea01563bc3d",
        "FairBias_scaler": "42236586cddc6dd9bf36d46d3209ef40e830c03493a3f041bb429aedc5d4856c",
        "FairBias_LR": "cb0e9a360d68b1705bc1b3b54d508afe80803648fdb22aafc468e245b6045bca",
    },
}

_LEDGER_ANCHOR_KEY_MAP: Dict[str, str] = {
    "changed_dict": "changed_dict",
    "baseline_scaler": "baseline_scaler",
    "baseline_LR": "baseline_logistic_regression",
    "FairBias_scaler": "fairbias_scaler",
    "FairBias_LR": "fairbias_logistic_regression",
}

FROZEN_D6_SCIENTIFIC_PATHS: Tuple[str, ...] = (
    "src/fairbias/**",
    "src/nhis_fairbias/adapter.py",
    "src/nhis_fairbias/preprocessing.py",
    "src/nhis_fairbias/evaluation.py",
    "src/nhis_fairbias/features.py",
    "src/nhis_fairbias/schema.py",
    "src/nhis_fairbias/d6_temporal_runner.py",
)
FROZEN_D6_ARCHIVE_RELATIVE_PATH = "docs/releases/" + D6_TRAIN_VAL_RELEASE_ID

REQUIRED_PER_ARM_ARCHIVE_FILES: Tuple[str, ...] = (
    "arm_config.json",
    "final_changed_dict.json",
    "input_provenance.json",
    "train_dphi_before_after.json",
    "train_fairbias_trace.json",
    "validation_comparison.json",
    "validation_dphi_before_after.json",
    "validation_group_metrics_baseline.csv",
    "validation_group_metrics_fairbias.csv",
    "validation_metrics_baseline.json",
    "validation_metrics_fairbias.json",
)
REQUIRED_TOP_LEVEL_ARCHIVE_FILES: Tuple[str, ...] = (
    "d6_temporal_train_val_manifest.json",
    "preprocessing_provenance.json",
    "release_state.json",
)

FUTURE_PER_ARM_TEST_ARTIFACTS: Tuple[str, ...] = (
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
FUTURE_PER_ARM_ARTIFACT_COUNT = 10
FUTURE_MANIFEST_TRACKED_ARTIFACT_COUNT = 41
FUTURE_COMPLETE_FILE_COUNT = 43


class D6HarnessError(RuntimeError):
    """Base error for fail-closed D6.1 harness behavior."""


class FrozenArchiveIntegrityError(D6HarnessError):
    """Raised when the immutable D6 TRAIN/VALIDATION archive is invalid."""


class AnchorMismatchError(D6HarnessError):
    """Raised when a 2022 reproduction does not exactly match frozen anchors."""


class GlobalBarrierError(D6HarnessError):
    """Raised when the all-four-arm reproduction barrier is not satisfied."""


class ReleaseCollisionError(D6HarnessError):
    """Raised when a synthetic release directory already exists."""


def compute_canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-serializable logical state using the gate's canonical encoding."""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FrozenArchiveIntegrityError(
            f"State is not canonical-JSON serializable without nonfinite values: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def compute_file_sha256(path: pathlib.Path) -> str:
    """Compute a raw-byte SHA-256 without importing scientific/data libraries."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FrozenArchiveIntegrityError(f"Missing required archive file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FrozenArchiveIntegrityError(f"Invalid JSON in archive file {path}: {exc}") from exc


def _finite_numeric_paths(value: Any, path: str = "$") -> List[str]:
    """Return locations containing JSON nonfinite floats, if a parser permits them."""
    import math

    found: List[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        found.append(path)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            found.extend(_finite_numeric_paths(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_finite_numeric_paths(child, f"{path}[{index}]"))
    return found


def _git_revision(repo_root: pathlib.Path, revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", revision],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise FrozenArchiveIntegrityError(
            f"Git revision verification failed for {revision!r}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _git_diff_clean(repo_root: pathlib.Path, base: str, target: str, paths: Sequence[str]) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", f"{base}..{target}", "--", *paths],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise FrozenArchiveIntegrityError(
        f"Git boundary check failed: {result.stderr.strip()}"
    )


def _git_worktree_clean(repo_root: pathlib.Path, paths: Sequence[str]) -> bool:
    """Require no staged, unstaged, or untracked changes under frozen paths."""
    for diff_args in (
        ["git", "diff", "--quiet", "--", *paths],
        ["git", "diff", "--cached", "--quiet", "--", *paths],
    ):
        result = subprocess.run(
            diff_args,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 1:
            return False
        if result.returncode != 0:
            raise FrozenArchiveIntegrityError(
                f"Git worktree boundary check failed: {result.stderr.strip()}"
            )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *paths],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        raise FrozenArchiveIntegrityError(
            f"Git status boundary check failed: {status.stderr.strip()}"
        )
    return not bool(status.stdout.strip())


class FrozenD6ArchiveLoader:
    """Read-only loader and verifier for the immutable D6.0c archive.

    This class intentionally has no adapter field and no cohort-access method.
    """

    def __init__(
        self,
        archive_dir: Optional[pathlib.Path | str] = None,
        *,
        repo_root: Optional[pathlib.Path | str] = None,
        enforce_git_identity: bool = True,
    ) -> None:
        self.repo_root = pathlib.Path(repo_root).resolve() if repo_root else _REPO_ROOT
        self.archive_dir = (
            pathlib.Path(archive_dir).resolve() if archive_dir else D6_TRAIN_VAL_ARCHIVE_DIR.resolve()
        )
        self.enforce_git_identity = bool(enforce_git_identity)
        self._last_audit: Optional[Dict[str, Any]] = None

    @property
    def archive_manifest_path(self) -> pathlib.Path:
        return self.archive_dir / "d6_temporal_train_val_manifest.json"

    @property
    def archive_ledger_path(self) -> pathlib.Path:
        return self.archive_dir / "archive_ledger.json"

    def verify_git_identity(self) -> Dict[str, Any]:
        """Verify the annotated D6 tag and unchanged D6 scientific/archive boundary."""
        if not self.enforce_git_identity:
            return {
                "tag_object": "BYPASS",
                "tag_commit": "BYPASS",
                "archive_boundary_clean": "BYPASS",
                "scientific_boundary_clean": "BYPASS",
            }
        tag_object = _git_revision(self.repo_root, D6_TRAIN_VAL_TAG)
        if tag_object != D6_TRAIN_VAL_TAG_OBJECT:
            raise FrozenArchiveIntegrityError(
                f"D6 tag object mismatch: expected {D6_TRAIN_VAL_TAG_OBJECT}, got {tag_object}"
            )
        tag_commit = _git_revision(self.repo_root, f"{D6_TRAIN_VAL_TAG}^{{commit}}")
        if tag_commit != D6_TRAIN_VAL_ARCHIVE_COMMIT:
            raise FrozenArchiveIntegrityError(
                f"D6 tag commit mismatch: expected {D6_TRAIN_VAL_ARCHIVE_COMMIT}, got {tag_commit}"
            )
        target = _git_revision(self.repo_root, "HEAD")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", D6_TRAIN_VAL_ARCHIVE_COMMIT, target],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if ancestor.returncode != 0:
            raise FrozenArchiveIntegrityError(
                f"D6 archive commit is not an ancestor of current HEAD {target}"
            )
        archive_clean = _git_diff_clean(
            self.repo_root,
            D6_TRAIN_VAL_ARCHIVE_COMMIT,
            target,
            [FROZEN_D6_ARCHIVE_RELATIVE_PATH],
        )
        archive_worktree_clean = _git_worktree_clean(
            self.repo_root,
            [FROZEN_D6_ARCHIVE_RELATIVE_PATH],
        )
        scientific_clean = _git_diff_clean(
            self.repo_root,
            D6_TRAIN_VAL_ARCHIVE_COMMIT,
            target,
            FROZEN_D6_SCIENTIFIC_PATHS,
        )
        scientific_worktree_clean = _git_worktree_clean(
            self.repo_root,
            FROZEN_D6_SCIENTIFIC_PATHS,
        )
        if not archive_clean or not archive_worktree_clean:
            raise FrozenArchiveIntegrityError(
                "The frozen D6 TRAIN/VALIDATION archive changed after D6.0c."
            )
        if not scientific_clean or not scientific_worktree_clean:
            raise FrozenArchiveIntegrityError(
                "A frozen D6 scientific execution path changed after D6.0c."
            )
        return {
            "tag_object": tag_object,
            "tag_commit": tag_commit,
            "current_head": target,
            "archive_boundary_clean": True,
            "scientific_boundary_clean": True,
            "archive_worktree_clean": True,
            "scientific_worktree_clean": True,
            "scientific_paths_checked": list(FROZEN_D6_SCIENTIFIC_PATHS),
        }

    def _expected_canonical_paths(self) -> set[str]:
        paths = set(REQUIRED_TOP_LEVEL_ARCHIVE_FILES)
        for arm_id in FROZEN_D6_ARM_IDS:
            paths.update(f"{arm_id}/{name}" for name in REQUIRED_PER_ARM_ARCHIVE_FILES)
        return paths

    def load_arm_config(self, arm_id: str) -> Dict[str, Any]:
        if arm_id not in FROZEN_D6_ARMS:
            raise KeyError(f"Unknown frozen D6 arm: {arm_id}")
        return copy.deepcopy(_read_json(self.archive_dir / arm_id / "arm_config.json"))

    def load_preprocessing_state(self) -> Dict[str, Any]:
        state = _read_json(self.archive_dir / "preprocessing_provenance.json")
        if not isinstance(state, dict):
            raise FrozenArchiveIntegrityError("Archived preprocessing state must be a JSON object")
        return copy.deepcopy(state)

    def expected_train_digests(self) -> Dict[str, str]:
        return copy.deepcopy(EXPECTED_TRAIN_SOURCE_ROW_DIGESTS)

    def expected_state_anchors(self) -> Dict[str, Dict[str, str]]:
        return copy.deepcopy(EXPECTED_TRAINING_STATE_ANCHORS)

    def verify(self) -> Dict[str, Any]:
        """Verify archive structure, hashes, state anchors, and immutable provenance."""
        if not self.archive_dir.is_dir():
            raise FrozenArchiveIntegrityError(
                f"Frozen D6 archive directory not found: {self.archive_dir}"
            )

        git_info = self.verify_git_identity()
        actual_files = sorted(
            path.relative_to(self.archive_dir).as_posix()
            for path in self.archive_dir.rglob("*")
            if path.is_file()
        )
        canonical_files = [rel for rel in actual_files if rel != "archive_ledger.json"]
        expected_paths = self._expected_canonical_paths()
        if len(canonical_files) != 47:
            raise FrozenArchiveIntegrityError(
                f"Frozen archive has {len(canonical_files)} canonical files; expected 47"
            )
        if set(canonical_files) != expected_paths:
            missing = sorted(expected_paths - set(canonical_files))
            extra = sorted(set(canonical_files) - expected_paths)
            raise FrozenArchiveIntegrityError(
                f"Frozen archive path set mismatch; missing={missing}, extra={extra}"
            )
        if actual_files.count("archive_ledger.json") != 1 or len(actual_files) != 48:
            raise FrozenArchiveIntegrityError(
                "Frozen archive must contain exactly 47 canonical files plus archive_ledger.json"
            )

        manifest_sha = compute_file_sha256(self.archive_manifest_path)
        if manifest_sha != D6_TRAIN_VAL_MANIFEST_SHA256:
            raise FrozenArchiveIntegrityError(
                f"D6 manifest SHA mismatch: expected {D6_TRAIN_VAL_MANIFEST_SHA256}, got {manifest_sha}"
            )
        ledger_sha = compute_file_sha256(self.archive_ledger_path)
        if ledger_sha != D6_TRAIN_VAL_LEDGER_SHA256:
            raise FrozenArchiveIntegrityError(
                f"D6 archive ledger SHA mismatch: expected {D6_TRAIN_VAL_LEDGER_SHA256}, got {ledger_sha}"
            )

        state = _read_json(self.archive_dir / "release_state.json")
        if state.get("status") != "COMPLETE":
            raise FrozenArchiveIntegrityError(
                f"D6 release state is {state.get('status')!r}; expected 'COMPLETE'"
            )
        if state.get("release_id") != D6_TRAIN_VAL_RELEASE_ID:
            raise FrozenArchiveIntegrityError("D6 release_state release_id mismatch")
        if state.get("manifest_sha256") != D6_TRAIN_VAL_MANIFEST_SHA256:
            raise FrozenArchiveIntegrityError("D6 release_state manifest SHA mismatch")

        manifest = _read_json(self.archive_manifest_path)
        required_manifest_values = {
            "gate": "D6 temporal TRAIN/VALIDATION",
            "release_id": D6_TRAIN_VAL_RELEASE_ID,
            "status": "COMPLETE",
            "train_year": 2022,
            "validation_year": 2023,
            "future_test_year": 2024,
            "temporal_robustness_analysis": True,
            "repeated_cross_sectional": True,
            "longitudinal": False,
            "causal_analysis": False,
            "primary_analysis": False,
            "replaces_d4_primary": False,
            "survey_weighted_geometry": False,
            "classifier_weighted": False,
            "evaluation_weighted": False,
            "validation_used_for_selection": False,
            "test_year_requested": False,
            "test_year_evaluated": False,
        }
        for key, expected in required_manifest_values.items():
            if manifest.get(key) != expected:
                raise FrozenArchiveIntegrityError(
                    f"D6 manifest field {key!r} mismatch: "
                    f"expected {expected!r}, got {manifest.get(key)!r}"
                )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or len(artifacts) != 45:
            raise FrozenArchiveIntegrityError("D6 manifest must contain exactly 45 artifacts")
        expected_manifest_paths = {
            f"{arm_id}/{name}"
            for arm_id in FROZEN_D6_ARM_IDS
            for name in REQUIRED_PER_ARM_ARCHIVE_FILES
        }
        expected_manifest_paths.add("preprocessing_provenance.json")
        if set(artifacts) != expected_manifest_paths:
            raise FrozenArchiveIntegrityError("D6 manifest artifact path set mismatch")
        bad_artifacts: List[str] = []
        for rel_path, metadata in artifacts.items():
            target = self.archive_dir / rel_path
            if not target.is_file():
                bad_artifacts.append(rel_path)
                continue
            if metadata.get("sha256") != compute_file_sha256(target):
                bad_artifacts.append(rel_path)
            if metadata.get("size_bytes") != target.stat().st_size:
                bad_artifacts.append(rel_path)
        if bad_artifacts:
            raise FrozenArchiveIntegrityError(
                f"D6 manifest artifact mismatch: {sorted(set(bad_artifacts))}"
            )

        ledger = _read_json(self.archive_ledger_path)
        required_ledger_values = {
            "archive_gate": D6_ARCHIVE_GATE,
            "release_id": D6_TRAIN_VAL_RELEASE_ID,
            "raw_canonical_file_count": 47,
            "manifest_tracked_artifact_count": 45,
            "per_arm_artifact_count": 44,
            "execution_commit": D6_SCIENTIFIC_EXECUTION_COMMIT,
            "train_year": 2022,
            "validation_year": 2023,
            "future_test_year": 2024,
            "repeated_cross_sectional": True,
            "longitudinal": False,
            "causal_analysis": False,
            "survey_weighted_geometry": False,
            "classifier_weighted": False,
            "evaluation_weighted": False,
            "canonical_execution_count": 1,
            "temporal_train_validation_rerun": False,
            "validation_driven_tuning": False,
            "test_year_requested": False,
            "test_year_evaluated": False,
        }
        for key, expected in required_ledger_values.items():
            if ledger.get(key) != expected:
                raise FrozenArchiveIntegrityError(
                    f"D6 ledger field {key!r} mismatch: expected {expected!r}, got {ledger.get(key)!r}"
                )
        ledger_entries = ledger.get("canonical_file_hashes")
        if not isinstance(ledger_entries, list) or len(ledger_entries) != 47:
            raise FrozenArchiveIntegrityError("D6 ledger must list exactly 47 canonical file hashes")
        ledger_paths = [item.get("relative_path") for item in ledger_entries]
        if ledger_paths != sorted(ledger_paths) or set(ledger_paths) != set(canonical_files):
            raise FrozenArchiveIntegrityError("D6 ledger canonical file paths are not the exact sorted set")
        ledger_hashes = {item["relative_path"]: item.get("sha256") for item in ledger_entries}
        ledger_mismatches = [
            rel for rel in canonical_files
            if ledger_hashes.get(rel) != compute_file_sha256(self.archive_dir / rel)
        ]
        if ledger_mismatches:
            raise FrozenArchiveIntegrityError(
                f"D6 ledger canonical file hash mismatch: {ledger_mismatches}"
            )

        warning = ledger.get("numerical_warning_reconciliation", {})
        if warning.get("reconciliation_verdict") != "PASS":
            raise FrozenArchiveIntegrityError("D6 numerical-warning reconciliation is not PASS")
        if warning.get("unexpected_nonfinite_frozen_state_count") != 0:
            raise FrozenArchiveIntegrityError("D6 ledger records unexpected nonfinite frozen state values")
        if warning.get("expected_undefined_group_metrics_count") != 4:
            raise FrozenArchiveIntegrityError("D6 ledger expected undefined group metric count mismatch")

        anchors = ledger.get("model_state_anchors")
        normalized_anchors = {
            arm_id: {
                public_name: anchors.get(arm_id, {}).get(ledger_name)
                for public_name, ledger_name in _LEDGER_ANCHOR_KEY_MAP.items()
            }
            for arm_id in FROZEN_D6_ARM_IDS
        }
        if normalized_anchors != EXPECTED_TRAINING_STATE_ANCHORS:
            raise FrozenArchiveIntegrityError("D6 ledger model-state anchors do not match frozen constants")
        digests = ledger.get("cohort_source_row_digests")
        if not isinstance(digests, dict):
            raise FrozenArchiveIntegrityError("D6 ledger cohort digest record is missing")
        for arm_id, expected_train_digest in EXPECTED_TRAIN_SOURCE_ROW_DIGESTS.items():
            if digests.get(arm_id, {}).get("2022") != expected_train_digest:
                raise FrozenArchiveIntegrityError(f"D6 train digest mismatch for {arm_id}")
        # The 2023 values are archived evidence, not a future cohort request.  They
        # are checked for presence and nonempty SHA shape without treating them as
        # a D6.1 input to materialize.
        for arm_id in FROZEN_D6_ARM_IDS:
            value = digests.get(arm_id, {}).get("2023")
            if not isinstance(value, str) or len(value) != 64:
                raise FrozenArchiveIntegrityError(f"D6 archived validation digest missing for {arm_id}")

        preprocessor_state = self.load_preprocessing_state()
        observed_preprocessing_hash = compute_canonical_json_sha256(preprocessor_state)
        if observed_preprocessing_hash != PREPROCESSING_STATE_SHA256:
            raise FrozenArchiveIntegrityError(
                "D6 preprocessing provenance logical-state hash mismatch"
            )
        if preprocessor_state.get("fit_year") != 2022 or preprocessor_state.get("fit_study_role") != "development_train":
            raise FrozenArchiveIntegrityError("D6 preprocessing provenance fit role/year mismatch")
        if preprocessor_state.get("row_count") != 27651:
            raise FrozenArchiveIntegrityError("D6 preprocessing provenance row_count mismatch")
        nonfinite = _finite_numeric_paths(preprocessor_state)
        if nonfinite:
            raise FrozenArchiveIntegrityError(f"D6 preprocessing state contains nonfinite values: {nonfinite}")

        arm_configs: Dict[str, Dict[str, Any]] = {}
        for arm_id, expected in FROZEN_D6_ARMS.items():
            config = self.load_arm_config(arm_id)
            arm_configs[arm_id] = config
            for key in (
                "arm_id", "outcome", "protected_attribute", "feature_set", "disability_arm",
                "expected_predictors", "expected_groups", "expected_group_count",
                "expected_pair_count",
            ):
                if config.get(key) != expected.get(key):
                    raise FrozenArchiveIntegrityError(f"D6 arm config mismatch for {arm_id}: {key}")
            common_config = {
                "train_year": 2022,
                "validation_year": 2023,
                "future_test_year": 2024,
                "repeated_cross_sectional": True,
                "longitudinal": False,
                "causal_analysis": False,
                "survey_weighting": "NONE",
                "prediction_threshold": PREDICTION_THRESHOLD,
                "algorithm_mode": FAIRBIAS_ALGORITHM_MODE,
                "temporal_robustness_analysis": True,
            }
            if any(config.get(key) != value for key, value in common_config.items()):
                raise FrozenArchiveIntegrityError(f"D6 arm config scientific semantics mismatch for {arm_id}")
            classifier = config.get("classifier", {})
            expected_classifier = {
                "type": "LR",
                "random_state": 0,
                "max_iter": 1000,
                "solver": "lbfgs",
                "sample_weight": None,
            }
            if classifier != expected_classifier:
                raise FrozenArchiveIntegrityError(f"D6 classifier semantics mismatch for {arm_id}")
            if arm_id == "D6_ARM_004":
                if config.get("expected_predictors") != 15:
                    raise FrozenArchiveIntegrityError("D6 ARM004 predictor count mismatch")
                if any(name in config.get("categorical_features", []) for name in EXCLUDED_DISABILITY_COMPONENTS):
                    raise FrozenArchiveIntegrityError("D6 ARM004 disability exclusion mismatch")
            else:
                if config.get("expected_predictors") != 21:
                    raise FrozenArchiveIntegrityError(f"D6 predictor count mismatch for {arm_id}")

        self._last_audit = {
            "status": "PASS",
            "gate": D6_GATE,
            "release_id": D6_TRAIN_VAL_RELEASE_ID,
            "tag_object": git_info["tag_object"],
            "tag_commit": git_info["tag_commit"],
            "archive_manifest_sha256": manifest_sha,
            "archive_ledger_sha256": ledger_sha,
            "archive_canonical_file_count": 47,
            "archive_total_file_count": 48,
            "manifest_tracked_artifact_count": 45,
            "per_arm_artifact_count": 44,
            "expected_train_digests": self.expected_train_digests(),
            "expected_training_state_anchors": self.expected_state_anchors(),
            "preprocessing_state_expected_hash": PREPROCESSING_STATE_SHA256,
            "preprocessing_state_observed_hash": observed_preprocessing_hash,
            "preprocessing_state_match": True,
            "temporal_train_year": 2022,
            "frozen_validation_year": 2023,
            "future_test_year": 2024,
            "validation_access_prohibited": True,
            "global_training_state_reproduction_barrier": True,
            "real_2022_state_reproduction_executed": False,
            "validation_2023_cohort_requested": False,
            "test_2024_cohort_requested": False,
            "test_2024_evaluated": False,
            "canonical_test_release_created": False,
            "arm_configs": arm_configs,
            "disclosure": TEMPORAL_2024_DISCLOSURE,
        }
        return copy.deepcopy(self._last_audit)

    def run_audit_only(self) -> Dict[str, Any]:
        """Alias used by the CLI and tests; remains read-only and adapter-free."""
        return self.verify()

    @property
    def last_audit(self) -> Optional[Dict[str, Any]]:
        return copy.deepcopy(self._last_audit)


def _as_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return converted
    return None


def _invoke_injected(function: Callable[..., Any], available: Mapping[str, Any]) -> Any:
    """Invoke an injected synthetic/future callback without guessing scientific semantics."""
    signature = inspect.signature(function)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return function(**dict(available))
    kwargs = {
        name: value
        for name, value in available.items()
        if name in parameters
    }
    return function(**kwargs)


def _cohort_source_digest(cohort: Any) -> Optional[str]:
    if isinstance(cohort, Mapping):
        for key in ("source_row_digest", "train_source_row_digest", "observed_train_source_row_digest"):
            value = cohort.get(key)
            if isinstance(value, str):
                return value
    if hasattr(cohort, "source_row_digest"):
        value = getattr(cohort, "source_row_digest")
        if isinstance(value, str):
            return value
    if isinstance(cohort, (tuple, list)) and cohort:
        first = cohort[0]
        index = getattr(first, "index", None)
        if index is not None:
            values = [f"{TEMPORAL_TRAIN_YEAR}:{item}" for item in index]
            return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
    return None


_STATE_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "changed_dict": ("changed_dict", "changed_dict_sha256", "observed_changed_dict_sha256"),
    "baseline_scaler": ("baseline_scaler", "baseline_scaler_sha256", "observed_baseline_scaler_sha256"),
    "baseline_LR": (
        "baseline_LR", "baseline_lr", "baseline_logistic_regression",
        "baseline_logistic_regression_sha256", "observed_baseline_LR_sha256",
    ),
    "FairBias_scaler": (
        "FairBias_scaler", "fairbias_scaler", "fairbias_scaler_sha256",
        "observed_FairBias_scaler_sha256",
    ),
    "FairBias_LR": (
        "FairBias_LR", "fairbias_LR", "fairbias_lr", "fairbias_logistic_regression",
        "fairbias_logistic_regression_sha256", "observed_FairBias_LR_sha256",
    ),
}


def _hash_observed_state(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    mapping = _as_mapping(value)
    if mapping is not None:
        if "state" in mapping:
            return compute_canonical_json_sha256(mapping["state"])
        if "sha256" in mapping and isinstance(mapping["sha256"], str):
            # A caller may provide a serialized state plus its independently
            # computed hash.  Prefer recomputing when the state is present.
            return mapping["sha256"]
        return compute_canonical_json_sha256(dict(mapping))
    return None


def _summary_has_global_barrier(summary: Mapping[str, Any]) -> bool:
    """Validate the complete 4/4 and 20/20 proof before allowing 2024 access."""
    global_state = summary.get("global")
    arms = summary.get("arms")
    if not isinstance(global_state, Mapping) or not isinstance(arms, Mapping):
        return False
    if global_state.get("cohort_digest_match_count") != 4:
        return False
    if global_state.get("cohort_digest_expected_count") != 4:
        return False
    if global_state.get("cohort_digest_matches") != "4/4":
        return False
    if global_state.get("training_state_anchor_match_count") != 20:
        return False
    if global_state.get("training_state_anchor_expected_count") != 20:
        return False
    if global_state.get("training_state_anchor_matches") != "20/20":
        return False
    if global_state.get("all_training_state_anchors_verified") is not True:
        return False
    for arm_id in FROZEN_D6_ARM_IDS:
        arm_state = arms.get(arm_id)
        if not isinstance(arm_state, Mapping):
            return False
        if arm_state.get("digest_match") is not True:
            return False
        if arm_state.get("preprocessing_state_match") is not True:
            return False
        if arm_state.get("all_five_state_hashes_match") is not True:
            return False
        for logical_name, expected_hash in EXPECTED_TRAINING_STATE_ANCHORS[arm_id].items():
            anchor = arm_state.get(logical_name)
            if not isinstance(anchor, Mapping):
                return False
            if anchor.get("expected_sha256") != expected_hash:
                return False
            if anchor.get("observed_sha256") != expected_hash:
                return False
            if anchor.get("match") is not True:
                return False
    return True


@dataclass
class TrainingStateObservation:
    """Normalized result of one injected 2022 state reconstruction."""

    arm_id: str
    expected_train_source_row_digest: str
    observed_train_source_row_digest: Optional[str]
    preprocessing_state_expected_hash: str
    preprocessing_state_observed_hash: Optional[str]
    state_hashes: Dict[str, Optional[str]]
    digest_match: bool
    preprocessing_state_match: bool
    all_five_state_hashes_match: bool
    error: Optional[str] = None

    @property
    def all_matches(self) -> bool:
        return (
            self.digest_match
            and self.preprocessing_state_match
            and self.all_five_state_hashes_match
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "arm_id": self.arm_id,
            "expected_2022_source_row_digest": self.expected_train_source_row_digest,
            "observed_2022_source_row_digest": self.observed_train_source_row_digest,
            "digest_match": self.digest_match,
            "preprocessing_state_expected_hash": self.preprocessing_state_expected_hash,
            "preprocessing_state_observed_hash": self.preprocessing_state_observed_hash,
            "preprocessing_state_match": self.preprocessing_state_match,
            "all_five_state_hashes_match": self.all_five_state_hashes_match,
            "frozen_prediction_threshold": PREDICTION_THRESHOLD,
            "error": self.error,
        }
        for logical_name, expected_hash in EXPECTED_TRAINING_STATE_ANCHORS[self.arm_id].items():
            observed_hash = self.state_hashes.get(logical_name)
            payload[logical_name] = {
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "match": observed_hash == expected_hash,
            }
            payload[f"{logical_name}_expected_sha256"] = expected_hash
            payload[f"{logical_name}_observed_sha256"] = observed_hash
        return payload


class D6TrainingStateReproducer:
    """Reproduce all four 2022 training states before any 2024 request.

    ``reproduction_fn`` is intentionally injected.  It receives a 2022 cohort
    and must return the observed digest, preprocessing state, and five logical
    state hashes.  No real-data implementation is bundled into Gate D6.1a.
    """

    def __init__(
        self,
        archive_loader: FrozenD6ArchiveLoader,
        adapter: Any,
        *,
        reproduction_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
        event_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.archive_loader = archive_loader
        self.adapter = adapter
        self.reproduction_fn = reproduction_fn
        self.event_callback = event_callback
        self.events: List[str] = []
        self.observations: Dict[str, TrainingStateObservation] = {}
        self.failures: Dict[str, str] = {}

    def _emit(self, event: str) -> None:
        self.events.append(event)
        if self.event_callback is not None:
            self.event_callback(event)

    def _request_2022(self, arm_id: str) -> Any:
        arm = FROZEN_D6_ARMS[arm_id]
        self._emit(f"request_2022_{arm_id}")
        if self.adapter is None or not hasattr(self.adapter, "get_cohort"):
            raise D6HarnessError(
                "D6.1 training-state reproduction requires an injected adapter; "
                "the audit path never constructs one."
            )
        return self.adapter.get_cohort(
            year=TEMPORAL_TRAIN_YEAR,
            outcome=arm["outcome"],
            protected_attribute=arm["protected_attribute"],
            feature_set=arm["feature_set"].lower(),
            disability_arm=arm["disability_arm"],
        )

    def _call_reproduction(self, arm_id: str, cohort: Any) -> Mapping[str, Any]:
        arm = FROZEN_D6_ARMS[arm_id]
        self._emit(f"reproduce_{arm_id}")
        function = self.reproduction_fn
        if function is None:
            candidate = getattr(self.adapter, "reproduce_training_state", None)
            if callable(candidate):
                function = candidate
        if function is None:
            raise D6HarnessError(
                "No injected 2022 state reproducer is available. "
                "The real D6.1b reproducer is outside Gate D6.1a."
            )
        result = _invoke_injected(
            function,
            {
                "arm_id": arm_id,
                "arm_config": copy.deepcopy(arm),
                "cohort": cohort,
                "archive_loader": self.archive_loader,
                "expected_train_source_row_digest": self.archive_loader.expected_train_digests()[arm_id],
                "expected_state_anchors": self.archive_loader.expected_state_anchors()[arm_id],
            },
        )
        mapping = _as_mapping(result)
        if mapping is None:
            raise AnchorMismatchError(f"Reproducer returned no mapping for {arm_id}")
        return mapping

    def reproduce_arm(self, arm_id: str) -> TrainingStateObservation:
        """Request exactly one 2022 arm and validate every frozen anchor."""
        if arm_id not in FROZEN_D6_ARMS:
            raise KeyError(f"Unknown frozen D6 arm: {arm_id}")
        expected_digest = self.archive_loader.expected_train_digests()[arm_id]
        expected_states = self.archive_loader.expected_state_anchors()[arm_id]
        cohort = self._request_2022(arm_id)
        payload = self._call_reproduction(arm_id, cohort)

        nested = payload.get("training_state_reproduction", payload)
        nested_mapping = _as_mapping(nested) or payload
        observed_digest: Optional[str] = None
        for key in (
            "observed_2022_source_row_digest",
            "observed_train_source_row_digest",
            "train_source_row_digest",
            "source_row_digest",
        ):
            if isinstance(nested_mapping.get(key), str):
                observed_digest = nested_mapping[key]
                break
        if observed_digest is None:
            observed_digest = _cohort_source_digest(cohort)

        preprocessing_state = None
        for key in ("preprocessing_state", "preprocessing_record", "observed_preprocessing_state"):
            if key in nested_mapping:
                preprocessing_state = nested_mapping[key]
                break
        if preprocessing_state is None:
            preprocessor = getattr(self.adapter, "preprocessor", None)
            fitted_record = getattr(preprocessor, "fitted_record", None)
            if fitted_record is not None and hasattr(fitted_record, "to_dict"):
                preprocessing_state = fitted_record.to_dict()
        preprocessing_observed_hash: Optional[str]
        if isinstance(preprocessing_state, str):
            preprocessing_observed_hash = preprocessing_state
        elif preprocessing_state is not None:
            preprocessing_observed_hash = compute_canonical_json_sha256(preprocessing_state)
        else:
            preprocessing_observed_hash = nested_mapping.get("preprocessing_state_observed_hash")
        preprocessing_expected_hash = PREPROCESSING_STATE_SHA256

        state_container = nested_mapping.get("state_hashes", nested_mapping.get("model_state_hashes", nested_mapping))
        state_mapping = _as_mapping(state_container) or {}
        observed_states: Dict[str, Optional[str]] = {}
        for logical_name in expected_states:
            observed_value = None
            for alias in _STATE_KEY_ALIASES[logical_name]:
                if alias in state_mapping:
                    observed_value = _hash_observed_state(state_mapping[alias])
                    break
            observed_states[logical_name] = observed_value

        observation = TrainingStateObservation(
            arm_id=arm_id,
            expected_train_source_row_digest=expected_digest,
            observed_train_source_row_digest=observed_digest,
            preprocessing_state_expected_hash=preprocessing_expected_hash,
            preprocessing_state_observed_hash=preprocessing_observed_hash,
            state_hashes=observed_states,
            digest_match=observed_digest == expected_digest,
            preprocessing_state_match=preprocessing_observed_hash == preprocessing_expected_hash,
            all_five_state_hashes_match=all(
                observed_states[name] == expected_hash
                for name, expected_hash in expected_states.items()
            ),
        )
        self.observations[arm_id] = observation
        if not observation.all_matches:
            failures: List[str] = []
            if not observation.digest_match:
                failures.append("2022 source-row digest")
            if not observation.preprocessing_state_match:
                failures.append("preprocessing state")
            if not observation.all_five_state_hashes_match:
                failures.append("five training-state hashes")
            observation.error = "Mismatch: " + ", ".join(failures)
            self.failures[arm_id] = observation.error
            raise AnchorMismatchError(f"{arm_id} {observation.error}")
        return observation

    def build_summary(self) -> Dict[str, Any]:
        arm_payload: Dict[str, Any] = {}
        digest_matches = 0
        state_matches = 0
        preprocessing_hashes: List[str] = []
        for arm_id in FROZEN_D6_ARM_IDS:
            observation = self.observations.get(arm_id)
            if observation is None:
                observation = TrainingStateObservation(
                    arm_id=arm_id,
                    expected_train_source_row_digest=self.archive_loader.expected_train_digests()[arm_id],
                    observed_train_source_row_digest=None,
                    preprocessing_state_expected_hash=PREPROCESSING_STATE_SHA256,
                    preprocessing_state_observed_hash=None,
                    state_hashes={name: None for name in EXPECTED_TRAINING_STATE_ANCHORS[arm_id]},
                    digest_match=False,
                    preprocessing_state_match=False,
                    all_five_state_hashes_match=False,
                    error=self.failures.get(arm_id, "arm reproduction did not complete"),
                )
            if observation.digest_match:
                digest_matches += 1
            if observation.all_five_state_hashes_match:
                state_matches += 5
            if observation.preprocessing_state_observed_hash is not None:
                preprocessing_hashes.append(observation.preprocessing_state_observed_hash)
            arm_payload[arm_id] = observation.to_dict()

        preprocessing_observed = preprocessing_hashes[0] if preprocessing_hashes else None
        preprocessing_match = (
            bool(preprocessing_hashes)
            and len(set(preprocessing_hashes)) == 1
            and preprocessing_observed == PREPROCESSING_STATE_SHA256
            and all(item["preprocessing_state_match"] for item in arm_payload.values())
        )
        all_verified = digest_matches == 4 and state_matches == 20 and preprocessing_match
        return {
            "d6_train_val_archive_tag": D6_TRAIN_VAL_TAG,
            "d6_archive_commit": D6_TRAIN_VAL_ARCHIVE_COMMIT,
            "d6_archive_manifest_sha": D6_TRAIN_VAL_MANIFEST_SHA256,
            "preprocessing_state_expected_hash": PREPROCESSING_STATE_SHA256,
            "preprocessing_state_observed_hash": preprocessing_observed,
            "preprocessing_match": preprocessing_match,
            "frozen_prediction_threshold": PREDICTION_THRESHOLD,
            "arms": arm_payload,
            "global": {
                "cohort_digest_match_count": digest_matches,
                "cohort_digest_expected_count": 4,
                "cohort_digest_matches": f"{digest_matches}/4",
                "training_state_anchor_match_count": state_matches,
                "training_state_anchor_expected_count": 20,
                "training_state_anchor_matches": f"{state_matches}/20",
                "all_training_state_anchors_verified": all_verified,
            },
        }

    def reproduce_all(self, *, fail_closed: bool = True) -> Dict[str, Any]:
        """Run Phase 1 for every arm; never transition to 2024 on mismatch."""
        for arm_id in FROZEN_D6_ARM_IDS:
            try:
                self.reproduce_arm(arm_id)
            except Exception as exc:
                self.failures.setdefault(arm_id, str(exc))
        summary = self.build_summary()
        if fail_closed and not summary["global"]["all_training_state_anchors_verified"]:
            raise GlobalBarrierError(
                "ALL_D6_TRAINING_STATE_ANCHORS_VERIFIED was not reached; "
                "2024 access is forbidden."
            )
        return summary


class D6TemporalTestEvaluator:
    """2024-only evaluator that can operate only after the global barrier."""

    def __init__(
        self,
        archive_loader: FrozenD6ArchiveLoader,
        adapter: Any,
        training_state_summary: Mapping[str, Any],
        *,
        barrier_passed: Optional[bool] = None,
        score_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
        event_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.archive_loader = archive_loader
        self.adapter = adapter
        self.training_state_summary = copy.deepcopy(dict(training_state_summary))
        inferred_barrier = _summary_has_global_barrier(self.training_state_summary)
        self.barrier_passed = (
            inferred_barrier
            if barrier_passed is None
            else bool(barrier_passed) and inferred_barrier
        )
        self.score_fn = score_fn
        self.event_callback = event_callback
        self.events: List[str] = []

    def _emit(self, event: str) -> None:
        self.events.append(event)
        if self.event_callback is not None:
            self.event_callback(event)

    def get_cohort(self, year: int, arm_id: Optional[str] = None) -> Any:
        """Strict temporal gate: 2023 is forbidden; 2024 needs the barrier."""
        year_int = int(year)
        if year_int == TEMPORAL_VALIDATION_YEAR:
            raise AssertionError(
                "FATAL: D6.1 attempted to access frozen 2023 validation cohort"
            )
        if year_int != TEMPORAL_TEST_YEAR:
            raise ValueError("D6.1 test evaluator accepts only the fixed 2024 TEST year")
        if not self.barrier_passed:
            raise AssertionError(
                "FATAL: D6.1 attempted temporal TEST access before all frozen training states were reproduced"
            )
        return self._request_2024(arm_id)

    def _request_2024(self, arm_id: Optional[str] = None) -> Any:
        if not self.barrier_passed:
            raise AssertionError(
                "FATAL: D6.1 attempted temporal TEST access before all frozen training states were reproduced"
            )
        if self.adapter is None or not hasattr(self.adapter, "get_cohort"):
            raise D6HarnessError("D6.1 2024 evaluation requires an injected adapter")
        if arm_id is None:
            raise ValueError("arm_id is required for a 2024 request")
        arm = FROZEN_D6_ARMS[arm_id]
        self._emit(f"request_2024_{arm_id}")
        return self.adapter.get_cohort(
            year=TEMPORAL_TEST_YEAR,
            outcome=arm["outcome"],
            protected_attribute=arm["protected_attribute"],
            feature_set=arm["feature_set"].lower(),
            disability_arm=arm["disability_arm"],
        )

    def evaluate_arm(self, arm_id: str) -> Dict[str, Any]:
        """Request 2024 only after the barrier and delegate scoring to an injected function."""
        if arm_id not in FROZEN_D6_ARMS:
            raise KeyError(f"Unknown frozen D6 arm: {arm_id}")
        cohort = self._request_2024(arm_id)
        arm_summary = self.training_state_summary.get("arms", {}).get(arm_id, {})
        state_snapshot = compute_canonical_json_sha256(arm_summary)
        state_copy = copy.deepcopy(arm_summary)
        if self.score_fn is None:
            result: Mapping[str, Any] = {
                "arm_id": arm_id,
                "test_year": TEMPORAL_TEST_YEAR,
                "diagnostic_only": True,
                "prediction_threshold": PREDICTION_THRESHOLD,
            }
        else:
            result = _as_mapping(
                _invoke_injected(
                    self.score_fn,
                    {
                        "arm_id": arm_id,
                        "arm_config": copy.deepcopy(FROZEN_D6_ARMS[arm_id]),
                        "cohort": cohort,
                        "frozen_state": state_copy,
                        "training_state": state_copy,
                        "threshold": PREDICTION_THRESHOLD,
                        "prediction_threshold": PREDICTION_THRESHOLD,
                    },
                )
            ) or {}
        if "prediction_threshold" in result and result["prediction_threshold"] != PREDICTION_THRESHOLD:
            raise D6HarnessError("D6.1 2024 evaluation attempted to change threshold 0.5")
        returned_state = result.get("frozen_state", result.get("training_state"))
        if returned_state is not None and compute_canonical_json_sha256(returned_state) != state_snapshot:
            raise D6HarnessError("D6.1 2024 evaluation attempted to change frozen training state")
        if compute_canonical_json_sha256(state_copy) != state_snapshot:
            raise D6HarnessError("D6.1 2024 evaluation mutated frozen training state")
        if compute_canonical_json_sha256(arm_summary) != state_snapshot:
            raise D6HarnessError("D6.1 2024 evaluation mutated frozen training state")
        self._emit(f"evaluate_2024_{arm_id}")
        return copy.deepcopy(dict(result))

    def evaluate_all(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate all four 2024 arms in order, after the barrier."""
        if not self.barrier_passed:
            raise AssertionError(
                "FATAL: D6.1 attempted temporal TEST access before all frozen training states were reproduced"
            )
        return {arm_id: self.evaluate_arm(arm_id) for arm_id in FROZEN_D6_ARM_IDS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class NHISD6TemporalTestReleaseManager:
    """Global two-phase manager for future synthetic/authorized D6.1 execution.

    ``execute_release`` requires explicit opt-in and an injected adapter.  The CLI
    does not call it in Gate D6.1a.  Any output is written only to a caller-supplied
    temporary directory and is never treated as a canonical D6.1b release.
    """

    def __init__(
        self,
        *,
        archive_loader: Optional[FrozenD6ArchiveLoader] = None,
        repo_root: Optional[pathlib.Path | str] = None,
        adapter: Any = None,
        reproduction_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
        score_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
    ) -> None:
        self.repo_root = pathlib.Path(repo_root).resolve() if repo_root else _REPO_ROOT
        self.archive_loader = archive_loader or FrozenD6ArchiveLoader(repo_root=self.repo_root)
        self.adapter = adapter
        self.reproduction_fn = reproduction_fn
        self.score_fn = score_fn
        self.events: List[str] = []
        self.last_result: Optional[Dict[str, Any]] = None

    def _record_event(self, event: str) -> None:
        self.events.append(event)

    def run_audit_only(self) -> Dict[str, Any]:
        """Perform the complete static audit without adapter construction or cohort access."""
        archive_audit = self.archive_loader.run_audit_only()
        result = {
            "status": "PASS",
            "audit_name": "D6 TEMPORAL TEST HARNESS AUDIT",
            "frozen_d6_train_val_tag": "VERIFIED",
            "frozen_d6_train_val_archive": "VERIFIED",
            "d6_train_val_manifest": "VERIFIED",
            "expected_train_digests_loaded": "4 / 4",
            "expected_train_digests_count": 4,
            "expected_training_state_anchors_loaded": "20 / 20",
            "expected_training_state_anchors_count": 20,
            "temporal_train_year": 2022,
            "frozen_validation_year": "2023 — ACCESS PROHIBITED",
            "future_test_year": 2024,
            "global_training_state_reproduction_barrier": "ENFORCED",
            "real_2022_state_reproduction_executed": False,
            "2023_cohort_requested": False,
            "2024_test_cohort_requested": False,
            "2024_test_evaluated": False,
            "canonical_test_release_created": False,
            "archive": archive_audit,
        }
        return result

    def execute_release(
        self,
        output_dir: pathlib.Path | str,
        *,
        allow_substantive_execution: bool = False,
        adapter: Any = None,
        reproduction_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
        score_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run the two-phase synthetic/future path in a fresh temporary directory.

        This method is not used by the D6.1a CLI.  It exists to make the global
        barrier dynamically testable without providing a real-data default path.
        """
        if not allow_substantive_execution:
            raise D6HarnessError(
                "Substantive D6.1 execution is not authorized by the D6.1a audit-only CLI."
            )
        selected_adapter = self.adapter if adapter is None else adapter
        if selected_adapter is None:
            raise D6HarnessError(
                "Synthetic/future execution requires an explicitly injected adapter; "
                "real NHIS access is not a D6.1a default."
            )
        target = pathlib.Path(output_dir).resolve()
        if target.exists():
            raise ReleaseCollisionError(
                f"D6.1 release output directory already exists: {target}; silent overwrite/retry is forbidden"
            )

        # Static archive preconditions are checked before creating any output.
        archive_audit = self.archive_loader.verify()
        target.mkdir(parents=True, exist_ok=False)
        started_at = _utc_now()
        _write_json(
            target / "release_state.json",
            {
                "release_id": "SYNTHETIC_D6_TEMPORAL_TEST",
                "status": "STARTED",
                "started_at": started_at,
                "test_year": 2024,
                "validation_year_requested": False,
                "test_year_requested": False,
                "test_year_evaluated": False,
            },
        )
        self.events = []
        selected_reproduction_fn = self.reproduction_fn if reproduction_fn is None else reproduction_fn
        selected_score_fn = self.score_fn if score_fn is None else score_fn
        reproducer = D6TrainingStateReproducer(
            self.archive_loader,
            selected_adapter,
            reproduction_fn=selected_reproduction_fn,
            event_callback=self._record_event,
        )
        try:
            summary = reproducer.reproduce_all(fail_closed=False)
            _write_json(target / "training_state_reproduction_summary.json", summary)
            if not summary["global"]["all_training_state_anchors_verified"]:
                self._record_event("GLOBAL_BARRIER_BLOCKED")
                failed = {
                    "release_id": "SYNTHETIC_D6_TEMPORAL_TEST",
                    "status": "FAILED",
                    "failed_at": _utc_now(),
                    "error": "Global 2022 training-state reproduction barrier failed",
                    "2022_request_count": sum(
                        1 for event in self.events if event.startswith("request_2022_")
                    ),
                    "2023_request_count": 0,
                    "2024_request_count": 0,
                    "test_year_requested": False,
                    "test_year_evaluated": False,
                }
                _write_json(target / "release_state.json", failed)
                self.last_result = failed
                raise GlobalBarrierError(failed["error"])

            self._record_event("ALL_D6_TRAINING_STATE_ANCHORS_VERIFIED")
            evaluator = D6TemporalTestEvaluator(
                self.archive_loader,
                selected_adapter,
                summary,
                barrier_passed=True,
                score_fn=selected_score_fn,
                event_callback=self._record_event,
            )
            test_results = evaluator.evaluate_all()
            completed = {
                "release_id": "SYNTHETIC_D6_TEMPORAL_TEST",
                "status": "COMPLETE",
                "completed_at": _utc_now(),
                "test_year": 2024,
                "validation_year_requested": False,
                "test_year_requested": True,
                "test_year_evaluated": True,
                "training_state_anchor_matches": "20/20",
                "cohort_digest_matches": "4/4",
            }
            _write_json(target / "release_state.json", completed)
            self.last_result = {
                "release_id": completed["release_id"],
                "status": "COMPLETE",
                "release_dir": str(target),
                "archive_audit": archive_audit,
                "training_state_reproduction_summary": summary,
                "test_results": test_results,
                "events": list(self.events),
            }
            return copy.deepcopy(self.last_result)
        except Exception as exc:
            if self.last_result is not None and self.last_result.get("status") == "FAILED":
                raise
            failed = {
                "release_id": "SYNTHETIC_D6_TEMPORAL_TEST",
                "status": "FAILED",
                "failed_at": _utc_now(),
                "error": str(exc),
                "2022_request_count": sum(
                    1 for event in self.events if event.startswith("request_2022_")
                ),
                "2023_request_count": 0,
                "2024_request_count": sum(
                    1 for event in self.events if event.startswith("request_2024_")
                ),
                "test_year_requested": any(event.startswith("request_2024_") for event in self.events),
                "test_year_evaluated": False,
            }
            _write_json(target / "release_state.json", failed)
            self.last_result = failed
            raise


__all__ = [
    "AnchorMismatchError",
    "CLASSIFIER_MAX_ITER",
    "CLASSIFIER_SOLVER",
    "D6_ARCHIVE_GATE",
    "D6HarnessError",
    "D6_GATE",
    "D6TemporalTestEvaluator",
    "D6TrainingStateReproducer",
    "D6_TRAIN_VAL_ARCHIVE_DIR",
    "D6_TRAIN_VAL_ARCHIVE_COMMIT",
    "D6_SCIENTIFIC_EXECUTION_COMMIT",
    "D6_TRAIN_VAL_LEDGER_SHA256",
    "D6_TRAIN_VAL_MANIFEST_SHA256",
    "D6_TRAIN_VAL_RELEASE_ID",
    "D6_TRAIN_VAL_TAG",
    "D6_TRAIN_VAL_TAG_OBJECT",
    "DEFAULT_RANDOM_SEED",
    "EXCLUDED_DISABILITY_COMPONENTS",
    "EXPECTED_TRAINING_STATE_ANCHORS",
    "EXPECTED_TRAIN_SOURCE_ROW_DIGESTS",
    "FAIRBIAS_ALGORITHM_MODE",
    "FROZEN_D6_ARCHIVE_RELATIVE_PATH",
    "FROZEN_D6_ARMS",
    "FROZEN_D6_ARM_IDS",
    "FROZEN_D6_SCIENTIFIC_PATHS",
    "FUTURE_COMPLETE_FILE_COUNT",
    "FUTURE_MANIFEST_TRACKED_ARTIFACT_COUNT",
    "FUTURE_PER_ARM_ARTIFACT_COUNT",
    "FUTURE_PER_ARM_TEST_ARTIFACTS",
    "FrozenArchiveIntegrityError",
    "FrozenD6ArchiveLoader",
    "GlobalBarrierError",
    "NHISD6TemporalTestReleaseManager",
    "POWER_REVISIT_POLICY",
    "POWER_SEQUENCE_POLICY",
    "PREDICTION_THRESHOLD",
    "PREPROCESSING_STATE_SHA256",
    "ReleaseCollisionError",
    "TEMPORAL_2024_DISCLOSURE",
    "TEMPORAL_ROLES",
    "TEMPORAL_TEST_YEAR",
    "TEMPORAL_TRAIN_YEAR",
    "TEMPORAL_VALIDATION_YEAR",
    "compute_canonical_json_sha256",
    "compute_file_sha256",
]
