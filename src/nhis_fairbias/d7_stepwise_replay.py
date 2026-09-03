"""Gate D7.2a.1: Frozen-Design Path-Dependent Stepwise FairBias Representation Replay Audit Harness.

Scientific Question:
"Along the actual accepted FairBias transformation path learned from NHIS 2022,
at which path states do predictive discrimination, class-conditional score separation,
and prediction volume change, and how do those path-dependent changes transport to
the unchanged 2023 and 2024 repeated cross-sections?"

Scientific Identity:
- Post-primary, post-TEST, exploratory explanatory, path-dependent, temporal repeated-cross-sectional.
- NOT causal identification, NOT an independent TEST, NOT model selection, NOT a new fairness algorithm,
  NOT threshold optimization.
- INTERMEDIATE MODEL REFITTING: At each intermediate representation state k, fit a NEW MinMaxScaler
  and LogisticRegression using 2022 ONLY.
- ANTI-LEAKAGE: 2023 and 2024 never enter .fit() or .fit_transform().
- DESCRIPTIVE MARGINAL CHANGES: Differences across path states are strictly "pathwise marginal changes",
  never "causal effects" or "effects holding all else constant".
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler

from fairbias.evaluator import FairEvaluator
from fairbias.transform import FairTransform, apply_power_transform

from .d6_temporal_runner import (
    compute_canonical_json_sha256,
    compute_cohort_source_row_digest,
    compute_sha256,
    extract_logistic_regression_state,
    extract_minmax_scaler_state,
)

# -----------------------------------------------------------------------------
# Scientific Disclosures, Terminology & Questions
# -----------------------------------------------------------------------------
D7_2_SCIENTIFIC_QUESTION: str = (
    "Along the actual accepted FairBias transformation path learned from NHIS 2022, "
    "at which path states do predictive discrimination, class-conditional score separation, "
    "and prediction volume change, and how do those path-dependent changes transport to "
    "the unchanged 2023 and 2024 repeated cross-sections?"
)

MANDATORY_D7_2_DISCLOSURE: str = (
    "D7.2 is an exploratory explanatory, path-dependent post-primary audit conducted along the "
    "actual accepted FairBias transformation path learned from NHIS 2022. Intermediate representation "
    "states have no historical model; each state fits a fresh scaler and logistic regression on 2022 only. "
    "Evaluations on 2023 and 2024 repeated cross-sections are descriptive and order-dependent, not causal."
)

D7_2_SCIENTIFIC_TERMINOLOGY: Dict[str, bool] = {
    "post_primary": True,
    "post_test": True,
    "exploratory_explanatory": True,
    "path_dependent": True,
    "temporal_repeated_cross_sectional": True,
    "longitudinal": False,
    "panel": False,
    "causal_identification": False,
    "independent_test": False,
    "model_selection": False,
    "new_fairness_algorithm": False,
    "threshold_optimization": False,
}

# -----------------------------------------------------------------------------
# Refitting Configuration & Protocol Invariants
# -----------------------------------------------------------------------------
D7_2_REFITS_INTERMEDIATE_CLASSIFIERS: bool = True
TRAINING_YEAR: int = 2022
VALIDATION_YEAR: int = 2023
DESCRIPTIVE_TEST_YEAR: int = 2024
TEMPORAL_YEARS: Tuple[int, ...] = (2022, 2023, 2024)

CANONICAL_DECISION_THRESHOLD: float = 0.5
LR_RANDOM_STATE: int = 0
LR_SOLVER: str = "lbfgs"
LR_MAX_ITER: int = 1000
LR_FIT_INTERCEPT: bool = True
SURVEY_WEIGHTS: Optional[Any] = None

MATRIX_NUMERICAL_TOLERANCE: float = 1e-12
METRIC_REPRODUCTION_TOLERANCE: float = 1e-12

# -----------------------------------------------------------------------------
# Starting Identity, Tags, and Upstream Anchors
# -----------------------------------------------------------------------------
STARTING_HEAD_COMMIT: str = "b1fcbb07d04d0d53bfc61ebcd80b8ed16ed985c3"
D7_1_TAG: str = "nhis-d7-terminal-mechanism-v1"
D7_1_TAG_OBJECT: str = "bd1067ee1e4530d303e8ea4d6e605d4afcc45d53"
D7_1_COMMIT: str = "a5f0d81c4cceb50b02d40ad6a2f2e4f88f2eefd9"

D6_TRAIN_VAL_RELEASE_ID: str = "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
D6_TRAIN_VAL_MANIFEST_SHA256: str = (
    "773d43b2d893232eb4cbba9dfcdcc639ecdc2fbc8efb2e353f7d2a68514159c4"
)
D6_TRAIN_VAL_TAG: str = "nhis-d6-temporal-train-val-v1"
D6_TRAIN_VAL_TAG_OBJECT: str = "e23941edf2085c84290868f69289d30af62b7e95"
D6_TRAIN_VAL_COMMIT: str = "bdf154c541d365b216a732bc3a58415af41cdcc7"

D6_TEST_RELEASE_ID: str = "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"
D6_TEST_MANIFEST_SHA256: str = (
    "2a134eb1c23421fd7e50f8c255f2d55395ecfda8149700c78f76dac3da553fd5"
)
D6_TEST_TAG: str = "nhis-d6-temporal-test-v1"
D6_TEST_TAG_OBJECT: str = "b3aaae6559028e83aea9c329d585c1607e87364c"
D6_TEST_COMMIT: str = "e24685cbde26497d0209f26fcf1d82b131dbe726"

D6_ARM_IDS: Tuple[str, ...] = (
    "D6_ARM_001",
    "D6_ARM_002",
    "D6_ARM_003",
    "D6_ARM_004",
)

ARM_PROTECTED_ATTRIBUTES: Dict[str, str] = {
    "D6_ARM_001": "SEX_A",
    "D6_ARM_002": "HISPALLP_A",
    "D6_ARM_003": "DISAB3_A",
    "D6_ARM_004": "DISAB3_A",
}

ARM_DISABILITY_POLICIES: Dict[str, str] = {
    "D6_ARM_001": "full_feature",
    "D6_ARM_002": "full_feature",
    "D6_ARM_003": "full_feature",
    "D6_ARM_004": "exclude_disability_components",
}

# Expected path lengths by arm
EXPECTED_ARCHIVED_STEP_COUNTS: Dict[str, int] = {
    "D6_ARM_001": 15,  # SEX
    "D6_ARM_002": 12,  # HISP
    "D6_ARM_003": 7,   # DISAB full
    "D6_ARM_004": 9,   # DISAB excl
}

TOTAL_EXPECTED_ACCEPTED_STEPS: int = 43
TOTAL_EXPECTED_BASELINE_STATES: int = 4
TOTAL_EXPECTED_REPRESENTATION_STATES: int = 47

# Prepared data and preprocessing anchors
FROZEN_FEATURES_PARQUET_PATH: Path = Path("data/processed/nhis/nhis_2022_2024_features.parquet")
FROZEN_FEATURES_PARQUET_SHA256: str = (
    "49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383"
)
PREPROCESSING_STATE_SHA256: str = (
    "f106967a8ed9bf46ff1c3ff009e40763280fa1dbc78983d3a479a1508fd83db9"
)

# 12 Mandatory Expected Cohort Source-Row Digests
EXPECTED_COHORT_SOURCE_ROW_DIGESTS: Dict[int, Dict[str, str]] = {
    2022: {
        "D6_ARM_001": "fbf4c5b0cadd74b7dfa565082e5d6577c8ec75fbdbf5abbdc8e48d712896ff6e",
        "D6_ARM_002": "30a71c454f54871cde9c03e34eabe936ce0c1a55a789e31017d9b31e81f23aa4",
        "D6_ARM_003": "e54056b34627b15440af95bcdb8e3f6ca1f282a909d3a59a3f47d62e7959185d",
        "D6_ARM_004": "e54056b34627b15440af95bcdb8e3f6ca1f282a909d3a59a3f47d62e7959185d",
    },
    2023: {
        "D6_ARM_001": "f66e94714e614c8ffd6ddeb2f9d466e616164e7b9aaf023e58b34c776006716a",
        "D6_ARM_002": "31197a19f03bf65a75cb7a6db2c97de8c7bb6a4878aa5260f9d1ee5ab56d82ad",
        "D6_ARM_003": "b24ede4a8a5a71c798612be991a1b3032a868b24da5f5793a201372b211d747a",
        "D6_ARM_004": "b24ede4a8a5a71c798612be991a1b3032a868b24da5f5793a201372b211d747a",
    },
    2024: {
        "D6_ARM_001": "f1d4386c9c14d939482bebaae94001f126fd388a8e55c494c65532f306b2ad4f",
        "D6_ARM_002": "0162b49440230ce4047ff2ebc1c2e0261479344615af282d9674f07e91786708",
        "D6_ARM_003": "0a9efcfd5a18647c55614d515066177ea6fabf8f5292472f4558ad87692079e9",
        "D6_ARM_004": "0a9efcfd5a18647c55614d515066177ea6fabf8f5292472f4558ad87692079e9",
    },
}

# 20 Mandatory Expected Training State Anchors
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

# -----------------------------------------------------------------------------
# Predeclared Future Artifact Schema (10 Files Total, 8 Manifest-Tracked)
# -----------------------------------------------------------------------------
D7_2_MANIFEST_TRACKED_ARTIFACTS: Tuple[str, ...] = (
    "provenance_summary.json",
    "archived_trace_inventory.json",
    "terminal_replay_verification.json",
    "endpoint_reproduction.json",
    "stepwise_metrics.csv",
    "stepwise_deltas.csv",
    "protected_fairness_stepwise.csv",
    "key_path_summary.json",
)

D7_2_UNTRACKED_CONTROL_FILES: Tuple[str, ...] = (
    "d7_stepwise_manifest.json",
    "release_state.json",
)

D7_2_ALL_RELEASE_FILES: Tuple[str, ...] = (
    D7_2_MANIFEST_TRACKED_ARTIFACTS + D7_2_UNTRACKED_CONTROL_FILES
)

# Forbid causal terminology in generated text/summaries
FORBIDDEN_CAUSAL_TERMS: Tuple[str, ...] = (
    "caused",
    "causes",
    "causal effect",
    "causal impact",
    "independent effect",
    "effect holding all else constant",
    "ceteris paribus",
    "H1 proven",
    "H2 proven",
    "H3 proven",
    "H4 proven",
)


# -----------------------------------------------------------------------------
# Custom Exceptions
# -----------------------------------------------------------------------------
class D7StepwiseReplayError(Exception):
    """Base error for D7.2 stepwise replay harness."""


class IllegalReplayError(D7StepwiseReplayError):
    """Raised when an illegal operation occurs during trace replay (e.g. reappearing dropped feature)."""


class TerminalReplayBarrierError(D7StepwiseReplayError):
    """Raised when sequential replay through state K fails to match the frozen terminal representation."""


class EndpointReproductionBarrierError(D7StepwiseReplayError):
    """Raised when refitting on state 0 or state K fails to reproduce archived D6 models/metrics."""


class DataLeakageError(D7StepwiseReplayError):
    """Raised when temporal holdout data (2023 or 2024) is accessed during model training."""


class ProvenanceVerificationError(D7StepwiseReplayError):
    """Raised when repository state, tags, or upstream archives fail verification."""


# -----------------------------------------------------------------------------
# Trace Step and Representation State Data Classes
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ArchivedTraceStep:
    """Immutable representation of one accepted step from train_fairbias_trace.json."""

    iteration: int
    selected_feature: str
    feature_semantic_type: str
    d_phi_before: float
    epsilon: float
    proposed_transformation: Any
    accepted_transformation: Any
    numerical_exponent: Optional[float] = None
    categorical_merge_mapping: Optional[Dict[str, str]] = None
    d_phi_after: Optional[float] = None
    dropped: bool = False
    stopped_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "selected_feature": self.selected_feature,
            "feature_semantic_type": self.feature_semantic_type,
            "d_phi_before": self.d_phi_before,
            "epsilon": self.epsilon,
            "proposed_transformation": self.proposed_transformation,
            "accepted_transformation": self.accepted_transformation,
            "numerical_exponent": self.numerical_exponent,
            "categorical_merge_mapping": self.categorical_merge_mapping,
            "d_phi_after": self.d_phi_after,
            "dropped": self.dropped,
            "stopped_reason": self.stopped_reason,
        }


@dataclass(frozen=True)
class RepresentationState:
    """Representation state along the FairBias transformation path."""

    arm_id: str
    state_index: int
    preceding_step_index: Optional[int]
    selected_feature: Optional[str]
    feature_semantic_type: Optional[str]
    accepted_transformation: Any
    dropped: Optional[bool]
    numerical_exponent: Optional[float]
    categorical_merge_mapping: Optional[Dict[str, str]]
    active_feature_count: int
    active_features: Tuple[str, ...]
    cumulative_changed_dict: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "state_index": self.state_index,
            "preceding_step_index": self.preceding_step_index,
            "selected_feature": self.selected_feature,
            "feature_semantic_type": self.feature_semantic_type,
            "accepted_transformation": self.accepted_transformation,
            "dropped": self.dropped,
            "numerical_exponent": self.numerical_exponent,
            "categorical_merge_mapping": self.categorical_merge_mapping,
            "active_feature_count": self.active_feature_count,
            "active_features": list(self.active_features),
            "cumulative_changed_dict": self.cumulative_changed_dict,
        }


# -----------------------------------------------------------------------------
# Trace Parsing & Loading
# -----------------------------------------------------------------------------
def load_archived_trace(trace_path: Union[str, Path]) -> List[ArchivedTraceStep]:
    """Parse and validate train_fairbias_trace.json into structured ArchivedTraceStep objects."""
    p = Path(trace_path)
    if not p.is_file():
        raise FileNotFoundError(f"Trace file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_steps = data.get("steps", [])
    steps: List[ArchivedTraceStep] = []
    dropped_features = set()

    for idx, raw in enumerate(raw_steps):
        iter_num = int(raw.get("iteration", idx + 1))
        feat = str(raw["selected_feature"])
        sem_type = str(raw["feature_semantic_type"])
        dropped = bool(raw.get("dropped", False))
        prop = raw.get("proposed_transformation")
        acc = raw.get("accepted_transformation")
        num_exp = (
            float(raw["numerical_exponent"])
            if raw.get("numerical_exponent") is not None
            else None
        )
        cat_map = (
            {str(k): str(v) for k, v in raw["categorical_merge_mapping"].items()}
            if raw.get("categorical_merge_mapping") is not None
            else None
        )
        d_phi_before = float(raw["d_phi_before"])
        d_phi_after = (
            float(raw["d_phi_after"]) if raw.get("d_phi_after") is not None else None
        )
        eps = float(raw.get("epsilon", 0.0005))
        stopped = (
            str(raw["stopped_reason"]) if raw.get("stopped_reason") is not None else None
        )

        # Invariant checks on archived trace
        if feat in dropped_features:
            raise IllegalReplayError(
                f"Archived trace contains illegal step on already-dropped feature {feat!r} "
                f"at iteration {iter_num}."
            )
        if sem_type not in ("numerical", "categorical"):
            raise ValueError(f"Unknown feature_semantic_type {sem_type!r} for {feat!r}.")

        if dropped:
            dropped_features.add(feat)

        if sem_type == "numerical" and not dropped:
            if num_exp is None or num_exp <= 0:
                raise ValueError(
                    f"Numerical transform on {feat!r} at iteration {iter_num} has non-positive exponent: {num_exp}"
                )

        steps.append(
            ArchivedTraceStep(
                iteration=iter_num,
                selected_feature=feat,
                feature_semantic_type=sem_type,
                d_phi_before=d_phi_before,
                epsilon=eps,
                proposed_transformation=prop,
                accepted_transformation=acc,
                numerical_exponent=num_exp,
                categorical_merge_mapping=cat_map,
                d_phi_after=d_phi_after,
                dropped=dropped,
                stopped_reason=stopped,
            )
        )

    return steps


def load_all_archived_traces(
    repo_root: Optional[Path] = None,
) -> Dict[str, List[ArchivedTraceStep]]:
    """Load and validate archived traces for all 4 D6 arms from frozen train-val release."""
    root = Path(repo_root or ".")
    base_dir = root / "docs" / "releases" / D6_TRAIN_VAL_RELEASE_ID
    traces: Dict[str, List[ArchivedTraceStep]] = {}

    for arm_id in D6_ARM_IDS:
        trace_path = base_dir / arm_id / "train_fairbias_trace.json"
        arm_steps = load_archived_trace(trace_path)
        expected_len = EXPECTED_ARCHIVED_STEP_COUNTS[arm_id]
        if len(arm_steps) != expected_len:
            raise TerminalReplayBarrierError(
                f"Arm {arm_id} step count mismatch: expected {expected_len}, observed {len(arm_steps)}"
            )
        traces[arm_id] = arm_steps

    return traces


# -----------------------------------------------------------------------------
# Sequential Replay State Machine
# -----------------------------------------------------------------------------
class SequentialReplayStateMachine:
    """State machine executing step-by-step FairBias representation replay.

    Semantics:
    - State 0: Original frozen baseline representation.
    - State k: Formed by applying accepted step k to the representation of state k-1.
    - Categorical merge: Applies archived mapping to the current category state.
    - Numerical power: Follows the frozen transform implementation where the accepted power
      replaces any earlier power (evaluated sign-preserving from the baseline values).
    - Drop: Feature is removed from all subsequent states and cannot reappear.
    - Index & Columns: Non-dropped features preserve their initial column order; row index is strictly preserved.
    """

    def __init__(
        self,
        arm_id: str,
        trace_steps: Sequence[ArchivedTraceStep],
        base_feature_order: Sequence[str],
        num_attrs: Optional[Sequence[str]] = None,
        cate_attrs: Optional[Sequence[str]] = None,
        validate_step_count: bool = True,
    ) -> None:
        self.arm_id = arm_id
        self.trace_steps = list(trace_steps)
        self.base_feature_order = list(base_feature_order)
        self.num_attrs = set(num_attrs or [])
        self.cate_attrs = set(cate_attrs or [])

        # Validate step count if requested and arm_id is recognized
        if validate_step_count:
            expected_steps = EXPECTED_ARCHIVED_STEP_COUNTS.get(arm_id)
            if expected_steps is not None and len(self.trace_steps) != expected_steps:
                raise TerminalReplayBarrierError(
                    f"Arm {arm_id} has {len(self.trace_steps)} steps; expected {expected_steps}."
                )

        # Build state definitions
        self.states: List[RepresentationState] = self._build_representation_states()

    def _build_representation_states(self) -> List[RepresentationState]:
        states: List[RepresentationState] = []
        active_features = list(self.base_feature_order)
        dropped_features = set()
        cumulative_changed_dict: Dict[str, Any] = {}

        # State 0: Baseline
        states.append(
            RepresentationState(
                arm_id=self.arm_id,
                state_index=0,
                preceding_step_index=None,
                selected_feature=None,
                feature_semantic_type=None,
                accepted_transformation=None,
                dropped=None,
                numerical_exponent=None,
                categorical_merge_mapping=None,
                active_feature_count=len(active_features),
                active_features=tuple(active_features),
                cumulative_changed_dict={},
            )
        )

        for step_idx, step in enumerate(self.trace_steps, start=1):
            feat = step.selected_feature
            if feat in dropped_features:
                raise IllegalReplayError(
                    f"Illegal replay in arm {self.arm_id} at step {step_idx}: "
                    f"feature {feat!r} was already dropped and cannot be transformed."
                )
            if feat not in active_features:
                raise IllegalReplayError(
                    f"Illegal replay in arm {self.arm_id} at step {step_idx}: "
                    f"feature {feat!r} is not active in the current representation."
                )

            next_cum_changed = copy.deepcopy(cumulative_changed_dict)
            if step.dropped:
                dropped_features.add(feat)
                active_features = [f for f in active_features if f != feat]
                next_cum_changed[feat] = "dropped"
            elif step.feature_semantic_type == "numerical":
                power = float(step.numerical_exponent)
                next_cum_changed[feat] = {"power": power}
            elif step.feature_semantic_type == "categorical":
                # In the trace, accepted_transformation is the composed category mapping
                next_cum_changed[feat] = step.accepted_transformation
            else:
                raise ValueError(f"Unknown semantic type: {step.feature_semantic_type}")

            cumulative_changed_dict = next_cum_changed

            states.append(
                RepresentationState(
                    arm_id=self.arm_id,
                    state_index=step_idx,
                    preceding_step_index=step_idx - 1,
                    selected_feature=feat,
                    feature_semantic_type=step.feature_semantic_type,
                    accepted_transformation=step.accepted_transformation,
                    dropped=step.dropped,
                    numerical_exponent=step.numerical_exponent,
                    categorical_merge_mapping=step.categorical_merge_mapping,
                    active_feature_count=len(active_features),
                    active_features=tuple(active_features),
                    cumulative_changed_dict=copy.deepcopy(cumulative_changed_dict),
                )
            )

        return states

    def reconstruct_terminal_changed_dict(self) -> Dict[str, Any]:
        """Reconstruct the terminal transformation dictionary implied by the trace."""
        return copy.deepcopy(self.states[-1].cumulative_changed_dict)

    def replay_step(
        self,
        prev_df: pd.DataFrame,
        step: ArchivedTraceStep,
        base_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply a single step to the preceding DataFrame."""
        feat = step.selected_feature
        if feat not in prev_df.columns:
            raise IllegalReplayError(
                f"Cannot apply step {step.iteration} to feature {feat!r}: not present in columns."
            )

        if step.dropped:
            return prev_df.drop(columns=[feat])

        df_out = prev_df.copy()
        transformer = FairTransform()

        if step.feature_semantic_type == "categorical":
            # Apply accepted transformation mapping
            mapping = step.accepted_transformation
            s_transformed = transformer.transform_series(
                prev_df[feat], feat, mapping, is_categorical=True
            )
            df_out[feat] = s_transformed
        elif step.feature_semantic_type == "numerical":
            # Frozen _make_candidate() semantics from fairbias/mitigation.py:
            #   temp_changed[attr] = change
            #   FairTransform.transform_data(raw_X, temp_changed, ...)
            # In the original FairBias training, candidate transforms are applied directly to the
            # baseline raw feature matrix via the updated changed_dict entry. Therefore, numerical
            # revisits update the positive power relative to base_df[feat], rather than compounding
            # powers (i.e. power 3 followed by power 5 produces x^5, not x^15).
            power = float(step.numerical_exponent)
            s_transformed = apply_power_transform(base_df[feat], power)
            df_out[feat] = s_transformed
        else:
            raise ValueError(f"Unknown semantic type: {step.feature_semantic_type}")

        return df_out

    def replay_all(self, base_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Sequentially replay all representation states 0..K on base_df."""
        # Ensure base columns match expected order
        missing = [f for f in self.base_feature_order if f not in base_df.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing base features: {missing}")

        current_df = base_df[self.base_feature_order].copy()
        replayed: Dict[int, pd.DataFrame] = {0: current_df.copy()}

        for k, step in enumerate(self.trace_steps, start=1):
            next_df = self.replay_step(current_df, step, base_df)
            replayed[k] = next_df.copy()
            current_df = next_df

        return replayed


# -----------------------------------------------------------------------------
# Terminal Replay Barrier (Section 7 & 8)
# -----------------------------------------------------------------------------
def verify_terminal_replay_barrier(
    arm_id: str,
    trace_steps: Sequence[ArchivedTraceStep],
    final_changed_dict: Dict[str, Any],
    expected_changed_dict_sha256: str,
    sample_df: Optional[pd.DataFrame] = None,
    num_attrs: Optional[Sequence[str]] = None,
    cate_attrs: Optional[Sequence[str]] = None,
    tolerance: float = MATRIX_NUMERICAL_TOLERANCE,
) -> Dict[str, Any]:
    """Verify that sequential replay through state K reproduces the frozen terminal representation.

    Two-way barrier:
    A. Transformation-state equivalence:
       Reconstructed terminal state dict matches archived final_changed_dict.json and its frozen hash.
    B. Matrix equivalence (if sample_df is provided):
       Sequentially applying all K steps produces the same matrix as applying the canonical FairTransform
       using final_changed_dict, within tolerance.
    """
    # 1. Check step count
    expected_count = EXPECTED_ARCHIVED_STEP_COUNTS.get(arm_id)
    if expected_count is not None and len(trace_steps) != expected_count:
        raise TerminalReplayBarrierError(
            f"Terminal barrier failed for {arm_id}: step count {len(trace_steps)} != expected {expected_count}"
        )

    # 2. Reconstruct terminal changed_dict
    reconstructed: Dict[str, Any] = {}
    for step in trace_steps:
        feat = step.selected_feature
        if step.dropped:
            reconstructed[feat] = "dropped"
        elif step.feature_semantic_type == "numerical":
            reconstructed[feat] = {"power": float(step.numerical_exponent)}
        elif step.feature_semantic_type == "categorical":
            reconstructed[feat] = step.accepted_transformation

    # A. Check dictionary key and value equality
    if set(reconstructed.keys()) != set(final_changed_dict.keys()):
        diff_keys = set(reconstructed.keys()) ^ set(final_changed_dict.keys())
        raise TerminalReplayBarrierError(
            f"Terminal transformation-state equivalence failed for {arm_id}: "
            f"key set mismatch: {diff_keys}"
        )

    for k in final_changed_dict:
        v_exp = final_changed_dict[k]
        v_rec = reconstructed.get(k)
        if v_exp != v_rec:
            raise TerminalReplayBarrierError(
                f"Terminal transformation-state equivalence failed for {arm_id} on {k!r}: "
                f"expected {v_exp}, reconstructed {v_rec}"
            )

    # Check hash equality
    rec_hash = compute_canonical_json_sha256(reconstructed)
    if rec_hash != expected_changed_dict_sha256:
        raise TerminalReplayBarrierError(
            f"Terminal transformation-state hash mismatch for {arm_id}: "
            f"reconstructed {rec_hash} != expected {expected_changed_dict_sha256}"
        )

    # B. Matrix equivalence if sample_df provided
    matrix_eq_verified = False
    max_numeric_diff = 0.0
    if sample_df is not None:
        base_features = list(sample_df.columns)
        r_machine = SequentialReplayStateMachine(
            arm_id=arm_id,
            trace_steps=trace_steps,
            base_feature_order=base_features,
            num_attrs=num_attrs,
            cate_attrs=cate_attrs,
        )
        all_replayed = r_machine.replay_all(sample_df)
        X_k = all_replayed[len(trace_steps)]

        # Canonical terminal transform
        transformer = FairTransform()
        X_term = transformer.transform_data(
            sample_df,
            final_changed_dict,
            num_attrs=list(num_attrs or []),
            cate_attrs=list(cate_attrs or []),
        )

        # Check column names and order
        if list(X_k.columns) != list(X_term.columns):
            raise TerminalReplayBarrierError(
                f"Matrix equivalence failed for {arm_id}: column order mismatch.\n"
                f"Replayed: {list(X_k.columns)}\nCanonical: {list(X_term.columns)}"
            )

        # Check index
        if not X_k.index.equals(X_term.index):
            raise TerminalReplayBarrierError(
                f"Matrix equivalence failed for {arm_id}: index mismatch."
            )

        # Check values
        for c in X_k.columns:
            s_k = X_k[c]
            s_term = X_term[c]
            if c in (num_attrs or []):
                diff = float(np.max(np.abs(s_k.to_numpy() - s_term.to_numpy())))
                if diff > max_numeric_diff:
                    max_numeric_diff = diff
                if diff > tolerance:
                    raise TerminalReplayBarrierError(
                        f"Matrix equivalence failed for {arm_id} on numeric column {c!r}: "
                        f"max diff {diff:.3e} > tolerance {tolerance:.3e}"
                    )
            else:
                # Deterministic categorical column exact equality
                if not s_k.equals(s_term):
                    raise TerminalReplayBarrierError(
                        f"Matrix equivalence failed for {arm_id} on categorical column {c!r}: values do not match exactly."
                    )

        matrix_eq_verified = True

    return {
        "arm_id": arm_id,
        "transformation_state_equivalence": True,
        "reconstructed_changed_dict_sha256": rec_hash,
        "expected_changed_dict_sha256": expected_changed_dict_sha256,
        "matrix_equivalence_verified": matrix_eq_verified,
        "max_numeric_diff": max_numeric_diff,
        "tolerance": tolerance,
    }


# -----------------------------------------------------------------------------
# Intermediate Model Refitting & Anti-Leakage (Section 3, 8, 19)
# -----------------------------------------------------------------------------
def fit_intermediate_model_2022(
    X_train_2022: pd.DataFrame,
    y_train_2022: Union[pd.Series, np.ndarray],
    arm_id: str,
    state_index: int,
) -> Tuple[MinMaxScaler, LogisticRegression]:
    """Fit a new MinMaxScaler and LogisticRegression using 2022 state representation ONLY.

    Guarantees:
    - 2022 data only enters .fit().
    - random_state = 0, solver = 'lbfgs', max_iter = 1000.
    - No survey weights.
    - In-memory only: models are returned to caller and never persisted to disk.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_train_2022)

    y_arr = np.asarray(y_train_2022, dtype=int)
    lr = LogisticRegression(
        random_state=LR_RANDOM_STATE,
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        fit_intercept=LR_FIT_INTERCEPT,
    )
    lr.fit(X_scaled, y_arr)

    return scaler, lr


def predict_proba_fitted_model(
    scaler: MinMaxScaler,
    model: LogisticRegression,
    X: pd.DataFrame,
) -> np.ndarray:
    """Compute predicted positive probabilities for input X using fitted scaler and LR."""
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    return probs


# -----------------------------------------------------------------------------
# Endpoint Model-Reproduction Barrier (Section 8 & 10)
# -----------------------------------------------------------------------------
def verify_endpoint_model_reproduction_barrier(
    arm_id: str,
    state_0_scaler: MinMaxScaler,
    state_0_lr: LogisticRegression,
    state_0_features: Sequence[str],
    state_K_scaler: MinMaxScaler,
    state_K_lr: LogisticRegression,
    state_K_features: Sequence[str],
    archived_2023_baseline_metrics: Dict[str, Any],
    archived_2023_fairbias_metrics: Dict[str, Any],
    archived_2024_baseline_metrics: Dict[str, Any],
    archived_2024_fairbias_metrics: Dict[str, Any],
    observed_2023_state_0_metrics: Dict[str, Any],
    observed_2023_state_K_metrics: Dict[str, Any],
    observed_2024_state_0_metrics: Dict[str, Any],
    observed_2024_state_K_metrics: Dict[str, Any],
    tolerance: float = METRIC_REPRODUCTION_TOLERANCE,
) -> Dict[str, Any]:
    """Verify that fitting state 0 and state K reproduces archived D6 baseline & FairBias models.

    Barrier checks:
    1. State 0 scaler hash == EXPECTED_TRAINING_STATE_ANCHORS[arm_id]["baseline_scaler"]
    2. State 0 LR hash == EXPECTED_TRAINING_STATE_ANCHORS[arm_id]["baseline_LR"]
    3. State K scaler hash == EXPECTED_TRAINING_STATE_ANCHORS[arm_id]["FairBias_scaler"]
    4. State K LR hash == EXPECTED_TRAINING_STATE_ANCHORS[arm_id]["FairBias_LR"]
    5. 2023 baseline metrics: predicted_positive count exact, AUROC/AUPRC/selection_rate abs diff <= 1e-12
    6. 2023 FairBias metrics: predicted_positive count exact, AUROC/AUPRC/selection_rate abs diff <= 1e-12
    7. 2024 baseline metrics: predicted_positive count exact, AUROC/AUPRC/selection_rate abs diff <= 1e-12
    8. 2024 FairBias metrics: predicted_positive count exact, AUROC/AUPRC/selection_rate abs diff <= 1e-12
    """
    anchors = EXPECTED_TRAINING_STATE_ANCHORS[arm_id]

    # Compute hashes of state 0
    s0_scaler_state = extract_minmax_scaler_state(state_0_scaler, state_0_features)
    s0_scaler_sha = compute_canonical_json_sha256(s0_scaler_state)
    s0_lr_state = extract_logistic_regression_state(state_0_lr, state_0_features)
    s0_lr_sha = compute_canonical_json_sha256(s0_lr_state)

    # Compute hashes of state K
    sK_scaler_state = extract_minmax_scaler_state(state_K_scaler, state_K_features)
    sK_scaler_sha = compute_canonical_json_sha256(sK_scaler_state)
    sK_lr_state = extract_logistic_regression_state(state_K_lr, state_K_features)
    sK_lr_sha = compute_canonical_json_sha256(sK_lr_state)

    if s0_scaler_sha != anchors["baseline_scaler"]:
        raise EndpointReproductionBarrierError(
            f"State 0 scaler hash mismatch for {arm_id}: observed {s0_scaler_sha}, expected {anchors['baseline_scaler']}"
        )
    if s0_lr_sha != anchors["baseline_LR"]:
        raise EndpointReproductionBarrierError(
            f"State 0 LR hash mismatch for {arm_id}: observed {s0_lr_sha}, expected {anchors['baseline_LR']}"
        )
    if sK_scaler_sha != anchors["FairBias_scaler"]:
        raise EndpointReproductionBarrierError(
            f"State K scaler hash mismatch for {arm_id}: observed {sK_scaler_sha}, expected {anchors['FairBias_scaler']}"
        )
    if sK_lr_sha != anchors["FairBias_LR"]:
        raise EndpointReproductionBarrierError(
            f"State K LR hash mismatch for {arm_id}: observed {sK_lr_sha}, expected {anchors['FairBias_LR']}"
        )

    # Verification helper for metrics
    comparisons = [
        ("2023 State 0 (Baseline)", observed_2023_state_0_metrics, archived_2023_baseline_metrics.get("utility", archived_2023_baseline_metrics)),
        ("2023 State K (FairBias)", observed_2023_state_K_metrics, archived_2023_fairbias_metrics.get("utility", archived_2023_fairbias_metrics)),
        ("2024 State 0 (Baseline)", observed_2024_state_0_metrics, archived_2024_baseline_metrics.get("utility", archived_2024_baseline_metrics)),
        ("2024 State K (FairBias)", observed_2024_state_K_metrics, archived_2024_fairbias_metrics.get("utility", archived_2024_fairbias_metrics)),
    ]

    for label, obs, exp in comparisons:
        # Check predicted positive count
        obs_pos = int(obs["count_predicted_positive"])
        exp_pos = int(exp["count_predicted_positive"])
        if obs_pos != exp_pos:
            raise EndpointReproductionBarrierError(
                f"{label} for {arm_id}: predicted positive count mismatch: observed {obs_pos}, expected {exp_pos}"
            )

        # Check auroc, auprc, selection_rate within tolerance
        for m in ("auroc", "auprc", "selection_rate"):
            obs_val = float(obs[m])
            exp_val = float(exp[m])
            diff = abs(obs_val - exp_val)
            if diff > tolerance:
                raise EndpointReproductionBarrierError(
                    f"{label} for {arm_id}: {m} diff {diff:.3e} > tolerance {tolerance:.3e} "
                    f"(observed {obs_val}, expected {exp_val})"
                )

    return {
        "arm_id": arm_id,
        "endpoint_reproduction_status": "PASS",
        "state_0_scaler_sha256": s0_scaler_sha,
        "state_0_LR_sha256": s0_lr_sha,
        "state_K_scaler_sha256": sK_scaler_sha,
        "state_K_LR_sha256": sK_lr_sha,
        "metric_tolerance": tolerance,
    }


# -----------------------------------------------------------------------------
# Stepwise Diagnostic Metrics & Score Separation (Section 9)
# -----------------------------------------------------------------------------
def compute_stepwise_metrics(
    arm_id: str,
    state_index: int,
    year: int,
    probs: np.ndarray,
    y_true: Union[pd.Series, np.ndarray],
    X_state: Optional[pd.DataFrame] = None,
    O_df: Optional[pd.DataFrame] = None,
    cate_attrs: Optional[Sequence[str]] = None,
    num_attrs: Optional[Sequence[str]] = None,
    protected_attr_series: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Compute utility, score separation, and representation dependence for a given arm x state x year.

    P1-C Fix: Active semantic lists (active_cate_attrs, active_num_attrs) are strictly filtered
    internally to match X_state.columns so that dropped features never enter calculate_epsilon.
    """
    y_arr = np.asarray(y_true, dtype=int)
    scores = np.asarray(probs, dtype=float)
    n_samples = len(y_arr)

    # 1. Predictions at canonical threshold 0.5
    preds = (scores >= CANONICAL_DECISION_THRESHOLD).astype(int)
    count_pred_pos = int(np.sum(preds))
    sel_rate = float(count_pred_pos / n_samples) if n_samples > 0 else 0.0

    # 2. Utility metrics
    has_two_classes = len(np.unique(y_arr)) == 2
    auroc = float(roc_auc_score(y_arr, scores)) if has_two_classes else None
    auprc = float(average_precision_score(y_arr, scores)) if has_two_classes else None
    acc = float(accuracy_score(y_arr, preds))
    bal_acc = float(balanced_accuracy_score(y_arr, preds))
    f1 = float(f1_score(y_arr, preds, zero_division=0))

    # 3. Score separation metrics
    scores_y0 = scores[y_arr == 0]
    scores_y1 = scores[y_arr == 1]

    if len(scores_y0) > 0 and len(scores_y1) > 0:
        ks_res = scipy.stats.ks_2samp(scores_y0, scores_y1)
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)
    else:
        ks_stat = None
        ks_pval = None

    score_mean = float(np.mean(scores)) if n_samples > 0 else None
    score_median = float(np.median(scores)) if n_samples > 0 else None
    if n_samples > 0:
        q75, q25 = np.percentile(scores, [75, 25])
        score_iqr = float(q75 - q25)
    else:
        score_iqr = None

    y0_mean = float(np.mean(scores_y0)) if len(scores_y0) > 0 else None
    y0_median = float(np.median(scores_y0)) if len(scores_y0) > 0 else None
    y1_mean = float(np.mean(scores_y1)) if len(scores_y1) > 0 else None
    y1_median = float(np.median(scores_y1)) if len(scores_y1) > 0 else None

    # 4. Representation dependence (max d_phi) - P1-C INTERNAL FILTERING
    max_d_phi = None
    if X_state is not None and O_df is not None:
        evaluator = FairEvaluator()
        active_cate_attrs = [f for f in (cate_attrs or []) if f in X_state.columns]
        active_num_attrs = [f for f in (num_attrs or []) if f in X_state.columns]
        eps_dict = evaluator.calculate_epsilon(
            X_state,
            O_df,
            cate_attrs=active_cate_attrs,
            num_attrs=active_num_attrs,
            sample_weight=None,
        )
        if eps_dict:
            max_d_phi = float(max(v for gd in eps_dict.values() for v in gd.values()))
        else:
            max_d_phi = 0.0

    # 5. Optional protected group fairness metrics (Section 13)
    dp_gap = None
    tpr_gap = None
    fpr_gap = None
    eo_max_gap = None

    if protected_attr_series is not None:
        a_arr = np.asarray(protected_attr_series)
        groups = np.unique(a_arr)
        group_sel_rates = {}
        group_tprs = {}
        group_fprs = {}

        for g in groups:
            mask_g = a_arr == g
            y_g = y_arr[mask_g]
            p_g = preds[mask_g]
            if len(y_g) > 0:
                group_sel_rates[g] = float(np.mean(p_g))
            pos_mask = y_g == 1
            if np.sum(pos_mask) > 0:
                group_tprs[g] = float(np.mean(p_g[pos_mask]))
            neg_mask = y_g == 0
            if np.sum(neg_mask) > 0:
                group_fprs[g] = float(np.mean(p_g[neg_mask]))

        if len(group_sel_rates) >= 2:
            dp_gap = float(max(group_sel_rates.values()) - min(group_sel_rates.values()))
        if len(group_tprs) >= 2:
            tpr_gap = float(max(group_tprs.values()) - min(group_tprs.values()))
        if len(group_fprs) >= 2:
            fpr_gap = float(max(group_fprs.values()) - min(group_fprs.values()))
        if tpr_gap is not None and fpr_gap is not None:
            eo_max_gap = float(max(tpr_gap, fpr_gap))

    return {
        "arm_id": arm_id,
        "state_index": state_index,
        "year": year,
        "threshold": CANONICAL_DECISION_THRESHOLD,
        "max_d_phi": max_d_phi,
        "auroc": auroc,
        "auprc": auprc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "accuracy": acc,
        "count_predicted_positive": count_pred_pos,
        "selection_rate": sel_rate,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pval,
        "score_mean": score_mean,
        "score_median": score_median,
        "score_iqr": score_iqr,
        "y0_mean": y0_mean,
        "y0_median": y0_median,
        "y1_mean": y1_mean,
        "y1_median": y1_median,
        "dp_gap": dp_gap,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "equalized_odds_max_gap": eo_max_gap,
    }


# -----------------------------------------------------------------------------
# Stepwise Deltas Computation (Section 10)
# -----------------------------------------------------------------------------
def compute_stepwise_deltas(
    metrics_records: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    """Compute pathwise marginal changes: metric(state k) - metric(state k-1) and delta from baseline.

    Mandatory Nomenclature:
    These differences are strictly designated as "pathwise marginal changes" or
    "pathwise marginal delta", NEVER "causal effect" or "effect holding all else constant".
    """
    df = pd.DataFrame(metrics_records)
    delta_rows: List[Dict[str, Any]] = []

    tracked_metrics = (
        "d_phi",
        "auroc",
        "auprc",
        "ks_statistic",
        "selection_rate",
        "count_predicted_positive",
    )

    # Group by arm and year
    for (arm_id, year), arm_year_df in df.groupby(["arm_id", "year"]):
        arm_year_df = arm_year_df.sort_values("state_index").reset_index(drop=True)
        state_0_row = arm_year_df[arm_year_df["state_index"] == 0].iloc[0]

        for i in range(1, len(arm_year_df)):
            curr_row = arm_year_df.iloc[i]
            prev_row = arm_year_df.iloc[i - 1]
            k = int(curr_row["state_index"])

            delta_record: Dict[str, Any] = {
                "arm_id": arm_id,
                "year": int(year),
                "state_index": k,
                "preceding_state_index": k - 1,
            }

            for m in tracked_metrics:
                col_name = "max_d_phi" if m == "d_phi" else m
                curr_val = curr_row.get(col_name)
                prev_val = prev_row.get(col_name)
                s0_val = state_0_row.get(col_name)

                marginal_delta = (
                    float(curr_val - prev_val)
                    if (curr_val is not None and prev_val is not None)
                    else None
                )
                cum_delta = (
                    float(curr_val - s0_val)
                    if (curr_val is not None and s0_val is not None)
                    else None
                )

                delta_record[f"pathwise_marginal_delta_{m}"] = marginal_delta
                delta_record[f"cumulative_delta_from_baseline_{m}"] = cum_delta

            delta_rows.append(delta_record)

    return pd.DataFrame(delta_rows)


# -----------------------------------------------------------------------------
# Key Path Temporal Summary Generator (Section 13 & 15 Repair)
# -----------------------------------------------------------------------------
def generate_key_path_summary(
    deltas_df: pd.DataFrame,
    traces: Dict[str, List[ArchivedTraceStep]],
) -> Dict[str, Any]:
    """Generate descriptive key path summary identifying largest pathwise changes.

    Repair:
    - Retains separate 2023 and 2024 rankings (does NOT rank 2023 and 2024 together or drop duplicates).
    - For each selected transformation step, records the corresponding delta in BOTH years where available,
      plus same_direction = True/False/None.
    - Descriptive language only (coincides_with, pathwise_association, PI_REVIEW_REQUIRED).
    - Absolutely NO causal assertions or hypotheses declarations.
    """
    summary: Dict[str, Any] = {
        "documentation": (
            "Descriptive pathwise summary identifying transformation steps associated with "
            "the largest absolute pathwise changes in AUROC, AUPRC, KS separation, and selection rate. "
            "These steps coincide with path-dependent shifts along the greedily chosen FairBias trajectory. "
            "All reported metrics remain descriptive and order-dependent."
        ),
        "status": "PI_REVIEW_REQUIRED",
        "causal_claims": False,
        "arms": {},
    }

    metrics_to_rank = [
        ("pathwise_marginal_delta_auroc", "auroc"),
        ("pathwise_marginal_delta_auprc", "auprc"),
        ("pathwise_marginal_delta_ks_statistic", "ks_statistic"),
        ("pathwise_marginal_delta_selection_rate", "selection_rate"),
    ]

    for arm_id in D6_ARM_IDS:
        arm_deltas = deltas_df[deltas_df["arm_id"] == arm_id]
        arm_trace = traces.get(arm_id, [])
        arm_summary: Dict[str, Any] = {
            "total_accepted_steps": len(arm_trace),
            "protected_attribute": ARM_PROTECTED_ATTRIBUTES[arm_id],
        }

        # Build lookup for state_index -> year -> delta
        for delta_col, metric_name in metrics_to_rank:
            if delta_col not in arm_deltas.columns:
                arm_summary[f"largest_2023_pathwise_{metric_name}_changes"] = []
                arm_summary[f"largest_2024_pathwise_{metric_name}_changes"] = []
                continue

            # Separate rankings for 2023 and 2024
            for year in (2023, 2024):
                year_df = arm_deltas[arm_deltas["year"] == year].dropna(subset=[delta_col]).copy()
                if year_df.empty:
                    arm_summary[f"largest_{year}_pathwise_{metric_name}_changes"] = []
                    continue

                year_df["abs_delta"] = year_df[delta_col].abs()
                top_steps = year_df.sort_values("abs_delta", ascending=False).head(3)

                ranked_list: List[Dict[str, Any]] = []
                for _, r in top_steps.iterrows():
                    st_idx = int(r["state_index"])
                    step_obj = arm_trace[st_idx - 1] if 0 < st_idx <= len(arm_trace) else None

                    # Find deltas for both years
                    row_2023 = arm_deltas[(arm_deltas["year"] == 2023) & (arm_deltas["state_index"] == st_idx)]
                    row_2024 = arm_deltas[(arm_deltas["year"] == 2024) & (arm_deltas["state_index"] == st_idx)]

                    d23 = float(row_2023[delta_col].iloc[0]) if not row_2023.empty and pd.notna(row_2023[delta_col].iloc[0]) else None
                    d24 = float(row_2024[delta_col].iloc[0]) if not row_2024.empty and pd.notna(row_2024[delta_col].iloc[0]) else None

                    same_dir = None
                    if d23 is not None and d24 is not None:
                        same_dir = bool((d23 >= 0 and d24 >= 0) or (d23 <= 0 and d24 <= 0))

                    ranked_list.append({
                        "state_index": st_idx,
                        "selected_feature": step_obj.selected_feature if step_obj else None,
                        "feature_semantic_type": step_obj.feature_semantic_type if step_obj else None,
                        "accepted_transformation": (
                            "dropped"
                            if step_obj and step_obj.dropped
                            else (step_obj.accepted_transformation if step_obj else None)
                        ),
                        "ranking_year": year,
                        "delta_2023": d23,
                        "delta_2024": d24,
                        "same_direction": same_dir,
                        "coincides_with": (
                            f"step_{st_idx}_{step_obj.selected_feature}" if step_obj else f"step_{st_idx}"
                        ),
                        "pathwise_association": True,
                        "status": "PI_REVIEW_REQUIRED",
                    })

                arm_summary[f"largest_{year}_pathwise_{metric_name}_changes"] = ranked_list

        summary["arms"][arm_id] = arm_summary

    # Text audit check for forbidden causal terms
    summary_str = json.dumps(summary, sort_keys=True)
    for forbidden in FORBIDDEN_CAUSAL_TERMS:
        if forbidden.lower() in summary_str.lower():
            raise ValueError(f"Forbidden causal terminology detected in summary: {forbidden!r}")

    return summary


# -----------------------------------------------------------------------------
# Fail-Closed Upstream Provenance Verification (P1-A)
# -----------------------------------------------------------------------------
def verify_git_execution_preconditions(
    repo_root: Path,
    expected_sha: str,
) -> Dict[str, Any]:
    """Verify local Git HEAD, tracked worktree cleanliness, and expected execution SHA."""
    root = Path(repo_root)

    # 1. Local HEAD
    res_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if res_head.returncode != 0:
        raise ProvenanceVerificationError(f"Failed to query git HEAD: {res_head.stderr.strip()}")
    current_head = res_head.stdout.strip()
    if current_head != expected_sha:
        raise ProvenanceVerificationError(
            f"Git HEAD mismatch: current {current_head} != expected {expected_sha}"
        )

    # 2. Tracked worktree cleanliness (ignoring untracked files)
    res_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if res_status.returncode != 0:
        raise ProvenanceVerificationError(f"Failed to query git status: {res_status.stderr.strip()}")
    if res_status.stdout.strip():
        raise ProvenanceVerificationError(
            f"Tracked git worktree is dirty:\n{res_status.stdout.strip()}"
        )

    return {
        "git_head": current_head,
        "expected_head": expected_sha,
        "tracked_worktree_clean": True,
    }


def verify_upstream_provenance(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Verify repository HEAD, D7.1 frozen tag, and D6 upstream releases. Fail-closed on any error."""
    root = Path(repo_root or ".")

    def run_git_cmd(cmd: List[str]) -> str:
        res = subprocess.run(
            ["git"] + cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise ProvenanceVerificationError(
                f"Git command {' '.join(cmd)} failed: {res.stderr.strip()}"
            )
        return res.stdout.strip()

    # 1. Verify D7.1 tag object and commit
    d7_1_tag_obj = run_git_cmd(["rev-parse", D7_1_TAG])
    if d7_1_tag_obj != D7_1_TAG_OBJECT:
        raise ProvenanceVerificationError(
            f"D7.1 tag object mismatch: observed {d7_1_tag_obj}, expected {D7_1_TAG_OBJECT}"
        )
    d7_1_commit = run_git_cmd(["rev-parse", f"{D7_1_TAG}^{{commit}}"])
    if d7_1_commit != D7_1_COMMIT:
        raise ProvenanceVerificationError(
            f"D7.1 dereferenced commit mismatch: observed {d7_1_commit}, expected {D7_1_COMMIT}"
        )

    # 2. Verify D6 Train/Val tag object and commit
    d6_tv_tag_obj = run_git_cmd(["rev-parse", D6_TRAIN_VAL_TAG])
    if d6_tv_tag_obj != D6_TRAIN_VAL_TAG_OBJECT:
        raise ProvenanceVerificationError(
            f"D6 train/val tag object mismatch: observed {d6_tv_tag_obj}, expected {D6_TRAIN_VAL_TAG_OBJECT}"
        )
    d6_tv_commit = run_git_cmd(["rev-parse", f"{D6_TRAIN_VAL_TAG}^{{commit}}"])
    if d6_tv_commit != D6_TRAIN_VAL_COMMIT:
        raise ProvenanceVerificationError(
            f"D6 train/val commit mismatch: observed {d6_tv_commit}, expected {D6_TRAIN_VAL_COMMIT}"
        )

    # 3. Verify D6 Test tag object and commit
    d6_test_tag_obj = run_git_cmd(["rev-parse", D6_TEST_TAG])
    if d6_test_tag_obj != D6_TEST_TAG_OBJECT:
        raise ProvenanceVerificationError(
            f"D6 test tag object mismatch: observed {d6_test_tag_obj}, expected {D6_TEST_TAG_OBJECT}"
        )
    d6_test_commit = run_git_cmd(["rev-parse", f"{D6_TEST_TAG}^{{commit}}"])
    if d6_test_commit != D6_TEST_COMMIT:
        raise ProvenanceVerificationError(
            f"D6 test commit mismatch: observed {d6_test_commit}, expected {D6_TEST_COMMIT}"
        )

    # 4. Verify D6 Train/Val manifest and every tracked artifact
    tv_base = root / "docs" / "releases" / D6_TRAIN_VAL_RELEASE_ID
    tv_manifest_path = tv_base / "d6_temporal_train_val_manifest.json"
    if not tv_manifest_path.is_file():
        raise ProvenanceVerificationError(f"Missing D6 train/val manifest: {tv_manifest_path}")
    tv_manifest_sha = compute_sha256(tv_manifest_path)
    if tv_manifest_sha != D6_TRAIN_VAL_MANIFEST_SHA256:
        raise ProvenanceVerificationError(
            f"D6 train/val manifest SHA mismatch: observed {tv_manifest_sha}, expected {D6_TRAIN_VAL_MANIFEST_SHA256}"
        )
    tv_manifest_data = json.loads(tv_manifest_path.read_text(encoding="utf-8"))
    tv_artifacts = tv_manifest_data.get("artifacts", {})
    for rel_p, spec in tv_artifacts.items():
        art_path = tv_base / rel_p
        if not art_path.is_file():
            raise ProvenanceVerificationError(f"Missing D6 train/val artifact: {art_path}")
        if art_path.stat().st_size != spec["size_bytes"]:
            raise ProvenanceVerificationError(
                f"Size mismatch on {art_path}: observed {art_path.stat().st_size}, expected {spec['size_bytes']}"
            )
        art_sha = compute_sha256(art_path)
        if art_sha != spec["sha256"]:
            raise ProvenanceVerificationError(
                f"SHA-256 mismatch on {art_path}: observed {art_sha}, expected {spec['sha256']}"
            )

    # 5. Verify D6 Test manifest and every tracked artifact
    test_base = root / "docs" / "releases" / D6_TEST_RELEASE_ID
    test_manifest_path = test_base / "d6_temporal_test_manifest.json"
    if not test_manifest_path.is_file():
        raise ProvenanceVerificationError(f"Missing D6 test manifest: {test_manifest_path}")
    test_manifest_sha = compute_sha256(test_manifest_path)
    if test_manifest_sha != D6_TEST_MANIFEST_SHA256:
        raise ProvenanceVerificationError(
            f"D6 test manifest SHA mismatch: observed {test_manifest_sha}, expected {D6_TEST_MANIFEST_SHA256}"
        )
    test_manifest_data = json.loads(test_manifest_path.read_text(encoding="utf-8"))
    test_artifacts = test_manifest_data.get("artifacts", {})
    for rel_p, spec in test_artifacts.items():
        art_path = test_base / rel_p
        if not art_path.is_file():
            raise ProvenanceVerificationError(f"Missing D6 test artifact: {art_path}")
        if art_path.stat().st_size != spec["size_bytes"]:
            raise ProvenanceVerificationError(
                f"Size mismatch on {art_path}: observed {art_path.stat().st_size}, expected {spec['size_bytes']}"
            )
        art_sha = compute_sha256(art_path)
        if art_sha != spec["sha256"]:
            raise ProvenanceVerificationError(
                f"SHA-256 mismatch on {art_path}: observed {art_sha}, expected {spec['sha256']}"
            )

    return {
        "d7_1_tag": D7_1_TAG,
        "d7_1_tag_object": d7_1_tag_obj,
        "d7_1_commit": d7_1_commit,
        "d6_train_val_tag": D6_TRAIN_VAL_TAG,
        "d6_train_val_tag_object": d6_tv_tag_obj,
        "d6_train_val_commit": d6_tv_commit,
        "d6_train_val_manifest_sha256": tv_manifest_sha,
        "d6_train_val_artifacts_verified": len(tv_artifacts),
        "d6_test_tag": D6_TEST_TAG,
        "d6_test_tag_object": d6_test_tag_obj,
        "d6_test_commit": d6_test_commit,
        "d6_test_manifest_sha256": test_manifest_sha,
        "d6_test_artifacts_verified": len(test_artifacts),
        "provenance_status": "VERIFIED",
    }


def verify_12_cohort_provenance_barrier(
    observed_digests: Dict[int, Dict[str, str]],
) -> Dict[str, Any]:
    """Verify that all 12 arm x year cohorts match frozen source-row digests."""
    for year in (2022, 2023, 2024):
        if year not in observed_digests:
            raise ProvenanceVerificationError(f"Missing cohort digests for year {year}")
        for arm_id in D6_ARM_IDS:
            if arm_id not in observed_digests[year]:
                raise ProvenanceVerificationError(f"Missing cohort digest for {arm_id} in {year}")
            obs = observed_digests[year][arm_id]
            exp = EXPECTED_COHORT_SOURCE_ROW_DIGESTS[year][arm_id]
            if obs != exp:
                raise ProvenanceVerificationError(
                    f"Cohort source-row digest mismatch for {arm_id} ({year}): "
                    f"observed {obs}, expected {exp}"
                )
    return {"cohort_provenance_barrier": "PASS", "cohorts_verified": 12}


# -----------------------------------------------------------------------------
# D7.2 Stepwise Replay Harness (Audit-Only & Production Guard)
# -----------------------------------------------------------------------------
class NHISD7StepwiseReplayHarness:
    """Harness orchestrating Gate D7.2a/a.1 design audit and verification.

    Guarantees:
    - Zero real NHIS cohort access.
    - Zero model fits in audit-only mode.
    - Zero FairBias relearning / FairBiasMitigation executions.
    - Full verification of archived traces, step counts, and terminal replay equivalence.
    """

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self.repo_root = Path(repo_root or ".")
        self.real_nhis_cohort_accessed = False
        self.nhis_cohort_count = 0
        self.models_fit_count = 0
        self.fairbias_mitigation_count = 0
        self.scientific_release_produced = False

    def run_audit_only(self) -> Dict[str, Any]:
        """Execute strict read-only audit of archived traces and state invariants."""
        # 1. Verify upstream provenance fail-closed
        prov_results = verify_upstream_provenance(self.repo_root)

        # 2. Load archived traces and verify step counts
        traces = load_all_archived_traces(self.repo_root)
        step_counts = {arm: len(steps) for arm, steps in traces.items()}
        total_steps = sum(step_counts.values())

        if step_counts != EXPECTED_ARCHIVED_STEP_COUNTS:
            raise TerminalReplayBarrierError(
                f"Step count distribution mismatch: {step_counts} != {EXPECTED_ARCHIVED_STEP_COUNTS}"
            )
        if total_steps != TOTAL_EXPECTED_ACCEPTED_STEPS:
            raise TerminalReplayBarrierError(
                f"Total accepted steps mismatch: observed {total_steps}, expected {TOTAL_EXPECTED_ACCEPTED_STEPS}"
            )

        # 3. Verify terminal transformation-state equivalence for all 4 arms
        base_dir = self.repo_root / "docs" / "releases" / D6_TRAIN_VAL_RELEASE_ID
        terminal_barrier_results: Dict[str, Any] = {}
        for arm_id in D6_ARM_IDS:
            cd_path = base_dir / arm_id / "final_changed_dict.json"
            cd_dict = json.loads(cd_path.read_text(encoding="utf-8"))
            expected_hash = EXPECTED_TRAINING_STATE_ANCHORS[arm_id]["changed_dict"]

            res = verify_terminal_replay_barrier(
                arm_id=arm_id,
                trace_steps=traces[arm_id],
                final_changed_dict=cd_dict,
                expected_changed_dict_sha256=expected_hash,
                sample_df=None,  # Zero cohort access in audit-only
            )
            terminal_barrier_results[arm_id] = res

        return {
            "scientific_question": D7_2_SCIENTIFIC_QUESTION,
            "mandatory_disclosure": MANDATORY_D7_2_DISCLOSURE,
            "provenance": prov_results,
            "step_counts_by_arm": step_counts,
            "total_accepted_steps": total_steps,
            "total_representation_states": TOTAL_EXPECTED_REPRESENTATION_STATES,
            "terminal_replay_barrier": terminal_barrier_results,
            "real_nhis_cohort_accessed": self.real_nhis_cohort_accessed,
            "nhis_cohort_count": self.nhis_cohort_count,
            "models_fit_count": self.models_fit_count,
            "fairbias_mitigation_count": self.fairbias_mitigation_count,
            "scientific_release_produced": self.scientific_release_produced,
            "d7_2_substantive_execution": "NOT_STARTED",
            "audit_status": "PASS",
        }


# -----------------------------------------------------------------------------
# Release Manager & Manifest Builder (P1-B)
# -----------------------------------------------------------------------------
def build_d7_stepwise_manifest(release_dir: Path) -> Dict[str, Any]:
    """Construct deterministic SHA-256 manifest tracking exactly the 8 primary release artifacts."""
    manifest_records: Dict[str, Dict[str, Any]] = {}
    for filename in D7_2_MANIFEST_TRACKED_ARTIFACTS:
        file_path = release_dir / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Manifest target artifact missing: {file_path}")
        manifest_records[filename] = {
            "sha256": compute_sha256(file_path),
            "size_bytes": file_path.stat().st_size,
        }

    manifest = {
        "release_schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tracked_artifact_count": len(D7_2_MANIFEST_TRACKED_ARTIFACTS),
        "artifacts": manifest_records,
    }
    return manifest


class ProductionD7StepwiseReplayRuntime:
    """Production runtime orchestrating complete 12-cohort, 47-state replay, barriers, and diagnostics."""

    def __init__(
        self,
        repo_root: Path,
        adapter_factory: Optional[Callable[[], Any]] = None,
        train_val_dir: Optional[Path] = None,
        test_dir: Optional[Path] = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.adapter_factory = adapter_factory
        self.train_val_dir = train_val_dir or (
            self.repo_root / "docs" / "releases" / D6_TRAIN_VAL_RELEASE_ID
        )
        self.test_dir = test_dir or (
            self.repo_root / "docs" / "releases" / D6_TEST_RELEASE_ID
        )

        self.adapter: Optional[Any] = None
        self.traces: Dict[str, List[ArchivedTraceStep]] = {}
        self.raw_cohorts: Dict[int, Dict[str, Dict[str, Any]]] = {2022: {}, 2023: {}, 2024: {}}
        self.cohort_digests: Dict[int, Dict[str, str]] = {2022: {}, 2023: {}, 2024: {}}
        self.observed_preprocessing_sha256: Optional[str] = None
        self.replayed_states: Dict[str, Dict[int, Dict[int, pd.DataFrame]]] = {}
        self.fitted_models: Dict[str, Dict[int, Tuple[MinMaxScaler, LogisticRegression]]] = {}
        self.endpoint_barrier_results: Dict[str, Any] = {}
        self.terminal_matrix_results: Dict[str, Any] = {}
        self.metrics_records: List[Dict[str, Any]] = []
        self.deltas_df: Optional[pd.DataFrame] = None
        self.key_path_summary: Optional[Dict[str, Any]] = None

    def construct_adapter(self) -> Any:
        """Construct real NHISStudyAdapter or test adapter factory, verifying preprocessing anchor."""
        if self.adapter_factory is not None:
            self.adapter = self.adapter_factory()
        else:
            from nhis_fairbias.adapter import NHISStudyAdapter
            pq_path = self.repo_root / FROZEN_FEATURES_PARQUET_PATH
            if not pq_path.is_file():
                raise FileNotFoundError(f"Prepared features parquet not found: {pq_path}")
            pq_sha = compute_sha256(pq_path)
            if pq_sha != FROZEN_FEATURES_PARQUET_SHA256:
                raise ProvenanceVerificationError(
                    f"Features parquet SHA mismatch: observed {pq_sha}, expected {FROZEN_FEATURES_PARQUET_SHA256}"
                )
            self.adapter = NHISStudyAdapter(features_parquet_path=pq_path)

        if not hasattr(self.adapter, "preprocessor") or getattr(self.adapter.preprocessor, "fitted_record", None) is None:
            raise ProvenanceVerificationError("Adapter preprocessor missing required fitted_record")

        fitted_rec = self.adapter.preprocessor.fitted_record
        if hasattr(fitted_rec, "to_dict") and callable(fitted_rec.to_dict):
            rec_dict = fitted_rec.to_dict()
        elif isinstance(fitted_rec, dict):
            rec_dict = fitted_rec
        else:
            rec_dict = dict(fitted_rec)

        obs_hash = compute_canonical_json_sha256(rec_dict)
        if obs_hash != PREPROCESSING_STATE_SHA256:
            raise ProvenanceVerificationError(
                f"Preprocessing state hash mismatch: observed {obs_hash}, expected {PREPROCESSING_STATE_SHA256}"
            )
        self.observed_preprocessing_sha256 = obs_hash
        return self.adapter

    def build_all_cohorts(self) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """Construct 12 cohorts and enforce global 12/12 source-row digest barrier."""
        if self.adapter is None:
            self.construct_adapter()

        for year in TEMPORAL_YEARS:
            for arm_id in D6_ARM_IDS:
                prot_attr = ARM_PROTECTED_ATTRIBUTES[arm_id]
                disab_policy = ARM_DISABILITY_POLICIES[arm_id]
                X_orig, y, a, w, meta = self.adapter.get_cohort(
                    year=year,
                    outcome="MEDDL12M_A",
                    protected_attribute=prot_attr,
                    feature_set="primary_core",
                    disability_arm=disab_policy,
                )
                digest = compute_cohort_source_row_digest(year, X_orig.index)
                self.cohort_digests[year][arm_id] = digest
                self.raw_cohorts[year][arm_id] = {
                    "X": X_orig,
                    "y": y,
                    "a": a,
                    "w": w,
                    "digest": digest,
                }

        # Enforce 12/12 provenance barrier
        verify_12_cohort_provenance_barrier(self.cohort_digests)
        return self.raw_cohorts

    def replay_and_verify_terminal_matrices(self) -> Dict[str, Any]:
        """Replay representations for all arms x years and verify 12/12 terminal matrix barrier."""
        self.traces = load_all_archived_traces(self.repo_root)
        transformer = FairTransform()

        for arm_id in D6_ARM_IDS:
            self.replayed_states[arm_id] = {}
            steps = self.traces[arm_id]
            cd_path = self.train_val_dir / arm_id / "final_changed_dict.json"
            final_cd = json.loads(cd_path.read_text(encoding="utf-8"))

            for year in TEMPORAL_YEARS:
                cohort_dict = self.raw_cohorts[year][arm_id]
                X_orig = cohort_dict["X"]

                all_cats, all_nums = self.adapter.preprocessor.get_feature_family_lists("primary_core")
                active_cols = set(X_orig.columns)
                cate_attrs = [c for c in all_cats if c in active_cols]
                num_attrs = [c for c in all_nums if c in active_cols]

                r_machine = SequentialReplayStateMachine(
                    arm_id=arm_id,
                    trace_steps=steps,
                    base_feature_order=list(X_orig.columns),
                    num_attrs=num_attrs,
                    cate_attrs=cate_attrs,
                    validate_step_count=True,
                )
                replayed = r_machine.replay_all(X_orig)
                self.replayed_states[arm_id][year] = replayed

                # Verify terminal matrix equivalence for this arm x year (12 total checks)
                X_k = replayed[len(steps)]
                X_term = transformer.transform_data(
                    X_orig,
                    final_cd,
                    num_attrs=num_attrs,
                    cate_attrs=cate_attrs,
                )

                if list(X_k.columns) != list(X_term.columns):
                    raise TerminalReplayBarrierError(
                        f"Terminal matrix column order mismatch for {arm_id} ({year})"
                    )
                if not X_k.index.equals(X_term.index):
                    raise TerminalReplayBarrierError(
                        f"Terminal matrix index mismatch for {arm_id} ({year})"
                    )

                max_diff = 0.0
                for col in X_k.columns:
                    s_k = X_k[col]
                    s_t = X_term[col]
                    if col in num_attrs:
                        d = float(np.max(np.abs(s_k.to_numpy() - s_t.to_numpy())))
                        if d > max_diff:
                            max_diff = d
                        if d > MATRIX_NUMERICAL_TOLERANCE:
                            raise TerminalReplayBarrierError(
                                f"Terminal matrix numeric diff {d:.3e} > tolerance on {col} for {arm_id} ({year})"
                            )
                    else:
                        if not s_k.equals(s_t):
                            raise TerminalReplayBarrierError(
                                f"Terminal matrix categorical mismatch on {col} for {arm_id} ({year})"
                            )

                self.terminal_matrix_results[f"{arm_id}_{year}"] = {
                    "verified": True,
                    "max_numeric_diff": max_diff,
                }

        return self.terminal_matrix_results

    def fit_and_verify_endpoints(self) -> Dict[str, Any]:
        """Fit state 0 and state K on 2022 for all arms, enforcing endpoint reproduction BEFORE intermediates."""
        for arm_id in D6_ARM_IDS:
            steps = self.traces[arm_id]
            k_terminal = len(steps)
            self.fitted_models[arm_id] = {}

            # 2022 Data for arm
            X_2022_0 = self.replayed_states[arm_id][2022][0]
            X_2022_K = self.replayed_states[arm_id][2022][k_terminal]
            y_2022 = self.raw_cohorts[2022][arm_id]["y"]

            # Fit state 0 on 2022 only
            scaler_0, lr_0 = fit_intermediate_model_2022(X_2022_0, y_2022, arm_id, 0)
            self.fitted_models[arm_id][0] = (scaler_0, lr_0)

            # Fit state K on 2022 only
            scaler_K, lr_K = fit_intermediate_model_2022(X_2022_K, y_2022, arm_id, k_terminal)
            self.fitted_models[arm_id][k_terminal] = (scaler_K, lr_K)

            # Evaluate state 0 and state K on 2023 and 2024
            obs_2023_0 = self._evaluate_model(arm_id, 0, 2023, scaler_0, lr_0)
            obs_2023_K = self._evaluate_model(arm_id, k_terminal, 2023, scaler_K, lr_K)
            obs_2024_0 = self._evaluate_model(arm_id, 0, 2024, scaler_0, lr_0)
            obs_2024_K = self._evaluate_model(arm_id, k_terminal, 2024, scaler_K, lr_K)

            # Load archived baseline and FairBias metrics
            val_base_path = self.train_val_dir / arm_id / "validation_metrics_baseline.json"
            val_fb_path = self.train_val_dir / arm_id / "validation_metrics_fairbias.json"
            test_base_path = self.test_dir / arm_id / "test_metrics_baseline.json"
            test_fb_path = self.test_dir / arm_id / "test_metrics_fairbias.json"

            arch_2023_base = json.loads(val_base_path.read_text(encoding="utf-8"))
            arch_2023_fb = json.loads(val_fb_path.read_text(encoding="utf-8"))
            arch_2024_base = json.loads(test_base_path.read_text(encoding="utf-8"))
            arch_2024_fb = json.loads(test_fb_path.read_text(encoding="utf-8"))

            ep_res = verify_endpoint_model_reproduction_barrier(
                arm_id=arm_id,
                state_0_scaler=scaler_0,
                state_0_lr=lr_0,
                state_0_features=list(X_2022_0.columns),
                state_K_scaler=scaler_K,
                state_K_lr=lr_K,
                state_K_features=list(X_2022_K.columns),
                archived_2023_baseline_metrics=arch_2023_base,
                archived_2023_fairbias_metrics=arch_2023_fb,
                archived_2024_baseline_metrics=arch_2024_base,
                archived_2024_fairbias_metrics=arch_2024_fb,
                observed_2023_state_0_metrics=obs_2023_0,
                observed_2023_state_K_metrics=obs_2023_K,
                observed_2024_state_0_metrics=obs_2024_0,
                observed_2024_state_K_metrics=obs_2024_K,
            )
            self.endpoint_barrier_results[arm_id] = ep_res

        return self.endpoint_barrier_results

    def fit_and_evaluate_all_intermediate_states(self) -> List[Dict[str, Any]]:
        """Fit intermediate models 1..K-1 on 2022 and evaluate all 47 states on 2022, 2023, and 2024."""
        # Check endpoint barrier was already verified
        if len(self.endpoint_barrier_results) != len(D6_ARM_IDS):
            raise EndpointReproductionBarrierError("Cannot fit intermediate models: endpoint barrier not verified.")

        self.metrics_records = []

        for arm_id in D6_ARM_IDS:
            steps = self.traces[arm_id]
            k_terminal = len(steps)
            y_2022 = self.raw_cohorts[2022][arm_id]["y"]

            all_cats, all_nums = self.adapter.preprocessor.get_feature_family_lists("primary_core")

            for k in range(k_terminal + 1):
                if k in (0, k_terminal):
                    scaler_k, lr_k = self.fitted_models[arm_id][k]
                else:
                    X_2022_k = self.replayed_states[arm_id][2022][k]
                    scaler_k, lr_k = fit_intermediate_model_2022(X_2022_k, y_2022, arm_id, k)
                    self.fitted_models[arm_id][k] = (scaler_k, lr_k)

                # Evaluate on 2022, 2023, 2024
                for year in TEMPORAL_YEARS:
                    X_year_k = self.replayed_states[arm_id][year][k]
                    y_year = self.raw_cohorts[year][arm_id]["y"]
                    a_year = self.raw_cohorts[year][arm_id]["a"]
                    probs_year = predict_proba_fitted_model(scaler_k, lr_k, X_year_k)

                    O_df = pd.DataFrame({ARM_PROTECTED_ATTRIBUTES[arm_id]: a_year})
                    metrics = compute_stepwise_metrics(
                        arm_id=arm_id,
                        state_index=k,
                        year=year,
                        probs=probs_year,
                        y_true=y_year,
                        X_state=X_year_k,
                        O_df=O_df,
                        cate_attrs=all_cats,
                        num_attrs=all_nums,
                        protected_attr_series=a_year,
                    )
                    self.metrics_records.append(metrics)

        self.deltas_df = compute_stepwise_deltas(self.metrics_records)
        self.key_path_summary = generate_key_path_summary(self.deltas_df, self.traces)
        return self.metrics_records

    def _evaluate_model(
        self,
        arm_id: str,
        state_index: int,
        year: int,
        scaler: MinMaxScaler,
        lr: LogisticRegression,
    ) -> Dict[str, Any]:
        X_df = self.replayed_states[arm_id][year][state_index]
        y_series = self.raw_cohorts[year][arm_id]["y"]
        probs = predict_proba_fitted_model(scaler, lr, X_df)
        return compute_stepwise_metrics(
            arm_id=arm_id,
            state_index=state_index,
            year=year,
            probs=probs,
            y_true=y_series,
        )


class D7StepwiseReplayReleaseManager:
    """Manager for substantive D7.2 execution enforcing strict preflight and release lifecycle."""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        releases_parent_dir: Optional[Path] = None,
        adapter_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.repo_root = Path(repo_root or ".")
        self.releases_parent_dir = releases_parent_dir or (
            self.repo_root / "runs" / "nhis_d7_stepwise_replay" / "releases"
        )
        self.adapter_factory = adapter_factory

    def execute_release(
        self,
        release_id: str,
        expected_execution_head: str,
    ) -> Dict[str, Any]:
        """Execute complete substantive D7.2 analysis."""
        # 1. Preconditions BEFORE directory creation
        verify_git_execution_preconditions(
            repo_root=self.repo_root,
            expected_sha=expected_execution_head,
        )
        verify_upstream_provenance(self.repo_root)

        # 2. Collision check
        target_release_dir = self.releases_parent_dir / release_id
        if target_release_dir.exists():
            raise FileExistsError(f"Release directory collision: {target_release_dir} already exists.")

        # 3. Create directory & STARTED state
        target_release_dir.mkdir(parents=True, exist_ok=False)
        state_file = target_release_dir / "release_state.json"
        state_payload: Dict[str, Any] = {
            "release_id": release_id,
            "status": "STARTED",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "execution_head": expected_execution_head,
            "manifest_sha256": None,
            "error": None,
        }
        state_file.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")

        runtime = ProductionD7StepwiseReplayRuntime(
            repo_root=self.repo_root,
            adapter_factory=self.adapter_factory,
        )

        try:
            # 4. Construct cohorts & enforce 12/12 cohort barrier
            runtime.build_all_cohorts()

            # 5. Replay representation states & enforce 12/12 terminal matrix barrier
            runtime.replay_and_verify_terminal_matrices()

            # 6. Fit & verify endpoint reproduction BEFORE any intermediate models
            runtime.fit_and_verify_endpoints()

            # 7. Fit & evaluate intermediate states (states 1..K-1 on 2022 only)
            runtime.fit_and_evaluate_all_intermediate_states()

            # 8. Write all 10 release artifacts
            self._write_release_artifacts(target_release_dir, release_id, expected_execution_head, runtime)

            # 9. Build and verify manifest tracking the 8 primary artifacts
            manifest = build_d7_stepwise_manifest(target_release_dir)
            manifest_file = target_release_dir / "d7_stepwise_manifest.json"
            manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            manifest_sha = compute_sha256(manifest_file)

            # 10. Update release_state.json to COMPLETE
            state_payload["status"] = "COMPLETE"
            state_payload["completed_at"] = datetime.now(timezone.utc).isoformat()
            state_payload["manifest_sha256"] = manifest_sha
            state_file.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")

            return {
                "release_id": release_id,
                "release_dir": str(target_release_dir),
                "manifest_sha256": manifest_sha,
                "status": "COMPLETE",
            }

        except Exception as exc:
            # On failure: preserve directory and write FAILED state
            state_payload["status"] = "FAILED"
            state_payload["completed_at"] = datetime.now(timezone.utc).isoformat()
            state_payload["error"] = str(exc)
            state_file.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
            raise

    def _write_release_artifacts(
        self,
        release_dir: Path,
        release_id: str,
        execution_head: str,
        runtime: ProductionD7StepwiseReplayRuntime,
    ) -> None:
        """Write all 8 manifest-tracked release artifacts."""
        # 1. provenance_summary.json
        prov_summary = {
            "release_id": release_id,
            "execution_head": execution_head,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_question": D7_2_SCIENTIFIC_QUESTION,
            "mandatory_disclosure": MANDATORY_D7_2_DISCLOSURE,
            "scientific_terminology": D7_2_SCIENTIFIC_TERMINOLOGY,
            "refit_configuration": {
                "D7_2_refits_intermediate_classifiers": D7_2_REFITS_INTERMEDIATE_CLASSIFIERS,
                "training_year": TRAINING_YEAR,
                "validation_year": VALIDATION_YEAR,
                "descriptive_test_year": DESCRIPTIVE_TEST_YEAR,
                "threshold": CANONICAL_DECISION_THRESHOLD,
                "solver": LR_SOLVER,
                "max_iter": LR_MAX_ITER,
                "random_state": LR_RANDOM_STATE,
                "survey_weights": SURVEY_WEIGHTS,
            },
            "observed_cohort_source_row_digests": runtime.cohort_digests,
            "expected_cohort_source_row_digests": EXPECTED_COHORT_SOURCE_ROW_DIGESTS,
            "observed_preprocessing_sha256": runtime.observed_preprocessing_sha256,
            "expected_preprocessing_sha256": PREPROCESSING_STATE_SHA256,
        }
        (release_dir / "provenance_summary.json").write_text(
            json.dumps(prov_summary, indent=2, sort_keys=True), encoding="utf-8"
        )

        # 2. archived_trace_inventory.json
        trace_inv = {
            arm_id: [s.to_dict() for s in steps]
            for arm_id, steps in runtime.traces.items()
        }
        (release_dir / "archived_trace_inventory.json").write_text(
            json.dumps(trace_inv, indent=2, sort_keys=True), encoding="utf-8"
        )

        # 3. terminal_replay_verification.json
        (release_dir / "terminal_replay_verification.json").write_text(
            json.dumps(runtime.terminal_matrix_results, indent=2, sort_keys=True), encoding="utf-8"
        )

        # 4. endpoint_reproduction.json
        (release_dir / "endpoint_reproduction.json").write_text(
            json.dumps(runtime.endpoint_barrier_results, indent=2, sort_keys=True), encoding="utf-8"
        )

        # 5. stepwise_metrics.csv
        metrics_df = pd.DataFrame(runtime.metrics_records)
        primary_metrics_cols = [
            "arm_id", "state_index", "year", "threshold", "max_d_phi",
            "auroc", "auprc", "balanced_accuracy", "f1", "accuracy",
            "count_predicted_positive", "selection_rate",
            "ks_statistic", "ks_pvalue", "score_mean", "score_median", "score_iqr",
            "y0_mean", "y0_median", "y1_mean", "y1_median",
        ]
        metrics_df[primary_metrics_cols].to_csv(release_dir / "stepwise_metrics.csv", index=False)

        # 6. stepwise_deltas.csv
        runtime.deltas_df.to_csv(release_dir / "stepwise_deltas.csv", index=False)

        # 7. protected_fairness_stepwise.csv
        fairness_cols = [
            "arm_id", "state_index", "year", "dp_gap", "tpr_gap", "fpr_gap", "equalized_odds_max_gap"
        ]
        metrics_df[fairness_cols].to_csv(release_dir / "protected_fairness_stepwise.csv", index=False)

        # 8. key_path_summary.json
        (release_dir / "key_path_summary.json").write_text(
            json.dumps(runtime.key_path_summary, indent=2, sort_keys=True), encoding="utf-8"
        )
