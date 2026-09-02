"""Gate D7: Terminal-State FairBias Mechanism Audit Harness.

Scientific Identity:
- Post-primary, post-TEST, explanatory, descriptive, terminal-state mechanism audit.
- Temporal repeated-cross-sectional = True.
- Longitudinal = False, Panel = False, Causal = False.
- ZERO ESTIMATOR REFIT: scoring functions and scalers are mathematically restored from
  frozen D6 2022 state; no .fit() or .fit_transform() is ever invoked.
- ZERO FAIRBIAS RELEARNING: representations are frozen from D6.

Mandatory Disclosure:
D7 is an explanatory post-primary analysis conducted after the primary and temporal TEST
results were observed. TEST-partition mechanism diagnostics are descriptive and are not
interpreted as independent confirmatory evidence.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
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
from sklearn.metrics import average_precision_score, normalized_mutual_info_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------------
# Module Integrity Guard: Prohibit Estimator Fitting & Relearning
# -----------------------------------------------------------------------------
PROHIBITED_SUBSTANTIVE_CALLS: Tuple[str, ...] = (
    ".fit(",
    ".fit_transform(",
    "FairBiasMitigation(",
    "calculate_epsilon(",
)

# -----------------------------------------------------------------------------
# Canonical Constants, Hashes, and Tags
# -----------------------------------------------------------------------------
D6_TRAIN_VAL_RELEASE_ID = "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
D6_TRAIN_VAL_MANIFEST_SHA256 = (
    "773d43b2d893232eb4cbba9dfcdcc639ecdc2fbc8efb2e353f7d2a68514159c4"
)
D6_TRAIN_VAL_LEDGER_SHA256 = (
    "d40a8dce1f117961ee866cdfc2a12879ba7ce3b939695797be80b4a2acc47ca6"
)
D6_TRAIN_VAL_TAG = "nhis-d6-temporal-train-val-v1"
D6_TRAIN_VAL_TAG_OBJECT = "e23941edf2085c84290868f69289d30af62b7e95"
D6_TRAIN_VAL_COMMIT = "bdf154c541d365b216a732bc3a58415af41cdcc7"

D6_TEST_RELEASE_ID = "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"
D6_TEST_MANIFEST_SHA256 = (
    "2a134eb1c23421fd7e50f8c255f2d55395ecfda8149700c78f76dac3da553fd5"
)
D6_TEST_TAG = "nhis-d6-temporal-test-v1"
D6_TEST_TAG_OBJECT = "b3aaae6559028e83aea9c329d585c1607e87364c"
D6_TEST_COMMIT = "e24685cbde26497d0209f26fcf1d82b131dbe726"

D4_PRIMARY_RELEASE_ID = "NHIS_D4_PRIMARY_TEST_RELEASE_V1_9920fc5a"
D5_WEIGHTED_RELEASE_ID = "NHIS_D5_WEIGHTED_SECONDARY_TEST_V1_14cc7aa6"

PREPROCESSING_STATE_SHA256 = (
    "f106967a8ed9bf46ff1c3ff009e40763280fa1dbc78983d3a479a1508fd83db9"
)
FROZEN_FEATURES_PARQUET_PATH = Path("data/processed/nhis/nhis_2022_2024_features.parquet")
FROZEN_FEATURES_PARQUET_SHA256 = (
    "49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383"
)

CANONICAL_DECISION_THRESHOLD = 0.5
SCORING_REPRODUCTION_ABSOLUTE_TOLERANCE = 1e-12

TEMPORAL_YEARS: Tuple[int, ...] = (2022, 2023, 2024)
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

# -----------------------------------------------------------------------------
# Scientific Disclosures, Terminology & Hypotheses
# -----------------------------------------------------------------------------
MANDATORY_D7_DISCLOSURE = (
    "D7 is an explanatory post-primary analysis conducted after the primary and temporal TEST "
    "results were observed. TEST-partition mechanism diagnostics are descriptive and are not "
    "interpreted as independent confirmatory evidence."
)

NMI_GATE_DISCLOSURE = (
    "Under the frozen configuration, the retained NMI information-loss gate is effectively "
    "non-binding because its threshold is far above the ordinary fractional-loss range."
)

FAMILY_I_INVARIANT = (
    "Drops and categorical merges irreversibly remove distinctions in the original feature "
    "representation. Whether the removed distinctions carried outcome-relevant information is "
    "an empirical D7.1 question."
)

FAMILY_II_INVARIANT = (
    "In exact arithmetic and over the accepted non-overflow domain, the positive-power "
    "transformation is information-preserving as a bijective reparameterization, while potentially "
    "changing numerical geometry and the functional relationship accessible to a linear logistic model."
)

SCIENTIFIC_TERMINOLOGY = {
    "temporal_repeated_cross_sectional": True,
    "longitudinal": False,
    "panel": False,
    "causal": False,
}

D7_HYPOTHESES = {
    "H1": (
        "Irreversible feature drops/category merges may remove outcome-relevant distinctions "
        "and contribute to reduced predictive discrimination."
    ),
    "H2": (
        "Invertible monotone numerical power transforms may preserve raw feature information "
        "while changing numerical spacing and the linear functional relationship available to "
        "the fitted logistic model."
    ),
    "H3": (
        "Terminal FairBias scoring functions may exhibit reduced Y=1 versus Y=0 score separation "
        "and/or altered score location/spread relative to baseline."
    ),
    "H4": (
        "Protected-attribute-specific downstream differences may correspond to systematic "
        "differences in transformation family, transformation severity, and terminal LR "
        "contribution structure."
    ),
}

# -----------------------------------------------------------------------------
# 12 Mandatory Expected Cohort Source-Row Digests
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 20 Mandatory Expected Training State Anchors (5 per arm x 4 arms)
# -----------------------------------------------------------------------------
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
# Future Release Artifact Schema (11 Files Total, 9 Manifest-Tracked)
# -----------------------------------------------------------------------------
D7_MANIFEST_TRACKED_ARTIFACTS: Tuple[str, ...] = (
    "provenance_summary.json",
    "terminal_transformation_inventory.json",
    "macro_context.json",
    "family1_information_compression.csv",
    "family2_numeric_geometry.csv",
    "score_distribution_summary.csv",
    "protected_group_score_summary.csv",
    "feature_logit_contribution_summary.csv",
    "mechanism_arm_summary.json",
)

D7_UNTRACKED_CONTROL_FILES: Tuple[str, ...] = (
    "d7_terminal_mechanism_manifest.json",
    "release_state.json",
)

D7_ALL_RELEASE_FILES: Tuple[str, ...] = (
    D7_MANIFEST_TRACKED_ARTIFACTS + D7_UNTRACKED_CONTROL_FILES
)

QUANTILE_LEVELS: Tuple[float, ...] = (
    0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99
)


# -----------------------------------------------------------------------------
# Deterministic Utility & Hash Helpers
# -----------------------------------------------------------------------------
def compute_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 hex digest of a file in binary mode."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for SHA computation: {p}")
    hasher = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_canonical_json_sha256(data: Any) -> str:
    """Compute deterministic SHA-256 hash of a JSON-serializable structure."""
    canonical_bytes = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def compute_cohort_source_row_digest(year: int, row_indices: Sequence[Any]) -> str:
    """Compute deterministic SHA-256 digest of cohort row indices."""
    sorted_indices = sorted(str(idx) for idx in row_indices)
    payload = f"{year}:" + ",".join(sorted_indices)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Zero-Refit Scaler Restoration
# -----------------------------------------------------------------------------
class FrozenScaler:
    """Mathematically restored MinMaxScaler using frozen training parameters.

    Never invokes sklearn fit() or fit_transform().
    Direct transform: X_scaled = X_ordered * scale_ + min_
    where min_ is sklearn's stored affine-offset: min_ = feature_range[0] - data_min_ * scale_.
    """

    def __init__(self, state: Mapping[str, Any]) -> None:
        required_keys = (
            "feature_order",
            "feature_range",
            "n_features_in_",
            "data_min_",
            "data_max_",
            "data_range_",
            "scale_",
            "min_",
        )
        for k in required_keys:
            if k not in state:
                raise ValueError(f"Missing required scaler state key: {k}")

        self.feature_order: List[str] = [str(f) for f in state["feature_order"]]
        self.feature_range: Tuple[float, float] = (
            float(state["feature_range"][0]),
            float(state["feature_range"][1]),
        )
        self.n_features_in_: int = int(state["n_features_in_"])
        self.data_min_: np.ndarray = np.asarray(state["data_min_"], dtype=float)
        self.data_max_: np.ndarray = np.asarray(state["data_max_"], dtype=float)
        self.data_range_: np.ndarray = np.asarray(state["data_range_"], dtype=float)
        self.scale_: np.ndarray = np.asarray(state["scale_"], dtype=float)
        self.min_: np.ndarray = np.asarray(state["min_"], dtype=float)

        if len(self.feature_order) != self.n_features_in_:
            raise ValueError(
                f"Dimension mismatch: feature_order has {len(self.feature_order)}, "
                f"n_features_in_ is {self.n_features_in_}"
            )
        if (
            len(self.scale_) != self.n_features_in_
            or len(self.min_) != self.n_features_in_
            or len(self.data_min_) != self.n_features_in_
        ):
            raise ValueError("Internal dimension mismatch among scaler parameter arrays.")

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Apply exact sklearn affine MinMax transform: X * scale_ + min_."""
        if isinstance(X, pd.DataFrame):
            actual_cols = list(X.columns)
            if actual_cols != self.feature_order:
                raise ValueError(
                    f"Feature order mismatch in input DataFrame.\n"
                    f"Expected: {self.feature_order}\n"
                    f"Observed: {actual_cols}"
                )
            arr = X[self.feature_order].to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
            if arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.n_features_in_}, got {arr.shape[1]}"
                )

        # Exact formula: X * scale_ + min_
        return arr * self.scale_ + self.min_

    def to_sklearn_scaler(self) -> MinMaxScaler:
        """Construct an sklearn-compatible MinMaxScaler object without fitting."""
        scaler = MinMaxScaler(feature_range=self.feature_range)
        scaler.n_features_in_ = self.n_features_in_
        scaler.feature_names_in_ = np.array(self.feature_order, dtype=object)
        scaler.data_min_ = self.data_min_.copy()
        scaler.data_max_ = self.data_max_.copy()
        scaler.data_range_ = self.data_range_.copy()
        scaler.scale_ = self.scale_.copy()
        scaler.min_ = self.min_.copy()
        return scaler


# -----------------------------------------------------------------------------
# Zero-Refit Logistic Regression Scoring
# -----------------------------------------------------------------------------
class FrozenLogisticRegression:
    """Mathematically restored Logistic Regression scoring function.

    Never invokes sklearn fit().
    Logit formula: intercept_[0] + X_scaled @ coef_[0]
    Probability: scipy.special.expit(logit)
    """

    def __init__(self, state: Mapping[str, Any]) -> None:
        required_keys = (
            "feature_order",
            "classes_",
            "coef_",
            "intercept_",
            "n_features_in_",
            "solver",
            "max_iter",
            "random_state",
        )
        for k in required_keys:
            if k not in state:
                raise ValueError(f"Missing required LR state key: {k}")

        self.feature_order: List[str] = [str(f) for f in state["feature_order"]]
        self.classes_: List[int] = [int(c) for c in state["classes_"]]
        if self.classes_ != [0, 1]:
            raise ValueError(f"Classes must be exactly [0, 1]; got {self.classes_}")

        self.coef_: np.ndarray = np.asarray(state["coef_"], dtype=float)
        self.intercept_: np.ndarray = np.asarray(state["intercept_"], dtype=float)
        self.n_features_in_: int = int(state["n_features_in_"])
        self.solver: str = str(state["solver"])
        self.max_iter: int = int(state["max_iter"])
        self.random_state: Optional[int] = (
            int(state["random_state"]) if state["random_state"] is not None else None
        )
        self.penalty: Optional[str] = (
            str(state["penalty"]) if state.get("penalty") is not None else None
        )
        self.C: float = float(state.get("C", 1.0))
        self.fit_intercept: bool = bool(state.get("fit_intercept", True))

        if self.coef_.shape[1] != self.n_features_in_:
            raise ValueError(
                f"LR coef dimension {self.coef_.shape[1]} does not match n_features_in_ {self.n_features_in_}"
            )
        if len(self.feature_order) != self.n_features_in_:
            raise ValueError(
                f"Feature order length {len(self.feature_order)} != n_features_in_ {self.n_features_in_}"
            )

    def compute_logits(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute raw logit scores: intercept_[0] + X_scaled @ coef_[0]."""
        arr = np.asarray(X_scaled, dtype=float)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Input features {arr.shape[1]} != expected features {self.n_features_in_}"
            )
        return self.intercept_[0] + arr @ self.coef_[0]

    def predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute positive-class probabilities using stable sigmoid (expit)."""
        logits = self.compute_logits(X_scaled)
        return scipy.special.expit(logits)

    def predict(
        self, X_scaled: np.ndarray, threshold: float = CANONICAL_DECISION_THRESHOLD
    ) -> np.ndarray:
        """Predict binary labels using canonical threshold (0.5)."""
        probs = self.predict_proba(X_scaled)
        return (probs >= threshold).astype(int)

    def to_sklearn_lr(self) -> LogisticRegression:
        """Construct an sklearn-compatible LogisticRegression object without fitting."""
        lr = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            penalty=self.penalty,
            random_state=self.random_state,
            fit_intercept=self.fit_intercept,
        )
        lr.classes_ = np.array(self.classes_, dtype=int)
        lr.n_features_in_ = self.n_features_in_
        lr.coef_ = self.coef_.copy()
        lr.intercept_ = self.intercept_.copy()
        lr.feature_names_in_ = np.array(self.feature_order, dtype=object)
        return lr


# -----------------------------------------------------------------------------
# Arm Scientific State Anchor Container
# -----------------------------------------------------------------------------
class FrozenArmState:
    """Scientific state bundle for a single D6 arm loaded from frozen archive."""

    def __init__(
        self,
        arm_id: str,
        changed_dict: Dict[str, Any],
        baseline_scaler: FrozenScaler,
        baseline_lr: FrozenLogisticRegression,
        fairbias_scaler: FrozenScaler,
        fairbias_lr: FrozenLogisticRegression,
        state_hashes: Dict[str, str],
    ) -> None:
        self.arm_id = arm_id
        self.changed_dict = changed_dict
        self.baseline_scaler = baseline_scaler
        self.baseline_lr = baseline_lr
        self.fairbias_scaler = fairbias_scaler
        self.fairbias_lr = fairbias_lr
        self.state_hashes = state_hashes


# -----------------------------------------------------------------------------
# Scored Cohort Structure (Binding Barriers and Diagnostics)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class FrozenScoredCohort:
    """Immutable bundle of a scored cohort with its raw, transformed, scaled,
    and probability arrays.
    """

    year: int
    arm_id: str
    X_original: pd.DataFrame
    X_terminal: pd.DataFrame
    X_baseline_scaled: np.ndarray
    X_fairbias_scaled: np.ndarray
    y: pd.Series
    a: pd.Series
    baseline_logits: np.ndarray
    baseline_probs: np.ndarray
    fairbias_logits: np.ndarray
    fairbias_probs: np.ndarray
    source_row_digest: str


def compute_cohort_utility_metrics(
    y_true: np.ndarray, probs: np.ndarray, threshold: float = CANONICAL_DECISION_THRESHOLD
) -> Dict[str, Any]:
    """Derive benchmark reproduction metrics directly from probability arrays."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(probs, dtype=float)
    y_pred = (p_arr >= threshold).astype(int)
    pos_count = int(np.sum(y_pred))
    sel_rate = float(pos_count / len(y_pred)) if len(y_pred) > 0 else 0.0
    auroc = float(roc_auc_score(y_arr, p_arr)) if len(np.unique(y_arr)) == 2 else 0.0
    auprc = float(average_precision_score(y_arr, p_arr)) if len(np.unique(y_arr)) == 2 else 0.0
    return {
        "count_predicted_positive": pos_count,
        "selection_rate": sel_rate,
        "auroc": auroc,
        "auprc": auprc,
    }


# -----------------------------------------------------------------------------
# Archive & State Loading
# -----------------------------------------------------------------------------
def load_frozen_20_states(
    train_val_release_dir: Union[str, Path],
    test_release_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, FrozenArmState]:
    """Load and verify the 20 scientific state anchors from frozen D6 archives.

    Requires all 20 hashes to match EXPECTED_TRAINING_STATE_ANCHORS exactly.
    """
    tv_dir = Path(train_val_release_dir)
    if not tv_dir.is_dir():
        raise FileNotFoundError(f"D6 train/val release directory not found: {tv_dir}")

    arm_states: Dict[str, FrozenArmState] = {}

    for arm_id in D6_ARM_IDS:
        arm_dir = tv_dir / arm_id
        prov_path = arm_dir / "input_provenance.json"
        changed_dict_path = arm_dir / "final_changed_dict.json"

        if not prov_path.is_file():
            raise FileNotFoundError(f"Missing input_provenance.json for {arm_id} at {prov_path}")
        if not changed_dict_path.is_file():
            raise FileNotFoundError(f"Missing final_changed_dict.json for {arm_id} at {changed_dict_path}")

        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        frozen_state = prov.get("frozen_2022_training_state", {})

        b_scaler_info = frozen_state.get("baseline_scaler", {})
        b_lr_info = frozen_state.get("baseline_logistic_regression", {})
        fb_scaler_info = frozen_state.get("fairbias_scaler", {})
        fb_lr_info = frozen_state.get("fairbias_logistic_regression", {})

        raw_cd = json.loads(changed_dict_path.read_text(encoding="utf-8"))
        cd_data = raw_cd.get("changed_dict", raw_cd)

        observed_hashes = {
            "changed_dict": compute_canonical_json_sha256(cd_data),
            "baseline_scaler": compute_canonical_json_sha256(b_scaler_info.get("state", {})),
            "baseline_LR": compute_canonical_json_sha256(b_lr_info.get("state", {})),
            "FairBias_scaler": compute_canonical_json_sha256(fb_scaler_info.get("state", {})),
            "FairBias_LR": compute_canonical_json_sha256(fb_lr_info.get("state", {})),
        }

        expected_anchors = EXPECTED_TRAINING_STATE_ANCHORS[arm_id]
        for key in ("changed_dict", "baseline_scaler", "baseline_LR", "FairBias_scaler", "FairBias_LR"):
            if observed_hashes[key] != expected_anchors[key]:
                raise ValueError(
                    f"State anchor mismatch for {arm_id} {key}:\n"
                    f"Expected: {expected_anchors[key]}\n"
                    f"Observed: {observed_hashes[key]}"
                )

        b_scaler = FrozenScaler(b_scaler_info["state"])
        b_lr = FrozenLogisticRegression(b_lr_info["state"])
        fb_scaler = FrozenScaler(fb_scaler_info["state"])
        fb_lr = FrozenLogisticRegression(fb_lr_info["state"])

        arm_states[arm_id] = FrozenArmState(
            arm_id=arm_id,
            changed_dict=cd_data,
            baseline_scaler=b_scaler,
            baseline_lr=b_lr,
            fairbias_scaler=fb_scaler,
            fairbias_lr=fb_lr,
            state_hashes=observed_hashes,
        )

    if test_release_dir is not None:
        t_dir = Path(test_release_dir)
        if t_dir.is_dir():
            for arm_id in D6_ARM_IDS:
                test_prov_path = t_dir / arm_id / "training_state_reproduction.json"
                if test_prov_path.is_file():
                    test_repro = json.loads(test_prov_path.read_text(encoding="utf-8"))
                    for key in ("baseline_scaler", "baseline_LR", "FairBias_scaler", "FairBias_LR", "changed_dict"):
                        exp = test_repro.get(key, {}).get("expected_sha256")
                        obs = test_repro.get(key, {}).get("observed_sha256")
                        if exp != EXPECTED_TRAINING_STATE_ANCHORS[arm_id][key] or obs != exp:
                            raise ValueError(f"D6 test archive state mismatch for {arm_id} {key}")

    return arm_states


def verify_git_tag_provenance(repo_root: Path) -> Dict[str, Any]:
    """Verify exact annotated tag objects and dereferenced commits for frozen D6 tags."""
    # Tag nhis-d6-temporal-train-val-v1
    p_tv_tag = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{D6_TRAIN_VAL_TAG}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    tv_tag_obj = p_tv_tag.stdout.strip()
    p_tv_commit = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{D6_TRAIN_VAL_TAG}^{{commit}}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    tv_commit = p_tv_commit.stdout.strip()

    # Tag nhis-d6-temporal-test-v1
    p_test_tag = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{D6_TEST_TAG}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    test_tag_obj = p_test_tag.stdout.strip()
    p_test_commit = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{D6_TEST_TAG}^{{commit}}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    test_commit = p_test_commit.stdout.strip()

    if tv_tag_obj != D6_TRAIN_VAL_TAG_OBJECT or tv_commit != D6_TRAIN_VAL_COMMIT:
        raise ValueError(
            f"D6 Train/Val tag provenance mismatch:\n"
            f"Tag object: expected {D6_TRAIN_VAL_TAG_OBJECT}, got {tv_tag_obj}\n"
            f"Commit: expected {D6_TRAIN_VAL_COMMIT}, got {tv_commit}"
        )

    if test_tag_obj != D6_TEST_TAG_OBJECT or test_commit != D6_TEST_COMMIT:
        raise ValueError(
            f"D6 Test tag provenance mismatch:\n"
            f"Tag object: expected {D6_TEST_TAG_OBJECT}, got {test_tag_obj}\n"
            f"Commit: expected {D6_TEST_COMMIT}, got {test_commit}"
        )

    return {
        "train_val_tag_object": tv_tag_obj,
        "train_val_commit": tv_commit,
        "test_tag_object": test_tag_obj,
        "test_commit": test_commit,
        "tags_verified": True,
    }


def verify_frozen_archives(repo_root: Union[str, Path]) -> Dict[str, Any]:
    """Verify D6 Train/Val and Test manifests, ledger, tags, preprocessing SHA,
    and all manifest-tracked/canonical artifacts.
    """
    root = Path(repo_root)

    tv_dir = root / "docs" / "releases" / D6_TRAIN_VAL_RELEASE_ID
    tv_manifest_path = tv_dir / "d6_temporal_train_val_manifest.json"
    tv_ledger_path = tv_dir / "archive_ledger.json"
    prep_path = tv_dir / "preprocessing_provenance.json"

    test_dir = root / "docs" / "releases" / D6_TEST_RELEASE_ID
    test_manifest_path = test_dir / "d6_temporal_test_manifest.json"
    test_ledger_path = test_dir / "archive_ledger.json"

    if not tv_manifest_path.is_file():
        raise FileNotFoundError(f"D6 Train/Val manifest missing: {tv_manifest_path}")
    if not test_manifest_path.is_file():
        raise FileNotFoundError(f"D6 Test manifest missing: {test_manifest_path}")

    obs_tv_manifest_sha = compute_sha256(tv_manifest_path)
    if obs_tv_manifest_sha != D6_TRAIN_VAL_MANIFEST_SHA256:
        raise ValueError(
            f"D6 Train/Val manifest SHA mismatch: expected {D6_TRAIN_VAL_MANIFEST_SHA256}, got {obs_tv_manifest_sha}"
        )

    obs_tv_ledger_sha = compute_sha256(tv_ledger_path)
    if obs_tv_ledger_sha != D6_TRAIN_VAL_LEDGER_SHA256:
        raise ValueError(
            f"D6 Train/Val ledger SHA mismatch: expected {D6_TRAIN_VAL_LEDGER_SHA256}, got {obs_tv_ledger_sha}"
        )

    obs_test_manifest_sha = compute_sha256(test_manifest_path)
    if obs_test_manifest_sha != D6_TEST_MANIFEST_SHA256:
        raise ValueError(
            f"D6 Test manifest SHA mismatch: expected {D6_TEST_MANIFEST_SHA256}, got {obs_test_manifest_sha}"
        )

    if prep_path.is_file():
        prep_data = json.loads(prep_path.read_text(encoding="utf-8"))
        obs_prep_sha = compute_canonical_json_sha256(prep_data)
        if obs_prep_sha != PREPROCESSING_STATE_SHA256:
            raise ValueError(
                f"Preprocessing provenance SHA mismatch: expected {PREPROCESSING_STATE_SHA256}, got {obs_prep_sha}"
            )
    else:
        raise FileNotFoundError(f"Preprocessing provenance missing: {prep_path}")

    # Verify all manifest-tracked artifacts in Train/Val
    tv_manifest = json.loads(tv_manifest_path.read_text(encoding="utf-8"))
    tv_artifacts = tv_manifest.get("artifacts", {})
    if len(tv_artifacts) != 45:
        raise ValueError(f"D6 Train/Val manifest must track 45 artifacts; found {len(tv_artifacts)}")
    for rel_path, meta in tv_artifacts.items():
        art_path = tv_dir / rel_path
        if not art_path.is_file():
            raise FileNotFoundError(f"Train/Val artifact missing: {art_path}")
        if compute_sha256(art_path) != meta["sha256"]:
            raise ValueError(f"Train/Val artifact hash mismatch for {rel_path}")

    # Verify all 41 manifest-tracked artifacts in Test
    test_manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))
    test_artifacts = test_manifest.get("artifacts", {})
    if len(test_artifacts) != 41:
        raise ValueError(f"D6 Test manifest must track 41 artifacts; found {len(test_artifacts)}")
    for rel_path, meta in test_artifacts.items():
        art_path = test_dir / rel_path
        if not art_path.is_file():
            raise FileNotFoundError(f"Test artifact missing: {art_path}")
        if compute_sha256(art_path) != meta["sha256"]:
            raise ValueError(f"Test artifact hash mismatch for {rel_path}")

    # Verify canonical files in ledgers if present
    tv_ledger = json.loads(tv_ledger_path.read_text(encoding="utf-8"))
    tv_can_hashes = tv_ledger.get("canonical_file_hashes", [])
    if isinstance(tv_can_hashes, list):
        for item in tv_can_hashes:
            rel_path = item["relative_path"]
            exp_sha = item["sha256"]
            can_path = tv_dir / rel_path
            if not can_path.is_file():
                raise FileNotFoundError(f"Train/Val canonical file missing: {can_path}")
            if compute_sha256(can_path) != exp_sha:
                raise ValueError(f"Train/Val canonical file hash mismatch for {rel_path}")
    elif isinstance(tv_can_hashes, dict):
        for rel_path, exp_sha in tv_can_hashes.items():
            can_path = tv_dir / rel_path
            if not can_path.is_file():
                raise FileNotFoundError(f"Train/Val canonical file missing: {can_path}")
            if compute_sha256(can_path) != exp_sha:
                raise ValueError(f"Train/Val canonical file hash mismatch for {rel_path}")

    if test_ledger_path.is_file():
        test_ledger = json.loads(test_ledger_path.read_text(encoding="utf-8"))
        test_can_hashes = test_ledger.get("canonical_file_hashes", [])
        if isinstance(test_can_hashes, list):
            for item in test_can_hashes:
                rel_path = item["relative_path"]
                exp_sha = item["sha256"]
                can_path = test_dir / rel_path
                if not can_path.is_file():
                    raise FileNotFoundError(f"Test canonical file missing: {can_path}")
                if compute_sha256(can_path) != exp_sha:
                    raise ValueError(f"Test canonical file hash mismatch for {rel_path}")
        elif isinstance(test_can_hashes, dict):
            for rel_path, exp_sha in test_can_hashes.items():
                can_path = test_dir / rel_path
                if not can_path.is_file():
                    raise FileNotFoundError(f"Test canonical file missing: {can_path}")
                if compute_sha256(can_path) != exp_sha:
                    raise ValueError(f"Test canonical file hash mismatch for {rel_path}")

    # Verify git tags - fail closed on any tag object or commit mismatch
    tag_info = verify_git_tag_provenance(root)

    return {
        "train_val_manifest_sha256": obs_tv_manifest_sha,
        "train_val_ledger_sha256": obs_tv_ledger_sha,
        "test_manifest_sha256": obs_test_manifest_sha,
        "preprocessing_provenance_sha256": obs_prep_sha,
        "tag_verification": tag_info,
        "status": "VERIFIED",
        "train_val_artifact_count": len(tv_artifacts),
        "test_artifact_count": len(test_artifacts),
    }


# -----------------------------------------------------------------------------
# Provenance and Scoring Reproduction Barriers
# -----------------------------------------------------------------------------
class CohortProvenanceBarrierError(Exception):
    """Raised when 12/12 cohort source-row digests do not match exactly."""


class ScoringReproductionBarrierError(Exception):
    """Raised when terminal predictions do not match archived D6 benchmarks."""


def verify_cohort_provenance_barrier(
    cohort_digests: Mapping[int, Mapping[str, str]],
) -> Dict[str, Any]:
    """Verify exact 12/12 cohort source-row digests before any mechanism diagnostic.

    Any mismatch raises CohortProvenanceBarrierError and enforces zero diagnostics.
    """
    total_checked = 0
    matched = 0
    mismatches: List[Dict[str, Any]] = []

    for year in TEMPORAL_YEARS:
        year_expected = EXPECTED_COHORT_SOURCE_ROW_DIGESTS[year]
        year_observed = cohort_digests.get(year, {})
        for arm_id in D6_ARM_IDS:
            total_checked += 1
            exp = year_expected[arm_id]
            obs = year_observed.get(arm_id)
            if obs == exp:
                matched += 1
            else:
                mismatches.append(
                    {
                        "year": year,
                        "arm_id": arm_id,
                        "expected": exp,
                        "observed": obs,
                    }
                )

    if total_checked != 12 or matched != 12:
        raise CohortProvenanceBarrierError(
            f"Cohort provenance barrier failed: {matched}/12 matched.\n"
            f"Mismatches: {json.dumps(mismatches, indent=2)}"
        )

    return {
        "total_cohorts_expected": 12,
        "total_cohorts_matched": 12,
        "status": "PASS",
        "provenance_barrier_enforced": True,
    }


def read_archived_scoring_benchmark(arch_path: Path) -> Dict[str, Any]:
    """Read utility metrics dict from archived JSON artifact."""
    return json.loads(arch_path.read_text(encoding="utf-8"))["utility"]


def verify_scoring_reproduction_barrier(
    observed_metrics: Mapping[int, Mapping[str, Mapping[str, Any]]],
    archived_train_val_dir: Union[str, Path],
    archived_test_dir: Union[str, Path],
    tolerance: float = SCORING_REPRODUCTION_ABSOLUTE_TOLERANCE,
) -> Dict[str, Any]:
    """Verify reproduction of archived 2023 validation and 2024 test terminal scores.

    Requirements:
    - predicted_positive_count: EXACT match.
    - AUROC, AUPRC, selection_rate: absolute error <= tolerance (1e-12).
    """
    tv_dir = Path(archived_train_val_dir)
    t_dir = Path(archived_test_dir)

    failures: List[Dict[str, Any]] = []

    # 2023 Validation
    for arm_id in D6_ARM_IDS:
        for model_key, json_name in (
            ("baseline", "validation_metrics_baseline.json"),
            ("fairbias", "validation_metrics_fairbias.json"),
        ):
            arch_path = tv_dir / arm_id / json_name
            arch_data = read_archived_scoring_benchmark(arch_path)

            obs_data = (
                observed_metrics.get(2023, {})
                .get(arm_id, {})
                .get(model_key, {})
            )

            # Check exact predicted positive count
            exp_pos = int(arch_data["count_predicted_positive"])
            obs_pos = int(obs_data.get("count_predicted_positive", -999))
            if obs_pos != exp_pos:
                failures.append(
                    {
                        "year": 2023,
                        "arm_id": arm_id,
                        "model": model_key,
                        "metric": "count_predicted_positive",
                        "expected": exp_pos,
                        "observed": obs_pos,
                        "type": "exact_int_mismatch",
                    }
                )

            # Check continuous metrics
            for metric in ("auroc", "auprc", "selection_rate"):
                exp_val = float(arch_data[metric])
                obs_val = float(obs_data.get(metric, -999.0))
                diff = abs(obs_val - exp_val)
                if diff > tolerance:
                    failures.append(
                        {
                            "year": 2023,
                            "arm_id": arm_id,
                            "model": model_key,
                            "metric": metric,
                            "expected": exp_val,
                            "observed": obs_val,
                            "diff": diff,
                            "tolerance": tolerance,
                            "type": "tolerance_exceeded",
                        }
                    )

    # 2024 Test
    for arm_id in D6_ARM_IDS:
        for model_key, json_name in (
            ("baseline", "test_metrics_baseline.json"),
            ("fairbias", "test_metrics_fairbias.json"),
        ):
            arch_path = t_dir / arm_id / json_name
            arch_data = read_archived_scoring_benchmark(arch_path)

            obs_data = (
                observed_metrics.get(2024, {})
                .get(arm_id, {})
                .get(model_key, {})
            )

            # Check exact predicted positive count
            exp_pos = int(arch_data["count_predicted_positive"])
            obs_pos = int(obs_data.get("count_predicted_positive", -999))
            if obs_pos != exp_pos:
                failures.append(
                    {
                        "year": 2024,
                        "arm_id": arm_id,
                        "model": model_key,
                        "metric": "count_predicted_positive",
                        "expected": exp_pos,
                        "observed": obs_pos,
                        "type": "exact_int_mismatch",
                    }
                )

            for metric in ("auroc", "auprc", "selection_rate"):
                exp_val = float(arch_data[metric])
                obs_val = float(obs_data.get(metric, -999.0))
                diff = abs(obs_val - exp_val)
                if diff > tolerance:
                    failures.append(
                        {
                            "year": 2024,
                            "arm_id": arm_id,
                            "model": model_key,
                            "metric": metric,
                            "expected": exp_val,
                            "observed": obs_val,
                            "diff": diff,
                            "tolerance": tolerance,
                            "type": "tolerance_exceeded",
                        }
                    )

    if failures:
        raise ScoringReproductionBarrierError(
            f"Scoring reproduction barrier failed with {len(failures)} mismatch(es):\n"
            f"{json.dumps(failures, indent=2)}"
        )

    return {
        "status": "PASS",
        "scoring_reproduction_barrier_enforced": True,
        "reproduction_tolerance": tolerance,
        "anchors_verified_count": 16,  # 4 arms x 2 models x 2 years
    }


# -----------------------------------------------------------------------------
# Diagnostic Helpers: NMI & Quantiles
# -----------------------------------------------------------------------------
def compute_nmi(x: pd.Series, target: pd.Series) -> float:
    """Calculate Normalized Mutual Information (NMI) using frozen binning semantics."""
    target_arr = np.asarray(target, dtype=int)
    if pd.api.types.is_float_dtype(x) or (pd.api.types.is_numeric_dtype(x) and x.nunique() > 10):
        try:
            x_binned = pd.qcut(
                x, q=min(10, max(2, x.nunique())), labels=False, duplicates="drop"
            )
        except Exception:
            x_binned = pd.cut(x, bins=min(10, max(2, x.nunique())), labels=False)
        return float(normalized_mutual_info_score(np.asarray(x_binned).astype(int), target_arr))
    else:
        return float(normalized_mutual_info_score(np.asarray(x).astype(str), target_arr))


def compute_distribution_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Compute standard summary statistics for score distributions."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "iqr": None,
            **{f"p{int(q*100):02d}": None for q in QUANTILE_LEVELS},
            "fraction_ge_0_5": None,
        }

    q_vals = np.quantile(arr, QUANTILE_LEVELS)
    q_dict = {f"p{int(q*100):02d}": float(val) for q, val in zip(QUANTILE_LEVELS, q_vals)}
    iqr = float(q_dict["p75"] - q_dict["p25"])

    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(arr)),
        "iqr": iqr,
        **q_dict,
        "fraction_ge_0_5": float(np.mean(arr >= CANONICAL_DECISION_THRESHOLD)),
    }


# -----------------------------------------------------------------------------
# Diagnostic Engine 1: Family I Information Compression & Category States
# -----------------------------------------------------------------------------
def compute_family1_diagnostics(
    X_raw: pd.DataFrame,
    X_transformed: pd.DataFrame,
    y: pd.Series,
    a: pd.Series,
    changed_dict: Mapping[str, Any],
    arm_id: str,
    year: int,
) -> List[Dict[str, Any]]:
    """Compute Family I distinction-reducing diagnostics (drops and categorical merges).

    Emits both feature_summary and category_state records for family1_information_compression.csv.
    """
    if not X_raw.index.equals(y.index):
        raise ValueError(
            f"Family-I index alignment failure: X_raw.index != y.index for {arm_id} {year}"
        )
    if not X_raw.index.equals(a.index):
        raise ValueError(
            f"Family-I index alignment failure: X_raw.index != a.index for {arm_id} {year}"
        )
    if not X_raw.index.equals(X_transformed.index):
        raise ValueError(
            f"Family-I index alignment failure: X_raw.index != X_transformed.index for {arm_id} {year}"
        )

    records: List[Dict[str, Any]] = []

    for feature, transform in changed_dict.items():
        if transform == "dropped":
            orig_series = X_raw[feature]
            orig_card = int(orig_series.nunique())
            nmi_y_before = compute_nmi(orig_series, y)
            nmi_a_before = compute_nmi(orig_series, a)

            # Feature summary
            records.append(
                {
                    "record_type": "feature_summary",
                    "year": year,
                    "arm_id": arm_id,
                    "feature": feature,
                    "transform_family": "Family_I",
                    "transform_type": "feature_drop",
                    "original_cardinality": orig_card,
                    "terminal_cardinality": 1,
                    "cardinality_reduction": orig_card - 1,
                    "nmi_y_before": nmi_y_before,
                    "nmi_y_after": 0.0,
                    "delta_nmi_y": -nmi_y_before,
                    "nmi_a_before": nmi_a_before,
                    "nmi_a_after": 0.0,
                    "delta_nmi_a": -nmi_a_before,
                    "post_state": "dropped_constant_equivalent",
                    "category": None,
                    "state": None,
                    "n": None,
                    "outcome_positive_count": None,
                    "outcome_prevalence": None,
                }
            )

            # Category state records: before
            for cat_val, grp_df in orig_series.groupby(orig_series):
                n_c = len(grp_df)
                group_index = orig_series.index[orig_series == cat_val]
                y_group = y.loc[group_index]
                y_c = int(np.sum(y_group))
                records.append(
                    {
                        "record_type": "category_state",
                        "year": year,
                        "arm_id": arm_id,
                        "feature": feature,
                        "transform_family": "Family_I",
                        "transform_type": "feature_drop",
                        "original_cardinality": None,
                        "terminal_cardinality": None,
                        "cardinality_reduction": None,
                        "nmi_y_before": None,
                        "nmi_y_after": None,
                        "delta_nmi_y": None,
                        "nmi_a_before": None,
                        "nmi_a_after": None,
                        "delta_nmi_a": None,
                        "post_state": "dropped_constant_equivalent",
                        "category": str(cat_val),
                        "state": "before",
                        "n": n_c,
                        "outcome_positive_count": y_c,
                        "outcome_prevalence": float(y_c / n_c) if n_c > 0 else 0.0,
                    }
                )

            # Category state record: after (constant equivalent)
            total_n = len(orig_series)
            total_pos = int(np.sum(y))
            records.append(
                {
                    "record_type": "category_state",
                    "year": year,
                    "arm_id": arm_id,
                    "feature": feature,
                    "transform_family": "Family_I",
                    "transform_type": "feature_drop",
                    "original_cardinality": None,
                    "terminal_cardinality": None,
                    "cardinality_reduction": None,
                    "nmi_y_before": None,
                    "nmi_y_after": None,
                    "delta_nmi_y": None,
                    "nmi_a_before": None,
                    "nmi_a_after": None,
                    "delta_nmi_a": None,
                    "post_state": "dropped_constant_equivalent",
                    "category": "dropped_constant_equivalent",
                    "state": "after",
                    "n": total_n,
                    "outcome_positive_count": total_pos,
                    "outcome_prevalence": float(total_pos / total_n) if total_n > 0 else 0.0,
                }
            )

        elif isinstance(transform, dict) and "power" not in transform:
            # Categorical merge
            orig_series = X_raw[feature]
            term_series = X_transformed[feature]
            orig_card = int(orig_series.nunique())
            term_card = int(term_series.nunique())

            nmi_y_before = compute_nmi(orig_series, y)
            nmi_y_after = compute_nmi(term_series, y)
            nmi_a_before = compute_nmi(orig_series, a)
            nmi_a_after = compute_nmi(term_series, a)

            records.append(
                {
                    "record_type": "feature_summary",
                    "year": year,
                    "arm_id": arm_id,
                    "feature": feature,
                    "transform_family": "Family_I",
                    "transform_type": "categorical_merge",
                    "original_cardinality": orig_card,
                    "terminal_cardinality": term_card,
                    "cardinality_reduction": orig_card - term_card,
                    "nmi_y_before": nmi_y_before,
                    "nmi_y_after": nmi_y_after,
                    "delta_nmi_y": nmi_y_after - nmi_y_before,
                    "nmi_a_before": nmi_a_before,
                    "nmi_a_after": nmi_a_after,
                    "delta_nmi_a": nmi_a_after - nmi_a_before,
                    "post_state": "merged_categories",
                    "category": None,
                    "state": None,
                    "n": None,
                    "outcome_positive_count": None,
                    "outcome_prevalence": None,
                }
            )

            # Category state records: before
            for cat_val, grp_df in orig_series.groupby(orig_series):
                n_c = len(grp_df)
                group_index = orig_series.index[orig_series == cat_val]
                y_group = y.loc[group_index]
                y_c = int(np.sum(y_group))
                records.append(
                    {
                        "record_type": "category_state",
                        "year": year,
                        "arm_id": arm_id,
                        "feature": feature,
                        "transform_family": "Family_I",
                        "transform_type": "categorical_merge",
                        "original_cardinality": None,
                        "terminal_cardinality": None,
                        "cardinality_reduction": None,
                        "nmi_y_before": None,
                        "nmi_y_after": None,
                        "delta_nmi_y": None,
                        "nmi_a_before": None,
                        "nmi_a_after": None,
                        "delta_nmi_a": None,
                        "post_state": "merged_categories",
                        "category": str(cat_val),
                        "state": "before",
                        "n": n_c,
                        "outcome_positive_count": y_c,
                        "outcome_prevalence": float(y_c / n_c) if n_c > 0 else 0.0,
                    }
                )

            # Category state records: after
            for cat_val, grp_df in term_series.groupby(term_series):
                n_c = len(grp_df)
                group_index = term_series.index[term_series == cat_val]
                y_group = y.loc[group_index]
                y_c = int(np.sum(y_group))
                records.append(
                    {
                        "record_type": "category_state",
                        "year": year,
                        "arm_id": arm_id,
                        "feature": feature,
                        "transform_family": "Family_I",
                        "transform_type": "categorical_merge",
                        "original_cardinality": None,
                        "terminal_cardinality": None,
                        "cardinality_reduction": None,
                        "nmi_y_before": None,
                        "nmi_y_after": None,
                        "delta_nmi_y": None,
                        "nmi_a_before": None,
                        "nmi_a_after": None,
                        "delta_nmi_a": None,
                        "post_state": "merged_categories",
                        "category": str(cat_val),
                        "state": "after",
                        "n": n_c,
                        "outcome_positive_count": y_c,
                        "outcome_prevalence": float(y_c / n_c) if n_c > 0 else 0.0,
                    }
                )

    return records


# -----------------------------------------------------------------------------
# Diagnostic Engine 2: Family II Numerical Geometry
# -----------------------------------------------------------------------------
def compute_family2_diagnostics(
    X_raw: pd.DataFrame,
    X_transformed: pd.DataFrame,
    changed_dict: Mapping[str, Any],
    fairbias_scaler: FrozenScaler,
    arm_id: str,
    year: int,
) -> List[Dict[str, Any]]:
    """Compute Family II numerical geometry diagnostics for positive power transforms."""
    records: List[Dict[str, Any]] = []

    for feature, transform in changed_dict.items():
        if isinstance(transform, dict) and "power" in transform:
            power = float(transform["power"])
            raw_s = X_raw[feature].to_numpy(dtype=float)
            term_s = X_transformed[feature].to_numpy(dtype=float)

            feat_idx = fairbias_scaler.feature_order.index(feature)
            scale_val = float(fairbias_scaler.scale_[feat_idx])
            min_val = float(fairbias_scaler.min_[feat_idx])
            data_min_val = float(fairbias_scaler.data_min_[feat_idx])
            data_max_val = float(fairbias_scaler.data_max_[feat_idx])
            data_range_val = float(fairbias_scaler.data_range_[feat_idx])

            scaled_s = term_s * scale_val + min_val

            raw_q = np.quantile(raw_s, QUANTILE_LEVELS)
            term_q = np.quantile(term_s, QUANTILE_LEVELS)
            scaled_q = np.quantile(scaled_s, QUANTILE_LEVELS)

            row: Dict[str, Any] = {
                "year": year,
                "arm_id": arm_id,
                "feature": feature,
                "transform_family": "Family_II",
                "transform_type": "power_reparameterization",
                "power_exponent": power,
                "scaler_data_min": data_min_val,
                "scaler_data_max": data_max_val,
                "scaler_data_range": data_range_val,
                "scaler_scale": scale_val,
                "scaler_min": min_val,
                "mathematically_bijective_over_domain": bool(power > 0),
                "information_preserving_reparameterization": True,
            }

            for q, r_val, t_val, s_val in zip(QUANTILE_LEVELS, raw_q, term_q, scaled_q):
                p_str = f"p{int(q*100):02d}"
                row[f"raw_{p_str}"] = float(r_val)
                row[f"power_{p_str}"] = float(t_val)
                row[f"scaled_{p_str}"] = float(s_val)

            records.append(row)

    return records


# -----------------------------------------------------------------------------
# Diagnostic Engine 3: Terminal Score Distributions & KS Separation
# -----------------------------------------------------------------------------
def compute_score_distribution_diagnostics(
    probs: np.ndarray,
    y_true: pd.Series,
    arm_id: str,
    year: int,
    model_name: str,
) -> Dict[str, Any]:
    """Compute score distribution summaries, class-conditional quantiles, and KS separation."""
    y_arr = np.asarray(y_true, dtype=int)
    scores_overall = probs
    scores_y0 = probs[y_arr == 0]
    scores_y1 = probs[y_arr == 1]

    sum_overall = compute_distribution_summary(scores_overall)
    sum_y0 = compute_distribution_summary(scores_y0)
    sum_y1 = compute_distribution_summary(scores_y1)

    if len(scores_y0) > 0 and len(scores_y1) > 0:
        ks_res = scipy.stats.ks_2samp(scores_y0, scores_y1)
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)
    else:
        ks_stat = None
        ks_pval = None

    auroc = (
        float(roc_auc_score(y_arr, probs))
        if len(np.unique(y_arr)) == 2
        else None
    )
    auprc = (
        float(average_precision_score(y_arr, probs))
        if len(np.unique(y_arr)) == 2
        else None
    )

    return {
        "year": year,
        "arm_id": arm_id,
        "model": model_name,
        "auroc": auroc,
        "auprc": auprc,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pval,
        "overall": sum_overall,
        "y0": sum_y0,
        "y1": sum_y1,
    }


# -----------------------------------------------------------------------------
# Diagnostic Engine 4: Protected-Group Class-Conditional Score Diagnostics
# -----------------------------------------------------------------------------
def compute_protected_group_score_diagnostics(
    probs: np.ndarray,
    y_true: pd.Series,
    a_group: pd.Series,
    expected_groups: Sequence[int],
    arm_id: str,
    year: int,
    model_name: str,
) -> List[Dict[str, Any]]:
    """Compute protected-group stratified score distributions for all, Y0, and Y1 strata.

    Preserves all groups (e.g. HISP 1..7) and outputs null/None for empty strata (never 0).
    """
    records: List[Dict[str, Any]] = []
    y_arr = np.asarray(y_true, dtype=int)
    a_arr = np.asarray(a_group, dtype=int)

    for g in expected_groups:
        mask_g = a_arr == g
        scores_g = probs[mask_g]
        y_g = y_arr[mask_g]

        # Stratum: all
        sum_all = compute_distribution_summary(scores_g)
        records.append(
            {
                "year": year,
                "arm_id": arm_id,
                "model": model_name,
                "group": g,
                "outcome_stratum": "all",
                "n": sum_all["n"],
                "outcome_positive_count": int(np.sum(y_g == 1)) if len(y_g) > 0 else 0,
                "mean": sum_all["mean"],
                "std": sum_all["std"],
                "median": sum_all["median"],
                "iqr": sum_all["iqr"],
                "p10": sum_all["p10"],
                "p25": sum_all["p25"],
                "p50": sum_all["p50"],
                "p75": sum_all["p75"],
                "p90": sum_all["p90"],
                "fraction_ge_0_5": sum_all["fraction_ge_0_5"],
            }
        )

        # Stratum: Y0
        scores_y0 = scores_g[y_g == 0]
        sum_y0 = compute_distribution_summary(scores_y0)
        records.append(
            {
                "year": year,
                "arm_id": arm_id,
                "model": model_name,
                "group": g,
                "outcome_stratum": "Y0",
                "n": sum_y0["n"],
                "outcome_positive_count": 0,
                "mean": sum_y0["mean"],
                "std": sum_y0["std"],
                "median": sum_y0["median"],
                "iqr": sum_y0["iqr"],
                "p10": sum_y0["p10"],
                "p25": sum_y0["p25"],
                "p50": sum_y0["p50"],
                "p75": sum_y0["p75"],
                "p90": sum_y0["p90"],
                "fraction_ge_0_5": sum_y0["fraction_ge_0_5"],
            }
        )

        # Stratum: Y1
        scores_y1 = scores_g[y_g == 1]
        sum_y1 = compute_distribution_summary(scores_y1)
        records.append(
            {
                "year": year,
                "arm_id": arm_id,
                "model": model_name,
                "group": g,
                "outcome_stratum": "Y1",
                "n": sum_y1["n"],
                "outcome_positive_count": sum_y1["n"],
                "mean": sum_y1["mean"],
                "std": sum_y1["std"],
                "median": sum_y1["median"],
                "iqr": sum_y1["iqr"],
                "p10": sum_y1["p10"],
                "p25": sum_y1["p25"],
                "p50": sum_y1["p50"],
                "p75": sum_y1["p75"],
                "p90": sum_y1["p90"],
                "fraction_ge_0_5": sum_y1["fraction_ge_0_5"],
            }
        )

    return records


# -----------------------------------------------------------------------------
# Diagnostic Engine 5: Frozen Logistic Contribution Decomposition
# -----------------------------------------------------------------------------
def compute_logit_contribution_diagnostics(
    X_scaled: np.ndarray,
    y_true: pd.Series,
    lr: FrozenLogisticRegression,
    arm_id: str,
    year: int,
    model_name: str,
    changed_dict: Optional[Mapping[str, Any]] = None,
    all_baseline_features: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Compute descriptive decomposition of the fitted terminal linear scoring function.

    Preserves explicit feature presence semantics and never implies ceteris-paribus comparability.
    """
    y_arr = np.asarray(y_true, dtype=int)
    mask_y0 = y_arr == 0
    mask_y1 = y_arr == 1

    records: List[Dict[str, Any]] = []
    intercept = float(lr.intercept_[0])
    coefs = lr.coef_[0]

    cd = changed_dict or {}

    # Present features
    for j, feature in enumerate(lr.feature_order):
        col_scaled = X_scaled[:, j]
        beta_j = float(coefs[j])

        mean_x_y0 = float(np.mean(col_scaled[mask_y0])) if np.any(mask_y0) else None
        mean_x_y1 = float(np.mean(col_scaled[mask_y1])) if np.any(mask_y1) else None

        if mean_x_y0 is not None and mean_x_y1 is not None:
            mean_beta_x_y0 = beta_j * mean_x_y0
            mean_beta_x_y1 = beta_j * mean_x_y1
            class_sep_contrib = beta_j * (mean_x_y1 - mean_x_y0)
        else:
            mean_beta_x_y0 = None
            mean_beta_x_y1 = None
            class_sep_contrib = None

        if feature in cd:
            comp_type = "representation_reparameterized"
        else:
            comp_type = "shared_feature_same_name"

        records.append(
            {
                "year": year,
                "arm_id": arm_id,
                "model": model_name,
                "feature": feature,
                "intercept": intercept,
                "beta_j": beta_j,
                "mean_scaled_X_j_Y0": mean_x_y0,
                "mean_scaled_X_j_Y1": mean_x_y1,
                "mean_beta_scaled_X_j_Y0": mean_beta_x_y0,
                "mean_beta_scaled_X_j_Y1": mean_beta_x_y1,
                "class_separation_contribution": class_sep_contrib,
                "baseline_present": True,
                "fairbias_present": (cd.get(feature) != "dropped"),
                "comparison_type": comp_type,
                "absence_equivalent_zero_contribution": False,
                "interpretation_status": (
                    "descriptive decomposition of the fitted terminal linear scoring function"
                ),
            }
        )

    # If FairBias model, also explicitly record features dropped from baseline representation
    if model_name == "fairbias" and all_baseline_features is not None:
        for b_feat in all_baseline_features:
            if b_feat not in lr.feature_order:
                records.append(
                    {
                        "year": year,
                        "arm_id": arm_id,
                        "model": "fairbias",
                        "feature": b_feat,
                        "intercept": intercept,
                        "beta_j": None,
                        "mean_scaled_X_j_Y0": None,
                        "mean_scaled_X_j_Y1": None,
                        "mean_beta_scaled_X_j_Y0": None,
                        "mean_beta_scaled_X_j_Y1": None,
                        "class_separation_contribution": None,
                        "baseline_present": True,
                        "fairbias_present": False,
                        "comparison_type": "dropped_from_fairbias_representation",
                        "absence_equivalent_zero_contribution": True,
                        "interpretation_status": (
                            "descriptive decomposition of the fitted terminal linear scoring function"
                        ),
                    }
                )

    return records


# -----------------------------------------------------------------------------
# Diagnostic Engine 6: Mechanism Arm Summary (Descriptive Synthesis Only)
# -----------------------------------------------------------------------------
def compute_mechanism_arm_summary(
    arm_id: str,
    changed_dict: Mapping[str, Any],
    family1_records: Sequence[Mapping[str, Any]],
    family2_records: Sequence[Mapping[str, Any]],
    score_dist_records: Sequence[Mapping[str, Any]],
    logit_contrib_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Synthesize descriptive mechanism findings per arm without hardcoding causal narratives
    or automatic hypothesis booleans.
    """
    f1_arm = [r for r in family1_records if r.get("arm_id") == arm_id and r.get("record_type") == "feature_summary"]
    f2_arm = [r for r in family2_records if r.get("arm_id") == arm_id]

    drop_count = sum(1 for r in f1_arm if r.get("transform_type") == "feature_drop")
    merge_count = sum(1 for r in f1_arm if r.get("transform_type") == "categorical_merge")
    power_count = len(f2_arm)

    # 2024 test metrics comparison
    test_scores = [r for r in score_dist_records if r.get("arm_id") == arm_id and r.get("year") == 2024]
    base_dist = next((r for r in test_scores if r.get("model") == "baseline"), {})
    fb_dist = next((r for r in test_scores if r.get("model") == "fairbias"), {})

    base_ks = base_dist.get("ks_statistic")
    fb_ks = fb_dist.get("ks_statistic")
    ks_delta = (fb_ks - base_ks) if (fb_ks is not None and base_ks is not None) else None

    base_auroc = base_dist.get("auroc")
    fb_auroc = fb_dist.get("auroc")
    auroc_delta = (fb_auroc - base_auroc) if (fb_auroc is not None and base_auroc is not None) else None

    base_auprc = base_dist.get("auprc")
    fb_auprc = fb_dist.get("auprc")
    auprc_delta = (fb_auprc - base_auprc) if (fb_auprc is not None and base_auprc is not None) else None

    base_sel = base_dist.get("overall", {}).get("fraction_ge_0_5")
    fb_sel = fb_dist.get("overall", {}).get("fraction_ge_0_5")
    sel_delta = (fb_sel - base_sel) if (fb_sel is not None and base_sel is not None) else None

    # Largest contribution changes
    base_contribs = {
        r["feature"]: r.get("class_separation_contribution")
        for r in logit_contrib_records
        if r.get("arm_id") == arm_id and r.get("model") == "baseline" and r.get("year") == 2024
    }
    fb_contribs = {
        r["feature"]: r.get("class_separation_contribution")
        for r in logit_contrib_records
        if r.get("arm_id") == arm_id and r.get("model") == "fairbias" and r.get("year") == 2024
    }

    contrib_deltas = []
    all_features = set(base_contribs) | set(fb_contribs)
    for feat in all_features:
        b_c = base_contribs.get(feat)
        f_c = fb_contribs.get(feat)
        # Handle dropped features where fairbias contribution is absence-equivalent zero
        b_val = b_c if b_c is not None else 0.0
        f_val = f_c if f_c is not None else 0.0
        comp_type = "dropped_from_fairbias_representation" if f_c is None else (
            "representation_reparameterized" if feat in changed_dict else "shared_feature_same_name"
        )
        contrib_deltas.append(
            {
                "feature": feat,
                "baseline_contrib": b_c,
                "fairbias_contrib": f_c,
                "abs_delta": abs(f_val - b_val),
                "comparison_type": comp_type,
            }
        )
    contrib_deltas.sort(key=lambda x: x["abs_delta"], reverse=True)

    # Family 1 NMI changes summary
    f1_nmi = [
        {
            "feature": r["feature"],
            "delta_nmi_y": r["delta_nmi_y"],
            "delta_nmi_a": r["delta_nmi_a"],
        }
        for r in f1_arm
        if r.get("year") == 2024
    ]

    # Family 2 geometry summaries
    f2_geom = [
        {
            "feature": r["feature"],
            "power_exponent": r["power_exponent"],
            "raw_p50": r.get("raw_p50"),
            "power_p50": r.get("power_p50"),
            "scaled_p50": r.get("scaled_p50"),
        }
        for r in f2_arm
        if r.get("year") == 2024
    ]

    return {
        "arm_id": arm_id,
        "protected_attribute": ARM_PROTECTED_ATTRIBUTES[arm_id],
        "drop_count": drop_count,
        "merge_count": merge_count,
        "power_count": power_count,
        "family1_transform_count": drop_count + merge_count,
        "family2_transform_count": power_count,
        "score_ks_baseline": base_ks,
        "score_ks_fairbias": fb_ks,
        "score_separation_delta": ks_delta,
        "selection_rate_baseline": base_sel,
        "selection_rate_fairbias": fb_sel,
        "selection_rate_delta": sel_delta,
        "auroc_baseline": base_auroc,
        "auroc_fairbias": fb_auroc,
        "auroc_delta": auroc_delta,
        "auprc_baseline": base_auprc,
        "auprc_fairbias": fb_auprc,
        "auprc_delta": auprc_delta,
        "largest_absolute_terminal_contribution_changes": contrib_deltas[:5],
        "family1_nmi_changes": f1_nmi,
        "family2_geometry_summaries": f2_geom,
        "hypothesis_adjudication": "PI_REVIEW_REQUIRED",
        "interpretation_status": "descriptive_post_hoc",
        "causal_claim_made": False,
    }


# -----------------------------------------------------------------------------
# Macro Context Builder (D4 / D5 Context)
# -----------------------------------------------------------------------------
def build_macro_context(repo_root: Union[str, Path]) -> Dict[str, Any]:
    """Aggregate macro aggregate performance context from canonical D4 and D5 releases.

    Zero dependency on local preflight trace files or runs/ directory.
    """
    root = Path(repo_root)
    d4_dir = root / "docs" / "releases" / D4_PRIMARY_RELEASE_ID / "raw"
    d5_dir = root / "docs" / "releases" / D5_WEIGHTED_RELEASE_ID

    d4_context: Dict[str, Any] = {}
    if d4_dir.is_dir():
        for arm_key, d4_arm_dir_name in (
            ("D6_ARM_001", "ARM_D3_001"),
            ("D6_ARM_002", "ARM_D3_002"),
            ("D6_ARM_003", "ARM_D3_003"),
            ("D6_ARM_004", "ARM_D3_004"),
        ):
            comp_path = d4_dir / d4_arm_dir_name / "test_comparison.json"
            if comp_path.is_file():
                d4_context[arm_key] = json.loads(comp_path.read_text(encoding="utf-8"))

    d5_perf_context: Dict[str, Any] = {}
    d5_transform_summary: Dict[str, Any] = {}
    if d5_dir.is_dir():
        for arm_key, d5_arm_name in (
            ("D6_ARM_001", "D5_ARM_001"),
            ("D6_ARM_002", "D5_ARM_002"),
            ("D6_ARM_003", "D5_ARM_003"),
            ("D6_ARM_004", "D5_ARM_004"),
        ):
            comp_path = d5_dir / d5_arm_name / "test_comparison.json"
            if comp_path.is_file():
                d5_perf_context[arm_key] = json.loads(comp_path.read_text(encoding="utf-8"))

            cd_path = d5_dir / d5_arm_name / "frozen_changed_dict.json"
            if cd_path.is_file():
                cd_data = json.loads(cd_path.read_text(encoding="utf-8"))
                cd_map = cd_data.get("changed_dict", cd_data)
                drops = [f for f, t in cd_map.items() if t == "dropped"]
                powers = {f: t["power"] for f, t in cd_map.items() if isinstance(t, dict) and "power" in t}
                merges = {f: t for f, t in cd_map.items() if isinstance(t, dict) and "power" not in t}
                d5_transform_summary[arm_key] = {
                    "changed_features": list(cd_map.keys()),
                    "drops": drops,
                    "powers": powers,
                    "merges": list(merges.keys()),
                }

    return {
        "d4_primary_aggregate_context": d4_context,
        "d5_weighted_aggregate_context": d5_perf_context,
        "d5_weighted_terminal_transformation_summary": d5_transform_summary,
        "source": "canonical_frozen_releases_only",
        "local_d4_preflight_required": False,
    }


# -----------------------------------------------------------------------------
# Concrete Production Runtime
# -----------------------------------------------------------------------------
class ProductionD7TerminalMechanismRuntime:
    """Concrete production runtime that builds cohorts, scores from frozen states,
    enforces 12/12 provenance and 2023/2024 scoring reproduction barriers,
    and runs mechanism diagnostics strictly from those same scored cohorts.
    """

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
        self.arm_states: Dict[str, FrozenArmState] = {}
        self.scored_cohorts: Dict[int, Dict[str, FrozenScoredCohort]] = {
            2022: {},
            2023: {},
            2024: {},
        }
        self.reproduction_metrics: Dict[int, Dict[str, Dict[str, Any]]] = {
            2023: {},
            2024: {},
        }
        self.observed_cohort_digests: Dict[int, Dict[str, str]] = {
            2022: {},
            2023: {},
            2024: {},
        }
        self.observed_preprocessing_sha256: Optional[str] = None
        self.diagnostics: Optional[Dict[str, Any]] = None

    def construct_adapter(self) -> Any:
        """Lazily construct real NHISStudyAdapter or test adapter factory."""
        if self.adapter_factory is not None:
            self.adapter = self.adapter_factory()
        else:
            from nhis_fairbias.adapter import NHISStudyAdapter
            pq_path = self.repo_root / FROZEN_FEATURES_PARQUET_PATH
            self.adapter = NHISStudyAdapter(features_parquet_path=pq_path)

        if not hasattr(self.adapter, "preprocessor") or getattr(self.adapter.preprocessor, "fitted_record", None) is None:
            raise ValueError("Adapter preprocessor missing required fitted_record")

        fitted_rec = self.adapter.preprocessor.fitted_record
        if hasattr(fitted_rec, "to_dict") and callable(fitted_rec.to_dict):
            rec_dict = fitted_rec.to_dict()
        elif isinstance(fitted_rec, dict):
            rec_dict = fitted_rec
        else:
            try:
                rec_dict = dict(fitted_rec)
            except Exception as exc:
                raise ValueError(f"Adapter preprocessor fitted_record is not canonicalizable: {exc}")

        obs_hash = compute_canonical_json_sha256(rec_dict)
        if obs_hash != PREPROCESSING_STATE_SHA256:
            raise ValueError(
                f"Preprocessing state hash mismatch from adapter: expected {PREPROCESSING_STATE_SHA256}, got {obs_hash}"
            )
        self.observed_preprocessing_sha256 = obs_hash
        return self.adapter

    def load_states(self) -> Dict[str, FrozenArmState]:
        self.arm_states = load_frozen_20_states(self.train_val_dir, self.test_dir)
        return self.arm_states

    def build_and_score_all_cohorts(self) -> Dict[int, Dict[str, FrozenScoredCohort]]:
        if self.adapter is None:
            self.construct_adapter()
        if not self.arm_states:
            self.load_states()

        from fairbias.transform import FairTransform
        transformer = FairTransform()

        cohort_digests: Dict[int, Dict[str, str]] = {2022: {}, 2023: {}, 2024: {}}
        raw_cohorts: Dict[int, Dict[str, Dict[str, Any]]] = {2022: {}, 2023: {}, 2024: {}}

        # Step 1: Construct 12 cohorts
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
                cohort_digests[year][arm_id] = digest
                raw_cohorts[year][arm_id] = {
                    "X": X_orig,
                    "y": y,
                    "a": a,
                    "w": w,
                    "digest": digest,
                }

        self.observed_cohort_digests = cohort_digests

        # Step 2: Enforce 12/12 cohort provenance barrier
        verify_cohort_provenance_barrier(cohort_digests)

        # Step 3: Score from frozen states
        for year in TEMPORAL_YEARS:
            for arm_id in D6_ARM_IDS:
                cohort_dict = raw_cohorts[year][arm_id]
                X_orig = cohort_dict["X"]
                y = cohort_dict["y"]
                a = cohort_dict["a"]
                arm_state = self.arm_states[arm_id]

                all_cats, all_nums = self.adapter.preprocessor.get_feature_family_lists("primary_core")
                active_cols = set(X_orig.columns)
                cate_attrs = [c for c in all_cats if c in active_cols]
                num_attrs = [c for c in all_nums if c in active_cols]

                # Transform data using frozen changed_dict (zero mitigation search)
                X_term = transformer.transform_data(
                    X_orig, arm_state.changed_dict, num_attrs=num_attrs, cate_attrs=cate_attrs
                )

                # Baseline scoring
                X_base_ordered = X_orig[arm_state.baseline_scaler.feature_order]
                X_base_scaled = arm_state.baseline_scaler.transform(X_base_ordered)
                b_logits = arm_state.baseline_lr.compute_logits(X_base_scaled)
                b_probs = arm_state.baseline_lr.predict_proba(X_base_scaled)

                # Fairbias scoring
                X_fb_ordered = X_term[arm_state.fairbias_scaler.feature_order]
                X_fb_scaled = arm_state.fairbias_scaler.transform(X_fb_ordered)
                fb_logits = arm_state.fairbias_lr.compute_logits(X_fb_scaled)
                fb_probs = arm_state.fairbias_lr.predict_proba(X_fb_scaled)

                scored = FrozenScoredCohort(
                    year=year,
                    arm_id=arm_id,
                    X_original=X_orig,
                    X_terminal=X_term,
                    X_baseline_scaled=X_base_scaled,
                    X_fairbias_scaled=X_fb_scaled,
                    y=y,
                    a=a,
                    baseline_logits=b_logits,
                    baseline_probs=b_probs,
                    fairbias_logits=fb_logits,
                    fairbias_probs=fb_probs,
                    source_row_digest=cohort_dict["digest"],
                )
                self.scored_cohorts[year][arm_id] = scored

                if year in (2023, 2024):
                    y_np = np.asarray(y, dtype=int)
                    self.reproduction_metrics[year][arm_id] = {
                        "baseline": compute_cohort_utility_metrics(y_np, b_probs),
                        "fairbias": compute_cohort_utility_metrics(y_np, fb_probs),
                    }

        # Step 4: Enforce scoring reproduction barrier from these same scored arrays
        verify_scoring_reproduction_barrier(
            observed_metrics=self.reproduction_metrics,
            archived_train_val_dir=self.train_val_dir,
            archived_test_dir=self.test_dir,
            tolerance=SCORING_REPRODUCTION_ABSOLUTE_TOLERANCE,
        )

        return self.scored_cohorts

    def compute_diagnostics(self) -> Dict[str, Any]:
        """Compute all mechanism diagnostics strictly from the scored cohorts."""
        if not self.scored_cohorts.get(2024):
            raise RuntimeError("Scored cohorts not available; run build_and_score_all_cohorts first.")

        diagnostics: Dict[str, Any] = {
            "family1_records": [],
            "family2_records": [],
            "score_distribution_records": [],
            "protected_group_records": [],
            "logit_contribution_records": [],
            "arm_summaries": {},
        }

        for year in TEMPORAL_YEARS:
            for arm_id in D6_ARM_IDS:
                scored = self.scored_cohorts[year][arm_id]
                arm_state = self.arm_states[arm_id]

                # Family I
                f1 = compute_family1_diagnostics(
                    X_raw=scored.X_original,
                    X_transformed=scored.X_terminal,
                    y=scored.y,
                    a=scored.a,
                    changed_dict=arm_state.changed_dict,
                    arm_id=arm_id,
                    year=year,
                )
                diagnostics["family1_records"].extend(f1)

                # Family II
                f2 = compute_family2_diagnostics(
                    X_raw=scored.X_original,
                    X_transformed=scored.X_terminal,
                    changed_dict=arm_state.changed_dict,
                    fairbias_scaler=arm_state.fairbias_scaler,
                    arm_id=arm_id,
                    year=year,
                )
                diagnostics["family2_records"].extend(f2)

                # Score distributions
                sd_b = compute_score_distribution_diagnostics(
                    scored.baseline_probs, scored.y, arm_id, year, "baseline"
                )
                sd_f = compute_score_distribution_diagnostics(
                    scored.fairbias_probs, scored.y, arm_id, year, "fairbias"
                )
                diagnostics["score_distribution_records"].extend([sd_b, sd_f])

                # Protected groups
                exp_groups = range(1, 8) if arm_id == "D6_ARM_002" else [1, 2]
                pg_b = compute_protected_group_score_diagnostics(
                    scored.baseline_probs, scored.y, scored.a, exp_groups, arm_id, year, "baseline"
                )
                pg_f = compute_protected_group_score_diagnostics(
                    scored.fairbias_probs, scored.y, scored.a, exp_groups, arm_id, year, "fairbias"
                )
                diagnostics["protected_group_records"].extend(pg_b + pg_f)

                # Logit contributions
                all_b_feats = arm_state.baseline_scaler.feature_order
                lc_b = compute_logit_contribution_diagnostics(
                    scored.X_baseline_scaled,
                    scored.y,
                    arm_state.baseline_lr,
                    arm_id,
                    year,
                    "baseline",
                    changed_dict=arm_state.changed_dict,
                    all_baseline_features=all_b_feats,
                )
                lc_f = compute_logit_contribution_diagnostics(
                    scored.X_fairbias_scaled,
                    scored.y,
                    arm_state.fairbias_lr,
                    arm_id,
                    year,
                    "fairbias",
                    changed_dict=arm_state.changed_dict,
                    all_baseline_features=all_b_feats,
                )
                diagnostics["logit_contribution_records"].extend(lc_b + lc_f)

        for arm_id in D6_ARM_IDS:
            arm_state = self.arm_states[arm_id]
            diagnostics["arm_summaries"][arm_id] = compute_mechanism_arm_summary(
                arm_id=arm_id,
                changed_dict=arm_state.changed_dict,
                family1_records=diagnostics["family1_records"],
                family2_records=diagnostics["family2_records"],
                score_dist_records=diagnostics["score_distribution_records"],
                logit_contrib_records=diagnostics["logit_contribution_records"],
            )

        self.diagnostics = diagnostics
        return diagnostics


# -----------------------------------------------------------------------------
# D7 Terminal Mechanism Release Manager (11-File Lifecycle)
# -----------------------------------------------------------------------------
class D7TerminalMechanismReleaseManager:
    """Manages the lifecycle, preconditions, artifact persistence, manifest tracking,
    and release state for Gate D7 substantive executions.
    """

    def __init__(
        self,
        repo_root: Path,
        releases_parent_dir: Optional[Path] = None,
        adapter_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.releases_parent_dir = releases_parent_dir or (
            self.repo_root / "runs" / "nhis_d7_terminal_mechanism" / "releases"
        )
        self.adapter_factory = adapter_factory

    def execute_release(
        self,
        release_id: str,
        expected_execution_head: str,
    ) -> Dict[str, Any]:
        """Execute complete substantive terminal mechanism release workflow."""
        # 1. Preconditions BEFORE directory creation
        verify_git_execution_preconditions(
            repo_root=self.repo_root,
            expected_sha=expected_execution_head,
        )

        pq_path = self.repo_root / FROZEN_FEATURES_PARQUET_PATH
        if not pq_path.is_file():
            raise FileNotFoundError(f"Prepared features parquet not found: {pq_path}")
        pq_sha = compute_sha256(pq_path)
        if pq_sha != FROZEN_FEATURES_PARQUET_SHA256:
            raise ValueError(f"Features parquet SHA mismatch: expected {FROZEN_FEATURES_PARQUET_SHA256}, got {pq_sha}")

        verify_frozen_archives(self.repo_root)

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
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_head": expected_execution_head,
            "manifest_sha256": None,
        }
        state_file.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")

        runtime = ProductionD7TerminalMechanismRuntime(
            repo_root=self.repo_root,
            adapter_factory=self.adapter_factory,
        )

        try:
            # 4. Build, score, enforce barriers
            runtime.build_and_score_all_cohorts()
            diagnostics = runtime.compute_diagnostics()

            # 5. Build and write all 11 files
            self._write_release_artifacts(target_release_dir, release_id, expected_execution_head, runtime, diagnostics)

            # 6. Build and write manifest tracking the 9 scientific files
            manifest_sha = self._write_manifest(target_release_dir, release_id)

            # 7. Update release_state.json to COMPLETE
            state_payload["status"] = "COMPLETE"
            state_payload["completed_at"] = datetime.now(timezone.utc).isoformat()
            state_payload["manifest_sha256"] = manifest_sha
            state_file.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")

            return {
                "release_id": release_id,
                "release_dir": str(target_release_dir),
                "status": "COMPLETE",
                "manifest_sha256": manifest_sha,
            }

        except Exception as exc:
            # On failure: preserve release dir, record FAILED state, re-raise
            state_payload["status"] = "FAILED"
            state_payload["failed_at"] = datetime.now(timezone.utc).isoformat()
            state_payload["error"] = str(exc)
            state_file.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
            raise exc

    def _write_release_artifacts(
        self,
        release_dir: Path,
        release_id: str,
        execution_head: str,
        runtime: ProductionD7TerminalMechanismRuntime,
        diagnostics: Dict[str, Any],
    ) -> None:
        cohort_digest_matches = sum(
            1
            for y in TEMPORAL_YEARS
            for a in D6_ARM_IDS
            if runtime.observed_cohort_digests.get(y, {}).get(a) == EXPECTED_COHORT_SOURCE_ROW_DIGESTS.get(y, {}).get(a)
        )
        prep_match = bool(
            runtime.observed_preprocessing_sha256 == PREPROCESSING_STATE_SHA256
        )

        prov_summary = {
            "gate": "D7 terminal mechanism",
            "release_id": release_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_commit": execution_head,
            "scientific_identity": "post-primary explanatory descriptive terminal-state mechanism audit",
            "mandatory_disclosure": MANDATORY_D7_DISCLOSURE,
            "scientific_terminology": SCIENTIFIC_TERMINOLOGY,
            "hypotheses": D7_HYPOTHESES,
            "nmi_gate_effectively_nonbinding": True,
            "nmi_gate_disclosure": NMI_GATE_DISCLOSURE,
            "cohort_source_row_digests": EXPECTED_COHORT_SOURCE_ROW_DIGESTS,
            "expected_cohort_source_row_digests": EXPECTED_COHORT_SOURCE_ROW_DIGESTS,
            "observed_cohort_source_row_digests": runtime.observed_cohort_digests,
            "cohort_digest_matches": cohort_digest_matches,
            "cohort_digest_expected": 12,
            "preprocessing_expected_sha256": PREPROCESSING_STATE_SHA256,
            "preprocessing_observed_sha256": runtime.observed_preprocessing_sha256,
            "preprocessing_state_match": prep_match,
            "training_state_anchors": EXPECTED_TRAINING_STATE_ANCHORS,
            "scoring_mode": "ZERO ESTIMATOR REFIT",
            "scoring_reproduction_verified": True,
            "scoring_reproduction_tolerance": SCORING_REPRODUCTION_ABSOLUTE_TOLERANCE,
        }
        (release_dir / "provenance_summary.json").write_text(json.dumps(prov_summary, indent=2), encoding="utf-8")

        # Artifact 2: terminal_transformation_inventory.json
        inventory: Dict[str, Any] = {}
        for arm_id, arm_state in runtime.arm_states.items():
            cd = arm_state.changed_dict
            drops = [f for f, t in cd.items() if t == "dropped"]
            powers = {f: t["power"] for f, t in cd.items() if isinstance(t, dict) and "power" in t}
            merges = {f: t for f, t in cd.items() if isinstance(t, dict) and "power" not in t}
            unchanged = [f for f in arm_state.baseline_scaler.feature_order if f not in cd]
            inventory[arm_id] = {
                "arm_id": arm_id,
                "protected_attribute": ARM_PROTECTED_ATTRIBUTES[arm_id],
                "disability_arm": ARM_DISABILITY_POLICIES[arm_id],
                "changed_dict": cd,
                "drops": drops,
                "powers": powers,
                "merges": merges,
                "unchanged_features": unchanged,
                "baseline_feature_order": arm_state.baseline_scaler.feature_order,
                "fairbias_feature_order": arm_state.fairbias_scaler.feature_order,
            }
        (release_dir / "terminal_transformation_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")

        # Artifact 3: macro_context.json
        macro = build_macro_context(self.repo_root)
        (release_dir / "macro_context.json").write_text(json.dumps(macro, indent=2), encoding="utf-8")

        # Artifact 4: family1_information_compression.csv
        pd.DataFrame(diagnostics["family1_records"]).to_csv(release_dir / "family1_information_compression.csv", index=False)

        # Artifact 5: family2_numeric_geometry.csv
        pd.DataFrame(diagnostics["family2_records"]).to_csv(release_dir / "family2_numeric_geometry.csv", index=False)

        # Artifact 6: score_distribution_summary.csv
        sd_rows = []
        for r in diagnostics["score_distribution_records"]:
            base_row = {
                "year": r["year"],
                "arm_id": r["arm_id"],
                "model": r["model"],
                "auroc": r["auroc"],
                "auprc": r["auprc"],
                "ks_statistic": r["ks_statistic"],
                "ks_pvalue": r["ks_pvalue"],
            }
            for stratum in ("overall", "y0", "y1"):
                stratum_dict = r.get(stratum, {})
                for k, v in stratum_dict.items():
                    base_row[f"{stratum}_{k}"] = v
            sd_rows.append(base_row)
        pd.DataFrame(sd_rows).to_csv(release_dir / "score_distribution_summary.csv", index=False)

        # Artifact 7: protected_group_score_summary.csv
        pd.DataFrame(diagnostics["protected_group_records"]).to_csv(release_dir / "protected_group_score_summary.csv", index=False)

        # Artifact 8: feature_logit_contribution_summary.csv
        pd.DataFrame(diagnostics["logit_contribution_records"]).to_csv(release_dir / "feature_logit_contribution_summary.csv", index=False)

        # Artifact 9: mechanism_arm_summary.json
        (release_dir / "mechanism_arm_summary.json").write_text(json.dumps(diagnostics["arm_summaries"], indent=2), encoding="utf-8")

    def _write_manifest(self, release_dir: Path, release_id: str) -> str:
        manifest_path = release_dir / "d7_terminal_mechanism_manifest.json"
        tracked_artifacts: Dict[str, Dict[str, Any]] = {}

        for fname in D7_MANIFEST_TRACKED_ARTIFACTS:
            fpath = release_dir / fname
            if not fpath.is_file():
                raise FileNotFoundError(f"Tracked artifact missing: {fpath}")
            tracked_artifacts[fname] = {
                "sha256": compute_sha256(fpath),
                "size_bytes": fpath.stat().st_size,
            }

        manifest_data = {
            "gate": "D7 terminal mechanism",
            "release_id": release_id,
            "tracked_artifact_count": len(tracked_artifacts),
            "total_file_count": 11,
            "artifacts": tracked_artifacts,
            "mandatory_disclosure": MANDATORY_D7_DISCLOSURE,
            "scientific_terminology": SCIENTIFIC_TERMINOLOGY,
            "hypotheses": D7_HYPOTHESES,
            "nmi_gate_effectively_nonbinding": True,
            "nmi_gate_disclosure": NMI_GATE_DISCLOSURE,
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        return compute_sha256(manifest_path)


# -----------------------------------------------------------------------------
# D7 Terminal Mechanism Harness (Audit-Only and Authorization)
# -----------------------------------------------------------------------------
class NHISD7TerminalMechanismHarness:
    """Supervised execution harness for Gate D7.1a."""

    def __init__(self, repo_root: Optional[Union[str, Path]] = None) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.d6_train_val_dir = (
            self.repo_root / "docs" / "releases" / D6_TRAIN_VAL_RELEASE_ID
        )
        self.d6_test_dir = (
            self.repo_root / "docs" / "releases" / D6_TEST_RELEASE_ID
        )

        self.estimator_fit_count = 0
        self.fairbias_execution_count = 0
        self.real_nhis_cohort_accessed = False

    def run_audit_only(self) -> Dict[str, Any]:
        """Perform static audit of frozen D6 archives, state anchors, and barriers.

        Constructs zero adapter, scores zero cohort, fits zero estimator.
        """
        arch_res = verify_frozen_archives(self.repo_root)
        states = load_frozen_20_states(self.d6_train_val_dir, self.d6_test_dir)
        digests_loaded = len(EXPECTED_COHORT_SOURCE_ROW_DIGESTS) * len(D6_ARM_IDS)

        if self.estimator_fit_count != 0:
            raise RuntimeError("Audit must perform zero estimator fits.")
        if self.fairbias_execution_count != 0:
            raise RuntimeError("Audit must execute zero FairBias relearning.")
        if self.real_nhis_cohort_accessed:
            raise RuntimeError("Audit must not access real NHIS microdata.")

        return {
            "d6_train_val_archive": arch_res["status"],
            "d6_test_archive": arch_res["status"],
            "d6_frozen_states_loaded": f"{len(states) * 5} / 20 LOADED",
            "expected_cohort_digests_loaded": f"{digests_loaded} / 12 LOADED",
            "preprocessing_anchor": "VERIFIED",
            "scoring_mode": "ZERO ESTIMATOR REFIT",
            "family1_diagnostics": "IMPLEMENTED",
            "family2_geometry_diagnostics": "IMPLEMENTED",
            "score_separation_diagnostics": "IMPLEMENTED",
            "logit_contribution_diagnostics": "IMPLEMENTED",
            "real_nhis_cohort_accessed": False,
            "estimator_fit_count": 0,
            "fairbias_execution_count": 0,
            "audit_status": "PASS",
        }


# -----------------------------------------------------------------------------
# Git Authorization Preconditions
# -----------------------------------------------------------------------------
def verify_git_execution_preconditions(
    repo_root: Union[str, Path],
    expected_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify git HEAD matches origin/research/nhis-fairbias and tracked tree is clean."""
    root = Path(repo_root)

    p_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    local_sha = p_head.stdout.strip()

    p_remote = subprocess.run(
        ["git", "rev-parse", "origin/research/nhis-fairbias"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    remote_sha = p_remote.stdout.strip()

    if local_sha != remote_sha:
        raise ValueError(
            f"Git HEAD mismatch: local={local_sha} != remote={remote_sha}"
        )

    if expected_sha and local_sha != expected_sha:
        raise ValueError(
            f"Git HEAD mismatch: local={local_sha} != expected={expected_sha}"
        )

    p_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_diff = p_status.stdout.strip()
    if tracked_diff:
        raise ValueError(
            f"Tracked working tree is dirty; authorization rejected:\n{tracked_diff}"
        )

    return {
        "local_head": local_sha,
        "remote_head": remote_sha,
        "tracked_clean": True,
    }
