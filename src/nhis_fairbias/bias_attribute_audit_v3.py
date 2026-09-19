"""Contracts for the NHIS limited-versus-expanded fairness audit.

This module is intentionally isolated from the historical four-arm runners.  It
contains only registry, alignment, inference-reporting, and validation-lock
contracts needed before any real A3 performance scan is allowed.

The functions here do not load NHIS microdata or model predictions.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


class AuditContractError(ValueError):
    """Raised when a frozen audit contract would be violated."""


CONCLUSION_WITHIN = "SUPPORTED_WITHIN_TOLERANCE"
CONCLUSION_EXCEEDS = "SUPPORTED_EXCEEDS_TOLERANCE"
CONCLUSION_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def canonical_sha256(value: Any) -> str:
    """Hash strict, canonical JSON without accepting NaN or Infinity."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AuditContractError("value is not strict canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for an existing regular file."""

    target = Path(path)
    if not target.is_file():
        raise AuditContractError("hash target is not an existing regular file")
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_keys(values: Iterable[Any], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise AuditContractError(f"{label} must be a sequence of record keys")
    raw_keys = tuple(values)
    if any(not isinstance(value, str) for value in raw_keys):
        raise AuditContractError(f"{label} must contain canonical string keys only")
    keys = raw_keys
    if not keys:
        raise AuditContractError(f"{label} must not be empty")
    if any(not key.strip() for key in keys):
        raise AuditContractError(f"{label} contains an empty record key")
    if len(set(keys)) != len(keys):
        raise AuditContractError(f"{label} contains duplicate record keys")
    return keys


def record_order_sha256(record_keys: Iterable[Any]) -> str:
    """Hash exact ordered record identity using collision-safe length framing."""

    keys = _record_keys(record_keys, "record_keys")
    digest = hashlib.sha256()
    digest.update(len(keys).to_bytes(8, byteorder="big", signed=False))
    for key in keys:
        encoded = key.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def verify_record_alignment(
    expected_record_keys: Iterable[Any],
    observed_record_keys: Iterable[Any],
    *,
    expected_order_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed on reordering, duplication, omission, or hash mismatch.

    Errors report only the mismatch position, never a respondent identifier.
    """

    expected = _record_keys(expected_record_keys, "expected_record_keys")
    observed = _record_keys(observed_record_keys, "observed_record_keys")
    expected_hash = record_order_sha256(expected)
    observed_hash = record_order_sha256(observed)
    if expected_order_sha256 is not None and expected_hash != expected_order_sha256:
        raise AuditContractError("expected record-order hash does not match expected keys")
    if len(expected) != len(observed):
        raise AuditContractError("record alignment length mismatch")
    if expected != observed:
        mismatch = next(i for i, pair in enumerate(zip(expected, observed)) if pair[0] != pair[1])
        raise AuditContractError(f"record alignment mismatch at position {mismatch}")
    if expected_hash != observed_hash:
        raise AuditContractError("record-order hash mismatch")
    return {"status": "ALIGNED", "rows": len(expected), "record_order_sha256": expected_hash}


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise AuditContractError(f"{label} must be a lowercase SHA-256 hex string")
    digest = value
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise AuditContractError(f"{label} must be a lowercase SHA-256 hex digest")
    return digest


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditContractError(f"{label} must be a nonempty string")
    return value


def _validate_seed_policy(seed_policy: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(seed_policy, Mapping):
        raise AuditContractError(f"{label} must be a mapping")
    policy = copy.deepcopy(dict(seed_policy))
    required = {
        "status",
        "seeds",
        "aggregation_estimand",
        "training_randomness_inference",
    }
    if set(policy) != required:
        raise AuditContractError(f"{label} fields do not match the registered schema")
    if policy["status"] != "FROZEN":
        raise AuditContractError(f"{label} must be frozen")
    seeds = policy["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise AuditContractError(f"{label} requires a nonempty ordered seeds list")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise AuditContractError(f"{label} seeds must be non-boolean integers")
    if len(seeds) != len(set(seeds)):
        raise AuditContractError(f"{label} seeds must be unique in their declared order")
    _require_nonempty_string(policy["aggregation_estimand"], f"{label} aggregation_estimand")
    _require_nonempty_string(
        policy["training_randomness_inference"],
        f"{label} training_randomness_inference",
    )
    return policy


def freeze_threshold_policy(
    policy_id: str,
    threshold: float,
    *,
    comparison_operator: str = ">=",
    selection_role: str = "FROZEN_WITHOUT_EVALUATION_RESULT_ACCESS",
) -> dict[str, Any]:
    """Create a hash-bound threshold policy for event-risk decisions."""

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise AuditContractError("threshold must be a real number, not boolean")
    threshold_value = float(threshold)
    policy_name = _require_nonempty_string(policy_id, "threshold policy_id")
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise AuditContractError("threshold must be finite and lie in [0, 1]")
    if comparison_operator != ">=":
        raise AuditContractError("only the frozen >= threshold operator is supported")
    if selection_role != "FROZEN_WITHOUT_EVALUATION_RESULT_ACCESS":
        raise AuditContractError("threshold selection_role is not the preregistered no-result-access state")
    policy = {
        "status": "FROZEN",
        "policy_id": policy_name,
        "threshold": threshold_value,
        "comparison_operator": comparison_operator,
        "selection_role": str(selection_role),
    }
    policy["threshold_policy_sha256"] = canonical_sha256(policy)
    return policy


def _verify_threshold_policy(policy: Mapping[str, Any]) -> tuple[float, str]:
    payload = copy.deepcopy(dict(policy))
    claimed_hash = payload.pop("threshold_policy_sha256", None)
    required = {
        "status",
        "policy_id",
        "threshold",
        "comparison_operator",
        "selection_role",
    }
    if set(payload) != required:
        raise AuditContractError("threshold policy fields do not match the registered schema")
    if payload.get("status") != "FROZEN" or not claimed_hash:
        raise AuditContractError("threshold policy is not frozen and hash-bound")
    _require_sha256(claimed_hash, "threshold_policy_sha256")
    if canonical_sha256(payload) != claimed_hash:
        raise AuditContractError("threshold policy hash mismatch")
    if payload.get("comparison_operator") != ">=":
        raise AuditContractError("threshold policy comparison operator must be >=")
    _require_nonempty_string(payload.get("policy_id"), "threshold policy_id")
    if payload.get("selection_role") != "FROZEN_WITHOUT_EVALUATION_RESULT_ACCESS":
        raise AuditContractError("threshold selection_role is not the preregistered no-result-access state")
    if not isinstance(payload.get("threshold"), (int, float)) or isinstance(payload.get("threshold"), bool):
        raise AuditContractError("threshold policy value must be numeric, not boolean")
    threshold = float(payload.get("threshold"))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise AuditContractError("threshold policy value must lie in [0, 1]")
    return threshold, claimed_hash


def validate_binary_decisions(values: Sequence[Any], *, label: str = "y_hat") -> np.ndarray:
    """Require bool or integer 0/1 decisions for FNR/FPR/BA inputs."""

    decisions = np.asarray(values)
    if decisions.ndim != 1 or not len(decisions):
        raise AuditContractError(f"{label} must be a nonempty one-dimensional array")
    if decisions.dtype.kind not in {"b", "i", "u"}:
        raise AuditContractError(f"{label} must contain bool or integer 0/1 decisions, not scores")
    if not np.all(np.isin(decisions, [0, 1])):
        raise AuditContractError(f"{label} contains values outside 0/1")
    return decisions.astype(bool, copy=False)


def bind_thresholded_decisions(
    record_keys: Iterable[Any],
    p_event: Sequence[float],
    threshold_policy: Mapping[str, Any],
    *,
    expected_order_sha256: str | None = None,
    expected_prediction_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bind p_event -> frozen threshold -> binary y_hat and all identities."""

    keys = _record_keys(record_keys, "record_keys")
    probabilities = np.asarray(p_event, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != len(keys):
        raise AuditContractError("p_event must be one-dimensional and aligned to record_keys")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise AuditContractError("p_event must contain finite probabilities in [0, 1]")
    threshold, threshold_hash = _verify_threshold_policy(threshold_policy)
    order_hash = record_order_sha256(keys)
    if expected_order_sha256 is not None:
        _require_sha256(expected_order_sha256, "expected_order_sha256")
        if order_hash != expected_order_sha256:
            raise AuditContractError("record-order hash mismatch before thresholding")
    prediction_hash = canonical_sha256([float(value) for value in probabilities])
    if expected_prediction_sha256 is not None:
        _require_sha256(expected_prediction_sha256, "expected_prediction_sha256")
        if prediction_hash != expected_prediction_sha256:
            raise AuditContractError("p_event prediction hash mismatch")
    decisions = validate_binary_decisions(probabilities >= threshold)
    decision_hash = canonical_sha256([int(value) for value in decisions])
    receipt = {
        "status": "BOUND_BINARY_DECISION",
        "risk_output_type": "event_probability_p",
        "decision_output_type": "binary_decision_y_hat",
        "threshold_comparison_operator": ">=",
        "rows": len(keys),
        "record_order_sha256": order_hash,
        "prediction_sha256": prediction_hash,
        "threshold_policy_sha256": threshold_hash,
        "y_hat_sha256": decision_hash,
        "decision_metrics_allowed": ["fnr", "fpr", "ba"],
        "risk_quality_input": "event_probability_p",
    }
    return decisions, receipt


def validate_metric_input(metric: str, values: Sequence[Any], *, output_type: str) -> np.ndarray:
    """Keep decision-rate metrics and probability-quality metrics disjoint."""

    metric_name = str(metric).lower()
    if metric_name in {"fnr", "fpr", "ba", "balanced_accuracy", "tpr", "tnr"}:
        if output_type != "binary_decision_y_hat":
            raise AuditContractError("FNR/FPR/BA-family metrics require binary_decision_y_hat")
        return validate_binary_decisions(values, label="decision-metric input")
    if metric_name in {"brier", "calibration", "log_loss"}:
        probabilities = np.asarray(values, dtype=float)
        if output_type != "event_probability_p":
            raise AuditContractError("risk-quality metrics require event_probability_p")
        if probabilities.ndim != 1 or not len(probabilities) or not np.all(np.isfinite(probabilities)):
            raise AuditContractError("risk-quality input must be a nonempty finite probability vector")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise AuditContractError("risk-quality probabilities must lie in [0, 1]")
        return probabilities
    raise AuditContractError("metric is not registered for this prediction-semantics contract")


def per_attribute_common_domain(
    outcome_valid: Sequence[bool],
    attribute_values: Sequence[Any],
    allowed_groups: Sequence[Any],
    prediction_valid_masks: Mapping[str, Sequence[bool]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build one O-specific domain shared by every member of a panel.

    The function accepts exactly one audit attribute.  Callers must invoke it
    separately for different O values; there is no all-O complete-case path.
    """

    y_mask = np.asarray(outcome_valid)
    o_values = np.asarray(attribute_values)
    if y_mask.ndim != 1 or o_values.ndim != 1 or len(y_mask) != len(o_values):
        raise AuditContractError("outcome and attribute arrays must be aligned one-dimensional arrays")
    if y_mask.dtype.kind != "b":
        raise AuditContractError("outcome_valid must be boolean")
    groups = tuple(allowed_groups)
    if not groups or len(set(groups)) != len(groups):
        raise AuditContractError("allowed_groups must be a nonempty unique sequence")
    domain = y_mask.copy() & np.isin(o_values, np.asarray(groups, dtype=object))
    method_counts: dict[str, int] = {}
    if not prediction_valid_masks:
        raise AuditContractError("at least one panel prediction mask is required")
    for method, values in prediction_valid_masks.items():
        mask = np.asarray(values)
        if mask.ndim != 1 or len(mask) != len(domain) or mask.dtype.kind != "b":
            raise AuditContractError("prediction-valid masks must be aligned boolean arrays")
        method_counts[str(method)] = int(mask.sum())
        domain &= mask
    return domain, {
        "status": "VALID" if np.any(domain) else "EMPTY_DOMAIN",
        "annual_rows": int(len(domain)),
        "domain_rows": int(domain.sum()),
        "allowed_groups": list(groups),
        "panel_prediction_valid_rows": method_counts,
    }


@dataclasses.dataclass(frozen=True)
class AuditAttributeSpec:
    variable: str
    groups: tuple[Any, ...]
    reference_group: Any
    audit_role: str
    O_group_rule_sha256: str
    audit_role_sha256: str
    O_registry_version: str
    O_registry_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.variable, "audit variable name")
        if len(self.groups) < 2 or len(set(self.groups)) != len(self.groups):
            raise AuditContractError("audit groups must contain at least two unique values")
        if self.reference_group not in self.groups:
            raise AuditContractError("reference group must be one of the declared groups")
        if self.audit_role not in {"anchor", "expanded_candidate"}:
            raise AuditContractError("audit_role must be anchor or expanded_candidate")
        _require_sha256(self.O_group_rule_sha256, "O_group_rule_sha256")
        _require_nonempty_string(self.O_registry_version, "O_registry_version")
        _require_sha256(self.O_registry_sha256, "O_registry_sha256")
        _require_sha256(self.audit_role_sha256, "audit_role_sha256")
        if self.audit_role_sha256 != audit_role_sha256(
            self.variable,
            self.audit_role,
            self.O_registry_version,
            self.O_registry_sha256,
        ):
            raise AuditContractError("audit_role_sha256 does not bind the attribute role and O registry")


ANCHOR_O_UNIVERSE = ("SEX_A", "HISPALLP_A", "DISAB3_A")
_CONTRAST_UNIVERSE_SCHEMA_VERSION = "nhis_bias_audit_contrast_universe_v3"
_CONTRAST_UNIVERSE_STATUS = "FROZEN_WITHOUT_PERFORMANCE_RESULT_ACCESS"


def audit_role_sha256(
    variable: str,
    audit_role: str,
    O_registry_version: str,
    O_registry_sha256: str,
) -> str:
    """Bind an audit attribute's anchor/new-O role to one O registry."""

    return canonical_sha256(
        {
            "O_audit": _require_nonempty_string(variable, "audit variable name"),
            "audit_role": _require_nonempty_string(audit_role, "audit_role"),
            "O_registry_version": _require_nonempty_string(
                O_registry_version,
                "O_registry_version",
            ),
            "O_registry_sha256": _require_sha256(
                O_registry_sha256,
                "O_registry_sha256",
            ),
        }
    )


def _attribute_record(attribute: AuditAttributeSpec) -> dict[str, Any]:
    return {
        "O_audit": attribute.variable,
        "groups": list(attribute.groups),
        "reference_group": attribute.reference_group,
        "audit_role": attribute.audit_role,
        "audit_role_sha256": attribute.audit_role_sha256,
        "O_group_rule_sha256": attribute.O_group_rule_sha256,
        "O_registry_version": attribute.O_registry_version,
        "O_registry_sha256": attribute.O_registry_sha256,
    }


def _attribute_from_record(record: Mapping[str, Any], *, label: str) -> AuditAttributeSpec:
    if not isinstance(record, Mapping):
        raise AuditContractError(f"{label} must be a mapping")
    required = {
        "O_audit",
        "groups",
        "reference_group",
        "audit_role",
        "audit_role_sha256",
        "O_group_rule_sha256",
        "O_registry_version",
        "O_registry_sha256",
    }
    if set(record) != required:
        raise AuditContractError(f"{label} fields do not match the registered schema")
    groups = record["groups"]
    if not isinstance(groups, list):
        raise AuditContractError(f"{label} groups must be a JSON list")
    return AuditAttributeSpec(
        variable=record["O_audit"],
        groups=tuple(groups),
        reference_group=record["reference_group"],
        audit_role=record["audit_role"],
        O_group_rule_sha256=record["O_group_rule_sha256"],
        audit_role_sha256=record["audit_role_sha256"],
        O_registry_version=record["O_registry_version"],
        O_registry_sha256=record["O_registry_sha256"],
    )


def _normalize_method_pairs(
    values: Sequence[tuple[str, str]] | Sequence[Sequence[str]],
    label: str,
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for raw_pair in values:
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
            raise AuditContractError(f"{label} entries must be method-baseline pairs")
        method = _require_nonempty_string(raw_pair[0], f"{label} method_id")
        baseline = _require_nonempty_string(raw_pair[1], f"{label} matched_baseline_id")
        if method == baseline:
            raise AuditContractError(f"{label} cannot contain a self-comparison")
        pairs.append((method, baseline))
    normalized = tuple(pairs)
    if len(set(normalized)) != len(normalized):
        raise AuditContractError(f"{label} must contain unique pairs")
    return normalized


def _contrast_universe_entry(
    *,
    family_name: str,
    question: str,
    identity: Mapping[str, Any],
    attribute: Mapping[str, Any],
    group: Any,
    endpoint: str,
    model_id: str | None = None,
    method_id: str | None = None,
    matched_baseline_id: str | None = None,
) -> dict[str, Any]:
    body = {
        "family_name": family_name,
        "question": question,
        "year": identity["year"],
        "evidence_role": identity["evidence_role"],
        "panel_id": identity["panel_id"],
        "panel_sha256": identity["panel_sha256"],
        "O_train": identity["O_train"],
        "seed_policy_sha256": identity["seed_policy_sha256"],
        "delta_rationale_id": identity["delta_rationale_id"],
        "O_audit": attribute["O_audit"],
        "audit_role": attribute["audit_role"],
        "audit_role_sha256": attribute["audit_role_sha256"],
        "O_group_rule_sha256": attribute["O_group_rule_sha256"],
        "O_registry_version": attribute["O_registry_version"],
        "O_registry_sha256": attribute["O_registry_sha256"],
        "group": group,
        "reference_group": attribute["reference_group"],
        "endpoint": endpoint,
    }
    if model_id is not None:
        body["model_id"] = model_id
        prefix = "q1"
    else:
        body["method_id"] = method_id
        body["matched_baseline_id"] = matched_baseline_id
        prefix = "method"
    return {**body, "contrast_id": _contrast_id(prefix, body)}


def _expected_contrast_universe(
    attributes: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    configurations = (
        (
            "q1_confirmatory",
            "Q1_FIXED_MODEL_GROUP_DIFFERENCE",
            ({"model_id": value} for value in identity["q1_model_ids"]),
        ),
        (
            "q3_q4_confirmatory",
            "Q3_Q4_CONFIRMATORY_METHOD_CHANGE",
            (
                {"method_id": pair[0], "matched_baseline_id": pair[1]}
                for pair in identity["confirmatory_method_pairs"]
            ),
        ),
        (
            "q3_q4_descriptive",
            "Q3_Q4_DESCRIPTIVE_METHOD_CHANGE",
            (
                {"method_id": pair[0], "matched_baseline_id": pair[1]}
                for pair in identity["descriptive_method_pairs"]
            ),
        ),
    )
    output: dict[str, list[dict[str, Any]]] = {name: [] for name in _STATISTICAL_FAMILY_NAMES}
    for family_name, question, member_iterator in configurations:
        members = tuple(member_iterator)
        for attribute in attributes:
            for group in attribute["groups"]:
                if group == attribute["reference_group"]:
                    continue
                for endpoint in identity["endpoints"]:
                    for member in members:
                        output[family_name].append(
                            _contrast_universe_entry(
                                family_name=family_name,
                                question=question,
                                identity=identity,
                                attribute=attribute,
                                group=group,
                                endpoint=endpoint,
                                **member,
                            )
                        )
    return output


def freeze_contrast_universe_receipt(
    anchor_attributes: Sequence[AuditAttributeSpec],
    new_O_attributes: Sequence[AuditAttributeSpec],
    *,
    O_registry_version: str,
    O_registry_sha256: str,
    selection_rule_id: str,
    selection_rule_sha256: str,
    q1_model_ids: Sequence[str],
    confirmatory_method_pairs: Sequence[tuple[str, str]],
    descriptive_method_pairs: Sequence[tuple[str, str]] = (),
    endpoints: Sequence[str] = ("fnr", "fpr"),
    year: int,
    evidence_role: str,
    panel_id: str,
    panel_sha256: str,
    O_train: str,
    seed_policy_sha256: str,
    delta_rationale_id: str,
) -> dict[str, Any]:
    """Freeze the exact anchor/new-O and contrast-slot universe without results."""

    registry_version = _require_nonempty_string(O_registry_version, "O_registry_version")
    registry_hash = _require_sha256(O_registry_sha256, "O_registry_sha256")
    anchor_records = [_attribute_record(value) for value in anchor_attributes]
    new_records = [_attribute_record(value) for value in new_O_attributes]
    anchor_names = [row["O_audit"] for row in anchor_records]
    new_names = [row["O_audit"] for row in new_records]
    if anchor_names != list(ANCHOR_O_UNIVERSE):
        raise AuditContractError("anchor attributes must equal the fixed ordered anchor universe")
    if len(new_names) != len(set(new_names)) or set(anchor_names) & set(new_names):
        raise AuditContractError("anchor and new-O universes must be unique and disjoint")
    if new_names != sorted(new_names):
        raise AuditContractError("new-O attributes must use canonical variable-name order")
    for record in anchor_records + new_records:
        expected_role = "anchor" if record in anchor_records else "expanded_candidate"
        if record["audit_role"] != expected_role:
            raise AuditContractError("attribute audit_role contradicts its frozen universe")
        if record["O_registry_version"] != registry_version or record["O_registry_sha256"] != registry_hash:
            raise AuditContractError("attribute O registry identity contradicts the universe registry")
    if not isinstance(year, int) or isinstance(year, bool) or year < 2000:
        raise AuditContractError("contrast universe year is invalid")
    if evidence_role not in {
        "training_calibration_history",
        "discovery",
        "known_retrospective_replication",
        "locked_one_shot_validation",
    }:
        raise AuditContractError("contrast universe evidence_role is invalid")
    model_ids = tuple(_require_nonempty_string(value, "q1 model_id") for value in q1_model_ids)
    if not model_ids or len(model_ids) != len(set(model_ids)):
        raise AuditContractError("q1_model_ids must be nonempty and unique")
    endpoint_values = tuple(_require_nonempty_string(value, "endpoint").lower() for value in endpoints)
    if not endpoint_values or len(endpoint_values) != len(set(endpoint_values)):
        raise AuditContractError("endpoints must be nonempty and unique")
    confirmatory_pairs = _normalize_method_pairs(confirmatory_method_pairs, "confirmatory_method_pairs")
    descriptive_pairs = _normalize_method_pairs(descriptive_method_pairs, "descriptive_method_pairs")
    if set(confirmatory_pairs) & set(descriptive_pairs):
        raise AuditContractError("confirmatory and descriptive method pairs must be disjoint")
    family_identity = {
        "year": year,
        "evidence_role": evidence_role,
        "panel_id": _require_nonempty_string(panel_id, "panel_id"),
        "panel_sha256": _require_sha256(panel_sha256, "panel_sha256"),
        "O_train": _require_nonempty_string(O_train, "O_train"),
        "seed_policy_sha256": _require_sha256(seed_policy_sha256, "seed_policy_sha256"),
        "delta_rationale_id": _require_nonempty_string(delta_rationale_id, "delta_rationale_id"),
        "endpoints": list(endpoint_values),
        "q1_model_ids": list(model_ids),
        "confirmatory_method_pairs": [list(value) for value in confirmatory_pairs],
        "descriptive_method_pairs": [list(value) for value in descriptive_pairs],
    }
    all_records = anchor_records + new_records
    receipt = {
        "schema_version": _CONTRAST_UNIVERSE_SCHEMA_VERSION,
        "status": _CONTRAST_UNIVERSE_STATUS,
        "O_registry": {"version": registry_version, "sha256": registry_hash},
        "selection_rule": {
            "selection_rule_id": _require_nonempty_string(selection_rule_id, "selection_rule_id"),
            "sha256": _require_sha256(selection_rule_sha256, "selection_rule_sha256"),
        },
        "panel_family_identity": family_identity,
        "anchor_O_set": anchor_names,
        "new_O_set": new_names,
        "attribute_identities": all_records,
        "expected_contrasts": _expected_contrast_universe(all_records, family_identity),
    }
    receipt["contrast_universe_sha256"] = canonical_sha256(receipt)
    verify_contrast_universe_receipt(receipt)
    return receipt


def verify_contrast_universe_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on role, registry, membership, or exact-slot drift."""

    value = copy.deepcopy(dict(receipt))
    claimed_hash = value.pop("contrast_universe_sha256", None)
    _require_sha256(claimed_hash, "contrast_universe_sha256")
    if canonical_sha256(value) != claimed_hash:
        raise AuditContractError("contrast universe receipt hash mismatch")
    required = {
        "schema_version",
        "status",
        "O_registry",
        "selection_rule",
        "panel_family_identity",
        "anchor_O_set",
        "new_O_set",
        "attribute_identities",
        "expected_contrasts",
    }
    if set(value) != required:
        raise AuditContractError("contrast universe receipt fields do not match the registered schema")
    if value["schema_version"] != _CONTRAST_UNIVERSE_SCHEMA_VERSION or value["status"] != _CONTRAST_UNIVERSE_STATUS:
        raise AuditContractError("contrast universe receipt schema or status is invalid")
    registry = value["O_registry"]
    if not isinstance(registry, Mapping) or set(registry) != {"version", "sha256"}:
        raise AuditContractError("contrast universe O_registry is invalid")
    registry_version = _require_nonempty_string(registry["version"], "O_registry version")
    registry_hash = _require_sha256(registry["sha256"], "O_registry sha256")
    selection = value["selection_rule"]
    if not isinstance(selection, Mapping) or set(selection) != {"selection_rule_id", "sha256"}:
        raise AuditContractError("contrast universe selection_rule is invalid")
    _require_nonempty_string(selection["selection_rule_id"], "selection_rule_id")
    _require_sha256(selection["sha256"], "selection_rule sha256")
    identity = value["panel_family_identity"]
    expected_identity_fields = {
        "year",
        "evidence_role",
        "panel_id",
        "panel_sha256",
        "O_train",
        "seed_policy_sha256",
        "delta_rationale_id",
        "endpoints",
        "q1_model_ids",
        "confirmatory_method_pairs",
        "descriptive_method_pairs",
    }
    if not isinstance(identity, Mapping) or set(identity) != expected_identity_fields:
        raise AuditContractError("contrast universe panel_family_identity is invalid")
    if not isinstance(identity["year"], int) or isinstance(identity["year"], bool) or identity["year"] < 2000:
        raise AuditContractError("contrast universe year is invalid")
    if identity["evidence_role"] not in {
        "training_calibration_history",
        "discovery",
        "known_retrospective_replication",
        "locked_one_shot_validation",
    }:
        raise AuditContractError("contrast universe evidence_role is invalid")
    for field in ("panel_id", "O_train", "delta_rationale_id"):
        _require_nonempty_string(identity[field], f"contrast universe {field}")
    _require_sha256(identity["panel_sha256"], "contrast universe panel_sha256")
    _require_sha256(identity["seed_policy_sha256"], "contrast universe seed_policy_sha256")
    endpoints = identity["endpoints"]
    models = identity["q1_model_ids"]
    if not isinstance(endpoints, list) or not endpoints or any(not isinstance(value, str) or not value for value in endpoints):
        raise AuditContractError("contrast universe endpoints are invalid")
    if len(endpoints) != len(set(endpoints)):
        raise AuditContractError("contrast universe endpoints must be unique")
    if not isinstance(models, list) or not models or any(not isinstance(value, str) or not value for value in models):
        raise AuditContractError("contrast universe q1_model_ids are invalid")
    if len(models) != len(set(models)):
        raise AuditContractError("contrast universe q1_model_ids must be unique")
    confirmatory_pairs = _normalize_method_pairs(identity["confirmatory_method_pairs"], "confirmatory_method_pairs")
    descriptive_pairs = _normalize_method_pairs(identity["descriptive_method_pairs"], "descriptive_method_pairs")
    if set(confirmatory_pairs) & set(descriptive_pairs):
        raise AuditContractError("contrast universe method families overlap")
    if value["anchor_O_set"] != list(ANCHOR_O_UNIVERSE):
        raise AuditContractError("contrast universe is missing or changing the fixed anchor O set")
    new_names = value["new_O_set"]
    if not isinstance(new_names, list) or new_names != sorted(new_names) or len(new_names) != len(set(new_names)):
        raise AuditContractError("contrast universe new-O set is not canonical and unique")
    if set(value["anchor_O_set"]) & set(new_names):
        raise AuditContractError("the same O cannot appear in anchor and new-O universes")
    records = value["attribute_identities"]
    if not isinstance(records, list):
        raise AuditContractError("contrast universe attribute_identities must be a list")
    validated_records = []
    seen_names: set[str] = set()
    for record in records:
        attribute = _attribute_from_record(record, label="contrast universe attribute")
        if attribute.variable in seen_names:
            raise AuditContractError("contrast universe contains a duplicate O_audit")
        seen_names.add(attribute.variable)
        expected_role = "anchor" if attribute.variable in ANCHOR_O_UNIVERSE else "expanded_candidate"
        if attribute.audit_role != expected_role:
            raise AuditContractError("contrast universe audit_role contradicts O membership")
        if attribute.O_registry_version != registry_version or attribute.O_registry_sha256 != registry_hash:
            raise AuditContractError("contrast universe attribute registry identity mismatch")
        validated_records.append(_attribute_record(attribute))
    if [row["O_audit"] for row in validated_records] != list(ANCHOR_O_UNIVERSE) + new_names:
        raise AuditContractError("contrast universe attribute records do not match the declared O sets")
    expected_contrasts = _expected_contrast_universe(validated_records, identity)
    if value["expected_contrasts"] != expected_contrasts:
        raise AuditContractError("contrast universe expected slots do not match frozen identities")
    all_ids = [
        row["contrast_id"]
        for family_name in _STATISTICAL_FAMILY_NAMES
        for row in expected_contrasts[family_name]
    ]
    if len(all_ids) != len(set(all_ids)):
        raise AuditContractError("contrast universe identifiers are not globally unique")
    return {**value, "contrast_universe_sha256": claimed_hash}


def _contrast_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{canonical_sha256(payload)[:20]}"


_STATISTICAL_FAMILY_NAMES = (
    "q1_confirmatory",
    "q3_q4_confirmatory",
    "q3_q4_descriptive",
)


_FAMILY_MANIFEST_SCHEMA_VERSION = "nhis_bias_audit_families_v3"
_FAMILY_MANIFEST_STATUS = "STRUCTURE_GENERATED_CONTENT_REQUIRES_SCIENTIFIC_FREEZE"


def generate_statistical_families(
    attributes: Sequence[AuditAttributeSpec],
    *,
    contrast_universe_receipt: Mapping[str, Any],
    q1_model_ids: Sequence[str],
    confirmatory_method_pairs: Sequence[tuple[str, str]],
    descriptive_method_pairs: Sequence[tuple[str, str]] = (),
    endpoints: Sequence[str] = ("fnr", "fpr"),
    year: int,
    evidence_role: str,
    scope: str,
    panel_id: str,
    panel_sha256: str,
    O_train: str,
    seed_policy: Mapping[str, Any],
    delta_rationale_id: str,
) -> dict[str, Any]:
    """Generate complete, hash-bound family manifests with explicit NA slots.

    Q1 tests d(model, O, group) = 0.  The method family tests
    d(method, O, group) - d(matched baseline, O, group) = 0.  These nulls are
    never pooled into a single family.
    """

    if not attributes:
        raise AuditContractError("at least one audit attribute is required")
    verified_universe = verify_contrast_universe_receipt(contrast_universe_receipt)
    universe_hash = verified_universe["contrast_universe_sha256"]
    attribute_records = [_attribute_record(value) for value in attributes]
    attribute_names = [row["O_audit"] for row in attribute_records]
    if len(attribute_names) != len(set(attribute_names)):
        raise AuditContractError("attributes cannot repeat an O_audit")
    model_ids = tuple(_require_nonempty_string(value, "q1 model_id") for value in q1_model_ids)
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise AuditContractError("q1_model_ids must be nonempty and unique")
    endpoint_values = tuple(_require_nonempty_string(value, "endpoint").lower() for value in endpoints)
    if not endpoint_values or len(set(endpoint_values)) != len(endpoint_values):
        raise AuditContractError("endpoints must be nonempty and unique")
    if not isinstance(year, int) or isinstance(year, bool) or year < 2000:
        raise AuditContractError("year must be an explicit survey year")
    evidence_roles = {
        "training_calibration_history",
        "discovery",
        "known_retrospective_replication",
        "locked_one_shot_validation",
    }
    if evidence_role not in evidence_roles:
        raise AuditContractError("evidence_role is not registered")
    if scope not in {"anchor", "expanded"}:
        raise AuditContractError("scope must be anchor or expanded")
    _require_nonempty_string(panel_id, "panel_id")
    _require_nonempty_string(O_train, "O_train")
    _require_nonempty_string(delta_rationale_id, "delta_rationale_id")
    panel_digest = _require_sha256(panel_sha256, "panel_sha256")
    seed_contract = _validate_seed_policy(seed_policy, label="seed_policy")
    seed_policy_hash = canonical_sha256(seed_contract)

    confirmatory_pairs = _normalize_method_pairs(confirmatory_method_pairs, "confirmatory_method_pairs")
    descriptive_pairs = _normalize_method_pairs(descriptive_method_pairs, "descriptive_method_pairs")
    if set(confirmatory_pairs) & set(descriptive_pairs):
        raise AuditContractError("confirmatory and descriptive method pairs must be disjoint")
    receipt_identity = verified_universe["panel_family_identity"]
    supplied_identity = {
        "year": int(year),
        "evidence_role": evidence_role,
        "panel_id": panel_id,
        "panel_sha256": panel_digest,
        "O_train": O_train,
        "seed_policy_sha256": seed_policy_hash,
        "delta_rationale_id": delta_rationale_id,
        "endpoints": list(endpoint_values),
        "q1_model_ids": list(model_ids),
        "confirmatory_method_pairs": [list(value) for value in confirmatory_pairs],
        "descriptive_method_pairs": [list(value) for value in descriptive_pairs],
    }
    if supplied_identity != receipt_identity:
        raise AuditContractError("family inputs do not match the frozen contrast universe identity")
    anchor_records = [
        row for row in verified_universe["attribute_identities"] if row["audit_role"] == "anchor"
    ]
    new_records = [
        row
        for row in verified_universe["attribute_identities"]
        if row["audit_role"] == "expanded_candidate"
    ]
    expected_records = anchor_records if scope == "anchor" else anchor_records + new_records
    if attribute_records != expected_records:
        raise AuditContractError("scope attributes do not exactly match the frozen contrast universe")

    shared_identity = {
        "year": int(year),
        "evidence_role": evidence_role,
        "scope": scope,
        "panel_id": panel_id,
        "panel_sha256": panel_digest,
        "O_train": O_train,
        "seed_policy": seed_contract,
        "seed_policy_sha256": seed_policy_hash,
        "delta_rationale_id": delta_rationale_id,
        "contrast_universe_sha256": universe_hash,
    }

    q1: list[dict[str, Any]] = []
    method_confirmatory: list[dict[str, Any]] = []
    method_descriptive: list[dict[str, Any]] = []
    for attribute in attributes:
        for group in attribute.groups:
            if group == attribute.reference_group:
                continue
            for endpoint in endpoint_values:
                for model_id in model_ids:
                    body = {
                        **shared_identity,
                        "family_name": "q1_confirmatory",
                        "question": "Q1_FIXED_MODEL_GROUP_DIFFERENCE",
                        "model_id": model_id,
                        "O_audit": attribute.variable,
                        "audit_role": attribute.audit_role,
                        "audit_role_sha256": attribute.audit_role_sha256,
                        "O_group_rule_sha256": attribute.O_group_rule_sha256,
                        "O_registry_version": attribute.O_registry_version,
                        "O_registry_sha256": attribute.O_registry_sha256,
                        "group": group,
                        "reference_group": attribute.reference_group,
                        "endpoint": endpoint,
                    }
                    expected = _contrast_universe_entry(
                        family_name="q1_confirmatory",
                        question="Q1_FIXED_MODEL_GROUP_DIFFERENCE",
                        identity=receipt_identity,
                        attribute=_attribute_record(attribute),
                        group=group,
                        endpoint=endpoint,
                        model_id=model_id,
                    )
                    q1.append(
                        {
                            **body,
                            "contrast_id": expected["contrast_id"],
                            "null_hypothesis": "d_model_O_group = 0",
                            "slot_status": "PREDECLARED_NOT_YET_ESTIMATED",
                            "estimate": None,
                        }
                    )
                for family, pairs, output in (
                    ("Q3_Q4_CONFIRMATORY_METHOD_CHANGE", confirmatory_pairs, method_confirmatory),
                    ("Q3_Q4_DESCRIPTIVE_METHOD_CHANGE", descriptive_pairs, method_descriptive),
                ):
                    for method_id, baseline_id in pairs:
                        body = {
                            **shared_identity,
                            "family_name": (
                                "q3_q4_confirmatory"
                                if family == "Q3_Q4_CONFIRMATORY_METHOD_CHANGE"
                                else "q3_q4_descriptive"
                            ),
                            "question": family,
                            "method_id": method_id,
                            "matched_baseline_id": baseline_id,
                            "O_audit": attribute.variable,
                            "audit_role": attribute.audit_role,
                            "audit_role_sha256": attribute.audit_role_sha256,
                            "O_group_rule_sha256": attribute.O_group_rule_sha256,
                            "O_registry_version": attribute.O_registry_version,
                            "O_registry_sha256": attribute.O_registry_sha256,
                            "group": group,
                            "reference_group": attribute.reference_group,
                            "endpoint": endpoint,
                        }
                        expected = _contrast_universe_entry(
                            family_name=body["family_name"],
                            question=family,
                            identity=receipt_identity,
                            attribute=_attribute_record(attribute),
                            group=group,
                            endpoint=endpoint,
                            method_id=method_id,
                            matched_baseline_id=baseline_id,
                        )
                        output.append(
                            {
                                **body,
                                "contrast_id": expected["contrast_id"],
                                "null_hypothesis": "d_method_O_group - d_matched_baseline_O_group = 0",
                                "slot_status": "PREDECLARED_NOT_YET_ESTIMATED",
                                "estimate": None,
                            }
                        )
    all_ids = [row["contrast_id"] for row in q1 + method_confirmatory + method_descriptive]
    if len(all_ids) != len(set(all_ids)):
        raise AuditContractError("generated contrast identifiers are not unique")
    families = {
        "q1_confirmatory": q1,
        "q3_q4_confirmatory": method_confirmatory,
        "q3_q4_descriptive": method_descriptive,
    }
    family_manifests: dict[str, Any] = {}
    for family_name, slots in families.items():
        identity = {
            **shared_identity,
            "family_name": family_name,
            "confirmatory": family_name != "q3_q4_descriptive",
            "slot_count_including_not_estimable": len(slots),
        }
        family_id = _contrast_id("family", identity)
        family_hash = canonical_sha256({"identity": identity, "slots": slots})
        for slot in slots:
            slot["family_id"] = family_id
            slot["family_sha256"] = family_hash
        family_manifests[family_name] = {
            **identity,
            "family_id": family_id,
            "family_sha256": family_hash,
            "na_slots_retained": True,
        }
    manifest = {
        "schema_version": _FAMILY_MANIFEST_SCHEMA_VERSION,
        "manifest_status": _FAMILY_MANIFEST_STATUS,
        "manifest_identity": shared_identity,
        "contrast_universe_receipt": verified_universe,
        "family_manifests": family_manifests,
        "q1_confirmatory": q1,
        "q3_q4_confirmatory": method_confirmatory,
        "q3_q4_descriptive": method_descriptive,
        "family_sizes": {
            "q1_confirmatory": len(q1),
            "q3_q4_confirmatory": len(method_confirmatory),
            "q3_q4_descriptive": len(method_descriptive),
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def verify_statistical_family_manifest(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Validate and hash-bind the complete manifest, not only child families."""

    manifest_copy = copy.deepcopy(dict(manifest))
    claimed_manifest_hash = manifest_copy.pop("manifest_sha256", None)
    _require_sha256(claimed_manifest_hash, "manifest_sha256")
    if canonical_sha256(manifest_copy) != claimed_manifest_hash:
        raise AuditContractError("manifest_sha256 mismatch")
    if manifest_copy.get("schema_version") != _FAMILY_MANIFEST_SCHEMA_VERSION:
        raise AuditContractError("family manifest schema_version is not registered")
    if manifest_copy.get("manifest_status") != _FAMILY_MANIFEST_STATUS:
        raise AuditContractError("family manifest status is not registered")
    expected_top_level = {
        "schema_version",
        "manifest_status",
        "manifest_identity",
        "contrast_universe_receipt",
        "family_manifests",
        "family_sizes",
        *_STATISTICAL_FAMILY_NAMES,
    }
    if set(manifest_copy) != expected_top_level:
        raise AuditContractError("family manifest top-level fields do not match the registered schema")
    shared_identity = manifest_copy.get("manifest_identity")
    if not isinstance(shared_identity, dict):
        raise AuditContractError("manifest_identity must be a mapping")
    required_shared_identity = {
        "year",
        "evidence_role",
        "scope",
        "panel_id",
        "panel_sha256",
        "O_train",
        "seed_policy",
        "seed_policy_sha256",
        "delta_rationale_id",
        "contrast_universe_sha256",
    }
    if set(shared_identity) != required_shared_identity:
        raise AuditContractError("manifest_identity fields do not match the registered schema")
    if (
        not isinstance(shared_identity["year"], int)
        or isinstance(shared_identity["year"], bool)
        or shared_identity["year"] < 2000
    ):
        raise AuditContractError("manifest_identity year is invalid")
    if shared_identity["evidence_role"] not in {
        "training_calibration_history",
        "discovery",
        "known_retrospective_replication",
        "locked_one_shot_validation",
    }:
        raise AuditContractError("manifest_identity evidence_role is invalid")
    if shared_identity["scope"] not in {"anchor", "expanded"}:
        raise AuditContractError("manifest_identity scope is invalid")
    for field in ("panel_id", "O_train", "delta_rationale_id"):
        if not isinstance(shared_identity[field], str) or not shared_identity[field].strip():
            raise AuditContractError(f"manifest_identity {field} is invalid")
    _require_sha256(shared_identity["panel_sha256"], "panel_sha256")
    validated_seed_policy = _validate_seed_policy(
        shared_identity["seed_policy"],
        label="manifest seed_policy",
    )
    if validated_seed_policy != shared_identity["seed_policy"]:
        raise AuditContractError("manifest seed_policy is not canonical")
    if canonical_sha256(shared_identity["seed_policy"]) != shared_identity["seed_policy_sha256"]:
        raise AuditContractError("manifest seed_policy_sha256 mismatch")
    _require_sha256(shared_identity["contrast_universe_sha256"], "contrast_universe_sha256")
    verified_universe = verify_contrast_universe_receipt(manifest_copy["contrast_universe_receipt"])
    if verified_universe["contrast_universe_sha256"] != shared_identity["contrast_universe_sha256"]:
        raise AuditContractError("manifest does not bind the embedded contrast universe receipt")
    universe_identity = verified_universe["panel_family_identity"]
    for field in (
        "year",
        "evidence_role",
        "panel_id",
        "panel_sha256",
        "O_train",
        "seed_policy_sha256",
        "delta_rationale_id",
    ):
        if shared_identity[field] != universe_identity[field]:
            raise AuditContractError("manifest identity contradicts contrast universe identity")
    allowed_roles = {"anchor"} if shared_identity["scope"] == "anchor" else {"anchor", "expanded_candidate"}

    expected_names = set(_STATISTICAL_FAMILY_NAMES)
    family_manifests = manifest_copy.get("family_manifests")
    if not isinstance(family_manifests, dict) or set(family_manifests) != expected_names:
        raise AuditContractError("family_manifests must contain the exact registered family set")
    family_sizes = manifest_copy.get("family_sizes")
    if not isinstance(family_sizes, dict) or set(family_sizes) != expected_names:
        raise AuditContractError("family_sizes must contain the exact registered family set")
    verified: dict[str, str] = {}
    all_contrast_ids: list[str] = []
    for family_name in _STATISTICAL_FAMILY_NAMES:
        family = family_manifests[family_name]
        slots = manifest_copy.get(family_name)
        if not isinstance(slots, list):
            raise AuditContractError("family manifest has no aligned slot list")
        family_row = dict(family)
        claimed_id = family_row.pop("family_id", None)
        claimed_hash = family_row.pop("family_sha256", None)
        if family_row.pop("na_slots_retained", None) is not True:
            raise AuditContractError("family manifest must retain NA slots")
        family_shared = {key: family_row.get(key) for key in required_shared_identity}
        if family_shared != shared_identity:
            raise AuditContractError("family identity contradicts manifest_identity")
        if family_row.get("family_name") != family_name:
            raise AuditContractError("family_name mismatch")
        expected_confirmatory = family_name != "q3_q4_descriptive"
        if family_row.get("confirmatory") is not expected_confirmatory:
            raise AuditContractError("family confirmatory flag mismatch")
        if family_row.get("slot_count_including_not_estimable") != len(slots):
            raise AuditContractError("family slot count mismatch")
        if family_sizes[family_name] != len(slots):
            raise AuditContractError("family_sizes does not match actual slots")
        if claimed_id != _contrast_id("family", family_row):
            raise AuditContractError("family_id mismatch")
        _require_sha256(claimed_hash, "family_sha256")
        stripped_slots = []
        expected_question = {
            "q1_confirmatory": "Q1_FIXED_MODEL_GROUP_DIFFERENCE",
            "q3_q4_confirmatory": "Q3_Q4_CONFIRMATORY_METHOD_CHANGE",
            "q3_q4_descriptive": "Q3_Q4_DESCRIPTIVE_METHOD_CHANGE",
        }[family_name]
        expected_null = (
            "d_model_O_group = 0"
            if family_name == "q1_confirmatory"
            else "d_method_O_group - d_matched_baseline_O_group = 0"
        )
        expected_slots = [
            row
            for row in verified_universe["expected_contrasts"][family_name]
            if row["audit_role"] in allowed_roles
        ]
        expected_by_id = {row["contrast_id"]: row for row in expected_slots}
        observed_ids: set[str] = set()
        for slot in slots:
            slot_copy = dict(slot)
            if slot_copy.pop("family_id", None) != claimed_id:
                raise AuditContractError("slot family_id mismatch")
            if slot_copy.pop("family_sha256", None) != claimed_hash:
                raise AuditContractError("slot family_sha256 mismatch")
            if {key: slot_copy.get(key) for key in required_shared_identity} != shared_identity:
                raise AuditContractError("slot identity contradicts manifest_identity")
            if slot_copy.get("question") != expected_question:
                raise AuditContractError("slot question does not match its family")
            if slot_copy.get("null_hypothesis") != expected_null:
                raise AuditContractError("slot null hypothesis does not match its family")
            if slot_copy.get("slot_status") != "PREDECLARED_NOT_YET_ESTIMATED":
                raise AuditContractError("generated manifest contains an invalid slot_status")
            if slot_copy.get("estimate") is not None:
                raise AuditContractError("predeclared unestimated slots must have estimate=null")
            if not slot_copy.get("O_audit") or not slot_copy.get("endpoint"):
                raise AuditContractError("slot is missing its audit attribute or endpoint")
            if slot_copy.get("audit_role") not in allowed_roles:
                raise AuditContractError("slot audit_role is outside the manifest scope")
            _require_sha256(slot_copy.get("audit_role_sha256"), "slot audit_role_sha256")
            _require_sha256(slot_copy.get("O_group_rule_sha256"), "slot O_group_rule_sha256")
            _require_nonempty_string(slot_copy.get("O_registry_version"), "slot O_registry_version")
            _require_sha256(slot_copy.get("O_registry_sha256"), "slot O_registry_sha256")
            if slot_copy.get("group") == slot_copy.get("reference_group"):
                raise AuditContractError("slot group must differ from its reference group")
            if family_name == "q1_confirmatory":
                if not slot_copy.get("model_id") or "method_id" in slot_copy or "matched_baseline_id" in slot_copy:
                    raise AuditContractError("Q1 slot must bind exactly one model_id")
            else:
                if (
                    not slot_copy.get("method_id")
                    or not slot_copy.get("matched_baseline_id")
                    or slot_copy["method_id"] == slot_copy["matched_baseline_id"]
                    or "model_id" in slot_copy
                ):
                    raise AuditContractError("method slot must bind one non-self method-baseline pair")
            contrast_id = slot_copy.get("contrast_id")
            expected_slot = expected_by_id.get(contrast_id)
            if expected_slot is None:
                raise AuditContractError("slot is outside the frozen contrast universe")
            observed_projection = {field: slot_copy.get(field) for field in expected_slot if field != "contrast_id"}
            expected_projection = {field: value for field, value in expected_slot.items() if field != "contrast_id"}
            if observed_projection != expected_projection:
                raise AuditContractError("slot identity contradicts the frozen contrast universe")
            observed_ids.add(contrast_id)
            all_contrast_ids.append(contrast_id)
            stripped_slots.append(slot_copy)
        if observed_ids != set(expected_by_id):
            raise AuditContractError("family slots omit or add frozen contrast-universe entries")
        observed_hash = canonical_sha256({"identity": family_row, "slots": stripped_slots})
        if observed_hash != claimed_hash:
            raise AuditContractError("family_sha256 mismatch")
        verified[str(family_name)] = observed_hash
    if len(all_contrast_ids) != len(set(all_contrast_ids)):
        raise AuditContractError("contrast identifiers must be unique across the full manifest")
    return verified


def holm_adjust(p_values: Mapping[str, float | None]) -> dict[str, Any]:
    """Holm-adjust p-values while preserving NA slots in the family size.

    Missing/non-finite p-values receive a separate conservative value of 1.0
    for multiplicity accounting.  This function deliberately returns no
    confidence intervals: Holm p-values are not simultaneous intervals.
    """

    if not p_values:
        raise AuditContractError("p-value family must not be empty")
    rows = []
    for contrast_id, raw in p_values.items():
        if not contrast_id:
            raise AuditContractError("contrast identifiers must be nonempty")
        raw_value = None if raw is None else float(raw)
        valid = raw_value is not None and math.isfinite(raw_value) and 0.0 <= raw_value <= 1.0
        if raw_value is not None and not valid and math.isfinite(raw_value):
            raise AuditContractError("finite p-values must lie in [0, 1]")
        rows.append(
            {
                "contrast_id": str(contrast_id),
                "raw_p": raw_value if valid else None,
                "family_p_for_adjustment": raw_value if valid else 1.0,
                "raw_status": "VALID" if valid else "NOT_ESTIMABLE",
            }
        )
    family_size = len(rows)
    ordered = sorted(range(family_size), key=lambda index: (rows[index]["family_p_for_adjustment"], rows[index]["contrast_id"]))
    running = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (family_size - rank) * rows[index]["family_p_for_adjustment"])
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running
    return {
        "method": "HOLM_P_VALUES_ONLY",
        "family_size_including_not_estimable": family_size,
        "rows": sorted(rows, key=lambda row: row["contrast_id"]),
        "confidence_interval_note": "No confidence interval is produced by Holm adjustment.",
    }


def simultaneous_intervals_from_replicates(
    point_estimates: Sequence[float],
    replicate_estimates: Sequence[Sequence[float]],
    *,
    labels: Sequence[str] | None = None,
    alpha: float = 0.05,
    min_complete_fraction: float = 0.95,
) -> dict[str, Any]:
    """Build separate ordinary and max-standardized replicate intervals.

    This is an A2 candidate implementation for a predeclared contrast family.
    Production use still requires method/reference validation and coverage QA.
    A post-hoc selected maximum pair must not be passed as a one-element family.
    """

    points = np.asarray(point_estimates, dtype=float)
    replicates = np.asarray(replicate_estimates, dtype=float)
    if points.ndim != 1 or not len(points) or not np.all(np.isfinite(points)):
        raise AuditContractError("point_estimates must be a nonempty finite vector")
    if replicates.ndim != 2 or replicates.shape[1] != len(points) or replicates.shape[0] < 20:
        raise AuditContractError("replicate_estimates must be B x M with B >= 20")
    if not (0.0 < float(alpha) < 1.0):
        raise AuditContractError("alpha must lie strictly between 0 and 1")
    if not (0.0 < float(min_complete_fraction) <= 1.0):
        raise AuditContractError("min_complete_fraction must lie in (0, 1]")
    coordinate_labels = tuple(labels) if labels is not None else tuple(f"contrast_{i}" for i in range(len(points)))
    if len(coordinate_labels) != len(points) or len(set(coordinate_labels)) != len(coordinate_labels):
        raise AuditContractError("labels must be unique and aligned to point_estimates")
    complete_mask = np.all(np.isfinite(replicates), axis=1)
    complete = replicates[complete_mask]
    complete_fraction = len(complete) / float(len(replicates))
    if len(complete) < 20 or complete_fraction < min_complete_fraction:
        raise AuditContractError("insufficient complete joint replicates for simultaneous inference")
    standard_errors = np.std(complete, axis=0, ddof=1)
    centered = complete - np.mean(complete, axis=0, keepdims=True)
    standardized = np.zeros_like(centered)
    positive = standard_errors > 0
    standardized[:, positive] = centered[:, positive] / standard_errors[positive]
    max_statistics = np.max(np.abs(standardized), axis=1)
    try:
        empirical_critical = float(np.quantile(max_statistics, 1.0 - alpha, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        empirical_critical = float(np.quantile(max_statistics, 1.0 - alpha, interpolation="higher"))
    degrees_of_freedom = len(complete) - 1
    ordinary_critical = float(stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom))
    simultaneous_critical = max(empirical_critical, ordinary_critical)
    rows = []
    for label, point, standard_error in zip(coordinate_labels, points, standard_errors):
        ordinary_half = ordinary_critical * standard_error
        simultaneous_half = simultaneous_critical * standard_error
        rows.append(
            {
                "label": str(label),
                "point_estimate": float(point),
                "standard_error": float(standard_error),
                "ordinary_ci": [float(point - ordinary_half), float(point + ordinary_half)],
                "simultaneous_ci": [float(point - simultaneous_half), float(point + simultaneous_half)],
            }
        )
    return {
        "status": "A2_CANDIDATE_NOT_PRODUCTION_FROZEN",
        "method": "EMPIRICAL_MAX_STANDARDIZED_JOINT_REPLICATES",
        "alpha": float(alpha),
        "family_size": len(points),
        "replicates_total": int(len(replicates)),
        "replicates_complete": int(len(complete)),
        "complete_fraction": float(complete_fraction),
        "ordinary_critical": ordinary_critical,
        "simultaneous_critical": simultaneous_critical,
        "rows": rows,
        "warning": "Validate coverage/reference behavior before production freeze; do not substitute Holm-adjusted p-values for these intervals.",
    }


def project_max_absolute_interval(components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Project a simultaneous component family to max absolute disparity."""

    if len(components) < 2:
        raise AuditContractError("max disparity projection requires the full predeclared component family")
    lower_candidates = []
    upper_candidates = []
    point_candidates = []
    for component in components:
        point = float(component["point_estimate"])
        lower, upper = (float(value) for value in component["simultaneous_ci"])
        if not all(math.isfinite(value) for value in (point, lower, upper)) or lower > upper:
            raise AuditContractError("component intervals must be finite and ordered")
        point_candidates.append(abs(point))
        lower_candidates.append(0.0 if lower <= 0.0 <= upper else min(abs(lower), abs(upper)))
        upper_candidates.append(max(abs(lower), abs(upper)))
    return {
        "point_estimate": float(max(point_candidates)),
        "simultaneous_ci": [float(max(lower_candidates)), float(max(upper_candidates))],
        "projection": "MAX_ABSOLUTE_OVER_PREDECLARED_COMPONENTS",
        "component_count": len(components),
        "posthoc_selected_pair_interval": False,
    }


def classify_tolerance_interval(
    ci_lower: float,
    ci_upper: float,
    delta: float,
    *,
    nonnegative_estimand: bool = False,
) -> str:
    """Return the preregistered three-state conclusion for one interval."""

    lower, upper, tolerance = float(ci_lower), float(ci_upper), float(delta)
    if not all(math.isfinite(value) for value in (lower, upper, tolerance)) or lower > upper or tolerance < 0:
        raise AuditContractError("interval and delta must be finite, ordered, and nonnegative where required")
    if nonnegative_estimand:
        if lower < 0:
            raise AuditContractError("a nonnegative estimand cannot have a negative projected lower bound")
        if upper <= tolerance:
            return CONCLUSION_WITHIN
        if lower > tolerance:
            return CONCLUSION_EXCEEDS
        return CONCLUSION_INSUFFICIENT
    if lower >= -tolerance and upper <= tolerance:
        return CONCLUSION_WITHIN
    if lower > tolerance or upper < -tolerance:
        return CONCLUSION_EXCEEDS
    return CONCLUSION_INSUFFICIENT


_SCOPE_STATE_PRIORITY = {
    CONCLUSION_WITHIN: 0,
    CONCLUSION_INSUFFICIENT: 1,
    CONCLUSION_EXCEEDS: 2,
}


_Q2_SCIENTIFIC_IDENTITY_FIELDS = (
    "year",
    "evidence_role",
    "family_name",
    "question",
    "panel_id",
    "panel_sha256",
    "O_train",
    "O_audit",
    "audit_role",
    "audit_role_sha256",
    "O_group_rule_sha256",
    "O_registry_version",
    "O_registry_sha256",
    "contrast_universe_sha256",
    "group",
    "reference_group",
    "endpoint",
    "seed_policy_sha256",
    "delta",
    "delta_units",
    "delta_rationale_id",
    "contrast_id",
)


_Q2_SHARED_SCOPE_FIELDS = (
    "year",
    "evidence_role",
    "family_name",
    "question",
    "panel_id",
    "panel_sha256",
    "O_train",
    "O_registry_version",
    "O_registry_sha256",
    "contrast_universe_sha256",
    "endpoint",
    "seed_policy_sha256",
    "delta",
    "delta_units",
    "delta_rationale_id",
)


_Q2_FAMILY_DEPENDENT_INFERENCE_FIELDS = {
    "conclusion_state",
    "family_id",
    "family_sha256",
    "family_scope",
    "simultaneous_ci",
    "simultaneous_ci_lower",
    "simultaneous_ci_upper",
    "simultaneous_critical",
    "holm_adjusted_p",
    "multiplicity_adjusted_p",
}


_NO_NEW_O_SCHEMA_VERSION = "nhis_bias_audit_no_new_O_receipt_v3"
_NO_NEW_O_STATUS = "FROZEN_NO_NEW_O_SELECTED"
_NO_NEW_O_PRECISION_INTERPRETATIONS = {
    "RULED_OUT_PREDECLARED_MATERIAL_EFFECTS",
    "INSUFFICIENT_PRECISION_TO_RULE_OUT_PREDECLARED_MATERIAL_EFFECTS",
    "MIXED_PRECISION_ACROSS_CANDIDATES",
}


def freeze_no_new_O_receipt(
    *,
    shared_scope_identity: Mapping[str, Any],
    atlas_source: Mapping[str, Any],
    candidate_registry_source: Mapping[str, Any],
    excluded_candidate_count: int,
    not_estimable_candidate_count: int,
    precision_interpretation: str,
    precision_evidence_source: Mapping[str, Any],
    selection_rule: Mapping[str, Any],
    O_registry: Mapping[str, Any],
    contrast_universe_sha256: str,
) -> dict[str, Any]:
    """Create the only receipt that permits an empty frozen new-O set."""

    receipt = {
        "schema_version": _NO_NEW_O_SCHEMA_VERSION,
        "status": _NO_NEW_O_STATUS,
        "shared_scope_identity": copy.deepcopy(dict(shared_scope_identity)),
        "atlas_source": copy.deepcopy(dict(atlas_source)),
        "candidate_registry_source": copy.deepcopy(dict(candidate_registry_source)),
        "excluded_candidate_count": excluded_candidate_count,
        "not_estimable_candidate_count": not_estimable_candidate_count,
        "selected_new_O_count": 0,
        "precision_interpretation": precision_interpretation,
        "precision_evidence_source": copy.deepcopy(dict(precision_evidence_source)),
        "selection_rule": copy.deepcopy(dict(selection_rule)),
        "O_registry": copy.deepcopy(dict(O_registry)),
        "contrast_universe_sha256": _require_sha256(
            contrast_universe_sha256,
            "contrast_universe_sha256",
        ),
        "candidate_universe_scope": "PROJECT_ATLAS_NOT_CODEBOOK_WIDE_ELIGIBILITY_LEDGER",
        "codebook_wide_eligibility_ledger_complete": False,
    }
    _verify_no_new_O_receipt({**receipt, "receipt_sha256": canonical_sha256(receipt)})
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _verify_no_new_O_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(receipt))
    claimed_hash = payload.pop("receipt_sha256", None)
    _require_sha256(claimed_hash, "no-new-O receipt_sha256")
    if canonical_sha256(payload) != claimed_hash:
        raise AuditContractError("no-new-O receipt hash mismatch")
    expected_fields = {
        "schema_version",
        "status",
        "shared_scope_identity",
        "atlas_source",
        "candidate_registry_source",
        "excluded_candidate_count",
        "not_estimable_candidate_count",
        "selected_new_O_count",
        "precision_interpretation",
        "precision_evidence_source",
        "selection_rule",
        "O_registry",
        "contrast_universe_sha256",
        "candidate_universe_scope",
        "codebook_wide_eligibility_ledger_complete",
    }
    if set(payload) != expected_fields:
        raise AuditContractError("no-new-O receipt fields do not match the registered schema")
    if payload["schema_version"] != _NO_NEW_O_SCHEMA_VERSION or payload["status"] != _NO_NEW_O_STATUS:
        raise AuditContractError("no-new-O receipt schema or status is invalid")
    shared_scope = _validate_q2_shared_scope_identity(
        payload["shared_scope_identity"],
        label="no-new-O shared_scope_identity",
    )
    if shared_scope != payload["shared_scope_identity"]:
        raise AuditContractError("no-new-O shared_scope_identity is not canonical")
    atlas_source = payload["atlas_source"]
    if not isinstance(atlas_source, Mapping) or set(atlas_source) != {"path", "sha256", "variable_count"}:
        raise AuditContractError("atlas_source must bind path, sha256, and variable_count")
    _require_nonempty_string(atlas_source["path"], "atlas_source path")
    _require_sha256(atlas_source["sha256"], "atlas_source sha256")
    candidate_source = payload["candidate_registry_source"]
    if not isinstance(candidate_source, Mapping) or set(candidate_source) != {
        "path",
        "sha256",
        "candidate_denominator",
    }:
        raise AuditContractError("candidate_registry_source must bind path, sha256, and denominator")
    _require_nonempty_string(candidate_source["path"], "candidate_registry_source path")
    _require_sha256(candidate_source["sha256"], "candidate_registry_source sha256")
    precision_source = payload["precision_evidence_source"]
    if not isinstance(precision_source, Mapping) or set(precision_source) != {"path", "sha256", "schema_version"}:
        raise AuditContractError("precision_evidence_source must bind path, sha256, and schema_version")
    _require_nonempty_string(precision_source["path"], "precision_evidence_source path")
    _require_sha256(precision_source["sha256"], "precision_evidence_source sha256")
    _require_nonempty_string(precision_source["schema_version"], "precision_evidence_source schema_version")
    selection_rule = payload["selection_rule"]
    if not isinstance(selection_rule, Mapping) or set(selection_rule) != {
        "selection_rule_id",
        "path",
        "schema_version",
        "sha256",
    }:
        raise AuditContractError("selection_rule must bind id, path, schema_version, and sha256")
    for field in ("selection_rule_id", "path", "schema_version"):
        _require_nonempty_string(selection_rule[field], f"selection_rule {field}")
    _require_sha256(selection_rule["sha256"], "selection_rule sha256")
    o_registry = payload["O_registry"]
    if not isinstance(o_registry, Mapping) or set(o_registry) != {"version", "path", "sha256"}:
        raise AuditContractError("O_registry must bind version, path, and sha256")
    for field in ("version", "path"):
        _require_nonempty_string(o_registry[field], f"O_registry {field}")
    _require_sha256(o_registry["sha256"], "O_registry sha256")
    _require_sha256(payload["contrast_universe_sha256"], "contrast_universe_sha256")
    count_fields = (
        "excluded_candidate_count",
        "not_estimable_candidate_count",
        "selected_new_O_count",
    )
    source_counts = (
        atlas_source["variable_count"],
        candidate_source["candidate_denominator"],
    )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in source_counts):
        raise AuditContractError("no-new-O source denominators must be integers")
    if any(not isinstance(payload[field], int) or isinstance(payload[field], bool) for field in count_fields):
        raise AuditContractError("no-new-O receipt counts must be integers")
    if atlas_source["variable_count"] <= 0 or candidate_source["candidate_denominator"] < 0:
        raise AuditContractError("no-new-O receipt denominators are invalid")
    if candidate_source["candidate_denominator"] > atlas_source["variable_count"]:
        raise AuditContractError("candidate denominator cannot exceed the atlas denominator")
    if payload["excluded_candidate_count"] < 0 or payload["not_estimable_candidate_count"] < 0:
        raise AuditContractError("no-new-O receipt counts must be nonnegative")
    if payload["selected_new_O_count"] != 0:
        raise AuditContractError("an empty new-O path requires selected_new_O_count=0")
    accounted = (
        payload["excluded_candidate_count"]
        + payload["not_estimable_candidate_count"]
        + payload["selected_new_O_count"]
    )
    if accounted != candidate_source["candidate_denominator"]:
        raise AuditContractError("no-new-O receipt candidate counts do not reconcile")
    if payload["precision_interpretation"] not in _NO_NEW_O_PRECISION_INTERPRETATIONS:
        raise AuditContractError("no-new-O precision interpretation is not registered")
    if payload["candidate_universe_scope"] != "PROJECT_ATLAS_NOT_CODEBOOK_WIDE_ELIGIBILITY_LEDGER":
        raise AuditContractError("no-new-O receipt must disclose its non-codebook-wide candidate scope")
    if payload["codebook_wide_eligibility_ledger_complete"] is not False:
        raise AuditContractError("no-new-O receipt cannot claim a complete codebook-wide eligibility ledger")
    return {**payload, "receipt_sha256": claimed_hash}


def _q2_scientific_identity(row: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    missing = [field for field in _Q2_SCIENTIFIC_IDENTITY_FIELDS if field not in row]
    if missing:
        raise AuditContractError(f"{label} is missing scientific identity fields: {missing}")
    identity = {field: copy.deepcopy(row[field]) for field in _Q2_SCIENTIFIC_IDENTITY_FIELDS}
    if not isinstance(identity["year"], int) or isinstance(identity["year"], bool) or identity["year"] < 2000:
        raise AuditContractError(f"{label} year must be an explicit survey year")
    if identity["evidence_role"] not in {
        "training_calibration_history",
        "discovery",
        "known_retrospective_replication",
        "locked_one_shot_validation",
    }:
        raise AuditContractError(f"{label} evidence_role is not registered")
    for field in (
        "family_name",
        "question",
        "panel_id",
        "O_train",
        "O_audit",
        "audit_role",
        "O_registry_version",
        "endpoint",
        "delta_units",
        "delta_rationale_id",
        "contrast_id",
    ):
        _require_nonempty_string(identity[field], f"{label} {field}")
    expected_question = {
        "q1_confirmatory": "Q1_FIXED_MODEL_GROUP_DIFFERENCE",
        "q3_q4_confirmatory": "Q3_Q4_CONFIRMATORY_METHOD_CHANGE",
        "q3_q4_descriptive": "Q3_Q4_DESCRIPTIVE_METHOD_CHANGE",
    }.get(identity["family_name"])
    if identity["question"] != expected_question:
        raise AuditContractError(f"{label} family_name/question identity is invalid")
    if identity["audit_role"] not in {"anchor", "expanded_candidate"}:
        raise AuditContractError(f"{label} audit_role is invalid")
    _require_sha256(identity["panel_sha256"], "panel_sha256")
    _require_sha256(identity["seed_policy_sha256"], "seed_policy_sha256")
    _require_sha256(identity["O_group_rule_sha256"], "O_group_rule_sha256")
    _require_sha256(identity["audit_role_sha256"], "audit_role_sha256")
    _require_sha256(identity["O_registry_sha256"], "O_registry_sha256")
    _require_sha256(identity["contrast_universe_sha256"], "contrast_universe_sha256")
    if identity["audit_role_sha256"] != audit_role_sha256(
        identity["O_audit"],
        identity["audit_role"],
        identity["O_registry_version"],
        identity["O_registry_sha256"],
    ):
        raise AuditContractError(f"{label} audit_role_sha256 mismatch")
    delta = identity["delta"]
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(float(delta)) or float(delta) < 0:
        raise AuditContractError(f"{label} delta must be finite and nonnegative")
    if identity["group"] == identity["reference_group"]:
        raise AuditContractError(f"{label} group must differ from reference_group")
    model_id = row.get("model_id")
    method_id = row.get("method_id")
    baseline_id = row.get("matched_baseline_id")
    has_model = isinstance(model_id, str) and bool(model_id.strip())
    has_pair = (
        isinstance(method_id, str)
        and bool(method_id.strip())
        and isinstance(baseline_id, str)
        and bool(baseline_id.strip())
        and method_id != baseline_id
    )
    if has_model == has_pair:
        raise AuditContractError(f"{label} must bind exactly one model_id or method-baseline pair")
    if has_model:
        if method_id is not None or baseline_id is not None:
            raise AuditContractError(f"{label} model contrast cannot also contain method fields")
        identity["model_id"] = model_id
    else:
        if model_id is not None:
            raise AuditContractError(f"{label} method contrast cannot also contain model_id")
        identity["method_id"] = method_id
        identity["matched_baseline_id"] = baseline_id
    return identity


def _validate_q2_shared_scope_identity(identity: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise AuditContractError(f"{label} must be a mapping")
    value = copy.deepcopy(dict(identity))
    base_fields = set(_Q2_SHARED_SCOPE_FIELDS)
    model_fields = {"model_id"}
    method_fields = {"method_id", "matched_baseline_id"}
    if set(value) == base_fields | model_fields:
        _require_nonempty_string(value["model_id"], f"{label} model_id")
    elif set(value) == base_fields | method_fields:
        method = _require_nonempty_string(value["method_id"], f"{label} method_id")
        baseline = _require_nonempty_string(
            value["matched_baseline_id"],
            f"{label} matched_baseline_id",
        )
        if method == baseline:
            raise AuditContractError(f"{label} method-baseline pair cannot be self-comparing")
    else:
        raise AuditContractError(f"{label} fields do not match a model or method shared-scope schema")
    if not isinstance(value["year"], int) or isinstance(value["year"], bool) or value["year"] < 2000:
        raise AuditContractError(f"{label} year is invalid")
    if value["evidence_role"] not in {
        "training_calibration_history",
        "discovery",
        "known_retrospective_replication",
        "locked_one_shot_validation",
    }:
        raise AuditContractError(f"{label} evidence_role is invalid")
    for field in (
        "family_name",
        "question",
        "panel_id",
        "O_train",
        "O_registry_version",
        "endpoint",
        "delta_units",
        "delta_rationale_id",
    ):
        _require_nonempty_string(value[field], f"{label} {field}")
    expected_question = {
        "q1_confirmatory": "Q1_FIXED_MODEL_GROUP_DIFFERENCE",
        "q3_q4_confirmatory": "Q3_Q4_CONFIRMATORY_METHOD_CHANGE",
        "q3_q4_descriptive": "Q3_Q4_DESCRIPTIVE_METHOD_CHANGE",
    }.get(value["family_name"])
    if value["question"] != expected_question:
        raise AuditContractError(f"{label} family_name/question identity is invalid")
    _require_sha256(value["panel_sha256"], f"{label} panel_sha256")
    _require_sha256(value["seed_policy_sha256"], f"{label} seed_policy_sha256")
    _require_sha256(value["O_registry_sha256"], f"{label} O_registry_sha256")
    _require_sha256(value["contrast_universe_sha256"], f"{label} contrast_universe_sha256")
    delta = value["delta"]
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(float(delta)) or delta < 0:
        raise AuditContractError(f"{label} delta must be finite and nonnegative")
    return value


def _q2_shared_scope_identity(row: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    scientific = _q2_scientific_identity(row, label=label)
    shared = {field: copy.deepcopy(scientific[field]) for field in _Q2_SHARED_SCOPE_FIELDS}
    if "model_id" in scientific:
        shared["model_id"] = scientific["model_id"]
    else:
        shared["method_id"] = scientific["method_id"]
        shared["matched_baseline_id"] = scientific["matched_baseline_id"]
    return _validate_q2_shared_scope_identity(shared, label=label)


def _validated_scope_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    expected_layer: str,
    allow_empty: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not rows:
        if allow_empty:
            return {}, {}
        raise AuditContractError(f"{label} must contain at least one predeclared contrast slot")
    by_key: dict[str, dict[str, Any]] = {}
    coverage: dict[str, dict[str, Any]] = {}
    expected_role = "anchor" if expected_layer == "anchor" else "expanded_candidate"
    for raw_row in rows:
        row = dict(raw_row)
        key = row.get("contrast_key")
        attribute = row.get("O_audit")
        state = row.get("conclusion_state")
        if not isinstance(key, str) or not key or key in by_key:
            raise AuditContractError(f"{label} contrast_key values must be nonempty and unique")
        _q2_scientific_identity(row, label=label)
        if key != row["contrast_id"]:
            raise AuditContractError(f"{label} contrast_key must equal the predeclared contrast_id")
        if not isinstance(attribute, str) or not attribute:
            raise AuditContractError(f"{label} requires O_audit on every row")
        if row.get("audit_layer") != expected_layer:
            raise AuditContractError(f"{label} contains a row from the wrong audit layer")
        if row.get("audit_role") != expected_role:
            raise AuditContractError(f"{label} audit_role contradicts its audit layer")
        if state not in _SCOPE_STATE_PRIORITY:
            raise AuditContractError(f"{label} contains an invalid conclusion_state")
        estimation_status = row.get("estimation_status")
        if estimation_status != "VALID" and state != CONCLUSION_INSUFFICIENT:
            raise AuditContractError("non-valid estimates must remain INSUFFICIENT_EVIDENCE")
        annual_rows = row.get("annual_rows")
        domain_rows = row.get("domain_rows")
        if not isinstance(annual_rows, int) or not isinstance(domain_rows, int):
            raise AuditContractError(f"{label} requires integer annual_rows and domain_rows")
        if annual_rows <= 0 or domain_rows < 0 or domain_rows > annual_rows:
            raise AuditContractError(f"{label} contains invalid per-O coverage counts")
        current = coverage.setdefault(
            attribute,
            {
                "O_audit": attribute,
                "O_group_rule_sha256": row["O_group_rule_sha256"],
                "audit_role": row["audit_role"],
                "audit_role_sha256": row["audit_role_sha256"],
                "audit_layer": expected_layer,
                "annual_rows": annual_rows,
                "domain_rows": domain_rows,
                "contrast_count": 0,
                "not_estimable_contrasts": 0,
                "conclusion_counts": {
                    CONCLUSION_WITHIN: 0,
                    CONCLUSION_EXCEEDS: 0,
                    CONCLUSION_INSUFFICIENT: 0,
                },
            },
        )
        if (
            current["annual_rows"] != annual_rows
            or current["domain_rows"] != domain_rows
            or current["O_group_rule_sha256"] != row["O_group_rule_sha256"]
            or current["audit_role_sha256"] != row["audit_role_sha256"]
        ):
            raise AuditContractError(f"{label} has inconsistent domain or group-rule identity within one O")
        current["contrast_count"] += 1
        current["not_estimable_contrasts"] += int(estimation_status != "VALID")
        current["conclusion_counts"][state] += 1
        by_key[key] = row
    for summary in coverage.values():
        summary["coverage_fraction"] = summary["domain_rows"] / float(summary["annual_rows"])
    return by_key, coverage


def _expected_q2_entries(
    universe: Mapping[str, Any],
    shared_scope: Mapping[str, Any],
    *,
    audit_role: str,
) -> dict[str, dict[str, Any]]:
    family_name = shared_scope["family_name"]
    entries = universe["expected_contrasts"][family_name]
    selected = []
    for row in entries:
        if row["audit_role"] != audit_role or row["endpoint"] != shared_scope["endpoint"]:
            continue
        if "model_id" in shared_scope:
            if row.get("model_id") != shared_scope["model_id"]:
                continue
        elif (
            row.get("method_id") != shared_scope["method_id"]
            or row.get("matched_baseline_id") != shared_scope["matched_baseline_id"]
        ):
            continue
        selected.append(row)
    return {row["contrast_id"]: row for row in selected}


def _verify_q2_rows_equal_universe(
    rows: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> None:
    if set(rows) != set(expected):
        raise AuditContractError(f"{label} omit or add frozen contrast-universe slots")
    for contrast_id, row in rows.items():
        expected_row = expected[contrast_id]
        observed_projection = {field: row.get(field) for field in expected_row}
        if observed_projection != expected_row:
            raise AuditContractError(f"{label} row identity contradicts the frozen contrast universe")


def _aggregate_scope_state(rows: Mapping[str, Mapping[str, Any]]) -> str:
    return max(
        (str(row["conclusion_state"]) for row in rows.values()),
        key=lambda state: _SCOPE_STATE_PRIORITY[state],
    )


def _state_change(before: str, after: str) -> dict[str, Any]:
    return {
        "from": before,
        "to": after,
        "changed": before != after,
        "priority_change": _SCOPE_STATE_PRIORITY[after] - _SCOPE_STATE_PRIORITY[before],
    }


def aggregate_q2_scope_conclusions(
    anchor_native_rows: Sequence[Mapping[str, Any]],
    anchor_under_expanded_family_rows: Sequence[Mapping[str, Any]],
    new_O_under_expanded_family_rows: Sequence[Mapping[str, Any]],
    *,
    contrast_universe_receipt: Mapping[str, Any],
    no_new_O_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate Q2 while separating content from multiplicity/precision effects.

    The fail-closed priority is EXCEEDS > INSUFFICIENT > WITHIN.  Anchor rows
    are evaluated both in their native family and under the expanded family.
    New-O content is then added only to the latter so the two mechanisms are
    reported separately.  Per-O coverage is never collapsed to a global
    all-attribute complete-case domain.
    """

    verified_universe = verify_contrast_universe_receipt(contrast_universe_receipt)
    anchor_native, native_coverage = _validated_scope_rows(
        anchor_native_rows, label="anchor_native_rows", expected_layer="anchor"
    )
    anchor_expanded, expanded_anchor_coverage = _validated_scope_rows(
        anchor_under_expanded_family_rows,
        label="anchor_under_expanded_family_rows",
        expected_layer="anchor",
    )
    new_o, new_o_coverage = _validated_scope_rows(
        new_O_under_expanded_family_rows,
        label="new_O_under_expanded_family_rows",
        expected_layer="new_O",
        allow_empty=True,
    )
    if set(anchor_native) != set(anchor_expanded):
        raise AuditContractError("anchor native and expanded-family contrast slots must match exactly")
    q2_shared_scope: dict[str, Any] | None = None
    for key in anchor_native:
        native = anchor_native[key]
        expanded = anchor_expanded[key]
        native_identity = _q2_scientific_identity(native, label="anchor_native_rows")
        expanded_identity = _q2_scientific_identity(expanded, label="anchor_under_expanded_family_rows")
        if native_identity != expanded_identity:
            raise AuditContractError("family expansion must not change the scientific contrast identity")
        if any(native[field] != expanded[field] for field in ("annual_rows", "domain_rows", "estimation_status")):
            raise AuditContractError("family expansion must not change anchor contrast identity or domain")
        stable_fields = (set(native) | set(expanded)) - _Q2_FAMILY_DEPENDENT_INFERENCE_FIELDS
        if any(native.get(field) != expanded.get(field) for field in stable_fields):
            raise AuditContractError("only registered family-dependent inference fields may change")
        current_shared = _q2_shared_scope_identity(
            expanded,
            label="anchor_under_expanded_family_rows",
        )
        if q2_shared_scope is None:
            q2_shared_scope = current_shared
        elif current_shared != q2_shared_scope:
            raise AuditContractError("all anchor rows in one Q2 aggregation must share one scope identity")
    if q2_shared_scope is None:
        raise AuditContractError("Q2 aggregation requires at least one anchor contrast")
    if q2_shared_scope["contrast_universe_sha256"] != verified_universe["contrast_universe_sha256"]:
        raise AuditContractError("Q2 rows do not bind the supplied contrast universe receipt")
    universe_identity = verified_universe["panel_family_identity"]
    for field in (
        "year",
        "evidence_role",
        "panel_id",
        "panel_sha256",
        "O_train",
        "seed_policy_sha256",
        "delta_rationale_id",
    ):
        if q2_shared_scope[field] != universe_identity[field]:
            raise AuditContractError("Q2 shared scope contradicts the frozen contrast universe")
    if q2_shared_scope["O_registry_version"] != verified_universe["O_registry"]["version"]:
        raise AuditContractError("Q2 O registry version contradicts the frozen contrast universe")
    if q2_shared_scope["O_registry_sha256"] != verified_universe["O_registry"]["sha256"]:
        raise AuditContractError("Q2 O registry hash contradicts the frozen contrast universe")
    if q2_shared_scope["endpoint"] not in universe_identity["endpoints"]:
        raise AuditContractError("Q2 endpoint is outside the frozen contrast universe")
    if "model_id" in q2_shared_scope:
        if q2_shared_scope["family_name"] != "q1_confirmatory":
            raise AuditContractError("Q2 model contrast must use the frozen Q1 family")
        if q2_shared_scope["model_id"] not in universe_identity["q1_model_ids"]:
            raise AuditContractError("Q2 model is outside the frozen contrast universe")
    else:
        pair = [q2_shared_scope["method_id"], q2_shared_scope["matched_baseline_id"]]
        pair_field = (
            "confirmatory_method_pairs"
            if q2_shared_scope["family_name"] == "q3_q4_confirmatory"
            else "descriptive_method_pairs"
        )
        if pair not in universe_identity[pair_field]:
            raise AuditContractError("Q2 method-baseline pair is outside the frozen contrast universe")
    expected_anchor = _expected_q2_entries(
        verified_universe,
        q2_shared_scope,
        audit_role="anchor",
    )
    expected_new_o = _expected_q2_entries(
        verified_universe,
        q2_shared_scope,
        audit_role="expanded_candidate",
    )
    _verify_q2_rows_equal_universe(
        anchor_native,
        expected_anchor,
        label="anchor_native_rows",
    )
    _verify_q2_rows_equal_universe(
        anchor_expanded,
        expected_anchor,
        label="anchor_under_expanded_family_rows",
    )
    _verify_q2_rows_equal_universe(
        new_o,
        expected_new_o,
        label="new_O_under_expanded_family_rows",
    )
    if set(anchor_expanded) & set(new_o):
        raise AuditContractError("anchor and new-O contrast keys must be disjoint")
    for row in new_o.values():
        if _q2_shared_scope_identity(row, label="new_O_under_expanded_family_rows") != q2_shared_scope:
            raise AuditContractError("new-O contrast does not match the anchor shared-scope identity")

    if new_o and no_new_O_receipt is not None:
        raise AuditContractError("no-new-O receipt cannot accompany nonempty new-O contrasts")
    verified_no_new_o = None
    if not new_o:
        if no_new_O_receipt is None:
            raise AuditContractError("empty new-O contrasts require a hash-bound no-new-O receipt")
        verified_no_new_o = _verify_no_new_O_receipt(no_new_O_receipt)
        if verified_no_new_o["shared_scope_identity"] != q2_shared_scope:
            raise AuditContractError("no-new-O receipt does not match the anchor shared-scope identity")
        if verified_universe["new_O_set"]:
            raise AuditContractError("empty new-O rows require a frozen empty new-O universe")
        if verified_no_new_o["contrast_universe_sha256"] != verified_universe["contrast_universe_sha256"]:
            raise AuditContractError("no-new-O receipt does not bind the supplied contrast universe")
        if (
            verified_no_new_o["O_registry"]["version"] != verified_universe["O_registry"]["version"]
            or verified_no_new_o["O_registry"]["sha256"] != verified_universe["O_registry"]["sha256"]
        ):
            raise AuditContractError("no-new-O receipt O registry contradicts the contrast universe")
        if (
            verified_no_new_o["selection_rule"]["selection_rule_id"]
            != verified_universe["selection_rule"]["selection_rule_id"]
            or verified_no_new_o["selection_rule"]["sha256"]
            != verified_universe["selection_rule"]["sha256"]
        ):
            raise AuditContractError("no-new-O receipt selection rule contradicts the contrast universe")

    anchor_native_state = _aggregate_scope_state(anchor_native)
    anchor_expanded_state = _aggregate_scope_state(anchor_expanded)
    new_o_state = _aggregate_scope_state(new_o) if new_o else "NOT_APPLICABLE_NO_NEW_O"
    all_expanded = {**anchor_expanded, **new_o}
    expanded_state = _aggregate_scope_state(all_expanded)
    limiting_priority = _SCOPE_STATE_PRIORITY[expanded_state]
    new_limiting_o = sorted(
        {
            str(row["O_audit"])
            for row in new_o.values()
            if _SCOPE_STATE_PRIORITY[str(row["conclusion_state"])] == limiting_priority
        }
    )
    coverage_rows = [
        *sorted(expanded_anchor_coverage.values(), key=lambda item: item["O_audit"]),
        *sorted(new_o_coverage.values(), key=lambda item: item["O_audit"]),
    ]
    return {
        "schema_version": "nhis_bias_audit_q2_scope_aggregator_v3",
        "scope_priority_high_to_low": [
            CONCLUSION_EXCEEDS,
            CONCLUSION_INSUFFICIENT,
            CONCLUSION_WITHIN,
        ],
        "anchor_native_conclusion": anchor_native_state,
        "anchor_under_expanded_family_conclusion": anchor_expanded_state,
        "new_O_content_conclusion": new_o_state,
        "expanded_conclusion": expanded_state,
        "multiplicity_precision_change": _state_change(anchor_native_state, anchor_expanded_state),
        "new_O_content_change": (
            _state_change(anchor_expanded_state, expanded_state)
            if new_o
            else {
                "status": "NOT_APPLICABLE_NO_NEW_O",
                "from": anchor_expanded_state,
                "to": anchor_expanded_state,
                "changed": False,
                "priority_change": 0,
                "affirmative_new_O_change_allowed": False,
            }
        ),
        "overall_anchor_to_expanded_change": _state_change(anchor_native_state, expanded_state),
        "new_limiting_O": new_limiting_o,
        "per_O_coverage": coverage_rows,
        "coverage_is_per_O_not_global_complete_case": True,
        "native_anchor_coverage_verified_equal": True,
        "shared_scope_identity": q2_shared_scope,
        "contrast_universe_sha256": verified_universe["contrast_universe_sha256"],
        "expanded_equals_anchor_when_no_new_O": not bool(new_o),
        "no_new_O_receipt": verified_no_new_o,
    }


def precision_diagnostics(
    point_estimate: float,
    ci_lower: float,
    ci_upper: float,
    delta: float,
    *,
    standard_error: float | None = None,
    critical_value: float | None = None,
) -> dict[str, Any]:
    """Report interval precision without mislabeling margin of error as MDE."""

    point, lower, upper, tolerance = map(float, (point_estimate, ci_lower, ci_upper, delta))
    if (
        not all(math.isfinite(value) for value in (point, lower, upper, tolerance))
        or lower > upper
        or not lower <= point <= upper
        or tolerance < 0
    ):
        raise AuditContractError("invalid precision inputs")
    half_width = (upper - lower) / 2.0
    margin_from_point = max(point - lower, upper - point)
    supplied_critical_margin = None
    if standard_error is not None or critical_value is not None:
        if standard_error is None or critical_value is None:
            raise AuditContractError("standard_error and critical_value must be supplied together")
        se, critical = float(standard_error), float(critical_value)
        if not math.isfinite(se) or not math.isfinite(critical) or se < 0 or critical <= 0:
            raise AuditContractError("standard_error and critical_value must be finite and valid")
        supplied_critical_margin = critical * se
    largest_absolute = float(max(abs(lower), abs(upper)))
    smallest_absolute = 0.0 if lower <= 0.0 <= upper else float(min(abs(lower), abs(upper)))
    rules_out_material_effect = largest_absolute <= tolerance
    rules_out_within_tolerance = lower > tolerance or upper < -tolerance
    if rules_out_material_effect:
        delta_statement = "INTERVAL_RULES_OUT_EFFECTS_WITH_ABSOLUTE_MAGNITUDE_ABOVE_DELTA"
    elif rules_out_within_tolerance:
        delta_statement = "INTERVAL_RULES_OUT_THE_WITHIN_TOLERANCE_REGION"
    else:
        delta_statement = "INTERVAL_CROSSES_A_TOLERANCE_BOUNDARY"
    return {
        "ci_half_width": float(half_width),
        "simultaneous_margin_of_error_from_point": float(margin_from_point),
        "supplied_critical_margin_of_error": (
            None if supplied_critical_margin is None else float(supplied_critical_margin)
        ),
        "minimum_detectable_effect": None,
        "minimum_detectable_effect_status": "NOT_COMPUTED_REQUIRES_ALPHA_POWER_ALTERNATIVE_AND_DESIGN",
        "compatible_effect_interval": [lower, upper],
        "smallest_absolute_effect_compatible_with_interval": smallest_absolute,
        "largest_absolute_effect_compatible_with_interval": largest_absolute,
        "rules_out_effects_with_absolute_magnitude_above_delta": bool(rules_out_material_effect),
        "rules_out_within_tolerance_region": bool(rules_out_within_tolerance),
        "delta_exclusion_statement": delta_statement,
        "tolerance_conclusion_state": classify_tolerance_interval(lower, upper, tolerance),
        "support_thresholds_are_precision_claim": False,
    }


def classify_method_tradeoff(
    *,
    gap_change: float,
    disadvantaged_group_performance_change: float,
    advantaged_group_performance_change: float,
    overall_ba_change: float,
    overall_risk_quality_change: float | None,
    any_other_O_worsened: bool,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Classify gap changes without hiding absolute performance trade-offs."""

    numeric = [gap_change, disadvantaged_group_performance_change, advantaged_group_performance_change, overall_ba_change, tolerance]
    if overall_risk_quality_change is not None:
        numeric.append(overall_risk_quality_change)
    values = [float(value) for value in numeric]
    if not all(math.isfinite(value) for value in values) or float(tolerance) < 0:
        raise AuditContractError("trade-off inputs must be finite and tolerance nonnegative")
    tol = float(tolerance)
    gap_narrowed = float(gap_change) < -tol
    disadvantaged_improved = float(disadvantaged_group_performance_change) > tol
    disadvantaged_worse = float(disadvantaged_group_performance_change) < -tol
    advantaged_worse = float(advantaged_group_performance_change) < -tol
    if disadvantaged_worse and advantaged_worse:
        category = "ALL_GROUPS_WORSE"
    elif gap_narrowed and disadvantaged_improved and bool(any_other_O_worsened):
        category = "TARGET_O_IMPROVED_OTHER_O_WORSENED"
    elif gap_narrowed and disadvantaged_improved:
        category = "GAP_NARROWED_DISADVANTAGED_GROUP_IMPROVED"
    elif gap_narrowed and advantaged_worse and not disadvantaged_improved:
        category = "GAP_NARROWED_BY_ADVANTAGED_GROUP_LOSS"
    else:
        category = "NO_PREDECLARED_IMPROVEMENT_PATTERN"
    return {
        "category": category,
        "gap_change": float(gap_change),
        "disadvantaged_group_performance_change": float(disadvantaged_group_performance_change),
        "advantaged_group_performance_change": float(advantaged_group_performance_change),
        "overall_ba_change": float(overall_ba_change),
        "overall_risk_quality_change": None if overall_risk_quality_change is None else float(overall_risk_quality_change),
        "any_other_O_worsened": bool(any_other_O_worsened),
    }


def _panel_payload(panel: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(panel))
    payload.pop("panel_sha256", None)
    return payload


def _validate_complete_panel_member(
    member: Mapping[str, Any],
    *,
    panel_O_train: str,
    panel_seed_policy: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
) -> None:
    if not isinstance(member, Mapping):
        raise AuditContractError("panel member must be a mapping")
    required = {
        "member_id",
        "model_ids",
        "source_binding",
        "O_train",
        "seed_policy_sha256",
        "seed_model_bindings",
        "feature_sha256",
        "preprocessing_sha256",
        "training_source_sha256",
        "prediction_source_code_sha256",
        "backbone",
        "output_type",
        "threshold_policy",
        "O_at_inference",
    }
    missing = required - set(member)
    if missing:
        raise AuditContractError(f"panel member contract is missing fields: {sorted(missing)}")
    if member.get("member_contract_complete") is not True:
        raise AuditContractError("panel member contract must be explicitly complete before freeze")
    _require_nonempty_string(member["member_id"], "panel member_id")
    _require_nonempty_string(member["O_train"], "panel member O_train")
    if member["O_train"] != panel_O_train:
        raise AuditContractError("panel member O_train differs from the panel contract")
    source_binding = member["source_binding"]
    if not isinstance(source_binding, Mapping):
        raise AuditContractError("source_binding must be a structured path+sha256 object")
    if set(source_binding) - {"path", "sha256", "json_pointer"} or not {"path", "sha256"}.issubset(source_binding):
        raise AuditContractError("source_binding must contain path and sha256 with optional json_pointer")
    _require_nonempty_string(source_binding["path"], "source_binding path")
    _require_sha256(source_binding["sha256"], "source_binding sha256")
    if "json_pointer" in source_binding and (
        not isinstance(source_binding["json_pointer"], str)
        or not source_binding["json_pointer"].startswith("/")
    ):
        raise AuditContractError("source_binding json_pointer must be an absolute JSON pointer")
    expected_seed_hash = canonical_sha256(panel_seed_policy)
    if member["seed_policy_sha256"] != expected_seed_hash:
        raise AuditContractError("panel member seed policy hash mismatch")
    bindings = member["seed_model_bindings"]
    if not isinstance(bindings, list) or not bindings:
        raise AuditContractError("panel member requires seed-to-model bindings")
    if any(not isinstance(binding, Mapping) for binding in bindings):
        raise AuditContractError("seed bindings must be mappings")
    expected_seeds = panel_seed_policy["seeds"]
    observed_seeds = [binding.get("seed") for binding in bindings]
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in observed_seeds):
        raise AuditContractError("seed binding values must be non-boolean integers")
    if observed_seeds != expected_seeds or len(observed_seeds) != len(set(observed_seeds)):
        raise AuditContractError("panel member seed ordering must match the frozen seed policy")
    model_ids = []
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise AuditContractError("seed binding must be a mapping")
        model_id = _require_nonempty_string(binding.get("model_id"), "seed binding model_id")
        _require_sha256(binding.get("model_sha256"), "model_sha256")
        model_ids.append(model_id)
    if not isinstance(member["model_ids"], list) or not member["model_ids"]:
        raise AuditContractError("panel member model_ids must be a nonempty list")
    for model_id in member["model_ids"]:
        _require_nonempty_string(model_id, "panel member model_id")
    if model_ids != member["model_ids"]:
        raise AuditContractError("model_ids must match ordered seed-to-model bindings")
    if len(model_ids) != len(set(model_ids)):
        raise AuditContractError("model_ids must be unique within a panel member")
    for field in (
        "feature_sha256",
        "preprocessing_sha256",
        "training_source_sha256",
        "prediction_source_code_sha256",
    ):
        _require_sha256(member[field], field)
    _require_nonempty_string(member["backbone"], "panel member backbone")
    if member["backbone"] != deployment_contract["backbone"]:
        raise AuditContractError("panel member backbone differs from deployment_contract")
    if member["feature_sha256"] != deployment_contract["feature_sha256"]:
        raise AuditContractError("panel member feature contract differs from deployment_contract")
    if member["preprocessing_sha256"] != deployment_contract["preprocessing_sha256"]:
        raise AuditContractError("panel member preprocessing contract differs from deployment_contract")
    if member["output_type"] != deployment_contract["output_type"]:
        raise AuditContractError("panel member output_type differs from deployment_contract")
    if member["O_at_inference"] is not deployment_contract["O_at_inference"]:
        raise AuditContractError("panel member O_at_inference differs from deployment_contract")
    _, threshold_hash = _verify_threshold_policy(member["threshold_policy"])
    if member["threshold_policy"].get("threshold_policy_sha256") != threshold_hash:
        raise AuditContractError("threshold policy binding is invalid")
    if member["threshold_policy"].get("comparison_operator") != deployment_contract["threshold_comparison_operator"]:
        raise AuditContractError("member threshold operator differs from deployment_contract")


def _validate_complete_panel_contract(panel: Mapping[str, Any]) -> None:
    required = {
        "panel_id",
        "status",
        "baseline_member_id",
        "members",
        "deployment_contract",
        "common_domain_contract",
        "O_train",
        "seed_policy",
        "model_id_uniqueness_policy",
    }
    missing = required - set(panel)
    if missing:
        raise AuditContractError(f"comparison panel is missing required fields: {sorted(missing)}")
    if panel["status"] != "FROZEN":
        raise AuditContractError("comparison panel status must be FROZEN")
    _require_nonempty_string(panel["panel_id"], "comparison panel_id")
    _require_nonempty_string(panel["O_train"], "comparison panel O_train")
    _require_nonempty_string(panel["baseline_member_id"], "baseline_member_id")
    deployment = panel["deployment_contract"]
    required_deployment = {
        "backbone",
        "feature_sha256",
        "preprocessing_sha256",
        "output_type",
        "O_at_inference",
        "threshold_policy_mode",
        "threshold_comparison_operator",
    }
    if not isinstance(deployment, Mapping) or not required_deployment.issubset(deployment):
        raise AuditContractError("deployment_contract is incomplete")
    _require_nonempty_string(deployment["backbone"], "deployment_contract backbone")
    _require_sha256(deployment["feature_sha256"], "deployment feature_sha256")
    _require_sha256(deployment["preprocessing_sha256"], "deployment preprocessing_sha256")
    if deployment["output_type"] != "event_probability_p":
        raise AuditContractError("main risk deployment requires event_probability_p")
    if deployment["O_at_inference"] is not False:
        raise AuditContractError("main risk deployment requires O_at_inference=false")
    if deployment["threshold_policy_mode"] != "REUSE_EACH_MEMBER_FROZEN_POLICY_NO_RETUNING":
        raise AuditContractError("deployment threshold policy mode is not registered")
    if deployment["threshold_comparison_operator"] != ">=":
        raise AuditContractError("deployment threshold comparison operator must be >=")
    common_domain = panel["common_domain_contract"]
    required_domain = {
        "scope",
        "all_panel_members_required",
        "annual_design_rows_retained",
        "global_all_O_complete_case_prohibited",
    }
    if not isinstance(common_domain, Mapping) or set(common_domain) != required_domain:
        raise AuditContractError("common_domain_contract fields do not match the registered schema")
    if common_domain["scope"] != "per_O":
        raise AuditContractError("main panel common domain must be per_O")
    if common_domain["all_panel_members_required"] is not True:
        raise AuditContractError("main panel common domain must require every member")
    if common_domain["annual_design_rows_retained"] is not True:
        raise AuditContractError("main panel common domain must retain annual design rows")
    if common_domain["global_all_O_complete_case_prohibited"] is not True:
        raise AuditContractError("main panel must prohibit global all-O complete-case domains")
    members = panel["members"]
    if not isinstance(members, list) or len(members) < 2:
        raise AuditContractError("comparison panel must contain at least two members")
    member_ids = [
        _require_nonempty_string(member.get("member_id"), "panel member_id")
        for member in members
        if isinstance(member, Mapping)
    ]
    if len(member_ids) != len(members) or len(member_ids) != len(set(member_ids)):
        raise AuditContractError("panel member identifiers must be nonempty and unique")
    if panel["baseline_member_id"] not in member_ids:
        raise AuditContractError("baseline_member_id must identify a panel member")
    seed_policy = _validate_seed_policy(panel["seed_policy"], label="panel seed_policy")
    if panel["model_id_uniqueness_policy"] != "UNIQUE_ACROSS_PANEL_MEMBERS_AND_SEEDS":
        raise AuditContractError("model ID uniqueness policy is not registered")
    all_model_ids: list[str] = []
    for member in members:
        _validate_complete_panel_member(
            member,
            panel_O_train=panel["O_train"],
            panel_seed_policy=seed_policy,
            deployment_contract=deployment,
        )
        all_model_ids.extend(member["model_ids"])
    if len(all_model_ids) != len(set(all_model_ids)):
        raise AuditContractError("model IDs must be unique across panel members and seeds")


def freeze_comparison_panel(panel: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze only a complete member-level comparison contract."""

    frozen = _panel_payload(panel)
    frozen.pop("status", None)
    frozen["status"] = "FROZEN"
    _validate_complete_panel_contract(frozen)
    frozen["panel_sha256"] = canonical_sha256(_panel_payload(frozen))
    return frozen


def verify_frozen_panel(panel: Mapping[str, Any]) -> str:
    panel_copy = copy.deepcopy(dict(panel))
    if panel_copy.get("status") != "FROZEN" or not panel_copy.get("panel_sha256"):
        raise AuditContractError("comparison panel is not frozen")
    claimed_hash = panel_copy.get("panel_sha256")
    _require_sha256(claimed_hash, "panel_sha256")
    observed = canonical_sha256(_panel_payload(panel_copy))
    if observed != claimed_hash:
        raise AuditContractError("frozen comparison panel hash mismatch")
    validation_payload = _panel_payload(panel_copy)
    _validate_complete_panel_contract(validation_payload)
    return observed


def validate_supplemental_panel(main_panel: Mapping[str, Any], supplemental_panel: Mapping[str, Any]) -> None:
    """Require late low-coverage methods to live in a separate child panel."""

    main_hash = verify_frozen_panel(main_panel)
    if supplemental_panel.get("panel_id") == main_panel.get("panel_id"):
        raise AuditContractError("supplemental panel must have a distinct panel_id")
    if supplemental_panel.get("parent_panel_id") != main_panel.get("panel_id"):
        raise AuditContractError("supplemental panel must name the main panel as parent")
    if supplemental_panel.get("parent_panel_sha256") != main_hash:
        raise AuditContractError("supplemental panel parent hash mismatch")
    if supplemental_panel.get("may_redefine_parent_domain") is not False:
        raise AuditContractError("supplemental panel must not redefine the parent common domain")


def assert_2025_performance_access_allowed(freeze_receipt: Mapping[str, Any]) -> None:
    """Fail closed unless every 2025 performance-release condition is frozen."""

    required_true = (
        "candidate_O_frozen",
        "delta_approved_with_domain_rationale",
        "models_frozen",
        "metrics_frozen",
        "families_frozen",
        "source_code_frozen",
        "source_hashes_verified",
        "record_order_contract_frozen",
        "human_release_decision_recorded",
    )
    missing = [field for field in required_true if freeze_receipt.get(field) is not True]
    if missing:
        raise AuditContractError(f"2025 micro-outcome/performance access remains locked: {','.join(missing)}")
    if freeze_receipt.get("validation_year") != 2025:
        raise AuditContractError("freeze receipt does not bind validation year 2025")
    if freeze_receipt.get("status") != "APPROVED_FOR_ONE_SHOT_2025_EVALUATION":
        raise AuditContractError("2025 release status is not approved")
