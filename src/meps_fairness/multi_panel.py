"""Pure-synthetic contracts for the Gate 14B multi-panel design.

This module deliberately has no dependency on the MEPS data layer.  Its data
interfaces accept already-materialized in-memory records only; they do not
open data files, run a pipeline, fit a model, or calculate a population
estimate.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterable, Mapping
from numbers import Integral
from pathlib import Path
from typing import Any, Final, TypeAlias


GATE14A_DECISION: Final = "DESIGN_APPROVED_FOR_SYNTHETIC_IMPLEMENTATION"
SYNTHETIC_IMPLEMENTATION_READY: Final = "SYNTHETIC_IMPLEMENTATION_READY_FOR_CODEX_REVIEW"

FIXED_CANDIDATE_PANELS: Final = (23, 24, 25, 26)
LOCKED_PANEL_27: Final = 27
ALLOWED_PARTITIONS: Final = ("train", "validation", "calibration")
PRIMARY_AUDIT_DIMENSIONS: Final = ("race_ethnicity", "sex")

MIN_TOTAL_POSITIVE_EVENTS: Final = 200
MIN_PRIMARY_AUDIT_N: Final = 100
MIN_PRIMARY_AUDIT_POSITIVE_EVENTS: Final = 20
MIN_PRIMARY_AUDIT_NEGATIVE_EVENTS: Final = 20
MIN_PRIMARY_AUDIT_KISH_N: Final = 50.0

MODEL_DEVELOPMENT_LOSS_WEIGHT_SEMANTICS: Final = (
    "MODEL_DEVELOPMENT_LOSS_WEIGHT_ONLY; NOT_A_POOLED_POPULATION_WEIGHT"
)

DEFAULT_CONFIG_PATH: Final = (
    Path(__file__).resolve().parents[2] / "configs" / "development_panels.json"
)

Record: TypeAlias = Mapping[str, Any] | Any


class Gate14BError(ValueError):
    """Stable, machine-readable failure for a Gate 14B contract violation."""

    def __init__(self, code: str, message: str) -> None:
        if not code or any(character.isspace() for character in code):
            raise ValueError("Gate 14B error codes must be non-empty and whitespace-free")
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclasses.dataclass(frozen=True)
class Gate14AContract:
    """The small, validated portion of the Gate 14A configuration used here."""

    decision: str
    candidate_panels: tuple[int, ...]
    locked_panel: int
    primary_audit_dimensions: tuple[str, ...]
    minimum_total_positive_events: int
    primary_audit_n_min: int
    primary_audit_positive_min: int
    primary_audit_negative_min: int
    primary_audit_kish_n_min: float


@dataclasses.dataclass(frozen=True)
class NamespacedRecordKeys:
    """Literal composite keys for one in-memory synthetic record."""

    person: tuple[Any, ...]
    household: tuple[Any, ...]
    pid: tuple[Any, ...]
    stratum: tuple[Any, ...]
    psu: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class SemanticContractCheck:
    """Metadata about a successful standardized semantic-contract comparison."""

    panels: tuple[int, ...]
    consistent: bool = True


@dataclasses.dataclass(frozen=True)
class PanelEventSummary:
    """Synthetic eligible-record and event counts for one development panel."""

    panel: int
    n: int
    positive_events: int
    negative_events: int

    @property
    def panel_number(self) -> int:
        return self.panel

    @property
    def positive(self) -> int:
        return self.positive_events

    @property
    def negative(self) -> int:
        return self.negative_events


@dataclasses.dataclass(frozen=True)
class PrimaryAuditUnitSummary:
    """Power and estimability counts for one primary audit unit."""

    panel: int
    audit_unit: Any
    n: int
    positive_events: int
    negative_events: int
    kish_effective_n: float

    @property
    def panel_number(self) -> int:
        return self.panel

    @property
    def unit(self) -> Any:
        return self.audit_unit

    @property
    def positive(self) -> int:
        return self.positive_events

    @property
    def negative(self) -> int:
        return self.negative_events

    @property
    def kish_n(self) -> float:
        return self.kish_effective_n


@dataclasses.dataclass(frozen=True)
class MultiPanelReadinessReport:
    """Counts from a successful synthetic readiness check.

    The report contains only contract counts and loss-weight semantics.  It
    intentionally has no population estimate, model result, or Panel 27
    indicator field.
    """

    status: str
    decision: str
    panels: tuple[int, ...]
    total_n: int
    total_positive_events: int
    total_negative_events: int
    per_panel: tuple[PanelEventSummary, ...]
    primary_audit_units: tuple[PrimaryAuditUnitSummary, ...]
    loss_weight_semantics: str = MODEL_DEVELOPMENT_LOSS_WEIGHT_SEMANTICS

    @property
    def total_positive(self) -> int:
        return self.total_positive_events

    @property
    def total_negative(self) -> int:
        return self.total_negative_events

    @property
    def panel_counts(self) -> tuple[PanelEventSummary, ...]:
        return self.per_panel

    @property
    def audit_units(self) -> tuple[PrimaryAuditUnitSummary, ...]:
        return self.primary_audit_units


_MISSING = object()

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "PANEL": ("PANEL", "panel"),
    "DUID": ("DUID", "duid", "household_id"),
    "DUPERSID": ("DUPERSID", "dupersid", "person_id"),
    "PID": ("PID", "pid"),
    "VARSTR": ("VARSTR", "varstr"),
    "VARPSU": ("VARPSU", "varpsu"),
    "LONGWT": ("LONGWT", "longwt", "weight"),
    "OUTCOME": ("OUTCOME", "outcome", "Y", "y", "target"),
    "AUDIT_UNIT": (
        "AUDIT_UNIT",
        "audit_unit",
        "PRIMARY_AUDIT_UNIT",
        "primary_audit_unit",
    ),
    "PARTITION": ("PARTITION", "partition", "split"),
}


def _field_candidates(field: str) -> tuple[str, ...]:
    names: list[str] = []
    for name in (field, *_FIELD_ALIASES.get(field, ())):
        if name not in names:
            names.append(name)
    for name in (field.lower(), field.upper()):
        if name not in names:
            names.append(name)
    return tuple(names)


def _read_field(record: Record, field: str, *, required: bool = True) -> Any:
    """Read one configured field from a mapping or a simple record object."""

    candidates = _field_candidates(field)
    if isinstance(record, Mapping):
        for name in candidates:
            if name in record:
                return record[name]
    else:
        for name in candidates:
            if hasattr(record, name):
                return getattr(record, name)
    if required:
        raise Gate14BError("RECORD_FIELD_MISSING", f"required field is missing: {field}")
    return _MISSING


def _materialize_records(records: Iterable[Record] | Record) -> tuple[Record, ...]:
    if isinstance(records, (str, bytes)):
        raise Gate14BError("RECORDS_INVALID", "records must be in-memory records, not text")
    if isinstance(records, Mapping):
        materialized = (records,)
    else:
        try:
            materialized = tuple(records)  # type: ignore[arg-type]
        except TypeError:
            materialized = (records,)
    if not materialized:
        raise Gate14BError("NO_RECORDS", "at least one in-memory record is required")
    return materialized


def _read_panel_literal(record: Record, panel_field: str) -> int:
    value = _read_field(record, panel_field)
    if value is None or isinstance(value, bool) or not isinstance(value, Integral):
        raise Gate14BError("PANEL_VALUE_INVALID", "PANEL must be an integer literal")
    panel = int(value)
    if panel == LOCKED_PANEL_27:
        raise Gate14BError(
            "PANEL_27_LOCKED",
            "Panel 27 is a locked temporal holdout and cannot enter Gate 14B",
        )
    return panel


def _supported_panel(panel: Any) -> int:
    if isinstance(panel, bool) or not isinstance(panel, Integral):
        raise Gate14BError("PANEL_VALUE_INVALID", "PANEL must be an integer literal")
    normalized = int(panel)
    if normalized == LOCKED_PANEL_27:
        raise Gate14BError(
            "PANEL_27_LOCKED",
            "Panel 27 is a locked temporal holdout and cannot enter Gate 14B",
        )
    if normalized not in FIXED_CANDIDATE_PANELS:
        raise Gate14BError(
            "PANEL_NOT_CANDIDATE",
            "only fixed development candidate panels are allowed",
        )
    return normalized


def _hashable_literal(value: Any, code: str, label: str) -> Any:
    if value is None:
        raise Gate14BError(code, f"{label} cannot be missing")
    try:
        hash(value)
    except TypeError as exc:
        raise Gate14BError(code, f"{label} must be a hashable literal") from exc
    return value


def namespace_key(panel: Any, *components: Any) -> tuple[Any, ...]:
    """Return a literal tuple prefixed by the panel; no offsets or joins."""

    if not components:
        raise Gate14BError("NAMESPACE_COMPONENTS_MISSING", "a composite key needs a component")
    normalized_panel = _supported_panel(panel)
    checked = tuple(
        _hashable_literal(value, "NAMESPACE_VALUE_INVALID", "key component")
        for value in components
    )
    return (normalized_panel, *checked)


def person_key(panel: Any, dupersid: Any) -> tuple[Any, ...]:
    """Namespace ``DUPERSID`` as the literal composite key ``(PANEL, DUPERSID)``."""

    return namespace_key(panel, dupersid)


def household_key(panel: Any, duid: Any) -> tuple[Any, ...]:
    """Namespace ``DUID`` as the literal composite key ``(PANEL, DUID)``."""

    return namespace_key(panel, duid)


def pid_key(panel: Any, pid: Any) -> tuple[Any, ...]:
    """Namespace ``PID`` as the literal composite key ``(PANEL, PID)``."""

    return namespace_key(panel, pid)


def stratum_key(panel: Any, varstr: Any) -> tuple[Any, ...]:
    """Namespace ``VARSTR`` as the literal composite key ``(PANEL, VARSTR)``."""

    return namespace_key(panel, varstr)


def psu_key(panel: Any, varstr: Any, varpsu: Any) -> tuple[Any, ...]:
    """Namespace PSU as the literal composite key ``(PANEL, VARSTR, VARPSU)``."""

    return namespace_key(panel, varstr, varpsu)


def namespace_record_keys(
    record: Record,
    *,
    panel_field: str = "PANEL",
    household_field: str = "DUID",
    person_field: str = "DUPERSID",
    pid_field: str = "PID",
    stratum_field: str = "VARSTR",
    psu_field: str = "VARPSU",
) -> NamespacedRecordKeys:
    """Build all five namespaced keys for one in-memory record."""

    panel = _supported_panel(_read_field(record, panel_field))
    duid = _hashable_literal(
        _read_field(record, household_field), "IDENTIFIER_VALUE_INVALID", household_field
    )
    dupersid = _hashable_literal(
        _read_field(record, person_field), "IDENTIFIER_VALUE_INVALID", person_field
    )
    pid = _hashable_literal(
        _read_field(record, pid_field), "IDENTIFIER_VALUE_INVALID", pid_field
    )
    varstr = _hashable_literal(
        _read_field(record, stratum_field), "IDENTIFIER_VALUE_INVALID", stratum_field
    )
    varpsu = _hashable_literal(
        _read_field(record, psu_field), "IDENTIFIER_VALUE_INVALID", psu_field
    )
    return NamespacedRecordKeys(
        person=person_key(panel, dupersid),
        household=household_key(panel, duid),
        pid=pid_key(panel, pid),
        stratum=stratum_key(panel, varstr),
        psu=psu_key(panel, varstr, varpsu),
    )


def validate_namespaced_keys(
    records: Iterable[Record] | Record,
    *,
    panel_field: str = "PANEL",
    household_field: str = "DUID",
    person_field: str = "DUPERSID",
    pid_field: str = "PID",
    stratum_field: str = "VARSTR",
    psu_field: str = "VARPSU",
) -> tuple[NamespacedRecordKeys, ...]:
    """Validate and return namespaced keys without linking or deduplicating panels."""

    rows = _materialize_records(records)
    # Read every PANEL first so a holdout row is rejected before other fields
    # from that row are inspected.
    for row in rows:
        _supported_panel(_read_panel_literal(row, panel_field))

    namespaced = tuple(
        namespace_record_keys(
            row,
            panel_field=panel_field,
            household_field=household_field,
            person_field=person_field,
            pid_field=pid_field,
            stratum_field=stratum_field,
            psu_field=psu_field,
        )
        for row in rows
    )

    seen_person_keys: set[tuple[Any, ...]] = set()
    for keys in namespaced:
        if keys.person in seen_person_keys:
            raise Gate14BError(
                "DUPLICATE_PERSON_WITHIN_PANEL",
                "a person key occurs more than once within a panel",
            )
        seen_person_keys.add(keys.person)

    # The panel prefix makes cross-panel intersections impossible for ordinary
    # literal values.  Keep this explicit as an auditable contract check.
    by_role: dict[str, dict[int, set[tuple[Any, ...]]]] = {
        "person": {},
        "household": {},
        "pid": {},
        "stratum": {},
        "psu": {},
    }
    for keys in namespaced:
        for role in by_role:
            value = getattr(keys, role)
            panel = int(value[0])
            by_role[role].setdefault(panel, set()).add(value)

    for role, panel_sets in by_role.items():
        panels = tuple(panel_sets)
        for index, left_panel in enumerate(panels):
            for right_panel in panels[index + 1 :]:
                if panel_sets[left_panel].intersection(panel_sets[right_panel]):
                    raise Gate14BError(
                        "NAMESPACE_COLLISION",
                        f"namespaced {role} keys collide across panels",
                    )
    return namespaced


def validate_household_partition_integrity(
    records: Iterable[Record] | Record,
    *,
    panel_field: str = "PANEL",
    household_field: str = "DUID",
    partition_field: str = "PARTITION",
) -> None:
    """Fail if one namespaced household occurs in multiple loss partitions."""

    rows = _materialize_records(records)
    assignments: dict[tuple[Any, ...], str] = {}
    for row in rows:
        panel = _supported_panel(_read_panel_literal(row, panel_field))
        duid = _hashable_literal(
            _read_field(row, household_field), "IDENTIFIER_VALUE_INVALID", household_field
        )
        partition = _read_field(row, partition_field)
        if not isinstance(partition, str) or partition not in ALLOWED_PARTITIONS:
            raise Gate14BError(
                "PARTITION_VALUE_INVALID",
                "partition must be train, validation, or calibration",
            )
        key = household_key(panel, duid)
        previous = assignments.get(key)
        if previous is not None and previous != partition:
            raise Gate14BError(
                "CROSS_PARTITION_LEAKAGE",
                "one namespaced (PANEL, DUID) occurs in multiple partitions",
            )
        assignments[key] = partition


def _freeze_for_comparison(value: Any) -> Any:
    """Create a deterministic, read-only comparison form for semantic specs."""

    if isinstance(value, Mapping):
        pairs = [
            (repr(key), _freeze_for_comparison(item))
            for key, item in value.items()
        ]
        return ("mapping", tuple(sorted(pairs, key=lambda pair: pair[0])))
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_freeze_for_comparison(item) for item in value))
    if isinstance(value, (set, frozenset)):
        frozen = [_freeze_for_comparison(item) for item in value]
        return ("set", tuple(sorted(frozen, key=repr)))
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if math.isinf(value):
            return ("float", "positive_inf" if value > 0 else "negative_inf")
    return value


def _semantic_panel_key(value: Any) -> int:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            value = int(stripped)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise Gate14BError(
            "SEMANTIC_CONTRACT_INVALID",
            "semantic contract panel keys must be integer panel literals",
        )
    return int(value)


def validate_standardized_semantic_contract(
    semantic_contracts: Mapping[Any, Mapping[str, Any]] | None,
    *,
    required_panels: Iterable[int] = FIXED_CANDIDATE_PANELS,
) -> SemanticContractCheck:
    """Require one identical standardized semantic contract for every panel."""

    if not isinstance(semantic_contracts, Mapping):
        raise Gate14BError(
            "SEMANTIC_CONTRACT_MISSING",
            "standardized semantic contracts are required for every candidate panel",
        )

    required = tuple(int(panel) for panel in required_panels)
    if len(set(required)) != len(required) or any(
        panel == LOCKED_PANEL_27 for panel in required
    ):
        raise Gate14BError(
            "SEMANTIC_CONTRACT_INVALID",
            "required semantic panels must be distinct development candidates",
        )

    normalized: dict[int, Mapping[str, Any]] = {}
    for raw_panel, contract in semantic_contracts.items():
        panel = _semantic_panel_key(raw_panel)
        if panel == LOCKED_PANEL_27:
            raise Gate14BError(
                "PANEL_27_LOCKED",
                "Panel 27 semantic information is outside the Gate 14B boundary",
            )
        if panel not in FIXED_CANDIDATE_PANELS:
            raise Gate14BError(
                "PANEL_SET_MISMATCH",
                "semantic contracts may contain only fixed candidate panels",
            )
        if panel in normalized:
            raise Gate14BError(
                "SEMANTIC_CONTRACT_INVALID",
                "duplicate semantic contract panel key",
            )
        if not isinstance(contract, Mapping) or not contract:
            raise Gate14BError(
                "SEMANTIC_CONTRACT_INVALID",
                "each semantic contract must be a non-empty mapping",
            )
        normalized[panel] = contract

    required_set = set(required)
    provided_set = set(normalized)
    if provided_set - required_set:
        raise Gate14BError(
            "SEMANTIC_CONTRACT_EXTRA_PANEL",
            "a semantic contract is present for a non-required panel",
        )
    if required_set - provided_set:
        raise Gate14BError(
            "SEMANTIC_CONTRACT_MISSING",
            "a required candidate panel has no semantic contract",
        )

    reference = _freeze_for_comparison(normalized[required[0]])
    for panel in required[1:]:
        if _freeze_for_comparison(normalized[panel]) != reference:
            raise Gate14BError(
                "SEMANTIC_CONTRACT_CONFLICT",
                "standardized semantic contracts conflict across candidate panels",
            )
    return SemanticContractCheck(panels=required)


# Short aliases keep the public contract easy to discover without changing the
# single implementation or its stable error codes.
validate_semantic_contract = validate_standardized_semantic_contract
validate_semantic_alignment_contract = validate_standardized_semantic_contract


def _config_value(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    try:
        for key in path:
            current = current[key]
    except (KeyError, IndexError, TypeError) as exc:
        joined = ".".join(path)
        raise Gate14BError(
            "GATE14A_CONFIG_INVALID",
            f"required Gate 14A configuration field is missing: {joined}",
        ) from exc
    return current


def _expect_config(
    config: Mapping[str, Any],
    path: tuple[str, ...],
    expected: Any,
    *,
    code: str = "GATE14A_CONFIG_INVALID",
) -> None:
    actual = _config_value(config, path)
    if isinstance(expected, bool):
        matches = type(actual) is bool and actual is expected
    else:
        matches = actual == expected
    if not matches:
        joined = ".".join(path)
        raise Gate14BError(code, f"Gate 14A configuration disagrees at {joined}")


def read_gate14a_decision(config_path: str | Path | None = None) -> Gate14AContract:
    """Read and validate the Gate 14A decision state from JSON only."""

    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    try:
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError as exc:
        raise Gate14BError(
            "GATE14A_CONFIG_UNREADABLE", "Gate 14A configuration file is missing"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise Gate14BError(
            "GATE14A_CONFIG_INVALID", "Gate 14A configuration is not valid JSON"
        ) from exc

    if not isinstance(config, Mapping):
        raise Gate14BError("GATE14A_CONFIG_INVALID", "Gate 14A configuration must be an object")
    if config.get("gate") != "14A" or config.get("decision") != GATE14A_DECISION:
        raise Gate14BError(
            "GATE14A_DECISION_MISMATCH",
            "Gate 14A is not authorized for synthetic implementation",
        )

    scope_path = ("design_scope",)
    _expect_config(config, scope_path + ("metadata_only",), True)
    for field in (
        "microdata_read_allowed",
        "new_downloads_allowed",
        "outcome_values_read_allowed",
        "real_pipeline_allowed",
        "model_training_allowed",
        "bootstrap_allowed",
        "panel_27_access_allowed",
    ):
        _expect_config(config, scope_path + (field,), False)

    expected_pufs = ("HC-217", "HC-225", "HC-234", "HC-244")
    expected_candidates = (
        ("HC-217", 23, 2018, 2019),
        ("HC-225", 24, 2019, 2020),
        ("HC-234", 25, 2020, 2021),
        ("HC-244", 26, 2021, 2022),
    )
    selection = _config_value(config, ("panel_selection",))
    if not isinstance(selection, Mapping):
        raise Gate14BError("GATE14A_CONFIG_INVALID", "panel_selection must be an object")
    if selection.get("fixed_candidate_sequence") != list(expected_pufs):
        raise Gate14BError(
            "GATE14A_CONFIG_INVALID", "fixed candidate sequence is not the Gate 14A sequence"
        )
    for field in ("candidate_set_frozen_before_outcome_access", "selection_may_not_use_event_counts", "all_four_candidates_required_for_selected_design"):
        _expect_config(config, ("panel_selection", field), True)

    candidates = config.get("candidate_panels")
    if not isinstance(candidates, list):
        raise Gate14BError("GATE14A_CONFIG_INVALID", "candidate_panels must be a list")
    observed_candidates: list[tuple[Any, Any, Any, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise Gate14BError("GATE14A_CONFIG_INVALID", "candidate panel entries must be objects")
        try:
            observed_candidates.append(
                (
                    candidate["puf_id"],
                    candidate["panel_number"],
                    candidate["baseline_year"],
                    candidate["follow_up_year"],
                )
            )
        except (KeyError, TypeError) as exc:
            raise Gate14BError(
                "GATE14A_CONFIG_INVALID", "candidate panel metadata is incomplete"
            ) from exc
    if tuple(observed_candidates) != expected_candidates:
        raise Gate14BError(
            "GATE14A_CONFIG_INVALID", "candidate panel metadata does not match Gate 14A"
        )

    namespace_keys = {
        "person": ["PANEL", "DUPERSID"],
        "household_split_group": ["PANEL", "DUID"],
        "pid": ["PANEL", "PID"],
        "stratum": ["PANEL", "VARSTR"],
        "psu": ["PANEL", "VARSTR", "VARPSU"],
    }
    _expect_config(
        config,
        ("identity_and_design_namespaces", "required_composite_keys"),
        namespace_keys,
    )
    _expect_config(
        config,
        ("identity_and_design_namespaces", "panel_prefix_required_for_cross_panel_operations"),
        True,
    )
    _expect_config(
        config,
        ("identity_and_design_namespaces", "no_numeric_offset_or_record_linkage"),
        True,
    )

    protected_dimensions = _config_value(
        config,
        ("canonical_estimand", "protected_audit_dimensions"),
    )
    if not isinstance(protected_dimensions, Mapping):
        raise Gate14BError(
            "GATE14A_CONFIG_INVALID",
            "protected_audit_dimensions must be an object",
        )
    configured_primary_dimensions = protected_dimensions.get("primary")
    if not isinstance(configured_primary_dimensions, list) or any(
        not isinstance(dimension, str) for dimension in configured_primary_dimensions
    ):
        raise Gate14BError(
            "GATE14A_CONFIG_INVALID",
            "protected_audit_dimensions.primary must be a list of strings",
        )
    if tuple(configured_primary_dimensions) != PRIMARY_AUDIT_DIMENSIONS:
        raise Gate14BError(
            "GATE14A_CONFIG_INVALID",
            "primary audit dimensions do not match the frozen Gate 14A contract",
        )

    _expect_config(
        config,
        ("longwt_and_pooling", "development_model_loss_only", "not_a_population_estimator"),
        True,
    )
    _expect_config(
        config,
        ("longwt_and_pooling", "primary_population_estimation", "cross_panel_population_pooling"),
        False,
    )

    power = _config_value(config, ("power_and_estimability",))
    if not isinstance(power, Mapping):
        raise Gate14BError("GATE14A_CONFIG_INVALID", "power_and_estimability must be an object")
    thresholds = _config_value(power, ("estimability_rules",))
    if not isinstance(thresholds, Mapping):
        raise Gate14BError("GATE14A_CONFIG_INVALID", "estimability_rules must be an object")
    if power.get("minimum_total_positive_events") != MIN_TOTAL_POSITIVE_EVENTS:
        raise Gate14BError("GATE14A_CONFIG_INVALID", "total event threshold disagrees with Gate 14A")
    expected_thresholds = {
        "unweighted_subgroup_n_min": MIN_PRIMARY_AUDIT_N,
        "positive_events_min": MIN_PRIMARY_AUDIT_POSITIVE_EVENTS,
        "negative_events_min": MIN_PRIMARY_AUDIT_NEGATIVE_EVENTS,
        "kish_effective_n_min": MIN_PRIMARY_AUDIT_KISH_N,
    }
    for field, expected in expected_thresholds.items():
        if thresholds.get(field) != expected:
            raise Gate14BError("GATE14A_CONFIG_INVALID", f"estimability threshold disagrees at {field}")

    holdout = _config_value(config, ("locked_holdout",))
    if not isinstance(holdout, Mapping):
        raise Gate14BError("GATE14A_LOCK_INVALID", "locked_holdout must be an object")
    for field, expected in (
        ("puf_id", "HC-252"),
        ("panel_number", LOCKED_PANEL_27),
        ("included_in_development_candidates", False),
        ("outcome_values_allowed", False),
        ("protected_group_distributions_allowed", False),
        ("predictor_distributions_allowed", False),
        ("performance_metrics_allowed", False),
        ("tuning_or_selection_allowed", False),
    ):
        if holdout.get(field) != expected:
            raise Gate14BError("GATE14A_LOCK_INVALID", f"Panel 27 lock disagrees at {field}")

    _expect_config(
        config,
        ("gate_14b_boundary", "authorized_only_if_decision_matches"),
        GATE14A_DECISION,
    )
    _expect_config(config, ("gate_14b_boundary", "real_data_still_prohibited"), True)

    return Gate14AContract(
        decision=GATE14A_DECISION,
        candidate_panels=FIXED_CANDIDATE_PANELS,
        locked_panel=LOCKED_PANEL_27,
        primary_audit_dimensions=PRIMARY_AUDIT_DIMENSIONS,
        minimum_total_positive_events=MIN_TOTAL_POSITIVE_EVENTS,
        primary_audit_n_min=MIN_PRIMARY_AUDIT_N,
        primary_audit_positive_min=MIN_PRIMARY_AUDIT_POSITIVE_EVENTS,
        primary_audit_negative_min=MIN_PRIMARY_AUDIT_NEGATIVE_EVENTS,
        primary_audit_kish_n_min=MIN_PRIMARY_AUDIT_KISH_N,
    )


load_gate14a_contract = read_gate14a_decision
validate_gate14a_decision = read_gate14a_decision


def validate_fixed_candidate_panels(panel_numbers: Iterable[int]) -> tuple[int, ...]:
    """Require exactly the fixed four candidate panels, with no replacement."""

    try:
        observed_raw = tuple(panel_numbers)
    except TypeError as exc:
        raise Gate14BError("PANEL_SET_INVALID", "panel numbers must be iterable") from exc
    normalized: list[int] = []
    for value in observed_raw:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise Gate14BError("PANEL_SET_INVALID", "panel numbers must be integer literals")
        panel = int(value)
        if panel == LOCKED_PANEL_27:
            raise Gate14BError(
                "PANEL_27_LOCKED",
                "Panel 27 is a locked temporal holdout and cannot enter Gate 14B",
            )
        normalized.append(panel)
    if len(normalized) != len(set(normalized)):
        raise Gate14BError("PANEL_SET_MISMATCH", "candidate panel list contains duplicates")
    observed = set(normalized)
    expected = set(FIXED_CANDIDATE_PANELS)
    if observed == expected:
        return FIXED_CANDIDATE_PANELS
    if observed < expected:
        raise Gate14BError(
            "PANEL_SET_INCOMPLETE",
            "the fixed candidate set is missing one or more required panels",
        )
    raise Gate14BError(
        "PANEL_SET_MISMATCH",
        "candidate panels contain a replacement or non-candidate panel",
    )


def _validate_weight(value: Any) -> float:
    if isinstance(value, (str, bytes, bool)) or value is None:
        raise Gate14BError("LONGWT_VALUE_INVALID", "LONGWT must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Gate14BError("LONGWT_VALUE_INVALID", "LONGWT must be numeric") from exc
    if not math.isfinite(numeric):
        raise Gate14BError("LONGWT_NOT_FINITE", "LONGWT must be finite")
    if numeric <= 0.0:
        raise Gate14BError("LONGWT_NOT_POSITIVE", "LONGWT must be strictly positive")
    return numeric


def _validate_outcome(value: Any) -> int:
    if isinstance(value, (str, bytes)) or value is None:
        raise Gate14BError("OUTCOME_INVALID", "OUTCOME must be binary 0 or 1")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Gate14BError("OUTCOME_INVALID", "OUTCOME must be binary 0 or 1") from exc
    if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise Gate14BError("OUTCOME_INVALID", "OUTCOME must be binary 0 or 1")
    return int(numeric)


def _single_loss_partition(
    rows: tuple[Record, ...],
    *,
    partition_field: str,
    partition_name: str | None,
) -> None:
    values: list[str] = []
    missing = 0
    for row in rows:
        value = _read_field(row, partition_field, required=False)
        if value is _MISSING or value is None:
            missing += 1
            continue
        if not isinstance(value, str) or value not in ALLOWED_PARTITIONS:
            raise Gate14BError(
                "LOSS_PARTITION_INVALID",
                "loss weights accept only train, validation, or calibration",
            )
        values.append(value)

    if partition_name is not None:
        if partition_name not in ALLOWED_PARTITIONS:
            raise Gate14BError(
                "LOSS_PARTITION_INVALID",
                "loss partition must be train, validation, or calibration",
            )
        if missing or len(values) != len(rows) or any(value != partition_name for value in values):
            raise Gate14BError(
                "LOSS_PARTITION_NOT_SINGLE",
                "all records must belong to the requested single loss partition",
            )
    elif missing:
        raise Gate14BError(
            "LOSS_PARTITION_NOT_SINGLE",
            "loss partition labels are incomplete",
        )
    elif len(set(values)) > 1:
        raise Gate14BError(
            "LOSS_PARTITION_NOT_SINGLE",
            "one loss-weight call may contain only one partition",
        )


def _normalized_positive_weights(weights: list[float]) -> list[float]:
    if not weights:
        raise Gate14BError("NO_RECORDS", "at least one weight is required")
    maximum = max(weights)
    scaled = [weight / maximum for weight in weights]
    denominator = math.fsum(scaled)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise Gate14BError("LONGWT_SUM_INVALID", "LONGWT cannot be normalized safely")
    return [value / denominator for value in scaled]


def compute_panel_balanced_loss_weights(
    records: Iterable[Record] | Record,
    *,
    panel_field: str = "PANEL",
    weight_field: str = "LONGWT",
    partition_field: str = "PARTITION",
    partition: str | None = None,
    partition_name: str | None = None,
    config_path: str | Path | None = None,
) -> tuple[float, ...]:
    """Compute effective model-development loss weights for one partition.

    Within each observed candidate panel, weights are normalized to sum to
    one and then multiplied by ``1/K``.  The returned vector therefore sums
    to one, with each panel contributing ``1/K``.  These are loss weights for
    model development only, never pooled population weights.
    """

    contract = read_gate14a_decision(config_path)
    rows = _materialize_records(records)
    panels = tuple(_supported_panel(_read_panel_literal(row, panel_field)) for row in rows)
    if any(panel not in contract.candidate_panels for panel in panels):
        raise Gate14BError("PANEL_NOT_CANDIDATE", "loss weights accept only fixed candidate panels")
    if partition is not None and partition_name is not None and partition != partition_name:
        raise Gate14BError("LOSS_PARTITION_INVALID", "partition aliases disagree")
    requested_partition = partition if partition is not None else partition_name
    _single_loss_partition(
        rows,
        partition_field=partition_field,
        partition_name=requested_partition,
    )

    raw_weights = [
        _validate_weight(_read_field(row, weight_field))
        for row in rows
    ]
    observed_panels = tuple(sorted(set(panels)))
    panel_count = len(observed_panels)
    result = [0.0] * len(rows)
    for panel in observed_panels:
        indices = [index for index, row_panel in enumerate(panels) if row_panel == panel]
        normalized = _normalized_positive_weights([raw_weights[index] for index in indices])
        for index, normalized_weight in zip(indices, normalized):
            result[index] = normalized_weight / panel_count

    for panel in observed_panels:
        panel_sum = math.fsum(
            result[index] for index, row_panel in enumerate(panels) if row_panel == panel
        )
        if not math.isclose(panel_sum, 1.0 / panel_count, rel_tol=1e-12, abs_tol=1e-15):
            raise Gate14BError("LOSS_WEIGHT_NORMALIZATION_FAILED", "panel loss weights do not sum to 1/K")
    total = math.fsum(result)
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-15):
        raise Gate14BError("LOSS_WEIGHT_NORMALIZATION_FAILED", "loss weights do not sum to one")
    return tuple(result)


panel_balanced_loss_weights = compute_panel_balanced_loss_weights


def _audit_cells(
    value: Any,
    *,
    required_dimensions: tuple[str, ...],
) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(value, Mapping):
        raise Gate14BError(
            "PRIMARY_AUDIT_DIMENSIONS_MISSING",
            "each record must provide the required primary audit dimensions",
        )
    missing_dimensions = tuple(
        dimension for dimension in required_dimensions if dimension not in value
    )
    if missing_dimensions:
        raise Gate14BError(
            "PRIMARY_AUDIT_DIMENSIONS_MISSING",
            "each record must provide race_ethnicity and sex audit groups",
        )

    # Only the frozen primary dimensions are estimability-gated here.  Extra
    # secondary audit dimensions may be carried by a record, but cannot become
    # required primary cells merely because they were supplied.
    return tuple(
        (
            dimension,
            _hashable_literal(value[dimension], "AUDIT_UNIT_INVALID", "audit group"),
        )
        for dimension in required_dimensions
    )


def _kish_effective_n(weights: list[float]) -> float:
    maximum = max(weights)
    scaled = [weight / maximum for weight in weights]
    sum_scaled = math.fsum(scaled)
    sum_squared_scaled = math.fsum(value * value for value in scaled)
    if sum_squared_scaled <= 0.0 or not math.isfinite(sum_squared_scaled):
        raise Gate14BError("KISH_EFFECTIVE_N_INVALID", "Kish effective n cannot be computed")
    return float((sum_scaled * sum_scaled) / sum_squared_scaled)


def validate_multi_panel_readiness(
    records: Iterable[Record] | Record,
    *,
    semantic_contracts: Mapping[Any, Mapping[str, Any]] | None,
    config_path: str | Path | None = None,
    panel_field: str = "PANEL",
    household_field: str = "DUID",
    person_field: str = "DUPERSID",
    pid_field: str = "PID",
    stratum_field: str = "VARSTR",
    psu_field: str = "VARPSU",
    weight_field: str = "LONGWT",
    outcome_field: str = "OUTCOME",
    audit_unit_field: str = "AUDIT_UNIT",
    partition_field: str = "PARTITION",
) -> MultiPanelReadinessReport:
    """Run the complete Gate 14B contract on pure in-memory records only."""

    contract = read_gate14a_decision(config_path)
    rows = _materialize_records(records)

    # The first pass reads only PANEL.  In particular, a Panel 27 row is
    # rejected before its outcome, identifiers, or any other field is read.
    panels = tuple(_read_panel_literal(row, panel_field) for row in rows)
    # Records repeat panel labels by design; candidate-set validation applies
    # to the distinct observed panels, not to the row-wise label sequence.
    validate_fixed_candidate_panels(set(panels))

    validate_standardized_semantic_contract(
        semantic_contracts,
        required_panels=contract.candidate_panels,
    )
    validate_namespaced_keys(
        rows,
        panel_field=panel_field,
        household_field=household_field,
        person_field=person_field,
        pid_field=pid_field,
        stratum_field=stratum_field,
        psu_field=psu_field,
    )
    validate_household_partition_integrity(
        rows,
        panel_field=panel_field,
        household_field=household_field,
        partition_field=partition_field,
    )

    weights = [
        _validate_weight(_read_field(row, weight_field))
        for row in rows
    ]
    outcomes = [
        _validate_outcome(_read_field(row, outcome_field))
        for row in rows
    ]

    panel_stats: dict[int, list[int]] = {
        panel: [0, 0, 0] for panel in contract.candidate_panels
    }
    audit_stats: dict[tuple[int, tuple[Any, ...]], list[Any]] = {}
    for panel, weight, outcome, row in zip(panels, weights, outcomes, rows):
        panel_stats[panel][0] += 1
        panel_stats[panel][1] += outcome
        panel_stats[panel][2] += 1 - outcome
        for cell in _audit_cells(
            _read_field(row, audit_unit_field),
            required_dimensions=contract.primary_audit_dimensions,
        ):
            stats = audit_stats.setdefault((panel, cell), [0, 0, 0, []])
            stats[0] += 1
            stats[1] += outcome
            stats[2] += 1 - outcome
            stats[3].append(weight)

    total_n = len(rows)
    total_positive = sum(outcomes)
    total_negative = total_n - total_positive
    if total_positive < contract.minimum_total_positive_events:
        raise Gate14BError(
            "TOTAL_POSITIVE_EVENTS_INSUFFICIENT",
            "total positive events are below the frozen minimum of 200",
        )

    audit_summaries: list[PrimaryAuditUnitSummary] = []
    for panel, cell in sorted(audit_stats, key=repr):
        n, positive, negative, cell_weights = audit_stats[(panel, cell)]
        kish = _kish_effective_n(cell_weights)
        if n < contract.primary_audit_n_min:
            raise Gate14BError(
                "AUDIT_UNIT_N_INSUFFICIENT",
                "a primary audit unit is below the frozen n threshold",
            )
        if positive < contract.primary_audit_positive_min:
            raise Gate14BError(
                "AUDIT_UNIT_POSITIVE_EVENTS_INSUFFICIENT",
                "a primary audit unit is below the frozen positive-event threshold",
            )
        if negative < contract.primary_audit_negative_min:
            raise Gate14BError(
                "AUDIT_UNIT_NEGATIVE_EVENTS_INSUFFICIENT",
                "a primary audit unit is below the frozen negative-event threshold",
            )
        if kish < contract.primary_audit_kish_n_min:
            raise Gate14BError(
                "AUDIT_UNIT_KISH_N_INSUFFICIENT",
                "a primary audit unit is below the frozen Kish effective-n threshold",
            )
        audit_summaries.append(
            PrimaryAuditUnitSummary(
                panel=panel,
                audit_unit=cell,
                n=n,
                positive_events=positive,
                negative_events=negative,
                kish_effective_n=kish,
            )
        )

    panel_summaries = tuple(
        PanelEventSummary(
            panel=panel,
            n=panel_stats[panel][0],
            positive_events=panel_stats[panel][1],
            negative_events=panel_stats[panel][2],
        )
        for panel in contract.candidate_panels
    )
    return MultiPanelReadinessReport(
        status=SYNTHETIC_IMPLEMENTATION_READY,
        decision=contract.decision,
        panels=contract.candidate_panels,
        total_n=total_n,
        total_positive_events=total_positive,
        total_negative_events=total_negative,
        per_panel=panel_summaries,
        primary_audit_units=tuple(audit_summaries),
    )


assess_multi_panel_readiness = validate_multi_panel_readiness
check_multi_panel_readiness = validate_multi_panel_readiness


__all__ = [
    "ALLOWED_PARTITIONS",
    "DEFAULT_CONFIG_PATH",
    "FIXED_CANDIDATE_PANELS",
    "Gate14AContract",
    "Gate14BError",
    "MODEL_DEVELOPMENT_LOSS_WEIGHT_SEMANTICS",
    "MultiPanelReadinessReport",
    "NamespacedRecordKeys",
    "PanelEventSummary",
    "PRIMARY_AUDIT_DIMENSIONS",
    "PrimaryAuditUnitSummary",
    "SYNTHETIC_IMPLEMENTATION_READY",
    "SemanticContractCheck",
    "assess_multi_panel_readiness",
    "check_multi_panel_readiness",
    "compute_panel_balanced_loss_weights",
    "household_key",
    "load_gate14a_contract",
    "namespace_key",
    "namespace_record_keys",
    "panel_balanced_loss_weights",
    "person_key",
    "pid_key",
    "psu_key",
    "read_gate14a_decision",
    "stratum_key",
    "validate_fixed_candidate_panels",
    "validate_gate14a_decision",
    "validate_household_partition_integrity",
    "validate_multi_panel_readiness",
    "validate_namespaced_keys",
    "validate_semantic_alignment_contract",
    "validate_semantic_contract",
    "validate_standardized_semantic_contract",
]
