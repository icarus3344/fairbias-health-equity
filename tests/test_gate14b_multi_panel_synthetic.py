"""Pure in-memory tests for the Gate 14B multi-panel contract.

No test in this file opens a MEPS file, imports the data pipeline, trains a
model, runs bootstrap inference, or creates a population estimate.
"""

from __future__ import annotations

import copy
import pathlib
import sys
import unittest


_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from meps_fairness.multi_panel import (
    FIXED_CANDIDATE_PANELS,
    GATE14A_DECISION,
    MODEL_DEVELOPMENT_LOSS_WEIGHT_SEMANTICS,
    SYNTHETIC_IMPLEMENTATION_READY,
    Gate14BError,
    compute_panel_balanced_loss_weights,
    read_gate14a_decision,
    validate_fixed_candidate_panels,
    validate_household_partition_integrity,
    validate_multi_panel_readiness,
    validate_namespaced_keys,
    validate_standardized_semantic_contract,
)


SEMANTIC_SPEC = {
    "eligibility": {
        "age": {"canonical_name": "AGEY1X", "timing": "baseline_y1", "type": "continuous"},
        "both_years": {"canonical_name": "YEARIND", "valid_code": 1},
    },
    "outcome": {
        "canonical_name": "any_uninsured_month_y2",
        "timing": "follow_up_y2",
        "codes": {"insured": 1, "uninsured": 2},
    },
    "survey_design": {
        "weight": {"canonical_name": "LONGWT", "role": "longitudinal_weight"},
        "stratum": {"canonical_name": "VARSTR", "role": "stratum"},
        "psu": {"canonical_name": "VARPSU", "role": "psu"},
    },
}


def _semantic_contracts() -> dict[int, dict]:
    return {panel: copy.deepcopy(SEMANTIC_SPEC) for panel in FIXED_CANDIDATE_PANELS}


def _record(
    panel: int,
    index: int,
    outcome: int,
    audit_unit: str | dict[str, str],
    *,
    weight: float = 1.0,
    duid: int | None = None,
    partition: str = "train",
) -> dict:
    raw_id = index if duid is None else duid
    audit_value = copy.deepcopy(audit_unit) if isinstance(audit_unit, dict) else {
        "race_ethnicity": audit_unit,
        "sex": "all",
    }
    return {
        "PANEL": panel,
        "DUID": raw_id,
        "DUPERSID": index,
        "PID": index,
        "VARSTR": index % 7,
        "VARPSU": index % 3,
        "LONGWT": weight,
        "OUTCOME": outcome,
        "AUDIT_UNIT": audit_value,
        "PARTITION": partition,
    }


def _full_records() -> list[dict]:
    rows: list[dict] = []
    partitions = ("train", "validation", "calibration")
    for panel in FIXED_CANDIDATE_PANELS:
        for index in range(100):
            rows.append(
                _record(
                    panel,
                    index,
                    int(index < 50),
                    {"race_ethnicity": "A", "sex": "F"},
                    duid=index,
                    partition=partitions[index % 3],
                )
            )
    return rows


def _audit_spec_records(specs: list[tuple[str, int, int]], *, heavy_first_unit: bool = False) -> list[dict]:
    rows: list[dict] = []
    partitions = ("train", "validation", "calibration")
    row_index = 0
    for panel in FIXED_CANDIDATE_PANELS:
        for audit_unit, n, positive_count in specs:
            for within_unit_index in range(n):
                weight = 1.0
                if heavy_first_unit and audit_unit == "low_kish" and within_unit_index == 0:
                    weight = 1_000_000_000.0
                rows.append(
                    _record(
                        panel,
                        row_index,
                        int(within_unit_index < positive_count),
                        audit_unit,
                        weight=weight,
                        duid=row_index,
                        partition=partitions[row_index % 3],
                    )
                )
                row_index += 1
    return rows


class TestGate14BMultiPanelSynthetic(unittest.TestCase):
    def assert_code(self, code: str, function, *args, **kwargs) -> None:
        with self.assertRaises(Gate14BError) as context:
            function(*args, **kwargs)
        self.assertEqual(context.exception.code, code)

    def readiness(self, rows: list[dict]):
        return validate_multi_panel_readiness(rows, semantic_contracts=_semantic_contracts())

    def test_gate14a_decision_is_read_and_validated(self) -> None:
        contract = read_gate14a_decision()
        self.assertEqual(contract.decision, GATE14A_DECISION)
        self.assertEqual(contract.candidate_panels, FIXED_CANDIDATE_PANELS)
        self.assertEqual(contract.locked_panel, 27)
        self.assertEqual(contract.primary_audit_dimensions, ("race_ethnicity", "sex"))

    def test_single_panel_namespace_and_weights_work_but_readiness_rejects_incomplete_set(self) -> None:
        rows = [_record(23, index, index % 2, "A", weight=index + 1) for index in range(4)]
        keys = validate_namespaced_keys(rows)
        self.assertEqual(len(keys), 4)
        weights = compute_panel_balanced_loss_weights(rows)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assert_code(
            "PANEL_SET_INCOMPLETE",
            self.readiness,
            rows,
        )

    def test_fixed_candidate_replacement_and_panel27_fail_closed(self) -> None:
        self.assert_code("PANEL_SET_MISMATCH", validate_fixed_candidate_panels, [23, 24, 25, 99])
        self.assert_code("PANEL_27_LOCKED", validate_fixed_candidate_panels, [23, 24, 25, 27])

    def test_same_raw_ids_across_four_panels_have_unique_literal_namespaces(self) -> None:
        rows = [_record(panel, 7, 0, "A", duid=11) for panel in FIXED_CANDIDATE_PANELS]
        keys = validate_namespaced_keys(rows)
        self.assertEqual(len({item.person for item in keys}), 4)
        self.assertEqual(len({item.household for item in keys}), 4)
        self.assertEqual(len({item.pid for item in keys}), 4)
        self.assertEqual(len({item.stratum for item in keys}), 4)
        self.assertEqual(len({item.psu for item in keys}), 4)

    def test_duplicate_person_within_panel_cannot_inflate_readiness_counts(self) -> None:
        rows = [
            _record(
                panel,
                1,
                int(within_panel_index < 50),
                {"race_ethnicity": "A", "sex": "S"},
                duid=1,
            )
            for panel in FIXED_CANDIDATE_PANELS
            for within_panel_index in range(100)
        ]
        self.assert_code("DUPLICATE_PERSON_WITHIN_PANEL", self.readiness, rows)

    def test_primary_audit_cells_are_checked_within_each_panel(self) -> None:
        rows: list[dict] = []
        partitions = ("train", "validation", "calibration")
        row_index = 0
        for panel in FIXED_CANDIDATE_PANELS:
            for within_panel_index in range(25):
                rows.append(
                    _record(
                        panel,
                        row_index,
                        int(within_panel_index < 5),
                        {"race_ethnicity": "sparse", "sex": "F"},
                        duid=row_index,
                        partition=partitions[row_index % 3],
                    )
                )
                row_index += 1
            for within_panel_index in range(100):
                rows.append(
                    _record(
                        panel,
                        row_index,
                        int(within_panel_index < 45),
                        {"race_ethnicity": "adequate", "sex": "F"},
                        duid=row_index,
                        partition=partitions[row_index % 3],
                    )
                )
                row_index += 1
        self.assert_code("AUDIT_UNIT_N_INSUFFICIENT", self.readiness, rows)

    def test_primary_audit_dimensions_cannot_be_omitted(self) -> None:
        rows = _full_records()
        rows[0] = {
            **rows[0],
            "AUDIT_UNIT": {"race_ethnicity": "A"},
        }
        self.assert_code("PRIMARY_AUDIT_DIMENSIONS_MISSING", self.readiness, rows)

    def test_same_namespaced_household_across_partitions_fails(self) -> None:
        rows = [
            _record(23, 1, 0, "A", duid=55, partition="train"),
            _record(23, 2, 1, "A", duid=55, partition="validation"),
        ]
        self.assert_code("CROSS_PARTITION_LEAKAGE", validate_household_partition_integrity, rows)

    def test_semantic_contract_conflict_fails_closed(self) -> None:
        contracts = _semantic_contracts()
        contracts[24]["outcome"]["codes"]["uninsured"] = 9
        self.assert_code("SEMANTIC_CONTRACT_CONFLICT", validate_standardized_semantic_contract, contracts)

    def test_nan_infinite_zero_and_negative_longwt_fail(self) -> None:
        for value, expected_code in (
            (float("nan"), "LONGWT_NOT_FINITE"),
            (float("inf"), "LONGWT_NOT_FINITE"),
            (0.0, "LONGWT_NOT_POSITIVE"),
            (-1.0, "LONGWT_NOT_POSITIVE"),
        ):
            with self.subTest(value=value):
                self.assert_code(
                    expected_code,
                    compute_panel_balanced_loss_weights,
                    [_record(23, 1, 0, "A", weight=value)],
                )

    def test_panel_balanced_loss_weights_are_one_over_k_per_panel(self) -> None:
        rows = [
            _record(23, 1, 0, "A", weight=1.0),
            _record(23, 2, 0, "A", weight=3.0),
            _record(24, 3, 0, "A", weight=2.0),
            _record(24, 4, 0, "A", weight=2.0),
        ]
        weights = compute_panel_balanced_loss_weights(rows, partition="train")
        panel_23_sum = sum(weight for row, weight in zip(rows, weights) if row["PANEL"] == 23)
        panel_24_sum = sum(weight for row, weight in zip(rows, weights) if row["PANEL"] == 24)
        self.assertAlmostEqual(panel_23_sum, 0.5)
        self.assertAlmostEqual(panel_24_sum, 0.5)
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_total_positive_event_threshold_is_frozen(self) -> None:
        rows = _full_records()
        rows[0] = {**rows[0], "OUTCOME": 0}
        self.assert_code("TOTAL_POSITIVE_EVENTS_INSUFFICIENT", self.readiness, rows)

    def test_primary_audit_unit_n_threshold_is_frozen(self) -> None:
        rows = _full_records()
        rows[0] = {
            **rows[0],
            "AUDIT_UNIT": {"race_ethnicity": "tiny", "sex": "F"},
        }
        self.assert_code("AUDIT_UNIT_N_INSUFFICIENT", self.readiness, rows)

    def test_primary_audit_unit_positive_threshold_is_frozen(self) -> None:
        rows = _audit_spec_records(
            [("low_pos", 200, 19), ("good", 200, 20), ("bulk", 200, 161)]
        )
        self.assert_code("AUDIT_UNIT_POSITIVE_EVENTS_INSUFFICIENT", self.readiness, rows)

    def test_primary_audit_unit_negative_threshold_is_frozen(self) -> None:
        rows = _audit_spec_records([("low_neg", 200, 181), ("good", 200, 20)])
        self.assert_code("AUDIT_UNIT_NEGATIVE_EVENTS_INSUFFICIENT", self.readiness, rows)

    def test_primary_audit_unit_kish_threshold_is_frozen(self) -> None:
        rows = _audit_spec_records(
            [("low_kish", 200, 100), ("good", 200, 100)],
            heavy_first_unit=True,
        )
        self.assert_code("AUDIT_UNIT_KISH_N_INSUFFICIENT", self.readiness, rows)

    def test_synthetic_data_meeting_all_frozen_requirements_passes_and_retains_panel_counts(self) -> None:
        report = self.readiness(_full_records())
        self.assertEqual(report.status, SYNTHETIC_IMPLEMENTATION_READY)
        self.assertEqual(report.decision, GATE14A_DECISION)
        self.assertEqual(report.panels, FIXED_CANDIDATE_PANELS)
        self.assertEqual(report.total_positive_events, 200)
        self.assertEqual(report.total_negative_events, 200)
        self.assertEqual(
            [(item.panel, item.n, item.positive_events, item.negative_events) for item in report.per_panel],
            [(23, 100, 50, 50), (24, 100, 50, 50), (25, 100, 50, 50), (26, 100, 50, 50)],
        )
        self.assertEqual(
            {(item.panel, item.audit_unit) for item in report.primary_audit_units},
            {
                (panel, ("race_ethnicity", "A"))
                for panel in FIXED_CANDIDATE_PANELS
            }
            | {
                (panel, ("sex", "F"))
                for panel in FIXED_CANDIDATE_PANELS
            },
        )
        self.assertEqual(report.loss_weight_semantics, MODEL_DEVELOPMENT_LOSS_WEIGHT_SEMANTICS)
        self.assertNotIn("pooled_population_estimate", report.__dataclass_fields__)
        self.assertNotIn("model_results", report.__dataclass_fields__)
        self.assertNotIn("panel_27_indicators", report.__dataclass_fields__)

    def test_panel27_is_rejected_even_after_two_hundred_positive_events(self) -> None:
        rows = _full_records()
        rows.append({"PANEL": 27, "OUTCOME": 1})
        self.assert_code("PANEL_27_LOCKED", self.readiness, rows)

    def test_raw_cross_panel_collisions_are_not_household_leakage(self) -> None:
        rows = [
            _record(panel, 1, 0, "A", duid=88, partition=partition)
            for panel, partition in zip(
                FIXED_CANDIDATE_PANELS,
                ("train", "validation", "calibration", "train"),
            )
        ]
        validate_namespaced_keys(rows)
        validate_household_partition_integrity(rows)

    def test_loss_weights_require_complete_partition_labels(self) -> None:
        rows = [
            _record(23, 1, 0, "A", weight=1.0),
            _record(23, 2, 0, "A", weight=3.0),
            _record(24, 3, 0, "A", weight=2.0),
            _record(24, 4, 0, "A", weight=2.0),
        ]
        with self.subTest(scenario="complete_single_train_partition_passes"):
            weights = compute_panel_balanced_loss_weights(rows)
            self.assertAlmostEqual(sum(weights), 1.0)
        unlabeled_rows = [
            {key: value for key, value in row.items() if key != "PARTITION"}
            for row in rows
        ]
        with self.subTest(scenario="all_records_missing_partition"):
            self.assert_code(
                "LOSS_PARTITION_NOT_SINGLE",
                compute_panel_balanced_loss_weights,
                unlabeled_rows,
            )
        partially_labeled_rows = [dict(row) for row in rows]
        del partially_labeled_rows[0]["PARTITION"]
        del partially_labeled_rows[3]["PARTITION"]
        with self.subTest(scenario="some_records_missing_partition"):
            self.assert_code(
                "LOSS_PARTITION_NOT_SINGLE",
                compute_panel_balanced_loss_weights,
                partially_labeled_rows,
            )


if __name__ == "__main__":
    unittest.main()
