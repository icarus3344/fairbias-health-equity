"""Non-data regression tests for the MEPS pre-experiment repair gate.

These tests must not load MEPS microdata, fit models, or execute the pipeline.
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = str(REPO_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from meps_fairness.data.cohort import (  # noqa: E402
    ALL_BASELINE_PREDICTOR_COLUMNS,
    BASELINE_INSURANCE_MONTHS,
    FOLLOWUP_INSURANCE_MONTHS,
)
from meps_fairness.pipeline import (  # noqa: E402
    APPARENT_CALIBRATION_EVIDENCE_NATURE,
    MIN_DEVELOPMENT_POSITIVE_EVENTS,
    PANEL27_LOCKED_PENDING_STATUS,
    evaluate_development_power_gate,
    locked_panel27_status,
)


class TestStudyConfigurationReconciliation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.study = json.loads((REPO_ROOT / "configs/study.json").read_text(encoding="utf-8"))
        cls.variables = json.loads(
            (REPO_ROOT / "configs/cohort_and_variables.json").read_text(encoding="utf-8")
        )

    def test_exact_variable_mappings_match_source_constants(self) -> None:
        population = self.study["population_eligibility"]
        self.assertEqual(population["age_variable_name"], "AGEY1X")
        self.assertEqual(
            tuple(population["baseline_insurance_variables"]),
            BASELINE_INSURANCE_MONTHS,
        )
        self.assertEqual(
            tuple(self.study["target_variables"]["primary_outcome"]["exact_monthly_variables"]),
            FOLLOWUP_INSURANCE_MONTHS,
        )
        self.assertEqual(
            tuple(self.study["predictors"]["exact_predictor_list"]),
            ALL_BASELINE_PREDICTOR_COLUMNS,
        )

    def test_predictor_list_matches_canonical_variable_dictionary(self) -> None:
        dictionary_predictors: list[str] = []
        for group in self.variables["baseline_predictors"].values():
            dictionary_predictors.extend(group["continuous"])
            dictionary_predictors.extend(group["categorical"])
        self.assertEqual(
            sorted(dictionary_predictors),
            list(ALL_BASELINE_PREDICTOR_COLUMNS),
        )
        self.assertEqual(len(dictionary_predictors), 74)

    def test_only_honestly_named_implemented_arms_remain(self) -> None:
        arms = self.study["mitigation_arms"]
        self.assertEqual(
            arms,
            [
                "unmitigated_survey_weighted_logistic_regression",
                "exploratory_survey_weighted_group_aware_centering_requires_protected_attribute_at_inference",
            ],
        )
        serialized = json.dumps(self.study).lower()
        self.assertNotIn("paper_informed_reconstruction", serialized)
        self.assertNotIn("survey_weighted_mitigation_extension", serialized)
        self.assertNotIn("primary_comparison", self.study)
        comparison = self.study["implemented_comparison"]
        self.assertIn("not_fairbias_or_tang", comparison["scientific_role"])
        self.assertIn("protected_group_required", comparison["inference_requirement"])

    def test_development_panel_sequence_is_frozen_without_pooling_authority(self) -> None:
        policy = self.study["panels"]["development_panel_policy"]
        self.assertEqual(
            policy["fixed_candidate_sequence"],
            ["HC-217", "HC-225", "HC-234", "HC-244"],
        )
        self.assertEqual(policy["outcome_access_in_this_gate"], "prohibited")
        self.assertTrue(policy["pooling_status"].startswith("not_authorized"))
        candidates = self.study["panels"]["additional_development_candidates"]
        self.assertEqual(
            [(p["puf_id"], p["panel_number"], p["years"]) for p in candidates],
            [
                ("HC-217", 23, [2018, 2019]),
                ("HC-225", 24, [2019, 2020]),
                ("HC-234", 25, [2020, 2021]),
            ],
        )
        power_gate = self.study["development_power_gate"]
        self.assertEqual(
            power_gate["minimum_positive_events"],
            MIN_DEVELOPMENT_POSITIVE_EVENTS,
        )
        self.assertIn("keep_panel_27_locked", power_gate["threshold_met_action"])

    def test_calibration_evidence_is_apparent_fit_only(self) -> None:
        calibration = self.study["calibration"]
        self.assertEqual(calibration["fit_partition"], "calibration")
        self.assertEqual(
            calibration["development_diagnostic_partition"],
            "same_calibration_partition",
        )
        self.assertEqual(
            calibration["development_evidence_nature"],
            APPARENT_CALIBRATION_EVIDENCE_NATURE,
        )

    def test_current_protocol_documents_reject_old_method_arm_names(self) -> None:
        for relative_path in (
            "docs/research/RESEARCH_PROTOCOL.md",
            "docs/research/STATISTICAL_ANALYSIS_PLAN.md",
        ):
            content = (REPO_ROOT / relative_path).read_text(encoding="utf-8").lower()
            self.assertNotIn("paper-informed reconstruction", content)
            self.assertNotIn("survey-weighted mitigation extension", content)
            self.assertIn("group-aware centering", content)
            self.assertIn("not fairbias", content)
            self.assertIn("tang", content)


class TestDataDerivedPowerGate(unittest.TestCase):
    def test_underpowered_status_uses_runtime_values(self) -> None:
        record = evaluate_development_power_gate(7, threshold=11)
        self.assertEqual(record["positive_events_count"], 7)
        self.assertEqual(record["minimum_positive_events"], 11)
        self.assertFalse(record["power_threshold_met"])
        self.assertEqual(
            locked_panel27_status(record),
            "LOCKED_UNDERPOWERED_STOP_CONDITION (development eligible positives = 7 < 11)",
        )

    def test_threshold_met_still_does_not_unlock_panel27(self) -> None:
        for count in (MIN_DEVELOPMENT_POSITIVE_EVENTS, 999):
            record = evaluate_development_power_gate(count)
            self.assertTrue(record["power_threshold_met"])
            self.assertEqual(locked_panel27_status(record), PANEL27_LOCKED_PENDING_STATUS)
            self.assertNotIn("UNLOCKED", locked_panel27_status(record))

    def test_invalid_power_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_development_power_gate(-1)
        with self.assertRaises(ValueError):
            evaluate_development_power_gate(0, threshold=0)

    def test_pipeline_source_contains_no_observed_event_literal(self) -> None:
        source = (REPO_ROOT / "src/meps_fairness/pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("136", source)
        self.assertIn("evaluate_development_power_gate", source)
        self.assertIn("locked_panel27_status", source)


if __name__ == "__main__":
    unittest.main()
