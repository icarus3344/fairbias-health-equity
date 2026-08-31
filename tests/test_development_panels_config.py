"""Pure configuration checks for the Gate 14A multi-panel design.

These tests intentionally load only JSON and Markdown configuration artifacts.
They never open MEPS data, inspect outcomes, run the pipeline, train a model,
or access Panel 27.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "configs" / "development_panels.json"
STUDY_PATH = ROOT / "configs" / "study.json"
CANONICAL_PATH = ROOT / "configs" / "cohort_and_variables.json"
DECISION_PATH = ROOT / "docs" / "decisions" / "0007-multi-panel-development-design.md"
PLAN_PATH = ROOT / "docs" / "research" / "MULTI_PANEL_DEVELOPMENT_PLAN.md"


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected JSON object in {path}")
    return value


class TestDevelopmentPanelsConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = _load_json(DESIGN_PATH)
        cls.study = _load_json(STUDY_PATH)
        cls.canonical = _load_json(CANONICAL_PATH)

    def test_json_is_valid_and_gate_is_metadata_only(self) -> None:
        self.assertEqual(self.design["gate"], "14A")
        self.assertEqual(
            self.design["decision"],
            "DESIGN_APPROVED_FOR_SYNTHETIC_IMPLEMENTATION",
        )
        scope = self.design["design_scope"]
        for key in (
            "metadata_only",
            "microdata_read_allowed",
            "new_downloads_allowed",
            "outcome_values_read_allowed",
            "real_pipeline_allowed",
            "model_training_allowed",
            "bootstrap_allowed",
            "panel_27_access_allowed",
        ):
            expected = True if key == "metadata_only" else False
            self.assertEqual(scope[key], expected, key)

    def test_fixed_candidates_match_official_metadata_records(self) -> None:
        expected = [
            ("HC-217", 23, 2018, 2019),
            ("HC-225", 24, 2019, 2020),
            ("HC-234", 25, 2020, 2021),
            ("HC-244", 26, 2021, 2022),
        ]
        candidates = self.design["candidate_panels"]
        self.assertEqual(self.design["panel_selection"]["fixed_candidate_sequence"], [x[0] for x in expected])
        self.assertEqual(len(candidates), len(expected))
        for candidate, (puf_id, panel, baseline, follow_up) in zip(candidates, expected):
            self.assertEqual(
                (candidate["puf_id"], candidate["panel_number"], candidate["baseline_year"], candidate["follow_up_year"]),
                (puf_id, panel, baseline, follow_up),
            )
            self.assertEqual(candidate["candidate_order"], panel - 22)
            self.assertEqual(candidate["metadata_status"].split("_")[0], "frozen")
            self.assertEqual(candidate["source_artifacts_downloaded_in_gate_14a"], False)
            self.assertTrue(candidate["official_metadata_url"].startswith("https://meps.ahrq.gov/"))

        self.assertTrue(self.design["panel_selection"]["candidate_set_frozen_before_outcome_access"])
        self.assertTrue(self.design["panel_selection"]["selection_may_not_use_event_counts"])
        self.assertTrue(self.design["panel_selection"]["all_four_candidates_required_for_selected_design"])
        self.assertIn("event count", self.design["panel_selection"]["failure_action"])

        study_candidates = self.study["panels"]["additional_development_candidates"] + [
            self.study["panels"]["development"]
        ]
        study_by_puf = {item["puf_id"]: item for item in study_candidates}
        for candidate in candidates:
            study_record = study_by_puf[candidate["puf_id"]]
            self.assertEqual(candidate["panel_number"], study_record["panel_number"])
            self.assertEqual(
                [candidate["baseline_year"], candidate["follow_up_year"]],
                study_record["years"],
            )

    def test_canonical_estimand_matches_existing_variable_contract(self) -> None:
        estimand = self.design["canonical_estimand"]
        survey = self.canonical["survey_design"]
        self.assertEqual(estimand["survey_design"]["weight"], survey["weight_variable"])
        self.assertEqual(estimand["survey_design"]["strata"], survey["strata_variable"])
        self.assertEqual(estimand["survey_design"]["psu"], survey["psu_variable"])

        eligibility = self.canonical["cohort_eligibility"]
        joined_eligibility = " ".join(estimand["eligibility"])
        for token in (
            eligibility["year_indicator"]["variable"],
            eligibility["five_rounds"]["variable"],
            eligibility["analysis_weight"]["variable"],
            eligibility["baseline_age"]["variable"],
        ):
            self.assertIn(token, joined_eligibility)

        expected_baseline = [x["variable"] for x in eligibility["baseline_continuous_coverage"]["monthly_variables"]]
        expected_follow_up = [x["variable"] for x in self.canonical["target_outcomes"]["primary_any_uninsured_y2"]["monthly_variables"]]
        self.assertEqual(estimand["baseline_month_variables"], expected_baseline)
        self.assertEqual(estimand["follow_up_month_variables"], expected_follow_up)

        predictors = estimand["predictors"]
        all_continuous: list[str] = []
        all_categorical: list[str] = []
        for group in self.canonical["baseline_predictors"].values():
            all_continuous.extend(group["continuous"])
            all_categorical.extend(group["categorical"])
        self.assertEqual(len(all_continuous), predictors["expected_type_breakdown"]["continuous"])
        self.assertEqual(len(all_categorical), predictors["expected_type_breakdown"]["categorical"])
        self.assertEqual(len(all_continuous) + len(all_categorical), predictors["expected_count"])
        self.assertEqual(predictors["expected_count"], 74)
        self.assertEqual(predictors["canonical_source"], "configs/cohort_and_variables.json#/baseline_predictors")
        for flag in ("panel_variable_is_predictor", "survey_design_variables_are_predictors", "protected_variables_are_predictors", "identifiers_are_predictors"):
            self.assertFalse(predictors[flag], flag)

    def test_protected_dimensions_and_namespaces_are_frozen(self) -> None:
        dims = self.design["canonical_estimand"]["protected_audit_dimensions"]
        canonical_dims = self.canonical["audit_dimensions"]
        self.assertEqual(dims["primary"], list(canonical_dims["primary"].keys()))
        self.assertEqual(dims["secondary"], list(canonical_dims["secondary"].keys()))
        self.assertEqual(dims["canonical_source"], "configs/cohort_and_variables.json#/audit_dimensions")

        keys = self.design["identity_and_design_namespaces"]["required_composite_keys"]
        self.assertEqual(keys["person"], ["PANEL", "DUPERSID"])
        self.assertEqual(keys["household_split_group"], ["PANEL", "DUID"])
        self.assertEqual(keys["stratum"], ["PANEL", "VARSTR"])
        self.assertEqual(keys["psu"], ["PANEL", "VARSTR", "VARPSU"])
        self.assertTrue(self.design["identity_and_design_namespaces"]["panel_prefix_required_for_cross_panel_operations"])
        self.assertTrue(self.design["identity_and_design_namespaces"]["no_numeric_offset_or_record_linkage"])

    def test_longwt_pooling_rule_is_explicit_and_not_naive(self) -> None:
        pooling = self.design["longwt_and_pooling"]
        self.assertTrue(pooling["primary_population_estimation"]["use_original_longwt"])
        self.assertEqual(pooling["primary_population_estimation"]["mode"], "panel_stratified")
        self.assertFalse(pooling["primary_population_estimation"]["cross_panel_population_pooling"])
        self.assertFalse(pooling["raw_concatenation"]["allowed"])
        self.assertFalse(pooling["simple_division_by_panel_count"]["allowed"])
        model_loss = pooling["development_model_loss_only"]
        self.assertTrue(model_loss["allowed_after_gate_15"])
        self.assertTrue(model_loss["not_a_population_estimator"])
        self.assertIn("LONGWT_pi /", model_loss["per_panel_weight"])
        self.assertIn("1/K", model_loss["aggregate_objective"])
        self.assertIn("Never replace", model_loss["implementation_requirement"])

    def test_validation_alternatives_and_selected_structure(self) -> None:
        validation = self.design["development_validation"]
        ids = [item["id"] for item in validation["alternatives_considered"]]
        self.assertEqual(
            ids,
            ["panel_stratified_models", "panel_aware_pooling", "per_panel_sensitivity", "leave_one_panel_out"],
        )
        self.assertEqual(validation["selected_structure"], "panel_aware_pooling")
        self.assertTrue(validation["partitioning"]["performed_within_each_panel"])
        self.assertEqual(validation["partitioning"]["group_key"], ["PANEL", "DUID"])
        self.assertEqual(validation["partitioning"]["ratios"], {"train": 0.6, "validation": 0.2, "calibration": 0.2})
        self.assertIn("no data-driven panel selection", validation["selection_rule"].lower())
        self.assertIn("apparent calibration-fit", validation["calibration_boundary"])

    def test_power_is_joint_and_panel_27_is_locked(self) -> None:
        power = self.design["power_and_estimability"]
        self.assertEqual(power["minimum_total_positive_events"], 200)
        for key in ("total_event_count_required", "per_panel_event_counts_required", "subgroup_estimability_required", "minimum_threshold_is_not_sufficient", "no_panel_selection_by_power"):
            self.assertTrue(power[key], key)
        self.assertEqual(
            power["required_power_report"],
            [
                "total eligible positive events across the fixed development set",
                "eligible positive and negative events for each panel",
                "unweighted subgroup n, positive events, negative events, and Kish effective n for every primary audit cell",
                "secondary-cell suppression status using the frozen rules",
            ],
        )
        holdout = self.design["locked_holdout"]
        self.assertEqual((holdout["puf_id"], holdout["panel_number"]), ("HC-252", 27))
        for key in ("outcome_values_allowed", "protected_group_distributions_allowed", "predictor_distributions_allowed", "performance_metrics_allowed", "tuning_or_selection_allowed"):
            self.assertFalse(holdout[key], key)
        self.assertFalse(holdout["included_in_development_candidates"])
        self.assertEqual(holdout["gate_14a_effect"], "no_change_to_lock")

    def test_no_observed_event_count_is_embedded_in_design(self) -> None:
        serialized = json.dumps(self.design, ensure_ascii=False)
        for key in ("observed_event_count", "event_count_observed", "eligible_positive_events_observed", "row_count_observed"):
            self.assertNotIn(key, serialized)
        self.assertNotRegex(serialized, re.compile(r"\b136\b"))

    def test_decision_and_plan_record_the_same_conclusion_and_boundary(self) -> None:
        decision_text = DECISION_PATH.read_text(encoding="utf-8")
        plan_text = PLAN_PATH.read_text(encoding="utf-8")
        for text in (decision_text, plan_text):
            self.assertIn("DESIGN_APPROVED_FOR_SYNTHETIC_IMPLEMENTATION", text)
            self.assertIn("HC-252", text)
            self.assertIn("Panel 27", text)
            self.assertIn("STOP — waiting for Codex review.", text)
            self.assertRegex(text.lower(), re.compile(r"no\s+new\s+(meps|network|data)"))
        self.assertNotIn("STOP_NO_DEFENSIBLE_MULTI_PANEL_DESIGN", decision_text)


if __name__ == "__main__":
    unittest.main()
