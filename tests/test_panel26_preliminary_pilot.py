"""Focused tests for the HC-244 / Panel 26 preliminary pilot boundary."""

from __future__ import annotations

import pathlib
import sys
import inspect
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meps_fairness.pilot import (  # noqa: E402
    EXPECTED_ELIGIBLE_RECORD_COUNT,
    EXPECTED_POSITIVE_EVENTS,
    RESULT_LABEL,
    _validate_panel26_stop_condition,
    fixed_threshold_subgroup_audit,
    fit_validation_calibrator_and_freeze_threshold,
    select_model_by_validation,
)
from scripts.run_panel26_preliminary_pilot import _is_baseline_ancestor  # noqa: E402


class TestPanel26PreliminaryPilot(unittest.TestCase):
    def test_baseline_ancestry_accepts_descendant_and_rejects_reverse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo = pathlib.Path(temporary_directory)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            commit_args = [
                "git",
                "-c",
                "user.name=pilot-test",
                "-c",
                "user.email=pilot-test@example.invalid",
                "commit",
                "--allow-empty",
                "-m",
            ]
            subprocess.run([*commit_args, "baseline"], cwd=repo, check=True, capture_output=True)
            baseline = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            subprocess.run([*commit_args, "descendant"], cwd=repo, check=True, capture_output=True)
            descendant = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            self.assertTrue(_is_baseline_ancestor(repo, baseline, descendant))
            self.assertFalse(_is_baseline_ancestor(repo, descendant, baseline))

    def test_panel26_stop_condition_rejects_count_or_event_mismatch(self) -> None:
        valid_y = np.array([1] * EXPECTED_POSITIVE_EVENTS + [0] * (EXPECTED_ELIGIBLE_RECORD_COUNT - EXPECTED_POSITIVE_EVENTS))
        valid_cohort = type(
            "CohortStub",
            (),
            {
                "eligible_record_count": EXPECTED_ELIGIBLE_RECORD_COUNT,
                "y": valid_y,
            },
        )()
        _validate_panel26_stop_condition(valid_cohort)

        invalid_count = type(
            "CohortStub",
            (),
            {
                "eligible_record_count": EXPECTED_ELIGIBLE_RECORD_COUNT - 1,
                "y": valid_y,
            },
        )()
        with self.assertRaisesRegex(ValueError, "eligible records"):
            _validate_panel26_stop_condition(invalid_count)

        invalid_events = type(
            "CohortStub",
            (),
            {
                "eligible_record_count": EXPECTED_ELIGIBLE_RECORD_COUNT,
                "y": np.array([1] * (EXPECTED_POSITIVE_EVENTS - 1) + [0] * (EXPECTED_ELIGIBLE_RECORD_COUNT - EXPECTED_POSITIVE_EVENTS + 1)),
            },
        )()
        with self.assertRaisesRegex(ValueError, "positive events"):
            _validate_panel26_stop_condition(invalid_events)

    def test_validation_only_model_selection_and_tie_break(self) -> None:
        rows = [
            {
                "model_name": "WeightedRandomForestClassifier",
                "validation_weighted_auprc": 0.70,
                "validation_weighted_auroc": 0.90,
                "test_weighted_auprc": 0.99,
            },
            {
                "model_name": "WeightedLogisticClassifier",
                "validation_weighted_auprc": 0.80,
                "validation_weighted_auroc": 0.80,
                "test_weighted_auprc": 0.01,
            },
            {
                "model_name": "WeightedGradientBoostingClassifier",
                "validation_weighted_auprc": 0.80,
                "validation_weighted_auroc": 0.80,
                "test_weighted_auprc": 1.00,
            },
        ]
        selected = select_model_by_validation(rows)
        self.assertEqual(selected["model_name"], "WeightedGradientBoostingClassifier")
        self.assertEqual(selected["validation_selection_rank"], 1)

    def test_calibrator_and_threshold_use_validation_only(self) -> None:
        validation_raw = np.array([0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 0.90, 0.95])
        validation_y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        validation_w = np.ones(8)

        self.assertEqual(
            set(inspect.signature(fit_validation_calibrator_and_freeze_threshold).parameters),
            {
                "validation_raw_probs",
                "validation_y",
                "validation_weights",
                "seed",
                "capacity_fraction",
            },
        )

        _, _, threshold_a = fit_validation_calibrator_and_freeze_threshold(
            validation_raw, validation_y, validation_w
        )
        # The public function has no test argument.  Changing any hypothetical
        # final-test values therefore cannot alter the threshold provenance.
        _, _, threshold_b = fit_validation_calibrator_and_freeze_threshold(
            validation_raw, validation_y, validation_w
        )
        self.assertEqual(threshold_a, threshold_b)

    def test_fixed_threshold_fairness_does_not_recompute_capacity(self) -> None:
        # Four valid groups, each satisfying all suppression thresholds.  The
        # supplied threshold selects only the high-probability records; a
        # test-derived 10% capacity threshold would select a different set.
        n_per_group = 100
        groups = np.repeat([1, 2], n_per_group)
        y = np.tile(np.array([1] * 50 + [0] * 50), 2)
        probs = np.tile(np.array([0.95] * 50 + [0.05] * 50), 2)
        weights = np.ones_like(probs, dtype=float)
        audit = pd.DataFrame({"RACETHX": groups, "SEX": groups})

        result = fixed_threshold_subgroup_audit(
            y,
            probs,
            audit,
            threshold=0.90,
            sample_weight=weights,
        )
        self.assertEqual(result["threshold"], 0.90)
        self.assertFalse(result["threshold_recomputed_on_test"])
        self.assertEqual(result["dimensions"]["RACETHX"]["subgroups"]["1"]["selection_rate"], 0.5)

    def test_suppression_propagates_none_and_not_estimable_endpoint(self) -> None:
        y = np.array([1] * 5 + [0] * 45)
        probs = np.linspace(0.01, 0.99, 50)
        weights = np.ones(50)
        audit = pd.DataFrame({"RACETHX": [1] * 50, "SEX": [1] * 50})

        result = fixed_threshold_subgroup_audit(
            y,
            probs,
            audit,
            threshold=0.5,
            sample_weight=weights,
        )
        subgroup = result["dimensions"]["RACETHX"]["subgroups"]["1"]
        self.assertEqual(subgroup["status"], "Suppressed (Insufficient Sample/Power)")
        self.assertIsNone(subgroup["tpr"])
        self.assertIsNone(subgroup["fpr"])
        self.assertIsNone(result["primary_fairness_max_tpr_gap"])
        self.assertEqual(
            result["primary_fairness_endpoint"],
            "PRIMARY_FAIRNESS_ENDPOINT = NOT_ESTIMABLE_SUPPRESSED",
        )

    def test_result_label_and_no_forbidden_data_path(self) -> None:
        self.assertEqual(RESULT_LABEL, "PRELIMINARY_SINGLE_PANEL_DEVELOPMENT_ONLY")
        module_text = (SRC_ROOT / "meps_fairness" / "pilot.py").read_text(encoding="utf-8").lower()
        script_text = (
            SRC_ROOT.parent / "scripts" / "run_panel26_preliminary_pilot.py"
        ).read_text(encoding="utf-8").lower()
        self.assertNotIn("data/interim/meps/h252", module_text)
        self.assertNotIn("data/interim/meps/h252", script_text)
        self.assertNotIn("h252.dta", module_text)
        self.assertNotIn("h252.dta", script_text)


if __name__ == "__main__":
    unittest.main()
