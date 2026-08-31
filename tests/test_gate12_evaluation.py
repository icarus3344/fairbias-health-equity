"""Tests for Gate 12: Formal Panel 26 Evaluation, Gated Holdout Audit & Stop Condition Verification."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from meps_fairness.pipeline import run_pipeline


class TestGate12Evaluation(unittest.TestCase):
    """Unit and integration tests for Gate 12 formal evaluation and holdout lock enforcement."""

    def setUp(self) -> None:
        self.repo_root = pathlib.Path(__file__).resolve().parents[1]
        self.dta_path = self.repo_root / "data/interim/meps/h244/h244.dta"

    def test_holdout_remains_locked_under_underpowering_stop_condition(self) -> None:
        """Verify Panel 27 remains locked when the runtime development count is below threshold."""
        if not self.dta_path.is_file():
            self.skipTest(f"Missing Panel 26 Stata file: {self.dta_path}")

        output = run_pipeline(
            mode="formal",
            seed=20260828,
            repo_root=self.repo_root,
            n_bootstraps=25,
            output_root="runs",
        )

        self.assertIn("LOCKED_UNDERPOWERED_STOP_CONDITION", output.panel27_status)
        self.assertIsNone(output.panel27_metrics)

        manifest = output.pre_unlock_manifest
        self.assertFalse(manifest["holdout_unlock_prerequisites"]["prerequisites_satisfied"])
        self.assertLess(
            manifest["development_panel"]["positive_events_count"],
            manifest["development_panel"]["minimum_positive_events"],
        )
        self.assertFalse(manifest["development_panel"]["power_threshold_met"])

    def test_multi_seed_reproducibility_panel26(self) -> None:
        """Verify formal multi-seed pipeline runs deterministically across seeds."""
        if not self.dta_path.is_file():
            self.skipTest(f"Missing Panel 26 Stata file: {self.dta_path}")

        out_28_a = run_pipeline(mode="smoke", seed=20260828, repo_root=self.repo_root, n_bootstraps=5)
        out_28_b = run_pipeline(mode="smoke", seed=20260828, repo_root=self.repo_root, n_bootstraps=5)

        # Exact numerical match on identical seed
        self.assertEqual(
            out_28_a.pre_unlock_manifest["frozen_pipeline_hashes"]["split_assignment_hash"],
            out_28_b.pre_unlock_manifest["frozen_pipeline_hashes"]["split_assignment_hash"],
        )
        self.assertAlmostEqual(
            out_28_a.panel26_metrics["unmitigated"]["auprc"],
            out_28_b.panel26_metrics["unmitigated"]["auprc"],
            places=6,
        )

    def test_subgroup_suppression_truth_on_panel26(self) -> None:
        """Verify that all race/ethnicity and sex subgroups on the evaluation partition are suppressed and fairness is not estimable."""
        if not self.dta_path.is_file():
            self.skipTest(f"Missing Panel 26 Stata file: {self.dta_path}")

        output = run_pipeline(mode="smoke", seed=20260828, repo_root=self.repo_root, n_bootstraps=5)
        p26 = output.panel26_metrics
        race_audit = p26["unmitigated"]["primary_fairness"]["race_ethnicity"]
        sex_audit = p26["unmitigated"]["primary_fairness"]["sex"]

        self.assertEqual(race_audit["unsuppressed_subgroups_count"], 0)
        self.assertEqual(sex_audit["unsuppressed_subgroups_count"], 0)
        self.assertIsNone(race_audit["max_tpr_gap"])
        self.assertIsNone(sex_audit["max_tpr_gap"])
        self.assertIsNone(p26["unmitigated"]["primary_fairness"]["primary_fairness_max_tpr_gap"])
        self.assertIsNone(p26["mitigated"]["primary_fairness"]["primary_fairness_max_tpr_gap"])

        for _, sub_info in race_audit["subgroups"].items():
            self.assertEqual(sub_info["status"], "Suppressed (Insufficient Sample/Power)")
            self.assertIsNone(sub_info["tpr_at_capacity"])
        for _, sub_info in sex_audit["subgroups"].items():
            self.assertEqual(sub_info["status"], "Suppressed (Insufficient Sample/Power)")
            self.assertIsNone(sub_info["tpr_at_capacity"])

        # Verify bootstrap fairness inference is Not Estimable and propagates without fabricating scalars or CIs
        boot = p26["bootstrap_inference"]
        fair_boot = boot["fairness_max_tpr_gap"]
        self.assertEqual(fair_boot["status"], "NOT_ESTIMABLE_SUPPRESSED")
        self.assertIsNone(fair_boot["unmitigated"])
        self.assertIsNone(fair_boot["mitigated"])
        self.assertIsNone(fair_boot["paired_difference"])
        self.assertIsNone(fair_boot["std_error"])
        self.assertIsNone(fair_boot["ci_95_lower"])
        self.assertIsNone(fair_boot["ci_95_upper"])

        # Verify utility inference remains valid and estimable
        util_boot = boot["utility_auprc"]
        self.assertEqual(util_boot["status"], "ESTIMABLE")
        self.assertIsNotNone(util_boot["unmitigated"])
        self.assertIsNotNone(util_boot["mitigated"])
        self.assertIsNotNone(util_boot["paired_difference"])


    def test_no_clobber_manifest_and_pointer_behavior(self) -> None:
        """Verify immutable unique run directories and mutable root pointer behavior."""
        if not self.dta_path.is_file():
            self.skipTest(f"Missing Panel 26 Stata file: {self.dta_path}")

        out1 = run_pipeline(mode="smoke", seed=20260828, repo_root=self.repo_root, n_bootstraps=2)
        out2 = run_pipeline(mode="smoke", seed=20260828, repo_root=self.repo_root, n_bootstraps=2)

        # Separate directories
        self.assertNotEqual(out1.output_directory, out2.output_directory)
        self.assertTrue((pathlib.Path(out1.output_directory) / "pre_unlock_manifest.json").is_file())
        self.assertTrue((pathlib.Path(out2.output_directory) / "pre_unlock_manifest.json").is_file())

        # Check root pointer
        pointer_file = self.repo_root / "runs/latest_run_manifest_pointer.json"
        self.assertTrue(pointer_file.is_file())
        with open(pointer_file, encoding="utf-8") as f:
            pointer_data = json.load(f)
        self.assertEqual(pointer_data["artifact_nature"], "mutable_latest_run_pointer")
        self.assertEqual(pointer_data["latest_run_id"], out2.run_id)


if __name__ == "__main__":
    unittest.main()
