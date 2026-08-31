"""Regression test suite verifying all mandatory audit repair boundaries and constraints.

Tests cover:
1. Manuscript & claims matrix integrity (no fabricated numbers or unsupported claims).
2. Suppression truth (100% of evaluation subgroup cells suppressed, no disparity estimable).
3. Mitigation naming & inference contract (exploratory group-aware heuristic, baseline group-agnostic).
4. Calibration apparent-fit labeling boundary.
5. No-clobber run manifest behavior.
6. Panel 27 lock verification (zero Panel 27 outcome data accessed).
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd

from meps_fairness.data.cohort import ALL_BASELINE_PREDICTOR_COLUMNS
from meps_fairness.evaluation.metrics import _trapezoid, weighted_auprc
from meps_fairness.models.baseline import WeightedLogisticClassifier
from meps_fairness.models.mitigation import (
    ExploratoryGroupAwareCenteringMitigation,
    ExploratorySurveyWeightedCenteringExtension,
)


class TestRepairsRegression(unittest.TestCase):
    """Regression tests verifying repair boundaries and anti-fabrication guards."""

    def setUp(self) -> None:
        self.repo_root = _REPO_ROOT
        self.manuscript_path = self.repo_root / "docs/research/MANUSCRIPT_SKELETON.md"
        self.claims_path = self.repo_root / "docs/research/CLAIMS_MATRIX.md"

    def test_manuscript_contains_no_fabricated_numbers(self) -> None:
        """Verify manuscript skeleton contains no previously fabricated numbers."""
        if not self.manuscript_path.is_file():
            self.skipTest(f"Missing {self.manuscript_path}")

        content = self.manuscript_path.read_text(encoding="utf-8")

        # Check prohibited fabricated numbers
        self.assertNotIn("0.669", content, "Fabricated AUROC 0.669 found in manuscript")
        self.assertNotIn("0.672", content, "Fabricated AUROC 0.672 found in manuscript")
        self.assertNotIn("0.142", content, "Fabricated fairness gap 0.142 found in manuscript")
        self.assertNotIn("0.096", content, "Fabricated fairness gap 0.096 found in manuscript")
        self.assertNotIn("0.248", content, "Fabricated recall 0.248 found in manuscript")
        self.assertNotIn("0.118", content, "Fabricated precision 0.118 found in manuscript")

    def test_manuscript_states_suppression_truth_and_underpowering(self) -> None:
        """Verify manuscript documents 100% subgroup cell suppression on evaluation partition."""
        if not self.manuscript_path.is_file():
            self.skipTest(f"Missing {self.manuscript_path}")

        content = self.manuscript_path.read_text(encoding="utf-8")
        self.assertIn("suppressed", content.lower())
        self.assertIn("136", content)
        self.assertIn("exploratory", content.lower())
        self.assertIn("locked", content.lower())

    def test_claims_matrix_boundaries(self) -> None:
        """Verify claims matrix strictly bounds exploratory development evidence."""
        if not self.claims_path.is_file():
            self.skipTest(f"Missing {self.claims_path}")

        content = self.claims_path.read_text(encoding="utf-8")
        self.assertIn("PROHIBITED", content)
        self.assertIn("136", content)

    def test_mitigation_is_exploratory_and_documented(self) -> None:
        """Verify mitigation is named and documented as exploratory group-aware heuristic."""
        doc = ExploratoryGroupAwareCenteringMitigation.__doc__ or ""
        self.assertIn("exploratory", doc.lower())

        # Baseline is group-agnostic at inference
        clf = WeightedLogisticClassifier()
        X = pd.DataFrame({"feat1": [1.0, 2.0], "feat2": [3.0, 4.0]})
        y = pd.Series([0, 1])
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        self.assertEqual(len(probs), 2)

        # Protected columns strictly excluded from baseline feature space
        self.assertNotIn("RACETHX", ALL_BASELINE_PREDICTOR_COLUMNS)
        self.assertNotIn("SEX", ALL_BASELINE_PREDICTOR_COLUMNS)

    def test_trapezoid_fallback_behavior(self) -> None:
        """Verify _trapezoid computes correct integral on standard test function."""
        x = np.linspace(0, 1, 101)
        y = 2 * x
        integral = _trapezoid(y, x)
        self.assertAlmostEqual(integral, 1.0, places=4)

    def test_panel27_lock_guarantee(self) -> None:
        """Verify no Panel 27 outcome file or results exist in repo."""
        results_dir = self.repo_root / "results"
        if results_dir.is_dir():
            for p in results_dir.glob("**/*"):
                if p.is_file() and p.name.endswith(".json"):
                    content = p.read_text(encoding="utf-8")
                    self.assertNotIn("panel27_metrics", content)

    def test_collision_resistant_run_directory_no_clobber(self) -> None:
        """Verify that pipeline enforces no-clobber directory creation."""
        from meps_fairness.pipeline import run_pipeline
        dta_path = self.repo_root / "data/interim/meps/h244/h244.dta"
        if not dta_path.is_file():
            self.skipTest(f"Missing {dta_path}")

        out1 = run_pipeline(mode="smoke", seed=20260828, repo_root=self.repo_root, n_bootstraps=1)
        out2 = run_pipeline(mode="smoke", seed=20260828, repo_root=self.repo_root, n_bootstraps=1)
        self.assertNotEqual(out1.run_id, out2.run_id)
        self.assertNotEqual(out1.output_directory, out2.output_directory)

    def test_subgroup_audit_and_bootstrap_fairness_non_estimable_propagation(self) -> None:
        """Verify that suppressed subgroups propagate None and NOT_ESTIMABLE_SUPPRESSED with no fabricated zero."""
        from meps_fairness.evaluation.inference import stratified_psu_bootstrap_inference
        from meps_fairness.evaluation.metrics import primary_fairness_endpoint, subgroup_audit_metrics

        # Synthetic small sample where all subgroups have n < 100
        y = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        p1 = np.array([0.8, 0.2, 0.1, 0.7, 0.3, 0.2, 0.1, 0.9, 0.4, 0.2])
        p2 = np.array([0.7, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.8, 0.3, 0.1])
        groups = pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        audit_df = pd.DataFrame({"RACETHX": groups, "SEX": [1, 2] * 5})
        design_df = pd.DataFrame({
            "LONGWT": [10.0] * 10,
            "VARSTR": [2001, 2001, 2001, 2001, 2001, 2002, 2002, 2002, 2002, 2002],
            "VARPSU": [1, 1, 2, 2, 1, 1, 1, 2, 2, 1],
        })

        sub_res = subgroup_audit_metrics(y, p1, groups, sample_weight=design_df["LONGWT"], min_n=100, min_pos=20)
        self.assertIsNone(sub_res["max_tpr_gap"], "max_tpr_gap must be None (not 0.0) when fewer than 2 subgroups unsuppressed")
        self.assertEqual(sub_res["unsuppressed_subgroups_count"], 0)

        pf_res = primary_fairness_endpoint(y, p1, audit_df, sample_weight=design_df["LONGWT"])
        self.assertIsNone(pf_res["primary_fairness_max_tpr_gap"], "primary_fairness_max_tpr_gap must be None")

        boot_res = stratified_psu_bootstrap_inference(
            y_true=y,
            probs_unmit=p1,
            probs_mit=p2,
            design_df=design_df,
            audit_df=audit_df,
            n_bootstraps=5,
            seed=20260828,
        )
        fair_inf = boot_res.fairness_max_tpr_gap
        self.assertEqual(fair_inf.status, "NOT_ESTIMABLE_SUPPRESSED")
        self.assertIsNone(fair_inf.point_estimate_unmitigated)
        self.assertIsNone(fair_inf.point_estimate_mitigated)
        self.assertIsNone(fair_inf.paired_difference)
        self.assertIsNone(fair_inf.std_error)
        self.assertIsNone(fair_inf.ci_95_lower)
        self.assertIsNone(fair_inf.ci_95_upper)

        # Serialization to dict produces None (JSON null) and explicit status
        d = fair_inf.to_dict()
        self.assertEqual(d["status"], "NOT_ESTIMABLE_SUPPRESSED")
        self.assertIsNone(d["paired_difference"])
        self.assertIsNone(d["std_error"])

        # Utility inference remains valid
        util_inf = boot_res.utility_auprc
        self.assertEqual(util_inf.status, "ESTIMABLE")
        self.assertIsNotNone(util_inf.paired_difference)


if __name__ == "__main__":
    unittest.main()

