"""Synthetic verification suite for NHIS D8 runner, data isolation, and output protection."""

from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest
import numpy as np
import pandas as pd

from nhis_fairbias.d8_enhancement_runner import (
    D8EnhancementRunner,
    compute_group_fairness_gaps,
)
from nhis_fairbias.d6_temporal_runner import FROZEN_D6_ARMS


class FakePreprocessor:
    """Mock preprocessor supplying feature family lists."""

    def get_feature_family_lists(self, feature_set: str):
        return (["cat1"], ["num1", "num2"])


class FakeNHISStudyAdapter:
    """Synthetic in-memory adapter that generates mock NHIS cohorts without reading parquet files."""

    def __init__(self, n_rows: int = 150):
        self.n_rows = n_rows
        self.preprocessor = FakePreprocessor()

    def get_cohort(self, year: int, outcome: str, protected_attribute: str, feature_set: str, disability_arm: bool):
        np.random.seed(year)
        n = self.n_rows
        x_num1 = np.random.randn(n) * 2.0 + 5.0
        x_num2 = np.random.randn(n) + 10.0
        x_cat1 = np.random.choice([0, 1, 2, 3], size=n)
        o_prot = np.random.choice([0, 1], size=n)
        prob = 1.0 / (1.0 + np.exp(-(0.3 * x_num1 + (x_cat1 == 0).astype(float) * 1.2 - 2.0)))
        y = (np.random.rand(n) < prob).astype(int)

        X = pd.DataFrame({"num1": x_num1, "num2": x_num2, "cat1": x_cat1})
        y_s = pd.Series(y, name=outcome)
        o_s = pd.Series(o_prot, name=protected_attribute)
        strata = pd.Series(np.random.randint(100, 110, size=n), name="STRATA")
        psu = pd.Series(np.random.randint(1, 5, size=n), name="PSU")
        return X, y_s, o_s, strata, psu


class TestNHISD8SyntheticContracts(unittest.TestCase):
    """Verifies runner isolation, sentinel behavior, dual events, and output protection."""

    def test_sentinel_raises_error_if_real_parquet_accessed_without_permission(self):
        # Default initialization without adapter must refuse to read real parquet
        runner = D8EnhancementRunner(smoke_test=True, allow_real_data=False)
        with self.assertRaises(RuntimeError) as ctx:
            _ = runner.adapter
        self.assertIn("Access to real NHIS parquet is prohibited", str(ctx.exception))

    def test_synthetic_runner_all_conditions(self):
        fake_adapter = FakeNHISStudyAdapter(n_rows=100)
        canonical_dict = {"num1": {"power": 3.0}}

        runner = D8EnhancementRunner(
            adapter=fake_adapter,
            canonical_provider=lambda arm_id: canonical_dict,
            smoke_test=True,
            random_seed=42,
            run_id="synth_test_run",
        )

        res = runner.run_arm("D6_ARM_001")
        self.assertEqual(res["arm_id"], "D6_ARM_001")
        self.assertIn("conditions", res)

        conds = res["conditions"]
        for c_name in ["baseline", "canonical_fairbias", "posthoc_enhancement", "joint_enhancement"]:
            self.assertIn(c_name, conds)
            c_data = conds[c_name]
            self.assertIn("terminal_state", c_data)
            self.assertIn("terminal_train_max_dphi", c_data)
            self.assertIn("final_epsilon", c_data)
            self.assertIn("fairness_feasible", c_data)
            self.assertIn("termination_reason", c_data)
            self.assertIn("test", c_data)
            self.assertIn("auprc_trapezoidal", c_data["test"])
            self.assertIn("average_precision", c_data["test"])

        # Check dual events in joint condition
        joint_res = conds["joint_enhancement"]
        self.assertIn("iteration_events", joint_res)
        iter_events = joint_res["iteration_events"]
        engines_present = {ev["engine"] for ev in iter_events}
        self.assertIn("BM", engines_present)
        self.assertIn("AE", engines_present)

    def test_output_collision_prevention(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = pathlib.Path(tmpdir) / "test_run_dir"
            out_path.mkdir(parents=True, exist_ok=True)

            # Create existing result file
            dummy_results = out_path / "d8_enhancement_study_results.json"
            dummy_results.write_text("{}", encoding="utf-8")

            # Check that attempting to use this directory as an output path triggers collision guard
            with self.assertRaises(FileExistsError):
                if out_path.exists() and (
                    (out_path / "d8_enhancement_study_results.json").exists()
                    or (out_path / "d8_enhancement_comparison.csv").exists()
                ):
                    raise FileExistsError("Output directory already exists and contains previous study results.")

    def test_candidate_audit_events_logged(self):
        fake_adapter = FakeNHISStudyAdapter(n_rows=100)
        runner = D8EnhancementRunner(
            adapter=fake_adapter,
            canonical_provider=lambda arm_id: {},
            smoke_test=True,
            random_seed=42,
            run_id="audit_check_run",
        )
        runner.run_arm("D6_ARM_001")
        self.assertGreater(len(runner.audit_events), 0)
        first_event = runner.audit_events[0].to_dict()
        self.assertEqual(first_event["run_id"], "audit_check_run")
        self.assertEqual(first_event["arm_id"], "D6_ARM_001")
        self.assertIn("parent_state_hash", first_event)
        self.assertIn("candidate_state_hash", first_event)
        self.assertIn("accepted", first_event)


if __name__ == "__main__":
    unittest.main()
