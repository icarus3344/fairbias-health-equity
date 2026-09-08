"""Synthetic verification suite for NHIS D8 runner, data isolation, and output protection."""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
import shutil
import unittest
import unittest.mock
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCKED_ACCESS: list[str] = []


def data_guard(event: str, args: tuple) -> None:
    if event == "open" and args and isinstance(args[0], (str, bytes, pathlib.Path)):
        p = pathlib.Path(args[0]).resolve()
        if (
            p.is_relative_to(ROOT / "data")
            or p in [ROOT / "data_COMPAS.csv", ROOT / "data_Credit_Card.csv"]
            or p.suffix == ".parquet"
        ):
            BLOCKED_ACCESS.append(p.name)
            raise RuntimeError(f"DATA_ACCESS_BLOCKED: {p.name}")


sys.addaudithook(data_guard)

import scripts.run_nhis_enhancement_study as cli
from scripts.run_nhis_enhancement_study import build_comparison_dataframe
from fairbias.config import FairBiasConfig
from fairbias.evaluator import FairEvaluator
from fairbias.transform import FairTransform
from fairbias.enhancement_state import changed_dict_hash
from nhis_fairbias.d8_enhancement_runner import (
    D8EnhancementRunner,
    compute_group_fairness_gaps,
    evaluate_representation,
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
        class Dummy:
            def __init__(self, **kwargs): pass
            def run_arm(self, arm): raise RuntimeError("SYNTHETIC_INJECTED_STOP")

        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td) / "existing"
            d.mkdir()
            (d / "execution_manifest.json").write_text('{"status":"failed"}', encoding="utf-8")
            with unittest.mock.patch.object(sys, "argv", ["run_d8", "--arm", "D6_ARM_001", "--output-dir", str(d)]), \
                 unittest.mock.patch.object(cli, "D8EnhancementRunner", Dummy):
                with self.assertRaises(FileExistsError):
                    cli.main()

    def test_real_cli_records_midrun_failure_manifest(self):
        class Dummy:
            def __init__(self, **kwargs): pass
            def run_arm(self, arm): raise RuntimeError("SYNTHETIC_INJECTED_STOP")

        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td) / "new"
            with unittest.mock.patch.object(sys, "argv", ["run_d8", "--arm", "D6_ARM_001", "--output-dir", str(d)]), \
                 unittest.mock.patch.object(cli, "D8EnhancementRunner", Dummy):
                with self.assertRaisesRegex(RuntimeError, "SYNTHETIC_INJECTED_STOP"):
                    cli.main()
            self.assertTrue((d / "execution_manifest.json").exists())
            manifest = json.loads((d / "execution_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "FAILED")
            self.assertIn("SYNTHETIC_INJECTED_STOP", manifest["error"])

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

    def test_r1c_04_downstream_eval_stopped_on_failure(self):
        fake_adapter = FakeNHISStudyAdapter(n_rows=100)
        runner = D8EnhancementRunner(
            adapter=fake_adapter,
            canonical_provider=lambda arm_id: {},
            smoke_test=True,
            random_seed=42,
            run_id="fail_eval_run",
        )

        with unittest.mock.patch(
            "fairbias.enhancement.FairAccuracyEnhancement.enhance_step",
            side_effect=ValueError("Synthetic injected step failure"),
        ):
            res = runner.run_arm("D6_ARM_001")

        post_res = res["conditions"]["posthoc_enhancement"]
        self.assertFalse(post_res["terminal_evaluation_performed"])
        self.assertIsNone(post_res["train"])
        self.assertIsNone(post_res["validation"])
        self.assertIsNone(post_res["test"])
        self.assertIsNone(post_res["terminal_train_max_dphi"])
        self.assertFalse(post_res["fairness_feasible"])
        self.assertEqual(post_res["termination_reason"], "evaluation_failed")

        # build_comparison_dataframe must handle None metrics without throwing
        df_cmp = build_comparison_dataframe({"D6_ARM_001": res})
        post_row = df_cmp[df_cmp["condition"] == "posthoc_enhancement"].iloc[0]
        self.assertTrue(pd.isna(post_row["test_auroc"]))
        self.assertEqual(post_row["termination_reason"], "evaluation_failed")

    def test_r1c_05_joint_cycle_detection(self):
        fake_adapter = FakeNHISStudyAdapter(n_rows=100)
        runner = D8EnhancementRunner(
            adapter=fake_adapter,
            canonical_provider=lambda arm_id: {},
            smoke_test=True,
            random_seed=42,
            run_id="cycle_check_run",
        )

        def cycling_mitigate_step(X, Y, O, nmi_org, changed_dict, current_epsilon, epsilon_threshold, iteration):
            return X, {"num1": {"power": 3.0}}, "SEX_A", "num1"

        def cycling_enhance_step(X_train, Y_train, changed_dict, O_train, epsilon_threshold, current_epsilon, iteration, partition):
            # Revert to empty state {} which is already in committed_joint_states_set
            return X_train, {}, "num1"

        with unittest.mock.patch(
            "fairbias.mitigation.FairBiasMitigation.mitigate_step",
            side_effect=cycling_mitigate_step,
        ), unittest.mock.patch(
            "fairbias.enhancement.FairAccuracyEnhancement.enhance_step",
            side_effect=cycling_enhance_step,
        ):
            res = runner.run_arm("D6_ARM_001")

        joint_res = res["conditions"]["joint_enhancement"]
        self.assertEqual(joint_res["termination_reason"], "cycle_detected")
        events = joint_res["iteration_events"]
        cycle_events = [ev for ev in events if ev.get("cycle_detected") is True]
        self.assertEqual(len(cycle_events), 1)
        self.assertEqual(cycle_events[0]["engine"], "AE")

    def test_r1c_06_geometry_accounting_breakdown_and_replay(self):
        fake_adapter = FakeNHISStudyAdapter(n_rows=100)
        runner = D8EnhancementRunner(
            adapter=fake_adapter,
            canonical_provider=lambda arm_id: {"num1": {"power": 3.0}},
            smoke_test=True,
            random_seed=42,
            run_id="accounting_check_run",
        )
        res = runner.run_arm("D6_ARM_001")
        joint_res = res["conditions"]["joint_enhancement"]
        breakdown = joint_res["geometry_eval_breakdown"]

        self.assertIn("ae_guard_geometry_evals", breakdown)
        self.assertIn("runner_state_refresh_geometry_evals", breakdown)
        self.assertIn("terminal_evaluation_geometry_evals", breakdown)
        self.assertIsNone(breakdown["bm_geometry_evals"])

        expected_sum = (
            breakdown["ae_guard_geometry_evals"]
            + breakdown["runner_state_refresh_geometry_evals"]
            + breakdown["terminal_evaluation_geometry_evals"]
        )
        self.assertEqual(breakdown["total_observable_geometry_evals"], expected_sum)

        # State replay check
        events = joint_res["iteration_events"]
        current_state = {}
        for ev in events:
            if ev.get("trajectory_committed", ev.get("accepted", False)):
                current_state = copy.deepcopy(ev["changed_dict_snapshot"])
                self.assertEqual(ev.get("resulting_state_hash", ev.get("state_hash")), changed_dict_hash(current_state))
        self.assertEqual(changed_dict_hash(current_state), changed_dict_hash(joint_res["terminal_state"]))

        # JSON round-trip
        json_str = json.dumps(res)
        loaded = json.loads(json_str)
        self.assertEqual(loaded["arm_id"], "D6_ARM_001")

    def test_r1c_07_evaluate_representation_contracts(self):
        cfg = FairBiasConfig.compas_default(random_seed=42).resolved()
        ev = FairEvaluator(cfg, ["prot"], "target", ["cat1"], ["num1"])
        tr = FairTransform()

        # 1. Non-numeric column rejected
        df_str = pd.DataFrame({"num1": [1.0, 2.0, 3.0, 4.0], "cat1": ["A", "B", "A", "B"]})
        y = np.array([0, 1, 0, 1])
        o = np.array([0, 1, 0, 1])
        with self.assertRaises(ValueError) as ctx:
            evaluate_representation(
                model=None, scaler=None,
                X_train_raw=df_str, y_train=y,
                X_val_raw=df_str, y_val=y,
                X_test_raw=df_str, y_test=y,
                o_train=o, o_val=o, o_test=o,
                changed_dict={}, evaluator=ev, transformer=tr,
                cate_attrs=["cat1"], num_attrs=["num1"], protected_attr="prot",
            )
        self.assertIn("Non-numeric representation encountered", str(ctx.exception))

        # 2. Model lacking predict_proba rejected
        class NoProbaModel:
            def fit(self, X, y): return self
            def predict(self, X): return np.zeros(len(X))

        df_num = pd.DataFrame({"num1": [1.0, 2.0, 3.0, 4.0], "cat1": [0, 1, 0, 1]})
        with self.assertRaises(ValueError) as ctx:
            evaluate_representation(
                model=NoProbaModel(), scaler=None,
                X_train_raw=df_num, y_train=y,
                X_val_raw=df_num, y_val=y,
                X_test_raw=df_num, y_test=y,
                o_train=o, o_val=o, o_test=o,
                changed_dict={}, evaluator=ev, transformer=tr,
                cate_attrs=["cat1"], num_attrs=["num1"], protected_attr="prot",
            )
        self.assertIn("Model does not support predict_proba", str(ctx.exception))

        # 3. Single-class target rejected
        y_single = np.array([1, 1, 1, 1])
        with self.assertRaises(ValueError) as ctx:
            evaluate_representation(
                model=None, scaler=None,
                X_train_raw=df_num, y_train=y_single,
                X_val_raw=df_num, y_val=y,
                X_test_raw=df_num, y_test=y,
                o_train=o, o_val=o, o_test=o,
                changed_dict={}, evaluator=ev, transformer=tr,
                cate_attrs=["cat1"], num_attrs=["num1"], protected_attr="prot",
            )
        self.assertIn("Single class encountered in target labels", str(ctx.exception))

    def test_r1c_08_cli_directory_exclusivity_and_manifests(self):
        fake_adapter = FakeNHISStudyAdapter(n_rows=50)

        # 1. Existing empty directory rejected
        with tempfile.TemporaryDirectory() as td:
            empty_d = pathlib.Path(td) / "empty_dir"
            empty_d.mkdir()
            with unittest.mock.patch.object(sys, "argv", ["run_d8", "--arm", "D6_ARM_001", "--output-dir", str(empty_d)]):
                with self.assertRaises(FileExistsError):
                    cli.main()

        # 2. Existing non-empty directory rejected
        with tempfile.TemporaryDirectory() as td:
            non_empty_d = pathlib.Path(td) / "non_empty_dir"
            non_empty_d.mkdir()
            (non_empty_d / "prior_data.txt").write_text("prior_content", encoding="utf-8")
            with unittest.mock.patch.object(sys, "argv", ["run_d8", "--arm", "D6_ARM_001", "--output-dir", str(non_empty_d)]):
                with self.assertRaises(FileExistsError):
                    cli.main()

        # 3. Constructor failure writes FAILED manifest
        class ConstructorFailRunner:
            def __init__(self, **kwargs):
                raise RuntimeError("INJECTED_CONSTRUCTOR_FAILURE")

        with tempfile.TemporaryDirectory() as td:
            target_d = pathlib.Path(td) / "run_fail_dir"
            with unittest.mock.patch.object(sys, "argv", ["run_d8", "--arm", "D6_ARM_001", "--output-dir", str(target_d)]), \
                 unittest.mock.patch.object(cli, "D8EnhancementRunner", ConstructorFailRunner):
                with self.assertRaises(RuntimeError):
                    cli.main()
            manifest_p = target_d / "execution_manifest.json"
            self.assertTrue(manifest_p.exists())
            manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "FAILED")
            self.assertEqual(manifest["error_type"], "RuntimeError")
            self.assertIn("INJECTED_CONSTRUCTOR_FAILURE", manifest["error"])
            self.assertIn("Traceback", manifest["traceback"])

        # 4. run_arm failure writes FAILED manifest
        class RunArmFailRunner(D8EnhancementRunner):
            def __init__(self, **kwargs):
                super().__init__(
                    adapter=fake_adapter,
                    canonical_provider=lambda arm_id: {},
                    smoke_test=True,
                    allow_real_data=False,
                )
            def run_arm(self, arm_id):
                raise RuntimeError("INJECTED_RUN_ARM_FAILURE")

        with tempfile.TemporaryDirectory() as td:
            arm_fail_d = pathlib.Path(td) / "arm_fail_dir"
            with unittest.mock.patch.object(sys, "argv", ["run_d8", "--arm", "D6_ARM_001", "--output-dir", str(arm_fail_d)]), \
                 unittest.mock.patch.object(cli, "D8EnhancementRunner", RunArmFailRunner):
                with self.assertRaises(RuntimeError):
                    cli.main()
            manifest_p = arm_fail_d / "execution_manifest.json"
            self.assertTrue(manifest_p.exists())
            manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "FAILED")
            self.assertEqual(manifest["error_type"], "RuntimeError")
            self.assertIn("INJECTED_RUN_ARM_FAILURE", manifest["error"])
            self.assertIn("Traceback", manifest["traceback"])

        # 5. Output-write failure writes FAILED manifest
        class WorkingRunner(D8EnhancementRunner):
            def __init__(self, **kwargs):
                super().__init__(
                    adapter=fake_adapter,
                    canonical_provider=lambda arm_id: {},
                    smoke_test=True,
                    allow_real_data=False,
                    run_id=kwargs.get("run_id", "d8_study"),
                )

        with tempfile.TemporaryDirectory() as td:
            write_fail_d = pathlib.Path(td) / "write_fail_dir"
            orig_open = open
            def failing_open(file, mode="r", *args, **kwargs):
                if "d8_enhancement_study_results.json" in str(file) and "w" in mode:
                    raise IOError("INJECTED_WRITE_FAILURE")
                return orig_open(file, mode, *args, **kwargs)

            with unittest.mock.patch.object(sys, "argv", ["run_d8", "--arm", "D6_ARM_001", "--output-dir", str(write_fail_d)]), \
                 unittest.mock.patch.object(cli, "D8EnhancementRunner", WorkingRunner), \
                 unittest.mock.patch("builtins.open", side_effect=failing_open):
                with self.assertRaises(IOError):
                    cli.main()
            manifest_p = write_fail_d / "execution_manifest.json"
            self.assertTrue(manifest_p.exists())
            manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "FAILED")
            self.assertIn(manifest["error_type"], ["IOError", "OSError"])
            self.assertIn("INJECTED_WRITE_FAILURE", manifest["error"])

        # 6. Full successful run writes COMPLETED manifest
        with tempfile.TemporaryDirectory() as td:
            success_d = pathlib.Path(td) / "success_dir"
            with unittest.mock.patch.object(sys, "argv", ["run_d8", "--arm", "D6_ARM_001", "--smoke-test", "--output-dir", str(success_d)]), \
                 unittest.mock.patch.object(cli, "D8EnhancementRunner", WorkingRunner):
                cli.main()
            manifest_p = success_d / "execution_manifest.json"
            self.assertTrue(manifest_p.exists())
            manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "COMPLETED")
            self.assertIn("output_files", manifest)
            self.assertIn("results_json", manifest["output_files"])
            self.assertIn("comparison_csv", manifest["output_files"])
            self.assertIn("audit_events_jsonl", manifest["output_files"])
            for f_info in manifest["output_files"].values():
                self.assertEqual(len(f_info["sha256"]), 64)

        # 7. Successive runs without --output-dir produce distinct directories
        created_runs = []
        class TrackingRunner(WorkingRunner):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                created_runs.append(self.run_id)

        try:
            with unittest.mock.patch.object(sys, "argv", ["run_d8", "--arm", "D6_ARM_001", "--smoke-test"]), \
                 unittest.mock.patch.object(cli, "D8EnhancementRunner", TrackingRunner):
                cli.main()
                cli.main()
            self.assertEqual(len(created_runs), 2)
            self.assertNotEqual(created_runs[0], created_runs[1])
            self.assertTrue(created_runs[0].startswith("d8_run_"))
            self.assertTrue(created_runs[1].startswith("d8_run_"))
        finally:
            for rid in created_runs:
                dirname = rid.removeprefix("d8_run_")
                p = pathlib.Path("runs/d8_enhancement_study") / dirname
                if p.exists():
                    shutil.rmtree(p)

    def test_r1d_03_joint_trajectory_chain_and_cycle_reconstruction(self):
        """R1D-03: Transition schema, parent/resulting hash chain, cycle detection, and trajectory reconstruction."""
        fake_adapter = FakeNHISStudyAdapter(n_rows=50)
        runner = D8EnhancementRunner(
            adapter=fake_adapter,
            canonical_provider=lambda arm_id: {},
            smoke_test=True,
            random_seed=42,
            run_id="r1d03_trajectory_run",
        )

        state_A = {}
        state_B = {"num1": {"power": 2.0}}
        state_C = {"num1": {"power": 2.0}, "num2": {"power": 0.5}}

        hash_A = changed_dict_hash(state_A)
        hash_B = changed_dict_hash(state_B)
        hash_C = changed_dict_hash(state_C)

        # 1. Unbroken chain: A -> B -> C
        mock_bm = unittest.mock.MagicMock()
        mock_ae = unittest.mock.MagicMock()

        mock_bm.mitigate_step.side_effect = [
            (None, state_B, "prot", "num1"),
            (None, state_C, None, None),
        ]
        mock_ae.enhance_step.side_effect = [
            (None, state_C, "num2"),
            (None, state_C, None),
        ]
        mock_ae.audit_trail = []
        mock_ae.total_model_fits = 0
        mock_ae.total_geometry_evals = 0

        mock_ae_post = unittest.mock.MagicMock()
        mock_ae_post.enhance_step.return_value = (None, {}, None)
        mock_ae_post.audit_trail = []
        mock_ae_post.total_model_fits = 0
        mock_ae_post.total_geometry_evals = 0

        def ae_init_chain(*args, **kwargs):
            if kwargs.get("condition") == "joint_enhancement":
                return mock_ae
            return mock_ae_post

        with unittest.mock.patch("nhis_fairbias.d8_enhancement_runner.FairBiasMitigation", return_value=mock_bm), \
             unittest.mock.patch("nhis_fairbias.d8_enhancement_runner.FairAccuracyEnhancement", side_effect=ae_init_chain), \
             unittest.mock.patch("nhis_fairbias.d8_enhancement_runner.evaluate_representation", return_value={
                 "terminal_evaluation_performed": True,
                 "train": {"auroc": 0.8, "max_dphi": 0.01},
                 "validation": {"auroc": 0.8, "max_dphi": 0.01},
                 "test": {"auroc": 0.8, "max_dphi": 0.01},
             }):
            res_chain = runner.run_arm("D6_ARM_001")

        joint_chain = res_chain["conditions"]["joint_enhancement"]
        self.assertEqual(joint_chain["bm_steps_accepted"], 1)
        self.assertEqual(joint_chain["ae_steps_accepted"], 1)
        committed_events = [e for e in joint_chain["iteration_events"] if e["trajectory_committed"]]
        self.assertEqual(len(committed_events), 2)

        # Verify A -> B -> C chain
        self.assertEqual(committed_events[0]["parent_state_hash"], hash_A)
        self.assertEqual(committed_events[0]["resulting_state_hash"], hash_B)
        self.assertTrue(committed_events[0]["engine_accepted"])
        self.assertTrue(committed_events[0]["trajectory_committed"])
        self.assertFalse(committed_events[0]["cycle_detected"])

        self.assertEqual(committed_events[1]["parent_state_hash"], hash_B)
        self.assertEqual(committed_events[1]["resulting_state_hash"], hash_C)
        self.assertTrue(committed_events[1]["engine_accepted"])
        self.assertTrue(committed_events[1]["trajectory_committed"])
        self.assertFalse(committed_events[1]["cycle_detected"])

        # 2. Cycle detection: A -> B -> A
        mock_bm_cycle = unittest.mock.MagicMock()
        mock_ae_cycle = unittest.mock.MagicMock()

        mock_bm_cycle.mitigate_step.return_value = (None, state_B, "prot", "num1")
        mock_ae_cycle.enhance_step.return_value = (None, state_A, "num1")
        mock_ae_cycle.audit_trail = []
        mock_ae_cycle.total_model_fits = 0
        mock_ae_cycle.total_geometry_evals = 0

        def ae_init_cycle(*args, **kwargs):
            if kwargs.get("condition") == "joint_enhancement":
                return mock_ae_cycle
            return mock_ae_post

        with unittest.mock.patch("nhis_fairbias.d8_enhancement_runner.FairBiasMitigation", return_value=mock_bm_cycle), \
             unittest.mock.patch("nhis_fairbias.d8_enhancement_runner.FairAccuracyEnhancement", side_effect=ae_init_cycle), \
             unittest.mock.patch("nhis_fairbias.d8_enhancement_runner.evaluate_representation", return_value={
                 "terminal_evaluation_performed": True,
                 "train": {"auroc": 0.8, "max_dphi": 0.01},
                 "validation": {"auroc": 0.8, "max_dphi": 0.01},
                 "test": {"auroc": 0.8, "max_dphi": 0.01},
             }):
            res_cycle = runner.run_arm("D6_ARM_001")

        joint_cycle = res_cycle["conditions"]["joint_enhancement"]
        self.assertEqual(joint_cycle["termination_reason"], "cycle_detected")
        self.assertEqual(joint_cycle["bm_steps_accepted"], 1)
        self.assertEqual(joint_cycle["ae_steps_accepted"], 0)

        cycle_events = joint_cycle["iteration_events"]
        # Event 0: BM committed A -> B
        self.assertTrue(cycle_events[0]["trajectory_committed"])
        self.assertFalse(cycle_events[0]["cycle_detected"])
        # Event 1: AE cycle B -> A
        self.assertTrue(cycle_events[1]["engine_accepted"])
        self.assertFalse(cycle_events[1]["trajectory_committed"])
        self.assertTrue(cycle_events[1]["cycle_detected"])
        self.assertEqual(cycle_events[1]["parent_state_hash"], hash_B)
        self.assertEqual(cycle_events[1]["resulting_state_hash"], hash_A)

        # Reconstructed committed trajectory using ONLY trajectory_committed=True events
        reconstructed = [hash_A]
        for e in cycle_events:
            if e["trajectory_committed"]:
                self.assertEqual(e["parent_state_hash"], reconstructed[-1])
                reconstructed.append(e["resulting_state_hash"])
        self.assertEqual(reconstructed, [hash_A, hash_B])

    def test_r1d_p2_02_independent_geometry_eval_counting_oracle(self):
        """R1D-P2-02: Counting evaluator double verifies runner geometry eval breakdown and grand total."""
        fake_adapter = FakeNHISStudyAdapter(n_rows=60)

        class CountingFairEvaluator(FairEvaluator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.actual_calculate_epsilon_calls = 0
                self.call_callers = []

            def calculate_epsilon(self, *args, **kwargs):
                import inspect
                caller = inspect.stack()[1].function
                self.actual_calculate_epsilon_calls += 1
                self.call_callers.append(caller)
                return super().calculate_epsilon(*args, **kwargs)

        runner = D8EnhancementRunner(
            adapter=fake_adapter,
            canonical_provider=lambda arm_id: {},
            smoke_test=True,
            random_seed=42,
            run_id="geom_count_oracle_run",
        )

        counting_evaluator = CountingFairEvaluator(
            FairBiasConfig.compas_default(random_seed=42).resolved(),
            ["protected"], "target", ["cat1"], ["num1"]
        )

        with unittest.mock.patch("nhis_fairbias.d8_enhancement_runner.FairEvaluator", return_value=counting_evaluator):
            res = runner.run_arm("D6_ARM_001")

        # 1. Condition 3 breakdown checks
        post_res = res["conditions"]["posthoc_enhancement"]
        post_breakdown = post_res["geometry_eval_breakdown"]
        self.assertEqual(
            post_breakdown["total_observable_geometry_evals"],
            post_breakdown["ae_guard_geometry_evals"]
            + post_breakdown["runner_state_refresh_geometry_evals"]
            + post_breakdown["terminal_evaluation_geometry_evals"]
        )

        # 2. Condition 4 breakdown checks
        joint_res = res["conditions"]["joint_enhancement"]
        joint_breakdown = joint_res["geometry_eval_breakdown"]
        self.assertEqual(
            joint_breakdown["total_observable_geometry_evals"],
            joint_breakdown["ae_guard_geometry_evals"]
            + joint_breakdown["runner_state_refresh_geometry_evals"]
            + joint_breakdown["terminal_evaluation_geometry_evals"]
        )

        # 3. Separate observable runner calls vs unobservable internal BM calls
        observable_callers = [c for c in counting_evaluator.call_callers if c in ("run_arm", "evaluate_representation", "_is_fairness_acceptable")]
        unobservable_bm_callers = [c for c in counting_evaluator.call_callers if c == "_epsilon_of"]

        # Expected runner observable calls:
        # 1 (init_eps_dict) + 3 (baseline terminal) + 3 (canonical terminal)
        # + Condition 3 observable total + Condition 4 observable total
        expected_observable_total = (
            1
            + 3
            + 3
            + post_breakdown["total_observable_geometry_evals"]
            + joint_breakdown["total_observable_geometry_evals"]
        )
        self.assertEqual(len(observable_callers), expected_observable_total)
        self.assertEqual(
            counting_evaluator.actual_calculate_epsilon_calls,
            expected_observable_total + len(unobservable_bm_callers)
        )


class TestD8R2BaselineReproductionContracts(unittest.TestCase):
    """Rigorous preflight contracts for Gate D8-R2 baseline reproduction mode."""

    def test_baseline_reproduction_only_structural_bypass(self):
        """Verify baseline_reproduction_only executes only conditions 1 & 2 with 0 enhancement calls."""
        fake_adapter = FakeNHISStudyAdapter(n_rows=100)
        canonical_dict = {"num1": {"power": 3.0}}
        runner = D8EnhancementRunner(
            adapter=fake_adapter,
            canonical_provider=lambda arm_id: canonical_dict,
            smoke_test=True,
            random_seed=0,
            allow_real_data=False,
            baseline_reproduction_only=True,
        )

        with unittest.mock.patch("nhis_fairbias.d8_enhancement_runner.FairAccuracyEnhancement") as mock_ae, \
             unittest.mock.patch("nhis_fairbias.d8_enhancement_runner.FairBiasMitigation") as mock_bm:
            res = runner.run_arm("D6_ARM_001")

            # 1. Structural bypass: 0 construction / calls of AE and BM
            self.assertEqual(mock_ae.call_count, 0)
            self.assertEqual(mock_bm.call_count, 0)

        # 2. Conditions in result strictly limited to baseline and canonical_fairbias
        self.assertIn("conditions", res)
        conds = res["conditions"]
        self.assertEqual(set(conds.keys()), {"baseline", "canonical_fairbias"})

        # Baseline executed
        self.assertTrue(conds["baseline"]["terminal_evaluation_performed"])
        self.assertEqual(conds["baseline"]["termination_reason"], "baseline_untransformed")
        self.assertIn("auroc", conds["baseline"]["test"])
        self.assertIn("auprc", conds["baseline"]["test"])
        self.assertIn("accuracy", conds["baseline"]["test"])
        self.assertIn("balanced_accuracy", conds["baseline"]["test"])

        # Canonical FairBias executed
        self.assertTrue(conds["canonical_fairbias"]["terminal_evaluation_performed"])
        self.assertEqual(conds["canonical_fairbias"]["termination_reason"], "d6_canonical_frozen")
        self.assertIn("auroc", conds["canonical_fairbias"]["test"])
        self.assertIn("auprc", conds["canonical_fairbias"]["test"])

        # 3. Candidate model fit count == 0 and candidate audit events == 0
        self.assertEqual(len(runner.audit_events), 0)
        self.assertIn("audit", res)
        self.assertTrue(res["audit"]["baseline_reproduction_only"])
        self.assertFalse(res["audit"]["posthoc_enhancement_called"])
        self.assertFalse(res["audit"]["joint_enhancement_called"])
        self.assertEqual(res["audit"]["candidate_fits_performed"], 0)

    def test_real_data_authorization_guards_and_defaults(self):
        """Verify real-data authorization defaults to OFF and prevents full 4-condition study on real data."""
        # 1. Defaults to OFF
        runner_default = D8EnhancementRunner(allow_real_data=False)
        self.assertFalse(runner_default.allow_real_data)
        self.assertFalse(runner_default.baseline_reproduction_only)

        # 2. Real data cannot be accessed without explicit authorization
        with self.assertRaises(RuntimeError) as ctx:
            _ = runner_default.adapter
        self.assertIn("prohibited", str(ctx.exception).lower())

        # 3. baseline-only mode cannot silently fall through to the full four-condition study on real data
        with self.assertRaises(RuntimeError) as ctx_fallthrough:
            D8EnhancementRunner(allow_real_data=True, baseline_reproduction_only=False)
        self.assertIn("restricted to --baseline-reproduction-only", str(ctx_fallthrough.exception))

        # 4. CLI parser defaults
        cli_args = cli.parse_args.__wrapped__() if hasattr(cli.parse_args, "__wrapped__") else None
        # Test CLI flag enforcement directly
        with unittest.mock.patch("sys.argv", ["cli", "--allow-real-data"]):
            with self.assertRaises(ValueError) as ctx_cli:
                cli.main()
            self.assertIn("strictly prohibited without --baseline-reproduction-only", str(ctx_cli.exception))

    def test_cli_lifecycle_and_directory_exclusivity_in_baseline_only(self):
        """Verify RUNNING -> COMPLETED lifecycle and directory exclusivity in baseline-only mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = pathlib.Path(tmpdir) / "repro_run"

            # 1. Output directory exclusive creation: fails if exists
            run_dir.mkdir()
            with unittest.mock.patch("sys.argv", [
                "cli",
                "--output-dir", str(run_dir),
                "--baseline-reproduction-only",
            ]):
                with self.assertRaises(FileExistsError):
                    cli.main()

            # Remove to test clean run
            run_dir.rmdir()

            # Mock D8EnhancementRunner to verify lifecycle
            mock_runner = unittest.mock.MagicMock()
            mock_runner.audit_events = []
            mock_runner.run_arm.return_value = {
                "arm_id": "D6_ARM_001",
                "protected_attribute": "SEX_A",
                "epsilon_threshold": 0.005,
                "conditions": {
                    "baseline": {
                        "num_transforms": 0,
                        "termination_reason": "baseline_untransformed",
                        "test": {"auroc": 0.75, "auprc": 0.21, "accuracy": 0.92, "predicted_positive_count": 100, "predicted_positive_rate": 0.01, "max_dphi": 0.003, "demographic_parity_difference": 0.001, "equal_opportunity_difference": 0.001},
                        "train": {"max_dphi": 0.004},
                    },
                    "canonical_fairbias": {
                        "num_transforms": 5,
                        "termination_reason": "d6_canonical_frozen",
                        "test": {"auroc": 0.66, "auprc": 0.14, "accuracy": 0.92, "predicted_positive_count": 5, "predicted_positive_rate": 0.001, "max_dphi": 0.001, "demographic_parity_difference": 0.0001, "equal_opportunity_difference": 0.0},
                        "train": {"max_dphi": 0.0004},
                    },
                },
            }

            with unittest.mock.patch("scripts.run_nhis_enhancement_study.D8EnhancementRunner", return_value=mock_runner), \
                 unittest.mock.patch("sys.argv", [
                     "cli",
                     "--output-dir", str(run_dir),
                     "--arm", "D6_ARM_001",
                     "--baseline-reproduction-only",
                 ]):
                cli.main()

            # Verify manifest lifecycle COMPLETED
            manifest_path = run_dir / "execution_manifest.json"
            self.assertTrue(manifest_path.exists())
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertEqual(manifest["status"], "COMPLETED")
            self.assertTrue(manifest["baseline_reproduction_only"])
            self.assertIn("started_at_utc", manifest)
            self.assertIn("completed_at_utc", manifest)
            self.assertIn("results_json", manifest["output_files"])
            self.assertIn("comparison_csv", manifest["output_files"])


class TestD8R2BEvidenceIntegrityContracts(unittest.TestCase):
    """R2B evidence integrity, cohort/schema barrier, and complete temporal metric contracts."""

    def test_r2b_01_observed_n_cannot_be_sourced_from_reference(self):
        """Test 1: Observed N must derive directly from partition DataFrames, not reference variables."""
        import scripts.reproduce_d6_baselines as repro

        # Test with size 80
        adapter_80 = FakeNHISStudyAdapter(n_rows=80)
        runner_80 = D8EnhancementRunner(
            adapter=adapter_80,
            canonical_provider=lambda arm_id: {},
            smoke_test=False,
            baseline_reproduction_only=True,
            random_seed=42,
        )
        res_80 = runner_80.run_arm("D6_ARM_001")
        obs_80 = res_80["observed_cohort"]
        self.assertEqual(obs_80["train_n"], 80)
        self.assertEqual(obs_80["validation_n"], 80)
        self.assertEqual(obs_80["test_n"], 80)

        # Test with size 115
        adapter_115 = FakeNHISStudyAdapter(n_rows=115)
        runner_115 = D8EnhancementRunner(
            adapter=adapter_115,
            canonical_provider=lambda arm_id: {},
            smoke_test=False,
            baseline_reproduction_only=True,
            random_seed=42,
        )
        res_115 = runner_115.run_arm("D6_ARM_001")
        obs_115 = res_115["observed_cohort"]
        self.assertEqual(obs_115["train_n"], 115)
        self.assertEqual(obs_115["validation_n"], 115)
        self.assertEqual(obs_115["test_n"], 115)

        # Confirm compare_cohorts preserves independent observed values vs fixed reference
        tv_prov = {
            "train_n": 27450,
            "train_outcome_positive_count": 1769,
            "validation_n": 29277,
            "validation_outcome_positive_count": 1929,
        }
        te_prov = {
            "test_n": 32350,
            "test_outcome_positive_count": 2563,
        }
        c_rec, c_match = repro.compare_cohorts(obs_80, tv_prov, te_prov)
        self.assertFalse(c_match)
        self.assertEqual(c_rec["train_n_observed"], 80)
        self.assertEqual(c_rec["train_n_reference"], 27450)
        self.assertNotEqual(c_rec["train_n_observed"], c_rec["train_n_reference"])

    def test_r2b_02_wrong_observed_n_fails_barrier(self):
        """Test 2: Perturbed observed N or positive counts must fail the cohort barrier."""
        import scripts.reproduce_d6_baselines as repro

        tv_prov = {
            "train_n": 27450,
            "train_outcome_positive_count": 1769,
            "validation_n": 29277,
            "validation_outcome_positive_count": 1929,
        }
        te_prov = {
            "test_n": 32350,
            "test_outcome_positive_count": 2563,
        }

        exact_cohort = {
            "train_n": 27450,
            "train_positives": 1769,
            "validation_n": 29277,
            "validation_positives": 1929,
            "test_n": 32350,
            "test_positives": 2563,
        }
        _, match = repro.compare_cohorts(exact_cohort, tv_prov, te_prov)
        self.assertTrue(match)

        # Perturb each field individually
        for field in ["train_n", "train_positives", "validation_n", "validation_positives", "test_n", "test_positives"]:
            bad_cohort = copy.deepcopy(exact_cohort)
            bad_cohort[field] += 1
            rec, match = repro.compare_cohorts(bad_cohort, tv_prov, te_prov)
            self.assertFalse(match, f"Cohort barrier should fail when {field} is perturbed")

    def test_r2b_03_wrong_observed_feature_order_fails_barrier(self):
        """Test 3: Wrong observed feature order must fail schema barrier and hash equality."""
        import scripts.reproduce_d6_baselines as repro

        d6_cfg = {
            "expected_predictors": 3,
            "protected_attribute": "prot_a",
            "outcome": "out_y",
            "disability_arm": "full_feature",
            "categorical_features": ["cat1"],
            "numerical_features": ["num1", "num2"],
        }
        tv_prov = {
            "frozen_2022_training_state": {
                "baseline_logistic_regression": {
                    "state": {"feature_order": ["num1", "num2", "cat1"]}
                }
            }
        }
        valid_schema = {
            "feature_order": ["num1", "num2", "cat1"],
            "categorical_features": ["cat1"],
            "numerical_features": ["num1", "num2"],
            "protected_attribute": "prot_a",
            "outcome": "out_y",
            "disability_arm": "full_feature",
            "feature_count": 3,
        }
        rec, match = repro.compare_schemas(valid_schema, d6_cfg, tv_prov)
        self.assertTrue(match)
        self.assertTrue(rec["schema_hashes_match"])

        # Perturb feature order
        bad_order_schema = copy.deepcopy(valid_schema)
        bad_order_schema["feature_order"] = ["num2", "num1", "cat1"]
        bad_rec, bad_match = repro.compare_schemas(bad_order_schema, d6_cfg, tv_prov)
        self.assertFalse(bad_match)
        self.assertFalse(bad_rec["feature_order_match"])
        self.assertFalse(bad_rec["schema_hashes_match"])

    def test_r2b_04_wrong_categorical_numerical_family_fails_barrier(self):
        """Test 4: Wrong categorical or numerical family assignment must fail schema barrier."""
        import scripts.reproduce_d6_baselines as repro

        d6_cfg = {
            "expected_predictors": 3,
            "protected_attribute": "prot_a",
            "outcome": "out_y",
            "disability_arm": "full_feature",
            "categorical_features": ["cat1"],
            "numerical_features": ["num1", "num2"],
        }
        tv_prov = {
            "frozen_2022_training_state": {
                "baseline_logistic_regression": {
                    "state": {"feature_order": ["num1", "num2", "cat1"]}
                }
            }
        }
        # Swap family
        bad_family_schema = {
            "feature_order": ["num1", "num2", "cat1"],
            "categorical_features": ["cat1", "num2"],
            "numerical_features": ["num1"],
            "protected_attribute": "prot_a",
            "outcome": "out_y",
            "disability_arm": "full_feature",
            "feature_count": 3,
        }
        bad_rec, bad_match = repro.compare_schemas(bad_family_schema, d6_cfg, tv_prov)
        self.assertFalse(bad_match)
        self.assertFalse(bad_rec["categorical_features_match"])
        self.assertFalse(bad_rec["numerical_features_match"])
        self.assertFalse(bad_rec["schema_hashes_match"])

    def test_r2b_05_schema_failure_forces_global_barrier_false(self):
        """Test 5: Any schema failure forces global reproduction barrier to False."""
        import scripts.reproduce_d6_baselines as repro

        # Test with mismatched feature count
        d6_cfg = {
            "expected_predictors": 21,
            "protected_attribute": "SEX_A",
            "outcome": "MEDDL12M_A",
            "disability_arm": "full_feature",
            "categorical_features": ["c1"],
            "numerical_features": ["n1"],
        }
        tv_prov = {
            "frozen_2022_training_state": {
                "baseline_logistic_regression": {
                    "state": {"feature_order": ["n1", "c1"]}
                }
            }
        }
        mismatched_schema = {
            "feature_order": ["n1", "c1"],
            "categorical_features": ["c1"],
            "numerical_features": ["n1"],
            "protected_attribute": "SEX_A",
            "outcome": "MEDDL12M_A",
            "disability_arm": "full_feature",
            "feature_count": 2, # expected 21
        }
        rec, match = repro.compare_schemas(mismatched_schema, d6_cfg, tv_prov)
        self.assertFalse(match)
        self.assertFalse(rec["feature_count_match"])

    def test_r2b_06_validation_metric_mismatch_forces_global_barrier_false(self):
        """Test 6: Validation metric mismatch beyond tolerance forces barrier FAIL."""
        import scripts.reproduce_d6_baselines as repro

        base_res = {
            "train": {"N": 27450, "count_predicted_positive": 10, "count_outcome_positive": 100, "accuracy": 0.9, "max_dphi": 0.005},
            "validation": {
                "auroc": 0.779,
                "auprc": 0.200,
                "accuracy": 0.932,
                "predicted_positive_count": 111,
                "selection_rate": 0.00379,
                "max_dphi": 0.00419,
                "demographic_parity_gap": 0.00101,
                "equal_opportunity_gap": 0.00048,
            },
            "test": {
                "auroc": 0.751, "auprc": 0.215, "accuracy": 0.919, "predicted_positive_count": 130,
                "selection_rate": 0.00401, "max_dphi": 0.00392, "demographic_parity_gap": 0.00058, "equal_opportunity_gap": 0.00029
            }
        }
        canon_res = copy.deepcopy(base_res)
        canon_res["train"]["max_dphi"] = 0.0005
        canon_res["validation"]["max_dphi"] = 0.00082
        canon_res["test"]["max_dphi"] = 0.00140
        tv_prov = {"train_n": 27450, "train_outcome_positive_count": 100}
        d6_train_dphi = {"initial_max_dphi": 0.005, "final_max_dphi": 0.0005}

        d6_val_base = {
            "utility": {"auroc": 0.779, "auprc": 0.200, "accuracy": 0.932, "count_predicted_positive": 111, "selection_rate": 0.00379},
            "fairness_gaps": {"demographic_parity_gap": 0.00101, "equal_opportunity_gap": 0.00048},
        }
        d6_val_fb = copy.deepcopy(d6_val_base)
        d6_val_dphi = {"original_validation_max_dphi": 0.00419, "transformed_validation_max_dphi": 0.00082}

        d6_test_base = {
            "utility": {"auroc": 0.751, "auprc": 0.215, "accuracy": 0.919, "count_predicted_positive": 130, "selection_rate": 0.00401},
            "fairness_gaps": {"demographic_parity_gap": 0.00058, "equal_opportunity_gap": 0.00029},
        }
        d6_test_fb = copy.deepcopy(d6_test_base)
        d6_test_dphi = {"original_test_max_dphi": 0.00392, "transformed_test_max_dphi": 0.00140}

        # Clean check passes
        rows_clean = repro.compare_temporal_metrics(
            "D6_ARM_001", "SEX", base_res, canon_res, tv_prov,
            d6_train_dphi, d6_val_base, d6_val_fb, d6_val_dphi,
            d6_test_base, d6_test_fb, d6_test_dphi
        )
        self.assertTrue(all(r["status"] == "PASS" for r in rows_clean))

        # Perturb validation AUROC by 0.01 (beyond 1e-10)
        bad_base_res = copy.deepcopy(base_res)
        bad_base_res["validation"]["auroc"] = 0.769
        rows_bad = repro.compare_temporal_metrics(
            "D6_ARM_001", "SEX", bad_base_res, canon_res, tv_prov,
            d6_train_dphi, d6_val_base, d6_val_fb, d6_val_dphi,
            d6_test_base, d6_test_fb, d6_test_dphi
        )
        val_auroc_row = [r for r in rows_bad if r["condition_row"] == "SEX validation baseline" and r["metric"] == "auroc"][0]
        self.assertEqual(val_auroc_row["status"], "FAIL")
        self.assertFalse(all(r["status"] == "PASS" for r in rows_bad))

    def test_r2b_07_train_dphi_mismatch_forces_global_barrier_false(self):
        """Test 7: Train max dphi mismatch beyond tolerance forces barrier FAIL."""
        import scripts.reproduce_d6_baselines as repro

        base_res = {
            "train": {"N": 27450, "count_predicted_positive": 10, "count_outcome_positive": 100, "accuracy": 0.9, "max_dphi": 0.009}, # Mismatched: expected 0.005
            "validation": {"auroc": 0.7, "auprc": 0.2, "accuracy": 0.9, "predicted_positive_count": 10, "selection_rate": 0.01, "max_dphi": 0.01, "demographic_parity_gap": 0.001, "equal_opportunity_gap": 0.001},
            "test": {"auroc": 0.7, "auprc": 0.2, "accuracy": 0.9, "predicted_positive_count": 10, "selection_rate": 0.01, "max_dphi": 0.01, "demographic_parity_gap": 0.001, "equal_opportunity_gap": 0.001}
        }
        canon_res = copy.deepcopy(base_res)
        canon_res["train"]["max_dphi"] = 0.0005
        tv_prov = {"train_n": 27450, "train_outcome_positive_count": 100}
        d6_train_dphi = {"initial_max_dphi": 0.005, "final_max_dphi": 0.0005}

        d6_val = {"utility": {"auroc": 0.7, "auprc": 0.2, "accuracy": 0.9, "count_predicted_positive": 10, "selection_rate": 0.01}, "fairness_gaps": {"demographic_parity_gap": 0.001, "equal_opportunity_gap": 0.001}}
        d6_val_dphi = {"original_validation_max_dphi": 0.01, "transformed_validation_max_dphi": 0.01}
        d6_test = {"utility": {"auroc": 0.7, "auprc": 0.2, "accuracy": 0.9, "count_predicted_positive": 10, "selection_rate": 0.01}, "fairness_gaps": {"demographic_parity_gap": 0.001, "equal_opportunity_gap": 0.001}}
        d6_test_dphi = {"original_test_max_dphi": 0.01, "transformed_test_max_dphi": 0.01}

        rows = repro.compare_temporal_metrics(
            "D6_ARM_001", "SEX", base_res, canon_res, tv_prov,
            d6_train_dphi, d6_val, d6_val, d6_val_dphi, d6_test, d6_test, d6_test_dphi
        )
        train_dphi_row = [r for r in rows if r["condition_row"] == "SEX train baseline" and r["metric"] == "max_dphi"][0]
        self.assertEqual(train_dphi_row["status"], "FAIL")
        self.assertFalse(all(r["status"] == "PASS" for r in rows))

    def test_r2b_08_conditions_3_4_call_count_remains_zero(self):
        """Test 8: Structural bypass guarantees 0 candidate fits and 0 calls to enhancement engines."""
        fake_adapter = FakeNHISStudyAdapter(n_rows=50)
        runner = D8EnhancementRunner(
            adapter=fake_adapter,
            canonical_provider=lambda arm_id: {},
            baseline_reproduction_only=True,
            random_seed=42,
        )
        with unittest.mock.patch("nhis_fairbias.d8_enhancement_runner.FairAccuracyEnhancement") as mock_ae, \
             unittest.mock.patch("nhis_fairbias.d8_enhancement_runner.FairBiasMitigation") as mock_bm:
            res = runner.run_arm("D6_ARM_001")
            self.assertEqual(mock_ae.call_count, 0)
            self.assertEqual(mock_bm.call_count, 0)
            self.assertEqual(res["audit"]["candidate_fits_performed"], 0)
            self.assertFalse(res["audit"]["posthoc_enhancement_called"])
            self.assertFalse(res["audit"]["joint_enhancement_called"])
            self.assertTrue(res["audit"]["baseline_reproduction_only"])

    def test_r2b_09_no_real_data_access_in_synthetic_preflight(self):
        """Test 9: Process-level audit hook confirms 0 real data access attempts in preflight."""
        # Confirm audit hook intercepted any blocked attempts
        self.assertEqual(len(BLOCKED_ACCESS), 0)
        with self.assertRaises(RuntimeError) as ctx:
            with open(ROOT / "data" / "processed" / "nhis" / "nhis_2022_2024_features.parquet", "rb"):
                pass
        self.assertIn("DATA_ACCESS_BLOCKED", str(ctx.exception))
        # Clear sentinel list after verify
        BLOCKED_ACCESS.clear()


if __name__ == "__main__":
    unittest.main()
