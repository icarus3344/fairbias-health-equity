import builtins
import copy
import io
import os
import pathlib
import socket
import subprocess
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from _fairbias_r1_guard import FairBiasR1Guard, GuardViolationError
from fairbias.enhancement_contracts import EvaluationPartition


def test_isolation_1_data_read_prohibited():
    """ISOLATION-1: Reading forbidden paths is strictly blocked by guard; no skips, synthetic targets only."""
    guard = FairBiasR1Guard.get_instance()
    assert guard is not None, "FairBiasR1Guard must be installed and active in this execution context (fail-closed)"

    # Synthetic forbidden paths (non-code files in code directories, or data directories)
    forbidden_synthetic_paths = [
        guard.repo_root / "data" / "synthetic_forbidden.parquet",
        guard.repo_root / "scripts" / "synthetic_non_runner.csv",
        guard.repo_root / "src" / "synthetic_forbidden_data.txt",
    ]

    for p in forbidden_synthetic_paths:
        with guard.expect_denial(action="file_read", target_pattern=str(p)):
            with pytest.raises((GuardViolationError, PermissionError)):
                with builtins.open(p, "r"):
                    pass


def test_isolation_2_partition_mutation_and_st_perturbation():
    """ISOLATION-2: Partition mutation raises violation; S/T selection perturbation does not affect F fit."""
    df_fit = pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.1, 0.2]}, index=[1, 2])
    y_fit = pd.Series([0, 1], index=[1, 2])
    df_sel1 = pd.DataFrame({"f1": [3.0, 4.0], "f2": [0.3, 0.4]}, index=[3, 4])
    y_sel1 = pd.Series([0, 1], index=[3, 4])

    part1 = EvaluationPartition(fit_X=df_fit, fit_y=y_fit, selection_X=df_sel1, selection_y=y_sel1)
    fit_fp1 = part1.fit_fingerprint()

    # Perturbed selection partition S2
    df_sel2 = pd.DataFrame({"f1": [99.0, 100.0], "f2": [9.9, 10.0]}, index=[5, 6])
    y_sel2 = pd.Series([1, 0], index=[5, 6])

    part2 = EvaluationPartition(fit_X=df_fit, fit_y=y_fit, selection_X=df_sel2, selection_y=y_sel2)
    fit_fp2 = part2.fit_fingerprint()

    # S/T perturbation must NOT affect F (fit) fingerprint
    assert fit_fp1 == fit_fp2, "Perturbation in selection partition must not alter fit partition fingerprint"
    assert part1.selection_fingerprint() != part2.selection_fingerprint(), "Selection fingerprints must reflect perturbation"

    # In-place mutation of fit_X must be caught by verify_not_mutated
    part1.fit_X.iloc[0, 0] = 999.0
    with pytest.raises(ValueError, match="mutated post-construction"):
        part1.verify_not_mutated()


def test_isolation_3_open_network_subprocess_blocked():
    """ISOLATION-3: Guard cannot be bypassed during test execution; network, subprocess, and write routes blocked."""
    guard = FairBiasR1Guard.get_instance()
    assert guard is not None, "FairBiasR1Guard must be installed and active (no skip permitted)"

    # Socket connect blocked
    with guard.expect_denial(action="socket.connect"):
        with pytest.raises((GuardViolationError, PermissionError)):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", 9999))

    # Subprocess execution blocked
    with guard.expect_denial(action="subprocess.Popen"):
        with pytest.raises((GuardViolationError, PermissionError)):
            subprocess.Popen(["echo", "probe"])

    # Modifying files in protected areas blocked
    with guard.expect_denial(action="file_write"):
        with pytest.raises((GuardViolationError, PermissionError)):
            with builtins.open(guard.repo_root / "src" / "malicious_write.py", "w"):
                pass


def test_isolation_4_dir_fd_relative_unlink_blocked():
    """ISOLATION-4: Using non-default dir_fd for relative unlink is fail-closed blocked by the guard."""
    guard = FairBiasR1Guard.get_instance()
    assert guard is not None

    dfd = os.open(str(guard.output_dir), os.O_RDONLY)
    try:
        with guard.expect_denial(action="os.remove", target_pattern=f"dir_fd={dfd}"):
            with pytest.raises((GuardViolationError, PermissionError)):
                os.unlink("victim.txt", dir_fd=dfd)
    finally:
        os.close(dfd)


def test_t_log_fd_alias_and_expect_denial_enforcement():
    """T-LOG: Opening log_fd via io.open blocked; unfulfilled expect_denial raises AssertionError."""
    guard = FairBiasR1Guard.get_instance()
    assert guard is not None

    # Reopening guard log descriptor via io.open is blocked by guard
    with guard.expect_denial(action="log_fd_access", target_pattern=f"fd={guard._log_fd}"):
        with pytest.raises(GuardViolationError, match="Direct access to guard log descriptor"):
            with io.open(guard._log_fd, "wb", closefd=False) as handle:
                handle.write(b"untracked\n")

    # Unfulfilled expected denial declaration must raise AssertionError on exit (not silently discarded)
    with pytest.raises(AssertionError, match="Expected denial contract violated"):
        with guard.expect_denial(action="intentionally_never_attempted", target_pattern="synthetic_target", count=1):
            pass


def test_r4_06_exact_denial_target_matching_rejects_substring():
    """R4-06: Expect denial contract requires exact target/reason/count match; substring does not consume."""
    guard = FairBiasR1Guard.get_instance()
    assert guard is not None

    # Use an isolated test guard instance for fault testing, NEVER modifying the singleton guard of the test run!
    isolated_guard = FairBiasR1Guard(
        output_dir=guard.output_dir / "isolated_test_guard",
        repo_root=guard.repo_root,
    )
    try:
        # 1. Substring target match rejected: expecting "synthetic_expected", action on "synthetic_expected_DIFFERENT"
        # must NOT consume expectation, must raise AssertionError on exit and record unexpected denial in isolated guard.
        with pytest.raises(AssertionError, match="Expected denial contract violated"):
            with isolated_guard.expect_denial(action="synthetic_test_action", target_pattern="synthetic_expected", count=1):
                isolated_guard.log_event(action="synthetic_test_action", target="synthetic_expected_DIFFERENT", decision="DENIED", reason="policy")
        assert len(isolated_guard.unexpected_denials) == 1

        # 2. Reason mismatch rejected: expecting reason "exact_reason", action with "other_reason"
        with pytest.raises(AssertionError, match="Expected denial contract violated"):
            with isolated_guard.expect_denial(action="synthetic_test_action", target_pattern="synthetic_expected", reason_pattern="exact_reason", count=1):
                isolated_guard.log_event(action="synthetic_test_action", target="synthetic_expected", decision="DENIED", reason="other_reason")
        assert len(isolated_guard.unexpected_denials) == 2

        # 3. Count mismatch: expecting 2, but only 1 occurs -> AssertionError
        with pytest.raises(AssertionError, match="Expected denial contract violated"):
            with isolated_guard.expect_denial(action="synthetic_test_action", target_pattern="synthetic_expected", count=2):
                isolated_guard.log_event(action="synthetic_test_action", target="synthetic_expected", decision="DENIED", reason="policy")
    finally:
        isolated_guard.close_log_sink()

    # 4. Exact match succeeds cleanly with remaining=0 on main guard without leaving unexpected denials
    with guard.expect_denial(action="synthetic_test_action", target_pattern="synthetic_expected", count=1):
        guard.log_event(action="synthetic_test_action", target="synthetic_expected", decision="DENIED", reason="policy")


def test_isolation_f_c_and_s_t_perturbation():
    """ISOLATION / R5-04: FairBias core learning/selection on F/C strictly invariant to independent S/T perturbation;
    operational threshold is constant 0.5; evaluation consumes labels, groups, and weights."""
    import copy
    import json
    from fairbias.config import FairBiasConfig
    from fairbias.evaluator import FairEvaluator
    from fairbias.transform import FairTransform
    from fairbias.enhancement import FairAccuracyEnhancement
    from fairbias.application_metrics import compute_application_group_fairness
    from nhis_fairbias.survey import weighted_binary_proportion
    from nhis_fairbias.d8_enhancement_runner import evaluate_representation
    from sklearn.preprocessing import MinMaxScaler

    # F (fit partition: 8 samples, 2 numerical features)
    X_f = pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
        index=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    y_f = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=[1, 2, 3, 4, 5, 6, 7, 8])

    # C (candidate validation partition: 8 samples)
    X_c = pd.DataFrame(
        {"f1": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], "f2": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]},
        index=[9, 10, 11, 12, 13, 14, 15, 16],
    )
    y_c = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=[9, 10, 11, 12, 13, 14, 15, 16])

    # S1 / T1 (baseline evaluation partitions with labels y, protected groups prot, and survey weights w)
    X_s1 = pd.DataFrame({"f1": [1.0, 2.0, 7.0, 8.0], "f2": [0.1, 0.2, 0.7, 0.8]}, index=[17, 18, 19, 20])
    y_s1 = pd.Series([0, 1, 0, 1], index=[17, 18, 19, 20])
    prot_s1 = pd.Series(["a", "a", "b", "b"], index=[17, 18, 19, 20])
    w_s1 = pd.Series([1.0, 2.0, 1.5, 0.5], index=[17, 18, 19, 20])

    X_t1 = pd.DataFrame({"f1": [1.1, 2.1, 7.1, 8.1], "f2": [0.11, 0.21, 0.71, 0.81]}, index=[21, 22, 23, 24])
    y_t1 = pd.Series([1, 0, 1, 0], index=[21, 22, 23, 24])
    prot_t1 = pd.Series(["a", "a", "b", "b"], index=[21, 22, 23, 24])
    w_t1 = pd.Series([1.0, 1.0, 1.0, 1.0], index=[21, 22, 23, 24])

    # S2 / T2 (perturbed independent evaluation partitions: perturbed X, y, prot, and w)
    # y and prot values are truly perturbed, not identical
    X_s2 = pd.DataFrame({"f1": [-999.0, -888.0, -777.0, -666.0], "f2": [-99.0, -88.0, -77.0, -66.0]}, index=[25, 26, 27, 28])
    y_s2 = pd.Series([1, 1, 0, 0], index=[25, 26, 27, 28])
    prot_s2 = pd.Series(["b", "b", "a", "a"], index=[25, 26, 27, 28])
    w_s2 = pd.Series([5.0, 0.1, 3.0, 2.0], index=[25, 26, 27, 28])

    X_t2 = pd.DataFrame({"f1": [-50.0, -60.0, 70.0, 80.0], "f2": [-5.0, -6.0, 7.0, 8.0]}, index=[29, 30, 31, 32])
    y_t2 = pd.Series([0, 0, 1, 1], index=[29, 30, 31, 32])
    prot_t2 = pd.Series(["b", "a", "a", "b"], index=[29, 30, 31, 32])
    w_t2 = pd.Series([2.0, 4.0, 1.0, 3.0], index=[29, 30, 31, 32])

    assert not y_s1.equals(y_s2), "y_s1 and y_s2 must be genuinely perturbed"
    assert not prot_s1.equals(prot_s2), "prot_s1 and prot_s2 must be genuinely perturbed"
    assert not y_t1.equals(y_t2), "y_t1 and y_t2 must be genuinely perturbed"
    assert not prot_t1.equals(prot_t2), "prot_t1 and prot_t2 must be genuinely perturbed"

    part1 = EvaluationPartition(fit_X=X_f, fit_y=y_f, selection_X=X_c, selection_y=y_c, fit_source="F_train", selection_source="C_val")
    part2 = EvaluationPartition(fit_X=X_f, fit_y=y_f, selection_X=X_c, selection_y=y_c, fit_source="F_train", selection_source="C_val")

    # 1. Setup FairBias enhancement engine on F/C with pre-AE fit spy and internal model/scaler spies (R7-03, R8-01)
    cfg = FairBiasConfig(classifier="LR", random_seed=42)
    ev = FairEvaluator(config=cfg, label_O=["prot"], label_Y="target", cate_attrs=[], num_attrs=["f1", "f2"])
    tr = FairTransform()

    ae_spy_records = []
    class SpyFairAccuracyEnhancement(FairAccuracyEnhancement):
        def enhance_step(self, X, y, changed_dict=None, partition=None):
            ae_spy_records.append({
                "X_shape": tuple(X.shape),
                "X_values": np.asarray(X).copy(),
                "y_values": np.asarray(y).copy(),
                "fit_source": partition.fit_source if partition else None,
            })
            return super().enhance_step(X, y, changed_dict=changed_dict, partition=partition)

    import fairbias.enhancement as enhancement_module

    curr_candidate_changed = [{}]
    orig_eval_cand = enhancement_module.evaluate_candidate_utility

    def spy_evaluate_candidate_utility(partition, changed_dict, num_attrs, cate_attrs, transformer, evaluator, *args, **kwargs):
        curr_candidate_changed[0] = copy.deepcopy(changed_dict or {})
        return orig_eval_cand(partition, changed_dict, num_attrs, cate_attrs, transformer, evaluator, *args, **kwargs)

    internal_fit_records = []
    internal_scaler_records = []
    prev_lr_fit = LogisticRegression.fit
    prev_scaler_fit = MinMaxScaler.fit

    def spy_scaler_fit(scaler_self, X, y=None, *args, **kwargs):
        X_arr = np.asarray(X, dtype=float)
        assert X_arr.shape == (8, 2), f"Scaler fit received unexpected shape: {X_arr.shape}"

        cand_dict = curr_candidate_changed[0]
        expected_cand_df = tr.transform_data(X_f, cand_dict, ["f1", "f2"], [])
        expected_raw = expected_cand_df.to_numpy(dtype=float)
        np.testing.assert_allclose(X_arr, expected_raw, err_msg="Scaler fit input X does not match expected transformed F_train")

        res = prev_scaler_fit(scaler_self, X, y=y, *args, **kwargs)

        exp_min = np.min(expected_raw, axis=0)
        exp_max = np.max(expected_raw, axis=0)
        exp_scale = 1.0 / (exp_max - exp_min)

        if not np.allclose(scaler_self.data_min_, exp_min):
            raise AssertionError(
                f"Internal scaler fit input integrity violation: learned data_min_ {scaler_self.data_min_} != expected {exp_min} "
                f"(learned min shift = {scaler_self.data_min_ - exp_min})"
            )
        if not np.allclose(scaler_self.data_max_, exp_max):
            raise AssertionError(
                f"Internal scaler fit input integrity violation: learned data_max_ {scaler_self.data_max_} != expected {exp_max}"
            )
        if not np.allclose(scaler_self.scale_, exp_scale):
            raise AssertionError(
                f"Internal scaler fit input integrity violation: learned scale_ {scaler_self.scale_} != expected {exp_scale}"
            )

        internal_scaler_records.append({
            "scaler": scaler_self,
            "raw_X": X_arr.copy(),
            "data_min": scaler_self.data_min_.copy().tolist(),
            "data_max": scaler_self.data_max_.copy().tolist(),
            "scale": scaler_self.scale_.copy().tolist(),
            "candidate_changed": copy.deepcopy(cand_dict),
            "content_verified": True,
            "verified": True,
        })
        return res

    def spy_lr_fit(model_self, X, y, *args, **kwargs):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        np.testing.assert_array_equal(y_arr, y_f.values, err_msg="Internal model fit received non-F labels")
        assert X_arr.shape == (8, 2), f"Internal model fit received unexpected shape: {X_arr.shape}"

        cand_dict = curr_candidate_changed[0]
        expected_cand_df = tr.transform_data(X_f, cand_dict, ["f1", "f2"], [])
        expected_raw = expected_cand_df.to_numpy(dtype=float)
        exp_min = np.min(expected_raw, axis=0)
        exp_max = np.max(expected_raw, axis=0)
        exp_scale = 1.0 / (exp_max - exp_min)
        expected_scaled = (expected_raw - exp_min) * exp_scale

        if not np.allclose(X_arr, expected_scaled):
            raise AssertionError(
                f"Internal model fit input integrity violation: model fit features do not match expected F_train scaled features! "
                f"Max feature diff = {np.max(np.abs(X_arr - expected_scaled))}"
            )

        res = prev_lr_fit(model_self, X, y, *args, **kwargs)

        # Verify internal fit identity: Logistic Regression MLE satisfies sum(p_i) == sum(y_i) on training data.
        p_pred = model_self.predict_proba(X_arr)[:, 1]
        sum_p = float(np.sum(p_pred))
        sum_y = float(np.sum(y_arr))
        grad_intercept = abs(sum_p - sum_y)
        if grad_intercept >= 0.05:
            raise AssertionError(
                f"Internal model fit input integrity violation: model was fitted with altered/corrupted features! "
                f"Sum of predicted probabilities on untainted features {sum_p:.4f} != sum(y) {sum_y:.4f} "
                f"(gradient deviation {grad_intercept:.4f} >= 0.05)"
            )

        internal_fit_records.append({
            "shape": list(X_arr.shape),
            "coef": model_self.coef_.copy().tolist(),
            "intercept": model_self.intercept_.copy().tolist(),
            "grad_intercept": grad_intercept,
            "candidate_changed": copy.deepcopy(cand_dict),
            "content_verified": True,
            "verified": True,
        })
        return res

    ae1 = SpyFairAccuracyEnhancement(ev, tr, "target", [], ["f1", "f2"])
    ae2 = SpyFairAccuracyEnhancement(ev, tr, "target", [], ["f1", "f2"])

    try:
        enhancement_module.evaluate_candidate_utility = spy_evaluate_candidate_utility
        MinMaxScaler.fit = spy_scaler_fit
        LogisticRegression.fit = spy_lr_fit

        # 2. Perform actual FairBias learning/candidate selection on F/C
        status1, changed1, attr1 = ae1.enhance_step(X_f, y_f, changed_dict={}, partition=part1)
        status2, changed2, attr2 = ae2.enhance_step(X_f, y_f, changed_dict={}, partition=part2)
    finally:
        enhancement_module.evaluate_candidate_utility = orig_eval_cand
        MinMaxScaler.fit = prev_scaler_fit
        LogisticRegression.fit = prev_lr_fit

    # Invariance across identical F/C executions
    assert changed1 == changed2
    assert attr1 == attr2

    # Assert pre-AE spy records: AE strictly consumed F data (8 samples), never S or T (R7-03)
    assert len(ae_spy_records) == 2
    for rec in ae_spy_records:
        assert rec["X_shape"] == (8, 2)
        assert rec["fit_source"] == "F_train"
        np.testing.assert_allclose(rec["X_values"], X_f.values)
        np.testing.assert_allclose(rec["y_values"], y_f.values)

    # Assert internal model and scaler fit records: all internal fits strictly verified from F
    assert len(internal_fit_records) >= 26, f"Expected at least 26 internal fits, got {len(internal_fit_records)}"
    assert all(rec["verified"] and rec["content_verified"] for rec in internal_fit_records)
    assert len(internal_scaler_records) >= 26, f"Expected at least 26 internal scaler fits, got {len(internal_scaler_records)}"
    assert all(rec["verified"] and rec["content_verified"] for rec in internal_scaler_records)

    # 3. Model fit on transformed F with spy tracking to verify data identity and provenance
    fit_spy_records = []
    class SpyLogisticRegression(LogisticRegression):
        def fit(self, X, y, sample_weight=None):
            fit_spy_records.append({
                "X_shape": tuple(X.shape),
                "X_values": np.asarray(X).copy(),
                "y_values": np.asarray(y).copy(),
            })
            return super().fit(X, y, sample_weight=sample_weight)

    t_X_f1 = tr.transform_data(X_f, changed1, ["f1", "f2"], [])
    t_X_f2 = tr.transform_data(X_f, changed2, ["f1", "f2"], [])
    clf1 = SpyLogisticRegression(random_state=42).fit(t_X_f1, y_f)
    clf2 = SpyLogisticRegression(random_state=42).fit(t_X_f2, y_f)

    # Verify fit spy records: all fits came strictly from F (8 samples), never S or T
    assert len(fit_spy_records) == 2
    for rec in fit_spy_records:
        assert rec["X_shape"] == (8, 2)
        np.testing.assert_allclose(rec["y_values"], y_f.values)

    frozen_coef = copy.deepcopy(clf1.coef_)
    frozen_intercept = copy.deepcopy(clf1.intercept_)
    np.testing.assert_allclose(clf1.coef_, clf2.coef_)
    np.testing.assert_allclose(clf1.intercept_, clf2.intercept_)

    # 4. Production decision threshold contract (0.5) verified via production evaluate_representation call
    class ControlledProbModel:
        def __init__(self, p_vals):
            self.p_vals = p_vals
            self.classes_ = np.array([0, 1])
        def fit(self, X, y): pass
        def predict_proba(self, X):
            return np.column_stack([1.0 - self.p_vals[:len(X)], self.p_vals[:len(X)]])

    controlled_probs = np.array([0.49, 0.50, 0.51])
    prod_eval_test = evaluate_representation(
        model=ControlledProbModel(controlled_probs),
        scaler=MinMaxScaler(),
        X_train_raw=pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=[1, 2, 3]),
        y_train=np.array([0, 1, 1]),
        X_val_raw=pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=[1, 2, 3]),
        y_val=np.array([0, 1, 1]),
        X_test_raw=pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=[1, 2, 3]),
        y_test=np.array([0, 1, 1]),
        o_train=np.array(["a", "b", "b"]),
        o_val=np.array(["a", "b", "b"]),
        o_test=np.array(["a", "b", "b"]),
        changed_dict={},
        evaluator=FairEvaluator(config=FairBiasConfig(random_seed=42), label_O=["prot"]),
        transformer=FairTransform(),
        cate_attrs=[],
        num_attrs=["f1"],
        protected_attr="prot",
        expected_groups=["a", "b"],
    )
    assert prod_eval_test["terminal_evaluation_performed"] is True
    assert prod_eval_test["validation"]["predicted_positive_count"] == 2
    assert prod_eval_test["test"]["predicted_positive_count"] == 2
    operational_threshold = 0.5

    # 5. Synthetic S evaluation partitions (S1 and S2)
    X_s1 = pd.DataFrame({"f1": [1.0, 2.0, 7.0, 8.0], "f2": [0.1, 0.2, 0.7, 0.8]}, index=[17, 18, 19, 20])
    y_s1 = pd.Series([0, 1, 0, 1], index=[17, 18, 19, 20])
    prot_s1 = pd.Series(["a", "a", "b", "b"], index=[17, 18, 19, 20])
    w_s1 = pd.Series([1.0, 2.0, 1.5, 0.5], index=[17, 18, 19, 20])

    X_s2 = pd.DataFrame({"f1": [-999.0, -888.0, -777.0, -666.0], "f2": [-99.0, -88.0, -77.0, -66.0]}, index=[25, 26, 27, 28])
    y_s2 = pd.Series([1, 1, 0, 0], index=[25, 26, 27, 28])
    prot_s2 = pd.Series(["b", "b", "a", "a"], index=[25, 26, 27, 28])
    w_s2 = pd.Series([5.0, 0.1, 3.0, 2.0], index=[25, 26, 27, 28])

    t_s1 = tr.transform_data(X_s1, changed1, ["f1", "f2"], [])
    p_s1 = clf1.predict_proba(t_s1)[:, 1]
    yhat_s1 = (p_s1 >= operational_threshold).astype(int)
    eval_s1 = compute_application_group_fairness(
        y_true=y_s1.values, y_pred=yhat_s1, protected_vals=prot_s1.values, expected_groups=["a", "b"]
    )
    w_prop_s1 = weighted_binary_proportion(pd.Series(yhat_s1, index=w_s1.index), w_s1, valid_codes=(0, 1), code=1)

    t_s2 = tr.transform_data(X_s2, changed2, ["f1", "f2"], [])
    p_s2 = clf2.predict_proba(t_s2)[:, 1]
    yhat_s2 = (p_s2 >= operational_threshold).astype(int)
    eval_s2 = compute_application_group_fairness(
        y_true=y_s2.values, y_pred=yhat_s2, protected_vals=prot_s2.values, expected_groups=["a", "b"]
    )
    w_prop_s2 = weighted_binary_proportion(pd.Series(yhat_s2, index=w_s2.index), w_s2, valid_codes=(0, 1), code=1)

    # 6. Synthetic T evaluation partitions (T1 and T2)
    X_t1 = pd.DataFrame({"f1": [1.1, 2.1, 7.1, 8.1], "f2": [0.11, 0.21, 0.71, 0.81]}, index=[21, 22, 23, 24])
    y_t1 = pd.Series([1, 0, 1, 0], index=[21, 22, 23, 24])
    prot_t1 = pd.Series(["a", "a", "b", "b"], index=[21, 22, 23, 24])
    w_t1 = pd.Series([1.0, 1.0, 1.0, 1.0], index=[21, 22, 23, 24])

    X_t2 = pd.DataFrame({"f1": [-50.0, -60.0, 70.0, 80.0], "f2": [-5.0, -6.0, 7.0, 8.0]}, index=[29, 30, 31, 32])
    y_t2 = pd.Series([0, 0, 1, 1], index=[29, 30, 31, 32])
    prot_t2 = pd.Series(["b", "a", "a", "b"], index=[29, 30, 31, 32])
    w_t2 = pd.Series([2.0, 4.0, 1.0, 3.0], index=[29, 30, 31, 32])

    t_t1 = tr.transform_data(X_t1, changed1, ["f1", "f2"], [])
    p_t1 = clf1.predict_proba(t_t1)[:, 1]
    yhat_t1 = (p_t1 >= operational_threshold).astype(int)
    eval_t1 = compute_application_group_fairness(
        y_true=y_t1.values, y_pred=yhat_t1, protected_vals=prot_t1.values, expected_groups=["a", "b"]
    )
    w_prop_t1 = weighted_binary_proportion(pd.Series(yhat_t1, index=w_t1.index), w_t1, valid_codes=(0, 1), code=1)

    t_t2 = tr.transform_data(X_t2, changed2, ["f1", "f2"], [])
    p_t2 = clf2.predict_proba(t_t2)[:, 1]
    yhat_t2 = (p_t2 >= operational_threshold).astype(int)
    eval_t2 = compute_application_group_fairness(
        y_true=y_t2.values, y_pred=yhat_t2, protected_vals=prot_t2.values, expected_groups=["a", "b"]
    )
    w_prop_t2 = weighted_binary_proportion(pd.Series(yhat_t2, index=w_t2.index), w_t2, valid_codes=(0, 1), code=1)

    # 7. Verification: Core F/C model parameters remained 100% INVARIANT despite S1/S2/T1/T2 evaluations
    np.testing.assert_allclose(clf1.coef_, frozen_coef)
    np.testing.assert_allclose(clf1.intercept_, frozen_intercept)
    assert part1.fit_source == "F_train"
    assert part1.selection_source == "C_val"

    # S/T evaluations actively reflected their different datasets (metric sensitivity to perturbation)
    assert eval_s1["group_selection_rates"] != eval_s2["group_selection_rates"]
    assert eval_t1["group_selection_rates"] != eval_t2["group_selection_rates"]
    assert w_prop_s1 != w_prop_s2
    assert w_prop_t1 != w_prop_t2

    # 8. Single-variable perturbations on S partition (R7-03)
    # 8a. y-only perturbation on S:
    y_s_yonly = pd.Series([1, 1, 0, 0], index=y_s1.index)
    eval_s_yonly = compute_application_group_fairness(
        y_true=y_s_yonly.values, y_pred=yhat_s1, protected_vals=prot_s1.values, expected_groups=["a", "b"]
    )
    assert eval_s_yonly["eo_estimable"] != eval_s1["eo_estimable"] or eval_s_yonly["equal_opportunity_difference"] != eval_s1["equal_opportunity_difference"], \
        "y-only perturbation must change at least the estimability or value of equal opportunity"
    np.testing.assert_allclose(clf1.coef_, frozen_coef)

    # 8b. A-only (protected attribute) perturbation on S
    prot_s_Aonly = pd.Series(["a", "b", "a", "b"], index=prot_s1.index)
    eval_s_Aonly = compute_application_group_fairness(
        y_true=y_s1.values, y_pred=yhat_s1, protected_vals=prot_s_Aonly.values, expected_groups=["a", "b"]
    )
    assert eval_s_Aonly["demographic_parity_difference"] != eval_s1["demographic_parity_difference"]
    np.testing.assert_allclose(clf1.coef_, frozen_coef)

    # 8c. w-only perturbation on S with hand-calculated analytical benchmark (R7-03)
    y_hand = pd.Series([1, 0], index=[1, 2])
    w_hand_75 = pd.Series([3.0, 1.0], index=[1, 2])
    w_hand_25 = pd.Series([1.0, 3.0], index=[1, 2])
    prop_75 = weighted_binary_proportion(y_hand, w_hand_75, code=1, valid_codes=(0, 1))
    prop_25 = weighted_binary_proportion(y_hand, w_hand_25, code=1, valid_codes=(0, 1))
    assert abs(prop_75 - 0.75) < 1e-12, f"Expected 0.75 for weights [3, 1], got {prop_75}"
    assert abs(prop_25 - 0.25) < 1e-12, f"Expected 0.25 for weights [1, 3], got {prop_25}"

    unweighted_mean = float(np.mean(y_hand.values == 1))
    assert unweighted_mean == 0.5
    assert prop_75 != unweighted_mean and prop_25 != unweighted_mean, "Weighted binary proportion must be sensitive to weights and fail mutant"

    w_s_wonly = pd.Series([0.1, 10.0, 0.2, 5.0], index=w_s1.index)
    w_prop_s_wonly = weighted_binary_proportion(pd.Series(yhat_s1, index=w_s_wonly.index), w_s_wonly, valid_codes=(0, 1), code=1)
    assert w_prop_s_wonly != w_prop_s1
    np.testing.assert_allclose(clf1.coef_, frozen_coef)

    # 8d. X-only perturbation on S
    X_s_Xonly = pd.DataFrame({"f1": [100.0, 200.0, 300.0, 400.0], "f2": [10.0, 20.0, 30.0, 40.0]}, index=X_s1.index)
    t_s_Xonly = tr.transform_data(X_s_Xonly, changed1, ["f1", "f2"], [])
    p_s_Xonly = clf1.predict_proba(t_s_Xonly)[:, 1]
    yhat_s_Xonly = (p_s_Xonly >= operational_threshold).astype(int)
    assert not np.array_equal(yhat_s_Xonly, yhat_s1)
    np.testing.assert_allclose(clf1.coef_, frozen_coef)

    # 9. Single-variable perturbations on T partition (R8-01, Batch 2 item 4)
    # 9a. y-only perturbation on T
    y_t_yonly = pd.Series([0, 0, 1, 1], index=y_t1.index)
    eval_t_yonly = compute_application_group_fairness(
        y_true=y_t_yonly.values, y_pred=yhat_t1, protected_vals=prot_t1.values, expected_groups=["a", "b"]
    )
    assert eval_t_yonly["equal_opportunity_difference"] != eval_t1["equal_opportunity_difference"] or eval_t_yonly["eo_estimable"] != eval_t1["eo_estimable"]
    np.testing.assert_allclose(clf1.coef_, frozen_coef)

    # 9b. A-only perturbation on T
    prot_t_Aonly = pd.Series(["a", "b", "a", "b"], index=prot_t1.index)
    eval_t_Aonly = compute_application_group_fairness(
        y_true=y_t1.values, y_pred=yhat_t1, protected_vals=prot_t_Aonly.values, expected_groups=["a", "b"]
    )
    assert eval_t_Aonly["demographic_parity_difference"] != eval_t1["demographic_parity_difference"]
    np.testing.assert_allclose(clf1.coef_, frozen_coef)

    # 9c. w-only perturbation on T
    w_t_wonly = pd.Series([10.0, 0.1, 5.0, 0.2], index=w_t1.index)
    w_prop_t_wonly = weighted_binary_proportion(pd.Series(yhat_t1, index=w_t_wonly.index), w_t_wonly, valid_codes=(0, 1), code=1)
    assert w_prop_t_wonly != w_prop_t1
    np.testing.assert_allclose(clf1.coef_, frozen_coef)

    # 9d. X-only perturbation on T
    X_t_Xonly = pd.DataFrame({"f1": [50.0, 60.0, 70.0, 80.0], "f2": [5.0, 6.0, 7.0, 8.0]}, index=X_t1.index)
    t_t_Xonly = tr.transform_data(X_t_Xonly, changed1, ["f1", "f2"], [])
    p_t_Xonly = clf1.predict_proba(t_t_Xonly)[:, 1]
    yhat_t_Xonly = (p_t_Xonly >= operational_threshold).astype(int)
    assert not np.array_equal(yhat_t_Xonly, yhat_t1)
    np.testing.assert_allclose(clf1.coef_, frozen_coef)

    # Save isolation trace artifact
    guard = FairBiasR1Guard.get_instance()
    out_dir = guard.output_dir if guard is not None else pathlib.Path("runs")
    trace_path = out_dir / "isolation_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    all_from_f = bool(
        len(internal_fit_records) >= 26
        and len(internal_scaler_records) >= 26
        and all(rec.get("content_verified") for rec in internal_fit_records)
        and all(rec.get("content_verified") for rec in internal_scaler_records)
    )
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump({
            "fit_source": part1.fit_source,
            "selection_source": part1.selection_source,
            "operational_threshold": operational_threshold,
            "production_threshold_verified_via_call": True,
            "fit_calls_count": len(fit_spy_records),
            "ae_spy_calls_count": len(ae_spy_records),
            "internal_fit_calls_count": len(internal_fit_records),
            "internal_scaler_fit_calls_count": len(internal_scaler_records),
            "fit_all_from_f": all_from_f,
            "frozen_model_coef": frozen_coef.tolist(),
            "frozen_model_intercept": frozen_intercept.tolist(),
            "eval_s1_dp": eval_s1["demographic_parity_difference"],
            "eval_s2_dp": eval_s2["demographic_parity_difference"],
            "eval_t1_dp": eval_t1["demographic_parity_difference"],
            "eval_t2_dp": eval_t2["demographic_parity_difference"],
            "weighted_prop_s1": w_prop_s1,
            "weighted_prop_s2": w_prop_s2,
            "weighted_prop_t1": w_prop_t1,
            "weighted_prop_t2": w_prop_t2,
            "core_parameters_invariant": bool(np.allclose(clf1.coef_, frozen_coef)),
            "metric_sensitivity_verified": True,
            "internal_fit_verification_summary": {
                "internal_model_fits_count": len(internal_fit_records),
                "internal_scaler_fits_count": len(internal_scaler_records),
                "all_scaler_min_max_scale_verified": True,
                "all_model_features_y_verified": True,
                "all_candidates_f_derived": True,
            },
        }, f, indent=2)


def test_r5_05_executed_bytes_evidence_integrity_verification(tmp_path):
    """R5-05/R6-01: Controller evidence verification strictly rejects absent, empty, malformed, unrecorded, and mismatched executed bytes evidence."""
    import json
    from run_fairbias_r1_guarded_tests import verify_executed_bytes_evidence

    pre_hashes = {
        "scripts/run_fairbias_r1_guarded_tests.py": {"sha256": "runner_hash", "bytes": 500},
        "tests/synthetic/test_r1a_guard.py": {"sha256": "test_hash", "bytes": 400},
        "src/fairbias/mitigation.py": {"sha256": "aaaa1111", "bytes": 100},
        "src/fairbias/enhancement.py": {"sha256": "bbbb2222", "bytes": 200},
    }

    # 1. Absent evidence file
    absent_p = tmp_path / "non_existent.json"
    ok, mismatches, unrec, err, info = verify_executed_bytes_evidence(absent_p, pre_hashes)
    assert ok is False
    assert "MISSING_EVIDENCE_FILE" in err

    # 2. Empty evidence file
    empty_p = tmp_path / "empty.json"
    empty_p.write_text("", encoding="utf-8")
    ok, mismatches, unrec, err, info = verify_executed_bytes_evidence(empty_p, pre_hashes)
    assert ok is False
    assert "MALFORMED_JSON" in err

    # 3. Malformed JSON syntax
    malformed_p = tmp_path / "malformed.json"
    malformed_p.write_text("{not valid json", encoding="utf-8")
    ok, mismatches, unrec, err, info = verify_executed_bytes_evidence(malformed_p, pre_hashes)
    assert ok is False
    assert "MALFORMED_JSON" in err

    # 4. Wrong structure: JSON is a list, not dict
    list_p = tmp_path / "list.json"
    list_p.write_text("[]", encoding="utf-8")
    ok, mismatches, unrec, err, info = verify_executed_bytes_evidence(list_p, pre_hashes)
    assert ok is False
    assert "MALFORMED_STRUCTURE" in err

    # 5. Missing executed_sources key
    missing_key_p = tmp_path / "missing_key.json"
    missing_key_p.write_text(json.dumps({"some_other_key": {}}), encoding="utf-8")
    ok, mismatches, unrec, err, info = verify_executed_bytes_evidence(missing_key_p, pre_hashes)
    assert ok is False
    assert "missing executed_sources" in err

    # 6. Empty executed_sources dictionary
    empty_dict_p = tmp_path / "empty_dict.json"
    empty_dict_p.write_text(json.dumps({"executed_sources": {}}), encoding="utf-8")
    ok, mismatches, unrec, err, info = verify_executed_bytes_evidence(empty_dict_p, pre_hashes)
    assert ok is False
    assert "EMPTY_EXECUTED_SOURCES" in err

    # 7. Declared count mismatch: declared 999 vs 1 actual
    count_mismatch_p = tmp_path / "count_mismatch.json"
    count_mismatch_data = {
        "executed_module_count": 999,
        "executed_sources": {
            "src/fairbias/mitigation.py": {
                "sha256": "aaaa1111",
                "bytes": 100,
                "loader_source": "fresh_source_loader",
            },
        },
    }
    count_mismatch_p.write_text(json.dumps(count_mismatch_data), encoding="utf-8")
    ok, mismatches, unrec, err, info = verify_executed_bytes_evidence(count_mismatch_p, pre_hashes)
    assert ok is False
    assert "COUNT_MISMATCH" in err

    # 8. Negative bytes: bytes = -1
    neg_bytes_p = tmp_path / "neg_bytes.json"
    neg_bytes_data = {
        "executed_module_count": 1,
        "executed_sources": {
            "src/fairbias/mitigation.py": {
                "sha256": "aaaa1111",
                "bytes": -1,
                "loader_source": "fresh_source_loader",
            },
        },
    }
    neg_bytes_p.write_text(json.dumps(neg_bytes_data), encoding="utf-8")
    ok, mismatches, unrec, err, info = verify_executed_bytes_evidence(neg_bytes_p, pre_hashes)
    assert ok is False
    assert "INVALID_BYTE_COUNT" in err

    # 9. Missing required runner entrypoint
    no_runner_p = tmp_path / "no_runner.json"
    no_runner_data = {
        "executed_module_count": 1,
        "executed_sources": {
            "src/fairbias/mitigation.py": {
                "sha256": "aaaa1111",
                "bytes": 100,
                "loader_source": "fresh_source_loader",
            },
        },
    }
    no_runner_p.write_text(json.dumps(no_runner_data), encoding="utf-8")
    ok, mismatches, unrec, err, info = verify_executed_bytes_evidence(
        no_runner_p, pre_hashes, expected_runner="scripts/run_fairbias_r1_guarded_tests.py"
    )
    assert ok is False
    assert "MISSING_RUNNER_ENTRYPOINT" in err

    # 10. Missing expected test files
    missing_tests_p = tmp_path / "missing_tests.json"
    missing_tests_data = {
        "executed_module_count": 2,
        "executed_sources": {
            "scripts/run_fairbias_r1_guarded_tests.py": {
                "sha256": "runner_hash",
                "bytes": 500,
                "loader_source": "entrypoint_initial_load",
            },
            "src/fairbias/mitigation.py": {
                "sha256": "aaaa1111",
                "bytes": 100,
                "loader_source": "fresh_source_loader",
            },
        },
    }
    missing_tests_p.write_text(json.dumps(missing_tests_data), encoding="utf-8")
    ok, mismatches, unrec, err, info = verify_executed_bytes_evidence(
        missing_tests_p,
        pre_hashes,
        expected_runner="scripts/run_fairbias_r1_guarded_tests.py",
        expected_test_files=["tests/synthetic/test_r1a_guard.py"],
    )
    assert ok is False
    assert "MISSING_REQUIRED_TEST_FILES" in err

    # 11. Unrecorded module present in executed_sources but not in pre_hashes
    unrec_p = tmp_path / "unrecorded.json"
    unrec_data = {
        "executed_module_count": 2,
        "executed_sources": {
            "src/fairbias/mitigation.py": {
                "sha256": "aaaa1111",
                "bytes": 100,
                "loader_source": "fresh_source_loader",
            },
            "src/unexpected_module.py": {
                "sha256": "cccc3333",
                "bytes": 300,
                "loader_source": "fresh_source_loader",
            },
        },
    }
    unrec_p.write_text(json.dumps(unrec_data), encoding="utf-8")
    ok, mismatches, unrec_list, err, info = verify_executed_bytes_evidence(unrec_p, pre_hashes)
    assert ok is False
    assert "src/unexpected_module.py" in unrec_list

    # 12. SHA256 mismatch between executed module and pre_hashes
    mismatch_p = tmp_path / "mismatch.json"
    mismatch_data = {
        "executed_module_count": 1,
        "executed_sources": {
            "src/fairbias/mitigation.py": {
                "sha256": "wrong_hash",
                "bytes": 100,
                "loader_source": "fresh_source_loader",
            },
        },
    }
    mismatch_p.write_text(json.dumps(mismatch_data), encoding="utf-8")
    ok, mismatches, unrec_list, err, info = verify_executed_bytes_evidence(mismatch_p, pre_hashes)
    assert ok is False
    assert "src/fairbias/mitigation.py" in mismatches

    # 13. Invalid/unrecognized loader_source
    bad_loader_p = tmp_path / "bad_loader.json"
    bad_loader_data = {
        "executed_module_count": 1,
        "executed_sources": {
            "src/fairbias/mitigation.py": {
                "sha256": "aaaa1111",
                "bytes": 100,
                "loader_source": "bogus_loader",
            },
        },
    }
    bad_loader_p.write_text(json.dumps(bad_loader_data), encoding="utf-8")
    ok, mismatches, unrec_list, err, info = verify_executed_bytes_evidence(bad_loader_p, pre_hashes)
    assert ok is False
    assert "UNRECOGNIZED_LOADER_SOURCE" in err

    # 14. Legitimate complete evidence matching pre_hashes
    valid_p = tmp_path / "valid.json"
    valid_data = {
        "executed_module_count": 4,
        "executed_sources": {
            "scripts/run_fairbias_r1_guarded_tests.py": {
                "sha256": "runner_hash",
                "bytes": 500,
                "loader_source": "entrypoint_initial_load",
            },
            "tests/synthetic/test_r1a_guard.py": {
                "sha256": "test_hash",
                "bytes": 400,
                "loader_source": "pytest_assertion_rewrite_live",
            },
            "src/fairbias/mitigation.py": {
                "sha256": "aaaa1111",
                "bytes": 100,
                "loader_source": "fresh_source_loader",
            },
            "src/fairbias/enhancement.py": {
                "sha256": "bbbb2222",
                "bytes": 200,
                "loader_source": "fresh_source_loader",
            },
        },
    }
    valid_loaded = {
        "scripts/run_fairbias_r1_guarded_tests.py": {
            "sha256": "runner_hash",
            "bytes": 500,
        },
        "tests/synthetic/test_r1a_guard.py": {
            "sha256": "test_hash",
            "bytes": 400,
        },
        "src/fairbias/mitigation.py": {
            "sha256": "aaaa1111",
            "bytes": 100,
        },
        "src/fairbias/enhancement.py": {
            "sha256": "bbbb2222",
            "bytes": 200,
        },
    }
    valid_p.write_text(json.dumps(valid_data), encoding="utf-8")
    ok, mismatches, unrec_list, err, info = verify_executed_bytes_evidence(
        valid_p,
        pre_hashes,
        expected_runner="scripts/run_fairbias_r1_guarded_tests.py",
        expected_test_files=["tests/synthetic/test_r1a_guard.py"],
        min_required_modules=3,
        actual_loaded_sources=valid_loaded,
    )
    assert ok is True
    assert len(mismatches) == 0
    assert len(unrec_list) == 0
    assert err == ""

    # 15. Loaded without bytecode evidence (R7-02)
    loaded_uncompiled = {
        "scripts/run_fairbias_r1_guarded_tests.py": {"sha256": "runner_hash", "bytes": 500},
        "tests/synthetic/test_r1a_guard.py": {"sha256": "test_hash", "bytes": 400},
        "src/fairbias/mitigation.py": {"sha256": "aaaa1111", "bytes": 100},
        "src/fairbias/uncompiled_extra.py": {"sha256": "extra_hash", "bytes": 50},  # In sys.modules but not compiled!
    }
    ok, mismatches, unrec_list, err, info = verify_executed_bytes_evidence(
        valid_p,
        pre_hashes,
        actual_loaded_sources=loaded_uncompiled,
    )
    assert ok is False
    assert "LOADED_WITHOUT_BYTECODE_EVIDENCE" in err

    # 16. Loaded is None or empty rejected (R7-02, R9-01)
    ok, _, _, err, _ = verify_executed_bytes_evidence(valid_p, pre_hashes, actual_loaded_sources=None)
    assert ok is False
    assert "EMPTY_OR_MISSING_LOADED_SOURCES" in err

    ok, _, _, err, _ = verify_executed_bytes_evidence(valid_p, pre_hashes, actual_loaded_sources={})
    assert ok is False
    assert "EMPTY_OR_MISSING_LOADED_SOURCES" in err

    # 17. Loaded missing sha256 or bytes rejected (R7-02, R9-01)
    missing_fields_loaded = copy.deepcopy(valid_loaded)
    missing_fields_loaded["src/fairbias/enhancement.py"] = {}
    ok, _, _, err, _ = verify_executed_bytes_evidence(valid_p, pre_hashes, actual_loaded_sources=missing_fields_loaded)
    assert ok is False
    assert "LOADED_INTEGRITY_MISMATCH" in err

    # 18. Loaded wrong sha256 or bytes rejected (R7-02, R9-01)
    wrong_hash_loaded = copy.deepcopy(valid_loaded)
    wrong_hash_loaded["src/fairbias/enhancement.py"] = {"sha256": "wrong", "bytes": -1}
    ok, _, _, err, _ = verify_executed_bytes_evidence(valid_p, pre_hashes, actual_loaded_sources=wrong_hash_loaded)
    assert ok is False
    assert "LOADED_INTEGRITY_MISMATCH" in err


def test_r6_01_guard_log_integrity_verification(tmp_path):
    """R6-01 / R7-02 / R9-01: Guard log verification strictly rejects bad JSON, missing keys, count regression, denial mismatches, and closing delta violations."""
    import json
    from run_fairbias_r1_guarded_tests import verify_guard_log_events

    def _make_dummy_guard_metrics(**overrides):
        m = {
            "total_events": 1,
            "denied_events": 0,
            "allowed_events": 1,
            "truncated_events": 0,
            "disk_events_lost": 0,
            "memory_window_dropped": 0,
            "unconsumed_expected_denials_count": 0,
            "denials_in_sentinels": 0,
            "denials_in_tests": 0,
            "unexpected_denials": [],
            "consumed_expected_denials": [],
            "log_healthy": True,
            "log_errors": [],
        }
        m.update(overrides)
        return m

    # 1. Missing log file
    missing_log = tmp_path / "missing_guard.jsonl"
    ok, err, metrics = verify_guard_log_events(missing_log, {})
    assert ok is False
    assert "MISSING_GUARD_LOG_FILE" in err

    # 2. Empty log file
    empty_log = tmp_path / "empty_guard.jsonl"
    empty_log.write_text("", encoding="utf-8")
    ok, err, metrics = verify_guard_log_events(empty_log, {})
    assert ok is False
    assert "EMPTY_GUARD_LOG" in err

    # 3. Bad JSON line
    bad_json_log = tmp_path / "bad_json.jsonl"
    bad_json_log.write_text("not json\n", encoding="utf-8")
    ok, err, metrics = verify_guard_log_events(bad_json_log, {})
    assert ok is False
    assert "MALFORMED_LOG_LINE" in err

    # 4. Missing mandatory keys (e.g. missing 'reason')
    missing_keys_log = tmp_path / "missing_keys.jsonl"
    missing_keys_log.write_text(
        json.dumps({"timestamp": "2026-09-15T00:00:00Z", "action": "open", "target": "foo", "decision": "ALLOWED"}) + "\n",
        encoding="utf-8",
    )
    ok, err, metrics = verify_guard_log_events(missing_keys_log, {})
    assert ok is False
    assert "MISSING_MANDATORY_KEYS" in err

    # 5. Invalid decision value
    invalid_dec_log = tmp_path / "invalid_decision.jsonl"
    invalid_dec_log.write_text(
        json.dumps({"timestamp": "2026-09-15T00:00:00Z", "action": "open", "target": "foo", "decision": "MAYBE", "reason": "test"}) + "\n",
        encoding="utf-8",
    )
    ok, err, metrics = verify_guard_log_events(invalid_dec_log, {})
    assert ok is False
    assert "INVALID_DECISION_VALUE" in err

    # 6. Count regression: 1 line log but summary claims 999 events
    single_ev = {
        "timestamp": "2026-09-15T00:00:00Z",
        "action": "open",
        "target": "foo",
        "decision": "ALLOWED",
        "reason": "whitelisted",
    }
    one_line_log = tmp_path / "one_line.jsonl"
    one_line_log.write_text(json.dumps(single_ev) + "\n", encoding="utf-8")
    summary_regression = {
        "guard_metrics": _make_dummy_guard_metrics(total_events=999),
        "events_at_summary_cutoff": 999,
    }
    ok, err, metrics = verify_guard_log_events(one_line_log, summary_regression)
    assert ok is False
    assert "LOG_COUNT_REGRESSION" in err

    # 7. Denied count mismatch: log has 1 DENIED, summary claims 0
    denied_ev = {
        "timestamp": "2026-09-15T00:00:00Z",
        "action": "open",
        "target": "data_COMPAS.csv",
        "decision": "DENIED",
        "reason": "forbidden",
    }
    denied_mismatch_log = tmp_path / "denied_mismatch.jsonl"
    denied_mismatch_log.write_text(json.dumps(denied_ev) + "\n", encoding="utf-8")
    summary_zero_denied = {
        "guard_metrics": _make_dummy_guard_metrics(total_events=1, denied_events=0),
        "events_at_summary_cutoff": 1,
    }
    ok, err, metrics = verify_guard_log_events(denied_mismatch_log, summary_zero_denied)
    assert ok is False
    assert "DENIED_COUNT_MISMATCH" in err

    # 8. Summary reports unexpected denials
    summary_unexpected = {
        "guard_metrics": _make_dummy_guard_metrics(
            total_events=1,
            denied_events=1,
            unexpected_denials=[{"action": "open", "target": "forbidden.py"}],
            consumed_expected_denials=[{
                "action": "open",
                "target": "data_COMPAS.csv",
                "reason": "forbidden",
                "expected_action": "open",
                "expected_target_pattern": "data_COMPAS.csv",
                "expected_reason_pattern": "forbidden",
                "match_mode": "exact",
            }],
        ),
        "events_at_summary_cutoff": 1,
    }
    ok, err, metrics = verify_guard_log_events(denied_mismatch_log, summary_unexpected)
    assert ok is False
    assert "UNEXPECTED_DENIALS_REPORTED" in err

    # 9. Disk events lost > 0
    summary_lost = {
        "guard_metrics": _make_dummy_guard_metrics(
            total_events=1,
            denied_events=1,
            disk_events_lost=3,
            consumed_expected_denials=[{
                "action": "open",
                "target": "data_COMPAS.csv",
                "reason": "forbidden",
                "expected_action": "open",
                "expected_target_pattern": "data_COMPAS.csv",
                "expected_reason_pattern": "forbidden",
                "match_mode": "exact",
            }],
        ),
        "events_at_summary_cutoff": 1,
    }
    ok, err, metrics = verify_guard_log_events(denied_mismatch_log, summary_lost)
    assert ok is False
    assert "DISK_EVENTS_LOST" in err

    # 10. Legitimate log matching summary cutoff (6 lines in log, cutoff at 4, exactly 2 closing events to test_summary.json)
    valid_log = tmp_path / "valid_guard.jsonl"
    events = [
        {"timestamp": "2026-09-15T00:00:01Z", "action": "open", "target": "a.py", "decision": "ALLOWED", "reason": "ok"},
        {"timestamp": "2026-09-15T00:00:02Z", "action": "open", "target": "b.py", "decision": "ALLOWED", "reason": "ok"},
        {"timestamp": "2026-09-15T00:00:03Z", "action": "open", "target": "data_COMPAS.csv", "decision": "DENIED", "reason": "sentinel"},
        {"timestamp": "2026-09-15T00:00:04Z", "action": "open", "target": "c.py", "decision": "ALLOWED", "reason": "ok"},
        {"timestamp": "2026-09-15T00:00:05Z", "action": "file_write", "target": str(tmp_path / "test_summary.json"), "decision": "ALLOWED", "reason": "Write target is within output_dir"},
        {"timestamp": "2026-09-15T00:00:06Z", "action": "file_write", "target": str(tmp_path / "test_summary.json"), "decision": "ALLOWED", "reason": "Write target is within output_dir"},
    ]
    valid_log.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    valid_summary = {
        "expected_summary_path": str(tmp_path / "test_summary.json"),
        "summary_path": str(tmp_path / "test_summary.json"),
        "guard_metrics": {
            "total_events": 4,
            "denied_events": 1,
            "allowed_events": 3,
            "truncated_events": 0,
            "disk_events_lost": 0,
            "memory_window_dropped": 0,
            "unconsumed_expected_denials_count": 0,
            "denials_in_sentinels": 1,
            "denials_in_tests": 0,
            "unexpected_denials": [],
            "consumed_expected_denials": [
                {
                    "action": "open",
                    "target": "data_COMPAS.csv",
                    "reason": "sentinel",
                    "expected_action": "open",
                    "expected_target_pattern": "data_COMPAS.csv",
                    "expected_reason_pattern": "sentinel",
                    "match_mode": "exact",
                }
            ],
            "log_healthy": True,
            "log_errors": [],
        },
        "events_at_summary_cutoff": 4,
    }
    ok, err, metrics = verify_guard_log_events(valid_log, valid_summary)
    assert ok is True
    assert err == ""
    assert metrics["final_log_events_count"] == 6
    assert metrics["allowed_count"] == 5
    assert metrics["denied_count"] == 1

    # 11. Missing one closing event (delta == 1 != 2) rejected (R7-02, R9-01)
    missing_one_log = tmp_path / "missing_one.jsonl"
    missing_one_log.write_text("\n".join(json.dumps(e) for e in events[:-1]) + "\n", encoding="utf-8")
    ok, err, metrics = verify_guard_log_events(missing_one_log, valid_summary)
    assert ok is False
    assert "MISSING_CLOSING_EVENTS" in err

    # 12. Missing both closing events (delta == 0 != 2) rejected (R7-02, R9-01)
    missing_two_log = tmp_path / "missing_two.jsonl"
    missing_two_log.write_text("\n".join(json.dumps(e) for e in events[:-2]) + "\n", encoding="utf-8")
    ok, err, metrics = verify_guard_log_events(missing_two_log, valid_summary)
    assert ok is False
    assert "MISSING_CLOSING_EVENTS" in err

    # 13. Unrelated closing events rejected (R7-02, R9-01)
    unrelated_events = copy.deepcopy(events)
    unrelated_events[-1] = {"timestamp": "2026-09-15T00:00:06Z", "action": "unrelated", "target": "/unrelated", "decision": "ALLOWED", "reason": "unrelated"}
    unrelated_log = tmp_path / "unrelated.jsonl"
    unrelated_log.write_text("\n".join(json.dumps(e) for e in unrelated_events) + "\n", encoding="utf-8")
    ok, err, metrics = verify_guard_log_events(unrelated_log, valid_summary)
    assert ok is False
    assert "INVALID_CLOSING_EVENT" in err

    # 14. Wrong reason registration rejected (R7-02, R9-01)
    wrong_reason_summary = copy.deepcopy(valid_summary)
    wrong_reason_summary["guard_metrics"]["consumed_expected_denials"][0]["expected_reason_pattern"] = "THIS_REASON_NEVER_OCCURRED"
    ok, err, metrics = verify_guard_log_events(valid_log, wrong_reason_summary)
    assert ok is False
    assert "DENIED_EVENT_REGISTRATION_MISMATCH" in err

    # 15. Missing registration fields rejected (R7-02, R9-01)
    missing_reg_summary = copy.deepcopy(valid_summary)
    missing_reg_summary["guard_metrics"]["consumed_expected_denials"][0].pop("expected_action")
    ok, err, metrics = verify_guard_log_events(valid_log, missing_reg_summary)
    assert ok is False
    assert "MISSING_REGISTRATION_FIELDS" in err

    # 16. Unknown match mode rejected (R7-02, R9-01)
    unknown_mode_summary = copy.deepcopy(valid_summary)
    unknown_mode_summary["guard_metrics"]["consumed_expected_denials"][0]["match_mode"] = "invalid_mode"
    ok, err, metrics = verify_guard_log_events(valid_log, unknown_mode_summary)
    assert ok is False
    assert "DENIED_EVENT_REGISTRATION_MISMATCH" in err

    # 17. Closing same basename but wrong directory rejected (R7-02, R10)
    wrong_dir_events = copy.deepcopy(events)
    for ev in wrong_dir_events[-2:]:
        ev["target"] = "/unrelated/run/test_summary.json"
    wrong_dir_log = tmp_path / "wrong_dir.jsonl"
    wrong_dir_log.write_text("\n".join(json.dumps(e) for e in wrong_dir_events) + "\n", encoding="utf-8")
    ok, err, metrics = verify_guard_log_events(wrong_dir_log, valid_summary)
    assert ok is False
    assert "INVALID_CLOSING_EVENT" in err

    # 18. Closing wrong filename suffix rejected (R7-02, R10)
    wrong_suffix_events = copy.deepcopy(events)
    for ev in wrong_suffix_events[-2:]:
        ev["target"] = "/unrelated/run/not_test_summary.json"
    wrong_suffix_log = tmp_path / "wrong_suffix.jsonl"
    wrong_suffix_log.write_text("\n".join(json.dumps(e) for e in wrong_suffix_events) + "\n", encoding="utf-8")
    ok, err, metrics = verify_guard_log_events(wrong_suffix_log, valid_summary)
    assert ok is False
    assert "INVALID_CLOSING_EVENT" in err







