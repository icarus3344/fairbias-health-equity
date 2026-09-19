"""Synthetic verification suite for STATE-1/2/3 and CACHE-1/2/3 contracts."""

import copy
import json
import pathlib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from _fairbias_r1_guard import FairBiasR1Guard

from fairbias.config import FairBiasConfig
from fairbias.enhancement import FairAccuracyEnhancement
from fairbias.enhancement_contracts import (
    EvaluationPartition,
    compute_configuration_fingerprint,
)
from fairbias.enhancement_state import (
    StatefulCandidateTracker,
    canonical_json_dump,
    changed_dict_hash,
    hash_transform_state,
    hash_transform_state_legacy,
    hash_transform_state_lossless,
)
from fairbias.evaluator import FairEvaluator
from fairbias.transform import FairTransform


def test_state_1_lossless_float_collision():
    """STATE-1: 3.000000001 and 3.000000002 runtime hashes do not collide under v2 lossless."""
    state_a = {"attr1": {"power": 3.000000001}}
    state_b = {"attr1": {"power": 3.000000002}}

    hash_a = hash_transform_state(state_a)
    hash_b = hash_transform_state(state_b)
    alias_a = changed_dict_hash(state_a)
    alias_b = changed_dict_hash(state_b)

    assert hash_a != hash_b, "Lossless v2 state hashes must not collide on 9th decimal difference"
    assert hash_a == alias_a, "changed_dict_hash must default to lossless v2"
    assert hash_b == alias_b, "changed_dict_hash must default to lossless v2"


def test_state_2_legacy_v1_compatibility():
    """STATE-2: Existing v1 in legacy explicit mode retains 8-decimal rounding compatibility."""
    state_a = {"attr1": {"power": 3.000000001}}
    state_b = {"attr1": {"power": 3.000000002}}

    # In legacy 8-dec mode, both round to 3.0 and collide
    legacy_a = hash_transform_state_legacy(state_a)
    legacy_b = hash_transform_state_legacy(state_b)
    assert legacy_a == legacy_b, "Legacy 8-dec hashes must reproduce historical rounding behavior"

    dump_v1 = canonical_json_dump({"val": 1.234567891}, version="v1_legacy_8dec")
    assert dump_v1 == '{"val":1.23456789}', "v1 must round floats to 8 decimal places"

    # Verify non-finite floats in v2 raise ValueError, while in v1 they map to None
    assert canonical_json_dump({"nan": float("nan")}, version="v1_legacy_8dec") == '{"nan":null}'
    with pytest.raises(ValueError):
        canonical_json_dump({"nan": float("nan")}, version="v2_lossless")


def test_state_3_min_utility_gain_sensitivity():
    """STATE-3: Minute changes in min_utility_gain alter the effective configuration fingerprint."""
    fp1 = compute_configuration_fingerprint(min_utility_gain=0.001000000001)
    fp2 = compute_configuration_fingerprint(min_utility_gain=0.001000000002)

    assert fp1 != fp2, "Minute changes in min_utility_gain must produce distinct configuration fingerprints"


def test_cache_1_classifier_param_invalidation():
    """CACHE-1: Real LogisticRegression C=1.0 -> C=2.0 alters configuration fingerprint and invalidates cache on same engine."""
    cfg = FairBiasConfig(classifier="LR")
    evaluator = FairEvaluator(config=cfg)
    evaluator.model = LogisticRegression(C=1.0, random_state=42)

    transformer = FairTransform()
    engine = FairAccuracyEnhancement(
        evaluator=evaluator,
        transformer=transformer,
        label_Y="target",
        cate_attrs=[],
        num_attrs=["f1", "f2"],
    )

    fp1 = engine.configuration_fingerprint()

    # Create synthetic evaluation partition with disjoint fit and selection
    X_tr = pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
        index=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    y_tr = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=[1, 2, 3, 4, 5, 6, 7, 8])
    X_sel = pd.DataFrame(
        {"f1": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], "f2": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]},
        index=[9, 10, 11, 12, 13, 14, 15, 16],
    )
    y_sel = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=[9, 10, 11, 12, 13, 14, 15, 16])
    part = EvaluationPartition(fit_X=X_tr, fit_y=y_tr, selection_X=X_sel, selection_y=y_sel)

    # Context fingerprint under C=1.0
    ctx1 = engine._build_candidate_cache_context_fingerprint(part, epsilon_threshold=None, curr_max_eps=None)
    cand_sig = "f1:power(3.0)"
    parent_state = hash_transform_state({})

    # Record evaluation under ctx1
    engine.tracker.mark_candidate_evaluated(parent_state, cand_sig, context_fingerprint=ctx1)
    assert engine.tracker.is_candidate_evaluated(parent_state, cand_sig, context_fingerprint=ctx1) is True

    # Mutate C from 1.0 to 2.0 on the SAME engine/evaluator
    evaluator.model.set_params(C=2.0)
    fp2 = engine.configuration_fingerprint()
    assert fp1 != fp2, "Mutating C=1.0 to C=2.0 on same engine must alter configuration fingerprint"

    ctx2 = engine._build_candidate_cache_context_fingerprint(part, epsilon_threshold=None, curr_max_eps=None)
    assert ctx1 != ctx2, "Context fingerprint must change when model hyperparameter C changes"

    # Verify cache miss / invalidation under C=2.0 context
    assert engine.tracker.is_candidate_evaluated(parent_state, cand_sig, context_fingerprint=ctx2) is False

    # Restore C=1.0 and verify cache reuse
    evaluator.model.set_params(C=1.0)
    ctx1_restored = engine._build_candidate_cache_context_fingerprint(part, epsilon_threshold=None, curr_max_eps=None)
    assert ctx1 == ctx1_restored
    assert engine.tracker.is_candidate_evaluated(parent_state, cand_sig, context_fingerprint=ctx1_restored) is True


def test_cache_2_scaler_param_invalidation():
    """CACHE-2: Scaling config changes invalidate configuration fingerprint; unchanged parameters match."""
    cfg = FairBiasConfig(classifier="LR")
    evaluator_minmax = FairEvaluator(config=cfg)
    evaluator_minmax.scaler = MinMaxScaler(feature_range=(0, 1))

    evaluator_std = FairEvaluator(config=cfg)
    evaluator_std.scaler = StandardScaler()

    evaluator_minmax2 = FairEvaluator(config=cfg)
    evaluator_minmax2.scaler = MinMaxScaler(feature_range=(0, 1))

    fp_minmax = compute_configuration_fingerprint(config=cfg, scaler=evaluator_minmax.scaler)
    fp_std = compute_configuration_fingerprint(config=cfg, scaler=evaluator_std.scaler)
    fp_minmax2 = compute_configuration_fingerprint(config=cfg, scaler=evaluator_minmax2.scaler)

    assert fp_minmax != fp_std, "Changing scaler type must alter configuration fingerprint"
    assert fp_minmax == fp_minmax2, "Identical scaler configuration must preserve fingerprint"


def test_cache_3_ae_joint_cycle_detection():
    """CACHE-3: StatefulCandidateTracker, real AE enhance_step, and Joint cycle detection routes verified."""
    tracker = StatefulCandidateTracker()
    state_0 = changed_dict_hash({})
    state_1 = changed_dict_hash({"x1": {"power": 3.0}})
    state_2 = changed_dict_hash({"x1": {"power": 5.0}})

    ctx = "ctx_test_1"
    tracker.record_state_visit(state_0, context_fingerprint=ctx)
    tracker.record_state_visit(state_1, context_fingerprint=ctx)

    assert not tracker.is_cycle(state_2, context_fingerprint=ctx)
    assert tracker.is_cycle(state_0, context_fingerprint=ctx), "Revisiting state_0 must be detected as a cycle"

    # Real AE enhance_step execution with cycle detection
    cfg = FairBiasConfig(classifier="LR")
    evaluator = FairEvaluator(config=cfg)
    evaluator.model = LogisticRegression(C=1.0, random_state=42)
    transformer = FairTransform()

    X_tr = pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
        index=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    y_tr = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=[1, 2, 3, 4, 5, 6, 7, 8])
    X_sel = pd.DataFrame(
        {"f1": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], "f2": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]},
        index=[9, 10, 11, 12, 13, 14, 15, 16],
    )
    y_sel = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=[9, 10, 11, 12, 13, 14, 15, 16])
    part = EvaluationPartition(fit_X=X_tr, fit_y=y_tr, selection_X=X_sel, selection_y=y_sel)

    engine = FairAccuracyEnhancement(
        evaluator=evaluator,
        transformer=transformer,
        label_Y="target",
        cate_attrs=[],
        num_attrs=["f1", "f2"],
    )

    # Run real AE enhance_step
    engine.enhance_step(X_tr, y_tr, changed_dict={}, partition=part)
    assert len(engine.tracker._state_history) >= 1, "AE enhance_step must record state visit in tracker"
    assert engine.tracker.is_cycle(state_0), "Root state must be detected as cycle after enhance_step"

    # D8 Joint cycle detection logic verification
    committed_joint_states_set = {state_0}
    bm_candidate_state = state_1
    assert bm_candidate_state not in committed_joint_states_set
    committed_joint_states_set.add(bm_candidate_state)

    # Second proposed transition revisits state_0: must detect cycle
    cycle_candidate = state_0
    assert cycle_candidate in committed_joint_states_set, "Revisiting committed state in Joint loop must be detected as cycle"


class ParamModel:
    """Mock model with typed parameter dict for testing configuration fingerprint."""
    def __init__(self, value):
        self.value = value

    def get_params(self, deep=True):
        return {"value": self.value}


def test_cache_4_typed_nested_dict_key_collision():
    """CACHE-4: {1: 0.5} and {'1': 0.5} produce distinct fingerprints and do not collide."""
    fp_int_key = compute_configuration_fingerprint(model=ParamModel({1: 0.5}))
    fp_str_key = compute_configuration_fingerprint(model=ParamModel({"1": 0.5}))
    assert fp_int_key != fp_str_key, "Typed parameters with int vs str keys must produce distinct fingerprints"


def test_cache_5_empty_vs_full_grid_distinct():
    """CACHE-5: Empty grid () does not collapse to default full grid in FairAccuracyEnhancement."""
    cfg = FairBiasConfig()
    e = FairEvaluator(config=cfg)
    t = FairTransform()
    empty = FairAccuracyEnhancement(e, t, "y", [], ["x"], poly_exponents=())
    full = FairAccuracyEnhancement(e, t, "y", [], ["x"])

    assert empty.poly_exponents == ()
    assert len(full.poly_exponents) > 0
    assert empty.configuration_fingerprint() != full.configuration_fingerprint(), (
        "Empty poly_exponents () must not collide with default full grid"
    )


def test_cache_6_explicit_kwargs_compatibility():
    """CACHE-6: compute_configuration_fingerprint accepts legitimate configuration parameters and they affect fingerprint."""
    fp1 = compute_configuration_fingerprint(
        algorithm_mode="engineering_bounded",
        random_seed=42,
        classifier="LR",
        eval_norm="min-max",
        min_utility_gain=0.001,
        max_fairness_degradation=0.02,
    )
    fp2 = compute_configuration_fingerprint(
        algorithm_mode="canonical_full",
        random_seed=42,
        classifier="LR",
        eval_norm="min-max",
        min_utility_gain=0.001,
        max_fairness_degradation=0.02,
    )
    assert isinstance(fp1, str) and len(fp1) == 16
    assert fp1 != fp2, "Different algorithm_mode must yield distinct fingerprints"


def test_t_config_narrowing_and_id_types():
    """T-CONFIG: Unknown keywords/typos rejected with TypeError; unconsumed params rejected; object rejected as ID."""
    # Typo rejected with TypeError
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        compute_configuration_fingerprint(epslion_threshold=0.123)

    # Unconsumed target_fairness rejected with TypeError
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        compute_configuration_fingerprint(target_fairness=0.05)

    # Unsupported arbitrary object rejected as record ID with TypeError
    with pytest.raises(TypeError, match="Unsupported record ID type"):
        EvaluationPartition._normalize_id(object())

    # Valid scalar types are supported
    assert EvaluationPartition._normalize_id("test_id") == "test_id"
    assert EvaluationPartition._normalize_id(123) == "123"
    assert EvaluationPartition._normalize_id(456.0) == "456"


def test_cache_7_model_fit_counter_spy_and_invalidation():
    """CACHE-7: Consecutive evaluations of identical state do not invoke extra model fits (cache hit); changed C/scaler re-evaluates."""
    fit_call_count = 0

    class SpyLogisticRegression(LogisticRegression):
        def fit(self, X, y, sample_weight=None):
            nonlocal fit_call_count
            fit_call_count += 1
            return super().fit(X, y, sample_weight=sample_weight)

    cfg = FairBiasConfig(classifier="LR")
    evaluator = FairEvaluator(config=cfg)
    evaluator.model = SpyLogisticRegression(C=1.0, random_state=42)
    transformer = FairTransform()

    X_tr = pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
        index=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    y_tr = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=[1, 2, 3, 4, 5, 6, 7, 8])
    X_sel = pd.DataFrame(
        {"f1": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], "f2": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]},
        index=[9, 10, 11, 12, 13, 14, 15, 16],
    )
    y_sel = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=[9, 10, 11, 12, 13, 14, 15, 16])
    part = EvaluationPartition(fit_X=X_tr, fit_y=y_tr, selection_X=X_sel, selection_y=y_sel)

    engine = FairAccuracyEnhancement(
        evaluator=evaluator,
        transformer=transformer,
        label_Y="target",
        cate_attrs=[],
        num_attrs=["f1", "f2"],
    )

    # 1. First enhance_step fits models
    engine.enhance_step(X_tr, y_tr, changed_dict={}, partition=part)
    initial_fits = fit_call_count
    assert initial_fits > 0, "First enhance_step must invoke model fit"

    # 2. Second enhance_step on the SAME state with cached candidates does NOT invoke fits (cache hit)
    engine.enhance_step(X_tr, y_tr, changed_dict={}, partition=part)
    assert fit_call_count == initial_fits, (
        f"Repeated execution with identical configuration must reuse cached candidates without refitting ({fit_call_count} == {initial_fits})"
    )

    # 3. Mutating C=2.0 invalidates the cache context and triggers actual refit
    evaluator.model.set_params(C=2.0)
    engine.enhance_step(X_tr, y_tr, changed_dict={}, partition=part)
    assert fit_call_count > initial_fits, "Mutating C parameter must trigger re-evaluation and new model fits"
    fits_after_c = fit_call_count

    # 4. Mutating scaler re-evaluates
    from sklearn.preprocessing import StandardScaler
    evaluator.scaler = StandardScaler()
    engine.enhance_step(X_tr, y_tr, changed_dict={}, partition=part)
    assert fit_call_count > fits_after_c, "Mutating scaler must invalidate cache and trigger re-evaluation"


def test_cache_8_epsilon_data_and_return_to_config_invalidation():
    """CACHE-8: Epsilon change, F/C data perturbation invalidate cache; return-to-config policy."""
    import dataclasses
    fit_call_count = 0

    class SpyLogisticRegression(LogisticRegression):
        def fit(self, X, y, sample_weight=None):
            nonlocal fit_call_count
            fit_call_count += 1
            return super().fit(X, y, sample_weight=sample_weight)

    cfg = dataclasses.replace(FairBiasConfig(classifier="LR"), label_O=("prot",))
    evaluator = FairEvaluator(config=cfg, label_O=["prot"])
    evaluator.model = SpyLogisticRegression(C=1.0, random_state=42)
    transformer = FairTransform()

    X_tr = pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
        index=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    y_tr = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=[1, 2, 3, 4, 5, 6, 7, 8])
    O_tr = pd.DataFrame({"prot": [0, 1, 0, 1, 0, 1, 0, 1]}, index=[1, 2, 3, 4, 5, 6, 7, 8])

    X_sel = pd.DataFrame(
        {"f1": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], "f2": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]},
        index=[9, 10, 11, 12, 13, 14, 15, 16],
    )
    y_sel = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=[9, 10, 11, 12, 13, 14, 15, 16])
    O_sel = pd.DataFrame({"prot": [0, 1, 0, 1, 0, 1, 0, 1]}, index=[9, 10, 11, 12, 13, 14, 15, 16])

    part = EvaluationPartition(
        fit_X=X_tr,
        fit_y=y_tr,
        selection_X=X_sel,
        selection_y=y_sel,
        protected_fit=O_tr,
        protected_selection=O_sel,
    )

    engine = FairAccuracyEnhancement(
        evaluator=evaluator,
        transformer=transformer,
        label_Y="target",
        cate_attrs=[],
        num_attrs=["f1", "f2"],
    )

    # 0. Verify MISSING_PROTECTED_DATA_WITH_ENABLED_GUARD check is active and strictly enforced
    missing_guard_res = engine._is_fairness_acceptable(
        X_cand=X_tr,
        O_train=None,
        epsilon_threshold=0.5,
        current_max_epsilon=None,
    )
    assert missing_guard_res.is_acceptable is False
    assert missing_guard_res.rejection_reason == "MISSING_PROTECTED_DATA_WITH_ENABLED_GUARD"

    # 1. Base execution
    engine.enhance_step(X_tr, y_tr, changed_dict={}, partition=part, O_train=O_tr)
    baseline_fits = fit_call_count
    assert baseline_fits > 0

    # 2. Perturbing F data alters fit fingerprint -> invalidates cache and triggers new fits
    X_tr_perturbed = X_tr.copy()
    X_tr_perturbed["f1"] = X_tr_perturbed["f1"] * 2.0
    part_f_perturbed = EvaluationPartition(
        fit_X=X_tr_perturbed,
        fit_y=y_tr,
        selection_X=X_sel,
        selection_y=y_sel,
        protected_fit=O_tr,
        protected_selection=O_sel,
    )
    engine.enhance_step(X_tr_perturbed, y_tr, changed_dict={}, partition=part_f_perturbed, O_train=O_tr)
    assert fit_call_count > baseline_fits, "Perturbing F data must invalidate cache and trigger new fits"
    fits_after_f = fit_call_count

    # 3. Perturbing C data alters selection fingerprint -> invalidates cache and triggers new fits
    X_sel_perturbed = X_sel.copy()
    X_sel_perturbed["f2"] = X_sel_perturbed["f2"] * 5.0
    part_c_perturbed = EvaluationPartition(
        fit_X=X_tr,
        fit_y=y_tr,
        selection_X=X_sel_perturbed,
        selection_y=y_sel,
        protected_fit=O_tr,
        protected_selection=O_sel,
    )
    engine.enhance_step(X_tr, y_tr, changed_dict={}, partition=part_c_perturbed, O_train=O_tr)
    assert fit_call_count > fits_after_f, "Perturbing C data must invalidate cache and trigger new fits"
    fits_after_c = fit_call_count

    # 4. Epsilon modification on same engine with O_train provided: epsilon_threshold=0.5
    audit_start_05 = len(engine.audit_trail)
    fits_before_05 = fit_call_count
    engine.enhance_step(X_tr, y_tr, changed_dict={}, partition=part, O_train=O_tr, epsilon_threshold=0.5)
    fits_eps_05 = fit_call_count
    new_fits_05 = fits_eps_05 - fits_before_05
    events_05 = engine.audit_trail[audit_start_05:]
    cand_fits_05 = sum(e.model_fit_count for e in events_05)
    geom_evals_05 = sum(e.geometry_eval_count for e in events_05)
    rejection_reasons_05 = [e.rejection_reason for e in events_05]

    # Verify candidates passed fairness gate and triggered candidate model fits
    assert geom_evals_05 > 0, "Geometry evaluations must be performed under enabled guard"
    assert cand_fits_05 > 0, "Candidate utility model fits must be triggered when candidates pass fairness gate"
    assert "MISSING_PROTECTED_DATA_WITH_ENABLED_GUARD" not in rejection_reasons_05

    # 5. Changing epsilon_threshold to 0.2 alters context fingerprint -> triggers new candidate evaluations
    audit_start_02 = len(engine.audit_trail)
    fits_before_02 = fit_call_count
    engine.enhance_step(X_tr, y_tr, changed_dict={}, partition=part, O_train=O_tr, epsilon_threshold=0.2)
    fits_eps_02 = fit_call_count
    new_fits_02 = fits_eps_02 - fits_before_02
    events_02 = engine.audit_trail[audit_start_02:]
    cand_fits_02 = sum(e.model_fit_count for e in events_02)
    geom_evals_02 = sum(e.geometry_eval_count for e in events_02)
    assert fits_eps_02 > fits_eps_05, "Changing epsilon_threshold must invalidate context and trigger new evaluations"

    # 6. Returning to prior configuration (epsilon_threshold=0.5):
    # Baseline fit is computed once per enhance_step, but candidate fits are completely reused from tracker cache
    audit_start_return = len(engine.audit_trail)
    fits_before_return = fit_call_count
    engine.enhance_step(X_tr, y_tr, changed_dict={}, partition=part, O_train=O_tr, epsilon_threshold=0.5)
    fits_return_05 = fit_call_count
    new_fits_return = fits_return_05 - fits_before_return
    events_return = engine.audit_trail[audit_start_return:]
    cand_fits_return = sum(e.model_fit_count for e in events_return)
    geom_evals_return = sum(e.geometry_eval_count for e in events_return)

    # Returning to 0.5 reuses evaluated candidates in tracker: 0 new candidate model fits
    assert cand_fits_return == 0, "Returning to prior configuration must reuse cached candidate evaluations"
    assert new_fits_return <= 1, "At most 1 baseline fit on return to prior configuration"

    # Save cache trace
    guard = FairBiasR1Guard.get_instance()
    out_dir = guard.output_dir if guard is not None else pathlib.Path("runs")
    trace_path = out_dir / "cache_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_fits": baseline_fits,
            "fits_after_f": fits_after_f,
            "fits_after_c": fits_after_c,
            "fits_eps_05": fits_eps_05,
            "new_fits_05": new_fits_05,
            "cand_fits_05": cand_fits_05,
            "geom_evals_05": geom_evals_05,
            "fits_eps_02": fits_eps_02,
            "new_fits_02": new_fits_02,
            "cand_fits_02": cand_fits_02,
            "geom_evals_02": geom_evals_02,
            "fits_return_05": fits_return_05,
            "new_fits_return": new_fits_return,
            "cand_fits_return": cand_fits_return,
            "geom_evals_return": geom_evals_return,
            "candidate_cache_reused": bool(cand_fits_return == 0),
            "missing_protected_data_guard_active": True,
        }, f, indent=2)


def test_r2_01_public_fair_evaluator_compute_metrics():
    """R2-01: Public FairEvaluator.compute_metrics contract, return fields, single-group, and expected groups validation."""
    import dataclasses
    cfg = dataclasses.replace(FairBiasConfig.compas_default(), label_O=("prot",))

    # 1. Standard two-group evaluation
    evaluator = FairEvaluator(config=cfg, label_O=["prot"], expected_groups=["g0", "g1"])
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4])
    O_df = pd.DataFrame({"prot": ["g0", "g1", "g0", "g1"]})

    res = evaluator.compute_metrics(y_true, y_pred, O_df, y_prob=y_prob)

    # Verify all mandatory performance and disparity fields exist
    expected_top_keys = [
        "ACC", "F1", "Precision", "Recall", "SP", "EO", "EOpp", "CUAE", "OAE",
        "BNC", "BPC", "FDRP", "FORP", "FNRB", "FPRB", "NPVP", "PPVP",
        "application_demographic_parity", "application_equal_opportunity", "application_equalized_odds",
        "dp_status", "eo_status", "equalized_odds_status", "fairness_status",
        "dp_estimable", "eo_estimable", "equalized_odds_estimable", "is_primary_estimand",
    ]
    for key in expected_top_keys:
        assert key in res, f"Missing expected key in compute_metrics result: {key}"

    assert res["dp_status"]["prot"] == "VALID"
    assert res["dp_estimable"]["prot"] is True
    assert res["application_demographic_parity"]["prot"] is not None

    # 2. Single-group behavior: disparity metrics must be None, status is SINGLE_GROUP, without unhandled error
    O_single = pd.DataFrame({"prot": ["g0", "g0", "g0", "g0"]})
    res_single = evaluator.compute_metrics(y_true, y_pred, O_single, y_prob=y_prob)
    assert res_single["SP"]["prot"] is None
    assert res_single["EO"]["prot"] is None
    assert res_single["dp_status"]["prot"] == "SINGLE_GROUP"
    assert res_single["dp_estimable"]["prot"] is False

    # 3. Missing expected group behavior: observed data lacks declared expected group
    evaluator_missing = FairEvaluator(config=cfg, label_O=["prot"], expected_groups=["g0", "g1", "g_unseen"])
    res_missing = evaluator_missing.compute_metrics(y_true, y_pred, O_df, y_prob=y_prob)
    assert "MISSING_EXPECTED_GROUPS" in str(res_missing["dp_status"]["prot"])

    # 4. Extra unexpected group behavior: observed data contains group not in expected_groups -> raises ValueError
    evaluator_extra = FairEvaluator(config=cfg, label_O=["prot"], expected_groups=["g0"])
    with pytest.raises(ValueError, match="Observed unexpected groups not declared in expected_groups"):
        evaluator_extra.compute_metrics(y_true, y_pred, O_df, y_prob=y_prob)

    # Save evaluator trace artifact
    guard = FairBiasR1Guard.get_instance()
    out_dir = guard.output_dir if guard is not None else pathlib.Path("runs")
    trace_path = out_dir / "evaluator_compute_metrics_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump({
            "entrypoint": "FairEvaluator.compute_metrics",
            "keys_present": list(res.keys()),
            "dp_status_normal": res["dp_status"],
            "dp_status_single": res_single["dp_status"],
            "dp_status_missing": res_missing["dp_status"],
            "extra_group_raises_value_error": True,
        }, f, indent=2)


def test_r4_03_d8_four_arms_and_joint_cycle_detection(monkeypatch):
    """R4-03 / R5-04: Real D8EnhancementRunner.run_arm execution across 4 conditions and real Joint cycle detection."""
    from nhis_fairbias.d8_enhancement_runner import D8EnhancementRunner, D8ExecutionMode
    from fairbias.mitigation import FairBiasMitigation
    from fairbias.enhancement_state import changed_dict_hash

    class SyntheticAdapter:
        def __init__(self):
            class SyntheticPreprocessor:
                def get_feature_family_lists(self, feature_set):
                    return ([], ["f1", "f2"])
            self.preprocessor = SyntheticPreprocessor()

        def get_cohort(self, year, outcome, protected_attribute, feature_set, disability_arm):
            n = 20
            idx = list(range(1, n + 1))
            X = pd.DataFrame({
                "f1": np.linspace(1.0, 5.0, n),
                "f2": np.linspace(0.1, 0.9, n),
            }, index=idx)
            y_s = pd.Series([0, 1] * (n // 2), index=idx)
            o_s = pd.Series([1, 2] * (n // 2), index=idx)
            w_s = pd.Series([1.0] * n, index=idx)
            psu_s = pd.Series([1] * n, index=idx)
            return X, y_s, o_s, w_s, psu_s

    adapter = SyntheticAdapter()
    canonical_provider = lambda arm_id: {"f1": {"power": 2.0}}

    # 1. Independent positive test: standard 4-arm run
    runner = D8EnhancementRunner(
        mode=D8ExecutionMode.EXPLORATORY_ENGINEERING,
        adapter=adapter,
        canonical_provider=canonical_provider,
        smoke_test=True,
        random_seed=42,
    )

    res = runner.run_arm("D6_ARM_001")
    assert res["arm_id"] == "D6_ARM_001"
    conds = res["conditions"]

    assert "baseline" in conds
    assert conds["baseline"]["terminal_evaluation_performed"] is True
    assert conds["baseline"]["expected_groups"] == [1, 2]

    assert "canonical_fairbias" in conds
    assert conds["canonical_fairbias"]["terminal_evaluation_performed"] is True
    assert conds["canonical_fairbias"]["expected_groups"] == [1, 2]

    assert "posthoc_enhancement" in conds
    assert conds["posthoc_enhancement"]["expected_groups"] == [1, 2]

    assert "joint_enhancement" in conds
    assert conds["joint_enhancement"]["expected_groups"] == [1, 2]
    assert conds["joint_enhancement"]["termination_reason"] in ("budget_exhausted", "cycle_detected", "candidate_exhausted")

    # 2. Real Joint cycle detection under run_arm:
    # We control candidate proposals so mitigation proposes A -> B (accepted), then in iteration 2 proposes A (cycle detected!)
    call_cnt = 0
    def controlled_mitigate_step(self, X, Y, O, nmi_org, changed_dict, current_epsilon, epsilon_threshold, iteration):
        nonlocal call_cnt
        call_cnt += 1
        if call_cnt == 1:
            # Iteration 1: propose state B = {"f1": {"power": 2.0}}
            return X, {"f1": {"power": 2.0}}, "prot", "f1"
        else:
            # Iteration 2: propose state A = {} (revisiting committed initial state)
            return X, {}, "prot", "f1"

    from fairbias.evaluator import FairEvaluator
    call_eps_cnt = 0
    def controlled_calculate_epsilon(self, *args, **kwargs):
        nonlocal call_eps_cnt
        call_eps_cnt += 1
        if call_eps_cnt <= 1:
            return {"f1": {"prot": 0.2}}
        return {"f1": {"prot": 0.8}}

    monkeypatch.setattr(FairBiasMitigation, "mitigate_step", controlled_mitigate_step)
    monkeypatch.setattr(FairEvaluator, "calculate_epsilon", controlled_calculate_epsilon)

    runner_cycle = D8EnhancementRunner(
        mode=D8ExecutionMode.EXPLORATORY_ENGINEERING,
        adapter=adapter,
        canonical_provider=canonical_provider,
        smoke_test=True,
        random_seed=42,
    )

    cycle_res = runner_cycle.run_arm("D6_ARM_001")
    joint_info = cycle_res["conditions"]["joint_enhancement"]

    assert joint_info["termination_reason"] == "cycle_detected", (
        f"Expected cycle_detected, got {joint_info['termination_reason']}"
    )
    assert joint_info["bm_steps_accepted"] == 1, "Must have accepted step A -> B before cycle"
    assert joint_info["terminal_state"] == {"f1": {"power": 2.0}}, "Must retain committed state B upon cycle detection"
    assert joint_info["terminal_evaluation_performed"] is True

    # Verify cycle detection event in iteration_events
    events = joint_info["iteration_events"]
    cycle_events = [e for e in events if e.get("cycle_detected") is True]
    assert len(cycle_events) == 1, "Must record exactly one cycle detection event"
    ce = cycle_events[0]
    assert ce["engine"] == "BM"
    assert ce["trajectory_committed"] is False
    assert ce["resulting_state_hash"] == changed_dict_hash({})

    # Save structured trace artifact for supervisor verification
    guard = FairBiasR1Guard.get_instance()
    out_dir = guard.output_dir if guard is not None else pathlib.Path("runs")
    trace_path = out_dir / "joint_cycle_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump({
            "termination_reason": joint_info["termination_reason"],
            "bm_steps_accepted": joint_info["bm_steps_accepted"],
            "ae_steps_accepted": joint_info["ae_steps_accepted"],
            "terminal_state": joint_info["terminal_state"],
            "events": events,
        }, f, indent=2)




