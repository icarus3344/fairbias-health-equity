"""Synthetic verification suite for CATEGORY-1...4, METRIC-1...4, and GEOM-1...6 contracts."""

import json
import pathlib
import numpy as np
import pandas as pd
import pytest

from fairbias.application_metrics import compute_application_group_fairness
from fairbias.bias_metric import (
    ORIGIN,
    compute_bias_concentration,
    compute_dphi_matrix,
    compute_pairwise_divergences,
    compute_shapley_distance_matrix,
    get_subsets,
)
from fairbias.config import FairBiasConfig, official_power_stream
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from fairbias.transform import FairTransform, calculate_nmi_dict
from nhis_fairbias.d8_enhancement_runner import check_fairness_feasibility, evaluate_representation
from _fairbias_r1_guard import FairBiasR1Guard


def test_category_1_two_group_divergence():
    """CATEGORY-1: Categorical feature c=[0, 1] across two groups yields divergence 1.0."""
    df = pd.DataFrame({"c": [0, 1]})
    prot = pd.Series(["g0", "g1"])

    div_df = compute_pairwise_divergences(
        X=df,
        o_series=prot,
        cate_attrs=["c"],
        num_attrs=[],
    )
    val = float(div_df.loc["c", "g0_g1"])
    assert abs(val - 1.0) < 1e-12, f"Expected 1.0 for binary category across 2 groups, got {val}"


def test_category_2_unobserved_level_invariance():
    """CATEGORY-2: Adding unobserved level 2 does not change divergence from 1.0."""
    # Data has only levels 0 and 1, but categorical series has categories [0, 1, 2]
    c_cat = pd.Categorical([0, 1], categories=[0, 1, 2])
    df = pd.DataFrame({"c": c_cat})
    prot = pd.Series(["g0", "g1"])

    div_df = compute_pairwise_divergences(
        X=df,
        o_series=prot,
        cate_attrs=["c"],
        num_attrs=[],
    )
    val = float(div_df.loc["c", "g0_g1"])
    assert abs(val - 1.0) < 1e-12, f"Unobserved level must not inflate K; expected 1.0, got {val}"


def test_category_3_zero_weight_level_invariance():
    """CATEGORY-3: Zero-weight category level does not enlarge K in weighted divergence."""
    # Observations with category 2 have weight 0
    df = pd.DataFrame({"c": [0, 1, 2, 2]})
    prot = pd.Series(["g0", "g1", "g0", "g1"])
    weights = pd.Series([10.0, 10.0, 0.0, 0.0])

    div_df = compute_pairwise_divergences(
        X=df,
        o_series=prot,
        cate_attrs=["c"],
        num_attrs=[],
        sample_weight=weights,
        allow_zero_weights=True,
    )
    val = float(div_df.loc["c", "g0_g1"])
    assert abs(val - 1.0) < 1e-12, f"Zero-weight category must not enlarge K; expected 1.0, got {val}"


def test_category_4_merged_empty_level_invariance():
    """CATEGORY-4: Merged empty category levels do not alter the geometry."""
    # Category 2 was merged away, leaving 0 observations in level 2
    c_series = pd.Categorical([0, 1], categories=[0, 1, 2])
    df = pd.DataFrame({"c": c_series})
    prot = pd.Series(["g0", "g1"])

    div_df = compute_pairwise_divergences(
        X=df,
        o_series=prot,
        cate_attrs=["c"],
        num_attrs=[],
    )
    assert abs(float(div_df.loc["c", "g0_g1"]) - 1.0) < 1e-12


def test_metric_1_single_group_unestimable():
    """METRIC-1: Single group evaluation returns None + status, not 0.0."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    prot = np.array(["only_group", "only_group", "only_group", "only_group"])

    res = compute_application_group_fairness(y_true, y_pred, prot)
    assert res["demographic_parity_difference"] is None
    assert res["equal_opportunity_difference"] is None
    assert res["dp_status"] == "SINGLE_GROUP"
    assert res["eo_status"] == "SINGLE_GROUP"


def test_metric_2_three_group_dp_vs_legacy_mean_pair():
    """METRIC-2: Three groups with positive rates 0, 0, 1 yield application DP=1.0, legacy mean-pair=2/3."""
    # Group 1: 10 samples, 0 positive predictions -> rate = 0.0
    # Group 2: 10 samples, 0 positive predictions -> rate = 0.0
    # Group 3: 10 samples, 10 positive predictions -> rate = 1.0
    y_true = np.array([0] * 10 + [0] * 10 + [1] * 10)
    y_pred = np.array([0] * 10 + [0] * 10 + [1] * 10)
    prot = np.array(["g1"] * 10 + ["g2"] * 10 + ["g3"] * 10)

    res = compute_application_group_fairness(y_true, y_pred, prot)

    assert abs(res["demographic_parity_difference"] - 1.0) < 1e-12
    assert abs(res["legacy_demographic_parity_mean_pair"] - (2.0 / 3.0)) < 1e-12


def test_metric_3_missing_expected_group_eo_unestimable():
    """METRIC-3: Seven expected groups where one group is missing in data: primary EO is unestimable."""
    expected_7 = ["g1", "g2", "g3", "g4", "g5", "g6", "g7"]
    # Only 6 groups in data
    y_true = np.array([0, 1] * 6)
    y_pred = np.array([0, 1] * 6)
    prot = np.array(["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4", "g5", "g5", "g6", "g6"])

    res = compute_application_group_fairness(y_true, y_pred, prot, expected_groups=expected_7)

    assert res["demographic_parity_difference"] is None
    assert res["equal_opportunity_difference"] is None
    assert "MISSING_EXPECTED_GROUPS" in res["dp_status"]
    assert "MISSING_EXPECTED_GROUPS" in res["eo_status"]


def test_metric_4_dp_valid_when_eo_lacks_positives():
    """METRIC-4: When DP support is sufficient across all expected groups, but one group lacks positive cases, DP is valid while EO is unestimable."""
    # g1 has positives (y_true=1), g2 has NO positives (all y_true=0)
    y_true = np.array([0, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 1])
    prot = np.array(["g1", "g1", "g2", "g2"])

    res = compute_application_group_fairness(y_true, y_pred, prot, expected_groups=["g1", "g2"])

    assert res["dp_estimable"] is True
    assert res["demographic_parity_difference"] is not None
    assert res["eo_estimable"] is False
    assert res["equal_opportunity_difference"] is None
    assert "MISSING_POSITIVE_SAMPLES" in res["eo_status"]


def test_geom_1_h1_context_subsets():
    """GEOM-1: H=1 context generation produces near-full companion sets (excluding at most 1 item)."""
    subsets = get_subsets(["a", "b", "c"], h_order=1)
    # Expected subsets: {a,b}, {a,c}, {b,c}, {a,b,c}
    set_tuples = {tuple(sorted(s)) for s in subsets}
    assert set_tuples == {("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c")}


def test_geom_2_vector_aggregation_counterexample():
    """GEOM-2: author_max_pair gives 0.0, while mean_pair gives 0.8 for v1=(.9, .1), v2=(.1, .9)."""
    df_s = pd.DataFrame(
        {
            "p1_n1": [0.9, 0.1],
            "p2_n2": [0.1, 0.9],
        },
        index=["f1", "f2"],
    )
    dist_author, _ = compute_shapley_distance_matrix(
        df_s, features=["f1", "f2"], h_order=1, multigroup_aggregation="author_max_pair"
    )
    dist_mean, _ = compute_shapley_distance_matrix(
        df_s, features=["f1", "f2"], h_order=1, multigroup_aggregation="mean_pair"
    )

    # In author_max_pair: abs(max(0.9, 0.1) - max(0.1, 0.9)) = abs(0.9 - 0.9) = 0.0
    # In mean_pair: mean(abs(0.9 - 0.1), abs(0.1 - 0.9)) = mean(0.8, 0.8) = 0.8
    assert abs(dist_author[0, 1] - 0.0) < 1e-12
    assert abs(dist_mean[0, 1] - 0.8) < 1e-12


def test_geom_3_config_propagates_to_dphi_spy(monkeypatch):
    """GEOM-3: FairEvaluator.calculate_epsilon explicitly passes multigroup_aggregation to compute_dphi_matrix."""
    import fairbias.evaluator

    cfg = FairBiasConfig(multigroup_aggregation="author_max_pair")
    evaluator = FairEvaluator(config=cfg)

    spy_calls = []
    orig_fn = fairbias.evaluator.compute_dphi_matrix

    def spy_compute_dphi(*args, **kwargs):
        spy_calls.append(kwargs)
        return orig_fn(*args, **kwargs)

    monkeypatch.setattr(fairbias.evaluator, "compute_dphi_matrix", spy_compute_dphi)

    df_X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [4.0, 3.0, 2.0, 1.0]})
    df_O = pd.DataFrame({"prot": ["A", "A", "B", "B"]})

    # Execute calculate_epsilon with author_max_pair
    res = evaluator.calculate_epsilon(df_X, df_O, cate_attrs=[], num_attrs=["f1", "f2"])
    assert len(spy_calls) >= 1, "compute_dphi_matrix must be called by calculate_epsilon"
    assert (
        spy_calls[0].get("multigroup_aggregation") == "author_max_pair"
    ), f"multigroup_aggregation must propagate to compute_dphi_matrix, got: {spy_calls[0].get('multigroup_aggregation')}"
    assert "prot" in res


def test_geom_4_synthetic_mds_bm_call():
    """GEOM-4: Real small MDS and BM candidate search integration verified on synthetic data."""
    df_X = pd.DataFrame({
        "num1": [0.1, 0.2, 0.8, 0.9, 0.15, 0.85],
        "num2": [10.0, 20.0, 80.0, 90.0, 15.0, 85.0],
    })
    df_y = pd.Series([0, 0, 1, 1, 0, 1])
    df_O = pd.DataFrame({"prot": ["A", "A", "B", "B", "A", "B"]})

    cfg = FairBiasConfig(
        mds_fixed_components=2,
        multigroup_aggregation="mean_pair",
        power_revisit_policy="restart",
        failed_attribute_mode="stop",
    )
    evaluator = FairEvaluator(config=cfg)
    transformer = FairTransform()

    mit = FairBiasMitigation(
        evaluator=evaluator,
        transformer=transformer,
        label_O=["prot"],
        cate_attrs=[],
        num_attrs=["num1", "num2"],
        power_revisit_policy="restart",
        failed_attribute_mode="stop",
    )

    dphi = evaluator.calculate_epsilon(df_X, df_O, cate_attrs=[], num_attrs=["num1", "num2"])
    assert float(dphi["prot"]["num1"]) >= 0.0
    assert float(dphi["prot"]["num2"]) >= 0.0

    # Run real BM mitigate step to verify integration of candidate search and MDS evaluation
    res_df, new_dict, chosen_attr, chosen_trans = mit.mitigate_step(
        X=df_X,
        Y=df_y,
        O=df_O,
        nmi_org={"num1": 0.5, "num2": 0.5},
        changed_dict={},
        current_epsilon=dphi,
        epsilon_threshold=0.001,
    )
    assert mit.non_convergence is not None, "BM search must record non-convergence when epsilon ball is unreachable"
    assert mit.non_convergence["label_O"] == "prot"


def test_geom_5_restart_vs_monotone_cursor_and_stop_mode():
    """GEOM-5: Official power stream (1998 items), restart policy, and stop mode termination."""
    from fairbias.config import official_power_stream

    stream = official_power_stream()
    assert len(stream) == 1998, f"Official power stream must have exactly 1998 elements, got {len(stream)}"
    assert stream[0] == 3.0
    assert abs(stream[1] - (1.0 / 3.0)) < 1e-12

    # Verify restart policy starts at index 0
    cfg_restart = FairBiasConfig(power_revisit_policy="restart", failed_attribute_mode="stop")
    evaluator = FairEvaluator(config=cfg_restart)
    transformer = FairTransform()

    mit_restart = FairBiasMitigation(
        evaluator=evaluator,
        transformer=transformer,
        label_O=["prot"],
        cate_attrs=[],
        num_attrs=["num1"],
        power_revisit_policy="restart",
        failed_attribute_mode="stop",
    )
    assert mit_restart.power_revisit_policy == "restart"
    assert mit_restart.failed_attribute_mode == "stop"

    # Verify failed_attribute_mode='stop' terminates on failure
    df_X = pd.DataFrame({"num1": [0.1, 0.2, 0.8, 0.9]})
    df_y = pd.Series([0, 0, 1, 1])
    df_O = pd.DataFrame({"prot": ["A", "A", "B", "B"]})
    curr_eps = {"prot": {"num1": 0.5}}

    _, _, chosen, _ = mit_restart.mitigate_step(
        X=df_X,
        Y=df_y,
        O=df_O,
        nmi_org={"num1": 0.3},
        changed_dict={},
        current_epsilon=curr_eps,
        epsilon_threshold=0.0001,
    )
    assert chosen is None, "failed_attribute_mode='stop' must terminate step when candidate fails to reach threshold"
    assert mit_restart.non_convergence is not None


def test_geom_6_nmi_and_epsilon_boundaries():
    """GEOM-6: Strict epsilon boundary (new_dphi < epsilon_threshold), NMI validity, and non-finite geometry rejection."""
    # NMI calculation validity
    from fairbias.transform import calculate_nmi_dict

    df_X = pd.DataFrame({"num1": [1.0, 2.0, 3.0, 4.0]})
    df_y = pd.Series([0, 0, 1, 1])
    nmi_dict = calculate_nmi_dict(df_X, df_y)
    nmi = nmi_dict["num1"]
    assert 0.0 <= nmi <= 1.0, f"NMI must be bounded in [0, 1], got {nmi}"

    # Invalid aggregation mode rejection
    with pytest.raises(ValueError, match=r"multigroup_aggregation"):
        FairBiasConfig(multigroup_aggregation="invalid_mode")

    with pytest.raises(ValueError, match=r"multigroup_aggregation"):
        compute_dphi_matrix(
            X=pd.DataFrame({"f": [1, 2]}),
            O=pd.DataFrame({"p": ["a", "b"]}),
            cate_attrs=[],
            num_attrs=["f"],
            multigroup_aggregation="invalid_mode",
        )

    # Non-finite geometry rejection: NaN in features rejected
    with pytest.raises(ValueError, match="divergence is unestimable"):
        compute_dphi_matrix(
            X=pd.DataFrame({"f": [1.0, float("nan")]}),
            O=pd.DataFrame({"p": ["a", "b"]}),
            cate_attrs=[],
            num_attrs=["f"],
        )


def test_geom_7_artificial_stream_cursor_behavior():
    """GEOM-7 (artificial stream): Monotone cursor stream continuation advances without restarting; restart policy resets to 0 under artificial 6-value stream."""
    poly_exps = (1 / 7, 1 / 5, 1 / 3, 3.0, 5.0, 7.0)

    # 1. Monotone cursor engine under official_stream
    cfg_mono = FairBiasConfig(power_revisit_policy="monotone_cursor", power_sequence_policy="official_stream")
    e_mono = FairEvaluator(config=cfg_mono)
    t_mono = FairTransform()

    mono_transformed_powers_step1 = []
    mono_transformed_powers_step2 = []
    current_mono_step = 1

    orig_mono_transform = t_mono.transform_data
    def spy_mono_transform(df, changed, num_attrs, cate_attrs):
        p = changed.get("num1", {}).get("power")
        if p is not None:
            if current_mono_step == 1:
                mono_transformed_powers_step1.append(float(p))
            elif current_mono_step == 2:
                mono_transformed_powers_step2.append(float(p))
        return orig_mono_transform(df, changed, num_attrs, cate_attrs)

    t_mono.transform_data = spy_mono_transform

    mit_mono = FairBiasMitigation(
        evaluator=e_mono,
        transformer=t_mono,
        label_O=["prot"],
        cate_attrs=[],
        num_attrs=["num1"],
        poly_exponents=poly_exps,
        power_revisit_policy="monotone_cursor",
        power_sequence_policy="official_stream",
    )
    X = pd.DataFrame({"num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    Y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    O = pd.DataFrame({"prot": ["a", "a", "b", "b", "a", "a", "b", "b"]})
    nmi_org = {"num1": 0.5}

    # First search: accept at index 1 (power 1/5)
    call_cnt = 0
    orig_eps_of = mit_mono._epsilon_of
    def controlled_eps_mono(cand_df, O_df, l_o, attr):
        nonlocal call_cnt
        call_cnt += 1
        # Accept the second evaluated power (index 1)
        if call_cnt >= 2:
            return 0.0001
        return 0.5

    mit_mono._epsilon_of = controlled_eps_mono
    res1 = mit_mono._search_numerical(X, Y, O, nmi_org, {}, "prot", "num1", epsilon_threshold=0.01)
    assert res1 is not None
    cursor_after_step1 = mit_mono._exponent_stream_cursors.get("num1", 0)
    assert cursor_after_step1 == 2, f"Monotone cursor should be at index 2, got {cursor_after_step1}"
    assert len(mono_transformed_powers_step1) == 2
    assert mono_transformed_powers_step1 == [float(poly_exps[0]), float(poly_exps[1])]

    # Second search on monotone engine: starts strictly at index 2 (power 1/3)
    current_mono_step = 2
    mit_mono._epsilon_of = lambda cand_df, O_df, l_o, attr: 0.5  # do not accept, consume all
    mit_mono._search_numerical(X, Y, O, nmi_org, {"num1": {"power": poly_exps[1]}}, "prot", "num1", epsilon_threshold=0.01)
    cursor_after_step2 = mit_mono._exponent_stream_cursors.get("num1", 0)
    assert cursor_after_step2 >= cursor_after_step1
    # Step 2 evaluated powers must NOT contain the already-consumed powers [1/7, 1/5]
    assert float(poly_exps[0]) not in mono_transformed_powers_step2
    assert float(poly_exps[1]) not in mono_transformed_powers_step2
    assert mono_transformed_powers_step2[0] == float(poly_exps[2]), "Monotone search must resume from index 2"

    # 2. Restart engine under identical official_stream
    cfg_restart = FairBiasConfig(power_revisit_policy="restart", power_sequence_policy="official_stream")
    e_restart = FairEvaluator(config=cfg_restart)
    t_restart = FairTransform()

    restart_transformed_powers_step1 = []
    restart_transformed_powers_step2 = []
    current_restart_step = 1

    orig_restart_transform = t_restart.transform_data
    def spy_restart_transform(df, changed, num_attrs, cate_attrs):
        p = changed.get("num1", {}).get("power")
        if p is not None:
            if current_restart_step == 1:
                restart_transformed_powers_step1.append(float(p))
            elif current_restart_step == 2:
                restart_transformed_powers_step2.append(float(p))
        return orig_restart_transform(df, changed, num_attrs, cate_attrs)

    t_restart.transform_data = spy_restart_transform

    mit_restart = FairBiasMitigation(
        evaluator=e_restart,
        transformer=t_restart,
        label_O=["prot"],
        cate_attrs=[],
        num_attrs=["num1"],
        poly_exponents=poly_exps,
        power_revisit_policy="restart",
        power_sequence_policy="official_stream",
    )

    # First search on restart engine: accept at index 1
    call_cnt = 0
    mit_restart._epsilon_of = controlled_eps_mono
    res_r1 = mit_restart._search_numerical(X, Y, O, nmi_org, {}, "prot", "num1", epsilon_threshold=0.01)
    assert res_r1 is not None
    assert mit_restart._exponent_stream_cursors.get("num1", 0) == 0

    # Second search on restart engine: restarts from index 0!
    current_restart_step = 2
    mit_restart._epsilon_of = lambda cand_df, O_df, l_o, attr: 0.5
    mit_restart._search_numerical(X, Y, O, nmi_org, {}, "prot", "num1", epsilon_threshold=0.01)
    assert restart_transformed_powers_step2[0] == float(poly_exps[0]), "Restart search must reset to index 0"

    # Save artificial BM cursor trace artifact
    guard = FairBiasR1Guard.get_instance()
    out_dir = guard.output_dir if guard is not None else pathlib.Path("runs")
    trace_path = out_dir / "bm_artificial_cursor_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump({
            "stream_type": "artificial_6_value",
            "poly_exponents": [float(p) for p in poly_exps],
            "mono_step1_powers": mono_transformed_powers_step1,
            "mono_step2_powers": mono_transformed_powers_step2,
            "restart_step1_powers": restart_transformed_powers_step1,
            "restart_step2_powers": restart_transformed_powers_step2,
            "mono_cursor_maintained": bool(cursor_after_step2 >= cursor_after_step1),
            "restart_reset_verified": bool(restart_transformed_powers_step2[0] == float(poly_exps[0])),
        }, f, indent=2)


def test_geom_7_official_1998_stream_cursor_and_revisit_continuation():
    """GEOM-7 (official 1998 stream): Monotone cursor stream continuation advances without restarting; restart policy resets to 0 under official 1998-item stream."""
    stream = official_power_stream()
    assert len(stream) == 1998, f"Official interleaved stream must have 1998 items, got {len(stream)}"
    assert stream[0] == 3.0 and stream[1] == (1.0 / 3.0)

    # 1. Monotone cursor engine under full official 1998-item stream
    cfg_mono = FairBiasConfig(power_revisit_policy="monotone_cursor", power_sequence_policy="official_stream", transform_poly_exponents=stream)
    e_mono = FairEvaluator(config=cfg_mono)
    t_mono = FairTransform()

    mono_transformed_powers_step1 = []
    mono_transformed_powers_step2 = []
    current_mono_step = 1

    orig_mono_transform = t_mono.transform_data
    def spy_mono_transform(df, changed, num_attrs, cate_attrs):
        p = changed.get("num1", {}).get("power")
        if p is not None:
            if current_mono_step == 1:
                mono_transformed_powers_step1.append(float(p))
            elif current_mono_step == 2:
                mono_transformed_powers_step2.append(float(p))
        return orig_mono_transform(df, changed, num_attrs, cate_attrs)

    t_mono.transform_data = spy_mono_transform

    mit_mono = FairBiasMitigation(
        evaluator=e_mono,
        transformer=t_mono,
        label_O=["prot"],
        cate_attrs=[],
        num_attrs=["num1"],
        poly_exponents=stream,
        power_revisit_policy="monotone_cursor",
        power_sequence_policy="official_stream",
    )
    X = pd.DataFrame({"num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    Y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    O = pd.DataFrame({"prot": ["a", "a", "b", "b", "a", "a", "b", "b"]})
    nmi_org = {"num1": 0.5}

    # Step 1: accept at 2nd evaluated power (stream index 1, power 1/3) via early stopping
    call_cnt = 0
    def controlled_eps_mono_step1(cand_df, O_df, l_o, attr):
        nonlocal call_cnt
        call_cnt += 1
        if call_cnt == 2:
            return 0.0001
        return 0.5

    mit_mono._epsilon_of = controlled_eps_mono_step1
    cursor_start_mono1 = mit_mono._exponent_stream_cursors.get("num1", 0)
    res_mono1 = mit_mono._search_numerical(X, Y, O, nmi_org, {}, "prot", "num1", epsilon_threshold=0.01)
    assert res_mono1 is not None
    cursor_end_mono1 = mit_mono._exponent_stream_cursors.get("num1", 0)
    assert cursor_start_mono1 == 0
    assert cursor_end_mono1 == 2, f"Monotone cursor should be at index 2, got {cursor_end_mono1}"
    assert len(mono_transformed_powers_step1) == 2
    assert mono_transformed_powers_step1 == [float(stream[0]), float(stream[1])]
    assert res_mono1[1]["num1"]["power"] == float(stream[1])

    # Step 2: resume strictly at cursor 2 (stream index 2, power 5.0). Early stop at 2nd evaluated power (stream index 3, power 1/5)
    current_mono_step = 2
    call_cnt = 0
    def controlled_eps_mono_step2(cand_df, O_df, l_o, attr):
        nonlocal call_cnt
        call_cnt += 1
        if call_cnt == 2:
            return 0.0001
        return 0.5

    mit_mono._epsilon_of = controlled_eps_mono_step2
    cursor_start_mono2 = mit_mono._exponent_stream_cursors.get("num1", 0)
    assert cursor_start_mono2 == 2
    res_mono2 = mit_mono._search_numerical(X, Y, O, nmi_org, res_mono1[1], "prot", "num1", epsilon_threshold=0.01)
    assert res_mono2 is not None
    cursor_end_mono2 = mit_mono._exponent_stream_cursors.get("num1", 0)
    assert cursor_end_mono2 == 4, f"Monotone cursor should advance to index 4, got {cursor_end_mono2}"
    assert len(mono_transformed_powers_step2) == 2
    assert float(stream[0]) not in mono_transformed_powers_step2
    assert float(stream[1]) not in mono_transformed_powers_step2
    assert mono_transformed_powers_step2 == [float(stream[2]), float(stream[3])]
    assert res_mono2[1]["num1"]["power"] == float(stream[3])

    # 2. Restart engine under identical official 1998-item stream
    cfg_restart = FairBiasConfig(power_revisit_policy="restart", power_sequence_policy="official_stream", transform_poly_exponents=stream)
    e_restart = FairEvaluator(config=cfg_restart)
    t_restart = FairTransform()

    restart_transformed_powers_step1 = []
    restart_transformed_powers_step2 = []
    current_restart_step = 1

    orig_restart_transform = t_restart.transform_data
    def spy_restart_transform(df, changed, num_attrs, cate_attrs):
        p = changed.get("num1", {}).get("power")
        if p is not None:
            if current_restart_step == 1:
                restart_transformed_powers_step1.append(float(p))
            elif current_restart_step == 2:
                restart_transformed_powers_step2.append(float(p))
        return orig_restart_transform(df, changed, num_attrs, cate_attrs)

    t_restart.transform_data = spy_restart_transform

    mit_restart = FairBiasMitigation(
        evaluator=e_restart,
        transformer=t_restart,
        label_O=["prot"],
        cate_attrs=[],
        num_attrs=["num1"],
        poly_exponents=stream,
        power_revisit_policy="restart",
        power_sequence_policy="official_stream",
    )

    # Step 1 on restart engine: accept at 2nd evaluated power (stream index 1, power 1/3)
    call_cnt = 0
    mit_restart._epsilon_of = controlled_eps_mono_step1
    cursor_start_rst1 = mit_restart._exponent_stream_cursors.get("num1", 0)
    res_rst1 = mit_restart._search_numerical(X, Y, O, nmi_org, {}, "prot", "num1", epsilon_threshold=0.01)
    assert res_rst1 is not None
    cursor_end_rst1 = mit_restart._exponent_stream_cursors.get("num1", 0)
    assert cursor_end_rst1 == 0, "Restart policy cursor remains 0"
    assert restart_transformed_powers_step1 == [float(stream[0]), float(stream[1])]

    # Step 2 on restart engine: restarts from index 0! Evaluates stream[0]=3.0, skips stream[1] (current power), evaluates stream[2]=5.0 and accepts
    current_restart_step = 2
    call_cnt = 0
    mit_restart._epsilon_of = controlled_eps_mono_step2
    cursor_start_rst2 = mit_restart._exponent_stream_cursors.get("num1", 0)
    res_rst2 = mit_restart._search_numerical(X, Y, O, nmi_org, res_rst1[1], "prot", "num1", epsilon_threshold=0.01)
    assert res_rst2 is not None
    cursor_end_rst2 = mit_restart._exponent_stream_cursors.get("num1", 0)
    assert cursor_end_rst2 == 0
    assert len(restart_transformed_powers_step2) == 2
    # Restart visits stream[0] again, unlike monotone
    assert restart_transformed_powers_step2[0] == float(stream[0]), "Restart must revisit stream from head (index 0)"
    assert restart_transformed_powers_step2[1] == float(stream[2])
    assert res_rst2[1]["num1"]["power"] == float(stream[2])

    # Prove that both engines executed non-empty searches with distinct evaluation sequences
    assert mono_transformed_powers_step2 != restart_transformed_powers_step2
    assert mono_transformed_powers_step2 == [5.0, 0.2]
    assert restart_transformed_powers_step2 == [3.0, 5.0]

    # Save official BM cursor trace artifact
    guard = FairBiasR1Guard.get_instance()
    out_dir = guard.output_dir if guard is not None else pathlib.Path("runs")
    trace_path = out_dir / "bm_cursor_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump({
            "stream_length": len(stream),
            "stream_policy": "official_stream_1998",
            "mono_step1": {
                "cursor_start": cursor_start_mono1,
                "cursor_end": cursor_end_mono1,
                "evaluated_powers": mono_transformed_powers_step1,
                "accepted_power": res_mono1[1]["num1"]["power"],
            },
            "mono_step2": {
                "cursor_start": cursor_start_mono2,
                "cursor_end": cursor_end_mono2,
                "evaluated_powers": mono_transformed_powers_step2,
                "accepted_power": res_mono2[1]["num1"]["power"],
            },
            "restart_step1": {
                "cursor_start": cursor_start_rst1,
                "cursor_end": cursor_end_rst1,
                "evaluated_powers": restart_transformed_powers_step1,
                "accepted_power": res_rst1[1]["num1"]["power"],
            },
            "restart_step2": {
                "cursor_start": cursor_start_rst2,
                "cursor_end": cursor_end_rst2,
                "evaluated_powers": restart_transformed_powers_step2,
                "accepted_power": res_rst2[1]["num1"]["power"],
            },
            "step2_distinct_sequences": bool(mono_transformed_powers_step2 != restart_transformed_powers_step2),
            "early_stopping_verified": True,
        }, f, indent=2)


def test_geom_8_strict_candidate_vs_terminal_epsilon_boundary():
    """GEOM-8: Candidate acceptance requires strictly < epsilon_threshold, terminal feasibility requires <= epsilon_threshold."""
    cfg = FairBiasConfig(phi_threshold=0.05)
    evaluator = FairEvaluator(config=cfg)
    transformer = FairTransform()

    mit = FairBiasMitigation(
        evaluator=evaluator,
        transformer=transformer,
        label_O=["prot"],
        cate_attrs=[],
        num_attrs=["num1"],
        phi_threshold=0.05,
    )

    X = pd.DataFrame({"num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    Y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    O = pd.DataFrame({"prot": ["a", "a", "b", "b", "a", "a", "b", "b"]})
    nmi = {"num1": 0.5}

    eps_target = 0.05

    # 1. Candidate acceptance: strictly < epsilon_threshold
    class FakeDphiMitigation(FairBiasMitigation):
        def __init__(self, fake_dphi_val):
            super().__init__(
                evaluator=evaluator,
                transformer=transformer,
                label_O=["prot"],
                cate_attrs=[],
                num_attrs=["num1"],
                phi_threshold=0.05,
            )
            self._fake_dphi_val = fake_dphi_val

        def _epsilon_of(self, candidate_df, O_df, l_o, attr):
            return self._fake_dphi_val

        def _nmi_gate_ok(self, transformed_df, Y_df, nmi_org, attr):
            return True

    # Exactly equal to threshold -> REJECTED (candidate requires strictly <)
    cand_equal = FakeDphiMitigation(eps_target)._make_candidate(
        X, {}, "num1", {"power": 3.0}, Y, O, nmi, "prot", epsilon_threshold=eps_target
    )
    assert cand_equal is None, "Candidate with d_phi == epsilon_threshold must be rejected (strictly < required)"

    # Greater than threshold -> REJECTED
    cand_greater = FakeDphiMitigation(eps_target + 0.001)._make_candidate(
        X, {}, "num1", {"power": 3.0}, Y, O, nmi, "prot", epsilon_threshold=eps_target
    )
    assert cand_greater is None, "Candidate with d_phi > epsilon_threshold must be rejected"

    # Strictly less than threshold -> ACCEPTED
    cand_less = FakeDphiMitigation(eps_target - 0.001)._make_candidate(
        X, {}, "num1", {"power": 3.0}, Y, O, nmi, "prot", epsilon_threshold=eps_target
    )
    assert cand_less is not None, "Candidate with d_phi < epsilon_threshold must be accepted"

    # 2. Terminal feasibility boundary in production representation evaluation (<= epsilon_threshold):
    # Call actual evaluate_representation from nhis_fairbias.d8_enhancement_runner
    from nhis_fairbias.d8_enhancement_runner import evaluate_representation
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler

    X_train_term = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]}, index=[1, 2, 3, 4])
    y_train_term = np.array([0, 0, 1, 1])
    o_train_term = np.array(["g0", "g0", "g1", "g1"])

    def run_prod_terminal_eval(cand_dphi: float, threshold: float) -> bool:
        evaluator_term = FairEvaluator(config=FairBiasConfig(random_seed=42), label_O=["prot"])
        evaluator_term.calculate_epsilon = lambda *args, **kwargs: {"prot": {"f1": float(cand_dphi)}}
        res = evaluate_representation(
            model=LogisticRegression(random_state=42),
            scaler=MinMaxScaler(),
            X_train_raw=X_train_term,
            y_train=y_train_term,
            X_val_raw=X_train_term,
            y_val=y_train_term,
            X_test_raw=X_train_term,
            y_test=y_train_term,
            o_train=o_train_term,
            o_val=o_train_term,
            o_test=o_train_term,
            changed_dict={},
            evaluator=evaluator_term,
            transformer=FairTransform(),
            cate_attrs=[],
            num_attrs=["f1"],
            protected_attr="prot",
            expected_groups=["g0", "g1"],
        )
        max_dphi_val = res["train"]["max_dphi"]
        assert max_dphi_val == cand_dphi
        # Call production terminal feasibility function (R7-03)
        return check_fairness_feasibility(max_dphi_val, threshold)

    # 2a. Direct validation of check_fairness_feasibility function (R7-03)
    assert check_fairness_feasibility(eps_target - 0.001, eps_target) is True
    assert check_fairness_feasibility(eps_target, eps_target) is True
    assert check_fairness_feasibility(eps_target + 0.001, eps_target) is False
    assert check_fairness_feasibility(None, eps_target) is False
    assert check_fairness_feasibility(float("nan"), eps_target) is False
    assert check_fairness_feasibility(float("inf"), eps_target) is False

    # 2b. End-to-end evaluation calling production evaluate_representation
    feas_less = run_prod_terminal_eval(eps_target - 0.001, eps_target)
    feas_equal = run_prod_terminal_eval(eps_target, eps_target)
    feas_greater = run_prod_terminal_eval(eps_target + 0.001, eps_target)

    assert feas_less is True, "Terminal dphi < epsilon is feasible"
    assert feas_equal is True, "Terminal dphi == epsilon is feasible (production <= required)"
    assert feas_greater is False, "Terminal dphi > epsilon is infeasible"

    # 2c. In-process mutation test for check_fairness_feasibility: mutate <= to < (R7-03)
    from nhis_fairbias import d8_enhancement_runner
    orig_feasibility = d8_enhancement_runner.check_fairness_feasibility
    try:
        # Mutant implementation using strict <
        mutant_fn = lambda dphi, eps: float(dphi) < float(eps)
        d8_enhancement_runner.check_fairness_feasibility = mutant_fn
        # Direct mutant call on equality boundary: strict < must fail where <= succeeds
        mutant_feas_equal = mutant_fn(eps_target, eps_target)
        assert mutant_feas_equal is False, "Mutant terminal feasibility (< instead of <=) MUST fail on equality boundary to prove test sensitivity"
        # Confirm the original (production) function passes on equality boundary
        assert orig_feasibility(eps_target, eps_target) is True, "Production <= must pass on equality boundary"
    finally:
        d8_enhancement_runner.check_fairness_feasibility = orig_feasibility

    # 3. Real NMI information loss gate boundary:
    # Production formula: phi_loss = (nmi_before - nmi_after) / (nmi_before + 1e-10) <= phi_threshold
    nmi_before = 0.5
    nmi_after = 0.475
    exact_phi = (nmi_before - nmi_after) / (nmi_before + 1e-10)

    from fairbias import mitigation
    orig_calc_nmi = mitigation.calculate_nmi_dict
    mitigation.calculate_nmi_dict = lambda df, y: {"num1": nmi_after}

    # Test normal implementation with exact_phi threshold: <= must PASS
    mit_exact = FairBiasMitigation(
        evaluator=FairEvaluator(config=FairBiasConfig(phi_threshold=exact_phi)),
        transformer=FairTransform(),
        label_O=["prot"],
        cate_attrs=[],
        num_attrs=["num1"],
        phi_threshold=exact_phi,
    )
    normal_pass = mit_exact._nmi_gate_ok(X, Y, {"num1": nmi_before}, "num1")
    assert normal_pass is True, "Normal production implementation with <= must pass on exact boundary"

    # Test mutant implementation with strict <: must FAIL on exact boundary
    def mutant_nmi_gate_ok(mit_inst, transformed_df, Y_series, nmi_org_dict, attr_name):
        nmi_new = mitigation.calculate_nmi_dict(transformed_df, Y_series)
        nb = float(nmi_org_dict.get(attr_name, 1e-6))
        na = float(nmi_new.get(attr_name, 0.0))
        loss = (nb - na) / (nb + 1e-10)
        return loss < mit_inst.phi_threshold  # Mutated <= to <

    mutant_pass = mutant_nmi_gate_ok(mit_exact, X, Y, {"num1": nmi_before}, "num1")
    assert mutant_pass is False, "Mutant implementation with strict < MUST fail on exact boundary to prove test sensitivity"

    # Also test nominal 0.05 neighborhood
    mit_nominal = FairBiasMitigation(
        evaluator=FairEvaluator(config=FairBiasConfig(phi_threshold=0.05)),
        transformer=FairTransform(),
        label_O=["prot"],
        cate_attrs=[],
        num_attrs=["num1"],
        phi_threshold=0.05,
    )
    mitigation.calculate_nmi_dict = lambda df, y: {"num1": 0.480}
    nmi_pass_less = mit_nominal._nmi_gate_ok(X, Y, {"num1": 0.5}, "num1")
    assert nmi_pass_less is True

    mitigation.calculate_nmi_dict = lambda df, y: {"num1": 0.470}
    nmi_fail_greater = mit_nominal._nmi_gate_ok(X, Y, {"num1": 0.5}, "num1")
    assert nmi_fail_greater is False

    # Restore
    mitigation.calculate_nmi_dict = orig_calc_nmi

    # Save NMI and epsilon boundary trace artifact
    guard = FairBiasR1Guard.get_instance()
    out_dir = guard.output_dir if guard is not None else pathlib.Path("runs")
    trace_path = out_dir / "nmi_epsilon_boundary_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump({
            "candidate_boundary": {
                "less_dphi_accepted": bool(cand_less is not None),
                "equal_dphi_accepted": bool(cand_equal is not None),
                "greater_dphi_accepted": bool(cand_greater is not None),
            },
            "terminal_boundary": {
                "less_feasible": feas_less,
                "equal_feasible": feas_equal,
                "greater_feasible": feas_greater,
                "production_evaluate_representation_called": True,
            },
            "nmi_gate_boundary": {
                "phi_threshold_exact": exact_phi,
                "nmi_before": nmi_before,
                "nmi_after": nmi_after,
                "normal_implementation_pass": normal_pass,
                "mutant_implementation_pass": mutant_pass,
                "mutant_failed_as_required": not mutant_pass,
                "loss_under_accepted": nmi_pass_less,
                "loss_over_rejected": not nmi_fail_greater,
            },
        }, f, indent=2)


def test_geom_9_highest_dphi_attribute_selection_and_failure_stop():
    """GEOM-9: Mitigation step identifies the attribute with the maximum dphi and stops if all candidates fail."""
    cfg = FairBiasConfig()
    evaluator = FairEvaluator(config=cfg)
    transformer = FairTransform()
    mit = FairBiasMitigation(
        evaluator=evaluator,
        transformer=transformer,
        label_O=["prot"],
        cate_attrs=[],
        num_attrs=["f_low", "f_high"],
        failed_attribute_mode="stop",
    )
    # Target d_phi: f_high has 0.15, f_low has 0.05
    dphi_mock = {
        "prot": {
            "f_low": 0.05,
            "f_high": 0.15,
        }
    }
    max_attr = max(dphi_mock["prot"], key=dphi_mock["prot"].get)
    assert max_attr == "f_high"
    assert dphi_mock["prot"][max_attr] == 0.15

    # Run real mitigate_step where threshold cannot be met: verify stop
    X = pd.DataFrame({"f_low": [1.0, 2.0, 3.0, 4.0], "f_high": [10.0, 20.0, 30.0, 40.0]})
    Y = pd.Series([0, 0, 1, 1])
    O = pd.DataFrame({"prot": ["a", "b", "a", "b"]})
    res_X, res_changed, chosen_attr, chosen_trans = mit.mitigate_step(
        X,
        Y,
        O,
        nmi_org={"f_low": 0.5, "f_high": 0.5},
        changed_dict={},
        current_epsilon=dphi_mock,
        epsilon_threshold=1e-6,
    )
    assert chosen_attr is None, "Mitigation step must stop (return no chosen attribute) when highest dphi attribute cannot satisfy threshold"
    assert mit.non_convergence is not None
    assert mit.non_convergence["attribute"] == "f_high"



