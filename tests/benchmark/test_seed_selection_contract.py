from nhis_fairbias.benchmark.experiment_selection import aggregate_configuration, select_aggregate


def test_incomplete_seed_set_cannot_win_and_mean_gap_is_not_gap_of_ensemble():
    config = {'candidate_id': 'A', 'seeds': [0, 7], 'complexity': 1.}
    record = lambda seed, ba, gap: {'seed': seed, 'status': 'VALID', 'metrics_S': {'balanced_accuracy': ba, 'eo_gap': gap, 'dp_gap': gap}, 'risk_S': {'average_precision': .7}}
    one = aggregate_configuration(config, [record(0, .99, .01)])
    assert one['status'] == 'INCOMPLETE_SEEDS'
    two = aggregate_configuration(config, [record(0, .9, .2), record(7, .7, .2)])
    assert two['eo_gap'] == .2  # opposite signed disparities could cancel in mean q
    assert two['balanced_accuracy'] == .8
    result = select_aggregate([one, two], .1)
    assert result['status'] == 'NO_FEASIBLE_CONFIGURATION'
    assert result['selected_candidate_id'] is None
    assert result['boundary_candidate_id'] == 'A'
