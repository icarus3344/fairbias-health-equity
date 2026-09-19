"""Synthetic completion-policy tests; no study data or remote execution."""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.manifold import MDS

from fairbias import bias_metric
from nhis_fairbias.benchmark.fairbias_completion import adaptive_mds_completion, FairBiasCompletionAdapter
from nhis_fairbias.benchmark.adapters.adapter_fairbias_ae import FairBiasAEAdapter
from nhis_fairbias.benchmark.adapters.geometry_audit import audited_mds
from test_joint_ae_recovery import data


def test_adaptive_mds_retains_all_failed_caps_and_matches_fresh_final_budget(tmp_path):
    D = np.array([[0., 1., 1.], [1., 0., np.sqrt(2.)], [1., np.sqrt(2.), 0.]])
    params = dict(n_components=2, n_init=4, random_state=7, dissimilarity='precomputed',
                  max_iter=1, eps=1e6, normalized_stress='auto')
    original = bias_metric.MDS
    before = np.random.get_state()
    events = tmp_path / 'progress.jsonl'
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        with adaptive_mds_completion(progress_path=events) as stats:
            with audited_mds(records, lambda: 1):
                model = bias_metric.MDS(**params)
                actual = model.fit_transform(D)
        expected = MDS(**{**params, 'max_iter': model.max_iter})
        expected_points = expected.fit_transform(D)
    assert model.max_iter == 4
    assert [r['actual_cap'] for r in stats['attempts']] == [1, 2, 4]
    assert actual.tobytes() == expected_points.tobytes()
    assert model.stress_ == expected.stress_ and model.n_iter_ == expected.n_iter_
    assert len(records) == 1 and records[0]['status'] == 'CONVERGED_BEFORE_CAP'
    assert len(events.read_text().splitlines()) == 2
    assert bias_metric.MDS is original
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_mds_interrupt_restores_and_does_not_invent_completion(monkeypatch):
    original = bias_metric.MDS
    def interrupt(*a, **k): raise KeyboardInterrupt()
    monkeypatch.setattr(MDS, 'fit_transform', interrupt)
    with pytest.raises(KeyboardInterrupt):
        with adaptive_mds_completion() as stats:
            bias_metric.MDS(random_state=0).fit_transform(np.eye(3))
    assert not stats['attempts'] and bias_metric.MDS is original


@pytest.mark.parametrize('mode', ['BM_AE', 'JOINT'])
def test_real_search_replays_geometry_ceiling_then_matches_uncapped_reference(tmp_path, mode):
    # The fixture performs a genuine categorical merge. A geometry limit of1
    # cannot finish; the new policy must extend it without changing the search.
    args = dict(mode=mode, epsilon_ratio=.75, random_state=0, max_outer_iterations=40)
    baseline = FairBiasAEAdapter(**args, max_geometry_evaluations=100)
    model = FairBiasCompletionAdapter(**args, max_geometry_evaluations=1,
        cache_dir=tmp_path / 'cache', progress_path=tmp_path / 'progress.jsonl')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        baseline.fit_development(*data())
        model.fit_development(*data())
    assert model.is_fitted_ and model.provenance_['status'] == 'COMPLETE_FEASIBLE'
    assert len(model.attempts_) > 1
    assert model._delegate.changed_dict_ == baseline.changed_dict_ == {'cat': {'d': 'a'}}
    assert model._delegate.candidate_traces_ == baseline.candidate_traces_
    assert model.predict_event_probability(data()[3]).tobytes() == baseline.predict_event_probability(data()[3]).tobytes()
    assert all(r['status'] == 'REPLAY_REQUIRED' for r in model.attempts_[:-1])
    assert model.recovery_result_['convergence_verified']
    assert not model.recovery_result_['formal_study_admitted']
    json.dumps(model.provenance_, allow_nan=False)
    saved = tmp_path / 'model.joblib'
    joblib.dump(model, saved)
    assert joblib.load(saved).predict_event_probability(data()[3]).tobytes() == model.predict_event_probability(data()[3]).tobytes()


def test_session_allowance_is_suspended_not_a_finished_model(tmp_path):
    model = FairBiasCompletionAdapter(mode='JOINT', random_state=0, epsilon_ratio=.75,
        max_geometry_evaluations=1, max_replays_per_session=1, cache_dir=tmp_path / 'cache')
    with warnings.catch_warnings(), pytest.raises(RuntimeError, match='SUSPENDED_REPLAY_REQUIRED'):
        warnings.simplefilter('ignore', FutureWarning)
        model.fit_development(*data())
    assert not model.is_fitted_
    assert model.provenance_['status'] == 'SUSPENDED_REPLAY_REQUIRED'
    assert model.provenance_['next_limits']['max_geometry_evaluations'] == 2
    with pytest.raises(RuntimeError, match='no admitted fitted model'):
        model.predict(data()[3])


@pytest.mark.parametrize('mode', ['BM_AE', 'JOINT'])
def test_real_ae_commit_cap_replays_to_actual_stopping_condition(tmp_path, mode):
    rng = np.random.default_rng(2)
    X = pd.DataFrame({'agep_a': rng.uniform(-1, 1, 400), 'pcnt18uptc': rng.uniform(-1, 1, 400)})
    y = (X.agep_a ** 3 + .8 * X.pcnt18uptc > 0).to_numpy(dtype=int)
    A = pd.DataFrame({'A': np.tile([0, 1], 200)})
    args = (X.iloc[:240], y[:240], A.iloc[:240], X.iloc[240:], y[240:], A.iloc[240:])
    reference = FairBiasAEAdapter(mode=mode, random_state=0, epsilon_ratio=100, max_outer_iterations=10)
    completion = FairBiasCompletionAdapter(mode=mode, random_state=0, epsilon_ratio=100,
        max_outer_iterations=1, cache_dir=tmp_path / 'cache')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        reference.fit_development(*args)
        completion.fit_development(*args)
    assert completion.attempts_[0]['exhausted_limit'] == 'max_outer_iterations'
    assert completion.attempts_[0]['status'] == 'REPLAY_REQUIRED'
    assert completion._delegate.changed_dict_ == reference.changed_dict_ == {'pcnt18uptc': {'power': 1 / 3}}
    assert completion._delegate.candidate_traces_ == reference.candidate_traces_
    assert completion._delegate.termination_reason_ == 'STRICT_FEASIBLE_SEARCH_EXHAUSTED'
    assert completion.predict_event_probability(X).tobytes() == reference.predict_event_probability(X).tobytes()
    assert completion.provenance_['cache']['hits'] > 0
