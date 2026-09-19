"""Generated-only production FairBias recovery fit/cache/reload and rejection."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace, ModuleType

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import FunctionTransformer

from nhis_fairbias.benchmark.data_contracts import PartitionDataset
from nhis_fairbias.benchmark import experiment_worker
from nhis_fairbias.benchmark.adapters.adapter_fairbias import FairBiasAdapter
from nhis_fairbias.benchmark.adapters.adapter_joint_ae_recovery import JointAERecoveryAdapter


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('fairbias_recovery_worker_tested', ROOT / 'scripts/run_nhis_fairbias_recovery_worker.py')
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def _part(role, n, year):
    frame = pd.DataFrame({'cat': ['a'] * 12 + ['b'] * 4 + ['c'] * 4 + ['d'] * 4
                          + ['a'] * 4 + ['b'] * 4 + ['c'] * 4 + ['d'] * 12,
                          'agep_a': np.tile([0.0, 1.0], 24)})[:n]
    return PartitionDataset(role=role, year=year, record_keys=np.array([role + ':' + str(i) for i in range(n)]),
        X_semantic=frame, y=np.tile([0, 1], n // 2), A=np.array(([1] * 24 + [2] * 24)[:n]),
        WTFA_A=np.tile([1., 3.], n // 2), PSTRAT=np.ones(n, dtype=int), PPSU=np.arange(n),
        feature_names=tuple(frame.columns), arm_id='arm_synthetic', metadata={'expected_categories': [1, 2]})


@pytest.fixture
def case(tmp_path, monkeypatch, request):
    for name, value in recovery.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    # PYTHONHASHSEED must genuinely be set at interpreter startup; run this
    # suite with PYTHONHASHSEED=0, just like the deployed worker CLI.
    assert sys.flags.hash_randomization == 0, 'Run generated tests with PYTHONHASHSEED=0'
    monkeypatch.setattr(sys, 'path', list(sys.path))
    options = getattr(request, 'param', {})
    policy = options.get('policy', 'joint_strict_exact_v1')
    method = recovery.POLICIES[policy]['method']
    original = tmp_path / 'original'
    prepared = original / 'prepared' / 'arm_synthetic.joblib'
    prepared.parent.mkdir(parents=True)
    parts = {r: _part(r, n, y) for r, n, y in [('fitting_F', 48, 2022), ('calibration_C', 48, 2022), ('selection_S', 32, 2023)]}
    data = {'data_identity': 'generated_fairbias_only', 'partitions': parts,
            **{'X_' + r[0].upper(): np.zeros((len(p), 1)) for r, p in parts.items()},
            'preprocessor': FunctionTransformer().fit(np.zeros((48, 1)))}
    # Production semantic FairBias consumes the semantic frame, not these arrays.
    data.update(X_F=np.zeros((48, 1)), X_C=np.zeros((48, 1)), X_S=np.zeros((32, 1)))
    joblib.dump(data, prepared)
    old_cache = original / 'representation_cache'
    old_cache.mkdir()
    (old_cache / 'preserved').write_bytes(b'original cache')
    configs = []
    for c in ([1., 2.] if method == 'FAIRBIAS_BM' else [1.]):
        params = ({'epsilon_ratio': .75, 'max_iterations': 50, 'max_geometry_evaluations': 100, 'C': c}
                  if method == 'FAIRBIAS_BM' else
                  {'mode': 'JOINT', 'epsilon_ratio': .75, 'max_geometry_evaluations': 100,
                   'max_utility_evaluations': 100, 'max_bm_steps': 50, 'max_outer_iterations': 10,
                   'estimator_params': {'C': c}})
        params.update(options.get('params', {}))
        config = {'method': method, 'backbone': 'LR', 'params': params, 'arm_id': 'arm_synthetic',
                  'seeds': [0], 'status': 'REGISTERED', 'training_weighted': method == 'FAIRBIAS_BM' and c == 1.}
        config['candidate_id'] = recovery.identity(config)[:20]
        configs.append(config)
    registered = ['src/nhis_fairbias/benchmark/experiment_worker.py',
                  'src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py',
                  'src/nhis_fairbias/benchmark/adapters/adapter_fairbias_ae.py']
    registration = {'candidates': configs, 'source_identity': 'generated_original_sources',
        'source_files': {p: recovery.file_sha(ROOT / p) for p in registered},
        'prepared': {'arm_synthetic': {'path': str(prepared), 'sha256': recovery.file_sha(prepared),
                                     'data_identity': data['data_identity']}}}
    regpath = original / 'registration.json'
    regpath.write_text(json.dumps(registration))
    files = recovery.build_runtime_source_manifest()
    runtime_id = recovery.runtime_identity(files, policy=policy)
    run = tmp_path / ('recovery_' + runtime_id)
    jobs = []
    for config in configs:
        name = config['candidate_id'] + '_s0'
        old = original / 'jobs' / name / 'job.json'
        old.parent.mkdir(parents=True)
        base = {'config': config, 'seed': 0, 'source_identity': registration['source_identity'],
                'data_identity': data['data_identity'], 'data_sha256': recovery.file_sha(prepared),
                'data_path': str(prepared), 'cache_path': str(old_cache)}
        old.write_text(json.dumps(base))
        previous = old.parent / 'result.json'
        previous.write_text(json.dumps({'candidate_id': config['candidate_id'], 'seed': 0,
            'status': recovery.POLICIES[policy]['original_statuses'][0]}))
        jobpath = run / 'jobs' / name / 'job.json'
        jobpath.parent.mkdir(parents=True)
        payload = {**base, 'cache_path': str(run / 'representation_cache'), 'recovery_run_path': str(run),
            'registration_path': str(regpath), 'registration_sha256': recovery.file_sha(regpath),
            'original_job_path': str(old), 'original_job_sha256': recovery.file_sha(old),
            'original_result_sha256': recovery.file_sha(previous), 'runtime_variant': recovery.VARIANT,
            'runtime_source_files': files, 'runtime_environment': recovery.THREAD_ENVIRONMENT,
            'runtime_source_identity': runtime_id, 'recovery_policy': policy,
            'execution_budget': recovery.DEFAULT_EXECUTION_BUDGET}
        jobpath.write_text(json.dumps(payload))
        jobs.append(jobpath)
    snapshot = {str(p.relative_to(original)): recovery.file_sha(p) for p in original.rglob('*') if p.is_file()}
    return SimpleNamespace(original=original, prepared=prepared, data=data, jobs=jobs, run=run,
                           original_cache=old_cache, files=files, before=snapshot, policy=policy)


def _rewrite(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def _assert_artifacts(case, path, receipt):
    assert receipt['status'] == 'VALID' and receipt['runtime_checks_passed'], receipt
    assert receipt['reload_exact_C'] and receipt['recovery_policy'] == case.policy
    assert receipt['execution_budget'] == recovery.DEFAULT_EXECUTION_BUDGET
    assert json.loads((path.parent / 'runtime_receipt.json').read_text()) == receipt
    result = json.loads((path.parent / 'result.json').read_text())
    assert result['reload_verified'] and result['status'] == 'VALID'
    for name, digest in receipt['files'].items():
        assert recovery.file_sha(path.parent / name) == digest
    artifact = joblib.load(path.parent / 'model.joblib')
    S = case.data['partitions']['selection_S']
    actual = artifact['policy'].predict(S.X_semantic, S.A)
    with np.load(path.parent / 'predictions_S.npz') as expected:
        np.testing.assert_array_equal(actual.q_decision, expected['q'])
        np.testing.assert_array_equal(actual.p_event, expected['p'])
    assert case.before == {str(p.relative_to(case.original)): recovery.file_sha(p)
                           for p in case.original.rglob('*') if p.is_file()}
    return artifact


def test_actual_strict_joint_production_fit_and_separate_process_reload(case):
    factory = experiment_worker.make_adapter
    calibration = experiment_worker.FrozenDecisionPolicy.fit_calibration
    receipt = recovery.execute_recovery_job(case.jobs[0])
    artifact = _assert_artifacts(case, case.jobs[0], receipt)
    assert receipt['model_admission']['status'] == 'COMPLETE_FEASIBLE'
    assert not receipt['representation_cache_reused'] and receipt['representation_key'] is None
    assert experiment_worker.make_adapter is factory
    assert experiment_worker.FrozenDecisionPolicy.fit_calibration is calibration
    model_path = case.jobs[0].parent / 'model.joblib'
    C = case.data['partitions']['calibration_C']
    probe = case.run / 'generated_C_probe.joblib'
    joblib.dump((C.X_semantic, C.A, artifact['policy'].predict(C.X_semantic, C.A)), probe)
    child = subprocess.run([sys.executable, '-B', '-c',
        'import sys,joblib,numpy as np; a=joblib.load(sys.argv[1]); X,A,p=joblib.load(sys.argv[2]); '
        'q=a["policy"].predict(X,A); assert np.array_equal(q.q_decision,p.q_decision); '
        'assert np.array_equal(q.p_event,p.p_event)', str(model_path), str(probe)],
        env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr


@pytest.mark.parametrize('case', [{'policy': 'bm_mds_retry_v1'}], indirect=True)
def test_actual_weighted_bm_predictors_share_new_runtime_representation(case, monkeypatch):
    original = FairBiasAdapter.fit_predictor
    weights = []
    def fit(self, X, y, sample_weight=None):
        weights.append(None if sample_weight is None else np.array(sample_weight, copy=True))
        return original(self, X, y, sample_weight=sample_weight)
    monkeypatch.setattr(FairBiasAdapter, 'fit_predictor', fit)
    first = recovery.execute_recovery_job(case.jobs[0])
    second = recovery.execute_recovery_job(case.jobs[1])
    models = [_assert_artifacts(case, p, r) for p, r in zip(case.jobs, (first, second))]
    assert not first['representation_cache_reused'] and second['representation_cache_reused']
    assert first['representation_key'] == second['representation_key']
    assert [x['policy']._adapter.C for x in models] == [1., 2.]
    expected = case.data['partitions']['fitting_F'].WTFA_A / 2
    assert len(weights) == 2
    np.testing.assert_array_equal(weights[0], expected)
    assert weights[1] is None, 'Second predictor preserves its original unweighted config'
    job = json.loads(case.jobs[0].read_text())
    assert recovery.verify_cache_entry(job, first['representation_key'])


@pytest.mark.parametrize('case', [{'params': {'max_geometry_evaluations': 1}},
    {'policy': 'bm_mds_retry_v1', 'params': {'max_iterations': 0}}], indirect=True)
def test_actual_budget_exit_is_never_valid_and_failed_cache_is_bound(case):
    receipt = recovery.execute_recovery_job(case.jobs[0])
    assert receipt['status'] == 'BUDGET_EXHAUSTED' and receipt['runtime_checks_passed']
    assert not (case.jobs[0].parent / 'model.joblib').exists()
    if receipt['representation_key']:
        job = json.loads(case.jobs[0].read_text())
        assert recovery.verify_cache_entry(job, receipt['representation_key'])
        assert set(receipt['representation_cache_evidence']['files']) == {'.json'}
        second = recovery.execute_recovery_job(case.jobs[1])
        assert second['status'] == 'BUDGET_EXHAUSTED' and second['representation_cache_reused']


@pytest.mark.parametrize('bad', ['blocked_policy', 'unknown_policy', 'method', 'weighted_joint', 'budget',
    'missing_budget', 'runtime_hash', 'source_hash', 'source_missing', 'data_hash', 'data_path',
    'job_hash', 'result_hash', 'registration_hash', 'old_cache', 'old_run', 'seed', 'config', 'output'])
def test_invalid_contract_rejected_before_fit(case, monkeypatch, bad):
    path = case.jobs[0]
    changes = {
        'blocked_policy': lambda j: j.update(recovery_policy='bmae_cap40_v1'),
        'unknown_policy': lambda j: j.update(recovery_policy='scheduled_joint'),
        'method': lambda j: j['config'].update(method='FAIRBIAS_BM_AE'),
        'weighted_joint': lambda j: j['config'].update(training_weighted=True),
        'budget': lambda j: j['execution_budget'].update(fit_seconds=3600),
        'missing_budget': lambda j: j.pop('execution_budget'),
        'runtime_hash': lambda j: j.update(runtime_source_identity='0'*64),
        'source_hash': lambda j: j['runtime_source_files'].update({recovery.WORKER_SOURCE: '0'*64}),
        'source_missing': lambda j: j['runtime_source_files'].pop(recovery.PRIMITIVES_SOURCE),
        'data_hash': lambda j: j.update(data_sha256='0'*64),
        'data_path': lambda j: j.update(data_path=str(path.parent / 'other.joblib')),
        'job_hash': lambda j: j.update(original_job_sha256='0'*64),
        'result_hash': lambda j: j.update(original_result_sha256='0'*64),
        'registration_hash': lambda j: j.update(registration_sha256='0'*64),
        'old_cache': lambda j: j.update(cache_path=str(case.original_cache)),
        'old_run': lambda j: j.update(recovery_run_path=str(case.original)),
        'seed': lambda j: j.update(seed=1),
        'config': lambda j: j['config']['params'].update(epsilon_ratio=3),
    }
    if bad == 'output':
        (path.parent / 'result.json').write_text('preserve existing result')
    else:
        _rewrite(path, changes[bad])
    monkeypatch.setattr(experiment_worker, 'execute_job', lambda *a: pytest.fail('Invalid job reached fit'))
    with pytest.raises((ValueError, FileNotFoundError)):
        recovery.execute_recovery_job(path)
    assert not (path.parent / 'runtime_started.json').exists()


def test_legacy_valid_and_wrong_original_failure_status_are_refused(case):
    job = json.loads(case.jobs[0].read_text())
    previous = Path(job['original_job_path']).parent / 'result.json'
    for status in ('VALID', 'BUDGET_EXHAUSTED'):
        _rewrite(previous, lambda r: r.update(status=status))
        _rewrite(case.jobs[0], lambda j: j.update(original_result_sha256=recovery.file_sha(previous)))
        with pytest.raises(ValueError, match='outside this recovery policy'):
            recovery.validate_job(case.jobs[0])


@pytest.mark.parametrize('fault', ['incumbent', 'empty_features'])
def test_strict_guard_refuses_returned_incumbent_or_empty_representation(case, monkeypatch, fault):
    fit = JointAERecoveryAdapter.fit_development
    def corrupt(self, *args, **kwargs):
        fit(self, *args, **kwargs)
        if fault == 'incumbent':
            self.recovery_result_['status'] = 'FEASIBLE_BUDGET_LIMITED'
        else:
            self._delegate.preprocessor_.feature_names = ()
        return self
    monkeypatch.setattr(JointAERecoveryAdapter, 'fit_development', corrupt)
    receipt = recovery.execute_recovery_job(case.jobs[0])
    assert receipt['status'] == 'FAILED' and receipt['runtime_checks_passed']
    assert not (case.jobs[0].parent / 'model.joblib').exists()
    assert not (case.jobs[0].parent / 'predictions_S.npz').exists()


@pytest.mark.parametrize('case', [{'policy': 'bm_mds_retry_v1'}], indirect=True)
def test_invalid_cached_bm_model_never_admitted_even_with_matching_sidecar(case):
    first = recovery.execute_recovery_job(case.jobs[0])
    assert first['status'] == 'VALID'
    job = json.loads(case.jobs[1].read_text())
    key, cache = first['representation_key'], Path(job['cache_path'])
    modelpath = cache / (key + '.joblib')
    adapter = joblib.load(modelpath)
    adapter.converged_ = False
    joblib.dump(adapter, modelpath)
    _rewrite(cache / (key + '.json'), lambda r: r.update(model_sha256=recovery.file_sha(modelpath)))
    (cache / (key + '.runtime.json')).write_text(json.dumps(recovery._cache_evidence(job, key)))
    second = recovery.execute_recovery_job(case.jobs[1])
    assert second['status'] == 'FAILED' and second['runtime_checks_passed']
    assert not (case.jobs[1].parent / 'model.joblib').exists()


def test_shadow_project_module_rejected_before_worker(case, monkeypatch, tmp_path):
    shadow = ModuleType('fairbias.unlisted_test_shadow')
    shadow.__file__ = str(tmp_path / 'shadow.py')
    monkeypatch.setitem(sys.modules, shadow.__name__, shadow)
    with pytest.raises(ValueError, match='outside source tree'):
        recovery.execute_recovery_job(case.jobs[0])


def test_policy_and_external_budget_are_runtime_identity_not_silent_options(case):
    before = recovery.runtime_identity(case.files, policy='joint_strict_exact_v1')
    assert before != recovery.runtime_identity(case.files, policy='bm_mds_retry_v1')
    assert before != recovery.runtime_identity(case.files, policy='joint_strict_exact_v1',
        execution_budget={'fit_seconds': 3600, 'worker_rss_bytes': 4*1024**3})


def test_import_does_not_mutate_numerical_worker_namespace():
    path = ROOT / recovery.PRIMITIVES_SOURCE
    spec = importlib.util.spec_from_file_location('unmodified_numerical_worker_tested', path)
    numerical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(numerical)
    files = numerical.build_runtime_source_manifest()
    before = numerical.runtime_identity(files)
    second_spec = importlib.util.spec_from_file_location('second_fairbias_worker_tested', ROOT / recovery.WORKER_SOURCE)
    second = importlib.util.module_from_spec(second_spec)
    second_spec.loader.exec_module(second)
    assert numerical.VARIANT == 'eg_lfr_numerical_recovery_v1_20260917'
    assert numerical.runtime_identity(files) == before
    assert numerical._cache_namespace({'runtime_source_identity': 'r', 'registration_sha256': 'g'}) == {
        'runtime_variant': numerical.VARIANT, 'runtime_source_identity': 'r', 'registration_sha256': 'g'}
    assert second._base is not numerical and second._base is not recovery._base


@pytest.mark.parametrize('case', [{'policy': 'bm_mds_retry_v1'}], indirect=True)
def test_bm_cache_tamper_and_repeat_output_are_rejected(case):
    first = recovery.execute_recovery_job(case.jobs[0])
    assert first['status'] == 'VALID'
    snapshot = {p.name: recovery.file_sha(p) for p in case.jobs[0].parent.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match='already has output'):
        recovery.execute_recovery_job(case.jobs[0])
    assert snapshot == {p.name: recovery.file_sha(p) for p in case.jobs[0].parent.iterdir() if p.is_file()}
    cache = case.run / 'representation_cache'
    with (cache / (first['representation_key'] + '.joblib')).open('ab') as stream:
        stream.write(b'tampered bytes')
    with pytest.raises(ValueError, match='cache integrity mismatch'):
        recovery.execute_recovery_job(case.jobs[1])


def test_refuses_unbound_cache_contents(case):
    cache = case.run / 'representation_cache'
    cache.mkdir()
    (cache / 'old.joblib').write_bytes(b'old scientific cache')
    with pytest.raises(ValueError, match='without runtime provenance'):
        recovery.execute_recovery_job(case.jobs[0])


def test_source_failure_after_worker_valid_cannot_be_receipt_valid(case, monkeypatch):
    original_validate = recovery.validate_runtime_sources
    calls = []
    def validate(job):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError('synthetic post-fit source change')
        return original_validate(job)
    monkeypatch.setattr(recovery, 'validate_runtime_sources', validate)
    receipt = recovery.execute_recovery_job(case.jobs[0])
    assert receipt['worker_result_status'] == 'VALID'
    assert receipt['status'] == 'RUNTIME_FAILED' and not receipt['runtime_checks_passed']
    assert receipt['files']['result.json'] == recovery.file_sha(case.jobs[0].parent / 'result.json')


def test_actual_cli_job_uses_startup_hash_seed_and_persists_receipt(case):
    child = subprocess.run([sys.executable, '-B', str(ROOT / recovery.WORKER_SOURCE), 'worker',
                            '--job', str(case.jobs[0])], env={**os.environ, **recovery.THREAD_ENVIRONMENT,
                            'PYTHONPATH': str(ROOT / 'src')}, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr[-3000:] + child.stdout[-1000:]
    receipt = json.loads((case.jobs[0].parent / 'runtime_receipt.json').read_text())
    _assert_artifacts(case, case.jobs[0], receipt)
