"""Generated-only formal worker recovery, cache sharing and provenance guards."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import FunctionTransformer

from nhis_fairbias.benchmark.data_contracts import PartitionDataset
from nhis_fairbias.benchmark import experiment_worker
from nhis_fairbias.benchmark.adapters.adapter_lfr_recovery import LFRAnalyticRecoveryAdapter


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'numerical_recovery_worker_tested', ROOT / 'scripts/run_nhis_numerical_recovery_worker.py'
)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def _part(role, n, seed, year):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n, 6))
    y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)
    split = 17 if n == 48 else n // 2
    a = np.r_[np.ones(split, dtype=int), np.full(n - split, 2, dtype=int)]
    return PartitionDataset(
        role=role, year=year, record_keys=np.array([role + ':' + str(i) for i in range(n)]),
        X_semantic=pd.DataFrame(x), y=y, A=a, WTFA_A=np.ones(n),
        PSTRAT=np.ones(n, dtype=int), PPSU=np.arange(n), feature_names=tuple('f' + str(i) for i in range(6)),
        arm_id='arm_synthetic', metadata={'expected_categories': [1, 2]},
    ), x


@pytest.fixture
def case(tmp_path, monkeypatch, request):
    for key, value in recovery.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    original = tmp_path / 'original_registration'
    prepared = original / 'prepared' / 'arm_synthetic.joblib'
    prepared.parent.mkdir(parents=True)
    F, xf = _part('fitting_F', 48, 9, 2022)
    C, xc = _part('calibration_C', 24, 19, 2022)
    S, xs = _part('selection_S', 32, 29, 2023)
    data = {'data_identity': 'generated_worker_only',
            'partitions': {'fitting_F': F, 'calibration_C': C, 'selection_S': S},
            'X_F': xf, 'X_C': xc, 'X_S': xs,
            'preprocessor': FunctionTransformer().fit(xf)}
    joblib.dump(data, prepared)
    original_cache = original / 'representation_cache'
    original_cache.mkdir()
    (original_cache / 'untouched.txt').write_text('historical cache remains unchanged')
    configs = []
    for method, c in [('LFR_RECONSTRUCTED', 1.0), ('LFR_RECONSTRUCTED', 2.0), ('EG_DP', 1.0)]:
        params = ({'k': 3, 'Ax': 0.01, 'Ay': 1.0, 'Az': 0.5, 'maxiter': 200, 'maxfun': 300, 'C': c}
                  if method == 'LFR_RECONSTRUCTED' else
                  {'constraint_type': 'dp', 'difference_bound': 0.2, 'eps': 0.01, 'max_iter': 3, 'C': c})
        if method == 'LFR_RECONSTRUCTED':
            params.update(getattr(request, 'param', {}))
        config = {'method': method, 'backbone': 'LR', 'params': params,
                  'arm_id': 'arm_synthetic', 'seeds': [7], 'status': 'REGISTERED', 'training_weighted': False}
        config['candidate_id'] = recovery.identity(config)[:20]
        configs.append(config)
    registered_sources = [
        'src/nhis_fairbias/benchmark/experiment_worker.py',
        'src/nhis_fairbias/benchmark/experiment_registry.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_lfr.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_reductions.py',
    ]
    registration = {'candidates': configs, 'source_identity': 'synthetic-original-source',
                    'source_files': {p: recovery.file_sha(ROOT / p) for p in registered_sources},
                    'prepared': {'arm_synthetic': {'sha256': recovery.file_sha(prepared),
                                                  'data_identity': data['data_identity']}}}
    registration_path = original / 'registration.json'
    registration_path.write_text(json.dumps(registration))
    files = recovery.build_runtime_source_manifest()
    runtime_id = recovery.runtime_identity(files)
    run = tmp_path / ('recovery_' + runtime_id)
    run.mkdir()
    jobs = []
    for config in configs:
        name = config['candidate_id'] + '_s7'
        original_job_path = original / 'jobs' / name / 'job.json'
        original_job_path.parent.mkdir(parents=True)
        base = {'config': config, 'seed': 7, 'source_identity': registration['source_identity'],
                'data_identity': data['data_identity'], 'data_sha256': recovery.file_sha(prepared),
                'data_path': str(prepared), 'cache_path': str(original_cache)}
        original_job_path.write_text(json.dumps(base))
        original_result = {'candidate_id': config['candidate_id'], 'seed': 7,
                           'status': 'FAILED' if config['method'].startswith('EG_') else 'BUDGET_EXHAUSTED'}
        original_result_path = original_job_path.parent / 'result.json'
        original_result_path.write_text(json.dumps(original_result))
        job_path = run / 'jobs' / name / 'job.json'
        job_path.parent.mkdir(parents=True)
        job = {**base, 'cache_path': str(run / 'representation_cache'), 'recovery_run_path': str(run),
               'registration_path': str(registration_path), 'registration_sha256': recovery.file_sha(registration_path),
               'original_job_path': str(original_job_path), 'original_job_sha256': recovery.file_sha(original_job_path),
               'original_result_sha256': recovery.file_sha(original_result_path),
               'runtime_variant': recovery.VARIANT, 'runtime_source_files': files,
               'runtime_environment': recovery.THREAD_ENVIRONMENT, 'runtime_source_identity': runtime_id}
        job_path.write_text(json.dumps(job))
        jobs.append(job_path)
    before = {str(p.relative_to(original)): recovery.file_sha(p) for p in original.rglob('*') if p.is_file()}
    return types.SimpleNamespace(original=original, run=run, jobs=jobs, data=data, prepared=prepared,
                                 files=files, runtime_id=runtime_id, before=before,
                                 registration_path=registration_path, original_cache=original_cache)


def _rewrite(path, mutate):
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))


def test_real_worker_lfr_fit_then_runtime_bound_cache_share_and_reload(case, monkeypatch):
    original_factory = experiment_worker.make_adapter
    real_fit = LFRAnalyticRecoveryAdapter.fit
    fitted = []

    def count_fit(self, *args, **kwargs):
        fitted.append(len(args[0]))
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(LFRAnalyticRecoveryAdapter, 'fit', count_fit)
    first = recovery.execute_recovery_job(case.jobs[0])
    second = recovery.execute_recovery_job(case.jobs[1])
    assert first['status'] == second['status'] == 'VALID', (first, second)
    assert fitted == [48], 'Second predictor should reuse only this runtime representation'
    assert first['representation_cache_reused'] is False
    assert second['representation_cache_reused'] is True
    assert first['representation_key'] == second['representation_key']
    assert experiment_worker.make_adapter is original_factory
    models = []
    for path, receipt in zip(case.jobs, (first, second)):
        out = path.parent
        record = json.loads((out / 'result.json').read_text())
        persisted = json.loads((out / 'runtime_receipt.json').read_text())
        assert persisted == receipt
        assert record['reload_verified'] and receipt['runtime_checks_passed']
        assert receipt['legacy_valid_results_admitted'] is False
        assert record['optimization']['recovery_variant'] == 'lfr_analytic_recovery_v1'
        assert record['sample_sizes'] == {'F': 48, 'C': 24, 'S': 32}
        assert receipt['loaded_project_modules_before'] and receipt['loaded_project_modules_after']
        for name, digest in receipt['files'].items():
            assert recovery.file_sha(out / name) == digest
        artifact = joblib.load(out / 'model.joblib')
        models.append(artifact['policy']._adapter)
        observed = artifact['policy'].predict(case.data['X_S'], case.data['partitions']['selection_S'].A)
        with np.load(out / 'predictions_S.npz') as arrays:
            np.testing.assert_array_equal(observed.q_decision, arrays['q'])
            np.testing.assert_array_equal(observed.p_event, arrays['p'])
    np.testing.assert_array_equal(models[0].lfr.learned_model, models[1].lfr.learned_model)
    assert models[0].C == 1.0 and models[1].C == 2.0
    current_original = {str(p.relative_to(case.original)): recovery.file_sha(p)
                        for p in case.original.rglob('*') if p.is_file()}
    assert current_original == case.before


def test_real_eg_worker_fit_reload_receipt_has_no_legacy_valid_admission(case):
    receipt = recovery.execute_recovery_job(case.jobs[2])
    assert receipt['status'] == 'VALID', receipt
    assert receipt['representation_key'] is None
    assert not receipt['legacy_valid_results_admitted']
    artifact = joblib.load(case.jobs[2].parent / 'model.joblib')
    adapter = artifact['policy']._adapter
    assert adapter.name.endswith('_NUMERICAL_RECOVERY')
    assert len(adapter.model.predictors_) > 0
    with np.load(case.jobs[2].parent / 'predictions_S.npz') as arrays:
        assert set(arrays.files) == {'q'}


@pytest.mark.parametrize('bad', [
    'variant', 'runtime_identity', 'source_missing', 'source_hash', 'environment',
    'registration_hash', 'original_job_hash', 'original_result_hash', 'data_hash',
    'candidate', 'weighted', 'method', 'old_cache', 'independent_wrong_cache', 'old_run',
    'result_exists', 'model_exists', 'started_exists',
])
def test_invalid_contracts_reject_before_frozen_worker(case, monkeypatch, bad):
    path = case.jobs[0]
    mutate = {
        'variant': lambda j: j.update(runtime_variant='unknown'),
        'runtime_identity': lambda j: j.update(runtime_source_identity='0' * 64),
        'source_missing': lambda j: j['runtime_source_files'].pop(recovery.WORKER_SOURCE),
        'source_hash': lambda j: j['runtime_source_files'].update({recovery.WORKER_SOURCE: '0' * 64}),
        'environment': lambda j: j.update(runtime_environment={}),
        'registration_hash': lambda j: j.update(registration_sha256='0' * 64),
        'original_job_hash': lambda j: j.update(original_job_sha256='0' * 64),
        'original_result_hash': lambda j: j.update(original_result_sha256='0' * 64),
        'data_hash': lambda j: j.update(data_sha256='0' * 64),
        'candidate': lambda j: j['config'].update(candidate_id='unregistered'),
        'weighted': lambda j: j['config'].update(training_weighted=True),
        'method': lambda j: j['config'].update(method='UNMITIGATED'),
        'old_cache': lambda j: j.update(cache_path=str(case.original_cache)),
        'independent_wrong_cache': lambda j: j.update(cache_path=str(case.run / 'some_other_cache')),
        'old_run': lambda j: j.update(recovery_run_path=str(case.original)),
    }
    if bad.endswith('_exists'):
        name = {'result_exists': 'result.json', 'model_exists': 'model.joblib',
                'started_exists': 'runtime_started.json'}[bad]
        (path.parent / name).write_text('old output must survive')
    else:
        _rewrite(path, mutate[bad])
    monkeypatch.setattr(experiment_worker, 'execute_job', lambda *a: pytest.fail('Invalid job reached frozen worker'))
    with pytest.raises(ValueError):
        recovery.execute_recovery_job(path)
    assert not (path.parent / 'runtime_receipt.json').exists()


def test_eg_existing_valid_result_is_not_admitted(case, monkeypatch):
    path = case.jobs[2]
    job = json.loads(path.read_text())
    previous = Path(job['original_job_path']).parent / 'result.json'
    _rewrite(previous, lambda p: p.update(status='VALID'))
    _rewrite(path, lambda j: j.update(original_result_sha256=recovery.file_sha(previous)))
    monkeypatch.setattr(experiment_worker, 'execute_job', lambda *a: pytest.fail('Legacy VALID was admitted'))
    with pytest.raises(ValueError, match='failed jobs only'):
        recovery.execute_recovery_job(path)


def test_existing_unbound_cache_contents_are_rejected(case, monkeypatch):
    cache = case.run / 'representation_cache'
    cache.mkdir()
    (cache / 'historical.joblib').write_bytes(b'not from this runtime')
    monkeypatch.setattr(experiment_worker, 'execute_job', lambda *a: pytest.fail('Old cache was admitted'))
    with pytest.raises(ValueError, match='without runtime provenance'):
        recovery.execute_recovery_job(case.jobs[0])
    assert (cache / 'historical.joblib').read_bytes() == b'not from this runtime'


@pytest.mark.parametrize('corruption', ['model', 'status', 'sidecar_missing', 'namespace'])
def test_shared_cache_rejects_tampered_or_unbound_artifacts(case, monkeypatch, corruption):
    first = recovery.execute_recovery_job(case.jobs[0])
    assert first['status'] == 'VALID'
    cache, key = case.run / 'representation_cache', first['representation_key']
    if corruption == 'model':
        with (cache / (key + '.joblib')).open('ab') as handle:
            handle.write(b'tampered')
    elif corruption == 'status':
        _rewrite(cache / (key + '.json'), lambda p: p.update(key='tampered'))
    elif corruption == 'sidecar_missing':
        (cache / (key + '.runtime.json')).unlink()
    else:
        _rewrite(cache / recovery.CACHE_NAMESPACE_FILE, lambda p: p.update(runtime_source_identity='other'))
    monkeypatch.setattr(experiment_worker, 'execute_job', lambda *a: pytest.fail('Invalid cache reached frozen worker'))
    with pytest.raises(ValueError, match='cache'):
        recovery.execute_recovery_job(case.jobs[1])


def test_repeat_attempt_cannot_overwrite_results(case):
    first = recovery.execute_recovery_job(case.jobs[0])
    assert first['status'] == 'VALID'
    hashes = {p.name: recovery.file_sha(p) for p in case.jobs[0].parent.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match='already has output'):
        recovery.execute_recovery_job(case.jobs[0])
    assert hashes == {p.name: recovery.file_sha(p) for p in case.jobs[0].parent.iterdir() if p.is_file()}


def test_shadow_module_is_rejected_before_fit(case, monkeypatch, tmp_path):
    fake = types.ModuleType('nhis_fairbias.recovery_shadow')
    fake.__file__ = str(tmp_path / 'outside.py')
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(experiment_worker, 'execute_job', lambda *a: pytest.fail('Shadow source reached worker'))
    with pytest.raises(ValueError, match='outside source tree'):
        recovery.execute_recovery_job(case.jobs[0])


def test_wrapper_restores_factory_and_binds_failed_worker_result(case, monkeypatch):
    original_factory = experiment_worker.make_adapter

    def failed_worker(path):
        result = {'status': 'FAILED'}
        (Path(path).parent / 'result.json').write_text(json.dumps(result))
        return result

    monkeypatch.setattr(experiment_worker, 'execute_job', failed_worker)
    receipt = recovery.execute_recovery_job(case.jobs[0])
    assert receipt['status'] == 'FAILED'
    assert receipt['runtime_checks_passed']
    assert receipt['files']['result.json'] == recovery.file_sha(case.jobs[0].parent / 'result.json')
    assert experiment_worker.make_adapter is original_factory


@pytest.mark.parametrize('case', [{'maxiter': 1, 'maxfun': 1}], indirect=True)
def test_real_budget_failure_and_cached_dependents_stay_failed(case):
    first = recovery.execute_recovery_job(case.jobs[0])
    second = recovery.execute_recovery_job(case.jobs[1])
    assert first['status'] == second['status'] == 'BUDGET_EXHAUSTED'
    assert not first['representation_cache_reused'] and second['representation_cache_reused']
    for path in case.jobs[:2]:
        result = json.loads((path.parent / 'result.json').read_text())
        assert result['status'] == 'BUDGET_EXHAUSTED'
        assert not (path.parent / 'model.joblib').exists()
        assert not (path.parent / 'predictions_S.npz').exists()
    evidence = first['representation_cache_evidence']
    assert set(evidence['files']) == {'.json'}


def test_original_registered_source_hash_is_checked_independently(case, monkeypatch):
    _rewrite(case.registration_path, lambda p: p['source_files'].update({
        'src/nhis_fairbias/benchmark/experiment_worker.py': '0' * 64,
    }))
    _rewrite(case.jobs[0], lambda p: p.update(registration_sha256=recovery.file_sha(case.registration_path)))
    monkeypatch.setattr(experiment_worker, 'execute_job', lambda *a: pytest.fail('Wrong registration source reached fit'))
    with pytest.raises(ValueError, match='Supervisor-registered source hash mismatch'):
        recovery.execute_recovery_job(case.jobs[0])


def test_module_introduced_by_prepared_load_cannot_fit(case, monkeypatch, tmp_path):
    actual_load = joblib.load

    def load_and_introduce_module(path, *args, **kwargs):
        data = actual_load(path, *args, **kwargs)
        if Path(path) == case.prepared:
            fake = types.ModuleType('nhis_fairbias.prepared_shadow')
            fake.__file__ = str(tmp_path / 'outside.py')
            monkeypatch.setitem(sys.modules, fake.__name__, fake)
        return data

    monkeypatch.setattr(joblib, 'load', load_and_introduce_module)
    monkeypatch.setattr(LFRAnalyticRecoveryAdapter, 'fit', lambda *a, **k: pytest.fail('Shadow module reached fit'))
    receipt = recovery.execute_recovery_job(case.jobs[0])
    assert receipt['status'] == 'RUNTIME_FAILED'
    assert not receipt['runtime_checks_passed']
    assert not (case.jobs[0].parent / 'model.joblib').exists()


def test_job_change_during_fit_invalidates_receipt_even_if_original_worker_returns_valid(case, monkeypatch):
    real_fit = LFRAnalyticRecoveryAdapter.fit

    def fit_and_mutate(self, *args, **kwargs):
        real_fit(self, *args, **kwargs)
        _rewrite(case.jobs[0], lambda p: p.update(synthetic_unexpected_change=True))
        return self

    monkeypatch.setattr(LFRAnalyticRecoveryAdapter, 'fit', fit_and_mutate)
    receipt = recovery.execute_recovery_job(case.jobs[0])
    assert receipt['worker_result_status'] == 'VALID'
    assert receipt['status'] == 'RUNTIME_FAILED'
    assert not receipt['runtime_checks_passed']
    assert not (case.run / 'representation_cache' / (receipt['representation_key'] + '.runtime.json')).exists()


def test_frozen_snapshot_allows_unimported_addition_but_refuses_loaded_unlisted_module(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    for name, value in recovery.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    files = {}
    for relative in recovery.REQUIRED_SOURCES:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('# Generated test source\n')
        files[relative] = recovery.file_sha(source)
    job = {'runtime_variant': recovery.VARIANT, 'runtime_source_files': files,
           'runtime_environment': recovery.THREAD_ENVIRONMENT,
           'runtime_source_identity': recovery.runtime_identity(files)}
    recovery.validate_runtime_sources(job)
    unrelated = tmp_path / 'src/nhis_fairbias/unrelated_later_repair.py'
    unrelated.write_text('# Newly added but not imported\n')
    recovery.validate_runtime_sources(job)
    modules = {name: module for name, module in sys.modules.items()
               if name.split('.')[0] not in {'fairbias', 'nhis_fairbias'}}
    fake = types.ModuleType('nhis_fairbias.unrelated_later_repair')
    fake.__file__ = str(unrelated)
    modules[fake.__name__] = fake
    monkeypatch.setattr(sys, 'modules', modules)
    with pytest.raises(ValueError, match='source mismatch'):
        recovery.runtime_modules(files)
    (tmp_path / recovery.WORKER_SOURCE).write_text('# Changed listed source\n')
    with pytest.raises(ValueError, match='source manifest/hash mismatch'):
        recovery.validate_runtime_sources(job)
