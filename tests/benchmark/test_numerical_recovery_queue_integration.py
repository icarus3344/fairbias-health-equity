"""Real generated-data children: one LFR budget failure and one valid job.

Only external host-resource readings are simulated. The scheduler, worker,
optimizer, metrics, cache, model serialization and reload all execute normally.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

import scripts.run_nhis_numerical_recovery_queue as queue
import scripts.run_nhis_numerical_recovery_worker as worker
from nhis_fairbias.benchmark.data_contracts import PartitionDataset
from nhis_fairbias.benchmark.parallel_execution import ResourceLimits, _representation_key, verify_completed_job


GIB = 1024 ** 3


def _generated_partition(role, n, seed, year):
    x = np.random.RandomState(seed).normal(size=(n, 6))
    y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)
    split = 17 if n == 48 else n // 2
    groups = np.r_[np.ones(split, dtype=int), np.full(n - split, 2, dtype=int)]
    return PartitionDataset(
        role=role, year=year, record_keys=np.array([role + ':' + str(i) for i in range(n)]),
        X_semantic=pd.DataFrame(x), y=y, A=groups, WTFA_A=np.ones(n),
        PSTRAT=np.ones(n, dtype=int), PPSU=np.arange(n), feature_names=tuple('f' + str(i) for i in range(6)),
        arm_id='arm_synthetic', metadata={'expected_categories': [1, 2]},
    ), x


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def test_real_two_process_queue_keeps_success_after_other_optimizer_exhausts_budget(tmp_path, monkeypatch):
    original = tmp_path / 'original_generated_run'
    (original / 'prepared').mkdir(parents=True)
    (original / 'representation_cache').mkdir()
    (original / 'representation_cache' / 'historical.txt').write_text('Preserve this original cache.')
    parts, arrays = {}, {}
    for role, tag, n, seed, year in (
        ('fitting_F', 'F', 48, 9, 2022), ('calibration_C', 'C', 24, 19, 2022),
        ('selection_S', 'S', 32, 29, 2023),
    ):
        parts[role], arrays['X_' + tag] = _generated_partition(role, n, seed, year)
    data = {'data_identity': 'generated_two_process_integration', 'partitions': parts,
            'preprocessor': FunctionTransformer().fit(arrays['X_F']), **arrays}
    prepared = original / 'prepared' / 'arm_synthetic.joblib'
    joblib.dump(data, prepared)
    configs = []
    for budget in (1, 300):
        config = {'arm_id': 'arm_synthetic', 'backbone': 'LR', 'method': 'LFR_RECONSTRUCTED',
                  'status': 'REGISTERED', 'seeds': [7], 'training_weighted': False,
                  'params': {'k': 3, 'Ax': 0.01, 'Ay': 1.0, 'Az': 0.5, 'C': 1.0,
                             'maxiter': 200, 'maxfun': budget}}
        config['candidate_id'] = worker.identity(config)[:20]
        configs.append(config)
    registered_sources = (
        'src/nhis_fairbias/benchmark/experiment_worker.py',
        'src/nhis_fairbias/benchmark/experiment_registry.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_lfr.py',
    )
    registration = {
        'candidates': configs, 'source_identity': 'generated_original_source',
        'source_files': {name: worker.file_sha(worker.ROOT / name) for name in registered_sources},
        'prepared': {'arm_synthetic': {'path': str(prepared), 'sha256': worker.file_sha(prepared),
                                     'data_identity': data['data_identity']}},
        'resources': {'fit_seconds': 60.0, 'worker_rss_bytes': 4 * GIB, 'concurrent_fits': 1},
    }
    registration_path = original / 'registration.json'
    _write(registration_path, registration)
    failures = []
    for config in configs:
        name = config['candidate_id'] + '_s7'
        path = original / 'jobs' / name
        path.mkdir(parents=True)
        key = _representation_key(config, 7, data['data_identity'])
        original_job = {
            'config': config, 'seed': 7, 'source_identity': registration['source_identity'],
            'data_path': str(prepared), 'data_sha256': worker.file_sha(prepared),
            'data_identity': data['data_identity'], 'cache_path': str(original / 'representation_cache'),
            'representation_key': key,
        }
        _write(path / 'job.json', original_job)
        _write(path / 'result.json', {'candidate_id': config['candidate_id'], 'seed': 7,
                                    'status': 'BUDGET_EXHAUSTED'})
        failures.append({'job_id': name, 'job_path': str(path / 'job.json'), 'config': config, 'seed': 7,
                         'job_sha256': worker.file_sha(path / 'job.json'),
                         'result_sha256': worker.file_sha(path / 'result.json'),
                         'status': 'BUDGET_EXHAUSTED', 'representation_key': key})
    inventory_path = tmp_path / 'generated_failure_inventory.json'
    _write(inventory_path, {'schema': 'failure_recovery_inventory_v1',
                            'registration_sha256': worker.file_sha(registration_path), 'failures': failures})
    files = worker.build_runtime_source_manifest()
    runtime_id = worker.runtime_identity(files)
    plan = {'schema_version': queue.RECOVERY_PLAN_VERSION, 'runtime_variant': worker.VARIANT,
            'runtime_environment': worker.THREAD_ENVIRONMENT, 'runtime_source_files': files,
            'runtime_source_identity': runtime_id, 'worker_script': worker.WORKER_SOURCE,
            'failure_manifest_path': str(inventory_path)}
    plan_path = tmp_path / 'generated_runtime_plan.json'
    _write(plan_path, plan)
    policy_path = tmp_path / 'generated_parallel_policy.json'
    _write(policy_path, {'status': 'SUPERVISOR_AUTHORIZED_PARALLEL_DEVELOPMENT',
                         'registration_sha256': worker.file_sha(registration_path),
                         'max_workers': 2, 'max_total_rss_bytes': 8 * GIB,
                         'worker_rss_bytes': 4 * GIB, 'fit_seconds': 60.0})
    namespace = tmp_path / ('lfr_recovery_' + runtime_id)
    namespace.mkdir()
    (namespace / 'jobs').mkdir()
    (namespace / 'representation_cache').mkdir()
    snapshot = {str(path.relative_to(original)): worker.file_sha(path)
                for path in original.rglob('*') if path.is_file()}
    monkeypatch.setattr(queue, 'external_worker_count', lambda runs: 0)
    monkeypatch.setattr(queue, 'cgroup_memory', lambda: (0, 60 * GIB))
    scheduler = queue.NumericalRecoveryScheduler(
        namespace, original_run=original,
        selected=queue._failure_jobs(original, inventory_path, registration, expected_count=2),
        runtime_plan=queue._runtime_plan(plan_path), runtime_plan_path=plan_path,
        external_runs=(tmp_path / 'external_frappe_a', tmp_path / 'external_frappe_b'),
        worker=worker.ROOT / worker.WORKER_SOURCE,
        source_script=worker.ROOT / 'scripts/run_nhis_benchmark_parallel.py',
        limits=ResourceLimits(workers=2, max_workers=2, total_rss_bytes=8 * GIB,
                              worker_rss_bytes=4 * GIB, fit_seconds=60.0, poll_seconds=0.05),
        methods=['LFR_RECONSTRUCTED'], policy_path=policy_path, python_executable=sys.executable,
    )
    launches = []
    real_spawn = scheduler._spawn

    def trace_real_spawn(spec):
        live = [state.process.pid for state in scheduler._running.values() if state.process.poll() is None]
        state = real_spawn(spec)
        launches.append((state.process.pid, live))
        return state

    monkeypatch.setattr(scheduler, '_spawn', trace_real_spawn)
    counts = scheduler.run()
    assert counts == {'scheduled': 2, 'completed': 2, 'resumed': 0, 'failed': 1}
    assert len({pid for pid, _ in launches}) == 2
    assert all(pid != os.getpid() for pid, _ in launches)
    assert launches[1][1] == [launches[0][0]], 'Both genuine child processes must run concurrently'
    statuses = {}
    for config in configs:
        budget = config['params']['maxfun']
        output = namespace / 'jobs' / (config['candidate_id'] + '_s7')
        result = verify_completed_job(output, config, 7, registration)
        statuses[budget] = result['status']
        receipt = json.loads((output / 'runtime_receipt.json').read_text())
        scheduler_receipt = json.loads((output / 'receipt.json').read_text())
        assert receipt['runtime_checks_passed'] and receipt['status'] == result['status']
        assert result['resource_telemetry']['thread_controls_all_one']
        assert scheduler_receipt['termination'] is None
        assert scheduler_receipt['returncode'] == (0 if budget == 300 else 1)
        assert scheduler_receipt['files']['runtime_receipt.json'] == worker.file_sha(output / 'runtime_receipt.json')
        for evidence in (receipt, scheduler_receipt):
            for name, digest in evidence['files'].items():
                assert worker.file_sha(output / name) == digest
        key = _representation_key(config, 7, data['data_identity'])
        cache = namespace / 'representation_cache'
        marker = json.loads((cache / worker.CACHE_NAMESPACE_FILE).read_text())
        sidecar = json.loads((cache / (key + '.runtime.json')).read_text())
        assert marker['runtime_source_identity'] == sidecar['runtime_source_identity'] == runtime_id
        assert marker['registration_sha256'] == worker.file_sha(registration_path)
        for suffix, digest in sidecar['files'].items():
            assert worker.file_sha(cache / (key + suffix)) == digest
        if budget == 300:
            assert set(sidecar['files']) == {'.json', '.joblib'}
            assert result['reload_verified']
            assert result['optimization']['warnflag'] == 0
            artifact = joblib.load(output / 'model.joblib')
            actual = artifact['policy'].predict(arrays['X_S'], parts['selection_S'].A)
            with np.load(output / 'predictions_S.npz') as prediction:
                np.testing.assert_array_equal(actual.q_decision, prediction['q'])
                np.testing.assert_array_equal(actual.p_event, prediction['p'])
        else:
            assert set(sidecar['files']) == {'.json'}
            status = json.loads((cache / (key + '.json')).read_text())
            assert status['diagnostics']['warnflag'] != 0
            assert status['diagnostics']['funcalls'] > budget
            assert not (output / 'model.joblib').exists()
    assert statuses == {1: 'BUDGET_EXHAUSTED', 300: 'VALID'}
    assert json.loads((namespace / 'parallel_scheduler_live_status.json').read_text())['status'] == 'COMPLETE'
    assert not (namespace / 'registration.json').exists()
    assert snapshot == {str(path.relative_to(original)): worker.file_sha(path)
                        for path in original.rglob('*') if path.is_file()}
