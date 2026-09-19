"""Strict EG failure metadata and generated-data real worker integration."""
import copy
import json
from pathlib import Path
import sys

import pytest

import scripts.run_nhis_eg_recovery_queue as queue
from test_numerical_recovery_worker import case as worker_case
from nhis_fairbias.benchmark.parallel_execution import verify_completed_job

GIB = 1024 ** 3


def write(path, value):
    Path(path).write_text(json.dumps(value))


def setup(case, count=23):
    registration = queue.base._json(case.registration_path)
    template = copy.deepcopy(registration['candidates'][-1])
    registration['candidates'] = registration['candidates'][:-1]
    failures = []
    for number in range(count):
        config = copy.deepcopy(template)
        config['method'] = 'EG_DP' if number < 15 else 'EG_EO'
        config['params']['constraint_type'] = 'dp' if number < 15 else 'eo'
        config['params']['C'] = 1.0 + number / 100
        config['candidate_id'] = queue.base.identity(config)[:20]
        registration['candidates'].append(config)
        name = config['candidate_id'] + '_s7'
        folder = case.original / 'jobs' / name
        folder.mkdir(parents=True, exist_ok=True)
        job = {'config': config, 'seed': 7, 'source_identity': registration['source_identity'],
               'data_identity': case.data['data_identity'], 'data_sha256': queue.base.file_sha(case.prepared),
               'data_path': str(case.prepared), 'cache_path': str(case.original_cache),
               'representation_key': None}
        write(folder / 'job.json', job)
        write(folder / 'result.json', {'candidate_id': config['candidate_id'], 'seed': 7, 'status': 'FAILED'})
        failures.append({**job, 'job_id': name, 'job_path': str(folder / 'job.json'), 'status': 'FAILED',
                         'job_sha256': queue.base.file_sha(folder / 'job.json'),
                         'result_sha256': queue.base.file_sha(folder / 'result.json')})
    registration['prepared']['arm_synthetic']['path'] = str(case.prepared)
    registration['resources'] = {'fit_seconds': 60.0, 'worker_rss_bytes': 4 * GIB}
    write(case.registration_path, registration)
    manifest = {'schema': 'failure_recovery_inventory_v1', 'original_run': str(case.original),
                'registration_sha256': queue.base.file_sha(case.registration_path), 'failures': failures}
    path = case.original.parent / 'eg_inventory.json'
    write(path, manifest)
    plan = {'schema_version': queue.base.RECOVERY_PLAN_VERSION,
            'runtime_variant': queue.base.RECOVERY_VARIANT,
            'runtime_environment': queue.base.RECOVERY_THREAD_ENVIRONMENT,
            'runtime_source_files': case.files, 'runtime_source_identity': case.runtime_id,
            'worker_script': queue.FROZEN_WORKER, 'failure_manifest_path': str(path)}
    plan_path = case.original.parent / 'eg_plan.json'
    write(plan_path, plan)
    target = queue.namespace(case.original.parent, case.runtime_id)
    target.mkdir()
    selected = queue.failure_jobs(case.original, path, registration, expected_count=count)
    scheduler = queue.EGRecoveryScheduler(target, original_run=case.original, selected=selected,
        runtime_plan=plan, runtime_plan_path=plan_path,
        external_runs=(case.original.parent / 'frappe', case.original.parent / 'lfr'),
        worker=queue.ROOT / queue.FROZEN_WORKER, source_script=Path(queue.__file__),
        limits=queue.base.ResourceLimits(workers=1, max_workers=28, total_rss_bytes=48 * GIB,
            worker_rss_bytes=4 * GIB, fit_seconds=60, poll_seconds=.05),
        methods=sorted(queue.EG_METHODS), python_executable=sys.executable)
    return scheduler, registration, manifest, path


def test_exact_all_23_failed_eg_jobs_have_bound_original_evidence_and_no_cache_key(worker_case):
    scheduler, _, _, _ = setup(worker_case)
    jobs = scheduler.discover()
    assert len(jobs) == len({job.path.name for job in jobs}) == 23
    assert sum(job.config['method'] == 'EG_DP' for job in jobs) == 15
    assert sum(job.config['method'] == 'EG_EO' for job in jobs) == 8
    assert scheduler.run_path.name == 'eg_recovery_' + worker_case.runtime_id
    assert scheduler.cache_path != worker_case.original_cache
    assert not (scheduler.run_path / 'registration.json').exists()
    for job in jobs:
        payload = scheduler._job_payload(job)
        assert payload['representation_key'] is payload['original_representation_key'] is None
        assert payload['original_job_sha256'] == queue.base.file_sha(payload['original_job_path'])
        assert payload['original_result_sha256'] == queue.base.file_sha(Path(payload['original_job_path']).parent / 'result.json')
        assert payload['runtime_source_identity'] == worker_case.runtime_id
        assert payload['runtime_source_files'][queue.FROZEN_WORKER] == queue.FROZEN_WORKER_SHA256


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'non_eg', 'seed', 'job_hash', 'result_hash',
                                 'config', 'source_identity', 'cache_key', 'original_valid'])
def test_incomplete_non_eg_or_mutated_inventory_cannot_enter_dispatch(worker_case, bad):
    _, registration, manifest, path = setup(worker_case)
    row = manifest['failures'][0]
    if bad == 'missing':
        manifest['failures'].pop()
    elif bad == 'duplicate':
        manifest['failures'].append(copy.deepcopy(row))
    elif bad == 'non_eg':
        row['config']['method'] = 'LFR_RECONSTRUCTED'
    elif bad == 'seed':
        row['seed'] = 99
    elif bad == 'job_hash':
        row['job_sha256'] = '0' * 64
    elif bad == 'result_hash':
        row['result_sha256'] = '0' * 64
    elif bad == 'config':
        row['config']['params']['C'] = 999
    elif bad == 'source_identity':
        row['source_identity'] = 'other'
    elif bad == 'cache_key':
        row['representation_key'] = 'wrong'
    else:
        result_path = Path(row['job_path']).parent / 'result.json'
        result = queue.base._json(result_path)
        result['status'] = 'VALID'
        write(result_path, result)
        row['result_sha256'] = queue.base.file_sha(result_path)
    write(path, manifest)
    with pytest.raises(queue.base.SchedulerError):
        queue.failure_jobs(worker_case.original, path, registration)


def test_discover_rejects_non_eg_selected_job(worker_case):
    scheduler, _, _, _ = setup(worker_case, count=1)
    next(iter(scheduler.selected.values()))['config']['method'] = 'LFR_RECONSTRUCTED'
    with pytest.raises(queue.base.SchedulerError, match='non-EG'):
        scheduler.discover()


def test_external_workers_and_memory_reserve_control_slots(worker_case, monkeypatch):
    scheduler, _, _, _ = setup(worker_case, count=1)
    scheduler.discover()
    scheduler.limits = queue.base.ResourceLimits(workers=28, max_workers=28, total_rss_bytes=48 * GIB)
    monkeypatch.setattr(queue.base, 'cgroup_memory', lambda: (0, 60 * GIB))
    monkeypatch.setattr(queue.base, 'external_worker_count', lambda runs: 28)
    assert scheduler._launchable_index(scheduler.jobs, 1) is None
    monkeypatch.setattr(queue.base, 'external_worker_count', lambda runs: 27)
    assert scheduler._launchable_index(scheduler.jobs, 1) == 0
    monkeypatch.setattr(queue.base, 'cgroup_memory', lambda: (49 * GIB, 60 * GIB))
    assert scheduler._launchable_index(scheduler.jobs, 1) is None


def test_server_wait_requires_complete_bound_lfr_and_no_surviving_workers(tmp_path, monkeypatch):
    runtime_id, registration_sha = 'a' * 64, 'b' * 64
    run = tmp_path / ('lfr_recovery_' + runtime_id)
    run.mkdir()
    status_path = run / 'parallel_scheduler_live_status.json'
    write(status_path, {'status': 'RUNNING'})
    sleeps = []
    def advance(seconds):
        sleeps.append(seconds)
        write(status_path, {'status': 'COMPLETE', 'run': str(run), 'methods': ['LFR_RECONSTRUCTED'],
                            'scheduled_count': 960, 'finalized_count': 960, 'active_count': 0})
        write(run / 'recovery_dispatch_manifest.json', {'runtime_source_identity': runtime_id,
              'registration_sha256': registration_sha, 'selected_jobs': 960})
    monkeypatch.setattr(queue.time, 'sleep', advance)
    monkeypatch.setattr(queue.base, 'external_worker_count', lambda runs: 0)
    queue.wait_for_lfr(run, runtime_id, registration_sha)
    assert sleeps == [15.0]
    monkeypatch.setattr(queue.base, 'external_worker_count', lambda runs: 1)
    assert not queue.lfr_complete(run, runtime_id, registration_sha)
    write(status_path, {'status': 'INTERRUPTED'})
    with pytest.raises(queue.base.SchedulerError, match='interrupted'):
        queue.wait_for_lfr(run, runtime_id, registration_sha)


def test_real_generated_eg_subprocess_runs_through_queue_and_preserves_original(worker_case, monkeypatch):
    scheduler, registration, _, _ = setup(worker_case, count=1)
    before = {str(p.relative_to(worker_case.original)): queue.base.file_sha(p)
              for p in worker_case.original.rglob('*') if p.is_file()}
    monkeypatch.setattr(queue.base, 'external_worker_count', lambda runs: 0)
    monkeypatch.setattr(queue.base, 'cgroup_memory', lambda: (0, 60 * GIB))
    summary = scheduler.run()
    assert summary == {'scheduled': 1, 'completed': 1, 'resumed': 0, 'failed': 0}
    spec = scheduler.jobs[0]
    result = verify_completed_job(spec.path, spec.config, spec.seed, registration)
    assert result['status'] == 'VALID' and result['reload_verified']
    receipt = queue.base._json(spec.path / 'runtime_receipt.json')
    assert receipt['status'] == 'VALID' and receipt['runtime_checks_passed']
    assert receipt['representation_key'] is None and not receipt['legacy_valid_results_admitted']
    assert queue.base._json(scheduler._live_status_path)['status'] == 'COMPLETE'
    assert before == {str(p.relative_to(worker_case.original)): queue.base.file_sha(p)
                      for p in worker_case.original.rglob('*') if p.is_file()}


def test_cli_refuses_manifest_mutation_while_waiting_before_any_dispatch(worker_case, monkeypatch):
    scheduler, _, manifest, path = setup(worker_case)
    policy = worker_case.original.parent / 'eg_policy.json'
    write(policy, {'max_total_rss_bytes': 48 * GIB, 'worker_rss_bytes': 4 * GIB, 'fit_seconds': 60})
    def change_inventory(*args):
        manifest['unexpected_change'] = True
        write(path, manifest)
    monkeypatch.setattr(queue, 'wait_for_lfr', change_inventory)
    monkeypatch.setattr(queue.base, '_effective_cpu_count', lambda: 28)
    monkeypatch.setattr(queue.base, 'cgroup_memory', lambda: (0, 60 * GIB))
    monkeypatch.setattr(queue.EGRecoveryScheduler, 'run', lambda self: pytest.fail('Changed wait inputs dispatched'))
    output_root = worker_case.original.parent / 'new_cli_output'
    with pytest.raises(SystemExit) as exc:
        queue.main(['--run', str(output_root), '--original-run', str(worker_case.original),
                    '--failure-manifest', str(path), '--runtime-plan', str(scheduler.runtime_plan_path),
                    '--policy', str(policy), '--wait-for-lfr-run', str(worker_case.original.parent / 'lfr'),
                    '--external-run', str(worker_case.original.parent / 'frappe'), '--workers', '1'])
    assert exc.value.code == 2
    assert not output_root.exists()
