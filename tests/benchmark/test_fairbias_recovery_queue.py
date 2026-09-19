"""Generated metadata, injected /proc identities, and real synthetic workers."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_nhis_fairbias_recovery_queue as queue
from test_fairbias_recovery_worker import case

base, worker, GiB = queue.base, queue.worker, queue.GiB


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup(tmp_path, policy='bm_mds_retry_v1', count=22):
    original = tmp_path / 'original'
    method = worker.policy_spec(policy)['method']
    config = {'candidate_id': 'generated', 'method': method, 'backbone': 'LR', 'arm_id': 'a',
              'status': 'REGISTERED', 'seeds': list(range(count)), 'params': {},
              'training_weighted': policy == 'bm_mds_retry_v1'}
    registration = {'source_files': {}, 'source_identity': 'synthetic-source', 'candidates': [config],
                    'prepared': {'a': {'sha256': 'generated-only', 'data_identity': 'synthetic-data'}}}
    save(original / 'registration.json', registration)
    entries = []
    for seed in range(count):
        path = original / 'jobs' / f'generated_s{seed}' / 'job.json'
        job = {'config': config, 'seed': seed, 'source_identity': registration['source_identity'],
               'data_identity': 'synthetic-data', 'data_sha256': 'generated-only',
               'representation_key': base._representation_key(config, seed, 'synthetic-data')}
        status = 'BUDGET_EXHAUSTED' if policy == 'bm_mds_retry_v1' else 'TIME_LIMIT'
        previous = {'status': status, 'candidate_id': 'generated', 'seed': seed, 'error': 'MDS_ITERATION_CAP'}
        save(path, job)
        save(path.parent / 'result.json', previous)
        entries.append({**job, 'job_id': path.parent.name, 'job_path': str(path), 'status': status,
                        'error': previous['error'], 'job_sha256': base.file_sha(path),
                        'result_sha256': base.file_sha(path.parent / 'result.json')})
    inventory = tmp_path / 'inventory.json'
    save(inventory, {'schema': 'failure_recovery_inventory_v1', 'original_run': str(original),
                    'registration_sha256': base.file_sha(original / 'registration.json'), 'failures': entries})
    return original, registration, inventory


def plan_for(tmp_path, original, inventory, policy):
    files = worker.build_runtime_source_manifest()
    plan = {'schema_version': queue.PLAN_SCHEMA, 'runtime_variant': worker.VARIANT,
            'recovery_policy': policy, 'recovery_policy_spec': worker.policy_spec(policy),
            'execution_budget': worker.DEFAULT_EXECUTION_BUDGET,
            'runtime_source_files': files, 'runtime_environment': worker.THREAD_ENVIRONMENT,
            'worker_script': worker.WORKER_SOURCE,
            'runtime_source_identity': worker.runtime_identity(files, policy=policy),
            'registration_sha256': base.file_sha(original / 'registration.json'),
            'failure_manifest_sha256': base.file_sha(inventory)}
    path = tmp_path / 'plan.json'
    save(path, plan)
    return path, plan


def scheduler_for(tmp_path, original, inventory, policy, count):
    registration = base._json(original / 'registration.json')
    plan_path, plan = plan_for(tmp_path, original, inventory, policy)
    selected = queue.failure_jobs(original, inventory, registration, policy, count)
    policy_path = tmp_path / 'policy.json'
    save(policy_path, {'status': 'SUPERVISOR_AUTHORIZED_PARALLEL_DEVELOPMENT',
        'registration_sha256': base.file_sha(original / 'registration.json'),
        'max_workers': 28, 'max_total_rss_bytes': 48 * GiB, 'external_supervisors': [],
        'admitted_runtime_identities': [plan['runtime_source_identity']]})
    plan['failure_manifest_path'] = str(inventory)
    return queue.FairBiasRecoveryScheduler(
        queue.namespace(tmp_path / 'new', policy, plan['runtime_source_identity']),
        original_run=original, selected=selected, runtime_plan=plan, runtime_plan_path=plan_path,
        external_runs=(), worker=queue.ROOT / worker.WORKER_SOURCE, source_script=Path(queue.__file__),
        limits=base.ResourceLimits(workers=28, max_workers=28, total_rss_bytes=48 * GiB),
        methods=[worker.policy_spec(policy)['method']], policy_path=policy_path)


@pytest.mark.parametrize('policy,count', [('bm_mds_retry_v1', 22), ('joint_strict_exact_v1', 38)])
def test_exact_policy_coverage_and_fresh_cache(tmp_path, policy, count):
    original, registration, inventory = setup(tmp_path, policy, count)
    selected = queue.failure_jobs(original, inventory, registration, policy)
    assert len(selected) == count
    scheduler = scheduler_for(tmp_path, original, inventory, policy, count)
    jobs = scheduler.discover()
    payload = scheduler._job_payload(jobs[0])
    assert payload['config'] == registration['candidates'][0]
    assert payload['execution_budget'] == worker.DEFAULT_EXECUTION_BUDGET
    assert payload['recovery_policy'] == policy
    assert Path(payload['data_path']) == original / 'prepared' / 'a.joblib'
    assert Path(payload['cache_path']).is_relative_to(scheduler.run_path)
    assert not (scheduler.run_path / 'registration.json').exists()
    assert (jobs[0].representation_key is None) == (policy == 'joint_strict_exact_v1')
    with pytest.raises(base.SchedulerError, match='already exists'):
        scheduler.discover()


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'config', 'result', 'noncap'])
def test_exact_inventory_rejects_changed_evidence(tmp_path, change):
    original, registration, inventory = setup(tmp_path)
    data = base._json(inventory)
    if change == 'missing':
        data['failures'].pop()
    elif change == 'duplicate':
        data['failures'].append(data['failures'][0])
    elif change == 'config':
        data['failures'][0]['config']['training_weighted'] = False
    elif change == 'noncap':
        data['failures'][0]['error'] = 'AE_ITERATION_CAP'
    else:
        path = original / 'jobs' / 'generated_s0' / 'result.json'
        save(path, {'status': 'VALID'})
    save(inventory, data)
    with pytest.raises(base.SchedulerError):
        queue.failure_jobs(original, inventory, registration, 'bm_mds_retry_v1')


def test_unadmitted_policies_and_joint_mds_cannot_enter(tmp_path):
    original, registration, inventory = setup(tmp_path, 'joint_strict_exact_v1', 38)
    data = base._json(inventory)
    data['failures'][0]['status'] = 'BUDGET_EXHAUSTED'
    save(inventory, data)
    with pytest.raises(base.SchedulerError, match='exactly cover'):
        queue.failure_jobs(original, inventory, registration, 'joint_strict_exact_v1')
    for policy in ('bmae_cap40_v1', 'joint_mds_retry_v1'):
        with pytest.raises(ValueError):
            queue.failure_jobs(original, inventory, registration, policy)


def test_runtime_budget_and_source_identity_are_explicit(tmp_path):
    original, _, inventory = setup(tmp_path)
    path, plan = plan_for(tmp_path, original, inventory, 'bm_mds_retry_v1')
    assert queue.runtime_plan(path) == plan
    plan['execution_budget'] = {**plan['execution_budget'], 'fit_seconds': 3600.}
    save(path, plan)
    with pytest.raises(base.SchedulerError, match='identity mismatch'):
        queue.runtime_plan(path)
    plan['runtime_source_identity'] = worker.runtime_identity(plan['runtime_source_files'],
        policy=plan['recovery_policy'], execution_budget=plan['execution_budget'])
    save(path, plan)
    assert queue.runtime_plan(path) == plan
    plan['runtime_source_files'][worker.WORKER_SOURCE] = '0' * 64
    save(path, plan)
    with pytest.raises(base.SchedulerError, match='source mismatch'):
        queue.runtime_plan(path)


def test_ownership_blocks_duplicate_phase_and_separate_launch(tmp_path):
    original, registration, inventory = setup(tmp_path)
    _, plan = plan_for(tmp_path, original, inventory, 'bm_mds_retry_v1')
    selected = queue.failure_jobs(original, inventory, registration, plan['recovery_policy'])
    with pytest.raises(base.SchedulerError, match='Duplicate ownership'):
        queue.claim_jobs(original, [(plan, selected), (plan, selected)], tmp_path / 'new')
    queue.claim_jobs(original, [(plan, selected)], tmp_path / 'new')
    with pytest.raises(base.SchedulerError, match='already owned'):
        queue.claim_jobs(original, [(plan, selected)], tmp_path / 'another')


def supervisor(tmp_path, status='RUNNING'):
    path = tmp_path / 'pilot.json'
    command = ['/absolute/python', '-B', '/absolute/pilot.py']
    item = {'supervisor_receipt': str(path), 'command': command,
            'command_sha256': base.identity(command), 'memory_limit_bytes': 4 * GiB}
    receipt = {'supervisor_only': True, 'status': status, 'command': command,
               'memory_limit_bytes': 4 * GiB, 'child_pid': 123, 'process_group_id': 123,
               'started_utc': '2026-09-17T00:00:00+00:00'}
    save(path, receipt)
    process = {'command': command, 'start_ticks': 999, 'process_group_id': 123,
               'started_epoch': 1789603200.}
    # Derive epoch rather than relying on an unverified literal date conversion.
    process['started_epoch'] = queue.datetime.fromisoformat(receipt['started_utc']).timestamp()
    return path, item, receipt, process


@pytest.mark.parametrize('status,alive,reserved', [('STARTING', False, 1), ('RUNNING', False, 1),
    ('RUNNING', True, 1), ('CHILD_PROCESS_EXITED', True, 1), ('CHILD_PROCESS_EXITED', False, 0)])
def test_supervisor_reservation_requires_terminal_and_exit(tmp_path, status, alive, reserved):
    path, item, receipt, process = supervisor(tmp_path, status)
    if status == 'STARTING':
        receipt.pop('child_pid')
        receipt.pop('process_group_id')
        save(path, receipt)
    monitor = queue.ExternalSupervisors([item], probe=lambda pid: process if alive else None)
    sample = monitor.sample()
    assert sample['reserved_slots'] == reserved
    assert sample['reserved_bytes'] == reserved * 4 * GiB
    assert not sample['blocked_receipts']


@pytest.mark.parametrize('change', ['command', 'ticks', 'pid', 'start_time'])
def test_pid_reuse_or_command_mismatch_fails_closed(tmp_path, change):
    path, item, receipt, process = supervisor(tmp_path)
    clock = [0.]
    monitor = queue.ExternalSupervisors([item], probe=lambda pid: process, clock=lambda: clock[0])
    assert monitor.sample()['supervisors'][0]['error'] is None
    if change == 'command':
        process['command'] = ['unrelated']
    elif change == 'ticks':
        process['start_ticks'] += 1
    elif change == 'pid':
        receipt['child_pid'] = receipt['process_group_id'] = 124
        save(path, receipt)
    else:
        process['started_epoch'] += 3600
    assert monitor.sample()['reserved_slots'] == 1
    clock[0] = 31.
    assert monitor.sample()['blocked_receipts'] == [str(path)]


def test_truncated_receipt_retries_and_recovers_without_release(tmp_path, monkeypatch):
    path, item, receipt, process = supervisor(tmp_path)
    original = base._json
    calls = []
    def read(value):
        calls.append(value)
        if len(calls) < 3:
            raise base.SchedulerError('transient incomplete JSON')
        return original(value)
    monkeypatch.setattr(base, '_json', read)
    monitor = queue.ExternalSupervisors([item], probe=lambda pid: process)
    assert monitor.sample()['reserved_slots'] == 1
    assert not monitor.bad_since
    assert len(calls) == 3


def test_resource_accounting_releases_eight_and_preserves_running_on_bad_receipt(tmp_path, monkeypatch):
    original, _, inventory = setup(tmp_path)
    scheduler = scheduler_for(tmp_path, original, inventory, 'bm_mds_retry_v1', 22)
    scheduler.discover()
    monkeypatch.setattr(base, 'external_worker_count', lambda runs: 0)
    monkeypatch.setattr(base, 'cgroup_memory', lambda: (0, 60 * GiB))
    bindings = []
    for i in range(9):
        path, item, receipt, _ = supervisor(tmp_path / str(i), 'CHILD_PROCESS_EXITED' if i < 8 else 'RUNNING')
        # Dead terminal PIDs may be reused across separately finished pilots.
        receipt['child_pid'] = receipt['process_group_id'] = 123 + i
        save(path, receipt)
        bindings.append(item)
    scheduler.supervisors = queue.ExternalSupervisors(bindings, probe=lambda pid: None, bad_grace_seconds=0)
    assert scheduler._launchable_index(scheduler.jobs, 28) == 0
    assert scheduler.limits.total_rss_bytes == 44 * GiB
    scheduler._running = {i: SimpleNamespace(last_rss_bytes=GiB) for i in range(27)}
    assert scheduler._launchable_index(scheduler.jobs, 1) is None
    Path(bindings[0]['supervisor_receipt']).write_text('{')
    assert scheduler._launchable_index(scheduler.jobs, 1) is None
    assert scheduler.limits.total_rss_bytes == 44 * GiB
    scheduler._running = {}
    with pytest.raises(base.SchedulerError, match='running jobs drained'):
        scheduler._launchable_index(scheduler.jobs, 28)


@pytest.mark.parametrize('case', [{'policy': 'joint_strict_exact_v1'}, {'policy': 'bm_mds_retry_v1'}], indirect=True)
def test_actual_generated_worker_queue_and_receipts(case, tmp_path, monkeypatch):
    """The real subprocess fits generated F/C/S and verifies reload/cache receipts."""
    entries = []
    registration = base._json(case.original / 'registration.json')
    for newpath in case.jobs:
        payload = base._json(newpath)
        oldpath = Path(payload['original_job_path'])
        job = base._json(oldpath)
        key = base._representation_key(job['config'], job['seed'], job['data_identity'])
        job['representation_key'] = key
        save(oldpath, job)
        previous = base._json(oldpath.parent / 'result.json')
        previous['error'] = 'MDS_ITERATION_CAP'
        save(oldpath.parent / 'result.json', previous)
        entries.append({**job, 'job_id': oldpath.parent.name, 'job_path': str(oldpath),
            'job_sha256': base.file_sha(oldpath), 'result_sha256': base.file_sha(oldpath.parent / 'result.json'),
            'status': previous['status'], 'error': previous['error']})
    inventory = tmp_path / 'inventory.json'
    save(inventory, {'schema': 'failure_recovery_inventory_v1', 'original_run': str(case.original),
                    'registration_sha256': base.file_sha(case.original / 'registration.json'), 'failures': entries})
    before = {str(p): base.file_sha(p) for p in case.original.rglob('*') if p.is_file()}
    scheduler = scheduler_for(tmp_path, case.original, inventory, case.policy, len(entries))
    scheduler.limits = queue.replace(scheduler.limits, workers=2, poll_seconds=.02)
    monkeypatch.setattr(base, 'external_worker_count', lambda runs: 0)
    monkeypatch.setattr(base, 'cgroup_memory', lambda: (0, 60 * GiB))
    result = scheduler.run()
    assert result == {'scheduled': len(entries), 'completed': len(entries), 'resumed': 0, 'failed': 0}
    for spec in scheduler.jobs:
        state = SimpleNamespace(spec=spec, termination=None, process=SimpleNamespace(returncode=0))
        assert scheduler._receipt_error(state) is None
        path = spec.path / 'runtime_receipt.json'
        receipt = base._json(path)
        assert receipt['status'] == 'VALID'
        assert receipt['execution_budget'] == worker.DEFAULT_EXECUTION_BUDGET
        receipt['execution_budget']['fit_seconds'] = 17.
        save(path, receipt)
        assert scheduler._receipt_error(state) == 'FairBias receipt policy/budget/schema mismatch'
    assert before == {str(p): base.file_sha(p) for p in case.original.rglob('*') if p.is_file()}
