#!/usr/bin/env python3
"""Run separately identified FairBias recovery phases over exact historical failures.

Only BM's 22 MDS-cap failures and Joint's 38 strict wall failures are admitted.
Policies/budgets never share runtime identities or caches. External supervised
pilots reserve their declared memory until terminal evidence and process state
both permit release. This command never selects candidates using S/T metrics.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import scripts.run_nhis_numerical_recovery_queue as base
import scripts.run_nhis_fairbias_recovery_worker as worker
from nhis_fairbias.benchmark.parallel_execution import ExclusiveRunLock

GiB = 1024 ** 3
PLAN_SCHEMA = 'nhis_fairbias_recovery_plan_v1_20260917'
COUNTS = {'bm_mds_retry_v1': 22, 'joint_strict_exact_v1': 38}
TERMINAL = {'CHILD_PROCESS_EXITED', 'CHILD_TERMINATED_BY_SUPERVISOR', 'SUPERVISOR_EXCEPTION'}
QUEUE_SOURCES = ('scripts/run_nhis_fairbias_recovery_queue.py',
                 'scripts/run_nhis_numerical_recovery_queue.py',
                 'src/nhis_fairbias/benchmark/parallel_execution.py')


def runtime_plan(path):
    plan = base._json(path)
    policy = plan.get('recovery_policy')
    spec = worker.policy_spec(policy)
    if (plan.get('schema_version') != PLAN_SCHEMA or policy not in COUNTS
            or plan.get('runtime_variant') != worker.VARIANT
            or plan.get('runtime_environment') != worker.THREAD_ENVIRONMENT
            or plan.get('recovery_policy_spec') != spec
            or plan.get('worker_script') != worker.WORKER_SOURCE):
        raise base.SchedulerError('FairBias runtime plan contract mismatch')
    if 'execution_budget' not in plan:
        raise base.SchedulerError('FairBias plan must explicitly declare execution budget')
    budget = worker.execution_budget_spec(plan['execution_budget'])
    files = plan.get('runtime_source_files')
    if not isinstance(files, dict) or not worker.REQUIRED_SOURCES.issubset(files):
        raise base.SchedulerError('FairBias runtime source manifest is incomplete')
    for relative, digest in files.items():
        source = (ROOT / relative).resolve()
        if (Path(relative).is_absolute() or not source.is_relative_to(ROOT)
                or not source.is_file() or base.file_sha(source) != digest):
            raise base.SchedulerError('FairBias runtime source mismatch: ' + relative)
    if plan.get('runtime_source_identity') != worker.runtime_identity(
            files, plan['runtime_environment'], policy=policy, execution_budget=budget):
        raise base.SchedulerError('FairBias policy/budget/runtime identity mismatch')
    return plan


def _eligible(value, policy):
    if policy == 'bm_mds_retry_v1':
        return (value.get('status') == 'BUDGET_EXHAUSTED'
                and 'MDS_ITERATION_CAP' in str(value.get('error', '')))
    return value.get('status') == 'TIME_LIMIT'


def failure_jobs(original_run, manifest_path, registration, policy, expected_count=None):
    """Verify the complete policy subset of the frozen failure inventory."""
    spec = worker.policy_spec(policy)
    if policy not in COUNTS:
        raise base.SchedulerError('FairBias policy has no admitted failure set')
    expected_count = COUNTS[policy] if expected_count is None else expected_count
    original = Path(original_run).resolve()
    manifest = base._json(manifest_path)
    if (manifest.get('schema') != 'failure_recovery_inventory_v1'
            or Path(manifest.get('original_run', '')).resolve() != original
            or manifest.get('registration_sha256') != base.file_sha(original / 'registration.json')
            or not isinstance(manifest.get('failures'), list)):
        raise base.SchedulerError('FairBias failure inventory origin/registration mismatch')
    registered = {}
    for config in registration['candidates']:
        if config.get('method') != spec['method'] or config.get('status') != 'REGISTERED':
            continue
        for seed in config['seeds']:
            name = f"{config['candidate_id']}_s{seed}"
            if name in registered:
                raise base.SchedulerError('Duplicate registered FairBias job')
            registered[name] = (config, seed)
    selected, seen = {}, set()
    for entry in manifest['failures']:
        if not isinstance(entry, dict):
            raise base.SchedulerError('Malformed failure entry')
        name = entry.get('job_id')
        if not isinstance(name, str) or name in seen:
            raise base.SchedulerError('Duplicate/invalid inventory job')
        seen.add(name)
        if entry.get('config', {}).get('method') != spec['method'] or not _eligible(entry, policy):
            continue
        if name not in registered:
            raise base.SchedulerError('Failure is not a registered policy job')
        config, seed = registered[name]
        old = original / 'jobs' / name / 'job.json'
        result_path = old.parent / 'result.json'
        if (entry.get('config') != config or entry.get('seed') != seed
                or Path(entry.get('job_path', '')).resolve() != old
                or entry.get('job_sha256') != base.file_sha(old)
                or entry.get('result_sha256') != base.file_sha(result_path)):
            raise base.SchedulerError('Failure inventory job/config/seed/hash mismatch')
        previous, job = base._json(result_path), base._json(old)
        if (not _eligible(previous, policy) or previous.get('status') != entry.get('status')
                or previous.get('candidate_id') != config['candidate_id'] or previous.get('seed') != seed):
            raise base.SchedulerError('Original result is not the claimed policy failure')
        data = registration['prepared'][config['arm_id']]
        key = base._representation_key(config, seed, data['data_identity'])
        required = {'config': config, 'seed': seed, 'source_identity': registration['source_identity'],
                    'data_identity': data['data_identity'], 'data_sha256': data['sha256'],
                    'representation_key': key}
        if any(job.get(k) != v or entry.get(k) != v for k, v in required.items()):
            raise base.SchedulerError('Failure/original job differs from registered identity')
        if config.get('training_weighted', False) and not spec['training_weighted_supported']:
            raise base.SchedulerError('Weighted policy is not supported')
        selected[name] = {'config': config, 'seed': seed, 'original_job_path': str(old),
                          'original_job_sha256': base.file_sha(old),
                          'original_result_sha256': base.file_sha(result_path),
                          'original_representation_key': key}
    if len(selected) != expected_count:
        raise base.SchedulerError('Failure inventory does not exactly cover the admitted policy count')
    return selected


def process_identity(pid):
    """Read Linux PID identity; start ticks prevent reuse of an observed PID."""
    path = Path('/proc') / str(pid)
    try:
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        argv = [v.decode() for v in (path / 'cmdline').read_bytes().split(b'\0') if v]
        ticks = int(stat[19])
        boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                    if line.startswith('btime '))
        return {'command': argv, 'start_ticks': ticks, 'process_group_id': int(stat[2]),
                'started_epoch': boot + ticks / os.sysconf('SC_CLK_TCK')}
    except (FileNotFoundError, ProcessLookupError):
        return None


class ExternalSupervisors:
    """Conservative reservations, without signalling another supervisor's child."""
    def __init__(self, bindings, *, probe=None, clock=None, bad_grace_seconds=30.0):
        self.bindings = list(bindings)
        self.probe, self.clock = probe or process_identity, clock or time.monotonic
        self.bad_grace_seconds = bad_grace_seconds
        self.observed, self.bad_since, self.released = {}, {}, set()
        paths = set()
        for item in self.bindings:
            path = str(Path(item['supervisor_receipt']).resolve())
            command = item.get('command')
            if (path in paths or not isinstance(command, list) or not command
                    or any(not isinstance(v, str) for v in command)
                    or item.get('command_sha256') != base.identity(command)
                    or type(item.get('memory_limit_bytes')) is not int
                    or not 0 < item['memory_limit_bytes'] <= 4 * GiB):
                raise base.SchedulerError('Invalid/duplicate external supervisor binding')
            paths.add(path)

    def _one(self, item):
        path = str(Path(item['supervisor_receipt']).resolve())
        receipt = None
        for attempt in range(3):
            try:
                receipt = base._json(path)
                break
            except base.SchedulerError:
                if attempt < 2:
                    time.sleep(.02)  # Existing supervisor writes by seek/truncate.
        if not isinstance(receipt, dict):
            raise ValueError('missing or temporarily unreadable supervisor receipt')
        if (receipt.get('supervisor_only') is not True or receipt.get('command') != item['command']
                or receipt.get('memory_limit_bytes') != item['memory_limit_bytes']
                or receipt.get('status') not in TERMINAL | {'STARTING', 'RUNNING'}):
            raise ValueError('supervisor receipt command/limit/status mismatch')
        status, pid = receipt['status'], receipt.get('child_pid')
        if pid is None:
            if status in TERMINAL and status != 'SUPERVISOR_EXCEPTION':
                raise ValueError('terminal receipt lacks child PID')
            # A supervisor exception without a child is terminal only if it
            # explicitly reports no launch; otherwise preserve the reservation.
            return True, status, None
        if type(pid) is not int or pid <= 0 or receipt.get('process_group_id') != pid:
            raise ValueError('invalid supervisor process identity')
        if path in self.observed and self.observed[path][0] != pid:
            raise ValueError('supervised PID identity changed')
        actual = self.probe(pid)
        if actual is None:
            if status in TERMINAL:
                self.released.add(path)
            return status not in TERMINAL, status, pid
        if path in self.released:
            raise ValueError('released supervisor has a live child or reused PID')
        if actual['command'] != item['command'] or actual['process_group_id'] != pid:
            raise ValueError('live child command/process-group mismatch or PID reuse')
        started = datetime.fromisoformat(receipt['started_utc']).timestamp()
        if not started - 2 <= actual['started_epoch'] <= started + 30:
            raise ValueError('child start time mismatches supervisor launch or PID reuse')
        identity = (pid, actual['start_ticks'])
        if path in self.observed and self.observed[path] != identity:
            raise ValueError('supervised PID identity changed')
        self.observed[path] = identity
        return True, status, pid  # A terminal receipt cannot release a live child.

    def sample(self):
        slots, memory, rows, blocked, pids = 0, 0, [], [], set()
        for item in self.bindings:
            path = str(Path(item['supervisor_receipt']).resolve())
            error = None
            try:
                active, status, pid = self._one(item)
                if pid is not None and pid in pids:
                    raise ValueError('duplicate supervisor child PID')
                if pid is not None:
                    pids.add(pid)
                self.bad_since.pop(path, None)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                active, status, pid, error = True, 'UNVERIFIED_RESERVED', None, str(exc)
                first = self.bad_since.setdefault(path, self.clock())
                if self.clock() - first >= self.bad_grace_seconds:
                    blocked.append(path)
            if active:
                slots += 1
                memory += item['memory_limit_bytes']
            rows.append({'supervisor_receipt': path, 'status': status, 'pid': pid,
                         'reserved': active, 'error': error})
        return {'reserved_slots': slots, 'reserved_bytes': memory,
                'blocked_receipts': blocked, 'supervisors': rows}


def namespace(root, policy, runtime_id):
    worker.policy_spec(policy)
    if len(runtime_id) != 64 or any(v not in '0123456789abcdef' for v in runtime_id):
        raise base.SchedulerError('Full runtime SHA256 required')
    return Path(root).resolve() / ('fairbias_' + policy + '_' + runtime_id)


class FairBiasRecoveryScheduler(base.NumericalRecoveryScheduler):
    def __init__(self, *args, external_supervisors=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.supervisors = ExternalSupervisors(external_supervisors)
        self.total_ceiling = self.limits.total_rss_bytes
        self._external_status = None

    def _validate_parallel_policy(self):
        policy = base._json(self.policy_path)
        budget = worker.execution_budget_spec(self.runtime_plan['execution_budget'])
        if (policy.get('status') != 'SUPERVISOR_AUTHORIZED_PARALLEL_DEVELOPMENT'
                or policy.get('registration_sha256') != base.file_sha(self.registration_path)
                or not self.limits.workers <= int(policy.get('max_workers', 0)) <= 28
                or not self.total_ceiling <= int(policy.get('max_total_rss_bytes', 0)) <= 48 * GiB
                or self.limits.fit_seconds != budget['fit_seconds']
                or self.limits.worker_rss_bytes != budget['worker_rss_bytes']
                or budget['worker_rss_bytes'] > 4 * GiB
                or policy.get('external_supervisors', []) != self.supervisors.bindings
                or self.runtime_plan['runtime_source_identity'] not in policy.get('admitted_runtime_identities', [])):
            raise base.SchedulerError('FairBias resource/admission policy mismatch')

    def discover(self):
        self._validate_sources()
        self._validate_parallel_policy()
        self._write_manifest()
        spec = worker.policy_spec(self.runtime_plan['recovery_policy'])
        jobs = []
        for name, selected in sorted(self.selected.items()):
            config, seed = selected['config'], selected['seed']
            data = dict(self.registration['prepared'][config['arm_id']])
            data.setdefault('path', str(self.original_run / 'prepared' / (config['arm_id'] + '.joblib')))
            key = base._representation_key(config, seed, data['data_identity'])
            if (config['method'] != spec['method'] or selected['original_representation_key'] != key
                    or name != f"{config['candidate_id']}_s{seed}"):
                raise base.SchedulerError('FairBias discovery policy/cache identity mismatch')
            jobs.append(base.JobSpec(config=config, seed=seed, path=self.run_path / 'jobs' / name,
                data=data, source_identity=self.registration['source_identity'], representation_key=key))
        self.jobs = jobs
        return jobs

    def _job_payload(self, spec):
        return {**super()._job_payload(spec), 'recovery_policy': self.runtime_plan['recovery_policy'],
                'execution_budget': self.runtime_plan['execution_budget']}

    def _write_manifest(self):
        base.ParallelBenchmarkScheduler._write_manifest(self)
        marker = {'runtime_variant': self.runtime_plan['runtime_variant'],
                  'runtime_source_identity': self._runtime_identity,
                  'registration_sha256': base.file_sha(self.registration_path)}
        contract = {**marker, 'schema_version': PLAN_SCHEMA,
            'recovery_policy': self.runtime_plan['recovery_policy'],
            'recovery_policy_spec': worker.policy_spec(self.runtime_plan['recovery_policy']),
            'execution_budget': self.runtime_plan['execution_budget'],
            'runtime_plan_sha256': base.file_sha(self.runtime_plan_path),
            'failure_manifest_sha256': base.file_sha(self.runtime_plan['failure_manifest_path']),
            'queue_source_files': {p: base.file_sha(ROOT / p) for p in QUEUE_SOURCES},
            'selected_jobs': len(self.selected), 'selected_original_evidence': self.selected,
            'external_supervisors': self.supervisors.bindings,
            'external_runs': [str(p) for p in self.external_runs],
            'resource_rule': 'global_slots<=28; formal_peak_RSS+active_pilot_declared_reserve<=48GiB; cgroup_free>=12GiB',
            'sampled_limits_not_kernel_caps': True}
        for path, value in ((self.cache_path / '.runtime_namespace.json', marker),
                            (self.run_path / 'recovery_dispatch_manifest.json', contract)):
            if path.exists() or not base.write_json_once(path, value):
                raise base.SchedulerError('Fresh FairBias dispatch/cache evidence already exists')

    def _receipt_error(self, state):
        error = super()._receipt_error(state)
        if error:
            return error
        receipt = base._json(state.spec.path / 'runtime_receipt.json')
        expected = {'schema_version': 'nhis_fairbias_recovery_receipt_v1',
                    'recovery_policy': self.runtime_plan['recovery_policy'],
                    'recovery_policy_spec': worker.policy_spec(self.runtime_plan['recovery_policy']),
                    'execution_budget': self.runtime_plan['execution_budget'],
                    'runtime_source_files': self.runtime_plan['runtime_source_files'],
                    'runtime_environment': self.runtime_plan['runtime_environment'],
                    'data_identity': state.spec.data['data_identity'],
                    'data_sha256': state.spec.data['sha256'],
                    'representation_key': state.spec.representation_key,
                    'cache_path': str(self.cache_path)}
        if any(receipt.get(k) != v for k, v in expected.items()):
            return 'FairBias receipt policy/budget/schema mismatch'
        return None

    def _write_scheduler_failure(self, state, reason):
        super()._write_scheduler_failure(state, reason)
        path = state.spec.path / 'runtime_receipt.json'
        receipt = base._json(path)
        receipt.update(schema_version='nhis_fairbias_recovery_receipt_v1',
                       recovery_policy=self.runtime_plan['recovery_policy'],
                       recovery_policy_spec=worker.policy_spec(self.runtime_plan['recovery_policy']),
                       execution_budget=self.runtime_plan['execution_budget'])
        base.replace_json(path, receipt)

    def _launchable_index(self, pending, slots):
        external = self.supervisors.sample()
        if external != self._external_status:
            base.replace_json(self.run_path / 'external_resources_live.json', external)
            self._external_status = external
        if any(row['error'] for row in external['supervisors']):
            # A truncated or bad receipt pauses dispatch, never shrinks the
            # budget under running children based on uncertain evidence.
            if external['blocked_receipts'] and not self._running:
                raise base.SchedulerError('External supervisor evidence remains invalid; running jobs drained')
            return None
        other_workers = base.external_worker_count(self.external_runs)
        remaining = max(1, self.total_ceiling - external['reserved_bytes'] - other_workers * 4 * GiB)
        self.limits = replace(self.limits, total_rss_bytes=remaining)
        if len(self._running) + other_workers + external['reserved_slots'] >= self.limits.workers:
            return None
        # Reserve a full worker limit before launch, using observed formal peak
        # RSS for workers already running, plus full limits before first sample.
        formal = sum(state.last_rss_bytes or self.limits.worker_rss_bytes for state in self._running.values())
        if formal + self.limits.worker_rss_bytes > remaining:
            return None
        current, maximum = base.cgroup_memory()
        if current + self.limits.worker_rss_bytes > maximum - base.HOST_RESERVE_BYTES:
            return None
        return base.ParallelBenchmarkScheduler._launchable_index(self, pending, slots)


def claim_jobs(original, phases, run):
    """Persistent claims across queue launches; no implicit rerun/resume release."""
    directory = Path(original).resolve().parent / 'fairbias_recovery_ownership'
    directory.mkdir(exist_ok=True)
    with ExclusiveRunLock(directory / 'claim.lock'):
        claims = []
        for plan, selected in phases:
            for name, entry in selected.items():
                key = base.identity({'registration_sha256': plan['registration_sha256'], 'job': name})
                path = directory / (key + '.json')
                if path.exists():
                    raise base.SchedulerError('Original job already owned by another recovery: ' + name)
                claims.append((path, {'job_id': name, 'run': str(Path(run).resolve()),
                    'runtime_source_identity': plan['runtime_source_identity'],
                    'original_job_sha256': entry['original_job_sha256'],
                    'original_result_sha256': entry['original_result_sha256']}))
        if len({p for p, _ in claims}) != len(claims):
            raise base.SchedulerError('Duplicate ownership across requested phases')
        for path, value in claims:
            if not base.write_json_once(path, value):
                raise base.SchedulerError('Recovery ownership claim raced')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'original-run', 'failure-manifest', 'policy'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--runtime-plan', required=True, action='append', type=Path)
    parser.add_argument('--external-run', action='append', type=Path, default=[])
    parser.add_argument('--workers', type=int, default=28)
    args = parser.parse_args(argv)
    try:
        if not sys.platform.startswith('linux'):
            raise base.SchedulerError('FairBias production queue requires Linux /proc and cgroup accounting')
        if not 1 <= args.workers <= min(28, base._effective_cpu_count()):
            raise base.SchedulerError('FairBias workers exceed CPU/28-slot bound')
        original, run = args.original_run.resolve(), args.run.resolve()
        if (run.exists() or original == run or original in run.parents or run in original.parents
                or any(p.resolve() == run or p.resolve() in run.parents or run in p.resolve().parents
                       for p in args.external_run)):
            raise base.SchedulerError('FairBias run must be fresh and independent')
        registration = base._json(original / 'registration.json')
        policy = base._json(args.policy)
        ceiling = int(policy['max_total_rss_bytes'])
        if not 0 < ceiling <= min(48 * GiB, base.cgroup_memory()[1] - base.HOST_RESERVE_BYTES):
            raise base.SchedulerError('FairBias memory ceiling must preserve 12GiB cgroup reserve')
        monitor = ExternalSupervisors(policy.get('external_supervisors', []))
        phases, schedulers = [], []
        for path in args.runtime_plan:
            plan = runtime_plan(path)
            if (plan.get('registration_sha256') != base.file_sha(original / 'registration.json')
                    or plan.get('failure_manifest_sha256') != base.file_sha(args.failure_manifest)):
                raise base.SchedulerError('FairBias plan evidence binding mismatch')
            if any(plan['runtime_source_files'].get(p) != digest
                   for p, digest in registration.get('source_files', {}).items()):
                raise base.SchedulerError('FairBias plan omits or changes registered source hashes')
            selected = failure_jobs(original, args.failure_manifest, registration, plan['recovery_policy'])
            plan['failure_manifest_path'] = str(args.failure_manifest.resolve())
            budget = plan['execution_budget']
            scheduler = FairBiasRecoveryScheduler(namespace(run, plan['recovery_policy'], plan['runtime_source_identity']),
                original_run=original, selected=selected, runtime_plan=plan, runtime_plan_path=path,
                external_runs=args.external_run, worker=ROOT / worker.WORKER_SOURCE,
                external_supervisors=monitor.bindings, source_script=Path(__file__).resolve(),
                limits=base.ResourceLimits(workers=args.workers, max_workers=28, total_rss_bytes=ceiling,
                    worker_rss_bytes=budget['worker_rss_bytes'], fit_seconds=budget['fit_seconds']),
                methods=[worker.policy_spec(plan['recovery_policy'])['method']], policy_path=args.policy)
            scheduler._validate_sources()
            scheduler._validate_parallel_policy()
            scheduler.supervisors = monitor  # PID identity persists across phases.
            phases.append((plan, selected))
            schedulers.append(scheduler)
        watched = [original / 'registration.json', args.failure_manifest, args.policy,
                   *args.runtime_plan, *(ROOT / p for p in QUEUE_SOURCES)]
        hashes = {str(p.resolve()): base.file_sha(p) for p in watched}
        run.mkdir(parents=True, exist_ok=False)
        claim_jobs(original, phases, run)
        base.write_json_once(run / 'ownership_manifest.json', {'bound_files': hashes,
            'phases': [{'runtime_source_identity': p['runtime_source_identity'],
                        'recovery_policy': p['recovery_policy'], 'selected_jobs': sorted(s)} for p, s in phases]})
        with ExclusiveRunLock(run / 'fairbias_queue.lock'):
            summaries = []
            for scheduler in schedulers:
                if any(base.file_sha(p) != digest for p, digest in hashes.items()):
                    raise base.SchedulerError('Queue plan/source/evidence changed between phases')
                runtime_plan(scheduler.runtime_plan_path)
                fresh = failure_jobs(original, args.failure_manifest, registration,
                                     scheduler.runtime_plan['recovery_policy'])
                if fresh != scheduler.selected:
                    raise base.SchedulerError('Failure evidence changed between phases')
                scheduler.run_path.mkdir(parents=True, exist_ok=False)
                summaries.append({'runtime_source_identity': scheduler._runtime_identity, **scheduler.run()})
            base.write_json_once(run / 'queue_completion.json', {'status': 'COMPLETE',
                'fit_success_established': False, 'phases': summaries})
            print(base.json.dumps(summaries), flush=True)
        return 0
    except (base.SchedulerError, ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        parser.exit(2, f'FairBias recovery queue refused to run: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
