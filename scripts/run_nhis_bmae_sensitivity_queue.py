#!/usr/bin/env python3
"""Dispatch all 40 registered BM_AE jobs under the separately bound budget policy.

No old result is replaced or selected by performance. Admission requires the
previous Joint controller and every bound pilot to have exited. This controller
then owns the complete matrix in a fresh run and enforces 7200s/4GiB per job.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import scripts.run_nhis_bmae_sensitivity_worker as worker

FROZEN_QUEUE = 'scripts/run_nhis_fairbias_recovery_queue.py'
_spec = importlib.util.spec_from_file_location('_bmae_frozen_scheduler', ROOT / FROZEN_QUEUE)
_q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_q)
base, GiB = _q.base, _q.GiB
PLAN_SCHEMA = 'nhis_bmae_sensitivity_plan_v1_20260917'
QUEUE_SOURCES = ('scripts/run_nhis_bmae_sensitivity_queue.py', FROZEN_QUEUE,
                'scripts/run_nhis_numerical_recovery_queue.py',
                'src/nhis_fairbias/benchmark/parallel_execution.py')
_q.worker, _q.PLAN_SCHEMA, _q.COUNTS = worker, PLAN_SCHEMA, {worker.POLICY: 40}
runtime_plan = _q.runtime_plan


def build_sensitivity_manifest(original_run):
    """Metadata-only complete matrix, including the original 19 VALID attempts."""
    original = Path(original_run).resolve()
    registration = base._json(original / 'registration.json')
    jobs = worker.registered_jobs(registration)
    selected = {}
    for name, value in sorted(jobs.items()):
        config, seed = value['config'], value['seed']
        old = original / 'jobs' / name / 'job.json'
        result_path = old.parent / 'result.json'
        job, result = base._json(old), base._json(result_path)
        data = registration['prepared'][config['arm_id']]
        expected_data_path = Path(data.get('path', original / 'prepared' / (config['arm_id'] + '.joblib'))).resolve()
        required = {'config': config, 'seed': seed, 'source_identity': registration['source_identity'],
                    'data_identity': data['data_identity'], 'data_sha256': data['sha256']}
        if (any(job.get(k) != v for k, v in required.items())
                or result.get('candidate_id') != config['candidate_id'] or result.get('seed') != seed
                or result.get('status') not in worker.POLICY_SPEC['original_statuses']
                or job.get('representation_key') is not None
                or Path(job.get('data_path', '')).resolve() != expected_data_path):
            raise base.SchedulerError('Original BM_AE metadata differs from complete registration: ' + name)
        selected[name] = {**value, 'original_job_path': str(old), 'original_job_sha256': base.file_sha(old),
            'original_result_sha256': base.file_sha(result_path), 'original_representation_key': None,
            'original_status': result['status'], 'source_identity': job['source_identity'],
            'data_identity': data['data_identity'], 'data_sha256': data['sha256']}
    return {'schema_version': worker.MANIFEST_SCHEMA, 'recovery_policy': worker.POLICY,
            'original_run': str(original), 'registration_sha256': base.file_sha(original / 'registration.json'),
            'jobs': selected}


def sensitivity_jobs(original_run, manifest_path):
    expected = build_sensitivity_manifest(original_run)
    if base._json(manifest_path) != expected:
        raise base.SchedulerError('BM_AE sensitivity manifest must exactly bind all 40 original jobs/results')
    return expected['jobs']


def live_training(exclude_run=None, proc_root=Path('/proc')):
    """Count actual project workers and pilots, not only --job worker patterns."""
    found = []
    for proc in Path(proc_root).iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z':
                continue
            argv = [v.decode() for v in (proc / 'cmdline').read_bytes().split(b'\0') if v]
            scripts = [Path(v) for v in argv if v.endswith('.py')]
            if not any(p.is_absolute() and p.parent == ROOT / 'scripts' for p in scripts):
                continue
            is_worker = 'worker' in argv and '--job' in argv
            is_pilot = any(p.name.startswith('pilot_') for p in scripts)
            if not (is_worker or is_pilot):
                continue
            if is_worker and exclude_run is not None:
                job = Path(argv[argv.index('--job') + 1]).resolve()
                if job.parent.parent == Path(exclude_run).resolve() / 'jobs':
                    continue
            found.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return sorted(found)


def predecessor_ready(binding, *, probe=None, training=None):
    """Read-only hard gate: scheduler completion is separate from fit success."""
    root = Path(binding['queue_root']).resolve()
    joint = Path(binding['joint_run']).resolve()
    if not joint.is_relative_to(root) or not joint.name.endswith('_' + binding['joint_runtime_source_identity']):
        raise base.SchedulerError('BM_AE predecessor run/runtime binding mismatch')
    completion_path = root / 'queue_completion.json'
    if not completion_path.is_file() or base.file_sha(completion_path) != binding['completion_sha256']:
        raise base.SchedulerError('BM_AE predecessor completion evidence missing or changed')
    completion = base._json(completion_path)
    phases = [p for p in completion.get('phases', [])
              if p.get('runtime_source_identity') == binding['joint_runtime_source_identity']]
    if (completion.get('status') != 'COMPLETE' or len(phases) != 1
            or phases[0].get('scheduled') != 38 or phases[0].get('completed') != 38):
        raise base.SchedulerError('BM_AE predecessor Joint phase is not complete')
    status = base._json(joint / 'parallel_scheduler_live_status.json')
    if any(status.get(k) != v for k, v in {'status': 'COMPLETE', 'scheduled_count': 38,
            'finalized_count': 38, 'active_count': 0, 'run': str(joint)}.items()):
        raise base.SchedulerError('BM_AE predecessor Joint live status is not complete')
    pid = binding['controller_pid']
    if type(pid) is not int or pid <= 0:
        raise base.SchedulerError('BM_AE predecessor controller PID is invalid')
    if (probe or _q.process_identity)(pid) is not None:
        raise base.SchedulerError('Previous controller remains alive or PID was reused')
    sample = _q.ExternalSupervisors(binding['pilot_supervisors'], probe=probe).sample()
    if sample['reserved_slots'] or sample['blocked_receipts']:
        raise base.SchedulerError('A pilot is active or its terminal evidence is unverified')
    active = (training or live_training)()
    if active:
        raise base.SchedulerError('Residual training/pilot processes block BM_AE sensitivity')
    return {'predecessor_completion_sha256': binding['completion_sha256'],
            'joint_runtime_source_identity': binding['joint_runtime_source_identity'],
            'controller_absent': True, 'pilots_terminal_and_absent': True, 'other_training_absent': True}


class BMAESensitivityScheduler(_q.FairBiasRecoveryScheduler):
    def _job_payload(self, spec):
        return {**super()._job_payload(spec),
                'sensitivity_manifest_path': self.runtime_plan['sensitivity_manifest_path'],
                'sensitivity_manifest_sha256': self.runtime_plan['sensitivity_manifest_sha256']}

    def _write_manifest(self):
        base.ParallelBenchmarkScheduler._write_manifest(self)
        marker = {'runtime_variant': worker.VARIANT, 'runtime_source_identity': self._runtime_identity,
                  'registration_sha256': base.file_sha(self.registration_path)}
        contract = {**marker, 'schema_version': PLAN_SCHEMA, 'recovery_policy': worker.POLICY,
            'recovery_policy_spec': worker.policy_spec(), 'execution_budget': worker.DEFAULT_EXECUTION_BUDGET,
            'runtime_plan_sha256': base.file_sha(self.runtime_plan_path),
            'sensitivity_manifest_sha256': self.runtime_plan['sensitivity_manifest_sha256'],
            'queue_source_files': {p: base.file_sha(ROOT / p) for p in QUEUE_SOURCES},
            'selected_jobs': len(self.selected), 'selected_original_evidence': self.selected,
            'complete_matrix_policy': True, 'old_results_preserved': True,
            'resource_rule': 'global_slots<=28; formal_peak_RSS<=48GiB; cgroup_free>=12GiB',
            'sampled_limits_not_kernel_caps': True}
        for path, value in ((self.cache_path / '.runtime_namespace.json', marker),
                            (self.run_path / 'recovery_dispatch_manifest.json', contract)):
            if path.exists() or not base.write_json_once(path, value):
                raise base.SchedulerError('BM_AE dispatch/cache namespace is not fresh')

    def _receipt_error(self, state):
        error = base.NumericalRecoveryScheduler._receipt_error(self, state)
        if error:
            return error
        receipt = base._json(state.spec.path / 'runtime_receipt.json')
        expected = {'schema_version': worker.RECEIPT_SCHEMA, 'recovery_policy': worker.POLICY,
            'recovery_policy_spec': worker.policy_spec(), 'execution_budget': worker.DEFAULT_EXECUTION_BUDGET,
            'runtime_source_files': self.runtime_plan['runtime_source_files'],
            'runtime_environment': worker.THREAD_ENVIRONMENT, 'data_identity': state.spec.data['data_identity'],
            'data_sha256': state.spec.data['sha256'], 'representation_key': None,
            'representation_cache_reused': False, 'cache_path': str(self.cache_path),
            'sensitivity_manifest_sha256': self.runtime_plan['sensitivity_manifest_sha256'],
            'sensitivity_manifest_path': self.runtime_plan['sensitivity_manifest_path'],
            'original_status': self.selected[state.spec.path.name]['original_status'],
            'original_valid_job_refit': self.selected[state.spec.path.name]['original_status'] == 'VALID',
            'legacy_valid_results_admitted': False}
        if any(receipt.get(k) != v for k, v in expected.items()):
            return 'BM_AE sensitivity receipt identity/policy mismatch'
        if receipt['status'] == 'VALID':
            admission = receipt.get('model_admission', {})
            if (receipt.get('reload_exact_C') is not True or admission.get('status') != 'COMPLETE_FEASIBLE'
                    or admission.get('search_complete') is not True
                    or admission.get('geometry_search_incomplete') is not False):
                return 'BM_AE sensitivity VALID lacks complete model/C evidence'
        return None

    def _write_scheduler_failure(self, state, reason):
        base.NumericalRecoveryScheduler._write_scheduler_failure(self, state, reason)
        path = state.spec.path / 'runtime_receipt.json'
        receipt = base._json(path)
        receipt.update(schema_version=worker.RECEIPT_SCHEMA, recovery_policy=worker.POLICY,
            recovery_policy_spec=worker.policy_spec(), execution_budget=worker.DEFAULT_EXECUTION_BUDGET,
            sensitivity_manifest_path=self.runtime_plan['sensitivity_manifest_path'],
            sensitivity_manifest_sha256=self.runtime_plan['sensitivity_manifest_sha256'])
        base.replace_json(path, receipt)

    def _launchable_index(self, pending, slots):
        # A newly introduced external training process pauses admissions while
        # existing jobs remain governed by their already declared limits.
        if live_training(exclude_run=self.run_path):
            return None
        return super()._launchable_index(pending, slots)


def claim_jobs(original, selected, plan, run):
    # Independent scientific variant: original recovery ownership remains
    # immutable, while all sensitivity attempts have their own permanent claims.
    directory = Path(original).resolve().parent / 'bmae_sensitivity_ownership'
    directory.mkdir(exist_ok=True)
    with _q.ExclusiveRunLock(directory / 'claim.lock'):
        paths = {n: directory / (base.identity({'registration_sha256': plan['registration_sha256'],
                 'variant': worker.VARIANT, 'job': n}) + '.json') for n in selected}
        if any(p.exists() for p in paths.values()):
            raise base.SchedulerError('BM_AE sensitivity original job already has an owner')
        for name, path in paths.items():
            if not base.write_json_once(path, {'job_id': name, 'run': str(run),
                'runtime_source_identity': plan['runtime_source_identity'], **selected[name]}):
                raise base.SchedulerError('BM_AE sensitivity ownership claim raced')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'original-run', 'sensitivity-manifest', 'runtime-plan', 'policy'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--workers', type=int, default=28)
    args = parser.parse_args(argv)
    try:
        if not sys.platform.startswith('linux') or not 1 <= args.workers <= min(28, base._effective_cpu_count()):
            raise base.SchedulerError('BM_AE queue requires Linux and at most 28 available CPU slots')
        original, root = args.original_run.resolve(), args.run.resolve()
        if root.exists() or original == root or original in root.parents or root in original.parents:
            raise base.SchedulerError('BM_AE output root must be fresh and independent')
        plan, policy = runtime_plan(args.runtime_plan), base._json(args.policy)
        if (plan.get('registration_sha256') != base.file_sha(original / 'registration.json')
                or plan.get('sensitivity_manifest_sha256') != base.file_sha(args.sensitivity_manifest)
                or Path(plan.get('sensitivity_manifest_path', '')).resolve() != args.sensitivity_manifest.resolve()):
            raise base.SchedulerError('BM_AE runtime plan manifest/registration mismatch')
        selected = sensitivity_jobs(original, args.sensitivity_manifest)
        registration = base._json(original / 'registration.json')
        if any(plan['runtime_source_files'].get(p) != digest
               for p, digest in registration.get('source_files', {}).items()):
            raise base.SchedulerError('BM_AE plan omits or changes registered source hashes')
        ceiling = int(policy['max_total_rss_bytes'])
        if not 0 < ceiling <= min(48 * GiB, base.cgroup_memory()[1] - 12 * GiB):
            raise base.SchedulerError('BM_AE policy must preserve 12GiB host memory')
        target = root / ('bmae_sensitivity_' + plan['runtime_source_identity'])
        scheduler = BMAESensitivityScheduler(target, original_run=original, selected=selected,
            runtime_plan=plan, runtime_plan_path=args.runtime_plan, external_runs=(),
            worker=ROOT / worker.WORKER_SOURCE, source_script=Path(__file__),
            limits=base.ResourceLimits(workers=args.workers, max_workers=28, total_rss_bytes=ceiling,
                worker_rss_bytes=4*GiB, fit_seconds=7200),
            methods=['FAIRBIAS_BM_AE'], policy_path=args.policy)
        scheduler._validate_sources()
        scheduler._validate_parallel_policy()
        gate = predecessor_ready(policy['predecessor'])
        watched = [original / 'registration.json', args.sensitivity_manifest, args.runtime_plan, args.policy,
                   *(ROOT / p for p in QUEUE_SOURCES)]
        hashes = {str(p.resolve()): base.file_sha(p) for p in watched}
        root.mkdir(parents=True, exist_ok=False)
        claim_jobs(original, selected, plan, root)
        base.write_json_once(root / 'ownership_manifest.json', {'bound_files': hashes, 'gate': gate,
            'runtime_source_identity': plan['runtime_source_identity'], 'jobs': selected})
        with _q.ExclusiveRunLock(root / 'bmae_queue.lock'):
            if any(base.file_sha(p) != digest for p, digest in hashes.items()):
                raise base.SchedulerError('BM_AE bound inputs changed before dispatch')
            predecessor_ready(policy['predecessor'])
            target.mkdir(exist_ok=False)
            summary = scheduler.run()
            base.write_json_once(root / 'queue_completion.json', {'status': 'COMPLETE',
                'fit_success_established': False, 'runtime_source_identity': plan['runtime_source_identity'],
                'summary': summary})
            print(base.json.dumps(summary), flush=True)
        return 0
    except (base.SchedulerError, ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        parser.exit(2, f'BM_AE sensitivity queue refused to run: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
