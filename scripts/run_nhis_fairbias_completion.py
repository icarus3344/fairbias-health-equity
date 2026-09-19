"""Register all 80 full-method tasks and run one isolated F/C completion pilot.

No selection or T evaluation is exposed. Old registration and jobs are read
only. Run each pilot in a fresh, single-thread process. No wall-time timeout
is imposed here; external interruption remains an incomplete attempt.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_SHA256 = '340cd60f6ab0d5919e2ac7bc22b62bdb3cc758faaa986650abdb3978b964732a'
VERSION = 'full_method_completion_fc_v1'
METHODS = ('FAIRBIAS_BM_AE', 'FAIRBIAS_JOINT')
SEEDS = (0, 7, 19, 37, 73)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def source_hashes():
    paths = [p for package in ('fairbias', 'nhis_fairbias')
             for p in (ROOT / 'src' / package).rglob('*.py')]
    paths += [Path(__file__).resolve(), ROOT / 'scripts' / 'pilot_scheduled_joint.py']
    paths += [ROOT / 'scripts' / 'run_nhis_completion_pilot_queue.py']
    paths += [ROOT / p for p in ('scripts/run_nhis_benchmark.py', 'configs/nhis/features.json',
                                'configs/nhis/study.json', 'configs/nhis/variables.json')]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def validate_coverage(jobs):
    expected = {(method, f'arm_{arm:03d}', backbone, seed)
                for method in METHODS for arm in range(1, 5)
                for backbone in ('LR', 'GBDT') for seed in SEEDS}
    keys = [(j['config']['method'], j['config']['arm_id'], j['config']['backbone'], j['seed']) for j in jobs]
    if len(keys) != 80 or set(keys) != expected or len({j['job_id'] for j in jobs}) != 80:
        raise ValueError('Completion requires exactly the original 80 unique tasks')


def prepare(registration_path, output):
    registration_path, output = Path(registration_path).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Plan output must be fresh')
    if sha(registration_path) != REGISTRATION_SHA256:
        raise ValueError('Original registration hash differs')
    original = json.loads(registration_path.read_text())
    jobs = []
    for config in original['candidates']:
        if config['method'] not in METHODS:
            continue
        for seed in config['seeds']:
            job_id = f"{config['candidate_id']}_s{seed}"
            path = registration_path.parent / 'jobs' / job_id / 'job.json'
            job = json.loads(path.read_text())
            prepared = original['prepared'][config['arm_id']]
            if (job['config'] != config or job['seed'] != seed
                    or job['source_identity'] != original['source_identity']
                    or job['data_identity'] != prepared['data_identity']
                    or job['data_sha256'] != prepared['sha256']):
                raise ValueError('Original job does not match registration')
            jobs.append({'job_id': job_id, 'config': config, 'seed': seed,
                         'original_job_sha256': sha(path), 'prepared': prepared})
    validate_coverage(jobs)
    sources = source_hashes()
    for path, digest in original['source_files'].items():
        if sources.get(path) != digest:
            raise ValueError('Original registered source was changed: ' + path)
    value = {'version': VERSION, 'status': 'FC_PILOT_READY_NOT_FORMAL_ADMISSION',
             'registration_sha256': REGISTRATION_SHA256,
             'original_source_identity': original['source_identity'],
             'sources': sources, 'jobs': sorted(jobs, key=lambda j: j['job_id']),
             'initial_ae_cap': 40, 'budget_extension': 'exhausted_limit_times_two',
             'mds_extension': 'same_seed_all_initializations_max_iter_times_two',
             'max_replays_per_session': None, 'wall_timeout_seconds': None,
             'cache_namespace': 'fairbias_completion_v1_' + REGISTRATION_SHA256,
             'T_already_known': True, 'repair_based_on': 'F_C_and_failure_evidence_only',
             'formal_benchmark_admission': False, 'S_T_evaluation_authorized': False}
    value['manifest_sha256'] = seal(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return value


def load_manifest(path):
    value = json.loads(Path(path).read_text())
    digest = value.pop('manifest_sha256')
    if seal(value) != digest or value.get('version') != VERSION:
        raise ValueError('Completion manifest integrity/version failure')
    if (value['registration_sha256'] != REGISTRATION_SHA256 or value['initial_ae_cap'] != 40
            or value['S_T_evaluation_authorized'] is not False
            or value['formal_benchmark_admission'] is not False
            or value['budget_extension'] != 'exhausted_limit_times_two'
            or value['mds_extension'] != 'same_seed_all_initializations_max_iter_times_two'
            or value['max_replays_per_session'] is not None or value['wall_timeout_seconds'] is not None):
        raise ValueError('Completion policy contract mismatch')
    if source_hashes() != value['sources']:
        raise ValueError('Completion source closure changed after registration')
    validate_coverage(value['jobs'])
    value['manifest_sha256'] = digest
    return value


class ExternalStop(BaseException):
    pass


def pilot(manifest_path, job_id, prepared_path, output, cache_dir):
    manifest = load_manifest(manifest_path)
    matches = [j for j in manifest['jobs'] if j['job_id'] == job_id]
    if len(matches) != 1:
        raise ValueError('One known completion task is required')
    job = matches[0]
    output, prepared_path = Path(output).resolve(), Path(prepared_path).resolve()
    if output.exists() or sha(prepared_path) != job['prepared']['sha256']:
        raise ValueError('Output must be fresh and prepared input hash must match')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError('Pilot requires a single-thread numerical environment')
    import joblib
    import numpy as np
    from pilot_scheduled_joint import runtime_modules
    from nhis_fairbias.benchmark.fairbias_completion import make_completion_adapter
    from nhis_fairbias.benchmark.predictions import FrozenDecisionPolicy
    project = {p: h for p, h in manifest['sources'].items() if p.startswith('src/')}
    runtime_modules(ROOT, project)
    # The registered payload contains F/C/S. Its bytes are hashed and unpickled;
    # S is immediately discarded without inspecting arrays, labels or metrics.
    data = joblib.load(prepared_path)
    if (data['data_identity'] != job['prepared']['data_identity']
            or set(data['partitions']) != {'fitting_F', 'calibration_C', 'selection_S'}
            or 'X_T' in data):
        raise ValueError('Prepared data identity or partition mismatch')
    F, C = data['partitions']['fitting_F'], data['partitions']['calibration_C']
    del data
    for partition, role in ((F, 'fitting_F'), (C, 'calibration_C')):
        sizes = [len(partition.X_semantic), len(partition.y), len(partition.A), len(partition.record_keys)]
        if (partition.role != role or partition.year != 2022
                or partition.arm_id != job['config']['arm_id'] or not sizes[0]
                or len(set(sizes)) != 1 or len(set(partition.record_keys)) != sizes[0]):
            raise ValueError('F/C partition contract mismatch')
    if set(F.record_keys) & set(C.record_keys):
        raise ValueError('F/C record identity overlap')
    runtime_modules(ROOT, project)
    output.mkdir(parents=True, exist_ok=False)
    adapter = make_completion_adapter(job['config'], job['seed'], cache_dir=cache_dir,
                                     progress_path=output / 'progress.jsonl',
                                     namespace=manifest['cache_namespace'])
    record = {'version': VERSION, 'status': 'STARTED', 'job_id': job_id,
              'manifest_sha256': manifest['manifest_sha256'], 'config': job['config'], 'seed': job['seed'],
              'prepared_sha256': sha(prepared_path), 'data_identity': job['prepared']['data_identity'],
              'packages': {name: importlib.metadata.version(name) for name in (
                  'numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib', 'threadpoolctl')},
              'formal_benchmark_admission': False, 'S_T_evaluated': False,
              'partitions_used_for_fitting': ['fitting_F', 'calibration_C']}
    def save():
        pending = output / 'result.json.part'
        with pending.open('w') as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        pending.replace(output / 'result.json')
    def stop(signum, frame):
        raise ExternalStop()
    previous = signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    save()
    try:
        adapter.fit_development(F.X_semantic, F.y, F.A, C.X_semantic, C.y, C.A,
                                metadata={'F_ids': F.record_keys, 'C_ids': C.record_keys})
        policy = FrozenDecisionPolicy().fit_calibration(adapter, C.X_semantic, C.y, C.A)
        before = policy.predict(C.X_semantic, C.A)
        model_path = output / 'policy.joblib'
        joblib.dump(policy, model_path)
        after = joblib.load(model_path).predict(C.X_semantic, C.A)
        if (before.p_event is None or after.p_event is None
                or not np.array_equal(before.q_decision, after.q_decision)
                or not np.array_equal(before.p_event, after.p_event)):
            raise ValueError('Reload predictions are not exact')
        record.update(status='FC_COMPLETE_FEASIBLE', reload_exact=True, model_sha256=sha(model_path))
    except BaseException as exc:
        record.update(status='FAILED_REQUIRES_REPAIR' if isinstance(exc, Exception) else 'INTERRUPTED_REPLAY_REQUIRED',
                      error_type=type(exc).__name__)
    finally:
        signal.signal(signal.SIGTERM, previous)
        record.update(elapsed_seconds=time.monotonic() - started,
                      completion=getattr(adapter, 'provenance_', {}))
        try:
            record['loaded_modules'] = runtime_modules(ROOT, project)
            if source_hashes() != manifest['sources']:
                raise ValueError('Source changed during completion pilot')
            if sha(prepared_path) != job['prepared']['sha256']:
                raise ValueError('Prepared input changed during completion pilot')
        except Exception as exc:
            record.update(status='FAILED_REQUIRES_REPAIR', error_type=type(exc).__name__, failure_phase='source_integrity')
        save()
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--registration', type=Path, required=True)
    plan.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('pilot')
    for arg in ('manifest', 'prepared', 'output', 'cache-dir'):
        run.add_argument('--' + arg, type=Path, required=True)
    run.add_argument('--job-id', required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        result = prepare(args.registration, args.output)
        print(json.dumps({'status': result['status'], 'jobs': len(result['jobs']),
                          'methods': dict(Counter(j['config']['method'] for j in result['jobs'])),
                          'manifest_sha256': result['manifest_sha256']}))
    else:
        result = pilot(args.manifest, args.job_id, args.prepared, args.output, args.cache_dir)
        print(json.dumps({'status': result['status'], 'job_id': args.job_id}))
        raise SystemExit(0 if result['status'] == 'FC_COMPLETE_FEASIBLE' else 1)
