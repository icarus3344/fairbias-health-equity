"""One opt-in FairBias recovery job over the frozen production worker.

Policies are immutable runtime identities, not overrides of scientific config.
Joint retains strict search and every registered budget. BM permits the reviewed
single fresh doubled-cap MDS retry, with original predictor weighting preserved.
BM_AE cap40 is deliberately unavailable until its pilot is independently admitted.
The callable is process-scoped, not thread-safe; imports never launch work.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
WORKER_SOURCE = 'scripts/run_nhis_fairbias_recovery_worker.py'
PRIMITIVES_SOURCE = 'scripts/run_nhis_numerical_recovery_worker.py'
VARIANT = 'fairbias_recovery_v1_20260917'
DEFAULT_EXECUTION_BUDGET = {'fit_seconds': 1800.0, 'worker_rss_bytes': 4 * 1024 ** 3}
POLICIES = {
    'joint_strict_exact_v1': {
        'method': 'FAIRBIAS_JOINT', 'controller': 'strict',
        'exact_kernel': 'joint_numpy_mds_v1', 'use_mds_retry': False,
        'ae_cap_retry': None, 'registered_budgets_unchanged': True,
        'original_statuses': ['TIME_LIMIT'], 'training_weighted_supported': False,
    },
    'bm_mds_retry_v1': {
        'method': 'FAIRBIAS_BM', 'exact_kernel': 'joint_numpy_mds_v1',
        'use_mds_retry': True, 'mds_retry': 'mds_budget_retry_v1',
        'retry_policy': 'one_fresh_seeded_refit_at_double_iteration_cap',
        'max_factor': 2, 'registered_outer_budgets_unchanged': True,
        'original_statuses': ['BUDGET_EXHAUSTED'],
        'representation_weighted': False, 'training_weighted_supported': True,
    },
}
BLOCKED_POLICIES = {'bmae_cap40_v1': 'BM_AE cap40 pilot has no completed model; separate admission required'}

# Reuse the already tested helpers in a private module namespace. Rebinding its
# constants cannot alter the live EG/LFR worker's module or scientific sources.
_spec = importlib.util.spec_from_file_location('_fairbias_recovery_primitives', ROOT / PRIMITIVES_SOURCE)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
THREAD_ENVIRONMENT = {**_base.THREAD_ENVIRONMENT, 'PYTHONHASHSEED': '0'}
_base.VARIANT = VARIANT
_base.WORKER_SOURCE = WORKER_SOURCE
_base.THREAD_ENVIRONMENT = THREAD_ENVIRONMENT
file_sha, identity = _base.file_sha, _base.identity
_read_json, _write_once, _lock = _base._read_json, _base._write_once, _base._lock
runtime_modules = _base.runtime_modules
initialize_cache, verify_cache_entry = _base.initialize_cache, _base.verify_cache_entry
_cache_evidence = _base._cache_evidence
_cache_namespace = _base._cache_namespace
CACHE_NAMESPACE_FILE = _base.CACHE_NAMESPACE_FILE
REQUIRED_SOURCES = {
    WORKER_SOURCE, PRIMITIVES_SOURCE, 'scripts/run_nhis_benchmark.py',
    'src/nhis_fairbias/benchmark/experiment_worker.py',
    'src/nhis_fairbias/benchmark/experiment_registry.py',
    'src/nhis_fairbias/benchmark/predictions.py',
    'src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py',
    'src/nhis_fairbias/benchmark/adapters/adapter_fairbias_ae.py',
    'src/nhis_fairbias/benchmark/adapters/adapter_bm_recovery.py',
    'src/nhis_fairbias/benchmark/adapters/adapter_joint_ae_recovery.py',
    'src/nhis_fairbias/benchmark/adapters/geometry_audit.py',
    'src/nhis_fairbias/benchmark/joint_budget_optimization.py',
    'src/nhis_fairbias/benchmark/joint_mds_numpy.py',
    'src/nhis_fairbias/benchmark/mds_budget_retry.py',
}


def policy_spec(policy):
    if not isinstance(policy, str) or policy not in POLICIES:
        reason = BLOCKED_POLICIES.get(policy, 'Unknown FairBias recovery policy') if isinstance(policy, str) else 'Invalid recovery policy'
        raise ValueError(reason)
    return copy.deepcopy(POLICIES[policy])


def build_runtime_source_manifest(root=ROOT):
    files = _base.build_runtime_source_manifest(root)
    files[PRIMITIVES_SOURCE] = file_sha(Path(root) / PRIMITIVES_SOURCE)
    return files


def execution_budget_spec(value=None):
    value = DEFAULT_EXECUTION_BUDGET if value is None else value
    if (not isinstance(value, dict) or set(value) != {'fit_seconds', 'worker_rss_bytes'}
            or isinstance(value['fit_seconds'], bool) or not isinstance(value['fit_seconds'], (int, float))
            or not math.isfinite(value['fit_seconds']) or value['fit_seconds'] <= 0
            or type(value['worker_rss_bytes']) is not int or value['worker_rss_bytes'] <= 0):
        raise ValueError('Invalid explicitly declared execution budget')
    return {'fit_seconds': float(value['fit_seconds']), 'worker_rss_bytes': value['worker_rss_bytes']}


def runtime_identity(files, environment=None, *, policy, execution_budget=None):
    return identity({'variant': VARIANT, 'policy': policy, 'policy_spec': policy_spec(policy),
                     'execution_budget': execution_budget_spec(execution_budget),
                     'files': files, 'environment': THREAD_ENVIRONMENT if environment is None else environment})


def validate_runtime_sources(job):
    policy = job.get('recovery_policy')
    policy_spec(policy)
    if 'execution_budget' not in job:
        raise ValueError('Missing explicitly declared execution budget')
    budget = execution_budget_spec(job['execution_budget'])
    if job.get('runtime_variant') != VARIANT or job.get('runtime_environment') != THREAD_ENVIRONMENT:
        raise ValueError('FairBias runtime variant/environment mismatch')
    if any(os.environ.get(k) != v for k, v in THREAD_ENVIRONMENT.items()) or sys.flags.hash_randomization != 0:
        raise ValueError('FairBias requires single threads and PYTHONHASHSEED=0 at process startup')
    if (Path(__file__).resolve() != ROOT / WORKER_SOURCE
            or Path(_base.__file__).resolve() != ROOT / PRIMITIVES_SOURCE):
        raise ValueError('Recovery scripts loaded outside the source checkout')
    files = job.get('runtime_source_files')
    if not isinstance(files, dict) or not REQUIRED_SOURCES.issubset(files):
        raise ValueError('Runtime source manifest is incomplete')
    for rel, digest in files.items():
        path = (ROOT / rel).resolve()
        if (Path(rel).is_absolute() or not path.is_relative_to(ROOT)
                or not path.is_file() or file_sha(path) != digest):
            raise ValueError('Runtime source manifest/hash mismatch: ' + rel)
    if job.get('runtime_source_identity') != runtime_identity(files, job['runtime_environment'], policy=policy, execution_budget=budget):
        raise ValueError('Runtime source identity mismatch')


def validate_job(job_path):
    """Keep original candidate/seed/data immutable; weighted BM is intentional."""
    job_path = Path(job_path).resolve()
    job = _read_json(job_path)
    validate_runtime_sources(job)
    spec = policy_spec(job['recovery_policy'])
    run, rid = Path(job['recovery_run_path']).resolve(), job['runtime_source_identity']
    if run.name != rid and not run.name.endswith('_' + rid):
        raise ValueError('Recovery run path must contain its full runtime identity')
    if job_path.name != 'job.json' or job_path.parent.parent != run / 'jobs':
        raise ValueError('Recovery job must be in its independent run/jobs directory')
    registration_path = Path(job['registration_path']).resolve()
    original_run = registration_path.parent
    if run == original_run or original_run in run.parents or run in original_run.parents:
        raise ValueError('Recovery run must be independent of the original run')
    if file_sha(registration_path) != job['registration_sha256']:
        raise ValueError('Original registration hash mismatch')
    registration = _read_json(registration_path)
    original_path = Path(job['original_job_path']).resolve()
    if (original_path.name != 'job.json' or original_path.parent.parent != original_run / 'jobs'
            or job_path.parent.name != original_path.parent.name):
        raise ValueError('Original/recovery job paths disagree')
    if file_sha(original_path) != job['original_job_sha256']:
        raise ValueError('Original job hash mismatch')
    original = _read_json(original_path)
    config = job['config']
    if config.get('method') != spec['method']:
        raise ValueError('Recovery method/policy mismatch')
    if config.get('training_weighted', False) and not spec['training_weighted_supported']:
        raise ValueError('Weighted strict Joint is not supported by the registered adapter')
    candidates = [c for c in registration['candidates'] if c.get('candidate_id') == config.get('candidate_id')]
    if (candidates != [config] or config.get('status') != 'REGISTERED'
            or job['seed'] not in config['seeds']):
        raise ValueError('Recovery candidate/seed differs from original registration')
    for key in ('config', 'seed', 'source_identity', 'data_identity', 'data_sha256'):
        if job.get(key) != original.get(key):
            raise ValueError('Recovery changed original job identity: ' + key)
    if job['source_identity'] != registration['source_identity']:
        raise ValueError('Registered source identity mismatch')
    sources = registration.get('source_files')
    if not isinstance(sources, dict) or not sources or not set(sources).issubset(job['runtime_source_files']):
        raise ValueError('Runtime snapshot omits supervisor-registered source files')
    if any(job['runtime_source_files'][name] != digest for name, digest in sources.items()):
        raise ValueError('Supervisor-registered source hash mismatch')
    prepared = registration['prepared'][config['arm_id']]
    data_path = Path(job['data_path']).resolve()
    expected_path = Path(prepared.get('path', original_run / 'prepared' / (config['arm_id'] + '.joblib'))).resolve()
    if data_path != Path(original['data_path']).resolve() or data_path != expected_path:
        raise ValueError('Recovery prepared path differs from original registration')
    if (prepared['sha256'] != job['data_sha256'] or prepared['data_identity'] != job['data_identity']
            or file_sha(data_path) != job['data_sha256']):
        raise ValueError('Registered prepared data hash/identity mismatch')
    previous_path = original_path.parent / 'result.json'
    if file_sha(previous_path) != job['original_result_sha256']:
        raise ValueError('Original result hash mismatch')
    previous = _read_json(previous_path)
    if (previous.get('candidate_id') != config['candidate_id'] or previous.get('seed') != job['seed']
            or previous.get('status') not in spec['original_statuses']):
        raise ValueError('Original result identity/status is outside this recovery policy')
    cache, old_cache = Path(job['cache_path']).resolve(), Path(original['cache_path']).resolve()
    if cache != run / 'representation_cache' or cache == old_cache or old_cache in cache.parents or cache in old_cache.parents:
        raise ValueError('Recovery requires an independent runtime cache namespace')
    outputs = ('result.json', 'model.joblib', 'predictions_S.npz', 'receipt.json', 'runtime_receipt.json', 'runtime_started.json')
    if any((job_path.parent / name).exists() for name in outputs):
        raise ValueError('Recovery attempt already has output; use a fresh attempt')
    return job, run, previous


def _admit_adapter(adapter, policy):
    """Fail before C calibration or S prediction if geometry/model is incomplete."""
    from nhis_fairbias.benchmark.adapters.adapter_bm_recovery import BMRecoveryAdapter
    from nhis_fairbias.benchmark.adapters.adapter_joint_ae_recovery import JointAERecoveryAdapter, classify_recovery_fit
    if policy == 'joint_strict_exact_v1':
        if (type(adapter) is not JointAERecoveryAdapter or adapter.controller != 'strict'
                or adapter.mode != 'JOINT' or adapter.use_mds_retry or adapter.ae_cap_retry is not None
                or not adapter.is_fitted_):
            raise ValueError('Strict Joint model/policy admission failed')
        outcome = classify_recovery_fit(adapter._delegate, controller='strict')
        if adapter.recovery_result_ != outcome:
            raise ValueError('Strict Joint recovery result contradicts fitted model')
        encoder = adapter._delegate.preprocessor_
    elif policy == 'bm_mds_retry_v1':
        if (type(adapter) is not BMRecoveryAdapter or not adapter.use_mds_retry
                or not adapter.converged_ or not adapter.fitted_
                or adapter.termination_reason_ != 'epsilon_reached'):
            raise ValueError('BM recovery model is not converged and fitted')
        manifest, receipt = adapter.fit_manifest_, adapter.recovery_receipt_
        final, threshold = manifest.get('final_max_dphi'), manifest.get('epsilon_threshold')
        if (not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in (final, threshold))
                or final > threshold or receipt.get('status') != 'COMPLETE_FEASIBLE'
                or receipt.get('mds_budget_changed') is not True):
            raise ValueError('BM recovery lacks finite final geometry feasibility')
        diagnostics = adapter.mds_diagnostics_
        retry = receipt.get('mds_retry')
        if (any(row.get('status') != 'CONVERGED_BEFORE_CAP' for row in diagnostics)
                or not isinstance(retry, dict) or retry.get('version') != 'mds_budget_retry_v1'
                or any(row.get('complete') is not True for row in retry.get('fits', []))):
            raise ValueError('BM recovery has incomplete geometry or MDS retry')
        outcome = {'status': 'COMPLETE_FEASIBLE', 'final_max_dphi': final, 'epsilon_threshold': threshold,
                   'mds_retry_policy': 'mds_budget_retry_v1', 'global_infeasibility_claim': False}
        encoder = adapter.preprocessor_
    else:
        policy_spec(policy)
        raise ValueError('Unsupported model admission policy')
    if (not getattr(encoder, 'fitted_', False) or not getattr(encoder, 'feature_names', ())
            or not getattr(encoder, 'transformed_feature_names_', ())):
        raise ValueError('Recovery refuses empty or unfitted feature representation')
    return outcome


def execute_recovery_job(job_path):
    job_path = Path(job_path).resolve()
    job, _, previous = validate_job(job_path)
    sys.path.insert(0, str(ROOT / 'src'))
    import joblib
    import numpy as np
    from nhis_fairbias.benchmark import experiment_worker as worker
    from nhis_fairbias.benchmark.adapters.adapter_bm_recovery import BMRecoveryAdapter
    from nhis_fairbias.benchmark.adapters.adapter_joint_ae_recovery import make_joint_ae_recovery_adapter
    sources, policy = job['runtime_source_files'], job['recovery_policy']
    before = runtime_modules(sources)
    initialize_cache(job)
    key = worker.representation_key(job['config'], job['seed'], job['data_identity'])
    cache, out = Path(job['cache_path']), job_path.parent
    job_hash = file_sha(job_path)
    with _lock(cache / ((key or out.name) + '.lock')):
        reused = verify_cache_entry(job, key)
        _write_once(out / 'runtime_started.json', {'runtime_variant': VARIANT, 'recovery_policy': policy,
            'runtime_source_identity': job['runtime_source_identity'], 'job_sha256': job_hash, 'pid': os.getpid()})
        receipt = {k: job[k] for k in ('runtime_source_identity', 'runtime_source_files', 'runtime_environment',
            'registration_sha256', 'original_job_sha256', 'original_result_sha256', 'data_sha256', 'data_identity')}
        receipt.update(schema_version='nhis_fairbias_recovery_receipt_v1', runtime_variant=VARIANT,
            recovery_policy=policy, recovery_policy_spec=policy_spec(policy), original_status=previous['status'],
            execution_budget=job['execution_budget'], execution_budget_enforcement='external_scheduler',
            candidate_id=job['config']['candidate_id'], seed=job['seed'], cache_path=str(cache),
            representation_key=key, representation_cache_reused=reused, loaded_project_modules_before=before,
            legacy_valid_results_admitted=False, status='RUNTIME_FAILED', runtime_checks_passed=False)
        calibration = {}
        original_calibrate = worker.FrozenDecisionPolicy.fit_calibration

        def factory(config, seed):
            runtime_modules(sources)
            if config != job['config'] or seed != job['seed']:
                raise ValueError('Unexpected candidate in process-local recovery factory')
            if policy == 'joint_strict_exact_v1':
                adapter = make_joint_ae_recovery_adapter(config, seed, controller='strict', use_mds_retry=False)
            else:
                params = dict(config['params'], backbone=config['backbone'], arm_id=config['arm_id'], random_state=seed)
                adapter = BMRecoveryAdapter(**params, use_mds_retry=True)
            runtime_modules(sources)
            return adapter

        def checked_calibrate(self, adapter, X, y, A, *args, **kwargs):
            runtime_modules(sources)  # Includes modules introduced by cached-model unpickling.
            calibration['admission'] = _admit_adapter(adapter, policy)
            fitted = original_calibrate(self, adapter, X, y, A, *args, **kwargs)
            calibration.update(X=X, A=A, prediction=fitted.predict(X, A))
            return fitted

        try:
            with patch.object(worker, 'make_adapter', factory), patch.object(worker.FrozenDecisionPolicy, 'fit_calibration', checked_calibrate):
                result = worker.execute_job(job_path)
            receipt['worker_result_status'] = result['status']
            if result['status'] == 'VALID':
                if result.get('reload_verified') is not True or 'prediction' not in calibration:
                    raise ValueError('Worker VALID lacks model admission or reload evidence')
                artifact = joblib.load(out / 'model.joblib')
                runtime_modules(sources)
                _admit_adapter(artifact['policy']._adapter, policy)
                after = artifact['policy'].predict(calibration['X'], calibration['A'])
                expected = calibration['prediction']
                if (not np.array_equal(expected.q_decision, after.q_decision)
                        or not np.array_equal(expected.p_event, after.p_event)):
                    raise ValueError('Persisted recovery policy differs on C')
                receipt.update(reload_exact_C=True, model_admission=calibration['admission'])
            validate_runtime_sources(job)
            receipt['loaded_project_modules_after'] = runtime_modules(sources)
            if (file_sha(job_path) != job_hash or file_sha(job['registration_path']) != job['registration_sha256']
                    or file_sha(job['data_path']) != job['data_sha256']
                    or file_sha(job['original_job_path']) != job['original_job_sha256']
                    or file_sha(Path(job['original_job_path']).parent / 'result.json') != job['original_result_sha256']):
                raise ValueError('Recovery inputs changed during execution')
            if key is not None:
                evidence = _cache_evidence(job, key)
                if reused:
                    verify_cache_entry(job, key)
                elif evidence['files']:
                    _write_once(cache / (key + '.runtime.json'), evidence)
                receipt['representation_cache_evidence'] = _cache_evidence(job, key)
            receipt.update(status=result['status'], runtime_checks_passed=True)
        except Exception as exc:
            receipt.update(status='RUNTIME_FAILED', error_type=type(exc).__name__)
        finally:
            names = ('job.json', 'runtime_started.json', 'result.json', 'model.joblib', 'predictions_S.npz')
            receipt['files'] = {name: file_sha(out / name) for name in names if (out / name).is_file()}
            if receipt['status'] == 'VALID' and not {'result.json', 'model.joblib', 'predictions_S.npz'}.issubset(receipt['files']):
                receipt.update(status='RUNTIME_FAILED', runtime_checks_passed=False, error_type='MissingOutput')
            _write_once(out / 'runtime_receipt.json', receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['worker'])
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args()
    receipt = execute_recovery_job(args.job)
    print(json.dumps({k: receipt[k] for k in ('status', 'recovery_policy', 'runtime_source_identity', 'candidate_id', 'seed')}))
    return 0 if receipt['status'] == 'VALID' and receipt['runtime_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
