"""Complete registered BM_AE matrix under one explicit conditional AE budget.

Old VALID and failed jobs are refit alike. The original cap10 path is retained;
only its exact AE-cap exit permits one fresh same-seed cap40 fit. The frozen
production worker still owns F/C fitting, calibration, S output and reload.
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
WORKER_SOURCE = 'scripts/run_nhis_bmae_sensitivity_worker.py'
FROZEN_SOURCE = 'scripts/run_nhis_fairbias_recovery_worker.py'
VARIANT = 'bmae_budget_sensitivity_v1_20260917'
POLICY = 'bmae_strict_conditional_10_to_40_v1'
MANIFEST_SCHEMA = 'nhis_bmae_sensitivity_registration_v1'
RECEIPT_SCHEMA = 'nhis_bmae_sensitivity_receipt_v1'
DEFAULT_EXECUTION_BUDGET = {'fit_seconds': 7200.0, 'worker_rss_bytes': 4 * 1024**3}
POLICY_SPEC = {
    'method': 'FAIRBIAS_BM_AE', 'controller': 'strict', 'exact_kernel': 'joint_numpy_mds_v1',
    'ae_cap_retry': 40, 'first_ae_commit_cap': 10,
    'retry_policy': 'fresh_same_seed_refit_after_exact_ae_commit_cap',
    'use_mds_retry': False, 'search_order_changed': False, 'mds_iteration_budget_changed': False,
    'original_statuses': ['VALID', 'BUDGET_EXHAUSTED', 'TIME_LIMIT'],
    'training_weighted_supported': False, 'complete_registered_matrix': True, 'registered_job_count': 40,
    'registered_other_limits': {'utility': 500, 'geometry': 20000, 'bm': 50},
    'original_budget_results_preserved': True,
}

# Private helper module copies keep the frozen running worker untouched.
_spec = importlib.util.spec_from_file_location('_bmae_frozen_helpers', ROOT / FROZEN_SOURCE)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)
file_sha, identity = _core.file_sha, _core.identity
_read_json, _write_once, _lock = _core._read_json, _core._write_once, _core._lock
runtime_modules = _core.runtime_modules
THREAD_ENVIRONMENT = dict(_core.THREAD_ENVIRONMENT)
REQUIRED_SOURCES = set(_core.REQUIRED_SOURCES) | {WORKER_SOURCE, FROZEN_SOURCE}
_core._base.VARIANT = VARIANT
_core._base.WORKER_SOURCE = WORKER_SOURCE
initialize_cache, _cache_namespace = _core.initialize_cache, _core._cache_namespace


def policy_spec(policy=POLICY):
    if policy != POLICY:
        raise ValueError('Only the registered full BM_AE conditional sensitivity policy is supported')
    return copy.deepcopy(POLICY_SPEC)


def execution_budget_spec(value=None):
    value = DEFAULT_EXECUTION_BUDGET if value is None else value
    checked = _core.execution_budget_spec(value)
    if checked != DEFAULT_EXECUTION_BUDGET:
        raise ValueError('BM_AE sensitivity requires explicit 7200 second / 4GiB budget')
    return checked


def build_runtime_source_manifest(root=ROOT):
    files = _core.build_runtime_source_manifest(root)
    files[FROZEN_SOURCE] = file_sha(Path(root) / FROZEN_SOURCE)
    return files


def runtime_identity(files, environment=None, *, policy=POLICY, execution_budget=None):
    return identity({'variant': VARIANT, 'policy': policy, 'policy_spec': policy_spec(policy),
        'execution_budget': execution_budget_spec(execution_budget), 'files': files,
        'environment': THREAD_ENVIRONMENT if environment is None else environment})


def registered_jobs(registration):
    jobs = {}
    for config in registration['candidates']:
        if config.get('method') != 'FAIRBIAS_BM_AE' or config.get('status') != 'REGISTERED':
            continue
        params = config['params']
        if (config.get('training_weighted', False) or params.get('mode') != 'BM_AE'
                or any(params.get(k) != v for k, v in {'max_outer_iterations': 10,
                    'max_utility_evaluations': 500, 'max_geometry_evaluations': 20000, 'max_bm_steps': 50}.items())):
            raise ValueError('BM_AE registered scientific budgets/mode/weighting differ')
        for seed in config['seeds']:
            name = f"{config['candidate_id']}_s{seed}"
            if name in jobs:
                raise ValueError('Duplicate registered BM_AE job')
            jobs[name] = {'config': config, 'seed': seed}
    if len(jobs) != 40:
        raise ValueError('BM_AE sensitivity requires all 40 registered jobs')
    return jobs


def validate_runtime_sources(job):
    policy_spec(job.get('recovery_policy'))
    if 'execution_budget' not in job:
        raise ValueError('Missing explicitly declared execution budget')
    execution_budget_spec(job['execution_budget'])
    if job.get('runtime_variant') != VARIANT or job.get('runtime_environment') != THREAD_ENVIRONMENT:
        raise ValueError('BM_AE sensitivity runtime variant/environment mismatch')
    if any(os.environ.get(k) != v for k, v in THREAD_ENVIRONMENT.items()) or sys.flags.hash_randomization != 0:
        raise ValueError('BM_AE sensitivity requires single threads and PYTHONHASHSEED=0 at startup')
    if Path(__file__).resolve() != ROOT / WORKER_SOURCE or Path(_core.__file__).resolve() != ROOT / FROZEN_SOURCE:
        raise ValueError('BM_AE helper/source origin mismatch')
    files = job.get('runtime_source_files')
    if not isinstance(files, dict) or not REQUIRED_SOURCES.issubset(files):
        raise ValueError('BM_AE runtime source manifest is incomplete')
    for rel, digest in files.items():
        path = (ROOT / rel).resolve()
        if (Path(rel).is_absolute() or not path.is_relative_to(ROOT)
                or not path.is_file() or file_sha(path) != digest):
            raise ValueError('BM_AE runtime source/hash mismatch: ' + rel)
    if job.get('runtime_source_identity') != runtime_identity(files, job['runtime_environment'],
            policy=job['recovery_policy'], execution_budget=job['execution_budget']):
        raise ValueError('BM_AE runtime policy/budget identity mismatch')


def validate_job(job_path):
    job, run, previous = _core.validate_job(job_path)
    manifest_path = Path(job['sensitivity_manifest_path']).resolve()
    if file_sha(manifest_path) != job['sensitivity_manifest_sha256']:
        raise ValueError('BM_AE complete-matrix manifest hash mismatch')
    manifest = _read_json(manifest_path)
    registration = _read_json(job['registration_path'])
    expected = registered_jobs(registration)
    entries = manifest.get('jobs', {})
    if (manifest.get('schema_version') != MANIFEST_SCHEMA
            or manifest.get('registration_sha256') != job['registration_sha256']
            or Path(manifest.get('original_run', '')).resolve() != Path(job['registration_path']).resolve().parent
            or manifest.get('recovery_policy') != POLICY or set(entries) != set(expected)):
        raise ValueError('BM_AE sensitivity manifest is not the complete registered matrix')
    for name, value in expected.items():
        if any(entries[name].get(k) != v for k, v in value.items()):
            raise ValueError('BM_AE sensitivity manifest config/seed mismatch')
    own = entries[Path(job_path).parent.name]
    for key in ('original_job_path', 'original_job_sha256', 'original_result_sha256'):
        if own.get(key) != job.get(key):
            raise ValueError('BM_AE job differs from complete-matrix manifest')
    if job.get('representation_key') is not None or job.get('original_representation_key') is not None:
        raise ValueError('BM_AE sensitivity must not reuse a representation cache')
    return job, run, previous


_core.policy_spec = policy_spec
_core.validate_runtime_sources = validate_runtime_sources


def _admit_adapter(adapter, job):
    from nhis_fairbias.benchmark.adapters.adapter_joint_ae_recovery import JointAERecoveryAdapter, classify_recovery_fit
    if (type(adapter) is not JointAERecoveryAdapter or adapter.mode != 'BM_AE'
            or adapter.controller != 'strict' or adapter.use_mds_retry or adapter.ae_cap_retry != 40
            or not adapter.is_fitted_):
        raise ValueError('BM_AE sensitivity model/policy admission failed')
    expected_params = copy.deepcopy(job['config']['params'])
    expected_params.pop('mode')
    expected_params.update(backbone=job['config']['backbone'], random_state=job['seed'])
    if adapter._original_adapter_params != expected_params:
        raise ValueError('BM_AE fitted model changed registered parameters')
    outcome = classify_recovery_fit(adapter._delegate, controller='strict')
    if adapter.recovery_result_ != outcome or outcome['status'] != 'COMPLETE_FEASIBLE':
        raise ValueError('BM_AE sensitivity requires complete feasible search')
    attempts = adapter.fit_attempts_
    states = [(a.get('ae_commit_cap'), a.get('status')) for a in attempts]
    if states not in [[(10, 'FIT_RETURNED')], [(10, 'AE_COMMIT_CAP'), (40, 'FIT_RETURNED')]]:
        raise ValueError('BM_AE sensitivity has an invalid conditional retry trace')
    if len(attempts) == 2 and attempts[0].get('retry_eligible') is not True:
        raise ValueError('BM_AE cap40 was not enabled by the exact original cap exit')
    if (adapter._delegate.max_outer_iterations != attempts[-1]['ae_commit_cap']
            or adapter._delegate.max_utility_evaluations != 500
            or adapter._delegate.max_geometry_evaluations != 20000 or adapter._delegate.max_bm_steps != 50):
        raise ValueError('BM_AE fitted model changed declared search budgets')
    for attempt in attempts:
        if attempt.get('seed') != job['seed'] or not attempt.get('mds_fits'):
            raise ValueError('BM_AE attempt seed/geometry evidence is missing')
        if any(row.get('status') != 'CONVERGED_BEFORE_CAP'
               or not math.isfinite(row.get('stress', float('nan')))
               or row.get('iterations', 1) >= row.get('max_iter', 0)
               for row in attempt['mds_fits']):
            raise ValueError('BM_AE sensitivity has incomplete geometry')
    encoder = adapter._delegate.preprocessor_
    if (not getattr(encoder, 'fitted_', False) or not getattr(encoder, 'feature_names', ())
            or not getattr(encoder, 'transformed_feature_names_', ())):
        raise ValueError('BM_AE sensitivity refuses an empty representation')
    return outcome


def execute_recovery_job(job_path):
    job_path = Path(job_path).resolve()
    job, _, previous = validate_job(job_path)
    sys.path.insert(0, str(ROOT / 'src'))
    import joblib
    import numpy as np
    from nhis_fairbias.benchmark import experiment_worker as production
    from nhis_fairbias.benchmark.adapters.adapter_joint_ae_recovery import make_joint_ae_recovery_adapter
    sources = job['runtime_source_files']
    before = runtime_modules(sources)
    initialize_cache(job)
    out, cache, job_hash = job_path.parent, Path(job['cache_path']), file_sha(job_path)
    with _lock(cache / (out.name + '.lock')):
        _write_once(out / 'runtime_started.json', {'runtime_variant': VARIANT, 'recovery_policy': POLICY,
            'runtime_source_identity': job['runtime_source_identity'], 'job_sha256': job_hash, 'pid': os.getpid()})
        receipt = {k: job[k] for k in ('runtime_source_identity', 'runtime_source_files', 'runtime_environment',
            'registration_sha256', 'original_job_sha256', 'original_result_sha256', 'data_sha256', 'data_identity',
            'sensitivity_manifest_path', 'sensitivity_manifest_sha256')}
        receipt.update(schema_version=RECEIPT_SCHEMA, runtime_variant=VARIANT, recovery_policy=POLICY,
            recovery_policy_spec=policy_spec(), original_status=previous['status'],
            execution_budget=job['execution_budget'], execution_budget_enforcement='external_scheduler',
            candidate_id=job['config']['candidate_id'], seed=job['seed'], cache_path=str(cache),
            representation_key=None, representation_cache_reused=False, loaded_project_modules_before=before,
            legacy_valid_results_admitted=False, original_valid_job_refit=previous['status'] == 'VALID',
            status='RUNTIME_FAILED', runtime_checks_passed=False)
        calibration = {}
        original_calibrate = production.FrozenDecisionPolicy.fit_calibration

        def factory(config, seed):
            runtime_modules(sources)
            if config != job['config'] or seed != job['seed']:
                raise ValueError('Unexpected candidate in BM_AE sensitivity process')
            return make_joint_ae_recovery_adapter(config, seed, controller='strict', ae_cap_retry=40, use_mds_retry=False)

        def checked_calibrate(self, adapter, X, y, A, *args, **kwargs):
            runtime_modules(sources)
            calibration['admission'] = _admit_adapter(adapter, job)
            fitted = original_calibrate(self, adapter, X, y, A, *args, **kwargs)
            calibration.update(X=X, A=A, prediction=fitted.predict(X, A))
            return fitted

        try:
            with patch.object(production, 'make_adapter', factory), patch.object(production.FrozenDecisionPolicy, 'fit_calibration', checked_calibrate):
                result = production.execute_job(job_path)
            receipt['worker_result_status'] = result['status']
            if result['status'] == 'VALID':
                if result.get('reload_verified') is not True or 'prediction' not in calibration:
                    raise ValueError('BM_AE VALID lacks admission/reload evidence')
                artifact = joblib.load(out / 'model.joblib')
                runtime_modules(sources)
                _admit_adapter(artifact['policy']._adapter, job)
                after = artifact['policy'].predict(calibration['X'], calibration['A'])
                expected = calibration['prediction']
                if not np.array_equal(expected.q_decision, after.q_decision) or not np.array_equal(expected.p_event, after.p_event):
                    raise ValueError('Persisted BM_AE policy differs on C')
                receipt.update(reload_exact_C=True, model_admission=calibration['admission'])
            validate_runtime_sources(job)
            receipt['loaded_project_modules_after'] = runtime_modules(sources)
            paths = {job_path: job_hash, Path(job['registration_path']): job['registration_sha256'],
                Path(job['data_path']): job['data_sha256'], Path(job['original_job_path']): job['original_job_sha256'],
                Path(job['original_job_path']).parent / 'result.json': job['original_result_sha256'],
                Path(job['sensitivity_manifest_path']): job['sensitivity_manifest_sha256']}
            if any(file_sha(p) != digest for p, digest in paths.items()):
                raise ValueError('BM_AE sensitivity inputs changed during execution')
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
    print(json.dumps({k: receipt[k] for k in ('status', 'runtime_source_identity', 'candidate_id', 'seed')}))
    return 0 if receipt['status'] == 'VALID' and receipt['runtime_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
