"""Isolated F/C-only EG/LFR repair pilot. Never selects models or evaluates S/T."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    """Keep numerical diagnostics numeric; never stringify unknown objects."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Unsupported pilot JSON value: ' + type(value).__name__)


def validate_job(job_file, out):
    job_file, out = Path(job_file).resolve(), Path(out).resolve()
    run = job_file.parents[2]
    registration = json.loads((run / 'registration.json').read_text())
    job = json.loads(job_file.read_text())
    if out.exists() or out == run or run in out.parents:
        raise ValueError('Pilot output must be fresh and outside original run')
    matches = [c for c in registration['candidates']
               if c['candidate_id'] == job['config']['candidate_id']]
    if len(matches) != 1:
        raise ValueError('Job requires one known registered candidate')
    candidate = matches[0]
    if candidate != job['config'] or job['seed'] not in candidate['seeds']:
        raise ValueError('Job differs from registered candidate/seed')
    if job['source_identity'] != registration['source_identity']:
        raise ValueError('Job/source registration identity mismatch')
    if not registration.get('source_files'):
        raise ValueError('Missing registered source files')
    for path, digest in registration['source_files'].items():
        resolved = (ROOT / path).resolve()
        if Path(path).is_absolute() or not resolved.is_relative_to(ROOT) or not resolved.is_file():
            raise ValueError('Unknown registered source path: ' + path)
        if sha(resolved) != digest:
            raise ValueError('Registered source mismatch: ' + path)
    expected = registration['prepared'][candidate['arm_id']]
    if expected['sha256'] != job['data_sha256'] or expected['data_identity'] != job['data_identity']:
        raise ValueError('Job input differs from registered input')
    expected_path = Path(expected.get('path', run / 'prepared' / (candidate['arm_id'] + '.joblib'))).resolve()
    if Path(job['data_path']).resolve() != expected_path:
        raise ValueError('Job prepared path differs from registered input')
    if sha(job['data_path']) != job['data_sha256']:
        raise ValueError('Prepared data hash mismatch')
    return job, run


def run_pilot(job_file, output, variant):
    if variant not in {'eg_roundoff_v1', 'lfr_analytic_v1'}:
        raise ValueError('Unknown numerical recovery variant')
    job, original_run = validate_job(job_file, output)
    method = job['config']['method']
    if (variant == 'eg_roundoff_v1' and method not in {'EG_DP', 'EG_EO'}
            or variant == 'lfr_analytic_v1' and method != 'LFR_RECONSTRUCTED'):
        raise ValueError('Recovery variant/method mismatch')
    if job['config'].get('training_weighted', False):
        raise ValueError('Numerical recovery pilot does not support survey-weighted training')
    threads = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    if any(os.environ.get(k) != '1' for k in threads):
        raise ValueError('Pilot requires single-thread environment before startup')
    project_hashes = {str(p.relative_to(ROOT)): sha(p)
                      for package in ('fairbias', 'nhis_fairbias')
                      for p in (ROOT / 'src' / package).rglob('*.py')}
    # Bind fresh imports to this checkout; loaded shadow modules are rejected
    # by the source-origin check below instead of silently replaced.
    for directory in (ROOT / 'src', ROOT / 'scripts'):
        sys.path.insert(0, str(directory))
    import joblib
    import numpy as np
    import pilot_scheduled_joint as runtime_helper
    helper_path = ROOT / 'scripts' / 'pilot_scheduled_joint.py'
    if Path(runtime_helper.__file__).resolve() != helper_path.resolve():
        raise ValueError('Runtime source helper loaded outside this checkout')
    helper_sha256 = sha(helper_path)
    runtime_modules = runtime_helper.runtime_modules
    from nhis_fairbias.benchmark.predictions import FrozenDecisionPolicy
    if variant == 'eg_roundoff_v1':
        from nhis_fairbias.benchmark.adapters.adapter_reductions_numerical import NumericallyRecoveredExponentiatedGradientAdapter as Adapter
    else:
        from nhis_fairbias.benchmark.adapters.adapter_lfr_recovery import LFRAnalyticRecoveryAdapter as Adapter
    modules_before_load = runtime_modules(ROOT, project_hashes)
    data = joblib.load(job['data_path'])
    modules_after_load = runtime_modules(ROOT, project_hashes)
    if (data['data_identity'] != job['data_identity']
            or 'evaluation_T' in data['partitions'] or 'X_T' in data):
        raise ValueError('Prepared identity/partition contract mismatch')
    F, C = data['partitions']['fitting_F'], data['partitions']['calibration_C']
    if F.year != 2022 or C.year != 2022 or F.role != 'fitting_F' or C.role != 'calibration_C':
        raise ValueError('Numerical pilots require development F/C 2022')
    X_F, X_C = data['X_F'], data['X_C']
    if any(p.arm_id != job['config']['arm_id'] for p in (F, C)):
        raise ValueError('F/C arm differs from registered candidate')
    if any(np.ndim(x) != 2 or len(x) != len(p) or len(p.A) != len(p)
           or len(p.record_keys) != len(p) or len(p) == 0 for x, p in ((X_F, F), (X_C, C))):
        raise ValueError('F/C feature, label, group or record-key rows do not align')
    if X_F.shape[1] != X_C.shape[1] or set(F.record_keys) & set(C.record_keys):
        raise ValueError('F/C feature dimensions or disjoint record identities disagree')
    del data  # drop the unused S object immediately; no S arrays or labels accessed
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    record = {
        'diagnostic_only': True, 'formal_benchmark_admission': False,
        'variant': variant, 'job_id': Path(job_file).parent.name,
        'job_sha256': sha(job_file), 'source_identity': job['source_identity'],
        'data_identity': job['data_identity'], 'data_sha256': job['data_sha256'],
        'registration_sha256': sha(original_run / 'registration.json'),
        'config': job['config'], 'seed': job['seed'],
        'partitions_used': ['fitting_F', 'calibration_C'],
        'rows': {'F': len(F), 'C': len(C)},
        'status': 'STARTED', 'script_sha256': sha(__file__),
        'project_source_hashes': project_hashes,
        'runtime_helper_sha256': helper_sha256,
        'loaded_project_modules_before_load': modules_before_load,
        'loaded_project_modules_before_fit': modules_after_load,
        'S_T_evaluated': False,
        'package_versions': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'fairlearn', 'aif360')},
    }
    def save():
        serialized = json.dumps(record, indent=2, allow_nan=False, default=json_default) + '\n'
        pending = out / 'result.json.part'
        pending.write_text(serialized)
        pending.replace(out / 'result.json')
    save()
    try:
        params = dict(job['config']['params'])
        params.update(backbone=job['config']['backbone'], random_state=job['seed'])
        adapter = Adapter(**params)
        adapter.fit(X_F, F.y, F.A)
        fit_seconds = time.monotonic() - start
        if variant == 'lfr_analytic_v1':
            record['optimization'] = adapter.optimization_result_
            if not adapter.converged_:
                record['status'] = 'OPTIMIZATION_INCOMPLETE'
                return record
        else:
            raw = np.asarray(adapter.model._pmf_predict(X_C))[:, 1]
            recovered = adapter.predict_decision_proba(X_C, C.A)
            interior = (raw >= 0) & (raw <= 1)
            if not np.array_equal(raw[interior], recovered[interior]):
                raise ValueError('EG recovery changed an interior probability')
            record['roundoff'] = {
                'repaired_count_C': int(np.count_nonzero(raw != recovered)),
                'max_correction_C': float(np.max(np.abs(raw - recovered))),
                'all_interior_unchanged': True,
                'mixture_weight_sum': float(np.asarray(adapter.model.weights_).sum()),
            }
        policy = FrozenDecisionPolicy().fit_calibration(adapter, X_C, C.y, C.A)
        before = policy.predict(X_C, C.A)
        model_path = out / 'policy.joblib'
        joblib.dump(policy, model_path)
        after = joblib.load(model_path).predict(X_C, C.A)
        if not np.array_equal(before.q_decision, after.q_decision):
            raise ValueError('Reloaded decisions differ')
        if before.p_event is not None and not np.array_equal(before.p_event, after.p_event):
            raise ValueError('Reloaded risks differ')
        record.update(status='FC_PILOT_PASS', fit_seconds=fit_seconds,
                      reload_exact=True, model_sha256=sha(model_path))
        return record
    except Exception as exc:
        # Arbitrary estimator messages can contain data values; retain type.
        record.update(status='FAILED', error_type=type(exc).__name__)
        return record
    finally:
        record['elapsed_seconds'] = time.monotonic() - start
        record['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform == 'linux' else 1))
        try:
            record['loaded_project_modules'] = runtime_modules(ROOT, project_hashes)
            if sha(helper_path) != helper_sha256:
                raise ValueError('Runtime source helper changed during the pilot')
        except Exception as exc:
            record.update(status='FAILED', error_type=type(exc).__name__, failure_phase='source_integrity')
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--variant', required=True, choices=['eg_roundoff_v1', 'lfr_analytic_v1'])
    args = parser.parse_args()
    result = run_pilot(args.job, args.output, args.variant)
    print(json.dumps({k: result[k] for k in ('status', 'variant', 'job_id', 'elapsed_seconds')}))
    raise SystemExit(0 if result['status'] == 'FC_PILOT_PASS' else 1)
