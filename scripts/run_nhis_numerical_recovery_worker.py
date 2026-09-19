"""Hash-bound, isolated EG/LFR recovery over the unchanged experiment worker.

The supervisor supplies immutable original registration/job/result hashes and
an independently identified recovery run. This worker neither edits ALL70 nor
admits legacy VALID results. The CLI runs one job in one process; the callable
entry point exists for generated-data verification and is not thread-safe.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
VARIANT = 'eg_lfr_numerical_recovery_v1_20260917'
THREAD_ENVIRONMENT = {name: '1' for name in (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
)}
WORKER_SOURCE = 'scripts/run_nhis_numerical_recovery_worker.py'
REQUIRED_SOURCES = {
    WORKER_SOURCE,
    'scripts/run_nhis_benchmark.py',
    'src/nhis_fairbias/benchmark/experiment_worker.py',
    'src/nhis_fairbias/benchmark/experiment_registry.py',
    'src/nhis_fairbias/benchmark/predictions.py',
    'src/nhis_fairbias/benchmark/adapters/adapter_lfr.py',
    'src/nhis_fairbias/benchmark/adapters/adapter_lfr_recovery.py',
    'src/nhis_fairbias/benchmark/lfr_analytic_objective.py',
    'src/nhis_fairbias/benchmark/adapters/adapter_reductions.py',
    'src/nhis_fairbias/benchmark/adapters/adapter_reductions_numerical.py',
}
CACHE_NAMESPACE_FILE = '.runtime_namespace.json'
ALLOWED_METHODS = {'EG_DP', 'EG_EO', 'LFR_RECONSTRUCTED'}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build_runtime_source_manifest(root=ROOT):
    """Supervisor-facing helper: snapshot all project Python and this worker."""
    root = Path(root).resolve()
    paths = [root / WORKER_SOURCE, root / 'scripts/run_nhis_benchmark.py']
    for package in ('fairbias', 'nhis_fairbias'):
        paths.extend(sorted((root / 'src' / package).rglob('*.py')))
    if len(paths) == 2:
        raise ValueError('Missing project source packages')
    paths.extend(sorted((root / 'configs/nhis').glob('*.json')))
    if any(not p.resolve().is_relative_to(root) for p in paths):
        raise ValueError('Runtime source escapes checkout')
    return {str(p.relative_to(root)): file_sha(p) for p in paths}


def runtime_identity(files, environment=None):
    return identity({'variant': VARIANT, 'files': files,
                     'environment': THREAD_ENVIRONMENT if environment is None else environment})


def _read_json(path):
    return json.loads(Path(path).read_text())


def _write_once(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def validate_runtime_sources(job):
    if job.get('runtime_variant') != VARIANT:
        raise ValueError('Unknown numerical recovery runtime variant')
    if job.get('runtime_environment') != THREAD_ENVIRONMENT:
        raise ValueError('Missing declared numerical recovery thread environment')
    if any(os.environ.get(name) != value for name, value in THREAD_ENVIRONMENT.items()):
        raise ValueError('Numerical recovery requires single-thread startup environment')
    files = job.get('runtime_source_files')
    if not isinstance(files, dict) or not REQUIRED_SOURCES.issubset(files):
        raise ValueError('Runtime source manifest is incomplete')
    # This is a frozen snapshot, not a live directory-set equality check.
    # Unrelated, unimported additions may coexist with running recovery jobs.
    # Every listed file must remain unchanged; any imported unlisted project
    # module is independently refused by runtime_modules before/after fitting.
    for relative, expected in files.items():
        path = (ROOT / relative).resolve()
        if (Path(relative).is_absolute() or not path.is_relative_to(ROOT)
                or not path.is_file() or file_sha(path) != expected):
            raise ValueError('Runtime source manifest/hash mismatch: ' + relative)
    if job.get('runtime_source_identity') != runtime_identity(files, job['runtime_environment']):
        raise ValueError('Runtime source identity mismatch')


def runtime_modules(expected):
    observed = {}
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in {'fairbias', 'nhis_fairbias'}:
            continue
        origin = getattr(module, '__file__', None)
        if origin is None:
            raise ValueError('Project module has no verifiable source: ' + name)
        path = Path(origin).resolve()
        if not path.is_relative_to(ROOT / 'src'):
            raise ValueError('Project module loaded outside source tree: ' + name)
        relative = str(path.relative_to(ROOT))
        if relative not in expected or file_sha(path) != expected[relative]:
            raise ValueError('Loaded project module source mismatch: ' + name)
        observed[name] = {'source': relative, 'sha256': expected[relative]}
    return observed


def validate_job(job_path):
    job_path = Path(job_path).resolve()
    job = _read_json(job_path)
    validate_runtime_sources(job)
    runtime_id = job['runtime_source_identity']
    run = Path(job['recovery_run_path']).resolve()
    if run.name != runtime_id and not run.name.endswith('_' + runtime_id):
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
    original_job_path = Path(job['original_job_path']).resolve()
    if (original_job_path.name != 'job.json'
            or original_job_path.parent.parent != original_run / 'jobs'
            or job_path.parent.name != original_job_path.parent.name):
        raise ValueError('Original/recovery job paths disagree')
    if file_sha(original_job_path) != job['original_job_sha256']:
        raise ValueError('Original job hash mismatch')
    original_job = _read_json(original_job_path)
    config = job['config']
    if config.get('method') not in ALLOWED_METHODS or config.get('training_weighted', False):
        raise ValueError('Recovery accepts only unweighted registered EG/LFR jobs')
    candidates = [c for c in registration['candidates'] if c.get('candidate_id') == config.get('candidate_id')]
    if (len(candidates) != 1 or candidates[0] != config or config.get('status') != 'REGISTERED'
            or job['seed'] not in config['seeds']):
        raise ValueError('Recovery candidate/seed differs from original registration')
    for key in ('config', 'seed', 'source_identity', 'data_identity', 'data_sha256'):
        if job.get(key) != original_job.get(key):
            raise ValueError('Recovery changed original job identity: ' + key)
    if job['source_identity'] != registration['source_identity']:
        raise ValueError('Registered source identity mismatch')
    sources = registration.get('source_files')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Missing supervisor-registered source manifest')
    if not set(sources).issubset(job['runtime_source_files']):
        raise ValueError('Runtime snapshot omits supervisor-registered source files')
    for relative, expected in sources.items():
        path = (ROOT / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(ROOT) or file_sha(path) != expected:
            raise ValueError('Supervisor-registered source hash mismatch: ' + relative)
    prepared = registration['prepared'][config['arm_id']]
    if (prepared['sha256'] != job['data_sha256'] or prepared['data_identity'] != job['data_identity']
            or file_sha(job['data_path']) != job['data_sha256']):
        raise ValueError('Registered prepared data hash/identity mismatch')
    original_result_path = original_job_path.parent / 'result.json'
    if file_sha(original_result_path) != job['original_result_sha256']:
        raise ValueError('Original result hash mismatch')
    original_result = _read_json(original_result_path)
    if (original_result.get('candidate_id') != config['candidate_id']
            or original_result.get('seed') != job['seed']):
        raise ValueError('Original result candidate/seed mismatch')
    if config['method'].startswith('EG_') and original_result.get('status') != 'FAILED':
        raise ValueError('EG recovery admits failed jobs only; legacy VALID needs a separate gate')
    cache = Path(job['cache_path']).resolve()
    if cache != run / 'representation_cache':
        raise ValueError('Recovery cache must use the independent runtime namespace')
    original_cache = Path(original_job['cache_path']).resolve()
    if cache == original_cache or original_cache in cache.parents or cache in original_cache.parents:
        raise ValueError('Original representation cache cannot be reused')
    existing = {'result.json', 'model.joblib', 'predictions_S.npz', 'receipt.json',
                'runtime_receipt.json', 'runtime_started.json'}
    if any((job_path.parent / name).exists() for name in existing):
        raise ValueError('Recovery attempt already has output; use a fresh attempt')
    return job, run, original_result


@contextmanager
def _lock(path):
    with Path(path).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _cache_namespace(job):
    return {'runtime_variant': VARIANT, 'runtime_source_identity': job['runtime_source_identity'],
            'registration_sha256': job['registration_sha256']}


def initialize_cache(job):
    cache = Path(job['cache_path']).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with _lock(cache / '.namespace.lock'):
        marker = cache / CACHE_NAMESPACE_FILE
        expected = _cache_namespace(job)
        if marker.exists():
            if _read_json(marker) != expected:
                raise ValueError('Recovery cache namespace mismatch')
        else:
            if set(p.name for p in cache.iterdir()) != {'.namespace.lock'}:
                raise ValueError('Recovery refuses cache contents without runtime provenance')
            _write_once(marker, expected)


def _cache_evidence(job, key):
    cache = Path(job['cache_path'])
    files = {suffix: file_sha(cache / (key + suffix)) for suffix in ('.json', '.joblib')
             if (cache / (key + suffix)).is_file()}
    return {**_cache_namespace(job), 'representation_key': key, 'files': files}


def verify_cache_entry(job, key):
    if key is None:
        return False
    cache = Path(job['cache_path'])
    evidence = _cache_evidence(job, key)
    sidecar = cache / (key + '.runtime.json')
    if sidecar.exists():
        if not evidence['files'] or _read_json(sidecar) != evidence:
            raise ValueError('Recovery representation cache integrity mismatch')
        if '.json' not in evidence['files']:
            raise ValueError('Recovery cache lacks completion status')
        return True
    if evidence['files']:
        raise ValueError('Recovery refuses representation cache without runtime provenance')
    return False


def execute_recovery_job(job_path):
    """Run one independently registered recovery attempt, returning its receipt."""
    job_path = Path(job_path).resolve()
    job, run, previous = validate_job(job_path)
    sys.path.insert(0, str(ROOT / 'src'))
    from nhis_fairbias.benchmark import experiment_worker as worker
    from nhis_fairbias.benchmark.adapters.adapter_lfr_recovery import LFRAnalyticRecoveryAdapter
    from nhis_fairbias.benchmark.adapters.adapter_reductions_numerical import NumericallyRecoveredExponentiatedGradientAdapter

    sources = job['runtime_source_files']
    before = runtime_modules(sources)
    initialize_cache(job)
    key = worker.representation_key(job['config'], job['seed'], job['data_identity'])
    cache = Path(job['cache_path'])
    job_hash = file_sha(job_path)
    registration_hash = job['registration_sha256']
    lock_path = cache / ((key if key is not None else job_path.parent.name) + '.lock')
    with _lock(lock_path):
        reused = verify_cache_entry(job, key)
        _write_once(job_path.parent / 'runtime_started.json', {
            'runtime_variant': VARIANT, 'runtime_source_identity': job['runtime_source_identity'],
            'job_sha256': job_hash, 'pid': os.getpid(),
        })
        receipt = {
            'schema_version': 'nhis_numerical_recovery_receipt_v1',
            'runtime_variant': VARIANT, 'runtime_source_identity': job['runtime_source_identity'],
            'runtime_source_files': sources, 'runtime_environment': job['runtime_environment'],
            'registration_sha256': registration_hash, 'original_job_sha256': job['original_job_sha256'],
            'original_result_sha256': job['original_result_sha256'], 'original_status': previous['status'],
            'candidate_id': job['config']['candidate_id'], 'seed': job['seed'],
            'data_sha256': job['data_sha256'], 'data_identity': job['data_identity'],
            'cache_path': str(cache), 'representation_key': key, 'representation_cache_reused': reused,
            'loaded_project_modules_before': before, 'legacy_valid_results_admitted': False,
            'status': 'RUNTIME_FAILED', 'runtime_checks_passed': False,
        }

        def make_adapter(config, seed):
            # Called by the original worker after loading prepared data. A
            # deserialized module must pass origin verification before fit.
            runtime_modules(sources)
            if config != job['config'] or seed != job['seed']:
                raise ValueError('Unexpected candidate in process-local recovery factory')
            params = dict(config['params'])
            params.update(backbone=config['backbone'], random_state=seed)
            adapter_type = (LFRAnalyticRecoveryAdapter if config['method'] == 'LFR_RECONSTRUCTED'
                            else NumericallyRecoveredExponentiatedGradientAdapter)
            adapter = adapter_type(**params)
            runtime_modules(sources)
            return adapter

        try:
            with patch.object(worker, 'make_adapter', make_adapter):
                result = worker.execute_job(job_path)
            receipt['worker_result_status'] = result['status']
            validate_runtime_sources(job)
            receipt['loaded_project_modules_after'] = runtime_modules(sources)
            if (file_sha(job_path) != job_hash
                    or file_sha(job['registration_path']) != registration_hash
                    or file_sha(job['data_path']) != job['data_sha256']
                    or file_sha(job['original_job_path']) != job['original_job_sha256']
                    or file_sha(Path(job['original_job_path']).parent / 'result.json') != job['original_result_sha256']):
                raise ValueError('Recovery inputs changed during execution')
            if key is not None:
                sidecar = cache / (key + '.runtime.json')
                if reused:
                    verify_cache_entry(job, key)
                else:
                    evidence = _cache_evidence(job, key)
                    if evidence['files']:
                        _write_once(sidecar, evidence)
                receipt['representation_cache_evidence'] = _cache_evidence(job, key)
            receipt.update(status=result['status'], runtime_checks_passed=True)
        except Exception as exc:
            # The unchanged worker retains its own diagnostic conventions;
            # this extra receipt never emits arbitrary data-bearing messages.
            receipt.update(status='RUNTIME_FAILED', error_type=type(exc).__name__)
        finally:
            names = ('job.json', 'runtime_started.json', 'result.json', 'model.joblib', 'predictions_S.npz')
            receipt['files'] = {name: file_sha(job_path.parent / name) for name in names
                                if (job_path.parent / name).is_file()}
            if receipt['status'] == 'VALID' and not {'result.json', 'model.joblib', 'predictions_S.npz'}.issubset(receipt['files']):
                receipt.update(status='RUNTIME_FAILED', runtime_checks_passed=False, error_type='MissingOutput')
            _write_once(job_path.parent / 'runtime_receipt.json', receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['worker'])
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args()
    receipt = execute_recovery_job(args.job)
    print(json.dumps({key: receipt[key] for key in ('status', 'runtime_source_identity', 'candidate_id', 'seed')}))
    return 0 if receipt['status'] == 'VALID' and receipt['runtime_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
