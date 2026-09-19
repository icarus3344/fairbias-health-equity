"""Read-only, metric-blind provenance catalog for explicitly declared runs.

This module never imports estimators, deserializes models, selects candidates,
or authorizes evaluation. Roles, versions, expected jobs and path relocation
are supplied before scanning, rather than inferred from successful outcomes.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


MANIFEST_SCHEMA = 'nhis_result_catalog_manifest_v1'
CATALOG_SCHEMA = 'nhis_result_catalog_v1'
SCHEDULER = 'parallel_scheduler_v1_20260916'
PROFILES = {'serial_v1', 'parallel_v1', 'frappe_v1', 'numerical_recovery_v1', 'fairbias_recovery_v1'}
ROLES = {'original', 'replacement', 'sensitivity', 'pending'}
TERMINAL = {'VALID', 'FAILED', 'BUDGET_EXHAUSTED', 'NOT_SUPPORTED', 'TIME_LIMIT',
            'MEMORY_LIMIT', 'TOTAL_MEMORY_LIMIT', 'WORKER_FAILED', 'SCHEDULER_INTERRUPTED',
            'SCHEDULER_RUNTIME_FAILED'}
NUMERICAL_VARIANT = 'eg_lfr_numerical_recovery_v1_20260917'
FRAPPE_VARIANT = 'frappe_pipeline_deterministic_v1_20260917'
FAIRBIAS_VARIANT = 'fairbias_recovery_v1_20260917'
EXTERNAL_STOPS = {'TIME_LIMIT', 'MEMORY_LIMIT', 'TOTAL_MEMORY_LIMIT', 'SCHEDULER_INTERRUPTED'}
NUMERICAL_ENVIRONMENT = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
_BASE_REQUIRED = {'scripts/run_nhis_benchmark.py',
    'src/nhis_fairbias/benchmark/experiment_worker.py',
    'src/nhis_fairbias/benchmark/experiment_registry.py',
    'src/nhis_fairbias/benchmark/predictions.py'}
RUNTIME_REQUIRED = {
    'frappe_v1': {'scripts/run_nhis_runtime_worker.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_frappe_pipeline.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_frappe.py'},
    'numerical_recovery_v1': _BASE_REQUIRED | {'scripts/run_nhis_numerical_recovery_worker.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_lfr.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_lfr_recovery.py',
        'src/nhis_fairbias/benchmark/lfr_analytic_objective.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_reductions.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_reductions_numerical.py'},
    'fairbias_recovery_v1': _BASE_REQUIRED | {'scripts/run_nhis_fairbias_recovery_worker.py',
        'scripts/run_nhis_numerical_recovery_worker.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_fairbias_ae.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_bm_recovery.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_joint_ae_recovery.py',
        'src/nhis_fairbias/benchmark/adapters/geometry_audit.py',
        'src/nhis_fairbias/benchmark/joint_budget_optimization.py',
        'src/nhis_fairbias/benchmark/joint_mds_numpy.py',
        'src/nhis_fairbias/benchmark/mds_budget_retry.py'},
}
FAIRBIAS_POLICIES = {
    'joint_strict_exact_v1': {'method': 'FAIRBIAS_JOINT', 'controller': 'strict',
        'exact_kernel': 'joint_numpy_mds_v1', 'use_mds_retry': False, 'ae_cap_retry': None,
        'registered_budgets_unchanged': True, 'original_statuses': ['TIME_LIMIT'], 'training_weighted_supported': False},
    'bm_mds_retry_v1': {'method': 'FAIRBIAS_BM', 'exact_kernel': 'joint_numpy_mds_v1',
        'use_mds_retry': True, 'mds_retry': 'mds_budget_retry_v1',
        'retry_policy': 'one_fresh_seeded_refit_at_double_iteration_cap', 'max_factor': 2,
        'registered_outer_budgets_unchanged': True, 'original_statuses': ['BUDGET_EXHAUSTED'],
        'representation_weighted': False, 'training_weighted_supported': True},
}
RESULT_FIELDS = {'candidate_id', 'seed', 'status', 'source_identity', 'data_identity',
                 'reload_verified', 'output_type', 'error_type', 'termination_reason'}
REGISTRATION_FIELDS = {'version', 'source_files', 'source_identity', 'prepared', 'candidates', 'resources'}
LABEL = re.compile(r'^[A-Za-z0-9_.-]+$')
SHA = re.compile(r'^[0-9a-f]{64}$')
Q_METHODS = {'EG_DP', 'EG_EO', 'TO_EO', 'OXONFAIR_EO'}
P_METHODS = {'UNMITIGATED', 'FAIRBIAS_BM', 'REWEIGHING', 'LFR_RECONSTRUCTED',
             'FAIRGBM_EO', 'FAIRRET_EO', 'FRAPPE_EO', 'FAIRBIAS_BM_AE', 'FAIRBIAS_JOINT'}


class CatalogError(ValueError):
    """A bounded error code, without data-bearing exception text."""


def _require(condition, code):
    if not condition:
        raise CatalogError(code)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


class _JSONStream:
    """Validate/skip unselected JSON values without decoding their contents."""

    def __init__(self, handle):
        self.handle, self.buffer, self.pos = handle, '', 0

    def peek(self):
        if self.pos == len(self.buffer):
            self.buffer, self.pos = self.handle.read(65536), 0
        return self.buffer[self.pos:self.pos + 1]

    def take(self):
        value = self.peek()
        _require(bool(value), 'TRUNCATED_JSON')
        self.pos += 1
        return value

    def space(self):
        while self.peek() and self.peek() in ' \t\r\n':
            self.take()

    def value(self, keep=False, depth=0):
        _require(depth < 100, 'JSON_NESTING_LIMIT')
        self.space()
        saved = [] if keep else None
        def emit(char):
            if saved is not None:
                saved.append(char)
        token = self.peek()
        if token == '"':
            emit(self.take())
            while True:
                char = self.take()
                emit(char)
                if char == '"':
                    break
                _require(ord(char) >= 32, 'INVALID_JSON_STRING')
                if char == '\\':
                    escaped = self.take()
                    emit(escaped)
                    _require(escaped in '"\\/bfnrtu', 'INVALID_JSON_ESCAPE')
                    if escaped == 'u':
                        for _ in range(4):
                            digit = self.take()
                            _require(digit in '0123456789abcdefABCDEF', 'INVALID_JSON_ESCAPE')
                            emit(digit)
        elif token in ('{', '['):
            opening = self.take()
            closing = '}' if opening == '{' else ']'
            emit(opening)
            self.space()
            if self.peek() != closing:
                while True:
                    if opening == '{':
                        _require(self.peek() == '"', 'INVALID_JSON_OBJECT')
                        key = self.value(keep, depth + 1)
                        if keep:
                            saved.append(key)
                        self.space()
                        _require(self.take() == ':', 'INVALID_JSON_OBJECT')
                        emit(':')
                    child = self.value(keep, depth + 1)
                    if keep:
                        saved.append(child)
                    self.space()
                    if self.peek() == closing:
                        break
                    _require(self.take() == ',', 'INVALID_JSON_COLLECTION')
                    emit(',')
                    self.space()
            emit(self.take())
        else:
            chars = []
            while self.peek() and self.peek() not in ',]} \t\r\n':
                chars.append(self.take())
                _require(len(chars) <= 1024, 'INVALID_JSON_SCALAR')
            text = ''.join(chars)
            _require(text in ('true', 'false', 'null') or re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?', text),
                     'INVALID_JSON_SCALAR')
            emit(text)
        return ''.join(saved) if keep else None

    def object(self, fields=None):
        self.space()
        _require(self.take() == '{', 'JSON_OBJECT_REQUIRED')
        result, seen = {}, set()
        self.space()
        if self.peek() != '}':
            while True:
                _require(self.peek() == '"', 'INVALID_JSON_OBJECT')
                key = json.loads(self.value(True))
                _require(key not in seen, 'DUPLICATE_JSON_KEY')
                seen.add(key)
                self.space()
                _require(self.take() == ':', 'INVALID_JSON_OBJECT')
                keep = fields is None or key in fields
                raw = self.value(keep)
                if keep:
                    result[key] = json.loads(raw, object_pairs_hook=_unique_object,
                                             parse_constant=lambda _: (_ for _ in ()).throw(CatalogError('NONFINITE_JSON')))
                self.space()
                if self.peek() == '}':
                    break
                _require(self.take() == ',', 'INVALID_JSON_OBJECT')
                self.space()
        self.take()
        self.space()
        _require(not self.peek(), 'TRAILING_JSON_CONTENT')
        return result


def read_metadata(path, fields=None):
    """Only explicitly selected top-level values are decoded."""
    # Lazy import keeps the unchanged streaming reader available as a bounded
    # fallback without introducing a circular module initialization.
    from .catalog_json_reader import read_metadata as bounded_read_metadata
    return bounded_read_metadata(path, fields)


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class _Files:
    def __init__(self, roots, relocations):
        _require(isinstance(roots, dict) and roots, 'MISSING_ROOTS')
        self.roots, self.hashes = {}, {}
        for name, value in roots.items():
            _require(isinstance(name, str) and LABEL.fullmatch(name), 'INVALID_ROOT_ID')
            path = Path(value)
            _require(path.is_absolute() and path.is_dir(), 'INVALID_ROOT_DIRECTORY')
            resolved = path.resolve()
            _require(resolved.parent != resolved, 'FILESYSTEM_ROOT_FORBIDDEN')
            self.roots[name] = resolved
        self.relocations = []
        seen = set()
        for item in relocations:
            prefix = PurePosixPath(item['recorded_prefix'])
            _require(prefix.is_absolute() and '..' not in prefix.parts and str(prefix) not in seen,
                     'INVALID_OR_DUPLICATE_RELOCATION')
            seen.add(str(prefix))
            self.relocations.append((prefix, self.ref(item['target'])))
        self.relocations.sort(key=lambda pair: len(pair[0].parts), reverse=True)

    def ref(self, reference):
        _require(isinstance(reference, dict) and set(reference) == {'root', 'path'}, 'INVALID_PATH_REFERENCE')
        _require(reference['root'] in self.roots, 'UNKNOWN_ROOT')
        relative = PurePosixPath(reference['path'])
        _require(not relative.is_absolute() and '..' not in relative.parts, 'PATH_ESCAPE')
        root = self.roots[reference['root']]
        path = (root / str(relative)).resolve()
        _require(path.is_relative_to(root), 'PATH_ESCAPE')
        return path

    def child(self, directory, relative):
        _require(isinstance(relative, str), 'INVALID_RELATIVE_PATH')
        rel = PurePosixPath(relative)
        _require(not rel.is_absolute() and '..' not in rel.parts, 'PATH_ESCAPE')
        directory = Path(directory).resolve()
        path = (directory / str(rel)).resolve()
        _require(path.is_relative_to(directory), 'PATH_ESCAPE')
        return path

    def recorded(self, value):
        _require(isinstance(value, str), 'INVALID_RECORDED_PATH')
        path = PurePosixPath(value)
        _require(path.is_absolute() and '..' not in path.parts, 'INVALID_RECORDED_PATH')
        for prefix, target in self.relocations:
            if path.is_relative_to(prefix):
                return self.child(target, str(path.relative_to(prefix)))
        raise CatalogError('UNMAPPED_RECORDED_PATH')

    def sha(self, path):
        path = Path(path)
        _require(path.is_file(), 'MISSING_FILE')
        before = path.stat()
        stamp = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if self.hashes.get(path, (None,))[0] == stamp:
            return self.hashes[path][1]
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        after = path.stat()
        _require(stamp == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'FILE_CHANGED_DURING_SCAN')
        self.hashes[path] = (stamp, digest.hexdigest())
        return digest.hexdigest()

    def verify(self, path, expected):
        _require(isinstance(expected, str) and SHA.fullmatch(expected), 'INVALID_SHA256')
        _require(self.sha(path) == expected, 'FILE_HASH_MISMATCH')

    def source_manifest(self, root, values):
        _require(isinstance(values, dict) and values, 'MISSING_SOURCE_MANIFEST')
        for name, digest in values.items():
            self.verify(self.child(root, name), digest)

    def evidence(self, directory, values, mandatory):
        _require(isinstance(values, dict) and mandatory <= values.keys(), 'MISSING_REQUIRED_HASH')
        for name, digest in values.items():
            _require(isinstance(name, str) and PurePosixPath(name).name == name and name not in {'.', '..'}, 'UNSAFE_ARTIFACT_PATH')
            self.verify(self.child(directory, name), digest)


def _execution(value):
    _require(isinstance(value, dict) and set(value) == {'fit_seconds', 'worker_rss_bytes'}, 'INVALID_EXECUTION_BUDGET')
    seconds, rss = value['fit_seconds'], value['worker_rss_bytes']
    _require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0
             and type(rss) is int and rss > 0, 'INVALID_EXECUTION_BUDGET')
    return {'fit_seconds': float(seconds), 'worker_rss_bytes': rss}


def _runtime(files, run, registration, source_root):
    profile, budget = run['profile'], run['budget_policy']
    execution = _execution(budget['execution'])
    expected_budget_names = {'serial_v1': 'registered_v1', 'parallel_v1': 'registered_v1',
        'frappe_v1': 'frappe_deterministic_v1', 'numerical_recovery_v1': 'numerical_same_budgets_v1'}
    if profile in expected_budget_names:
        _require(budget.get('name') == expected_budget_names[profile], 'UNKNOWN_BUDGET_POLICY')
        original = registration['resources']
        _require(execution == _execution({k: original[k] for k in execution}), 'REGISTERED_BUDGET_CHANGED')
    runtime = run.get('runtime')
    if profile in {'serial_v1', 'parallel_v1'}:
        _require(runtime is None, 'UNEXPECTED_RUNTIME')
        return None
    _require(isinstance(runtime, dict), 'MISSING_DECLARED_RUNTIME')
    variant = {'frappe_v1': FRAPPE_VARIANT, 'numerical_recovery_v1': NUMERICAL_VARIANT,
               'fairbias_recovery_v1': FAIRBIAS_VARIANT}[profile]
    _require(runtime.get('variant') == variant, 'UNKNOWN_RUNTIME_VARIANT')
    sources, environment = runtime.get('source_files'), runtime.get('environment')
    files.source_manifest(source_root, sources)
    _require(RUNTIME_REQUIRED[profile] <= sources.keys(), 'INCOMPLETE_RUNTIME_SOURCE_MANIFEST')
    if profile == 'frappe_v1':
        _require(set(sources) == RUNTIME_REQUIRED[profile], 'FRAPPE_RUNTIME_SOURCE_SET_MISMATCH')
    _require(isinstance(environment, dict) and environment, 'MISSING_RUNTIME_ENVIRONMENT')
    expected_environment = ({'TF_DETERMINISTIC_OPS': '1'} if profile == 'frappe_v1' else
        {**NUMERICAL_ENVIRONMENT, 'PYTHONHASHSEED': '0'} if profile == 'fairbias_recovery_v1' else NUMERICAL_ENVIRONMENT)
    _require(environment == expected_environment, 'RUNTIME_ENVIRONMENT_MISMATCH')
    payload = {'variant': variant, 'files': sources, 'environment': environment}
    if profile == 'fairbias_recovery_v1':
        policy, spec = runtime.get('recovery_policy'), runtime.get('recovery_policy_spec')
        _require(policy in {'joint_strict_exact_v1', 'bm_mds_retry_v1'} and budget.get('name') == policy,
                 'UNKNOWN_BUDGET_POLICY')
        _require(spec == FAIRBIAS_POLICIES[policy], 'FAIRBIAS_POLICY_SPEC_MISMATCH')
        if policy == 'bm_mds_retry_v1':
            _require(run['role'] == 'sensitivity', 'BM_RETRY_REQUIRES_SENSITIVITY_ROLE')
        payload.update(policy=policy, policy_spec=spec, execution_budget=execution)
    _require(runtime.get('source_identity') == _identity(payload), 'RUNTIME_IDENTITY_MISMATCH')
    return runtime


def _representation_key(config, seed, data_identity):
    method, params = config['method'], config['params']
    if method == 'FAIRBIAS_BM' or method.startswith('FAIRBIAS_GEOMETRY_'):
        defaults = {'epsilon_ratio': .5, 'max_iterations': 50, 'max_geometry_evaluations': 20000,
                    'geometry_profile': 'stress_elbow', 'phi_threshold': 100., 'algorithm_version': 'application_v1'}
        spec, family = {k: params.get(k, v) for k, v in defaults.items()}, 'FAIRBIAS_BM'
    elif method == 'LFR_RECONSTRUCTED':
        spec, family = {k: params[k] for k in ('k', 'Az', 'Ax', 'Ay', 'maxiter', 'maxfun')}, method
    else:
        return None
    return _identity({'data': data_identity, 'method': family, 'seed': seed, 'representation': spec})


def _runtime_evidence(files, run, directory, job, result, registration_path, runtime):
    for field, expected in {'runtime_variant': runtime['variant'], 'runtime_source_identity': runtime['source_identity'],
                            'runtime_source_files': runtime['source_files'], 'runtime_environment': runtime['environment']}.items():
        _require(job.get(field) == expected, 'JOB_RUNTIME_MISMATCH')
    if run['profile'] == 'frappe_v1':
        _require(job['config']['method'] == 'FRAPPE_EO', 'RUNTIME_METHOD_MISMATCH')
        return None
    original_job_path = files.recorded(job['original_job_path'])
    files.verify(original_job_path, job['original_job_sha256'])
    original_result_path = files.child(original_job_path.parent, 'result.json')
    files.verify(original_result_path, job['original_result_sha256'])
    original = read_metadata(original_job_path)
    previous = read_metadata(original_result_path, RESULT_FIELDS)
    _require(original_job_path.name == 'job.json' and original_job_path.parent.name == directory.name,
             'ORIGINAL_JOB_PATH_MISMATCH')
    _require(original_job_path.parent.parent == registration_path.parent / 'jobs', 'ORIGINAL_JOB_PATH_MISMATCH')
    for field in ('config', 'seed', 'source_identity', 'data_identity', 'data_sha256'):
        _require(job.get(field) == original.get(field), 'ORIGINAL_JOB_IDENTITY_MISMATCH')
    _require(previous.get('candidate_id') == result['candidate_id'] and previous.get('seed') == job['seed'],
             'ORIGINAL_RESULT_IDENTITY_MISMATCH')
    _require(files.recorded(original['data_path']) == files.recorded(job['data_path']), 'ORIGINAL_PREPARED_PATH_MISMATCH')
    _require(files.recorded(job['registration_path']) == registration_path
             and job.get('registration_sha256') == files.sha(registration_path), 'RUNTIME_REGISTRATION_MISMATCH')
    runtime_run = files.recorded(job['recovery_run_path'])
    _require(directory.parent.parent == runtime_run, 'RUNTIME_JOB_PATH_MISMATCH')
    rid = runtime['source_identity']
    _require(runtime_run.name == rid or runtime_run.name.endswith('_' + rid), 'RUNTIME_NAMESPACE_IDENTITY_MISMATCH')
    _require(files.recorded(job['cache_path']) == runtime_run / 'representation_cache', 'RUNTIME_CACHE_PATH_MISMATCH')
    provenance = {'original_job_path': str(original_job_path), 'original_result_path': str(original_result_path),
                  'original_status': previous['status'], 'runtime_verified': False}
    rpath = files.child(directory, 'runtime_receipt.json')
    if not rpath.exists():
        _require(result['status'] in EXTERNAL_STOPS, 'MISSING_RUNTIME_RECEIPT')
        return provenance
    receipt = read_metadata(rpath)
    schema = 'nhis_fairbias_recovery_receipt_v1' if run['profile'] == 'fairbias_recovery_v1' else 'nhis_numerical_recovery_receipt_v1'
    _require(receipt.get('schema_version') == schema, 'RUNTIME_RECEIPT_SCHEMA_MISMATCH')
    required = {'runtime_variant': runtime['variant'], 'runtime_source_identity': runtime['source_identity'],
        'registration_sha256': job['registration_sha256'], 'original_job_sha256': job['original_job_sha256'],
        'original_result_sha256': job['original_result_sha256'], 'candidate_id': result['candidate_id'],
        'seed': job['seed'], 'status': result['status']}
    _require(all(receipt.get(k) == v for k, v in required.items()), 'RUNTIME_RECEIPT_IDENTITY_MISMATCH')
    failed = result['status'] == 'SCHEDULER_RUNTIME_FAILED'
    mandatory = {'job.json'} if failed else {'job.json', 'runtime_started.json', 'result.json'}
    if result['status'] == 'VALID':
        mandatory |= {'model.joblib', 'predictions_S.npz'}
    files.evidence(directory, receipt.get('files'), mandatory)
    if failed:
        _require(receipt.get('runtime_checks_passed') is False and result.get('status') != 'VALID', 'FAILED_RUNTIME_PROMOTED')
    else:
        expected = {'runtime_checks_passed': True, 'worker_result_status': result['status'],
            'runtime_source_files': runtime['source_files'], 'runtime_environment': runtime['environment'],
            'data_identity': job['data_identity'], 'data_sha256': job['data_sha256'],
            'original_status': previous['status'], 'legacy_valid_results_admitted': False}
        _require(all(receipt.get(k) == v for k, v in expected.items()), 'RUNTIME_EVIDENCE_MISMATCH')
        _require(files.recorded(receipt.get('cache_path')) == files.recorded(job['cache_path']), 'RUNTIME_CACHE_PATH_MISMATCH')
        started = read_metadata(files.child(directory, 'runtime_started.json'))
        _require(started.get('runtime_variant') == runtime['variant'] and started.get('runtime_source_identity') == runtime['source_identity']
                 and started.get('job_sha256') == files.sha(files.child(directory, 'job.json'))
                 and type(started.get('pid')) is int and started['pid'] > 0, 'RUNTIME_START_IDENTITY_MISMATCH')
        if run['profile'] == 'numerical_recovery_v1':
            method = job['config']['method']
            _require(method in {'LFR_RECONSTRUCTED', 'EG_DP', 'EG_EO'}, 'RUNTIME_METHOD_MISMATCH')
            _require(not method.startswith('EG_') or previous['status'] == 'FAILED', 'LEGACY_VALID_REUSE_NOT_ADMITTED')
        for group in ('loaded_project_modules_before', 'loaded_project_modules_after'):
            modules = receipt.get(group)
            _require(isinstance(modules, dict) and modules, 'MISSING_LOADED_MODULE_EVIDENCE')
            for item in modules.values():
                _require(isinstance(item, dict) and item.get('source') in runtime['source_files']
                         and item.get('sha256') == runtime['source_files'][item['source']], 'LOADED_MODULE_HASH_MISMATCH')
        key = _representation_key(job['config'], job['seed'], job['data_identity'])
        _require(receipt.get('representation_key') == key, 'REPRESENTATION_KEY_MISMATCH')
        if key is not None:
            cache = files.recorded(job['cache_path'])
            marker = {'runtime_variant': runtime['variant'], 'runtime_source_identity': runtime['source_identity'],
                      'registration_sha256': job['registration_sha256']}
            _require(read_metadata(files.child(cache, '.runtime_namespace.json')) == marker, 'CACHE_NAMESPACE_MISMATCH')
            sidecar = read_metadata(files.child(cache, key + '.runtime.json'))
            _require(sidecar == receipt.get('representation_cache_evidence') and all(sidecar.get(k) == v for k, v in marker.items())
                     and sidecar.get('representation_key') == key, 'CACHE_SIDECAR_MISMATCH')
            entries = sidecar.get('files')
            _require(isinstance(entries, dict) and '.json' in entries and set(entries) <= {'.json', '.joblib'}, 'CACHE_ARTIFACTS_MISSING')
            for suffix, digest in entries.items():
                files.verify(files.child(cache, key + suffix), digest)
            cache_status = read_metadata(files.child(cache, key + '.json'), {'status', 'key', 'model_sha256'})
            if cache_status.get('status') == 'VALID':
                _require(cache_status.get('key') == key and '.joblib' in entries
                         and cache_status.get('model_sha256') == entries['.joblib'], 'VALID_CACHE_MODEL_BINDING_MISMATCH')
            if result['status'] == 'VALID':
                _require(cache_status.get('status') == 'VALID', 'FAILED_CACHE_PROMOTED')
    if run['profile'] == 'fairbias_recovery_v1':
        expected = {'recovery_policy': runtime['recovery_policy'], 'recovery_policy_spec': runtime['recovery_policy_spec'],
                    'execution_budget': run['budget_policy']['execution']}
        _require(all(receipt.get(k) == v for k, v in expected.items()), 'FAIRBIAS_POLICY_RECEIPT_MISMATCH')
        _require(job.get('recovery_policy') == runtime['recovery_policy'] and job.get('execution_budget') == expected['execution_budget'],
                 'FAIRBIAS_POLICY_JOB_MISMATCH')
        spec = runtime['recovery_policy_spec']
        _require(job['config']['method'] == spec['method'] and previous['status'] in spec['original_statuses'],
                 'FAIRBIAS_POLICY_ORIGINAL_SCOPE_MISMATCH')
        if result['status'] == 'VALID':
            admission = receipt.get('model_admission', {})
            _require(isinstance(admission, dict) and receipt.get('reload_exact_C') is True and admission.get('status') == 'COMPLETE_FEASIBLE',
                     'INCOMPLETE_FAIRBIAS_MODEL')
            final, threshold = admission.get('final_max_dphi'), admission.get('epsilon_threshold')
            _require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (final, threshold))
                     and final <= threshold, 'INFEASIBLE_FAIRBIAS_MODEL')
            if runtime['recovery_policy'] == 'joint_strict_exact_v1':
                _require(all(admission.get(k) is True for k in ('model_available', 'final_geometry_feasible', 'search_complete', 'convergence_verified'))
                         and admission.get('geometry_search_incomplete') is False
                         and admission.get('termination_reason') == 'STRICT_FEASIBLE_SEARCH_EXHAUSTED', 'INCOMPLETE_FAIRBIAS_MODEL')
    return {**provenance, 'runtime_verified': not failed, 'runtime_receipt_sha256': files.sha(rpath)}


def _scan_job(files, run, directory, config, seed, registration, registration_path, runtime):
    job_path = files.child(directory, 'job.json')
    row = {'run_id': run['id'], 'registration_id': run['registration_id'], 'job_id': directory.name,
           'candidate_id': config['candidate_id'], 'seed': seed, 'method': config['method'],
           'backbone': config['backbone'], 'arm_id': config['arm_id'], 'training_weighted': config.get('training_weighted', False),
           'method_version': run['method_version'], 'role': run['role'], 'profile': run['profile'],
           'budget_policy': run['budget_policy'], 'directory': str(directory), 'status': 'PENDING',
           'evidence_state': 'PENDING', 'evaluation_authorized': False}
    if not job_path.exists():
        _require(run['lifecycle'] == 'active', 'MISSING_EXPECTED_JOB')
        row['pending_reason'] = 'NOT_STARTED'
        return row
    job = read_metadata(job_path)
    data = registration['prepared'][config['arm_id']]
    expected = {'config': config, 'seed': seed, 'source_identity': registration['source_identity'],
                'data_identity': data['data_identity'], 'data_sha256': data['sha256']}
    _require(type(job.get('seed')) is int and all(job.get(k) == v for k, v in expected.items()), 'JOB_REGISTRATION_IDENTITY_MISMATCH')
    data_path = files.recorded(job['data_path'])
    _require(data_path == files.recorded(data['path']), 'PREPARED_PATH_MISMATCH')
    files.verify(data_path, data['sha256'])
    row.update(job_sha256=files.sha(job_path), data_identity=job['data_identity'], data_sha256=job['data_sha256'],
               source_identity=job['source_identity'])
    if runtime is not None:
        for key, value in {'runtime_variant': runtime['variant'], 'runtime_source_identity': runtime['source_identity'],
                           'runtime_source_files': runtime['source_files'], 'runtime_environment': runtime['environment']}.items():
            _require(job.get(key) == value, 'JOB_RUNTIME_MISMATCH')
    receipt_path = files.child(directory, 'receipt.json')
    if not receipt_path.exists():
        _require(run['lifecycle'] == 'active', 'MISSING_TERMINAL_RECEIPT')
        row['pending_reason'] = 'NO_TERMINAL_RECEIPT'
        return row
    result_path = files.child(directory, 'result.json')
    result, receipt = read_metadata(result_path, RESULT_FIELDS), read_metadata(receipt_path)
    status = result.get('status')
    _require(isinstance(status, str) and status in TERMINAL, 'INVALID_TERMINAL_STATUS')
    _require(type(result.get('seed')) is int and result.get('candidate_id') == config['candidate_id'] and result.get('seed') == seed, 'RESULT_IDENTITY_MISMATCH')
    for field in ('source_identity', 'data_identity'):
        _require(field not in result or result[field] == expected[field], 'RESULT_SOURCE_DATA_MISMATCH')
    if run['profile'] != 'serial_v1':
        _require(receipt.get('scheduler_version') == SCHEDULER and job.get('parallel_scheduler_version') == SCHEDULER,
                 'SCHEDULER_SCHEMA_MISMATCH')
        _require(isinstance(receipt.get('session_id'), str) and receipt['session_id'], 'MISSING_SCHEDULER_SESSION')
    else:
        _require('scheduler_version' not in receipt and 'parallel_scheduler_version' not in job, 'SERIAL_SCHEMA_MISMATCH')
    _require(type(receipt.get('returncode')) is int and 'termination' in receipt, 'INVALID_SCHEDULER_EXIT')
    _require(receipt['termination'] is None or receipt['termination'] in EXTERNAL_STOPS, 'UNKNOWN_SCHEDULER_TERMINATION')
    mandatory = {'job.json', 'result.json', 'worker.log'}
    external_without_runtime = (run['profile'] in {'numerical_recovery_v1', 'fairbias_recovery_v1'}
        and status in EXTERNAL_STOPS and not files.child(directory, 'runtime_receipt.json').exists())
    if external_without_runtime:
        _require(receipt['termination'] == status, 'EXTERNAL_STOP_RECEIPT_MISMATCH')
    elif run['profile'] in {'numerical_recovery_v1', 'fairbias_recovery_v1'}:
        mandatory.add('runtime_receipt.json')
    if status == 'VALID':
        mandatory |= {'model.joblib', 'predictions_S.npz'}
        _require(receipt['returncode'] == 0 and receipt['termination'] is None and result.get('reload_verified') is True
                 and result.get('source_identity') == expected['source_identity']
                 and result.get('data_identity') == expected['data_identity'], 'VALID_WITHOUT_COMPLETE_EVIDENCE')
        method = config['method']
        _require(method in Q_METHODS | P_METHODS or method.startswith('FAIRBIAS_GEOMETRY_'), 'UNKNOWN_METHOD_OUTPUT_CONTRACT')
        expected_output = 'decision_probability_q' if method in Q_METHODS else 'event_probability_p'
        _require(result.get('output_type') == expected_output, 'INVALID_P_Q_OUTPUT_IDENTITY')
    files.evidence(directory, receipt.get('files'), mandatory)
    provenance = _runtime_evidence(files, run, directory, job, result, registration_path, runtime) if runtime else None
    row.update(status=status, evidence_state='VERIFIED_TERMINAL', result_sha256=files.sha(result_path),
        receipt_sha256=files.sha(receipt_path), output_type=result.get('output_type'), error_type=result.get('error_type'),
        termination_reason=result.get('termination_reason'), scheduler_termination=receipt['termination'],
        runtime_source_identity=None if runtime is None else runtime['source_identity'], original_evidence=provenance)
    if external_without_runtime:
        row['evidence_state'] = 'VERIFIED_EXTERNAL_STOP'
    if status == 'VALID':
        row['artifacts'] = {name: {'path': str(files.child(directory, name)), 'sha256': receipt['files'][name]}
                            for name in ('model.joblib', 'predictions_S.npz')}
    return row


def _build_catalog(manifest_path):
    """Scan explicit evidence once; rejected attempts remain issues, never VALID.

    Manifest/global-contract errors raise CatalogError. Per-job evidence failures
    are retained as REJECTED rows, with ``integrity_passed=False`` on the catalog.
    Unattempted registration cells appear in coverage; replacements do not erase
    the original run or merge seeds into a supposedly complete method version.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = read_metadata(manifest_path)
    _require(manifest.get('schema_version') == MANIFEST_SCHEMA, 'UNKNOWN_MANIFEST_SCHEMA')
    _require(manifest.get('evaluation_authorized') is False, 'EVALUATION_MUST_REMAIN_DISABLED')
    files = _Files(manifest.get('roots'), manifest.get('relocations', []))
    registrations, registered, registration_hashes = {}, {}, set()
    for item in manifest.get('registrations', []):
        rid = item['id']
        _require(isinstance(rid, str) and LABEL.fullmatch(rid) and rid not in registrations, 'DUPLICATE_OR_INVALID_REGISTRATION')
        path, source_root = files.ref(item['path']), files.ref(item['source_root'])
        _require(item['sha256'] not in registration_hashes, 'DUPLICATE_REGISTRATION_CONTENT')
        registration_hashes.add(item['sha256'])
        files.verify(path, item['sha256'])
        registration = read_metadata(path, REGISTRATION_FIELDS)
        _require(registration.get('version') == 'codex_application_v1_20260916', 'UNKNOWN_REGISTRATION_SCHEMA')
        _require(registration.get('source_identity') == _identity(registration.get('source_files')), 'REGISTRATION_SOURCE_IDENTITY_MISMATCH')
        files.source_manifest(source_root, registration['source_files'])
        jobs = {}
        for config in registration.get('candidates', []):
            if config.get('status') != 'REGISTERED':
                continue
            candidate = config.get('candidate_id')
            _require(isinstance(candidate, str) and LABEL.fullmatch(candidate), 'INVALID_CANDIDATE_ID')
            _require(isinstance(config.get('seeds'), list) and config['seeds'], 'INVALID_REGISTERED_SEEDS')
            for seed in config['seeds']:
                _require(type(seed) is int and seed >= 0, 'INVALID_REGISTERED_SEED')
                name = f'{candidate}_s{seed}'
                _require(name not in jobs, 'DUPLICATE_REGISTERED_JOB')
                jobs[name] = (config, seed)
        _require(jobs, 'EMPTY_REGISTRATION')
        registrations[rid] = (registration, path, source_root)
        registered[rid] = jobs
    _require(registrations, 'MISSING_REGISTRATIONS')
    seen_ids, ownership, physical, version_contracts = set(), set(), set(), {}
    rows, issues = [], []
    for run in manifest.get('runs', []):
        run_id = run['id']
        _require(isinstance(run_id, str) and LABEL.fullmatch(run_id) and run_id not in seen_ids, 'DUPLICATE_OR_INVALID_RUN')
        seen_ids.add(run_id)
        _require(run.get('registration_id') in registrations and run.get('role') in ROLES
                 and run.get('profile') in PROFILES and run.get('lifecycle') in {'active', 'closed'}, 'INVALID_RUN_CONTRACT')
        _require(isinstance(run.get('method_version'), str) and LABEL.fullmatch(run['method_version']), 'INVALID_METHOD_VERSION')
        registration, registration_path, source_root = registrations[run['registration_id']]
        runtime = _runtime(files, run, registration, files.ref(run['source_root']) if 'source_root' in run else source_root)
        directory = files.ref(run['directory'])
        jobs = run.get('expected_jobs')
        _require(isinstance(jobs, list) and jobs and all(isinstance(name, str) for name in jobs), 'MISSING_EXPLICIT_JOB_OWNERSHIP')
        for name in jobs:
            _require(name in registered[run['registration_id']], 'UNKNOWN_EXPECTED_JOB')
            key = (run['registration_id'], run['method_version'], name)
            job_directory = files.child(directory, 'jobs/' + name)
            _require(key not in ownership and job_directory not in physical, 'DUPLICATE_JOB_OWNERSHIP')
            ownership.add(key)
            physical.add(job_directory)
            config, seed = registered[run['registration_id']][name]
            version_key = (run['registration_id'], config['method'], run['method_version'])
            contract = {'role': run['role'], 'budget_policy': run['budget_policy'], 'runtime': runtime}
            _require(version_key not in version_contracts or version_contracts[version_key] == contract,
                     'CONFLICTING_METHOD_VERSION_CONTRACT')
            version_contracts[version_key] = contract
            try:
                row = _scan_job(files, run, job_directory, config, seed, registration, registration_path, runtime)
            except (CatalogError, KeyError, TypeError, ValueError, OSError) as exc:
                code = str(exc) if isinstance(exc, CatalogError) else 'MALFORMED_REQUIRED_EVIDENCE'
                issues.append({'run_id': run_id, 'job_id': name, 'code': code})
                row = {'run_id': run_id, 'registration_id': run['registration_id'], 'job_id': name,
                    'candidate_id': config['candidate_id'], 'seed': seed, 'method': config['method'],
                    'method_version': run['method_version'], 'role': run['role'], 'directory': str(job_directory),
                    'status': 'REJECTED', 'evidence_state': 'REJECTED', 'issue_code': code, 'evaluation_authorized': False}
            rows.append(row)
    _require(rows, 'EMPTY_RUN_DECLARATIONS')
    counts = Counter((r['method'], r['method_version'], r['role'], r['status']) for r in rows)
    summary = [{'method': method, 'method_version': version, 'role': role, 'status': status, 'jobs': count}
               for (method, version, role, status), count in sorted(counts.items())]
    covered = {(r['registration_id'], r['job_id']) for r in rows}
    coverage = []
    for rid, jobs in registered.items():
        missing = sorted(name for name in jobs if (rid, name) not in covered)
        original_names = {r['job_id'] for r in rows if r['registration_id'] == rid and r['role'] == 'original'}
        coverage.append({'registration_id': rid, 'registered_jobs': len(jobs),
                         'declared_jobs': len(jobs) - len(missing), 'undeclared_jobs': missing,
                         'original_declared_jobs': len(original_names),
                         'original_undeclared_jobs': sorted(set(jobs) - original_names),
                         'note': 'Coverage counts declared sources only; versions and failed outcomes remain separate.'})
    return {'schema_version': CATALOG_SCHEMA, 'evaluation_authorized': False,
        'selection_performed': False, 'metrics_decoded': False, 'models_deserialized': False,
        'manifest_sha256': files.sha(manifest_path), 'integrity_passed': not issues,
        'jobs': rows, 'summary': summary, 'issues': issues, 'registration_coverage': coverage,
        'unique_files_hashed': len(files.hashes)}


def build_catalog(manifest_path):
    """Return a metric-free inventory or reject a malformed global contract."""
    try:
        return _build_catalog(manifest_path)
    except CatalogError:
        raise
    except (OSError, ValueError, TypeError, KeyError):
        raise CatalogError('MALFORMED_MANIFEST') from None
