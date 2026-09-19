"""Supervisor-admitted, cross-run S selection; never a T release or evaluator.

``validate_selection_admission(path)`` is metric blind. Only an explicit call
to ``freeze_catalog_selection(path)`` reads S metrics. It returns a JSON-ready
artifact; the caller owns writing a fresh output and obtaining independent
review. No historical files are written, models loaded, or T data accessed.

An admission uses root/path references as in result_catalog and contains::

    schema_version: nhis_catalog_selection_admission_v1
    decision: SUPERVISOR_ADMITTED
    evaluation_authorized: false
    roots: {evidence: /absolute/evidence/root}
    catalog: {path: {root: evidence, path: catalog.json}, sha256: ...}
    catalog_manifest: {path: {root: evidence, path: manifest.json}, sha256: ...}
    registration: {id: original, sha256: ...}
    methods:
      LFR_RECONSTRUCTED:
        analysis_version: lfr_analytic_v1
        versions:
          - method_version: lfr_analytic_v1
            role: replacement
            budget_policy: {name: numerical_same_budgets_v1, execution: {...}}
            runtime_source_identity: ...
    resolved_jobs: {candidate_s0: explicitly_named_run, ...}
    extension_resolutions: {extension: {status: INCLUDED, reason: ...}}

Every registered job is resolved explicitly, including terminal failures.
Ordinary methods have exactly one version contract, across all configurations
and seeds. Sensitivity/pending roles cannot enter this main-analysis interface.
Only EG_DP/EG_EO may combine versions, via the dedicated hash-bound gate below;
there is no generic compatibility allowlist or success/latest fallback.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

from . import result_catalog as catalog

ADMISSION_SCHEMA = 'nhis_catalog_selection_admission_v1'
SELECTION_SCHEMA = 'nhis_catalog_selection_v1'
EG_GATE_SCHEMA = 'nhis_eg_selection_compatibility_v1'
_VERSION_KEYS = {'method_version', 'role', 'budget_policy', 'runtime_source_identity'}
_ADMISSION_KEYS = {'schema_version', 'decision', 'evaluation_authorized', 'roots',
    'catalog', 'catalog_manifest', 'registration', 'methods', 'resolved_jobs', 'extension_resolutions'}
_SOURCE_NAMES = ('catalog_selection.py', 'result_catalog.py', 'catalog_json_reader.py',
                 'experiment_selection.py', 'experiment_registry.py')


class SelectionAdmissionError(ValueError):
    """Bounded error code; never embed metric or raw-record content."""


def _require(condition, code):
    if not condition:
        raise SelectionAdmissionError(code)


def _label(value):
    return isinstance(value, str) and catalog.LABEL.fullmatch(value)


def _read_bound(files, item):
    _require(isinstance(item, dict) and set(item) == {'path', 'sha256'}, 'INVALID_BOUND_FILE')
    path = files.ref(item['path'])
    files.verify(path, item['sha256'])
    result = catalog.read_metadata(path)
    files.verify(path, item['sha256'])
    return path, result


def _contract(row):
    return {key: row.get(key) for key in _VERSION_KEYS}


def _check_eg_gate(files, specification, method, admission, resolutions, versions):
    """Evidence is supervisor-authored, not an algorithmic compatibility claim.

    The gate must bind the exact admission inputs and exact EG job mapping.
    Its evidence files carry the independently reviewed compatibility report;
    this function verifies their hashes, not the scientific claim itself.
    """
    _require(method in {'EG_DP', 'EG_EO'}, 'CROSS_VERSION_METHOD_FORBIDDEN')
    path, gate = _read_bound(files, specification)
    expected = {
        'schema_version': EG_GATE_SCHEMA, 'decision': 'ADMITTED_EG_NUMERICAL_COMPATIBILITY',
        'catalog_sha256': admission['catalog']['sha256'],
        'registration_sha256': admission['registration']['sha256'],
        'method': method, 'output_type': 'decision_probability_q',
        'resolved_jobs': resolutions, 'versions': versions,
        'registered_budgets_unchanged': True,
    }
    _require(set(gate) == set(expected) | {'evidence'}, 'INVALID_EG_GATE_SCHEMA')
    _require(all(gate.get(key) == value for key, value in expected.items()), 'EG_GATE_BINDING_MISMATCH')
    _require(isinstance(gate['evidence'], list) and gate['evidence'], 'MISSING_EG_COMPATIBILITY_EVIDENCE')
    evidence_paths = set()
    for evidence in gate['evidence']:
        _require(isinstance(evidence, dict) and set(evidence) == {'path', 'sha256'}, 'INVALID_EG_EVIDENCE')
        evidence_path = files.ref(evidence['path'])
        _require(evidence_path != path and evidence_path not in evidence_paths, 'DUPLICATE_EG_EVIDENCE')
        evidence_paths.add(evidence_path)
        files.verify(evidence_path, evidence['sha256'])
    return {'path': str(path), 'sha256': specification['sha256'], 'evidence': gate['evidence']}


def _validate(path):
    path = Path(path).resolve()
    admission = catalog.read_metadata(path)
    _require(set(admission) == _ADMISSION_KEYS and admission.get('schema_version') == ADMISSION_SCHEMA,
             'INVALID_ADMISSION_SCHEMA')
    _require(admission.get('decision') == 'SUPERVISOR_ADMITTED', 'SUPERVISOR_ADMISSION_REQUIRED')
    _require(admission.get('evaluation_authorized') is False, 'EVALUATION_MUST_REMAIN_DISABLED')
    files = catalog._Files(admission['roots'], [])
    admission_sha = files.sha(path)
    catalog_path, snapshot = _read_bound(files, admission['catalog'])
    manifest_path, manifest = _read_bound(files, admission['catalog_manifest'])
    _require(snapshot.get('schema_version') == catalog.CATALOG_SCHEMA
             and snapshot.get('integrity_passed') is True and snapshot.get('issues') == []
             and snapshot.get('evaluation_authorized') is False
             and snapshot.get('selection_performed') is False
             and snapshot.get('metrics_decoded') is False
             and snapshot.get('models_deserialized') is False, 'CATALOG_NOT_ADMISSIBLE')
    _require(snapshot.get('manifest_sha256') == admission['catalog_manifest']['sha256'], 'CATALOG_MANIFEST_MISMATCH')
    # Reuse the deployed receipt/source/data/cache verifier. It never decodes S/T
    # metrics or models. No selection value is read until ALL checks below pass.
    rebuilt = catalog.build_catalog(manifest_path)
    _require(rebuilt == snapshot, 'CATALOG_NO_LONGER_MATCHES_EVIDENCE')
    _require(all(row['status'] in catalog.TERMINAL and row['role'] != 'pending'
                 for row in snapshot['jobs']), 'UNRESOLVED_CATALOG_ATTEMPTS')
    ref = admission['registration']
    _require(isinstance(ref, dict) and set(ref) == {'id', 'sha256'}, 'INVALID_REGISTRATION_REFERENCE')
    registrations = [r for r in manifest['registrations'] if r['id'] == ref['id']]
    _require(len(registrations) == 1 and registrations[0]['sha256'] == ref['sha256'], 'REGISTRATION_BINDING_MISMATCH')
    storage = catalog._Files(manifest['roots'], manifest.get('relocations', []))
    registration_path = storage.ref(registrations[0]['path'])
    storage.verify(registration_path, ref['sha256'])
    registration = catalog.read_metadata(registration_path)
    configs, expected_jobs, method_jobs = {}, {}, defaultdict(dict)
    for config in registration['candidates']:
        candidate = config['candidate_id']
        _require(candidate not in configs, 'DUPLICATE_CANDIDATE_ID')
        _require(config['status'] in {'REGISTERED', 'NOT_SUPPORTED'}, 'UNRESOLVED_REGISTRATION_CONFIGURATION')
        configs[candidate] = config
        if config['status'] != 'REGISTERED':
            continue
        _require(type(config.get('complexity')) in (int, float) and math.isfinite(config['complexity']), 'INVALID_COMPLEXITY')
        for seed in config['seeds']:
            name = f'{candidate}_s{seed}'
            expected_jobs[name] = (config, seed)
            method_jobs[config['method']][name] = None
    resolutions = admission['resolved_jobs']
    _require(isinstance(resolutions, dict) and set(resolutions) == set(expected_jobs)
             and all(_label(run) for run in resolutions.values()), 'INCOMPLETE_OR_INVALID_JOB_RESOLUTION')
    methods = admission['methods']
    _require(isinstance(methods, dict) and set(methods) == set(method_jobs), 'METHOD_ADMISSION_COVERAGE_MISMATCH')
    attempts = {(r['run_id'], r['job_id']): r for r in snapshot['jobs'] if r['registration_id'] == ref['id']}
    rows = {}
    for name, run_id in resolutions.items():
        _require((run_id, name) in attempts, 'UNKNOWN_RESOLVED_ATTEMPT')
        row = attempts[run_id, name]
        _require(row['role'] in {'original', 'replacement'}, 'SENSITIVITY_CANNOT_REPLACE_MAIN_BUDGET')
        if row['method'] in {'FAIRBIAS_JOINT', 'FAIRBIAS_BM_AE'}:
            _require(row['role'] == 'original' and row['profile'] in {'serial_v1', 'parallel_v1'}
                and row['budget_policy']['name'] == 'registered_v1' and row['runtime_source_identity'] is None,
                'ABLATION_RECOVERY_REQUIRES_SEPARATE_STUDY')
        _require(row['status'] in catalog.TERMINAL, 'UNRESOLVED_REGISTERED_JOB')
        rows[name] = row
        method_jobs[row['method']][name] = run_id
    gates = {}
    for method, jobs in method_jobs.items():
        policy = methods[method]
        _require(isinstance(policy, dict) and set(policy) in (
            {'analysis_version', 'versions'}, {'analysis_version', 'versions', 'compatibility_gate'}), 'INVALID_METHOD_POLICY')
        _require(_label(policy['analysis_version']), 'INVALID_ANALYSIS_VERSION')
        versions = policy['versions']
        _require(isinstance(versions, list) and versions and all(isinstance(v, dict) and set(v) == _VERSION_KEYS for v in versions),
                 'INVALID_VERSION_CONTRACT')
        actual = {catalog._identity(_contract(rows[name])) for name in jobs}
        declared = [catalog._identity(v) for v in versions]
        _require(len(set(declared)) == len(declared) and set(declared) == actual, 'VERSION_CONTRACT_MISMATCH')
        registered_execution = {k: registration['resources'][k] for k in ('fit_seconds', 'worker_rss_bytes')}
        _require(all(v['budget_policy']['execution'] == registered_execution for v in versions), 'MAIN_EXECUTION_BUDGET_CHANGED')
        if len(versions) > 1:
            _require('compatibility_gate' in policy, 'EG_COMPATIBILITY_GATE_REQUIRED')
            _require({v['budget_policy']['name'] for v in versions} <= {'registered_v1', 'numerical_same_budgets_v1'},
                     'EG_BUDGET_POLICY_MISMATCH')
            gates[method] = _check_eg_gate(files, policy['compatibility_gate'], method, admission, jobs, versions)
        else:
            _require('compatibility_gate' not in policy, 'UNNECESSARY_COMPATIBILITY_GATE')
    extensions = admission['extension_resolutions']
    _require(isinstance(extensions, dict) and set(extensions) == set(registration.get('pending_predeclared_extensions', [])),
             'UNRESOLVED_PREDECLARED_EXTENSIONS')
    for resolution in extensions.values():
        _require(isinstance(resolution, dict) and set(resolution) == {'status', 'reason'}
            and resolution['status'] in {'INCLUDED', 'EXCLUDED_UNSUPPORTED'}
            and isinstance(resolution['reason'], str) and resolution['reason'].strip(), 'INVALID_EXTENSION_RESOLUTION')
    # Detect changed binding documents even when they were mutated during the scan.
    files.verify(path, admission_sha)
    files.verify(catalog_path, admission['catalog']['sha256'])
    files.verify(manifest_path, admission['catalog_manifest']['sha256'])
    return dict(admission=admission, admission_path=path, admission_sha256=admission_sha,
        snapshot=snapshot, manifest=manifest, files=storage, registration=registration,
        registration_path=registration_path, rows=rows, configs=configs, gates=gates)


def _guard(function, *args):
    try:
        return function(*args)
    except SelectionAdmissionError:
        raise
    except catalog.CatalogError as exc:
        raise SelectionAdmissionError(str(exc)) from None
    except (KeyError, TypeError, ValueError, OSError, OverflowError):
        raise SelectionAdmissionError('MALFORMED_SELECTION_EVIDENCE') from None


def validate_selection_admission(path):
    """Verify all provenance and explicit resolutions, without decoding metrics."""
    context = _guard(_validate, path)
    return {'schema_version': ADMISSION_SCHEMA, 'status': 'ADMISSION_VERIFIED',
        'admission_sha256': context['admission_sha256'], 'registered_jobs': len(context['rows']),
        'methods': context['admission']['methods'], 'metrics_decoded': False,
        'models_deserialized': False, 'evaluation_authorized': False}


def _sources(files):
    base = Path(__file__).resolve().parent
    return {name: files.sha(base / name) for name in _SOURCE_NAMES}


def _model_reference(context, row):
    files, manifest = context['files'], context['manifest']
    run = next(r for r in manifest['runs'] if r['id'] == row['run_id'])
    directory = files.child(files.ref(run['directory']), 'jobs/' + row['job_id'])
    job_path = files.child(directory, 'job.json')
    files.verify(job_path, row['job_sha256'])
    job = catalog.read_metadata(job_path)
    artifacts = {}
    for name, artifact in row['artifacts'].items():
        actual_path = files.child(directory, name)
        _require(str(actual_path) == artifact['path'], 'ARTIFACT_PATH_MISMATCH')
        files.verify(actual_path, artifact['sha256'])
        artifacts[name] = dict(artifact)
    return {
        **{key: row[key] for key in ('registration_id', 'run_id', 'job_id', 'candidate_id', 'seed',
           'method', 'backbone', 'arm_id', 'training_weighted', 'method_version', 'role', 'profile',
           'budget_policy', 'output_type', 'source_identity', 'data_identity', 'data_sha256',
           'job_sha256', 'result_sha256', 'receipt_sha256', 'original_evidence')},
        'analysis_version': context['admission']['methods'][row['method']]['analysis_version'],
        'job_path': str(job_path), 'result_path': str(files.child(directory, 'result.json')),
        'registration_path': str(context['registration_path']),
        'registered_source_files': context['registration']['source_files'],
        'artifacts': artifacts, 'runtime': run.get('runtime'),
        'runtime_receipt': ({'path': str(files.child(directory, 'runtime_receipt.json')),
            'sha256': row['original_evidence']['runtime_receipt_sha256']}
            if row['original_evidence'] and row['original_evidence'].get('runtime_receipt_sha256') else None),
        'runtime_job_metadata': {key: value for key, value in job.items() if key.startswith('runtime_')
            or key in {'recovery_policy', 'recovery_policy_spec', 'execution_budget', 'cache_path',
                       'representation_key', 'original_job_path', 'original_job_sha256',
                       'original_result_sha256', 'registration_path', 'registration_sha256', 'recovery_run_path'}},
    }


def _metric(value):
    _require(value is None or (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1),
             'INVALID_SELECTION_METRIC')
    return value


def _freeze(path):
    context = _validate(path)
    # Lazy import: provenance-only users do not need to import the old worker.
    from .experiment_selection import aggregate_configuration, select_aggregate, OPERATING_POINTS
    summaries, conditions = [], defaultdict(list)
    for config in context['registration']['candidates']:
        if config['status'] != 'REGISTERED':
            summaries.append(config)
            continue
        records = []
        for seed in config['seeds']:
            row = context['rows'][f"{config['candidate_id']}_s{seed}"]
            record = {'seed': seed, 'status': row['status']}
            if row['status'] == 'VALID':
                result_path = context['files'].child(row['directory'], 'result.json')
                context['files'].verify(result_path, row['result_sha256'])
                metrics = catalog.read_metadata(result_path, {'metrics_S', 'risk_S'})
                _require(isinstance(metrics.get('metrics_S'), dict) and isinstance(metrics.get('risk_S'), dict),
                         'MISSING_SELECTION_METRICS')
                record['metrics_S'] = {name: _metric(metrics['metrics_S'].get(name))
                    for name in ('balanced_accuracy', 'eo_gap', 'dp_gap')}
                record['risk_S'] = {'average_precision': _metric(metrics['risk_S'].get('average_precision'))}
                context['files'].verify(result_path, row['result_sha256'])
            records.append(record)
        aggregated = aggregate_configuration(config, records)
        summaries.append({'config': config, **aggregated})
        key = (config['arm_id'], config['backbone'], config['method'], config['training_weighted'])
        conditions[key].append(aggregated)
    selections = []
    for (arm, backbone, method, weighted), candidates in sorted(conditions.items()):
        identity = dict(arm_id=arm, backbone=backbone, method=method, training_weighted=weighted,
            analysis_version=context['admission']['methods'][method]['analysis_version'])
        for tau in OPERATING_POINTS:
            selections.append({**select_aggregate(candidates, tau), **identity})
        if method == 'UNMITIGATED':
            valid = [c for c in candidates if c['status'] == 'VALID' and c['average_precision'] is not None]
            # This is the registered risk-reference rule from freeze_selection.
            chosen = min(valid, key=lambda c: (-c['average_precision'], -c['balanced_accuracy'],
                c['complexity'], c['candidate_id'])) if valid else None
            selections.append({**identity, 'status': 'PREDICTION_REFERENCE' if chosen else 'NOT_ESTIMABLE',
                'tau': None, 'selected_candidate_id': chosen['candidate_id'] if chosen else None})
    selected = {s.get('selected_candidate_id') or s.get('boundary_candidate_id') for s in selections} - {None}
    models = [_model_reference(context, row) for name, row in sorted(context['rows'].items())
              if row['candidate_id'] in selected]
    return {'schema_version': SELECTION_SCHEMA, 'status': 'SELECTION_ARTIFACT_BUILT_REVIEW_REQUIRED',
        'evaluation_authorized': False, 'models_deserialized': False, 'metrics_decoded': True,
        'admission': {'path': str(context['admission_path']), 'sha256': context['admission_sha256']},
        'catalog_sha256': context['admission']['catalog']['sha256'],
        'catalog_manifest_sha256': context['admission']['catalog_manifest']['sha256'],
        'registration': context['admission']['registration'],
        'analysis_source_files': _sources(context['files']),
        'aggregation': "mean of each seed's metric; incomplete seed sets excluded",
        'operating_points': list(OPERATING_POINTS), 'selections': selections,
        'candidate_summaries': summaries, 'selected_models': models,
        'resolved_jobs': context['admission']['resolved_jobs'], 'methods': context['admission']['methods'],
        'compatibility_gates': context['gates'],
        'extension_resolutions': context['admission']['extension_resolutions'],
        'resolved_statuses': {name: row['status'] for name, row in sorted(context['rows'].items())}}


def freeze_catalog_selection(admission_path):
    """Explicitly read S after admission; return an artifact, never authorize T."""
    return _guard(_freeze, admission_path)


def _selected_candidates(context, selections):
    """Check downstream reference structure without reconsidering S rankings."""
    from .experiment_registry import OPERATING_POINTS
    groups = defaultdict(set)
    for config in context['configs'].values():
        if config['status'] == 'REGISTERED':
            key = (config['arm_id'], config['backbone'], config['method'], config['training_weighted'])
            groups[key].add(config['candidate_id'])
    expected_cells = {(key, tau) for key in groups for tau in OPERATING_POINTS}
    expected_cells |= {(key, None) for key in groups if key[2] == 'UNMITIGATED'}
    selected, cells = set(), set()
    _require(isinstance(selections, list), 'INVALID_SELECTION_ROWS')
    for row in selections:
        _require(isinstance(row, dict), 'INVALID_SELECTION_ROW')
        key = tuple(row[k] for k in ('arm_id', 'backbone', 'method', 'training_weighted'))
        cell = (key, row['tau'])
        _require(cell in expected_cells and cell not in cells, 'SELECTION_CONDITION_MISMATCH')
        cells.add(cell)
        _require(row['analysis_version'] == context['admission']['methods'][row['method']]['analysis_version'],
                 'SELECTION_VERSION_MISMATCH')
        chosen, boundary, status = row.get('selected_candidate_id'), row.get('boundary_candidate_id'), row['status']
        if row['tau'] is None:
            _require(status in {'PREDICTION_REFERENCE', 'NOT_ESTIMABLE'} and boundary is None,
                     'INVALID_REFERENCE_SELECTION')
        else:
            _require(status in {'FEASIBLE', 'NO_FEASIBLE_CONFIGURATION', 'NOT_ESTIMABLE'}, 'INVALID_SELECTION_STATUS')
        _require((chosen is not None) == (status in {'FEASIBLE', 'PREDICTION_REFERENCE'})
            and (boundary is not None) == (status == 'NO_FEASIBLE_CONFIGURATION'), 'INVALID_SELECTED_CANDIDATE_STATE')
        candidate = chosen or boundary
        if candidate is not None:
            _require(candidate in groups[key], 'SELECTED_CANDIDATE_CONDITION_MISMATCH')
            selected.add(candidate)
    _require(cells == expected_cells, 'INCOMPLETE_SELECTION_CONDITIONS')
    return selected


def _resolve_with_context(path, expected_sha256):
    path = Path(path).resolve()
    files = catalog._Files({'artifact': str(path.parent)}, [])
    files.verify(path, expected_sha256)
    artifact = catalog.read_metadata(path)
    _require(artifact.get('schema_version') == SELECTION_SCHEMA
        and artifact.get('status') == 'SELECTION_ARTIFACT_BUILT_REVIEW_REQUIRED'
        and artifact.get('evaluation_authorized') is False
        and artifact.get('models_deserialized') is False, 'INVALID_SELECTION_ARTIFACT')
    admission_path = Path(artifact['admission']['path'])
    _require(admission_path.is_absolute() and '..' not in admission_path.parts, 'INVALID_ADMISSION_PATH')
    files.verify(admission_path, artifact['admission']['sha256'])
    context = _validate(admission_path)
    from .experiment_registry import OPERATING_POINTS
    for key, expected in {'catalog_sha256': context['admission']['catalog']['sha256'],
        'catalog_manifest_sha256': context['admission']['catalog_manifest']['sha256'],
        'registration': context['admission']['registration'], 'analysis_source_files': _sources(files),
        'resolved_jobs': context['admission']['resolved_jobs'], 'methods': context['admission']['methods'],
        'compatibility_gates': context['gates'], 'operating_points': list(OPERATING_POINTS),
        'extension_resolutions': context['admission']['extension_resolutions'],
        'resolved_statuses': {name: row['status'] for name, row in sorted(context['rows'].items())}}.items():
        _require(artifact.get(key) == expected, 'SELECTION_ARTIFACT_BINDING_MISMATCH')
    selected = _selected_candidates(context, artifact['selections'])
    rows = [r for name, r in sorted(context['rows'].items()) if r['candidate_id'] in selected]
    _require(all(r['status'] == 'VALID' for r in rows), 'FAILED_MODEL_SELECTED')
    expected_models = [_model_reference(context, row) for row in rows]
    _require(artifact.get('selected_models') == expected_models, 'SELECTED_MODEL_REFERENCE_MISMATCH')
    files.verify(path, expected_sha256)
    return ({'selection_sha256': expected_sha256, 'evaluation_authorized': False,
             'models_deserialized': False, 'models': expected_models}, context)


def _resolve(path, expected_sha256):
    return _resolve_with_context(path, expected_sha256)[0]


def resolve_selected_models(selection_path, expected_sha256):
    """Revalidate references without reading result metrics or loading models.

    The expected artifact hash must come from the independent caller's gate.
    Returned references are inputs to a future evaluator; they grant no release.
    """
    return _guard(_resolve, selection_path, expected_sha256)


def resolve_selected_models_context(selection_path, expected_sha256):
    """Return the same validated references and their in-call metadata context.

    Study freezing needs fixed, parameter-selected ablations in addition to the
    tuned selections. Reusing this context avoids re-scanning the complete
    catalog twice within one freeze. It is not an admission bypass or a cached
    authorization: every invocation repeats the original full validation.
    """
    return _guard(_resolve_with_context, selection_path, expected_sha256)
