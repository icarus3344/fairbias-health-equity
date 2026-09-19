"""Independent, provenance-bound paper export from cross-run aggregate results.

No fitting, selection, model deserialization, prediction-array loading, or
microdata access occurs here. Reporting does not grant evaluation authority.
Historical runs are inputs only; every report uses a fresh output directory.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

from . import result_catalog as catalog
from .paper_reporting import _csv, _flatten, _key, _nested

REPORT_SCHEMA = 'nhis_catalog_paper_report_v2'
RISK_METRICS = ('average_precision', 'auroc', 'brier')
FAIRNESS_METRICS = ('balanced_accuracy', 'eo_gap', 'dp_gap')
ANALYSIS_REQUIRED = {'src/nhis_fairbias/benchmark/' + name for name in (
    'catalog_reporting.py', 'paper_reporting.py', 'catalog_evaluation.py', 'catalog_inference.py',
    'catalog_selection.py', 'result_catalog.py', 'experiment_selection.py', 'experiment_registry.py',
    'experiment_evaluation.py', 'survey_batch.py', 'survey_linearization.py', 'survey_inference.py',
    'metrics.py', 'risk_metrics.py', 'data_contracts.py', 'cohort_reporting.py', 'predictions.py')}
ANALYSIS_REQUIRED.add('scripts/evaluate_nhis_catalog.py')


class ReportingError(ValueError):
    """A bounded provenance/schema error without data-bearing exception text."""


def _require(condition, code):
    if not condition:
        raise ReportingError(code)


def _object(value, required, code):
    _require(isinstance(value, dict) and set(required) <= value.keys(), code)
    return value


def _path(value):
    _require(isinstance(value, str), 'INVALID_ABSOLUTE_PATH')
    path = Path(value)
    _require(path.is_absolute() and '..' not in path.parts, 'INVALID_ABSOLUTE_PATH')
    return path.resolve()


def _number(value):
    return value is None or type(value) in (int, float) and math.isfinite(value)


def _mean(value):
    _object(value, {'mean', 'seed_sd', 'seed_range', 'status'}, 'MISSING_AGGREGATE_METRIC')
    _require(_number(value['mean']) and (value['mean'] is None or 0 <= value['mean'] <= 1), 'INVALID_AGGREGATE_METRIC')
    _require(_number(value['seed_sd']) and (value['seed_sd'] is None or value['seed_sd'] >= 0), 'INVALID_SEED_SD')
    _require(value['seed_range'] is None or isinstance(value['seed_range'], list)
        and len(value['seed_range']) == 2 and all(_number(v) and v is not None for v in value['seed_range']), 'INVALID_SEED_RANGE')
    return value['mean']


def _interval(value, *, projection=False):
    _object(value, {'status'}, 'MISSING_INTERVAL_STATUS')
    _require(value['status'] in {'VALID', 'NOT_ESTIMABLE'}, 'INVALID_INTERVAL_STATUS')
    if value['status'] == 'VALID':
        _object(value, {'lower', 'upper'}, 'MISSING_VALID_INTERVAL_BOUNDS')
        lo, hi = value['lower'], value['upper']
        _require(_number(lo) and lo is not None and _number(hi) and hi is not None and lo <= hi, 'INVALID_INTERVAL_BOUNDS')
        _require(not projection or 0 <= lo <= hi <= 1, 'INVALID_PROJECTION_BOUNDS')
        if 'se' in value:
            _require(_number(value['se']) and value['se'] is not None and value['se'] >= 0, 'INVALID_INTERVAL_SE')
    else:
        _require(value.get('lower') is None and value.get('upper') is None, 'NONESTIMABLE_INTERVAL_HAS_BOUNDS')


def _flat_selection(row, selected):
    # Match by the frozen selection identity before using legacy formatting;
    # method/tau alone cannot distinguish fixed or budget-sensitive branches.
    flat = _flatten(row, {_key(row): selected})
    for key in ('selection_id', 'study_branch', 'analysis_version'):
        flat[key] = row[key]
    flat['candidate_id'] = selected.get('selected_candidate_id') or selected.get('boundary_candidate_id')
    flat['model_ids'] = json.dumps(row['frozen_seed_models'], separators=(',', ':'))
    return flat


def _flat_pair(row):
    flat = {key: row[key] for key in ('arm_id', 'backbone', 'tau', 'training_weighted',
        'reference_method', 'comparison_method', 'feasible_on_S', 'in_primary_family_20',
        'reference_selection_id', 'comparison_selection_id')}
    flat.update(reference_study_branch=row['reference_study_branch'],
                comparison_study_branch=row['comparison_study_branch'],
                reference_model_ids=json.dumps(row['reference_model_ids'], separators=(',', ':')),
                comparison_model_ids=json.dumps(row['comparison_model_ids'], separators=(',', ':')))
    for metric in RISK_METRICS:
        flat['delta_' + metric] = row['delta_' + metric]
    for metric in ('balanced_accuracy', 'eo_gap'):
        flat['delta_' + metric] = _nested(row, 'delta_' + metric, 'estimate')
    flat['delta_ba_se'] = _nested(row, 'delta_balanced_accuracy', 'se')
    for side in ('lower', 'upper'):
        flat['delta_ba_' + side] = _nested(row, 'delta_balanced_accuracy', side)
        flat['delta_ba_family20_' + side] = _nested(row, 'delta_balanced_accuracy', 'family_20_interval', side)
        flat['delta_eo_projection_' + side] = _nested(row, 'delta_eo_gap', 'projection_95', side)
        flat['delta_eo_family20_' + side] = _nested(row, 'delta_eo_gap', 'projection_primary_family', side)
    return flat


def _inventory(selection, files, *, catalog=catalog):
    """Verify explicit cross-run denominator evidence without decoding metrics."""
    admission_path = _path(selection['admission']['path'])
    files.verify(admission_path, selection['admission']['sha256'])
    admission = catalog.read_metadata(admission_path)
    _require(admission.get('schema_version') == 'nhis_catalog_selection_admission_v1'
        and admission.get('decision') == 'SUPERVISOR_ADMITTED'
        and admission.get('evaluation_authorized') is False, 'INVALID_SELECTION_ADMISSION')
    roots = catalog._Files(admission['roots'], [])
    bound = {}
    for name in ('catalog', 'catalog_manifest'):
        item = admission[name]
        path = roots.ref(item['path'])
        roots.verify(path, item['sha256'])
        _require(selection[name + '_sha256'] == item['sha256'], 'SELECTION_CATALOG_BINDING_MISMATCH')
        bound[name] = catalog.read_metadata(path)
    snapshot, manifest = bound['catalog'], bound['catalog_manifest']
    _require(snapshot.get('schema_version') == catalog.CATALOG_SCHEMA and snapshot.get('integrity_passed') is True
        and snapshot.get('issues') == [] and snapshot.get('metrics_decoded') is False
        and snapshot.get('evaluation_authorized') is False
        and snapshot.get('manifest_sha256') == admission['catalog_manifest']['sha256'], 'CATALOG_NOT_ADMISSIBLE')
    _require(manifest.get('schema_version') == catalog.MANIFEST_SCHEMA, 'INVALID_CATALOG_MANIFEST')
    storage = catalog._Files(manifest['roots'], manifest.get('relocations', []))
    registration_id = selection['registration']['id']
    _require(selection['registration'] == admission['registration'], 'REGISTRATION_BINDING_MISMATCH')
    declarations = [r for r in manifest['registrations'] if r['id'] == registration_id]
    _require(len(declarations) == 1 and declarations[0]['sha256'] == selection['registration']['sha256'],
             'REGISTRATION_BINDING_MISMATCH')
    registration_path = storage.ref(declarations[0]['path'])
    storage.verify(registration_path, declarations[0]['sha256'])
    registration = catalog.read_metadata(registration_path)
    registered = {f"{config['candidate_id']}_s{seed}": (config, seed)
        for config in registration['candidates'] if config['status'] == 'REGISTERED' for seed in config['seeds']}
    _require(registered and selection['resolved_jobs'] == admission['resolved_jobs']
        and set(selection['resolved_jobs']) == set(registered)
        and set(selection['resolved_statuses']) == set(registered), 'REGISTERED_JOB_COVERAGE_MISMATCH')
    runs = {run['id']: run for run in manifest['runs']}
    attempts, paths = {}, set()
    for row in snapshot['jobs']:
        key = (row['registration_id'], row['run_id'], row['job_id'])
        _require(key not in attempts and row['status'] in catalog.TERMINAL, 'DUPLICATE_OR_UNRESOLVED_ATTEMPT')
        run = runs[row['run_id']]
        directory = storage.child(storage.ref(run['directory']), 'jobs/' + row['job_id'])
        _require(str(directory) == row['directory'] and directory not in paths
            and row['registration_id'] == run['registration_id'] and row['job_id'] in run['expected_jobs'],
            'ATTEMPT_PATH_OR_OWNERSHIP_MISMATCH')
        paths.add(directory)
        for name, field in (('job.json', 'job_sha256'), ('result.json', 'result_sha256'), ('receipt.json', 'receipt_sha256')):
            storage.verify(storage.child(directory, name), row[field])
        result = catalog.read_metadata(storage.child(directory, 'result.json'), catalog.RESULT_FIELDS)
        _require(all(result.get(k) == row[k] for k in ('candidate_id', 'seed', 'status')), 'RESULT_IDENTITY_MISMATCH')
        receipt = catalog.read_metadata(storage.child(directory, 'receipt.json'))
        _require(receipt.get('files', {}).get('job.json') == row['job_sha256']
            and receipt.get('files', {}).get('result.json') == row['result_sha256'], 'RECEIPT_BINDING_MISMATCH')
        attempts[key] = row
    expected_attempts = {(run['registration_id'], run['id'], job) for run in manifest['runs'] for job in run['expected_jobs']}
    _require(set(attempts) == expected_attempts, 'ATTEMPT_COVERAGE_MISMATCH')
    resolved = []
    for name, run_id in sorted(selection['resolved_jobs'].items()):
        key = (registration_id, run_id, name)
        _require(key in attempts, 'MISSING_RESOLVED_ATTEMPT')
        row = attempts[key]
        config, seed = registered[name]
        _require(row['candidate_id'] == config['candidate_id'] and row['seed'] == seed
            and row['method'] == config['method'] and row['status'] == selection['resolved_statuses'][name],
            'RESOLVED_JOB_IDENTITY_MISMATCH')
        resolved.append({**row, 'job_path': str(storage.child(row['directory'], 'job.json')),
                         'result_path': str(storage.child(row['directory'], 'result.json'))})
    return {'registration': registration, 'registration_path': str(registration_path),
            'resolved': resolved, 'attempts': list(attempts.values()), 'attempt_index': attempts,
            'catalog': snapshot, 'runs': runs}


def _verify_bound(files, item, expected_path=None, *, decode_json=True, catalog=catalog):
    _object(item, {'path', 'sha256'}, 'MISSING_BOUND_FILE')
    path = _path(item['path'])
    _require(expected_path is None or path == Path(expected_path).resolve(), 'BOUND_PATH_MISMATCH')
    files.verify(path, item['sha256'])
    # Supplemental evidence may be a plan or other opaque artifact. Integrity
    # is mandatory for every file; JSON object semantics apply only to inputs
    # whose contents the reporting contract actually interprets.
    return catalog.read_metadata(path) if decode_json else None


def _validate_models(study, inventory, *, catalog=catalog):
    models, identity_keys = {}, set()
    configs = {c['candidate_id']: c for c in inventory['registration']['candidates'] if c['status'] == 'REGISTERED'}
    _require(isinstance(study['selected_models'], list), 'INVALID_MODEL_LIST')
    for model in study['selected_models']:
        _object(model, {'model_id', 'registration_id', 'run_id', 'job_id', 'candidate_id', 'seed', 'method',
            'arm_id', 'backbone', 'training_weighted', 'method_version', 'analysis_version', 'role', 'profile',
            'budget_policy', 'output_type', 'runtime', 'runtime_job_metadata', 'artifacts', 'source_identity',
            'data_identity', 'data_sha256', 'job_path', 'result_path', 'job_sha256', 'result_sha256', 'receipt_sha256',
            'original_evidence', 'registration_path', 'registered_source_files', 'runtime_receipt'}, 'MISSING_MODEL_PROVENANCE')
        model_id = model['model_id']
        _require(isinstance(model_id, str) and model_id and model_id not in models, 'DUPLICATE_OR_INVALID_MODEL_ID')
        _require(model_id == 'model_' + catalog._identity({k: v for k, v in model.items() if k != 'model_id'}),
                 'MODEL_ID_PROVENANCE_MISMATCH')
        key = (model['analysis_version'], model['candidate_id'], model['seed'])
        _require(key not in identity_keys, 'DUPLICATE_VERSIONED_CANDIDATE_SEED')
        identity_keys.add(key)
        _require(model['candidate_id'] in configs, 'UNKNOWN_MODEL_CANDIDATE')
        config = configs[model['candidate_id']]
        _require(model['seed'] in config['seeds'] and all(model[k] == config[k]
            for k in ('method', 'arm_id', 'backbone', 'training_weighted')), 'MODEL_CONFIGURATION_MISMATCH')
        attempt = inventory['attempt_index'].get((model['registration_id'], model['run_id'], model['job_id']))
        _require(attempt is not None and attempt['status'] == 'VALID', 'MODEL_WITHOUT_VALID_ATTEMPT')
        _require(all(model[k] == attempt[k] for k in ('candidate_id', 'seed', 'method_version', 'role', 'profile',
            'budget_policy', 'output_type', 'source_identity', 'data_identity', 'data_sha256', 'original_evidence',
            'job_sha256', 'result_sha256', 'receipt_sha256')),
            'MODEL_ATTEMPT_PROVENANCE_MISMATCH')
        _require(model['artifacts'] == attempt['artifacts'], 'MODEL_ARTIFACT_BINDING_MISMATCH')
        _require(model['runtime'] == inventory['runs'][model['run_id']].get('runtime'), 'MODEL_RUNTIME_MISMATCH')
        directory = _path(attempt['directory'])
        _require(_path(model['job_path']) == directory / 'job.json'
            and _path(model['result_path']) == directory / 'result.json', 'MODEL_PATH_MISMATCH')
        _require(model['registered_source_files'] == inventory['registration']['source_files']
            and model['registration_path'] == inventory['registration_path'], 'MODEL_REGISTRATION_PROVENANCE_MISMATCH')
        job = catalog.read_metadata(directory / 'job.json')
        runtime_keys = {'recovery_policy', 'recovery_policy_spec', 'execution_budget', 'cache_path', 'representation_key',
            'original_job_path', 'original_job_sha256', 'original_result_sha256', 'registration_path', 'registration_sha256', 'recovery_run_path'}
        _require(model['runtime_job_metadata'] == {k: v for k, v in job.items() if k.startswith('runtime_') or k in runtime_keys},
                 'MODEL_RUNTIME_JOB_METADATA_MISMATCH')
        rr_sha = (attempt.get('original_evidence') or {}).get('runtime_receipt_sha256')
        expected_receipt = {'path': str(directory / 'runtime_receipt.json'), 'sha256': rr_sha} if rr_sha else None
        _require(model['runtime_receipt'] == expected_receipt, 'MODEL_RUNTIME_RECEIPT_MISMATCH')
        for name, artifact in model['artifacts'].items():
            _require(name in {'model.joblib', 'predictions_S.npz'} and _path(artifact['path']) == directory / name
                and catalog.SHA.fullmatch(artifact['sha256']), 'UNSAFE_MODEL_ARTIFACT')
        _require(model['output_type'] == ('decision_probability_q' if model['method'] in catalog.Q_METHODS else 'event_probability_p'),
                 'MODEL_P_Q_IDENTITY_MISMATCH')
        models[model_id] = model
    grouped_seeds = {}
    for model in models.values():
        grouped_seeds.setdefault((model['analysis_version'], model['candidate_id']), set()).add(model['seed'])
    _require(all(seeds == set(configs[cid]['seeds']) for (_, cid), seeds in grouped_seeds.items()), 'INCOMPLETE_FROZEN_SEEDS')
    _require(isinstance(study['arm_ids'], list) and len(study['arm_ids']) == len(set(study['arm_ids'])), 'INVALID_STUDY_ARMS')
    expected = {arm: [] for arm in study['arm_ids']}
    for model_id, model in models.items():
        _require(model['arm_id'] in expected, 'MODEL_ARM_NOT_REGISTERED')
        expected[model['arm_id']].append(model_id)
    expected = {arm: sorted(ids) for arm, ids in expected.items()}
    _require({arm: sorted(ids) for arm, ids in study['expected_modelsets'].items()} == expected
        and all(len(ids) == len(set(ids)) for ids in study['expected_modelsets'].values()), 'STUDY_MODELSET_MISMATCH')
    return models


def _validate_selections(study, selection, models):
    for reference in selection['selected_models']:
        _require(any({k: v for k, v in model.items() if k != 'model_id'} == reference for model in models.values()),
                 'ORIGINAL_SELECTED_MODEL_MISSING_OR_CHANGED')
    selections = {}
    used_models = set()
    for row in study['selections']:
        _object(row, {'selection_id', 'model_ids', 'study_branch', 'analysis_version', 'status',
            'arm_id', 'method', 'backbone', 'training_weighted', 'tau', 'selected_candidate_id'}, 'MISSING_FROZEN_SELECTION_FIELD')
        identity = row['selection_id']
        _require(isinstance(identity, str) and identity and identity not in selections, 'DUPLICATE_SELECTION_ID')
        _require(identity == 'selection_' + catalog._identity({k: v for k, v in row.items() if k != 'selection_id'}),
                 'SELECTION_ID_PROVENANCE_MISMATCH')
        _require(row['study_branch'] == study['study_branch'], 'MIXED_STUDY_BRANCH')
        ids = row['model_ids']
        _require(isinstance(ids, list) and len(ids) == len(set(ids)) and set(ids) <= models.keys(), 'INVALID_SELECTION_MODELSET')
        cid = row.get('selected_candidate_id') or row.get('boundary_candidate_id')
        expected = {mid for mid, model in models.items() if model['candidate_id'] == cid
            and model['analysis_version'] == row['analysis_version']}
        _require(set(ids) == expected, 'INCOMPLETE_SELECTION_MODELS')
        for mid in ids:
            _require(all(row[k] == models[mid][k] for k in ('arm_id', 'method', 'backbone', 'training_weighted', 'analysis_version')),
                     'SELECTION_MODEL_IDENTITY_MISMATCH')
        used_models.update(ids)
        selections[identity] = row
    _require(used_models == set(models), 'EXTRA_OR_UNUSED_MODEL')
    # Study may add parameter-fixed ablations, but may not drop or mutate the
    # already frozen S-selected conditions to produce a more favorable table.
    for original in selection['selections']:
        matches = [row for row in selections.values() if all(row.get(k) == v for k, v in original.items())]
        _require(len(matches) == 1, 'FROZEN_SELECTION_OMITTED_OR_CHANGED')
    return selections


def _validate_summary(summary, frozen, models):
    rows = {}
    for row in summary['selections']:
        _object(row, {'selection_id', 'model_ids', 'frozen_seed_models', 'evaluation_status'}, 'MISSING_SUMMARY_FIELD')
        identity = row['selection_id']
        _require(identity in frozen and identity not in rows, 'EXTRA_OR_DUPLICATE_SUMMARY_SELECTION')
        selected = frozen[identity]
        _require(all(row.get(k) == value for k, value in selected.items()), 'SUMMARY_SELECTION_BINDING_MISMATCH')
        _require(row['frozen_seed_models'] == selected['model_ids'], 'SUMMARY_MODELSET_MISMATCH')
        ids = selected['model_ids']
        if ids:
            _require(row['evaluation_status'] in {'VALID', 'PARTIALLY_ESTIMABLE', 'DESIGN_NOT_ESTIMABLE'}, 'INVALID_EVALUATION_STATUS')
            _object(row, {'sample_n', 'df'}, 'MISSING_DESIGN_FIELDS')
            _require(type(row['sample_n']) is int and row['sample_n'] >= 0 and _number(row['df']), 'INVALID_DESIGN_FIELDS')
            output_types = {models[mid]['output_type'] for mid in ids}
            _require(len(output_types) == 1, 'MIXED_P_Q_SEEDS')
            q_only = output_types == {'decision_probability_q'}
            for prefix in ('', 'unweighted_'):
                for name in (*FAIRNESS_METRICS, *RISK_METRICS):
                    _require(prefix + name in row, 'MISSING_REQUIRED_METRIC')
                    value = _mean(row[prefix + name])
                    _require(not (q_only and name in RISK_METRICS and value is not None), 'Q_ONLY_RISK_METRIC_PRESENT')
                    if q_only and name in RISK_METRICS:
                        _require(row[prefix + name]['seed_sd'] is None and row[prefix + name]['seed_range'] is None,
                                 'Q_ONLY_RISK_VARIATION_PRESENT')
            for name in RISK_METRICS:
                for prefix in ('untouched_base_', 'unweighted_untouched_base_'):
                    if prefix + name in row:
                        _mean(row[prefix + name])
            _require('taylor_95' in row['balanced_accuracy'], 'MISSING_TAYLOR_INTERVAL')
            _interval(row['balanced_accuracy']['taylor_95'])  # Taylor intervals need not lie in [0, 1].
            for name in ('eo_gap', 'dp_gap'):
                for band in ('projection_95', 'projection_primary_family'):
                    _require(band in row[name], 'MISSING_PROJECTION_INTERVAL')
                    _interval(row[name][band], projection=True)
        else:
            _require(row['evaluation_status'] == 'NO_VALID_MODEL', 'FAILED_SELECTION_PROMOTED')
            _require(all(name not in row or _mean(row[name]) is None for name in (*FAIRNESS_METRICS, *RISK_METRICS)),
                     'METRICS_WITHOUT_FROZEN_MODEL')
        rows[identity] = row
    _require(set(rows) == set(frozen), 'MISSING_SUMMARY_SELECTION')
    expected_pairs = {(a['selection_id'], b['selection_id']) for a in rows.values() for b in rows.values()
        if a['method'] == 'FAIRBIAS_BM' and b['method'] != 'FAIRBIAS_BM' and a['model_ids'] and b['model_ids']
        and all(a[k] == b[k] for k in ('arm_id', 'backbone', 'tau', 'training_weighted'))
        and (a['status'] == 'FIXED_ABLATION') == (b['status'] == 'FIXED_ABLATION')}
    paired, seen = [], set()
    for row in summary['paired_contrasts']:
        _object(row, {'reference_selection_id', 'comparison_selection_id', 'reference_method', 'comparison_method',
            'arm_id', 'backbone', 'tau', 'training_weighted', 'feasible_on_S', 'in_primary_family_20',
            'reference_model_ids', 'comparison_model_ids', 'study_branch', 'delta_balanced_accuracy', 'delta_eo_gap',
            *('delta_' + name for name in RISK_METRICS)}, 'MISSING_PAIRED_FIELD')
        key = (row['reference_selection_id'], row['comparison_selection_id'])
        _require(key in expected_pairs and key not in seen, 'INVALID_OR_DUPLICATE_PAIR')
        seen.add(key)
        reference, comparison = rows[key[0]], rows[key[1]]
        _require(reference['model_ids'] and comparison['model_ids'], 'PAIR_WITHOUT_MODELS')
        _require(row['reference_model_ids'] == reference['model_ids'] and row['comparison_model_ids'] == comparison['model_ids']
            and row['study_branch'] == reference['study_branch'] == comparison['study_branch'], 'PAIRED_MODEL_BINDING_MISMATCH')
        _require(row['reference_method'] == reference['method'] and row['comparison_method'] == comparison['method']
            and all(row[k] == reference[k] == comparison[k] for k in ('arm_id', 'backbone', 'tau', 'training_weighted')),
            'PAIRED_CONDITION_MISMATCH')
        _require(row['feasible_on_S'] == (reference['status'] == comparison['status'] == 'FEASIBLE'), 'PAIR_FEASIBILITY_MISMATCH')
        primary = (row['arm_id'] in {'arm_001', 'arm_003'} and row['backbone'] == 'LR' and not row['training_weighted']
            and row['tau'] == .1 and row['comparison_method'] in {'REWEIGHING', 'LFR_RECONSTRUCTED', 'EG_DP', 'EG_EO', 'TO_EO'})
        _require(type(row['in_primary_family_20']) is bool and row['in_primary_family_20'] == primary, 'PRIMARY_FAMILY_MISMATCH')
        for name in ('balanced_accuracy', 'eo_gap'):
            _object(row['delta_' + name], {'estimate'}, 'MISSING_PAIRED_ESTIMATE')
            point = row['delta_' + name]['estimate']
            _require(_number(point) and (point is None or -1 <= point <= 1), 'INVALID_PAIRED_ESTIMATE')
        _interval(row['delta_balanced_accuracy'])
        for band in ('projection_95', 'projection_primary_family'):
            _require(band in row['delta_eo_gap'], 'MISSING_PAIRED_PROJECTION')
            _interval(row['delta_eo_gap'][band])
        has_q = any(models[mid]['output_type'] == 'decision_probability_q' for mid in reference['model_ids'] + comparison['model_ids'])
        _require(all(_number(row['delta_' + name]) and (row['delta_' + name] is None or -1 <= row['delta_' + name] <= 1)
            and (not has_q or row['delta_' + name] is None) for name in RISK_METRICS),
                 'INVALID_PAIRED_RISK_ESTIMAND')
        paired.append({**row, 'reference_study_branch': reference['study_branch'],
                       'comparison_study_branch': comparison['study_branch']})
    _require(seen == expected_pairs, 'MISSING_PAIRED_CONTRAST')
    return list(rows.values()), paired


def _validate(study_path, selection_path, evaluation_path, summary_path,
              study_sha, selection_sha, summary_sha, analysis_root, *, catalog=catalog):
    paths = [Path(p).resolve() for p in (study_path, selection_path, evaluation_path, summary_path)]
    _require(len(set(paths)) == len(paths), 'DUPLICATE_INPUT_PATH')
    study_path, selection_path, evaluation_path, summary_path = paths
    files = catalog._Files({'input': str(study_path.parent)}, [])
    for path, expected in ((study_path, study_sha), (selection_path, selection_sha), (summary_path, summary_sha)):
        files.verify(path, expected)
    study, selection, evaluation, summary = map(catalog.read_metadata, paths)
    _require(study.get('schema_version') == 'nhis_catalog_study_freeze_v1'
        and study.get('status') == 'STUDY_FREEZE_BUILT_REVIEW_REQUIRED', 'INVALID_STUDY_FREEZE')
    _object(study, {'study_branch', 'selection', 'admission', 'catalog', 'catalog_manifest', 'registration',
        'analysis_root', 'analysis_files', 'selected_models', 'selections', 'expected_modelsets',
        'resolved_jobs', 'inference_policies', 't_source_provenance', 'extension_resolutions', 'arm_ids'}, 'MISSING_STUDY_FIELD')
    _require(isinstance(study['study_branch'], str) and catalog.LABEL.fullmatch(study['study_branch']), 'INVALID_STUDY_BRANCH')
    _require(study['selection']['sha256'] == selection_sha, 'STUDY_SELECTION_HASH_MISMATCH')
    _verify_bound(files, study['selection'], selection_path, catalog=catalog)
    _require(selection.get('schema_version') == 'nhis_catalog_selection_v1'
        and selection.get('evaluation_authorized') is False, 'INVALID_SELECTION_ARTIFACT')
    _require(evaluation.get('schema_version') == 'nhis_catalog_evaluation_manifest_v1'
        and evaluation.get('scope') == 'complete_study' and evaluation.get('status') == 'COMPLETE', 'COMPLETE_STUDY_EVALUATION_REQUIRED')
    _require(summary.get('schema_version') == 'nhis_catalog_summary_T_v1', 'INVALID_SUMMARY_SCHEMA')
    for document in (evaluation, summary):
        _require(document.get('study_freeze_sha256') == study_sha
            and document.get('selection_sha256') == selection_sha
            and document.get('study_branch') == study['study_branch'], 'STUDY_EVALUATION_BINDING_MISMATCH')
    files.verify(evaluation_path, summary['evaluation_manifest_sha256'])
    root = _path(str(analysis_root)) if analysis_root is not None else _path(study['analysis_root'])
    sources = catalog._Files({'analysis': str(root)}, [])
    _require(isinstance(study['analysis_files'], dict) and ANALYSIS_REQUIRED <= study['analysis_files'].keys(),
             'INCOMPLETE_ANALYSIS_SOURCE_BINDING')
    current_sources = {p.relative_to(root).as_posix() for package in ('nhis_fairbias', 'fairbias')
                       for p in (root / 'src' / package).rglob('*.py')}
    current_sources.add('scripts/evaluate_nhis_catalog.py')
    _require(set(study['analysis_files']) == current_sources, 'ANALYSIS_SOURCE_CLOSURE_MISMATCH')
    sources.source_manifest(root, study['analysis_files'])
    _require(evaluation.get('analysis_files') == study['analysis_files'], 'ANALYSIS_SOURCE_BINDING_MISMATCH')
    _require(evaluation.get('source_provenance') == study['t_source_provenance'], 'T_SOURCE_PROVENANCE_MISMATCH')
    observed = evaluation.get('observed_source_provenance')
    _require(isinstance(observed, dict), 'MISSING_OBSERVED_T_PROVENANCE')
    declared = study['t_source_provenance']
    _require(set(observed) == set(declared), 'OBSERVED_T_PROVENANCE_MISMATCH')
    for key in declared:
        if key != 'raw_sources':
            _require(observed[key] == declared[key], 'OBSERVED_T_PROVENANCE_MISMATCH')
    _require(set(observed['raw_sources']) == set(declared['raw_sources']) == {'2024'}, 'OBSERVED_T_PROVENANCE_MISMATCH')
    expected_source, actual_source = declared['raw_sources']['2024'], observed['raw_sources']['2024']
    _require(set(actual_source) <= set(expected_source) | {'rows'} and all(actual_source.get(k) == v for k, v in expected_source.items())
        and type(actual_source.get('rows')) is int and actual_source['rows'] > 0, 'OBSERVED_T_PROVENANCE_MISMATCH')
    release = _verify_bound(files, evaluation['release'], catalog=catalog)
    _require(set(release) == {'schema_version', 'decision', 'study_freeze_sha256', 'selection_sha256', 'study_branch', 'arm_ids'}
        and release.get('schema_version') == 'nhis_catalog_t_release_v1' and release.get('decision') == 'SUPERVISOR_RELEASED_T'
        and release.get('study_freeze_sha256') == study_sha and release.get('selection_sha256') == selection_sha
        and release.get('study_branch') == study['study_branch'] and set(release.get('arm_ids', [])) == set(study['arm_ids'])
        and len(release['arm_ids']) == len(set(release['arm_ids'])),
             'MISSING_STUDY_EVALUATION_RELEASE')
    for key, value in {'bootstrap_replicates': 2000, 'bootstrap_seed': 20260914, 'primary_contrast_family_size': 20}.items():
        _require(study.get(key) == value and evaluation.get(key) == value, 'REGISTERED_STATISTICS_MISMATCH')
    branches = set()
    for branch in study.get('supplemental_evidence', []):
        _require(set(branch) == {'study_branch', 'status', 'evidence_files'}
            and branch['study_branch'] != study['study_branch'] and branch['study_branch'] not in branches
            and branch['status'] == 'AWAITING_SEPARATE_FROZEN_EVALUATION'
            and isinstance(branch['evidence_files'], list) and branch['evidence_files'], 'INVALID_SUPPLEMENTAL_BRANCH')
        branches.add(branch['study_branch'])
        for item in branch['evidence_files']:
            _verify_bound(files, item, decode_json=False, catalog=catalog)
    inventory = _inventory(selection, files, catalog=catalog)
    _require(study['admission'] == selection['admission'], 'STUDY_ADMISSION_BINDING_MISMATCH')
    for name in ('catalog', 'catalog_manifest'):
        _verify_bound(files, study[name], catalog=catalog)
        _require(study[name]['sha256'] == selection[name + '_sha256'], 'STUDY_CATALOG_BINDING_MISMATCH')
    _verify_bound(files, study['registration'], inventory['registration_path'], catalog=catalog)
    _require(study['registration']['sha256'] == selection['registration']['sha256'], 'STUDY_REGISTRATION_MISMATCH')
    # Evaluator freezes resolved metadata; strict comparison prevents a selected
    # subset from being advertised as the original registered denominator.
    _require(study['resolved_jobs'] == inventory['resolved'], 'STUDY_RESOLVED_DENOMINATOR_MISMATCH')
    models = _validate_models(study, inventory, catalog=catalog)
    frozen = _validate_selections(study, selection, models)
    _require(evaluation.get('expected_modelsets') == study['expected_modelsets']
        and evaluation.get('selected_models') == study['selected_models']
        and set(evaluation.get('arm_ids', [])) == set(study['arm_ids'])
        and len(evaluation['arm_ids']) == len(set(evaluation['arm_ids'])), 'EVALUATION_MODELSET_MISMATCH')
    rows, paired = _validate_summary(summary, frozen, models)
    return dict(study=study, selection=selection, evaluation=evaluation, summary=summary,
        inventory=inventory, rows=rows, paired=paired, frozen=frozen, files=files,
        paths=paths, analysis_root=str(root), hashes={str(path): files.sha(path) for path in paths})


def _counts(records, fields):
    counts = Counter(tuple(r[field] for field in fields) + (r['status'],) for r in records)
    return [{**dict(zip((*fields, 'status'), key)), 'jobs': count} for key, count in sorted(counts.items())]


def _write(context, output):
    out = Path(output).resolve()
    _require(not out.exists(), 'OUTPUT_ALREADY_EXISTS')
    protected = {_path(r['directory']).parent.parent for r in context['inventory']['attempts']}
    _require(all(not out.is_relative_to(path) for path in protected), 'OUTPUT_INSIDE_EVIDENCE')
    rows, paired, inventory = context['rows'], context['paired'], context['inventory']
    primary = [r for r in rows if r['tau'] == .10 and r['study_branch'] == 'registered_budget_main']
    auxiliary = [r for r in rows if r not in primary]
    flattened = [_flat_selection(r, context['frozen'][r['selection_id']]) for r in rows]
    fields = list(flattened[0]) if flattened else list(_flatten({}, {})) + ['selection_id', 'study_branch', 'analysis_version', 'candidate_id', 'model_ids']
    paired_rows = [_flat_pair(row) for row in paired]
    paired_fields = list(paired_rows[0]) if paired_rows else ['reference_selection_id', 'comparison_selection_id']
    unsupported = [{key: config.get(key) for key in ('candidate_id', 'arm_id', 'backbone', 'method', 'status', 'reason')}
        for config in inventory['registration']['candidates'] if config['status'] != 'REGISTERED']
    failure_fields = ['registration_id', 'job_id', 'candidate_id', 'seed', 'arm_id', 'backbone', 'method',
        'training_weighted', 'run_id', 'method_version', 'role', 'status', 'error_type', 'termination_reason',
        'result_path', 'result_sha256']
    failures = [{field: row.get(field) for field in failure_fields} for row in inventory['resolved'] if row['status'] != 'VALID']
    status_counts = _counts(inventory['resolved'], ('method', 'method_version', 'role'))
    attempt_counts = _counts(inventory['attempts'], ('method', 'method_version', 'role'))
    denominators = {'registered_jobs': len(inventory['resolved']), 'catalog_attempts': len(inventory['attempts']),
        'resolved_valid_jobs': sum(r['status'] == 'VALID' for r in inventory['resolved']),
        'resolved_failed_jobs': len(failures), 'not_supported_configurations': len(unsupported),
        'note': 'Registered job denominator and historical/versioned attempts are distinct; failures are not removed.'}
    tables = {
        'paper_results_tau_010.csv': ([_flat_selection(r, context['frozen'][r['selection_id']]) for r in primary], fields),
        'paper_results_auxiliary.csv': ([_flat_selection(r, context['frozen'][r['selection_id']]) for r in auxiliary], fields),
        'paper_paired_contrasts.csv': (paired_rows, paired_fields),
        'not_supported_conditions.csv': (unsupported, ['candidate_id', 'arm_id', 'backbone', 'method', 'status', 'reason']),
        'paper_failures.csv': (failures, failure_fields),
        'registered_job_status_counts.csv': (status_counts, ['method', 'method_version', 'role', 'status', 'jobs']),
        'attempt_status_counts.csv': (attempt_counts, ['method', 'method_version', 'role', 'status', 'jobs']),
    }
    out.mkdir(parents=True)
    for filename, (values, columns) in tables.items():
        _csv(out / filename, values, columns)
    methods = (
        'We used the explicitly admitted, hash-bound study, selection and runtime versions. '
        'The registered job denominator includes every resolved seed job, including failures; '
        'historical and recovery attempts are counted separately. No model was refitted or selected on T. '
        'Primary operating-point tables use tau=0.10 in the registered-budget main study; '
        'fixed ablations and budget sensitivity retain separate frozen identities. '
        'Metrics are means across the complete frozen seed set. Seed SD describes training variation; '
        'Taylor design SE and simultaneous EO projection intervals describe fixed-policy survey uncertainty. '
        'Risk metrics use event probability p only: q-only methods retain missing risk values, '
        'with untouched-base p shown separately. No-feasible, unsupported and failed conditions are retained.'
    )
    draft = ('# NHIS FairBias application results draft\n\n'
        'Aggregate-only export; source integrity is checked, and no individual data, model or prediction array is loaded. '
        'This draft does not assert a FairBias advantage or authorize further evaluation.\n\n'
        f"Study branch: `{context['study']['study_branch']}`. Registered jobs: {denominators['registered_jobs']}; "
        f"catalog attempts: {denominators['catalog_attempts']}; resolved failures: {denominators['resolved_failed_jobs']}.\n\n"
        '## Methods draft\n\n' + methods + '\n\n'
        'The main, auxiliary, paired, unsupported and failure tables preserve their frozen source/version identities.\n')
    (out / 'paper_results_draft.md').write_text(draft, encoding='utf-8')
    outputs = {path.name: context['files'].sha(path) for path in sorted(out.iterdir())}
    manifest = {'schema_version': REPORT_SCHEMA, 'status': 'PAPER_EXPORT_BUILT_REVIEW_REQUIRED',
        'evaluation_authorized': False, 'study_branch': context['study']['study_branch'],
        'study_freeze_sha256': context['evaluation']['study_freeze_sha256'],
        'selection_sha256': context['evaluation']['selection_sha256'],
        'input_hashes': context['hashes'], 'analysis_files': context['study']['analysis_files'],
        'analysis_root': context['analysis_root'], 'selected_models': context['study']['selected_models'],
        'supplemental_evidence': context['study'].get('supplemental_evidence', []),
        'denominators': denominators, 'resolved_job_status_counts': status_counts, 'attempt_status_counts': attempt_counts,
        'primary_rows': len(primary), 'auxiliary_rows': len(auxiliary), 'paired_rows': len(paired),
        'output_files': outputs}
    (out / 'report_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return {'markdown': out / 'paper_results_draft.md', 'csv': out / 'paper_results_tau_010.csv',
            'manifest': out / 'report_manifest.json'}


def write_catalog_reporting(study_freeze_path, selection_path, evaluation_manifest_path, summary_t_path,
                            output_dir, *, expected_study_sha256, expected_selection_sha256,
                            expected_summary_sha256, analysis_root=None, archive=None):
    """Export one complete, explicitly hash-bound study to a fresh directory.

    This is a provenance-preserving formatter, not an evaluator. Model hashes
    are matched across the independently frozen study, catalog and evaluation
    manifest; model bytes and per-person prediction arrays are never opened.
    """
    try:
        _require(not Path(output_dir).exists(), 'OUTPUT_ALREADY_EXISTS')
        if archive is not None:
            archive.protect_output(output_dir)
        context = _validate(study_freeze_path, selection_path, evaluation_manifest_path, summary_t_path,
            expected_study_sha256, expected_selection_sha256, expected_summary_sha256, analysis_root,
            catalog=catalog if archive is None else archive)
        return _write(context, output_dir)
    except ReportingError:
        raise
    except catalog.CatalogError as exc:
        raise ReportingError(str(exc)) from None
    except (KeyError, TypeError, ValueError, OSError, OverflowError):
        raise ReportingError('MALFORMED_REPORTING_EVIDENCE') from None
