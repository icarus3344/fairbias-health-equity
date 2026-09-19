"""Generated metadata only; real catalog receipts and registered selection rules."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from nhis_fairbias.benchmark import catalog_selection as selection
from nhis_fairbias.benchmark import result_catalog as catalog
from nhis_fairbias.benchmark.experiment_selection import aggregate_configuration, select_aggregate, OPERATING_POINTS
from test_result_catalog import write, sha, seal


def _ref(path, root):
    return {'root': 'evidence', 'path': str(Path(path).relative_to(root))}


def _bound(path, root):
    return {'path': _ref(path, root), 'sha256': sha(path)}


def _refresh(case):
    write(case['manifest_path'], case['manifest'])
    output = catalog.build_catalog(case['manifest_path'])
    assert output['integrity_passed'], output['issues']
    write(case['catalog_path'], output)
    admission = case['admission']
    admission['catalog'] = _bound(case['catalog_path'], case['root'])
    admission['catalog_manifest'] = _bound(case['manifest_path'], case['root'])
    admission['registration']['sha256'] = sha(case['registration_path'])
    for method, policy in admission['methods'].items():
        if 'compatibility_gate' in policy:
            gate = {'schema_version': selection.EG_GATE_SCHEMA,
                'decision': 'ADMITTED_EG_NUMERICAL_COMPATIBILITY',
                'catalog_sha256': admission['catalog']['sha256'],
                'registration_sha256': admission['registration']['sha256'],
                'method': method, 'output_type': 'decision_probability_q',
                'registered_budgets_unchanged': True, 'versions': policy['versions'],
                'resolved_jobs': {name: run for name, run in admission['resolved_jobs'].items()
                    if case['job_configs'][name]['method'] == method},
                'evidence': [_bound(case['gate_evidence'], case['root'])]}
            write(case['gate_path'], gate)
            policy['compatibility_gate'] = _bound(case['gate_path'], case['root'])
    write(case['admission_path'], admission)
    return output


@pytest.fixture
def case(tmp_path):
    root = tmp_path
    old = root / 'archive/original'
    old.mkdir(parents=True)
    source = root / 'src/registered.py'
    source.parent.mkdir()
    source.write_text('# generated registered source\n')
    source_files = {'src/registered.py': sha(source)}
    source_identity = catalog._identity(source_files)
    data = root / 'archive/original/prepared/arm.joblib'
    data.parent.mkdir()
    data.write_bytes(b'opaque generated data, no model/data deserialization')
    remote = '/recorded/checkout'
    execution = {'fit_seconds': 1800.0, 'worker_rss_bytes': 4 * 1024**3}
    registered_budget = {'name': 'registered_v1', 'execution': execution}
    configs, record_values = [], {}
    specs = [
        ('bm_failed', 'FAIRBIAS_BM', 1, [(.99, .01, .9), None]),
        ('bm_winner', 'FAIRBIAS_BM', 1, [(.70, .04, .7), (.80, .08, .8)]),
        ('bm_strict', 'FAIRBIAS_BM', 1, [(.70, .02, .7), (.70, .02, .7)]),
        ('bm_tie', 'FAIRBIAS_BM', 2, [(.75, .06, .7), (.75, .06, .7)]),
        ('risk_ba', 'UNMITIGATED', 1, [(.85, .15, .65)]),
        ('risk_ap', 'UNMITIGATED', 1, [(.60, .03, .80)]),
        ('eg', 'EG_DP', 1, [None, (.70, .03, None)]),
    ]
    jobs, job_configs = {}, {}
    for name, method, complexity, values in specs:
        config = {'candidate_id': name, 'method': method, 'arm_id': 'arm', 'backbone': 'LR',
                  'status': 'REGISTERED', 'params': {}, 'training_weighted': False,
                  'complexity': complexity, 'seeds': list(range(len(values)))}
        configs.append(config)
        for seed, value in enumerate(values):
            jid = f'{name}_s{seed}'
            directory = old / 'jobs' / jid
            job = {'config': config, 'seed': seed, 'source_identity': source_identity,
                   'data_path': remote + '/archive/original/prepared/arm.joblib',
                   'data_identity': 'generated_data', 'data_sha256': sha(data),
                   'cache_path': remote + '/archive/original/representation_cache'}
            write(directory / 'job.json', job)
            result = {'candidate_id': name, 'seed': seed, 'status': 'VALID' if value else 'FAILED',
                'source_identity': source_identity, 'data_identity': 'generated_data',
                'reload_verified': bool(value),
                'output_type': 'decision_probability_q' if method == 'EG_DP' else 'event_probability_p',
                'risk_T': {'never_decode_T': 11111.2345}}
            if value:
                result.update(metrics_S={'balanced_accuracy': value[0], 'eo_gap': value[1], 'dp_gap': .1},
                              risk_S={'average_precision': value[2]})
                (directory / 'model.joblib').write_bytes(('opaque generated model ' + jid).encode())
                (directory / 'predictions_S.npz').write_bytes(b'opaque S bytes, never deserialize')
            write(directory / 'result.json', result)
            (directory / 'worker.log').write_text('generated worker log\n')
            seal(directory, parallel=False, status=result['status'])
            jobs[jid], job_configs[jid], record_values[jid] = directory, config, result
    configs.append({'candidate_id': 'lfr_unsupported', 'method': 'LFR_RECONSTRUCTED',
        'arm_id': 'arm_002', 'backbone': 'LR', 'status': 'NOT_SUPPORTED', 'seeds': [],
        'reason': 'official LFR is binary-group only'})
    registration = {'version': 'codex_application_v1_20260916', 'source_identity': source_identity,
        'source_files': source_files, 'candidates': configs, 'pending_predeclared_extensions': [],
        'prepared': {'arm': {'path': remote + '/archive/original/prepared/arm.joblib',
                     'data_identity': 'generated_data', 'sha256': sha(data)}},
        'resources': {**execution, 'threads': 1, 'concurrent_fits': 1}}
    regpath = write(old / 'registration.json', registration)
    # One repaired EG failure, using the deployed numerical receipt schema.
    runtime_files = dict(source_files)
    for name in catalog.RUNTIME_REQUIRED['numerical_recovery_v1']:
        generated = root / name
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text('# generated runtime source\n')
        runtime_files[name] = sha(generated)
    runtime = {'variant': catalog.NUMERICAL_VARIANT, 'source_files': runtime_files,
               'environment': dict(catalog.NUMERICAL_ENVIRONMENT)}
    runtime['source_identity'] = catalog._identity({'variant': runtime['variant'], 'files': runtime_files,
                                                   'environment': runtime['environment']})
    new = root / ('recovery_' + runtime['source_identity'])
    directory = new / 'jobs/eg_s0'
    original = jobs['eg_s0']
    job = json.loads((original / 'job.json').read_text())
    job.update(runtime_variant=runtime['variant'], runtime_source_identity=runtime['source_identity'],
        runtime_source_files=runtime_files, runtime_environment=runtime['environment'],
        registration_path=remote + '/archive/original/registration.json', registration_sha256=sha(regpath),
        original_job_path=remote + '/archive/original/jobs/eg_s0/job.json',
        original_job_sha256=sha(original / 'job.json'), original_result_sha256=sha(original / 'result.json'),
        recovery_run_path=remote + '/' + new.name, cache_path=remote + '/' + new.name + '/representation_cache',
        parallel_scheduler_version=catalog.SCHEDULER, representation_key=None)
    write(directory / 'job.json', job)
    result = dict(record_values['eg_s0'], status='VALID', reload_verified=True,
                  metrics_S={'balanced_accuracy': .8, 'eo_gap': .05, 'dp_gap': .1}, risk_S={})
    write(directory / 'result.json', result)
    (directory / 'model.joblib').write_bytes(b'opaque repaired EG model')
    (directory / 'predictions_S.npz').write_bytes(b'opaque generated q')
    (directory / 'worker.log').write_text('generated worker log')
    write(directory / 'runtime_started.json', {'runtime_variant': runtime['variant'],
        'runtime_source_identity': runtime['source_identity'], 'job_sha256': sha(directory / 'job.json'), 'pid': 1})
    receipt = {key: job[key] for key in ('runtime_variant', 'runtime_source_identity', 'runtime_source_files',
        'runtime_environment', 'registration_sha256', 'original_job_sha256', 'original_result_sha256',
        'data_identity', 'data_sha256')}
    receipt.update(schema_version='nhis_numerical_recovery_receipt_v1', candidate_id='eg', seed=0,
        status='VALID', worker_result_status='VALID', original_status='FAILED', runtime_checks_passed=True,
        legacy_valid_results_admitted=False, representation_key=None, cache_path=job['cache_path'],
        loaded_project_modules_before={'nhis_fairbias.example': {'source': 'src/registered.py', 'sha256': sha(source)}},
        loaded_project_modules_after={'nhis_fairbias.example': {'source': 'src/registered.py', 'sha256': sha(source)}},
        files={p.name: sha(p) for p in directory.iterdir() if p.name != 'worker.log'})
    write(directory / 'runtime_receipt.json', receipt)
    seal(directory)
    original_run = {'id': 'original', 'registration_id': 'registration', 'directory': _ref(old, root),
        'method_version': 'registered_v1', 'role': 'original', 'profile': 'serial_v1',
        'lifecycle': 'closed', 'expected_jobs': list(jobs), 'budget_policy': registered_budget}
    numerical_run = {'id': 'numerical', 'registration_id': 'registration', 'directory': _ref(new, root),
        'method_version': 'eg_endpoint_v1', 'role': 'replacement', 'profile': 'numerical_recovery_v1',
        'lifecycle': 'closed', 'expected_jobs': ['eg_s0'], 'runtime': runtime,
        'budget_policy': {'name': 'numerical_same_budgets_v1', 'execution': execution}}
    manifest = {'schema_version': catalog.MANIFEST_SCHEMA, 'evaluation_authorized': False,
        'roots': {'evidence': str(root)},
        'relocations': [{'recorded_prefix': remote, 'target': {'root': 'evidence', 'path': '.'}}],
        'registrations': [{'id': 'registration', 'path': _ref(regpath, root), 'sha256': sha(regpath),
                           'source_root': {'root': 'evidence', 'path': '.'}}],
        'runs': [original_run, numerical_run]}
    version = lambda run: {'method_version': run['method_version'], 'role': run['role'],
        'budget_policy': run['budget_policy'], 'runtime_source_identity': run.get('runtime', {}).get('source_identity')}
    admission = {'schema_version': selection.ADMISSION_SCHEMA, 'decision': 'SUPERVISOR_ADMITTED',
        'evaluation_authorized': False, 'roots': {'evidence': str(root)},
        'catalog': {}, 'catalog_manifest': {}, 'registration': {'id': 'registration', 'sha256': sha(regpath)},
        'methods': {method: {'analysis_version': 'registered_v1', 'versions': [version(original_run)]}
                    for method in ('FAIRBIAS_BM', 'UNMITIGATED', 'EG_DP')},
        'resolved_jobs': {name: 'original' for name in jobs}, 'extension_resolutions': {}}
    admission['resolved_jobs']['eg_s0'] = 'numerical'
    admission['methods']['EG_DP'] = {'analysis_version': 'eg_supervisor_compatible_v1',
        'versions': [version(original_run), version(numerical_run)], 'compatibility_gate': {}}
    case = dict(root=root, manifest=manifest, manifest_path=root / 'manifest.json',
        catalog_path=root / 'catalog.json', admission=admission, admission_path=root / 'admission.json',
        registration=registration, registration_path=regpath, jobs=jobs, job_configs=job_configs,
        new_job=directory, records=record_values, source=source, gate_path=root / 'eg_gate.json',
        gate_evidence=write(root / 'compatibility_review.json', {'synthetic_supervisor_evidence': True}))
    _refresh(case)
    return case


def _deny_result_metric_decode(monkeypatch):
    original = catalog.read_metadata
    seen = []
    def reader(path, fields=None):
        if Path(path).name == 'result.json':
            assert fields is not None and not {'metrics_S', 'risk_S'} & fields
            seen.append(str(path))
        return original(path, fields)
    monkeypatch.setattr(catalog, 'read_metadata', reader)
    return seen


def test_metric_blind_admission_and_exact_registered_selection(case, monkeypatch):
    with monkeypatch.context() as guard:
        seen = _deny_result_metric_decode(guard)
        proof = selection.validate_selection_admission(case['admission_path'])
        assert seen and proof['registered_jobs'] == 12
        assert not proof['evaluation_authorized'] and not proof['metrics_decoded']
    artifact = selection.freeze_catalog_selection(case['admission_path'])
    assert not artifact['evaluation_authorized'] and not artifact['models_deserialized']
    assert artifact['operating_points'] == list(OPERATING_POINTS)
    bm = {s['tau']: s for s in artifact['selections'] if s['method'] == 'FAIRBIAS_BM'}
    assert bm[.10]['selected_candidate_id'] == 'bm_winner'
    assert bm[.05]['selected_candidate_id'] == 'bm_strict'
    assert bm[.20]['selected_candidate_id'] == 'bm_winner'
    summaries = {s.get('candidate_id', s.get('config', {}).get('candidate_id')): s for s in artifact['candidate_summaries']}
    assert summaries['bm_failed']['status'] == 'INCOMPLETE_CONFIGURATION'
    assert summaries['bm_failed']['seed_statuses'] == {'0': 'VALID', '1': 'FAILED'}
    assert summaries['lfr_unsupported'] == case['registration']['candidates'][-1]
    for config in case['registration']['candidates'][:4]:
        expected = aggregate_configuration(config, [case['records'][f"{config['candidate_id']}_s{s}"] for s in config['seeds']])
        assert summaries[config['candidate_id']] == {'config': config, **expected}
    assert [select_aggregate([summaries[n] for n in ('bm_failed', 'bm_winner', 'bm_strict', 'bm_tie')], tau)
            for tau in OPERATING_POINTS] == [{k: v for k, v in bm[tau].items() if k in
                {'status', 'selected_candidate_id', 'boundary_candidate_id', 'tau'}} for tau in OPERATING_POINTS]
    references = [s for s in artifact['selections'] if s['status'] == 'PREDICTION_REFERENCE']
    assert references[0]['selected_candidate_id'] == 'risk_ap'
    assert not any(r['candidate_id'] == 'bm_failed' for r in artifact['selected_models'])
    assert 'never_decode_T' not in json.dumps(artifact)


def test_resolver_keeps_archive_real_paths_full_runtime_and_no_deserialization(case, monkeypatch):
    import joblib
    import numpy as np
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('must not deserialize models'))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('must not deserialize S predictions'))
    artifact = selection.freeze_catalog_selection(case['admission_path'])
    path = write(case['root'] / 'selection.json', artifact)
    _deny_result_metric_decode(monkeypatch)
    resolved = selection.resolve_selected_models(path, sha(path))
    assert resolved['models'] == artifact['selected_models'] and not resolved['evaluation_authorized']
    eg = next(r for r in resolved['models'] if r['job_id'] == 'eg_s0')
    assert eg['runtime'] == case['manifest']['runs'][1]['runtime']
    assert eg['runtime_job_metadata']['original_result_sha256'] == sha(case['jobs']['eg_s0'] / 'result.json')
    assert eg['artifacts']['model.joblib']['path'] == str(case['new_job'] / 'model.joblib')
    old = next(r for r in resolved['models'] if r['job_id'] == 'eg_s1')
    assert '/archive/original/' in old['artifacts']['model.joblib']['path']
    assert old['output_type'] == eg['output_type'] == 'decision_probability_q'
    assert {r['seed'] for r in resolved['models'] if r['candidate_id'] == 'eg'} == {0, 1}


@pytest.mark.parametrize('fault', ['schema', 'decision', 'evaluation', 'catalog_sha', 'registration_sha',
    'unknown_run', 'missing_seed', 'extra_seed', 'wrong_version', 'wrong_budget', 'wrong_runtime',
    'missing_gate', 'extra_method', 'path_escape', 'absolute_path', 'unknown_root'])
def test_admission_rejects_before_decoding_any_metrics(case, monkeypatch, fault):
    a = case['admission']
    if fault == 'schema': a['schema_version'] = 'unknown'
    elif fault == 'decision': a['decision'] = 'DRAFT'
    elif fault == 'evaluation': a['evaluation_authorized'] = True
    elif fault == 'catalog_sha': a['catalog']['sha256'] = '0'*64
    elif fault == 'registration_sha': a['registration']['sha256'] = '0'*64
    elif fault == 'unknown_run': a['resolved_jobs']['eg_s0'] = 'unknown'
    elif fault == 'missing_seed': del a['resolved_jobs']['bm_winner_s1']
    elif fault == 'extra_seed': a['resolved_jobs']['bm_winner_s9'] = 'original'
    elif fault in {'wrong_version', 'wrong_budget', 'wrong_runtime'}:
        v = a['methods']['FAIRBIAS_BM']['versions'][0]
        if fault == 'wrong_version': v['method_version'] = 'latest_success'
        elif fault == 'wrong_runtime': v['runtime_source_identity'] = '0'*64
        else: v['budget_policy'] = {'name': 'bigger_budget', 'execution': v['budget_policy']['execution']}
    elif fault == 'missing_gate': del a['methods']['EG_DP']['compatibility_gate']
    elif fault == 'extra_method': a['methods']['invented'] = {}
    elif fault == 'path_escape': a['catalog']['path']['path'] = '../catalog.json'
    elif fault == 'absolute_path': a['catalog']['path']['path'] = str(case['catalog_path'])
    elif fault == 'unknown_root': a['catalog']['path']['root'] = 'unknown'
    write(case['admission_path'], a)
    _deny_result_metric_decode(monkeypatch)
    with pytest.raises(selection.SelectionAdmissionError):
        selection.freeze_catalog_selection(case['admission_path'])


@pytest.mark.parametrize('fault', ['model', 'receipt', 'source', 'runtime_receipt', 'data', 'symlink_escape'])
def test_evidence_tamper_rejected_before_metric_read(case, monkeypatch, fault):
    paths = {'model': case['new_job'] / 'model.joblib', 'receipt': case['new_job'] / 'receipt.json',
             'runtime_receipt': case['new_job'] / 'runtime_receipt.json', 'source': case['source'],
             'data': case['registration_path'].parent / 'prepared/arm.joblib'}
    if fault == 'symlink_escape':
        path = case['new_job'] / 'model.joblib'
        outside = case['root'].parent / 'outside-generated-model'
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        path = paths[fault]
        path.write_bytes(path.read_bytes() + b'changed')
    _deny_result_metric_decode(monkeypatch)
    with pytest.raises(selection.SelectionAdmissionError):
        selection.freeze_catalog_selection(case['admission_path'])


@pytest.mark.parametrize('fault', ['pending', 'rejected', 'sensitivity', 'metrics_decoded', 'integrity_false'])
def test_unresolved_or_sensitivity_catalog_cannot_freeze(case, monkeypatch, fault):
    if fault == 'pending':
        (case['new_job'] / 'receipt.json').unlink()
        case['manifest']['runs'][1]['lifecycle'] = 'active'
        _refresh(case)
    elif fault == 'sensitivity':
        case['manifest']['runs'][1]['role'] = 'sensitivity'
        case['admission']['methods']['EG_DP']['versions'][1]['role'] = 'sensitivity'
        _refresh(case)
    else:
        snapshot = json.loads(case['catalog_path'].read_text())
        if fault == 'rejected': snapshot['jobs'][0]['status'] = 'REJECTED'
        elif fault == 'integrity_false': snapshot['integrity_passed'] = False
        else: snapshot['metrics_decoded'] = True
        write(case['catalog_path'], snapshot)
        case['admission']['catalog'] = _bound(case['catalog_path'], case['root'])
        write(case['admission_path'], case['admission'])
    _deny_result_metric_decode(monkeypatch)
    with pytest.raises(selection.SelectionAdmissionError):
        selection.freeze_catalog_selection(case['admission_path'])


@pytest.mark.parametrize('fault', ['audit_not_gate', 'mapping', 'catalog', 'evidence', 'output', 'budget'])
def test_eg_compatibility_is_specific_hash_bound_gate(case, monkeypatch, fault):
    gate = json.loads(case['gate_path'].read_text())
    if fault == 'audit_not_gate': gate['decision'] = 'COMPATIBLE_STORED_S_Q_ONLY'
    elif fault == 'mapping': gate['resolved_jobs']['eg_s0'] = 'original'
    elif fault == 'catalog': gate['catalog_sha256'] = '0'*64
    elif fault == 'evidence': gate['evidence'] = []
    elif fault == 'output': gate['output_type'] = 'event_probability_p'
    else: gate['registered_budgets_unchanged'] = False
    write(case['gate_path'], gate)
    case['admission']['methods']['EG_DP']['compatibility_gate'] = _bound(case['gate_path'], case['root'])
    write(case['admission_path'], case['admission'])
    _deny_result_metric_decode(monkeypatch)
    with pytest.raises(selection.SelectionAdmissionError):
        selection.freeze_catalog_selection(case['admission_path'])


def test_explicit_original_failure_is_preserved_without_success_fallback(case):
    case['admission']['resolved_jobs']['eg_s0'] = 'original'
    case['admission']['methods']['EG_DP'] = {'analysis_version': 'original_eg',
        'versions': [case['admission']['methods']['EG_DP']['versions'][0]]}
    write(case['admission_path'], case['admission'])
    artifact = selection.freeze_catalog_selection(case['admission_path'])
    assert artifact['resolved_statuses']['eg_s0'] == 'FAILED'
    assert all(s['status'] == 'NOT_ESTIMABLE' for s in artifact['selections'] if s['method'] == 'EG_DP')
    assert not any(r['method'] == 'EG_DP' for r in artifact['selected_models'])


def test_unresolved_extensions_block_even_with_all_jobs_terminal(case, monkeypatch):
    case['registration']['pending_predeclared_extensions'] = ['future_method']
    write(case['registration_path'], case['registration'])
    # Updating a registration would invalidate runtime binding. Use the original
    # only run to exercise this independent extension contract with honest hashes.
    case['manifest']['runs'] = case['manifest']['runs'][:1]
    case['manifest']['registrations'][0]['sha256'] = sha(case['registration_path'])
    case['admission']['resolved_jobs']['eg_s0'] = 'original'
    case['admission']['methods']['EG_DP'] = {'analysis_version': 'original_eg',
        'versions': [case['admission']['methods']['EG_DP']['versions'][0]]}
    _refresh(case)
    _deny_result_metric_decode(monkeypatch)
    with pytest.raises(selection.SelectionAdmissionError, match='UNRESOLVED_PREDECLARED_EXTENSIONS'):
        selection.validate_selection_admission(case['admission_path'])
    case['admission']['extension_resolutions'] = {'future_method': {'status': 'EXCLUDED_UNSUPPORTED',
                                                                 'reason': 'Synthetic predeclared exclusion'}}
    write(case['admission_path'], case['admission'])
    assert selection.validate_selection_admission(case['admission_path'])['status'] == 'ADMISSION_VERIFIED'


@pytest.mark.parametrize('value', [True, [], '0.9', 1.01, -0.1])
def test_invalid_metric_shape_and_range_rejected(case, value):
    directory = case['jobs']['bm_winner_s0']
    result = json.loads((directory / 'result.json').read_text())
    result['metrics_S']['balanced_accuracy'] = value
    write(directory / 'result.json', result)
    seal(directory, parallel=False)
    _refresh(case)
    with pytest.raises(selection.SelectionAdmissionError, match='INVALID_SELECTION_METRIC'):
        selection.freeze_catalog_selection(case['admission_path'])


@pytest.mark.parametrize('fault', ['hash', 'path', 'missing_seed', 'runtime', 'source',
    'missing_condition', 'wrong_tau', 'wrong_condition', 'unknown_status', 'wrong_selected_state', 'wrong_version'])
def test_resolver_rejects_corrupt_or_mismatched_artifact(case, fault):
    artifact = selection.freeze_catalog_selection(case['admission_path'])
    if fault == 'path': artifact['selected_models'][0]['artifacts']['model.joblib']['path'] = '/outside/model.joblib'
    elif fault == 'missing_seed': artifact['selected_models'].pop()
    elif fault == 'runtime': artifact['selected_models'][0]['runtime'] = {'invented': True}
    elif fault == 'source': artifact['analysis_source_files']['catalog_selection.py'] = '0'*64
    elif fault == 'missing_condition': artifact['selections'].pop()
    elif fault == 'wrong_tau': artifact['selections'][0]['tau'] = .123
    elif fault == 'wrong_condition': artifact['selections'][0]['selected_candidate_id'] = 'risk_ap'
    elif fault == 'unknown_status': artifact['selections'][0]['status'] = 'REVIEW_PENDING'
    elif fault == 'wrong_selected_state': artifact['selections'][0]['selected_candidate_id'] = None
    elif fault == 'wrong_version': artifact['selections'][0]['analysis_version'] = 'latest'
    path = write(case['root'] / 'selection.json', artifact)
    with pytest.raises(selection.SelectionAdmissionError):
        selection.resolve_selected_models(path, '0'*64 if fault == 'hash' else sha(path))


def test_no_feasible_condition_keeps_registered_boundary_models(case):
    for name, directory in case['jobs'].items():
        if not name.startswith('bm_'):
            continue
        result = json.loads((directory / 'result.json').read_text())
        if result['status'] == 'VALID':
            result['metrics_S']['eo_gap'] = .3 if name.startswith('bm_strict') else .4
            write(directory / 'result.json', result)
            seal(directory, parallel=False)
    _refresh(case)
    artifact = selection.freeze_catalog_selection(case['admission_path'])
    bm = [s for s in artifact['selections'] if s['method'] == 'FAIRBIAS_BM']
    assert all(s['status'] == 'NO_FEASIBLE_CONFIGURATION' and s['selected_candidate_id'] is None
               and s['boundary_candidate_id'] == 'bm_strict' for s in bm)
    models = [r for r in artifact['selected_models'] if r['method'] == 'FAIRBIAS_BM']
    assert {r['job_id'] for r in models} == {'bm_strict_s0', 'bm_strict_s1'}
    path = write(case['root'] / 'selection.json', artifact)
    assert selection.resolve_selected_models(path, sha(path))['models'] == artifact['selected_models']


def test_non_eg_versions_cannot_mix_seeds_even_with_gate_field(case, monkeypatch):
    new_run = copy.deepcopy(case['manifest']['runs'][0])
    new_directory = case['root'] / 'another_registered_variant'
    shutil.copytree(case['jobs']['bm_winner_s1'], new_directory / 'jobs/bm_winner_s1')
    new_run.update(id='another', directory=_ref(new_directory, case['root']),
                   method_version='another_version', role='replacement', expected_jobs=['bm_winner_s1'])
    case['manifest']['runs'].append(new_run)
    case['admission']['resolved_jobs']['bm_winner_s1'] = 'another'
    policy = case['admission']['methods']['FAIRBIAS_BM']
    policy['versions'].append({'method_version': 'another_version', 'role': 'replacement',
        'budget_policy': new_run['budget_policy'], 'runtime_source_identity': None})
    # The presence of a gate does not turn arbitrary methods into compatible ones.
    _refresh(case)
    policy['compatibility_gate'] = _bound(case['gate_path'], case['root'])
    write(case['admission_path'], case['admission'])
    _deny_result_metric_decode(monkeypatch)
    with pytest.raises(selection.SelectionAdmissionError, match='CROSS_VERSION_METHOD_FORBIDDEN'):
        selection.freeze_catalog_selection(case['admission_path'])


def test_candidate_id_is_final_tie_break(case):
    cfg = next(c for c in case['registration']['candidates'] if c['candidate_id'] == 'bm_tie')
    cfg['complexity'] = 1
    # Registration and job config equality is still enforced for this generated
    # change; remove the independent numerical run whose original hashes bind it.
    case['manifest']['runs'] = case['manifest']['runs'][:1]
    case['admission']['resolved_jobs']['eg_s0'] = 'original'
    case['admission']['methods']['EG_DP'] = {'analysis_version': 'original_eg',
        'versions': [case['admission']['methods']['EG_DP']['versions'][0]]}
    write(case['registration_path'], case['registration'])
    case['manifest']['registrations'][0]['sha256'] = sha(case['registration_path'])
    for seed in cfg['seeds']:
        directory = case['jobs'][f'bm_tie_s{seed}']
        job = json.loads((directory / 'job.json').read_text())
        job['config'] = cfg
        write(directory / 'job.json', job)
        seal(directory, parallel=False)
    _refresh(case)
    artifact = selection.freeze_catalog_selection(case['admission_path'])
    assert next(s for s in artifact['selections'] if s['method'] == 'FAIRBIAS_BM' and s['tau'] == .1)[
        'selected_candidate_id'] == 'bm_tie'


@pytest.mark.parametrize('method', ['FAIRBIAS_JOINT', 'FAIRBIAS_BM_AE'])
def test_main_analysis_retains_original_ablation_and_rejects_replacement(case, method, monkeypatch):
    case['manifest']['runs'] = case['manifest']['runs'][:1]
    case['admission']['resolved_jobs']['eg_s0'] = 'original'
    case['admission']['methods']['EG_DP'] = {'analysis_version': 'original_eg',
        'versions': [case['admission']['methods']['EG_DP']['versions'][0]]}
    case['admission']['methods'][method] = case['admission']['methods'].pop('FAIRBIAS_BM')
    for config in case['registration']['candidates']:
        if config['method'] != 'FAIRBIAS_BM':
            continue
        config['method'] = method
        for seed in config['seeds']:
            directory = case['jobs'][f"{config['candidate_id']}_s{seed}"]
            job = json.loads((directory / 'job.json').read_text())
            job['config'] = config
            write(directory / 'job.json', job)
            result = json.loads((directory / 'result.json').read_text())
            seal(directory, parallel=False, status=result['status'])
    write(case['registration_path'], case['registration'])
    case['manifest']['registrations'][0]['sha256'] = sha(case['registration_path'])
    _refresh(case)
    _deny_result_metric_decode(monkeypatch)
    assert selection.validate_selection_admission(case['admission_path'])['registered_jobs'] == 12
    # Even a same-execution-budget variant cannot relabel an ablation recovery
    # into this original-budget main analysis.
    case['manifest']['runs'][0]['role'] = 'replacement'
    for policy in case['admission']['methods'].values():
        policy['versions'][0]['role'] = 'replacement'
    _refresh(case)
    with pytest.raises(selection.SelectionAdmissionError, match='ABLATION_RECOVERY_REQUIRES_SEPARATE_STUDY'):
        selection.freeze_catalog_selection(case['admission_path'])
