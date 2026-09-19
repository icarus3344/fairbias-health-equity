"""Synthetic metadata matching deployed serial, parallel and runtime schemas."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nhis_fairbias.benchmark import result_catalog as catalog


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def mutate(path, change):
    value = json.loads(Path(path).read_text())
    change(value)
    write(path, value)


def seal(directory, parallel=True, status='VALID', termination=None):
    receipt = {'returncode': 0 if status == 'VALID' else (1 if termination is None else -15),
               'termination': termination, 'observed_peak_rss_bytes': 1024,
               'files': {p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.name != 'receipt.json'}}
    if parallel:
        receipt.update(scheduler_version=catalog.SCHEDULER, session_id='synthetic_session')
    write(directory / 'receipt.json', receipt)


@pytest.fixture
def case(tmp_path):
    # Physical archive paths intentionally differ from immutable recorded paths.
    root = tmp_path / 'archive'
    source = root / 'src/registered.py'
    source.parent.mkdir(parents=True)
    source.write_text('# generated source witness\n')
    extra = root / 'scripts/runtime.py'
    extra.parent.mkdir()
    extra.write_text('# generated runtime witness\n')
    for name in set().union(*catalog.RUNTIME_REQUIRED.values()):
        generated = root / name
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text('# generated source for schema test\n')
    recorded = '/remote/original_checkout'
    original = root / 'artifacts/original'
    data = original / 'prepared/arm.joblib'
    data.parent.mkdir(parents=True)
    data.write_bytes(b'generated opaque prepared bytes, never deserialize')
    source_files = {'src/registered.py': sha(source)}
    source_id = catalog._identity(source_files)
    configs = []
    for method, seeds in [('UNMITIGATED', [0]), ('EG_DP', [0, 1, 2])]:
        cfg = {'method': method, 'arm_id': 'arm', 'backbone': 'LR', 'params': {},
               'status': 'REGISTERED', 'seeds': seeds, 'training_weighted': False}
        cfg['candidate_id'] = catalog._identity(cfg)[:20]
        configs.append(cfg)
    budget = {'fit_seconds': 1800., 'worker_rss_bytes': 4*1024**3}
    reg = {'version': 'codex_application_v1_20260916', 'source_files': source_files,
           'source_identity': source_id, 'candidates': configs,
           'prepared': {'arm': {'path': recorded + '/artifacts/original/prepared/arm.joblib',
                               'sha256': sha(data), 'data_identity': 'generated_data_id'}},
           'resources': {**budget, 'threads': 1, 'concurrent_fits': 1}}
    regpath = write(original / 'registration.json', reg)
    originals = {}
    for cfg in configs:
        for seed in cfg['seeds']:
            if seed == 2:
                continue
            name = cfg['candidate_id'] + '_s' + str(seed)
            directory = original / 'jobs' / name
            job = {'config': cfg, 'seed': seed, 'source_identity': source_id,
                   'data_path': reg['prepared']['arm']['path'], 'data_identity': 'generated_data_id',
                   'data_sha256': sha(data), 'cache_path': recorded + '/artifacts/original/representation_cache'}
            parallel = cfg['method'] != 'UNMITIGATED'
            if parallel:
                job.update(parallel_scheduler_version=catalog.SCHEDULER, representation_key=None)
            write(directory / 'job.json', job)
            status = 'VALID' if cfg['method'] == 'UNMITIGATED' else 'FAILED'
            result = {'candidate_id': cfg['candidate_id'], 'seed': seed, 'status': status,
                      'source_identity': source_id, 'data_identity': 'generated_data_id',
                      'output_type': 'event_probability_p', 'reload_verified': status == 'VALID',
                      'metrics_S': {'forbidden_S_secret': [999.123]}, 'risk_T': {'forbidden_T_secret': 777.456}}
            write(directory / 'result.json', result)
            (directory / 'worker.log').write_text('generated log')
            if status == 'VALID':
                (directory / 'model.joblib').write_bytes(b'opaque generated model')
                (directory / 'predictions_S.npz').write_bytes(b'opaque generated predictions')
            seal(directory, parallel, status)
            originals[name] = directory
    runtime_files = {**source_files, 'scripts/runtime.py': sha(extra),
                     **{name: sha(root / name) for name in catalog.RUNTIME_REQUIRED['numerical_recovery_v1']}}
    env = dict(catalog.NUMERICAL_ENVIRONMENT)
    rid = catalog._identity({'variant': catalog.NUMERICAL_VARIANT, 'files': runtime_files, 'environment': env})
    run_dir = root / 'artifacts' / ('numerical_' + rid)
    runtime = {'variant': catalog.NUMERICAL_VARIANT, 'source_identity': rid,
               'source_files': runtime_files, 'environment': env}
    new_jobs = []
    cfg = configs[1]
    for seed in (0, 1):
        name = cfg['candidate_id'] + '_s' + str(seed)
        old = originals[name]
        directory = run_dir / 'jobs' / name
        job = json.loads((old / 'job.json').read_text())
        job.update(runtime_variant=runtime['variant'], runtime_source_identity=rid,
            runtime_source_files=runtime_files, runtime_environment=env,
            recovery_run_path=recorded + '/artifacts/' + run_dir.name,
            cache_path=recorded + '/artifacts/' + run_dir.name + '/representation_cache',
            registration_path=recorded + '/artifacts/original/registration.json', registration_sha256=sha(regpath),
            original_job_path=recorded + '/artifacts/original/jobs/' + name + '/job.json',
            original_job_sha256=sha(old / 'job.json'), original_result_sha256=sha(old / 'result.json'))
        write(directory / 'job.json', job)
        if seed == 0:
            write(directory / 'result.json', {'candidate_id': cfg['candidate_id'], 'seed': seed, 'status': 'VALID',
                'source_identity': source_id, 'data_identity': 'generated_data_id', 'reload_verified': True,
                'output_type': 'decision_probability_q', 'metrics_S': {'secret': [888.125]}})
            (directory / 'worker.log').write_text('generated runtime log')
            (directory / 'model.joblib').write_bytes(b'new opaque model')
            (directory / 'predictions_S.npz').write_bytes(b'new opaque predictions')
            write(directory / 'runtime_started.json', {'runtime_variant': runtime['variant'], 'runtime_source_identity': rid,
                  'job_sha256': sha(directory / 'job.json'), 'pid': 1})
            rr = {k: job[k] for k in ('runtime_variant', 'runtime_source_identity', 'runtime_source_files',
                'runtime_environment', 'registration_sha256', 'original_job_sha256', 'original_result_sha256',
                'data_identity', 'data_sha256')}
            rr.update(schema_version='nhis_numerical_recovery_receipt_v1', candidate_id=cfg['candidate_id'], seed=seed,
                status='VALID', worker_result_status='VALID', runtime_checks_passed=True, original_status='FAILED',
                legacy_valid_results_admitted=False, representation_key=None, cache_path=job['cache_path'],
                loaded_project_modules_before={'nhis_fairbias.example': {'source': 'src/registered.py', 'sha256': sha(source)}},
                loaded_project_modules_after={'nhis_fairbias.example': {'source': 'src/registered.py', 'sha256': sha(source)}},
                files={p.name: sha(p) for p in directory.iterdir() if p.name != 'worker.log'})
            write(directory / 'runtime_receipt.json', rr)
            seal(directory)
        else:
            # A worker result written before scheduler sealing is still pending.
            write(directory / 'result.json', {'status': 'VALID', 'metrics_S': {'secret': 123456}})
        new_jobs.append(directory)
    ref = lambda path: {'root': 'archive', 'path': str(path.relative_to(root))}
    common = {'registration_id': 'original', 'method_version': 'registered_v1', 'role': 'original',
              'directory': ref(original), 'lifecycle': 'closed', 'budget_policy': {'name': 'registered_v1', 'execution': budget}}
    manifest = {'schema_version': catalog.MANIFEST_SCHEMA, 'evaluation_authorized': False,
        'roots': {'archive': str(root)},
        'relocations': [{'recorded_prefix': recorded, 'target': {'root': 'archive', 'path': '.'}}],
        'registrations': [{'id': 'original', 'path': ref(regpath), 'sha256': sha(regpath),
                           'source_root': {'root': 'archive', 'path': '.'}}],
        'runs': [{**common, 'id': 'old_serial', 'profile': 'serial_v1',
                  'expected_jobs': [configs[0]['candidate_id'] + '_s0']},
                 {**common, 'id': 'old_parallel', 'profile': 'parallel_v1',
                  'expected_jobs': [cfg['candidate_id'] + '_s0', cfg['candidate_id'] + '_s1']},
                 {'id': 'numerical', 'registration_id': 'original', 'directory': ref(run_dir),
                  'method_version': 'eg_endpoint_v1', 'role': 'pending', 'lifecycle': 'active',
                  'profile': 'numerical_recovery_v1', 'runtime': runtime,
                  'expected_jobs': [p.name for p in new_jobs],
                  'budget_policy': {'name': 'numerical_same_budgets_v1', 'execution': budget}}]}
    mpath = write(tmp_path / 'manifest.json', manifest)
    return dict(root=root, manifest=manifest, path=mpath, original=original, originals=originals,
                new_jobs=new_jobs, regpath=regpath, source=source, config=cfg, runtime=runtime)


def test_multirun_archive_catalog_retains_failure_pending_and_registration_gaps(case):
    output = catalog.build_catalog(case['path'])
    assert output['integrity_passed'], output['issues']
    assert len(output['jobs']) == 5
    assert [r['status'] for r in output['jobs']] == ['VALID', 'FAILED', 'FAILED', 'VALID', 'PENDING']
    assert output['jobs'][-1]['pending_reason'] == 'NO_TERMINAL_RECEIPT'
    assert output['jobs'][3]['original_evidence']['original_status'] == 'FAILED'
    assert not output['evaluation_authorized'] and not output['selection_performed']
    assert not output['metrics_decoded'] and not output['models_deserialized']
    text = json.dumps(output)
    assert 'forbidden_S_secret' not in text and '999.123' not in text and '888.125' not in text
    coverage = output['registration_coverage'][0]
    assert coverage['registered_jobs'] == 4 and coverage['declared_jobs'] == 3
    assert coverage['original_undeclared_jobs'] == [case['config']['candidate_id'] + '_s2']
    assert output['jobs'][3]['output_type'] == 'decision_probability_q'
    assert output['jobs'][0]['output_type'] == 'event_probability_p'


def test_metric_values_are_skipped_without_json_decoding(case, monkeypatch):
    decoder = catalog.json.loads
    def guarded(text, *args, **kwargs):
        assert 'forbidden_S_secret' not in text and 'forbidden_T_secret' not in text
        assert '888.125' not in text and '123456' not in text
        return decoder(text, *args, **kwargs)
    monkeypatch.setattr(catalog.json, 'loads', guarded)
    assert catalog.build_catalog(case['path'])['integrity_passed']


@pytest.mark.parametrize('fault', ['model_hash', 'missing_model_hash', 'missing_result_hash', 'wrong_candidate',
    'wrong_data', 'wrong_source', 'scheduler', 'runtime_schema', 'runtime_failed', 'runtime_source',
    'missing_runtime', 'absolute_artifact', 'parent_artifact', 'p_q', 'original_hash', 'job_runtime'])
def test_bad_terminal_evidence_is_rejected_not_promoted(case, fault):
    directory = case['new_jobs'][0]
    result, receipt, runtime, job = [directory / n for n in ('result.json', 'receipt.json', 'runtime_receipt.json', 'job.json')]
    if fault == 'model_hash':
        (directory / 'model.joblib').write_bytes(b'tampered')
    elif fault == 'missing_model_hash':
        mutate(receipt, lambda x: x['files'].pop('model.joblib'))
    elif fault == 'missing_result_hash':
        mutate(receipt, lambda x: x['files'].pop('result.json'))
    elif fault in ('wrong_candidate', 'wrong_data', 'wrong_source', 'p_q'):
        field = {'wrong_candidate': 'candidate_id', 'wrong_data': 'data_identity',
                 'wrong_source': 'source_identity', 'p_q': 'output_type'}[fault]
        mutate(result, lambda x: x.update({field: 'invalid'}))
        mutate(runtime, lambda x: x['files'].update({'result.json': sha(result)}))
        seal(directory)
    elif fault == 'scheduler':
        mutate(receipt, lambda x: x.update(scheduler_version='unknown'))
    elif fault in ('runtime_schema', 'runtime_failed', 'runtime_source'):
        field = {'runtime_schema': 'schema_version', 'runtime_failed': 'status', 'runtime_source': 'runtime_source_identity'}[fault]
        mutate(runtime, lambda x: x.update({field: 'invalid'}))
        seal(directory)
    elif fault == 'missing_runtime':
        runtime.unlink()
        seal(directory)
    elif fault in ('absolute_artifact', 'parent_artifact'):
        mutate(receipt, lambda x: x['files'].update({('/tmp/escape' if fault == 'absolute_artifact' else '../escape'): '0'*64}))
    else:
        mutate(job, lambda x: x.update({('original_result_sha256' if fault == 'original_hash' else 'runtime_source_identity'): '0'*64}))
        mutate(runtime, lambda x: x['files'].update({'job.json': sha(job)}))
        seal(directory)
    output = catalog.build_catalog(case['path'])
    assert not output['integrity_passed']
    assert output['jobs'][3]['status'] == 'REJECTED'
    assert output['jobs'][1]['status'] == 'FAILED', 'Original failure remains in the denominator'
    assert not output['evaluation_authorized']


@pytest.mark.parametrize('fault', ['duplicate', 'conflicting_version', 'escape', 'unmapped', 'unknown_budget', 'enable_evaluation'])
def test_manifest_contract_errors_fail_closed(case, fault):
    m = case['manifest']
    if fault == 'duplicate':
        m['runs'].append({**m['runs'][1], 'id': 'duplicated'})
    elif fault == 'conflicting_version':
        m['runs'][2]['method_version'] = m['runs'][1]['method_version']
    elif fault == 'escape':
        m['runs'][0]['directory']['path'] = '../outside'
    elif fault == 'unmapped':
        m['relocations'] = []
    elif fault == 'unknown_budget':
        m['runs'][0]['budget_policy'] = {'name': 'invented_budget', 'execution': {'fit_seconds': 1800, 'worker_rss_bytes': 4*1024**3}}
    else:
        m['evaluation_authorized'] = True
    write(case['path'], m)
    if fault == 'unmapped':
        assert not catalog.build_catalog(case['path'])['integrity_passed']
    else:
        with pytest.raises(catalog.CatalogError):
            catalog.build_catalog(case['path'])


def test_missing_receipt_is_pending_only_for_declared_active_run(case):
    m = case['manifest']
    m['runs'][2]['lifecycle'] = 'closed'
    write(case['path'], m)
    output = catalog.build_catalog(case['path'])
    assert output['jobs'][-1]['status'] == 'REJECTED'
    assert output['issues'][-1]['code'] == 'MISSING_TERMINAL_RECEIPT'


def test_external_timeout_without_runtime_receipt_is_honest_failure(case):
    directory = case['new_jobs'][1]
    job = json.loads((directory / 'job.json').read_text())
    write(directory / 'result.json', {'candidate_id': job['config']['candidate_id'], 'seed': job['seed'],
        'status': 'TIME_LIMIT', 'source_identity': job['source_identity'], 'data_identity': job['data_identity']})
    (directory / 'worker.log').write_text('external timeout')
    seal(directory, status='TIME_LIMIT', termination='TIME_LIMIT')
    output = catalog.build_catalog(case['path'])
    assert output['integrity_passed'], output['issues']
    row = output['jobs'][-1]
    assert row['status'] == 'TIME_LIMIT' and row['evidence_state'] == 'VERIFIED_EXTERNAL_STOP'
    assert row['original_evidence']['runtime_verified'] is False
    assert 'artifacts' not in row


def test_symlink_model_escape_is_rejected_even_when_hash_matches(case, tmp_path):
    directory = case['new_jobs'][0]
    outside = tmp_path / 'outside_model'
    outside.write_bytes((directory / 'model.joblib').read_bytes())
    (directory / 'model.joblib').unlink()
    (directory / 'model.joblib').symlink_to(outside)
    output = catalog.build_catalog(case['path'])
    assert output['jobs'][3]['status'] == 'REJECTED'
    assert output['jobs'][3]['issue_code'] == 'PATH_ESCAPE'


def test_frappe_runtime_is_bound_by_job_and_parallel_receipt_without_extra_receipt(case):
    # Construct the actual deployed FRAPPE schema on a standalone generated registration.
    old = case['originals'][case['config']['candidate_id'] + '_s0']
    cfg = copy.deepcopy(case['config'])
    cfg.update(method='FRAPPE_EO', seeds=[0])
    cfg['candidate_id'] = catalog._identity(cfg)[:20]
    reg = json.loads(case['regpath'].read_text())
    reg['candidates'] = [cfg]
    write(case['regpath'], reg)
    runtime = copy.deepcopy(case['runtime'])
    runtime.update(variant=catalog.FRAPPE_VARIANT, environment={'TF_DETERMINISTIC_OPS': '1'})
    runtime['source_files'] = {name: sha(case['root'] / name) for name in catalog.RUNTIME_REQUIRED['frappe_v1']}
    runtime['source_identity'] = catalog._identity({'variant': runtime['variant'], 'files': runtime['source_files'], 'environment': runtime['environment']})
    directory = case['root'] / 'artifacts/frappe/jobs' / (cfg['candidate_id'] + '_s0')
    job = json.loads((old / 'job.json').read_text())
    job.update(config=cfg, runtime_variant=runtime['variant'], runtime_source_identity=runtime['source_identity'],
               runtime_source_files=runtime['source_files'], runtime_environment=runtime['environment'])
    write(directory / 'job.json', job)
    write(directory / 'result.json', {'candidate_id': cfg['candidate_id'], 'seed': 0, 'status': 'VALID',
        'source_identity': job['source_identity'], 'data_identity': job['data_identity'], 'reload_verified': True,
        'output_type': 'event_probability_p'})
    for name in ('worker.log', 'model.joblib', 'predictions_S.npz'):
        (directory / name).write_bytes(b'generated opaque witness')
    seal(directory)
    m = case['manifest']
    m['registrations'][0]['sha256'] = sha(case['regpath'])
    run = copy.deepcopy(m['runs'][2])
    run.update(id='frappe', profile='frappe_v1', runtime=runtime, role='replacement', lifecycle='closed',
        directory={'root': 'archive', 'path': 'artifacts/frappe'}, expected_jobs=[directory.name])
    run['budget_policy']['name'] = 'frappe_deterministic_v1'
    m['runs'] = [run]
    write(case['path'], m)
    output = catalog.build_catalog(case['path'])
    assert output['integrity_passed'], output['issues']
    assert output['jobs'][0]['status'] == 'VALID'


def test_cli_writes_fresh_small_inventory_and_refuses_overwrite(case, tmp_path):
    script = Path(__file__).resolve().parents[2] / 'scripts/build_nhis_result_catalog.py'
    output = tmp_path / 'catalog_output'
    args = [sys.executable, '-B', str(script), '--manifest', str(case['path']), '--output', str(output)]
    first = subprocess.run(args, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert {p.name for p in output.iterdir()} == {'catalog.json', 'summary.json', 'issues.json', 'jobs.jsonl'}
    assert len((output / 'jobs.jsonl').read_text().splitlines()) == 5
    assert subprocess.run(args, capture_output=True, text=True).returncode != 0


def test_cli_cannot_write_inside_historical_run(case):
    script = Path(__file__).resolve().parents[2] / 'scripts/build_nhis_result_catalog.py'
    output = case['original'] / 'new_catalog'
    child = subprocess.run([sys.executable, '-B', str(script), '--manifest', str(case['path']), '--output', str(output)],
                           capture_output=True, text=True)
    assert child.returncode != 0 and not output.exists()


def test_q_cannot_be_relabelled_as_event_risk(case):
    directory = case['new_jobs'][0]
    mutate(directory / 'result.json', lambda r: r.update(output_type='event_probability_p'))
    mutate(directory / 'runtime_receipt.json', lambda r: r['files'].update({'result.json': sha(directory / 'result.json')}))
    seal(directory)
    output = catalog.build_catalog(case['path'])
    assert output['jobs'][3]['issue_code'] == 'INVALID_P_Q_OUTPUT_IDENTITY'


def fairbias_case(case, policy):
    """Write the production FairBias worker+scheduler+cache receipt structure."""
    spec = catalog.FAIRBIAS_POLICIES[policy]
    cfg = {'method': spec['method'], 'arm_id': 'arm', 'backbone': 'LR', 'params': {},
           'status': 'REGISTERED', 'seeds': [0], 'training_weighted': policy == 'bm_mds_retry_v1'}
    cfg['candidate_id'] = catalog._identity(cfg)[:20]
    name = cfg['candidate_id'] + '_s0'
    reg = json.loads(case['regpath'].read_text())
    reg['candidates'] = [cfg]
    write(case['regpath'], reg)
    template = json.loads((next(iter(case['originals'].values())) / 'job.json').read_text())
    template.update(config=cfg, parallel_scheduler_version=catalog.SCHEDULER, representation_key=None)
    old = case['original'] / 'jobs' / name
    write(old / 'job.json', template)
    write(old / 'result.json', {'candidate_id': cfg['candidate_id'], 'seed': 0, 'status': spec['original_statuses'][0]})
    (old / 'worker.log').write_text('generated historical failure')
    seal(old, status=spec['original_statuses'][0], termination='TIME_LIMIT' if policy.startswith('joint') else None)
    runtime_files = {'src/registered.py': sha(case['source']),
                     **{p: sha(case['root'] / p) for p in catalog.RUNTIME_REQUIRED['fairbias_recovery_v1']}}
    environment = {**catalog.NUMERICAL_ENVIRONMENT, 'PYTHONHASHSEED': '0'}
    execution = {'fit_seconds': 1800., 'worker_rss_bytes': 4*1024**3}
    rid = catalog._identity({'variant': catalog.FAIRBIAS_VARIANT, 'files': runtime_files, 'environment': environment,
        'policy': policy, 'policy_spec': spec, 'execution_budget': execution})
    runtime = {'variant': catalog.FAIRBIAS_VARIANT, 'source_identity': rid, 'source_files': runtime_files,
               'environment': environment, 'recovery_policy': policy, 'recovery_policy_spec': spec}
    run = case['root'] / 'artifacts' / ('fairbias_' + rid)
    directory = run / 'jobs' / name
    base = '/remote/original_checkout'
    job = {**template, 'runtime_variant': runtime['variant'], 'runtime_source_identity': rid,
        'runtime_source_files': runtime_files, 'runtime_environment': environment,
        'recovery_policy': policy, 'execution_budget': execution,
        'recovery_run_path': base + '/artifacts/' + run.name,
        'cache_path': base + '/artifacts/' + run.name + '/representation_cache',
        'registration_path': base + '/artifacts/original/registration.json', 'registration_sha256': sha(case['regpath']),
        'original_job_path': base + '/artifacts/original/jobs/' + name + '/job.json',
        'original_job_sha256': sha(old / 'job.json'), 'original_result_sha256': sha(old / 'result.json')}
    key = catalog._representation_key(cfg, 0, job['data_identity'])
    job['representation_key'] = key
    write(directory / 'job.json', job)
    write(directory / 'result.json', {'candidate_id': cfg['candidate_id'], 'seed': 0, 'status': 'VALID',
        'source_identity': job['source_identity'], 'data_identity': job['data_identity'],
        'reload_verified': True, 'output_type': 'event_probability_p'})
    write(directory / 'runtime_started.json', {'runtime_variant': runtime['variant'], 'runtime_source_identity': rid,
          'job_sha256': sha(directory / 'job.json'), 'pid': 1})
    for filename in ('worker.log', 'model.joblib', 'predictions_S.npz'):
        (directory / filename).write_bytes(b'generated opaque witness')
    rr = {k: job[k] for k in ('runtime_variant', 'runtime_source_identity', 'runtime_source_files',
        'runtime_environment', 'data_identity', 'data_sha256', 'registration_sha256', 'original_job_sha256',
        'original_result_sha256', 'recovery_policy', 'execution_budget', 'cache_path')}
    rr.update(schema_version='nhis_fairbias_recovery_receipt_v1', recovery_policy_spec=spec,
        candidate_id=cfg['candidate_id'], seed=0, status='VALID', worker_result_status='VALID', runtime_checks_passed=True,
        original_status=spec['original_statuses'][0], legacy_valid_results_admitted=False, representation_key=key,
        loaded_project_modules_before={'nhis_fairbias.example': {'source': 'src/registered.py', 'sha256': sha(case['source'])}},
        loaded_project_modules_after={'nhis_fairbias.example': {'source': 'src/registered.py', 'sha256': sha(case['source'])}},
        reload_exact_C=True, model_admission={'status': 'COMPLETE_FEASIBLE', 'final_max_dphi': .1, 'epsilon_threshold': .2,
            'model_available': True, 'final_geometry_feasible': True, 'search_complete': True, 'convergence_verified': True,
            'geometry_search_incomplete': False, 'termination_reason': 'STRICT_FEASIBLE_SEARCH_EXHAUSTED'},
        files={p.name: sha(p) for p in directory.iterdir() if p.name != 'worker.log'})
    if key:
        cache = run / 'representation_cache'
        marker = {'runtime_variant': runtime['variant'], 'runtime_source_identity': rid, 'registration_sha256': sha(case['regpath'])}
        write(cache / '.runtime_namespace.json', marker)
        (cache / (key + '.joblib')).write_bytes(b'opaque generated shared representation')
        write(cache / (key + '.json'), {'key': key, 'status': 'VALID', 'model_sha256': sha(cache / (key + '.joblib'))})
        sidecar = {**marker, 'representation_key': key,
                   'files': {s: sha(cache / (key + s)) for s in ('.json', '.joblib')}}
        write(cache / (key + '.runtime.json'), sidecar)
        rr['representation_cache_evidence'] = sidecar
    write(directory / 'runtime_receipt.json', rr)
    seal(directory)
    manifest = case['manifest']
    manifest['registrations'][0]['sha256'] = sha(case['regpath'])
    original_run = {**manifest['runs'][1], 'expected_jobs': [name]}
    new = {**manifest['runs'][2], 'id': 'fairbias', 'profile': 'fairbias_recovery_v1', 'runtime': runtime,
        'role': 'sensitivity' if key else 'replacement', 'lifecycle': 'closed', 'expected_jobs': [name],
        'directory': {'root': 'archive', 'path': str(run.relative_to(case['root']))},
        'method_version': policy, 'budget_policy': {'name': policy, 'execution': execution}}
    manifest['runs'] = [original_run, new]
    write(case['path'], manifest)
    return directory, key


@pytest.mark.parametrize('policy', ['bm_mds_retry_v1', 'joint_strict_exact_v1'])
def test_actual_fairbias_receipt_schema_and_budget_roles(case, policy):
    directory, key = fairbias_case(case, policy)
    output = catalog.build_catalog(case['path'])
    assert output['integrity_passed'], output['issues']
    assert output['jobs'][1]['status'] == 'VALID'
    assert output['jobs'][0]['status'] != 'VALID'
    assert output['jobs'][1]['original_evidence']['runtime_verified']
    assert output['jobs'][1]['role'] == ('sensitivity' if key else 'replacement')


@pytest.mark.parametrize('fault', ['incomplete', 'infeasible', 'wrong_policy', 'changed_budget', 'missing_cache_model'])
def test_fairbias_incomplete_and_budget_or_cache_mismatch_refused(case, fault):
    directory, key = fairbias_case(case, 'bm_mds_retry_v1' if fault == 'missing_cache_model' else 'joint_strict_exact_v1')
    rp = directory / 'runtime_receipt.json'
    if fault == 'incomplete':
        mutate(rp, lambda r: r['model_admission'].update(search_complete=False))
    elif fault == 'infeasible':
        mutate(rp, lambda r: r['model_admission'].update(final_max_dphi=.3))
    elif fault == 'wrong_policy':
        mutate(rp, lambda r: r.update(recovery_policy='scheduled'))
    elif fault == 'changed_budget':
        mutate(rp, lambda r: r['execution_budget'].update(fit_seconds=3600))
    else:
        (directory.parent.parent / 'representation_cache' / (key + '.joblib')).unlink()
    seal(directory)
    output = catalog.build_catalog(case['path'])
    assert not output['integrity_passed']
    assert output['jobs'][1]['status'] == 'REJECTED'


def test_bm_budget_sensitivity_cannot_be_relabelled_original(case):
    fairbias_case(case, 'bm_mds_retry_v1')
    mutate(case['path'], lambda m: m['runs'][1].update(role='original'))
    with pytest.raises(catalog.CatalogError, match='SENSITIVITY'):
        catalog.build_catalog(case['path'])


def test_pending_job_with_wrong_runtime_identity_is_rejected(case):
    mutate(case['new_jobs'][1] / 'job.json', lambda j: j.update(runtime_source_identity='0'*64))
    output = catalog.build_catalog(case['path'])
    assert output['jobs'][-1]['status'] == 'REJECTED'


@pytest.mark.parametrize('text', ['[]', '{"x":NaN}', '{"x":1,"x":2}', '{"metrics_S":[1,]}', '{"metrics_S":"bad\\q"}'])
def test_malformed_json_is_rejected_even_in_skipped_field(tmp_path, text):
    path = tmp_path / 'bad.json'
    path.write_text(text)
    with pytest.raises(catalog.CatalogError):
        catalog.read_metadata(path, catalog.RESULT_FIELDS)
