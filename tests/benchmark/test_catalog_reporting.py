"""Synthetic cross-run study-to-paper export, without model or array loading."""
import copy
import csv
import json
from pathlib import Path

import pytest

from nhis_fairbias.benchmark import catalog_reporting as reporting
from nhis_fairbias.benchmark import catalog_selection, catalog_evaluation, result_catalog
from test_catalog_selection import case as selection_case, _refresh
from test_result_catalog import write, sha, seal
from test_catalog_evaluation import study_case, _loader, _evaluate


def _bound(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def _metrics(value):
    return {'mean': value, 'seed_sd': .01 if value is not None else None,
            'seed_range': [value, value] if value is not None else None,
            'status': 'VALID' if value is not None else 'NOT_ESTIMABLE'}


def _reseal(case):
    write(case['study_path'], case['study'])
    study_sha = sha(case['study_path'])
    case['release']['study_freeze_sha256'] = study_sha
    case['release']['study_branch'] = case['study']['study_branch']
    write(case['release_path'], case['release'])
    case['evaluation'].update(study_freeze_sha256=study_sha, release=_bound(case['release_path']),
                              study_branch=case['study']['study_branch'])
    write(case['evaluation_path'], case['evaluation'])
    case['summary'].update(study_freeze_sha256=study_sha, study_branch=case['study']['study_branch'],
                           evaluation_manifest_sha256=sha(case['evaluation_path']))
    write(case['summary_path'], case['summary'])


def _pairs(rows):
    paired = []
    for fair in rows:
        for base in rows:
            if fair['method'] != 'FAIRBIAS_BM' or base['method'] == 'FAIRBIAS_BM' or not fair['model_ids'] or not base['model_ids']:
                continue
            if any(fair[k] != base[k] for k in ('arm_id', 'backbone', 'tau', 'training_weighted')):
                continue
            if (fair['status'] == 'FIXED_ABLATION') != (base['status'] == 'FIXED_ABLATION'):
                continue
            paired.append(dict(arm_id=fair['arm_id'], backbone=fair['backbone'], tau=fair['tau'], training_weighted=fair['training_weighted'],
                reference_method='FAIRBIAS_BM', comparison_method=base['method'],
                feasible_on_S=fair['status'] == base['status'] == 'FEASIBLE', in_primary_family_20=False,
                reference_selection_id=fair['selection_id'], comparison_selection_id=base['selection_id'],
                reference_model_ids=fair['model_ids'], comparison_model_ids=base['model_ids'], study_branch=fair['study_branch'],
                delta_balanced_accuracy={'status': 'VALID', 'estimate': .05, 'se': .01, 'lower': .02, 'upper': .08},
                delta_eo_gap={'estimate': -.02, 'projection_95': {'status': 'VALID', 'lower': -.04, 'upper': .0},
                    'projection_primary_family': {'status': 'VALID', 'lower': -.08, 'upper': .04}},
                delta_average_precision=None, delta_auroc=None, delta_brier=None))
    return paired


def _make_case(selection_case, monkeypatch):
    base = selection_case
    root = base['root']
    selection_path = write(root / 'selection.json', catalog_selection.freeze_catalog_selection(base['admission_path']))
    # Exercise the real study builder, replacing only the synthetic arm registry.
    monkeypatch.setattr(catalog_evaluation, 'ARM_SPECS', {'arm': {}, 'empty_arm': {}})
    provenance = {'identity': 'official HHX + survey year', 'feature_registry_sha256': '1'*64,
        'study_registry_sha256': '2'*64, 'raw_sources': {'2024': {'path': 'generated/not_read.csv', 'sha256': '3'*64, 'rows': 8}}}
    study_path = root / 'study.json'
    study = catalog_evaluation.freeze_catalog_study(selection_path, sha(selection_path), study_path,
        t_source_provenance=provenance, extension_resolutions={})
    for cid, status in [('bm_winner', 'FIXED_ABLATION'), ('bm_failed', 'NO_VALID_FIXED_ABLATION')]:
        fixed = dict(arm_id='arm', backbone='LR', method='FAIRBIAS_BM', training_weighted=False,
            tau=None, status=status, selected_candidate_id=cid if status == 'FIXED_ABLATION' else None,
            registered_fixed_candidate_id=cid, analysis_version='registered_v1', study_branch=study['study_branch'],
            model_ids=[m['model_id'] for m in study['selected_models'] if m['candidate_id'] == cid])
        fixed['selection_id'] = 'selection_' + result_catalog._identity(fixed)
        study['selections'].append(fixed)
    release = {'schema_version': catalog_evaluation.RELEASE_SCHEMA, 'decision': 'SUPERVISOR_RELEASED_T',
        'study_freeze_sha256': sha(study_path), 'selection_sha256': sha(selection_path),
        'study_branch': study['study_branch'], 'arm_ids': list(study['arm_ids'])}
    release_path = write(root / 'release.json', release)
    evaluation = catalog_evaluation._manifest(study, sha(study_path), _bound(release_path), study['arm_ids'], 'complete_study')
    evaluation['observed_source_provenance'] = copy.deepcopy(provenance)
    models = {m['model_id']: m for m in study['selected_models']}
    rows = []
    for selected in study['selections']:
        row = dict(selected, frozen_seed_models=list(selected['model_ids']),
                   evaluation_status='VALID' if selected['model_ids'] else 'NO_VALID_MODEL')
        if selected['model_ids']:
            row.update(sample_n=8, design_n=12, df=4)
            q_only = models[selected['model_ids'][0]]['output_type'] == 'decision_probability_q'
            for prefix in ('', 'unweighted_'):
                for metric, value in [('balanced_accuracy', .75), ('eo_gap', .04), ('dp_gap', .03),
                    ('average_precision', None if q_only else .71), ('auroc', None if q_only else .73),
                    ('brier', None if q_only else .21)]:
                    row[prefix + metric] = _metrics(value)
            row['balanced_accuracy']['taylor_95'] = {'status': 'VALID', 'se': .03, 'lower': .68, 'upper': .82}
            for name in ('eo_gap', 'dp_gap'):
                for band in ('projection_95', 'projection_primary_family'):
                    row[name][band] = {'status': 'VALID', 'lower': .01, 'upper': .09}
            if q_only:
                for metric in reporting.RISK_METRICS:
                    row['untouched_base_' + metric] = _metrics(.6)
        rows.append(row)
    evaluation_path = root / 'evaluation_manifest.json'
    summary_path = root / 'summary_T.json'
    summary = {'schema_version': catalog_evaluation.SUMMARY_SCHEMA, 'study_freeze_sha256': sha(study_path),
        'selection_sha256': sha(selection_path), 'study_branch': study['study_branch'],
        'selections': rows, 'paired_contrasts': _pairs(rows)}
    case = dict(base=base, root=root, study=study, study_path=study_path, selection_path=selection_path,
        release=release, release_path=release_path, evaluation=evaluation, evaluation_path=evaluation_path,
        summary=summary, summary_path=summary_path)
    _reseal(case)
    return case


@pytest.fixture
def case(selection_case, monkeypatch):
    return _make_case(selection_case, monkeypatch)


def _export(case, output=None, **kwargs):
    return reporting.write_catalog_reporting(case['study_path'], case['selection_path'],
        case['evaluation_path'], case['summary_path'], output or case['root'] / 'report',
        expected_study_sha256=kwargs.pop('study_sha', sha(case['study_path'])),
        expected_selection_sha256=kwargs.pop('selection_sha', sha(case['selection_path'])),
        expected_summary_sha256=kwargs.pop('summary_sha', sha(case['summary_path'])), **kwargs)


def _csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def test_real_study_schema_export_preserves_denominators_risk_and_fixed_rows(case, monkeypatch):
    import joblib
    import numpy as np
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('No models may be deserialized'))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('No prediction arrays may be loaded'))
    reader = result_catalog.read_metadata
    def safe_read(path, fields=None):
        if Path(path).name == 'result.json':
            assert fields is not None and not {'metrics_S', 'risk_S', 'risk_T'} & fields
        return reader(path, fields)
    monkeypatch.setattr(result_catalog, 'read_metadata', safe_read)
    outputs = _export(case)
    proof = json.loads(outputs['manifest'].read_text())
    assert proof['denominators']['registered_jobs'] == 12
    assert proof['denominators']['catalog_attempts'] == 13
    assert proof['denominators']['resolved_failed_jobs'] == 1
    assert proof['denominators']['not_supported_configurations'] == 1
    assert proof['selected_models'] == case['study']['selected_models']
    assert not proof['evaluation_authorized']
    main = _csv(outputs['csv'])
    eg = next(row for row in main if row['method'] == 'EG_DP')
    assert eg['average_precision_p'] == eg['auroc_p'] == eg['brier_p'] == ''
    assert eg['untouched_base_average_precision'] == '0.6'
    bm = next(row for row in main if row['method'] == 'FAIRBIAS_BM')
    assert bm['seed_sd_balanced_accuracy'] == '0.01' and bm['design_se_balanced_accuracy'] == '0.03'
    aux = _csv(outputs['csv'].parent / 'paper_results_auxiliary.csv')
    assert {'FIXED_ABLATION', 'NO_VALID_FIXED_ABLATION'} <= {row['S_status'] for row in aux}
    failed = _csv(outputs['csv'].parent / 'paper_failures.csv')
    assert len(failed) == 1 and failed[0]['job_id'] == 'bm_failed_s1'
    assert '/archive/original/jobs/' in failed[0]['result_path']
    attempts = _csv(outputs['csv'].parent / 'attempt_status_counts.csv')
    assert sum(int(row['jobs']) for row in attempts) == 13
    assert any(row['method'] == 'EG_DP' and row['status'] == 'FAILED' for row in attempts)
    unsupported = _csv(outputs['csv'].parent / 'not_supported_conditions.csv')
    assert unsupported[0]['candidate_id'] == 'lfr_unsupported'
    pair = _csv(outputs['csv'].parent / 'paper_paired_contrasts.csv')[0]
    assert pair['delta_balanced_accuracy'] == '0.05' and pair['delta_average_precision'] == ''
    assert json.loads(pair['reference_model_ids'])
    for name, digest in proof['output_files'].items():
        assert sha(outputs['csv'].parent / name) == digest
    assert 'fixed-policy survey uncertainty' in outputs['markdown'].read_text()


@pytest.mark.parametrize('which', ['study_sha', 'selection_sha', 'summary_sha'])
def test_wrong_caller_hash_rejected_without_writing(case, which):
    with pytest.raises(reporting.ReportingError):
        _export(case, **{which: '0'*64})
    assert not (case['root'] / 'report').exists()


@pytest.mark.parametrize('fault', ['study_schema', 'study_selection_hash', 'missing_source', 'source_hash',
    'source_escape', 'manifest_source', 'manifest_study', 'manifest_scope', 'manifest_extra_model',
    'manifest_missing_model', 'manifest_duplicate_model', 'summary_missing', 'summary_extra', 'summary_duplicate',
    'summary_model', 'summary_missingfield', 'summary_wrong_branch', 'summary_q_risk', 'summary_risk_type',
    'paired_q_risk', 'paired_model', 'paired_missing', 'missing_denominator', 'changed_denominator',
    'changed_model_path', 'missing_model_hash', 'missing_model_metadata', 'duplicate_model', 'missing_seed',
    'release', 'statistics', 'missing_pair', 'unknown_pair', 'primary_family', 'observed_source', 'observed_rows',
    'source_closure', 'q_seed_sd', 'missing_metric_status', 'runtime_metadata'])
def test_schema_and_provenance_tampering_fail_closed(case, fault):
    study, manifest, summary = case['study'], case['evaluation'], case['summary']
    if fault == 'study_schema': study['schema_version'] = 'unknown'
    elif fault == 'study_selection_hash': study['selection']['sha256'] = '0'*64
    elif fault == 'missing_source': study['analysis_files'].pop('src/nhis_fairbias/benchmark/catalog_reporting.py')
    elif fault == 'source_hash': study['analysis_files']['src/nhis_fairbias/benchmark/catalog_reporting.py'] = '0'*64
    elif fault == 'source_escape': study['analysis_files']['../outside.py'] = '0'*64
    elif fault == 'manifest_source': manifest['analysis_files'] = {}
    elif fault == 'manifest_study': pass  # change after dependent hashes are sealed
    elif fault == 'manifest_scope': manifest['scope'] = 'arm'
    elif fault == 'manifest_extra_model': manifest['selected_models'] = manifest['selected_models'] + [{'model_id': 'extra'}]
    elif fault == 'manifest_missing_model': manifest['selected_models'] = manifest['selected_models'][:-1]
    elif fault == 'manifest_duplicate_model': manifest['selected_models'] = manifest['selected_models'] * 2
    elif fault == 'summary_missing': summary['selections'].pop()
    elif fault == 'summary_extra': summary['selections'].append({'selection_id': 'invented'})
    elif fault == 'summary_duplicate': summary['selections'].append(copy.deepcopy(summary['selections'][0]))
    elif fault == 'summary_model': summary['selections'][0]['frozen_seed_models'] = ['invented']
    elif fault == 'summary_missingfield': del summary['selections'][0]['balanced_accuracy']
    elif fault == 'summary_wrong_branch': summary['selections'][0]['study_branch'] = 'another'
    elif fault == 'summary_q_risk':
        next(r for r in summary['selections'] if r['method'] == 'EG_DP')['average_precision']['mean'] = .9
    elif fault == 'summary_risk_type': summary['selections'][0]['average_precision']['mean'] = True
    elif fault == 'paired_q_risk': summary['paired_contrasts'][0]['delta_average_precision'] = .1
    elif fault == 'paired_model': summary['paired_contrasts'][0]['reference_model_ids'] = []
    elif fault == 'paired_missing': del summary['paired_contrasts'][0]['reference_selection_id']
    elif fault == 'missing_denominator': study['resolved_jobs'].pop()
    elif fault == 'changed_denominator': study['resolved_jobs'][0]['status'] = 'FAILED'
    elif fault == 'changed_model_path': study['selected_models'][0]['artifacts']['model.joblib']['path'] = '/outside/model.joblib'
    elif fault == 'missing_model_hash': del study['selected_models'][0]['artifacts']['model.joblib']['sha256']
    elif fault == 'missing_model_metadata': del study['selected_models'][0]['runtime_job_metadata']
    elif fault == 'duplicate_model': study['selected_models'].append(copy.deepcopy(study['selected_models'][0]))
    elif fault == 'missing_seed': study['selected_models'].pop()
    elif fault == 'release': case['release']['decision'] = 'NOT_RELEASED'
    elif fault == 'statistics': study['bootstrap_replicates'] = 200
    elif fault == 'missing_pair': summary['paired_contrasts'].pop()
    elif fault == 'unknown_pair': summary['paired_contrasts'][0]['comparison_selection_id'] = 'unknown'
    elif fault == 'primary_family': summary['paired_contrasts'][0]['in_primary_family_20'] = True
    elif fault == 'observed_source': manifest['observed_source_provenance']['raw_sources']['2024']['sha256'] = '0'*64
    elif fault == 'observed_rows': del manifest['observed_source_provenance']['raw_sources']['2024']['rows']
    elif fault == 'source_closure': study['analysis_files'].pop('src/fairbias/__init__.py')
    elif fault == 'q_seed_sd':
        next(r for r in summary['selections'] if r['method'] == 'EG_DP')['average_precision']['seed_sd'] = .1
    elif fault == 'missing_metric_status': del summary['selections'][0]['average_precision']['status']
    elif fault == 'runtime_metadata':
        ref = study['selected_models'][0]
        ref['runtime_job_metadata'] = {'invented': True}
        ref['model_id'] = 'model_' + result_catalog._identity({k: v for k, v in ref.items() if k != 'model_id'})
    _reseal(case)
    if fault == 'manifest_study':
        manifest['study_freeze_sha256'] = '0'*64
        write(case['evaluation_path'], manifest)
        summary['evaluation_manifest_sha256'] = sha(case['evaluation_path'])
        write(case['summary_path'], summary)
    with pytest.raises(reporting.ReportingError):
        _export(case)
    assert not (case['root'] / 'report').exists()


@pytest.mark.parametrize('name', ['job.json', 'result.json', 'receipt.json'])
def test_historical_attempt_metadata_tamper_rejected(case, name):
    # This original EG failure was replaced, but remains in historical attempts.
    path = case['base']['jobs']['eg_s0'] / name
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(reporting.ReportingError, match='FILE_HASH_MISMATCH'):
        _export(case)


def test_existing_output_and_historical_run_output_refused(case):
    existing = case['root'] / 'existing'
    existing.mkdir()
    with pytest.raises(reporting.ReportingError, match='OUTPUT_ALREADY_EXISTS'):
        _export(case, existing)
    output = case['base']['jobs']['bm_winner_s0'].parent.parent / 'new-report'
    with pytest.raises(reporting.ReportingError, match='OUTPUT_INSIDE_EVIDENCE'):
        _export(case, output)
    assert not output.exists()


def test_no_feasible_boundary_and_unsupported_remain_visible(selection_case, monkeypatch):
    for name, directory in selection_case['jobs'].items():
        if name.startswith('bm_'):
            result = json.loads((directory / 'result.json').read_text())
            if result['status'] == 'VALID':
                result['metrics_S']['eo_gap'] = .3
                write(directory / 'result.json', result)
                seal(directory, parallel=False)
    _refresh(selection_case)
    case = _make_case(selection_case, monkeypatch)
    outputs = _export(case)
    rows = _csv(outputs['csv'])
    bm = next(row for row in rows if row['method'] == 'FAIRBIAS_BM')
    assert bm['S_status'] == 'NO_FEASIBLE_CONFIGURATION' and bm['candidate_id']
    assert _csv(outputs['csv'].parent / 'not_supported_conditions.csv')[0]['status'] == 'NOT_SUPPORTED'


def test_supplemental_budget_identity_is_bound_but_not_pooled(case):
    evidence = write(case['root'] / 'sensitivity_receipt.json', {'status': 'TERMINAL_RECEIPTS_ACCEPTED', 'jobs': 40})
    branch = {'study_branch': 'bmae_cap40_sensitivity', 'status': 'AWAITING_SEPARATE_FROZEN_EVALUATION',
              'evidence_files': [_bound(evidence)]}
    case['study']['supplemental_evidence'] = [branch]
    _reseal(case)
    outputs = _export(case)
    manifest = json.loads(outputs['manifest'].read_text())
    assert manifest['supplemental_evidence'] == [branch]
    assert manifest['denominators']['registered_jobs'] == 12
    assert not any(r['study_branch'] == 'bmae_cap40_sensitivity' for r in _csv(outputs['csv']))
    evidence.write_text('{}')
    with pytest.raises(reporting.ReportingError, match='FILE_HASH_MISMATCH'):
        _export(case, case['root'] / 'tampered_report')


def test_generated_actual_arm_evaluation_merge_and_report_interoperate(study_case, monkeypatch):
    case = study_case
    _loader(case, monkeypatch)
    shards = []
    for arm in catalog_evaluation.ARM_SPECS:
        out = case['root'] / arm
        _evaluate(case, out, arm)
        shards.append(_bound(out / 'summary_T.json'))
    merged = case['root'] / 'merged'
    catalog_evaluation.merge_catalog_evaluations(case['study_path'], sha(case['study_path']), shards, merged,
        release_path=case['release_path'], expected_release_sha256=sha(case['release_path']))
    import joblib
    import numpy as np
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('report must not load model'))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('report must not load prediction arrays'))
    monkeypatch.setattr(catalog_evaluation, 'load_local_nhis_cohort', lambda *a, **k: pytest.fail('report must not load T'))
    result = reporting.write_catalog_reporting(case['study_path'], case['selection_path'],
        merged / 'evaluation_manifest.json', merged / 'summary_T.json', case['root'] / 'paper',
        expected_study_sha256=sha(case['study_path']), expected_selection_sha256=sha(case['selection_path']),
        expected_summary_sha256=sha(merged / 'summary_T.json'))
    proof = json.loads(result['manifest'].read_text())
    assert proof['denominators']['registered_jobs'] == proof['denominators']['catalog_attempts'] == 12
    assert proof['denominators']['resolved_failed_jobs'] == 1
    assert len(proof['selected_models']) == 10
    rows = _csv(result['csv'])
    q_row = next(row for row in rows if row['method'] == 'TO_EO')
    assert q_row['average_precision_p'] == '' and q_row['untouched_base_average_precision']
    bm = next(row for row in rows if row['method'] == 'FAIRBIAS_BM')
    assert float(bm['eo_gap']) == 1.0  # per-seed gap mean, not mean-policy gap
    auxiliary = _csv(result['csv'].parent / 'paper_results_auxiliary.csv')
    assert any(row['S_status'] == 'NO_VALID_FIXED_ABLATION' for row in auxiliary)


def test_hash_only_declared_t_provenance_accepts_observed_row_count(case):
    case['study']['t_source_provenance'] = copy.deepcopy(case['study']['t_source_provenance'])
    case['study']['t_source_provenance']['raw_sources']['2024'].pop('rows')
    case['evaluation']['source_provenance'] = case['study']['t_source_provenance']
    _reseal(case)
    assert _export(case)['manifest'].exists()


@pytest.mark.parametrize('fault', ['point_high', 'point_low', 'risk_high', 'base_risk_high', 'delta_ba', 'delta_eo',
    'delta_risk', 'missing_taylor', 'missing_projection', 'missing_family_projection', 'missing_interval_status',
    'missing_valid_bound', 'reversed_bounds', 'projection_outside', 'missing_pair_interval'])
def test_illegal_points_or_missing_intervals_cannot_be_published(case, fault):
    row = case['summary']['selections'][0]
    pair = case['summary']['paired_contrasts'][0]
    if fault == 'point_high': row['balanced_accuracy']['mean'] = 1.01
    elif fault == 'point_low': row['eo_gap']['mean'] = -.01
    elif fault == 'risk_high': row['average_precision']['mean'] = 2.
    elif fault == 'base_risk_high':
        next(r for r in case['summary']['selections'] if r['method'] == 'EG_DP')['untouched_base_average_precision']['mean'] = 2.
    elif fault == 'delta_ba': pair['delta_balanced_accuracy']['estimate'] = 1.01
    elif fault == 'delta_eo': pair['delta_eo_gap']['estimate'] = -1.01
    elif fault == 'delta_risk': pair['delta_average_precision'] = 1.01
    elif fault == 'missing_taylor': del row['balanced_accuracy']['taylor_95']
    elif fault == 'missing_projection': del row['eo_gap']['projection_95']
    elif fault == 'missing_family_projection': del row['dp_gap']['projection_primary_family']
    elif fault == 'missing_interval_status': del row['balanced_accuracy']['taylor_95']['status']
    elif fault == 'missing_valid_bound': del row['balanced_accuracy']['taylor_95']['upper']
    elif fault == 'reversed_bounds': row['balanced_accuracy']['taylor_95'].update(lower=.9, upper=.1)
    elif fault == 'projection_outside': row['eo_gap']['projection_95']['lower'] = -.01
    elif fault == 'missing_pair_interval': del pair['delta_eo_gap']['projection_primary_family']
    _reseal(case)
    with pytest.raises(reporting.ReportingError):
        _export(case)
    assert not (case['root'] / 'report').exists()


def test_taylor_bounds_are_not_clipped_and_nonestimable_intervals_are_explicit(case):
    row = next(r for r in case['summary']['selections'] if r['method'] == 'FAIRBIAS_BM' and r['tau'] == .1)
    row['balanced_accuracy']['taylor_95'].update(lower=-.1, upper=1.2)
    row['eo_gap']['projection_95'] = {'status': 'NOT_ESTIMABLE', 'lower': None, 'upper': None}
    row['evaluation_status'] = 'PARTIALLY_ESTIMABLE'
    case['summary']['paired_contrasts'][0]['delta_balanced_accuracy'].update(lower=-1.2, upper=1.2)
    _reseal(case)
    output = _export(case)
    bm = next(r for r in _csv(output['csv']) if r['method'] == 'FAIRBIAS_BM')
    assert bm['ba_taylor_lower'] == '-0.1' and bm['ba_taylor_upper'] == '1.2'
    assert bm['eo_projection_lower'] == bm['eo_projection_upper'] == ''
