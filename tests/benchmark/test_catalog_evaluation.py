"""Synthetic catalog -> study -> explicit release -> annual-domain T -> merge."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from nhis_fairbias.benchmark import catalog_evaluation as ev
from nhis_fairbias.benchmark import catalog_selection as selection, result_catalog as catalog
from nhis_fairbias.benchmark.predictions import FrozenDecisionPolicy
from nhis_fairbias.benchmark.experiment_selection import freeze_selection
from test_result_catalog import write, sha, seal
from test_frozen_evaluation_contract import _annual


class SyntheticPreprocessor:
    def transform(self, X):
        return np.asarray(X)


class SyntheticAdapter:
    def __init__(self, q, output='event_probability_p'):
        self.q = np.asarray(q, dtype=float)
        self.output_type = output

    def predict_decision_proba(self, X, A=None):
        assert len(X) == len(self.q)
        return self.q

    def fit(self, *a, **k):
        pytest.fail('evaluation must never fit')


class SyntheticBaseAdapter(SyntheticAdapter):
    def predict_event_probability(self, X, A=None):
        return np.tile([.2, .8], len(X) // 2)


def _policy(q, output='event_probability_p', base=False):
    p = FrozenDecisionPolicy()
    p._adapter = (SyntheticBaseAdapter if base else SyntheticAdapter)(q, output)
    p._threshold = .5 if output == 'event_probability_p' else None
    return p


def _ref(path, root):
    return {'root': 'evidence', 'path': str(path.relative_to(root))}


@pytest.fixture
def study_case(tmp_path):
    root = tmp_path
    run = root / 'archive/original'
    run.mkdir(parents=True)
    source = root / 'src/registered.py'
    source.parent.mkdir()
    source.write_text('# generated source fixture\n')
    sources = {'src/registered.py': sha(source)}
    source_id = catalog._identity(sources)
    data = run / 'prepared/arm_001.joblib'
    data.parent.mkdir()
    joblib.dump({'synthetic': True}, data)
    configs, jobs = [], {}
    # Tuned BM differs from fixed epsilon=1 anchor; one fixed AE seed fails.
    specifications = [
        ('bm', 'FAIRBIAS_BM', {'epsilon_ratio': .5}, None, [.9, .9]),
        ('anchor', 'FAIRBIAS_BM', {'epsilon_ratio': 1., 'C': 1.}, None, [.7, .7]),
        ('to', 'TO_EO', {}, None, [.8, .8]),
        ('base', 'UNMITIGATED', {}, None, [.75, .75]),
        ('ae', 'FAIRBIAS_BM_AE', {}, 'fixed_ae_ablation', [.8, None]),
        ('geometry', 'FAIRBIAS_GEOMETRY_UNIT', {}, 'fixed_geometry_ablation', [.6, .6]),
    ]
    for cid, method, params, family, scores in specifications:
        config = dict(candidate_id=cid, method=method, arm_id='arm_001', backbone='LR', status='REGISTERED',
            params=params, training_weighted=False, complexity=1, seeds=[0, 1])
        if family:
            config['family'] = family
        configs.append(config)
        for seed, score in enumerate(scores):
            jid = f'{cid}_s{seed}'
            directory = run / 'jobs' / jid
            output = 'decision_probability_q' if method == 'TO_EO' else 'event_probability_p'
            job = dict(config=config, seed=seed, source_identity=source_id, data_path=str(data),
                data_identity='generated_data', data_sha256=sha(data), cache_path=str(run / 'representation_cache'))
            write(directory / 'job.json', job)
            result = dict(candidate_id=cid, seed=seed, status='VALID' if score else 'FAILED', source_identity=source_id,
                data_identity='generated_data', reload_verified=bool(score), output_type=output)
            if score:
                result.update(metrics_S=dict(balanced_accuracy=score, dp_gap=.02, eo_gap=.02), risk_S=dict(average_precision=score))
                q = np.tile([1., 1., 0., 0.] if seed == 0 else [0., 0., 1., 1.], 2) if cid == 'bm' else np.tile([.2, .8], 4)
                joblib.dump(dict(semantic_input=method.startswith('FAIRBIAS'), preprocessor=SyntheticPreprocessor(),
                    config=config, seed=seed, data_identity='generated_data', policy=_policy(q, output, method == 'TO_EO')), directory / 'model.joblib')
                np.savez(directory / 'predictions_S.npz', q=q)
            write(directory / 'result.json', result)
            (directory / 'worker.log').write_text('generated fixture\n')
            seal(directory, parallel=False, status=result['status'])
            jobs[jid] = directory
    execution = dict(fit_seconds=1800., worker_rss_bytes=4 * 1024**3)
    registration = dict(version='codex_application_v1_20260916', source_identity=source_id, source_files=sources,
        candidates=configs, pending_predeclared_extensions=[], registered_extensions=['BM_AE_FIXED', 'ARM004_GEOMETRY_FIXED'],
        prepared={'arm_001': dict(path=str(data), data_identity='generated_data', sha256=sha(data))},
        resources={**execution, 'threads': 1, 'concurrent_fits': 1})
    regpath = write(run / 'registration.json', registration)
    version = dict(method_version='registered_v1', role='original', budget_policy={'name': 'registered_v1', 'execution': execution})
    manifest = dict(schema_version=catalog.MANIFEST_SCHEMA, evaluation_authorized=False, roots={'evidence': str(root)}, relocations=[{'recorded_prefix': str(root), 'target': {'root': 'evidence', 'path': '.'}}],
        registrations=[dict(id='registration', path=_ref(regpath, root), sha256=sha(regpath), source_root={'root': 'evidence', 'path': '.'})],
        runs=[dict(id='original', registration_id='registration', directory=_ref(run, root), profile='serial_v1', lifecycle='closed', expected_jobs=list(jobs), **version)])
    manifest_path = write(root / 'manifest.json', manifest)
    snapshot = catalog.build_catalog(manifest_path)
    assert snapshot['integrity_passed'], snapshot['issues']
    catalog_path = write(root / 'catalog.json', snapshot)
    admission = dict(schema_version=selection.ADMISSION_SCHEMA, decision='SUPERVISOR_ADMITTED', evaluation_authorized=False,
        roots={'evidence': str(root)}, catalog={'path': _ref(catalog_path, root), 'sha256': sha(catalog_path)},
        catalog_manifest={'path': _ref(manifest_path, root), 'sha256': sha(manifest_path)},
        registration={'id': 'registration', 'sha256': sha(regpath)},
        methods={method: dict(analysis_version='registered_v1', versions=[{**version, 'runtime_source_identity': None}]) for _, method, *_ in specifications},
        resolved_jobs={jid: 'original' for jid in jobs}, extension_resolutions={})
    admission_path = write(root / 'admission.json', admission)
    selected = selection.freeze_catalog_selection(admission_path)
    selection_path = write(root / 'selection.json', selected)
    t_source = root / 'synthetic_t.txt'
    t_source.write_text('generated non-NHIS placeholder; loader replaced with generated annual design\n')
    study_registry = write(root / 'configs/nhis/study.json', {'years': {'2024': {'local_csv_file': t_source.name, 'expected_raw_rows': 8}}})
    features = write(root / 'configs/nhis/features.json', {'generated': True})
    provenance = dict(raw_sources={'2024': dict(path=t_source.name, sha256=sha(t_source), rows=8)},
        identity='official HHX + survey year', study_registry_sha256=sha(study_registry), feature_registry_sha256=sha(features))
    study_path = root / 'study.json'
    frozen = ev.freeze_catalog_study(selection_path, sha(selection_path), study_path, t_source_provenance=provenance, extension_resolutions={})
    release = dict(schema_version=ev.RELEASE_SCHEMA, decision='SUPERVISOR_RELEASED_T', study_freeze_sha256=sha(study_path),
        selection_sha256=sha(selection_path), study_branch=ev.STUDY_BRANCH, arm_ids=list(ev.ARM_SPECS))
    release_path = write(root / 'release.json', release)
    return dict(root=root, run=run, source=source, data=data, jobs=jobs, study=frozen, study_path=study_path,
        selection_path=selection_path, release_path=release_path, provenance=provenance, t_source=t_source, registration=registration)


def _loader(case, monkeypatch, T=None):
    seen = []
    T = T or _annual()
    cohort = SimpleNamespace(attrs={'source_provenance': case['provenance']})
    def load(root, *, years):
        assert root == case['root'] and years == (2024,)
        seen.append(years)
        return cohort
    monkeypatch.setattr(ev, 'load_local_nhis_cohort', load)
    monkeypatch.setattr(ev, 'load_arm_partitions', lambda cohort, arm: {'evaluation_T': T})
    monkeypatch.setattr(ev, 'summarize_eligibility', lambda cohort: {'synthetic': True})
    return seen


def _evaluate(case, output, arm='arm_001'):
    return ev.evaluate_catalog_arm(case['study_path'], sha(case['study_path']), case['release_path'], sha(case['release_path']),
        output, arm_id=arm, data_root=case['root'])


def test_freeze_fixed_anchors_all_seeds_and_no_t_io(study_case, monkeypatch):
    case = study_case
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('metadata must not deserialize'))
    monkeypatch.setattr(ev, '_verify_t_inputs', lambda *a: pytest.fail('metadata must not read T'))
    study = ev.validate_catalog_study(case['study_path'], sha(case['study_path']))
    fixed = {s['registered_fixed_candidate_id']: s for s in study['selections'] if 'registered_fixed_candidate_id' in s}
    assert set(fixed) == {'anchor', 'ae', 'geometry'}
    assert fixed['ae']['status'] == 'NO_VALID_FIXED_ABLATION' and fixed['ae']['model_ids'] == []
    assert len(fixed['anchor']['model_ids']) == len(fixed['geometry']['model_ids']) == 2
    assert any(r.get('selected_candidate_id') == 'bm' for r in study['selections'])
    assert len(study['selected_models']) == 10
    assert len(study['resolved_jobs']) == 12
    assert not study['evaluation_authorized']


@pytest.mark.parametrize('damage', ['release', 'model', 'source', 'prepared', 'selection', 'modelset', 'analysis', 'branch', 'extensions'])
def test_fail_closed_before_t_io(study_case, monkeypatch, damage):
    c = study_case
    monkeypatch.setattr(ev, '_verify_t_inputs', lambda *a: pytest.fail('T I/O before complete validation'))
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('model loaded before complete validation'))
    if damage in {'model', 'source', 'prepared', 'selection'}:
        path = {'model': c['jobs']['bm_s0'] / 'model.joblib', 'source': c['source'], 'prepared': c['data'], 'selection': c['selection_path']}[damage]
        path.write_bytes(path.read_bytes() + b'changed')
    elif damage == 'release':
        r = ev._read(c['release_path']); r['decision'] = 'NOT_RELEASED'; write(c['release_path'], r)
    else:
        s = ev._read(c['study_path'])
        if damage == 'modelset': s['expected_modelsets']['arm_001'].pop()
        if damage == 'analysis': s['analysis_files'].pop(next(iter(s['analysis_files'])))
        if damage == 'branch': s['study_branch'] = 'bmae_cap40_sensitivity'
        if damage == 'extensions': s['extension_resolutions']['invented'] = {'status': 'NOT_SUPPORTED', 'evidence_files': []}
        write(c['study_path'], s)
    with pytest.raises((ValueError, OSError)):
        _evaluate(c, c['root'] / 'rejected')
    assert not (c['root'] / 'rejected').exists()


def test_end_to_end_seed_eo_q_only_base_p_and_shared_bootstrap(study_case, monkeypatch):
    c = study_case
    seen = _loader(c, monkeypatch)
    out = c['root'] / 'evaluated'
    summary = _evaluate(c, out)
    assert seen == [(2024,)]
    bm = next(r for r in summary['selections'] if r.get('selected_candidate_id') == 'bm')
    assert bm['eo_gap']['mean'] == 1.  # mean-q would incorrectly give zero
    assert bm['balanced_accuracy']['mean'] == .5
    assert bm['balanced_accuracy']['taylor_95']['estimate'] == .5
    to = next(r for r in summary['selections'] if r['method'] == 'TO_EO')
    assert to['average_precision']['status'] == 'NOT_ESTIMABLE'
    assert to['untouched_base_average_precision']['status'] == 'VALID'
    assert to['unweighted_untouched_base_average_precision']['status'] == 'VALID'
    assert summary['paired_contrasts'] and all(p['reference_selection_id'] and p['comparison_selection_id'] for p in summary['paired_contrasts'])
    manifest = ev._read(out / 'evaluation_manifest.json')
    assert manifest['expected_modelsets']['arm_001'] == c['study']['expected_modelsets']['arm_001']
    assert manifest['bootstrap_replicates'] == 2000 and manifest['primary_contrast_family_size'] == 20
    arrays = np.load(out / 'arm_001_replicate_metric_arrays.npz')
    assert set(arrays.files) == set(c['study']['expected_modelsets']['arm_001'])
    assert all(arrays[k].shape == (2000, 3) for k in arrays.files)
    for ref in c['study']['selected_models']:
        prediction = np.load(out / ('arm_001_' + ref['model_id'] + '_predictions_T.npz'))
        assert ('base_p' in prediction) == (ref['method'] == 'TO_EO')
        assert ('p' in prediction) == (ref['method'] != 'TO_EO')
    with pytest.raises(ValueError, match='already exists'):
        _evaluate(c, out)


@pytest.mark.parametrize('mode,expected', [('singleton', 'DESIGN_NOT_ESTIMABLE'), ('missing', 'PARTIALLY_ESTIMABLE')])
def test_nonestimability_survives(study_case, monkeypatch, mode, expected):
    c = study_case
    _loader(c, monkeypatch, _annual(singleton=mode == 'singleton', missing_group=mode == 'missing'))
    summary = _evaluate(c, c['root'] / mode)
    assert {r['evaluation_status'] for r in summary['selections'] if r['model_ids']} == {expected}
    if mode == 'singleton':
        assert all(r['balanced_accuracy']['taylor_95']['status'] == 'NOT_ESTIMABLE' for r in summary['selections'] if r['model_ids'])


def test_exact_legacy_statistical_parity(study_case, monkeypatch):
    c = study_case
    seen = _loader(c, monkeypatch)
    current = _evaluate(c, c['root'] / 'new')
    old = ev.legacy
    monkeypatch.setattr(old, 'load_local_nhis_cohort', ev.load_local_nhis_cohort)
    monkeypatch.setattr(old, 'load_arm_partitions', ev.load_arm_partitions)
    monkeypatch.setattr(old, 'summarize_eligibility', ev.summarize_eligibility)
    freeze_selection(c['run'])
    old.freeze_study([c['run']], c['root'] / 'old_study.json', repo_root=c['root'])
    old.evaluate_frozen_run(c['run'], c['root'] / 'old', study_freeze=c['root'] / 'old_study.json', repo_root=c['root'])
    previous = ev._read(c['root'] / 'old/summary_T.json')
    fields = ('balanced_accuracy', 'dp_gap', 'eo_gap', 'average_precision', 'auroc', 'brier',
              'unweighted_balanced_accuracy', 'unweighted_eo_gap', 'untouched_base_average_precision')
    identify = lambda r: (r['arm_id'], r['method'], r['tau'], r['status'], r.get('selected_candidate_id'), r.get('boundary_candidate_id'))
    expected = {identify(r): r for r in previous['selections']}
    assert len(expected) == len(current['selections'])
    for row in current['selections']:
        prior = expected[identify(row)]
        for field in fields:
            assert row.get(field) == prior.get(field), field
    strip = lambda p: {k: v for k, v in p.items() if k not in {'reference_selection_id', 'comparison_selection_id', 'reference_model_ids', 'comparison_model_ids', 'study_branch'}}
    assert [strip(p) for p in current['paired_contrasts']] == previous['paired_contrasts']
    assert seen == [(2024,), (2024,)]


def test_merge_complete_arms_no_inference_and_tamper_rejection(study_case, monkeypatch):
    c = study_case
    _loader(c, monkeypatch)
    shards = []
    for arm in ev.ARM_SPECS:
        out = c['root'] / arm
        _evaluate(c, out, arm)
        shards.append(ev._bound(out / 'summary_T.json'))
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('merge cannot load models'))
    monkeypatch.setattr(ev, 'load_local_nhis_cohort', lambda *a, **k: pytest.fail('merge cannot load T'))
    kwargs = dict(release_path=c['release_path'], expected_release_sha256=sha(c['release_path']))
    merged = ev.merge_catalog_evaluations(c['study_path'], sha(c['study_path']), shards, c['root'] / 'merged', **kwargs)
    assert len(merged['selections']) == len(c['study']['selections'])
    manifest = ev._read(c['root'] / 'merged/evaluation_manifest.json')
    assert manifest['scope'] == 'complete_study' and manifest['selected_models'] == c['study']['selected_models']
    for damaged in (shards[:-1], [shards[0]] * 4):
        with pytest.raises(ValueError):
            ev.merge_catalog_evaluations(c['study_path'], sha(c['study_path']), damaged, c['root'] / 'bad_merge', **kwargs)
    value = ev._read(shards[0]['path'])
    value['selections'][0]['model_ids'].pop()
    write(shards[0]['path'], value)
    with pytest.raises(ValueError):
        ev.merge_catalog_evaluations(c['study_path'], sha(c['study_path']), shards, c['root'] / 'tampered', **kwargs)


def test_t_hash_mismatch_prevents_loader_and_model_load(study_case, monkeypatch):
    c = study_case
    c['t_source'].write_text('different generated bytes')
    monkeypatch.setattr(ev, 'load_local_nhis_cohort', lambda *a, **k: pytest.fail('wrong T bytes loaded'))
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('wrong T admitted'))
    with pytest.raises(ValueError, match='hash mismatch'):
        _evaluate(c, c['root'] / 'bad_t')


def test_hash_only_t_freeze_and_supplemental_branch_evidence(study_case, monkeypatch):
    c = study_case
    provenance = copy.deepcopy(c['provenance'])
    provenance['raw_sources']['2024'].pop('rows')
    evidence = write(c['root'] / 'supplemental_registration.json', {'synthetic_budget_sensitivity': 'cap10_then40'})
    supplemental = [{'study_branch': 'bmae40_sensitivity', 'status': 'AWAITING_SEPARATE_FROZEN_EVALUATION', 'evidence_files': [ev._bound(evidence)]}]
    path = c['root'] / 'hash_only_study.json'
    frozen = ev.freeze_catalog_study(c['selection_path'], sha(c['selection_path']), path,
        t_source_provenance=provenance, extension_resolutions={}, supplemental_evidence=supplemental)
    assert frozen['supplemental_evidence'] == supplemental
    assert frozen['selected_models'] == c['study']['selected_models']
    c['study_path'] = path
    r = ev._read(c['release_path']); r['study_freeze_sha256'] = sha(path); write(c['release_path'], r)
    _loader(c, monkeypatch)
    _evaluate(c, c['root'] / 'hash_only_evaluation')
    manifest = ev._read(c['root'] / 'hash_only_evaluation/evaluation_manifest.json')
    assert 'rows' not in manifest['source_provenance']['raw_sources']['2024']
    assert manifest['observed_source_provenance']['raw_sources']['2024']['rows'] == 8
    evidence.write_text('changed evidence')
    with pytest.raises(ValueError, match='hash mismatch'):
        ev.validate_catalog_study(path, sha(path))


def test_unreleased_arm_and_changed_during_prediction_leave_no_complete_manifest(study_case, monkeypatch):
    c = study_case
    release = ev._read(c['release_path']); release['arm_ids'] = ['arm_002']; write(c['release_path'], release)
    with monkeypatch.context() as guarded:
        guarded.setattr(ev, '_verify_t_inputs', lambda *a: pytest.fail('unreleased T access'))
        with pytest.raises(ValueError, match='Arm not released'):
            _evaluate(c, c['root'] / 'unreleased')
    release['arm_ids'] = list(ev.ARM_SPECS); write(c['release_path'], release)
    _loader(c, monkeypatch)
    original = ev.catalog_inference.predict_frozen_bundle
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        c['t_source'].write_text('changed during inference')
        return result
    monkeypatch.setattr(ev.catalog_inference, 'predict_frozen_bundle', changed)
    with pytest.raises(ValueError, match='hash mismatch'):
        _evaluate(c, c['root'] / 'partial')
    assert (c['root'] / 'partial').exists()
    assert not (c['root'] / 'partial/evaluation_manifest.json').exists()
    assert not (c['root'] / 'partial/summary_T.json').exists()


def test_distinct_admission_and_catalog_roots_are_resolved(study_case):
    c = study_case
    admission_path = c['root'] / 'admission.json'
    a = ev._read(admission_path)
    a['roots'] = {'intake': str(c['root'])}
    for key in ('catalog', 'catalog_manifest'):
        a[key]['path']['root'] = 'intake'
    write(admission_path, a)
    selected = selection.freeze_catalog_selection(admission_path)
    path = write(c['root'] / 'distinct_selection.json', selected)
    result = ev.freeze_catalog_study(path, sha(path), c['root'] / 'distinct_study.json',
        t_source_provenance=c['provenance'], extension_resolutions={})
    assert result['catalog']['path'] == str(c['root'] / 'catalog.json')
    assert result['selected_models'] == c['study']['selected_models']


def test_full_annual_domain_reaches_shared_linearization_and_bootstrap(study_case, monkeypatch):
    from nhis_fairbias.benchmark.data_contracts import AnnualSurveyDesign
    c = study_case
    T = _annual()
    mask = np.r_[np.ones(8, dtype=bool), np.zeros(4, dtype=bool)]
    T.annual_design = AnnualSurveyDesign(2024, np.arange(12), np.r_[T.PSTRAT, [3, 3, 3, 3]],
        np.r_[T.PPSU, [30, 30, 31, 31]], np.ones(12), mask)
    _loader(c, monkeypatch, T)
    seen = []
    original = ev.bootstrap_metric_arrays
    def bootstrap(y, qs, A, strata, psus, weights, groups, **kwargs):
        assert len(y) == len(A) == len(strata) == len(psus) == len(weights) == 12
        assert all(len(q) == 12 for q in qs.values())
        assert kwargs['domain_mask'].tolist() == mask.tolist()
        seen.append(set(qs))
        return original(y, qs, A, strata, psus, weights, groups, **kwargs)
    monkeypatch.setattr(ev, 'bootstrap_metric_arrays', bootstrap)
    result = _evaluate(c, c['root'] / 'domain')
    assert seen == [set(c['study']['expected_modelsets']['arm_001'])]
    assert all(r['sample_n'] == 8 and r['design_n'] == 12 and r['df'] == 3 for r in result['selections'])


def test_cli_freeze_and_missing_release_are_explicit(study_case):
    import os
    import subprocess
    import sys
    c = study_case
    provenance = write(c['root'] / 't_provenance.json', c['provenance'])
    extensions = write(c['root'] / 'extensions.json', {})
    supplemental = write(c['root'] / 'supplementals.json', [])
    output = c['root'] / 'cli_study.json'
    env = dict(os.environ, PYTHONPATH=str(ev.ROOT / 'src'))
    script = ev.ROOT / 'scripts/evaluate_nhis_catalog.py'
    command = [sys.executable, str(script), 'freeze', '--selection', str(c['selection_path']), '--selection-sha', sha(c['selection_path']),
        '--t-provenance', str(provenance), '--extension-resolutions', str(extensions), '--supplemental-evidence', str(supplemental), '--output', str(output)]
    process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    assert ev._read(output) == c['study']
    no_release = subprocess.run([sys.executable, str(script), 'evaluate-arm', '--study-freeze', str(output), '--study-sha', sha(output),
        '--arm', 'arm_001', '--data-root', str(c['root']), '--output', str(c['root'] / 'forbidden')], env=env, capture_output=True, text=True, timeout=10)
    assert no_release.returncode == 2 and '--release' in no_release.stderr
    assert not (c['root'] / 'forbidden').exists()


@pytest.mark.parametrize('damage', ['not_dict', 'config', 'seed', 'bool_seed', 'data_identity', 'semantic_input', 'int_semantic', 'preprocessor'])
def test_bound_model_artifact_schema_checked_before_prediction(study_case, monkeypatch, damage):
    c = study_case
    _loader(c, monkeypatch)
    original = joblib.load
    def wrong(path, *args, **kwargs):
        model = original(path, *args, **kwargs)
        if damage == 'not_dict': return []
        if damage == 'config': model['config']['params']['invented'] = 1
        elif damage == 'seed': model['seed'] = 12
        elif damage == 'bool_seed': model['seed'] = False
        elif damage == 'data_identity': model['data_identity'] = 'other'
        elif damage == 'semantic_input': model['semantic_input'] = not model['semantic_input']
        elif damage == 'int_semantic': model['semantic_input'] = int(model['semantic_input'])
        elif damage == 'preprocessor': model['preprocessor'] = None
        return model
    monkeypatch.setattr(joblib, 'load', wrong)
    original_predict = ev.catalog_inference.predict_frozen_bundle
    def guarded(model, *args, **kwargs):
        assert damage == 'preprocessor' and model['semantic_input']
        return original_predict(model, *args, **kwargs)
    monkeypatch.setattr(ev.catalog_inference, 'predict_frozen_bundle', guarded)
    with pytest.raises(ValueError, match='Frozen model|Missing frozen preprocessor'):
        _evaluate(c, c['root'] / 'wrong_artifact')
    assert not (c['root'] / 'wrong_artifact/evaluation_manifest.json').exists()


def test_validation_context_scans_catalog_exactly_once(study_case, monkeypatch):
    count = []
    original = catalog.build_catalog
    def scan(*args, **kwargs):
        count.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(catalog, 'build_catalog', scan)
    ev.validate_catalog_study(study_case['study_path'], sha(study_case['study_path']))
    assert count == [1]
