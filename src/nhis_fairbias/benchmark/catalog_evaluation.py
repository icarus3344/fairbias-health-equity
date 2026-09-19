"""Hash-bound catalog evaluation after a separate, explicit supervisor T release.

Only complete-arm shards are supported: all frozen seed models enter the same
survey-rate family and PSU bootstrap. No model is fitted or selected here.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
from scipy import stats

from . import catalog_selection, experiment_evaluation as legacy
from . import catalog_inference
from .data_contracts import ARM_SPECS, load_arm_partitions, load_local_nhis_cohort
from .experiment_worker import file_sha, write_json
from .metrics import compute_survey_fairness_metrics
from .predictions import PredictionBundle
from .risk_metrics import compute_risk_metrics
from .survey_batch import bootstrap_metric_arrays
from .survey_linearization import linearized_survey_inference
from .cohort_reporting import summarize_partition, summarize_eligibility

STUDY_SCHEMA = 'nhis_catalog_study_freeze_v1'
RELEASE_SCHEMA = 'nhis_catalog_t_release_v1'
MANIFEST_SCHEMA = 'nhis_catalog_evaluation_manifest_v1'
SUMMARY_SCHEMA = 'nhis_catalog_summary_T_v1'
STUDY_BRANCH = 'registered_budget_main'
B = 2000
BOOTSTRAP_SEED = 20260914
FAMILY_SIZE = 20
ROOT = Path(__file__).resolve().parents[3]
_REQUIRED_ANALYSIS = {f'src/nhis_fairbias/benchmark/{name}.py' for name in (
    'catalog_evaluation', 'catalog_selection', 'result_catalog', 'catalog_inference',
    'catalog_reporting', 'experiment_evaluation', 'experiment_selection', 'experiment_registry',
    'survey_batch', 'survey_linearization', 'survey_inference', 'metrics', 'risk_metrics',
    'data_contracts', 'cohort_reporting', 'paper_reporting', 'predictions')}
_STATISTICS = dict(bootstrap_replicates=B, bootstrap_seed=BOOTSTRAP_SEED,
    primary_contrast_family_size=FAMILY_SIZE,
    annual_design='complete_2024_annual_design_with_arm_domain',
    seed_aggregation='mean_seed_metrics; BA mean-q; EO simultaneous per-seed projections',
    resampling='common PSU factors across every frozen model within each arm',
    conditional_on_fixed_models=True)
_summary = legacy._summary
_replicate_interval = legacy._replicate_interval
_project_mean = legacy._project_mean
_projection_difference = legacy._projection_difference
_linear_record = legacy._linear_record


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _bound(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_sha(path)}


def _verify(item):
    _require(isinstance(item, dict) and set(item) == {'path', 'sha256'} and _sha(item['sha256']), 'Invalid bound file')
    path = Path(item['path'])
    _require(path.is_absolute() and path.is_file() and file_sha(path) == item['sha256'], 'Bound file hash mismatch')
    return path


def _fresh_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation; never truncate a historical artifact.
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def _environment():
    packages = ('numpy', 'scipy', 'pandas', 'scikit-learn', 'joblib', 'fairlearn', 'lightgbm', 'torch')
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'python': sys.version, 'packages': versions,
        'platform': platform.platform(), 'system': platform.system(), 'machine': platform.machine(),
        'execution_environment': {key: os.environ.get(key) for key in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'PYTHONHASHSEED')}}


def _analysis(root):
    root = Path(root).resolve()
    _require(root == ROOT, 'Analysis must bind the executing checkout')
    # Include transitive local prediction and loader dependencies, not only the
    # top-level evaluator. Changes after this freeze require a new study freeze.
    paths = sorted(p for package in ('nhis_fairbias', 'fairbias')
                   for p in (root / 'src' / package).rglob('*.py'))
    result = {p.relative_to(root).as_posix(): file_sha(p) for p in paths}
    cli = root / 'scripts/evaluate_nhis_catalog.py'
    if cli.is_file():
        result['scripts/evaluate_nhis_catalog.py'] = file_sha(cli)
    _require(_REQUIRED_ANALYSIS <= set(result), 'Incomplete analysis source closure')
    for name, module in tuple(sys.modules.items()):
        if name == 'nhis_fairbias' or name.startswith('nhis_fairbias.') or name == 'fairbias' or name.startswith('fairbias.'):
            origin = getattr(module, '__file__', None)
            if origin is not None:
                p = Path(origin).resolve()
                _require(p.is_relative_to(root) and p.relative_to(root).as_posix() in result,
                         'Imported project module has an unbound origin')
    return result


def _validate_t_provenance(value):
    _require(isinstance(value, dict) and set(value) == {'raw_sources', 'identity', 'feature_registry_sha256', 'study_registry_sha256'},
             'Exact T provenance is required')
    _require(value['identity'] == 'official HHX + survey year' and set(value['raw_sources']) == {'2024'}, 'Only registered 2024 T is permitted')
    source = value['raw_sources']['2024']
    _require(set(source) in ({'path', 'sha256'}, {'path', 'sha256', 'rows'}) and isinstance(source['path'], str)
             and not Path(source['path']).is_absolute() and '..' not in Path(source['path']).parts
             and ('rows' not in source or (type(source['rows']) is int and source['rows'] > 0)), 'Invalid annual T source declaration')
    _require(all(_sha(v) for v in (source['sha256'], value['feature_registry_sha256'], value['study_registry_sha256'])), 'Invalid T hashes')


def _fixed(config):
    params = config.get('params', {})
    base = params.get('estimator_params', {})
    bm = (config['method'] == 'FAIRBIAS_BM' and not config.get('training_weighted') and params.get('epsilon_ratio') == 1.
        and ((config['backbone'] == 'LR' and params.get('C', 1.) == 1.) or
             (config['backbone'] == 'GBDT' and base.get('n_estimators') == 100 and base.get('max_depth') == 2)))
    return bm or config.get('family') in {'fixed_ae_ablation', 'fixed_geometry_ablation'}


def _model_id(ref):
    # Includes run/runtime/version, not candidate+seed alone.
    return 'model_' + _digest(ref)


def _build_study(selection_path, selection_sha, *, t_source_provenance, extension_resolutions, supplemental_evidence, analysis_root):
    _validate_t_provenance(t_source_provenance)  # metadata only, no T file access
    selection_path = _verify({'path': str(Path(selection_path).resolve()), 'sha256': selection_sha})
    resolved, context = catalog_selection.resolve_selected_models_context(selection_path, selection_sha)
    selection = _read(selection_path)
    registration, files = context['registration'], context['files']
    _require(set(extension_resolutions) == set(registration.get('pending_predeclared_extensions', [])), 'Unresolved predeclared extensions')
    for key, resolution in extension_resolutions.items():
        _require(set(resolution) == {'status', 'evidence_files'} and resolution['status'] in {'ADMITTED_AND_FROZEN', 'NOT_SUPPORTED'}
                 and isinstance(resolution['evidence_files'], list) and resolution['evidence_files'], 'Final extension evidence required')
        admitted = selection['extension_resolutions'][key]['status'] == 'INCLUDED'
        _require(admitted == (resolution['status'] == 'ADMITTED_AND_FROZEN'), 'Extension disposition changed')
        for evidence in resolution['evidence_files']:
            _verify(evidence)
    _require(isinstance(supplemental_evidence, list), 'Invalid supplemental evidence')
    branches = set()
    for branch in supplemental_evidence:
        _require(set(branch) == {'study_branch', 'status', 'evidence_files'}
            and isinstance(branch['study_branch'], str) and branch['study_branch'] != STUDY_BRANCH
            and branch['study_branch'] not in branches
            and branch['status'] == 'AWAITING_SEPARATE_FROZEN_EVALUATION'
            and isinstance(branch['evidence_files'], list) and branch['evidence_files'], 'Invalid supplemental branch disposition')
        branches.add(branch['study_branch'])
        for evidence in branch['evidence_files']:
            _verify(evidence)
    selections = [dict(row) for row in selection['selections']]
    models = {ref['job_id']: ref for ref in resolved['models']}
    fixed_rows = []
    for config in registration['candidates']:
        if not _fixed(config):
            continue
        ids = [f"{config['candidate_id']}_s{seed}" for seed in config.get('seeds', [])]
        valid = config['status'] == 'REGISTERED' and bool(ids) and all(context['rows'][key]['status'] == 'VALID' for key in ids)
        fixed_rows.append(dict(arm_id=config['arm_id'], backbone=config['backbone'], method=config['method'],
            training_weighted=config['training_weighted'], tau=None,
            analysis_version=selection['methods'][config['method']]['analysis_version'],
            status='FIXED_ABLATION' if valid else 'NO_VALID_FIXED_ABLATION',
            selected_candidate_id=config['candidate_id'] if valid else None,
            registered_fixed_candidate_id=config['candidate_id']))
        if valid:
            for key in ids:
                models[key] = catalog_selection._model_reference(context, context['rows'][key])
    selections += fixed_rows
    refs = [{**ref, 'model_id': _model_id(ref)} for _, ref in sorted(models.items())]
    _require(all(row['arm_id'] in ARM_SPECS for row in selections), 'Unregistered evaluation arm')
    for row in selections:
        cid = row.get('selected_candidate_id') or row.get('boundary_candidate_id')
        row['study_branch'] = STUDY_BRANCH
        row['model_ids'] = [ref['model_id'] for ref in refs if ref['candidate_id'] == cid]
        if cid:
            config = context['configs'][cid]
            actual = [ref['seed'] for ref in refs if ref['candidate_id'] == cid]
            _require(sorted(actual) == sorted(config['seeds']) and len(actual) == len(set(actual)), 'Incomplete frozen seed set')
        row['selection_id'] = 'selection_' + _digest(row)
    _require(len({row['selection_id'] for row in selections}) == len(selections), 'Duplicate selection identity')
    admission = context['admission']
    admission_files = catalog_selection.catalog._Files(admission['roots'], [])
    jobs = []
    for key, row in sorted(context['rows'].items()):
        directory = Path(row['directory'])
        jobs.append({**row, 'job_path': str(directory / 'job.json'), 'result_path': str(directory / 'result.json')})
    result = dict(schema_version=STUDY_SCHEMA, status='STUDY_FREEZE_BUILT_REVIEW_REQUIRED', evaluation_authorized=False,
        study_branch=STUDY_BRANCH, selection=_bound(selection_path), admission=_bound(context['admission_path']),
        catalog=_bound(admission_files.ref(admission['catalog']['path'])), catalog_manifest=_bound(admission_files.ref(admission['catalog_manifest']['path'])),
        registration=_bound(context['registration_path']), registration_id=admission['registration']['id'],
        analysis_root=str(Path(analysis_root).resolve()), analysis_files=_analysis(analysis_root), environment=_environment(),
        selected_models=refs, selections=selections, resolved_jobs=jobs,
        expected_modelsets={arm: [ref['model_id'] for ref in refs if ref['arm_id'] == arm] for arm in ARM_SPECS},
        arm_ids=list(ARM_SPECS), methods=selection['methods'], compatibility_gates=selection['compatibility_gates'],
        supplemental_evidence=supplemental_evidence, registered_extensions=registration.get('registered_extensions', []), extension_resolutions=extension_resolutions,
        t_source_provenance=t_source_provenance,
        inference_policies={method: ('EG_BOUNDED_PMF_V1' if method in {'EG_DP', 'EG_EO'} else 'FROZEN_POLICY_PREDICT_V1') for method in selection['methods']},
        **_STATISTICS)
    _verify(result['selection'])
    return result


def freeze_catalog_study(selection_path, expected_selection_sha256, output, *, t_source_provenance,
                         extension_resolutions, supplemental_evidence=None, analysis_root=None):
    """Create a review artifact; does not authorize or access T, or load models."""
    _require(not Path(output).exists(), 'Study freeze output already exists')
    result = _build_study(selection_path, expected_selection_sha256, t_source_provenance=t_source_provenance,
        extension_resolutions=extension_resolutions, supplemental_evidence=[] if supplemental_evidence is None else supplemental_evidence, analysis_root=analysis_root or ROOT)
    _fresh_json(output, result)
    return result


def validate_catalog_study(study_path, expected_study_sha256):
    """Rebuild all metadata bindings before release/T I/O; never decode T."""
    path = _verify({'path': str(Path(study_path).resolve()), 'sha256': expected_study_sha256})
    study = _read(path)
    _require(study.get('schema_version') == STUDY_SCHEMA and study.get('study_branch') == STUDY_BRANCH, 'Unsupported study schema/branch')
    expected = _build_study(study['selection']['path'], study['selection']['sha256'],
        t_source_provenance=study['t_source_provenance'], extension_resolutions=study['extension_resolutions'],
        supplemental_evidence=study['supplemental_evidence'], analysis_root=study['analysis_root'])
    _require(study == expected, 'Study freeze does not match complete current evidence')
    _verify({'path': str(path), 'sha256': expected_study_sha256})
    return study


def _release(study, study_sha, release_path, release_sha, arms):
    path = _verify({'path': str(Path(release_path).resolve()), 'sha256': release_sha})
    release = _read(path)
    required = {'schema_version', 'decision', 'study_freeze_sha256', 'selection_sha256', 'study_branch', 'arm_ids'}
    _require(set(release) == required and release['schema_version'] == RELEASE_SCHEMA
        and release['decision'] == 'SUPERVISOR_RELEASED_T' and release['study_freeze_sha256'] == study_sha
        and release['selection_sha256'] == study['selection']['sha256'] and release['study_branch'] == STUDY_BRANCH,
        'Explicit supervisor T release required')
    _require(isinstance(release['arm_ids'], list) and len(set(release['arm_ids'])) == len(release['arm_ids'])
        and set(arms) <= set(release['arm_ids']) <= set(study['arm_ids']), 'Arm not released')
    return _bound(path)


def _verify_t_inputs(study, root):
    # Called ONLY after all metadata checks AND the explicit release.
    root = Path(root).resolve()
    p = study['t_source_provenance']
    for relative, sha in [('configs/nhis/study.json', p['study_registry_sha256']),
                          ('configs/nhis/features.json', p['feature_registry_sha256'])]:
        _verify({'path': str(root / relative), 'sha256': sha})
    source = p['raw_sources']['2024']
    spec = _read(root / 'configs/nhis/study.json')['years']['2024']
    _require(spec['local_csv_file'] == source['path'] and ('rows' not in source or spec['expected_raw_rows'] == source['rows']), 'T registry declaration mismatch')
    actual = (root / source['path']).resolve()
    _require(actual.is_relative_to(root), 'T source escapes data root')
    _verify({'path': str(actual), 'sha256': source['sha256']})


def _validate_observed_provenance(expected, actual):
    _require(isinstance(actual, dict), 'Loaded T provenance missing')
    _validate_t_provenance(actual)
    _require('rows' in actual['raw_sources']['2024'], 'Loaded T row count missing')
    compared = json.loads(json.dumps(actual))
    if 'rows' not in expected['raw_sources']['2024']:
        compared['raw_sources']['2024'].pop('rows')
    _require(compared == expected, 'Loaded T provenance mismatch')


def _manifest(study, study_sha, release, arms, scope):
    return dict(schema_version=MANIFEST_SCHEMA, status='COMPLETE', scope=scope, study_branch=STUDY_BRANCH,
        study_freeze_sha256=study_sha, selection_sha256=study['selection']['sha256'], release=release,
        arm_ids=arms, expected_modelsets={a: study['expected_modelsets'][a] for a in arms},
        selected_models=[r for r in study['selected_models'] if r['arm_id'] in arms],
        analysis_files=study['analysis_files'], source_provenance=study['t_source_provenance'], **_STATISTICS)


def _summary_document(study_sha, selection_sha, manifest_path, summaries, paired):
    return dict(schema_version=SUMMARY_SCHEMA, study_branch=STUDY_BRANCH, study_freeze_sha256=study_sha,
        selection_sha256=selection_sha, selections=summaries, paired_contrasts=paired,
        evaluation_manifest_sha256=file_sha(manifest_path),
        intervals='EO bootstrap is descriptive; projection bands address nonsmooth max/range using simultaneous rate intervals',
        training_variation='seed SD/range is separate from design SE; no seed is selected')


def _normalized_annual_design(T):
    """Normalize nullable-pandas boolean storage without changing domain membership.

    The frozen training loader can return an object array of actual booleans.
    NumPy cannot use that array as an index. Reject missing or non-boolean
    values instead of inventing an exclusion rule or coercing truthy values.
    """
    design = T.annual_design
    raw = np.asarray(design.domain_mask)
    _require(raw.ndim == 1 and (raw.dtype.kind == 'b' or
        raw.dtype.kind == 'O' and all(isinstance(v, (bool, np.bool_)) for v in raw)),
        'Annual domain must contain only explicit boolean values')
    domain = raw.astype(bool, copy=True)
    _require(int(domain.sum()) == len(T), 'Annual domain count differs from partition')
    for annual_field, partition_field in (('record_keys', 'record_keys'),
            ('strata', 'PSTRAT'), ('psus', 'PPSU'), ('weights', 'WTFA_A')):
        annual = np.asarray(getattr(design, annual_field))
        partition = np.asarray(getattr(T, partition_field))
        _require(annual.ndim == partition.ndim == 1 and len(annual) == len(domain)
            and np.array_equal(annual[domain], partition), 'Annual domain row alignment mismatch')
    return dataclasses.replace(design, domain_mask=domain)


def _evaluate_arm_statistics(T, arm, arm_selections, q_models, individual, output):
    """Registered arithmetic from experiment_evaluation, with explicit model IDs."""
    design, spec = _normalized_annual_design(T), ARM_SPECS[arm]
    summaries, all_paired = [], []
    args = (design.expand(T.y), q_models, design.expand(T.A), design.strata, design.psus, design.weights, spec["expected_categories"])
    linear = linearized_survey_inference(*args, domain_mask=design.domain_mask, alpha=.05)
    adjusted = linearized_survey_inference(*args, domain_mask=design.domain_mask, alpha=.05/20)
    design_valid = linear["status"] == "VALID"
    replicates = (bootstrap_metric_arrays(*args, domain_mask=design.domain_mask, B=B) if design_valid else
                  {k: np.full((B, 3), np.nan) for k in q_models})
    for key in individual:
        individual[key]["simultaneous_rate_intervals_95"] = {
            label: dataclasses.asdict(metric) for label, metric in
            linear.get("methods", {}).get(key, {}).get("rates", {}).items()}
    # No serialization of per-person arrays into public aggregate JSON.
    write_json(output / (arm + "_individual_model_metrics.json"), individual)
    df = linear.get("degrees_of_freedom", 0)
    condition_replicates, condition_q, arm_rows = {}, {}, []
    for position, selection in enumerate(arm_selections):
        cid = selection.get("selected_candidate_id") or selection.get("boundary_candidate_id")
        keys = selection["model_ids"]
        row = {**selection, "sample_n": len(T), "design_n": len(design.weights), "df": df,
               "evaluation_status": ("VALID" if design_valid else "DESIGN_NOT_ESTIMABLE") if keys else "NO_VALID_MODEL", "frozen_seed_models": keys}
        if keys:
            mean_reps = np.mean([replicates[k] for k in keys], axis=0)
            for j, metric in enumerate(("balanced_accuracy", "dp_gap", "eo_gap")):
                row[metric] = _summary([individual[k]["weighted"][metric] for k in keys])
                row["unweighted_"+metric] = _summary([individual[k]["unweighted"][metric] for k in keys])
                row[metric]["bootstrap_descriptive"] = _replicate_interval(row[metric]["mean"], mean_reps[:, j], df)
            for metric in ("average_precision", "auroc", "brier"):
                values = [individual[k]["risk"].get(metric, np.nan) for k in keys]
                row[metric] = _summary(values)
                row["unweighted_"+metric] = _summary([individual[k]["unweighted_risk"].get(metric, np.nan) for k in keys])
                if all("base_risk" in individual[k] for k in keys):
                    row["untouched_base_"+metric] = _summary([individual[k]["base_risk"].get(metric, np.nan) for k in keys])
                    row["unweighted_untouched_base_"+metric] = _summary([individual[k]["unweighted_base_risk"].get(metric, np.nan) for k in keys])
            for metric in ("dp_gap", "eo_gap"):
                row[metric]["projection_95"] = _project_mean(linear, keys, metric)
                row[metric]["projection_primary_family"] = _project_mean(adjusted, keys, metric)
            if design_valid and any(row[m]["projection_95"]["status"] != "VALID" for m in ("dp_gap", "eo_gap")):
                row["evaluation_status"] = "PARTIALLY_ESTIMABLE"
            condition_replicates[position] = mean_reps
            # BA is linear in q at fixed Y and weights; using mean q here
            # exactly equals mean seed BA. This identity is NOT used for EO.
            condition_q[position] = np.mean([q_models[k] for k in keys], axis=0)
            mean_ba = linearized_survey_inference(args[0], {"MEAN_POLICY": condition_q[position]}, *args[2:],
                domain_mask=design.domain_mask, alpha=.05)
            row["balanced_accuracy"]["taylor_95"] = _linear_record(mean_ba.get("methods", {}).get("MEAN_POLICY", {}).get("balanced_accuracy"))
        summaries.append(row)
        arm_rows.append(row)
    for i, fair in enumerate(arm_rows):
        if fair["method"] != "FAIRBIAS_BM" or i not in condition_replicates:
            continue
        for j, base in enumerate(arm_rows):
            if base["method"] == "FAIRBIAS_BM" or j not in condition_replicates:
                continue
            if any(fair.get(k) != base.get(k) for k in ("backbone", "training_weighted", "tau")):
                continue
            if (fair["status"] == "FIXED_ABLATION") != (base["status"] == "FIXED_ABLATION"):
                continue
            primary = (arm in ("arm_001", "arm_003") and fair["backbone"] == "LR" and not fair["training_weighted"]
                       and fair["tau"] == .10 and base["method"] in ("REWEIGHING", "LFR_RECONSTRUCTED", "EG_DP", "EG_EO", "TO_EO"))
            row = {"arm_id": arm, "backbone": fair["backbone"], "tau": fair["tau"], "training_weighted": fair["training_weighted"],
                   "reference_method": "FAIRBIAS_BM", "comparison_method": base["method"], "in_primary_family_20": primary,
                   "feasible_on_S": fair["status"] == base["status"] == "FEASIBLE"}
            diff_reps = condition_replicates[i] - condition_replicates[j]
            for k, idx in (("balanced_accuracy", 0), ("eo_gap", 2)):
                fp, bp = fair[k]["mean"], base[k]["mean"]
                point = fp-bp if fp is not None and bp is not None else None
                row["delta_"+k] = {"estimate": point, "bootstrap_descriptive": _replicate_interval(point, diff_reps[:, idx], df)}
            paired_args = (args[0], {"FAIRBIAS_BM": condition_q[i], "BASE": condition_q[j]}, *args[2:])
            paired_linear = linearized_survey_inference(*paired_args, domain_mask=design.domain_mask, alpha=.05)
            record = paired_linear.get("paired", {}).get("BASE", {}).get("balanced_accuracy")
            row["delta_balanced_accuracy"].update(_linear_record(record))
            if primary and record is not None:
                half = float(stats.t.ppf(1-(.05/20)/2, df))*record.std_error
                row["delta_balanced_accuracy"]["family_20_interval"] = {"status": record.status,
                    "lower": record.point_estimate-half, "upper": record.point_estimate+half, "alpha": .05/20}
            for band in ("projection_95", "projection_primary_family"):
                f, b = fair["eo_gap"][band], base["eo_gap"][band]
                row["delta_eo_gap"][band] = _projection_difference(f, b)
            for k in ("average_precision", "auroc", "brier"):
                fp, bp = fair[k]["mean"], base[k]["mean"]
                row["delta_"+k] = fp-bp if fp is not None and bp is not None else None
            row.update(reference_selection_id=fair['selection_id'], comparison_selection_id=base['selection_id'],
                reference_model_ids=fair['model_ids'], comparison_model_ids=base['model_ids'], study_branch=STUDY_BRANCH)
            all_paired.append(row)
    np.savez_compressed(output / (arm+"_replicate_metric_arrays.npz"), **replicates)
    return summaries, all_paired


def _validate_model_artifact(model, ref):
    job_path = _verify({'path': ref['job_path'], 'sha256': ref['job_sha256']})
    job = _read(job_path)
    _require(isinstance(model, dict) and {'policy', 'preprocessor', 'semantic_input', 'config', 'seed', 'data_identity'} <= set(model),
             'Frozen model artifact schema mismatch')
    _require(model['config'] == job['config'] and type(model['seed']) is int
             and model['seed'] == job['seed'] == ref['seed']
             and model['data_identity'] == job['data_identity'] == ref['data_identity'], 'Frozen model identity mismatch')
    # Preserve exactly the original worker's semantic-input rule.
    _require(type(model['semantic_input']) is bool
             and model['semantic_input'] == ref['method'].startswith('FAIRBIAS'), 'Frozen model feature interface mismatch')
    if not model['semantic_input']:
        _require(callable(getattr(model['preprocessor'], 'transform', None)), 'Missing frozen preprocessor')
    _verify({'path': ref['job_path'], 'sha256': ref['job_sha256']})


def evaluate_catalog_arm(study_path, expected_study_sha256, release_path, expected_release_sha256,
                         output, *, arm_id, data_root=None):
    """Evaluate exactly one complete frozen arm after explicit T admission."""
    _require(not Path(output).exists(), 'Evaluation output already exists')
    study = validate_catalog_study(study_path, expected_study_sha256)
    _require(arm_id in study['arm_ids'], 'Unknown arm')
    release = _release(study, expected_study_sha256, release_path, expected_release_sha256, [arm_id])
    root = Path(data_root or ROOT).resolve()
    _verify_t_inputs(study, root)
    # Reserve output only once pre-I/O authorization and exact T hashes passed.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cohort = load_local_nhis_cohort(root, years=(2024,))
    _validate_observed_provenance(study['t_source_provenance'], cohort.attrs.get('source_provenance'))
    write_json(output / 'cohort_eligibility_T.json', summarize_eligibility(cohort))
    T = load_arm_partitions(cohort, arm_id)['evaluation_T']
    design, spec = _normalized_annual_design(T), ARM_SPECS[arm_id]
    _require(design.year == 2024 and T.role == 'evaluation_T' and T.year == 2024, 'Incorrect annual T domain')
    arm_selections = [s for s in study['selections'] if s['arm_id'] == arm_id]
    refs = [r for r in study['selected_models'] if r['arm_id'] == arm_id]
    _require([r['model_id'] for r in refs] == study['expected_modelsets'][arm_id], 'Incomplete arm model set')
    q_models, individual = {}, {}
    for ref in refs:
        key = ref['model_id']
        path = _verify(ref['artifacts']['model.joblib'])
        model = joblib.load(path)
        _verify(ref['artifacts']['model.joblib'])
        _validate_model_artifact(model, ref)
        X = T.X_semantic if model['semantic_input'] else model['preprocessor'].transform(T.X_semantic)
        bundle = catalog_inference.predict_frozen_bundle(model, X, T.A, method=ref['method'],
            expected_output_type=ref['output_type'], inference_policy=study['inference_policies'][ref['method']])
        _require(bundle.n_samples == len(T), 'Prediction length differs from frozen T domain')
        q_models[key] = design.expand(bundle.q_decision)
        individual[key] = dict(model_id=key,
            weighted=compute_survey_fairness_metrics(T.y, bundle.q_decision, T.A, T.WTFA_A, spec['expected_categories']),
            unweighted=compute_survey_fairness_metrics(T.y, bundle.q_decision, T.A, np.ones(len(T)), spec['expected_categories']),
            risk=compute_risk_metrics(T.y, bundle, T.WTFA_A, T.A, spec['expected_categories']),
            unweighted_risk=compute_risk_metrics(T.y, bundle, np.ones(len(T)), T.A, spec['expected_categories']))
        arrays = {'q': bundle.q_decision}
        if bundle.p_event is not None:
            arrays['p'] = bundle.p_event
            individual[key]['threshold_05'] = compute_survey_fairness_metrics(T.y, (bundle.p_event >= .5).astype(float), T.A, T.WTFA_A, spec['expected_categories'])
        elif hasattr(model['policy']._adapter, 'predict_event_probability'):
            p = catalog_inference.predict_untouched_base_p(model, X, T.A)
            arrays['base_p'] = p
            individual[key]['base_risk'] = compute_risk_metrics(T.y, PredictionBundle(p, bundle.q_decision), T.WTFA_A, T.A, spec['expected_categories'])
            individual[key]['unweighted_base_risk'] = compute_risk_metrics(T.y, PredictionBundle(p, bundle.q_decision), np.ones(len(T)), T.A, spec['expected_categories'])
        prediction_path = output / (arm_id + '_' + key + '_predictions_T.npz')
        np.savez_compressed(prediction_path, **arrays)
    if refs:
        write_json(output / (arm_id + '_cohort_T.json'), summarize_partition(T, spec['expected_categories'], fitted_preprocessor=model['preprocessor']))
        rows, paired = _evaluate_arm_statistics(T, arm_id, arm_selections, q_models, individual, output)
    else:
        rows = [{**s, 'evaluation_status': 'NO_VALID_MODEL', 'frozen_seed_models': []} for s in arm_selections]
        paired = []
        write_json(output / (arm_id + '_individual_model_metrics.json'), {})
    # Final source/model/T checks: partial directories never receive COMPLETE
    # manifests if code or data changed while predictions were being produced.
    _require(_analysis(study['analysis_root']) == study['analysis_files'] and _environment() == study['environment'], 'Analysis changed during evaluation')
    for ref in refs:
        _verify(ref['artifacts']['model.joblib'])
    _verify_t_inputs(study, root)
    _verify(study['selection'])
    _verify({'path': str(Path(study_path).resolve()), 'sha256': expected_study_sha256})
    _verify(release)
    validate_catalog_study(study_path, expected_study_sha256)
    manifest = _manifest(study, expected_study_sha256, release, [arm_id], 'arm')
    manifest['observed_source_provenance'] = cohort.attrs['source_provenance']
    manifest['artifacts'] = {p.name: file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / 'evaluation_manifest.json', manifest)
    summary = _summary_document(expected_study_sha256, study['selection']['sha256'], output / 'evaluation_manifest.json', rows, paired)
    write_json(output / 'summary_T.json', summary)
    return summary


def merge_catalog_evaluations(study_path, expected_study_sha256, shards, output, *,
                             release_path, expected_release_sha256):
    """Merge exactly one hash-bound shard per registered arm; no T/model loading.

    ``shards`` is a list of {path: /absolute/summary_T.json, sha256: ...} supplied
    by the reviewer. Shard hashes are never inferred from a directory scan.
    """
    _require(not Path(output).exists(), 'Merged evaluation output already exists')
    study = validate_catalog_study(study_path, expected_study_sha256)
    release = _release(study, expected_study_sha256, release_path, expected_release_sha256, study['arm_ids'])
    _require(isinstance(shards, list) and len(shards) == len(study['arm_ids']), 'All arm shards required')
    seen, summaries, paired, bound_shards = set(), [], [], []
    for item in shards:
        path = _verify(item)
        summary = _read(path)
        mpath = path.with_name('evaluation_manifest.json')
        manifest = _read(_verify({'path': str(mpath), 'sha256': summary['evaluation_manifest_sha256']}))
        _require(manifest.get('scope') == 'arm' and len(manifest.get('arm_ids', [])) == 1, 'Expected complete arm shard')
        arm = manifest['arm_ids'][0]
        _require(arm in study['arm_ids'] and arm not in seen, 'Duplicate/unknown arm shard')
        seen.add(arm)
        shard_release = manifest['release']
        _release(study, expected_study_sha256, shard_release['path'], shard_release['sha256'], [arm])
        expected_manifest = _manifest(study, expected_study_sha256, shard_release, [arm], 'arm')
        _require({k: v for k, v in manifest.items() if k not in {'artifacts', 'observed_source_provenance'}} == expected_manifest, 'Shard provenance/model set mismatch')
        _validate_observed_provenance(study['t_source_provenance'], manifest.get('observed_source_provenance'))
        _require(summary.get('schema_version') == SUMMARY_SCHEMA and summary.get('study_branch') == STUDY_BRANCH
            and summary.get('study_freeze_sha256') == expected_study_sha256
            and summary.get('selection_sha256') == study['selection']['sha256'], 'Shard summary binding mismatch')
        expected_rows = [s for s in study['selections'] if s['arm_id'] == arm]
        actual_rows = summary['selections']
        _require(len(actual_rows) == len(expected_rows), 'Missing selection rows')
        by_id = {s['selection_id']: s for s in actual_rows}
        _require(len(by_id) == len(actual_rows), 'Duplicate selection row')
        for row in expected_rows:
            actual = by_id.get(row['selection_id'], {})
            _require(all(actual.get(k) == v for k, v in row.items()) and actual.get('frozen_seed_models') == row['model_ids'], 'Selection model membership changed')
        _require(isinstance(manifest.get('artifacts'), dict), 'Missing shard artifacts')
        for name, sha in manifest['artifacts'].items():
            _require(Path(name).name == name, 'Invalid shard artifact path')
            _verify({'path': str(path.parent / name), 'sha256': sha})
        individual_name = arm + '_individual_model_metrics.json'
        _require(individual_name in manifest['artifacts'], 'Individual model metrics are not bound')
        individual = _read(path.parent / individual_name)
        _require(set(individual) == set(study['expected_modelsets'][arm]), 'Incomplete individual metrics')
        expected_pairs = {(f['selection_id'], b['selection_id']) for f in expected_rows for b in expected_rows
            if f['method'] == 'FAIRBIAS_BM' and b['method'] != 'FAIRBIAS_BM' and f['model_ids'] and b['model_ids']
            and all(f.get(k) == b.get(k) for k in ('backbone', 'training_weighted', 'tau'))
            and (f['status'] == 'FIXED_ABLATION') == (b['status'] == 'FIXED_ABLATION')}
        actual_pairs = [(p.get('reference_selection_id'), p.get('comparison_selection_id')) for p in summary['paired_contrasts']]
        _require(len(actual_pairs) == len(set(actual_pairs)) and set(actual_pairs) == expected_pairs, 'Incomplete paired contrast family')
        for contrast in summary['paired_contrasts']:
            f = by_id.get(contrast.get('reference_selection_id'), {})
            b = by_id.get(contrast.get('comparison_selection_id'), {})
            _require(f and b and contrast.get('study_branch') == STUDY_BRANCH and contrast.get('arm_id') == arm
                and contrast.get('reference_model_ids') == f['model_ids'] and contrast.get('comparison_model_ids') == b['model_ids'], 'Unbound paired contrast')
        summaries.extend(actual_rows)
        paired.extend(summary['paired_contrasts'])
        bound_shards.append({'summary': dict(item), 'manifest': _bound(mpath)})
    _require(seen == set(study['arm_ids']), 'Incomplete arm coverage')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(study, expected_study_sha256, release, study['arm_ids'], 'complete_study')
    observed = [_read(item['manifest']['path'])['observed_source_provenance'] for item in bound_shards]
    _require(all(item == observed[0] for item in observed), 'Shards used different annual source observations')
    manifest['observed_source_provenance'] = observed[0]
    manifest['shards'] = bound_shards
    write_json(output / 'evaluation_manifest.json', manifest)
    result = _summary_document(expected_study_sha256, study['selection']['sha256'], output / 'evaluation_manifest.json', summaries, paired)
    write_json(output / 'summary_T.json', result)
    return result
