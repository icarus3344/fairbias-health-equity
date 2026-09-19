#!/usr/bin/env python3
"""Supervisor aggregate verification; no model deserialization or prediction.

Prepared F/C/S containers supply only membership after byte verification. Raw
public-use records stay in memory; the output contains aggregate evidence only.
The implementation is independent of both worker exporters.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'artifacts/nhis/manuscript_readiness_20260918/supervisor_verification.json'
PAPER = ROOT / 'docs/paper/manuscript_readiness_20260918'
COUNT = 0


def check(condition, label):
    global COUNT
    COUNT += 1
    if not condition:
        raise AssertionError(label)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def obj(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def same(x, y):
    return math.isclose(float(x), float(y), rel_tol=1e-12, abs_tol=1e-9)


def main():
    check(not OUT.exists(), 'unique verification output required')
    control = ROOT / 'artifacts/nhis/completion_evaluation_20260918/control'
    ap = control / 'admission_v1.json'
    check(sha(ap) == '2a83612b23048a54ce09196ec553b9881ee87bf5b5930f36fc53bc0e52d1a628', 'admission')
    admission = obj(ap)
    sp = control / 'study_v1.json'
    check(sha(sp) == '3c4572ec327d8d97a2f108fe3c82a8619370a78981c8fe1abdb496677b2933fb', 'study')
    study = obj(sp)
    for rel, h in study['analysis_files'].items():
        check(sha(ROOT / rel) == h, 'frozen evaluation source: ' + rel)
    tm = admission['training_manifest']
    check(sha(tm['path']) == tm['sha256'], 'training manifest')
    tr = Path(admission['training_sources_root'])
    train = obj(tm['path'])
    for rel, h in train['sources'].items():
        check(sha(tr / rel) == h, 'frozen training source: ' + rel)
    for name in ('features.json', 'study.json', 'variables.json'):
        check(sha(ROOT / 'configs/nhis' / name) == sha(tr / 'configs/nhis' / name), 'local config: ' + name)

    registry = obj(ROOT / 'configs/nhis/features.json')
    specs = {s['harmonized_name']: s for s in registry['primary_core'].values()}
    feature_names = registry['feature_lists']['primary_core_features']
    matrix = rows(PAPER / 'transformations/extraction/model_variable_matrix.csv')
    check(len(matrix) == 1680, 'transformation row count')
    lookup = {(r['job_id'], r['feature']): r for r in matrix}
    flagged = defaultdict(set)
    event_counts = Counter()
    final_counts = Counter()
    ae_hash_checks = 0
    for job in admission['jobs']:
        for role in ('result', 'policy'):
            check(sha(job[role]['path']) == job[role]['sha256'], role + ' hash')
        r = obj(job['result']['path'])
        p = r['completion']['model_provenance']
        check(r['job_id'] == job['job_id'] and r['status'] == 'FC_COMPLETE_FEASIBLE', 'result identity')
        final = p['changed_dict']
        accepted = [e for e in p['ae_audit'] if e['accepted']]
        timeline = sorted([(e['iteration'], 'BM', e) for e in p['bm_trace']] + [(e['iteration'], 'AE', e) for e in accepted])
        state = {}
        def signature():
            return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()[:16]
        for _, engine, e in timeline:
            event_counts[engine] += 1
            key = e['selected_feature']
            change = e['accepted_transformation'] if engine == 'BM' else e['proposed_transform']
            if engine == 'AE':
                check(signature() == e['parent_state_hash'], 'AE parent state')
                if isinstance(change, dict) and 'power' not in change:
                    old = state.get(key, {})
                    merged = {k: change.get(v, v) for k, v in old.items()}
                    for k, v in change.items():
                        if k not in merged:
                            merged[k] = v
                    state[key] = merged
                else:
                    state[key] = change
                check(signature() == e['candidate_state_hash'], 'AE candidate state')
                ae_hash_checks += 1
            else:
                state[key] = change
        check(state == final, 'full final replay')
        for feature in feature_names:
            row = lookup[(job['job_id'], feature)]
            transform = final.get(feature)
            encoded = row['final_transform']
            exported = None if encoded == '' else ('dropped' if encoded == 'dropped' else json.loads(encoded))
            check(exported == transform, 'final export mapping')
            if row['eligible'] == 'False':
                check(job['config']['arm_id'] == 'arm_004' and feature not in final and row['final_operation'] == 'NOT_IN_ARM', 'excluded representation')
                continue
            operation = 'unchanged' if transform is None else ('dropped' if transform == 'dropped' else ('numeric_power' if 'power' in transform else 'categorical_merge'))
            check(row['final_operation'] == operation, 'final operation')
            final_counts[operation] += 1
            flags = dict.fromkeys(('missing_substantive_pooled', 'structural_substantive_pooled', 'noncontiguous_ordinal_bin', 'all_substantive_collapsed'), False)
            if operation == 'categorical_merge':
                substantive = [str(v) for v in specs[feature]['substantive_codes']]
                full = substantive + (['-1', '-2'] if feature == 'empwrkft1_a' else ['MISSING'])
                groups = defaultdict(set)
                for value in full:
                    groups[transform.get(value, value)].add(value)
                for values in groups.values():
                    actual = values.intersection(substantive)
                    flags['missing_substantive_pooled'] |= bool(actual and values.intersection({'MISSING', '-2'}))
                    flags['structural_substantive_pooled'] |= bool(feature == 'empwrkft1_a' and actual and '-1' in values)
                    if specs[feature]['semantic_type'] == 'ordinal' and len(actual) > 1:
                        ordered = sorted(map(int, actual))
                        flags['noncontiguous_ordinal_bin'] |= any(b-a != 1 for a, b in zip(ordered, ordered[1:]))
                flags['all_substantive_collapsed'] = len({transform.get(v, v) for v in substantive}) == 1
            for flag, value in flags.items():
                check((row[flag] == 'True') == value, flag)
                if value:
                    flagged[flag].add(job['job_id'])
    check(event_counts == {'BM': 560, 'AE': 865}, 'event totals')
    check(final_counts == {'unchanged': 600, 'dropped': 60, 'numeric_power': 80, 'categorical_merge': 820}, 'final totals')
    check(len(flagged['missing_substantive_pooled']) == 80 and len(flagged['structural_substantive_pooled']) == 60 and len(flagged['noncontiguous_ordinal_bin']) == 70, 'headline flags')

    # Verify every recorded input before loading public raw columns or prepared membership.
    missing = PAPER / 'missingness/v2'
    mm = obj(missing / 'source_manifest.json')
    for name, entry in mm['inputs'].items():
        check(sha(entry['path']) == entry['sha256'], 'missingness source ' + name)
    mh = obj(missing / 'hash_manifest.json')
    for rel, h in mh['outputs'].items():
        check(sha(missing / rel) == h, 'missingness output ' + rel)
    import numpy as np
    import pandas as pd
    import joblib
    annual = {}
    for year in (2022, 2023, 2024):
        columns = ['HHX', 'WTFA_A', 'MEDDL12M_A', 'SEX_A', 'HISPALLP_A', 'DISAB3_A'] + [s['official_name'] for s in specs.values()]
        df = pd.read_csv(mm['inputs'][f'raw_{year}_csv']['path'], usecols=list(dict.fromkeys(columns)))
        df.index = 'nhis:' + str(year) + ':' + df.HHX.astype(str).str.strip()
        check(df.index.is_unique, 'raw unique membership')
        annual[year] = df
    expected = {(r['arm_id'], r['partition'], r['variable']): r for r in rows(missing / 'missingness_by_variable.csv')}
    table1 = {(r['arm_id'], r['partition']): r for r in rows(ROOT / 'docs/paper/cohort_tables_v2_20260918/Table1.csv') if r['group_code'] == 'ALL'}
    categories = ['observed', 'item_nonresponse', 'structural_niu', 'raw_null', 'unknown_unmapped']
    all_rows = included = excluded = 0
    for arm in ('arm_001', 'arm_002', 'arm_003', 'arm_004'):
        prepared = joblib.load(mm['inputs'][arm + '.prepared']['path'])
        protected = {'arm_001': 'SEX_A', 'arm_002': 'HISPALLP_A', 'arm_003': 'DISAB3_A', 'arm_004': 'DISAB3_A'}[arm]
        for part, role, year in [('F','fitting_F',2022),('C','calibration_C',2022),('S','selection_S',2023),('T','evaluation_T',2024)]:
            annual_frame = annual[year]
            if part == 'T':
                df = annual_frame.loc[annual_frame.MEDDL12M_A.isin([1,2]) & annual_frame[protected].isin(range(1,8) if arm == 'arm_002' else [1,2])]
            else:
                keys = set(map(str, prepared['partitions'][role].record_keys))
                df = annual_frame.loc[annual_frame.index.isin(keys)]
                check(len(df) == len(keys), 'prepared membership exact')
            w = df.WTFA_A.to_numpy(dtype=float)
            check(len(df) == int(table1[(arm,part)]['n']) and same(w.sum(), table1[(arm,part)]['weight_sum']), 'Table1 denominator')
            check(int(df.MEDDL12M_A.eq(1).sum()) == int(table1[(arm,part)]['event_count']), 'Table1 event count')
            for name, spec in specs.items():
                r = expected[(arm,part,name)]
                all_rows += 1
                if r['status'] == 'NA_EXCLUDED_ARM004':
                    check(arm == 'arm_004' and r['observed_n'] == r['denominator_n'] == 'NA', 'N/A distinct from zero')
                    excluded += 1
                    continue
                included += 1
                raw = df[spec['official_name']]
                codes = pd.to_numeric(raw, errors='coerce')
                label = np.full(len(df), 'unknown_unmapped', dtype=object)
                label[codes.isin(spec['substantive_codes'])] = 'observed'
                label[codes.isin(spec['missing_codes'])] = 'item_nonresponse'
                label[raw.isna()] = 'raw_null'
                if name == 'empwrkft1_a':
                    status = df.EMPWRKLSW1_A
                    label[status.isin([7,8,9])] = 'item_nonresponse'
                    label[status.isna()] = 'raw_null'
                    label[status.notna() & ~status.isin([1,2,7,8,9])] = 'unknown_unmapped'
                    label[status.eq(2)] = 'structural_niu'
                check(int(r['denominator_n']) == len(df) and same(r['denominator_weight'], w.sum()), 'variable denominator')
                for cat in categories + ['preprocessing_missing']:
                    mask = label != 'observed' if cat == 'preprocessing_missing' else label == cat
                    n, wt = int(mask.sum()), float(w[mask].sum())
                    check(n == int(r[cat + '_n']), 'raw category count ' + cat)
                    check(same(wt, r[cat + '_weight']), 'raw category weighted sum ' + cat)
                    check(same(n / len(df), r[cat + '_fraction']), 'raw category fraction ' + cat)
                    check(same(wt / w.sum(), r[cat + '_weighted_fraction']), 'raw weighted fraction ' + cat)
    check((all_rows,included,excluded) == (336,312,24), 'missingness coverage')
    for row in rows(missing / 'raw_omitted_code_counts.csv'):
        # No record export; the official registry omissions have no support in these files.
        df = annual[int(row['year'])]
        col = row.get('variable', row.get('official_name'))
        count = int(df[col].eq(int(row['official_code'])).sum())
        check(count == int(row['unweighted_n']) == 0, 'omitted official code observed count')

    lh = obj(PAPER / 'literature/local_source_hashes.json')['files']
    for rel, entry in lh.items():
        if rel == 'docs/plans/FAIRBIAS_MANUSCRIPT_READINESS_EXECUTION_20260918.md':
            prior = (ROOT / rel).read_text().split('\n## Primary-source host addition')[0].rstrip() + '\n'
            check(hashlib.sha256(prior.encode()).hexdigest() == entry['sha256'], 'gate only appended authorized institutional-source host')
        else:
            check(sha(ROOT / rel) == entry['sha256'], 'reviewed literature code binding: ' + rel)
    for d, manifest_name in [('literature','output_hashes.json'),('transformations/extraction','output_hashes.json')]:
        manifest = obj(PAPER / d / manifest_name)
        entries = manifest['files'] if 'files' in manifest else manifest
        for rel, entry in entries.items():
            digest = entry['sha256'] if isinstance(entry, dict) else entry
            check(sha(PAPER / d / rel) == digest, 'worker output unchanged: ' + rel)
    old_bundle = ROOT / 'artifacts/nhis/completion_evaluation_20260918/paper_bundle_20260918.zip'
    check(sha(old_bundle) == '460b0b78469b632c8ddc5e791c2d8c07215fd32bf477881388ddae25a93c669d', 'old accepted bundle unchanged')
    result = {'status':'PASS', 'assertions':COUNT, 'scope':'independent aggregate recomputation and final transformation replay; no model/prediction load or refit', 'evaluation_sources':len(study['analysis_files']), 'training_sources':len(train['sources']), 'verified_result_policy_files':160, 'replayed_models':80, 'ae_parent_candidate_pairs':ae_hash_checks, 'committed_event_counts':dict(event_counts), 'final_feature_model_counts':dict(final_counts), 'models_with_flags':{k:len(v) for k,v in flagged.items()}, 'missingness_rows':all_rows, 'included_missingness_rows':included, 'arm004_na_rows':excluded, 'table1_domains_reconciled':16, 'script_sha256':sha(__file__), 'missingness_manifest_sha256':sha(missing/'hash_manifest.json'), 'transformation_manifest_sha256':sha(PAPER/'transformations/extraction/extraction_manifest.json'), 'literature_manifest_sha256':sha(PAPER/'literature/output_hashes.json'), 'gate_append_only_amendment':'authorized author institutional repository; prior prefix SHA matches worker binding', 'old_paper_bundle_unchanged':True}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
