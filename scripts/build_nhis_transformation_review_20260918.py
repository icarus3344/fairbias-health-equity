#!/usr/bin/env python3
"""Read only sealed JSON and byte hashes; export all80 transformation evidence.

No runtime model module is imported. No model/prepared object is deserialized,
and no row-level data, predictions, outcomes, or utility values are extracted.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ADMISSION_SHA = "2a83612b23048a54ce09196ec553b9881ee87bf5b5930f36fc53bc0e52d1a628"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)

def state_hash(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()[:16]

def compose(old, new):
    result = {k: new.get(v, v) for k, v in old.items()}
    result.update({k: v for k, v in new.items() if k not in result})
    return result

def op(x):
    if x is None: return "unchanged"
    if x == "dropped": return "dropped"
    if isinstance(x, dict) and set(x) == {"power"}: return "numeric_power"
    if isinstance(x, dict) and x and "power" not in x: return "categorical_merge"
    raise ValueError(f"Unsupported transformation: {x!r}")

def flags(feature, spec, transform):
    if op(transform) != "categorical_merge": return {}
    states = list(map(str, spec["substantive_codes"]))
    if feature == "empwrkft1_a": states += ["-1", "-2"]
    else: states += ["MISSING"]
    groups = defaultdict(list)
    for value in states: groups[transform.get(value, value)].append(value)
    substantive = set(map(str, spec["substantive_codes"]))
    missing = {"MISSING", "-2"} if feature == "empwrkft1_a" else {"MISSING"}
    mixed_missing = any(set(g) & missing and set(g) & substantive for g in groups.values())
    noncontiguous = False
    if spec["semantic_type"] == "ordinal":
        for g in groups.values():
            xs = sorted(int(v) for v in g if v in substantive)
            if len(xs) > 1 and xs[-1] - xs[0] + 1 != len(xs): noncontiguous = True
    valid_keys = set(states) | {"UNKNOWN"}
    assert set(transform) <= valid_keys and set(transform.values()) <= valid_keys, (feature, transform)
    return {"missing_substantive_pooled": mixed_missing,
            "noncontiguous_ordinal_bin": noncontiguous,
            "all_substantive_collapsed": len({transform.get(v, v) for v in substantive}) == 1,
            "structural_substantive_pooled": feature == "empwrkft1_a" and any("-1" in g and set(g) & substantive for g in groups.values()),
            "representation_groups": dict(groups)}

def csv_dump(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: canon(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); root = args.root.resolve(); out = args.output
    if out.exists(): raise FileExistsError("Refusing to overwrite existing extraction")
    admission_path = root / "artifacts/nhis/completion_evaluation_20260918/control/admission_v1.json"
    assert sha(admission_path) == ADMISSION_SHA
    admission = json.loads(admission_path.read_text()); jobs = admission["jobs"]
    assert len(jobs) == 80 and len({j["job_id"] for j in jobs}) == 80
    inputs = [{"role": "admission", "path": str(admission_path), "sha256": ADMISSION_SHA}]
    # Complete byte verification of all80 sources precedes result extraction.
    for job in jobs:
        for role in ("result", "policy"):
            item = job[role]; assert sha(item["path"]) == item["sha256"], item["path"]
            inputs.append({"role": role, "job_id": job["job_id"], **item})
    runtime = Path(admission["training_sources_root"])
    manifest_path = Path(admission["training_manifest"]["path"])
    assert sha(manifest_path) == admission["training_manifest"]["sha256"]
    manifest = json.loads(manifest_path.read_text())
    inputs.append({"role": "training_manifest", **admission["training_manifest"]})
    source_paths = ["configs/nhis/features.json", "src/nhis_fairbias/preprocessing.py",
                   "src/nhis_fairbias/benchmark/preprocessing.py", "src/nhis_fairbias/benchmark/data_contracts.py",
                   "src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py",
                   "src/nhis_fairbias/benchmark/adapters/adapter_fairbias_ae.py",
                   "src/fairbias/transform.py", "src/fairbias/enhancement.py", "src/fairbias/enhancement_state.py"]
    for rel in source_paths:
        assert sha(runtime / rel) == manifest["sources"][rel], rel
        inputs.append({"role": "frozen_source", "path": str(runtime / rel), "sha256": manifest["sources"][rel]})
    registry = json.loads((runtime / source_paths[0]).read_text())
    features = registry["feature_lists"]["primary_core_features"]
    specs = {s["harmonized_name"]: s for s in registry["primary_core"].values()}
    constants = {}
    for node in ast.parse((runtime / source_paths[3]).read_text()).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("PRIMARY_CORE_21", "DISABILITY_COMPONENTS_6"):
                    constants[t.id] = ast.literal_eval(node.value)
    assert tuple(features) == constants["PRIMARY_CORE_21"] and len(features) == 21
    excluded = set(constants["DISABILITY_COMPONENTS_6"])
    models, rows, events = [], [], []
    for job in jobs:
        record = json.loads(Path(job["result"]["path"]).read_text())
        assert record["job_id"] == job["job_id"] and record["status"] == "FC_COMPLETE_FEASIBLE" and record["reload_exact"] is True
        assert record["config"] == job["config"]
        p = record["completion"]["model_provenance"]
        bm = p["bm_trace"]; ae = [r for r in p["ae_audit"] if r["accepted"]]
        assert len(bm) == p["bm_commits"] and len(ae) == p["ae_commits"]
        assert len(record["completion"]["attempts"]) == 1
        base = {"job_id": job["job_id"], "mode": p["mode"], "arm": job["config"]["arm_id"], "backbone": job["config"]["backbone"], "seed": job["seed"]}
        chronology = [(r["iteration"], "BM", r) for r in bm] + [(r["iteration"], "AE", r) for r in ae]
        assert len({i for i, _, _ in chronology}) == len(chronology), job["job_id"]
        state = {}
        for iteration, engine, event in sorted(chronology):
            feature = event["selected_feature"]; assert feature in features
            tr = event["accepted_transformation"] if engine == "BM" else event["proposed_transform"]
            operation = op(tr)
            if engine == "AE":
                assert event["validity_status"] == "VALID" and state_hash(state) == event["parent_state_hash"], (job["job_id"], iteration, "parent")
                state[feature] = compose(state.get(feature, {}), tr) if operation == "categorical_merge" else tr
                assert state_hash(state) == event["candidate_state_hash"], (job["job_id"], iteration, "candidate")
            else:
                assert (event["dropped"] is True) == (operation == "dropped")
                state[feature] = tr
            events.append({**base, "engine": engine, "iteration": iteration, "feature": feature, "operation": operation,
                           "reported_transform": tr, "composed_feature_state_after": state[feature]})
        assert state == p["changed_dict"], (job["job_id"], "final replay")
        if base["arm"] == "arm_004": assert not (set(state) & excluded)
        assert set(state) <= set(features)
        models.append({**base, "bm_events": len(bm), "ae_committed_events": len(ae), "ae_audit_rows": len(p["ae_audit"]),
                       "final_state_hash": state_hash(state), "replayed_final_matches": True, "changed_dict": state})
        for feature in features:
            available = not (base["arm"] == "arm_004" and feature in excluded)
            bmf = [r for r in bm if r["selected_feature"] == feature]
            aef = [r for r in ae if r["selected_feature"] == feature]
            final = state.get(feature)
            lastbm = bmf[-1]["accepted_transformation"] if bmf else None
            f = flags(feature, specs[feature], final)
            rows.append({**base, "feature": feature, "eligible": available, "bm_events": len(bmf), "bm_touched": bool(bmf),
                         "ae_committed_events": len(aef), "ae_touched": bool(aef),
                         "last_bm_transform": lastbm, "final_operation": op(final) if available else "NOT_IN_ARM",
                         "final_transform": final, "final_differs_last_bm": available and final != lastbm,
                         "missing_substantive_pooled": f.get("missing_substantive_pooled", False),
                         "structural_substantive_pooled": f.get("structural_substantive_pooled", False),
                         "noncontiguous_ordinal_bin": f.get("noncontiguous_ordinal_bin", False),
                         "all_substantive_collapsed": f.get("all_substantive_collapsed", False),
                         "representation_groups": f.get("representation_groups", {})})
    summary, matrix, patterns = {}, [], []
    for mode in sorted({r["mode"] for r in rows}):
        mm = [m for m in models if m["mode"] == mode]
        er = [r for r in rows if r["mode"] == mode and r["eligible"]]
        ee = [e for e in events if e["mode"] == mode]
        summary[mode] = {"models": len(mm), "eligible_model_features": len(er), "excluded_model_features": 40 * 21 - len(er),
                         "bm_event_operations": dict(Counter(e["operation"] for e in ee if e["engine"] == "BM")),
                         "ae_event_operations": dict(Counter(e["operation"] for e in ee if e["engine"] == "AE")),
                         "final_operations": dict(Counter(r["final_operation"] for r in er)),
                         "flags": {k: sum(bool(r[k]) for r in er) for k in ("bm_touched", "ae_touched", "final_differs_last_bm", "missing_substantive_pooled", "structural_substantive_pooled", "noncontiguous_ordinal_bin", "all_substantive_collapsed")},
                         "models_with_flags": {k: len({r["job_id"] for r in er if r[k]}) for k in ("missing_substantive_pooled", "structural_substantive_pooled", "noncontiguous_ordinal_bin", "all_substantive_collapsed")}}
        for feature in features:
            rr = [r for r in er if r["feature"] == feature]; spec = specs[feature]
            m = {"mode": mode, "feature": feature, "official_description": spec["official_description"], "semantic_type": spec["semantic_type"],
                 "eligible_models": len(rr), "excluded_models": 40-len(rr), "bm_events": sum(r["bm_events"] for r in rr),
                 "bm_touched_models": sum(r["bm_touched"] for r in rr), "ae_committed_events": sum(r["ae_committed_events"] for r in rr),
                 "ae_touched_models": sum(r["ae_touched"] for r in rr)}
            m.update({"final_"+k+"_models": sum(r["final_operation"] == k for r in rr) for k in ("unchanged", "dropped", "numeric_power", "categorical_merge")})
            m.update({k+"_models": sum(r[k] for r in rr) for k in ("final_differs_last_bm", "missing_substantive_pooled", "structural_substantive_pooled", "noncontiguous_ordinal_bin", "all_substantive_collapsed")})
            matrix.append(m)
            for encoded, n in sorted(Counter(canon(r["final_transform"]) for r in rr).items()):
                tr = json.loads(encoded)
                matched = [r for r in rr if canon(r["final_transform"]) == encoded]
                patterns.append({"mode": mode, "feature": feature, "model_count": n, "final_operation": op(tr), "final_transform": tr,
                                 "arms_backbones": sorted({r["arm"]+"/"+r["backbone"] for r in matched}),
                                 "representation_groups": flags(feature, spec, tr).get("representation_groups", {})})
    cells=[]
    for mode in sorted(summary):
        for arm in ("arm_001", "arm_002", "arm_003", "arm_004"):
            for backbone in ("LR", "GBDT"):
                for feature in features:
                    rr=[r for r in rows if (r["mode"],r["arm"],r["backbone"],r["feature"])==(mode,arm,backbone,feature)]
                    assert len(rr)==5 and {r["seed"] for r in rr}=={0,7,19,37,73}
                    eligible=all(r["eligible"] for r in rr)
                    cells.append({"mode":mode,"arm":arm,"backbone":backbone,"feature":feature,"eligible":eligible,
                                  "seed_denominator":5 if eligible else None,
                                  "bm_touched_seeds":sum(r["bm_touched"] for r in rr) if eligible else None,
                                  "distinct_final_states":len({canon(r["final_transform"]) for r in rr}) if eligible else None})
    assert len(rows)==1680 and len(matrix)==42 and len(cells)==336
    assert sum(m["bm_events"] for m in models)==560 and sum(m["ae_committed_events"] for m in models)==865
    assert sum(m["ae_audit_rows"] for m in models)==37515
    out.mkdir(parents=True)
    for name, data in [("model_variable_matrix.csv",rows),("variable_mode_summary.csv",matrix),("final_representation_patterns.csv",patterns),("committed_events.csv",events),("arm_backbone_seed_cells.csv",cells)]: csv_dump(out/name,data)
    (out/"model_states.json").write_text(json.dumps(models,indent=2,sort_keys=True)+"\n")
    payload={"admission_sha256":ADMISSION_SHA,"script_sha256":sha(__file__),"inputs":inputs,"summary":summary,
             "invariants":{"result_hashes_verified":80,"policy_hashes_verified_without_deserialization":80,"models_exactly_replayed":80,
                           "replayed_committed_events":len(events),"feature_mode_rows":len(matrix),"model_variable_rows":len(rows),
                           "seed_cells":len(cells),"not_in_arm_cells":sum(not c["eligible"] for c in cells),"raw_microdata_read":False,"predictions_generated":False},
             "definitions":{"final_differs_last_bm":"Final representation differs from the last committed BM transform for that variable, or was newly introduced by AE. This is not a separately trained BM-only model.",
                            "all_substantive_collapsed":"All registry substantive categories map to one code; a missing or structural category can keep the feature nonconstant. Not equivalent to an explicit drop.",
                            "representation_groups":"Full registered code-domain equivalence classes, including target code itself. Not observed row counts; no category-support assertion.",
                            "ordinal_noncontiguity":"At least one final bin contains substantive ordinal codes with an omitted intermediate code. AE adjacent means adjacent in target-rate ordering, not NHIS ordinal ordering."}}
    (out/"extraction_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    hashes={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out/"output_hashes.json").write_text(json.dumps(hashes,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"invariants":payload["invariants"],"summary":summary},indent=2))

if __name__ == "__main__": main()
