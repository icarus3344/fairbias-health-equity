"""Path-explicit, evaluation-only paper reporting helpers.

This module reads frozen registration/selection metadata and aggregate
evaluation JSON.  It never opens model files, microdata, or prediction arrays,
and it does not fit or select a model.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value(row: Mapping[str, Any], name: str) -> Any:
    value = row.get(name)
    if isinstance(value, Mapping):
        return value.get("mean")
    return value


def _status(row: Mapping[str, Any]) -> str:
    return str(row.get("evaluation_status", row.get("status", "UNKNOWN")))


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _failure_count(paths: Iterable[Path]) -> dict[str, int]:
    """Count every supplied job result, including failures omitted from S/T."""
    counts: dict[str, int] = {}
    total = 0
    for path in paths:
        record = _read(Path(path))
        status = str(record.get("status", record.get("evaluation_status", "UNKNOWN"))) if isinstance(record, Mapping) else "UNKNOWN"
        counts[status] = counts.get(status, 0) + 1
        total += 1
    counts["_total"] = total
    counts["_failures"] = sum(v for k, v in counts.items() if k not in {"_total", "_failures", "VALID"})
    return counts


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row.get("arm_id"), row.get("backbone"), row.get("method"),
            bool(row.get("training_weighted", False)), row.get("tau"))


def _nested(row, *keys):
    value = row
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _flatten(row, s_by_key):
    flat = {key: row.get(key) for key in ("arm_id", "method", "backbone", "tau", "training_weighted", "sample_n", "df")}
    flat.update(S_status=s_by_key.get(_key(row), row).get("status", "UNKNOWN"), T_status=_status(row),
                frozen_seed_count=len(row.get("frozen_seed_models", [])))
    for metric in ("balanced_accuracy", "eo_gap", "dp_gap", "average_precision", "auroc", "brier"):
        label = metric + "_p" if metric in ("average_precision", "auroc", "brier") else metric
        flat[label] = _value(row, metric)
        flat["seed_sd_"+metric] = _nested(row, metric, "seed_sd")
        flat["unweighted_"+label] = _value(row, "unweighted_"+metric)
    for metric in ("average_precision", "auroc", "brier"):
        flat["untouched_base_"+metric] = _value(row, "untouched_base_"+metric)
    flat["design_se_balanced_accuracy"] = _nested(row, "balanced_accuracy", "taylor_95", "se")
    for side in ("lower", "upper"):
        flat["ba_taylor_"+side] = _nested(row, "balanced_accuracy", "taylor_95", side)
        flat["eo_projection_"+side] = _nested(row, "eo_gap", "projection_95", side)
    return flat


def _csv(path, rows, fields):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _geometry_audit(registration, job_paths):
    """Describe the registered geometry intervention, including failed seeds."""
    configs = {c["candidate_id"]: c for c in registration.get("candidates", [])
               if c.get("family") == "fixed_geometry_ablation"}
    records, grouped = [], {}
    for path in job_paths:
        result = _read(path)
        config = configs.get(result.get("candidate_id"))
        if config is None:
            continue
        manifest = result.get("algorithm_manifest", {})
        trace = [{k: t.get(k) for k in ("selected_feature", "accepted_transformation", "dropped")}
                 for t in manifest.get("transform_trace", [])
                 if t.get("accepted_transformation") is not None or t.get("dropped")]
        record = {"candidate_id": result["candidate_id"], "seed": result["seed"], "backbone": config["backbone"],
            "geometry_profile": config["params"]["geometry_profile"], "status": result["status"],
            "epsilon_threshold": manifest.get("epsilon_threshold"), "initial_max_dphi": manifest.get("initial_max_dphi"),
            "final_max_dphi": manifest.get("final_max_dphi"), "termination": manifest.get("termination_reason"),
            "committed_trace": trace, "changed_dict": manifest.get("changed_dict"),
            "mds_fit_count": len(manifest.get("mds_fits", []))}
        records.append(record)
        grouped.setdefault((config["backbone"], result["seed"]), {})[record["geometry_profile"]] = record
    comparisons = []
    for (backbone, seed), profiles in sorted(grouped.items()):
        anchor = profiles.get("stress_elbow")
        for profile in ("fixed2_same_epsilon", "fixed2_own_epsilon"):
            other = profiles.get(profile)
            row = {"backbone": backbone, "seed": seed, "reference": "stress_elbow", "comparison": profile,
                   "status": "NOT_COMPARABLE", "first_different_commit": None}
            if anchor and other and anchor["status"] == other["status"] == "VALID":
                a, b = anchor["committed_trace"], other["committed_trace"]
                different = next((i+1 for i in range(max(len(a), len(b)))
                                  if (a[i] if i<len(a) else None) != (b[i] if i<len(b) else None)), None)
                row.update(status="COMPARABLE", first_different_commit=different,
                    identical_committed_trace=different is None,
                    reference_epsilon=anchor["epsilon_threshold"], comparison_epsilon=other["epsilon_threshold"])
            comparisons.append(row)
    return {"records": records, "comparisons": comparisons,
        "interpretation": "First committed-trace divergence under predeclared fixed configurations; endpoint differences alone do not identify a mechanism. The own-epsilon contrast changes geometry and threshold together."}


def write_paper_reporting(
    registration_path: str | Path,
    selection_freeze_path: str | Path,
    summary_t_path: str | Path,
    individual_metrics_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    job_result_paths: Sequence[str | Path] = (),
) -> dict[str, Path]:
    """Create a Markdown draft and CSV table from aggregate benchmark JSON.

    All input paths are explicit.  ``job_result_paths`` should enumerate every
    registered job result when available; its counts are intentionally kept
    separate from the selected S/T rows.  The output directory must not exist,
    preventing accidental overwrite of evidence.
    """
    registration_path, selection_freeze_path, summary_t_path = map(Path, (registration_path, selection_freeze_path, summary_t_path))
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing report directory: {out}")
    registration = _read(registration_path)
    selection = _read(selection_freeze_path)
    summary = _read(summary_t_path)
    if selection.get("registration_sha256") != _sha(registration_path):
        raise ValueError("selection_freeze registration_sha256 does not match registration.json")
    manifest_path = summary_t_path.parent / "evaluation_manifest.json"
    if not manifest_path.exists():
        raise ValueError("summary_T.json has no adjacent evaluation_manifest.json evidence")
    evaluation_manifest = _read(manifest_path)
    if evaluation_manifest.get("selection_sha256") != _sha(selection_freeze_path):
        raise ValueError("evaluation_manifest selection_sha256 does not match selection_freeze.json")
    if summary.get("evaluation_manifest_sha256") != _sha(manifest_path):
        raise ValueError("summary_T does not bind its evaluation manifest")
    individual = [_read(Path(path)) for path in individual_metrics_paths]

    selections = list(summary.get("selections", []))
    s_rows = list(selection.get("selections", []))
    s_by_key = {_key(row): row for row in s_rows}
    primary = [row for row in selections if row.get("tau") == 0.10]
    auxiliary = [row for row in selections if row not in primary]
    rows = [_flatten(row, s_by_key) for row in primary]
    csv_path = out / "paper_results_tau_010.csv"
    fields = list(_flatten({}, {}))

    job_paths = list(map(Path, job_result_paths))
    expected = {(c["candidate_id"], int(seed)) for c in registration.get("candidates", []) if c.get("status", "REGISTERED") == "REGISTERED" for seed in c.get("seeds", [])}
    observed = set()
    for path in job_paths:
        job = _read(path)
        if "candidate_id" not in job or "seed" not in job:
            raise ValueError(f"job result lacks candidate_id/seed: {path}")
        observed.add((job["candidate_id"], int(job["seed"])))
    if observed != expected or len(observed) != len(job_paths):
        raise ValueError(f"job result coverage mismatch: expected {len(expected)}, observed {len(observed)}")
    out.mkdir(parents=True)
    _csv(csv_path, rows, fields)
    failures = _failure_count(job_paths)
    s_counts: dict[str, int] = {}
    for row in s_rows:
        status = str(row.get("status", "UNKNOWN"))
        s_counts[status] = s_counts.get(status, 0) + 1
    t_counts: dict[str, int] = {}
    for row in selections:
        status = _status(row)
        t_counts[status] = t_counts.get(status, 0) + 1
    valid_t = t_counts.get("VALID", 0) + t_counts.get("PARTIALLY_ESTIMABLE", 0)
    report_path = out / "paper_results_draft.md"
    methods = (
        "We froze registered candidate configurations and selection artifacts before evaluation, "
        "then summarized fixed-policy aggregate metrics without refitting or selecting on T. "
        "The primary table uses tau=0.10 where registered; prediction risk is reported from the "
        "policy event probability p, with average precision recorded as NA when p is unavailable; untouched-base p "
        "shown separately. seed SD describes variation across frozen seed models, whereas design SE "
        "comes from the survey design inference and is not a substitute for seed variation."
    )
    lines = [
        "# NHIS FairBias 应用结果草稿",
        "",
        "本稿只汇总已冻结的注册、选择和 T 聚合 JSON；没有读取微观数据、模型文件或重新拟合。结果表以主 τ=0.10 配置为主，并把预测 AP 作为参考指标。",
        "",
        f"- S 状态计数（来自 selection_freeze）：{json.dumps(s_counts, ensure_ascii=False, sort_keys=True)}",
        f"- T 有效或部分可估记录数：{valid_t}",
        f"- T 状态计数（来自 summary_T）：{json.dumps(t_counts, ensure_ascii=False, sort_keys=True)}",
        f"- 所有显式 job result 总数：{failures['_total']}",
        f"- 所有显式 job result 失败数：{failures['_failures']}",
        f"- job result 状态计数：{json.dumps({k: v for k, v in failures.items() if not k.startswith('_')}, ensure_ascii=False, sort_keys=True)}",
        "",
        "表中 `average_precision_p` 只接受事件概率 p 的 AP；没有 p 时保留 NA。`untouched_base_average_precision` 单独表示 untouched-base p，不能与决策 q 混为同一 estimand。`seed_sd_balanced_accuracy` 是种子模型间 SD，`design_se_balanced_accuracy` 与区间来自 Taylor 设计推断；EO 使用 projection_95。`NOT_ESTIMABLE`、`NO_VALID_MODEL` 和失败状态保留在表中；本稿不默认 FairBias 胜出。",
        "",
        "## 可复制的英文 methods 段落",
        "",
        methods,
        "",
        f"输入 registration：`{Path(registration_path)}`；selection freeze：`{Path(selection_freeze_path)}`；summary_T：`{Path(summary_t_path)}`。individual metrics 文件数：{len(individual)}。",
        "",
        f"CSV：`{csv_path.name}`；τ=None 固定消融及预测参考另见 `paper_results_auxiliary.csv`。",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    aux_path = out / "paper_results_auxiliary.csv"
    _csv(aux_path, [_flatten(row, s_by_key) for row in auxiliary], fields)
    paired_path = out / "paper_paired_contrasts.csv"
    paired = list(summary.get("paired_contrasts", []))
    paired_fields = ["arm_id", "backbone", "tau", "training_weighted", "reference_method", "comparison_method", "feasible_on_S", "in_primary_family_20", "delta_balanced_accuracy", "delta_eo_gap", "delta_average_precision", "delta_auroc", "delta_brier", "delta_ba_se", "delta_ba_lower", "delta_ba_upper", "delta_ba_family20_lower", "delta_ba_family20_upper", "delta_eo_projection_lower", "delta_eo_projection_upper", "delta_eo_family20_lower", "delta_eo_family20_upper"]
    paired_rows = []
    for row in paired:
        flat = {field: row.get(field) for field in paired_fields[:13]}
        flat["delta_balanced_accuracy"] = _nested(row, "delta_balanced_accuracy", "estimate")
        flat["delta_eo_gap"] = _nested(row, "delta_eo_gap", "estimate")
        flat["delta_ba_se"] = _nested(row, "delta_balanced_accuracy", "se")
        for side in ("lower", "upper"):
            flat["delta_ba_"+side] = _nested(row, "delta_balanced_accuracy", side)
            flat["delta_ba_family20_"+side] = _nested(row, "delta_balanced_accuracy", "family_20_interval", side)
            flat["delta_eo_projection_"+side] = _nested(row, "delta_eo_gap", "projection_95", side)
            flat["delta_eo_family20_"+side] = _nested(row, "delta_eo_gap", "projection_primary_family", side)
        paired_rows.append(flat)
    _csv(paired_path, paired_rows, paired_fields)
    unsupported = [{"arm_id": c.get("arm_id"), "backbone": c.get("backbone"), "method": c.get("method"),
        "status": c["status"], "reason": c.get("reason") or "; ".join(c.get("notes", []))}
        for c in registration.get("candidates", []) if c.get("status") != "REGISTERED"]
    _csv(out / "not_supported_conditions.csv", unsupported, ["arm_id", "backbone", "method", "status", "reason"])
    (out / "arm004_geometry_trace_audit.json").write_text(json.dumps(_geometry_audit(registration, job_paths), indent=2) + "\n")
    (out / "report_manifest.json").write_text(json.dumps({
        "registration_path": str(registration_path), "registration_sha256": _sha(registration_path),
        "selection_freeze_path": str(selection_freeze_path), "selection_sha256": _sha(selection_freeze_path),
        "summary_T_path": str(summary_t_path), "summary_T_sha256": _sha(summary_t_path),
        "evaluation_manifest_path": str(manifest_path), "evaluation_manifest_sha256": _sha(manifest_path),
        "individual_metrics_paths": [str(Path(path)) for path in individual_metrics_paths],
        "job_result_paths": [str(Path(path)) for path in job_result_paths],
        "selection_count": len(selections),
        "primary_tau_rows": len(rows),
        "failure_counts": failures, "selection_status_counts": s_counts, "evaluation_status_counts": t_counts,
        "auxiliary_rows": len(auxiliary), "paired_rows": len(paired),
        "input_hashes": {str(path): _sha(path) for path in [*map(Path, individual_metrics_paths), *job_paths]},
    }, indent=2, ensure_ascii=False) + "\n")
    return {"markdown": report_path, "csv": csv_path, "manifest": out / "report_manifest.json"}
