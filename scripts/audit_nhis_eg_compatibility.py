#!/usr/bin/env python3
"""Fail-closed audit of stored Fairlearn EG S predictions."""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path, PurePosixPath
from typing import Any
import joblib
import numpy as np
from nhis_fairbias.benchmark.adapters.adapter_reductions import ExponentiatedGradientAdapter
from nhis_fairbias.benchmark.adapters.adapter_reductions_numerical import (
    recover_convex_mixture_q, NumericallyRecoveredExponentiatedGradientAdapter,
)
from nhis_fairbias.benchmark.experiment_worker import file_sha
from nhis_fairbias.benchmark.result_catalog import CatalogError, read_metadata

def _read(path: Path) -> Any: return json.loads(path.read_text())
def _fail(message: str) -> None: raise ValueError(message)
def _job_dirs(archive: Path) -> list[Path]:
    jobs = archive / "jobs"
    if not jobs.is_dir(): _fail("archive_original_run must contain jobs/")
    base = jobs.resolve(); result = []
    for path in jobs.iterdir():
        if not path.is_dir(): continue
        resolved = path.resolve()
        if resolved.parent != base or resolved == base: _fail(f"job path escapes archive: {path}")
        result.append(resolved)
    return sorted(result)
def _safe_root_path(root: Path, relative: str) -> Path:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts: _fail(f"path escapes repository root: {relative}")
    path = (root / Path(*parsed.parts)).resolve(); base = root.resolve()
    if path != base and base not in path.parents: _fail(f"path escapes repository root: {relative}")
    return path
def _exact_equal(left: Any, right: Any) -> bool:
    # Constructor normalization int->float is exact, not a tolerance policy.
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_exact_equal(left[k], right[k]) for k in left)
    if type(left) in (int, float) and type(right) in (int, float):
        return bool(np.isfinite(left) and np.isfinite(right) and left == right)
    return type(left) is type(right) and left == right

def audit_eg_archive(archive_original_run: str | Path, repo_root: str | Path, registration_sha256: str, expected_valid_count: int, output: str | Path) -> dict[str, Any]:
    archive, root, out = map(Path, (archive_original_run, repo_root, output))
    if out.exists(): _fail(f"output must be a fresh directory: {out}")
    if expected_valid_count < 0 or not archive.is_dir(): _fail("invalid archive or expected_valid_count")
    registration_path = archive / "registration.json"
    if not registration_path.is_file() or file_sha(registration_path) != registration_sha256: _fail("registration_sha256 mismatch or registration.json missing")
    registration = read_metadata(registration_path); source_files = registration.get("source_files", {})
    if not isinstance(source_files, dict) or not source_files: _fail("registration has no source_files hash manifest")
    for relative, expected_hash in source_files.items():
        path = _safe_root_path(root, relative)
        if not path.is_file() or file_sha(path) != expected_hash: _fail(f"source hash mismatch: {relative}")
    if NumericallyRecoveredExponentiatedGradientAdapter.fit is not ExponentiatedGradientAdapter.fit:
        _fail("numerical recovery overrides the registered fit implementation")
    analysis_sources = {str(Path(__file__).resolve()): file_sha(Path(__file__))}
    for module in ("nhis_fairbias.benchmark.result_catalog", "nhis_fairbias.benchmark.adapters.adapter_reductions_numerical"):
        path = Path(__import__('sys').modules[module].__file__).resolve()
        if not path.is_relative_to(root.resolve() / 'src') and root.resolve() == Path(__file__).resolve().parents[1]:
            _fail("audit dependency loaded outside the source checkout")
        analysis_sources[str(path)] = file_sha(path)
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in registration.get("candidates", []):
        if candidate.get("status", "REGISTERED") != "REGISTERED" or candidate.get("method") not in {"EG_DP", "EG_EO"}: continue
        for raw_seed in candidate.get("seeds", []):
            if type(raw_seed) is not int or raw_seed < 0: _fail("invalid registered seed")
            key = (str(candidate["candidate_id"]), int(raw_seed))
            if key in expected: _fail(f"duplicate registered EG candidate/seed: {key}")
            expected[key] = candidate
    rows: list[dict[str, Any]] = []; seen: set[tuple[str, int]] = set(); valid = 0
    for job_dir in _job_dirs(archive):
        job_path = _safe_root_path(job_dir, "job.json")
        if not job_path.is_file(): _fail(f"job.json missing: {job_dir.name}")
        job = read_metadata(job_path); config, seed = job.get("config"), job.get("seed")
        if type(seed) is not int: _fail("invalid job seed")
        if not isinstance(config, dict): _fail(f"invalid job config: {job_dir.name}")
        key = (str(config.get("candidate_id")), seed)
        if key not in expected:
            if config.get('method') in {'EG_DP', 'EG_EO'}: _fail("unregistered EG job")
            continue
        if key in seen: _fail(f"duplicate EG job: {key}")
        seen.add(key)
        if config != expected[key]: _fail(f"candidate config identity mismatch: {job_dir.name}")
        if job_dir.name != f'{key[0]}_s{seed}': _fail("job directory identity mismatch")
        result_path, receipt_path = (_safe_root_path(job_dir, name) for name in ("result.json", "receipt.json"))
        if not result_path.is_file() or not receipt_path.is_file(): _fail(f"required EG terminal metadata missing: {job_dir.name}")
        receipt = read_metadata(receipt_path); files = receipt.get("files", {})
        if not isinstance(files, dict) or not {'job.json', 'result.json', 'worker.log'} <= files.keys():
            _fail("receipt missing mandatory artifacts")
        for name, digest in files.items():
            if PurePosixPath(name).name != name: _fail("receipt artifact path must be a basename")
            path = _safe_root_path(job_dir, name)
            if not path.is_file() or file_sha(path) != digest: _fail(f"receipt hash mismatch: {job_dir.name}/{name}")
        if type(receipt.get('returncode')) is not int or 'termination' not in receipt: _fail("invalid scheduler receipt")
        try:
            result = read_metadata(result_path, fields={"candidate_id", "seed", "status", "reload_verified", "source_identity", "data_identity", "output_type", "threshold"})
        except CatalogError as exc: _fail(f"unreadable result metadata: {job_dir.name}: {exc}")
        if result.get("candidate_id") != key[0] or int(result.get("seed")) != seed: _fail(f"candidate/seed result identity mismatch: {job_dir.name}")
        if job.get("source_identity") != registration.get("source_identity") or result.get("source_identity") != registration.get("source_identity"): _fail(f"source identity mismatch: {job_dir.name}")
        prepared = registration.get("prepared", {}).get(config.get("arm_id"), {})
        if job.get("data_identity") != prepared.get("data_identity") or job.get("data_sha256") != prepared.get("sha256"): _fail(f"data identity mismatch: {job_dir.name}")
        if 'data_identity' in result and result['data_identity'] != job['data_identity']: _fail("result data identity mismatch")
        if result.get("status") != "VALID":
            terminal = {'FAILED', 'BUDGET_EXHAUSTED', 'TIME_LIMIT', 'MEMORY_LIMIT', 'NOT_SUPPORTED', 'WORKER_FAILED'}
            if result.get('status') not in terminal: _fail("not a terminal EG status")
            if receipt['termination'] is not None and receipt['termination'] != result['status']: _fail("terminal scheduler mismatch")
            rows.append({"candidate_id": key[0], "seed": seed, "job": job_dir.name, "status": result.get("status"), "compatibility": "TERMINAL_NOT_COMPATIBLE"}); continue
        if receipt['returncode'] != 0 or receipt['termination'] is not None: _fail("VALID has unsuccessful scheduler receipt")
        if result.get('data_identity') != job['data_identity']: _fail("VALID result data identity missing")
        if not result.get("reload_verified"): _fail(f"VALID result is not reload-verified: {job_dir.name}")
        for name in ("model.joblib", "predictions_S.npz", "worker.log"):
            if not (job_dir / name).is_file() or name not in files or file_sha(job_dir / name) != files[name]: _fail(f"required VALID artifact/hash missing: {job_dir.name}/{name}")
        artifact = joblib.load(job_dir / "model.joblib")
        if artifact.get("config") != config or int(artifact.get("seed", -1)) != seed or artifact.get("data_identity") != job.get("data_identity"): _fail(f"model artifact identity mismatch: {job_dir.name}")
        policy, adapter = artifact.get("policy"), getattr(artifact.get("policy"), "_adapter", None)
        if type(adapter) is not ExponentiatedGradientAdapter: _fail(f"stored policy adapter is not original ExponentiatedGradientAdapter: {job_dir.name}")
        if getattr(policy, "threshold", "MISSING") is not None or result.get("threshold") is not None: _fail(f"EG policy has a calibration threshold: {job_dir.name}")
        if getattr(adapter, "output_type", None) != "decision_probability_q" or result.get("output_type") != "decision_probability_q": _fail(f"EG output type is not decision_probability_q: {job_dir.name}")
        params = config.get("params", {})
        for name in ("eps", "max_iter", "C", "difference_bound", "constraint_type", "estimator_params"):
            if name in params and not _exact_equal(getattr(adapter, name), params[name]): _fail(f"adapter/config {name} mismatch: {job_dir.name}")
        if int(getattr(adapter, "random_state", -1)) != seed: _fail(f"adapter/random_state seed mismatch: {job_dir.name}")
        if str(getattr(adapter, "backbone", "")) != str(config.get("backbone", "")).upper(): _fail(f"adapter/config backbone mismatch: {job_dir.name}")
        weights = getattr(getattr(adapter, "model", None), "weights_", None)
        if weights is None: _fail(f"EG model has no weights_: {job_dir.name}")
        weights = weights.to_numpy() if hasattr(weights, "to_numpy") else weights
        with np.load(job_dir / "predictions_S.npz", allow_pickle=False) as npz:
            if set(npz.files) != {"q"}: _fail(f"predictions_S.npz must contain q only: {job_dir.name}")
            q = np.asarray(npz["q"])
        if q.ndim != 1 or np.iscomplexobj(q) or not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)) or q.size == 0: _fail(f"stored q is invalid: {job_dir.name}")
        recovered = recover_convex_mixture_q(np.column_stack((1.0 - q, q)), weights)
        if not np.array_equal(recovered, q): _fail(f"stored q does not survive bounded PMF recovery: {job_dir.name}")
        valid += 1; rows.append({"candidate_id": key[0], "seed": seed, "job": job_dir.name, "status": "VALID", "model_sha256": file_sha(job_dir / "model.joblib"), "predictions_sha256": file_sha(job_dir / "predictions_S.npz"), "q_count": int(q.size), "q_semantics": "decision_probability_q", "compatibility": "COMPATIBLE_STORED_S_Q_ONLY"})
        del artifact, policy, adapter, weights, q, recovered
        if valid % 50 == 0: gc.collect()
    if seen != set(expected): _fail(f"registered EG coverage mismatch: expected {len(expected)}, observed {len(seen)}")
    if valid != int(expected_valid_count): _fail(f"expected_valid_count mismatch: expected {expected_valid_count}, observed {valid}")
    if file_sha(registration_path) != registration_sha256: _fail("registration changed during audit")
    for relative, digest in source_files.items():
        if file_sha(_safe_root_path(root, relative)) != digest: _fail("registered source changed during audit")
    if any(file_sha(Path(path)) != digest for path, digest in analysis_sources.items()): _fail("analysis source changed during audit")
    report = {"status": "COMPATIBLE_STORED_S_Q_ONLY", "registration_sha256": registration_sha256, "expected_eg_job_count": len(expected), "expected_valid_count": int(expected_valid_count), "observed_valid_count": valid, "jobs": rows, "analysis_sources": analysis_sources, "fit_implementation_inherited": True, "evaluation_authorized": False, "claim_boundary": "Stored S q compatibility only; no T compatibility, input equivalence, cross-hardware training equivalence, or selection authorization."}
    out.mkdir(parents=True); (out / "eg_compatibility_report.json").write_text(json.dumps(report, indent=2) + "\n"); return report

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--archive-original-run", required=True, type=Path); parser.add_argument("--repo-root", required=True, type=Path); parser.add_argument("--registration-sha256", required=True); parser.add_argument("--expected-valid-count", required=True, type=int); parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); audit_eg_archive(args.archive_original_run, args.repo_root, args.registration_sha256, args.expected_valid_count, args.output); return 0
if __name__ == "__main__": raise SystemExit(main())
