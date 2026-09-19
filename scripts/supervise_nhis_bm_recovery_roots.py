#!/usr/bin/env python3
"""Gate and supervise two serial lanes of eight BM geometry diagnostics.

The default mode is a read-only preflight.  ``--release`` is required before
any child is started; each lane then runs four fresh F-only diagnostics in
series through ``supervise_joint_pilot.supervise_command``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOTS = 8
LANE_COUNT = 2
LANE_SECONDS = 900
MEMORY_BYTES = 4 * 1024**3
SOURCE_METHOD = "FAIRBIAS_BM"


class BMRecoveryGateError(RuntimeError):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:
        raise BMRecoveryGateError(f"unreadable JSON: {path}") from exc


def live_workers(run: Path) -> list[int]:
    """Find workers whose --job path belongs to one registered run."""
    run = Path(run).resolve()
    found: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [part.decode() for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
            if "worker" not in argv or "--job" not in argv:
                continue
            job = Path(argv[argv.index("--job") + 1]).resolve()
            if job.name == "job.json" and job.parent.parent == run / "jobs":
                found.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return sorted(set(found))


def source_ready(source_run: Path, *, expected_jobs: int = 100) -> tuple[bool, str]:
    status_path = Path(source_run) / "parallel_scheduler_live_status.json"
    if not status_path.is_file():
        return False, "source scheduler status is missing"
    status = read_json(status_path)
    required = {
        "status": "COMPLETE", "scheduled_count": expected_jobs,
        "finalized_count": expected_jobs, "failed_count": 0, "active_count": 0,
    }
    for key, expected in required.items():
        if status.get(key) != expected:
            return False, f"source status {key}={status.get(key)!r}, expected {expected!r}"
    workers = live_workers(source_run)
    if workers:
        return False, f"source run still has active workers: {workers}"
    return True, "source balanced run is COMPLETE with zero workers"


def _registered_jobs(registration: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    jobs: dict[str, Mapping[str, Any]] = {}
    for config in registration.get("candidates", []):
        if config.get("status") != "REGISTERED" or config.get("method") != SOURCE_METHOD:
            continue
        for seed in config.get("seeds", []):
            name = f"{config['candidate_id']}_s{int(seed)}"
            if name in jobs:
                raise BMRecoveryGateError(f"duplicate registered BM job: {name}")
            jobs[name] = {"config": config, "seed": int(seed)}
    return jobs


def select_geometry_roots(registered_run: Path, failure_manifest: Path,
                          *, expected_roots: int = EXPECTED_ROOTS) -> list[dict[str, Any]]:
    registered_run = Path(registered_run).resolve()
    registration_path = registered_run / "registration.json"
    registration = read_json(registration_path)
    manifest = read_json(failure_manifest)
    if manifest.get("schema") != "failure_recovery_inventory_v1":
        raise BMRecoveryGateError("unsupported failure inventory schema")
    if manifest.get("registration_sha256") != sha(registration_path):
        raise BMRecoveryGateError("failure inventory registration hash mismatch")
    registered = _registered_jobs(registration)
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest.get("failures", []):
        config = entry.get("config", {})
        cache_status = str(entry.get("cache_status", ""))
        if (config.get("method") != SOURCE_METHOD or entry.get("status") != "FAILED"
                or not cache_status.startswith("NOT_ESTIMABLE")):
            continue
        name = str(entry.get("job_id", ""))
        if name not in registered or entry.get("config") != registered[name]["config"]:
            raise BMRecoveryGateError(f"geometry failure is not a registered BM job: {name}")
        job_path = (registered_run / "jobs" / name / "job.json").resolve()
        if Path(entry.get("job_path", "")).resolve() != job_path or not job_path.is_file():
            raise BMRecoveryGateError(f"geometry failure job path mismatch: {name}")
        result_path = job_path.parent / "result.json"
        if (entry.get("job_sha256") != sha(job_path)
                or not result_path.is_file() or entry.get("result_sha256") != sha(result_path)):
            raise BMRecoveryGateError(f"geometry failure evidence hash mismatch: {name}")
        key = entry.get("representation_key")
        if not isinstance(key, str) or not key:
            raise BMRecoveryGateError(f"geometry failure lacks representation key: {name}")
        groups.setdefault(key, []).append({
            "job_id": name, "job_path": str(job_path), "job_sha256": sha(job_path),
            "result_sha256": sha(result_path), "representation_key": key,
            "config": config, "seed": int(entry["seed"]),
            "cache_status": cache_status,
        })
    if len(groups) != expected_roots:
        raise BMRecoveryGateError(f"expected {expected_roots} geometry roots, found {len(groups)}")
    selected = [min(entries, key=lambda item: item["job_id"]) for entries in groups.values()]
    selected.sort(key=lambda item: (item["representation_key"], item["job_id"]))
    if len({item["representation_key"] for item in selected}) != expected_roots:
        raise BMRecoveryGateError("selected geometry roots are not unique")
    return selected


def partition_lanes(roots: list[dict[str, Any]], lane_count: int = LANE_COUNT) -> list[list[dict[str, Any]]]:
    if len(roots) != EXPECTED_ROOTS or lane_count != LANE_COUNT:
        raise BMRecoveryGateError("BM recovery requires exactly eight roots and two lanes")
    midpoint = len(roots) // lane_count
    lanes = [roots[:midpoint], roots[midpoint:]]
    if any(len(lane) != midpoint for lane in lanes):
        raise BMRecoveryGateError("BM recovery lanes must be balanced")
    seen = [item["representation_key"] for lane in lanes for item in lane]
    if len(seen) != len(set(seen)):
        raise BMRecoveryGateError("representation root appears in more than one lane")
    return lanes


def _lane_command(root: Path, output: Path, python: Path, bm_script: Path) -> list[str]:
    return [str(python), "-B", str(bm_script), "--job", root["job_path"],
            "--output", str(output), "--recover"]


def run_lane(lane: list[dict[str, Any]], lane_index: int, *, output_root: Path,
             python: Path, bm_script: Path, supervisor_script: Path,
             registered_run: Path) -> list[dict[str, Any]]:
    sys.path.insert(0, str(supervisor_script.parent))
    from supervise_joint_pilot import supervise_command

    lane_root = output_root / f"lane_{lane_index}"
    lane_root.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    for item in lane:
        key = item["representation_key"]
        output = lane_root / key
        receipt = lane_root / f"{key}.supervisor.json"
        stdout = lane_root / f"{key}.stdout.log"
        environment = dict(os.environ, PYTHONHASHSEED="0",
                           PYTHONPATH=str(ROOT / "src"))
        result = supervise_command(
            _lane_command(item, output, python, bm_script),
            receipt_path=receipt, stdout_path=stdout,
            elapsed_limit_seconds=LANE_SECONDS, memory_limit_bytes=MEMORY_BYTES,
            sample_seconds=0.5, grace_seconds=5.0, environment=environment,
            cwd=ROOT, receipt_metadata={
                "lane": lane_index, "representation_key": key,
                "registered_run": str(registered_run),
                "job_sha256": item["job_sha256"], "diagnostic_only": True,
                "S_T_evaluated": False, "use_mds_retry": False,
            })
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--registered-run", required=True, type=Path)
    parser.add_argument("--failure-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--bm-script", type=Path, default=ROOT / "scripts/pilot_nhis_bm_recovery.py")
    parser.add_argument("--supervisor-script", type=Path, default=ROOT / "scripts/supervise_joint_pilot.py")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.python.is_absolute() or not args.python.is_file():
            raise BMRecoveryGateError("--python must be an existing absolute executable")
        if args.poll_seconds <= 0:
            raise BMRecoveryGateError("poll interval must be positive")
        roots = select_geometry_roots(args.registered_run, args.failure_manifest)
        lanes = partition_lanes(roots)
        ready, reason = source_ready(args.source_run)
        while args.release and args.wait and not ready:
            time.sleep(args.poll_seconds)
            ready, reason = source_ready(args.source_run)
        plan = {
            "schema": "nhis_bm_geometry_root_recovery_plan_v1",
            "source_ready": ready, "source_gate_reason": reason,
            "release_required": not args.release,
            "expected_roots": EXPECTED_ROOTS, "selected_roots": roots,
            "lanes": [[item["representation_key"] for item in lane] for lane in lanes],
            "python": str(args.python.resolve()), "python_hash_seed": "0",
            "elapsed_limit_seconds": LANE_SECONDS, "memory_limit_bytes": MEMORY_BYTES,
            "use_mds_retry": False, "S_T_evaluated": False,
        }
        if not args.release:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if not ready:
            raise BMRecoveryGateError(f"source gate not ready: {reason}")
        output_root = args.output_root.resolve()
        if output_root.exists():
            raise BMRecoveryGateError(f"fresh output root already exists: {output_root}")
        output_root.mkdir(parents=True)
        (output_root / "dispatch_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        with ThreadPoolExecutor(max_workers=LANE_COUNT) as pool:
            futures = [pool.submit(run_lane, lane, index, output_root=output_root,
                                    python=args.python.resolve(), bm_script=args.bm_script.resolve(),
                                    supervisor_script=args.supervisor_script.resolve(),
                                    registered_run=args.registered_run.resolve())
                       for index, lane in enumerate(lanes, 1)]
            results = [future.result() for future in futures]
        print(json.dumps({"status": "COMPLETE", "lanes": results}, sort_keys=True))
        return 0
    except (BMRecoveryGateError, FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"BM recovery gate refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
