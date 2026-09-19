"""Separate scheduled-Joint pilot with an outer audited MDS-budget retry.

Read BOTH result.json and mds_retry_receipt.json: the unchanged base runner's
variant describes its controller, while the retry receipt names the combined
method and records every MDS attempt. Neither receipt replaces the other.
Default execution uses the existing Linux external process supervisor.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EFFECTIVE_VARIANT = "scheduled_joint_v1_with_mds_budget_retry_v1"
SOURCE_NAMES = (
    "scripts/pilot_joint_mds_retry.py",
    "scripts/pilot_scheduled_joint.py",
    "scripts/supervise_joint_pilot.py",
    "src/nhis_fairbias/benchmark/mds_budget_retry.py",
    "src/nhis_fairbias/benchmark/joint_budget_optimization.py",
    "src/nhis_fairbias/benchmark/joint_mds_numpy.py",
    "src/nhis_fairbias/benchmark/adapters/adapter_fairbias_scheduled.py",
)


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _snapshot_sources():
    payloads = {name: (ROOT / name).read_bytes() for name in SOURCE_NAMES}
    return {name: _sha_bytes(value) for name, value in payloads.items()}, payloads


def _load_sibling(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_joint_retry_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _retry_context():
    from nhis_fairbias.benchmark.mds_budget_retry import mds_budget_retry
    return mds_budget_retry()


def _fresh_output(args):
    if args.variant != "scheduled_joint_v1":
        raise ValueError("MDS retry pilot requires scheduled_joint_v1")
    output, registered = args.output.resolve(), args.run.resolve()
    if output.exists() or output == registered or registered in output.parents:
        raise ValueError("Output must be fresh and outside the registered run")
    return output


def run_worker(args):
    """Call the unchanged base runner inside the outer retry scope."""
    output = _fresh_output(args)
    hashes, payloads = _snapshot_sources()
    base = _load_sibling("pilot_scheduled_joint")
    stats, error_type, returned = None, None, False
    try:
        with _retry_context() as stats:
            base.run(args)
        returned = True
    except BaseException as exc:
        error_type = type(exc).__name__
        raise
    finally:
        # Preflight can fail before the base runner creates its fresh directory.
        # In that case the external supervisor still records the nonzero exit.
        if output.is_dir():
            observed = {}
            for name in hashes:
                try:
                    observed[name] = _sha_bytes((ROOT / name).read_bytes())
                except OSError:
                    observed[name] = None
            unchanged = {name: observed[name] == digest for name, digest in hashes.items()}
            base_result = output / "result.json"
            result_hash, base_status, result_read_error = None, None, None
            if base_result.is_file():
                result_bytes = base_result.read_bytes()
                result_hash = _sha_bytes(result_bytes)
                try:
                    base_status = json.loads(result_bytes).get("status")
                except (ValueError, AttributeError) as exc:
                    result_read_error = type(exc).__name__
            receipt = {
                "diagnostic_only": True, "effective_variant": EFFECTIVE_VARIANT,
                "base_controller_variant": args.variant, "paper_equivalent": False,
                "consumers_must_read_both_receipts": True,
                "base_result_filename": "result.json",
                "base_result_sha256": result_hash, "base_result_status": base_status,
                "base_result_read_error_type": result_read_error,
                "base_result_rewritten": False, "fit_success_established": False,
                "wrapper_status": "BASE_RUNNER_RETURNED" if returned else "WRAPPER_EXCEPTION",
                "wrapper_error_type": error_type,
                "candidate_id": args.candidate, "seed": args.seed,
                "search_seconds": args.search_seconds, "hard_seconds": args.hard_seconds,
                "ae_interval": args.ae_interval, "geometry_prefix": args.geometry_prefix,
                "source_hashes_before": hashes, "source_hashes_after": observed,
                "source_unchanged_flags": unchanged, "all_sources_unchanged_after": all(unchanged.values()),
                "retry_stats": stats, "S_T_evaluated_by_wrapper": False,
            }
            if not all(unchanged.values()):
                receipt["wrapper_status"] = "INTEGRITY_FAILURE"
            # The base runner already snapshots its own sources. Add only these
            # two new source snapshots, preserving the bytes captured before fit.
            for name in (SOURCE_NAMES[0], SOURCE_NAMES[3]):
                with (output / Path(name).name).open("xb") as target:
                    target.write(payloads[name])
            with (output / "mds_retry_receipt.json").open("x", encoding="utf-8") as target:
                json.dump(receipt, target, indent=2, allow_nan=False)
                target.write("\n")


def run_supervised(args):
    if sys.platform != "linux":
        raise RuntimeError("The MDS retry pilot launcher is Linux-only")
    output = _fresh_output(args)
    receipt = output.with_name(output.name + ".supervisor.json")
    log = output.with_name(output.name + ".stdout.log")
    if receipt.exists() or log.exists():
        raise FileExistsError("Supervisor artifacts already exist")
    hashes, _ = _snapshot_sources()
    supervisor = _load_sibling("supervise_joint_pilot")
    command = [sys.executable, "-B", str(ROOT / SOURCE_NAMES[0]), "--worker"]
    for flag, value in (
        ("--run", args.run.resolve()), ("--candidate", args.candidate),
        ("--variant", args.variant), ("--seed", args.seed), ("--output", output),
        ("--hard-seconds", args.hard_seconds), ("--search-seconds", args.search_seconds),
        ("--ae-interval", args.ae_interval), ("--geometry-prefix", args.geometry_prefix),
    ):
        command.extend([flag, str(value)])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    output.parent.mkdir(parents=True, exist_ok=True)
    return supervisor.supervise_command(
        command, receipt_path=receipt, stdout_path=log,
        elapsed_limit_seconds=args.hard_seconds + 15, memory_limit_bytes=4 * 1024 ** 3,
        sample_seconds=0.5, grace_seconds=5.0, environment=environment, cwd=ROOT,
        receipt_metadata={
            "effective_variant": EFFECTIVE_VARIANT, "runtime_source_hashes": hashes,
            "python_executable": str(Path(sys.executable).resolve()), "python_version": sys.version,
            "child_output": str(output), "consumers_must_read_both_receipts": True,
            "required_child_receipts": ["result.json", "mds_retry_receipt.json"],
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--variant", choices=["scheduled_joint_v1"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-seconds", type=int, default=420)
    parser.add_argument("--search-seconds", type=float, default=360)
    parser.add_argument("--ae-interval", type=int, default=3)
    parser.add_argument("--geometry-prefix", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.hard_seconds <= 1800 or not 0 < args.search_seconds <= args.hard_seconds - 30:
        parser.error("hard deadline must be 1..1800 seconds and reserve 30 seconds after search")
    if args.ae_interval < 1 or args.geometry_prefix < 0:
        parser.error("interval must be positive and geometry prefix nonnegative")
    if sys.platform != "linux":
        parser.error("Production MDS retry pilot is Linux-only")
    if args.worker:
        run_worker(args)
        return 0
    result = run_supervised(args)
    print(json.dumps({key: result[key] for key in ("status", "termination_reason", "exit_code", "elapsed_seconds")}))
    return 0 if result["status"] == "CHILD_PROCESS_EXITED" and result["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
