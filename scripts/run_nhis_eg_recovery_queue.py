#!/usr/bin/env python3
"""Dispatch exactly the 23 historical FAILED EG jobs after LFR is COMPLETE.

Only metadata is read before dispatch. The frozen numerical worker owns the
fit/evaluation contract; old jobs and results remain immutable. Waiting occurs
in this server process, not an AI polling loop.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run_nhis_numerical_recovery_queue as base

EG_METHODS = {"EG_DP", "EG_EO"}
EG_FAILED_COUNT = 23
FROZEN_WORKER = "scripts/run_nhis_numerical_recovery_worker.py"
FROZEN_WORKER_SHA256 = "48befdf0eaf7be5c3250a13aed00d7b347d1b01f20be6293ca21747c65918dfe"


def failure_jobs(original_run, manifest_path, registration, expected_count=EG_FAILED_COUNT):
    """Bind all and only registered EG FAILED jobs to original evidence."""
    original_run = Path(original_run).resolve()
    manifest = base._json(manifest_path)
    if (manifest.get("schema") != "failure_recovery_inventory_v1"
            or Path(manifest.get("original_run", "")).resolve() != original_run
            or manifest.get("registration_sha256") != base.file_sha(original_run / "registration.json")):
        raise base.SchedulerError("EG failure inventory origin/registration mismatch")
    registered_failed = {}
    seen = set()
    for config in registration.get("candidates", []):
        if config.get("method") not in EG_METHODS or config.get("status") != "REGISTERED":
            continue
        for seed in config["seeds"]:
            name = f"{config['candidate_id']}_s{seed}"
            if name in seen:
                raise base.SchedulerError("Duplicate registered EG job")
            seen.add(name)
            result = base._json(original_run / "jobs" / name / "result.json")
            if result.get("candidate_id") != config["candidate_id"] or result.get("seed") != seed:
                raise base.SchedulerError("Original EG result identity mismatch")
            if result.get("status") == "FAILED":
                registered_failed[name] = (config, seed)
    if len(registered_failed) != expected_count:
        raise base.SchedulerError("Original registered EG FAILED count differs from expected coverage")
    if not isinstance(manifest.get("failures"), list):
        raise base.SchedulerError("EG inventory lacks failures list")
    selected = {}
    for entry in manifest["failures"]:
        if not isinstance(entry, dict):
            raise base.SchedulerError("Malformed failure inventory entry")
        if entry.get("config", {}).get("method") not in EG_METHODS:
            continue  # The shared inventory also contains unrelated methods.
        name = entry.get("job_id")
        if name in selected or name not in registered_failed:
            raise base.SchedulerError("Inventory contains duplicate or non-FAILED registered EG job")
        config, seed = registered_failed[name]
        original_job = original_run / "jobs" / name / "job.json"
        result_path = original_job.parent / "result.json"
        if (entry.get("config") != config or entry.get("seed") != seed or entry.get("status") != "FAILED"
                or Path(entry.get("job_path", "")).resolve() != original_job
                or entry.get("job_sha256") != base.file_sha(original_job)
                or entry.get("result_sha256") != base.file_sha(result_path)):
            raise base.SchedulerError("EG inventory job/config/seed/evidence mismatch")
        job = base._json(original_job)
        data = registration["prepared"][config["arm_id"]]
        required = {"config": config, "seed": seed, "source_identity": registration["source_identity"],
                    "data_identity": data["data_identity"], "data_sha256": data["sha256"]}
        if any(job.get(k) != value or entry.get(k) != value for k, value in required.items()):
            raise base.SchedulerError("EG inventory/original job differs from registered identity")
        if (entry.get("representation_key") is not None or job.get("representation_key") is not None
                or base._representation_key(config, seed, data["data_identity"]) is not None):
            raise base.SchedulerError("EG recovery must not adopt a representation cache key")
        selected[name] = {"config": config, "seed": seed, "original_job_path": str(original_job),
                          "original_job_sha256": base.file_sha(original_job),
                          "original_result_sha256": base.file_sha(result_path),
                          "original_representation_key": None}
    if set(selected) != set(registered_failed):
        raise base.SchedulerError("EG failure inventory does not exactly cover all original FAILED EG jobs")
    return selected


def lfr_complete(run, runtime_id, registration_sha, expected_count=960):
    """Only scheduler completion releases slots; no claim that fits succeeded."""
    run = Path(run).resolve()
    status_path = run / "parallel_scheduler_live_status.json"
    if not status_path.exists():
        return False
    status = base._json(status_path)
    if status.get("status") == "INTERRUPTED":
        raise base.SchedulerError("LFR queue interrupted; EG dispatch remains blocked")
    if status.get("status") != "COMPLETE":
        return False
    dispatch = base._json(run / "recovery_dispatch_manifest.json")
    if (not run.name.endswith("_" + runtime_id)
            or dispatch.get("runtime_source_identity") != runtime_id
            or dispatch.get("registration_sha256") != registration_sha
            or dispatch.get("selected_jobs") != expected_count
            or status.get("run") != str(run)
            or status.get("methods") != ["LFR_RECONSTRUCTED"]
            or status.get("scheduled_count") != expected_count
            or status.get("finalized_count") != expected_count or status.get("active_count") != 0):
        raise base.SchedulerError("LFR COMPLETE evidence is inconsistent")
    return base.external_worker_count((run,)) == 0


def wait_for_lfr(run, runtime_id, registration_sha, poll_seconds=15.0):
    if not 0 < poll_seconds <= 60:
        raise base.SchedulerError("LFR wait interval must be positive and at most 60 seconds")
    while not lfr_complete(run, runtime_id, registration_sha):
        time.sleep(poll_seconds)


class EGRecoveryScheduler(base.NumericalRecoveryScheduler):
    def discover(self):
        self._validate_sources()
        self._validate_parallel_policy()
        self._write_manifest()
        jobs = []
        for name in sorted(self._selected_names):
            selected = self.selected[name]
            config, seed = selected["config"], selected["seed"]
            if (config.get("method") not in EG_METHODS
                    or selected["original_representation_key"] is not None
                    or name != f"{config['candidate_id']}_s{seed}"):
                raise base.SchedulerError("EG discovery rejects non-EG or cache-bearing jobs")
            jobs.append(base.JobSpec(config=config, seed=seed, path=self.run_path / "jobs" / name,
                data=self.registration["prepared"][config["arm_id"]],
                source_identity=self.registration["source_identity"], representation_key=None))
        self.jobs = jobs
        return jobs

    def _write_manifest(self):
        super()._write_manifest()
        contract = {"schema": "eg_failed_recovery_dispatch_v1", "expected_failed_jobs": len(self.selected),
            "runtime_source_identity": self._runtime_identity,
            "queue_source_files": {"scripts/run_nhis_eg_recovery_queue.py": base.file_sha(__file__),
                "scripts/run_nhis_numerical_recovery_queue.py": base.file_sha(base.__file__)},
            "selected_original_evidence": {name: {k: value for k, value in row.items() if k != "config"}
                                           for name, row in sorted(self.selected.items())}}
        path = self.run_path / "eg_dispatch_contract.json"
        if path.exists() and base._json(path) != contract:
            raise base.SchedulerError("EG dispatch contract changed")
        if not path.exists() and not base.write_json_once(path, contract):
            raise base.SchedulerError("EG dispatch contract creation raced")


def namespace(root, runtime_id):
    if len(runtime_id) != 64 or any(c not in "0123456789abcdef" for c in runtime_id):
        raise base.SchedulerError("EG namespace needs the complete SHA256 runtime identity")
    return Path(root).resolve() / ("eg_recovery_" + runtime_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "original-run", "failure-manifest", "runtime-plan", "policy", "wait-for-lfr-run"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--external-run", type=Path, action="append", required=True,
                        help="Other live runs, including each FRAPPE run; LFR wait run is added automatically")
    parser.add_argument("--workers", type=int, default=28)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.workers <= min(28, base._effective_cpu_count()):
            raise base.SchedulerError("EG worker count exceeds the 28-slot or effective CPU bound")
        plan = base._runtime_plan(args.runtime_plan.resolve())
        if (plan.get("worker_script", FROZEN_WORKER) != FROZEN_WORKER
                or plan["runtime_source_files"].get(FROZEN_WORKER) != FROZEN_WORKER_SHA256):
            raise base.SchedulerError("EG plan must bind the frozen accepted numerical worker")
        original = args.original_run.resolve()
        registration = base._json(original / "registration.json")
        selected = failure_jobs(original, args.failure_manifest, registration)
        target = namespace(args.run, plan["runtime_source_identity"])
        lfr = args.wait_for_lfr_run.resolve()
        external = tuple(sorted({p.resolve() for p in args.external_run} | {lfr}))
        if (target.exists() or original == target or original in target.parents or target in original.parents
                or original in external
                or any(p == target or p in target.parents or target in p.parents for p in external)):
            raise base.SchedulerError("EG namespace must be fresh and independent of original/external runs")
        policy = base._json(args.policy)
        total_rss, worker_rss = int(policy.get("max_total_rss_bytes", 0)), int(policy.get("worker_rss_bytes", 0))
        fit_seconds = float(policy.get("fit_seconds", 0))
        if min(total_rss, worker_rss, fit_seconds) <= 0:
            raise base.SchedulerError("Invalid EG resource policy")
        _, maximum = base.cgroup_memory()
        if total_rss > maximum - base.HOST_RESERVE_BYTES:
            raise base.SchedulerError("EG policy must preserve the 12 GiB memory reserve")
        bound_paths = [original / "registration.json", args.failure_manifest.resolve(), args.policy.resolve(),
                       Path(__file__).resolve(), Path(base.__file__).resolve()]
        before_wait = {path: base.file_sha(path) for path in bound_paths}
        print("Waiting for the bound LFR queue to complete before EG dispatch.", flush=True)
        wait_for_lfr(lfr, plan["runtime_source_identity"], base.file_sha(original / "registration.json"))
        # Fail closed if the manifest or runtime changed during the server wait.
        if any(base.file_sha(path) != digest for path, digest in before_wait.items()):
            raise base.SchedulerError("EG metadata/policy/queue source changed during LFR wait")
        if base._runtime_plan(args.runtime_plan.resolve()) != plan:
            raise base.SchedulerError("Runtime plan changed during LFR wait")
        if failure_jobs(original, args.failure_manifest, registration) != selected:
            raise base.SchedulerError("Original EG failure set changed during LFR wait")
        plan["failure_manifest_path"] = str(args.failure_manifest.resolve())
        target.mkdir(parents=True, exist_ok=False)
        scheduler = EGRecoveryScheduler(target, original_run=original, selected=selected,
            runtime_plan=plan, runtime_plan_path=args.runtime_plan, external_runs=external,
            worker=ROOT / FROZEN_WORKER, source_script=Path(__file__).resolve(),
            limits=base.ResourceLimits(workers=args.workers, max_workers=args.workers,
                total_rss_bytes=total_rss, worker_rss_bytes=worker_rss, fit_seconds=fit_seconds),
            methods=sorted(EG_METHODS), policy_path=args.policy.resolve())
        print(base.json.dumps(scheduler.run()), flush=True)
        return 0
    except (base.SchedulerError, ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"EG recovery queue refused to run: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
