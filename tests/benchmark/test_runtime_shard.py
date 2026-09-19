"""Synthetic tests for the deterministic runtime recovery shard."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_nhis_runtime_shard import (ROOT, SHARD_AUDIT_VERSION,
    SHARD_PLAN_VERSION, ShardedScheduler)
from nhis_fairbias.benchmark.parallel_execution import JobSpec, ResourceLimits, SchedulerError


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_run(tmp_path):
    run = tmp_path / "run"
    (run / "jobs").mkdir(parents=True)
    data = run / "prepared.joblib"; data.write_bytes(b"prepared")
    configs = [
        {"candidate_id":"plain", "arm_id":"a", "backbone":"LR", "method":"UNMITIGATED",
         "status":"REGISTERED", "seeds":[0], "params":{}},
        {"candidate_id":"frappe", "arm_id":"a", "backbone":"LR", "method":"FRAPPE_EO",
         "status":"REGISTERED", "seeds":[0], "params":{}},
    ]
    registration = {"source_files":{}, "source_identity":"synthetic-source",
        "candidates":configs, "prepared":{"a":{"path":str(data), "data_identity":"d", "sha256":sha(data)}}}
    rp = run / "registration.json"; rp.write_text(json.dumps(registration))
    plan = {"schema_version":SHARD_PLAN_VERSION, "registration_sha256":sha(rp),
        "source_identity":"synthetic-source", "hostnames":{"source":os.uname().nodename, "cpu":os.uname().nodename},
        "assignments":{"source":["plain_s0","frappe_s0"],"cpu":[]},
        "runtime_variant":"frappe_pipeline_deterministic_v1_20260917",
        "runtime_environment":{"TF_DETERMINISTIC_OPS":"1"}}
    from nhis_fairbias.benchmark.experiment_registry import identity
    files = {name: sha(ROOT/name) for name in ('scripts/run_nhis_runtime_worker.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_frappe_pipeline.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_frappe.py')}
    plan['runtime_source_files'] = files
    plan['runtime_source_identity'] = identity({'variant':plan['runtime_variant'], 'files':files,
                                               'environment':plan['runtime_environment']})
    pp = run / "plan.json"; pp.write_text(json.dumps(plan))
    activation = {"schema_version":SHARD_AUDIT_VERSION,"role":"source","plan_sha256":sha(pp),
        "registration_sha256":sha(rp),"source_identity":"synthetic-source"}
    ap = run / "activation.json"; ap.write_text(json.dumps(activation))
    return run, pp, ap


def scheduler(run, plan, activation):
    return ShardedScheduler(run, runner=ROOT / "scripts/run_nhis_benchmark.py",
        source_script=ROOT / "scripts/run_nhis_benchmark_parallel.py",
        limits=ResourceLimits(workers=1), plan_path=plan, activation_path=activation, role="source")


def test_runtime_environment_only_applies_to_frappe_and_is_recorded(tmp_path):
    run, plan, activation = setup_run(tmp_path)
    s = scheduler(run, plan, activation)
    configs = {c["candidate_id"]: c for c in s.registration["candidates"]}
    data = s.registration["prepared"]["a"]
    frappe = JobSpec(configs["frappe"], 0, run / "jobs/frappe_s0", data, "synthetic-source", None)
    plain = JobSpec(configs["plain"], 0, run / "jobs/plain_s0", data, "synthetic-source", None)
    assert s._runtime_environment(frappe) == {"TF_DETERMINISTIC_OPS":"1"}
    assert s._runtime_environment(plain) == {}
    payload = s._job_payload(frappe)
    assert payload["runtime_variant"] == "frappe_pipeline_deterministic_v1_20260917"
    assert payload["runtime_environment"] == {"TF_DETERMINISTIC_OPS":"1"}
    assert payload["runtime_plan_sha256"] == sha(plan)
    assert "runtime_variant" not in s._job_payload(plain)


@pytest.mark.parametrize("field,value", [
    ("runtime_variant", "unknown"),
    ("runtime_environment", {"TF_DETERMINISTIC_OPS":"0"}),
])
def test_unknown_runtime_contract_rejected(tmp_path, field, value):
    run, plan, activation = setup_run(tmp_path)
    payload = json.loads(plan.read_text()); payload[field] = value; plan.write_text(json.dumps(payload))
    ap = json.loads(activation.read_text()); ap["plan_sha256"] = sha(plan); activation.write_text(json.dumps(ap))
    with pytest.raises(SchedulerError, match="runtime"):
        scheduler(run, plan, activation)


def test_runtime_plan_and_activation_tampering_rejected(tmp_path):
    run, plan, activation = setup_run(tmp_path)
    payload = json.loads(plan.read_text()); payload["source_identity"] = "tampered"; plan.write_text(json.dumps(payload))
    ap = json.loads(activation.read_text()); ap["plan_sha256"] = sha(plan); activation.write_text(json.dumps(ap))
    with pytest.raises(SchedulerError):
        scheduler(run, plan, activation)


def test_runtime_audit_binds_wrapper_and_activation_hash(tmp_path):
    run, plan, activation = setup_run(tmp_path)
    s = scheduler(run, plan, activation)
    s._write_shard_audit()
    audit = json.loads((run / "shard_source_activation.json").read_text())
    assert audit["activation_sha256"] == sha(activation)
    assert audit["wrapper_sha256"] == sha(ROOT / "scripts/run_nhis_runtime_shard.py")


def test_runtime_environment_preserves_scheduler_thread_contract(tmp_path):
    run, plan, activation = setup_run(tmp_path); s = scheduler(run, plan, activation)
    config = next(c for c in s.registration["candidates"] if c["method"] == "FRAPPE_EO")
    spec = JobSpec(config, 0, run / "jobs/frappe_s0", s.registration["prepared"]["a"], "synthetic-source", None)
    env = s._environment(spec)
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["TF_DETERMINISTIC_OPS"] == "1"


def test_runtime_discovery_keeps_registered_partition_and_filters_methods(tmp_path):
    run, plan, activation = setup_run(tmp_path)
    s = ShardedScheduler(run, runner=ROOT / "scripts/run_nhis_benchmark.py",
        source_script=ROOT / "scripts/run_nhis_benchmark_parallel.py", limits=ResourceLimits(workers=1),
        plan_path=plan, activation_path=activation, role="source", methods=["FRAPPE_EO"])
    assert [job.path.name for job in s.discover()] == ["frappe_s0"]


def test_runtime_missing_activation_is_fail_closed(tmp_path):
    run, plan, activation = setup_run(tmp_path)
    activation.unlink()
    with pytest.raises(SchedulerError, match="activation"):
        scheduler(run, plan, activation)


def test_runtime_hostname_binding_is_fail_closed(tmp_path):
    run, plan, activation = setup_run(tmp_path)
    payload = json.loads(plan.read_text()); payload["hostnames"]["source"] = "other-host"; plan.write_text(json.dumps(payload))
    ap = json.loads(activation.read_text()); ap["plan_sha256"] = sha(plan); activation.write_text(json.dumps(ap))
    with pytest.raises(SchedulerError, match="hostname"):
        scheduler(run, plan, activation)


def test_runtime_activation_registration_hash_is_fail_closed(tmp_path):
    run, plan, activation = setup_run(tmp_path)
    ap = json.loads(activation.read_text()); ap["registration_sha256"] = "0" * 64; activation.write_text(json.dumps(ap))
    with pytest.raises(SchedulerError, match="activation registration"):
        scheduler(run, plan, activation)
