"""Synthetic validation tests for the additive CPU shard wrapper."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts.run_nhis_cpu_shard import (  # noqa: E402
    ROOT, SHARD_AUDIT_VERSION, SHARD_PLAN_VERSION, ShardedScheduler,
)
from nhis_fairbias.benchmark.parallel_execution import ResourceLimits, SchedulerError
from nhis_fairbias.benchmark.parallel_execution import JobSpec, SCHEDULER_VERSION


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    run = tmp_path / "run"
    (run / "jobs").mkdir(parents=True)
    data = run / "prepared.joblib"
    data.write_bytes(b"prepared")
    configs = [
        {"candidate_id": "plain", "arm_id": "a", "method": "UNMITIGATED",
         "status": "REGISTERED", "seeds": [0], "params": {}},
        {"candidate_id": "frappe", "arm_id": "a", "method": "FRAPPE_EO",
         "status": "REGISTERED", "seeds": [0, 1], "params": {}},
    ]
    registration = {"source_files": {}, "source_identity": "src",
                    "candidates": configs,
                    "prepared": {"a": {"path": str(data), "data_identity": "d",
                                         "sha256": _sha(data)}}}
    registration_path = run / "registration.json"
    registration_path.write_text(json.dumps(registration))
    plan = {"schema_version": SHARD_PLAN_VERSION,
            "registration_sha256": _sha(registration_path), "source_identity": "src",
            "hostnames": {"source": os.uname().nodename, "cpu": os.uname().nodename},
            "assignments": {"source": ["plain_s0", "frappe_s0"], "cpu": ["frappe_s1"]}}
    plan_path = run / "plan.json"
    plan_path.write_text(json.dumps(plan))
    activation = {"schema_version": SHARD_AUDIT_VERSION, "role": "source",
                  "plan_sha256": _sha(plan_path), "registration_sha256": _sha(registration_path),
                  "source_identity": "src"}
    activation_path = run / "activation.json"
    activation_path.write_text(json.dumps(activation))
    return run, plan_path, activation_path


def make_scheduler(run, plan, activation, role="source"):
    return ShardedScheduler(run, runner=ROOT / "scripts/run_nhis_benchmark.py",
        source_script=ROOT / "scripts/run_nhis_benchmark_parallel.py",
        limits=ResourceLimits(workers=1), plan_path=plan, activation_path=activation,
        role=role)


def test_exact_allowlist_and_completed_resume(tmp_path):
    run, plan, activation = fixture(tmp_path)
    scheduler = make_scheduler(run, plan, activation)
    config = scheduler.registration["candidates"][0]
    data = scheduler.registration["prepared"]["a"]
    path = run / "jobs" / "plain_s0"
    path.mkdir()
    spec = JobSpec(config, 0, path, data, "src", None)
    (path / "job.json").write_text(json.dumps(scheduler._job_payload(spec)))
    (path / "result.json").write_text(json.dumps({"status": "NOT_SUPPORTED",
        "candidate_id": "plain", "seed": 0}))
    (path / "worker.log").write_text("done\n")
    files = {name: _sha(path / name) for name in ("job.json", "result.json", "worker.log")}
    (path / "receipt.json").write_text(json.dumps({"scheduler_version": SCHEDULER_VERSION,
        "returncode": 0, "termination": None, "files": files}))
    jobs = scheduler.discover()
    assert {j.path.name for j in jobs} == {"frappe_s0"}
    assert scheduler.resumed == 1


@pytest.mark.parametrize("mutation", ["registration", "activation", "hostname"])
def test_integrity_and_host_binding_rejected(tmp_path, mutation):
    run, plan, activation = fixture(tmp_path)
    if mutation == "registration":
        reg = run / "registration.json"
        payload = json.loads(reg.read_text()); payload["source_identity"] = "tampered"
        reg.write_text(json.dumps(payload))
    elif mutation == "activation":
        payload = json.loads(activation.read_text()); payload["plan_sha256"] = "0" * 64
        activation.write_text(json.dumps(payload))
    else:
        payload = json.loads(plan.read_text()); payload["hostnames"]["source"] = "wrong-host"
        plan.write_text(json.dumps(payload))
        ap = json.loads(activation.read_text()); ap["plan_sha256"] = _sha(plan)
        activation.write_text(json.dumps(ap))
    with pytest.raises(SchedulerError):
        make_scheduler(run, plan, activation)


def test_cpu_admission_throttles_on_conservative_reserve(tmp_path):
    run, plan, activation = fixture(tmp_path)
    ap = json.loads(activation.read_text())
    ap.update(role="cpu", source_drain_complete_sha256="a" * 64,
              source_request_sha256="b" * 64)
    activation.write_text(json.dumps(ap))
    scheduler = ShardedScheduler(run, runner=ROOT / "scripts/run_nhis_benchmark.py",
        source_script=ROOT / "scripts/run_nhis_benchmark_parallel.py",
        limits=ResourceLimits(workers=24, max_workers=24, total_rss_bytes=52 * 1024**3),
        plan_path=plan, activation_path=activation, role="cpu")
    scheduler._running = {i: SimpleNamespace(last_rss_bytes=4 * 1024**3) for i in range(12)}
    assert scheduler._launchable_index([SimpleNamespace(representation_key=None)], 1) is None


def test_representation_key_closure_rejects_cross_shard(monkeypatch, tmp_path):
    run, plan, activation = fixture(tmp_path)
    payload = json.loads(plan.read_text())
    payload["assignments"] = {"source": ["plain_s0"], "cpu": ["frappe_s0", "frappe_s1"]}
    plan.write_text(json.dumps(payload))
    ap = json.loads(activation.read_text()); ap["plan_sha256"] = _sha(plan); activation.write_text(json.dumps(ap))
    original = ShardedScheduler._universe
    def fake_universe(self):
        universe, _ = original(self)
        return universe, {"plain_s0": "shared", "frappe_s0": "shared", "frappe_s1": "other"}
    monkeypatch.setattr(ShardedScheduler, "_universe", fake_universe)
    with pytest.raises(SchedulerError, match="representation key split"):
        make_scheduler(run, plan, activation)


@pytest.mark.skipif(sys.platform != "linux", reason="requires /proc process identity and kernel wait status")
@pytest.mark.parametrize('exit_code', [0, 7])
@pytest.mark.parametrize('parent_kind', ['handoff', 'shard'])
def test_handoff_parent_is_captured_and_receipts_keep_exit_status(tmp_path, exit_code, parent_kind):
    """The wrapper accepts handoff_nhis_parallel.py as the active parent."""
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    for name in ("run_nhis_benchmark_parallel.py", "handoff_nhis_parallel.py", "run_nhis_cpu_shard.py", "run_nhis_runtime_shard.py"):
        shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
    (root / "scripts/run_nhis_benchmark.py").write_text(
        'import json,pathlib,sys,time,os\n'
        'p=pathlib.Path(sys.argv[sys.argv.index("--job")+1]); j=json.loads(p.read_text())\n'
        'deadline=time.monotonic()+15\n'
        'while pathlib.Path(f"/proc/{os.getppid()}/stat").read_text().rsplit(")",1)[1].split()[0] != "T":\n'
        ' if time.monotonic()>deadline: sys.exit(95)\n'
        ' time.sleep(.02)\n'
        'time.sleep(.5)\n'
        f'if j["seed"] == 0 and {exit_code}: sys.exit({exit_code})\n'
        '(p.parent/"result.json").write_text(json.dumps({"status":"NOT_SUPPORTED","candidate_id":j["config"]["candidate_id"],"seed":j["seed"]}))\n')
    run = root / "run"
    (run / "jobs").mkdir(parents=True); (run / "prepared").mkdir()
    data = run / "prepared/a.joblib"; data.write_bytes(b"d")
    config = {"candidate_id":"c", "arm_id":"a", "backbone":"LR", "method":"UNMITIGATED",
              "status":"REGISTERED", "seeds":[0,1], "params":{}}
    reg = {"source_files":{},"source_identity":"s","candidates":[config],
           "prepared":{"a":{"data_identity":"d","sha256":_sha(data)}}}
    (run / "registration.json").write_text(json.dumps(reg))
    policy = {"status":"SUPERVISOR_AUTHORIZED_PARALLEL_DEVELOPMENT",
              "registration_sha256":_sha(run/"registration.json"),"max_workers":3,
              "max_total_rss_bytes":64*1024**3,"worker_rss_bytes":4*1024**3,"fit_seconds":1800}
    (run / "parallel_policy.json").write_text(json.dumps(policy))
    (run / "parallel_scheduler_live_status.json").write_text('{}')
    names = ["c_s0", "c_s1"]
    plan = {"schema_version":SHARD_PLAN_VERSION,"registration_sha256":_sha(run/"registration.json"),
            "source_identity":"s","hostnames":{"source":os.uname().nodename,"cpu":os.uname().nodename},
            "assignments":{"source":names,"cpu":[]}}
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "activation.json").write_text(json.dumps({"schema_version":SHARD_AUDIT_VERSION,"role":"source",
        "plan_sha256":_sha(run/"plan.json"),"registration_sha256":_sha(run/"registration.json"),"source_identity":"s"}))
    parent_script = 'handoff_nhis_parallel.py' if parent_kind == 'handoff' else 'run_nhis_cpu_shard.py'
    parent_args = [sys.executable, '-B', str(root/'scripts'/parent_script), '--run', str(run),
                   '--policy', str(run/'parallel_policy.json'), '--workers', '2']
    if parent_kind == 'shard':
        parent_args += ['--plan', str(run/'plan.json'), '--activation', str(run/'activation.json'), '--role', 'source']
    handoff = subprocess.Popen(parent_args, cwd=root,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while len(list((run/"jobs").glob("*/job.json"))) < 2 and time.monotonic() < deadline:
            time.sleep(.02)
        child_script = 'run_nhis_cpu_shard.py' if parent_kind == 'handoff' else 'run_nhis_runtime_shard.py'
        done = subprocess.run([sys.executable,"-B",str(root/'scripts'/child_script),"--run",str(run),
            "--plan",str(run/"plan.json"),"--activation",str(run/"activation.json"),"--role","source",
            "--workers","1","--policy",str(run/"parallel_policy.json"),"--old-pid",str(handoff.pid)],
            cwd=root, capture_output=True, text=True, timeout=20)
        assert done.returncode == 0, done.stdout + done.stderr
        assert handoff.wait(timeout=5) == -9
        assert all((p/"receipt.json").exists() for p in (run/"jobs").iterdir())
        receipts = [json.loads((p/"receipt.json").read_text()) for p in (run/"jobs").iterdir()]
        assert sorted(r["returncode"] for r in receipts) == sorted([0, exit_code])
        audits = list((run / 'operational_handoffs').glob('*_cpu_shard'))
        assert len(audits) == 1
        kernel = [json.loads(p.read_text()) for p in audits[0].glob('kernel_exit_*.json')]
        assert len(kernel) == 2
        assert all(k['state'] == 'Z' for k in kernel)
        assert sorted(k['wait_status'] for k in kernel) == sorted([0, exit_code << 8])
        assert {p.name for p in (run / 'jobs').iterdir()} == set(names)
    finally:
        if handoff.poll() is None:
            handoff.kill(); handoff.wait()


@pytest.mark.parametrize("bad", [
    {"source": ["plain_s0", "frappe_s0"], "cpu": ["frappe_s0"]},
    {"source": ["plain_s0"], "cpu": ["frappe_s1"]},
    {"source": ["plain_s0", "frappe_s0"], "cpu": ["frappe_s1", "unknown_s9"]},
])
def test_invalid_partition_rejected_before_discovery(tmp_path, bad):
    run, plan, activation = fixture(tmp_path)
    payload = json.loads(plan.read_text())
    payload["assignments"] = bad
    plan.write_text(json.dumps(payload))
    activation_payload = json.loads(activation.read_text())
    activation_payload["plan_sha256"] = _sha(plan)
    activation.write_text(json.dumps(activation_payload))
    with pytest.raises(SchedulerError):
        make_scheduler(run, plan, activation)


def test_cpu_partial_preexisting_job_fails_on_discovery(tmp_path):
    run, plan, activation = fixture(tmp_path)
    cpu_activation = run / "cpu_activation.json"
    cpu_payload = json.loads(activation.read_text())
    cpu_payload["role"] = "cpu"
    cpu_payload["source_drain_complete_sha256"] = "a" * 64
    cpu_payload["source_request_sha256"] = "b" * 64
    cpu_activation.write_text(json.dumps(cpu_payload))
    (run / "jobs" / "frappe_s1").mkdir()
    scheduler = make_scheduler(run, plan, cpu_activation, role="cpu")
    with pytest.raises(SchedulerError, match="incomplete immutable attempt"):
        scheduler.discover()
