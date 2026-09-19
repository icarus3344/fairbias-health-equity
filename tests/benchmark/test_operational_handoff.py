"""Linux-only, synthetic process tests for immutable dispatcher handoff."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="kernel wait status requires Linux")
ROOT = Path(__file__).resolve().parents[2]


def fixture_run(tmp_path, failure=False):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    for name in ("run_nhis_benchmark_parallel.py", "handoff_nhis_parallel.py"):
        shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
    (root / "scripts/run_nhis_benchmark.py").write_text('''
import json, pathlib, sys, time
p = pathlib.Path(sys.argv[sys.argv.index("--job")+1]); j=json.loads(p.read_text())
time.sleep(2)
if j["seed"] == 0 and j["config"]["params"].get("failure"):
    sys.exit(7)
(p.parent/"model.joblib").write_bytes(b"synthetic model")
(p.parent/"predictions_S.npz").write_bytes(b"synthetic predictions")
(p.parent/"result.json").write_text(json.dumps({"status":"VALID", "candidate_id":j["config"]["candidate_id"],
"seed":j["seed"],"source_identity":j["source_identity"],"data_identity":j["data_identity"],"reload_verified":True}))
''')
    run = root / "run"
    for folder in ("jobs", "prepared", "representation_cache"):
        (run / folder).mkdir(parents=True)
    data = run / "prepared/arm_001.joblib"
    data.write_bytes(b"synthetic prepared data")
    config = {"candidate_id": "synthetic", "arm_id": "arm_001", "backbone": "LR", "method": "UNMITIGATED",
              "status": "REGISTERED", "seeds": list(range(8)), "params": {"failure": failure}}
    reg = {"source_files": {}, "source_identity": "synthetic-source", "candidates": [config],
           "prepared": {"arm_001": {"data_identity": "synthetic-data", "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}}}
    (run / "registration.json").write_text(json.dumps(reg))
    policy = {"status": "SUPERVISOR_AUTHORIZED_PARALLEL_DEVELOPMENT",
              "registration_sha256": hashlib.sha256((run / "registration.json").read_bytes()).hexdigest(),
              "max_workers": 4, "max_total_rss_bytes": 64*1024**3, "worker_rss_bytes": 4*1024**3, "fit_seconds": 1800}
    (run / "parallel_policy.json").write_text(json.dumps(policy))
    log = (root / "original.log").open("w")
    original = subprocess.Popen([sys.executable, "-B", str(root / "scripts/run_nhis_benchmark_parallel.py"),
        "develop", "--run", str(run), "--workers", "3", "--parallel-policy", str(run / "parallel_policy.json")],
        cwd=root, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic()+10
    while len(list((run / "jobs").glob("*/job.json"))) < 3:
        if original.poll() is not None or time.monotonic()>deadline:
            raise RuntimeError((root / "original.log").read_text())
        time.sleep(.01)
    return root, run, original, log


@pytest.mark.parametrize("failure", [False, True])
def test_drain_preserves_real_exit_status_and_resumes_without_duplicates(tmp_path, failure):
    root, run, original, log = fixture_run(tmp_path, failure)
    try:
        done = subprocess.run([sys.executable, "-B", str(root / "scripts/handoff_nhis_parallel.py"),
            "--run", str(run), "--policy", str(run / "parallel_policy.json"), "--workers", "4",
            "--old-pid", str(original.pid), "--methods", "UNMITIGATED"], cwd=root,
            capture_output=True, text=True, timeout=30)
        assert done.returncode == 0, done.stdout + done.stderr
        assert original.wait(timeout=5) == -9
        audit = next((run / "operational_handoffs").iterdir())
        drain = json.loads((audit / "drain_complete.json").read_text())
        assert drain["captured"] == drain["sealed"] == 3
        jobs = list((run / "jobs").iterdir())
        assert len(jobs) == 8
        assert all((p / "receipt.json").exists() for p in jobs)
        assert not list((run / "jobs").glob("*/.*.tmp"))
        receipt = json.loads((run / "jobs/synthetic_s0/receipt.json").read_text())
        assert receipt["returncode"] == (7 if failure else 0)
        evidence = [json.loads(p.read_text()) for p in audit.glob("kernel_exit_*.json")]
        assert all(e["state"] == "Z" for e in evidence)
        assert any(e["wait_status"] == (7 << 8 if failure else 0) for e in evidence)
        assert json.loads((audit / "resume_started.json").read_text())["workers"] == 4
        assert json.loads((run / "parallel_scheduler_live_status.json").read_text())["status"] == "COMPLETE"
        assert json.loads((audit / "queue_exit.json").read_text())["resumed"] == 3
    finally:
        if original.poll() is None:
            original.kill()
            original.wait()
        log.close()
