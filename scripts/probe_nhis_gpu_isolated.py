#!/usr/bin/env python3
"""Isolated synthetic TabM CUDA feasibility probe.

The probe is diagnostic-only.  It never loads NHIS data, benchmark runs,
representation caches, or selection/evaluation artifacts.  The default mode
compares the registered compact TabM CPU reference with an independent CUDA
training loop copied from the adapter's training semantics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TABM_SOURCE = ROOT / "artifacts/nhis/benchmark_dependencies_20260916/tabm_source/tabm.py"
TABM_ADAPTER = ROOT / "src/nhis_fairbias/benchmark/adapters/adapter_tabm.py"
MLP_ADAPTER = ROOT / "src/nhis_fairbias/benchmark/adapters/adapter_fairret.py"
PROBE_VERSION = "gpu_probe_v1_20260916"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_once(path: Path, value: Any) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _validate_output_dir(path: Path, *, fresh: bool = True) -> Path:
    resolved = Path(path).expanduser().resolve()
    if any(part.startswith("benchmark_autodl_") for part in resolved.parts):
        raise ValueError("output may not be inside a benchmark_autodl_* run directory")
    if fresh and resolved.exists():
        raise ValueError(f"--output must name a new directory: {resolved}")
    if not resolved.parent.exists():
        raise ValueError("output parent must already exist")
    if fresh:
        resolved.mkdir()
    return resolved


def _ensure_probe_output(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return _validate_output_dir(resolved)
    if not resolved.is_dir() or any(resolved.iterdir()):
        raise ValueError(f"probe output must be a new empty directory: {resolved}")
    if any(part.startswith("benchmark_autodl_") for part in resolved.parts):
        raise ValueError("output may not be inside a benchmark_autodl_* run directory")
    return resolved


def _source_manifest() -> dict[str, Any]:
    paths = {
        "probe_script": Path(__file__).resolve(),
        "tabm_adapter": TABM_ADAPTER,
        "mlp_adapter": MLP_ADAPTER,
        "official_tabm_source": TABM_SOURCE,
    }
    return {name: {"path": str(path), "sha256": file_sha(path)} for name, path in paths.items()}


def _import_references() -> tuple[Any, Any, Any, str | None]:
    from nhis_fairbias.benchmark.adapters.adapter_tabm import TabMClassifier, _load_official_tabm

    mlp_error = None
    try:
        from nhis_fairbias.benchmark.adapters.adapter_fairret import TorchMLPClassifier
    except Exception as exc:  # optional fairret environment
        TorchMLPClassifier = None
        mlp_error = f"{type(exc).__name__}: {exc}"
    return TabMClassifier, TorchMLPClassifier, _load_official_tabm, mlp_error


def _make_data(rows: int, features: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, features)).astype("float32")
    score = X[:, 0] - 0.35 * X[:, 1] + 0.2 * X[:, 2] + 0.15 * rng.normal(size=rows)
    y = (score > np.median(score)).astype("float32")
    return X, y


def _initial_state(torch: Any, TabM: Any, n_features: int, seed: int, cfg: dict[str, Any]):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TabM.make(
            n_num_features=n_features, cat_cardinalities=[], d_out=1,
            k=cfg["k"], n_blocks=cfg["n_blocks"], d_block=cfg["d_block"],
            dropout=cfg["dropout"], arch_type="tabm-packed",
        ).float()
        return copy.deepcopy(model.state_dict())


def _predict_model(torch: Any, model: Any, X: Any, batch_size: int):
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            logits = model(x_num=X[start : start + batch_size]).squeeze(-1)
            chunks.append(torch.sigmoid(logits).mean(dim=1))
    return torch.cat(chunks) if chunks else torch.empty(0, device=X.device)


def _gpu_train(torch: Any, TabM: Any, X_np: Any, y_np: Any, initial_state: dict[str, Any],
               seed: int, cfg: dict[str, Any]) -> tuple[Any, list[float], float, float, Any]:
    fit_started = time.perf_counter()
    device = torch.device("cuda")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TabM.make(
            n_num_features=X_np.shape[1], cat_cardinalities=[], d_out=1,
            k=cfg["k"], n_blocks=cfg["n_blocks"], d_block=cfg["d_block"],
            dropout=cfg["dropout"], arch_type="tabm-packed",
        ).float()
        model.load_state_dict(initial_state)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        criterion = torch.nn.BCEWithLogitsLoss()
        X = torch.from_numpy(X_np).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(y_np.reshape(-1, 1)).to(device=device, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            _ = model(x_num=X[: min(cfg["batch_size"], len(X))])
        torch.cuda.synchronize()
        model.load_state_dict(initial_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        losses: list[float] = []
        started = time.perf_counter()
        torch.cuda.synchronize()
        model.train()
        for _ in range(cfg["epochs"]):
            epoch_losses = []
            order = torch.randperm(len(X_np), device="cpu")
            for start in range(0, len(X_np), cfg["batch_size"]):
                stop = min(start + cfg["batch_size"], len(X_np))
                batch = order[start:stop].to(device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_num=X[batch]).squeeze(-1)
                loss = criterion(logits, y[batch].expand_as(logits))
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite CUDA TabM training loss")
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            losses.append(float(sum(epoch_losses) / len(epoch_losses)))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        fit_elapsed = time.perf_counter() - fit_started
        probabilities = _predict_model(torch, model, X, cfg["batch_size"])
        torch.cuda.synchronize()
    return model, losses, elapsed, fit_elapsed, probabilities.detach().cpu().numpy()


def _reload_check(output: Path) -> int:
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = json.loads((output / "probe_manifest.json").read_text())
    data = np.load(output / "synthetic_input.npz")
    X = torch.from_numpy(data["X"]).float()
    expected_cpu = np.load(output / "predictions_cpu.npz")["p"]
    expected_gpu = np.load(output / "predictions_gpu.npz")["p"]
    from nhis_fairbias.benchmark.adapters.adapter_tabm import _load_official_tabm

    cfg = metadata["configuration"]
    TabM = _load_official_tabm()
    checks = {}
    portability = {}
    for name, expected in (("cpu", expected_cpu), ("gpu", expected_gpu)):
        state = torch.load(output / f"model_{name}_state.pt", map_location="cpu", weights_only=True)
        model = TabM.make(
            n_num_features=cfg["features"], cat_cardinalities=[], d_out=1,
            k=cfg["k"], n_blocks=cfg["n_blocks"], d_block=cfg["d_block"],
            dropout=cfg["dropout"], arch_type="tabm-packed",
        ).float()
        model.load_state_dict(state["state_dict"])
        device = "cuda" if name == "gpu" else "cpu"
        model.to(device)
        p = _predict_model(torch, model, X.to(device), cfg["batch_size"]).cpu().numpy()
        checks[name] = bool(np.allclose(p, expected, rtol=0.0, atol=1e-7))
        if name == "gpu":
            model.cpu()
            portable_p = _predict_model(torch, model, X, cfg["batch_size"]).numpy()
            portability["gpu_state_on_cpu_max_probability_delta"] = float(np.max(np.abs(portable_p - expected)))
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "fresh_process": True, "reload_device_matches_training_device": True,
              "atol": 1e-7, "rtol": 0.0, "portability_diagnostic": portability}
    _json_once(output / "reload_check.json", result)
    return 0 if result["status"] == "PASS" else 2


def run_probe(output: Path, rows: int = 4096, epochs: int = 100, seed: int = 20260916) -> int:
    import numpy as np
    import torch

    output = _ensure_probe_output(output)
    TabMClassifier, TorchMLPClassifier, load_official_tabm, mlp_error = _import_references()
    source_hashes = _source_manifest()
    cfg = {
        "model": "TabM",
        "features": 24,
        "rows": rows,
        "epochs": epochs,
        "k": 4,
        "n_blocks": 2,
        "d_block": 64,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 1024,
        "random_state": seed,
        "dtype": "float32",
        "tf32": False,
        "mixed_precision": False,
    }
    manifest = {
        "probe_version": PROBE_VERSION,
        "status": "STARTING",
        "diagnostic_only": True,
        "real_data_loaded": False,
        "benchmark_paths_used": False,
        "source_hashes": source_hashes,
        "cpu_reference_imports": {
            "TabMClassifier": "imported",
            "TorchMLPClassifier": "imported" if TorchMLPClassifier is not None else f"unavailable: {mlp_error}",
        },
        "configuration": cfg,
    }
    _json_once(output / "probe_manifest.json", manifest)
    if not torch.cuda.is_available():
        result = {"status": "NO_CUDA", "diagnostic_only": True,
                  "cuda_available": False, "message": "torch.cuda.is_available() is false"}
        manifest["status"] = "NO_CUDA"
        (output / "probe_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _json_once(output / "result.json", result)
        return 0

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    X, y = _make_data(rows, cfg["features"], seed)
    np.savez_compressed(output / "synthetic_input.npz", X=X, y=y)
    initial = _initial_state(torch, load_official_tabm(), cfg["features"], seed, cfg)

    cpu = TabMClassifier(
        k=cfg["k"], n_blocks=cfg["n_blocks"], d_block=cfg["d_block"],
        dropout=cfg["dropout"], learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], random_state=seed,
    )
    cpu_started = time.perf_counter()
    cpu.fit(X, y)
    cpu_elapsed = time.perf_counter() - cpu_started
    with torch.no_grad():
        cpu_p = torch.from_numpy(cpu.predict_proba(X)[:, 1]).float().numpy()
    gpu, gpu_losses, gpu_elapsed, gpu_fit_elapsed, gpu_p = _gpu_train(
        torch, load_official_tabm(), X, y, initial, seed, cfg
    )
    cpu_state = {key: value.detach().cpu() for key, value in cpu.model_.state_dict().items()}
    gpu_state = {key: value.detach().cpu() for key, value in gpu.state_dict().items()}
    torch.save({"state_dict": cpu_state}, output / "model_cpu_state.pt")
    torch.save({"state_dict": gpu_state}, output / "model_gpu_state.pt")
    np.savez_compressed(output / "predictions_cpu.npz", p=cpu_p)
    np.savez_compressed(output / "predictions_gpu.npz", p=gpu_p)
    reload_command = [sys.executable, str(Path(__file__).resolve()), "--reload-check", str(output)]
    reload_run = subprocess.run(reload_command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                                capture_output=True, text=True, timeout=120)
    reload_status = "PASS" if reload_run.returncode == 0 else "FAIL"
    result = {
        "status": "COMPLETED" if reload_status == "PASS" and np.all(np.isfinite(gpu_p)) else "FAIL",
        "benchmark_equivalence": "NOT_ESTABLISHED",
        "diagnostic_only": True,
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "cpu_reference_fit_seconds": cpu_elapsed,
        "gpu_training_seconds": gpu_elapsed,
        "gpu_fit_total_seconds": gpu_fit_elapsed,
        "timing_scope": "CPU includes adapter fit setup and training. GPU total includes model setup, data transfer, warmup and training; both exclude prediction. GPU training alone excludes setup and warmup. One sequential pair under live server load; diagnostic only.",
        "cpu_final_loss": cpu.final_loss_,
        "gpu_final_loss": gpu_losses[-1],
        "cpu_finite_loss_count": cpu.finite_loss_count_,
        "gpu_finite_loss_count": len(gpu_losses),
        "event_probability_difference_max": float(np.max(np.abs(cpu_p - gpu_p))),
        "event_probability_difference_mean": float(np.mean(np.abs(cpu_p - gpu_p))),
        "threshold_05_label_mismatch": int(np.sum((cpu_p >= 0.5) != (gpu_p >= 0.5))),
        "reload_status": reload_status,
        "reload_stdout": reload_run.stdout[-1000:],
        "reload_stderr": reload_run.stderr[-1000:],
        "semantic_limitations": [
            "CPU reference uses TabMClassifier.fit; CUDA loop is an independent copied loop.",
            "CPU and CUDA floating-point reductions and optimizer kernels may differ.",
            "This probe does not establish benchmark equivalence, accuracy, fairness, or scientific validity.",
        ],
    }
    manifest["status"] = result["status"]
    (output / "probe_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _json_once(output / "result.json", result)
    return 0 if result["status"] == "COMPLETED" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--reload-check", type=Path)
    args = parser.parse_args(argv)
    if args.reload_check is not None:
        try:
            return _reload_check(_validate_output_dir(args.reload_check, fresh=False))
        except Exception as exc:
            print(f"reload check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    if args.output is None:
        parser.error("--output is required")
    if args.rows <= 0 or args.epochs <= 0:
        parser.error("--rows and --epochs must be positive")
    try:
        output = _validate_output_dir(args.output)
        return run_probe(output, rows=args.rows, epochs=args.epochs, seed=args.seed)
    except Exception as exc:
        print(f"GPU probe refused or failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
