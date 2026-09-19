"""Read the effective Linux environment and exercise one tiny CUDA operation."""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    import torch
    from threadpoolctl import threadpool_info

    cpu = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    memory = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    report = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_quota_cores": None if cpu[0] == "max" else int(cpu[0]) / int(cpu[1]),
        "memory_limit_bytes": None if memory == "max" else int(memory),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                     if d.metadata.get("Name")},
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "nvidia_smi": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True).strip(),
        "threadpools": threadpool_info(),
        "real_data_rows_read": 0,
    }
    if not report["cuda_available"]:
        raise RuntimeError("CUDA calculation admission requires an available GPU")
    torch.set_num_threads(1)
    x = torch.ones((256, 256), device="cuda", dtype=torch.float32)
    actual = (x @ x).cpu().numpy()
    np.testing.assert_array_equal(actual, np.full((256, 256), 256, dtype=np.float32))
    report["cuda_calculation_verified"] = True
    report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ("cpu_quota_cores", "memory_limit_bytes",
        "cuda_runtime", "cuda_calculation_verified", "nvidia_smi", "real_data_rows_read")}))


if __name__ == "__main__":
    main()
