# AutoDL migration and controlled parallel execution

The user has authorized deployment to their rented AutoDL instance, resource allocation, changing parallelism bounds, and starting the study. Local real-data training remains paused. This authorizes necessary SSH transfer of this project's NHIS public-use development data and source, and installation of dependencies on the designated instance. No MEPS data is included.

## Verified target and storage

- SSH alias: `fairbias-autodl`; root project: `/root/autodl-tmp/fairbias`.
- Ubuntu 22.04, Linux x86_64, RTX 4090 D 24564 MiB, driver 580.76.05.
- The cgroup CPU quota is 1600000/100000 (16 CPU equivalents); memory.max is 85899345920 (80 GiB). Host totals of 192 CPUs and about 1 TiB must not be used as job limits. The purchase screenshot advertises 15 cores; normal development parallelism is capped below both values.
- Project data, environments, logs and package caches go on the 50 GB data disk. Space is monitored; the disk is not a backup.

## Migration integrity

The live working tree contains uncommitted substantive work. Transfer the current source with an explicit SHA-256 manifest, never replace it with an old Git clone. Preserve frozen inherited files and all local historical artifacts. The previous batch remains an immutable historical execution; do not import its macOS model objects or incomplete job as successful Linux execution. Prepare new F/C/S partitions from identical 2022/2023 input hashes, then register a new Linux run with current source and package provenance. Keep 2024 off the initial development bundle.

## Environment

Create a dedicated Python 3.11 conda prefix and match the observed local numerical/fairness packages using `configs/server/autodl_main_constraints.txt`. PyTorch 2.13.0 CUDA 12.6 is supplied by the official index; the container's preinstalled CUDA toolkit does not set the wheel's runtime version. Record the resolved package report and verify an actual CUDA calculation. Build Linux native FairGBM from the retained official source; never copy macOS libraries. FRAPPE uses its own TensorFlow 2.14.0 runtime and pinned remediation package. R survey verification is a separate statistical acceptance step.

Authorized installation endpoints for this deployment: HTTPS `mirrors.tuna.tsinghua.edu.cn` (image's configured conda/PyPI mirror), `repo.anaconda.com`, `conda.anaconda.org`, `pypi.org`, `files.pythonhosted.org`, `download.pytorch.org` and its official download CDN redirects, and CRAN source mirrors when required for the statistical check. Retain dependency metadata, checksums and installation logs. The user-provided SSH host is authorized for project transfer and execution. Credentials and private keys stay outside the project.

## Resource policy and launch gates

1. Verify synthetic adapter fits, fresh-process serialization, package consistency, portable resource units, and scheduler/cache safety before formal execution.
2. Start at 8 independent CPU workers with one BLAS/OpenMP thread per worker; operational bounds 4–12 workers. Keep the registered 4 GiB per-worker and 1800 second limits unless a separately documented method budget assessment justifies an amendment. Resource utilization is measured inside the container, not inferred from host load averages.
3. Require exclusive scheduler ownership of a run; shared representation fits have a single writer and dependent jobs wait for a complete cache artifact. Validate result receipts when resuming; preserve interrupted attempts.
4. Adjust concurrency using throughput and memory observations, preserving reserve for SSH, scheduling, data loading and optional GPU workers. A chosen concurrency is an operational setting, not a claimed mathematical optimum.
5. Existing neural adapters run on CPU. CUDA availability alone does not change them. GPU use requires explicit device-path validation and a separately recorded policy; avoid silently mixing CPU and GPU runs in a method comparison.
6. Preserve the known LFR 5000-function-evaluation nonconvergence record. Do not present excluded LFR results as a FairBias win. Any recovered LFR budget receives training-only feasibility checks and a versioned amendment before selection.
7. Initial launch is development only. Candidate selection and 2024 evaluation require complete validated receipts, the existing selection freeze, and statistical acceptance.

## Acceptance evidence

Retain input/source manifests, environment installation reports, remote checks, synthetic test outputs, scheduling policy, launched PID/log path, and first verified real-job receipts. A running scheduler is not evidence of scientific superiority or study completion.
