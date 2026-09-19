# NHIS benchmark environment inventory (2026-09-16)

This document records the local environments and dependency artifacts used by the FairBias/NHIS benchmark work. It is an inventory and reproducibility boundary, not a one-command installation recipe. The observations below are from source files, dependency manifests, installation artifacts, build logs, and existing validation reports. This document did not install packages, run models, or read NHIS data.

## Main Python environment

The primary interpreter is:

`/Users/lkc/Downloads/code_v_0_3/.venv311/bin/python`

The observed package versions are:

| Package | Version |
|---|---:|
| Python | 3.11 (environment path; exact patch version was not recorded in the inspected report) |
| numpy | 1.26.4 |
| pandas | 2.1.1 |
| scipy | 1.10.1 |
| scikit-learn | 1.9.1 |
| joblib | 1.5.3 |
| torch | 2.13.0 |
| fairlearn | 0.14.0 |
| aif360 | 0.6.1 |
| pytest | 9.1.1 |

Repository modules are resolved with `PYTHONPATH=src` in the existing benchmark test and worker workflows. These versions describe the observed local environment; they are not a claim that the public repository contains a complete lockfile or that every historical result used exactly this environment.

## FRAPPE isolated TensorFlow environment

FRAPPE is kept outside the main environment at:

`/Users/lkc/Downloads/code_v_0_3/artifacts/nhis/benchmark_dependencies_20260916/frappe_env/.venv`

The recorded Apple Silicon TensorFlow artifact is:

`artifacts/nhis/benchmark_dependencies_20260916/frappe_env/wheels/tensorflow_macos-2.14.0-cp311-cp311-macosx_12_0_arm64.whl`

Its recorded size is 199,710,290 bytes and its SHA256 is `064e98b67d7a89e72c37c90254c0a322a0b8d0ce9b68f23286816210e3ef6685`. The recorded `tensorflow-model-remediation` 0.1.7.1 wheel is 142,041 bytes with SHA256 `0d90dcb8e27ab245ac61582d63f7422b30b344686cda698ffdd0a975daf631da`. The source of record is:

`artifacts/nhis/benchmark_dependencies_20260916/frappe_env/wheel_manifest.json`

The official FRAPPE source snapshot is under:

`artifacts/nhis/benchmark_dependencies_20260916/frappe_source/`

and is pinned to Google Research commit `dbddc6626ce5363f48139d461bd8d131216d5722`. Source hashes and the admission record are in [frappe_admission.md](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_codex_takeover_20260915_154733Z/frappe_admission.md).

The adapter used the TensorFlow interpreter while adding the main environment's site-packages through a `PYTHONPATH` bridge so repository sklearn/fairness modules remained importable. This is a bridged environment, not a fully independent lock: `frappe_env/.venv/pyvenv.cfg` records Python 3.11.15 arm64 and `include-system-site-packages = true`, created with `python -m venv --system-site-packages`. The recorded bridge was:

`PYTHONPATH=src:.venv311/lib/python3.11/site-packages artifacts/nhis/benchmark_dependencies_20260916/frappe_env/.venv/bin/python scripts/check_frappe_runtime.py`

The runtime check recorded a successful synthetic fit, fresh reload, and repeated-seed comparison (`fit_reload_ok`, `max_reload_delta=0.0`, `repeat_seed_delta=0.000000`). This is evidence for the bounded synthetic wrapper, not a claim that the two environments form a clean unified lockfile. The wheel manifest and `pip-resolution.json` lock the recorded TensorFlow artifacts and Python platform; the effective system-site package set is not treated as a complete FRAPPE lock. A fresh setup therefore requires both the isolated artifacts and the repository environment, with the bridge kept explicit.

## FairGBM native build

FairGBM artifacts are under:

`artifacts/nhis/benchmark_dependencies_20260916/fairgbm/`

The source snapshot is `fairgbm-0.9.14` at:

`artifacts/nhis/benchmark_dependencies_20260916/fairgbm/source/fairgbm-0.9.14/`

and the native local package path is:

`artifacts/nhis/benchmark_dependencies_20260916/fairgbm/local_site/lib/python3.11/site-packages`

The adapter records the FairGBM PyPI sdist SHA256 as `b9d69aef01740fea5956157f097aba6daeb6bdc35dba4449a41d4f9e3d59dc7a`; the corresponding package/download metadata is retained with the FairGBM artifacts. Build evidence is in `fairgbm/build/cmake_clang_configure.log`, `fairgbm/build/cmake_build.log`, `fairgbm/build/cmake_sdk_build.log`, and the source build cache.

The inspected cache records `/opt/homebrew/bin/cmake`, Apple Clang (`/usr/bin/clang` and `/usr/bin/clang++`), an arm64 build, and the Command Line Tools SDK path `/Library/Developer/CommandLineTools/SDKs/MacOSX26.2.sdk`, including the C++ standard-library include path. The extracted linker line includes `-arch arm64` and `-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`. An explicit `libomp` or `-fopenmp` flag/path was not present in the inspected linker line. The exact CMake version and any separate libomp setup must therefore be taken from the original build session log; they are not asserted here.

The logs contain more than one build attempt, including a clang path with a `cstdint` header failure and an SDK build with warnings. The presence of the native site-package artifact and later synthetic worker evidence does not make every logged build command interchangeable. A clean rebuild needs the recorded source, compiler/SDK selection, and any missing OpenMP setup from the original build record.

## R `survey` binary and R-version boundary

R survey artifacts are under:

`artifacts/nhis/benchmark_dependencies_20260916/r-survey/`

The local library is `r-survey/localRlib`; downloaded source and macOS ARM binaries are in `r-survey/downloads`. The retained packages include `survey` 4.5, `mitools` 2.7, and `minqa` 1.2.8. The package metadata shows:

* `survey` 4.5 was built under R 4.5.2 on `aarch64-apple-darwin20`.
* `minqa` 1.2.8 was built under R 4.5.0.
* `mitools` 2.7 was built under R 4.6.1.

The download checksum source is:

`artifacts/nhis/benchmark_dependencies_20260916/r-survey/downloads/SHA256SUMS.txt`

That file records, among others, `survey_4.5.tar.gz` as `8a2ab01759f9acf6000274255edf00e342dfbf320a39fb76d42594e4d262b519` and `macos-arm64-4.5-survey.tgz` as `92ace275320d5670f685f4e6651fac2791eb44e302ca496a55328eee6e543bd8`.

The completed R4.6.1 check is recorded in [luna_survey_linearization_report.md](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_codex_takeover_20260915_154733Z/luna_survey_linearization_report.md). Under R4.6.1, the local `survey` 4.5 package loaded successfully and was exercised on a 48-row synthetic three-stratum/two-PSU design. The independent Python/R covariance comparison reached maximum absolute differences of `5.42e-19` and `6.51e-19` for the reported ratio expressions; the ratio point estimate differed by `2.22e-16`, and paired balanced-accuracy point estimates and standard errors matched to displayed precision. The report also records the source and binary hashes used for that check.

The exact R4.6 binary index was unavailable (404), so the loaded ARM64 binary was the retained CRAN R4.5 build under R4.6.1. That is a documented local compatibility observation, not a general ABI guarantee. The comparison is synthetic/public-use design evidence only and does not establish an NHIS variance or survey-inference claim.

## Additional local Python artifacts

The benchmark dependency directory also retains wheel/source provenance for optional adapters. The `download_manifest.json` records `fairret` 0.1.3 (20,279 bytes, SHA256 `de47b7b50c3dbfb241b4b6e686d49543ab6c62ac3ea98a4b0c5cca765ca2b686`) and `oxonfair` 0.3 (51,981 bytes, SHA256 `b8667c73a78ce9e6199244fbebd84fbbb855b40424b290120fbbeedbce5103ab`). TabM source is retained under `tabm_source/`, pinned to official commit `28e47ae301c92ec37787dde1ce923a0793f405b4`, with source hashes in `tabm_source/manifest.json`; its recorded runtime dependency is `rtdl_num_embeddings==0.0.12`, wheel SHA256 `87fd61270118915cf40888f2164f3a1f0353edf702f15fd91e71a147d23c930d`.

These are local artifacts for the bounded adapters and are not asserted to be installed in every interpreter. Their installation locations and effective transitive dependencies must be checked against the target environment before reuse.

## Local artifacts, hashes, and public-clone boundary

The dependency artifacts above are local ignored material beneath `artifacts/nhis/benchmark_dependencies_20260916`. The public Git source tree supplies adapter and experiment code, but a normal clone does not supply these ignored wheels, native site-packages, R libraries, source snapshots, or their local installation logs. Reproduction therefore requires separately preserving or reacquiring the artifacts, verifying them against the manifests/checksum files, and rebuilding any native components.

The hash authorities used here are deliberately local and explicit: `frappe_env/wheel_manifest.json` for FRAPPE wheels, `r-survey/downloads/SHA256SUMS.txt` for R packages, and the FairGBM adapter plus its package metadata for the pinned FairGBM source distribution. Where an exact command, compiler flag, CMake version, or runtime compatibility log was not present in those sources, this document records the gap rather than inventing a one-click recipe.

STOP — waiting for Codex review.
