"""Run aggregate F/C-only FairBias geometry diagnostics for one prepared job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from nhis_fairbias.benchmark.bm_geometry_diagnostics import diagnose_bm_geometry, replay_fairbias_fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-joblib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categorical", nargs="+", help="semantic categorical columns; inferred when omitted")
    parser.add_argument("--numeric", nargs="+", help="semantic numeric columns; inferred when omitted")
    parser.add_argument("--replay-fit", action="store_true", help="also replay the registered F-only BM representation fit")
    parser.add_argument("--job-json", type=Path, help="registered job.json; required with --replay-fit")
    args = parser.parse_args()
    data = joblib.load(args.prepared_joblib)
    F = data["partitions"]["fitting_F"]
    C = data["partitions"]["calibration_C"]
    if args.numeric is None or args.categorical is None:
        # Infer only the diagnostic schema from the canonical semantic frame.
        # It is never fed back into the registered adapter/preprocessor.
        from nhis_fairbias.benchmark.preprocessing import NUMERICAL_FEATURES
        inferred_numeric = [c for c in F.X_semantic.columns if c in NUMERICAL_FEATURES]
        inferred_categorical = [c for c in F.X_semantic.columns if c not in inferred_numeric]
        numeric = args.numeric if args.numeric is not None else inferred_numeric
        categorical = args.categorical if args.categorical is not None else inferred_categorical
    else:
        numeric, categorical = args.numeric, args.categorical
    result = diagnose_bm_geometry(F.X_semantic, F.A, cate_attrs=categorical,
                                 num_attrs=numeric, X_C=C.X_semantic, A_C=C.A)
    if args.replay_fit:
        job_path = args.job_json
        if job_path is None or not job_path.exists():
            raise FileNotFoundError("--replay-fit requires --job-json pointing to a registered job")
        job = json.loads(job_path.read_text())
        result["fit_replay"] = replay_fairbias_fit(
            F.X_semantic, F.A, y_F=F.y, X_C=C.X_semantic, A_C=C.A,
            config=job["config"], seed=int(job["seed"]),
        )
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
