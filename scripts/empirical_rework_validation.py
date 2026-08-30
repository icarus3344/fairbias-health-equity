"""Empirical validation script for the reworked FairBias pipeline.

Runs run_fairbias_pipeline on COMPAS and Credit, printing per-iteration:
selected feature, max d_phi, ACC and EO, plus best iteration selection.
"""

import time

from fairbias.config import FairBiasConfig
from fairbias.pipeline import run_fairbias_pipeline


def _eo_mean(metrics):
    eo = metrics.get("EO", 0.0)
    if isinstance(eo, dict):
        vals = [float(v) for v in eo.values()]
        return sum(vals) / len(vals) if vals else 0.0
    return float(eo)


def run_and_report(cfg_factory, name, max_iterations=3):
    cfg = cfg_factory(max_iterations=max_iterations, output_dir="runs/rework_empirical")
    print(f"\n===== {name} (seed={cfg.random_seed}, iterations={max_iterations}) =====")
    t0 = time.time()
    res = run_fairbias_pipeline(cfg)
    elapsed = time.time() - t0

    print(f"[init] ACC={res.initial_metrics['ACC']:.4f} (validation) "
          f"EO={_eo_mean(res.initial_metrics):.4f} "
          f"max_dphi={max(v for g in res.initial_epsilon.values() for v in g.values()):.4f} "
          f"epsilon_threshold={res.epsilon_threshold:.4f}")
    for o_col, d in res.initial_epsilon.items():
        top = sorted(d.items(), key=lambda kv: -kv[1])[:3]
        print(f"  init d_phi top3 [{o_col}]: " + ", ".join(f"{k}={v:.4f}" for k, v in top))

    for it in res.iterations:
        sel = it["selected_attributes"]
        dropped = [k for k, v in it["changed_dict"].items() if v == "dropped"]
        print(f"[iter {it['iteration']}] attr={sel.get('selected_attribute')} "
              f"O={sel.get('selected_label_O')} "
              f"max_dphi={it['max_epsilon']:.4f} avg_dphi={it['avg_epsilon']:.4f} "
              f"ACC={it['metrics']['ACC']:.4f} (validation) EO={_eo_mean(it['metrics']):.4f} "
              f"dropped={dropped or '-'}")

    print(f"[pareto] best_iteration={res.best_iteration} reason={res.best_selection_reason}")
    print(f"[final] ACC={res.final_metrics['ACC']:.4f} EO={_eo_mean(res.final_metrics):.4f} (test, single locked-in evaluation)")
    print(f"[time] {elapsed:.1f}s")


if __name__ == "__main__":
    run_and_report(FairBiasConfig.compas_default, "COMPAS")
    run_and_report(FairBiasConfig.credit_default, "Credit")
