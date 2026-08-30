"""Empirical validation script for the reworked FairBias pipeline.

Runs run_fairbias_pipeline on COMPAS and Credit, printing per-iteration:
selected feature, max d_phi, ACC and EO, plus best iteration selection.
"""

import json
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

    with open(res.output_file, encoding="utf-8") as f:
        payload = json.load(f)
    split = payload["split"]
    rc = split["row_counts"]
    total = sum(rc.values())
    obs = split["observed_fractions"]
    print(f"  split (observed): train={rc['train']} ({obs['train']:.4f}) "
          f"validation={rc['validation']} ({obs['validation']:.4f}) "
          f"test={rc['test']} ({obs['test']:.4f}) / total={total}")

    for it in res.iterations:
        sel = it["selected_attributes"]
        dropped = [k for k, v in it["changed_dict"].items() if v == "dropped"]
        print(f"[iter {it['iteration']}] attr={sel.get('selected_attribute')} "
              f"O={sel.get('selected_label_O')} "
              f"max_dphi={it['max_epsilon']:.4f} avg_dphi={it['avg_epsilon']:.4f} "
              f"ACC={it['metrics']['ACC']:.4f} (validation) EO={_eo_mean(it['metrics']):.4f} "
              f"dropped={dropped or '-'}")

    print(f"[termination] converged={res.termination['converged']} "
          f"reason={res.termination['termination_reason']} "
          f"terminal_iteration={res.termination['terminal_iteration']} "
          f"terminal_max_dphi={res.termination['terminal_max_dphi']:.7f} "
          f"epsilon={res.termination['epsilon_threshold']:.7f}")

    print(f"[pareto_engineering] best_iteration={res.best_iteration} "
          f"reason={res.best_selection_reason}")
    print(f"[paper_strict] ACC={res.paper_strict_metrics['ACC']:.4f} "
          f"EO={_eo_mean(res.paper_strict_metrics):.4f} "
          f"(test, greedy termination state of the paper algorithm)")
    print(f"[pareto_engineering] ACC={res.pareto_engineering_metrics['ACC']:.4f} "
          f"EO={_eo_mean(res.pareto_engineering_metrics):.4f} "
          f"(test, ENGINEERING Pareto checkpoint — not the paper output)")
    if res.non_convergence is not None:
        print(f"[non-convergence] attribute={res.non_convergence['attribute']} "
              f"O={res.non_convergence['label_O']} "
              f"d_phi={res.non_convergence['d_phi']:.4f} "
              f"search_scope={res.non_convergence.get('search_scope')} "
              f"(configured grid exhausted; not a paper-level claim)")
    else:
        print("[non-convergence] none")
    print(f"[time] {elapsed:.1f}s")


if __name__ == "__main__":
    run_and_report(FairBiasConfig.compas_default, "COMPAS")
    run_and_report(FairBiasConfig.credit_default, "Credit")
