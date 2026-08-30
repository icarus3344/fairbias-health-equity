"""Empirical validation script for the reworked FairBias pipeline.

Runs run_fairbias_pipeline on COMPAS and Credit in BOTH algorithm modes
(Round 4.1; mode renamed twice — the second rename is the Round 4.1
REPAIR-2 verdict of 2026-08-30):

- ``official_code_derived_monotone_cursor_unweighted``: an
  OFFICIAL-CODE-DERIVED VARIANT WITH A TERMINATION-SAFETY EXTENSION —
  derived from the official code repository's behavior, NOT claimed to
  be behaviorally equivalent to it (the monotone per-attribute stream
  cursor is a deliberate deviation from the official restart-from-head
  search) and NOT a paper-text method reproduction (the paper text
  prescribes elbow-plot MDS dimension selection).  MDS fixed dim=2,
  official interleaved power stream [3, 1/3, ..., 1999, 1/1999] (order
  preserved), NO iteration budget, NO Pareto rollback — the greedy
  termination state is the sole reported state.
- ``engineering_bounded``: automatic MDS dim, six-value grid, bounded
  iterations, dual terminal states (configured greedy terminal + Pareto
  checkpoint).

One execution generates EXACTLY FOUR runs (both datasets × both modes).
The engineering runs use ``max_iterations=10`` (the pre-declared
comparable budget for the round-4/4.1 empirical comparison); the
official runs have no budget, so ``max_iterations`` is irrelevant for
them.

Prints per-iteration: selected feature, max d_phi, ACC and EO, plus
termination records and per-mode terminal-state metrics.
"""

import json
import time

from fairbias.config import ALGORITHM_MODE_OFFICIAL, FairBiasConfig
from fairbias.pipeline import run_fairbias_pipeline


def _eo_mean(metrics):
    eo = metrics.get("EO", 0.0)
    if isinstance(eo, dict):
        vals = [float(v) for v in eo.values()]
        return sum(vals) / len(vals) if vals else 0.0
    return float(eo)


def run_and_report(cfg_factory, name, max_iterations=10, mode="engineering"):
    # Default max_iterations=10: the pre-declared comparable budget for
    # the round-4/4.1 empirical comparison, so ONE execution of this
    # script generates exactly the four declared runs (Round 4.1
    # REPAIR-2: the previous default of 3 silently produced two extra
    # engineering runs that the report failed to disclose).
    cfg = cfg_factory(
        max_iterations=max_iterations,
        output_dir="runs/rework_empirical",
        mode=mode,
    )
    print(f"\n===== {name} [mode={cfg.algorithm_mode}] "
          f"(seed={cfg.random_seed}) =====")
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

    if res.algorithm_mode == ALGORITHM_MODE_OFFICIAL:
        print(f"[official_code_derived_monotone_cursor_unweighted] "
              f"ACC={res.greedy_terminal_metrics['ACC']:.4f} "
              f"EO={_eo_mean(res.greedy_terminal_metrics):.4f} "
              f"(test, greedy termination state — sole reported state, "
              f"no Pareto rollback in this mode)")
    else:
        print(f"[pareto_engineering] best_iteration={res.best_iteration} "
              f"reason={res.best_selection_reason}")
        print(f"[configured_greedy_terminal] ACC={res.greedy_terminal_metrics['ACC']:.4f} "
              f"EO={_eo_mean(res.greedy_terminal_metrics):.4f} "
              f"(test, configured greedy termination state — no "
              f"paper-alignment claim)")
        print(f"[pareto_engineering] ACC={res.pareto_engineering_metrics['ACC']:.4f} "
              f"EO={_eo_mean(res.pareto_engineering_metrics):.4f} "
              f"(test, ENGINEERING Pareto checkpoint — not the paper output)")
    if res.non_convergence is not None:
        scope = res.non_convergence.get("search_scope")
        if scope == "official_power_stream":
            scope_note = ("official power stream exhausted under the "
                          "monotone cursor")
        elif scope == "categorical_merge_chain":
            scope_note = "categorical merge chain exhausted"
        else:
            scope_note = ("configured grid exhausted; not a paper-level "
                          "claim")
        print(f"[non-convergence] attribute={res.non_convergence['attribute']} "
              f"O={res.non_convergence['label_O']} "
              f"d_phi={res.non_convergence['d_phi']:.4f} "
              f"search_scope={scope} ({scope_note})")
    else:
        print("[non-convergence] none")
    print(f"[time] {elapsed:.1f}s")


if __name__ == "__main__":
    for mode in ("official", "engineering"):
        run_and_report(FairBiasConfig.compas_default, "COMPAS", mode=mode)
        run_and_report(FairBiasConfig.credit_default, "Credit", mode=mode)
