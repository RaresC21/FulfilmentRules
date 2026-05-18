"""
Stochastic inventory allocation: comparison experiment.

For each (M, budget_factor) configuration we run T independent trials.
Each trial:
  1. Draw N_TRAIN demand scenarios  → find q via each method.
  2. Draw N_TEST  demand scenarios  → evaluate each q out-of-sample.

Gaps are relative to the SAA-LP solution evaluated on the same test set,
giving an apples-to-apples out-of-sample comparison.

Both evaluation metrics are recorded:
  - LP oracle  : each test scenario fulfilled by its own LP  (exact)
  - Rules      : each test scenario fulfilled by the heuristic (fast)

Output
------
  results/stochastic_results.json   — full per-trial data
  results/stochastic_summary.txt    — aggregate table (mean ± std, worst-case)
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))   # project root  (graph, fulfillment)
sys.path.insert(0, str(Path(__file__).parent))        # this folder   (stochastic_opt)

import numpy as np
import torch

from graph import WarehouseGraph, DemandSampler
from stochastic_opt import (
    solve_stochastic_lp,
    solve_gd,
    solve_fra,
    evaluate_lp_oracle,
    evaluate_rules,
)

# ── Experiment parameters ────────────────────────────────────────────────────

MEAN_DEMAND    = 10.0
STD_DEMAND     = 3.0
N_TRAIN_VALUES = [50, 100, 200, 500]  # scenarios used to find q (swept)
N_TEST         = 100    # held-out scenarios for out-of-sample evaluation
T_TRIALS       = 1      # independent train/test splits per config
M_VALUES       = [10, 15, 20, 40, 60, 80, 100]
# BUDGET_FACTORS = [0.8, 1.0, 1.2, 1.5]   # budget = factor * M * MEAN_DEMAND
BUDGET_FACTORS = [1.0, 1.5, 2.0]
SEED           = 42     # master seed for reproducibility

RESULTS_DIR = Path(__file__).parent / "results"


# ── Single trial ─────────────────────────────────────────────────────────────

def run_trial(
    graph: WarehouseGraph,
    budget: float,
    n_train: int,
    train_seed: int,
    test_seed: int,
) -> dict:
    """
    One train/test split.  Returns per-method costs and gaps on the test set.
    """
    train_demands = DemandSampler(graph, mean=MEAN_DEMAND, std=STD_DEMAND,
                                  seed=train_seed).sample(n_train)
    test_demands  = DemandSampler(graph, mean=MEAN_DEMAND, std=STD_DEMAND,
                                  seed=test_seed).sample(N_TEST)

    # ── Find q with each method (trained on train_demands) ────────────────
    q_lp,  _, t_lp  = solve_stochastic_lp(train_demands, graph, budget)
    q_gd,  _, t_gd  = solve_gd(train_demands, graph, budget)
    q_fra, t_fra    = np.array([1]), np.array([1]) #solve_fra(train_demands, graph, budget)

    # ── Evaluate all q's on held-out test_demands ─────────────────────────
    def eval_both(q):
        return evaluate_lp_oracle(q, test_demands, graph), \
               evaluate_rules(q, test_demands, graph)

    lp_ora,  lp_rul  = eval_both(q_lp)
    gd_ora,  gd_rul  = eval_both(q_gd)
    fra_ora, fra_rul = 1, 1 # eval_both(q_fra)

    # Gaps relative to LP on the same test set
    ref_ora = max(lp_ora, 1e-9)
    ref_rul = max(lp_rul, 1e-9)

    return {
        "lp"  : {"cost_oracle": lp_ora,  "cost_rules": lp_rul,
                 "solve_time": t_lp,  "gap_oracle": 1.0, "gap_rules": 1.0,
                 "q": q_lp.tolist()},
        "gd"  : {"cost_oracle": gd_ora,  "cost_rules": gd_rul,
                 "solve_time": t_gd,
                 "gap_oracle": gd_ora  / ref_ora,
                 "gap_rules" : gd_rul  / ref_rul,
                 "q": q_gd.tolist()},
        "fra" : {"cost_oracle": fra_ora, "cost_rules": fra_rul,
                 "solve_time": t_fra,
                 "gap_oracle": fra_ora / ref_ora,
                 "gap_rules" : fra_rul / ref_rul,
                 "q": q_fra.tolist()},
    }


# ── Aggregate T trials into summary stats ────────────────────────────────────

def _agg(trials: list[dict], method: str, metric: str) -> dict:
    vals = [t[method][metric] for t in trials]
    return {"mean": float(np.mean(vals)),
            "std" : float(np.std(vals)),
            "max" : float(np.max(vals)),
            "min" : float(np.min(vals)),
            "all" : vals}


def aggregate_trials(trials: list[dict]) -> dict:
    methods = ["lp", "gd", "fra"]
    metrics = ["cost_oracle", "cost_rules", "gap_oracle", "gap_rules", "solve_time"]
    return {
        m: {k: _agg(trials, m, k) for k in metrics}
        for m in methods
    }


# ── Formatting ───────────────────────────────────────────────────────────────

_HDR = (
    f"{'M':>4}  {'N_tr':>5}  {'Q/uM':>5}  "
    f"{'GD gap(ora)':>20}  {'FRA gap(ora)':>20}  "
    f"{'GD gap(rul)':>20}  {'FRA gap(rul)':>20}  "
    f"{'t_LP':>6}  {'t_GD':>6}  {'t_FRA':>6}"
)
_HDR2 = (
    f"{'':>4}  {'':>5}  {'':>5}  "
    f"{'mean+/-std  worst':>20}  {'mean+/-std  worst':>20}  "
    f"{'mean+/-std  worst':>20}  {'mean+/-std  worst':>20}  "
    f"{'':>6}  {'':>6}  {'':>6}"
)
_SEP = "-" * len(_HDR)


def _fmt_agg(a: dict) -> str:
    """Format 'mean+/-std  worst' for one metric aggregated over trials."""
    return f"{a['mean']:.4f}+/-{a['std']:.4f}  {a['max']:.4f}"


def _fmt_row(M: int, n_train: int, bf: float, agg: dict) -> str:
    gd  = agg["gd"]
    fra = agg["fra"]
    t_lp  = agg["lp"]["solve_time"]["mean"]
    t_gd  = agg["gd"]["solve_time"]["mean"]
    t_fra = agg["fra"]["solve_time"]["mean"]
    return (
        f"{M:>4}  {n_train:>5}  {bf:>5.2f}  "
        f"{_fmt_agg(gd['gap_oracle']):>20}  {_fmt_agg(fra['gap_oracle']):>20}  "
        f"{_fmt_agg(gd['gap_rules']):>20}  {_fmt_agg(fra['gap_rules']):>20}  "
        f"{t_lp:>5.1f}s  {t_gd:>5.1f}s  {t_fra:>5.1f}s"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    master_rng = np.random.RandomState(SEED)

    RESULTS_DIR.mkdir(exist_ok=True)

    all_configs: list[dict] = []
    lines: list[str] = [_HDR, _HDR2, _SEP]

    total_configs = len(M_VALUES) * len(BUDGET_FACTORS) * len(N_TRAIN_VALUES)
    done = 0

    for M in M_VALUES:
        graph = WarehouseGraph(M, edge_prob=0.2, seed=SEED)
        print(f"\n{'='*70}\nM = {M}  ({graph.n_edges} edges,  N_TEST={N_TEST}  T={T_TRIALS})")
        print(_HDR)
        print(_HDR2)
        print(_SEP)

        for bf in BUDGET_FACTORS:
            budget = bf * M * MEAN_DEMAND

            for n_train in N_TRAIN_VALUES:
                # Draw train/test seeds for each trial upfront (reproducible)
                trial_seeds = [
                    (int(master_rng.randint(0, 1_000_000)),
                     int(master_rng.randint(0, 1_000_000)))
                    for _ in range(T_TRIALS)
                ]

                trials: list[dict] = []
                t_config_start = time.perf_counter()

                for t, (train_seed, test_seed) in enumerate(trial_seeds):
                    print(f"  trial {t+1}/{T_TRIALS}  M={M} N_tr={n_train} Q={bf:.1f}uM ...", end=" ", flush=True)
                    t0 = time.perf_counter()
                    trial = run_trial(graph, budget, n_train, train_seed, test_seed)
                    trials.append(trial)
                    print(f"{time.perf_counter()-t0:.1f}s", flush=True)

                agg = aggregate_trials(trials)
                t_config = time.perf_counter() - t_config_start

                config_result = {
                    "M": M, "budget_factor": bf, "budget": budget,
                    "N_train": n_train, "N_test": N_TEST, "T": T_TRIALS,
                    "n_edges": graph.n_edges,
                    "trials": trials,
                    "agg": agg,
                }
                all_configs.append(config_result)

                row_line = _fmt_row(M, n_train, bf, agg)
                print(row_line)
                lines.append(row_line)

                done += 1
                print(
                    f"  [{done}/{total_configs}] config done in {t_config:.1f}s",
                    flush=True,
                )

    # ── Save ─────────────────────────────────────────────────────────────────
    def _to_serializable(obj):
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    results_file = RESULTS_DIR / "stochastic_results.json"
    with open(results_file, "w") as f:
        json.dump(_to_serializable(all_configs), f, indent=2)
    print(f"\n[OK] Results saved to {results_file}")

    summary_file = RESULTS_DIR / "stochastic_summary.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] Summary  saved to {summary_file}")

    # ── Overall aggregate stats ───────────────────────────────────────────────
    def _collect(metric, submeric):
        return [c["agg"][metric][submeric]["mean"] for c in all_configs]

    print(f"\nOverall (mean across all configs, of per-config means):")
    for method in ["gd", "fra"]:
        go  = np.mean(_collect(method, "gap_oracle"))
        gr  = np.mean(_collect(method, "gap_rules"))
        wco = max(c["agg"][method]["gap_oracle"]["max"] for c in all_configs)
        wcr = max(c["agg"][method]["gap_rules"]["max"]  for c in all_configs)
        print(f"  {method.upper():3s}  gap(oracle) mean={go:.4f}  worst={wco:.4f}  |"
              f"  gap(rules) mean={gr:.4f}  worst={wcr:.4f}")
