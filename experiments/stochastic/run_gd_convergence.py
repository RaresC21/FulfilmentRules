"""
GD convergence experiment: time vs. objective.

For each (M, N_train) configuration we run GD with patience-based early
stopping, recording (cumulative_wall_time, loss) at every step.

Output
------
  results/gd_convergence.json   — full per-config convergence curves
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from graph import WarehouseGraph, DemandSampler
from stochastic_opt import _rules_mean_cost

# ── Experiment parameters ────────────────────────────────────────────────────

MEAN_DEMAND  = 10.0
STD_DEMAND   = 3.0
BUDGET_FACTOR = 1.0          # budget = BUDGET_FACTOR * M * MEAN_DEMAND
SEED         = 42

M_VALUES       = [(i + 1) * 10 for i in range(20)]   # sweep problem sizes
N_TRAIN_VALUES = [(i + 1) * 10 for i in range(20)]    # sweep scenario counts

MAX_STEPS          = 1000
LR                 = 0.05
PATIENCE           = 20
MIN_REL_IMPROVEMENT = 1e-4

RESULTS_DIR = Path(__file__).parent / "results"


# ── GD with per-step timing ──────────────────────────────────────────────────

def solve_gd_convergence(
    demands: torch.Tensor,
    graph: WarehouseGraph,
    budget: float,
    max_steps: int = MAX_STEPS,
    lr: float = LR,
    patience: int = PATIENCE,
    min_rel_improvement: float = MIN_REL_IMPROVEMENT,
) -> dict:
    """
    Run Adam GD with patience-based early stopping, recording per-step timing.
    Returns a dict with:
      - 'times':  cumulative wall-clock seconds at each step
      - 'losses': training objective (rules mean cost) at each step
      - 'q':      best allocation seen
      - 'total_time': total elapsed seconds
    """
    M = demands.shape[1]
    theta = torch.zeros(M, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    best_val     = float("inf")
    best_q       = None
    steps_no_imp = 0
    times: list[float]  = []
    losses: list[float] = []

    t_start = time.perf_counter()
    for _ in range(max_steps):
        optimizer.zero_grad()
        q = budget * torch.softmax(theta, dim=0)
        loss = _rules_mean_cost(q, demands, graph)
        loss.backward()
        optimizer.step()

        val = loss.item()
        times.append(time.perf_counter() - t_start)
        losses.append(val)

        if val < best_val:
            rel_imp = (best_val - val) / max(abs(best_val), 1e-9)
            best_val = val
            best_q   = (budget * torch.softmax(theta.detach(), dim=0)).clone()
            steps_no_imp = 0 if rel_imp > min_rel_improvement else steps_no_imp + 1
        else:
            steps_no_imp += 1

        if steps_no_imp >= patience:
            break

    return {
        "times":      times,
        "losses":     losses,
        "q":          best_q.tolist(),
        "total_time": time.perf_counter() - t_start,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)

    rng = np.random.RandomState(SEED)
    all_configs: list[dict] = []

    total = len(M_VALUES) * len(N_TRAIN_VALUES)
    done = 0

    for M in M_VALUES:
        graph  = WarehouseGraph(M, edge_prob=1/M, seed=SEED)
        budget = BUDGET_FACTOR * M * MEAN_DEMAND

        train_seed = int(rng.randint(0, 1_000_000))

        print(f"\nM={M}  edges={graph.n_edges}  budget={budget:.1f}", flush=True)

        for N_train in N_TRAIN_VALUES:
            train_demands = DemandSampler(
                graph, mean=MEAN_DEMAND, std=STD_DEMAND, seed=train_seed
            ).sample(N_train)

            print(f"  GD  M={M}  N_train={N_train} ...", end=" ", flush=True)
            result = solve_gd_convergence(train_demands, graph, budget)
            print(f"{len(result['losses'])} steps  {result['total_time']:.2f}s  "
                  f"final_loss={result['losses'][-1]:.4f}", flush=True)

            all_configs.append({
                "M":             M,
                "N_train":       N_train,
                "budget_factor": BUDGET_FACTOR,
                "budget":        budget,
                "n_edges":       graph.n_edges,
                "n_steps":       len(result["losses"]),
                "lr":            LR,
                "gd_times":      result["times"],
                "gd_losses":     result["losses"],
                "gd_total_time": result["total_time"],
            })

            done += 1
            print(f"  [{done}/{total}] done", flush=True)

    out_file = RESULTS_DIR / "gd_convergence.json"
    with open(out_file, "w") as f:
        json.dump(all_configs, f, indent=2)
    print(f"\n[OK] Saved {out_file}")
