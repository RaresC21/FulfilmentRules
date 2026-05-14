"""
Accuracy test: fulfillment rules vs. optimal LP.

For each network size M, samples N demand realizations with a fixed
inventory allocation q, then computes the cost ratio
    cost_rules / cost_optimal
for every sample and reports summary statistics.

Also verifies that the PyTorch graph is intact by checking that
d(cost_rules)/d(q) is non-zero, confirming end-to-end differentiability.
"""

import time
import json
import os
from pathlib import Path

import numpy as np
import torch

from graph import DemandSampler, WarehouseGraph
from fulfillment import fulfillment_rules_cost, optimal_fulfillment_cost


# ── helpers ─────────────────────────────────────────────────────────────────

def run_comparison(
    M: int,
    n_samples: int = 500,
    mean_demand: float = 10.0,
    std_demand: float = 3.0,
    edge_prob: float = 1.0,
    seed: int = 42,
) -> dict:
    graph   = WarehouseGraph(M, edge_prob=edge_prob, seed=seed)
    sampler = DemandSampler(graph, mean=mean_demand, std=std_demand, seed=seed)

    # Fixed allocation: slightly above mean (a reasonable heuristic)
    q = torch.full((M,), mean_demand * 1.1)

    d_batch = sampler.sample(n_samples)   # (N, M)
    q_batch = q.unsqueeze(0).expand(n_samples, -1)  # (N, M)

    # ── fulfillment rules (batched, PyTorch) ─────────────────────────────
    t0 = time.perf_counter()
    _, cost_rules = fulfillment_rules_cost(d_batch, q_batch, graph)
    t_rules = time.perf_counter() - t0
    cost_rules_np = cost_rules.detach().numpy()

    # ── optimal LP (per-sample, scipy) ──────────────────────────────────
    t0 = time.perf_counter()
    cost_opt_list = []
    for n in range(n_samples):
        c_opt, _ = optimal_fulfillment_cost(d_batch[n], q, graph)
        cost_opt_list.append(c_opt)
    t_opt = time.perf_counter() - t0
    cost_opt_np = np.array(cost_opt_list)

    # Guard against zero optimal cost (demand = 0 samples)
    safe_opt = np.where(cost_opt_np < 1e-6, 1e-6, cost_opt_np)
    ratio = cost_rules_np / safe_opt

    return dict(
        M=M,
        edge_prob=edge_prob,
        n_edges=graph.n_edges,
        n=n_samples,
        mean_rules=cost_rules_np.mean(),
        mean_opt=cost_opt_np.mean(),
        mean_ratio=ratio.mean(),
        max_ratio=ratio.max(),
        pct_within_1pct=(ratio <= 1.01).mean() * 100,
        pct_within_5pct=(ratio <= 1.05).mean() * 100,
        t_rules=t_rules,
        t_opt=t_opt,
    )


def check_differentiable(M: int = 4, edge_prob: float = 0.5, seed: int = 0) -> bool:
    """Returns True if d(cost)/d(q) is non-zero for at least one element."""
    graph   = WarehouseGraph(M, edge_prob=edge_prob, seed=seed)
    sampler = DemandSampler(graph, seed=seed)

    d = sampler.sample(8)                           # (8, M)
    q = torch.full((M,), 10.0, requires_grad=True)
    q_batch = q.unsqueeze(0).expand(8, -1)

    _, cost = fulfillment_rules_cost(d, q_batch, graph)
    cost.sum().backward()

    return q.grad is not None and q.grad.abs().sum().item() > 0


# ── main ─────────────────────────────────────────────────────────────────────

# Column widths are fixed here so header and rows always align.
_ROW = "  {:>5}  {:>6}  {:>12}  {:>10}  {:>11}  {:>10}  {:>6}  {:>6}  {:>9}  {:>7}"
_HEADER = _ROW.format(
    "prob", "edges",
    "Cost(rules)", "Cost(opt)",
    "Mean ratio", "Max ratio",
    "<=1%", "<=5%",
    "t_rules", "t_opt",
)
_SEP = "  " + "-" * (len(_HEADER) - 2)


def _print_row(r: dict) -> None:
    print(_ROW.format(
        f"{r['edge_prob']:.2f}",
        r['n_edges'],
        f"{r['mean_rules']:.3f}",
        f"{r['mean_opt']:.3f}",
        f"{r['mean_ratio']:.4f}",
        f"{r['max_ratio']:.4f}",
        f"{r['pct_within_1pct']:.1f}%",
        f"{r['pct_within_5pct']:.1f}%",
        f"{r['t_rules']:.3f}s",
        f"{r['t_opt']:.2f}s",
    ))


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    ok = check_differentiable()
    print(f"Differentiability check: {'PASS' if ok else 'FAIL'}")

    N_SAMPLES = 20
    all_results = []

    for M in np.arange(5, 100, 5):
        probs = np.arange(0.05, 1, 0.05)
        print(f"\nM = {M}  ({M**2} possible edges,  N = {N_SAMPLES} samples)")
        print(_HEADER)
        print(_SEP)
        for prob in probs:
            r = run_comparison(M, edge_prob=prob, n_samples=N_SAMPLES)
            all_results.append(r)
            _print_row(r)

    # Save results to JSON (convert numpy types to native Python types)
    results_file = results_dir / "test_results.json"
    json_results = []
    for r in all_results:
        json_results.append({
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in r.items()
        })
    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n[OK] Results saved to {results_file}")
    