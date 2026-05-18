"""
Stochastic inventory allocation: three solvers.

Given N demand scenarios, all three methods find a single q that minimises
expected cost subject to sum(q) <= budget, q >= 0.

solve_stochastic_lp  -- SAA stochastic LP, ground-truth optimal for the sample
solve_gd             -- gradient descent on fulfillment-rules expected cost
solve_fra            -- coordinate descent with per-coordinate bisection (FRA)

evaluate_lp_oracle   -- evaluate any q with per-scenario LP fulfillment (true cost)
evaluate_rules       -- evaluate any q with fulfillment-rules heuristic
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).parents[2]))

from graph import WarehouseGraph
from fulfillment import fulfillment_rules_cost, optimal_fulfillment_cost


# ---------------------------------------------------------------------------
# Shared evaluation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Internal helper (used by both evaluators and solvers)
# ---------------------------------------------------------------------------

def _rules_mean_cost(
    q: torch.Tensor,
    demands: torch.Tensor,
    graph: WarehouseGraph,
) -> torch.Tensor:
    """Mean fulfillment-rules cost over demand batch (differentiable in q)."""
    N = demands.shape[0]
    _, costs = fulfillment_rules_cost(demands, q.unsqueeze(0).expand(N, -1), graph)
    return costs.mean()


# ---------------------------------------------------------------------------
# Shared evaluation
# ---------------------------------------------------------------------------

def evaluate_lp_oracle(
    q: torch.Tensor,
    demands: torch.Tensor,
    graph: WarehouseGraph,
) -> float:
    """
    True expected cost for allocation q: each scenario is fulfilled optimally
    by the LP oracle, independent of how q was produced.
    """
    costs = [
        optimal_fulfillment_cost(demands[n], q, graph)[0]
        for n in range(demands.shape[0])
    ]
    return float(np.mean(costs))


def evaluate_rules(
    q: torch.Tensor,
    demands: torch.Tensor,
    graph: WarehouseGraph,
) -> float:
    """
    Expected cost for allocation q using the fulfillment-rules heuristic.
    Fast (no LP solve per scenario), but approximate.
    """
    with torch.no_grad():
        return _rules_mean_cost(q, demands, graph).item()


# ---------------------------------------------------------------------------
# Method 1: SAA stochastic LP (ground truth)
# ---------------------------------------------------------------------------

def solve_stochastic_lp(
    demands: torch.Tensor,
    graph: WarehouseGraph,
    budget: float,
) -> tuple:
    """
    Solve the sample-average-approximation stochastic LP exactly.

    Variable layout per scenario n  (offset = M + n*(E+2M)):
        a^n_k  k=0..E-1   shipment on edge k
        s^n_i  i=0..M-1   unsold at warehouse i
        e^n_j  j=0..M-1   unmet demand at j

    Constraints:
        supply balance (=):  sum_j a^n_{ij} + s^n_i  =  q_i      for all i,n
        demand balance (<=): -sum_i a^n_{ij} - e^n_j  <= -d^n_j  for all j,n
        budget (<=):          sum_i q_i                <= budget

    Returns (q_opt, lp_obj_value, solve_time_seconds).
    """
    N, M = demands.shape
    edges = graph.edges
    E = len(edges)
    n_per = E + 2 * M
    n_vars = M + N * n_per

    def a_idx(n, k): return M + n * n_per + k
    def s_idx(n, i): return M + n * n_per + E + i
    def e_idx(n, j): return M + n * n_per + E + M + j

    # Objective: (1/N) * sum_n [c*a^n + h*s^n + u*e^n]
    obj = np.zeros(n_vars)
    c_edges = np.array([graph.cost_matrix[i, j].item() for i, j in edges])
    h_np = graph.h.numpy()
    u_np = graph.u.numpy()
    for n in range(N):
        base = M + n * n_per
        obj[base          : base + E    ] = c_edges / N
        obj[base + E      : base + E + M] = h_np    / N
        obj[base + E + M  : base + n_per] = u_np    / N

    # Edge lookup tables
    edges_from = {i: [] for i in range(M)}
    edges_to   = {j: [] for j in range(M)}
    for k, (ii, jj) in enumerate(edges):
        edges_from[ii].append(k)
        edges_to[jj].append(k)

    # Supply balance equality
    n_eq = N * M
    A_eq = lil_matrix((n_eq, n_vars))
    b_eq = np.zeros(n_eq)
    for n in range(N):
        for i in range(M):
            row = n * M + i
            A_eq[row, i] = -1.0               # -q_i
            A_eq[row, s_idx(n, i)] = 1.0      # s^n_i
            for k in edges_from[i]:
                A_eq[row, a_idx(n, k)] = 1.0  # a^n_{ij}

    # Demand balance inequality + budget
    n_ub = N * M + 1
    A_ub = lil_matrix((n_ub, n_vars))
    b_ub = np.zeros(n_ub)
    d_np = demands.numpy()
    for n in range(N):
        for j in range(M):
            row = n * M + j
            b_ub[row] = -d_np[n, j]
            A_ub[row, e_idx(n, j)] = -1.0     # -e^n_j
            for k in edges_to[j]:
                A_ub[row, a_idx(n, k)] = -1.0 # -a^n_{ij}
    for i in range(M):
        A_ub[N * M, i] = 1.0
    b_ub[N * M] = budget

    bounds = [(0.0, None)] * n_vars

    t0 = time.perf_counter()
    result = linprog(
        obj,
        A_ub=A_ub.tocsr(), b_ub=b_ub,
        A_eq=A_eq.tocsr(), b_eq=b_eq,
        bounds=bounds, method="highs",
    )
    t_elapsed = time.perf_counter() - t0

    if not result.success:
        raise RuntimeError(f"Stochastic LP failed: {result.message}")

    q_opt = torch.tensor(result.x[:M], dtype=torch.float32)
    return q_opt, float(result.fun), t_elapsed


# ---------------------------------------------------------------------------
# Method 2: Gradient descent on fulfillment-rules cost
# ---------------------------------------------------------------------------

def solve_gd(
    demands: torch.Tensor,
    graph: WarehouseGraph,
    budget: float,
    n_steps: int = 500,
    lr: float = 0.05,
    patience: int = 20,
    min_rel_improvement: float = 1e-4,
) -> tuple:
    """
    Minimise E[Z_FR(d, q)] via Adam with q = budget * softmax(theta).

    The softmax reparametrisation automatically enforces sum(q) = budget and
    q_i > 0 throughout training.

    Early stopping: if best loss does not improve by at least `min_rel_improvement`
    (relative) over the last `patience` steps, training halts.

    Returns (q_opt, convergence_curve, solve_time_seconds).
    """
    M = demands.shape[1]
    theta = torch.zeros(M, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    best_q        = None
    best_val      = float("inf")
    steps_no_imp  = 0
    curve: list[float] = []

    t0 = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad()
        q    = budget * torch.softmax(theta, dim=0)
        loss = _rules_mean_cost(q, demands, graph)
        loss.backward()
        optimizer.step()

        val = loss.item()
        curve.append(val)

        if val < best_val:
            rel_improvement = (best_val - val) / max(abs(best_val), 1e-9)
            best_val = val
            best_q   = (budget * torch.softmax(theta.detach(), dim=0)).clone()
            if rel_improvement > min_rel_improvement:
                steps_no_imp = 0
            else:
                steps_no_imp += 1
        else:
            steps_no_imp += 1
        if steps_no_imp >= patience:
            break

    return best_q, curve, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Method 3: Fulfillment Rules Algorithm (coordinate descent + bisection)
# ---------------------------------------------------------------------------

def solve_fra(
    demands: torch.Tensor,
    graph: WarehouseGraph,
    budget: float,
    max_outer: int = 30,
    bisect_iters: int = 40,
    tol: float = 1e-4,
) -> tuple:
    """
    Coordinate descent with per-coordinate bisection search (FRA, §3.3 of paper).

    For warehouse i (all others held fixed), the derivative
        L_i(q_i) = d/dq_i  E[Z_FR(d, q)]
    is monotone increasing (objective is convex in q).  We bisect q_i in
    [0, budget - sum(q_{-i})] until L_i = 0.  Gradients are computed via
    PyTorch autograd on fulfillment_rules_cost.

    Returns (q_opt, solve_time_seconds).
    """
    M = demands.shape[1]
    q_vals = [budget / M] * M  # plain Python floats, updated in-place

    def grad_coord(i: int) -> float:
        """d/dq_i E[Z_FR] at the current q_vals."""
        q_t = torch.tensor(q_vals, dtype=torch.float32, requires_grad=True)
        _rules_mean_cost(q_t, demands, graph).backward()
        return q_t.grad[i].item()

    t0 = time.perf_counter()
    for _ in range(max_outer):
        q_old = q_vals.copy()

        for i in range(M):
            remaining = budget - sum(q_vals) + q_vals[i]
            lo = 0.0
            hi = max(remaining, 0.0)

            # Gradient at lower bound
            q_vals[i] = lo
            if grad_coord(i) >= 0:
                continue  # optimal for this coord is 0

            # Gradient at upper bound
            q_vals[i] = hi
            if grad_coord(i) <= 0:
                continue  # optimal for this coord is the full remaining budget

            # Root is in (lo, hi) — bisect
            for _ in range(bisect_iters):
                mid = (lo + hi) / 2.0
                q_vals[i] = mid
                g = grad_coord(i)
                if g < 0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-6:
                    break
            q_vals[i] = (lo + hi) / 2.0

        if max(abs(q_vals[i] - q_old[i]) for i in range(M)) < tol:
            break

    return torch.tensor(q_vals, dtype=torch.float32), time.perf_counter() - t0
