"""
Fulfillment simulators.

fulfillment_rules_cost  — differentiable PyTorch implementation of the
                          cost-based fulfillment rule (paper §3).
optimal_fulfillment_cost — exact LP baseline via scipy HiGHS.
"""

from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linprog

from graph import WarehouseGraph


# ---------------------------------------------------------------------------
# Differentiable fulfillment rules
# ---------------------------------------------------------------------------

def fulfillment_rules_cost(
    d: torch.Tensor,
    q: torch.Tensor,
    graph: WarehouseGraph,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute fulfillment cost under the cost-based fulfillment rule.

    Iterates over (i, j) pairs in ascending shipping-cost order.  For each pair,
    allocates min(available supply at i, remaining demand at j) units.
    All operations are differentiable so gradients flow back to q (and d).

    Design notes
    ────────────
    • No in-place tensor mutations on nodes that carry gradients.
      Accumulation uses list reassignment: shipped[i] = shipped[i] + a_ij,
      which creates a new graph node at each step.
    • The final (B, M, M) allocation matrix is assembled with torch.stack +
      view — both fully differentiable — avoiding scatter_ in-place ops.

    Args:
        d     : demand (M,) or (B, M)
        q     : inventory allocation (M,) or (B, M)
        graph : WarehouseGraph carrying cost_matrix, h, u, fulfillment_order

    Returns:
        a     : allocation matrix, shape (M, M) or (B, M, M)
        cost  : total cost,        shape ()     or (B,)
    """
    single = d.dim() == 1
    if single:
        d = d.unsqueeze(0)
        q = q.unsqueeze(0)

    B, M = d.shape
    order = graph.fulfillment_order          # List[(i, j)] sorted by c_ij
    pair_to_rank = graph._pair_to_rank       # (i, j) -> index in order

    # Running tallies — reassigned (not mutated) at each step so autograd works
    shipped: list = [d.new_zeros(B) for _ in range(M)]   # shipped[i]: total out of i
    received: list = [d.new_zeros(B) for _ in range(M)]  # received[j]: total into j

    a_by_rank: list = []  # a_ij values in fulfillment_order order

    for i, j in order:
        avail = (q[:, i] - shipped[i]).clamp(min=0.0)   # remaining supply at i
        rem   = (d[:, j] - received[j]).clamp(min=0.0)  # remaining demand at j
        a_ij  = torch.minimum(avail, rem)                # allocate as much as possible

        a_by_rank.append(a_ij)
        shipped[i]  = shipped[i]  + a_ij    # non-in-place: creates new tensor node
        received[j] = received[j] + a_ij

    # Assemble (B, M, M) in natural row-major order: (0,0),(0,1),...,(M-1,M-1)
    # Non-edges never appear in fulfillment_order so their a_ij stays zero.
    a_natural = [
        a_by_rank[pair_to_rank[(i, j)]] if (i, j) in pair_to_rank
        else d.new_zeros(B)
        for i in range(M)
        for j in range(M)
    ]
    a = torch.stack(a_natural, dim=1).view(B, M, M)  # (B, M, M)

    # Cost components
    h = graph.h.to(q.device)
    u = graph.u.to(q.device)
    # Use cost_matrix_safe (0 on non-edges) to avoid inf * 0 = nan
    C = graph.cost_matrix_safe.to(q.device)

    # a.sum(2)[b,i] = total shipped FROM i in sample b
    # a.sum(1)[b,j] = total received AT  j in sample b
    unsold = (q - a.sum(dim=2)).clamp(min=0.0)  # (B, M)
    unmet  = (d - a.sum(dim=1)).clamp(min=0.0)  # (B, M)

    cost = (h * unsold).sum(1) + (u * unmet).sum(1) + (C * a).sum((1, 2))  # (B,)

    if single:
        return a.squeeze(0), cost.squeeze(0)
    return a, cost


# ---------------------------------------------------------------------------
# Exact LP baseline (not differentiable — comparison only)
# ---------------------------------------------------------------------------

def optimal_fulfillment_cost(
    d: torch.Tensor,
    q: torch.Tensor,
    graph: WarehouseGraph,
) -> Tuple[float, np.ndarray]:
    """
    Solve the fulfillment LP exactly for a single demand/allocation pair.

    LP formulation over existing edges only (variables: a_e for e in edges, s ∈ R^M, e ∈ R^M):

        min   Σ_e c_e · a_e + h·s + u·e
        s.t.  Σ_{j:(i,j)∈E} a_ij + s_i  =  q_i   ∀i   (supply balance; s_i = unsold)
              Σ_{i:(i,j)∈E} a_ij + e_j  ≥  d_j   ∀j   (demand balance; e_j = unmet)
              a_e, s, e ≥ 0

    Only edges present in graph.edges appear as LP variables, so the LP
    correctly handles sparse graphs and is smaller than the dense M² formulation.

    Args:
        d     : demand       (M,) tensor  — detached from graph
        q     : allocation   (M,) tensor  — detached from graph
        graph : WarehouseGraph

    Returns:
        cost  : optimal total cost (float)
        a_opt : (M, M) numpy array of optimal shipment quantities (0 on non-edges)
    """
    M = graph.M
    edges = graph.edges                    # existing (i, j) pairs, sorted by cost
    E = len(edges)
    edge_idx = {e: k for k, e in enumerate(edges)}  # (i,j) -> variable index

    d_np = d.detach().float().numpy()
    q_np = q.detach().float().numpy()
    h_np = graph.h.float().numpy()
    u_np = graph.u.float().numpy()

    # Variable layout: [a_e for e in edges (E vars),  s_0..s_{M-1},  e_0..e_{M-1}]
    n_vars = E + 2 * M
    c_edge = np.array([graph.cost_matrix[i, j].item() for i, j in edges])
    obj = np.concatenate([c_edge, h_np, u_np])

    # ── Equality: Σ_{j:(i,j)∈E} a_ij + s_i = q_i  for all i ───────────────
    A_eq = np.zeros((M, n_vars))
    for k, (i, j) in enumerate(edges):
        A_eq[i, k] = 1.0            # a_{ij} contributes to supply of i
    for i in range(M):
        A_eq[i, E + i] = 1.0        # s_i
    b_eq = q_np

    # ── Inequality: -Σ_{i:(i,j)∈E} a_ij - e_j ≤ -d_j ──────────────────────
    A_ub = np.zeros((M, n_vars))
    for k, (i, j) in enumerate(edges):
        A_ub[j, k] = -1.0           # -a_{ij} contributes to demand of j
    for j in range(M):
        A_ub[j, E + M + j] = -1.0   # -e_j
    b_ub = -d_np

    bounds = [(0.0, None)] * n_vars

    result = linprog(
        obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )

    if not result.success:
        raise RuntimeError(f"LP solver failed: {result.message}")

    a_opt = np.zeros((M, M))
    for k, (i, j) in enumerate(edges):
        a_opt[i, j] = result.x[k]
    return float(result.fun), a_opt
