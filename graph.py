import torch
from typing import List, Tuple


class WarehouseGraph:
    """
    2D warehouse network where shipping cost between nodes is Euclidean distance.

    Off-diagonal costs are scaled to [cost_lo, cost_hi].  The diagonal (local
    self-fulfillment) is set to self_cost < cost_lo, so the ordering
    h < self_cost < c_ij < u is always satisfied — matching the paper's assumption.

    Sparsity is controlled by edge_prob: each off-diagonal edge (i, j) is
    included independently with probability edge_prob.  The diagonal is always
    present (a warehouse can always fulfill its own demand).  Missing edges are
    represented as inf in cost_matrix and are excluded from fulfillment_order.

    Attributes:
        M                 : number of warehouses
        locations         : (M, 2) 2-D coordinates
        adjacency         : (M, M) bool — True where an edge exists
        cost_matrix       : (M, M) shipping costs; inf where no edge
        cost_matrix_safe  : (M, M) same but 0 on non-edges (safe for a·C products)
        h                 : (M,)   holding costs
        u                 : (M,)   lost-sale costs
        fulfillment_order : list of (i, j) pairs (existing edges) sorted by c_ij
        edges             : list of (i, j) pairs that exist (same set, unsorted)
    """

    def __init__(
        self,
        n_warehouses: int,
        h: float = 1.0,
        u: float = 10.0,
        cost_lo: float = 2.0,   # minimum cross-fulfillment cost  (> h)
        cost_hi: float = 8.0,   # maximum cross-fulfillment cost  (< u)
        self_cost: float = 1.2, # local self-fulfillment cost (< cost_lo)
        edge_prob: float = 1.0, # probability each off-diagonal edge exists; 1.0 = fully connected
        seed: int = 42,
    ):
        assert n_warehouses >= 2, "Need at least 2 warehouses"
        assert h < self_cost < cost_lo < cost_hi < u, (
            "Required: h < self_cost < cost_lo < cost_hi < u"
        )
        assert 0.0 < edge_prob <= 1.0, "edge_prob must be in (0, 1]"

        self.M = n_warehouses
        self.h_val = h
        self.u_val = u
        self.edge_prob = edge_prob

        rng = torch.Generator()
        rng.manual_seed(seed)

        # 2-D locations in [0, 1]^2
        self.locations = torch.rand(n_warehouses, 2, generator=rng)  # (M, 2)

        # Euclidean distances (M, M), diagonal = 0
        diff = self.locations.unsqueeze(0) - self.locations.unsqueeze(1)  # (M, M, 2)
        dist = diff.norm(dim=-1)  # (M, M)

        # Scale off-diagonal distances linearly into [cost_lo, cost_hi]
        off_diag_mask = ~torch.eye(n_warehouses, dtype=torch.bool)
        d_off = dist[off_diag_mask]
        d_min, d_max = d_off.min(), d_off.max()
        if d_max > d_min:
            scaled_off = (d_off - d_min) / (d_max - d_min) * (cost_hi - cost_lo) + cost_lo
        else:
            scaled_off = torch.full_like(d_off, (cost_lo + cost_hi) / 2.0)

        cost_matrix = dist.clone()
        cost_matrix[off_diag_mask] = scaled_off
        cost_matrix.fill_diagonal_(self_cost)

        # Adjacency: diagonal always present; off-diagonal sampled with edge_prob
        adjacency = torch.eye(n_warehouses, dtype=torch.bool)
        if edge_prob < 1.0:
            rand_edges = torch.rand(n_warehouses, n_warehouses, generator=rng)
            adjacency |= (rand_edges < edge_prob) & off_diag_mask
        else:
            adjacency |= off_diag_mask

        # Non-edges get inf cost so they sort last and are trivially excluded
        cost_matrix[~adjacency] = float("inf")

        self.adjacency: torch.Tensor = adjacency                        # (M, M) bool
        self.cost_matrix: torch.Tensor = cost_matrix                    # (M, M), inf on non-edges
        self.cost_matrix_safe: torch.Tensor = cost_matrix.nan_to_num(  # (M, M), 0 on non-edges
            posinf=0.0
        )
        self.h: torch.Tensor = torch.full((n_warehouses,), h)           # (M,)
        self.u: torch.Tensor = torch.full((n_warehouses,), u)           # (M,)

        # Fulfillment order: existing edges only, sorted by ascending cost
        flat = [
            (cost_matrix[i, j].item(), i, j)
            for i in range(n_warehouses)
            for j in range(n_warehouses)
            if adjacency[i, j].item()
        ]
        flat.sort()
        self.fulfillment_order: List[Tuple[int, int]] = [(i, j) for _, i, j in flat]
        self.edges: List[Tuple[int, int]] = [(i, j) for _, i, j in flat]  # same set

        # Reverse map: (i, j) -> position in fulfillment_order
        self._pair_to_rank = {(i, j): k for k, (i, j) in enumerate(self.fulfillment_order)}

    @property
    def n_edges(self) -> int:
        return len(self.fulfillment_order)

    def __repr__(self) -> str:
        density = self.n_edges / (self.M * self.M)
        return (
            f"WarehouseGraph(M={self.M}, edges={self.n_edges}/{self.M**2} "
            f"({density:.0%}), h={self.h_val}, u={self.u_val}, "
            f"edge_prob={self.edge_prob})"
        )


class DemandSampler:
    """
    Samples independent truncated-normal demand at each warehouse.

    Args:
        graph  : WarehouseGraph (used only for M)
        mean   : per-warehouse demand mean (scalar, broadcast to all M)
        std    : per-warehouse demand std  (scalar, broadcast to all M)
        seed   : RNG seed for reproducibility
    """

    def __init__(
        self,
        graph: WarehouseGraph,
        mean: float = 10.0,
        std: float = 3.0,
        seed: int = 0,
    ):
        self.M = graph.M
        self.mean = torch.full((graph.M,), mean)
        self.std = torch.full((graph.M,), std)
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def sample(self, n: int = 1) -> torch.Tensor:
        """
        Draw n demand realizations.

        Returns:
            (n, M) tensor if n > 1, else (M,) tensor.
        """
        noise = torch.randn(n, self.M, generator=self._rng)
        d = (self.mean + self.std * noise).clamp(min=0.0)
        return d.squeeze(0) if n == 1 else d
