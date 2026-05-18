"""
Plot GD convergence curves from gd_convergence.json.

Produces two figures:
  1. Fix M, vary N_train  — one subplot per M value
  2. Fix N_train, vary M  — one subplot per N_train value

Each curve is normalized cost (loss / loss[0]) vs. cumulative wall-clock time,
so all curves start at 1.0 and are directly comparable across problem sizes.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
data = json.loads((RESULTS_DIR / "gd_convergence.json").read_text())

M_VALUES      = sorted(set(c["M"]       for c in data))
N_TRAIN_VALUES = sorted(set(c["N_train"] for c in data))

COLORS = plt.cm.tab10.colors


# ── helper ───────────────────────────────────────────────────────────────────

def _get(M, N):
    for c in data:
        if c["M"] == M and c["N_train"] == N:
            return c
    return None


def _normalize(losses):
    """Divide by first step loss so curve starts at 1.0."""
    l0 = losses[0] if losses[0] != 0 else 1.0
    return [v / l0 for v in losses]


# ── Figure 1: fix M, vary N_train ────────────────────────────────────────────

ncols = len(M_VALUES)
fig1, axes1 = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=False)
if ncols == 1:
    axes1 = [axes1]

for ax, M in zip(axes1, M_VALUES):
    for color, N in zip(COLORS, N_TRAIN_VALUES):
        cfg = _get(M, N)
        if cfg is None:
            continue
        ax.plot(cfg["gd_times"], _normalize(cfg["gd_losses"]),
                color=color, label=f"N={N}", linewidth=1.5)

    ax.set_title(f"M = {M}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized cost (loss / loss₀)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig1.suptitle("GD convergence: vary N_train (fix M)", fontsize=13)
fig1.tight_layout()
p1 = RESULTS_DIR / "gd_convergence_vary_N.png"
fig1.savefig(p1, dpi=150)
print(f"Saved {p1}")


# ── Figure 2: fix N_train, vary M ────────────────────────────────────────────

ncols2 = len(N_TRAIN_VALUES)
fig2, axes2 = plt.subplots(1, ncols2, figsize=(5 * ncols2, 4), sharey=False)
if ncols2 == 1:
    axes2 = [axes2]

for ax, N in zip(axes2, N_TRAIN_VALUES):
    for color, M in zip(COLORS, M_VALUES):
        cfg = _get(M, N)
        if cfg is None:
            continue
        ax.plot(cfg["gd_times"], _normalize(cfg["gd_losses"]),
                color=color, label=f"M={M}", linewidth=1.5)

    ax.set_title(f"N_train = {N}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized cost (loss / loss₀)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig2.suptitle("GD convergence: vary M (fix N_train)", fontsize=13)
fig2.tight_layout()
p2 = RESULTS_DIR / "gd_convergence_vary_M.png"
fig2.savefig(p2, dpi=150)
print(f"Saved {p2}")

plt.show()
