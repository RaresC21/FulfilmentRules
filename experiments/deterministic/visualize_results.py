"""
Visualization script for fulfillment rules test results.

Creates two matplotlib plots:
1. Fix edge_prob = 0.25, vary M, show mean_ratio and max_ratio
2. Fix M = 32, vary edge_prob, show mean_ratio and max_ratio
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"


def load_results():
    results_file = RESULTS_DIR / "test_results.json"
    with open(results_file, "r") as f:
        return json.load(f)


def plot_fix_prob_vary_m(results, target_prob):
    filtered = [r for r in results if abs(r["edge_prob"] - target_prob) < 0.001]

    if not filtered:
        print(f"Warning: No results found for edge_prob = {target_prob}")
        return None

    filtered.sort(key=lambda x: x["M"])

    m_values   = [r["M"]          for r in filtered]
    mean_ratios = [r["mean_ratio"] for r in filtered]
    max_ratios  = [r["max_ratio"]  for r in filtered]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(m_values, mean_ratios, marker='o', linewidth=2, label="Mean Ratio", color="#1f77b4")
    ax.plot(m_values, max_ratios,  marker='s', linewidth=2, label="Max Ratio",  color="#ff7f0e")

    ax.set_xlabel("Network Size (M)", fontsize=12)
    ax.set_ylabel("Cost Ratio (Rules / Optimal)", fontsize=12)
    ax.set_title(f"Fulfillment Rules Accuracy: Fixed edge_prob = {target_prob}, Varying M", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output_file = RESULTS_DIR / "plot_1_fix_prob_vary_m.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {output_file}")
    return fig


def plot_fix_m_vary_prob(results, target_m=50):
    filtered = [r for r in results if abs(r["M"] - target_m) < 0.001]

    if not filtered:
        print(f"Warning: No results found for M = {target_m}")
        return None

    filtered.sort(key=lambda x: x["edge_prob"], reverse=True)

    prob_values = [r["edge_prob"]  for r in filtered]
    mean_ratios = [r["mean_ratio"] for r in filtered]
    max_ratios  = [r["max_ratio"]  for r in filtered]

    print(len(prob_values), len(mean_ratios), len(max_ratios))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prob_values, mean_ratios, marker='o', linewidth=2, label="Mean Ratio", color="#2ca02c")
    ax.plot(prob_values, max_ratios,  marker='s', linewidth=2, label="Max Ratio",  color="#d62728")

    ax.set_xlabel("Edge Probability", fontsize=12)
    ax.set_ylabel("Cost Ratio (Rules / Optimal)", fontsize=12)
    ax.set_title(f"Fulfillment Rules Accuracy: Fixed M = {target_m}, Varying edge_prob", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output_file = RESULTS_DIR / "plot_2_fix_m_vary_prob.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {output_file}")
    return fig


if __name__ == "__main__":
    results = load_results()
    print(f"Loaded {len(results)} test results")

    M    = 95
    prob = 0.95

    print("\nGenerating plots...")
    plot_fix_prob_vary_m(results, prob)
    plot_fix_m_vary_prob(results, M)

    print("\n[OK] Visualization complete!")
    print(f"  Output directory: {RESULTS_DIR}")
