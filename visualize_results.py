"""
Visualization script for fulfillment rules test results.

Creates two matplotlib plots:
1. Fix edge_prob = 0.25, vary M, show mean_ratio and max_ratio
2. Fix M = 32, vary edge_prob, show mean_ratio and max_ratio
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_results():
    """Load test results from JSON file."""
    results_file = Path("results") / "test_results.json"
    with open(results_file, "r") as f:
        return json.load(f)


def plot_fix_prob_vary_m(results, target_prob):
    """
    Plot 1: Fix prob, vary M
    X-axis: M (network size)
    Y-axis: mean_ratio and max_ratio
    """

    # Filter results for prob 
    filtered = [r for r in results if abs(r["edge_prob"] - target_prob) < 0.001]

    if not filtered:
        print(f"Warning: No results found for edge_prob = {target_prob}")
        return None

    # Sort by M
    filtered.sort(key=lambda x: x["M"])

    m_values = [r["M"] for r in filtered]
    mean_ratios = [r["mean_ratio"] for r in filtered]
    max_ratios = [r["max_ratio"] for r in filtered]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(m_values, mean_ratios, marker='o', linewidth=2, label="Mean Ratio", color="#1f77b4")
    ax.plot(m_values, max_ratios, marker='s', linewidth=2, label="Max Ratio", color="#ff7f0e")

    ax.set_xlabel("Network Size (M)", fontsize=12)
    ax.set_ylabel("Cost Ratio (Rules / Optimal)", fontsize=12)
    ax.set_title(f"Fulfillment Rules Accuracy: Fixed edge_prob = {target_prob}, Varying M", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output_file = Path("results") / "plot_1_fix_prob_vary_m.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {output_file}")

    return fig


def plot_fix_m_vary_prob(results, target_m = 50):
    """
    Plot 2: Fix M, vary edge_prob
    X-axis: edge_prob
    Y-axis: mean_ratio and max_ratio
    """

    # Filter results for M
    filtered = [r for r in results if (abs(r["M"] - target_m) < 0.001)]

    if not filtered:
        print(f"Warning: No results found for M = {target_m}")
        return None

    # Sort by edge_prob (descending for natural ordering)
    filtered.sort(key=lambda x: x["edge_prob"], reverse=True)

    prob_values = [r["edge_prob"] for r in filtered]
    mean_ratios = [r["mean_ratio"] for r in filtered]
    max_ratios = [r["max_ratio"] for r in filtered]

    print(len(prob_values), len(mean_ratios), len(max_ratios))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prob_values, mean_ratios, marker='o', linewidth=2, label="Mean Ratio", color="#2ca02c")
    ax.plot(prob_values, max_ratios, marker='s', linewidth=2, label="Max Ratio", color="#d62728")

    ax.set_xlabel("Edge Probability", fontsize=12)
    ax.set_ylabel("Cost Ratio (Rules / Optimal)", fontsize=12)
    ax.set_title(f"Fulfillment Rules Accuracy: Fixed M = {target_m}, Varying edge_prob", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output_file = Path("results") / "plot_2_fix_m_vary_prob.png"
    print(output_file) 
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {output_file}")

    return fig


if __name__ == "__main__":
    results = load_results()
    print(f"Loaded {len(results)} test results")

    M = 95
    prob = 0.2 

    print("\nGenerating plots...")
    fig1 = plot_fix_prob_vary_m(results, prob)
    fig2 = plot_fix_m_vary_prob(results, M)

    print("\n[OK] Visualization complete!")
    print("\nOutput files:")
    print("  - results/plot_1_fix_prob_vary_m.png")
    print("  - results/plot_2_fix_m_vary_prob.png")
