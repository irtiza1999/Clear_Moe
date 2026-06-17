#!/usr/bin/env python3
"""
Generate accuracy vs. latency scatter for L0-L10 layer policies (main paper Fig 3).
Replaces inline TikZ figure.

Reads outputs/ablation_layer/summaries/*.json.

Output: figures/tradeoff_scatter.pdf
Usage:  python figures/gen_tradeoff_scatter.py
"""

import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SUMMARIES_DIR = Path("outputs/ablation_layer/summaries")
OUT_PATH = Path("figures/tradeoff_scatter.pdf")

POLICY_LABELS = {
    "L0": "L0 Dense",
    "L1": "L1 Random",
    "L2": "L2 First-k",
    "L3": "L3 Last-k",
    "L4": "L4 Alternating",
    "L5": "L5 Sparsity",
    "L6": "L6 Clusterab.",
    "L7": "L7 Sensitivity",
    "L8": "L8 Sp+Cl",
    "L9": "L9 Cl-Se",
    "L10": "L10 Composite",
}


def load_data():
    results = {}
    for path in glob.glob(str(SUMMARIES_DIR / "*.json")):
        with open(path) as f:
            d = json.load(f)
        eid = d["experiment_id"]
        # Parse L-id
        parts = eid.split("_")
        lid = parts[0]  # e.g. L0, L1, L10
        if lid not in POLICY_LABELS:
            continue
        if lid not in results:
            results[lid] = {"top1_list": [], "p50_list": []}
        results[lid]["top1_list"].append(d["top1"])
        results[lid]["p50_list"].append(d["p50_ms"])

    # Average across seeds for each policy
    averaged = {}
    for lid, v in results.items():
        averaged[lid] = {
            "top1": float(np.mean(v["top1_list"])),
            "p50": float(np.mean(v["p50_list"])),
        }
    return averaged


def main():
    data = load_data()

    fig, ax = plt.subplots(figsize=(4.8, 3.5))

    dense_p50 = data["L0"]["p50"]
    dense_top1 = data["L0"]["top1"]

    # Dense baseline dashed line
    ax.axhline(dense_top1, color="black", linestyle="--", linewidth=1.2,
               label=f"Dense L0 ({dense_p50:.0f} ms)")

    # L1-L9 grey dots
    non_compose_x, non_compose_y = [], []
    for lid, v in data.items():
        if lid in ("L0", "L10"):
            continue
        non_compose_x.append(v["p50"])
        non_compose_y.append(v["top1"])
    ax.scatter(non_compose_x, non_compose_y,
               color="#888888", s=28, alpha=0.85,
               zorder=3, label="MoE L1–L9")

    # L10 composite — star
    ax.scatter(data["L10"]["p50"], data["L10"]["top1"],
               marker="*", color="black", s=160,
               zorder=5, label="Composite L10 (ours)")

    # Annotate a few key points
    for lid in ("L4", "L7", "L2"):
        v = data[lid]
        ax.annotate(lid, xy=(v["p50"], v["top1"]),
                    xytext=(3, 2), textcoords="offset points",
                    fontsize=7, color="#444444")
    ax.annotate("L10", xy=(data["L10"]["p50"], data["L10"]["top1"]),
                xytext=(4, -9), textcoords="offset points",
                fontsize=7.5, fontweight="bold")

    ax.set_xlabel("p50 Latency (ms)", fontsize=9)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=9)
    ax.set_title("Accuracy vs.\ latency — layer-selection policies", fontsize=9)
    ax.tick_params(labelsize=8)

    # Tight y-axis around the spread
    all_top1 = [v["top1"] for v in data.values()]
    ax.set_ylim(min(all_top1) - 0.10, max(all_top1) + 0.15)
    all_p50 = [v["p50"] for v in data.values() if v["p50"] > 60]
    ax.set_xlim(min(all_p50) - 3, max(all_p50) + 5)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
