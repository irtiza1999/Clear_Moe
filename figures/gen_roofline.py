#!/usr/bin/env python3
"""
Generate roofline figure for main paper (replaces inline TikZ).

GTX 960: peak FP32 2400 GFLOP/s, BW 112 GB/s, ridge 21.4 FLOPs/B.

Output: figures/roofline.pdf
Usage:  python figures/gen_roofline.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_PATH = Path("figures/roofline.pdf")

PEAK_COMPUTE = 2400   # GFLOP/s
BW = 112              # GB/s
RIDGE = PEAK_COMPUTE / BW  # 21.43 FLOPs/B

POINTS = [
    dict(ai=74.3,  label="Dense FFN",    color="#1565C0", marker="o",  mem_bound=False),
    dict(ai=22.7,  label="Expert GEMM",  color="#1E88E5", marker="^",  mem_bound=False),
    dict(ai=1.9,   label="Router gate",  color="#E53935", marker="D",  mem_bound=True),
    dict(ai=1.0,   label="Token sort",   color="#FB8C00", marker="s",  mem_bound=True),
]


def roofline_perf(ai):
    return min(BW * ai, PEAK_COMPUTE)


def main():
    fig, ax = plt.subplots(figsize=(4.8, 3.5))

    # Draw roofline envelope
    ai_vals = np.logspace(np.log10(0.05), np.log10(200), 400)
    perf_vals = np.minimum(BW * ai_vals, PEAK_COMPUTE)
    ax.loglog(ai_vals, perf_vals, "k--", linewidth=1.6, label="Roofline (GTX 960)")

    # Ridge point marker
    ax.axvline(RIDGE, color="gray", linewidth=0.8, linestyle=":")
    ax.text(RIDGE * 1.08, 80, f"Ridge\n{RIDGE:.1f} F/B",
            fontsize=7.5, color="gray", va="center")

    # Per-point label offsets (points, direction) — avoids ceiling crowding
    LABEL_CONFIG = {
        "Dense FFN":   dict(xytext=(10,  6),  ha="left",  va="bottom"),
        "Expert GEMM": dict(xytext=(10, -14), ha="left",  va="top"),
        "Router gate": dict(xytext=(10,  5),  ha="left",  va="center"),
        "Token sort":  dict(xytext=(10, -12), ha="left",  va="top"),
    }

    for p in POINTS:
        perf = roofline_perf(p["ai"])
        ax.scatter(p["ai"], perf, marker=p["marker"], color=p["color"],
                   s=70, zorder=5, linewidths=0.5, edgecolors="white")
        cfg = LABEL_CONFIG[p["label"]]
        ax.annotate(
            p["label"],
            xy=(p["ai"], perf),
            xytext=cfg["xytext"],
            textcoords="offset points",
            fontsize=7.5,
            color=p["color"],
            ha=cfg["ha"],
            va=cfg["va"],
        )

    ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)", fontsize=9)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=9)
    ax.set_title("Roofline analysis — GTX 960", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0.05, 200)
    ax.set_ylim(0.5, 8000)
    ax.grid(True, which="both", linestyle="--", alpha=0.25)

    # Move region labels away from point cluster
    ax.text(0.4, 800, "Memory-\nbound", fontsize=7.5, color="#B71C1C",
            ha="left", style="italic")
    ax.text(35, 120, "Compute-bound", fontsize=7.5, color="#0D47A1",
            ha="left", style="italic")

    # Legend entries (manual, to match colors)
    handles = [
        mpatches.Patch(color="#1565C0", label="Compute-bound (blue)"),
        mpatches.Patch(color="#E53935", label="Memory-bound (red/orange)"),
        plt.Line2D([0], [0], color="k", linestyle="--", label="Roofline"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
