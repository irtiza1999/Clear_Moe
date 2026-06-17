#!/usr/bin/env python3
"""
Generate dispatch throughput vs. imbalance figure for the appendix.

Reads outputs/dispatch_benchmark/summaries/*.json and plots
throughput (K tok/s) vs. imbalance fraction for each CUDA backend.

Output: figures/dispatch_curve.pdf
Usage:  python figures/gen_dispatch_curve.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUMMARIES_DIR = Path("outputs/dispatch_benchmark/summaries")
OUT_PATH = Path("figures/dispatch_curve.pdf")

BACKENDS = ["naive", "grouped", "cublas"]
IMBALANCE_LEVELS = [0, 40, 60, 80]
BACKEND_LABELS = {
    "naive": "Naive",
    "grouped": "Grouped",
    "cublas": "cuBLAS",
}
BACKEND_COLORS = {
    "naive": "#888888",
    "grouped": "#2196F3",
    "cublas": "#E53935",
}
BACKEND_MARKERS = {
    "naive": "s",
    "grouped": "o",
    "cublas": "^",
}


def load_result(backend, imb):
    fname = f"dispatch_{backend}_B8_imb{imb}.json"
    path = SUMMARIES_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with path.open() as f:
        return json.load(f)


def main():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for backend in BACKENDS:
        toks = []
        for imb in IMBALANCE_LEVELS:
            d = load_result(backend, imb)
            toks.append(d["tokens_per_second"] / 1000)  # K tok/s

        ax.plot(
            IMBALANCE_LEVELS,
            toks,
            marker=BACKEND_MARKERS[backend],
            color=BACKEND_COLORS[backend],
            linewidth=1.8,
            markersize=6,
            label=BACKEND_LABELS[backend],
        )

    ax.set_xlabel("Token-load imbalance (%)", fontsize=10)
    ax.set_ylabel("Throughput (K tok/s)", fontsize=10)
    ax.set_title("Dispatch throughput vs. load imbalance\n"
                 r"($B{=}8,\;T{=}1568,\;E{=}4$, GTX 960)",
                 fontsize=10)
    ax.set_xticks(IMBALANCE_LEVELS)
    ax.set_xticklabels([f"{x}%" for x in IMBALANCE_LEVELS], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
