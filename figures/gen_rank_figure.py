#!/usr/bin/env python3
"""
Generate SVD rank sweep figure for the appendix.

Plots Top-1 accuracy and reconstruction error vs. SVD rank on dual y-axes.
Reads outputs/rank_sweep/summaries/*.json; recon_err values are from
the measured values in paper Table tab:rank_sweep.

Output: figures/rank_sweep.pdf
Usage:  python figures/gen_rank_figure.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUMMARIES_DIR = Path("outputs/rank_sweep/summaries")
OUT_PATH = Path("figures/rank_sweep.pdf")

# Measured reconstruction error from paper (tab:rank_sweep)
RECON_ERRORS = {
    16: 0.907,
    32: 0.841,
    64: 0.734,
    96: 0.643,
    128: 0.562,
    192: 0.420,
    256: 0.294,
}


def load_rank_results():
    rows = []
    for rank, recon_err in sorted(RECON_ERRORS.items()):
        fname = f"rank_{rank}_seed42.json"
        path = SUMMARIES_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        with path.open() as f:
            d = json.load(f)
        rows.append({
            "rank": rank,
            "top1": d["top1"],
            "p50_ms": d["p50_ms"],
            "recon_err": recon_err,
        })
    return rows


def main():
    rows = load_rank_results()
    ranks = [r["rank"] for r in rows]
    top1s = [r["top1"] for r in rows]
    recons = [r["recon_err"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.8))

    color_acc = "#1565C0"
    color_rec = "#B71C1C"

    line1, = ax1.plot(ranks, top1s, color=color_acc, marker="o",
                      markersize=5, linewidth=1.8, label="Top-1 (%)")
    ax1.set_xlabel("SVD rank $r$", fontsize=10)
    ax1.set_ylabel("Top-1 accuracy (%)", color=color_acc, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=color_acc, labelsize=9)
    ax1.tick_params(axis="x", labelsize=9)
    ax1.set_xticks(ranks)
    y_lo = min(top1s) - 0.2
    y_hi = max(top1s) + 0.2
    ax1.set_ylim(y_lo, y_hi)

    ax2 = ax1.twinx()
    line2, = ax2.plot(ranks, recons, color=color_rec, marker="^",
                      markersize=5, linewidth=1.8, linestyle="--",
                      label="Recon. error")
    ax2.set_ylabel("Reconstruction error", color=color_rec, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=color_rec, labelsize=9)
    ax2.set_ylim(0, 1.0)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=9, loc="center right", framealpha=0.85)

    ax1.set_title("SVD rank vs. accuracy and reconstruction error\n"
                  "(DeiT-Small, Imagenette, $E{=}4$, seed 42)",
                  fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
