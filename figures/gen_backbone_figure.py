#!/usr/bin/env python3
"""
Generate multi-backbone accuracy retention figure.

Shows accuracy delta (pp) vs. backbone for both random (D5) and
learned linear (D6) routing, using outputs/multi_model/combined_results.json.

Output: figures/backbone_comparison.pdf
Usage:  python figures/gen_backbone_figure.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COMBINED = Path("outputs/multi_model/combined_results.json")
OUT_PATH = Path("figures/backbone_comparison.pdf")

# Parameters in millions (for x-axis label or secondary info)
BACKBONE_PARAMS = {
    "DeiT-T/16": 5.7,
    "DeiT-S/16": 22.1,
    "ViT-S/16": 22.1,
    "DeiT-B/16": 86.6,
    "ViT-B/16": 86.6,
}


def main():
    with COMBINED.open() as f:
        data = json.load(f)

    backbones = [d["backbone"] for d in data]
    d5_deltas = [d["runs"]["D5"]["delta_pp"] for d in data]
    d6_deltas = [d["runs"]["D6"]["delta_pp"] for d in data]
    dense_top1 = [d["runs"]["D0"]["top1"] for d in data]

    x = np.arange(len(backbones))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    bars1 = ax.bar(x - width / 2, d5_deltas, width, label="D5 (random routing)",
                   color="#90CAF9", edgecolor="#1565C0", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, d6_deltas, width, label="D6 (learned linear router)",
                   color="#EF9A9A", edgecolor="#B71C1C", linewidth=0.8)

    # Annotate dense Top-1 above each group
    for i, (bb, t1) in enumerate(zip(backbones, dense_top1)):
        ax.text(i, max(d5_deltas[i], d6_deltas[i]) + 0.015,
                f"dense={t1:.1f}%", ha="center", va="bottom",
                fontsize=7, color="#333333")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)

    # X-tick labels with param count
    tick_labels = [f"{bb}\n({BACKBONE_PARAMS[bb]:.1f}M)" for bb in backbones]
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_ylabel(r"$\Delta$ Top-1 (pp vs. dense)", fontsize=10)
    ax.set_title("Cross-backbone accuracy retention\n"
                 r"(Imagenette, $E{=}4$, $N{=}200$, seed 42)",
                 fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=9, framealpha=0.85, loc="lower right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    # Keep y-axis symmetric around 0 with small padding
    ylim = max(abs(min(d5_deltas + d6_deltas)),
               abs(max(d5_deltas + d6_deltas))) + 0.15
    ax.set_ylim(-ylim, ylim)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
