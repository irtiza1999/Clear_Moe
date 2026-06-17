#!/usr/bin/env python3
"""
Generate router-accuracy vs downstream delta scatter (appendix).

Data sources:
  - outputs/multi_model/combined_results.json  (D5/D6 per backbone)
  - outputs/router_comparison/summaries/*.json  (linear/mlp/adaptive on DeiT-S)

Thesis: routing accuracy (0.0 → 0.98) has NO predictive power over
downstream accuracy delta (r ≈ 0). Shared SVD basis is sole determiner.

Output: figures/router_scatter.pdf
Usage:  python figures/gen_router_scatter.py
"""

import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MULTI_MODEL_JSON = Path("outputs/multi_model/combined_results.json")
ROUTER_DIR = Path("outputs/router_comparison/summaries")
OUT_PATH = Path("figures/router_scatter.pdf")

BACKBONE_COLORS = {
    "DeiT-T/16": "#1565C0",
    "DeiT-S/16": "#E53935",
    "ViT-S/16":  "#2E7D32",
    "DeiT-B/16": "#F57F17",
    "ViT-B/16":  "#6A1B9A",
}


def load_multi_model():
    with open(MULTI_MODEL_JSON) as f:
        data = json.load(f)

    points = []
    for entry in data:
        backbone = entry["backbone"]
        runs = entry["runs"]

        dense_top1 = runs["D0"]["top1"]

        for cfg in ("D5", "D6"):
            if cfg not in runs:
                continue
            run = runs[cfg]
            router_acc = run.get("router_acc")
            if router_acc is None:
                continue
            delta_pp = run["top1"] - dense_top1
            points.append({
                "router_acc": router_acc,
                "delta_pp": delta_pp,
                "backbone": backbone,
                "cfg": cfg,
                "label": f"{backbone} {cfg}",
            })
    return points


def load_router_comparison():
    dense_top1 = 86.72611464968153  # DeiT-S D0 from combined_results

    router_names = {
        "router_linear_seed42": "Linear",
        "router_mlp_seed42": "MLP",
        "router_adaptive_seed42": "Adaptive",
    }

    points = []
    for path in glob.glob(str(ROUTER_DIR / "*.json")):
        with open(path) as f:
            d = json.load(f)
        stem = Path(path).stem
        if stem not in router_names:
            continue
        rname = router_names[stem]
        points.append({
            "router_acc": d["router_accuracy"],
            "delta_pp": d["top1"] - dense_top1,
            "backbone": "DeiT-S/16",
            "cfg": rname,
            "label": f"DeiT-S {rname}",
        })
    return points


def pearson_r(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    if xs.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def main():
    mm_pts = load_multi_model()
    rc_pts = load_router_comparison()
    all_pts = mm_pts + rc_pts

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    # --- Plot multi-model points ---
    plotted_backbones = set()
    for p in mm_pts:
        bk = p["backbone"]
        color = BACKBONE_COLORS.get(bk, "#888888")
        marker = "o" if p["cfg"] == "D5" else "^"
        label = bk if bk not in plotted_backbones else None
        plotted_backbones.add(bk)
        ax.scatter(p["router_acc"], p["delta_pp"],
                   color=color, marker=marker, s=55, zorder=4,
                   label=label, linewidths=0.4, edgecolors="white")

    # --- Plot router-comparison points (DeiT-S architecture variants) ---
    for p in rc_pts:
        ax.scatter(p["router_acc"], p["delta_pp"],
                   color=BACKBONE_COLORS["DeiT-S/16"], marker="s", s=55,
                   zorder=5, linewidths=0.4, edgecolors="black")
        ax.annotate(p["cfg"], xy=(p["router_acc"], p["delta_pp"]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=6.5, color="#555555")

    # --- Zero delta reference ---
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)

    # --- Linear trend (OLS) ---
    xs = [p["router_acc"] for p in all_pts]
    ys = [p["delta_pp"] for p in all_pts]
    m, b = np.polyfit(xs, ys, 1)
    x_fit = np.linspace(min(xs) - 0.02, max(xs) + 0.02, 100)
    ax.plot(x_fit, m * x_fit + b, color="#999999", linestyle=":",
            linewidth=1.2, label=f"OLS fit (r={pearson_r(xs, ys):.2f})")

    # --- Legend: backbone shapes ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
               markersize=6, label="D5 (random)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#555",
               markersize=6, label="D6 (learned linear)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#555",
               markeredgecolor="black", markersize=6, label="DeiT-S arch variant"),
    ]
    for bk, color in BACKBONE_COLORS.items():
        if any(p["backbone"] == bk for p in mm_pts):
            legend_elements.append(
                Line2D([0], [0], marker="D", color="w",
                       markerfacecolor=color, markersize=6, label=bk)
            )

    r_val = pearson_r(xs, ys)
    ax.set_xlabel("Router Accuracy (vs. k-means labels)", fontsize=9)
    ax.set_ylabel(r"$\Delta$ Top-1 vs. Dense (pp)", fontsize=9)
    ax.set_title(f"Routing accuracy vs. downstream accuracy (r={r_val:.2f})",
                 fontsize=9)
    ax.tick_params(labelsize=8)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.2f}"))
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.legend(handles=legend_elements, fontsize=6.5, loc="lower left",
              framealpha=0.85, ncol=2)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")
    print(f"Pearson r = {r_val:.4f} (flat = thesis confirmed)")


if __name__ == "__main__":
    main()
