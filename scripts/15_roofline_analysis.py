#!/usr/bin/env python3
"""
Script 15 — Roofline Analysis & Complexity Plots

Builds the roofline model for the GTX 960 (or any CUDA device) and overlays
the arithmetic intensity of each MoE operation:
    - Dense FFN forward pass
    - Expert GEMM (per expert)
    - Router linear
    - Token sort (argsort)
    - AllReduce (DP-2 simulation)
    - All-to-All (EP-2 simulation)

Also generates:
    - Theoretical vs measured scaling curves
    - Complexity table in CSV format
    - All figures as PNG (matplotlib)

GTX 960 hardware limits:
    Peak FP32:   ~2.4 TFLOP/s
    Mem BW:      ~112 GB/s
    Ridge point: ~21.4 FLOPs/byte

Usage:
    python scripts/15_roofline_analysis.py
    python scripts/15_roofline_analysis.py --results_dir outputs/benchmark_parallelism/xxx
    python scripts/15_roofline_analysis.py --device_flops 2.4e12 --device_bw 112e9
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Roofline analysis and complexity plots")
    p.add_argument("--results_dir", type=str, default=None,
                   help="Optional: directory with benchmark JSON results to overlay")
    p.add_argument("--device_flops", type=float, default=2.4e12,
                   help="Peak device FP32 FLOP/s (default: GTX 960 ~2.4 TFLOP/s)")
    p.add_argument("--device_bw", type=float, default=112e9,
                   help="Peak memory bandwidth bytes/s (default: GTX 960 ~112 GB/s)")
    p.add_argument("--hidden_dim", type=int, default=384,
                   help="Hidden dimension (DeiT-S=384)")
    p.add_argument("--inter_dim", type=int, default=1536,
                   help="FFN intermediate dimension (DeiT-S=1536)")
    p.add_argument("--num_tokens", type=int, default=196,
                   help="Tokens per image (14x14=196 for DeiT-S)")
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--num_workers_dp", type=int, default=2,
                   help="Workers for DP AllReduce analysis")
    p.add_argument("--num_params_m", type=float, default=22.0,
                   help="Model parameter count in millions (DeiT-S=22M)")
    p.add_argument("--output_dir", type=str, default="outputs/roofline")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Arithmetic intensity calculations
# ---------------------------------------------------------------------------

def compute_arithmetic_intensities(
    D: int,       # hidden dim
    I: int,       # intermediate dim
    T: int,       # tokens per image
    E: int,       # num experts
    B: int = 1,   # batch size
    N_workers: int = 2,
    num_params: int = 22_000_000,
) -> List[Dict]:
    """
    Compute (FLOPs, Bytes, AI) for each major operation.
    Returns list of operation dicts.
    """
    ops = []

    # ------------------------------------------------------------------
    # Dense FFN (fc1 + fc2): O(B*T*D*I) FLOPs, O(B*T*D + D*I) bytes
    # ------------------------------------------------------------------
    dense_flops = 2 * B * T * D * I + 2 * B * T * I * D   # fc1 + fc2
    # Weights: fc1 (I*D) + fc2 (D*I); Activations: input (B*T*D) + hidden (B*T*I)
    dense_bytes = (
        (I * D + D * I) * 4 +          # weights (fp32)
        (B * T * D + B * T * I) * 4    # activations
    )
    ops.append({
        "name": "Dense FFN (fc1+fc2)",
        "flops": dense_flops,
        "bytes": dense_bytes,
        "ai": dense_flops / max(dense_bytes, 1),
        "category": "compute",
    })

    # ------------------------------------------------------------------
    # Expert GEMM (one expert, T/E tokens): O(B*(T/E)*D*I)
    # ------------------------------------------------------------------
    tokens_per_expert = max(1, B * T // E)
    exp_flops = 2 * tokens_per_expert * D * I + 2 * tokens_per_expert * I * D
    exp_bytes = (
        (I * D + D * I) * 4 +
        (tokens_per_expert * D + tokens_per_expert * I) * 4
    )
    ops.append({
        "name": f"Expert GEMM (E={E}, T/E tokens)",
        "flops": exp_flops,
        "bytes": exp_bytes,
        "ai": exp_flops / max(exp_bytes, 1),
        "category": "compute",
    })

    # ------------------------------------------------------------------
    # Router linear: O(B*T*D*E) FLOPs
    # ------------------------------------------------------------------
    router_flops = 2 * B * T * D * E
    router_bytes = (D * E * 4) + (B * T * D * 4) + (B * T * E * 4)
    ops.append({
        "name": "Router linear (gate)",
        "flops": router_flops,
        "bytes": router_bytes,
        "ai": router_flops / max(router_bytes, 1),
        "category": "compute_small",
    })

    # ------------------------------------------------------------------
    # Token sort (argsort, B*T tokens): O(T log T) comparisons
    # ------------------------------------------------------------------
    sort_flops = int(B * T * np.log2(max(B * T, 2)))  # comparisons
    sort_bytes = 2 * B * T * 4  # read indices + write sorted indices
    ops.append({
        "name": "Token argsort (routing)",
        "flops": sort_flops,
        "bytes": sort_bytes,
        "ai": sort_flops / max(sort_bytes, 1),
        "category": "memory_bound",
    })

    # ------------------------------------------------------------------
    # AllReduce (DP-N): Ring-AllReduce, 2*(N-1)/N * P bytes transferred
    # ------------------------------------------------------------------
    P_bytes = num_params * 4  # fp32 gradients
    ar_transfer = 2 * (N_workers - 1) / N_workers * P_bytes
    # FLOPs: ~2*P additions in ring
    ar_flops = 2 * num_params
    ops.append({
        "name": f"AllReduce (DP-{N_workers}, {num_params//1e6:.0f}M params)",
        "flops": ar_flops,
        "bytes": ar_transfer,
        "ai": ar_flops / max(ar_transfer, 1),
        "category": "communication",
    })

    # ------------------------------------------------------------------
    # All-to-All (EP-2): scatter T*D tokens across N devices
    # ------------------------------------------------------------------
    aa_transfer = B * T * D * 4 * 2  # send + receive (both directions)
    aa_flops = B * T * D             # scatter indexing ops
    ops.append({
        "name": f"All-to-All (EP-{N_workers}, T={B*T}, D={D})",
        "flops": aa_flops,
        "bytes": aa_transfer,
        "ai": aa_flops / max(aa_transfer, 1),
        "category": "communication",
    })

    # ------------------------------------------------------------------
    # H-MoE coarse router
    # ------------------------------------------------------------------
    G = 2  # coarse groups
    hmoe_router_flops = 2 * B * T * D * G
    hmoe_router_bytes = (D * G * 4) + (B * T * D * 4) + (B * T * G * 4)
    ops.append({
        "name": "H-MoE coarse router (G=2)",
        "flops": hmoe_router_flops,
        "bytes": hmoe_router_bytes,
        "ai": hmoe_router_flops / max(hmoe_router_bytes, 1),
        "category": "compute_small",
    })

    # ------------------------------------------------------------------
    # Shared basis (SVD projection): O(B*T*D*r) where r = D/2
    # ------------------------------------------------------------------
    r = D // 2
    svd_flops = 2 * B * T * D * r * 2  # V projection + U projection
    svd_bytes = (D * r * 2 * 4) + (B * T * D * 4) + (B * T * r * 4)
    ops.append({
        "name": f"Shared basis SVD (rank={r})",
        "flops": svd_flops,
        "bytes": svd_bytes,
        "ai": svd_flops / max(svd_bytes, 1),
        "category": "compute",
    })

    return ops


# ---------------------------------------------------------------------------
# Roofline chart
# ---------------------------------------------------------------------------

def plot_roofline(
    ops: List[Dict],
    peak_flops: float,
    mem_bw: float,
    measured_points: Optional[List[Dict]] = None,
    save_path: Optional[str] = None,
):
    """
    Generate roofline chart (log-log plot).

    x-axis: arithmetic intensity (FLOPs/byte)
    y-axis: attainable FLOP/s
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available — skipping roofline plot")
        return

    ridge_point = peak_flops / mem_bw  # FLOPs/byte

    ai_range = np.logspace(-2, 4, 1000)
    roofline = np.minimum(mem_bw * ai_range, peak_flops)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Roofline
    ax.loglog(ai_range, roofline, "k-", linewidth=2.5, label="Roofline (GTX 960)")
    ax.axvline(ridge_point, color="gray", linestyle="--", alpha=0.5,
               label=f"Ridge point ({ridge_point:.1f} FLOPs/B)")

    # Operation points
    category_colors = {
        "compute": "#e74c3c",
        "compute_small": "#e67e22",
        "memory_bound": "#3498db",
        "communication": "#9b59b6",
    }
    category_markers = {
        "compute": "o",
        "compute_small": "s",
        "memory_bound": "^",
        "communication": "D",
    }

    for op in ops:
        ai = op["ai"]
        # Attainable FLOP/s = min(peak, BW * AI)
        attainable = min(peak_flops, mem_bw * ai)
        color = category_colors.get(op["category"], "gray")
        marker = category_markers.get(op["category"], "o")

        ax.scatter(ai, attainable, c=color, marker=marker, s=100, zorder=5)
        ax.annotate(
            op["name"],
            (ai, attainable),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            alpha=0.8,
        )

    # Measured points (if provided from profiling data)
    if measured_points:
        for mp in measured_points:
            ax.scatter(
                mp.get("ai", 0),
                mp.get("achieved_flops_per_s", 0),
                c="green",
                marker="*",
                s=200,
                zorder=6,
                label=mp.get("name", "measured"),
            )

    # Labels
    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)", fontsize=12)
    ax.set_ylabel("Attainable FLOP/s", fontsize=12)
    ax.set_title("Roofline Analysis — GTX 960 | CLEAR-MoE Operations", fontsize=13)

    # Legend
    patches = [
        mpatches.Patch(color=c, label=cat.replace("_", " ").title())
        for cat, c in category_colors.items()
    ]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0][:2],
              loc="upper left", fontsize=9)

    # Reference lines
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1e6, 1e14)
    ax.grid(True, alpha=0.3, which="both")

    # Annotate regions
    ax.text(0.02, 1e9, "Memory-bound\n(BW-limited)",
            fontsize=8, color="blue", alpha=0.6)
    ax.text(ridge_point * 10, peak_flops * 0.6, "Compute-bound\n(FLOPs-limited)",
            fontsize=8, color="red", alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Roofline chart saved to {save_path}")
    else:
        plt.savefig("roofline.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Complexity table
# ---------------------------------------------------------------------------

def build_complexity_table(ops: List[Dict]) -> str:
    """Build a markdown complexity table."""
    lines = [
        "# Arithmetic Intensity Analysis\n",
        f"| Operation | FLOPs | Bytes | AI (FLOPs/B) | Regime |",
        "|-----------|-------|-------|-------------|--------|",
    ]
    for op in ops:
        regime = op["category"].replace("_", " ").title()
        lines.append(
            f"| {op['name']} | {op['flops']:.2e} | {op['bytes']:.2e} | "
            f"{op['ai']:.1f} | {regime} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scaling curves
# ---------------------------------------------------------------------------

def plot_scaling_curves(
    results_data: Optional[Dict],
    peak_flops: float,
    mem_bw: float,
    D: int,
    I: int,
    save_path: Optional[str] = None,
):
    """Plot theoretical vs measured scaling for EP, DP, PP."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    N_range = np.array([1, 2, 4, 8])

    # Theoretical: ideal linear scaling
    ideal = N_range

    # EP theoretical: E-way parallel, limited by All-to-All communication
    # As N grows, communication volume = B*T*D*4*2 bytes per AllReduce step
    # Efficiency drops as comm volume ≈ sqrt(N) relative to compute
    ep_efficiency = 1.0 / (1 + 0.1 * (N_range - 1))  # heuristic communication penalty
    ep_speedup = N_range * ep_efficiency

    # DP theoretical: N-way replication, limited by AllReduce gradient sync
    # Ring-AllReduce volume = 2*(N-1)/N * P bytes; nearly constant for large N
    dp_efficiency = 1.0 / (1 + 0.05 * (N_range - 1))
    dp_speedup = N_range * dp_efficiency

    # PP theoretical: S stages, bubble fraction = (S-1)/(M+S-1), M=4 microbatches
    M = 4
    pp_bubble = (N_range - 1) / (M + N_range - 1)
    pp_speedup = N_range * (1 - pp_bubble)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Speedup
    ax1.plot(N_range, ideal, "k--", label="Ideal linear", linewidth=2)
    ax1.plot(N_range, ep_speedup, "r-o", label="Expert Parallel (EP)", linewidth=2)
    ax1.plot(N_range, dp_speedup, "b-s", label="Data Parallel (DP)", linewidth=2)
    ax1.plot(N_range, pp_speedup, "g-^", label="Pipeline Parallel (PP, M=4)", linewidth=2)

    # Overlay measured data if available
    if results_data:
        for mode, r in results_data.items():
            if not isinstance(r, dict):
                continue
            N = r.get("num_workers", 1)
            speedup = r.get("speedup_vs_baseline", 0)
            if N > 0 and speedup > 0:
                ax1.scatter(N, speedup, marker="*", s=200, zorder=5, color="purple",
                            label=f"Measured: {mode}")

    ax1.set_xlabel("Number of Workers N", fontsize=12)
    ax1.set_ylabel("Speedup vs 1-worker", fontsize=12)
    ax1.set_title("Strong Scaling: Speedup vs Workers", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(N_range)

    # Efficiency (speedup / N)
    ax2.plot(N_range, ideal / N_range, "k--", label="Ideal (1.0)", linewidth=2)
    ax2.plot(N_range, ep_speedup / N_range, "r-o", label="EP efficiency", linewidth=2)
    ax2.plot(N_range, dp_speedup / N_range, "b-s", label="DP efficiency", linewidth=2)
    ax2.plot(N_range, pp_speedup / N_range, "g-^", label="PP efficiency", linewidth=2)

    ax2.set_xlabel("Number of Workers N", fontsize=12)
    ax2.set_ylabel("Scaling Efficiency (speedup / N)", fontsize=12)
    ax2.set_title("Scaling Efficiency", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.2)
    ax2.set_xticks(N_range)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Scaling curves saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "roofline.log"),
        ],
    )

    D = args.hidden_dim
    I = args.inter_dim
    T = args.num_tokens
    E = args.num_experts
    peak_flops = args.device_flops
    mem_bw = args.device_bw
    ridge = peak_flops / mem_bw
    num_params = int(args.num_params_m * 1e6)

    logger.info("=" * 70)
    logger.info("CLEAR-MoE Roofline Analysis")
    logger.info("=" * 70)
    logger.info(f"Device: Peak FP32 = {peak_flops/1e12:.1f} TFLOP/s, "
                f"BW = {mem_bw/1e9:.0f} GB/s, "
                f"Ridge = {ridge:.1f} FLOPs/B")
    logger.info(f"Model: D={D}, I={I}, T={T}, E={E}, params={num_params//1e6:.0f}M")

    # Compute arithmetic intensities
    ops = compute_arithmetic_intensities(
        D=D, I=I, T=T, E=E, B=1,
        N_workers=args.num_workers_dp,
        num_params=num_params,
    )

    logger.info("\nArithmetic Intensity Summary:")
    logger.info(f"  {'Operation':<50} {'AI (FLOPs/B)':>14} {'Regime':<20}")
    logger.info("-" * 88)
    for op in ops:
        regime = "compute-bound" if op["ai"] > ridge else "memory-bound"
        logger.info(f"  {op['name']:<50} {op['ai']:>14.1f} {regime}")

    # Save complexity table
    table = build_complexity_table(ops)
    with open(out_dir / "complexity_table.md", "w") as f:
        f.write(table)

    # CSV version
    with open(out_dir / "arithmetic_intensity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "flops", "bytes", "ai", "category"])
        writer.writeheader()
        writer.writerows([{k: op[k] for k in ["name", "flops", "bytes", "ai", "category"]}
                          for op in ops])

    # Load optional measured results
    measured_points = None
    results_data = None
    if args.results_dir:
        results_path = Path(args.results_dir) / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                results_data = json.load(f)
            logger.info(f"Loaded benchmark results from {results_path}")

    # Generate roofline chart
    plot_roofline(
        ops=ops,
        peak_flops=peak_flops,
        mem_bw=mem_bw,
        measured_points=measured_points,
        save_path=str(out_dir / "roofline_chart.png"),
    )

    # Generate scaling curves
    scaling_data = results_data.get("scaling_efficiency") if results_data else None
    plot_scaling_curves(
        results_data=scaling_data,
        peak_flops=peak_flops,
        mem_bw=mem_bw,
        D=D,
        I=I,
        save_path=str(out_dir / "scaling_curves.png"),
    )

    # Print summary table to stdout
    print("\n" + "=" * 80)
    print("  ROOFLINE ANALYSIS — OPERATION SUMMARY")
    print("=" * 80)
    print(f"  {'Operation':<52} {'AI':>6} {'Regime'}")
    print("-" * 80)
    for op in ops:
        regime = "COMPUTE-BOUND" if op["ai"] > ridge else "MEMORY-BOUND"
        print(f"  {op['name']:<52} {op['ai']:>6.1f}  {regime}")
    print("=" * 80)
    print(f"\nRidge point: {ridge:.1f} FLOPs/byte")
    print(f"Peak FP32: {peak_flops/1e12:.1f} TFLOP/s")
    print(f"Memory BW: {mem_bw/1e9:.0f} GB/s")
    print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
