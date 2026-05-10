"""
Script 09: Comprehensive Parallelism Dispatch Benchmark

Compares five dispatch strategies across three load scenarios:
    1. Balanced load       (uniform token distribution across experts)
    2. Moderate imbalance  (dominant expert gets 50% tokens)
    3. Severe imbalance    (dominant expert gets 80% tokens)

For each scenario, measures:
    - Wall-clock latency: mean, p50, p90, std (ms)
    - Throughput: tokens/sec
    - Speedup vs naive sequential baseline
    - Sort overhead fraction (for sorted strategies)
    - Expert compute breakdown

Also sweeps over:
    - num_experts: [2, 4, 8]
    - batch_size: [1, 8, 32]
    - hidden_dim: [192, 384, 768]  (ViT-Ti, ViT-S, ViT-B)

Saves results to: outputs/benchmark_parallelism_<timestamp>/
    - results.json         — full numeric results
    - summary_table.md     — markdown comparison table
    - latency_vs_imbalance.json  — imbalance sweep data

Usage:
    python scripts/09_benchmark_parallelism.py
    python scripts/09_benchmark_parallelism.py --num_experts 8 --hidden_dim 384
    python scripts/09_benchmark_parallelism.py --sweep  # full sweep
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.parallel_expert_dispatch import (
    BenchmarkConfig,
    format_benchmark_table,
    run_dispatch_benchmark,
)
from clear_moe.utils import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dispatch parallelism benchmark")
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=384,
                   help="Hidden dim (192=ViT-Ti, 384=ViT-S, 768=ViT-B)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_tokens", type=int, default=196,
                   help="Tokens per image (196 = 14x14 patches)")
    p.add_argument("--num_warmup", type=int, default=30)
    p.add_argument("--num_iters", type=int, default=200)
    p.add_argument("--sweep", action="store_true",
                   help="Run full sweep over num_experts, hidden_dim, batch_size")
    p.add_argument("--output_dir", type=str, default="outputs/benchmark_parallelism")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load imbalance scenarios
# ---------------------------------------------------------------------------

LOAD_SCENARIOS = {
    "balanced":          0.00,   # uniform
    "mild_imbalance":    0.40,   # one expert: 40%
    "moderate_imbalance": 0.60,  # one expert: 60%
    "severe_imbalance":  0.80,   # one expert: 80%
}

# Strategy display names
STRATEGY_LABELS = {
    "naive_sequential":     "Naive Sequential (baseline)",
    "route_sorted_grouped": "Route-Sorted Grouped (existing)",
    "cuda_stream_parallel": "CUDA Stream Parallel (novel)",
    "expert_fusion":        "Expert Fusion (novel)",
    "adaptive":             "Adaptive Load-Balanced (novel)",
    "triton_scatter_gather": "Triton Scatter-Gather (NEW)",
    "cublas_batched_gemm":  "cuBLAS Batched GEMM (NEW)",
}


# ---------------------------------------------------------------------------
# Triton / cuBLAS extended benchmark
# ---------------------------------------------------------------------------

def run_triton_cublas_benchmark(cfg: BenchmarkConfig, load_imbalance: float = 0.0) -> dict:
    """Run Triton and cuBLAS dispatch benchmarks, returning combined results dict."""
    try:
        from runtime.triton_dispatch import TritonBenchmarkConfig, benchmark_triton_vs_pytorch
        tri_cfg = TritonBenchmarkConfig(
            num_experts=cfg.num_experts,
            hidden_dim=cfg.hidden_dim,
            intermediate_dim=cfg.intermediate_dim,
            num_tokens=cfg.num_tokens,
            batch_size=cfg.batch_size,
            num_warmup=cfg.num_warmup,
            num_iters=cfg.num_iters,
            device=cfg.device,
            load_imbalance=load_imbalance,
        )
        return benchmark_triton_vs_pytorch(tri_cfg)
    except Exception as e:
        logger.warning(f"Triton/cuBLAS benchmark skipped: {e}")
        return {}


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_scenario_benchmark(cfg: BenchmarkConfig) -> dict:
    """Run all strategies across all load scenarios. Returns nested dict."""
    all_results = {}
    for scenario_name, imbalance in LOAD_SCENARIOS.items():
        logger.info(f"  Scenario: {scenario_name} (imbalance={imbalance:.0%})")
        try:
            results = run_dispatch_benchmark(cfg, load_imbalance=imbalance)
            # Extend with Triton + cuBLAS kernel benchmarks
            kernel_results = run_triton_cublas_benchmark(cfg, load_imbalance=imbalance)
            results.update(kernel_results)
            all_results[scenario_name] = results
        except Exception as e:
            logger.warning(f"  Scenario {scenario_name} failed: {e}")
            all_results[scenario_name] = {}
    return all_results


def run_sweep(args) -> dict:
    """Sweep over (num_experts, hidden_dim, batch_size) configurations."""
    sweep_configs = []
    for E in [2, 4, 8]:
        for H in [192, 384, 768]:
            for B in [1, 8, 32]:
                sweep_configs.append((E, H, B))

    sweep_results = {}
    total = len(sweep_configs)

    for i, (E, H, B) in enumerate(sweep_configs):
        key = f"E{E}_H{H}_B{B}"
        logger.info(f"Sweep [{i+1}/{total}]: {key}")
        cfg = BenchmarkConfig(
            num_experts=E,
            hidden_dim=H,
            intermediate_dim=H * 4,
            batch_size=B,
            num_tokens=196,
            num_warmup=10,
            num_iters=50,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # Run balanced + severe imbalance only for sweep efficiency
        sweep_results[key] = {
            "balanced": run_dispatch_benchmark(cfg, load_imbalance=0.0),
            "severe_imbalance": run_dispatch_benchmark(cfg, load_imbalance=0.8),
            "config": {"num_experts": E, "hidden_dim": H, "batch_size": B},
        }

    return sweep_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_summary_table(scenario_results: dict) -> str:
    """Build a markdown summary table covering all scenarios and strategies."""
    lines = [
        "# Dispatch Strategy Benchmark — Summary\n",
        "Comparison of five dispatch strategies across load imbalance scenarios.\n",
        "All times in milliseconds (lower is better).\n",
    ]

    for scenario_name, results in scenario_results.items():
        if not results:
            continue
        imbalance = LOAD_SCENARIOS.get(scenario_name, 0.0)
        lines.append(format_benchmark_table(results, imbalance))
        lines.append("")

    return "\n".join(lines)


def compute_parallelism_gains(scenario_results: dict) -> dict:
    """
    Compute key gains highlighting parallelism impact:
    - Stream parallel vs grouped: latency reduction %
    - Fusion vs grouped under severe imbalance
    - Adaptive vs naive: overall improvement
    """
    gains = {}
    for scenario, results in scenario_results.items():
        if not results:
            continue
        naive = results.get("naive_sequential", {}).get("mean_ms", None)
        grouped = results.get("route_sorted_grouped", {}).get("mean_ms", None)
        stream = results.get("cuda_stream_parallel", {}).get("mean_ms", None)
        fusion = results.get("expert_fusion", {}).get("mean_ms", None)
        adaptive = results.get("adaptive", {}).get("mean_ms", None)

        gains[scenario] = {}
        if naive and grouped:
            gains[scenario]["grouped_vs_naive_speedup"] = round(naive / grouped, 3)
        if grouped and stream:
            gains[scenario]["stream_vs_grouped_speedup"] = round(grouped / stream, 3)
        if grouped and fusion:
            gains[scenario]["fusion_vs_grouped_speedup"] = round(grouped / fusion, 3)
        if naive and adaptive:
            gains[scenario]["adaptive_vs_naive_speedup"] = round(naive / adaptive, 3)
    return gains


def print_highlight_summary(gains: dict, scenario_results: dict):
    """Print key findings to stdout."""
    separator = "=" * 72
    print(f"\n{separator}")
    print("  PARALLELISM BENCHMARK — KEY FINDINGS")
    print(separator)

    for scenario, g in gains.items():
        imbalance = LOAD_SCENARIOS.get(scenario, 0.0)
        print(f"\n[{scenario.upper()} — imbalance={imbalance:.0%}]")
        if "grouped_vs_naive_speedup" in g:
            print(f"  Route-Sorted Grouped vs Naive Sequential:  {g['grouped_vs_naive_speedup']:.2f}×")
        if "stream_vs_grouped_speedup" in g:
            sv = g["stream_vs_grouped_speedup"]
            tag = "✓ STREAM WINS" if sv > 1.05 else "~ comparable"
            print(f"  CUDA Stream Parallel vs Route-Sorted Grouped: {sv:.2f}×  {tag}")
        if "fusion_vs_grouped_speedup" in g:
            fv = g["fusion_vs_grouped_speedup"]
            tag = "✓ FUSION WINS" if fv > 1.05 else "~ comparable"
            print(f"  Expert Fusion vs Route-Sorted Grouped:     {fv:.2f}×  {tag}")
        if "adaptive_vs_naive_speedup" in g:
            print(f"  Adaptive vs Naive Sequential:              {g['adaptive_vs_naive_speedup']:.2f}×")

    print(f"\n{separator}")
    print("  THROUGHPUT COMPARISON (tokens/sec, all scenarios)")
    print(separator)
    for scenario, results in scenario_results.items():
        if not results:
            continue
        print(f"\n  {scenario}:")
        for strat, r in results.items():
            thr = r.get("throughput_tokens_per_sec", 0)
            speedup = r.get("speedup_vs_naive", 1.0)
            label = STRATEGY_LABELS.get(strat, strat)
            print(f"    {label:<45} {thr:>10,.0f} tok/s  ({speedup:.2f}× vs naive)")

    print(f"\n{separator}\n")


def print_phase_breakdown(scenario_results: dict):
    """Print timing phase breakdown from first run of each strategy."""
    print("\n[TIMING PHASE BREAKDOWN — balanced scenario]")
    balanced = scenario_results.get("balanced", {})
    for strat, r in balanced.items():
        # Phase timing comes from DispatchResult.timing, but run_dispatch_benchmark
        # reports aggregate stats only. We log the strategy name here.
        p50 = r.get("p50_ms", 0)
        p90 = r.get("p90_ms", 0)
        print(f"  {STRATEGY_LABELS.get(strat, strat):<45}  p50={p50:.3f}ms  p90={p90:.3f}ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "run.log"),
        ],
    )
    logger.info("=" * 60)
    logger.info("CLEAR-MoE Dispatch Parallelism Benchmark")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("Running on CPU — stream parallelism disabled, times not representative")

    # -----------------------------------------------------------------------
    # Sweep mode
    # -----------------------------------------------------------------------
    if args.sweep:
        logger.info("Running full config sweep...")
        sweep_results = run_sweep(args)
        sweep_path = out_dir / "sweep_results.json"
        with open(sweep_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        logger.info(f"Sweep results saved to: {sweep_path}")
        _print_sweep_summary(sweep_results)
        return

    # -----------------------------------------------------------------------
    # Single config benchmark
    # -----------------------------------------------------------------------
    cfg = BenchmarkConfig(
        num_experts=args.num_experts,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.hidden_dim * 4,
        batch_size=args.batch_size,
        num_tokens=args.num_tokens,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
        device=device,
    )

    logger.info(f"Config: E={cfg.num_experts}, H={cfg.hidden_dim}, "
                f"B={cfg.batch_size}, T={cfg.num_tokens}, "
                f"N={cfg.batch_size * cfg.num_tokens} total tokens")
    logger.info(f"Warmup={cfg.num_warmup}, Iters={cfg.num_iters}")

    # Run benchmark across all load scenarios
    logger.info("\nRunning scenario benchmarks...")
    scenario_results = run_scenario_benchmark(cfg)

    # Compute gains
    gains = compute_parallelism_gains(scenario_results)

    # Print to terminal
    print_highlight_summary(gains, scenario_results)
    print_phase_breakdown(scenario_results)

    # Save results
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "num_experts": cfg.num_experts,
                "hidden_dim": cfg.hidden_dim,
                "intermediate_dim": cfg.intermediate_dim,
                "batch_size": cfg.batch_size,
                "num_tokens": cfg.num_tokens,
                "total_tokens": cfg.batch_size * cfg.num_tokens,
                "device": device,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "num_warmup": cfg.num_warmup,
                "num_iters": cfg.num_iters,
            },
            "scenario_results": scenario_results,
            "parallelism_gains": gains,
        }, f, indent=2)

    # Save markdown summary
    table_path = out_dir / "summary_table.md"
    with open(table_path, "w") as f:
        f.write(build_summary_table(scenario_results))

    logger.info(f"\nResults saved to: {out_dir}")
    logger.info(f"  {results_path.name}")
    logger.info(f"  {table_path.name}")


def _print_sweep_summary(sweep_results: dict):
    """Print top-level sweep insights."""
    print("\n[SWEEP SUMMARY — Stream Parallel vs Grouped speedup by config]")
    print(f"{'Config':<18} {'Balanced speedup':>18} {'Severe imb. speedup':>20}")
    print("-" * 60)
    for key, data in sweep_results.items():
        bal = data.get("balanced", {})
        sev = data.get("severe_imbalance", {})
        g_bal = bal.get("route_sorted_grouped", {}).get("mean_ms", 1.0)
        s_bal = bal.get("cuda_stream_parallel", {}).get("mean_ms", 1.0)
        g_sev = sev.get("route_sorted_grouped", {}).get("mean_ms", 1.0)
        f_sev = sev.get("expert_fusion", {}).get("mean_ms", 1.0)
        sp_bal = g_bal / s_bal if s_bal > 0 else 0.0
        sp_sev = g_sev / f_sev if f_sev > 0 else 0.0
        print(f"  {key:<16}  {sp_bal:>14.2f}×  {sp_sev:>16.2f}×")


if __name__ == "__main__":
    main()
