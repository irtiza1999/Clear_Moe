#!/usr/bin/env python3
"""
Script 10 — CPU vs GPU Comparison Benchmark

Runs the same MoE workload under all hardware + dispatch combinations:
    - CPU serial (1 thread)
    - CPU multi-threaded (up to physical core count)
    - GPU naive dispatch
    - GPU route-sorted grouped dispatch
    - GPU CUDA stream parallel dispatch
    - GPU expert fusion dispatch
    - GPU adaptive dispatch
    - (Optional) Heterogeneous CPU+GPU expert placement

Outputs:
    - normalised_speedup_table.md   — relative throughput vs CPU serial
    - pareto.csv                    — quality vs throughput data
    - results.json                  — full raw timing data

Usage:
    python scripts/10_cpu_vs_gpu_benchmark.py
    python scripts/10_cpu_vs_gpu_benchmark.py --num_experts 8 --hidden_dim 768
    python scripts/10_cpu_vs_gpu_benchmark.py --cpu_threads 1,4,8,16
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.parallel_expert_dispatch import BenchmarkConfig, run_dispatch_benchmark
from clear_moe.utils import set_seed

logger = logging.getLogger(__name__)
_INTEROP_THREADS_SET = False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CPU vs GPU dispatch benchmark")
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--num_tokens", type=int, default=196)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_warmup", type=int, default=50)
    p.add_argument("--num_iters", type=int, default=200)
    p.add_argument("--cpu_threads", type=str, default="1,4,8,16",
                   help="Comma-separated list of CPU thread counts to test")
    p.add_argument("--load_imbalance", type=float, default=0.0,
                   help="Fraction of tokens to dominant expert (0=balanced)")
    p.add_argument("--output_dir", type=str, default="outputs/cpu_vs_gpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# CPU thread benchmark helper
# ---------------------------------------------------------------------------

def bench_cpu_threads(
    cfg: BenchmarkConfig,
    thread_count: int,
    load_imbalance: float,
) -> dict:
    """Run all dispatch strategies on CPU with given thread count."""
    global _INTEROP_THREADS_SET
    torch.set_num_threads(thread_count)
    if not _INTEROP_THREADS_SET:
        torch.set_num_interop_threads(max(1, thread_count // 2))
        _INTEROP_THREADS_SET = True

    cpu_cfg = BenchmarkConfig(
        num_experts=cfg.num_experts,
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
        batch_size=cfg.batch_size,
        num_tokens=cfg.num_tokens,
        num_warmup=cfg.num_warmup,
        num_iters=cfg.num_iters,
        device="cpu",
    )

    results = run_dispatch_benchmark(cpu_cfg, load_imbalance=load_imbalance)
    return results


# ---------------------------------------------------------------------------
# GPU benchmark helper
# ---------------------------------------------------------------------------

def bench_gpu(
    cfg: BenchmarkConfig,
    load_imbalance: float,
) -> dict:
    """Run all dispatch strategies on GPU."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — skipping GPU benchmark")
        return {}

    gpu_cfg = BenchmarkConfig(
        num_experts=cfg.num_experts,
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
        batch_size=cfg.batch_size,
        num_tokens=cfg.num_tokens,
        num_warmup=cfg.num_warmup,
        num_iters=cfg.num_iters,
        device="cuda",
    )
    return run_dispatch_benchmark(gpu_cfg, load_imbalance=load_imbalance)


# ---------------------------------------------------------------------------
# Triton / cuBLAS optional benchmark
# ---------------------------------------------------------------------------

def bench_triton_cublas(
    cfg: BenchmarkConfig,
    load_imbalance: float,
) -> dict:
    """Benchmark Triton and cuBLAS dispatch if available."""
    if not torch.cuda.is_available():
        return {}

    try:
        from runtime.triton_dispatch import (
            TritonBenchmarkConfig,
            benchmark_triton_vs_pytorch,
        )
        tri_cfg = TritonBenchmarkConfig(
            num_experts=cfg.num_experts,
            hidden_dim=cfg.hidden_dim,
            intermediate_dim=cfg.intermediate_dim,
            num_tokens=cfg.num_tokens,
            batch_size=cfg.batch_size,
            num_warmup=cfg.num_warmup,
            num_iters=cfg.num_iters,
            device="cuda",
            load_imbalance=load_imbalance,
        )
        return benchmark_triton_vs_pytorch(tri_cfg)
    except Exception as e:
        logger.warning(f"Triton/cuBLAS benchmark failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Heterogeneous CPU+GPU benchmark
# ---------------------------------------------------------------------------

def bench_heterogeneous(
    cfg: BenchmarkConfig,
    load_imbalance: float,
) -> dict:
    """
    Simulate heterogeneous CPU+GPU expert placement.
    Half experts on CPU, half on GPU. Token routing crosses PCIe boundary.
    Measures the PCIe transfer overhead.
    """
    if not torch.cuda.is_available():
        return {}

    E = cfg.num_experts
    D = cfg.hidden_dim
    I = cfg.intermediate_dim
    N = cfg.batch_size * cfg.num_tokens

    # Build CPU experts
    cpu_experts = [(
        nn.Linear(D, I, bias=True),
        nn.Linear(I, D, bias=True),
    ) for _ in range(E // 2)]

    # Build GPU experts
    gpu_experts = [(
        nn.Linear(D, I, bias=True).cuda(),
        nn.Linear(I, D, bias=True).cuda(),
    ) for _ in range(E - E // 2)]

    activation = nn.GELU()

    def _run_heterogeneous(x_gpu, expert_ids):
        """Route tokens: CPU experts get half, GPU experts get rest."""
        out = torch.zeros_like(x_gpu)
        for e in range(E // 2):
            mask = expert_ids == e
            if not mask.any():
                continue
            # Transfer to CPU for processing
            x_cpu = x_gpu[mask].cpu()
            fc1, fc2 = cpu_experts[e]
            h = activation(fc1(x_cpu))
            r = fc2(h)
            out[mask] = r.cuda()  # Transfer back

        for e_local, e_global in enumerate(range(E // 2, E)):
            mask = expert_ids == e_global
            if not mask.any():
                continue
            fc1, fc2 = gpu_experts[e_local]
            h = activation(fc1(x_gpu[mask]))
            out[mask] = fc2(h)

        return out

    # Benchmark
    timings = []
    for _ in range(cfg.num_warmup):
        x = torch.randn(N, D, device="cuda")
        expert_ids = torch.randint(0, E, (N,), device="cuda")
        with torch.no_grad():
            _run_heterogeneous(x, expert_ids)
        torch.cuda.synchronize()

    for _ in range(cfg.num_iters):
        x = torch.randn(N, D, device="cuda")
        expert_ids = torch.randint(0, E, (N,), device="cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            _run_heterogeneous(x, expert_ids)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)

    arr = np.array(timings)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "throughput_tokens_per_sec": float(N / (arr.mean() / 1000)),
        "note": f"{E // 2}/{E} experts on CPU, rest on GPU (PCIe-limited)",
    }


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def build_speedup_table(all_results: dict, baseline_throughput: float) -> str:
    """Build markdown normalised speedup table."""
    lines = [
        "# CPU vs GPU vs Parallelism — Normalised Speedup Table",
        "",
        f"Baseline: CPU serial (1 thread) throughput = {baseline_throughput:.0f} tok/s",
        "",
        "| Configuration | Backend | Throughput (tok/s) | Speedup vs CPU-serial |",
        "|--------------|---------|-------------------|----------------------|",
    ]

    for config_name, results in sorted(all_results.items()):
        if not results:
            continue
        thr = results.get("throughput_tokens_per_sec", 0.0)
        speedup = thr / max(baseline_throughput, 1.0)
        backend = results.get("backend", config_name)
        lines.append(f"| {config_name} | {backend} | {thr:,.0f} | {speedup:.2f}× |")

    return "\n".join(lines)


def write_pareto_csv(all_results: dict, out_path: Path):
    """Write CSV for quality vs throughput Pareto plot."""
    rows = []
    for config_name, results in all_results.items():
        if not results:
            continue
        rows.append({
            "config": config_name,
            "throughput_tok_s": results.get("throughput_tokens_per_sec", 0),
            "latency_p50_ms": results.get("p50_ms", 0),
            "latency_p90_ms": results.get("p90_ms", 0),
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "throughput_tok_s",
                                               "latency_p50_ms", "latency_p90_ms"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "run.log"),
        ],
    )

    logger.info("=" * 70)
    logger.info("CLEAR-MoE CPU vs GPU vs Parallelism Benchmark")
    logger.info("=" * 70)

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    import multiprocessing
    phys_cores = multiprocessing.cpu_count()
    logger.info(f"CPU cores: {phys_cores}")

    cpu_thread_counts = [int(t) for t in args.cpu_threads.split(",")]
    cpu_thread_counts = [t for t in cpu_thread_counts if t <= phys_cores]

    base_cfg = BenchmarkConfig(
        num_experts=args.num_experts,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.hidden_dim * 4,
        batch_size=args.batch_size,
        num_tokens=args.num_tokens,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
        device="cpu",
    )

    all_results = {}

    # ------------------------------------------------------------------
    # CPU experiments
    # ------------------------------------------------------------------
    for t in cpu_thread_counts:
        label = f"cpu_t{t}"
        logger.info(f"\n[{label}] Running CPU benchmark with {t} threads...")
        res = bench_cpu_threads(base_cfg, t, args.load_imbalance)

        for strategy, r in res.items():
            key = f"{label}_{strategy}"
            r["backend"] = f"CPU {t}T"
            all_results[key] = r

    # ------------------------------------------------------------------
    # GPU experiments
    # ------------------------------------------------------------------
    logger.info("\n[GPU] Running GPU dispatch benchmarks...")
    gpu_res = bench_gpu(base_cfg, args.load_imbalance)
    for strategy, r in gpu_res.items():
        key = f"gpu_{strategy}"
        r["backend"] = "GPU (CUDA)"
        all_results[key] = r

    # ------------------------------------------------------------------
    # Triton / cuBLAS
    # ------------------------------------------------------------------
    logger.info("\n[Triton/cuBLAS] Running kernel dispatch benchmarks...")
    kernel_res = bench_triton_cublas(base_cfg, args.load_imbalance)
    for strategy, r in kernel_res.items():
        key = f"gpu_{strategy}"
        r["backend"] = "GPU (Triton/cuBLAS)"
        all_results[key] = r

    # ------------------------------------------------------------------
    # Heterogeneous CPU+GPU
    # ------------------------------------------------------------------
    logger.info("\n[Heterogeneous] Running CPU+GPU split benchmark...")
    het_res = bench_heterogeneous(base_cfg, args.load_imbalance)
    if het_res:
        het_res["backend"] = "CPU+GPU (PCIe)"
        all_results["heterogeneous_split"] = het_res

    # ------------------------------------------------------------------
    # Build baseline: CPU serial (1 thread, naive strategy)
    # ------------------------------------------------------------------
    baseline_key = f"cpu_t1_naive_sequential"
    baseline_thr = all_results.get(baseline_key, {}).get("throughput_tokens_per_sec", 1.0)
    if baseline_thr == 0:
        # Fallback: find any cpu_t1 result
        for k, v in all_results.items():
            if "cpu_t1" in k:
                baseline_thr = v.get("throughput_tokens_per_sec", 1.0)
                break

    logger.info(f"\nBaseline throughput: {baseline_thr:.0f} tok/s")

    # ------------------------------------------------------------------
    # Print summary table to stdout
    # ------------------------------------------------------------------
    sep = "=" * 80
    print(f"\n{sep}")
    print("  CPU vs GPU NORMALISED SPEEDUP SUMMARY")
    print(sep)
    print(f"  {'Config':<40} {'Throughput':>12}  {'Speedup':>10}")
    print(f"  {'-'*40} {'-'*12}  {'-'*10}")

    for key, r in sorted(all_results.items()):
        if not r:
            continue
        thr = r.get("throughput_tokens_per_sec", 0)
        speedup = thr / max(baseline_thr, 1.0)
        print(f"  {key:<40} {thr:>12,.0f}  {speedup:>9.2f}×")
    print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "config": {
                "num_experts": args.num_experts,
                "hidden_dim": args.hidden_dim,
                "num_tokens": args.num_tokens,
                "batch_size": args.batch_size,
                "load_imbalance": args.load_imbalance,
                "total_tokens": args.batch_size * args.num_tokens,
            },
            "baseline_throughput_tok_s": baseline_thr,
            "all_results": all_results,
        }, f, indent=2)

    table = build_speedup_table(all_results, baseline_thr)
    with open(out_dir / "normalised_speedup_table.md", "w") as f:
        f.write(table)

    write_pareto_csv(all_results, out_dir / "pareto.csv")

    logger.info(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
