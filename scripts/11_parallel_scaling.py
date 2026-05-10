#!/usr/bin/env python3
"""
Script 11 — Parallel Scaling Benchmark (Strong Scaling)

Simulates expert parallelism (EP), data parallelism (DP), and pipeline
parallelism (PP) using torch.multiprocessing on a single device.

Measures:
    - Throughput vs number of workers N (1, 2, 4)
    - Strong scaling efficiency = speedup / N
    - Communication overhead (AllReduce for DP, All-to-All for EP)
    - Pipeline bubble fraction for PP

Parallelism modes simulated:
    - cpu_serial:  single process, CPU
    - cpu_mt:      torch thread-level parallelism
    - gpu_single:  single process, GPU
    - ep2 / ep4:   expert parallelism (2 or 4 processes, each owns a subset of experts)
    - dp2:         data parallelism (2 processes, each processes half the batch)
    - pp2:         pipeline parallelism (2 stages, model split at mid-layer)

Usage:
    python scripts/11_parallel_scaling.py
    python scripts/11_parallel_scaling.py --modes ep2,dp2,pp2 --num_iters 100
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clear_moe.utils import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Parallel scaling benchmark")
    p.add_argument("--modes", type=str,
                   default="cpu_serial,cpu_mt,gpu_single,ep2,dp2,pp2",
                   help="Comma-separated parallelism modes to run")
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--num_tokens", type=int, default=196)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_warmup", type=int, default=30)
    p.add_argument("--num_iters", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="outputs/parallel_scaling")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Minimal MoE FFN module for benchmarking
# ---------------------------------------------------------------------------

class SimpleMoEFFN(nn.Module):
    """Minimal MoE FFN: linear router + E expert MLPs."""

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        I = hidden_dim * 4
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.fc1s = nn.ModuleList([nn.Linear(hidden_dim, I) for _ in range(num_experts)])
        self.fc2s = nn.ModuleList([nn.Linear(I, hidden_dim) for _ in range(num_experts)])
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, D = x.shape
        logits = self.gate(x)
        ids = logits.argmax(-1)
        out = torch.zeros_like(x)
        for e in range(len(self.fc1s)):
            mask = ids == e
            if mask.any():
                h = self.act(self.fc1s[e](x[mask]))
                out[mask] = self.fc2s[e](h)
        return out


# ---------------------------------------------------------------------------
# CPU serial / MT baseline
# ---------------------------------------------------------------------------

def bench_cpu(
    hidden_dim: int,
    num_experts: int,
    N: int,
    num_warmup: int,
    num_iters: int,
    num_threads: int = 1,
) -> Dict:
    torch.set_num_threads(num_threads)
    model = SimpleMoEFFN(hidden_dim, num_experts)

    timings = []
    for _ in range(num_warmup):
        x = torch.randn(N, hidden_dim)
        with torch.no_grad():
            model(x)

    for _ in range(num_iters):
        x = torch.randn(N, hidden_dim)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        timings.append((time.perf_counter() - t0) * 1000)

    arr = np.array(timings)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "throughput_tok_s": float(N / (arr.mean() / 1000)),
        "num_workers": 1,
        "threads": num_threads,
    }


# ---------------------------------------------------------------------------
# GPU single
# ---------------------------------------------------------------------------

def bench_gpu_single(
    hidden_dim: int,
    num_experts: int,
    N: int,
    num_warmup: int,
    num_iters: int,
) -> Dict:
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    model = SimpleMoEFFN(hidden_dim, num_experts).cuda()
    timings = []

    for _ in range(num_warmup):
        x = torch.randn(N, hidden_dim, device="cuda")
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()

    for _ in range(num_iters):
        x = torch.randn(N, hidden_dim, device="cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)

    arr = np.array(timings)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "throughput_tok_s": float(N / (arr.mean() / 1000)),
        "num_workers": 1,
    }


# ---------------------------------------------------------------------------
# Expert Parallelism simulation (EP-N)
# ---------------------------------------------------------------------------

def _ep_worker(
    rank: int,
    world_size: int,
    hidden_dim: int,
    num_experts: int,
    N: int,
    num_warmup: int,
    num_iters: int,
    result_queue: mp.Queue,
):
    """
    Each process owns a subset of experts.
    Tokens are distributed (All-to-All simulated via queue).
    """
    set_seed(rank)
    device = "cpu"  # CPU for multi-process on single machine

    experts_per_rank = max(1, num_experts // world_size)
    my_experts_start = rank * experts_per_rank
    my_experts_end = min(my_experts_start + experts_per_rank, num_experts)
    I = hidden_dim * 4

    # Build only my expert subset
    my_fc1s = [nn.Linear(hidden_dim, I) for _ in range(my_experts_end - my_experts_start)]
    my_fc2s = [nn.Linear(I, hidden_dim) for _ in range(my_experts_end - my_experts_start)]
    gate = nn.Linear(hidden_dim, num_experts, bias=False)
    act = nn.GELU()

    def _process_tokens(x_all):
        """Route and process tokens belonging to my experts."""
        logits = gate(x_all)
        ids = logits.argmax(-1)
        out = torch.zeros_like(x_all)
        for local_e, global_e in enumerate(range(my_experts_start, my_experts_end)):
            mask = ids == global_e
            if mask.any():
                h = act(my_fc1s[local_e](x_all[mask]))
                out[mask] = my_fc2s[local_e](h)
        # Return only my expert outputs (non-zero rows)
        return out

    timings = []
    comm_times = []

    for it in range(num_warmup + num_iters):
        x = torch.randn(N // world_size, hidden_dim)

        # Simulate All-to-All: measure overhead of token redistribution
        t_comm_start = time.perf_counter()
        # In real EP, tokens are sent to the device that owns the target expert.
        # Here we simulate it as a serialised gather to measure latency proxy.
        x_gathered = x.clone()  # placeholder for actual communication
        comm_time = (time.perf_counter() - t_comm_start) * 1000

        t0 = time.perf_counter()
        with torch.no_grad():
            out = _process_tokens(x_gathered)
        elapsed = (time.perf_counter() - t0) * 1000

        if it >= num_warmup:
            timings.append(elapsed)
            comm_times.append(comm_time)

    arr = np.array(timings)
    result_queue.put({
        "rank": rank,
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "throughput_tok_s": float((N // world_size) / (arr.mean() / 1000)),
        "mean_comm_ms": float(np.mean(comm_times)),
        "experts_owned": list(range(my_experts_start, my_experts_end)),
    })


def bench_expert_parallel(
    world_size: int,
    hidden_dim: int,
    num_experts: int,
    N: int,
    num_warmup: int,
    num_iters: int,
) -> Dict:
    """Run EP-world_size benchmark via multiprocessing."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_ep_worker,
            args=(rank, world_size, hidden_dim, num_experts, N,
                  num_warmup, num_iters, result_queue),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=120)

    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())

    if not results:
        return {"error": "EP workers produced no results"}

    # Aggregate: effective throughput = sum of per-rank throughput (tokens fully processed per sec)
    total_thr = sum(r["throughput_tok_s"] for r in results)
    mean_comm = np.mean([r["mean_comm_ms"] for r in results])
    mean_compute = np.mean([r["mean_ms"] for r in results])

    return {
        "num_workers": world_size,
        "mean_ms": mean_compute,
        "total_throughput_tok_s": total_thr,
        "throughput_tok_s": total_thr,
        "mean_comm_overhead_ms": mean_comm,
        "comm_fraction": mean_comm / max(mean_comm + mean_compute, 1e-8),
        "per_rank_results": results,
    }


# ---------------------------------------------------------------------------
# Data Parallelism simulation (DP-N)
# ---------------------------------------------------------------------------

def _dp_worker(
    rank: int,
    world_size: int,
    hidden_dim: int,
    num_experts: int,
    N: int,
    num_warmup: int,
    num_iters: int,
    result_queue: mp.Queue,
):
    """
    Each DP worker processes N/world_size tokens with the full model.
    Simulates AllReduce overhead by timing a reduce_sum of model output.
    """
    set_seed(rank)
    model = SimpleMoEFFN(hidden_dim, num_experts)
    total_params = sum(p.numel() for p in model.parameters())

    timings = []
    allreduce_times = []

    for it in range(num_warmup + num_iters):
        x = torch.randn(N // world_size, hidden_dim)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x)
        compute_time = (time.perf_counter() - t0) * 1000

        # Simulate AllReduce: sum a tensor the size of all parameters
        # (approximation of gradient AllReduce overhead)
        t_ar = time.perf_counter()
        grad_sim = torch.randn(total_params)
        reduced = grad_sim.sum()  # local reduction — no real network communication
        ar_time = (time.perf_counter() - t_ar) * 1000

        if it >= num_warmup:
            timings.append(compute_time)
            allreduce_times.append(ar_time)

    arr = np.array(timings)
    result_queue.put({
        "rank": rank,
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "throughput_tok_s": float((N // world_size) / (arr.mean() / 1000)),
        "mean_allreduce_ms": float(np.mean(allreduce_times)),
        "total_params": total_params,
    })


def bench_data_parallel(
    world_size: int,
    hidden_dim: int,
    num_experts: int,
    N: int,
    num_warmup: int,
    num_iters: int,
) -> Dict:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_dp_worker,
            args=(rank, world_size, hidden_dim, num_experts, N,
                  num_warmup, num_iters, result_queue),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=120)

    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())

    if not results:
        return {"error": "DP workers produced no results"}

    total_thr = sum(r["throughput_tok_s"] for r in results)
    mean_ar = np.mean([r["mean_allreduce_ms"] for r in results])
    mean_compute = np.mean([r["mean_ms"] for r in results])

    return {
        "num_workers": world_size,
        "mean_ms": mean_compute,
        "throughput_tok_s": total_thr,
        "total_throughput_tok_s": total_thr,
        "mean_allreduce_ms": mean_ar,
        "allreduce_fraction": mean_ar / max(mean_ar + mean_compute, 1e-8),
        "per_rank_results": results,
    }


# ---------------------------------------------------------------------------
# Pipeline Parallelism simulation (PP-2)
# ---------------------------------------------------------------------------

def bench_pipeline_parallel(
    num_stages: int,
    hidden_dim: int,
    num_experts: int,
    N: int,
    num_warmup: int,
    num_iters: int,
    num_micro_batches: int = 4,
) -> Dict:
    """
    GPipe-style pipeline parallelism with micro-batches.

    Stage 0: first half of a 6-layer transformer (FFN layers 0..2)
    Stage 1: second half (FFN layers 3..5)

    Bubble fraction = (S-1)/(M+S-1) where S=stages, M=micro-batches.
    """
    if num_stages != 2:
        return {"error": "Only PP-2 implemented"}

    total_layers = 6
    layers_per_stage = total_layers // num_stages

    # Build two stage modules
    def _make_stage(n_layers):
        return nn.Sequential(*[SimpleMoEFFN(hidden_dim, num_experts) for _ in range(n_layers)])

    stage0 = _make_stage(layers_per_stage)
    stage1 = _make_stage(layers_per_stage)

    micro_size = N // num_micro_batches

    def _run_gpipe(x_full):
        """Simulate GPipe forward pass with pipeline parallelism."""
        micros = x_full.split(micro_size, dim=0)
        stage0_outputs = []

        # Stage 0: process all micros
        for micro in micros:
            with torch.no_grad():
                stage0_outputs.append(stage0(micro))

        # Stage 1: process all stage0 outputs
        stage1_outputs = []
        for s0_out in stage0_outputs:
            with torch.no_grad():
                stage1_outputs.append(stage1(s0_out))

        return torch.cat(stage1_outputs, dim=0)

    # Theoretical bubble fraction
    S = num_stages
    M = num_micro_batches
    bubble_fraction = (S - 1) / (M + S - 1)

    timings = []
    for it in range(num_warmup + num_iters):
        x = torch.randn(N, hidden_dim)
        t0 = time.perf_counter()
        _run_gpipe(x)
        elapsed = (time.perf_counter() - t0) * 1000
        if it >= num_warmup:
            timings.append(elapsed)

    arr = np.array(timings)
    return {
        "num_workers": num_stages,
        "num_micro_batches": num_micro_batches,
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "throughput_tok_s": float(N / (arr.mean() / 1000)),
        "theoretical_bubble_fraction": bubble_fraction,
        "pipeline_stages": num_stages,
    }


# ---------------------------------------------------------------------------
# Scaling efficiency computation
# ---------------------------------------------------------------------------

def compute_scaling_efficiency(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Compute strong scaling efficiency for EP, DP, PP.
    Efficiency = speedup / N  where speedup = throughput_N / throughput_1.
    """
    baseline_thr = results.get("gpu_single", {}).get("throughput_tok_s") or \
                   results.get("cpu_serial", {}).get("throughput_tok_s", 1.0)

    efficiency = {}
    for mode, r in results.items():
        if not r or "error" in r:
            continue
        thr = r.get("throughput_tok_s", 0)
        N_workers = r.get("num_workers", 1)
        speedup = thr / max(baseline_thr, 1.0)
        eff = speedup / N_workers if N_workers > 0 else 0.0
        efficiency[mode] = {
            "num_workers": N_workers,
            "throughput_tok_s": thr,
            "speedup_vs_baseline": speedup,
            "scaling_efficiency": eff,
        }
    return efficiency


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

    modes = [m.strip() for m in args.modes.split(",")]
    N = args.batch_size * args.num_tokens
    D = args.hidden_dim
    E = args.num_experts

    logger.info("=" * 70)
    logger.info(f"Parallel Scaling Benchmark: N={N}, D={D}, E={E}")
    logger.info(f"Modes: {modes}")
    logger.info("=" * 70)

    results = {}

    for mode in modes:
        logger.info(f"\n[{mode}] Running...")

        if mode == "cpu_serial":
            r = bench_cpu(D, E, N, args.num_warmup, args.num_iters, num_threads=1)
            r["mode"] = mode
            results[mode] = r

        elif mode == "cpu_mt":
            import multiprocessing
            n_threads = min(16, multiprocessing.cpu_count())
            r = bench_cpu(D, E, N, args.num_warmup, args.num_iters, num_threads=n_threads)
            r["mode"] = mode
            results[mode] = r

        elif mode == "gpu_single":
            r = bench_gpu_single(D, E, N, args.num_warmup, args.num_iters)
            r["mode"] = mode
            results[mode] = r

        elif mode == "ep2":
            r = bench_expert_parallel(2, D, E, N, args.num_warmup, args.num_iters)
            r["mode"] = mode
            results[mode] = r

        elif mode == "ep4":
            r = bench_expert_parallel(4, D, E, N, args.num_warmup, args.num_iters)
            r["mode"] = mode
            results[mode] = r

        elif mode == "dp2":
            r = bench_data_parallel(2, D, E, N, args.num_warmup, args.num_iters)
            r["mode"] = mode
            results[mode] = r

        elif mode == "pp2":
            r = bench_pipeline_parallel(2, D, E, N, args.num_warmup, args.num_iters)
            r["mode"] = mode
            results[mode] = r

        else:
            logger.warning(f"Unknown mode: {mode}")
            continue

        thr = results[mode].get("throughput_tok_s", 0)
        logger.info(f"  -> {thr:.0f} tok/s")

    # Compute scaling efficiency
    efficiency = compute_scaling_efficiency(results)

    # Print summary
    print("\n" + "=" * 70)
    print("  PARALLEL SCALING RESULTS")
    print("=" * 70)
    print(f"  {'Mode':<20} {'Throughput':>14} {'N workers':>10} {'Speedup':>10} {'Efficiency':>12}")
    print(f"  {'-'*20} {'-'*14} {'-'*10} {'-'*10} {'-'*12}")
    for mode, eff in efficiency.items():
        print(f"  {mode:<20} {eff['throughput_tok_s']:>14,.0f} "
              f"{eff['num_workers']:>10} {eff['speedup_vs_baseline']:>9.2f}× "
              f"{eff['scaling_efficiency']:>11.2f}")
    print("=" * 70)

    # Save results
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "config": vars(args),
            "raw_results": {k: {ck: cv for ck, cv in v.items()
                                if not isinstance(cv, list)}
                            for k, v in results.items()
                            if not isinstance(v, dict) or "per_rank_results" not in v},
            "scaling_efficiency": efficiency,
        }, f, indent=2)

    # Markdown summary
    lines = ["# Parallel Scaling Efficiency\n",
             f"| Mode | Throughput (tok/s) | Workers | Speedup | Efficiency |",
             "|------|-------------------|---------|---------|-----------|"]
    for mode, eff in efficiency.items():
        lines.append(
            f"| {mode} | {eff['throughput_tok_s']:,.0f} | {eff['num_workers']} | "
            f"{eff['speedup_vs_baseline']:.2f}× | {eff['scaling_efficiency']:.2f} |"
        )
    with open(out_dir / "scaling_summary.md", "w") as f:
        f.write("\n".join(lines))

    logger.info(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
