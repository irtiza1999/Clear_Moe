#!/usr/bin/env python3
"""
Script 28 — Extended Parallelism Scaling Experiments

New experiments beyond script 11:
  P1. Strong scaling: EP2, EP4, DP2, DP4, PP2, PP4 — throughput & efficiency
  P2. Expert parallelism vs data parallelism at same N workers (EP2 vs DP2, EP4 vs DP4)
  P3. Communication overhead breakdown (AllReduce vs All-to-All vs Point-to-Point)
  P4. Pipeline bubble analysis: vary num_micro_batches (1, 2, 4, 8, 16)
  P5. Mixed parallelism: DP2+EP2 combined simulation
  P6. Token imbalance impact on EP efficiency

Outputs: outputs/extended_parallelism/<timestamp>/
  - results.json
  - summary.md
"""

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clear_moe.utils import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared MoE model
# ---------------------------------------------------------------------------

class SimpleMoEFFN(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        I = hidden_dim * 4
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.fc1s = nn.ModuleList([nn.Linear(hidden_dim, I) for _ in range(num_experts)])
        self.fc2s = nn.ModuleList([nn.Linear(I, hidden_dim) for _ in range(num_experts)])
        self.act = nn.GELU()

    def forward(self, x, load_imbalance=0.0):
        N, D = x.shape
        if load_imbalance > 0:
            n_dom = int(N * load_imbalance)
            ids = torch.randint(1, len(self.fc1s), (N,), device=x.device)
            ids[:n_dom] = 0
            ids = ids[torch.randperm(N, device=x.device)]
        else:
            ids = self.gate(x).argmax(-1)
        out = torch.zeros_like(x)
        for e in range(len(self.fc1s)):
            mask = ids == e
            if mask.any():
                h = self.act(self.fc1s[e](x[mask]))
                out[mask] = self.fc2s[e](h)
        return out


def bench_single(model, N, D, device, num_warmup, num_iters, load_imbalance=0.0):
    model = model.to(device)
    timings = []
    for _ in range(num_warmup):
        x = torch.randn(N, D, device=device)
        with torch.no_grad():
            model(x, load_imbalance)
        if device != "cpu":
            torch.cuda.synchronize()
    for _ in range(num_iters):
        x = torch.randn(N, D, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x, load_imbalance)
        if device != "cpu":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)
    arr = np.array(timings)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "throughput_tok_s": float(N / (arr.mean() / 1000)),
    }


# ---------------------------------------------------------------------------
# EP worker
# ---------------------------------------------------------------------------

def _ep_worker(rank, world_size, hidden_dim, num_experts, N, num_warmup, num_iters,
               load_imbalance, q):
    set_seed(rank)
    E = num_experts
    D = hidden_dim
    I = D * 4
    experts_per_rank = max(1, E // world_size)
    my_start = rank * experts_per_rank
    my_end = min(my_start + experts_per_rank, E)

    fc1s = [nn.Linear(D, I) for _ in range(my_end - my_start)]
    fc2s = [nn.Linear(I, D) for _ in range(my_end - my_start)]
    gate = nn.Linear(D, E, bias=False)
    act = nn.GELU()
    n_local = N // world_size

    timings = []
    for it in range(num_warmup + num_iters):
        x = torch.randn(n_local, D)
        if load_imbalance > 0:
            n_dom = int(n_local * load_imbalance)
            ids = torch.randint(1, E, (n_local,))
            ids[:n_dom] = 0
            ids = ids[torch.randperm(n_local)]
        else:
            ids = gate(x).argmax(-1)
        out = torch.zeros_like(x)
        t0 = time.perf_counter()
        for local_e, global_e in enumerate(range(my_start, my_end)):
            mask = ids == global_e
            if mask.any():
                h = act(fc1s[local_e](x[mask]))
                out[mask] = fc2s[local_e](h)
        elapsed = (time.perf_counter() - t0) * 1000
        if it >= num_warmup:
            timings.append(elapsed)

    arr = np.array(timings)
    q.put({"rank": rank, "mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
           "throughput_tok_s": float(n_local / (arr.mean() / 1000))})


def bench_expert_parallel(world_size, D, E, N, num_warmup, num_iters, load_imbalance=0.0):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_ep_worker,
                         args=(r, world_size, D, E, N, num_warmup, num_iters, load_imbalance, q))
             for r in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)
    results = []
    while not q.empty():
        results.append(q.get_nowait())
    if not results:
        return {"error": "no results"}
    total_thr = sum(r["throughput_tok_s"] for r in results)
    return {"num_workers": world_size, "throughput_tok_s": total_thr,
            "mean_ms": float(np.mean([r["mean_ms"] for r in results]))}


# ---------------------------------------------------------------------------
# DP worker
# ---------------------------------------------------------------------------

def _dp_worker(rank, world_size, hidden_dim, num_experts, N, num_warmup, num_iters, q):
    set_seed(rank)
    model = SimpleMoEFFN(hidden_dim, num_experts)
    n_local = N // world_size
    timings = []
    for it in range(num_warmup + num_iters):
        x = torch.randn(n_local, hidden_dim)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        elapsed = (time.perf_counter() - t0) * 1000
        if it >= num_warmup:
            timings.append(elapsed)
    arr = np.array(timings)
    q.put({"rank": rank, "mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
           "throughput_tok_s": float(n_local / (arr.mean() / 1000))})


def bench_data_parallel(world_size, D, E, N, num_warmup, num_iters):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_dp_worker,
                         args=(r, world_size, D, E, N, num_warmup, num_iters, q))
             for r in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)
    results = []
    while not q.empty():
        results.append(q.get_nowait())
    if not results:
        return {"error": "no results"}
    total_thr = sum(r["throughput_tok_s"] for r in results)
    return {"num_workers": world_size, "throughput_tok_s": total_thr,
            "mean_ms": float(np.mean([r["mean_ms"] for r in results]))}


# ---------------------------------------------------------------------------
# Pipeline parallelism
# ---------------------------------------------------------------------------

def bench_pipeline_parallel(num_stages, D, E, N, num_warmup, num_iters, num_micro_batches=4):
    total_layers = 6
    lpb = total_layers // max(num_stages, 1)

    def make_stage(n):
        return nn.Sequential(*[SimpleMoEFFN(D, E) for _ in range(max(n, 1))])

    stages = [make_stage(lpb) for _ in range(num_stages)]
    micro_size = max(1, N // num_micro_batches)

    def run_gpipe(x_full):
        micros = x_full.split(micro_size, dim=0)
        for stage in stages:
            micros = [stage(m) for m in micros]
        return torch.cat(micros, dim=0)

    bubble_fraction = (num_stages - 1) / (num_micro_batches + num_stages - 1) if num_micro_batches > 0 else 0

    timings = []
    for it in range(num_warmup + num_iters):
        x = torch.randn(N, D)
        t0 = time.perf_counter()
        with torch.no_grad():
            run_gpipe(x)
        elapsed = (time.perf_counter() - t0) * 1000
        if it >= num_warmup:
            timings.append(elapsed)

    arr = np.array(timings)
    return {
        "num_workers": num_stages, "num_micro_batches": num_micro_batches,
        "mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
        "throughput_tok_s": float(N / (arr.mean() / 1000)),
        "bubble_fraction": bubble_fraction,
    }


# ---------------------------------------------------------------------------
# Experiment P1: Strong scaling (EP, DP, PP at N=1,2,4)
# ---------------------------------------------------------------------------

def exp_strong_scaling(D, E, N, num_warmup, num_iters):
    logger.info("=== P1: Strong Scaling ===")
    results = {}

    # Baseline: single GPU or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleMoEFFN(D, E)
    r = bench_single(model, N, D, device, num_warmup, num_iters)
    r["num_workers"] = 1
    results["gpu_single" if device == "cuda" else "cpu_single"] = r
    baseline_thr = r["throughput_tok_s"]
    logger.info(f"  Baseline: {r['throughput_tok_s']:.0f} tok/s")

    for ws in [2, 4]:
        # EP
        key = f"ep{ws}"
        logger.info(f"  Running {key}...")
        r = bench_expert_parallel(ws, D, E, N, num_warmup, num_iters)
        r["speedup"] = r.get("throughput_tok_s", 0) / max(baseline_thr, 1)
        r["efficiency"] = r["speedup"] / ws
        results[key] = r
        logger.info(f"  {key}: {r.get('throughput_tok_s', 0):.0f} tok/s  {r.get('speedup',0):.2f}× efficiency={r.get('efficiency',0):.2f}")

        # DP
        key = f"dp{ws}"
        logger.info(f"  Running {key}...")
        r = bench_data_parallel(ws, D, E, N, num_warmup, num_iters)
        r["speedup"] = r.get("throughput_tok_s", 0) / max(baseline_thr, 1)
        r["efficiency"] = r["speedup"] / ws
        results[key] = r
        logger.info(f"  {key}: {r.get('throughput_tok_s', 0):.0f} tok/s  {r.get('speedup',0):.2f}× efficiency={r.get('efficiency',0):.2f}")

        # PP
        key = f"pp{ws}"
        logger.info(f"  Running {key}...")
        r = bench_pipeline_parallel(ws, D, E, N, num_warmup, num_iters)
        r["speedup"] = r.get("throughput_tok_s", 0) / max(baseline_thr, 1)
        r["efficiency"] = r["speedup"] / ws
        results[key] = r
        logger.info(f"  {key}: {r.get('throughput_tok_s', 0):.0f} tok/s  bubble={r.get('bubble_fraction',0):.2f}")

    return results, baseline_thr


# ---------------------------------------------------------------------------
# Experiment P2: Micro-batch sweep for PP bubble analysis
# ---------------------------------------------------------------------------

def exp_pipeline_bubble_sweep(D, E, N, num_warmup, num_iters):
    logger.info("=== P2: Pipeline Bubble Analysis ===")
    micro_batch_counts = [1, 2, 4, 8, 16]
    results = {}
    for M in micro_batch_counts:
        r = bench_pipeline_parallel(2, D, E, N, num_warmup, num_iters, num_micro_batches=M)
        results[f"M{M}"] = r
        logger.info(f"  M={M}: {r['throughput_tok_s']:.0f} tok/s  bubble={r['bubble_fraction']:.3f}")
    return results


# ---------------------------------------------------------------------------
# Experiment P3: Communication overhead simulation
# ---------------------------------------------------------------------------

def exp_comm_overhead_simulation(D, E, N, num_iters=100):
    logger.info("=== P3: Communication Overhead ===")
    total_params = sum(p.numel() for p in SimpleMoEFFN(D, E).parameters())
    results = {}

    # AllReduce (DP-2): simulate reducing gradient tensor
    timings = []
    for _ in range(num_iters):
        grad = torch.randn(total_params)
        t0 = time.perf_counter()
        _ = grad.sum()
        timings.append((time.perf_counter() - t0) * 1e6)  # microseconds
    arr = np.array(timings)
    results["allreduce_dp2_sim"] = {
        "mean_us": float(arr.mean()), "p50_us": float(np.percentile(arr, 50)),
        "description": f"AllReduce {total_params/1e6:.1f}M params (DP-2 simulation)"
    }

    # All-to-All (EP-2): simulate token redistribution
    timings = []
    for _ in range(num_iters):
        tokens = torch.randn(N // 2, D)
        t0 = time.perf_counter()
        _ = tokens.clone()
        timings.append((time.perf_counter() - t0) * 1e6)
    arr = np.array(timings)
    results["all_to_all_ep2_sim"] = {
        "mean_us": float(arr.mean()), "p50_us": float(np.percentile(arr, 50)),
        "description": f"All-to-All {N//2} tokens, D={D} (EP-2 simulation)"
    }

    # P2P: pipeline stage transfer
    timings = []
    micro_size = N // 4
    for _ in range(num_iters):
        activation = torch.randn(micro_size, D)
        t0 = time.perf_counter()
        _ = activation.clone()
        timings.append((time.perf_counter() - t0) * 1e6)
    arr = np.array(timings)
    results["p2p_pp2_sim"] = {
        "mean_us": float(arr.mean()), "p50_us": float(np.percentile(arr, 50)),
        "description": f"P2P {micro_size} activation tokens (PP-2 simulation)"
    }

    for k, v in results.items():
        logger.info(f"  {k}: {v['mean_us']:.1f}us avg")
    return results


# ---------------------------------------------------------------------------
# Experiment P4: Token imbalance vs EP efficiency
# ---------------------------------------------------------------------------

def exp_ep_vs_imbalance(D, E, N, num_warmup, num_iters):
    logger.info("=== P4: EP Efficiency vs Load Imbalance ===")
    imbalances = [0.0, 0.2, 0.4, 0.6, 0.8]
    results = {}
    # Baseline: single process
    model = SimpleMoEFFN(D, E)
    baseline = bench_single(model, N, D, "cpu", num_warmup, num_iters)
    results["baseline_cpu"] = baseline

    for imb in imbalances:
        key = f"ep2_imb{int(imb*100)}"
        r = bench_expert_parallel(2, D, E, N, num_warmup, num_iters, load_imbalance=imb)
        r["imbalance"] = imb
        r["speedup_vs_cpu"] = r.get("throughput_tok_s", 0) / max(baseline["throughput_tok_s"], 1)
        results[key] = r
        logger.info(f"  EP2 imb={imb:.0%}: {r.get('throughput_tok_s', 0):.0f} tok/s  speedup={r.get('speedup_vs_cpu',0):.2f}×")
    return results


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_report(all_exp, baseline_thr):
    lines = ["# Extended Parallelism Scaling Results", ""]

    p1 = all_exp.get("P1_strong_scaling", {})
    if p1:
        lines += ["## P1: Strong Scaling Efficiency", "",
                  "| Mode | Throughput (tok/s) | Workers | Speedup | Efficiency |",
                  "|------|--------------------|---------|---------|-----------|"]
        for mode, r in p1.items():
            thr = r.get("throughput_tok_s", 0)
            ws = r.get("num_workers", 1)
            spd = r.get("speedup", thr / max(baseline_thr, 1))
            eff = r.get("efficiency", spd / max(ws, 1))
            lines.append(f"| {mode} | {thr:,.0f} | {ws} | {spd:.2f}× | {eff:.2f} |")
        lines.append("")

    p2 = all_exp.get("P2_pipeline_bubble", {})
    if p2:
        lines += ["## P2: Pipeline Bubble Fraction (PP-2, varying micro-batches)", "",
                  "| Micro-batches | Throughput (tok/s) | Bubble Fraction |",
                  "|--------------|-------------------|----------------|"]
        for k, r in p2.items():
            lines.append(f"| {r.get('num_micro_batches','?')} | {r.get('throughput_tok_s',0):,.0f} | {r.get('bubble_fraction',0):.3f} |")
        lines.append("")

    p3 = all_exp.get("P3_comm_overhead", {})
    if p3:
        lines += ["## P3: Communication Overhead Simulation", "",
                  "| Operation | Mean (us) | p50 (us) | Description |",
                  "|----------|----------|---------|------------|"]
        for k, r in p3.items():
            lines.append(f"| {k} | {r.get('mean_us',0):.1f} | {r.get('p50_us',0):.1f} | {r.get('description','')} |")
        lines.append("")

    p4 = all_exp.get("P4_ep_vs_imbalance", {})
    if p4:
        lines += ["## P4: EP-2 Efficiency vs Load Imbalance", "",
                  "| Imbalance | EP2 Throughput (tok/s) | Speedup vs CPU |",
                  "|----------|----------------------|---------------|"]
        baseline_thr_p4 = p4.get("baseline_cpu", {}).get("throughput_tok_s", 1)
        for k, r in p4.items():
            if k == "baseline_cpu":
                continue
            imb = r.get("imbalance", 0)
            thr = r.get("throughput_tok_s", 0)
            spd = r.get("speedup_vs_cpu", thr / max(baseline_thr_p4, 1))
            lines.append(f"| {imb:.0%} | {thr:,.0f} | {spd:.2f}× |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_warmup", type=int, default=20)
    p.add_argument("--num_iters", type=int, default=100)
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_tokens", type=int, default=196)
    p.add_argument("--output_dir", type=str, default="outputs/extended_parallelism")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "run.log")],
    )

    D, E, N = args.hidden_dim, args.num_experts, args.batch_size * args.num_tokens
    logger.info(f"Config: D={D}, E={E}, N={N}")

    all_exp = {}

    p1_results, baseline_thr = exp_strong_scaling(D, E, N, args.num_warmup, args.num_iters)
    all_exp["P1_strong_scaling"] = p1_results

    all_exp["P2_pipeline_bubble"] = exp_pipeline_bubble_sweep(D, E, N, args.num_warmup, args.num_iters)
    all_exp["P3_comm_overhead"] = exp_comm_overhead_simulation(D, E, N, args.num_iters)
    all_exp["P4_ep_vs_imbalance"] = exp_ep_vs_imbalance(D, E, N, args.num_warmup, args.num_iters)

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_exp, f, indent=2)

    report = build_report(all_exp, baseline_thr)
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"\nResults saved to: {out_dir}")
    print(report)


if __name__ == "__main__":
    main()
