#!/usr/bin/env python3
"""
Script 27 — Extended CPU vs GPU Experiments

New experiments beyond script 10:
  E1. Scaling with hidden_dim: 192 (ViT-Ti), 384 (ViT-S), 768 (ViT-B)
  E2. Scaling with expert count: 2, 4, 8, 16
  E3. Load imbalance sweep: 0%, 20%, 40%, 60%, 80%
  E4. FP16 vs FP32 on GPU (best dispatch strategy)
  E5. Batch size impact: 1, 4, 8, 16, 32 tokens
  E6. Memory bandwidth stress: large hidden_dim=1536 (ViT-L-like)

Outputs: outputs/extended_cpu_gpu/<timestamp>/
  - results.json
  - summary.md
"""

import argparse
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
from clear_moe.utils import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal MoE FFN
# ---------------------------------------------------------------------------

class SimpleMoEFFN(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, intermediate_dim: int = None):
        super().__init__()
        I = intermediate_dim or hidden_dim * 4
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.fc1s = nn.ModuleList([nn.Linear(hidden_dim, I) for _ in range(num_experts)])
        self.fc2s = nn.ModuleList([nn.Linear(I, hidden_dim) for _ in range(num_experts)])
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, load_imbalance: float = 0.0) -> torch.Tensor:
        N, D = x.shape
        if load_imbalance > 0.0:
            # Force fraction of tokens to expert 0
            n_dom = int(N * load_imbalance)
            ids = torch.randint(1, len(self.fc1s), (N,), device=x.device)
            ids[:n_dom] = 0
            ids = ids[torch.randperm(N, device=x.device)]
        else:
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
# Timing utilities
# ---------------------------------------------------------------------------

def bench_model(model, N, hidden_dim, device, num_warmup, num_iters,
                dtype=torch.float32, load_imbalance=0.0):
    model = model.to(device=device, dtype=dtype)
    timings = []
    for _ in range(num_warmup):
        x = torch.randn(N, hidden_dim, device=device, dtype=dtype)
        with torch.no_grad():
            model(x, load_imbalance)
        if device != "cpu":
            torch.cuda.synchronize()

    for _ in range(num_iters):
        x = torch.randn(N, hidden_dim, device=device, dtype=dtype)
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
        "std_ms": float(arr.std()),
        "throughput_tok_s": float(N / (arr.mean() / 1000)),
    }


# ---------------------------------------------------------------------------
# Experiment E1: Vary hidden_dim (CPU vs GPU)
# ---------------------------------------------------------------------------

def exp_hidden_dim_scaling(num_warmup=20, num_iters=100):
    logger.info("=== E1: Hidden-Dim Scaling ===")
    dims = [192, 384, 768, 1536]
    N, E = 8 * 196, 4
    results = {}
    for D in dims:
        key = f"D{D}"
        model = SimpleMoEFFN(D, E)
        cpu_r = bench_model(model, N, D, "cpu", num_warmup, num_iters)
        cpu_r["device"] = "cpu"
        row = {"cpu": cpu_r}
        if torch.cuda.is_available():
            gpu_r = bench_model(model, N, D, "cuda", num_warmup, num_iters)
            gpu_r["device"] = "cuda"
            gpu_r["speedup_vs_cpu"] = cpu_r["mean_ms"] / max(gpu_r["mean_ms"], 1e-6)
            row["gpu"] = gpu_r
        results[key] = row
        logger.info(f"  D={D}: CPU={cpu_r['mean_ms']:.2f}ms  GPU={row.get('gpu', {}).get('mean_ms', 'N/A')}")
    return results


# ---------------------------------------------------------------------------
# Experiment E2: Vary expert count
# ---------------------------------------------------------------------------

def exp_expert_count_scaling(num_warmup=20, num_iters=100):
    logger.info("=== E2: Expert Count Scaling ===")
    expert_counts = [1, 2, 4, 8, 16]
    N, D = 8 * 196, 384
    results = {}
    for E in expert_counts:
        key = f"E{E}"
        model = SimpleMoEFFN(D, E)
        cpu_r = bench_model(model, N, D, "cpu", num_warmup, num_iters)
        cpu_r["device"] = "cpu"
        row = {"cpu": cpu_r}
        if torch.cuda.is_available():
            gpu_r = bench_model(model, N, D, "cuda", num_warmup, num_iters)
            gpu_r["device"] = "cuda"
            gpu_r["speedup_vs_cpu"] = cpu_r["mean_ms"] / max(gpu_r["mean_ms"], 1e-6)
            row["gpu"] = gpu_r
        results[key] = row
        logger.info(f"  E={E}: CPU={cpu_r['mean_ms']:.2f}ms  GPU={row.get('gpu', {}).get('mean_ms', 'N/A')}")
    return results


# ---------------------------------------------------------------------------
# Experiment E3: Load imbalance sweep
# ---------------------------------------------------------------------------

def exp_imbalance_sweep(num_warmup=20, num_iters=100):
    logger.info("=== E3: Load Imbalance Sweep ===")
    imbalance_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
    N, D, E = 8 * 196, 384, 4
    results = {}
    for imb in imbalance_levels:
        key = f"imb{int(imb*100)}"
        model = SimpleMoEFFN(D, E)
        cpu_r = bench_model(model, N, D, "cpu", num_warmup, num_iters, load_imbalance=imb)
        cpu_r["device"] = "cpu"
        row = {"cpu": cpu_r, "imbalance": imb}
        if torch.cuda.is_available():
            gpu_r = bench_model(model, N, D, "cuda", num_warmup, num_iters, load_imbalance=imb)
            gpu_r["device"] = "cuda"
            gpu_r["speedup_vs_cpu"] = cpu_r["mean_ms"] / max(gpu_r["mean_ms"], 1e-6)
            row["gpu"] = gpu_r
        results[key] = row
        logger.info(f"  imbalance={imb:.0%}: CPU={cpu_r['mean_ms']:.2f}ms  GPU={row.get('gpu', {}).get('mean_ms', 'N/A')}")
    return results


# ---------------------------------------------------------------------------
# Experiment E4: FP16 vs FP32 on GPU
# ---------------------------------------------------------------------------

def exp_fp16_vs_fp32(num_warmup=20, num_iters=100):
    logger.info("=== E4: FP16 vs FP32 on GPU ===")
    if not torch.cuda.is_available():
        logger.warning("  CUDA not available — skipping")
        return {}
    N, D, E = 8 * 196, 384, 4
    results = {}
    model = SimpleMoEFFN(D, E)
    for dtype, label in [(torch.float32, "fp32"), (torch.float16, "fp16")]:
        try:
            r = bench_model(model, N, D, "cuda", num_warmup, num_iters, dtype=dtype)
            r["dtype"] = label
            results[label] = r
            logger.info(f"  {label}: {r['mean_ms']:.2f}ms  {r['throughput_tok_s']:.0f} tok/s")
        except Exception as exc:
            logger.warning(f"  {label} failed: {exc}")
            results[label] = {"error": str(exc)}
    if "fp32" in results and "fp16" in results and "mean_ms" in results["fp16"]:
        results["fp16_speedup"] = results["fp32"]["mean_ms"] / max(results["fp16"]["mean_ms"], 1e-6)
    return results


# ---------------------------------------------------------------------------
# Experiment E5: Batch-size impact
# ---------------------------------------------------------------------------

def exp_batch_size(num_warmup=20, num_iters=100):
    logger.info("=== E5: Batch Size Impact ===")
    batch_sizes = [1, 4, 8, 16, 32]
    D, E, tokens_per_image = 384, 4, 196
    results = {}
    model = SimpleMoEFFN(D, E)
    for B in batch_sizes:
        N = B * tokens_per_image
        key = f"B{B}"
        cpu_r = bench_model(model, N, D, "cpu", num_warmup, num_iters)
        cpu_r["device"] = "cpu"
        cpu_r["batch_size"] = B
        row = {"cpu": cpu_r}
        if torch.cuda.is_available():
            gpu_r = bench_model(model, N, D, "cuda", num_warmup, num_iters)
            gpu_r["device"] = "cuda"
            gpu_r["batch_size"] = B
            gpu_r["speedup_vs_cpu"] = cpu_r["mean_ms"] / max(gpu_r["mean_ms"], 1e-6)
            row["gpu"] = gpu_r
        results[key] = row
        logger.info(f"  B={B}: CPU={cpu_r['mean_ms']:.2f}ms  GPU={row.get('gpu', {}).get('mean_ms', 'N/A')}")
    return results


# ---------------------------------------------------------------------------
# Experiment E6: CPU thread count scaling
# ---------------------------------------------------------------------------

def exp_cpu_thread_scaling(num_warmup=20, num_iters=100):
    logger.info("=== E6: CPU Thread Scaling ===")
    import multiprocessing
    max_threads = multiprocessing.cpu_count()
    thread_counts = [t for t in [1, 2, 4, 8, 16] if t <= max_threads]
    N, D, E = 8 * 196, 384, 4
    results = {}
    model = SimpleMoEFFN(D, E)
    for t in thread_counts:
        torch.set_num_threads(t)
        r = bench_model(model, N, D, "cpu", num_warmup, num_iters)
        r["num_threads"] = t
        results[f"cpu_t{t}"] = r
        logger.info(f"  CPU threads={t}: {r['mean_ms']:.2f}ms  {r['throughput_tok_s']:.0f} tok/s")
    return results


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def build_markdown_report(all_exp: dict, gpu_name: str) -> str:
    lines = [
        "# Extended CPU vs GPU Experiment Results",
        "",
        f"**GPU**: {gpu_name}",
        "",
    ]

    # E1
    e1 = all_exp.get("E1_hidden_dim", {})
    if e1:
        lines += ["## E1: Hidden-Dim Scaling (CPU vs GPU)", "",
                  "| Hidden Dim | CPU p50 (ms) | GPU p50 (ms) | GPU Speedup |",
                  "|-----------|-------------|-------------|------------|"]
        for k, v in e1.items():
            cpu_p50 = v.get("cpu", {}).get("p50_ms", 0)
            gpu_p50 = v.get("gpu", {}).get("p50_ms", "—")
            spd = v.get("gpu", {}).get("speedup_vs_cpu", "—")
            spd_str = f"{spd:.2f}×" if isinstance(spd, float) else "—"
            gpu_p50_str = f"{gpu_p50:.2f}" if isinstance(gpu_p50, float) else "—"
            lines.append(f"| {k} | {cpu_p50:.2f} | {gpu_p50_str} | {spd_str} |")
        lines.append("")

    # E2
    e2 = all_exp.get("E2_expert_count", {})
    if e2:
        lines += ["## E2: Expert Count Scaling (CPU vs GPU)", "",
                  "| Experts | CPU p50 (ms) | GPU p50 (ms) | GPU Speedup |",
                  "|--------|-------------|-------------|------------|"]
        for k, v in e2.items():
            cpu_p50 = v.get("cpu", {}).get("p50_ms", 0)
            gpu_p50 = v.get("gpu", {}).get("p50_ms", "—")
            spd = v.get("gpu", {}).get("speedup_vs_cpu", "—")
            spd_str = f"{spd:.2f}×" if isinstance(spd, float) else "—"
            gpu_p50_str = f"{gpu_p50:.2f}" if isinstance(gpu_p50, float) else "—"
            lines.append(f"| {k} | {cpu_p50:.2f} | {gpu_p50_str} | {spd_str} |")
        lines.append("")

    # E3
    e3 = all_exp.get("E3_imbalance_sweep", {})
    if e3:
        lines += ["## E3: Load Imbalance Sweep (CPU vs GPU)", "",
                  "| Imbalance | CPU p50 (ms) | GPU p50 (ms) | GPU Speedup |",
                  "|----------|-------------|-------------|------------|"]
        for k, v in e3.items():
            imb = v.get("imbalance", 0)
            cpu_p50 = v.get("cpu", {}).get("p50_ms", 0)
            gpu_p50 = v.get("gpu", {}).get("p50_ms", "—")
            spd = v.get("gpu", {}).get("speedup_vs_cpu", "—")
            spd_str = f"{spd:.2f}×" if isinstance(spd, float) else "—"
            gpu_p50_str = f"{gpu_p50:.2f}" if isinstance(gpu_p50, float) else "—"
            lines.append(f"| {imb:.0%} | {cpu_p50:.2f} | {gpu_p50_str} | {spd_str} |")
        lines.append("")

    # E4
    e4 = all_exp.get("E4_fp16_fp32", {})
    if e4 and "fp32" in e4 and "fp16" in e4:
        fp32_ms = e4["fp32"].get("p50_ms", 0)
        fp16_ms = e4["fp16"].get("p50_ms", 0)
        fp16_spd = e4.get("fp16_speedup", 0)
        lines += ["## E4: FP16 vs FP32 on GPU", "",
                  "| Dtype | p50 (ms) | Throughput (tok/s) | FP16 Speedup |",
                  "|------|---------|-------------------|-------------|",
                  f"| FP32 | {fp32_ms:.2f} | {e4['fp32'].get('throughput_tok_s', 0):.0f} | — |",
                  f"| FP16 | {fp16_ms:.2f} | {e4['fp16'].get('throughput_tok_s', 0):.0f} | {fp16_spd:.2f}× |",
                  ""]

    # E5
    e5 = all_exp.get("E5_batch_size", {})
    if e5:
        lines += ["## E5: Batch Size Impact (CPU vs GPU)", "",
                  "| Batch | CPU p50 (ms) | GPU p50 (ms) | GPU Speedup |",
                  "|------|-------------|-------------|------------|"]
        for k, v in e5.items():
            B = v.get("cpu", {}).get("batch_size", k)
            cpu_p50 = v.get("cpu", {}).get("p50_ms", 0)
            gpu_p50 = v.get("gpu", {}).get("p50_ms", "—")
            spd = v.get("gpu", {}).get("speedup_vs_cpu", "—")
            spd_str = f"{spd:.2f}×" if isinstance(spd, float) else "—"
            gpu_p50_str = f"{gpu_p50:.2f}" if isinstance(gpu_p50, float) else "—"
            lines.append(f"| {B} | {cpu_p50:.2f} | {gpu_p50_str} | {spd_str} |")
        lines.append("")

    # E6
    e6 = all_exp.get("E6_cpu_threads", {})
    if e6:
        lines += ["## E6: CPU Thread Scaling", "",
                  "| Threads | p50 (ms) | Throughput (tok/s) |",
                  "|--------|---------|-------------------|"]
        for k, v in e6.items():
            t = v.get("num_threads", k)
            lines.append(f"| {t} | {v.get('p50_ms', 0):.2f} | {v.get('throughput_tok_s', 0):.0f} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_warmup", type=int, default=20)
    p.add_argument("--num_iters", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="outputs/extended_cpu_gpu")
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

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    logger.info(f"GPU: {gpu_name}")

    all_exp = {}
    all_exp["E1_hidden_dim"] = exp_hidden_dim_scaling(args.num_warmup, args.num_iters)
    all_exp["E2_expert_count"] = exp_expert_count_scaling(args.num_warmup, args.num_iters)
    all_exp["E3_imbalance_sweep"] = exp_imbalance_sweep(args.num_warmup, args.num_iters)
    all_exp["E4_fp16_fp32"] = exp_fp16_vs_fp32(args.num_warmup, args.num_iters)
    all_exp["E5_batch_size"] = exp_batch_size(args.num_warmup, args.num_iters)
    all_exp["E6_cpu_threads"] = exp_cpu_thread_scaling(args.num_warmup, args.num_iters)

    with open(out_dir / "results.json", "w") as f:
        json.dump({"gpu": gpu_name, "experiments": all_exp}, f, indent=2)

    report = build_markdown_report(all_exp, gpu_name)
    with open(out_dir / "summary.md", "w") as f:
        f.write(report)

    logger.info(f"\nResults saved to: {out_dir}")
    print(report)


if __name__ == "__main__":
    main()
