"""
Experiment 6: End-to-End Dispatch Benchmark

Measures p50, p95, mean, std latency for each dispatch backend
across four token-load imbalance scenarios and multiple batch sizes.

Measurement protocol (per plan Section 3, Experiment 6):
  - At least 20 warm-up iterations
  - At least 100 timed iterations
  - torch.cuda.synchronize() before and after timed section
  - Record p50, p95, mean, std
  - Measure full model and dispatch-only latency separately
  - Measure peak VRAM
  - Record actual token counts and padding overhead

Run:
  python experiments/run_dispatch_benchmark.py
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.experiment_logger import ExperimentLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

WARMUP = 20
ITERATIONS = 100


def make_routing(
    total_tokens: int,
    num_experts: int,
    imbalance_frac: float,
    seed: int = 42,
) -> torch.Tensor:
    """
    Create a token→expert assignment tensor with controlled imbalance.

    Args:
        total_tokens: Total number of tokens
        num_experts: Number of experts
        imbalance_frac: Fraction of tokens routed to expert 0 (0.0 = perfectly balanced)
        seed: Random seed

    Returns:
        (total_tokens,) long tensor of expert assignments
    """
    rng = np.random.RandomState(seed)
    assignments = np.zeros(total_tokens, dtype=np.int64)

    if imbalance_frac <= 0.0:
        # Perfectly balanced
        for i in range(total_tokens):
            assignments[i] = i % num_experts
    else:
        n_hot = int(total_tokens * imbalance_frac)
        n_hot = min(n_hot, total_tokens)
        assignments[:n_hot] = 0  # Expert 0 gets the overloaded share
        remaining = total_tokens - n_hot
        if remaining > 0:
            # Distribute rest evenly among experts 1..E-1
            remaining_experts = list(range(1, num_experts))
            for i in range(remaining):
                assignments[n_hot + i] = remaining_experts[i % len(remaining_experts)]

    rng.shuffle(assignments)
    return torch.from_numpy(assignments)


def dispatch_naive(
    tokens: torch.Tensor,
    assignments: torch.Tensor,
    expert_weights: List[torch.Tensor],
) -> torch.Tensor:
    """Naive: sequential loop with boolean mask."""
    out = torch.zeros(tokens.shape[0], expert_weights[0].shape[0], device=tokens.device)
    for e, W in enumerate(expert_weights):
        mask = (assignments == e)
        if mask.any():
            out[mask] = tokens[mask] @ W.T
    return out


def dispatch_grouped(
    tokens: torch.Tensor,
    assignments: torch.Tensor,
    expert_weights: List[torch.Tensor],
) -> torch.Tensor:
    """Grouped (route-sorted): sort tokens by expert, GEMM per contiguous block."""
    num_experts = len(expert_weights)
    sorted_order = torch.argsort(assignments)
    sorted_tokens = tokens[sorted_order]
    sorted_assigns = assignments[sorted_order]

    boundaries = torch.searchsorted(sorted_assigns, torch.arange(num_experts, device=tokens.device))
    endings = torch.cat([boundaries[1:], torch.tensor([tokens.shape[0]], device=tokens.device)])

    out = torch.empty(tokens.shape[0], expert_weights[0].shape[0], device=tokens.device)
    for e, W in enumerate(expert_weights):
        start, end = boundaries[e].item(), endings[e].item()
        if start < end:
            out[sorted_order[start:end]] = sorted_tokens[start:end] @ W.T
    return out


def dispatch_cublas(
    tokens: torch.Tensor,
    assignments: torch.Tensor,
    expert_weights: List[torch.Tensor],
) -> torch.Tensor:
    """cuBLAS batched-GEMM: pad all sub-batches to uniform size."""
    num_experts = len(expert_weights)
    d_out = expert_weights[0].shape[0]
    d_in = tokens.shape[1]

    # Gather per-expert token groups
    expert_tokens = []
    expert_masks = []
    max_size = 0
    for e in range(num_experts):
        mask = (assignments == e).nonzero(as_tuple=True)[0]
        expert_masks.append(mask)
        expert_tokens.append(tokens[mask] if mask.shape[0] > 0 else torch.zeros(0, d_in, device=tokens.device))
        max_size = max(max_size, mask.shape[0])

    if max_size == 0:
        return torch.zeros(tokens.shape[0], d_out, device=tokens.device)

    # Pad to max_size (padding overhead = (max_size - actual_size) * d_in * E)
    padded = torch.zeros(num_experts, max_size, d_in, device=tokens.device)
    for e in range(num_experts):
        n = expert_tokens[e].shape[0]
        if n > 0:
            padded[e, :n, :] = expert_tokens[e]

    W_stack = torch.stack(expert_weights, dim=0)  # (E, d_out, d_in)
    # bmm expects (E, max_size, d_in) @ (E, d_in, d_out) -> (E, max_size, d_out)
    results = torch.bmm(padded, W_stack.permute(0, 2, 1))  # (E, max_size, d_out)

    out = torch.zeros(tokens.shape[0], d_out, device=tokens.device)
    for e in range(num_experts):
        n = expert_tokens[e].shape[0]
        if n > 0:
            out[expert_masks[e]] = results[e, :n, :]
    return out


BACKENDS = {
    "naive": dispatch_naive,
    "grouped": dispatch_grouped,
    "cublas": dispatch_cublas,
}

IMBALANCE_LEVELS = [0.0, 0.4, 0.6, 0.8]
BATCH_SIZES = [1, 2, 4, 8, 16]
TOKENS_PER_IMAGE = 196
HIDDEN_DIM = 384


def benchmark_backend(
    backend_fn,
    tokens: torch.Tensor,
    assignments: torch.Tensor,
    expert_weights: List[torch.Tensor],
    warmup: int = WARMUP,
    iterations: int = ITERATIONS,
) -> Dict:
    """Run warmup + timed iterations for one backend configuration."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        _ = backend_fn(tokens, assignments, expert_weights)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = backend_fn(tokens, assignments, expert_weights)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "tokens_per_second": float(tokens.shape[0] / (np.percentile(arr, 50) / 1000.0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Dispatch Backend Benchmark")
    parser.add_argument("--output_dir", default="outputs/dispatch_benchmark")
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(BACKENDS.keys()),
        choices=list(BACKENDS.keys()),
    )
    parser.add_argument(
        "--batch_sizes",
        nargs="+",
        type=int,
        default=BATCH_SIZES,
    )
    parser.add_argument(
        "--imbalance_levels",
        nargs="+",
        type=float,
        default=IMBALANCE_LEVELS,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    exp_logger = ExperimentLogger(args.output_dir)

    # Create fixed expert weight matrices
    torch.manual_seed(args.seed)
    expert_weights = [
        torch.randn(args.hidden_dim, args.hidden_dim, device=device) * 0.02
        for _ in range(args.num_experts)
    ]

    all_results = []

    for batch_size in args.batch_sizes:
        total_tokens = batch_size * TOKENS_PER_IMAGE

        for imbalance in args.imbalance_levels:
            assignments = make_routing(total_tokens, args.num_experts, imbalance, seed=args.seed).to(device)
            tokens = torch.randn(total_tokens, args.hidden_dim, device=device)

            # Log actual expert loads
            expert_loads = [(assignments == e).sum().item() for e in range(args.num_experts)]
            load_skew = max(expert_loads) / (total_tokens / args.num_experts + 1e-8)

            logger.info(
                f"B={batch_size}, T={total_tokens}, imbalance={imbalance:.0%}: "
                f"loads={expert_loads}, skew={load_skew:.2f}"
            )

            for backend_name in args.backends:
                backend_fn = BACKENDS[backend_name]

                try:
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats(device)

                    bench = benchmark_backend(backend_fn, tokens, assignments, expert_weights)

                    peak_vram = -1.0
                    if torch.cuda.is_available():
                        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

                    result = {
                        "backend": backend_name,
                        "batch_size": batch_size,
                        "total_tokens": total_tokens,
                        "imbalance_frac": imbalance,
                        "expert_loads": str(expert_loads),
                        "load_skew": load_skew,
                        "peak_vram_mb": peak_vram,
                        **bench,
                    }

                    exp_cfg = {
                        "backbone": "dispatch_microbench",
                        "dataset": "synthetic",
                        "calibration_size": 0,
                        "selected_layers": "all",
                        "expert_count": args.num_experts,
                        "basis_rank": -1,
                        "router": "fixed",
                        "dispatch_backend": backend_name,
                        "batch_size": batch_size,
                        "seed": args.seed,
                        "imbalance_frac": imbalance,
                        "total_tokens": total_tokens,
                        "load_skew": load_skew,
                    }
                    exp_results = {
                        "top1": -1, "top5": -1,
                        "router_accuracy": -1,
                        "load_skew": load_skew,
                        **bench,
                    }
                    exp_id = exp_logger.log(
                        exp_cfg, exp_results,
                        experiment_id=f"dispatch_{backend_name}_B{batch_size}_imb{int(imbalance*100)}"
                    )

                    all_results.append(result)
                    logger.info(
                        f"  {backend_name:20s} B={batch_size} imb={imbalance:.0%}: "
                        f"p50={bench['p50_ms']:.2f}ms p95={bench['p95_ms']:.2f}ms "
                        f"tok/s={bench['tokens_per_second']:.0f}"
                    )

                except Exception as e:
                    logger.error(f"  {backend_name} B={batch_size} imb={imbalance:.0%} FAILED: {e}")

    # Save full results
    out_path = Path(args.output_dir) / "dispatch_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"All dispatch results saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Backend':<22} {'B':>3} {'Imb':>5} {'p50 ms':>8} {'p95 ms':>8} {'tok/s':>12} {'skew':>6}")
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['backend']:<22} {r['batch_size']:>3} {r['imbalance_frac']:>5.0%} "
            f"{r['p50_ms']:>8.2f} {r['p95_ms']:>8.2f} "
            f"{r['tokens_per_second']:>12.0f} {r['load_skew']:>6.2f}"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
