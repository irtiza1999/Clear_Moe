"""Script 26: CLEAR-MoE++ Novelty Comparison.

Benchmarks three novel contributions vs. their direct baselines:
  N1. ALFRouter (DeepSeek-V3 inspired) vs. LinearRouter
      Gap: DeepSeek-V3 auxiliary-loss-free balancing never applied to
      post-training extracted vision MoE — aux-loss distorts pre-extracted
      expert specialization structure.

  N2. stream_fine (Comet inspired) vs. stream backend
      Gap: Comet fine-grained compute-communication overlap not applied
      to inference-only extracted MoE (only training in original paper).

  N3. DisaggregatedSim (MegaScale-Infer inspired) vs. serial execution
      Gap: MegaScale-Infer disaggregated attention-expert pipeline only
      demonstrated at cluster scale; single-GPU simulation missing.

Outputs: JSON + Markdown table to outputs/novelty_comparison/<timestamp>/

Usage:
    python scripts/26_novelty_comparison.py [--dry-run]
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from clear_moe.routers.linear import LinearRouter
from clear_moe.routers.alf_router import ALFRouter
from clear_moe.runtime.executor import MoEExecutor
from clear_moe.runtime.disaggregated import DisaggregatedSim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def time_fn(fn, warmup: int = 3, iters: int = 20) -> float:
    """Median wall-clock time for fn() in milliseconds."""
    for _ in range(warmup):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def make_experts(E: int, D: int) -> List[nn.Module]:
    return [nn.Linear(D, D, bias=False).to(DEVICE) for _ in range(E)]


# ─── N1: ALFRouter vs LinearRouter ────────────────────────────────────────────

def bench_n1(dry_run: bool = False) -> Dict:
    D, E = 384, 4
    B, S = (2, 8) if dry_run else (8, 196)
    N_passes = 5 if dry_run else 50
    iters = 5 if dry_run else 30

    # Build both routers with same gate initialisation
    torch.manual_seed(7)
    linear_r = LinearRouter(D, E, top_k=1).to(DEVICE)
    alf_r = ALFRouter(D, E, top_k=1, bias_update_rate=0.01).to(DEVICE)

    # Skew: force gate so expert 0 dominates
    with torch.no_grad():
        linear_r.gate.weight.fill_(0.0)
        linear_r.gate.weight[0, :] = 2.0
        alf_r.gate.weight.fill_(0.0)
        alf_r.gate.weight[0, :] = 2.0

    x_skewed = torch.randn(B, S, D, device=DEVICE)
    x_skewed[:, :, 0] += 3.0  # push tokens toward expert-0 direction

    def hotness_skew(idx: torch.Tensor) -> float:
        counts = torch.zeros(E)
        flat = idx.reshape(-1).cpu()
        for e in range(E):
            counts[e] = (flat == e).sum().float()
        return (counts.max() / counts.mean().clamp(min=1e-6)).item()

    # Simulate N forward passes; record final hotness skew
    for _ in range(N_passes):
        idx_l, _, _, _ = linear_r(x_skewed)
        idx_a, _, _, _ = alf_r(x_skewed)
    linear_skew = hotness_skew(idx_l)
    alf_skew = hotness_skew(idx_a)

    t_linear = time_fn(lambda: linear_r(x_skewed), warmup=2, iters=iters)
    t_alf = time_fn(lambda: alf_r(x_skewed), warmup=2, iters=iters)

    return {
        "component": "N1_ALFRouter",
        "baseline_name": "LinearRouter (no load balancing)",
        "novel_name": "ALFRouter (bias-correction, no aux loss)",
        "paper_gap": "DeepSeek-V3 ALF balancing never applied to post-training extracted vision MoE",
        "baseline_hotness_skew": round(linear_skew, 3),
        "novel_hotness_skew": round(alf_skew, 3),
        "skew_reduction_pct": round(
            100.0 * (linear_skew - alf_skew) / max(linear_skew, 1e-6), 2),
        "baseline_latency_ms": round(t_linear, 4),
        "novel_latency_ms": round(t_alf, 4),
        "latency_overhead_pct": round(
            100.0 * (t_alf - t_linear) / max(t_linear, 1e-6), 2),
    }


# ─── N2: stream_fine vs stream backend ────────────────────────────────────────

def bench_n2(dry_run: bool = False) -> Dict:
    E, D = 4, 384
    B, S = (2, 8) if dry_run else (8, 196)
    iters = 5 if dry_run else 30

    torch.manual_seed(7)
    x = torch.randn(B, S, D, device=DEVICE)
    indices = torch.randint(0, E, (B, S, 1), device=DEVICE)
    weights = torch.ones(B, S, 1, device=DEVICE)
    experts = make_experts(E, D)

    exec_grouped = MoEExecutor(backend="grouped", num_experts=E)
    exec_fine = MoEExecutor(backend="stream_fine", num_experts=E)

    t_grouped = time_fn(
        lambda: exec_grouped._run_grouped(x, indices, weights, experts),
        warmup=2, iters=iters,
    )
    t_fine = time_fn(
        lambda: exec_fine._run_stream_fine(x, indices, weights, experts),
        warmup=2, iters=iters,
    )

    return {
        "component": "N2_StreamFine",
        "baseline_name": "grouped dispatch (serial expert loop)",
        "novel_name": "stream_fine (per-expert CUDA stream + dependency-ordered combine)",
        "paper_gap": "Comet fine-grained overlap not applied to inference-only extracted MoE",
        "baseline_latency_ms": round(t_grouped, 4),
        "novel_latency_ms": round(t_fine, 4),
        "speedup": round(t_grouped / max(t_fine, 1e-6), 3),
    }


# ─── N3: DisaggregatedSim vs serial ───────────────────────────────────────────

def bench_n3(dry_run: bool = False) -> Dict:
    D, N_blocks = 384, 4
    B = 4 if dry_run else 16
    S = 4 if dry_run else 64
    M = 2 if dry_run else 8
    iters = 5 if dry_run else 20

    torch.manual_seed(7)
    x = torch.randn(B, S, D, device=DEVICE)
    attn = [nn.Linear(D, D, bias=False).to(DEVICE) for _ in range(N_blocks)]
    ffn = [nn.Linear(D, D, bias=False).to(DEVICE) for _ in range(N_blocks)]

    sim = DisaggregatedSim(attn, ffn, num_micro_batches=M)

    t_serial = time_fn(lambda: sim.forward_serial(x), warmup=2, iters=iters)
    t_pipe = time_fn(lambda: sim.forward(x), warmup=2, iters=iters)
    _, stats = sim.forward(x)

    return {
        "component": "N3_DisaggregatedSim",
        "baseline_name": "serial (attention then expert, no overlap)",
        "novel_name": f"DisaggregatedSim (micro-batch pipeline M={M})",
        "paper_gap": "MegaScale-Infer disaggregated EP not simulated on single GPU",
        "baseline_latency_ms": round(t_serial, 4),
        "novel_latency_ms": round(t_pipe, 4),
        "bubble_ratio": round(stats["bubble_ratio"], 4),
        "overlap_efficiency": round(stats["overlap_efficiency"], 4),
        "speedup": round(t_serial / max(t_pipe, 1e-6), 3),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CLEAR-MoE++ novelty comparison")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use tiny tensors for fast smoke-test")
    args = parser.parse_args()

    logger.info("Device: %s | dry_run: %s", DEVICE, args.dry_run)

    benchmarks = [
        ("N1 ALFRouter",       bench_n1),
        ("N2 StreamFine",      bench_n2),
        ("N3 Disaggregated",   bench_n3),
    ]

    results = []
    for label, fn in benchmarks:
        logger.info("Running %s ...", label)
        r = fn(dry_run=args.dry_run)
        results.append(r)
        pretty = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}
        logger.info("  %s", pretty)

    # ── Save outputs ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs/novelty_comparison") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Markdown table
    header = (
        "# CLEAR-MoE++ Novelty Comparison\n\n"
        "| Component | Baseline | Novel | Paper Gap Addressed | Speedup / Improvement |\n"
        "|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        speedup = r.get("speedup")
        skew_red = r.get("skew_reduction_pct")
        if speedup is not None:
            improvement = f"{speedup:.2f}x speedup"
        elif skew_red is not None:
            improvement = f"{skew_red:.1f}% hotness skew reduction"
        else:
            improvement = "—"
        rows.append(
            f"| {r['component']} | {r['baseline_name']} | {r['novel_name']} "
            f"| {r['paper_gap']} | {improvement} |"
        )

    md = header + "\n".join(rows) + "\n"
    (out_dir / "comparison_table.md").write_text(md)

    logger.info("Saved to %s", out_dir)
    print("\n" + md)


if __name__ == "__main__":
    main()
