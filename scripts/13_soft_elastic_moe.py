#!/usr/bin/env python3
"""
Script 13 — Soft-MoE and Elastic-MoE Evaluation

Implements and evaluates two alternative MoE routing strategies:

Soft-MoE:
    All E experts execute for every token. Outputs blended with softmax weights.
    Eliminates load imbalance. Upper-bound quality reference.
    Cost: E× the compute of hard top-1 routing.

Elastic-MoE:
    Active expert count varies per token based on confidence threshold.
    High-confidence → top-1; Low-confidence → top-k (up to k_max).
    Sweeps confidence_threshold to plot quality vs active FLOPs Pareto curve.

Outputs:
    - Quality vs active_flops Pareto CSV
    - Confidence threshold sweep table
    - Comparison with flat hard MoE at same E

Usage:
    python scripts/13_soft_elastic_moe.py --config configs/deit_s_imagenet.yaml \
        --activations_dir outputs/.../activations \
        --extraction_dir outputs/.../expertized_layers \
        --architecture soft
    python scripts/13_soft_elastic_moe.py ... --architecture elastic \
        --threshold_sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clear_moe.utils import (
    load_config, set_seed, setup_logging, get_device,
    log_gpu_info, save_json, get_run_name,
)
from clear_moe.models import load_model, get_ffn_layers, get_model_input_transform
from clear_moe.calibration import get_eval_dataloader
from clear_moe.extraction import load_expertized_layers
from clear_moe.metrics import (
    evaluate_classification, evaluate_segmentation,
    measure_latency, measure_throughput, estimate_flops,
)
from clear_moe.soft_elastic_moe import (
    SoftMoEFFN, ElasticMoEFFN,
    build_soft_moe_ffn, build_elastic_moe_ffn,
    count_soft_moe_flops, count_elastic_moe_expected_flops,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Soft-MoE and Elastic-MoE evaluation")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--activations_dir", type=str, required=True)
    p.add_argument("--extraction_dir", type=str, required=True,
                   help="Directory from script 04 (expertized_layers/)")
    p.add_argument("--architecture", type=str, default="soft",
                   choices=["soft", "elastic", "capacity", "all"],
                   help="Which architecture to evaluate")
    p.add_argument("--threshold_sweep", type=str,
                   default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
                   help="Confidence thresholds to sweep for Elastic-MoE")
    p.add_argument("--k_max", type=int, default=2,
                   help="Maximum active experts for Elastic-MoE uncertain tokens")
    p.add_argument("--capacity_factor", type=float, default=1.25,
                   help="Capacity factor for Capacity Buffer MoE")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Build MoE modules from extracted expert weights
# ---------------------------------------------------------------------------

def _build_expert_linears(exp_layer, device: torch.device):
    """Build nn.Linear modules from ExpertizedLayer residual weights."""
    fc1s = []
    fc2s = []
    E = exp_layer.num_experts
    for e in range(E):
        fc1 = nn.Linear(exp_layer.hidden_dim, exp_layer.intermediate_dim, bias=True)
        fc1.weight.data.copy_(exp_layer.expert_weights_1[e])
        fc1.bias.data.copy_(exp_layer.expert_biases_1[e])

        fc2 = nn.Linear(exp_layer.intermediate_dim, exp_layer.hidden_dim, bias=True)
        fc2.weight.data.copy_(exp_layer.expert_weights_2[e])
        fc2.bias.data.copy_(exp_layer.expert_biases_2[e])

        fc1s.append(fc1.to(device))
        fc2s.append(fc2.to(device))

    shared_U = exp_layer.shared_basis_U.to(device)
    shared_S = exp_layer.shared_basis_S.to(device)
    shared_V = exp_layer.shared_basis_V.to(device)

    return fc1s, fc2s, shared_U, shared_S, shared_V


def _patch_model(model: nn.Module, layer_name: str, new_module: nn.Module):
    parts = layer_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_architecture(
    model: nn.Module,
    config: dict,
    device: torch.device,
    arch_name: str,
) -> Dict:
    transform = get_model_input_transform(config)
    eval_loader = get_eval_dataloader(config, transform=transform)
    task = config["data"]["task"]

    results = {"architecture": arch_name}

    if task == "classification":
        cls_r = evaluate_classification(model, eval_loader, device)
        results.update(cls_r)
        sz = config["model"].get("input_size", 224)
        input_shape = (1, 3, sz, sz)
    else:
        num_cls = config["model"].get("num_classes", 19)
        seg_r = evaluate_segmentation(model, eval_loader, device, num_cls)
        results.update(seg_r)
        sz = config["model"].get("input_size", 512)
        input_shape = (1, 3, sz, sz)

    lat = measure_latency(model, input_shape, device, warmup=10, iterations=50)
    results.update(lat)

    thr = measure_throughput(model, eval_loader, device)
    results.update(thr)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or config.get("output_dir", "outputs"))
    run_dir = out_dir / f"soft_elastic_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(out_dir), args.run_name or "soft_elastic_moe")
    device = get_device(config)
    log_gpu_info()

    # ------------------------------------------------------------------
    # Load extracted experts
    # ------------------------------------------------------------------
    act_dir = Path(args.activations_dir)
    exp_dir = Path(args.extraction_dir)

    expertized_layers = load_expertized_layers(str(exp_dir))
    logger.info(f"Loaded {len(expertized_layers)} expertized layers")

    architectures = (
        ["soft", "elastic", "capacity"] if args.architecture == "all"
        else [args.architecture]
    )

    all_eval_results = {}
    pareto_rows = []

    for arch in architectures:
        logger.info(f"\n{'='*60}")
        logger.info(f"Architecture: {arch.upper()}-MoE")
        logger.info("=" * 60)

        thresholds = (
            [float(t) for t in args.threshold_sweep.split(",")]
            if arch == "elastic"
            else [None]
        )

        for thresh in thresholds:
            tag = f"{arch}" if thresh is None else f"{arch}_t{thresh:.1f}"
            logger.info(f"\nBuilding {tag} model...")

            # Fresh model for each config
            model = load_model(config)
            model = model.to(device)

            # Patch each expertized layer
            flop_estimates = {}
            for layer_name, exp_layer in expertized_layers.items():
                fc1s, fc2s, U, S, V = _build_expert_linears(exp_layer, device)
                E = exp_layer.num_experts
                D = exp_layer.hidden_dim
                I = exp_layer.intermediate_dim

                if arch == "soft":
                    module = build_soft_moe_ffn(fc1s, fc2s, U, S, V).to(device)
                    flop_est = count_soft_moe_flops(D, I, E, num_tokens=196)
                    flop_estimates[layer_name] = flop_est

                elif arch == "elastic":
                    module = build_elastic_moe_ffn(
                        fc1s, fc2s, U, S, V,
                        confidence_threshold=thresh,
                        k_max=args.k_max,
                    ).to(device)
                    # Estimate FLOPs assuming 50% confident tokens (placeholder)
                    flop_est = count_elastic_moe_expected_flops(D, I, E, 196, 0.5, args.k_max)
                    flop_estimates[layer_name] = flop_est

                elif arch == "capacity":
                    from clear_moe.soft_elastic_moe import build_capacity_moe_ffn
                    module = build_capacity_moe_ffn(
                        fc1s, fc2s, U, S, V,
                        capacity_factor=args.capacity_factor,
                    ).to(device)
                    flop_est = count_soft_moe_flops(D, I, 1, 196)  # approx: 1 expert path
                    flop_estimates[layer_name] = flop_est

                _patch_model(model, layer_name, module)

            model.eval()

            # Evaluate
            eval_r = evaluate_architecture(model, config, device, arch_name=tag)
            eval_r["confidence_threshold"] = thresh
            eval_r["flop_estimates"] = flop_estimates

            all_eval_results[tag] = eval_r

            # Pareto row
            quality_key = "top1_accuracy" if "top1_accuracy" in eval_r else "miou"
            pareto_rows.append({
                "architecture": tag,
                "quality": eval_r.get(quality_key, 0),
                "throughput_tok_s": eval_r.get("throughput_tok_s",
                                                eval_r.get("throughput_images_per_sec", 0)),
                "latency_p50_ms": eval_r.get("latency_p50_ms", 0),
                "confidence_threshold": thresh,
            })

            logger.info(f"  {quality_key}={eval_r.get(quality_key, 0):.4f}, "
                        f"latency_p50={eval_r.get('latency_p50_ms', 0):.2f}ms")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    save_json(all_eval_results, str(run_dir / "results.json"))

    with open(run_dir / "pareto.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pareto_rows[0].keys() if pareto_rows else [])
        writer.writeheader()
        writer.writerows(pareto_rows)

    # Markdown summary
    lines = ["# Soft-MoE / Elastic-MoE Evaluation\n",
             "| Architecture | Top-1 / mIoU | Latency p50 (ms) | Throughput (tok/s) |",
             "|---|---|---|---|"]
    for tag, r in all_eval_results.items():
        q_key = "top1_accuracy" if "top1_accuracy" in r else "miou"
        lines.append(
            f"| {tag} | {r.get(q_key, 0):.4f} | "
            f"{r.get('latency_p50_ms', 0):.2f} | "
            f"{r.get('throughput_images_per_sec', 0):.1f} |"
        )
    with open(run_dir / "summary.md", "w") as f:
        f.write("\n".join(lines))

    logger.info(f"\nAll results saved to: {run_dir}")


if __name__ == "__main__":
    main()
