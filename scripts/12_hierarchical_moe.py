#!/usr/bin/env python3
"""
Script 12 — Hierarchical MoE (H-MoE) Extraction & Evaluation

Implements and evaluates the two-level hierarchical MoE architecture:
    Level 1 (coarse): G groups selected by a lightweight linear router
    Level 2 (fine):   K experts per group, selected by a second router

Compared to flat MoE (same total experts G*K), H-MoE:
    - Reduces routing entropy on skewed distributions
    - Uses two serial router passes (small latency overhead)
    - Provides better load balance

Pipeline:
    1. Load pretrained dense model
    2. Log calibration activations for selected layers
    3. Extract H-MoE weights via two-level KMeans
    4. Build and train hierarchical routers
    5. Evaluate quality vs flat MoE at same expert count

Usage:
    python scripts/12_hierarchical_moe.py --config configs/deit_s_imagenet.yaml \
        --activations_dir outputs/logs/activations_xxx/activations \
        --num_groups 2 --experts_per_group 2
"""

import argparse
import json
import logging
import sys
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clear_moe.utils import (
    load_config, set_seed, setup_logging, get_device,
    log_gpu_info, save_json, get_run_name,
)
from clear_moe.models import load_model, get_ffn_layers, get_model_input_transform
from clear_moe.calibration import get_eval_dataloader
from clear_moe.metrics import (
    evaluate_classification, evaluate_segmentation,
    measure_latency, measure_throughput,
)
from clear_moe.hierarchical_moe import (
    extract_hmoe_layer,
    build_hmoe_ffn,
    fit_hmoe_router,
    compute_hmoe_routing_stats,
    HMoELayer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="H-MoE extraction and evaluation")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--activations_dir", type=str, required=True,
                   help="Directory with saved .pt activation files from script 02")
    p.add_argument("--num_groups", type=int, default=2,
                   help="Number of coarse groups (G)")
    p.add_argument("--experts_per_group", type=int, default=2,
                   help="Experts per group (K); total experts = G*K")
    p.add_argument("--shared_rank", type=int, default=None,
                   help="Shared basis SVD rank (None = hidden_dim//2)")
    p.add_argument("--router_epochs", type=int, default=5)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Layer patching helpers
# ---------------------------------------------------------------------------

def _find_and_patch_ffn(model: nn.Module, layer_name: str, replacement: nn.Module):
    """Replace the FFN module at layer_name with replacement."""
    parts = layer_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], replacement)
    logger.info(f"Patched {layer_name} with H-MoE module")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    run_name = args.run_name or (
        f"hmoe_G{args.num_groups}_K{args.experts_per_group}_{get_run_name(config)}"
    )
    out_dir = Path(args.output_dir or config.get("output_dir", "outputs"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"hmoe_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(out_dir), run_name)
    device = get_device(config)
    log_gpu_info()

    total_experts = args.num_groups * args.experts_per_group
    logger.info(f"H-MoE: G={args.num_groups}, K={args.experts_per_group}, "
                f"total_experts={total_experts}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = load_model(config)
    model = model.to(device)
    ffn_layers = get_ffn_layers(model, config)
    logger.info(f"Found {len(ffn_layers)} FFN layers")

    # Select layers (default: last half)
    strategy = config.get("extraction", {}).get("layer_selection", "last_half")
    total_n = len(ffn_layers)
    if strategy == "last_half":
        selected_layers = [l for l in ffn_layers if l.layer_index >= total_n // 2]
    elif strategy == "all_layers":
        selected_layers = ffn_layers
    else:
        selected_layers = ffn_layers[total_n // 2:]

    logger.info(f"Extracting H-MoE for {len(selected_layers)}/{total_n} layers")

    # ------------------------------------------------------------------
    # Load activations
    # ------------------------------------------------------------------
    act_dir = Path(args.activations_dir)
    activations = {}
    for linfo in selected_layers:
        clean = linfo.name.replace(".", "_")
        path = act_dir / f"{clean}.pt"
        if path.exists():
            activations[linfo.name] = torch.load(path, map_location="cpu", weights_only=True)
            logger.info(f"Loaded activations for {linfo.name}: {activations[linfo.name].shape}")
        else:
            logger.warning(f"No activation file for {linfo.name} at {path}")

    if not activations:
        raise FileNotFoundError(f"No activation files found in {act_dir}")

    # ------------------------------------------------------------------
    # Extract H-MoE layers
    # ------------------------------------------------------------------
    hmoe_results = {}
    hmoe_modules = {}

    for linfo in selected_layers:
        name = linfo.name
        if name not in activations:
            continue

        logger.info(f"\nExtracting H-MoE: {name}")
        hmoe_layer = extract_hmoe_layer(
            model=model,
            layer_name=name,
            activations=activations[name],
            num_groups=args.num_groups,
            experts_per_group=args.experts_per_group,
            shared_basis_rank=args.shared_rank,
            device=device,
        )

        # Build forward module
        hmoe_ffn = build_hmoe_ffn(hmoe_layer, device)

        # Train router
        logger.info(f"  Training hierarchical router for {name}...")
        history = fit_hmoe_router(
            hmoe_ffn=hmoe_ffn,
            activations=activations[name],
            hmoe_layer=hmoe_layer,
            epochs=args.router_epochs,
            lr=config.get("router", {}).get("lr", 1e-3),
            device=device,
        )

        # Routing stats
        stats = compute_hmoe_routing_stats(
            router=hmoe_ffn.router,
            activations=activations[name],
            device=device,
        )

        hmoe_results[name] = {
            "num_groups": args.num_groups,
            "experts_per_group": args.experts_per_group,
            "total_experts": total_experts,
            "hidden_dim": hmoe_layer.hidden_dim,
            "shared_rank": hmoe_layer.shared_rank,
            "reconstruction_error": hmoe_layer.reconstruction_error,
            "training_history": {
                "final_coarse_acc": history["coarse_acc"][-1],
                "final_fine_acc": history["fine_acc"][-1],
            },
            "routing_stats": stats,
        }
        hmoe_modules[name] = hmoe_ffn

        logger.info(
            f"  coarse_acc={history['coarse_acc'][-1]:.4f}, "
            f"fine_acc={history['fine_acc'][-1]:.4f}, "
            f"entropy={stats['load_balance_entropy_nats']:.4f} "
            f"(max={stats['max_possible_entropy_nats']:.4f})"
        )

    # ------------------------------------------------------------------
    # Patch model with H-MoE modules
    # ------------------------------------------------------------------
    for name, module in hmoe_modules.items():
        _find_and_patch_ffn(model, name, module)

    model.eval()

    # ------------------------------------------------------------------
    # Evaluate patched model
    # ------------------------------------------------------------------
    transform = get_model_input_transform(config)
    eval_loader = get_eval_dataloader(config, transform=transform)
    task = config["data"]["task"]

    logger.info("\n" + "=" * 60)
    logger.info("H-MoE MODEL EVALUATION")
    logger.info("=" * 60)

    eval_results = {"task": task, "architecture": "hmoe",
                    "num_groups": args.num_groups,
                    "experts_per_group": args.experts_per_group,
                    "total_experts": total_experts}

    if task == "classification":
        cls_r = evaluate_classification(model, eval_loader, device)
        eval_results.update(cls_r)
        input_shape = (1, 3, config["model"].get("input_size", 224),
                       config["model"].get("input_size", 224))
    else:
        num_classes = config["model"].get("num_classes", 19)
        seg_r = evaluate_segmentation(model, eval_loader, device, num_classes)
        eval_results.update(seg_r)
        sz = config["model"].get("input_size", 512)
        input_shape = (1, 3, sz, sz)

    # Latency
    lat = measure_latency(model, input_shape, device, warmup=10, iterations=50)
    eval_results.update(lat)

    thr = measure_throughput(model, eval_loader, device)
    eval_results.update(thr)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    full_results = {
        "eval_results": eval_results,
        "per_layer_extraction": hmoe_results,
    }
    save_json(full_results, str(run_dir / "hmoe_results.json"))

    logger.info("\nH-MoE SUMMARY:")
    for key, val in eval_results.items():
        if isinstance(val, float):
            logger.info(f"  {key}: {val:.4f}")
        else:
            logger.info(f"  {key}: {val}")
    logger.info(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
