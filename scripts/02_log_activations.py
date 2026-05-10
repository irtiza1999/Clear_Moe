#!/usr/bin/env python3
"""
Script 02 — Log Activations

Runs the pretrained model on calibration data and captures FFN input
activations for all FFN layers. These activations are used downstream for:
- Layer scoring (script 03)
- Expert extraction via clustering (script 04)
- Router fitting supervision (script 05)

Memory-efficient: activations are moved to CPU immediately and saved to disk.

Usage:
    python scripts/02_log_activations.py --config configs/deit_s_imagenet.yaml
    python scripts/02_log_activations.py --config configs/segformer_b0_cityscapes.yaml \
        --calib_size 500
"""

import argparse
import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from clear_moe.utils import (
    load_config, set_seed, setup_logging, get_device,
    log_gpu_info, save_json, get_run_name,
)
from clear_moe.models import load_model, get_ffn_layers, get_model_input_transform
from clear_moe.calibration import get_calibration_dataloader, collect_activations

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Log FFN activations")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--calib_size", type=int, default=None,
                        help="Override calibration subset size")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Max batches to process (for testing)")
    parser.add_argument("--parallel", action="store_true",
                        help="Compute stats and save tensors concurrently (ThreadPoolExecutor)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Worker threads for parallel post-processing (default: 4)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.calib_size:
        config["data"]["calib_size"] = args.calib_size
    set_seed(config.get("seed", 42))

    run_name = args.run_name or f"activations_{get_run_name(config)}"
    run_dir = setup_logging(config.get("output_dir", "outputs"), run_name)
    device = get_device(config)

    log_gpu_info()

    # Load model
    model = load_model(config)
    model = model.to(device)

    # Get FFN layers
    ffn_layers = get_ffn_layers(model, config)
    layer_names = [layer.name for layer in ffn_layers]

    logger.info(f"Will capture activations for {len(layer_names)} FFN layers:")
    for i, name in enumerate(layer_names):
        logger.info(f"  [{i}] {name}")

    # Build calibration dataloader
    transform = get_model_input_transform(config)
    calib_loader = get_calibration_dataloader(config, transform=transform)
    logger.info(f"Calibration loader: {len(calib_loader)} batches")

    # Collect activations
    save_dir = str(run_dir)
    activations = collect_activations(
        model=model,
        dataloader=calib_loader,
        layer_names=layer_names,
        device=device,
        save_dir=save_dir,
        max_batches=args.max_batches,
    )

    # Compute per-layer stats (parallel or serial)
    def _compute_layer_stats(name_act):
        name, act = name_act
        act_float = act.float()
        return name, {
            "shape": list(act.shape),
            "dtype": str(act.dtype),
            "mean": act_float.mean().item(),
            "std": act_float.std().item(),
            "min": act_float.min().item(),
            "max": act_float.max().item(),
            "sparsity_01": (act_float.abs() < 0.01).float().mean().item(),
            "sparsity_001": (act_float.abs() < 0.001).float().mean().item(),
            "l1_norm_mean": act_float.abs().mean().item(),
            "l2_norm_mean": act_float.norm(dim=-1).mean().item(),
            "size_MB": act.nbytes / 1e6,
        }

    stats = {}
    if args.parallel:
        logger.info(f"Computing stats in parallel ({args.num_workers} workers)")
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(_compute_layer_stats, item): item[0]
                       for item in activations.items()}
            for fut in as_completed(futures):
                name, layer_stats = fut.result()
                stats[name] = layer_stats
    else:
        for item in activations.items():
            name, layer_stats = _compute_layer_stats(item)
            stats[name] = layer_stats

    save_json(stats, str(run_dir / "activation_stats.json"))

    # Save activation tensors (parallel or serial)
    act_dir = run_dir / "activations"
    act_dir.mkdir(exist_ok=True)

    def _save_activation(name_act):
        name, act = name_act
        clean_name = name.replace(".", "_")
        torch.save(act, str(act_dir / f"{clean_name}.pt"))

    if args.parallel:
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            list(pool.map(_save_activation, activations.items()))
    else:
        for item in activations.items():
            _save_activation(item)

    total_size_mb = sum(act.nbytes for act in activations.values()) / 1e6
    logger.info(f"\nTotal activation data: {total_size_mb:.1f} MB")
    logger.info(f"Saved to {act_dir}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info(f"{'Layer':<40} {'Shape':<20} {'Mean':<10} {'Sparsity':<10} {'Size(MB)':<10}")
    logger.info("-" * 80)
    for name, s in stats.items():
        short_name = name[-38:] if len(name) > 38 else name
        logger.info(
            f"{short_name:<40} {str(s['shape']):<20} "
            f"{s['mean']:<10.4f} {s['sparsity_01']:<10.4f} {s['size_MB']:<10.1f}"
        )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
