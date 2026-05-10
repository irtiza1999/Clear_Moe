#!/usr/bin/env python3
"""
Script 03 — Score Layers

Loads saved activations and computes a composite expertization score for each
FFN layer based on:
1. Activation sparsity
2. Multimodality / clusterability
3. Output sensitivity

Selects the layers for expertization and generates a visualization.

Usage:
    python scripts/03_score_layers.py --config configs/deit_s_imagenet.yaml \
        --activations_dir outputs/logs/activations_deit_small.../activations
"""

import argparse
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from clear_moe.utils import (
    load_config, set_seed, setup_logging, get_device,
    log_gpu_info, save_json, get_run_name,
)
from clear_moe.models import load_model, get_ffn_layers, get_model_input_transform
from clear_moe.calibration import get_calibration_dataloader
from clear_moe.scoring import score_layers, select_layers, visualize_scores

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Score layers for expertization")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--activations_dir", type=str, required=True,
                        help="Directory containing saved activation .pt files")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None,
                        help="Override layer selection strategy")
    parser.add_argument("--skip_sensitivity", action="store_true",
                        help="Skip sensitivity analysis (faster, uses position proxy)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    run_name = args.run_name or f"scores_{get_run_name(config)}"
    run_dir = setup_logging(config.get("output_dir", "outputs"), run_name)
    device = get_device(config)

    log_gpu_info()

    # Load model and FFN layers
    model = load_model(config)
    model = model.to(device)
    ffn_layers = get_ffn_layers(model, config)

    # Load activations from disk
    act_dir = Path(args.activations_dir)
    activations = {}

    for layer_info in ffn_layers:
        clean_name = layer_info.name.replace(".", "_")
        act_path = act_dir / f"{clean_name}.pt"

        if act_path.exists():
            activations[layer_info.name] = torch.load(act_path, map_location="cpu",
                                                       weights_only=True)
            logger.info(f"Loaded {layer_info.name}: {activations[layer_info.name].shape}")
        else:
            logger.warning(f"No activation file for {layer_info.name} at {act_path}")

    if not activations:
        logger.error("No activations loaded! Check --activations_dir path.")
        return

    # Build calibration loader for sensitivity analysis
    calib_loader = None
    if not args.skip_sensitivity:
        transform = get_model_input_transform(config)
        calib_loader = get_calibration_dataloader(config, transform=transform)

    # Score layers
    num_clusters = config.get("extraction", {}).get("num_experts", 4)
    scores = score_layers(
        activations=activations,
        model=model,
        layer_infos=ffn_layers,
        device=device,
        calib_loader=calib_loader,
        num_clusters=num_clusters,
    )

    # Select layers
    strategy = args.strategy or config.get("extraction", {}).get("layer_selection", "last_half")
    top_k = config.get("extraction", {}).get("top_k")
    threshold = config.get("extraction", {}).get("threshold")

    scores = select_layers(
        scores,
        strategy=strategy,
        top_k=top_k,
        threshold=threshold,
        total_layers=len(ffn_layers),
    )

    # Save results
    scores_data = {
        "strategy": strategy,
        "total_layers": len(ffn_layers),
        "selected_layers": sum(1 for s in scores if s.selected),
        "scores": [s.to_dict() for s in scores],
    }
    save_json(scores_data, str(run_dir / "layer_scores.json"))

    # Generate visualization
    viz_path = str(run_dir / "layer_scores.png")
    try:
        visualize_scores(scores, save_path=viz_path)
    except Exception as e:
        logger.warning(f"Failed to generate visualization: {e}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("LAYER SCORING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Selected {sum(1 for s in scores if s.selected)}/{len(scores)} layers")
    logger.info("")
    logger.info(f"{'Layer':<40} {'Idx':>4} {'Spar':>6} {'Multi':>6} {'Sens':>6} {'Comp':>6} {'Sel':>4}")
    logger.info("-" * 80)
    for s in sorted(scores, key=lambda x: x.layer_index):
        sel = "Yes" if s.selected else ""
        logger.info(
            f"{s.name[-38:]:<40} {s.layer_index:>4} "
            f"{s.sparsity:>6.3f} {s.multimodality:>6.3f} "
            f"{s.sensitivity:>6.3f} {s.composite:>6.3f} {sel:>4}"
        )
    logger.info("=" * 80)
    logger.info(f"\nScores saved to {run_dir / 'layer_scores.json'}")


if __name__ == "__main__":
    main()
