#!/usr/bin/env python3
"""
Script 04 — Extract Experts

Performs the core CLEAR-MoE extraction: decomposes selected FFN layers into
shared-basis + residual experts.

For each selected layer:
1. Clusters calibration activations via KMeans
2. Computes shared basis via truncated SVD
3. Computes per-expert residual weights
4. Saves the decomposition

Usage:
    python scripts/04_extract_experts.py --config configs/deit_s_imagenet.yaml \
        --activations_dir outputs/logs/activations_.../activations \
        --scores_file outputs/logs/scores_.../layer_scores.json
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
    log_gpu_info, save_json, load_json, get_run_name,
)
from clear_moe.models import load_model, get_ffn_layers
from clear_moe.scoring import LayerScore
from clear_moe.extraction import (
    extract_experts, extract_all_experts, save_expertized_layers,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract experts from dense FFN layers")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--activations_dir", type=str, required=True)
    parser.add_argument("--scores_file", type=str, required=True,
                        help="Path to layer_scores.json from script 03")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_experts", type=int, default=None,
                        help="Override number of experts")
    parser.add_argument("--shared_rank", type=int, default=None,
                        help="Override shared basis SVD rank")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    run_name = args.run_name or f"extraction_{get_run_name(config)}"
    run_dir = setup_logging(config.get("output_dir", "outputs"), run_name)
    device = get_device(config)

    log_gpu_info()

    num_experts = args.num_experts or config.get("extraction", {}).get("num_experts", 4)
    shared_rank = args.shared_rank or config.get("extraction", {}).get("shared_basis_rank")

    # Load layer scores
    scores_data = load_json(args.scores_file)
    layer_scores = []
    for s in scores_data["scores"]:
        ls = LayerScore(
            name=s["name"],
            layer_index=s["layer_index"],
            sparsity=s["sparsity"],
            multimodality=s["multimodality"],
            sensitivity=s["sensitivity"],
            composite=s["composite"],
            selected=s["selected"],
        )
        layer_scores.append(ls)

    selected = [s for s in layer_scores if s.selected]
    logger.info(f"Selected {len(selected)} layers for extraction")

    if not selected:
        logger.error("No layers selected for extraction! Check scores file.")
        return

    # Load model
    model = load_model(config)

    # Load activations for selected layers
    act_dir = Path(args.activations_dir)
    activations = {}

    for score in selected:
        clean_name = score.name.replace(".", "_")
        act_path = act_dir / f"{clean_name}.pt"

        if act_path.exists():
            activations[score.name] = torch.load(act_path, map_location="cpu",
                                                   weights_only=True)
            logger.info(f"Loaded activations for {score.name}: {activations[score.name].shape}")
        else:
            logger.warning(f"No activations for {score.name}")

    # Extract experts
    logger.info("=" * 60)
    logger.info(f"EXTRACTING EXPERTS (E={num_experts})")
    logger.info("=" * 60)

    expertized_layers = extract_all_experts(
        model=model,
        selected_layers=layer_scores,
        activations=activations,
        num_experts=num_experts,
        shared_basis_rank=shared_rank,
        device=device,
    )

    # Save extracted experts
    save_expertized_layers(expertized_layers, str(run_dir))

    # Save extraction summary
    summary = {
        "num_experts": num_experts,
        "num_layers_extracted": len(expertized_layers),
        "layers": {},
    }

    for name, exp in expertized_layers.items():
        summary["layers"][name] = {
            "hidden_dim": exp.hidden_dim,
            "intermediate_dim": exp.intermediate_dim,
            "shared_rank": exp.shared_rank,
            "reconstruction_error": exp.reconstruction_error,
            "cluster_sizes": [
                int((exp.cluster_labels == e).sum())
                for e in range(exp.num_experts)
            ],
        }

    save_json(summary, str(run_dir / "extraction_summary.json"))

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 80)
    for name, info in summary["layers"].items():
        logger.info(f"\n  {name}:")
        logger.info(f"    Dims: {info['hidden_dim']} -> {info['intermediate_dim']}")
        logger.info(f"    Shared rank: {info['shared_rank']}")
        logger.info(f"    Reconstruction error: {info['reconstruction_error']:.4f}")
        logger.info(f"    Cluster sizes: {info['cluster_sizes']}")

    logger.info(f"\nExtracted experts saved to {run_dir / 'expertized_layers'}")


if __name__ == "__main__":
    main()
