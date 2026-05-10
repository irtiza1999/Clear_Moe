#!/usr/bin/env python3
"""
Script 05 — Fit Routers

Trains lightweight routers for each expertized layer. Routers learn to
predict expert assignments from hidden states, supervised by the KMeans
cluster labels from extraction.

This is the only training step in CLEAR-MoE — all other components are
extracted from the pretrained model without gradient-based learning.

Usage:
    python scripts/05_fit_router.py --config configs/deit_s_imagenet.yaml \
        --activations_dir outputs/logs/activations_.../activations \
        --extraction_dir outputs/logs/extraction_.../expertized_layers
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from clear_moe.utils import (
    load_config, set_seed, setup_logging, get_device,
    log_gpu_info, save_json, save_checkpoint, get_run_name,
)
from clear_moe.models import load_model, get_ffn_layers
from clear_moe.extraction import load_expertized_layers
from clear_moe.router import create_router, fit_router, fit_all_routers, compute_routing_stats

logger = logging.getLogger(__name__)


def _fit_routers_parallel(expertized_layers, activations, config, device, num_workers=4):
    """Fit routers for all layers concurrently using ThreadPoolExecutor.

    Each layer's router fit is independent (no shared GPU state at fit time —
    router training runs on CPU tensors by default in fit_router). Safe to
    parallelize.
    """
    from clear_moe.router import create_router, fit_router

    def _fit_one(name):
        exp_layer = expertized_layers[name]
        acts = activations.get(name)
        if acts is None:
            logger.warning(f"No activations for {name}, skipping")
            return name, None
        router = create_router(
            config,
            hidden_dim=exp_layer.hidden_dim,
            num_experts=exp_layer.num_experts,
        )
        router, history = fit_router(
            router=router,
            activations=acts,
            cluster_labels=exp_layer.cluster_labels,
            config=config,
            device=device,
        )
        logger.info(
            f"  {name}: acc={history['accuracy'][-1]:.4f} "
            f"ent={history['entropy'][-1]:.4f}"
        )
        return name, router

    routers = {}
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_fit_one, name): name for name in expertized_layers}
        for fut in as_completed(futures):
            name, router = fut.result()
            if router is not None:
                routers[name] = router
    return routers


def main():
    parser = argparse.ArgumentParser(description="Fit expert routers")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--activations_dir", type=str, required=True)
    parser.add_argument("--extraction_dir", type=str, required=True,
                        help="Directory containing expertized_layers/ from script 04")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--router_depth", type=int, default=None,
                        help="Router depth: 0=linear, 1=one hidden layer, 2=two hidden layers")
    parser.add_argument("--parallel_kmeans", action="store_true",
                        help="Fit routers for all layers concurrently (ThreadPoolExecutor, CPU)")
    parser.add_argument("--fit_workers", type=int, default=4,
                        help="Worker threads for parallel router fitting (default: 4)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config.setdefault("router", {})["epochs"] = args.epochs
    if args.lr:
        config.setdefault("router", {})["lr"] = args.lr
    if args.router_depth is not None:
        # depth 0 = linear router, depth 1/2 = MLP router
        router_cfg = config.setdefault("router", {})
        if args.router_depth == 0:
            router_cfg["type"] = "linear"
        else:
            router_cfg["type"] = "mlp"
            router_cfg["num_layers"] = args.router_depth
    set_seed(config.get("seed", 42))

    run_name = args.run_name or f"routers_{get_run_name(config)}"
    run_dir = setup_logging(config.get("output_dir", "outputs"), run_name)
    device = get_device(config)

    log_gpu_info()

    # Load extracted experts
    expertized_layers = load_expertized_layers(args.extraction_dir)
    logger.info(f"Loaded {len(expertized_layers)} expertized layers")

    # Load activations
    act_dir = Path(args.activations_dir)
    activations = {}

    for name in expertized_layers:
        clean_name = name.replace(".", "_")
        act_path = act_dir / f"{clean_name}.pt"

        if act_path.exists():
            activations[name] = torch.load(act_path, map_location="cpu", weights_only=True)
            logger.info(f"Loaded activations for {name}: {activations[name].shape}")

    # Fit routers
    logger.info("=" * 60)
    logger.info("FITTING ROUTERS" + (" (parallel)" if args.parallel_kmeans else ""))
    logger.info("=" * 60)

    if args.parallel_kmeans:
        routers = _fit_routers_parallel(
            expertized_layers=expertized_layers,
            activations=activations,
            config=config,
            device=device,
            num_workers=args.fit_workers,
        )
    else:
        routers = fit_all_routers(
            model=None,
            expertized_layers=expertized_layers,
            activations=activations,
            config=config,
            device=device,
        )

    # Compute and log routing statistics
    all_stats = {}
    for name, router in routers.items():
        logger.info(f"\nRouting stats for {name}:")
        stats = compute_routing_stats(router, activations[name], device)
        all_stats[name] = stats

        logger.info(f"  Expert loads: {stats['expert_load_fractions']}")
        logger.info(f"  Load balance std: {stats['load_balance_std']:.4f}")
        logger.info(f"  Mean entropy: {stats['mean_entropy']:.4f}")
        logger.info(f"  Mean top-1 weight: {stats['mean_top1_weight']:.4f}")

    # Save routers
    router_dir = run_dir / "routers"
    router_dir.mkdir(exist_ok=True)

    for name, router in routers.items():
        clean_name = name.replace(".", "_")
        router_path = router_dir / f"router_{clean_name}.pt"
        save_checkpoint(
            {
                "layer_name": name,
                "state_dict": router.state_dict(),
                "config": config.get("router", {}),
                "num_experts": expertized_layers[name].num_experts,
                "hidden_dim": expertized_layers[name].hidden_dim,
            },
            str(router_dir),
            filename=f"router_{clean_name}.pt",
        )

    # Save routing statistics
    save_json(all_stats, str(run_dir / "routing_stats.json"))

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("ROUTER FITTING SUMMARY")
    logger.info("=" * 80)
    for name, stats in all_stats.items():
        logger.info(f"\n  {name}:")
        logger.info(f"    Load distribution: {[f'{f:.3f}' for f in stats['expert_load_fractions']]}")
        logger.info(f"    Balance std: {stats['load_balance_std']:.4f}")
        logger.info(f"    Entropy: {stats['mean_entropy']:.4f}")

    logger.info(f"\nRouters saved to {router_dir}")


if __name__ == "__main__":
    main()
