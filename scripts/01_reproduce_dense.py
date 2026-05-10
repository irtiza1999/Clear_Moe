#!/usr/bin/env python3
"""
Script 01 — Reproduce Dense Baseline

Loads a pretrained dense vision backbone and evaluates it on the validation set.
This establishes the baseline accuracy, latency, throughput, and energy metrics
that the CLEAR-MoE extraction must match or beat.

Usage:
    python scripts/01_reproduce_dense.py --config configs/deit_s_imagenet.yaml
    python scripts/01_reproduce_dense.py --config configs/segformer_b0_cityscapes.yaml
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from clear_moe.utils import (
    load_config, set_seed, setup_logging, get_device, get_dtype,
    log_gpu_info, count_parameters, save_json, get_run_name,
)
from clear_moe.models import load_model, get_ffn_layers, get_model_input_transform
from clear_moe.calibration import get_eval_dataloader
from clear_moe.metrics import (
    evaluate_classification, evaluate_segmentation,
    measure_latency, measure_throughput, measure_energy,
    count_active_params, estimate_flops, compute_memory_usage,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Reproduce dense baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run_name", type=str, default=None, help="Name for this run")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    # Setup
    run_name = args.run_name or f"dense_baseline_{get_run_name(config)}"
    run_dir = setup_logging(config.get("output_dir", "outputs"), run_name)
    device = get_device(config)
    dtype = get_dtype(config)

    log_gpu_info()
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {run_dir}")

    # Load model
    model = load_model(config)
    model = model.to(device)
    param_info = count_parameters(model)
    logger.info(f"Parameters: {param_info}")

    # List FFN layers (for reference)
    ffn_layers = get_ffn_layers(model, config)
    logger.info(f"Found {len(ffn_layers)} FFN layers in the backbone")

    # Get transform and dataloader
    transform = get_model_input_transform(config)
    eval_loader = get_eval_dataloader(config, transform=transform)

    # Evaluate
    task = config["data"]["task"]
    results = {"task": task, "model": config["model"]["name"], "type": "dense_baseline"}

    if task == "classification":
        logger.info("=" * 60)
        logger.info("CLASSIFICATION EVALUATION")
        logger.info("=" * 60)

        cls_results = evaluate_classification(model, eval_loader, device)
        results.update(cls_results)

        # Latency
        input_shape = (1, 3, config["model"].get("input_size", 224),
                       config["model"].get("input_size", 224))

    elif task == "segmentation":
        logger.info("=" * 60)
        logger.info("SEGMENTATION EVALUATION")
        logger.info("=" * 60)

        num_classes = config["model"].get("num_classes", 19)
        seg_results = evaluate_segmentation(model, eval_loader, device, num_classes)
        results.update(seg_results)

        input_size = config["model"].get("input_size", 512)
        input_shape = (1, 3, input_size, input_size)

    else:
        raise ValueError(f"Unknown task: {task}")

    # Structural metrics
    logger.info("-" * 40)
    logger.info("STRUCTURAL METRICS")
    param_results = count_active_params(model)
    results.update(param_results)

    flop_results = estimate_flops(model, input_shape)
    results.update(flop_results)

    # Runtime metrics
    logger.info("-" * 40)
    logger.info("RUNTIME METRICS")

    eval_cfg = config.get("evaluation", {})
    latency_results = measure_latency(
        model, input_shape, device,
        warmup=eval_cfg.get("latency_warmup", 10),
        iterations=eval_cfg.get("latency_iterations", 50),
        dtype=dtype,
    )
    results.update(latency_results)

    throughput_results = measure_throughput(model, eval_loader, device)
    results.update(throughput_results)

    memory_results = compute_memory_usage(model, input_shape, device)
    results.update(memory_results)

    # Energy
    energy_results = measure_energy(model, input_shape, device, dtype=dtype)
    results.update(energy_results)

    # Save results
    save_json(results, str(run_dir / "dense_baseline_results.json"))

    logger.info("=" * 60)
    logger.info("DENSE BASELINE SUMMARY")
    logger.info("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        elif isinstance(value, list):
            logger.info(f"  {key}: [{len(value)} items]")
        else:
            logger.info(f"  {key}: {value}")

    logger.info(f"\nResults saved to {run_dir / 'dense_baseline_results.json'}")


if __name__ == "__main__":
    main()
