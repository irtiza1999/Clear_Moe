#!/usr/bin/env python3
"""
Script 07 — Benchmark Segmentation

Same flow as script 06 but for semantic segmentation on Cityscapes.
Tests whether expert extraction transfers to dense-prediction tasks
(the key scientific question of the paper).

Usage:
    python scripts/07_benchmark_seg.py --config configs/segformer_b0_cityscapes.yaml \
        --extraction_dir outputs/logs/extraction_.../expertized_layers \
        --router_dir outputs/logs/routers_.../routers \
        --baseline_results outputs/logs/dense_baseline_.../dense_baseline_results.json
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from clear_moe.utils import (
    load_config, set_seed, setup_logging, get_device, get_dtype,
    log_gpu_info, save_json, load_json, get_run_name, load_checkpoint,
)
from clear_moe.models import (
    load_model, get_ffn_layers, replace_ffn_with_moe, get_model_input_transform,
)
from clear_moe.calibration import get_eval_dataloader
from clear_moe.extraction import load_expertized_layers, build_moe_ffn
from clear_moe.router import create_router
from clear_moe.metrics import (
    evaluate_segmentation, measure_latency, measure_throughput,
    measure_energy, count_active_params, estimate_flops, compute_memory_usage,
)
from runtime.grouped_expert_mlp import MoEFFNWrapper

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Benchmark segmentation with MoE model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--extraction_dir", type=str, required=True)
    parser.add_argument("--router_dir", type=str, required=True)
    parser.add_argument("--baseline_results", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    run_name = args.run_name or f"benchmark_seg_{get_run_name(config)}"
    run_dir = setup_logging(config.get("output_dir", "outputs"), run_name)
    device = get_device(config)
    dtype = get_dtype(config)

    log_gpu_info()

    # Load model
    model = load_model(config)
    ffn_layers = get_ffn_layers(model, config)

    # Load extracted experts
    expertized_layers = load_expertized_layers(args.extraction_dir)

    # Load routers and assemble MoE modules
    from pathlib import Path
    router_dir = Path(args.router_dir)
    moe_modules = {}

    for name, exp_layer in expertized_layers.items():
        ffn_module = None
        for fl in ffn_layers:
            if fl.name == name:
                ffn_module = fl.module
                break

        if ffn_module is None:
            logger.warning(f"Could not find FFN module for {name}")
            continue

        moe_ffn = build_moe_ffn(exp_layer, ffn_module)

        clean_name = name.replace(".", "_")
        router_path = router_dir / f"router_{clean_name}.pt"

        if router_path.exists():
            ckpt = load_checkpoint(str(router_path))
            router = create_router(config, exp_layer.hidden_dim, exp_layer.num_experts)
            router.load_state_dict(ckpt["state_dict"])
        else:
            router = create_router(config, exp_layer.hidden_dim, exp_layer.num_experts)

        moe_wrapper = MoEFFNWrapper(moe_ffn, router)
        moe_modules[name] = moe_wrapper

    model = replace_ffn_with_moe(model, list(moe_modules.keys()), moe_modules)
    model = model.to(device)
    model.eval()

    # Evaluate
    transform = get_model_input_transform(config)
    eval_loader = get_eval_dataloader(config, transform=transform)
    num_classes = config["model"].get("num_classes", 19)

    logger.info("=" * 60)
    logger.info("SEGMENTATION EVALUATION (MoE Model)")
    logger.info("=" * 60)

    seg_results = evaluate_segmentation(model, eval_loader, device, num_classes)

    input_size = config["model"].get("input_size", 512)
    input_shape = (1, 3, input_size, input_size)

    # Metrics
    param_results = count_active_params(model)
    flop_results = estimate_flops(model, input_shape)

    eval_cfg = config.get("evaluation", {})
    latency_results = measure_latency(
        model, input_shape, device,
        warmup=eval_cfg.get("latency_warmup", 5),
        iterations=eval_cfg.get("latency_iterations", 20),
        dtype=dtype,
    )
    throughput_results = measure_throughput(model, eval_loader, device)
    memory_results = compute_memory_usage(model, input_shape, device)
    energy_results = measure_energy(model, input_shape, device, dtype=dtype)

    results = {
        "task": "segmentation",
        "model": config["model"]["name"],
        "type": "clear_moe",
        "num_expertized_layers": len(moe_modules),
        **seg_results,
        **param_results,
        **flop_results,
        **latency_results,
        **throughput_results,
        **memory_results,
        **energy_results,
    }

    # Compare against baseline
    if args.baseline_results and os.path.exists(args.baseline_results):
        baseline = load_json(args.baseline_results)
        comparison = {}

        if "miou" in baseline and "miou" in results:
            comparison["miou_delta"] = results["miou"] - baseline["miou"]

        if "latency_p50_ms" in baseline and "latency_p50_ms" in results:
            comparison["latency_reduction_pct"] = (
                (baseline["latency_p50_ms"] - results["latency_p50_ms"])
                / baseline["latency_p50_ms"] * 100
            )

        if "total_params" in baseline and "active_params" in results:
            comparison["param_reduction_pct"] = (
                (baseline["total_params"] - results["active_params"])
                / baseline["total_params"] * 100
            )

        results["comparison"] = comparison
        logger.info("\n" + "-" * 40)
        logger.info("COMPARISON vs DENSE BASELINE")
        for k, v in comparison.items():
            logger.info(f"  {k}: {v:+.2f}")

    save_json(results, str(run_dir / "moe_seg_results.json"))
    logger.info(f"\nResults saved to {run_dir / 'moe_seg_results.json'}")


if __name__ == "__main__":
    main()
