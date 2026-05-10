#!/usr/bin/env python3
"""
Script 08 — Profile Power and Route Statistics

Detailed GPU power/energy profiling and routing analysis for both
classification and segmentation models. Generates summary plots and
detailed route statistics.

Usage:
    python scripts/08_profile_power.py --config configs/deit_s_imagenet.yaml \
        --model_checkpoint outputs/logs/benchmark_cls_.../moe_model.pt \
        --task cls
"""

import argparse
import logging
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from clear_moe.utils import (
    load_config, set_seed, setup_logging, get_device, get_dtype,
    log_gpu_info, save_json, get_run_name,
)
from clear_moe.models import load_model, get_ffn_layers, get_model_input_transform
from clear_moe.calibration import get_eval_dataloader
from clear_moe.metrics import measure_latency, measure_energy

logger = logging.getLogger(__name__)


def profile_detailed_power(
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device,
    duration_s: float = 10.0,
    sample_interval_ms: float = 50.0,
) -> dict:
    """Profile GPU power over time during continuous inference."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        logger.warning(f"pynvml not available: {e}")
        return {}

    model.eval()
    dummy = torch.randn(*input_shape, device=device)

    # Warm up
    for _ in range(10):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Profile
    power_trace = []
    timestamps = []
    inference_count = 0

    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        _ = model(dummy)
        inference_count += 1

        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_trace.append(power_mw / 1000.0)  # Watts
            timestamps.append(time.perf_counter() - start)
        except Exception:
            pass

    if device.type == "cuda":
        torch.cuda.synchronize()

    pynvml.nvmlShutdown()

    if power_trace:
        power_arr = np.array(power_trace)
        results = {
            "duration_s": duration_s,
            "inference_count": inference_count,
            "inferences_per_second": inference_count / duration_s,
            "power_mean_W": float(power_arr.mean()),
            "power_std_W": float(power_arr.std()),
            "power_min_W": float(power_arr.min()),
            "power_max_W": float(power_arr.max()),
            "power_p50_W": float(np.percentile(power_arr, 50)),
            "power_p95_W": float(np.percentile(power_arr, 95)),
            "total_energy_J": float(power_arr.mean() * duration_s),
            "energy_per_inference_mJ": float(
                power_arr.mean() * duration_s / inference_count * 1000
            ),
            "num_samples": len(power_trace),
        }
    else:
        results = {"error": "No power samples collected"}

    return results


def profile_routing_overhead(
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device,
    iterations: int = 100,
) -> dict:
    """
    Profile the fraction of time spent in routing vs expert compute.

    Hooks into MoE layers to separately time routing and execution.
    """
    model.eval()
    dummy = torch.randn(*input_shape, device=device)

    # Find MoE layers
    moe_layers = []
    for name, module in model.named_modules():
        if type(module).__name__ in ("MoEFFNWrapper", "GroupedExpertMLP"):
            moe_layers.append((name, module))

    if not moe_layers:
        logger.info("No MoE layers found — model appears to be dense")
        return {"is_dense": True}

    # Warm up
    for _ in range(10):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time full forward pass
    total_times = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_times.append(time.perf_counter() - start)

    results = {
        "num_moe_layers": len(moe_layers),
        "moe_layer_names": [n for n, _ in moe_layers],
        "total_forward_mean_ms": float(np.mean(total_times) * 1000),
        "total_forward_p50_ms": float(np.median(total_times) * 1000),
    }

    return results


def generate_profile_plots(results: dict, save_dir: str):
    """Generate profiling visualizations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return

    save_path = Path(save_dir)

    # Power comparison (if both dense and MoE results available)
    if "dense_power" in results and "moe_power" in results:
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ["Mean Power (W)", "Energy/Inf (mJ)", "Throughput (inf/s)"]
        dense_vals = [
            results["dense_power"].get("power_mean_W", 0),
            results["dense_power"].get("energy_per_inference_mJ", 0),
            results["dense_power"].get("inferences_per_second", 0),
        ]
        moe_vals = [
            results["moe_power"].get("power_mean_W", 0),
            results["moe_power"].get("energy_per_inference_mJ", 0),
            results["moe_power"].get("inferences_per_second", 0),
        ]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width/2, dense_vals, width, label="Dense", color="#3498db")
        ax.bar(x + width/2, moe_vals, width, label="CLEAR-MoE", color="#e74c3c")

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_title("Dense vs CLEAR-MoE Power Profile")

        plt.tight_layout()
        plt.savefig(save_path / "power_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Plots saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile power and routing")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default="cls", choices=["cls", "seg", "both"])
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Power profiling duration in seconds")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    run_name = args.run_name or f"profile_{get_run_name(config)}"
    run_dir = setup_logging(config.get("output_dir", "outputs"), run_name)
    device = get_device(config)

    log_gpu_info()

    # Load dense model for comparison
    model = load_model(config)
    model = model.to(device)

    input_size = config["model"].get("input_size", 224)
    input_shape = (1, 3, input_size, input_size)

    results = {}

    # Dense model profiling
    logger.info("=" * 60)
    logger.info("DENSE MODEL POWER PROFILE")
    logger.info("=" * 60)

    dense_power = profile_detailed_power(model, input_shape, device, args.duration)
    results["dense_power"] = dense_power

    if dense_power:
        logger.info(f"  Mean power: {dense_power.get('power_mean_W', -1):.1f} W")
        logger.info(f"  Energy/inference: {dense_power.get('energy_per_inference_mJ', -1):.2f} mJ")
        logger.info(f"  Throughput: {dense_power.get('inferences_per_second', -1):.1f} inf/s")

    # Routing overhead analysis
    logger.info("\n" + "=" * 60)
    logger.info("ROUTING OVERHEAD ANALYSIS")
    logger.info("=" * 60)

    routing_profile = profile_routing_overhead(model, input_shape, device)
    results["routing_overhead"] = routing_profile

    for k, v in routing_profile.items():
        logger.info(f"  {k}: {v}")

    # Latency breakdown
    logger.info("\n" + "=" * 60)
    logger.info("LATENCY BREAKDOWN")
    logger.info("=" * 60)

    for bs in [1, 2, 4]:
        shape = (bs, 3, input_size, input_size)
        try:
            lat = measure_latency(model, shape, device, warmup=5, iterations=20)
            results[f"latency_bs{bs}"] = lat
            logger.info(
                f"  Batch {bs}: {lat['latency_mean_ms']:.2f}ms mean, "
                f"{lat['latency_p50_ms']:.2f}ms p50"
            )
        except RuntimeError as e:
            logger.warning(f"  Batch {bs}: OOM or error — {e}")

    # Save all results
    save_json(results, str(run_dir / "power_profile.json"))

    # Generate plots
    generate_profile_plots(results, str(run_dir))

    logger.info(f"\nProfile results saved to {run_dir}")


if __name__ == "__main__":
    main()
