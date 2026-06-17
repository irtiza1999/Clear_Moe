"""
Experiment 7: Repeated-Run and Statistical Robustness

Runs the main CLEAR-MoE++ configuration with 3 random seeds affecting:
  - calibration-image sampling
  - k-means initialization
  - router initialization

Reports mean ± std for Top-1, latency, load skew, router accuracy.

Run:
  python experiments/run_seeds.py --config configs/imagenette_deit_s_preprocessed.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import copy

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from clear_moe.calibration import collect_activations, get_calibration_dataloader, get_eval_dataloader
from clear_moe.extraction import extract_all_experts
from clear_moe.metrics import (
    count_active_params,
    evaluate_classification,
    measure_latency,
    compute_memory_usage,
)
from clear_moe.models import get_ffn_layers, get_model_input_transform, load_model, replace_ffn_with_moe
from clear_moe.moe_builder import build_moe_modules_from_expertized
from clear_moe.router import fit_all_routers, compute_routing_stats
from clear_moe.scoring import score_layers, select_layers
from experiments.experiment_logger import ExperimentLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(config: dict) -> torch.device:
    device_str = config.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def run_one_seed(config: dict, device: torch.device, seed: int) -> Dict:
    """Run the full pipeline once with the given seed and return metrics."""
    set_seed(seed)
    logger.info(f"\n{'='*60}")
    logger.info(f"Running seed {seed}")
    logger.info(f"{'='*60}")

    transform = get_model_input_transform(config)
    calib_loader = get_calibration_dataloader(config, transform=transform, seed=seed)
    eval_loader = get_eval_dataloader(config, transform=transform)

    model = load_model(config).to(device)
    layer_infos = get_ffn_layers(model, config)
    layer_names = [info.name for info in layer_infos]

    activations = collect_activations(model, calib_loader, layer_names, device)

    # Use composite (score_top_k) — the proposed full method
    k = len(layer_infos) // 2
    scores = score_layers(activations, model, layer_infos, device, calib_loader)
    scores = select_layers(scores, strategy="score_top_k", top_k=k, total_layers=len(layer_infos))
    selected = [s for s in scores if s.selected]
    selected_names = [s.name for s in selected]
    selected_indices = [s.layer_index for s in selected]
    logger.info(f"  Seed {seed}: selected layers = {selected_indices}")

    num_experts = config.get("extraction", {}).get("num_experts", 4)
    basis_rank = config.get("extraction", {}).get("shared_basis_rank") or None

    expertized = extract_all_experts(
        model, selected, activations,
        num_experts=num_experts,
        shared_basis_rank=basis_rank,
        device=device,
    )
    routers = fit_all_routers(model, expertized, activations, config, device)

    # Build the actual MoE model for evaluation
    moe_modules = build_moe_modules_from_expertized(expertized, routers, model, config)
    moe_model = copy.deepcopy(model)
    replace_ffn_with_moe(moe_model, selected_names, moe_modules)
    moe_model = moe_model.to(device)

    # Compute routing stats
    router_accs = []
    load_skews = []
    for name in selected_names:
        if name in routers and name in expertized:
            exp = expertized[name]
            router = routers[name]
            stats = compute_routing_stats(router, activations[name], device)
            load_skews.append(stats.get("load_balance_std", 0))
            with torch.no_grad():
                acts = activations[name].to(device)
                preds = []
                for i in range(0, acts.shape[0], 2048):
                    idx, _, _ = router(acts[i:i+2048])
                    if idx.dim() > 1:
                        idx = idx[:, 0]
                    preds.append(idx.cpu())
                preds_np = torch.cat(preds).numpy()
                acc = (preds_np == exp.cluster_labels).mean()
                router_accs.append(acc)

    cls_results = evaluate_classification(moe_model, eval_loader, device)
    input_shape = (
        config["data"].get("val_batch_size", 8),
        3, 224, 224,
    )
    lat_results = measure_latency(moe_model, input_shape, device)
    mem_results = compute_memory_usage(moe_model, input_shape, device)

    return {
        "seed": seed,
        "selected_layers": selected_indices,
        "top1": cls_results["top1"],
        "top5": cls_results["top5"],
        "p50_ms": lat_results["latency_p50_ms"],
        "p95_ms": lat_results["latency_p95_ms"],
        "mean_ms": lat_results["latency_mean_ms"],
        "std_ms": lat_results["latency_std_ms"],
        "peak_vram_mb": mem_results.get("peak_memory_MB", -1),
        "router_accuracy_mean": float(np.mean(router_accs)) if router_accs else -1.0,
        "router_accuracy_std": float(np.std(router_accs)) if router_accs else -1.0,
        "load_skew_mean": float(np.mean(load_skews)) if load_skews else -1.0,
        "load_skew_std": float(np.std(load_skews)) if load_skews else -1.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Robustness Experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="outputs/seed_robustness")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Random seeds to evaluate",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)
    exp_logger = ExperimentLogger(args.output_dir)

    seed_results = []
    for seed in args.seeds:
        result = run_one_seed(config, device, seed)
        seed_results.append(result)

        exp_cfg = {
            "backbone": config["model"]["name"],
            "dataset": config["data"]["dataset"],
            "calibration_size": config["data"].get("calib_size", 200),
            "selected_layers": str(result["selected_layers"]),
            "expert_count": config.get("extraction", {}).get("num_experts", 4),
            "basis_rank": config.get("extraction", {}).get("shared_basis_rank") or -1,
            "router": config.get("router", {}).get("type", "linear"),
            "dispatch_backend": config.get("runtime", {}).get("dispatch_backend", "grouped"),
            "batch_size": config["data"].get("val_batch_size", 8),
            "seed": seed,
        }
        exp_results = {
            "top1": result["top1"],
            "top5": result["top5"],
            "p50_ms": result["p50_ms"],
            "p95_ms": result["p95_ms"],
            "mean_ms": result["mean_ms"],
            "std_ms": result["std_ms"],
            "peak_vram_mb": result["peak_vram_mb"],
            "load_skew": result["load_skew_mean"],
            "router_accuracy": result["router_accuracy_mean"],
        }
        exp_logger.log(exp_cfg, exp_results, experiment_id=f"seed_{seed}")

    # Aggregate statistics
    top1s = [r["top1"] for r in seed_results]
    p50s = [r["p50_ms"] for r in seed_results]
    skews = [r["load_skew_mean"] for r in seed_results]

    print("\n" + "=" * 60)
    print("Multi-Seed Robustness Summary")
    print("=" * 60)
    for r in seed_results:
        print(
            f"  Seed {r['seed']:4d}: Top-1={r['top1']:.2f}% "
            f"p50={r['p50_ms']:.2f}ms "
            f"router_acc={r['router_accuracy_mean']:.4f} "
            f"skew={r['load_skew_mean']:.4f}"
        )
    print("-" * 60)
    print(f"  Mean ± Std Top-1: {np.mean(top1s):.2f}% ± {np.std(top1s):.2f}%")
    print(f"  Mean ± Std p50:   {np.mean(p50s):.2f}ms ± {np.std(p50s):.2f}ms")
    print(f"  Mean ± Std skew:  {np.mean(skews):.4f} ± {np.std(skews):.4f}")
    print("=" * 60)

    # Save aggregated summary
    agg = {
        "seeds": args.seeds,
        "top1_mean": float(np.mean(top1s)),
        "top1_std": float(np.std(top1s)),
        "p50_ms_mean": float(np.mean(p50s)),
        "p50_ms_std": float(np.std(p50s)),
        "load_skew_mean": float(np.mean(skews)),
        "load_skew_std": float(np.std(skews)),
        "per_seed": seed_results,
    }
    agg_path = Path(args.output_dir) / "summaries" / "seed_robustness_aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    logger.info(f"Aggregate summary saved to {agg_path}")


if __name__ == "__main__":
    main()
