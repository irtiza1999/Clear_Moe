"""
Experiment 3: Basis-Rank Study

Tests how shared SVD basis rank affects accuracy, latency, and active parameters.

Ranks: r ∈ {16, 32, 64, 96, 128, 192, 256}

Run:
  python experiments/run_rank_sweep.py --config configs/imagenette_deit_s_direct.yaml
"""

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from clear_moe.calibration import collect_activations, get_calibration_dataloader, get_eval_dataloader
from clear_moe.extraction import extract_all_experts
from clear_moe.metrics import count_active_params, evaluate_classification, measure_latency, compute_memory_usage
from clear_moe.models import get_ffn_layers, get_model_input_transform, load_model, replace_ffn_with_moe
from clear_moe.moe_builder import build_moe_modules_from_expertized
from clear_moe.router import fit_all_routers
from clear_moe.scoring import score_layers, select_layers
from experiments.experiment_logger import ExperimentLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RANKS = [16, 32, 64, 96, 128, 192, 256]


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
    d = config.get("device", "cuda")
    if d == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(d)


def main():
    parser = argparse.ArgumentParser(description="Basis Rank Sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="outputs/rank_sweep")
    parser.add_argument("--ranks", nargs="+", type=int, default=RANKS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = get_device(config)
    exp_logger = ExperimentLogger(args.output_dir)

    transform = get_model_input_transform(config)
    calib_loader = get_calibration_dataloader(config, transform=transform, seed=args.seed)
    eval_loader = get_eval_dataloader(config, transform=transform)

    base_model = load_model(config).to(device)
    layer_infos = get_ffn_layers(base_model, config)
    layer_names = [info.name for info in layer_infos]
    k = len(layer_infos) // 2

    activations = collect_activations(base_model, calib_loader, layer_names, device)
    scores = score_layers(activations, base_model, layer_infos, device, calib_loader)
    scores = select_layers(scores, strategy="score_top_k", top_k=k, total_layers=len(layer_infos))
    selected = [s for s in scores if s.selected]
    selected_names = [s.name for s in selected]
    selected_indices = [s.layer_index for s in selected]
    logger.info(f"Selected layers: {selected_indices}")

    num_experts = config.get("extraction", {}).get("num_experts", 4)

    all_results = []

    for rank in args.ranks:
        logger.info(f"\n{'='*60}\nRank r={rank}\n{'='*60}")

        cfg = copy.deepcopy(config)
        cfg.setdefault("extraction", {})
        cfg["extraction"]["shared_basis_rank"] = rank

        expertized = extract_all_experts(
            base_model, selected, activations,
            num_experts=num_experts,
            shared_basis_rank=rank,
            device=device,
        )
        routers = fit_all_routers(base_model, expertized, activations, cfg, device)

        moe_modules = build_moe_modules_from_expertized(expertized, routers, base_model, cfg)
        moe_model = copy.deepcopy(base_model)
        replace_ffn_with_moe(moe_model, selected_names, moe_modules)
        moe_model = moe_model.to(device)

        # Weight reconstruction error: average over selected layers
        recon_errors = []
        for name in selected_names:
            if name in expertized:
                exp = expertized[name]
                if hasattr(exp, "reconstruction_error"):
                    recon_errors.append(exp.reconstruction_error)

        cls = evaluate_classification(moe_model, eval_loader, device)
        input_shape = (config["data"].get("val_batch_size", 8), 3, 224, 224)
        lat = measure_latency(moe_model, input_shape, device)
        mem = compute_memory_usage(moe_model, input_shape, device)
        params = count_active_params(moe_model)

        result = {
            "rank": rank,
            "top1": cls["top1"],
            "top5": cls["top5"],
            "p50_ms": lat["latency_p50_ms"],
            "p95_ms": lat["latency_p95_ms"],
            "peak_vram_mb": mem.get("peak_memory_MB", -1),
            "total_params": params["total_params"],
            "active_params": params["active_params"],
            "recon_error_mean": float(np.mean(recon_errors)) if recon_errors else -1.0,
        }
        all_results.append(result)

        exp_cfg = {
            "backbone": config["model"]["name"],
            "dataset": config["data"]["dataset"],
            "calibration_size": config["data"].get("calib_size", 200),
            "selected_layers": str(selected_indices),
            "expert_count": num_experts,
            "basis_rank": rank,
            "router": config.get("router", {}).get("type", "linear"),
            "dispatch_backend": config.get("runtime", {}).get("dispatch_backend", "grouped"),
            "batch_size": config["data"].get("val_batch_size", 8),
            "seed": args.seed,
        }
        exp_results = {
            "top1": cls["top1"],
            "top5": cls["top5"],
            "p50_ms": lat["latency_p50_ms"],
            "p95_ms": lat["latency_p95_ms"],
            "mean_ms": lat["latency_mean_ms"],
            "std_ms": lat["latency_std_ms"],
            "peak_vram_mb": mem.get("peak_memory_MB", -1),
            "load_skew": -1,
            "router_accuracy": -1,
        }
        exp_id = exp_logger.log(exp_cfg, exp_results, experiment_id=f"rank_{rank}_seed{args.seed}")
        exp_logger.print_summary(exp_id)

    print("\n" + "=" * 70)
    print("Rank Sweep Summary")
    print("=" * 70)
    print(f"{'Rank':>6} {'Top-1':>8} {'p50 ms':>8} {'Active Params':>15} {'ReconErr':>10}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['rank']:>6} {r['top1']:>7.2f}% {r['p50_ms']:>8.2f} "
            f"{r['active_params']:>15,} {r['recon_error_mean']:>10.4f}"
        )
    print("=" * 70)
    logger.info(f"Rank sweep complete. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
