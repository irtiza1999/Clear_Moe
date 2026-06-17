"""
Experiment 5: Router Comparison

Compares implemented router types on the same backbone, layers, and eval split:
  - linear: Linear projection gate (cheapest)
  - mlp: 1-hidden-layer MLP gate
  - adaptive: Confidence-adaptive top-1/top-2 gate

Reports per-router: accuracy, latency, router accuracy vs k-means labels,
load skew, entropy, router parameter count.

Run:
  python experiments/run_router_comparison.py --config configs/imagenette_deit_s_direct.yaml
"""

import argparse
import copy
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from clear_moe.calibration import collect_activations, get_calibration_dataloader, get_eval_dataloader
from clear_moe.extraction import extract_all_experts
from clear_moe.metrics import count_active_params, evaluate_classification, measure_latency
from clear_moe.models import get_ffn_layers, get_model_input_transform, load_model, replace_ffn_with_moe
from clear_moe.moe_builder import build_moe_modules_from_expertized
from clear_moe.router import fit_all_routers, compute_routing_stats
from clear_moe.scoring import score_layers, select_layers
from experiments.experiment_logger import ExperimentLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ROUTER_TYPES = ["linear", "mlp", "adaptive"]


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


def count_router_params(routers: dict) -> int:
    total = 0
    for r in routers.values():
        if hasattr(r, "parameters"):
            total += sum(p.numel() for p in r.parameters())
    return total


def expert_usage_entropy(preds_np: np.ndarray, num_experts: int) -> float:
    counts = np.array([(preds_np == e).sum() for e in range(num_experts)], dtype=float)
    probs = counts / (counts.sum() + 1e-8)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def main():
    parser = argparse.ArgumentParser(description="Router Comparison")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="outputs/router_comparison")
    parser.add_argument("--router_types", nargs="+", default=ROUTER_TYPES, choices=ROUTER_TYPES)
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
    num_experts = config.get("extraction", {}).get("num_experts", 4)
    basis_rank = config.get("extraction", {}).get("shared_basis_rank") or None

    activations = collect_activations(base_model, calib_loader, layer_names, device)
    scores = score_layers(activations, base_model, layer_infos, device, calib_loader)
    scores = select_layers(scores, strategy="score_top_k", top_k=k, total_layers=len(layer_infos))
    selected = [s for s in scores if s.selected]
    selected_names = [s.name for s in selected]
    selected_indices = [s.layer_index for s in selected]
    logger.info(f"Selected layers: {selected_indices}")

    # Extract experts once; reuse for all router types
    expertized = extract_all_experts(
        base_model, selected, activations,
        num_experts=num_experts,
        shared_basis_rank=basis_rank,
        device=device,
    )

    all_results = []

    for router_type in args.router_types:
        logger.info(f"\n{'='*60}\nRouter: {router_type}\n{'='*60}")

        cfg = copy.deepcopy(config)
        cfg.setdefault("router", {})
        cfg["router"]["type"] = router_type

        try:
            set_seed(args.seed)
            routers = fit_all_routers(base_model, expertized, activations, cfg, device)

            moe_modules = build_moe_modules_from_expertized(expertized, routers, base_model, cfg)
            moe_model = copy.deepcopy(base_model)
            replace_ffn_with_moe(moe_model, selected_names, moe_modules)
            moe_model = moe_model.to(device)

            router_accs = []
            load_skews = []
            entropies = []
            max_loads = []
            min_loads = []
            for name in selected_names:
                if name in routers and name in expertized:
                    exp_obj = expertized[name]
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
                        acc = (preds_np == exp_obj.cluster_labels).mean()
                        router_accs.append(acc)

                    counts = [(preds_np == e).sum() for e in range(num_experts)]
                    max_loads.append(max(counts))
                    min_loads.append(min(counts))
                    entropies.append(expert_usage_entropy(preds_np, num_experts))

            cls = evaluate_classification(moe_model, eval_loader, device)
            input_shape = (config["data"].get("val_batch_size", 8), 3, 224, 224)
            lat = measure_latency(moe_model, input_shape, device)
            router_param_count = count_router_params(routers)

            result = {
                "router_type": router_type,
                "top1": cls["top1"],
                "top5": cls["top5"],
                "p50_ms": lat["latency_p50_ms"],
                "p95_ms": lat["latency_p95_ms"],
                "router_accuracy": float(np.mean(router_accs)) if router_accs else -1.0,
                "load_skew": float(np.mean(load_skews)) if load_skews else -1.0,
                "entropy": float(np.mean(entropies)) if entropies else -1.0,
                "max_load": float(np.mean(max_loads)) if max_loads else -1.0,
                "min_load": float(np.mean(min_loads)) if min_loads else -1.0,
                "router_params": router_param_count,
            }
            all_results.append(result)

            exp_cfg = {
                "backbone": config["model"]["name"],
                "dataset": config["data"]["dataset"],
                "calibration_size": config["data"].get("calib_size", 200),
                "selected_layers": str(selected_indices),
                "expert_count": num_experts,
                "basis_rank": basis_rank or -1,
                "router": router_type,
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
                "peak_vram_mb": -1,
                "load_skew": float(np.mean(load_skews)) if load_skews else -1.0,
                "router_accuracy": float(np.mean(router_accs)) if router_accs else -1.0,
            }
            exp_id = exp_logger.log(exp_cfg, exp_results, experiment_id=f"router_{router_type}_seed{args.seed}")
            exp_logger.print_summary(exp_id)

        except Exception as e:
            logger.error(f"Router {router_type} FAILED: {e}")
            all_results.append({"router_type": router_type, "top1": -1, "error": str(e)})

    print("\n" + "=" * 90)
    print("Router Comparison Summary")
    print("=" * 90)
    print(f"{'Router':>10} {'Top-1':>8} {'p50 ms':>8} {'RouterAcc':>10} {'Skew':>7} {'Entropy':>8} {'Params':>8}")
    print("-" * 90)
    for r in all_results:
        if r.get("top1", -1) < 0:
            print(f"{r['router_type']:>10}  FAILED: {r.get('error', '')[:60]}")
        else:
            print(
                f"{r['router_type']:>10} {r['top1']:>7.2f}% {r['p50_ms']:>8.2f} "
                f"{r['router_accuracy']:>10.4f} {r['load_skew']:>7.4f} "
                f"{r['entropy']:>8.4f} {r['router_params']:>8}"
            )
    print("=" * 90)
    logger.info(f"Router comparison complete. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
