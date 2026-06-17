"""
Experiment 2: Layer-Selection Policy Ablation (L0–L10)

Tests whether the proposed composite score selects better layers than simpler policies.

Policies:
  L0   No expertised layers (dense model)
  L1   Random k layers (repeat with 3 seeds)
  L2   First k layers
  L3   Last k layers
  L4   Alternating layers
  L5   Sparsity only
  L6   Clusterability only
  L7   Sensitivity only
  L8   Sparsity + clusterability
  L9   Clusterability - sensitivity
  L10  Full proposed composite score

All use the same: backbone, calibration images, router, dispatch backend, expert count.

Run:
  python experiments/run_layer_ablation.py --config configs/imagenette_deit_s_preprocessed.yaml
"""

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Dict, List

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
from clear_moe.models import (
    get_ffn_layers,
    get_model_input_transform,
    load_model,
    replace_ffn_with_moe,
)
from clear_moe.router import create_router, fit_router, compute_routing_stats
from clear_moe.scoring import score_layers, select_layers, LayerScore
from experiments.experiment_logger import ExperimentLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LAYER_POLICIES = {
    "L0": {"strategy": None, "label": "No expertised layers (dense)"},
    "L1": {"strategy": "random", "label": "Random k layers"},
    "L2": {"strategy": "first_k", "label": "First k layers"},
    "L3": {"strategy": "last_k", "label": "Last k layers"},
    "L4": {"strategy": "alternating", "label": "Alternating layers"},
    "L5": {"strategy": "sparsity_only", "label": "Sparsity only"},
    "L6": {"strategy": "clusterability_only", "label": "Clusterability only"},
    "L7": {"strategy": "sensitivity_only", "label": "Sensitivity only"},
    "L8": {"strategy": "sparsity_plus_clusterability", "label": "Sparsity + Clusterability"},
    "L9": {"strategy": "clusterability_minus_sensitivity", "label": "Clusterability - Sensitivity"},
    "L10": {"strategy": "score_top_k", "label": "Full composite score (proposed)"},
}


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


def apply_random_layer_selection(
    scores: List[LayerScore], k: int, seed: int
) -> List[LayerScore]:
    """Select k layers randomly."""
    rng = np.random.RandomState(seed)
    for s in scores:
        s.selected = False
    indices = rng.choice(len(scores), min(k, len(scores)), replace=False)
    for i in indices:
        scores[i].selected = True
    return scores


def build_moe_model(
    base_model,
    selected_scores,
    activations,
    config,
    device,
    seed,
):
    """Extract experts and fit routers for selected layers; return MoE model copy."""
    import copy

    selected_names = [s.name for s in selected_scores if s.selected]
    if not selected_names:
        return copy.deepcopy(base_model), {}, {}

    num_experts = config.get("extraction", {}).get("num_experts", 4)
    basis_rank = config.get("extraction", {}).get("shared_basis_rank") or None

    expertized = extract_all_experts(
        base_model, selected_scores, activations,
        num_experts=num_experts,
        shared_basis_rank=basis_rank,
        device=device,
    )

    # Fit routers
    from clear_moe.router import fit_all_routers
    routers = fit_all_routers(base_model, expertized, activations, config, device)

    from clear_moe.moe_builder import build_moe_modules_from_expertized
    moe_model = copy.deepcopy(base_model)
    moe_modules = build_moe_modules_from_expertized(expertized, routers, base_model, config)
    replace_ffn_with_moe(moe_model, selected_names, moe_modules)

    return moe_model, expertized, routers


def evaluate_moe_config(
    policy_id: str,
    base_model,
    selected_scores,
    activations,
    eval_loader,
    device,
    config,
    exp_logger: ExperimentLogger,
    seed: int,
) -> Dict:
    """Run evaluation for a single layer policy."""
    selected = [s for s in selected_scores if s.selected]
    selected_names = [s.name for s in selected]
    selected_indices = [s.layer_index for s in selected]

    # L1 is run with per-seed IDs like "L1_s42"; map back to base key for label lookup
    _policy_key = policy_id if policy_id in LAYER_POLICIES else policy_id.rsplit("_s", 1)[0]

    logger.info(f"\n{'='*60}")
    logger.info(f"{policy_id}: {LAYER_POLICIES[_policy_key]['label']}")
    logger.info(f"Selected layers ({len(selected)}): {selected_indices}")
    logger.info(f"{'='*60}")

    num_experts = config.get("extraction", {}).get("num_experts", 4)
    basis_rank = config.get("extraction", {}).get("shared_basis_rank") or None

    if not selected_names:
        # L0: no expertisation, use dense model
        model = copy.deepcopy(base_model)
        router_accuracy = 1.0
        load_skew = 1.0
    else:
        model, expertized, routers = build_moe_model(
            base_model, selected_scores, activations, config, device, seed
        )
        # Compute average router accuracy and load skew
        router_accs = []
        load_skews = []
        for name in selected_names:
            if name in routers and name in expertized:
                exp = expertized[name]
                router = routers[name]
                stats = compute_routing_stats(router, activations[name], device)
                load_skews.append(stats.get("load_balance_std", 0))
                # Router accuracy vs k-means labels
                with torch.no_grad():
                    acts = activations[name].to(device)
                    chunk = 2048
                    preds = []
                    for i in range(0, acts.shape[0], chunk):
                        idx, _, _ = router(acts[i:i+chunk])
                        if idx.dim() > 1:
                            idx = idx[:, 0]
                        preds.append(idx.cpu())
                    preds = torch.cat(preds).numpy()
                    labels = exp.cluster_labels
                    acc = (preds == labels).mean()
                    router_accs.append(acc)
        router_accuracy = float(np.mean(router_accs)) if router_accs else -1.0
        load_skew = float(np.mean(load_skews)) if load_skews else -1.0

    model = model.to(device)
    cls_results = evaluate_classification(model, eval_loader, device)
    input_shape = (
        config["data"].get("val_batch_size", 8),
        3,
        config["model"].get("input_size", 224),
        config["model"].get("input_size", 224),
    )
    lat_results = measure_latency(model, input_shape, device)
    mem_results = compute_memory_usage(model, input_shape, device)
    param_results = count_active_params(model)

    results = {
        "top1": cls_results["top1"],
        "top5": cls_results["top5"],
        "p50_ms": lat_results["latency_p50_ms"],
        "p95_ms": lat_results["latency_p95_ms"],
        "mean_ms": lat_results["latency_mean_ms"],
        "std_ms": lat_results["latency_std_ms"],
        "peak_vram_mb": mem_results.get("peak_memory_MB", -1),
        "load_skew": load_skew,
        "router_accuracy": router_accuracy,
        "total_params": param_results["total_params"],
        "active_params": param_results["active_params"],
    }

    exp_cfg = {
        "backbone": config["model"]["name"],
        "dataset": config["data"]["dataset"],
        "calibration_size": config["data"].get("calib_size", 200),
        "selected_layers": str(selected_indices),
        "expert_count": num_experts,
        "basis_rank": basis_rank or -1,
        "router": config.get("router", {}).get("type", "linear"),
        "dispatch_backend": config.get("runtime", {}).get("dispatch_backend", "grouped"),
        "batch_size": config["data"].get("val_batch_size", 8),
        "layer_policy": policy_id,
        "layer_policy_label": LAYER_POLICIES[_policy_key]["label"],
        "seed": seed,
    }

    exp_id = exp_logger.log(exp_cfg, results, experiment_id=f"{policy_id}_seed{seed}")
    exp_logger.print_summary(exp_id)
    # Save artifacts when available
    try:
        # Save model weights
        exp_logger.save_artifact(exp_id, "model", model.state_dict())
    except Exception:
        logger.debug("Could not save model state for artifact")

    try:
        # Save routing and expertized structures if present
        if 'expertized' in locals():
            exp_logger.save_artifact(exp_id, "expertized_layers", expertized)
        if 'routers' in locals():
            exp_logger.save_artifact(exp_id, "routing", routers)
    except Exception:
        logger.debug("Could not save expertized/routers artifacts")

    return results


def main():
    parser = argparse.ArgumentParser(description="Layer Selection Policy Ablation L0-L10")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="outputs/ablation_layer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(LAYER_POLICIES.keys()),
        help="Which policies to evaluate",
    )
    parser.add_argument(
        "--random_seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Seeds for L1 (random layer selection)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("seed", args.seed)
    set_seed(args.seed)

    device = get_device(config)
    exp_logger = ExperimentLogger(args.output_dir)

    transform = get_model_input_transform(config)
    calib_loader = get_calibration_dataloader(config, transform=transform, seed=args.seed)
    eval_loader = get_eval_dataloader(config, transform=transform)

    base_model = load_model(config).to(device)
    layer_infos = get_ffn_layers(base_model, config)
    layer_names = [info.name for info in layer_infos]
    total_layers = len(layer_infos)

    # Use k = total_layers // 2 for consistent comparison
    k = total_layers // 2

    # Collect activations once
    activations = collect_activations(base_model, calib_loader, layer_names, device)

    # Score layers (needed for scoring-based policies)
    scores = score_layers(activations, base_model, layer_infos, device, calib_loader)

    all_results = {}

    for policy_id in args.policies:
        if policy_id not in LAYER_POLICIES:
            logger.warning(f"Unknown policy {policy_id}, skipping")
            continue

        policy_cfg = LAYER_POLICIES[policy_id]
        strategy = policy_cfg["strategy"]

        if policy_id == "L1":
            # Random selection: repeat with multiple seeds
            for rseed in args.random_seeds:
                set_seed(rseed)
                import copy
                scores_copy = copy.deepcopy(scores)
                apply_random_layer_selection(scores_copy, k, seed=rseed)
                result = evaluate_moe_config(
                    f"L1_s{rseed}", base_model, scores_copy, activations,
                    eval_loader, device, config, exp_logger, rseed,
                )
                all_results[f"L1_seed{rseed}"] = result
        elif policy_id == "L0":
            import copy
            scores_copy = copy.deepcopy(scores)
            for s in scores_copy:
                s.selected = False
            result = evaluate_moe_config(
                "L0", base_model, scores_copy, activations,
                eval_loader, device, config, exp_logger, args.seed,
            )
            all_results["L0"] = result
        else:
            import copy
            scores_copy = copy.deepcopy(scores)
            scores_copy = select_layers(
                scores_copy,
                strategy=strategy,
                top_k=k,
                total_layers=total_layers,
            )
            result = evaluate_moe_config(
                policy_id, base_model, scores_copy, activations,
                eval_loader, device, config, exp_logger, args.seed,
            )
            all_results[policy_id] = result

    # Print comparison table
    print("\n" + "=" * 80)
    print("Layer Policy Comparison Summary")
    print("=" * 80)
    print(f"{'Policy':<12} {'Top-1':>8} {'p50 ms':>10} {'Active Params':>15} {'Router Acc':>12}")
    print("-" * 80)
    for pid, res in all_results.items():
        print(
            f"{pid:<12} {res.get('top1', -1):>7.2f}% "
            f"{res.get('p50_ms', -1):>10.2f} "
            f"{res.get('active_params', -1):>15,} "
            f"{res.get('router_accuracy', -1):>12.4f}"
        )
    print("=" * 80)

    logger.info(f"Layer ablation complete. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
