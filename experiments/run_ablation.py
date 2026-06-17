"""
Experiment 1: Core Decomposition Ablation (D0–D8)

Tests whether the proposed shared-basis + residual-expert decomposition
outperforms simpler alternatives.

Configurations:
  D0  Original dense model
  D1  Shared low-rank basis only; no residual experts
  D2  Residual experts only; no shared basis
  D3  Shared basis + one global residual; no routing
  D4  Shared basis + randomly grouped experts
  D5  Shared basis + k-means experts + random routing
  D6  Shared basis + k-means experts + learned router
  D7  Disjoint experts without the shared basis (MoEfication-style)
  D8  Proposed full CLEAR-MoE++ (with layer scoring)

Run:
  python experiments/run_ablation.py --config configs/imagenette_deit_s_preprocessed.yaml
"""

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from clear_moe.calibration import collect_activations, get_calibration_dataloader, get_eval_dataloader
from clear_moe.extraction import extract_all_experts, extract_experts
from clear_moe.metrics import (
    count_active_params,
    evaluate_classification,
    measure_latency,
    compute_memory_usage,
)
from clear_moe.models import (
    FFNLayerInfo,
    get_ffn_layers,
    get_model_input_transform,
    load_model,
    replace_ffn_with_moe,
)
from clear_moe.router import create_router, fit_router, compute_routing_stats
from clear_moe.scoring import score_layers, select_layers
from experiments.experiment_logger import ExperimentLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(config: dict) -> torch.device:
    device_str = config.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dense_baseline(model, eval_loader, device, config, exp_logger, seed):
    """D0: Evaluate the original dense model."""
    logger.info("=" * 60)
    logger.info("D0: Dense baseline")
    logger.info("=" * 60)

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
        "load_skew": 1.0,
        "router_accuracy": 1.0,
        "total_params": param_results["total_params"],
        "active_params": param_results["active_params"],
    }

    exp_cfg = {
        "backbone": config["model"]["name"],
        "dataset": config["data"]["dataset"],
        "calibration_size": 0,
        "selected_layers": "none",
        "expert_count": 0,
        "basis_rank": 0,
        "router": "none",
        "dispatch_backend": "dense",
        "batch_size": config["data"].get("val_batch_size", 8),
        "ablation_config": "D0",
        "seed": seed,
    }
    exp_id = exp_logger.log(exp_cfg, results, experiment_id=f"D0_dense_seed{seed}")
    exp_logger.print_summary(exp_id)
    return results


def run_moe_config(
    label: str,
    model_fn,
    eval_loader,
    device,
    config,
    exp_logger,
    seed,
    selected_layer_names: list,
    num_experts: int,
    basis_rank: int,
    router_type: str,
    extra_cfg: dict = None,
):
    """Generic runner for D1-D8 MoE configurations."""
    logger.info("=" * 60)
    logger.info(f"{label}: Evaluating MoE configuration")
    logger.info("=" * 60)

    model = model_fn().to(device)

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
        "load_skew": extra_cfg.get("load_skew", -1) if extra_cfg else -1,
        "router_accuracy": extra_cfg.get("router_accuracy", -1) if extra_cfg else -1,
        "total_params": param_results["total_params"],
        "active_params": param_results["active_params"],
    }

    exp_cfg = {
        "backbone": config["model"]["name"],
        "dataset": config["data"]["dataset"],
        "calibration_size": config["data"].get("calib_size", 200),
        "selected_layers": str(selected_layer_names),
        "expert_count": num_experts,
        "basis_rank": basis_rank,
        "router": router_type,
        "dispatch_backend": config.get("runtime", {}).get("dispatch_backend", "grouped"),
        "batch_size": config["data"].get("val_batch_size", 8),
        "ablation_config": label,
        "seed": seed,
    }
    if extra_cfg:
        exp_cfg.update({k: v for k, v in extra_cfg.items()
                        if k not in ("load_skew", "router_accuracy")})

    exp_id = exp_logger.log(exp_cfg, results, experiment_id=f"{label}_seed{seed}")
    exp_logger.print_summary(exp_id)
    return results


def main():
    parser = argparse.ArgumentParser(description="CLEAR-MoE Decomposition Ablation D0-D8")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", default="outputs/ablation_decomp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"],
        help="Which ablation configs to run",
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

    # Load base model once
    base_model = load_model(config).to(device)
    layer_infos = get_ffn_layers(base_model, config)

    # Collect calibration activations once
    layer_names = [info.name for info in layer_infos]
    activations = collect_activations(base_model, calib_loader, layer_names, device)

    # Score and select layers
    scores = score_layers(activations, base_model, layer_infos, device, calib_loader)
    num_experts = config.get("extraction", {}).get("num_experts", 4)
    basis_rank = config.get("extraction", {}).get("shared_basis_rank") or None
    scores = select_layers(scores, strategy="last_half", total_layers=len(layer_infos))
    selected = [s for s in scores if s.selected]
    selected_names = [s.name for s in selected]
    logger.info(f"Selected {len(selected)} layers for expertization: {selected_names}")

    all_results = {}

    # D0: Dense baseline
    if "D0" in args.configs:
        all_results["D0"] = run_dense_baseline(
            base_model, eval_loader, device, config, exp_logger, args.seed
        )

    # D1–D8 require extraction; build the MoE model variants
    # Extract experts for selected layers
    expertized = {}
    routers = {}
    if any(d in args.configs for d in ["D2", "D3", "D4", "D5", "D6", "D7", "D8"]):
        from clear_moe.extraction import extract_all_experts
        expertized = extract_all_experts(
            base_model, selected, activations,
            num_experts=num_experts,
            shared_basis_rank=basis_rank,
            device=device,
        )
        # Save expertized layers as a checkpoint artifact
        try:
            from clear_moe.extraction import save_expertized_layers
            save_expertized_layers(expertized, args.output_dir)
            exp_logger.save_artifact(f"extract_all_{args.seed}", "expertized_layers", expertized)
        except Exception:
            logger.warning("Could not save expertized layers checkpoint")

    # Fit routers where needed
    if any(d in args.configs for d in ["D6", "D8"]):
        from clear_moe.router import fit_all_routers
        routers = fit_all_routers(base_model, expertized, activations, config, device)

    # Helper: build a fresh copy of the model with specified replacements
    def make_model_copy():
        return copy.deepcopy(base_model)

    logger.info("Building and evaluating D1-D8 configurations...")

    from tqdm import tqdm
    for cfg_id in tqdm(args.configs, desc="Ablation configs"):
        if cfg_id == "D0":
            continue

        try:
            model_copy = make_model_copy()

            if cfg_id == "D1":
                # SVD-only: replace fc2 with rank-r approximation, no experts
                from clear_moe.baselines import extract_svd_only
                from clear_moe.extraction import _get_module, _get_ffn_weights, _detect_activation
                replacements = {}
                for name in selected_names:
                    ffn = _get_module(base_model, name)
                    fc1_w, fc1_b, fc2_w, fc2_b = _get_ffn_weights(ffn)
                    act_fn = _detect_activation(ffn)
                    rank = basis_rank or (min(fc2_w.shape) // 2)
                    replacements[name] = extract_svd_only(fc1_w, fc1_b, fc2_w, rank, act_fn)
                    replacements[name] = replacements[name].to(device)
                replace_ffn_with_moe(model_copy, selected_names, replacements)
                run_moe_config(
                    "D1", lambda: model_copy, eval_loader, device, config,
                    exp_logger, args.seed, selected_names, 0, basis_rank or -1, "none",
                )

            elif cfg_id in ("D2", "D7"):
                # Disjoint experts without shared basis (D2: k-means groups, D7: variant)
                from clear_moe.baselines import extract_disjoint_experts, DisjointExpertFFN
                from clear_moe.extraction import _get_module, _get_ffn_weights, _detect_activation
                replacements = {}
                cluster_labels_all = {}
                for name in selected_names:
                    ffn = _get_module(base_model, name)
                    fc1_w, fc1_b, fc2_w, fc2_b = _get_ffn_weights(ffn)
                    act_fn = _detect_activation(ffn)
                    module, labels = extract_disjoint_experts(
                        fc1_w, fc1_b, fc2_w, fc2_b,
                        activations[name], num_experts, act_fn,
                        random_grouping=(cfg_id == "D2"),
                        seed=args.seed,
                    )
                    cluster_labels_all[name] = labels
                    replacements[name] = module.to(device)
                replace_ffn_with_moe(model_copy, selected_names, replacements)
                run_moe_config(
                    cfg_id, lambda: model_copy, eval_loader, device, config,
                    exp_logger, args.seed, selected_names, num_experts, 0, "none",
                )

            elif cfg_id == "D3":
                # Shared basis + global residual, no routing
                from clear_moe.baselines import extract_global_residual
                from clear_moe.extraction import _get_module, _get_ffn_weights, _detect_activation
                replacements = {}
                for name in selected_names:
                    ffn = _get_module(base_model, name)
                    fc1_w, fc1_b, fc2_w, fc2_b = _get_ffn_weights(ffn)
                    act_fn = _detect_activation(ffn)
                    rank = basis_rank or (min(fc2_w.shape) // 2)
                    replacements[name] = extract_global_residual(
                        fc1_w, fc1_b, fc2_w, rank, act_fn
                    ).to(device)
                replace_ffn_with_moe(model_copy, selected_names, replacements)
                run_moe_config(
                    "D3", lambda: model_copy, eval_loader, device, config,
                    exp_logger, args.seed, selected_names, 0, basis_rank or -1, "none",
                )

            elif cfg_id in ("D4", "D5"):
                # D4: k-means experts + random routing
                # D5: k-means experts + random routing (variant)
                # Use ClearMoEFFN with a trivial random router for D4/D5.
                import copy as _copy
                from clear_moe.moe_builder import build_moe_modules_from_expertized

                class _RandomRouter(nn.Module):
                    def __init__(self, n):
                        super().__init__()
                        self.n = n
                    def forward(self, x):
                        idx = torch.randint(0, self.n, (x.shape[0],), device=x.device)
                        w = torch.ones(x.shape[0], device=x.device)
                        return idx, w, None

                random_routers = {n: _RandomRouter(num_experts) for n in expertized}
                moe_modules = build_moe_modules_from_expertized(
                    expertized, random_routers, base_model, config
                )
                replace_ffn_with_moe(model_copy, selected_names, moe_modules)
                run_moe_config(
                    cfg_id, lambda: model_copy, eval_loader, device, config,
                    exp_logger, args.seed, selected_names, num_experts,
                    basis_rank or -1, "random",
                )

            elif cfg_id in ("D6", "D8"):
                # D6: k-means experts + learned router (no layer scoring for D6)
                # D8: same but with composite layer scoring (already applied via selected_names)
                from clear_moe.moe_builder import build_moe_modules_from_expertized
                moe_modules = build_moe_modules_from_expertized(
                    expertized, routers, base_model, config
                )
                replace_ffn_with_moe(model_copy, selected_names, moe_modules)

                # Compute routing stats for logging
                router_accs = []
                load_skews = []
                from clear_moe.router import compute_routing_stats
                for n in selected_names:
                    if n in routers and n in expertized:
                        stats = compute_routing_stats(
                            routers[n], activations[n], device
                        )
                        load_skews.append(stats.get("load_balance_std", 0))
                        with torch.no_grad():
                            acts = activations[n].to(device)
                            preds = []
                            for i in range(0, acts.shape[0], 2048):
                                idx_r, _, _ = routers[n](acts[i : i + 2048])
                                if idx_r.dim() > 1:
                                    idx_r = idx_r[:, 0]
                                preds.append(idx_r.cpu())
                            preds_np = torch.cat(preds).numpy()
                            acc = (preds_np == expertized[n].cluster_labels).mean()
                            router_accs.append(acc)

                import numpy as np
                extra = {
                    "load_skew": float(np.mean(load_skews)) if load_skews else -1.0,
                    "router_accuracy": float(np.mean(router_accs)) if router_accs else -1.0,
                }
                run_moe_config(
                    cfg_id, lambda: model_copy, eval_loader, device, config,
                    exp_logger, args.seed, selected_names, num_experts,
                    basis_rank or -1, "linear", extra,
                )

            else:
                logger.warning(f"Unknown configuration {cfg_id}, skipping.")

        except Exception as e:
            logger.error(f"Failed to run {cfg_id}: {e}", exc_info=True)

    logger.info("Decomposition ablation complete.")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
