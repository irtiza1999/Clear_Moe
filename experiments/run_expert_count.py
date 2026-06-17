"""
Experiment 4: Expert-Count and Token-Coverage Study

Tests E ∈ {2, 4, 8, 16} experts and measures:
- Token distribution per expert
- Load skew, empty experts, router accuracy
- Top-1 accuracy, latency, active parameters

Run:
  python experiments/run_expert_count.py --config configs/imagenette_deit_s_direct.yaml
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
from clear_moe.router import fit_all_routers, compute_routing_stats
from clear_moe.scoring import score_layers, select_layers
from experiments.experiment_logger import ExperimentLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXPERT_COUNTS = [2, 4, 8, 16]


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
    parser = argparse.ArgumentParser(description="Expert Count Study")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="outputs/expert_count")
    parser.add_argument("--expert_counts", nargs="+", type=int, default=EXPERT_COUNTS)
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

    basis_rank = config.get("extraction", {}).get("shared_basis_rank") or None

    all_results = []

    for E in args.expert_counts:
        logger.info(f"\n{'='*60}\nE={E} experts\n{'='*60}")

        cfg = copy.deepcopy(config)
        cfg.setdefault("extraction", {})
        cfg["extraction"]["num_experts"] = E

        try:
            expertized = extract_all_experts(
                base_model, selected, activations,
                num_experts=E,
                shared_basis_rank=basis_rank,
                device=device,
            )
            routers = fit_all_routers(base_model, expertized, activations, cfg, device)

            moe_modules = build_moe_modules_from_expertized(expertized, routers, base_model, cfg)
            moe_model = copy.deepcopy(base_model)
            replace_ffn_with_moe(moe_model, selected_names, moe_modules)
            moe_model = moe_model.to(device)

            # Token distribution stats from calibration activations
            router_accs = []
            load_skews = []
            expert_loads_all = []
            empty_experts_per_layer = []
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

                    counts = [(preds_np == e).sum() for e in range(E)]
                    expert_loads_all.append(counts)
                    empty_experts_per_layer.append(sum(c == 0 for c in counts))

            all_counts = np.array(expert_loads_all)
            total_tokens = all_counts.sum(axis=1, keepdims=True)
            load_min = float(all_counts.min())
            load_max = float(all_counts.max())
            load_mean = float(all_counts.mean())
            load_std = float(all_counts.std())
            avg_empty = float(np.mean(empty_experts_per_layer))

            cls = evaluate_classification(moe_model, eval_loader, device)
            input_shape = (config["data"].get("val_batch_size", 8), 3, 224, 224)
            lat = measure_latency(moe_model, input_shape, device)
            mem = compute_memory_usage(moe_model, input_shape, device)
            params = count_active_params(moe_model)

            result = {
                "E": E,
                "top1": cls["top1"],
                "top5": cls["top5"],
                "p50_ms": lat["latency_p50_ms"],
                "p95_ms": lat["latency_p95_ms"],
                "peak_vram_mb": mem.get("peak_memory_MB", -1),
                "total_params": params["total_params"],
                "active_params": params["active_params"],
                "router_accuracy": float(np.mean(router_accs)) if router_accs else -1.0,
                "load_skew_mean": float(np.mean(load_skews)) if load_skews else -1.0,
                "expert_load_min": load_min,
                "expert_load_max": load_max,
                "expert_load_mean": load_mean,
                "expert_load_std": load_std,
                "avg_empty_experts": avg_empty,
            }
            all_results.append(result)

            exp_cfg = {
                "backbone": config["model"]["name"],
                "dataset": config["data"]["dataset"],
                "calibration_size": config["data"].get("calib_size", 200),
                "selected_layers": str(selected_indices),
                "expert_count": E,
                "basis_rank": basis_rank or -1,
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
                "load_skew": float(np.mean(load_skews)) if load_skews else -1.0,
                "router_accuracy": float(np.mean(router_accs)) if router_accs else -1.0,
            }
            exp_id = exp_logger.log(exp_cfg, exp_results, experiment_id=f"E{E}_seed{args.seed}")
            exp_logger.print_summary(exp_id)

        except Exception as e:
            logger.error(f"E={E} FAILED: {e}")
            all_results.append({"E": E, "top1": -1, "p50_ms": -1, "error": str(e)})

    print("\n" + "=" * 90)
    print("Expert Count Summary")
    print("=" * 90)
    print(f"{'E':>4} {'Top-1':>8} {'p50 ms':>8} {'RouterAcc':>10} {'Skew':>7} {'EmptyExp':>9} {'MaxLoad':>8}")
    print("-" * 90)
    for r in all_results:
        if r.get("top1", -1) < 0:
            print(f"{r['E']:>4}  FAILED: {r.get('error', '')[:60]}")
        else:
            print(
                f"{r['E']:>4} {r['top1']:>7.2f}% {r['p50_ms']:>8.2f} "
                f"{r['router_accuracy']:>10.4f} {r['load_skew_mean']:>7.4f} "
                f"{r['avg_empty_experts']:>9.2f} {r['expert_load_max']:>8.0f}"
            )
    print("=" * 90)
    logger.info(f"Expert count study complete. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
