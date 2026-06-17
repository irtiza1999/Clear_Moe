"""
Experiment 8: Calibration-Set Sensitivity

Tests how performance varies with calibration set size.

Calibration sizes: N_cal ∈ {50, 100, 200, 500}
Uses multiple random subsets per size to confirm stability.

Run:
  python experiments/run_calibration_sensitivity.py \
      --config configs/imagenette_deit_s_preprocessed.yaml
"""

import argparse
import copy
import json
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
from clear_moe.metrics import evaluate_classification, measure_latency
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

CALIB_SIZES = [50, 100, 200, 500]
SUBSETS_PER_SIZE = {50: 3, 100: 3, 200: 2, 500: 1}


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


def run_calib_size(
    config: dict,
    base_model,
    device: torch.device,
    calib_size: int,
    seed: int,
    eval_loader,
    exp_logger: ExperimentLogger,
) -> Dict:
    """Run pipeline with a given calibration subset size and seed."""
    set_seed(seed)
    logger.info(f"  calib_size={calib_size}, seed={seed}")

    cfg = copy.deepcopy(config)
    cfg["data"]["calib_size"] = calib_size

    transform = get_model_input_transform(cfg)
    calib_loader = get_calibration_dataloader(cfg, transform=transform, seed=seed)

    layer_infos = get_ffn_layers(base_model, cfg)
    layer_names = [info.name for info in layer_infos]

    activations = collect_activations(base_model, calib_loader, layer_names, device)

    k = len(layer_infos) // 2
    scores = score_layers(activations, base_model, layer_infos, device, calib_loader)
    scores = select_layers(scores, strategy="score_top_k", top_k=k, total_layers=len(layer_infos))
    selected = [s for s in scores if s.selected]
    selected_names = [s.name for s in selected]

    num_experts = cfg.get("extraction", {}).get("num_experts", 4)
    basis_rank = cfg.get("extraction", {}).get("shared_basis_rank") or None

    # Clustering quality (multimodality = clusterability proxy)
    silhouette_scores = []
    for name in selected_names:
        s = next((sc for sc in scores if sc.name == name), None)
        if s:
            silhouette_scores.append(s.multimodality)

    expertized = extract_all_experts(
        base_model, selected, activations,
        num_experts=num_experts,
        shared_basis_rank=basis_rank,
        device=device,
    )
    routers = fit_all_routers(base_model, expertized, activations, cfg, device)

    # Build actual MoE model
    moe_modules = build_moe_modules_from_expertized(expertized, routers, base_model, cfg)
    moe_model = copy.deepcopy(base_model)
    replace_ffn_with_moe(moe_model, selected_names, moe_modules)
    moe_model = moe_model.to(device)

    # Router accuracy
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
    lat_results = measure_latency(
        moe_model,
        (cfg["data"].get("val_batch_size", 8), 3, 224, 224),
        device,
    )

    result = {
        "calib_size": calib_size,
        "seed": seed,
        "selected_layers": [s.layer_index for s in selected],
        "silhouette_mean": float(np.mean(silhouette_scores)) if silhouette_scores else -1.0,
        "router_accuracy": float(np.mean(router_accs)) if router_accs else -1.0,
        "load_skew": float(np.mean(load_skews)) if load_skews else -1.0,
        "top1": cls_results["top1"],
        "top5": cls_results["top5"],
        "p50_ms": lat_results["latency_p50_ms"],
        "p95_ms": lat_results["latency_p95_ms"],
    }

    exp_cfg = {
        "backbone": cfg["model"]["name"],
        "dataset": cfg["data"]["dataset"],
        "calibration_size": calib_size,
        "selected_layers": str(result["selected_layers"]),
        "expert_count": num_experts,
        "basis_rank": basis_rank or -1,
        "router": cfg.get("router", {}).get("type", "linear"),
        "dispatch_backend": cfg.get("runtime", {}).get("dispatch_backend", "grouped"),
        "batch_size": cfg["data"].get("val_batch_size", 8),
        "seed": seed,
    }
    exp_results = {
        "top1": cls_results["top1"],
        "top5": cls_results["top5"],
        "p50_ms": lat_results["latency_p50_ms"],
        "p95_ms": lat_results["latency_p95_ms"],
        "mean_ms": lat_results["latency_mean_ms"],
        "std_ms": lat_results["latency_std_ms"],
        "peak_vram_mb": -1,
        "load_skew": result["load_skew"],
        "router_accuracy": result["router_accuracy"],
    }
    exp_logger.log(
        exp_cfg, exp_results,
        experiment_id=f"calib_N{calib_size}_seed{seed}",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Calibration Size Sensitivity")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="outputs/calibration_sensitivity")
    parser.add_argument(
        "--calib_sizes",
        nargs="+",
        type=int,
        default=CALIB_SIZES,
    )
    parser.add_argument("--base_seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)
    exp_logger = ExperimentLogger(args.output_dir)

    transform = get_model_input_transform(config)
    eval_loader = get_eval_dataloader(config, transform=transform)
    base_model = load_model(config).to(device)

    all_results = []

    for calib_size in args.calib_sizes:
        n_subsets = SUBSETS_PER_SIZE.get(calib_size, 1)
        size_results = []

        for i in range(n_subsets):
            seed = args.base_seed + i * 1000
            result = run_calib_size(config, base_model, device, calib_size, seed, eval_loader, exp_logger)
            size_results.append(result)
            all_results.append(result)

        top1s = [r["top1"] for r in size_results]
        sils = [r["silhouette_mean"] for r in size_results]
        router_accs = [r["router_accuracy"] for r in size_results]
        logger.info(
            f"N_cal={calib_size}: Top-1={np.mean(top1s):.2f}±{np.std(top1s):.2f}% "
            f"Silhouette={np.mean(sils):.3f}±{np.std(sils):.3f} "
            f"RouterAcc={np.mean(router_accs):.4f}±{np.std(router_accs):.4f}"
        )

    # Summary table
    print("\n" + "=" * 80)
    print("Calibration Size Sensitivity Summary")
    print("=" * 80)
    print(f"{'N_cal':>8} {'Seed':>6} {'Top-1':>8} {'Silhouette':>12} {'RouterAcc':>12} {'Skew':>8}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['calib_size']:>8} {r['seed']:>6} {r['top1']:>7.2f}% "
            f"{r['silhouette_mean']:>12.4f} {r['router_accuracy']:>12.4f} {r['load_skew']:>8.4f}"
        )
    print("=" * 80)

    # Save
    out_path = Path(args.output_dir) / "summaries" / "calibration_sensitivity_all.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
