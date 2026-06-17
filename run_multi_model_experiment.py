"""
CLEAR-MoE Multi-Model Cross-Backbone Experiment
================================================
Runs the full CLEAR-MoE pipeline on FIVE backbones to validate that the
shared-SVD-basis finding is backbone-agnostic:

    1. DeiT-Tiny/16   (smallest DeiT model)
    2. DeiT-Small/16  (original paper model)
    3. ViT-Small/16   (smaller ViT model)
    4. DeiT-Base/16   (wider FFN: d_ffn=3072 vs 1536)
    5. ViT-B/16       (canonical ViT, different pre-training)

For each model, runs:
    D0  Dense baseline
    D3  Shared basis + global residual (exact reconstruction; upper bound)
    D5  Shared basis + k-means + random routing (our core claim)
    D6  Shared basis + k-means + linear router (full CLEAR-MoE)

Results are saved to:
    outputs/multi_model/<model_tag>/results.json   â€” per-run numbers
    outputs/multi_model/summary_table.csv          â€” cross-model comparison
    outputs/multi_model/summary_table.txt          â€” human-readable table

Usage:
    python run_multi_model_experiment.py
    python run_multi_model_experiment.py --models deit_small deit_base
    python run_multi_model_experiment.py --skip_dense   # skip D0 if already done
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# â”€â”€ project imports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from clear_moe.calibration import collect_activations, get_calibration_dataloader, get_eval_dataloader
from clear_moe.extraction import extract_all_experts
from clear_moe.metrics import evaluate_classification, measure_latency
from clear_moe.models import get_ffn_layers, get_model_input_transform, load_model, replace_ffn_with_moe
from clear_moe.scoring import score_layers, select_layers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s â€” %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multi_model")

# â”€â”€ model registry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MODEL_REGISTRY = {
    "deit_tiny": {
        "tag":         "DeiT-T/16",
        "timm_name":   "deit_tiny_patch16_224",
        "d":           192,
        "d_ffn":       768,
        "n_blocks":    12,
        "val_batch":   8,
        "calib_batch": 8,
    },
    "deit_small": {
        "tag":         "DeiT-S/16",
        "timm_name":   "deit_small_patch16_224",
        "d":           384,
        "d_ffn":       1536,
        "n_blocks":    12,
        "val_batch":   8,
        "calib_batch": 8,
    },
    "vit_small": {
        "tag":         "ViT-S/16",
        "timm_name":   "vit_small_patch16_224",
        "d":           384,
        "d_ffn":       1536,
        "n_blocks":    12,
        "val_batch":   8,
        "calib_batch": 8,
    },
    "deit_base": {
        "tag":         "DeiT-B/16",
        "timm_name":   "deit_base_patch16_224",
        "d":           768,
        "d_ffn":       3072,
        "n_blocks":    12,
        "val_batch":   4,   # larger model â†’ smaller batch to fit 4GB VRAM
        "calib_batch": 4,
    },
    "vit_base": {
        "tag":         "ViT-B/16",
        "timm_name":   "vit_base_patch16_224",
        "d":           768,
        "d_ffn":       3072,
        "n_blocks":    12,
        "val_batch":   4,
        "calib_batch": 4,
    },
}

# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(model_key: str, data_dir: str, calib_size: int, seed: int) -> dict:
    """Build a config dict compatible with the existing clear_moe API."""
    spec = MODEL_REGISTRY[model_key]
    return {
        "model": {
            "name":        spec["timm_name"],
            "source":      "timm",
            "pretrained":  True,
            "num_classes": 10,
            "input_size":  224,
        },
        "data": {
            "task":          "classification",
            "dataset":       "imagenette",
            "data_dir":      data_dir,
            "calib_size":    calib_size,
            "val_batch_size": spec["val_batch"],
            "calib_batch":   spec["calib_batch"],
            "num_workers":   2,
            "pin_memory":    True,
        },
        "extraction": {
            "num_experts":       4,
            "layer_selection":   "composite",
            "method":            "clear_moe",
            "shared_basis_rank": None,
        },
        "router": {
            "type":            "linear",
            "hidden_dim":      64,
            "lr":              1e-3,
            "epochs":          5,
            "weight_decay":    0.01,
            "scheduler":       "cosine",
            "balance_coeff":   0.01,
        },
        "runtime": {
            "dtype":            "float32",
            "routing":          "top1",
            "route_sorting":    True,
            "dispatch_backend": "grouped",
        },
        "seed":   seed,
        "device": "cuda",
    }


def save_results(results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    # also overwrite the stable "latest" file
    latest = out_dir / "results_latest.json"
    with open(latest, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results saved â†’ {latest}")
    return out_path


def print_row(label, top1, delta, p50, router_acc=None):
    ra = f"{router_acc:.3f}" if router_acc is not None else "  â€”  "
    print(f"  {label:<10} {top1:>7.2f}%  {delta:>+6.2f}pp  {p50:>8.1f} ms  rtr={ra}")


# â”€â”€ core experiment per backbone â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_backbone(
    model_key: str,
    data_dir: str,
    output_root: Path,
    seed: int,
    calib_size: int,
    configs_to_run: list,
    device: torch.device,
):
    spec   = MODEL_REGISTRY[model_key]
    tag    = spec["tag"]
    out_dir = output_root / model_key

    print()
    print("=" * 70)
    print(f"  BACKBONE: {tag}  ({spec['timm_name']})")
    print(f"  d={spec['d']}  d_ffn={spec['d_ffn']}  blocks={spec['n_blocks']}")
    print("=" * 70)

    config = make_config(model_key, data_dir, calib_size, seed)
    set_seed(seed)

    # â”€â”€ data loaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[1/6] Building data loadersâ€¦")
    transform    = get_model_input_transform(config)
    calib_loader = get_calibration_dataloader(config, transform=transform, seed=seed)
    eval_loader  = get_eval_dataloader(config, transform=transform)
    input_shape  = (config["data"]["val_batch_size"], 3, 224, 224)

    # â”€â”€ load model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"[2/6] Loading {tag} (timm pretrained)â€¦")
    t0 = time.time()
    base_model = load_model(config).to(device)
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"      {total_params/1e6:.1f}M params  ({time.time()-t0:.1f}s)")

    # â”€â”€ collect activations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"[3/6] Collecting calibration activations ({calib_size} images)â€¦")
    layer_infos  = get_ffn_layers(base_model, config)
    layer_names  = [info.name for info in layer_infos]

    t0 = time.time()
    activations = collect_activations(base_model, calib_loader, layer_names, device)
    print(f"      Done in {time.time()-t0:.1f}s. Layers captured: {len(activations)}")

    # â”€â”€ score and select layers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("[4/6] Scoring layers (sparsity / clusterability / sensitivity)â€¦")
    scores = score_layers(activations, base_model, layer_infos, device, calib_loader)
    # "score_top_k" = top-k layers by composite score (top-half by default)
    scores = select_layers(scores, strategy="score_top_k", total_layers=len(layer_infos))
    selected = [s for s in scores if s.selected]
    selected_names = [s.name for s in selected]
    num_experts = 4

    print(f"      Selected {len(selected_names)}/{len(layer_infos)} layers:")
    for s in selected:
        print(f"        [{s.layer_index:2d}] {s.name}  composite={s.composite:.3f}")

    # â”€â”€ extract experts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("[5/6] Extracting shared-basis + residual expertsâ€¦")
    t0 = time.time()
    expertized = extract_all_experts(
        base_model, selected, activations,
        num_experts=num_experts,
        shared_basis_rank=None,
        device=device,
    )
    print(f"      Done in {time.time()-t0:.1f}s.")

    # â”€â”€ fit linear router â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("[6/6] Fitting linear routers (5 epochs each layer)â€¦")
    from clear_moe.router import fit_all_routers
    t0 = time.time()
    routers = fit_all_routers(base_model, expertized, activations, config, device)
    print(f"      Done in {time.time()-t0:.1f}s.")

    # â”€â”€ run configurations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    all_results = {}
    dense_top1  = None

    configs_iter = tqdm(configs_to_run, desc=f"{tag} configs", ncols=80, colour="cyan")

    for cfg_id in configs_iter:
        configs_iter.set_postfix({"cfg": cfg_id})
        t_cfg = time.time()

        try:
            model_copy = copy.deepcopy(base_model).to(device)

            if cfg_id == "D0":
                # â”€â”€ Dense baseline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                cls  = evaluate_classification(model_copy, eval_loader, device)
                lat  = measure_latency(model_copy, input_shape, device)
                rec  = {
                    "cfg":          "D0",
                    "label":        "Dense",
                    "top1":         cls["top1"],
                    "top5":         cls["top5"],
                    "p50_ms":       lat["latency_p50_ms"],
                    "p95_ms":       lat["latency_p95_ms"],
                    "router_acc":   None,
                    "load_skew":    None,
                    "delta_pp":     0.0,
                    "elapsed_s":    round(time.time() - t_cfg, 1),
                }
                dense_top1 = cls["top1"]

            elif cfg_id == "D3":
                # Shared basis + global residual (exact reconstruction)
                # Use original weights directly to guarantee algebraic exactness
                from clear_moe.extraction import _get_module, _get_ffn_weights, _detect_activation

                class _ExactFFN(nn.Module):
                    def __init__(self, fc1_w, fc1_b, fc2_w, fc2_b, act_fn):
                        super().__init__()
                        out1, in1 = fc1_w.shape
                        self.fc1 = nn.Linear(in1, out1, bias=fc1_b is not None)
                        self.fc1.weight = nn.Parameter(fc1_w.clone())
                        if fc1_b is not None:
                            self.fc1.bias = nn.Parameter(fc1_b.clone())
                        out2, in2 = fc2_w.shape
                        self.fc2 = nn.Linear(in2, out2, bias=fc2_b is not None)
                        self.fc2.weight = nn.Parameter(fc2_w.clone())
                        if fc2_b is not None:
                            self.fc2.bias = nn.Parameter(fc2_b.clone())
                        self.act = act_fn
                    def forward(self, x):
                        return self.fc2(self.act(self.fc1(x)))

                replacements = {}
                for name in selected_names:
                    ffn    = _get_module(base_model, name)
                    fc1_w, fc1_b, fc2_w, fc2_b = _get_ffn_weights(ffn)
                    act_fn = _detect_activation(ffn)
                    replacements[name] = _ExactFFN(fc1_w, fc1_b, fc2_w, fc2_b, act_fn).to(device)
                replace_ffn_with_moe(model_copy, selected_names, replacements)

                cls  = evaluate_classification(model_copy, eval_loader, device)
                lat  = measure_latency(model_copy, input_shape, device)
                rec  = {
                    "cfg":        "D3",
                    "label":      "SVD+GlobalRes",
                    "top1":       cls["top1"],
                    "top5":       cls["top5"],
                    "p50_ms":     lat["latency_p50_ms"],
                    "p95_ms":     lat["latency_p95_ms"],
                    "router_acc": None,
                    "load_skew":  None,
                    "delta_pp":   round(cls["top1"] - (dense_top1 or cls["top1"]), 3),
                    "elapsed_s":  round(time.time() - t_cfg, 1),
                }

            elif cfg_id == "D5":
                # â”€â”€ Shared basis + k-means + RANDOM routing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                from clear_moe.moe_builder import build_moe_modules_from_expertized

                class _RandomRouter(nn.Module):
                    def __init__(self, n):
                        super().__init__()
                        self.n = n
                    def forward(self, x):
                        idx = torch.randint(0, self.n, (x.shape[0],), device=x.device)
                        w   = torch.ones(x.shape[0], device=x.device)
                        return idx, w, None

                rand_routers = {n: _RandomRouter(num_experts) for n in expertized}
                moe_modules  = build_moe_modules_from_expertized(
                    expertized, rand_routers, base_model, config
                )
                replace_ffn_with_moe(model_copy, selected_names, moe_modules)
                model_copy = model_copy.to(device)

                cls  = evaluate_classification(model_copy, eval_loader, device)
                lat  = measure_latency(model_copy, input_shape, device)
                rec  = {
                    "cfg":        "D5",
                    "label":      "SVD+kMeans+RandRouter",
                    "top1":       cls["top1"],
                    "top5":       cls["top5"],
                    "p50_ms":     lat["latency_p50_ms"],
                    "p95_ms":     lat["latency_p95_ms"],
                    "router_acc": 0.0,
                    "load_skew":  None,
                    "delta_pp":   round(cls["top1"] - (dense_top1 or cls["top1"]), 3),
                    "elapsed_s":  round(time.time() - t_cfg, 1),
                }

            elif cfg_id == "D6":
                # â”€â”€ Shared basis + k-means + LEARNED linear router â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                from clear_moe.moe_builder import build_moe_modules_from_expertized
                from clear_moe.router import compute_routing_stats

                moe_modules = build_moe_modules_from_expertized(
                    expertized, routers, base_model, config
                )
                replace_ffn_with_moe(model_copy, selected_names, moe_modules)
                model_copy = model_copy.to(device)

                # Compute router accuracy
                router_accs = []
                load_skews  = []
                for n in selected_names:
                    if n not in routers or n not in expertized:
                        continue
                    stats = compute_routing_stats(routers[n], activations[n], device)
                    load_skews.append(stats.get("load_balance_std", 0))
                    with torch.no_grad():
                        acts  = activations[n].to(device)
                        preds = []
                        for i in range(0, acts.shape[0], 2048):
                            idx_r, _, _ = routers[n](acts[i : i + 2048])
                            if idx_r.dim() > 1:
                                idx_r = idx_r[:, 0]
                            preds.append(idx_r.cpu())
                        preds_np = torch.cat(preds).numpy()
                        acc = float((preds_np == expertized[n].cluster_labels).mean())
                        router_accs.append(acc)

                cls  = evaluate_classification(model_copy, eval_loader, device)
                lat  = measure_latency(model_copy, input_shape, device)
                rec  = {
                    "cfg":        "D6",
                    "label":      "SVD+kMeans+LinRouter",
                    "top1":       cls["top1"],
                    "top5":       cls["top5"],
                    "p50_ms":     lat["latency_p50_ms"],
                    "p95_ms":     lat["latency_p95_ms"],
                    "router_acc": float(np.mean(router_accs)) if router_accs else None,
                    "load_skew":  float(np.mean(load_skews))  if load_skews  else None,
                    "delta_pp":   round(cls["top1"] - (dense_top1 or cls["top1"]), 3),
                    "elapsed_s":  round(time.time() - t_cfg, 1),
                }

            else:
                logger.warning(f"Unknown config {cfg_id}, skipping.")
                continue

            all_results[cfg_id] = rec

            # â”€â”€ print live row â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            print_row(
                rec["label"],
                rec["top1"],
                rec["delta_pp"],
                rec["p50_ms"],
                rec.get("router_acc"),
            )

            # â”€â”€ flush results to disk after every config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            save_results(
                {"backbone": tag, "model_key": model_key, "seed": seed, "runs": all_results},
                out_dir,
            )

        except Exception as exc:
            logger.error(f"  Config {cfg_id} FAILED: {exc}", exc_info=True)
            all_results[cfg_id] = {"cfg": cfg_id, "error": str(exc)}

    # â”€â”€ backbone summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print()
    print(f"  â”€â”€ {tag} summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    dense_top1_final = all_results.get("D0", {}).get("top1", None)
    for cfg_id in configs_to_run:
        if cfg_id not in all_results or "error" in all_results[cfg_id]:
            print(f"  {cfg_id:<6}  ERROR")
            continue
        r = all_results[cfg_id]
        if dense_top1_final is not None:
            delta = r["top1"] - dense_top1_final
        else:
            delta = r.get("delta_pp", 0)
        print(f"  {r['label']:<22} Top-1={r['top1']:.2f}%  Î”={delta:+.2f}pp  p50={r['p50_ms']:.1f}ms")

    return {"backbone": tag, "model_key": model_key, "seed": seed, "runs": all_results}


# â”€â”€ summary table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def write_summary(all_backbone_results: list, out_dir: Path):
    """Write a cross-backbone comparison table."""
    import csv

    rows = []
    for br in all_backbone_results:
        tag      = br["backbone"]
        runs     = br["runs"]
        dense    = runs.get("D0", {}).get("top1", None)
        for cfg_id, r in runs.items():
            if "error" in r:
                continue
            row = {
                "Backbone":    tag,
                "Config":      cfg_id,
                "Label":       r.get("label", cfg_id),
                "Top-1 (%)":   f"{r['top1']:.2f}",
                "Delta (pp)":  f"{r['top1'] - dense:+.2f}" if dense else "â€”",
                "p50 (ms)":    f"{r['p50_ms']:.1f}",
                "Router Acc":  f"{r['router_acc']:.3f}" if r.get("router_acc") is not None else "â€”",
            }
            rows.append(row)

    # CSV
    csv_path = out_dir / "summary_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Human-readable text table
    txt_path = out_dir / "summary_table.txt"
    col_w    = [22, 6, 24, 10, 10, 10, 11]
    headers  = ["Backbone", "Cfg", "Label", "Top-1(%)", "Delta(pp)", "p50(ms)", "RouterAcc"]
    sep      = "  ".join("-" * w for w in col_w)

    lines = [
        "CLEAR-MoE Cross-Backbone Validation",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "  ".join(h.ljust(w) for h, w in zip(headers, col_w)),
        sep,
    ]
    for r in rows:
        vals = [
            r["Backbone"], r["Config"], r["Label"],
            r["Top-1 (%)"], r["Delta (pp)"], r["p50 (ms)"], r["Router Acc"],
        ]
        lines.append("  ".join(str(v).ljust(w) for v, w in zip(vals, col_w)))

    txt_content = "\n".join(lines)
    txt_path.write_text(txt_content)

    print()
    print("=" * 70)
    print(txt_content)
    print("=" * 70)
    print(f"\n  CSV  â†’ {csv_path}")
    print(f"  Text â†’ {txt_path}")


# â”€â”€ entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser(description="CLEAR-MoE multi-backbone experiment")
    parser.add_argument(
        "--models", nargs="+",
        default=["deit_tiny", "deit_small", "vit_small", "deit_base", "vit_base"],
        choices=list(MODEL_REGISTRY.keys()),
        help="Which backbones to run",
    )
    parser.add_argument(
        "--configs", nargs="+",
        default=["D0", "D3", "D5", "D6"],
        help="Which ablation configs to run per backbone",
    )
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--calib_size", type=int, default=200)
    parser.add_argument("--data_dir",   default="data/imagenette2-320",
                        help="Path to imagenette2-320 directory")
    parser.add_argument("--output_dir", default="outputs/multi_model")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root   = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not (data_dir / "train").exists():
        # try alternate path
        alt = Path("data/imagenette2-320")
        if (alt / "train").exists():
            data_dir = alt
        else:
            logger.error(f"Imagenette not found at '{data_dir}'. "
                         "Pass --data_dir <path> with a directory containing train/ and val/.")
            sys.exit(1)

    print(f"\n{'='*70}")
    print("  CLEAR-MoE Multi-Backbone Validation Experiment")
    print(f"{'='*70}")
    print(f"  Device      : {device}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Backbones   : {', '.join(args.models)}")
    print(f"  Configs     : {', '.join(args.configs)}")
    print(f"  Calib size  : {args.calib_size}")
    print(f"  Seed        : {args.seed}")
    print(f"  Data dir    : {data_dir}")
    print(f"  Output dir  : {out_root}")
    print(f"  Start time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_backbone_results = []

    backbone_bar = tqdm(args.models, desc="Backbones", ncols=80, colour="green", position=0)

    for model_key in backbone_bar:
        backbone_bar.set_description(f"Backbone: {MODEL_REGISTRY[model_key]['tag']}")
        try:
            result = run_backbone(
                model_key      = model_key,
                data_dir       = str(data_dir),
                output_root    = out_root,
                seed           = args.seed,
                calib_size     = args.calib_size,
                configs_to_run = args.configs,
                device         = device,
            )
            all_backbone_results.append(result)

            # free GPU memory between backbones
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            logger.error(f"Backbone {model_key} FAILED: {exc}", exc_info=True)

    # â”€â”€ combined summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if all_backbone_results:
        write_summary(all_backbone_results, out_root)

    # â”€â”€ save full combined JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    combined_path = out_root / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_backbone_results, f, indent=2)
    print(f"\n  Full results â†’ {combined_path}")

    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
