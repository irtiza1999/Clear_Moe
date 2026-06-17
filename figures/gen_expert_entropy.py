#!/usr/bin/env python3
"""
Generate expert routing entropy figure (appendix).

Runs the D6 pipeline (calibration → layer scoring → expert extraction →
router fitting) and records the mean per-token routing entropy per layer.
D5 (random routing) is shown as a horizontal reference at ln(4) = max entropy.

Low entropy for D6 means confident (decisive) routing.
Near-identical downstream accuracy despite this entropy gap proves that
routing quality does not drive accuracy — the shared SVD basis does.

Output: figures/expert_entropy.pdf
Usage:  python figures/gen_expert_entropy.py [--config CONFIG_YAML]
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from clear_moe.calibration import collect_activations, get_calibration_dataloader
from clear_moe.extraction import extract_all_experts
from clear_moe.models import get_ffn_layers, get_model_input_transform, load_model
from clear_moe.router import compute_routing_stats, fit_all_routers
from clear_moe.scoring import score_layers, select_layers

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUT_PATH = Path("figures/expert_entropy.pdf")
MAX_ENTROPY_4 = math.log(4)  # ln(4) ≈ 1.386 nats — random over 4 experts


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(config: dict) -> torch.device:
    d = config.get("device", "cuda")
    if d == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — using CPU")
        return torch.device("cpu")
    return torch.device(d)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/imagenette_deit_s_direct.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed
    set_seed(seed)
    device = get_device(config)

    num_experts = config.get("extraction", {}).get("num_experts", 4)
    basis_rank = config.get("extraction", {}).get("shared_basis_rank") or None

    logger.info("Loading model ...")
    transform = get_model_input_transform(config)
    base_model = load_model(config).to(device)
    layer_infos = get_ffn_layers(base_model, config)
    k = len(layer_infos) // 2

    logger.info("Collecting calibration activations ...")
    calib_loader = get_calibration_dataloader(config, transform=transform, seed=seed)
    layer_names = [info.name for info in layer_infos]
    activations = collect_activations(base_model, calib_loader, layer_names, device)

    logger.info("Scoring and selecting layers ...")
    scores = score_layers(activations, base_model, layer_infos, device, calib_loader)
    scores = select_layers(scores, strategy="score_top_k",
                           top_k=k, total_layers=len(layer_infos))
    selected = [s for s in scores if s.selected]
    selected_names = [s.name for s in selected]
    selected_indices = [s.layer_index for s in selected]
    logger.info(f"Selected layers: {selected_indices}")

    logger.info("Extracting experts ...")
    expertized = extract_all_experts(
        base_model, selected, activations,
        num_experts=num_experts,
        shared_basis_rank=basis_rank,
        device=device,
    )

    logger.info("Fitting D6 linear routers ...")
    routers = fit_all_routers(
        model=base_model,
        expertized_layers=expertized,
        activations=activations,
        config=config,
        device=device,
    )

    logger.info("Computing per-layer routing entropy ...")
    layer_labels = []
    d6_entropies = []

    for name in selected_names:
        if name not in routers or name not in activations:
            continue
        router = routers[name]
        acts = activations[name]
        stats = compute_routing_stats(router, acts, device)
        entropy = stats["mean_entropy"]
        d6_entropies.append(entropy)

        # Short label: "blk.N" from "blocks.N.mlp"
        parts = name.split(".")
        if len(parts) >= 2:
            lbl = f"blk.{parts[1]}"
        else:
            lbl = name
        layer_labels.append(lbl)

    logger.info("Generating figure ...")
    n = len(d6_entropies)
    x = np.arange(n)
    width = 0.4

    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    ax.bar(x - width / 2, d6_entropies, width,
           color="#1565C0", alpha=0.85, label="D6 — Learned linear router",
           edgecolor="white", linewidth=0.5, zorder=3)

    ax.axhline(MAX_ENTROPY_4, color="#E53935", linestyle="--", linewidth=1.4,
               label=f"D5 — Random router (max entropy, ln 4 ≈ {MAX_ENTROPY_4:.2f})",
               zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize=8)
    ax.set_xlabel("Expertized layer", fontsize=9)
    ax.set_ylabel("Mean routing entropy (nats)", fontsize=9)
    ax.set_title("Per-layer routing entropy: D5 (random) vs D6 (learned)",
                 fontsize=9)
    ax.set_ylim(0, MAX_ENTROPY_4 * 1.25)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")

    mean_d6 = float(np.mean(d6_entropies)) if d6_entropies else 0.0
    print(f"D6 mean entropy: {mean_d6:.4f} nats "
          f"(D5 reference: {MAX_ENTROPY_4:.4f} nats)")
    print("Layer entropies:")
    for lbl, h in zip(layer_labels, d6_entropies):
        print(f"  {lbl}: {h:.4f}")


if __name__ == "__main__":
    main()
