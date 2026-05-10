"""
Create paper-ready visuals for the CLEAR-MoE methodology using real dataset images
and actual run artifacts.

Outputs:
1) methodology_steps.png  - step-by-step methodology panel
2) full_pipeline_diagram.png - end-to-end pipeline diagram
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image


def _pick_dataset_image(data_root: Path) -> Path:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files: List[Path] = []
    for p in patterns:
        files.extend(sorted(data_root.rglob(p)))
    if not files:
        raise FileNotFoundError(f"No image files found under {data_root}")
    return files[0]


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _activation_heatmap(act_path: Path) -> np.ndarray:
    tensor = torch.load(act_path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor in {act_path}, got {type(tensor)}")
    x = tensor
    if x.ndim >= 2:
        row = x[0]
    else:
        row = x
    row = row.detach().float().flatten().cpu().numpy()
    if row.size < 256:
        row = np.pad(row, (0, 256 - row.size), mode="constant")
    else:
        row = row[:256]
    # 16x16 pseudo-grid from actual activation values.
    return row.reshape(16, 16)


def _best_elastic_by_latency(elastic: Dict) -> Tuple[str, Dict]:
    best_key = None
    best_val = None
    for k, v in elastic.items():
        p50 = v.get("latency_p50_ms", float("inf"))
        if best_val is None or p50 < best_val:
            best_val = p50
            best_key = k
    if best_key is None:
        raise ValueError("No elastic entries found")
    return best_key, elastic[best_key]


def build_methodology_figure(
    image_path: Path,
    layer_scores_png: Path,
    extraction_summary: Dict,
    routing_stats: Dict,
    hmoe_results: Dict,
    soft_results: Dict,
    elastic_results: Dict,
    roofline_png: Path,
    out_path: Path,
    activation_grid: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    # Step 1: real dataset image
    ax1 = fig.add_subplot(gs[0, 0])
    img = Image.open(image_path).convert("RGB")
    ax1.imshow(img)
    ax1.set_title("Step 1: Input Sample (Imagenette)")
    ax1.axis("off")
    ax1.text(
        0.01,
        -0.08,
        f"Actual image: {image_path.name}",
        transform=ax1.transAxes,
        fontsize=9,
    )

    # Step 2: activation logging view from actual tensor
    ax2 = fig.add_subplot(gs[0, 1])
    hm = ax2.imshow(activation_grid, cmap="magma", aspect="auto")
    ax2.set_title("Step 2: Activation Logging")
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.colorbar(hm, ax=ax2, fraction=0.046, pad=0.04)
    ax2.text(0.01, -0.08, "Source: blocks_11_mlp.pt", transform=ax2.transAxes, fontsize=9)

    # Step 3: layer scoring artifact
    ax3 = fig.add_subplot(gs[0, 2])
    score_img = Image.open(layer_scores_png)
    ax3.imshow(score_img)
    ax3.set_title("Step 3: Layer Scoring")
    ax3.axis("off")

    # Step 4: extraction stats (cluster sizes + reconstruction error)
    ax4 = fig.add_subplot(gs[0, 3])
    layer_name = "blocks.11.mlp"
    layer = extraction_summary["layers"][layer_name]
    clusters = layer["cluster_sizes"]
    ax4.bar(range(len(clusters)), clusters, color="#2c7fb8")
    ax4.set_title("Step 4: Expert Extraction")
    ax4.set_xlabel("Expert ID")
    ax4.set_ylabel("Tokens")
    ax4.text(
        0.02,
        0.94,
        f"Layer: {layer_name}\nRank={layer['shared_rank']}\nRecon err={layer['reconstruction_error']:.4f}",
        transform=ax4.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # Step 5: router load fractions
    ax5 = fig.add_subplot(gs[1, 0])
    r = routing_stats[layer_name]
    fracs = r["expert_load_fractions"]
    ax5.bar(range(len(fracs)), fracs, color="#41ab5d")
    ax5.set_ylim(0, 1)
    ax5.set_title("Step 5: Router Fitting")
    ax5.set_xlabel("Expert ID")
    ax5.set_ylabel("Load fraction")
    ax5.text(
        0.02,
        0.94,
        f"Entropy={r['mean_entropy']:.3f}\nLoad std={r['load_balance_std']:.3f}",
        transform=ax5.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # Step 6: variant metrics table
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.axis("off")
    best_k, best_elastic = _best_elastic_by_latency(elastic_results)
    table_text = (
        "Step 6: Variant Evaluation\n\n"
        f"H-MoE  top1={hmoe_results['eval_results']['top1']:.4f}%\n"
        f"       p50={hmoe_results['eval_results']['latency_p50_ms']:.2f} ms\n\n"
        f"Soft   top1={soft_results['soft']['top1']:.4f}%\n"
        f"       p50={soft_results['soft']['latency_p50_ms']:.2f} ms\n\n"
        f"Elastic best={best_k}\n"
        f"       top1={best_elastic['top1']:.4f}%\n"
        f"       p50={best_elastic['latency_p50_ms']:.2f} ms"
    )
    ax6.text(
        0.02,
        0.98,
        table_text,
        va="top",
        fontsize=11,
        family="monospace",
        bbox=dict(facecolor="#f7f7f7", edgecolor="#bdbdbd"),
    )

    # Step 7: systems analysis (roofline image)
    ax7 = fig.add_subplot(gs[1, 2])
    roof_img = Image.open(roofline_png)
    ax7.imshow(roof_img)
    ax7.set_title("Step 7: Roofline / Systems Analysis")
    ax7.axis("off")

    # Step 8: final deliverables
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis("off")
    ax8.text(
        0.02,
        0.98,
        "Step 8: Final Outputs\n\n"
        "- Classification benchmark\n"
        "- Segmentation benchmark\n"
        "- Dispatch benchmark\n"
        "- CPU vs GPU scaling\n"
        "- Ablation sweep\n"
        "- Roofline figures\n"
        "- Triton/cuBLAS comparison",
        va="top",
        fontsize=11,
        bbox=dict(facecolor="#f7f7f7", edgecolor="#bdbdbd"),
    )

    fig.suptitle("CLEAR-MoE Methodology Visualized on Real Run Artifacts", fontsize=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _draw_box(ax, x, y, w, h, label, fc="#e5f5f9", ec="#2b8cbe"):
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)


def build_pipeline_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Row 1: data/model prep
    _draw_box(ax, 0.03, 0.78, 0.18, 0.12, "Data + Dense Backbone")
    _draw_box(ax, 0.25, 0.78, 0.18, 0.12, "Activation Logging")
    _draw_box(ax, 0.47, 0.78, 0.18, 0.12, "Layer Scoring")
    _draw_box(ax, 0.69, 0.78, 0.18, 0.12, "Selected FFN Layers")

    # Row 2: extraction/routing
    _draw_box(ax, 0.14, 0.55, 0.22, 0.12, "Shared Basis + Residual Experts", fc="#edf8e9", ec="#31a354")
    _draw_box(ax, 0.42, 0.55, 0.18, 0.12, "Router Fitting", fc="#edf8e9", ec="#31a354")
    _draw_box(ax, 0.66, 0.55, 0.22, 0.12, "MoE Assembly", fc="#edf8e9", ec="#31a354")

    # Row 3: evaluations
    _draw_box(ax, 0.03, 0.30, 0.18, 0.12, "CLS Benchmark", fc="#fff7bc", ec="#d95f0e")
    _draw_box(ax, 0.25, 0.30, 0.18, 0.12, "SEG Benchmark", fc="#fff7bc", ec="#d95f0e")
    _draw_box(ax, 0.47, 0.30, 0.18, 0.12, "H/Soft/Elastic MoE", fc="#fff7bc", ec="#d95f0e")
    _draw_box(ax, 0.69, 0.30, 0.18, 0.12, "Ablation Runner", fc="#fff7bc", ec="#d95f0e")

    # Row 4: systems
    _draw_box(ax, 0.03, 0.07, 0.18, 0.12, "Parallel Dispatch", fc="#f2f0f7", ec="#756bb1")
    _draw_box(ax, 0.25, 0.07, 0.18, 0.12, "CPU vs GPU", fc="#f2f0f7", ec="#756bb1")
    _draw_box(ax, 0.47, 0.07, 0.18, 0.12, "Parallel Scaling", fc="#f2f0f7", ec="#756bb1")
    _draw_box(ax, 0.69, 0.07, 0.18, 0.12, "Roofline + Triton", fc="#f2f0f7", ec="#756bb1")

    # Arrows top row
    arrow = dict(arrowstyle="->", lw=1.5, color="#444444")
    ax.annotate("", xy=(0.25, 0.84), xytext=(0.21, 0.84), arrowprops=arrow)
    ax.annotate("", xy=(0.47, 0.84), xytext=(0.43, 0.84), arrowprops=arrow)
    ax.annotate("", xy=(0.69, 0.84), xytext=(0.65, 0.84), arrowprops=arrow)

    # Down arrows
    ax.annotate("", xy=(0.25, 0.67), xytext=(0.78, 0.78), arrowprops=arrow)
    ax.annotate("", xy=(0.51, 0.67), xytext=(0.36, 0.78), arrowprops=arrow)
    ax.annotate("", xy=(0.77, 0.67), xytext=(0.58, 0.78), arrowprops=arrow)

    # Row2 to row3
    ax.annotate("", xy=(0.12, 0.42), xytext=(0.22, 0.55), arrowprops=arrow)
    ax.annotate("", xy=(0.34, 0.42), xytext=(0.48, 0.55), arrowprops=arrow)
    ax.annotate("", xy=(0.56, 0.42), xytext=(0.75, 0.55), arrowprops=arrow)
    ax.annotate("", xy=(0.78, 0.42), xytext=(0.76, 0.55), arrowprops=arrow)

    # Row3 to row4
    ax.annotate("", xy=(0.12, 0.19), xytext=(0.12, 0.30), arrowprops=arrow)
    ax.annotate("", xy=(0.34, 0.19), xytext=(0.34, 0.30), arrowprops=arrow)
    ax.annotate("", xy=(0.56, 0.19), xytext=(0.56, 0.30), arrowprops=arrow)
    ax.annotate("", xy=(0.78, 0.19), xytext=(0.78, 0.30), arrowprops=arrow)

    fig.suptitle("CLEAR-MoE Full Pipeline Diagram", fontsize=18, y=0.98)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate methodology and pipeline visuals")
    parser.add_argument(
        "--tag",
        type=str,
        default="fullrun_20260423_141913",
        help="Run tag under outputs/full_runs and outputs/logs",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    tag = args.tag

    data_root = repo_root / "data" / "imagenette2-320" / "val"
    image_path = _pick_dataset_image(data_root)

    logs_root = repo_root / "outputs" / "logs"
    run_root = repo_root / "outputs" / "full_runs" / tag

    layer_scores_png = logs_root / f"{tag}_scores_cls" / "layer_scores.png"
    extraction_summary_path = logs_root / f"{tag}_extract_cls" / "extraction_summary.json"
    routing_stats_path = logs_root / f"{tag}_routers_cls" / "routing_stats.json"
    activation_path = logs_root / f"{tag}_acts_cls" / "activations" / "blocks_11_mlp.pt"
    hmoe_path = run_root / "hmoe" / "hmoe_20260423_143441" / "hmoe_results.json"
    soft_path = run_root / "soft_elastic" / "soft_elastic_20260423_155304" / "results.json"
    elastic_path = run_root / "soft_elastic" / "soft_elastic_20260423_155557" / "results.json"
    roofline_png = run_root / "roofline" / "roofline_chart.png"

    required = [
        layer_scores_png,
        extraction_summary_path,
        routing_stats_path,
        activation_path,
        hmoe_path,
        soft_path,
        elastic_path,
        roofline_png,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(str(p) for p in missing))

    extraction_summary = _load_json(extraction_summary_path)
    routing_stats = _load_json(routing_stats_path)
    hmoe_results = _load_json(hmoe_path)
    soft_results = _load_json(soft_path)
    elastic_results = _load_json(elastic_path)
    activation_grid = _activation_heatmap(activation_path)

    out_dir = run_root / "paper_visuals"
    methodology_png = out_dir / "methodology_steps.png"
    pipeline_png = out_dir / "full_pipeline_diagram.png"

    build_methodology_figure(
        image_path=image_path,
        layer_scores_png=layer_scores_png,
        extraction_summary=extraction_summary,
        routing_stats=routing_stats,
        hmoe_results=hmoe_results,
        soft_results=soft_results,
        elastic_results=elastic_results,
        roofline_png=roofline_png,
        out_path=methodology_png,
        activation_grid=activation_grid,
    )
    build_pipeline_diagram(pipeline_png)

    print("Generated:")
    print(methodology_png)
    print(pipeline_png)


if __name__ == "__main__":
    main()
