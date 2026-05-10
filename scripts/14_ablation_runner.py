#!/usr/bin/env python3
"""
Script 14 — Ablation Runner

Config-driven ablation harness. Reads a YAML grid, executes each cell by
invoking scripts 01–09 + 12–13 as subprocess commands, and collects all
results into a single results.csv.

Supports multi-axis ablation sweeps:
    - Layer selection strategy
    - Expert count E
    - Shared basis rank r
    - Router depth (0=linear, 1=1-layer MLP, 2=2-layer MLP)
    - Calibration set size
    - MoE architecture (flat_hard, hmoe, soft, elastic)

Usage:
    python scripts/14_ablation_runner.py --config configs/deit_s_imagenet.yaml \
        --ablation_config configs/ablation_deit_s.yaml

Ablation config YAML format:
    axes:
      - name: expert_count
        values: [2, 4, 8, 16]
        script_arg: "--num_experts"
        target_script: "04_extract_experts.py"
      - name: layer_strategy
        values: [last_half, score_top_k, all_layers, alternating]
        script_arg: "--strategy"
        target_script: "03_score_layers.py"
    fixed:
      model: deit_small_patch16_224
      E: 4
      strategy: last_half
    output_dir: outputs/ablations
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clear_moe.utils import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Config-driven ablation runner")
    p.add_argument("--config", type=str, required=True,
                   help="Base model config YAML")
    p.add_argument("--ablation_config", type=str, default=None,
                   help="Ablation grid YAML (optional; uses config defaults if not provided)")
    p.add_argument("--axes", type=str, default=None,
                   help="Comma-separated ablation axes: e.g. expert_count,layer_strategy")
    p.add_argument("--output_dir", type=str, default="outputs/ablations")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Ablation grid definition
# ---------------------------------------------------------------------------

DEFAULT_ABLATION_GRID = {
    "layer_strategy": {
        "values": ["last_half", "score_top_k", "all_layers", "alternating"],
        "script": "03_score_layers.py",
        "arg": "--strategy",
    },
    "expert_count": {
        "values": [2, 4, 8, 16],
        "script": "04_extract_experts.py",
        "arg": "--num_experts",
    },
    "basis_rank": {
        "values": [8, 16, 32, 64, 128],
        "script": "04_extract_experts.py",
        "arg": "--shared_rank",
    },
    "router_depth": {
        "values": [0, 1, 2],
        "script": "05_fit_router.py",
        "arg": "--router_depth",
    },
    "calib_size": {
        "values": [50, 200, 500, 2000],
        "script": "02_log_activations.py",
        "arg": "--calib_size",
    },
    "moe_architecture": {
        "values": ["flat_hard", "hmoe", "soft", "elastic"],
        "script": None,  # handled specially
        "arg": "--architecture",
    },
}


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

class AblationCell:
    """One cell in the ablation grid."""

    def __init__(
        self,
        axis: str,
        value: Any,
        base_config: str,
        output_dir: Path,
        extra_args: Dict[str, str] = None,
    ):
        self.axis = axis
        self.value = value
        self.base_config = base_config
        self.output_dir = output_dir
        self.cell_dir = output_dir / f"{axis}_{value}"
        self.extra_args = extra_args or {}

    def cell_id(self) -> str:
        return f"{self.axis}={self.value}"

    def result_path(self) -> Path:
        return self.cell_dir / "cell_result.json"

    def is_done(self) -> bool:
        return self.result_path().exists()


def _run_subprocess(cmd: List[str], cwd: Path, log_path: Path, dry_run: bool = False) -> bool:
    """Run a subprocess command, streaming output to log_path. Returns success."""
    if dry_run:
        logger.info(f"[DRY RUN] {' '.join(cmd)}")
        return True

    logger.info(f"Running: {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "w") as log_f:
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1 hour max per cell
            )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {' '.join(cmd)}")
        return False
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return False


def run_ablation_cell(
    cell: AblationCell,
    scripts_dir: Path,
    dry_run: bool,
) -> Dict:
    """
    Run a full pipeline for one ablation cell.

    For layer_strategy/expert_count/basis_rank/calib_size: modifies
    the corresponding pipeline stage and re-runs downstream steps.

    For moe_architecture: directly calls scripts 12/13.

    Returns a result dict for the CSV.
    """
    cell.cell_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    project_root = scripts_dir.parent
    cfg = cell.base_config

    result = {
        "axis": cell.axis,
        "value": str(cell.value),
        "cell_id": cell.cell_id(),
        "success": False,
        "top1_accuracy": None,
        "miou": None,
        "latency_p50_ms": None,
        "throughput_tok_s": None,
    }

    axis_def = DEFAULT_ABLATION_GRID.get(cell.axis, {})
    script_name = axis_def.get("script")
    arg_name = axis_def.get("arg")

    cell_out = str(cell.cell_dir)

    # ------------------------------------------------------------------
    # Strategy: just re-run scoring + downstream pipeline
    # ------------------------------------------------------------------
    if cell.axis == "layer_strategy":
        # Need activations_dir — look for a recent one or use extra_args
        acts_dir = cell.extra_args.get("activations_dir", "")
        if not acts_dir:
            logger.warning("No activations_dir specified for layer_strategy ablation")
            result["error"] = "missing activations_dir"
            return result

        score_cmd = [python, str(scripts_dir / "03_score_layers.py"),
                     "--config", cfg,
                     "--activations_dir", acts_dir,
                     "--strategy", str(cell.value),
                     "--run_name", f"scores_{cell.axis}_{cell.value}"]
        ok = _run_subprocess(score_cmd, project_root,
                             cell.cell_dir / "03_scores.log", dry_run)
        result["success"] = ok

    elif cell.axis == "expert_count":
        acts_dir = cell.extra_args.get("activations_dir", "")
        scores_file = cell.extra_args.get("scores_file", "")
        if not acts_dir or not scores_file:
            result["error"] = "missing activations_dir or scores_file"
            return result

        extract_cmd = [python, str(scripts_dir / "04_extract_experts.py"),
                       "--config", cfg,
                       "--activations_dir", acts_dir,
                       "--scores_file", scores_file,
                       "--num_experts", str(cell.value),
                       "--run_name", f"extraction_{cell.axis}_{cell.value}"]
        ok = _run_subprocess(extract_cmd, project_root,
                             cell.cell_dir / "04_extract.log", dry_run)
        result["success"] = ok

    elif cell.axis == "basis_rank":
        acts_dir = cell.extra_args.get("activations_dir", "")
        scores_file = cell.extra_args.get("scores_file", "")
        if not acts_dir or not scores_file:
            result["error"] = "missing activations_dir or scores_file"
            return result

        extract_cmd = [python, str(scripts_dir / "04_extract_experts.py"),
                       "--config", cfg,
                       "--activations_dir", acts_dir,
                       "--scores_file", scores_file,
                       "--shared_rank", str(cell.value),
                       "--run_name", f"extraction_{cell.axis}_{cell.value}"]
        ok = _run_subprocess(extract_cmd, project_root,
                             cell.cell_dir / "04_extract.log", dry_run)
        result["success"] = ok

    elif cell.axis == "calib_size":
        calib_cmd = [python, str(scripts_dir / "02_log_activations.py"),
                     "--config", cfg,
                     "--calib_size", str(cell.value),
                     "--run_name", f"activations_{cell.axis}_{cell.value}"]
        ok = _run_subprocess(calib_cmd, project_root,
                             cell.cell_dir / "02_activations.log", dry_run)
        result["success"] = ok

    elif cell.axis == "router_depth":
        acts_dir = cell.extra_args.get("activations_dir", "")
        ext_dir = cell.extra_args.get("extraction_dir", "")
        if not acts_dir or not ext_dir:
            result["error"] = "missing activations_dir or extraction_dir"
            return result

        router_cmd = [python, str(scripts_dir / "05_fit_router.py"),
                      "--config", cfg,
                      "--activations_dir", acts_dir,
                      "--extraction_dir", ext_dir,
                      "--router_depth", str(cell.value),
                      "--run_name", f"router_{cell.axis}_{cell.value}"]
        ok = _run_subprocess(router_cmd, project_root,
                             cell.cell_dir / "05_router.log", dry_run)
        result["success"] = ok

    elif cell.axis == "moe_architecture":
        acts_dir = cell.extra_args.get("activations_dir", "")
        ext_dir = cell.extra_args.get("extraction_dir", "")

        if cell.value == "hmoe":
            cmd = [python, str(scripts_dir / "12_hierarchical_moe.py"),
                   "--config", cfg,
                   "--activations_dir", acts_dir,
                   "--output_dir", cell_out]
        elif cell.value in ("soft", "elastic"):
            cmd = [python, str(scripts_dir / "13_soft_elastic_moe.py"),
                   "--config", cfg,
                   "--activations_dir", acts_dir,
                   "--extraction_dir", ext_dir,
                   "--architecture", str(cell.value),
                   "--output_dir", cell_out]
        else:
            # flat_hard: standard pipeline, just report
            result["success"] = True
            result["note"] = "flat_hard is the baseline pipeline output"
            return result

        ok = _run_subprocess(cmd, project_root,
                             cell.cell_dir / "arch.log", dry_run)
        result["success"] = ok

    else:
        result["error"] = f"Unknown ablation axis: {cell.axis}"

    # Try to collect quality metric from any result JSON in cell_dir
    for jf in cell.cell_dir.rglob("*.json"):
        try:
            with open(jf) as f:
                data = json.load(f)
            if "top1_accuracy" in data:
                result["top1_accuracy"] = data["top1_accuracy"]
            if "miou" in data:
                result["miou"] = data["miou"]
            if "latency_p50_ms" in data:
                result["latency_p50_ms"] = data["latency_p50_ms"]
            if "throughput_images_per_sec" in data:
                result["throughput_tok_s"] = data["throughput_images_per_sec"]
            if result.get("top1_accuracy") or result.get("miou"):
                break
        except Exception:
            continue

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "ablation_runner.log"),
        ],
    )

    scripts_dir = Path(__file__).resolve().parent

    # Load ablation config
    axes_to_run = []
    extra_args = {}

    if args.ablation_config and Path(args.ablation_config).exists():
        with open(args.ablation_config) as f:
            ab_cfg = yaml.safe_load(f)
        for axis_def in ab_cfg.get("axes", []):
            axes_to_run.append({
                "name": axis_def["name"],
                "values": axis_def["values"],
            })
        extra_args = ab_cfg.get("extra_args", {})
    else:
        # Use config-specified sweep fields or CLI --axes
        import yaml as _yaml
        with open(args.config) as f:
            base_cfg = _yaml.safe_load(f)

        selected_axes = (
            [a.strip() for a in args.axes.split(",")]
            if args.axes
            else list(DEFAULT_ABLATION_GRID.keys())
        )

        for ax in selected_axes:
            grid_def = DEFAULT_ABLATION_GRID.get(ax)
            if not grid_def:
                logger.warning(f"Unknown axis: {ax}")
                continue
            # Use config values if present, else default
            config_key_map = {
                "expert_count": "expert_counts",
                "basis_rank": "basis_ranks",
                "calib_size": "calib_sizes",
                "moe_architecture": "moe_architectures",
            }
            config_key = config_key_map.get(ax)
            values = base_cfg.get(config_key, grid_def["values"]) if config_key else grid_def["values"]
            axes_to_run.append({"name": ax, "values": values})

    logger.info(f"Ablation axes: {[a['name'] for a in axes_to_run]}")
    total_cells = sum(len(a["values"]) for a in axes_to_run)
    logger.info(f"Total cells: {total_cells}")

    all_results = []
    done = 0

    for axis_spec in axes_to_run:
        axis_name = axis_spec["name"]
        for value in axis_spec["values"]:
            done += 1
            logger.info(f"\n[{done}/{total_cells}] {axis_name}={value}")

            cell = AblationCell(
                axis=axis_name,
                value=value,
                base_config=args.config,
                output_dir=out_dir,
                extra_args=extra_args,
            )

            if cell.is_done():
                logger.info(f"  Already done, loading cached result")
                with open(cell.result_path()) as f:
                    r = json.load(f)
            else:
                r = run_ablation_cell(cell, scripts_dir, args.dry_run)
                with open(cell.result_path(), "w") as f:
                    json.dump(r, f, indent=2)

            all_results.append(r)
            logger.info(f"  success={r['success']} "
                        f"top1={r.get('top1_accuracy')} "
                        f"miou={r.get('miou')}")

    # Write master results CSV
    csv_path = out_dir / "results.csv"
    if all_results:
        fieldnames = sorted(set().union(*[r.keys() for r in all_results]))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        logger.info(f"\nResults CSV: {csv_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"ABLATION COMPLETE: {done} cells")
    print(f"{'='*60}")
    print(f"{'Axis':<25} {'Value':<12} {'Top-1':>8} {'mIoU':>8} {'p50ms':>8}")
    print("-" * 65)
    for r in all_results:
        top1 = f"{r.get('top1_accuracy', ''):.4f}" if r.get("top1_accuracy") else "N/A"
        miou = f"{r.get('miou', ''):.4f}" if r.get("miou") else "N/A"
        p50 = f"{r.get('latency_p50_ms', ''):.1f}" if r.get("latency_p50_ms") else "N/A"
        print(f"  {r['axis']:<23} {str(r['value']):<12} {top1:>8} {miou:>8} {p50:>8}")
    print(f"{'='*60}")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
