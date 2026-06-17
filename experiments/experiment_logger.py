"""
Experiment logging infrastructure for CLEAR-MoE reproducibility.

Saves all fields required by Section 10 of the improvement plan:
  experiment_id, date, commit_hash, seed, backbone, dataset,
  calibration_size, selected_layers, expert_count, basis_rank,
  router, dispatch_backend, batch_size, top1, top5,
  p50_ms, p95_ms, mean_ms, std_ms, peak_vram_mb,
  load_skew, router_accuracy
"""

import csv
import datetime
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class ExperimentLogger:
    """
    Logs experiment configuration and results to JSON + CSV.

    Directory layout (under output_dir):
        summaries/<experiment_id>.json
        summaries/all_experiments.csv
        configs/<experiment_id>.json
        predictions/<experiment_id>.pt (optional)
        routing_assignments/<experiment_id>.pt (optional)
        timings/<experiment_id>.npy (optional)
    """

    REQUIRED_FIELDS = [
        "experiment_id", "date", "commit_hash", "seed",
        "backbone", "dataset", "calibration_size", "selected_layers",
        "expert_count", "basis_rank", "router", "dispatch_backend",
        "batch_size", "top1", "top5", "p50_ms", "p95_ms",
        "mean_ms", "std_ms", "peak_vram_mb", "load_skew", "router_accuracy",
    ]

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        for sub in ["summaries", "configs", "predictions", "routing_assignments", "timings", "figures", "tables"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "summaries" / "all_experiments.csv"
        self._csv_initialized = self.csv_path.exists()

    def log(self, config: Dict, results: Dict, experiment_id: Optional[str] = None) -> str:
        """
        Log one experiment. Merges config + results and saves to JSON and CSV.

        Args:
            config: Experiment configuration dict (backbone, dataset, etc.)
            results: Measured results dict (top1, top5, latency, etc.)
            experiment_id: Optional ID; auto-generated if None.

        Returns:
            experiment_id string.
        """
        if experiment_id is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backbone = config.get("backbone", "unknown").replace("/", "-")
            experiment_id = f"{backbone}_{ts}"

        summary = {
            "experiment_id": experiment_id,
            "date": datetime.datetime.now().isoformat(),
            "commit_hash": _git_commit_hash(),
        }
        summary.update(config)
        summary.update(results)

        # Ensure all required fields present
        for field in self.REQUIRED_FIELDS:
            if field not in summary:
                summary[field] = None

        # Save JSON
        json_path = self.output_dir / "summaries" / f"{experiment_id}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved experiment summary to {json_path}")

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.REQUIRED_FIELDS,
                extrasaction="ignore",
            )
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow({k: summary.get(k) for k in self.REQUIRED_FIELDS})

        # Save config separately
        cfg_path = self.output_dir / "configs" / f"{experiment_id}.json"
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        return experiment_id

    def save_artifact(self, experiment_id: str, name: str, artifact: Any):
        """Save an arbitrary artifact for an experiment.

        Supported automatic handling:
          - torch tensors or modules saved via `torch.save`
          - numpy arrays saved via `np.save`
          - Python objects saved via JSON (if serializable)
        """
        from pathlib import Path
        import torch
        import numpy as _np

        folder_map = {
            "predictions": "predictions",
            "routing": "routing_assignments",
            "timings": "timings",
            "model": "models",
            "expertized_layers": "expertized_layers",
            "fig": "figures",
            "table": "tables",
        }

        sub = folder_map.get(name, name)
        dest_dir = self.output_dir / sub
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / f"{experiment_id}_{name}.pt"

        # Torch-friendly
        try:
            if hasattr(artifact, "state_dict") or isinstance(artifact, (torch.Tensor,)):
                torch.save(artifact, dest_path)
                logger.info(f"Saved artifact {name} to {dest_path}")
                return str(dest_path)
        except Exception:
            pass

        # Numpy arrays
        try:
            if isinstance(artifact, _np.ndarray):
                _np.save(str(dest_path.with_suffix(".npy")), artifact)
                logger.info(f"Saved numpy artifact {name} to {dest_path.with_suffix('.npy')}")
                return str(dest_path.with_suffix(".npy"))
        except Exception:
            pass

        # JSON-serializable fallback
        try:
            json_path = dest_path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(artifact, f, indent=2, default=str)
            logger.info(f"Saved json artifact {name} to {json_path}")
            return str(json_path)
        except Exception as e:
            logger.warning(f"Failed to save artifact {name} for {experiment_id}: {e}")
            return None

    def load_all(self) -> List[Dict]:
        """Load all experiment summaries from the CSV."""
        if not self.csv_path.exists():
            return []
        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def print_summary(self, experiment_id: str):
        """Print a formatted summary of one experiment."""
        json_path = self.output_dir / "summaries" / f"{experiment_id}.json"
        if not json_path.exists():
            logger.warning(f"No summary found for {experiment_id}")
            return
        with open(json_path) as f:
            summary = json.load(f)
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_id}")
        print(f"{'='*60}")
        for field in self.REQUIRED_FIELDS:
            val = summary.get(field)
            if val is not None:
                print(f"  {field:<22}: {val}")
        print()
