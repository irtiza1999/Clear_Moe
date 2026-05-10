"""
CLEAR-MoE Utilities — Configuration, seeding, logging, and checkpoint helpers.
"""

import os
import json
import random
import logging
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import numpy as np
import torch


def load_config(yaml_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return as a nested dict."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: Dict, args) -> Dict:
    """Merge CLI argparse args into config dict (CLI takes precedence)."""
    args_dict = vars(args) if hasattr(args, "__dict__") else args
    for key, value in args_dict.items():
        if value is not None:
            # Support dot-notation keys like 'data.calib_size'
            keys = key.split(".")
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
    return config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: Dict) -> torch.device:
    """Get the compute device from config, ensuring GPU is available."""
    device_str = config.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available! Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def get_dtype(config: Dict) -> torch.dtype:
    """Get the tensor dtype from config."""
    dtype_str = config.get("runtime", {}).get("dtype", "float32")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


def setup_logging(output_dir: str, run_name: Optional[str] = None) -> Path:
    """
    Set up logging to both console and file.

    Returns the run directory path.
    """
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    run_dir = Path(output_dir) / "logs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "run.log"),
        ],
    )

    return run_dir


def save_checkpoint(
    state: Dict[str, Any],
    path: str,
    filename: str = "checkpoint.pt",
):
    """Save a checkpoint dict (model state, config, metrics, etc.)."""
    save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)
    filepath = save_path / filename
    torch.save(state, filepath)
    logging.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a checkpoint dict."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    logging.info(f"Checkpoint loaded from {path}")
    return checkpoint


def save_json(data: Any, path: str):
    """Save data to a JSON file."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=_json_serializer)
    logging.info(f"JSON saved to {filepath}")


def load_json(path: str) -> Any:
    """Load data from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def _json_serializer(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def log_gpu_info():
    """Log GPU information for reproducibility."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"PyTorch version: {torch.__version__}")
    else:
        logging.warning("No CUDA GPU detected.")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_run_name(config: Dict) -> str:
    """Generate a descriptive run name from config."""
    model_name = config.get("model", {}).get("name", "model")
    task = config.get("data", {}).get("task", "task")
    n_experts = config.get("extraction", {}).get("num_experts", 0)
    method = config.get("extraction", {}).get("method", "")
    # Clean up model name for filesystem
    model_short = model_name.replace("/", "_").replace("-", "_")
    return f"{model_short}_{task}_E{n_experts}_{method}"
