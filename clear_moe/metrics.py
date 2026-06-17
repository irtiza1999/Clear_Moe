"""
CLEAR-MoE Metrics — Evaluation metrics for classification, segmentation,
latency, throughput, energy, and structural analysis.

Provides unified measurement for both the dense baseline and
the extracted MoE model to enable fair comparison.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Classification metrics
# ──────────────────────────────────────────────────────────────────────

# ImageNette class mapping for ImageNet-1K models
IMAGENETTE_TO_IMAGENET = {
    0: 0,    # tench
    1: 217,  # English springer
    2: 482,  # cassette player
    3: 491,  # chain saw
    4: 497,  # church
    5: 566,  # French horn
    6: 569,  # garbage truck
    7: 571,  # gas pump
    8: 574,  # golf ball
    9: 701,  # parachute
}


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    top_k: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """
    Evaluate classification accuracy (Top-1, Top-5).

    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        device: Torch device
        top_k: Tuple of k values for top-k accuracy

    Returns:
        Dict with 'top1', 'top5', 'total_samples'
    """
    model.eval()
    correct = {k: 0 for k in top_k}
    total = 0
    needs_mapping = None

    for images, targets in tqdm(dataloader, desc="Evaluating classification"):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        if hasattr(outputs, "logits"):
            outputs = outputs.logits

        # Detect mapping need on first batch
        if needs_mapping is None:
            needs_mapping = (outputs.shape[1] == 1000 and targets.max() < 10)

        if needs_mapping:
            mapped_targets = torch.tensor(
                [IMAGENETTE_TO_IMAGENET.get(t.item(), t.item()) for t in targets],
                device=device,
            )
        else:
            mapped_targets = targets

        batch_size = targets.shape[0]
        total += batch_size

        for k in top_k:
            _, preds = outputs.topk(k, dim=1, largest=True, sorted=True)
            correct_k = preds.eq(mapped_targets.view(-1, 1).expand_as(preds)).any(dim=1)
            correct[k] += correct_k.sum().item()

        del images, targets, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {}
    for k in top_k:
        acc = 100.0 * correct[k] / total
        results[f"top{k}"] = acc
        logger.info(f"  Top-{k} accuracy: {acc:.2f}%")

    results["total_samples"] = total
    return results


# ──────────────────────────────────────────────────────────────────────
# Segmentation metrics
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_segmentation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 19,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """
    Evaluate semantic segmentation (mIoU, per-class IoU).

    Args:
        model: Segmentation model
        dataloader: Validation dataloader
        device: Torch device
        num_classes: Number of semantic classes
        ignore_index: Label index to ignore

    Returns:
        Dict with 'miou', 'per_class_iou', 'mean_accuracy'
    """
    model.eval()

    # Confusion matrix for IoU computation
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for images, targets in tqdm(dataloader, desc="Evaluating segmentation"):
        images = images.to(device)

        outputs = model(images)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        # Upsample predictions to target size if needed
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = nn.functional.interpolate(
                logits, size=targets.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        preds = logits.argmax(dim=1).cpu()
        targets = targets.cpu()

        # Handle different target shapes
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        # Update confusion matrix using bincount (correct and efficient)
        for pred, target in zip(preds, targets):
            if target.shape == pred.shape:
                valid_mask = (target >= 0) & (target < num_classes) & (target != ignore_index)
                p = pred[valid_mask].long().clamp(0, num_classes - 1)
                t = target[valid_mask].long()
                combined = num_classes * t + p
                confusion += torch.bincount(combined, minlength=num_classes * num_classes).reshape(
                    num_classes, num_classes
                )

        del images, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute per-class IoU from confusion matrix
    intersection = confusion.diag()
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection

    per_class_iou = intersection.float() / (union.float() + 1e-8)

    # Only count classes that appear in ground truth
    valid_classes = union > 0
    miou = per_class_iou[valid_classes].mean().item() * 100

    # Mean accuracy
    per_class_acc = intersection.float() / (confusion.sum(dim=1).float() + 1e-8)
    mean_acc = per_class_acc[valid_classes].mean().item() * 100

    results = {
        "miou": miou,
        "per_class_iou": per_class_iou.tolist(),
        "mean_accuracy": mean_acc,
        "num_valid_classes": valid_classes.sum().item(),
    }

    logger.info(f"  mIoU: {miou:.2f}%")
    logger.info(f"  Mean accuracy: {mean_acc:.2f}%")

    return results


# ──────────────────────────────────────────────────────────────────────
# Latency and throughput
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    warmup: int = 20,
    iterations: int = 100,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """
    Measure model inference latency.

    Args:
        model: Model to benchmark
        input_shape: (batch, channels, height, width)
        device: Torch device
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        dtype: Tensor dtype

    Returns:
        Dict with p50, p90, mean, std latency in ms
    """
    model.eval()
    dummy_input = torch.randn(*input_shape, device=device, dtype=dtype)

    # Warmup
    for _ in range(warmup):
        _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)

    results = {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p90_ms": float(np.percentile(latencies, 90)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
    }

    logger.info(
        f"  Latency: mean={results['latency_mean_ms']:.2f}ms, "
        f"p50={results['latency_p50_ms']:.2f}ms, "
        f"p95={results['latency_p95_ms']:.2f}ms"
    )

    return results


@torch.no_grad()
def measure_throughput(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> Dict[str, float]:
    """
    Measure model throughput (images per second).

    Args:
        model: Model to benchmark
        dataloader: Data loader
        device: Torch device
        max_batches: Maximum batches to process

    Returns:
        Dict with images_per_second, batches_per_second
    """
    model.eval()
    total_images = 0
    total_time = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        images = images.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        _ = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        total_images += images.shape[0]
        total_time += end - start

    throughput = total_images / total_time if total_time > 0 else 0

    results = {
        "images_per_second": throughput,
        "batches_per_second": max_batches / total_time if total_time > 0 else 0,
        "total_images": total_images,
        "total_time_s": total_time,
    }

    logger.info(f"  Throughput: {throughput:.1f} images/s")

    return results


# ──────────────────────────────────────────────────────────────────────
# Energy measurement
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_energy(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    iterations: int = 50,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """
    Measure GPU energy consumption per inference using pynvml.

    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        device: Torch device
        iterations: Number of iterations
        dtype: Tensor dtype

    Returns:
        Dict with energy_per_image_mJ, avg_power_W
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        logger.warning(f"pynvml not available for energy measurement: {e}")
        return {"energy_per_image_mJ": -1.0, "avg_power_W": -1.0}

    model.eval()
    dummy_input = torch.randn(*input_shape, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    power_samples = []
    start_time = time.perf_counter()

    for _ in range(iterations):
        _ = model(dummy_input)
        # Sample power (milliwatts)
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_samples.append(power_mw / 1000.0)  # Convert to watts
        except Exception:
            pass

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    total_time_s = end_time - start_time

    pynvml.nvmlShutdown()

    if power_samples:
        avg_power_w = np.mean(power_samples)
        total_energy_j = avg_power_w * total_time_s
        energy_per_image_mj = (total_energy_j / (iterations * input_shape[0])) * 1000
    else:
        avg_power_w = -1.0
        energy_per_image_mj = -1.0

    results = {
        "energy_per_image_mJ": energy_per_image_mj,
        "avg_power_W": avg_power_w,
        "total_energy_J": total_energy_j if power_samples else -1.0,
        "num_power_samples": len(power_samples),
    }

    logger.info(
        f"  Energy: {energy_per_image_mj:.2f} mJ/image, "
        f"avg power: {avg_power_w:.1f} W"
    )

    return results


# ──────────────────────────────────────────────────────────────────────
# Structural metrics (parameter and FLOP counting)
# ──────────────────────────────────────────────────────────────────────

def count_active_params(model: nn.Module) -> Dict[str, int]:
    """
    Count total and active parameters in the model.

    For MoE models, 'active' counts only the shared basis + one expert
    (since only one expert is active per token in top-1 routing).
    """
    total_params = sum(p.numel() for p in model.parameters())

    # Try to identify MoE layers and compute active params
    active_params = total_params  # Default: all params are active (dense model)
    moe_total = 0
    moe_active = 0

    for name, module in model.named_modules():
        module_type = type(module).__name__
        if "GroupedExpertMLP" in module_type or "MoE" in module_type:
            mod_params = sum(p.numel() for p in module.parameters())
            moe_total += mod_params

            # Active = shared basis + one expert
            if hasattr(module, "shared_fc1"):
                shared = sum(p.numel() for p in module.shared_fc1.parameters())
                if hasattr(module, "shared_fc2"):
                    shared += sum(p.numel() for p in module.shared_fc2.parameters())
                if hasattr(module, "expert_fc1s") and len(module.expert_fc1s) > 0:
                    one_expert = sum(p.numel() for p in module.expert_fc1s[0].parameters())
                    if hasattr(module, "expert_fc2s") and len(module.expert_fc2s) > 0:
                        one_expert += sum(p.numel() for p in module.expert_fc2s[0].parameters())
                    moe_active += shared + one_expert
                else:
                    moe_active += shared
            else:
                moe_active += mod_params

    if moe_total > 0:
        active_params = total_params - moe_total + moe_active

    results = {
        "total_params": total_params,
        "active_params": active_params,
        "moe_total_params": moe_total,
        "moe_active_params": moe_active,
        "param_reduction_pct": 100 * (1 - active_params / total_params) if total_params > 0 else 0,
    }

    logger.info(
        f"  Params: {total_params:,} total, {active_params:,} active "
        f"({results['param_reduction_pct']:.1f}% reduction)"
    )

    return results


def estimate_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
) -> Dict[str, float]:
    """
    Estimate FLOPs (MACs) for the model.

    Uses a simple forward-hook based counter focused on Linear and Conv2d layers.

    Args:
        model: Model to analyze
        input_shape: (batch, channels, height, width)

    Returns:
        Dict with total_macs, ffn_macs, active_ffn_macs
    """
    macs = {"total": 0, "ffn": 0, "attention": 0, "other": 0}
    hooks = []

    def count_hook(module, input, output, name=""):
        if isinstance(module, nn.Linear):
            # MACs = in_features * out_features * batch_elements
            inp = input[0]
            batch_elements = inp.numel() // inp.shape[-1]
            layer_macs = module.in_features * module.out_features * batch_elements
            macs["total"] += layer_macs

            # Classify as FFN or attention
            if any(k in name for k in ["mlp", "ffn", "fc1", "fc2", "dense"]):
                macs["ffn"] += layer_macs
            elif any(k in name for k in ["attn", "attention", "qkv"]):
                macs["attention"] += layer_macs
            else:
                macs["other"] += layer_macs

        elif isinstance(module, nn.Conv2d):
            inp = input[0]
            batch_size = inp.shape[0]
            out_h, out_w = output.shape[2], output.shape[3]
            layer_macs = (
                module.in_channels * module.out_channels
                * module.kernel_size[0] * module.kernel_size[1]
                * out_h * out_w * batch_size
                // module.groups
            )
            macs["total"] += layer_macs
            macs["other"] += layer_macs

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hook = module.register_forward_hook(
                lambda m, i, o, n=name: count_hook(m, i, o, n)
            )
            hooks.append(hook)

    # Forward pass
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        _ = model(dummy)

    for hook in hooks:
        hook.remove()

    results = {
        "total_macs": macs["total"],
        "ffn_macs": macs["ffn"],
        "attention_macs": macs["attention"],
        "other_macs": macs["other"],
        "total_gmacs": macs["total"] / 1e9,
        "ffn_gmacs": macs["ffn"] / 1e9,
    }

    logger.info(
        f"  MACs: {results['total_gmacs']:.2f}G total, "
        f"{results['ffn_gmacs']:.2f}G FFN"
    )

    return results


def compute_memory_usage(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> Dict[str, float]:
    """Measure peak GPU memory usage during inference."""
    if device.type != "cuda":
        return {"peak_memory_MB": -1}

    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        _ = model(dummy)

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    results = {"peak_memory_MB": peak_mem}
    logger.info(f"  Peak memory: {peak_mem:.1f} MB")

    return results
