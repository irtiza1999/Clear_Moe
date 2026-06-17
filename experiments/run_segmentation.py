"""
Segmentation experiment: ADE20K with SegFormer-B0-ADE.

Evaluates dense baseline and CLEAR-MoE on ADE20K-150 using
nvidia/segformer-b0-finetuned-ade-512-512 + scene_parse_150.

HuggingFace SegformerImageProcessor handles:
  - Resize to 512×512
  - ImageNet normalisation
  - Label shift: ADE20K labels 1-150 → 0-149; 0 → ignore_index=255

Model output: 150 logits (0-indexed), ignore_index=255.

Usage:
    python experiments/run_segmentation.py --num_calib 200 --num_val 500
"""

import argparse
import json
import logging
import sys
import os
import time
from pathlib import Path

# Insert project root AFTER stdlib/site-packages so HuggingFace 'datasets'
# package is found before the local 'datasets/' directory in this repo.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)  # append, not prepend

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES = 150
IGNORE_INDEX = 255
MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
DATASET_ID = "scene_parse_150"
OUTPUT_DIR = Path("outputs/segmentation")

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

class ADE20KDataset(torch.utils.data.Dataset):
    """
    Wraps HuggingFace scene_parse_150 (ADE20K) and applies SegFormer processor.

    Processor does_reduce_labels=True shifts:
        label 0 (background/ignore) → 255
        labels 1-150               → 0-149
    So model output and label space are both 0-indexed 0-149, ignore=255.
    """

    def __init__(self, hf_dataset, processor):
        self.hf_dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"].convert("RGB")
        annotation = item["annotation"]

        encoding = self.processor(
            images=image,
            segmentation_maps=annotation,
            return_tensors="pt",
        )
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"].squeeze(0)  # (H/4, W/4) or (512, 512)
        return pixel_values, labels


def _try_load_hf_dataset(split: str):
    """Try multiple HuggingFace ADE20K sources (datasets 4.x dropped script-based loaders)."""
    from datasets import load_dataset

    # Newer Parquet-backed sources that don't use loading scripts
    candidates = [
        dict(path="scene_parse_150",      split=split),
        dict(path="lmms-lab/ADE20K",      split=split),
        dict(path="zhangjunwp3/ade20k",   split=split),
    ]
    for kwargs in candidates:
        try:
            logger.info(f"Trying dataset source: {kwargs['path']} split={split}...")
            ds = load_dataset(**kwargs)
            logger.info(f"Loaded {len(ds)} images from {kwargs['path']}")
            return ds
        except Exception as e:
            logger.warning(f"  Failed: {e}")

    return None


def get_dataloaders(num_calib: int, num_val: int, processor, batch_size: int = 2):
    train_hf = _try_load_hf_dataset("train")
    val_hf   = _try_load_hf_dataset("validation")

    if train_hf is None or val_hf is None:
        raise RuntimeError(
            "Could not load ADE20K from HuggingFace. "
            "The 'scene_parse_150' dataset requires a loading script which is no longer "
            "supported in datasets>=4.0. Options:\n"
            "  1. pip install 'datasets<4.0' (e.g. datasets==2.21.0)\n"
            "  2. Download ADE20K manually: \n"
            "     curl -O http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip\n"
            "     unzip ADEChallengeData2016.zip -d data/ade20k\n"
            "     Then rerun with --ade20k_dir data/ade20k/ADEChallengeData2016\n"
            "  3. Use Cityscapes: download from https://www.cityscapes-dataset.com\n"
            "     and rerun with --cityscapes_dir data/cityscapes"
        )

    # Reproducible calibration subset from training set
    rng = np.random.RandomState(42)
    calib_idx = rng.choice(len(train_hf), size=min(num_calib, len(train_hf)), replace=False).tolist()
    calib_subset = train_hf.select(calib_idx)

    # Validation subset
    val_idx = list(range(min(num_val, len(val_hf))))
    val_subset = val_hf.select(val_idx)

    calib_ds = ADE20KDataset(calib_subset, processor)
    val_ds   = ADE20KDataset(val_subset,   processor)

    calib_loader = DataLoader(calib_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Calibration: {len(calib_ds)} images | Val: {len(val_ds)} images")
    return calib_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_segmentation(model, dataloader, device, label_dtype=torch.long):
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

    for pixel_values, labels in tqdm(dataloader, desc="Segmentation eval"):
        pixel_values = pixel_values.to(device)

        out = model(pixel_values=pixel_values)
        logits = out.logits  # (B, 150, H/4, W/4)

        # Upsample to label resolution
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )

        preds = logits.argmax(dim=1).cpu()  # (B, H, W)

        for pred, target in zip(preds, labels):
            valid = (target != IGNORE_INDEX) & (target >= 0) & (target < NUM_CLASSES)
            p = pred[valid].long()
            t = target[valid].long()
            combined = NUM_CLASSES * t + p
            counts = torch.bincount(combined, minlength=NUM_CLASSES * NUM_CLASSES)
            confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES)

    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    valid = union > 0

    per_class_iou = intersection.float() / (union.float() + 1e-8)
    miou = per_class_iou[valid].mean().item() * 100

    per_class_acc = intersection.float() / (confusion.sum(1).float() + 1e-8)
    mean_acc = per_class_acc[valid].mean().item() * 100

    return {
        "miou": round(miou, 4),
        "mean_accuracy": round(mean_acc, 4),
        "num_valid_classes": valid.sum().item(),
    }


@torch.no_grad()
def measure_latency(model, device, input_shape=(1, 3, 512, 512), warmup=10, iters=50):
    model.eval()
    dummy = torch.randn(*input_shape, device=device)

    for _ in range(warmup):
        _ = model(pixel_values=dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(pixel_values=dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "img_per_sec": round(1000.0 / float(np.percentile(latencies, 50)), 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLEAR-MoE expert extraction (inline, no external config)
# ──────────────────────────────────────────────────────────────────────────────

def get_segformer_ffn_layers(model):
    """Return (name, module) pairs for SegFormer FFN dense layers."""
    layers = []
    for name, module in model.named_modules():
        # SegFormer FFN: segformer.encoder.block.*.*.layer.*.dense
        if isinstance(module, nn.Linear) and "dense" in name and "intermediate" not in name:
            if any(f"block.{i}" in name for i in range(12)):
                layers.append((name, module))
    return layers


def score_layers(model, calib_loader, device, num_experts=4):
    """
    Score SegFormer FFN blocks using calibration activations.
    Returns list of (score, layer_name, module) sorted descending.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans

    activations = {}
    hooks = []

    # Capture pre-FFN activations from each transformer block's output
    block_names = []
    for name, module in model.named_modules():
        # SegFormer: segformer.encoder.block.i.j = each attention+ffn block
        parts = name.split(".")
        if (len(parts) >= 4
                and parts[0] == "segformer"
                and parts[1] == "encoder"
                and parts[2] == "block"
                and parts[-1] == "layer_norm_1"):
            block_names.append(name)

    if not block_names:
        logger.warning("Could not find SegFormer block norms; falling back to all LayerNorm")
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm) and "encoder" in name:
                block_names.append(name)

    for name in block_names:
        def make_hook(n):
            def hook(m, inp, out):
                act = out.detach().cpu()
                if act.dim() == 3:
                    act = act.reshape(-1, act.shape[-1])
                activations.setdefault(n, []).append(act)
            return hook
        mod = dict(model.named_modules())[name]
        hooks.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    n_batches = 0
    with torch.no_grad():
        for pv, _ in calib_loader:
            model(pixel_values=pv.to(device))
            n_batches += 1
            if n_batches >= 10:
                break

    for h in hooks:
        h.remove()

    scores = []
    for name, acts in activations.items():
        act = torch.cat(acts, dim=0).numpy()
        if act.shape[0] < num_experts * 2:
            continue

        # Sparsity
        sparsity = float((np.abs(act) < 0.01).mean())

        # Clusterability (silhouette, subsample for speed)
        n_sub = min(2000, act.shape[0])
        idx = np.random.choice(act.shape[0], n_sub, replace=False)
        sub = act[idx]
        try:
            km = KMeans(n_clusters=num_experts, n_init=3, random_state=42)
            labels_km = km.fit_predict(sub)
            sil = float(silhouette_score(sub, labels_km))
        except Exception:
            sil = 0.0

        # Sensitivity: proxy = activation norm variance (high variance = sensitive)
        norms = np.linalg.norm(act, axis=1)
        sensitivity = float(np.std(norms) / (np.mean(norms) + 1e-8))
        sensitivity = min(1.0, sensitivity)

        composite = 0.4 * sparsity + 0.4 * sil - 0.2 * sensitivity
        scores.append((composite, name))
        logger.info(f"  {name}: sp={sparsity:.3f} sil={sil:.3f} se={sensitivity:.3f} → {composite:.3f}")

    scores.sort(reverse=True)
    return scores


class ClearMoESegFFN(nn.Module):
    """
    Shared-basis MoE replacement for a SegFormer FFN dense sub-layer.

    output = W_shared @ z + W_e_res[e] @ z
    where z is passed in directly (the FFN computes fc1 externally for SegFormer).
    """

    def __init__(self, W_shared: torch.Tensor, W_res_list: list, bias: torch.Tensor):
        """
        Args:
            W_shared: (out, in) shared SVD approximation of fc2 weight
            W_res_list: list of E (out, in) per-cluster residual weights
            bias: (out,) bias from original fc2
        """
        super().__init__()
        self.W_shared = nn.Parameter(W_shared)
        self.W_res = nn.ParameterList([nn.Parameter(w) for w in W_res_list])
        self.bias = nn.Parameter(bias)
        self.num_experts = len(W_res_list)

        # Simple linear router
        in_dim = W_shared.shape[1]
        self.router = nn.Linear(in_dim, self.num_experts, bias=True)
        nn.init.kaiming_uniform_(self.router.weight, a=0.01)
        nn.init.zeros_(self.router.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        shared = F.linear(z, self.W_shared, self.bias)

        logits = self.router(z)
        expert_idx = logits.argmax(dim=-1)

        res = torch.zeros_like(shared)
        for e in range(self.num_experts):
            mask = (expert_idx == e)
            if mask.any():
                res[mask] = F.linear(z[mask], self.W_res[e])

        return shared + res


def extract_moe_for_segformer(model, calib_loader, device,
                               num_experts=4, num_layers=4, rank_frac=0.5):
    """
    Extract CLEAR-MoE experts for the top-scored SegFormer FFN layers.
    Targets the 'dense' layers in SegFormer's FFN (intermediate output projections).
    """
    from sklearn.cluster import KMeans

    scores = score_layers(model, calib_loader, device, num_experts=num_experts)
    if not scores:
        logger.warning("No scoreable layers found; returning model unchanged")
        return model, []

    top_layer_names = [name for _, name in scores[:num_layers]]
    logger.info(f"Selected {len(top_layer_names)} layers for MoE conversion")
    for s, n in scores[:num_layers]:
        logger.info(f"  {n}: {s:.3f}")

    # Collect activations at selected blocks to do k-means
    block_acts = {}
    hooks = []

    for layer_name in top_layer_names:
        def make_hook(n):
            def hook(m, inp, out):
                act = out.detach().cpu()
                if act.dim() == 3:
                    act = act.reshape(-1, act.shape[-1])
                block_acts.setdefault(n, []).append(act)
            return hook
        mod = dict(model.named_modules())[layer_name]
        hooks.append(mod.register_forward_hook(make_hook(layer_name)))

    model.eval()
    n_batches = 0
    with torch.no_grad():
        for pv, _ in calib_loader:
            model(pixel_values=pv.to(device))
            n_batches += 1
            if n_batches >= 10:
                break

    for h in hooks:
        h.remove()

    # For each selected block, find the corresponding FFN output projection (dense)
    # SegFormer FFN structure: LayerNorm → dense → GELU → dense (output proj)
    # The block norm "block.i.j.layer_norm_1" is followed by block.i.j.layer.1
    # (the MixFFN). Within MixFFN: dense = linear output projection.

    replaced = []
    for layer_name in top_layer_names:
        acts_list = block_acts.get(layer_name, [])
        if not acts_list:
            continue
        acts = torch.cat(acts_list, dim=0).numpy()

        # Find the FFN output projection associated with this block
        # layer_name = "segformer.encoder.block.i.j.layer_norm_1"
        # FFN output proj = "segformer.encoder.block.i.j.layer.1.dense"
        block_prefix = layer_name.replace(".layer_norm_1", "")
        ffn_dense_name = block_prefix + ".layer.1.dense"

        # Try to get the module
        target_mod = None
        for n, m in model.named_modules():
            if n == ffn_dense_name and isinstance(m, nn.Linear):
                target_mod = (n, m)
                break

        if target_mod is None:
            logger.warning(f"Could not find FFN dense for {layer_name}; skipping")
            continue

        fname, fmod = target_mod
        W = fmod.weight.data.float().cpu()  # (out, in)
        b = fmod.bias.data.float().cpu() if fmod.bias is not None else torch.zeros(W.shape[0])

        # SVD decomposition
        r = max(1, int(W.shape[1] * rank_frac))
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = min(r, S.shape[0])
        W_shared = (U[:, :r] * S[:r]) @ Vh[:r, :]

        # k-means on calibration activations → cluster labels
        n_sub = min(4000, acts.shape[0])
        idx = np.random.choice(acts.shape[0], n_sub, replace=False)
        sub = acts[idx]
        km = KMeans(n_clusters=num_experts, n_init=5, random_state=42)
        cluster_labels = km.fit_predict(sub)

        # Per-cluster residuals
        W_res_list = []
        global_mean_norm = np.linalg.norm(sub, axis=1).mean() + 1e-8
        for e in range(num_experts):
            e_mask = (cluster_labels == e)
            if e_mask.sum() == 0:
                W_res_list.append(torch.zeros_like(W - W_shared))
                continue
            e_acts = sub[e_mask]
            e_norm = np.linalg.norm(e_acts, axis=1).mean()
            scale = float(e_norm / global_mean_norm)
            W_res_list.append((W - W_shared) * scale)

        moe_module = ClearMoESegFFN(W_shared, W_res_list, b)

        # Fit router on calibration activations
        logger.info(f"  Fitting router for {fname}...")
        _fit_router(moe_module.router, torch.tensor(sub, dtype=torch.float32),
                    torch.tensor(cluster_labels, dtype=torch.long))
        moe_module = moe_module.to(device)

        # Replace the dense layer with a wrapper
        _set_nested_attr(model, fname, _DenseReplacer(moe_module, fmod.in_features))
        replaced.append(fname)
        logger.info(f"  Replaced {fname} with ClearMoESegFFN ({num_experts} experts)")

    logger.info(f"MoE conversion complete: {len(replaced)} layers replaced")
    return model, replaced


class _DenseReplacer(nn.Module):
    """Drops into SegFormer FFN as a replacement for the output dense layer."""

    def __init__(self, moe_ffn: ClearMoESegFFN, in_features: int):
        super().__init__()
        self.moe = moe_ffn
        self.in_features = in_features
        self.out_features = moe_ffn.W_shared.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        z = x.reshape(-1, shape[-1])
        out = self.moe(z)
        return out.reshape(*shape[:-1], -1)


def _fit_router(router: nn.Linear, acts: torch.Tensor, labels: torch.Tensor,
                epochs=5, lr=1e-3):
    """Supervised router training from k-means labels."""
    from torch.utils.data import TensorDataset, DataLoader as DL

    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=0.01)
    ds = DL(TensorDataset(acts, labels), batch_size=256, shuffle=True)
    router.train()
    for _ in range(epochs):
        for xa, ya in ds:
            logits = router(xa)
            loss = F.cross_entropy(logits, ya)
            opt.zero_grad()
            loss.backward()
            opt.step()
    router.eval()


def _set_nested_attr(model: nn.Module, name: str, value: nn.Module):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], value)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_calib", type=int, default=200)
    parser.add_argument("--num_val",   type=int, default=500)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--num_layers",  type=int, default=4)
    parser.add_argument("--rank_frac",   type=float, default=0.5)
    parser.add_argument("--batch_size",  type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    (OUTPUT_DIR / "summaries").mkdir(parents=True, exist_ok=True)

    # Load model + processor
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    logger.info(f"Loading {MODEL_ID}...")
    processor = SegformerImageProcessor.from_pretrained(MODEL_ID, do_reduce_labels=True)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model = model.to(device)
    model.eval()

    # Data
    calib_loader, val_loader = get_dataloaders(
        args.num_calib, args.num_val, processor, batch_size=args.batch_size
    )

    # ── Dense baseline ──
    logger.info("=" * 60)
    logger.info("DENSE BASELINE")
    logger.info("=" * 60)

    dense_seg = evaluate_segmentation(model, val_loader, device)
    dense_lat = measure_latency(model, device)

    dense_results = {
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "config": "dense",
        "num_val": args.num_val,
        **dense_seg,
        **dense_lat,
    }
    logger.info(f"Dense: mIoU={dense_seg['miou']:.2f}%, mean_acc={dense_seg['mean_accuracy']:.2f}%")
    logger.info(f"Dense latency: p50={dense_lat['p50_ms']}ms, {dense_lat['img_per_sec']} img/s")

    with open(OUTPUT_DIR / "summaries" / "dense_seg_b0.json", "w") as f:
        json.dump(dense_results, f, indent=2)

    # ── CLEAR-MoE extraction ──
    logger.info("=" * 60)
    logger.info(f"CLEAR-MoE EXTRACTION (E={args.num_experts}, layers={args.num_layers})")
    logger.info("=" * 60)

    import copy
    moe_model = copy.deepcopy(model)
    moe_model, replaced_layers = extract_moe_for_segformer(
        moe_model, calib_loader, device,
        num_experts=args.num_experts,
        num_layers=args.num_layers,
        rank_frac=args.rank_frac,
    )
    moe_model.eval()

    # ── MoE evaluation ──
    logger.info("=" * 60)
    logger.info("CLEAR-MoE EVALUATION")
    logger.info("=" * 60)

    moe_seg = evaluate_segmentation(moe_model, val_loader, device)
    moe_lat = measure_latency(moe_model, device)

    delta_miou = moe_seg["miou"] - dense_seg["miou"]

    moe_results = {
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "config": "clear_moe",
        "num_experts": args.num_experts,
        "num_layers_replaced": len(replaced_layers),
        "replaced_layers": replaced_layers,
        "rank_frac": args.rank_frac,
        **moe_seg,
        **moe_lat,
        "delta_miou": round(delta_miou, 4),
    }
    logger.info(f"MoE: mIoU={moe_seg['miou']:.2f}% (Δ{delta_miou:+.2f}), "
                f"mean_acc={moe_seg['mean_accuracy']:.2f}%")
    logger.info(f"MoE latency: p50={moe_lat['p50_ms']}ms, {moe_lat['img_per_sec']} img/s")

    with open(OUTPUT_DIR / "summaries" / "moe_seg_b0.json", "w") as f:
        json.dump(moe_results, f, indent=2)

    # Summary
    summary = {
        "dense": dense_results,
        "moe": moe_results,
    }
    summary_path = OUTPUT_DIR / "summaries" / "segmentation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to {summary_path}")
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Dense   : mIoU={dense_seg['miou']:.2f}%, mean_acc={dense_seg['mean_accuracy']:.2f}%, "
                f"p50={dense_lat['p50_ms']}ms")
    logger.info(f"MoE(E={args.num_experts}): mIoU={moe_seg['miou']:.2f}%, mean_acc={moe_seg['mean_accuracy']:.2f}%, "
                f"p50={moe_lat['p50_ms']}ms, Δ={delta_miou:+.2f}pp")


if __name__ == "__main__":
    main()
