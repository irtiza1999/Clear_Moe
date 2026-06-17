"""
CLEAR-MoE Scoring — Layer scoring for expertization selection.

Scores each FFN layer on three factors:
1. Activation sparsity — fraction of near-zero activations
2. Multimodality / clusterability — how well activations cluster (silhouette score)
3. Output sensitivity — how much removing this FFN affects model output

Layers with highest composite scores are selected for expertization.
This implements the key insight from AdaMV-MoE: later layers tend to be
more specialized and better candidates for expertization.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class LayerScore:
    """Composite score for a single FFN layer."""
    name: str
    layer_index: int
    sparsity: float             # [0, 1] — higher = more sparse
    multimodality: float        # [0, 1] — higher = more clusterable
    sensitivity: float          # [0, 1] — higher = more sensitive (removal hurts more)
    composite: float = 0.0      # Weighted combination
    selected: bool = False      # Whether this layer is selected for expertization

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "layer_index": self.layer_index,
            "sparsity": round(self.sparsity, 4),
            "multimodality": round(self.multimodality, 4),
            "sensitivity": round(self.sensitivity, 4),
            "composite": round(self.composite, 4),
            "selected": self.selected,
        }


def score_layers(
    activations: Dict[str, torch.Tensor],
    model: nn.Module,
    layer_infos: list,
    device: torch.device,
    calib_loader=None,
    num_clusters: int = 4,
    sparsity_threshold: float = 0.01,
    weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
    max_samples_for_clustering: int = 5000,
) -> List[LayerScore]:
    """
    Score all FFN layers for expertization suitability.

    Args:
        activations: Dict mapping layer name to activation tensor (N, D)
        model: The pretrained model (needed for sensitivity analysis)
        layer_infos: List of FFNLayerInfo from models.get_ffn_layers
        device: Torch device
        calib_loader: Calibration dataloader (for sensitivity analysis)
        num_clusters: Number of clusters for multimodality scoring
        sparsity_threshold: Threshold for near-zero activation detection
        weights: (w_sparsity, w_multimodality, w_sensitivity) weights
        max_samples_for_clustering: Subsample limit for KMeans

    Returns:
        List of LayerScore, one per FFN layer, sorted by composite score (descending)
    """
    w_s, w_m, w_sens = weights
    scores = []

    for info in tqdm(layer_infos, desc="Scoring layers"):
        name = info.name
        act = activations.get(name)

        if act is None:
            logger.warning(f"No activations for layer {name}, skipping")
            continue

        # 1. Activation sparsity
        sparsity = compute_sparsity(act, threshold=sparsity_threshold)

        # 2. Multimodality / clusterability
        multimodality = compute_multimodality(
            act,
            num_clusters=num_clusters,
            max_samples=max_samples_for_clustering,
        )

        # 3. Output sensitivity
        if calib_loader is not None:
            sensitivity = compute_sensitivity(
                model, info, device, calib_loader
            )
        else:
            # Default: use layer position as proxy (later layers = more sensitive)
            num_layers = len(layer_infos)
            sensitivity = (info.layer_index + 1) / num_layers

        # Composite score: sensitivity is subtracted (high sensitivity = riskier to expertize)
        composite = w_s * sparsity + w_m * multimodality - w_sens * sensitivity

        scores.append(LayerScore(
            name=name,
            layer_index=info.layer_index,
            sparsity=sparsity,
            multimodality=multimodality,
            sensitivity=sensitivity,
            composite=composite,
        ))

    # Sort by composite score (descending)
    scores.sort(key=lambda s: s.composite, reverse=True)

    logger.info("Layer scores (sorted by composite):")
    for s in scores:
        logger.info(
            f"  [{s.layer_index:2d}] {s.name}: "
            f"sp={s.sparsity:.3f} mm={s.multimodality:.3f} "
            f"se={s.sensitivity:.3f} => {s.composite:.3f}"
        )

    return scores


def compute_sparsity(
    activations: torch.Tensor,
    threshold: float = 0.01,
) -> float:
    """
    Compute activation sparsity for a layer.

    Measures the fraction of activation values that are near-zero,
    indicating natural sparsity in the FFN's response.

    Args:
        activations: (N, D) tensor of FFN input activations
        threshold: Magnitude threshold for "near-zero"

    Returns:
        Sparsity score in [0, 1]
    """
    act_np = activations.float().numpy() if isinstance(activations, torch.Tensor) else activations

    # Fraction of elements below threshold
    near_zero = np.abs(act_np) < threshold
    sparsity = near_zero.mean()

    # Also compute the Hoyer sparsity (L1/L2 ratio based)
    # This is more robust than simple thresholding
    l1_norms = np.abs(act_np).sum(axis=-1)
    l2_norms = np.sqrt((act_np ** 2).sum(axis=-1))

    d = act_np.shape[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        hoyer = (np.sqrt(d) - l1_norms / (l2_norms + 1e-8)) / (np.sqrt(d) - 1 + 1e-8)
        hoyer = np.clip(hoyer, 0, 1)

    # Combine threshold-based and Hoyer sparsity
    combined = 0.5 * sparsity + 0.5 * hoyer.mean()

    return float(combined)


def compute_multimodality(
    activations: torch.Tensor,
    num_clusters: int = 4,
    max_samples: int = 5000,
) -> float:
    """
    Compute multimodality / clusterability score for a layer.

    Uses KMeans clustering and silhouette score to measure how well
    the activation space naturally partitions into distinct modes.
    Higher silhouette = more natural expert structure.

    Args:
        activations: (N, D) tensor of FFN input activations
        num_clusters: Number of clusters to try
        max_samples: Maximum samples for clustering (subsampled if needed)

    Returns:
        Multimodality score in [0, 1]
    """
    act_np = activations.float().numpy()

    # Subsample if too many activations
    if act_np.shape[0] > max_samples:
        indices = np.random.choice(act_np.shape[0], max_samples, replace=False)
        act_np = act_np[indices]

    # Dimensionality reduction if D is very large (for speed)
    if act_np.shape[1] > 256:
        # Use random projection for speed
        proj = np.random.randn(act_np.shape[1], 64).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        act_np = act_np @ proj

    try:
        # Use MiniBatchKMeans for efficiency
        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            batch_size=min(1024, act_np.shape[0]),
            n_init=3,
            random_state=42,
        )
        labels = kmeans.fit_predict(act_np)

        # Silhouette score: [-1, 1], remap to [0, 1]
        if len(np.unique(labels)) < 2:
            return 0.0

        # Use a subsample for silhouette computation (expensive)
        sil_samples = min(2000, act_np.shape[0])
        sil_indices = np.random.choice(act_np.shape[0], sil_samples, replace=False)
        sil = silhouette_score(
            act_np[sil_indices],
            labels[sil_indices],
            sample_size=None,
        )

        # Remap from [-1, 1] to [0, 1]
        multimodality = (sil + 1.0) / 2.0

    except Exception as e:
        logger.warning(f"Clustering failed: {e}")
        multimodality = 0.5  # Neutral default

    return float(multimodality)


@torch.no_grad()
def compute_sensitivity(
    model: nn.Module,
    layer_info,
    device: torch.device,
    calib_loader,
    max_batches: int = 5,
) -> float:
    """
    Compute output sensitivity when zeroing out an FFN layer.

    Measures how much the model's output changes when this layer's
    FFN is bypassed (set to zero). Higher sensitivity means the layer
    is more important and thus a riskier but potentially more impactful
    target for expertization.

    Args:
        model: The pretrained model
        layer_info: FFNLayerInfo for the layer to test
        device: Torch device
        calib_loader: Calibration dataloader
        max_batches: Number of batches to average over

    Returns:
        Sensitivity score in [0, 1]
    """
    model.eval()

    # Get the FFN module
    parts = layer_info.name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    ffn_module = getattr(parent, parts[-1])

    # Collect original outputs
    original_outputs = []
    zeroed_outputs = []

    for batch_idx, (images, _) in enumerate(calib_loader):
        if batch_idx >= max_batches:
            break

        images = images.to(device)

        # Forward with original FFN
        out_original = model(images)
        if hasattr(out_original, "logits"):
            out_original = out_original.logits
        original_outputs.append(out_original.detach().cpu())

    # Zero out the FFN (replace with identity-like behavior)
    # Save original forward
    original_forward = ffn_module.forward

    def zero_forward(*args, **kwargs):
        """Replace FFN output with zeros (effectively skipping the FFN)."""
        out = original_forward(*args, **kwargs)
        return torch.zeros_like(out)

    ffn_module.forward = zero_forward

    for batch_idx, (images, _) in enumerate(calib_loader):
        if batch_idx >= max_batches:
            break

        images = images.to(device)
        out_zeroed = model(images)
        if hasattr(out_zeroed, "logits"):
            out_zeroed = out_zeroed.logits
        zeroed_outputs.append(out_zeroed.detach().cpu())

    # Restore original forward
    ffn_module.forward = original_forward

    # Compute relative change
    if original_outputs and zeroed_outputs:
        orig = torch.cat(original_outputs, dim=0).float()
        zero = torch.cat(zeroed_outputs, dim=0).float()

        # Normalized L2 distance
        diff = (orig - zero).norm(dim=-1)
        orig_norm = orig.norm(dim=-1) + 1e-8
        relative_change = (diff / orig_norm).mean().item()

        # Clip and normalize to [0, 1] (empirically relative_change is often < 2.0)
        sensitivity = min(relative_change / 2.0, 1.0)
    else:
        sensitivity = 0.5

    return float(sensitivity)


def select_layers(
    scores: List[LayerScore],
    strategy: str = "last_half",
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    total_layers: Optional[int] = None,
) -> List[LayerScore]:
    """
    Select which layers to expertize based on scores and strategy.

    Strategies:
    - 'last_half': Expertize layers in the last half of the backbone
    - 'last_third': Expertize layers in the last third
    - 'top_k': Expertize the top-k scoring layers
    - 'threshold': Expertize all layers above a composite threshold
    - 'combined': Intersect top-scoring with positional preference (recommended)

    Args:
        scores: List of LayerScore (sorted by composite, descending)
        strategy: Selection strategy
        top_k: Number of layers for 'top_k' strategy
        threshold: Score threshold for 'threshold' strategy
        total_layers: Total number of FFN layers (for positional strategies)

    Returns:
        Updated list with selected=True for chosen layers
    """
    if total_layers is None:
        total_layers = len(scores)

    for s in scores:
        s.selected = False

    if strategy == "last_half":
        cutoff = total_layers // 2
        for s in scores:
            if s.layer_index >= cutoff:
                s.selected = True

    elif strategy == "last_third":
        cutoff = (2 * total_layers) // 3
        for s in scores:
            if s.layer_index >= cutoff:
                s.selected = True

    elif strategy == "top_k":
        k = top_k or (total_layers // 2)
        for i, s in enumerate(scores):
            if i < k:
                s.selected = True

    elif strategy == "threshold":
        thresh = threshold or 0.5
        for s in scores:
            if s.composite >= thresh:
                s.selected = True

    elif strategy == "combined":
        # Intersection: must be in the top half by score AND in the later half by position
        k = total_layers // 2
        cutoff = total_layers // 2
        top_by_score = {s.name for s in scores[:k]}
        for s in scores:
            if s.name in top_by_score and s.layer_index >= cutoff:
                s.selected = True

    elif strategy == "alternating":
        # Select every other layer starting from layer 1 (0-indexed)
        # Provides uniform coverage of the network depth
        for s in scores:
            if s.layer_index % 2 == 1:
                s.selected = True

    elif strategy == "score_top_k":
        k = top_k or max(1, total_layers // 2)
        sorted_by_score = sorted(scores, key=lambda s: s.composite, reverse=True)
        for i, s in enumerate(sorted_by_score):
            if i < k:
                s.selected = True

    elif strategy == "all_layers":
        for s in scores:
            s.selected = True

    elif strategy == "first_k":
        k = top_k or max(1, total_layers // 2)
        sorted_by_idx = sorted(scores, key=lambda s: s.layer_index)
        for i, s in enumerate(sorted_by_idx):
            if i < k:
                s.selected = True

    elif strategy == "last_k":
        k = top_k or max(1, total_layers // 2)
        sorted_by_idx = sorted(scores, key=lambda s: s.layer_index)
        for s in sorted_by_idx[-k:]:
            s.selected = True

    elif strategy == "sparsity_only":
        k = top_k or max(1, total_layers // 2)
        sorted_by_sp = sorted(scores, key=lambda s: s.sparsity, reverse=True)
        for i, s in enumerate(sorted_by_sp):
            if i < k:
                s.selected = True

    elif strategy == "clusterability_only":
        k = top_k or max(1, total_layers // 2)
        sorted_by_mm = sorted(scores, key=lambda s: s.multimodality, reverse=True)
        for i, s in enumerate(sorted_by_mm):
            if i < k:
                s.selected = True

    elif strategy == "sensitivity_only":
        k = top_k or max(1, total_layers // 2)
        # Select most sensitive layers (high sensitivity = important layers)
        sorted_by_se = sorted(scores, key=lambda s: s.sensitivity, reverse=True)
        for i, s in enumerate(sorted_by_se):
            if i < k:
                s.selected = True

    elif strategy == "sparsity_plus_clusterability":
        k = top_k or max(1, total_layers // 2)
        for s in scores:
            s._temp_score = s.sparsity + s.multimodality
        sorted_by_sc = sorted(scores, key=lambda s: s._temp_score, reverse=True)
        for i, s in enumerate(sorted_by_sc):
            if i < k:
                s.selected = True

    elif strategy == "clusterability_minus_sensitivity":
        k = top_k or max(1, total_layers // 2)
        for s in scores:
            s._temp_score = s.multimodality - s.sensitivity
        sorted_by_cs = sorted(scores, key=lambda s: s._temp_score, reverse=True)
        for i, s in enumerate(sorted_by_cs):
            if i < k:
                s.selected = True

    else:
        raise ValueError(f"Unknown layer selection strategy: {strategy}")

    selected = [s for s in scores if s.selected]
    logger.info(
        f"Selected {len(selected)}/{total_layers} layers for expertization "
        f"(strategy={strategy})"
    )
    for s in selected:
        logger.info(f"  ✓ [{s.layer_index}] {s.name} (score={s.composite:.3f})")

    return scores


def visualize_scores(
    scores: List[LayerScore],
    save_path: Optional[str] = None,
):
    """
    Create a bar chart visualization of layer scores.

    Shows sparsity, multimodality, and sensitivity as stacked bars,
    with selected layers highlighted.
    """
    import matplotlib.pyplot as plt

    # Sort by layer index for visualization
    sorted_scores = sorted(scores, key=lambda s: s.layer_index)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    indices = range(len(sorted_scores))
    names = [f"L{s.layer_index}" for s in sorted_scores]
    sparsities = [s.sparsity for s in sorted_scores]
    multimodalities = [s.multimodality for s in sorted_scores]
    sensitivities = [s.sensitivity for s in sorted_scores]
    composites = [s.composite for s in sorted_scores]
    colors = ["#2ecc71" if s.selected else "#95a5a6" for s in sorted_scores]

    # Stacked bar chart
    ax1.bar(indices, sparsities, label="Sparsity", color="#3498db", alpha=0.8)
    ax1.bar(indices, multimodalities, bottom=sparsities, label="Multimodality",
            color="#e74c3c", alpha=0.8)
    bottoms = [s + m for s, m in zip(sparsities, multimodalities)]
    ax1.bar(indices, sensitivities, bottom=bottoms, label="Sensitivity",
            color="#f39c12", alpha=0.8)

    ax1.set_xticks(indices)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_ylabel("Score Components")
    ax1.set_title("CLEAR-MoE Layer Scores — Component Breakdown")
    ax1.legend()

    # Composite score with selection highlighting
    ax2.bar(indices, composites, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(indices)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("Composite Score")
    ax2.set_title("Composite Score (green = selected for expertization)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Score visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()
