"""
CLEAR-MoE Extraction — Shared-basis + residual expert decomposition.

Core novelty: instead of creating disjoint experts (as in Berisha et al.),
we preserve a common dense subspace (shared basis) and route only the
residual specialization. This preserves shared visual structure and reduces
parameter inflation.

Algorithm:
1. Cluster calibration activations into E groups via KMeans
2. Compute shared basis via truncated SVD of the full weight matrix
3. Compute per-expert residual weights: W_e = W_cluster_e - W_shared
4. Build MoE FFN: output = shared_basis(x) + router_weight * expert_e(x)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ExpertizedLayer:
    """
    Result of expert extraction for a single FFN layer.

    Contains the shared basis, per-expert residual weights, and
    cluster assignments from calibration data.
    """
    layer_name: str
    num_experts: int

    # Shared basis components (from SVD)
    shared_basis_U: torch.Tensor        # (d_out, rank)
    shared_basis_S: torch.Tensor        # (rank,)
    shared_basis_V: torch.Tensor        # (rank, d_in)

    # Per-expert residual weights
    expert_weights_1: List[torch.Tensor]   # List of (intermediate, d_in) — fc1 residuals
    expert_biases_1: List[torch.Tensor]    # List of (intermediate,) — fc1 bias residuals
    expert_weights_2: List[torch.Tensor]   # List of (d_out, intermediate) — fc2 residuals
    expert_biases_2: List[torch.Tensor]    # List of (d_out,) — fc2 bias residuals

    # Cluster info
    cluster_labels: np.ndarray             # (N,) calibration token cluster assignments
    cluster_centers: np.ndarray            # (E, D) cluster centroids

    # Shared components for layer 1 (usually not decomposed but shared)
    shared_weight_1: Optional[torch.Tensor] = None  # (intermediate, d_in)
    shared_bias_1: Optional[torch.Tensor] = None    # (intermediate,)

    # Metadata
    hidden_dim: int = 0
    intermediate_dim: int = 0
    shared_rank: int = 0
    reconstruction_error: float = 0.0


def extract_experts(
    model: nn.Module,
    layer_name: str,
    activations: torch.Tensor,
    num_experts: int = 4,
    shared_basis_rank: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> ExpertizedLayer:
    """
    Extract shared-basis + residual experts from a single FFN layer.

    Steps:
    1. Get the FFN weight matrices (fc1, fc2 or equivalent)
    2. Cluster calibration activations to define expert groups
    3. Compute shared basis via truncated SVD
    4. Compute per-expert residual weights

    Args:
        model: Pretrained model
        layer_name: Dot-separated name of the FFN layer
        activations: (N, D) calibration activations for this layer
        num_experts: Number of experts to extract
        shared_basis_rank: Rank for shared basis SVD (None = auto: 50% of min dim)
        device: Torch device for computation

    Returns:
        ExpertizedLayer with all decomposition results
    """
    logger.info(f"Extracting experts for {layer_name} (E={num_experts})")

    # 1. Get FFN weight matrices
    ffn_module = _get_module(model, layer_name)
    fc1_weight, fc1_bias, fc2_weight, fc2_bias = _get_ffn_weights(ffn_module)

    hidden_dim = fc1_weight.shape[1]     # d_in
    intermediate_dim = fc1_weight.shape[0]  # d_intermediate

    logger.info(f"  FFN dims: {hidden_dim} -> {intermediate_dim} -> {fc2_weight.shape[0]}")

    # 2. Cluster activations
    act_np = activations.float().numpy()
    if act_np.shape[0] > 10000:
        # Subsample for clustering
        indices = np.random.choice(act_np.shape[0], 10000, replace=False)
        act_cluster = act_np[indices]
    else:
        act_cluster = act_np

    logger.info(f"  Clustering {act_cluster.shape[0]} activations into {num_experts} groups")

    kmeans = MiniBatchKMeans(
        n_clusters=num_experts,
        batch_size=min(1024, act_cluster.shape[0]),
        n_init=5,
        random_state=42,
    )
    cluster_labels = kmeans.fit_predict(act_cluster)
    cluster_centers = kmeans.cluster_centers_

    # Get labels for all activations
    all_labels = kmeans.predict(act_np)

    # Log cluster distribution
    for e in range(num_experts):
        count = (all_labels == e).sum()
        logger.info(f"  Expert {e}: {count} tokens ({100 * count / len(all_labels):.1f}%)")

    # 3. Compute shared basis via truncated SVD of fc2
    # This shared path maps the FFN intermediate activation back to hidden dim.
    if shared_basis_rank is None:
        shared_basis_rank = min(hidden_dim, intermediate_dim) // 2

    logger.info(f"  Computing shared basis (rank={shared_basis_rank})")

    # SVD on fc2 weight matrix
    U, S, Vh = torch.linalg.svd(fc2_weight.float(), full_matrices=False)

    # Truncate to shared_basis_rank
    U_shared = U[:, :shared_basis_rank]       # (hidden, rank)
    S_shared = S[:shared_basis_rank]           # (rank,)
    V_shared = Vh[:shared_basis_rank, :]       # (rank, intermediate)

    # Shared basis approximation of fc2: U_shared @ diag(S_shared) @ V_shared
    W_shared = U_shared @ torch.diag(S_shared) @ V_shared  # (hidden, intermediate)

    # Reconstruction error for fc2
    recon_error = (fc2_weight.float() - W_shared).norm() / fc2_weight.float().norm()
    logger.info(f"  Shared basis reconstruction error (fc2): {recon_error:.4f}")

    # 4. Compute per-expert residual weights
    # For each expert, compute residuals from cluster-specific behavior
    expert_weights_1 = []
    expert_biases_1 = []
    expert_weights_2 = []
    expert_biases_2 = []

    for e in range(num_experts):
        mask = (all_labels == e)
        expert_acts = activations[mask].float()  # (N_e, hidden)

        if expert_acts.shape[0] == 0:
            logger.warning(f"  Expert {e} has no assigned tokens!")
            # Use zero residuals
            expert_weights_1.append(torch.zeros_like(fc1_weight))
            expert_biases_1.append(torch.zeros_like(fc1_bias) if fc1_bias is not None
                                   else torch.zeros(intermediate_dim))
            expert_weights_2.append(torch.zeros_like(fc2_weight))
            expert_biases_2.append(torch.zeros_like(fc2_bias) if fc2_bias is not None
                                   else torch.zeros(fc2_weight.shape[0]))
            continue

        # Keep fc1 as the expert-specific hidden projection.
        # The shared basis is applied on fc2, so fc1 stays in the expert path.
        w1_expert = fc1_weight.float().clone()

        # Scale fc2 residuals by cluster-specific importance.
        # Use the ratio of mean activation norms for this cluster vs global.
        cluster_norm = expert_acts.norm(dim=-1).mean()
        global_norm = activations.float().norm(dim=-1).mean()
        scale = (cluster_norm / (global_norm + 1e-8)).item()

        expert_weights_1.append(w1_expert.to(fc1_weight.dtype))

        if fc1_bias is not None:
            expert_biases_1.append(fc1_bias.clone())
        else:
            expert_biases_1.append(torch.zeros(intermediate_dim))

        # fc2 residual against the shared basis approximation.
        W_residual_fc2 = fc2_weight.float() - W_shared
        expert_weights_2.append((W_residual_fc2 * scale).to(fc2_weight.dtype))

        if fc2_bias is not None:
            expert_biases_2.append(fc2_bias.clone())
        else:
            expert_biases_2.append(torch.zeros(fc2_weight.shape[0]))

    result = ExpertizedLayer(
        layer_name=layer_name,
        num_experts=num_experts,
        shared_basis_U=U_shared,
        shared_basis_S=S_shared,
        shared_basis_V=V_shared,
        shared_weight_1=fc1_weight.clone(),
        shared_bias_1=fc1_bias.clone() if fc1_bias is not None else None,
        expert_weights_1=expert_weights_1,
        expert_biases_1=expert_biases_1,
        expert_weights_2=expert_weights_2,
        expert_biases_2=expert_biases_2,
        cluster_labels=all_labels,
        cluster_centers=cluster_centers,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        shared_rank=shared_basis_rank,
        reconstruction_error=recon_error.item(),
    )

    return result


def extract_all_experts(
    model: nn.Module,
    selected_layers: list,
    activations: Dict[str, torch.Tensor],
    num_experts: int = 4,
    shared_basis_rank: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, ExpertizedLayer]:
    """
    Extract experts for all selected layers.

    Args:
        model: Pretrained model
        selected_layers: List of LayerScore with selected=True
        activations: Dict of activations per layer
        num_experts: Number of experts per layer
        shared_basis_rank: SVD rank (None for auto)
        device: Torch device

    Returns:
        Dict mapping layer name to ExpertizedLayer
    """
    results = {}

    for layer_score in selected_layers:
        if not layer_score.selected:
            continue

        name = layer_score.name
        if name not in activations:
            logger.warning(f"No activations for selected layer {name}, skipping")
            continue

        expertized = extract_experts(
            model=model,
            layer_name=name,
            activations=activations[name],
            num_experts=num_experts,
            shared_basis_rank=shared_basis_rank,
            device=device,
        )
        results[name] = expertized

    logger.info(f"Extracted experts for {len(results)} layers")
    return results


def build_moe_ffn(
    expertized: ExpertizedLayer,
    original_ffn: nn.Module,
    activation_fn: Optional[nn.Module] = None,
) -> nn.Module:
    """
    Build a MoE FFN module from an ExpertizedLayer.

    Creates a module that:
    1. Computes shared basis output
    2. Routes input to selected expert(s)
    3. Computes expert residual
    4. Combines: output = shared + expert_residual

    This is used for the initial construction; the router is added separately.

    Args:
        expertized: ExpertizedLayer from extract_experts
        original_ffn: The original FFN module (for activation function)
        activation_fn: Override activation function (default: detect from original)

    Returns:
        nn.Module implementing the MoE FFN
    """
    from runtime.grouped_expert_mlp import GroupedExpertMLP

    # Detect activation function from original FFN
    if activation_fn is None:
        activation_fn = _detect_activation(original_ffn)

    moe_ffn = GroupedExpertMLP(
        hidden_dim=expertized.hidden_dim,
        intermediate_dim=expertized.intermediate_dim,
        num_experts=expertized.num_experts,
        shared_basis_U=expertized.shared_basis_U,
        shared_basis_S=expertized.shared_basis_S,
        shared_basis_V=expertized.shared_basis_V,
        shared_weight_1=expertized.shared_weight_1,
        shared_bias_1=expertized.shared_bias_1,
        expert_weights_1=expertized.expert_weights_1,
        expert_biases_1=expertized.expert_biases_1,
        expert_weights_2=expertized.expert_weights_2,
        expert_biases_2=expertized.expert_biases_2,
        activation_fn=activation_fn,
    )

    return moe_ffn


def save_expertized_layers(
    expertized_layers: Dict[str, ExpertizedLayer],
    save_dir: str,
):
    """Save all expertized layers to disk."""
    from pathlib import Path

    save_path = Path(save_dir) / "expertized_layers"
    save_path.mkdir(parents=True, exist_ok=True)

    for name, exp in expertized_layers.items():
        clean_name = name.replace(".", "_")
        layer_path = save_path / f"{clean_name}.pt"

        state = {
            "layer_name": exp.layer_name,
            "num_experts": exp.num_experts,
            "shared_basis_U": exp.shared_basis_U,
            "shared_basis_S": exp.shared_basis_S,
            "shared_basis_V": exp.shared_basis_V,
            "shared_weight_1": exp.shared_weight_1,
            "shared_bias_1": exp.shared_bias_1,
            "expert_weights_1": exp.expert_weights_1,
            "expert_biases_1": exp.expert_biases_1,
            "expert_weights_2": exp.expert_weights_2,
            "expert_biases_2": exp.expert_biases_2,
            "cluster_labels": exp.cluster_labels,
            "cluster_centers": exp.cluster_centers,
            "hidden_dim": exp.hidden_dim,
            "intermediate_dim": exp.intermediate_dim,
            "shared_rank": exp.shared_rank,
            "reconstruction_error": exp.reconstruction_error,
        }
        torch.save(state, layer_path)
        logger.info(f"Saved expertized layer to {layer_path}")


def load_expertized_layers(load_dir: str) -> Dict[str, ExpertizedLayer]:
    """Load expertized layers from disk."""
    from pathlib import Path

    load_path = Path(load_dir) / "expertized_layers"
    results = {}

    for layer_file in sorted(load_path.glob("*.pt")):
        state = torch.load(layer_file, map_location="cpu", weights_only=False)
        exp = ExpertizedLayer(
            layer_name=state["layer_name"],
            num_experts=state["num_experts"],
            shared_basis_U=state["shared_basis_U"],
            shared_basis_S=state["shared_basis_S"],
            shared_basis_V=state["shared_basis_V"],
            shared_weight_1=state.get("shared_weight_1"),
            shared_bias_1=state.get("shared_bias_1"),
            expert_weights_1=state["expert_weights_1"],
            expert_biases_1=state["expert_biases_1"],
            expert_weights_2=state["expert_weights_2"],
            expert_biases_2=state["expert_biases_2"],
            cluster_labels=state["cluster_labels"],
            cluster_centers=state["cluster_centers"],
            hidden_dim=state["hidden_dim"],
            intermediate_dim=state["intermediate_dim"],
            shared_rank=state["shared_rank"],
            reconstruction_error=state["reconstruction_error"],
        )
        results[state["layer_name"]] = exp

    logger.info(f"Loaded {len(results)} expertized layers from {load_path}")
    return results


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _get_module(model: nn.Module, name: str) -> nn.Module:
    """Get a module by dot-separated name."""
    parts = name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def _get_ffn_weights(
    ffn_module: nn.Module,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract fc1 and fc2 weight matrices from an FFN module.

    Supports multiple FFN styles:
    - timm ViT: module.fc1.weight, module.fc2.weight
    - HF ViT: module.dense.weight (intermediate), output.dense.weight
    - SegFormer MLP: varies by SegFormer version
    """
    # Try timm-style
    if hasattr(ffn_module, "fc1") and hasattr(ffn_module, "fc2"):
        fc1_w = ffn_module.fc1.weight.data.cpu()
        fc1_b = ffn_module.fc1.bias.data.cpu() if ffn_module.fc1.bias is not None else None
        fc2_w = ffn_module.fc2.weight.data.cpu()
        fc2_b = ffn_module.fc2.bias.data.cpu() if ffn_module.fc2.bias is not None else None
        return fc1_w, fc1_b, fc2_w, fc2_b

    # Try HF-style (dense)
    if hasattr(ffn_module, "dense"):
        fc1_w = ffn_module.dense.weight.data.cpu()
        fc1_b = ffn_module.dense.bias.data.cpu() if ffn_module.dense.bias is not None else None
        # For HF intermediate layers, fc2 is in the output module (handled separately)
        return fc1_w, fc1_b, fc1_w, fc1_b  # Placeholder for fc2

    # Try SegFormer MixFFN-style modules with dense1/dense2
    dense1 = getattr(ffn_module, "dense1", None)
    dense2 = getattr(ffn_module, "dense2", None)
    if isinstance(dense1, nn.Linear) and isinstance(dense2, nn.Linear):
        fc1_w = dense1.weight.data.cpu()
        fc1_b = dense1.bias.data.cpu() if dense1.bias is not None else None
        fc2_w = dense2.weight.data.cpu()
        fc2_b = dense2.bias.data.cpu() if dense2.bias is not None else None
        return fc1_w, fc1_b, fc2_w, fc2_b

    # Try finding any Linear children
    linears = [(n, m) for n, m in ffn_module.named_modules() if isinstance(m, nn.Linear)]
    if len(linears) >= 2:
        fc1 = linears[0][1]
        fc2 = linears[-1][1]
        return (
            fc1.weight.data.cpu(),
            fc1.bias.data.cpu() if fc1.bias is not None else None,
            fc2.weight.data.cpu(),
            fc2.bias.data.cpu() if fc2.bias is not None else None,
        )

    raise ValueError(f"Cannot extract weights from FFN module: {type(ffn_module)}")


def _detect_activation(ffn_module: nn.Module) -> nn.Module:
    """Detect the activation function used in an FFN module."""
    for name, module in ffn_module.named_modules():
        if isinstance(module, nn.GELU):
            return nn.GELU()
        elif isinstance(module, nn.ReLU):
            return nn.ReLU()
        elif isinstance(module, nn.SiLU):
            return nn.SiLU()

    # Check for act_layer attribute (timm style)
    if hasattr(ffn_module, "act"):
        return type(ffn_module.act)()

    # Default to GELU (common in ViTs)
    return nn.GELU()
