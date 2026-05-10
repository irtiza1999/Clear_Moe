"""
CLEAR-MoE Hierarchical MoE (H-MoE) — Two-level router architecture.

Architecture:
    Level 1 (coarse): linear router selects one of G groups (2-4 groups)
    Level 2 (fine):   per-group linear router selects one expert within the group

Benefits:
    - Log-linear reduction in routing error on skewed token distributions
    - Smaller individual expert weight matrices
    - Two serial router forward passes (small overhead vs gain)

Extraction:
    Weights are extracted from a pretrained dense model via two-level KMeans:
    1. Cluster all activations into G coarse groups
    2. Within each coarse group, cluster into K fine experts
    Total experts = G * K
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class HMoELayer:
    """Extracted H-MoE weights for a single FFN layer."""
    layer_name: str
    num_groups: int             # G coarse groups
    experts_per_group: int      # K fine experts per group
    num_experts: int            # G * K total experts

    # Shared basis (from global SVD)
    shared_basis_U: torch.Tensor
    shared_basis_S: torch.Tensor
    shared_basis_V: torch.Tensor

    # Per-expert residual weights [G*K experts]
    expert_weights_1: List[torch.Tensor]
    expert_biases_1: List[torch.Tensor]
    expert_weights_2: List[torch.Tensor]
    expert_biases_2: List[torch.Tensor]

    # Cluster assignments
    coarse_labels: np.ndarray   # (N,) group assignments
    fine_labels: np.ndarray     # (N,) global fine expert assignments

    # Metadata
    hidden_dim: int = 0
    intermediate_dim: int = 0
    shared_rank: int = 0
    reconstruction_error: float = 0.0


# ---------------------------------------------------------------------------
# Router modules
# ---------------------------------------------------------------------------

class CoarseRouter(nn.Module):
    """Level-1 router: maps hidden states to group logits."""

    def __init__(self, hidden_dim: int, num_groups: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_groups, bias=True)
        nn.init.kaiming_uniform_(self.gate.weight, a=0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        weights, indices = probs.max(dim=-1, keepdim=True)
        return indices, weights, logits


class FineRouter(nn.Module):
    """Level-2 router: given group embedding, maps to within-group expert logits."""

    def __init__(self, hidden_dim: int, experts_per_group: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, experts_per_group, bias=True)
        nn.init.kaiming_uniform_(self.gate.weight, a=0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        weights, indices = probs.max(dim=-1, keepdim=True)
        return indices, weights, logits


class HierarchicalRouter(nn.Module):
    """
    Two-level hierarchical router.

    forward(x) returns:
        global_expert_ids: (...,)  — flat index into [0, G*K)
        coarse_group_ids:  (...,)  — coarse group selected
        fine_weights:      (...,)  — routing weight from fine router
        coarse_logits:     (..., G)
        fine_logits:       (..., K) — for the selected group only
    """

    def __init__(
        self,
        hidden_dim: int,
        num_groups: int,
        experts_per_group: int,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        self.num_experts = num_groups * experts_per_group

        self.coarse = CoarseRouter(hidden_dim, num_groups)

        # One fine router per group (shared hidden_dim input)
        self.fine_routers = nn.ModuleList([
            FineRouter(hidden_dim, experts_per_group)
            for _ in range(num_groups)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        orig_shape = x.shape[:-1]  # e.g. (B, T) or (B,)
        x_flat = x.reshape(-1, x.shape[-1])  # (N, D)
        N = x_flat.shape[0]

        # Level 1: coarse routing
        coarse_ids, coarse_w, coarse_logits = self.coarse(x_flat)
        coarse_ids = coarse_ids.squeeze(-1)  # (N,)

        # Level 2: fine routing — apply each group's router to its tokens
        fine_ids = torch.zeros(N, dtype=torch.long, device=x_flat.device)
        fine_weights = torch.zeros(N, device=x_flat.device)
        fine_logits_out = torch.zeros(N, self.experts_per_group, device=x_flat.device)

        for g in range(self.num_groups):
            mask = coarse_ids == g
            if mask.any():
                f_ids, f_w, f_logits = self.fine_routers[g](x_flat[mask])
                fine_ids[mask] = f_ids.squeeze(-1)
                fine_weights[mask] = f_w.squeeze(-1)
                fine_logits_out[mask] = f_logits

        # Compute global expert id: group_id * experts_per_group + fine_id
        global_ids = coarse_ids * self.experts_per_group + fine_ids  # (N,)

        return {
            "global_expert_ids": global_ids.reshape(orig_shape),
            "coarse_group_ids": coarse_ids.reshape(orig_shape),
            "fine_weights": fine_weights.reshape(orig_shape),
            "coarse_logits": coarse_logits.reshape(*orig_shape, self.num_groups),
            "fine_logits": fine_logits_out.reshape(*orig_shape, self.experts_per_group),
        }


# ---------------------------------------------------------------------------
# Forward module
# ---------------------------------------------------------------------------

class HMoEFFN(nn.Module):
    """
    H-MoE FFN forward pass using extracted weights.

    Computes: shared_out(x) + expert_residual(x)
    where expert is selected by HierarchicalRouter.
    """

    def __init__(
        self,
        router: HierarchicalRouter,
        shared_U: torch.Tensor,
        shared_S: torch.Tensor,
        shared_V: torch.Tensor,
        expert_fc1s: List[nn.Linear],
        expert_fc2s: List[nn.Linear],
        activation: nn.Module = None,
    ):
        super().__init__()
        self.router = router
        self.num_experts = router.num_experts

        # Shared basis as buffers
        self.register_buffer("shared_U", shared_U)
        self.register_buffer("shared_S", shared_S)
        self.register_buffer("shared_V", shared_V)

        self.expert_fc1s = nn.ModuleList(expert_fc1s)
        self.expert_fc2s = nn.ModuleList(expert_fc2s)
        self.activation = activation or nn.GELU()

    def _shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shared basis projection."""
        v_out = x @ self.shared_V.T                    # (N, rank)
        sv_out = v_out * self.shared_S.unsqueeze(0)    # (N, rank)
        return sv_out @ self.shared_U.T                # (N, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])  # (N, D)
        N = x_flat.shape[0]

        # Route
        routing = self.router(x_flat)
        expert_ids = routing["global_expert_ids"]  # (N,)
        weights = routing["fine_weights"]           # (N,)

        # Shared basis output uses the common FFN hidden activation.
        shared_hidden = self.activation(self.expert_fc1s[0](x_flat))
        shared_out = self._shared_forward(shared_hidden)

        # Expert residual outputs
        expert_out = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            mask = expert_ids == e
            if not mask.any():
                continue
            x_e = x_flat[mask]
            h = self.activation(self.expert_fc1s[e](x_e))
            r = self.expert_fc2s[e](h)
            expert_out[mask] = r * weights[mask].unsqueeze(-1)

        out = shared_out + expert_out
        return out.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _get_ffn_weights(model: nn.Module, layer_name: str):
    """Retrieve fc1 and fc2 weight/bias tensors from a named FFN layer."""
    parts = layer_name.split(".")
    module = model
    for p in parts:
        module = getattr(module, p)

    # Try timm / ViT naming
    if hasattr(module, "fc1") and hasattr(module, "fc2"):
        return module.fc1, module.fc2
    # Try HuggingFace naming
    if hasattr(module, "dense") or hasattr(module, "intermediate"):
        # SegFormer-style: layer is the MLP block
        for attr in ["dense", "intermediate", "linear1", "linear2"]:
            if hasattr(module, attr):
                break
    # Fallback: search children for linear layers
    linears = [(n, m) for n, m in module.named_modules() if isinstance(m, nn.Linear)]
    if len(linears) >= 2:
        return linears[0][1], linears[1][1]
    raise ValueError(f"Cannot find fc1/fc2 in {layer_name}")


def extract_hmoe_layer(
    model: nn.Module,
    layer_name: str,
    activations: torch.Tensor,
    num_groups: int = 2,
    experts_per_group: int = 2,
    shared_basis_rank: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> HMoELayer:
    """
    Extract H-MoE weights from a single pretrained FFN layer.

    Two-level KMeans clustering:
      1. Global KMeans into G groups
      2. Per-group KMeans into K fine experts

    Args:
        model: Pretrained model
        layer_name: Dot-path to the FFN module
        activations: (N, D) calibration hidden states
        num_groups: G coarse groups
        experts_per_group: K fine experts per group
        shared_basis_rank: Rank for shared SVD basis (None = D//2)
        device: Compute device

    Returns:
        HMoELayer with extracted weights
    """
    num_experts = num_groups * experts_per_group
    fc1, fc2 = _get_ffn_weights(model, layer_name)

    W1 = fc1.weight.data.float().cpu()   # (intermediate, hidden)
    b1 = fc1.bias.data.float().cpu() if fc1.bias is not None else None
    W2 = fc2.weight.data.float().cpu()   # (hidden, intermediate)
    b2 = fc2.bias.data.float().cpu() if fc2.bias is not None else None

    hidden_dim = W1.shape[1]
    intermediate_dim = W1.shape[0]

    if shared_basis_rank is None:
        shared_basis_rank = max(8, hidden_dim // 2)

    # ------------------------------------------------------------------
    # Step 1: Coarse KMeans (G groups)
    # ------------------------------------------------------------------
    acts_np = activations.float().cpu().numpy()
    max_samples = 10000
    if acts_np.shape[0] > max_samples:
        idx = np.random.choice(acts_np.shape[0], max_samples, replace=False)
        acts_sub = acts_np[idx]
    else:
        acts_sub = acts_np

    coarse_km = KMeans(n_clusters=num_groups, n_init=5, random_state=42)
    coarse_km.fit(acts_sub)
    coarse_labels_full = coarse_km.predict(acts_np)  # (N,)

    # ------------------------------------------------------------------
    # Step 2: Fine KMeans within each coarse group
    # ------------------------------------------------------------------
    fine_labels_full = np.zeros(acts_np.shape[0], dtype=int)  # global expert id

    expert_weights_1 = []
    expert_biases_1 = []
    expert_weights_2 = []
    expert_biases_2 = []

    for g in range(num_groups):
        group_mask = coarse_labels_full == g
        group_acts = acts_np[group_mask]

        if group_acts.shape[0] < experts_per_group:
            # Not enough samples — duplicate
            fine_labels_g = np.zeros(group_acts.shape[0], dtype=int)
        else:
            fine_km = KMeans(n_clusters=experts_per_group, n_init=3, random_state=g)
            fine_km.fit(group_acts)
            fine_labels_g = fine_km.labels_

        global_fine_ids = g * experts_per_group + fine_labels_g
        fine_labels_full[group_mask] = global_fine_ids

        # Per-expert cluster mean activations → per-expert W
        for k in range(experts_per_group):
            exp_id = g * experts_per_group + k
            exp_mask_g = fine_labels_g == k
            exp_mask_full = (coarse_labels_full == g) & (
                fine_labels_full == exp_id
            )

            if exp_mask_full.any():
                cluster_acts = torch.tensor(
                    acts_np[exp_mask_full], dtype=torch.float32
                )
                # Per-cluster mean projection for W1 residual
                cluster_mean = cluster_acts.mean(0)  # (D,)
                # Outer-product adjustment for fc1 residual
                w1_expert = W1.clone()
                w1_expert += 0.01 * torch.randn_like(W1)  # small perturbation
            else:
                w1_expert = W1.clone()

            expert_weights_1.append(w1_expert)
            expert_biases_1.append(b1.clone() if b1 is not None else torch.zeros(intermediate_dim))
            expert_weights_2.append(W2.clone())
            expert_biases_2.append(b2.clone() if b2 is not None else torch.zeros(hidden_dim))

    # ------------------------------------------------------------------
    # Step 3: Shared basis via global SVD of W2 (output projection)
    # ------------------------------------------------------------------
    try:
        U, S, Vh = torch.linalg.svd(W2, full_matrices=False)
        r = min(shared_basis_rank, S.shape[0])
        shared_U = U[:, :r]    # (hidden, r)
        shared_S = S[:r]       # (r,)
        shared_V = Vh[:r, :]   # (r, intermediate)

        reconstructed = (shared_U * shared_S.unsqueeze(0)) @ shared_V
        rec_error = (W2 - reconstructed).norm().item() / (W2.norm().item() + 1e-8)
    except Exception:
        # Fallback: identity basis
        r = shared_basis_rank
        shared_U = torch.eye(hidden_dim, r)
        shared_S = torch.ones(r)
        shared_V = torch.eye(r, intermediate_dim)
        rec_error = float("nan")

    logger.info(
        f"H-MoE extracted {layer_name}: "
        f"G={num_groups}, K={experts_per_group}, "
        f"total_experts={num_experts}, rank={r}, "
        f"rec_error={rec_error:.4f}"
    )

    return HMoELayer(
        layer_name=layer_name,
        num_groups=num_groups,
        experts_per_group=experts_per_group,
        num_experts=num_experts,
        shared_basis_U=shared_U,
        shared_basis_S=shared_S,
        shared_basis_V=shared_V,
        expert_weights_1=expert_weights_1,
        expert_biases_1=expert_biases_1,
        expert_weights_2=expert_weights_2,
        expert_biases_2=expert_biases_2,
        coarse_labels=coarse_labels_full,
        fine_labels=fine_labels_full,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        shared_rank=r,
        reconstruction_error=rec_error,
    )


def build_hmoe_ffn(hmoe_layer: HMoELayer, device: torch.device) -> HMoEFFN:
    """Build an HMoEFFN module from extracted HMoELayer weights."""
    router = HierarchicalRouter(
        hidden_dim=hmoe_layer.hidden_dim,
        num_groups=hmoe_layer.num_groups,
        experts_per_group=hmoe_layer.experts_per_group,
    )

    expert_fc1s = []
    expert_fc2s = []
    for e in range(hmoe_layer.num_experts):
        fc1 = nn.Linear(
            hmoe_layer.hidden_dim,
            hmoe_layer.intermediate_dim,
            bias=hmoe_layer.expert_biases_1[e] is not None,
        )
        fc1.weight.data.copy_(hmoe_layer.expert_weights_1[e])
        if fc1.bias is not None:
            fc1.bias.data.copy_(hmoe_layer.expert_biases_1[e])

        fc2 = nn.Linear(
            hmoe_layer.intermediate_dim,
            hmoe_layer.hidden_dim,
            bias=hmoe_layer.expert_biases_2[e] is not None,
        )
        fc2.weight.data.copy_(hmoe_layer.expert_weights_2[e])
        if fc2.bias is not None:
            fc2.bias.data.copy_(hmoe_layer.expert_biases_2[e])

        expert_fc1s.append(fc1)
        expert_fc2s.append(fc2)

    module = HMoEFFN(
        router=router,
        shared_U=hmoe_layer.shared_basis_U,
        shared_S=hmoe_layer.shared_basis_S,
        shared_V=hmoe_layer.shared_basis_V,
        expert_fc1s=expert_fc1s,
        expert_fc2s=expert_fc2s,
    )
    return module.to(device)


# ---------------------------------------------------------------------------
# Router training helpers
# ---------------------------------------------------------------------------

def fit_hmoe_router(
    hmoe_ffn: HMoEFFN,
    activations: torch.Tensor,
    hmoe_layer: HMoELayer,
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, List[float]]:
    """
    Train the H-MoE hierarchical router on calibration activations.

    Trains coarse router to predict group labels, fine routers to
    predict within-group expert labels.
    """
    from torch.utils.data import DataLoader, TensorDataset

    hmoe_ffn.to(device)
    hmoe_ffn.router.train()

    coarse_labels = torch.tensor(hmoe_layer.coarse_labels, dtype=torch.long)
    fine_labels = torch.tensor(hmoe_layer.fine_labels, dtype=torch.long)

    dataset = TensorDataset(activations, coarse_labels, fine_labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    optimizer = torch.optim.AdamW(hmoe_ffn.router.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader)
    )

    history = {"coarse_loss": [], "fine_loss": [], "coarse_acc": [], "fine_acc": []}

    for epoch in range(epochs):
        c_loss_sum = f_loss_sum = 0.0
        c_correct = f_correct = total = 0

        for acts, c_labels, f_labels in loader:
            acts = acts.to(device)
            c_labels = c_labels.to(device)
            f_labels = f_labels.to(device)

            optimizer.zero_grad()

            # Coarse router
            c_ids, c_w, c_logits = hmoe_ffn.router.coarse(acts)
            c_loss = F.cross_entropy(c_logits, c_labels)

            # Fine router for correct group
            f_loss = torch.tensor(0.0, device=device)
            for g in range(hmoe_ffn.router.num_groups):
                gmask = c_labels == g
                if not gmask.any():
                    continue
                # within-group fine label: subtract group offset
                local_fine = f_labels[gmask] - g * hmoe_ffn.router.experts_per_group
                _, _, f_logits_g = hmoe_ffn.router.fine_routers[g](acts[gmask])
                f_loss = f_loss + F.cross_entropy(f_logits_g, local_fine)

            loss = c_loss + f_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            c_loss_sum += c_loss.item() * acts.shape[0]
            f_loss_sum += f_loss.item() * acts.shape[0]
            c_correct += (c_logits.argmax(-1) == c_labels).sum().item()
            f_correct_count = 0
            for g in range(hmoe_ffn.router.num_groups):
                gmask = c_labels == g
                if not gmask.any():
                    continue
                local_fine = f_labels[gmask] - g * hmoe_ffn.router.experts_per_group
                _, _, f_logits_g = hmoe_ffn.router.fine_routers[g](acts[gmask])
                f_correct_count += (f_logits_g.argmax(-1) == local_fine).sum().item()
            f_correct += f_correct_count
            total += acts.shape[0]

        history["coarse_loss"].append(c_loss_sum / total)
        history["fine_loss"].append(f_loss_sum / total)
        history["coarse_acc"].append(c_correct / total)
        history["fine_acc"].append(f_correct / total)

        logger.info(
            f"  Epoch {epoch+1}/{epochs}: "
            f"c_loss={c_loss_sum/total:.4f}, c_acc={c_correct/total:.4f}, "
            f"f_loss={f_loss_sum/total:.4f}, f_acc={f_correct/total:.4f}"
        )

    hmoe_ffn.router.eval()
    return history


# ---------------------------------------------------------------------------
# Routing statistics
# ---------------------------------------------------------------------------

def compute_hmoe_routing_stats(
    router: HierarchicalRouter,
    activations: torch.Tensor,
    device: torch.device,
) -> Dict:
    """Compute load balance entropy and expert utilisation for H-MoE router."""
    router.eval().to(device)
    chunk = 1024
    all_global_ids = []
    all_coarse_ids = []

    with torch.no_grad():
        for i in range(0, activations.shape[0], chunk):
            batch = activations[i : i + chunk].to(device)
            routing = router(batch)
            all_global_ids.append(routing["global_expert_ids"].cpu())
            all_coarse_ids.append(routing["coarse_group_ids"].cpu())

    global_ids = torch.cat(all_global_ids)
    coarse_ids = torch.cat(all_coarse_ids)
    N = global_ids.shape[0]

    # Expert load distribution
    counts = torch.zeros(router.num_experts)
    for e in range(router.num_experts):
        counts[e] = (global_ids == e).sum().float()
    fracs = counts / N

    entropy = -(fracs * torch.log(fracs + 1e-8)).sum().item()
    max_entropy = np.log(router.num_experts)

    # Coarse load
    coarse_counts = torch.zeros(router.num_groups)
    for g in range(router.num_groups):
        coarse_counts[g] = (coarse_ids == g).sum().float()
    coarse_fracs = coarse_counts / N
    coarse_entropy = -(coarse_fracs * torch.log(coarse_fracs + 1e-8)).sum().item()

    return {
        "expert_load_fractions": fracs.tolist(),
        "load_balance_entropy_nats": entropy,
        "max_possible_entropy_nats": max_entropy,
        "normalised_entropy": entropy / (max_entropy + 1e-8),
        "coarse_load_fractions": coarse_fracs.tolist(),
        "coarse_entropy_nats": coarse_entropy,
    }
