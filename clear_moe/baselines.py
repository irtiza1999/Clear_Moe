"""
CLEAR-MoE Baselines — Simple extraction methods for ablation study.

Implements D0–D8 configurations from the decomposition ablation (Experiment 1):

  D0: Original dense model (no extraction)
  D1: Shared low-rank basis only; no residual experts (SVD-only)
  D2: Residual experts only; no shared basis (disjoint cluster experts)
  D3: Shared basis + one global residual; no routing
  D4: Shared basis + randomly grouped experts
  D5: Shared basis + k-means experts + random routing
  D6: Shared basis + k-means experts + learned router  [= standard CLEAR-MoE]
  D7: Disjoint experts without the shared basis (MoEfication-style)
  D8: Proposed full CLEAR-MoE++ (shared basis + k-means + learned router + scoring)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# D1: SVD-only baseline (shared basis, no experts, no routing)
# ──────────────────────────────────────────────────────────────────────

class SVDOnlyFFN(nn.Module):
    """
    Replace FFN fc2 with its rank-r SVD approximation.
    fc1 is kept dense. No experts, no routing.
    Output: fc2_approx(act(fc1(x)))
    """

    def __init__(
        self,
        shared_weight_1: torch.Tensor,
        shared_bias_1: Optional[torch.Tensor],
        shared_U: torch.Tensor,
        shared_S: torch.Tensor,
        shared_V: torch.Tensor,
        activation_fn: nn.Module,
    ):
        super().__init__()
        hidden_dim = shared_weight_1.shape[1]
        intermediate_dim = shared_weight_1.shape[0]
        rank = shared_U.shape[1]

        self.fc1 = nn.Linear(hidden_dim, intermediate_dim, bias=shared_bias_1 is not None)
        self.fc1.weight = nn.Parameter(shared_weight_1)
        if shared_bias_1 is not None:
            self.fc1.bias = nn.Parameter(shared_bias_1)

        # fc2 approximated as U @ diag(S) @ V
        W_approx = shared_U @ torch.diag(shared_S) @ shared_V
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.fc2.weight = nn.Parameter(W_approx)

        self.act = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ──────────────────────────────────────────────────────────────────────
# D2: Disjoint experts without shared basis (cluster entire fc1+fc2)
# ──────────────────────────────────────────────────────────────────────

class DisjointExpertFFN(nn.Module):
    """
    Split the FFN into E disjoint experts using k-means cluster centroids.
    Each expert handles a subset of neurons (MoEfication-style).
    No shared basis.
    """

    def __init__(
        self,
        expert_fc1_weights: List[torch.Tensor],
        expert_fc1_biases: List[Optional[torch.Tensor]],
        expert_fc2_weights: List[torch.Tensor],
        expert_fc2_biases: List[Optional[torch.Tensor]],
        activation_fn: nn.Module,
        router: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_experts = len(expert_fc1_weights)
        self.act = activation_fn
        self.router = router

        self.expert_fc1s = nn.ModuleList()
        self.expert_fc2s = nn.ModuleList()

        for e in range(self.num_experts):
            w1 = expert_fc1_weights[e]
            b1 = expert_fc1_biases[e]
            w2 = expert_fc2_weights[e]
            b2 = expert_fc2_biases[e]

            fc1 = nn.Linear(w1.shape[1], w1.shape[0], bias=b1 is not None)
            fc1.weight = nn.Parameter(w1)
            if b1 is not None:
                fc1.bias = nn.Parameter(b1)

            fc2 = nn.Linear(w2.shape[1], w2.shape[0], bias=b2 is not None)
            fc2.weight = nn.Parameter(w2)
            if b2 is not None:
                fc2.bias = nn.Parameter(b2)

            self.expert_fc1s.append(fc1)
            self.expert_fc2s.append(fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])

        if self.router is None:
            # Default to expert 0 (no routing)
            out = self.expert_fc2s[0](self.act(self.expert_fc1s[0](flat)))
        else:
            indices, weights, _ = self.router(flat)
            if indices.dim() > 1:
                indices = indices[:, 0]
            out = torch.zeros(flat.shape[0], self.expert_fc2s[0].out_features,
                              device=flat.device, dtype=flat.dtype)
            for e in range(self.num_experts):
                mask = (indices == e)
                if mask.any():
                    out[mask] = self.expert_fc2s[e](self.act(self.expert_fc1s[e](flat[mask])))

        return out.reshape(*shape[:-1], -1)


# ──────────────────────────────────────────────────────────────────────
# D3: Shared basis + single global residual, no routing
# ──────────────────────────────────────────────────────────────────────

class SharedBasisGlobalResidualFFN(nn.Module):
    """Shared basis + one global residual applied to ALL tokens. No routing."""

    def __init__(
        self,
        shared_weight_1: torch.Tensor,
        shared_bias_1: Optional[torch.Tensor],
        shared_U: torch.Tensor,
        shared_S: torch.Tensor,
        shared_V: torch.Tensor,
        global_residual: torch.Tensor,
        activation_fn: nn.Module,
    ):
        super().__init__()
        hidden_dim = shared_weight_1.shape[1]
        intermediate_dim = shared_weight_1.shape[0]

        self.fc1 = nn.Linear(hidden_dim, intermediate_dim, bias=shared_bias_1 is not None)
        self.fc1.weight = nn.Parameter(shared_weight_1)
        if shared_bias_1 is not None:
            self.fc1.bias = nn.Parameter(shared_bias_1)

        W_shared = shared_U @ torch.diag(shared_S) @ shared_V
        W_full = W_shared + global_residual
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.fc2.weight = nn.Parameter(W_full)

        self.act = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ──────────────────────────────────────────────────────────────────────
# Extraction functions for each baseline
# ──────────────────────────────────────────────────────────────────────

def extract_svd_only(
    fc1_weight: torch.Tensor,
    fc1_bias: Optional[torch.Tensor],
    fc2_weight: torch.Tensor,
    shared_basis_rank: int,
    activation_fn: nn.Module,
) -> SVDOnlyFFN:
    """D1: Truncated SVD of fc2 only. No experts."""
    U, S, Vh = torch.linalg.svd(fc2_weight.float(), full_matrices=False)
    return SVDOnlyFFN(
        shared_weight_1=fc1_weight.clone(),
        shared_bias_1=fc1_bias.clone() if fc1_bias is not None else None,
        shared_U=U[:, :shared_basis_rank],
        shared_S=S[:shared_basis_rank],
        shared_V=Vh[:shared_basis_rank, :],
        activation_fn=activation_fn,
    )


def extract_disjoint_experts(
    fc1_weight: torch.Tensor,
    fc1_bias: Optional[torch.Tensor],
    fc2_weight: torch.Tensor,
    fc2_bias: Optional[torch.Tensor],
    activations: torch.Tensor,
    num_experts: int,
    activation_fn: nn.Module,
    random_grouping: bool = False,
    seed: int = 42,
) -> Tuple[DisjointExpertFFN, np.ndarray]:
    """
    D2/D7: Disjoint expert extraction without shared basis.

    Partitions neurons into E disjoint groups (MoEfication-style).
    Each expert has a subset of fc2 columns and corresponding fc1 rows.

    Args:
        random_grouping: If True, assign neurons randomly (D4-style).
        seed: Random seed for k-means and random grouping.
    """
    intermediate_dim = fc1_weight.shape[0]
    tokens_per_expert = intermediate_dim // num_experts

    rng = np.random.RandomState(seed)

    if random_grouping:
        neuron_assignments = rng.permutation(intermediate_dim)
    else:
        # Cluster fc1 weight rows (one row per neuron)
        w1_np = fc1_weight.float().numpy()
        kmeans = MiniBatchKMeans(n_clusters=num_experts, n_init=5, random_state=seed)
        neuron_assignments_labels = kmeans.fit_predict(w1_np)
        neuron_assignments = neuron_assignments_labels

    expert_fc1_weights = []
    expert_fc1_biases = []
    expert_fc2_weights = []
    expert_fc2_biases = []

    for e in range(num_experts):
        if random_grouping:
            start = e * tokens_per_expert
            end = (e + 1) * tokens_per_expert if e < num_experts - 1 else intermediate_dim
            neuron_mask = np.zeros(intermediate_dim, dtype=bool)
            neuron_mask[neuron_assignments[start:end]] = True
        else:
            neuron_mask = (neuron_assignments == e)

        if not neuron_mask.any():
            neuron_mask[e] = True

        idx = torch.from_numpy(np.where(neuron_mask)[0]).long()
        expert_fc1_weights.append(fc1_weight[idx].clone())
        expert_fc1_biases.append(
            fc1_bias[idx].clone() if fc1_bias is not None else None
        )
        expert_fc2_weights.append(fc2_weight[:, idx].clone())
        expert_fc2_biases.append(
            fc2_bias.clone() if fc2_bias is not None else None
        )

    module = DisjointExpertFFN(
        expert_fc1_weights=expert_fc1_weights,
        expert_fc1_biases=expert_fc1_biases,
        expert_fc2_weights=expert_fc2_weights,
        expert_fc2_biases=expert_fc2_biases,
        activation_fn=activation_fn,
    )

    # Also compute cluster labels for router training
    if random_grouping:
        cluster_labels = np.zeros(activations.shape[0], dtype=np.int64)
    else:
        # Assign tokens to expert based on which neuron cluster fires most
        with torch.no_grad():
            act_np = activations.float().numpy()
            cluster_labels = kmeans.predict(
                fc1_weight.float().numpy()  # placeholder; real labels need token→neuron mapping
            )
            # Use activation-based k-means instead
            act_kmeans = MiniBatchKMeans(n_clusters=num_experts, n_init=5, random_state=seed)
            cluster_labels = act_kmeans.fit_predict(act_np)

    return module, cluster_labels


def extract_global_residual(
    fc1_weight: torch.Tensor,
    fc1_bias: Optional[torch.Tensor],
    fc2_weight: torch.Tensor,
    shared_basis_rank: int,
    activation_fn: nn.Module,
) -> SharedBasisGlobalResidualFFN:
    """D3: Shared basis + global residual (mean residual), no routing."""
    U, S, Vh = torch.linalg.svd(fc2_weight.float(), full_matrices=False)
    W_shared = U[:, :shared_basis_rank] @ torch.diag(S[:shared_basis_rank]) @ Vh[:shared_basis_rank, :]
    global_residual = fc2_weight.float() - W_shared

    return SharedBasisGlobalResidualFFN(
        shared_weight_1=fc1_weight.clone(),
        shared_bias_1=fc1_bias.clone() if fc1_bias is not None else None,
        shared_U=U[:, :shared_basis_rank],
        shared_S=S[:shared_basis_rank],
        shared_V=Vh[:shared_basis_rank, :],
        global_residual=global_residual,
        activation_fn=activation_fn,
    )
