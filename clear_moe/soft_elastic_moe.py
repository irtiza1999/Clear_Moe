"""
CLEAR-MoE Soft-MoE and Elastic-MoE implementations.

Soft-MoE:
    All E experts run for every token. Outputs are blended with softmax weights.
    No token dropping, no load imbalance — upper-bound quality reference.
    Cost: E× the compute of a single expert (all weights active).

Elastic-MoE:
    Active expert count varies per token based on router confidence.
    High-confidence tokens → top-1 expert only.
    Low-confidence tokens  → top-2 or top-k experts (up to k_max).
    Controlled by a confidence_threshold hyperparameter.
    Measures the Pareto frontier of quality vs active FLOPs.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===========================================================================
# Soft-MoE
# ===========================================================================

class SoftMoEFFN(nn.Module):
    """
    Soft (fully differentiable) MoE FFN.

    All experts run on all tokens. Output = weighted sum of expert outputs,
    weights given by router softmax probabilities.

    Reference: Puigcerver et al. "From Sparse to Soft Mixtures of Experts" (2023)

    Forward:
        x: (B, T, D) or (N, D)
        output: same shape as x
    """

    def __init__(
        self,
        expert_fc1s: List[nn.Linear],
        expert_fc2s: List[nn.Linear],
        shared_U: Optional[torch.Tensor] = None,
        shared_S: Optional[torch.Tensor] = None,
        shared_V: Optional[torch.Tensor] = None,
        activation: Optional[nn.Module] = None,
        use_shared_basis: bool = True,
    ):
        super().__init__()
        self.num_experts = len(expert_fc1s)
        self.expert_fc1s = nn.ModuleList(expert_fc1s)
        self.expert_fc2s = nn.ModuleList(expert_fc2s)
        self.activation = activation or nn.GELU()
        self.use_shared_basis = use_shared_basis and (shared_U is not None)

        # Gate: hidden_dim -> num_experts (soft)
        hidden_dim = expert_fc1s[0].in_features
        self.gate = nn.Linear(hidden_dim, self.num_experts, bias=True)
        nn.init.kaiming_uniform_(self.gate.weight, a=0.01)
        nn.init.zeros_(self.gate.bias)

        if self.use_shared_basis:
            self.register_buffer("shared_U", shared_U)
            self.register_buffer("shared_S", shared_S)
            self.register_buffer("shared_V", shared_V)

    def _shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        v_out = x @ self.shared_V.T
        sv_out = v_out * self.shared_S.unsqueeze(0)
        return sv_out @ self.shared_U.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])  # (N, D)
        N = x_flat.shape[0]

        # Compute soft routing weights
        logits = self.gate(x_flat)            # (N, E)
        weights = F.softmax(logits, dim=-1)   # (N, E)

        # Shared basis (optional) runs on the shared FFN hidden activation.
        shared_hidden = self.activation(self.expert_fc1s[0](x_flat))
        shared_out = self._shared_forward(shared_hidden) if self.use_shared_basis else torch.zeros_like(x_flat)

        # All experts run — blended output
        out = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            h = self.activation(self.expert_fc1s[e](x_flat))
            r = self.expert_fc2s[e](h)
            out = out + weights[:, e : e + 1] * r

        result = shared_out + out if self.use_shared_basis else out
        return result.reshape(orig_shape)

    def get_routing_stats(self, x: torch.Tensor) -> Dict:
        """Return per-expert load fractions from soft routing."""
        x_flat = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            logits = self.gate(x_flat)
            weights = F.softmax(logits, dim=-1)
        mean_weights = weights.mean(0)  # (E,)
        entropy = -(mean_weights * torch.log(mean_weights + 1e-8)).sum().item()
        return {
            "mean_expert_weights": mean_weights.tolist(),
            "soft_routing_entropy_nats": entropy,
            "all_experts_active": True,
        }


# ===========================================================================
# Elastic-MoE
# ===========================================================================

class ElasticMoEFFN(nn.Module):
    """
    Elastic (confidence-adaptive) MoE FFN.

    Active expert count per token varies between 1 and k_max based on
    router confidence (gap between top-1 and top-2 probabilities).

    Confidence score:   conf_i = p_{(1),i} - p_{(2),i}
    If conf_i >= threshold: use top-1 only (confident token)
    If conf_i <  threshold: escalate to top-k (up to k_max)

    This minimises active FLOPs for easy tokens while maintaining quality
    for ambiguous tokens, tracing a Pareto frontier of quality vs FLOPs.
    """

    def __init__(
        self,
        expert_fc1s: List[nn.Linear],
        expert_fc2s: List[nn.Linear],
        shared_U: Optional[torch.Tensor] = None,
        shared_S: Optional[torch.Tensor] = None,
        shared_V: Optional[torch.Tensor] = None,
        activation: Optional[nn.Module] = None,
        confidence_threshold: float = 0.4,
        k_max: int = 2,
        use_shared_basis: bool = True,
    ):
        super().__init__()
        self.num_experts = len(expert_fc1s)
        self.expert_fc1s = nn.ModuleList(expert_fc1s)
        self.expert_fc2s = nn.ModuleList(expert_fc2s)
        self.activation = activation or nn.GELU()
        self.confidence_threshold = confidence_threshold
        self.k_max = min(k_max, self.num_experts)
        self.use_shared_basis = use_shared_basis and (shared_U is not None)

        hidden_dim = expert_fc1s[0].in_features
        self.gate = nn.Linear(hidden_dim, self.num_experts, bias=True)
        nn.init.kaiming_uniform_(self.gate.weight, a=0.01)
        nn.init.zeros_(self.gate.bias)

        if self.use_shared_basis:
            self.register_buffer("shared_U", shared_U)
            self.register_buffer("shared_S", shared_S)
            self.register_buffer("shared_V", shared_V)

    def _shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        v_out = x @ self.shared_V.T
        sv_out = v_out * self.shared_S.unsqueeze(0)
        return sv_out @ self.shared_U.T

    def _expert_forward(
        self,
        x: torch.Tensor,
        expert_id: int,
    ) -> torch.Tensor:
        h = self.activation(self.expert_fc1s[expert_id](x))
        return self.expert_fc2s[expert_id](h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            output: same shape as x (tensor only, not tuple)
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])  # (N, D)
        N = x_flat.shape[0]

        logits = self.gate(x_flat)            # (N, E)
        probs = F.softmax(logits, dim=-1)     # (N, E)

        # Top-k_max probabilities and indices
        topk_probs, topk_ids = torch.topk(probs, self.k_max, dim=-1)  # (N, k_max)

        # Confidence: gap between top-1 and top-2 (or 1.0 if only 1 expert)
        if self.k_max >= 2:
            confidence = topk_probs[:, 0] - topk_probs[:, 1]  # (N,)
        else:
            confidence = torch.ones(N, device=x_flat.device)

        confident_mask = confidence >= self.confidence_threshold  # (N,)

        # Store stats for later access (e.g., via get_routing_stats)
        self._last_confidence = confidence
        self._last_confident_mask = confident_mask

        # Shared basis on the shared FFN hidden activation.
        shared_hidden = self.activation(self.expert_fc1s[0](x_flat))
        shared_out = self._shared_forward(shared_hidden) if self.use_shared_basis else torch.zeros_like(x_flat)

        out = torch.zeros_like(x_flat)

        # Group 1: confident tokens → top-1 only
        if confident_mask.any():
            x_conf = x_flat[confident_mask]
            top1_ids_conf = topk_ids[confident_mask, 0]  # (N_conf,)
            top1_w_conf = topk_probs[confident_mask, 0:1]  # (N_conf, 1)

            conf_out = torch.zeros_like(x_conf)
            for e in range(self.num_experts):
                emask = top1_ids_conf == e
                if emask.any():
                    r = self._expert_forward(x_conf[emask], e)
                    conf_out[emask] += r
            out[confident_mask] = conf_out

        # Group 2: uncertain tokens → top-k_max
        uncertain_mask = ~confident_mask
        if uncertain_mask.any():
            x_unc = x_flat[uncertain_mask]
            topk_ids_unc = topk_ids[uncertain_mask]    # (N_unc, k_max)
            topk_w_unc = topk_probs[uncertain_mask]    # (N_unc, k_max)
            # renormalize
            topk_w_unc = topk_w_unc / topk_w_unc.sum(-1, keepdim=True).clamp_min(1e-8)

            unc_out = torch.zeros_like(x_unc)
            for k in range(self.k_max):
                eids = topk_ids_unc[:, k]
                ws = topk_w_unc[:, k : k + 1]
                for e in range(self.num_experts):
                    emask = eids == e
                    if emask.any():
                        r = self._expert_forward(x_unc[emask], e)
                        unc_out[emask] += ws[emask] * r
            out[uncertain_mask] = unc_out

        result = shared_out + out if self.use_shared_basis else out
        return result.reshape(orig_shape)


# ===========================================================================
# Capacity-buffer MoE  (bonus: Section 3.4 in plan)
# ===========================================================================

class CapacityBufferMoEFFN(nn.Module):
    """
    MoE FFN with per-expert capacity buffers.

    Tokens that overflow an expert's capacity buffer are re-routed to
    their second-choice expert (or dropped if overflow_to_fallback=False).

    capacity_factor C: buffer_size = C * (total_tokens / num_experts)
    """

    def __init__(
        self,
        expert_fc1s: List[nn.Linear],
        expert_fc2s: List[nn.Linear],
        shared_U: Optional[torch.Tensor] = None,
        shared_S: Optional[torch.Tensor] = None,
        shared_V: Optional[torch.Tensor] = None,
        activation: Optional[nn.Module] = None,
        capacity_factor: float = 1.25,
        overflow_to_fallback: bool = True,
        use_shared_basis: bool = True,
    ):
        super().__init__()
        self.num_experts = len(expert_fc1s)
        self.expert_fc1s = nn.ModuleList(expert_fc1s)
        self.expert_fc2s = nn.ModuleList(expert_fc2s)
        self.activation = activation or nn.GELU()
        self.capacity_factor = capacity_factor
        self.overflow_to_fallback = overflow_to_fallback
        self.use_shared_basis = use_shared_basis and (shared_U is not None)

        hidden_dim = expert_fc1s[0].in_features
        self.gate = nn.Linear(hidden_dim, self.num_experts, bias=True)
        nn.init.kaiming_uniform_(self.gate.weight, a=0.01)
        nn.init.zeros_(self.gate.bias)

        if self.use_shared_basis:
            self.register_buffer("shared_U", shared_U)
            self.register_buffer("shared_S", shared_S)
            self.register_buffer("shared_V", shared_V)

    def _shared_forward(self, x):
        v = x @ self.shared_V.T
        return (v * self.shared_S) @ self.shared_U.T

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        N = x_flat.shape[0]

        capacity = max(1, int(self.capacity_factor * N / self.num_experts))

        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)
        top2_probs, top2_ids = torch.topk(probs, min(2, self.num_experts), dim=-1)

        # Assign tokens to experts with capacity enforcement
        assignment = torch.full((N,), -1, dtype=torch.long, device=x_flat.device)
        expert_fill = torch.zeros(self.num_experts, dtype=torch.long, device=x_flat.device)

        # Sort by top-1 confidence (highest first → they win capacity slots)
        order = top2_probs[:, 0].argsort(descending=True)
        for idx in order:
            e1 = top2_ids[idx, 0].item()
            if expert_fill[e1] < capacity:
                assignment[idx] = e1
                expert_fill[e1] += 1
            elif self.overflow_to_fallback and top2_ids.shape[1] > 1:
                e2 = top2_ids[idx, 1].item()
                if expert_fill[e2] < capacity:
                    assignment[idx] = e2
                    expert_fill[e2] += 1
            # else: token dropped (assignment stays -1)

        # Shared basis on the shared FFN hidden activation.
        shared_hidden = self.activation(self.expert_fc1s[0](x_flat))
        shared_out = self._shared_forward(shared_hidden) if self.use_shared_basis else torch.zeros_like(x_flat)
        out = torch.zeros_like(x_flat)

        for e in range(self.num_experts):
            emask = assignment == e
            if emask.any():
                h = self.activation(self.expert_fc1s[e](x_flat[emask]))
                out[emask] = self.expert_fc2s[e](h)

        result = shared_out + out if self.use_shared_basis else out

        n_dropped = (assignment == -1).sum().item()
        stats = {
            "capacity_factor": self.capacity_factor,
            "capacity_per_expert": capacity,
            "tokens_dropped": n_dropped,
            "drop_fraction": n_dropped / N,
            "expert_fill": expert_fill.tolist(),
        }

        return result.reshape(orig_shape), stats


# ===========================================================================
# Factory
# ===========================================================================

def build_soft_moe_ffn(
    expert_fc1s: List[nn.Linear],
    expert_fc2s: List[nn.Linear],
    shared_U: Optional[torch.Tensor] = None,
    shared_S: Optional[torch.Tensor] = None,
    shared_V: Optional[torch.Tensor] = None,
) -> SoftMoEFFN:
    return SoftMoEFFN(
        expert_fc1s=expert_fc1s,
        expert_fc2s=expert_fc2s,
        shared_U=shared_U,
        shared_S=shared_S,
        shared_V=shared_V,
    )


def build_elastic_moe_ffn(
    expert_fc1s: List[nn.Linear],
    expert_fc2s: List[nn.Linear],
    shared_U: Optional[torch.Tensor] = None,
    shared_S: Optional[torch.Tensor] = None,
    shared_V: Optional[torch.Tensor] = None,
    confidence_threshold: float = 0.4,
    k_max: int = 2,
) -> ElasticMoEFFN:
    return ElasticMoEFFN(
        expert_fc1s=expert_fc1s,
        expert_fc2s=expert_fc2s,
        shared_U=shared_U,
        shared_S=shared_S,
        shared_V=shared_V,
        confidence_threshold=confidence_threshold,
        k_max=k_max,
    )


def build_capacity_moe_ffn(
    expert_fc1s: List[nn.Linear],
    expert_fc2s: List[nn.Linear],
    shared_U: Optional[torch.Tensor] = None,
    shared_S: Optional[torch.Tensor] = None,
    shared_V: Optional[torch.Tensor] = None,
    capacity_factor: float = 1.25,
    overflow_to_fallback: bool = True,
) -> CapacityBufferMoEFFN:
    return CapacityBufferMoEFFN(
        expert_fc1s=expert_fc1s,
        expert_fc2s=expert_fc2s,
        shared_U=shared_U,
        shared_S=shared_S,
        shared_V=shared_V,
        capacity_factor=capacity_factor,
        overflow_to_fallback=overflow_to_fallback,
    )


# ===========================================================================
# FLOPs accounting helpers
# ===========================================================================

def count_soft_moe_flops(
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    num_tokens: int,
) -> Dict[str, int]:
    """Count FLOPs for a Soft-MoE forward pass (all experts active)."""
    # Gate: 2 * D * E per token
    gate_flops = 2 * hidden_dim * num_experts * num_tokens
    # Each expert: fc1 + activation(≈0) + fc2 = 2*D*I + 2*I*D per token
    expert_flops = 2 * 2 * hidden_dim * intermediate_dim * num_tokens * num_experts
    total = gate_flops + expert_flops
    return {
        "gate_flops": gate_flops,
        "expert_flops": expert_flops,
        "total_flops": total,
        "vs_dense_factor": num_experts,  # E× vs single expert
    }


def count_elastic_moe_expected_flops(
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    num_tokens: int,
    fraction_confident: float,
    k_max: int,
) -> Dict[str, float]:
    """Expected FLOPs for Elastic-MoE given confidence fraction."""
    mean_active = fraction_confident * 1.0 + (1 - fraction_confident) * k_max
    expert_flops = 2 * 2 * hidden_dim * intermediate_dim * num_tokens * mean_active
    gate_flops = 2 * hidden_dim * num_experts * num_tokens
    total = gate_flops + expert_flops
    return {
        "mean_active_experts": mean_active,
        "gate_flops": gate_flops,
        "expected_expert_flops": expert_flops,
        "expected_total_flops": total,
        "vs_dense_factor": mean_active,
    }
