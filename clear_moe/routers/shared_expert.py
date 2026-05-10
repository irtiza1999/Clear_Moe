"""SharedExpertRouter — 1 always-on shared expert + top-k routed experts.

Shared expert captures common/universal features.
Routed experts handle specialized patterns.
Inspired by DeepSeekMoE (Dai et al. arXiv:2401.06066).
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from clear_moe.routers.base import BaseRouter


class SharedExpertRouter(BaseRouter):
    """Router that always includes expert 0 (shared) plus top-(k-1) routed experts.

    The shared expert always receives all tokens with weight 0.5.
    The remaining top-(k-1) experts split weight 0.5.

    Args:
        shared_expert_id: which expert index is the always-on shared expert (default: 0)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        shared_expert_id: int = 0,
    ):
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        super().__init__(hidden_dim, num_experts, top_k)
        self.shared_expert_id = shared_expert_id
        # Gate only over the (E-1) routed experts
        self.gate = nn.Linear(hidden_dim, num_experts - 1, bias=True)
        nn.init.kaiming_uniform_(self.gate.weight, a=0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: Optional[int] = None,
        device_map: Optional[dict] = None,
        load_stats: Optional[torch.Tensor] = None,
    ):
        prefix = x.shape[:-1]   # (B, S) or (B,) or ()
        device = x.device

        # Routed expert scores (over E-1 experts)
        routed_logits = self.gate(x)          # (..., E-1)
        probs = F.softmax(routed_logits, dim=-1)

        k_routed = max(self.top_k - 1, 0)    # slots for routed experts

        # Shared expert slot
        shared_idx = torch.full(prefix + (1,), self.shared_expert_id, dtype=torch.long, device=device)
        shared_w = torch.full(prefix + (1,), 0.5, device=device)

        if k_routed > 0 and self.num_experts > 1:
            top_weights, top_idx_raw = torch.topk(probs, min(k_routed, self.num_experts - 1), dim=-1)
            # Shift indices to skip shared_expert_id slot
            top_idx = top_idx_raw + (top_idx_raw >= self.shared_expert_id).long()
            top_weights_norm = top_weights / (top_weights.sum(-1, keepdim=True) + 1e-8) * 0.5
            indices = torch.cat([shared_idx, top_idx], dim=-1)    # (..., top_k)
            weights = torch.cat([shared_w, top_weights_norm], dim=-1)
        else:
            indices = shared_idx
            weights = shared_w

        # Full logits (E dims) for logging
        full_logits = torch.zeros(*prefix, self.num_experts, device=device)
        if k_routed > 0 and self.num_experts > 1:
            shift = (top_idx_raw >= self.shared_expert_id).long()
            full_logits.scatter_(-1, top_idx_raw + shift, routed_logits)

        load_vec = probs.reshape(-1, self.num_experts - 1).mean(0).detach()
        # Pad to num_experts
        pad = torch.zeros(1, device=device)
        load_vec = torch.cat([pad, load_vec], dim=0) if self.shared_expert_id == 0 else load_vec

        entropy = -(probs * probs.log().clamp(min=-100)).sum(-1).mean().item()
        return indices, weights, full_logits, {
            'path_state': None,
            'comm_cost_est': 0.0,
            'entropy': entropy,
            'load_vec': load_vec[:self.num_experts],
        }
