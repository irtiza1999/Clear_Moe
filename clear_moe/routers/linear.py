"""LinearRouter and AdaptiveRouter — moved from clear_moe/router.py."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from clear_moe.routers.base import BaseRouter


class LinearRouter(BaseRouter):
    """Simple linear router: projects hidden states to expert logits."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
    ):
        super().__init__(hidden_dim, num_experts, top_k)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=True)
        nn.init.kaiming_uniform_(self.gate.weight, a=0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: Optional[int] = None,
        device_map: Optional[dict] = None,
        load_stats: Optional[torch.Tensor] = None,
    ):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(probs * probs.log().clamp(min=-100)).sum(-1).mean().item()
        return indices, weights, logits, {
            'path_state': None,
            'comm_cost_est': 0.0,
            'entropy': entropy,
            'load_vec': probs.detach().mean(0) if probs.dim() > 1 else probs.detach(),
        }


class AdaptiveRouter(BaseRouter):
    """Confidence-adaptive router: top-1 default, escalates to top-2 for uncertain tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        confidence_threshold: float = 0.6,
    ):
        super().__init__(hidden_dim, num_experts, top_k=2)
        self.confidence_threshold = confidence_threshold
        self.gate = nn.Linear(hidden_dim, num_experts, bias=True)
        nn.init.kaiming_uniform_(self.gate.weight, a=0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: Optional[int] = None,
        device_map: Optional[dict] = None,
        load_stats: Optional[torch.Tensor] = None,
    ):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top2_weights, top2_indices = torch.topk(probs, 2, dim=-1)

        # Confident tokens: use only top-1
        confident = top2_weights[..., 0] >= self.confidence_threshold
        weights = top2_weights.clone()
        indices = top2_indices.clone()
        # Zero out second expert weight for confident tokens
        weights[..., 1] = weights[..., 1] * (~confident).float()
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        entropy = -(probs * probs.log().clamp(min=-100)).sum(-1).mean().item()
        return indices, weights, logits, {
            'path_state': None,
            'comm_cost_est': 0.0,
            'entropy': entropy,
            'load_vec': probs.detach().mean(0) if probs.dim() > 1 else probs.detach(),
        }
