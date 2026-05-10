"""MLPRouter — moved from clear_moe/router.py."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from clear_moe.routers.base import BaseRouter


class MLPRouter(BaseRouter):
    """Two-layer MLP router with a small hidden dimension."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        router_hidden_dim: int = 64,
        top_k: int = 1,
    ):
        super().__init__(hidden_dim, num_experts, top_k)
        self.router_hidden_dim = router_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, router_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_experts, bias=True),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=0.01)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: Optional[int] = None,
        device_map: Optional[dict] = None,
        load_stats: Optional[torch.Tensor] = None,
    ):
        logits = self.mlp(x)
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
