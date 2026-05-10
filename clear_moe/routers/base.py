"""BaseRouter ABC — all routers implement this interface."""
from abc import abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class BaseRouter(nn.Module):
    """Abstract base for all CLEAR-MoE++ routers.

    forward() contract:
        Returns (indices, weights, logits, aux_dict)
        - indices:  (..., top_k) int64 expert IDs
        - weights:  (..., top_k) float32 routing weights, sum=1
        - logits:   (..., num_experts) float32 raw gate logits
        - aux_dict: dict with keys: path_state, comm_cost_est, entropy, load_vec
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        layer_idx: Optional[int] = None,
        device_map: Optional[dict] = None,
        load_stats: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Route tokens to experts.

        Args:
            x: (..., hidden_dim) token representations
            layer_idx: transformer layer index (for path-aware routers)
            device_map: dict mapping expert_id -> rank_id (for comm-aware routers)
            load_stats: (num_experts,) external load override; None = use internal EMA

        Returns:
            indices:  (..., top_k) selected expert IDs
            weights:  (..., top_k) normalized routing weights
            logits:   (..., num_experts) raw gate logits
            aux_dict: {'path_state': Tensor|None, 'comm_cost_est': float,
                       'entropy': float, 'load_vec': Tensor}
        """
        ...
