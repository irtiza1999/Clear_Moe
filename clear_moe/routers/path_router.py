"""PathConsistentRouter — N2 novelty: encourages stable routing across adjacent layers.

Wraps any BaseRouter and adds:
    L_path = mean_hamming_distance(r_l(x), r_{l+1}(x))

This is applied as an auxiliary loss during router fine-tuning (zero inference overhead).
Reduces path churn, improves token locality across layers.

Reference: CLEAR-MoE++ plan §4.2 N2; PathMoE arXiv:2603.18297
"""
from typing import Optional
import torch
import torch.nn as nn
from clear_moe.routers.base import BaseRouter


class PathConsistentRouter(nn.Module):
    """Wraps any BaseRouter and tracks cross-layer routing consistency.

    Usage:
        base = LinearRouter(D, E)
        router = PathConsistentRouter(base, path_penalty_coeff=0.01)
        # Call router.reset_path() between sequences/images
        indices, weights, logits, aux = router(x, layer_idx=l)
    """

    def __init__(
        self,
        base_router: BaseRouter,
        path_penalty_coeff: float = 0.01,
    ):
        super().__init__()
        self.base_router = base_router
        self.path_penalty_coeff = path_penalty_coeff
        # Expose base_router dims for compatibility
        self.hidden_dim = base_router.hidden_dim
        self.num_experts = base_router.num_experts
        self.top_k = base_router.top_k

        self._prev_indices: Optional[torch.Tensor] = None
        self._path_churn: float = 0.0

    def reset_path(self):
        """Reset stored path state. Call between images/sequences."""
        self._prev_indices = None
        self._path_churn = 0.0

    @property
    def path_churn(self) -> float:
        """Mean Hamming distance between last two consecutive layer assignments."""
        return self._path_churn

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: Optional[int] = None,
        device_map: Optional[dict] = None,
        load_stats: Optional[torch.Tensor] = None,
    ):
        indices, weights, logits, aux = self.base_router(
            x, layer_idx=layer_idx, device_map=device_map, load_stats=load_stats
        )

        # Compute Hamming distance to previous layer assignments
        path_churn = 0.0
        path_loss = 0.0
        if self._prev_indices is not None and self._prev_indices.shape == indices.shape:
            hamming = (indices != self._prev_indices.to(indices.device)).float()
            path_churn = hamming.mean().item()
            if self.training:
                path_loss = self.path_penalty_coeff * path_churn

        self._path_churn = path_churn
        self._prev_indices = indices.detach().clone()

        aux['path_state'] = self._prev_indices
        aux['path_churn'] = path_churn
        aux['path_loss'] = path_loss
        return indices, weights, logits, aux

    def parameters(self, recurse=True):
        return self.base_router.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return self.base_router.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )


def make_path_consistent(router: BaseRouter, path_penalty_coeff: float = 0.01) -> PathConsistentRouter:
    """Convenience wrapper: wraps any router with path consistency."""
    return PathConsistentRouter(router, path_penalty_coeff=path_penalty_coeff)
