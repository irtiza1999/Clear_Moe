"""SelfRoutingProxy — parameter-free routing from PCA of hidden states.

No learned router weights. Routing emerges from principal components of
token representations fitted on calibration data.

Reference: Mohamud et al. "Self-Routing" arXiv:2604.00421 (adapted for vision)
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from clear_moe.routers.base import BaseRouter


class SelfRoutingProxy(BaseRouter):
    """PCA-based parameter-free router.

    Fits principal components on calibration activations.
    At inference, projects tokens onto top-E PC directions → routing scores.
    Zero learned parameters (components are fixed after fit()).
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 1):
        super().__init__(hidden_dim, num_experts, top_k)
        # (E, D) PCA components — fitted, not learned
        self.register_buffer('components', torch.zeros(num_experts, hidden_dim))
        self.register_buffer('fitted', torch.tensor(False))

    def fit(self, activations: torch.Tensor):
        """Fit PCA components from calibration activations.

        Args:
            activations: (N, D) or (B, S, D) token representations
                         from calibration set (precomputed, not synthetic)
        """
        x = activations.reshape(-1, self.hidden_dim).float()
        x = x - x.mean(0, keepdim=True)
        # Covariance eigenvectors via SVD of X^T X
        cov = x.T @ x / x.shape[0]
        try:
            _, _, Vh = torch.linalg.svd(cov)
        except RuntimeError:
            # Fallback: random orthogonal components if SVD fails
            Vh = torch.linalg.qr(torch.randn(self.hidden_dim, self.hidden_dim))[0].T
        self.components.copy_(Vh[:self.num_experts])
        self.fitted.fill_(True)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: Optional[int] = None,
        device_map: Optional[dict] = None,
        load_stats: Optional[torch.Tensor] = None,
    ):
        if not self.fitted.item():
            raise RuntimeError(
                "SelfRoutingProxy not fitted. Call .fit(calibration_activations) first."
            )
        # Project: (..., D) @ (D, E) → (..., E)
        logits = x @ self.components.T
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(probs * probs.log().clamp(min=-100)).sum(-1).mean().item()
        load_vec = probs.reshape(-1, self.num_experts).mean(0).detach()
        return indices, weights, logits, {
            'path_state': None,
            'comm_cost_est': 0.0,
            'entropy': entropy,
            'load_vec': load_vec,
        }
