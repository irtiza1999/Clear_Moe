"""ExpertChoiceRouter — experts choose tokens rather than tokens choosing experts.

Provides deterministic expert capacity (no overflow/drop).
Each expert selects exactly floor(capacity_factor * N / E) tokens by affinity score.
Reference: Zhou et al. "Mixture-of-Experts with Expert Choice Routing" arXiv:2202.09368
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from clear_moe.routers.base import BaseRouter


class ExpertChoiceRouter(BaseRouter):
    """Experts select top-C tokens from the batch.

    Unlike token-choice routing, every expert gets exactly `capacity` tokens.
    Some tokens may be selected by 0 experts (handled via fallback to expert 0)
    or selected by multiple experts (weights averaged).

    For compatibility with the BaseRouter interface, output is reformatted
    to (N, top_k) per-token format with assignments from expert selection.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        top_k: int = 1,
    ):
        super().__init__(hidden_dim, num_experts, top_k)
        self.capacity_factor = capacity_factor
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
        orig_shape = x.shape
        # Flatten to (N, D)
        if x.dim() == 3:
            B, S, D = x.shape
            N = B * S
            x_flat = x.reshape(N, D)
        else:
            N = x.shape[0]
            x_flat = x

        # Affinity scores: (N, E)
        logits = self.gate(x_flat)
        scores = F.softmax(logits, dim=-1)  # (N, E)

        # Expert-view: each expert selects top-C tokens
        capacity = max(1, int(self.capacity_factor * N / self.num_experts))
        # scores_T: (E, N) — each row = one expert's affinity with all tokens
        scores_T = scores.T  # (E, N)
        top_scores, top_token_ids = torch.topk(scores_T, capacity, dim=-1)  # (E, C)

        # Build per-token assignments: which expert chose each token?
        # Initialize all tokens to expert 0 (fallback for unchosen tokens)
        indices = torch.zeros(N, 1, dtype=torch.long, device=x.device)
        weights = torch.zeros(N, 1, device=x.device)
        assigned = torch.zeros(N, dtype=torch.bool, device=x.device)

        for e in range(self.num_experts):
            chosen = top_token_ids[e]  # (C,) token indices chosen by expert e
            # For tokens already assigned, keep max weight; for new, assign
            new_mask = ~assigned[chosen]
            new_tokens = chosen[new_mask]
            if new_tokens.numel() > 0:
                indices[new_tokens, 0] = e
                weights[new_tokens, 0] = top_scores[e][new_mask]
                assigned[new_tokens] = True

        # Any remaining unassigned token → expert 0, weight = 1.0
        unassigned = ~assigned
        if unassigned.any():
            indices[unassigned, 0] = 0
            weights[unassigned, 0] = 1.0

        # Normalize weights
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)

        # Reshape back to original leading dims
        if x.dim() == 3:
            indices = indices.reshape(B, S, 1)
            weights = weights.reshape(B, S, 1)
            logits = logits.reshape(B, S, self.num_experts)

        # Load balance: count tokens per expert
        flat_idx = indices.reshape(-1)
        load_vec = torch.bincount(flat_idx, minlength=self.num_experts).float() / N
        entropy = -(scores * scores.log().clamp(min=-100)).sum(-1).mean().item()

        return indices, weights, logits, {
            'path_state': None,
            'comm_cost_est': 0.0,
            'entropy': entropy,
            'load_vec': load_vec,
            'capacity': capacity,
            'expert_token_map': top_token_ids,  # (E, C) for analysis
        }
