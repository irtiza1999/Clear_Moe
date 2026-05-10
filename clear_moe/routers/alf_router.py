"""ALFRouter — Auxiliary-Loss-Free load balancing router.

DeepSeek-V3 (arXiv:2412.19437) showed that bias correction in the gate
logits gives better expert specialization than auxiliary balancing losses,
which distort the routing distribution. This adapts that idea to
post-training extracted vision MoE.

Mechanism:
    adjusted_logit(x, e) = gate(x)[e] + bias[e]
    top_k on adjusted logits → indices, weights
    After each forward, update bias[e] -= lr * sign(load_count[e] - target_count)
    No auxiliary loss added to training objective.

Why this fills a gap: DeepSeek-V3 ALF only demonstrated for large language
MoE trained from scratch. Here we apply it to post-training extracted vision
experts where aux-loss distortion is especially harmful (experts already
specialized by KMeans; additional loss pressure deforms that structure).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from clear_moe.routers.base import BaseRouter


class ALFRouter(BaseRouter):
    """Auxiliary-loss-free load balancing via gate bias correction.

    Args:
        hidden_dim:       D — token feature dimension
        num_experts:      E
        top_k:            tokens select top_k experts
        bias_update_rate: step size for bias correction after each forward
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
        bias_update_rate: float = 0.01,
    ):
        super().__init__(hidden_dim, num_experts, top_k)
        self.bias_update_rate = bias_update_rate
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.kaiming_uniform_(self.gate.weight, a=0.01)
        # Bias buffer: one scalar per expert, updated in-place without grad
        self.register_buffer("bias", torch.zeros(num_experts))

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int = None,
        device_map: dict = None,
        load_stats: torch.Tensor = None,
    ):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])   # (N, D)
        N = flat.shape[0]

        # Gate logits + bias correction (bias not backpropagated)
        logits = self.gate(flat)                        # (N, E)
        adjusted = logits + self.bias.detach()          # (N, E)
        scores = F.softmax(adjusted, dim=-1)

        top_weights, top_indices = torch.topk(scores, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # Count tokens per expert this batch
        load_counts = torch.zeros(self.num_experts, device=x.device)
        for k in range(self.top_k):
            load_counts.scatter_add_(
                0, top_indices[:, k],
                torch.ones(N, device=x.device),
            )

        # Bias update: sign-based step, no gradient
        target = float(N * self.top_k) / self.num_experts
        with torch.no_grad():
            delta = load_counts - target          # positive → overloaded
            self.bias -= self.bias_update_rate * torch.sign(delta)

        # Reshape back to input spatial dims
        out_shape = shape[:-1] + (self.top_k,)
        top_indices = top_indices.reshape(out_shape)
        top_weights = top_weights.reshape(out_shape)
        logits = logits.reshape(shape[:-1] + (self.num_experts,))

        aux = {
            "bias": self.bias.clone(),
            "load_counts": load_counts,
        }
        return top_indices, top_weights, logits, aux

    def reset_bias(self):
        """Reset bias to zero (e.g., at start of new epoch)."""
        with torch.no_grad():
            self.bias.zero_()
