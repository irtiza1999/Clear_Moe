"""CommAwareRouter — N1 novelty: routing score penalized by dispatch cost.

Score: s(x, e) = s_task(x, e) - lambda_comm * c_device(e) - lambda_hot * c_load(e)

Where:
    s_task:    standard softmax routing score
    c_device:  1.0 if expert on remote rank, 0.0 if local (from device_map)
    c_load:    EMA-tracked token count per expert (hotness penalty)

Reference: CLEAR-MoE++ plan §4.2 N1
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from clear_moe.routers.base import BaseRouter


class CommAwareRouter(BaseRouter):
    """Communication-aware router: penalizes remote and overloaded experts.

    router aware of parallel execution cost at routing time — this is the
    key differentiator vs. standard load-balance loss.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
        lambda_comm: float = 0.1,
        lambda_hot: float = 0.1,
        ema_momentum: float = 0.9,
    ):
        super().__init__(hidden_dim, num_experts, top_k)
        self.lambda_comm = lambda_comm
        self.lambda_hot = lambda_hot
        self.ema_momentum = ema_momentum
        self.gate = nn.Linear(hidden_dim, num_experts, bias=True)
        nn.init.kaiming_uniform_(self.gate.weight, a=0.01)
        nn.init.zeros_(self.gate.bias)
        # EMA load tracker: (E,) normalized token fractions per expert
        self.register_buffer('load_ema', torch.ones(num_experts) / num_experts)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: Optional[int] = None,
        device_map: Optional[dict] = None,
        load_stats: Optional[torch.Tensor] = None,
    ):
        # Base task score
        logits = self.gate(x)  # (..., E)

        # Communication cost penalty: c_device(e) ∈ {0, 1}
        if device_map is not None:
            local_rank = device_map.get('local_rank', 0)
            c_device = torch.tensor(
                [0.0 if device_map.get(e, 0) == local_rank else 1.0
                 for e in range(self.num_experts)],
                device=x.device, dtype=x.dtype,
            )
        else:
            c_device = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)

        # Load hotness penalty: use external load_stats if provided, else internal EMA
        if load_stats is not None:
            c_load = load_stats.to(x.device, dtype=x.dtype)
        else:
            c_load = self.load_ema.to(x.device, dtype=x.dtype)

        # Penalized logits
        penalized_logits = logits - self.lambda_comm * c_device - self.lambda_hot * c_load

        probs = F.softmax(penalized_logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Update EMA with current batch routing
        with torch.no_grad():
            flat_indices = indices.reshape(-1).long()
            counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
            counts = counts / (counts.sum() + 1e-8)
            self.load_ema.mul_(self.ema_momentum).add_(
                counts.to(self.load_ema.device) * (1.0 - self.ema_momentum)
            )

        comm_cost_est = float(
            (self.lambda_comm * c_device + self.lambda_hot * c_load).mean().item()
        )
        entropy = -(probs * probs.log().clamp(min=-100)).sum(-1).mean().item()

        return indices, weights, penalized_logits, {
            'path_state': None,
            'comm_cost_est': comm_cost_est,
            'entropy': entropy,
            'load_vec': self.load_ema.clone(),
            'c_device': c_device,
            'c_load': c_load.clone(),
        }

    def reset_load_ema(self):
        """Reset EMA load tracker (call between epochs)."""
        nn.init.constant_(self.load_ema, 1.0 / self.num_experts)
