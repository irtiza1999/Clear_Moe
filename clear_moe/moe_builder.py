"""
CLEAR-MoE FFN Module and Builder.

ClearMoEFFN: drop-in replacement for a ViT FFN block.

Architecture:
    z          = activation(shared_fc1(x))
    shared_out = shared_fc2(z)           # low-rank SVD approx of original fc2
    expert_idx = argmax(router(x))
    residual   = expert_fc2s[idx](z)     # per-expert fc2 residual
    output     = shared_out + residual

build_moe_modules_from_expertized: factory that creates a ClearMoEFFN for
each expertized layer using pre-fitted routers.  Returns a dict compatible
with models.replace_ffn_with_moe.
"""
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from clear_moe.extraction import ExpertizedLayer

logger = logging.getLogger(__name__)


class ClearMoEFFN(nn.Module):
    """
    Shared-basis + routed-residual MoE FFN, self-contained replacement for a ViT FFN.

    Shared fc1 is applied to ALL tokens (no routing at fc1).
    Per-expert fc2 residuals are applied only to each token's assigned expert cluster.
    The router is called with grad disabled — only used for inference-time dispatch.
    """

    def __init__(
        self,
        shared_fc1: nn.Linear,
        shared_fc2: nn.Linear,
        expert_fc2s: nn.ModuleList,
        router: nn.Module,
        activation_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.shared_fc1 = shared_fc1
        self.shared_fc2 = shared_fc2
        self.expert_fc2s = expert_fc2s
        self.router = router
        self.activation = activation_fn or nn.GELU()
        self.num_experts = len(expert_fc2s)

        # Freeze shared basis — only router parameters are updated during fitting.
        for p in self.shared_fc1.parameters():
            p.requires_grad = False
        for p in self.shared_fc2.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])  # (N, D_hidden)

        # Shared path — identical for every token.
        z = self.activation(self.shared_fc1(x_flat))  # (N, D_inter)
        shared_out = self.shared_fc2(z)               # (N, D_hidden)

        # Route tokens (no grad; tolerates 3- or 4-value router returns).
        with torch.no_grad():
            router_out = self.router(x_flat)
            idx, weights = router_out[0], router_out[1]
            if idx.dim() > 1:
                idx = idx[:, 0]        # top-1
                weights = weights[:, 0]
        idx = idx.long()

        # Per-expert residual fc2.
        residual = torch.zeros_like(shared_out)
        for e in range(self.num_experts):
            mask = idx == e
            if mask.any():
                residual[mask] = self.expert_fc2s[e](z[mask])

        return (shared_out + residual).reshape(shape)


def build_moe_modules_from_expertized(
    expertized: Dict[str, ExpertizedLayer],
    routers: Dict[str, nn.Module],
    base_model: nn.Module,
    config: dict,
) -> Dict[str, nn.Module]:
    """
    Build one ClearMoEFFN per expertized layer.

    Args:
        expertized: Dict[layer_name -> ExpertizedLayer] from extract_all_experts.
        routers:    Dict[layer_name -> router nn.Module] from fit_all_routers.
        base_model: Original pretrained model (used to detect activation function).
        config:     Experiment config dict (unused beyond logging for now).

    Returns:
        Dict[layer_name -> ClearMoEFFN] ready for replace_ffn_with_moe.
    """
    from clear_moe.extraction import _detect_activation, _get_module

    modules: Dict[str, nn.Module] = {}

    for name, exp in expertized.items():
        if name not in routers:
            logger.warning(f"No router for {name} — skipping")
            continue

        dtype = exp.shared_basis_U.dtype

        # ── Shared fc1 ──────────────────────────────────────────────────
        shared_fc1 = nn.Linear(exp.hidden_dim, exp.intermediate_dim, bias=True)
        if exp.shared_weight_1 is not None:
            shared_fc1.weight.data.copy_(exp.shared_weight_1.to(dtype))
        if exp.shared_bias_1 is not None:
            shared_fc1.bias.data.copy_(exp.shared_bias_1.to(dtype))
        else:
            nn.init.zeros_(shared_fc1.bias)

        # ── Shared fc2 (SVD low-rank approx) ────────────────────────────
        W_shared = (
            exp.shared_basis_U
            @ torch.diag(exp.shared_basis_S)
            @ exp.shared_basis_V
        )  # (d_out, d_inter)
        d_out = W_shared.shape[0]
        shared_fc2 = nn.Linear(exp.intermediate_dim, d_out, bias=True)
        shared_fc2.weight.data.copy_(W_shared)
        nn.init.zeros_(shared_fc2.bias)

        # ── Per-expert fc2 residuals ─────────────────────────────────────
        expert_fc2s = nn.ModuleList()
        for e in range(exp.num_experts):
            fc2 = nn.Linear(exp.intermediate_dim, d_out, bias=True)
            fc2.weight.data.copy_(exp.expert_weights_2[e].to(dtype))
            if exp.expert_biases_2 and e < len(exp.expert_biases_2):
                fc2.bias.data.copy_(exp.expert_biases_2[e].to(dtype))
            else:
                nn.init.zeros_(fc2.bias)
            expert_fc2s.append(fc2)

        # ── Activation function ──────────────────────────────────────────
        try:
            original_ffn = _get_module(base_model, name)
            act_fn = _detect_activation(original_ffn)
        except Exception:
            act_fn = nn.GELU()

        moe_ffn = ClearMoEFFN(
            shared_fc1=shared_fc1,
            shared_fc2=shared_fc2,
            expert_fc2s=expert_fc2s,
            router=routers[name],
            activation_fn=act_fn,
        )
        modules[name] = moe_ffn
        logger.info(
            f"Built ClearMoEFFN for {name}: "
            f"E={exp.num_experts}, rank={exp.shared_rank}, d_out={d_out}"
        )

    return modules
