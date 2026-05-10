"""
CLEAR-MoE Grouped Expert MLP — The MoE FFN replacement module.

Architecture:
    output = shared_basis(x) + gate_weight * expert_e(x)

Where:
- shared_basis: low-rank approximation of the original FFN (via SVD)
- expert_e: residual expert specific to the routed cluster
- gate_weight: routing weight from the router

The shared basis preserves common visual structure, while residual
experts capture cluster-specific specialization. This avoids the
parameter inflation and quality degradation of fully disjoint experts.

Dispatch modes (set dispatch_mode in GroupedExpertMLP.__init__)
---------------------------------------------------------------
'grouped'       — Route-sorted sequential grouped GEMMs (default, existing).
'naive'         — No sorting, boolean-mask dispatch per expert (benchmark baseline).
'stream'        — CUDA stream parallel dispatch: all experts on dedicated streams.
'fusion'        — Fuse dominant expert with shared basis when load is imbalanced.
'adaptive'      — Runtime-adaptive: auto-select stream/fusion/grouped by load stats.

See runtime/parallel_expert_dispatch.py for full strategy documentation.
"""

import logging
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from runtime.route_sort import route_sort, route_sort_vectorized, unscramble

logger = logging.getLogger(__name__)

DispatchMode = Literal["grouped", "naive", "stream", "fusion", "adaptive"]


class GroupedExpertMLP(nn.Module):
    """
    MoE FFN module with shared basis + routed residual experts.

    Replaces a standard FFN (fc1 -> activation -> fc2) with:
    1. Shared basis fc1 + fc2 (applied to all tokens)
    2. Per-expert residual fc1 + fc2 (applied only to routed tokens)
    3. Route-sorted grouped execution for GPU efficiency

    forward(x, expert_indices, expert_weights) -> output
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        shared_basis_U: torch.Tensor,
        shared_basis_S: torch.Tensor,
        shared_basis_V: torch.Tensor,
        shared_weight_1: Optional[torch.Tensor] = None,
        shared_bias_1: Optional[torch.Tensor] = None,
        expert_weights_1: List[torch.Tensor] = None,
        expert_biases_1: List[torch.Tensor] = None,
        expert_weights_2: List[torch.Tensor] = None,
        expert_biases_2: List[torch.Tensor] = None,
        activation_fn: nn.Module = None,
        dispatch_mode: DispatchMode = "grouped",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.shared_rank = shared_basis_U.shape[1]
        self.dispatch_mode = dispatch_mode

        # Shared basis: decomposed as U @ diag(S) @ V
        # For efficiency, precompute W_shared = U @ diag(S) @ V
        # Note: shared_basis_U, S, V come from SVD of fc2, so W_shared has shape (hidden_dim, intermediate_dim)
        # This represents the mapping from intermediate_dim -> hidden_dim (i.e., fc2's output projection)
        W_shared = shared_basis_U @ torch.diag(shared_basis_S) @ shared_basis_V
        
        # Shared fc1: use provided weights if available (e.g. from original backbone)
        self.shared_fc1 = nn.Linear(hidden_dim, intermediate_dim, bias=True)
        if shared_weight_1 is not None:
            self.shared_fc1.weight.data.copy_(shared_weight_1)
            if shared_bias_1 is not None:
                self.shared_fc1.bias.data.copy_(shared_bias_1)
            else:
                nn.init.zeros_(self.shared_fc1.bias)
        else:
            nn.init.kaiming_uniform_(self.shared_fc1.weight, a=0.01)
            nn.init.zeros_(self.shared_fc1.bias)

        # Shared fc2: apply the low-rank shared basis approximation
        self.shared_fc2 = nn.Linear(intermediate_dim, hidden_dim, bias=True)
        self.shared_fc2.weight.data.copy_(W_shared)
        nn.init.zeros_(self.shared_fc2.bias)

        # Per-expert residual layers
        self.expert_fc1s = nn.ModuleList()
        self.expert_fc2s = nn.ModuleList()
        
        # Optimization flag: if expert FC1s are identical to shared FC1, we can skip them
        self.skip_expert_fc1 = True

        for e in range(num_experts):
            fc1 = nn.Linear(hidden_dim, intermediate_dim, bias=True)
            fc1.weight.data.copy_(expert_weights_1[e])
            fc1.bias.data.copy_(expert_biases_1[e])
            self.expert_fc1s.append(fc1)

            # Check if this expert's FC1 matches shared FC1 (for optimization)
            if self.skip_expert_fc1:
                if not torch.allclose(fc1.weight, self.shared_fc1.weight, atol=1e-6) or \
                   not torch.allclose(fc1.bias, self.shared_fc1.bias, atol=1e-6):
                    self.skip_expert_fc1 = False

            fc2 = nn.Linear(intermediate_dim, hidden_dim, bias=True)
            fc2.weight.data.copy_(expert_weights_2[e])
            fc2.bias.data.copy_(expert_biases_2[e])
            self.expert_fc2s.append(fc2)

        # Per-expert output scales (learnable during router fitting)
        self.expert_scales = nn.Parameter(torch.ones(num_experts))

        # Activation function
        self.activation = activation_fn or nn.GELU()

        # Freeze shared basis by default (only train router and scales)
        for p in self.shared_fc1.parameters():
            p.requires_grad = False
        for p in self.shared_fc2.parameters():
            p.requires_grad = False

        # Lazy-init dispatch module (needs device context at first forward)
        self._dispatch_module = None

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: Optional[torch.Tensor] = None,
        expert_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the MoE FFN.

        Args:
            x: (..., hidden_dim) input tensor
            expert_indices: (...,) or (..., top_k) expert assignments
                If None, runs as dense shared-basis only (fallback)
            expert_weights: (...,) or (..., top_k) routing weights
                If None, uses uniform weight of 1.0

        Returns:
            (..., hidden_dim) output tensor
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.hidden_dim)  # (N, D)

        # Compute shared basis and intermediate activation (N, D_inter)
        z = self.shared_fc1(x_flat)
        z = self.activation(z)
        shared_out = self.shared_fc2(z)  # (N, D)

        if expert_indices is None:
            return shared_out.reshape(original_shape)

        expert_ids_flat = expert_indices.reshape(-1)  # (N,)

        if expert_weights is not None:
            if expert_weights.numel() > x_flat.shape[0]:
                weights_flat = expert_weights[..., 0].reshape(-1)
            else:
                weights_flat = expert_weights.reshape(-1)
        else:
            weights_flat = torch.ones(x_flat.shape[0], device=x.device)

        output = self._dispatch(x_flat, z, shared_out, expert_ids_flat, weights_flat)
        return output.reshape(original_shape)

    def _dispatch(
        self,
        x_flat: torch.Tensor,
        z: torch.Tensor,
        shared_out: torch.Tensor,
        expert_ids_flat: torch.Tensor,
        weights_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Route tokens to experts using the configured dispatch_mode."""
        mode = self.dispatch_mode

        if mode == "naive":
            return self._dispatch_naive(x_flat, z, shared_out, expert_ids_flat, weights_flat)
        elif mode == "stream":
            return self._dispatch_stream(x_flat, z, shared_out, expert_ids_flat, weights_flat)
        elif mode == "fusion":
            return self._dispatch_fusion(x_flat, z, shared_out, expert_ids_flat, weights_flat)
        elif mode == "adaptive":
            return self._dispatch_adaptive(x_flat, z, shared_out, expert_ids_flat, weights_flat)
        else:  # 'grouped' (default)
            return self._dispatch_grouped(x_flat, z, shared_out, expert_ids_flat, weights_flat)

    def _dispatch_grouped(self, x_flat, z, shared_out, expert_ids_flat, weights_flat):
        """Route-sorted sequential grouped GEMMs (default)."""
        sorted_tokens, sorted_ids, restore_indices, boundaries = route_sort_vectorized(
            x_flat, expert_ids_flat, self.num_experts
        )
        # We need to sort z as well to use it for optimization
        sorted_z, _, _, _ = route_sort_vectorized(
            z, expert_ids_flat, self.num_experts
        )
        expert_output = torch.zeros_like(sorted_tokens)

        for e in range(self.num_experts):
            start = boundaries[e].item()
            end = boundaries[e + 1].item()
            if start >= end:
                continue
            
            if self.skip_expert_fc1:
                # Let's just use the fact that z is already computed and sort it.
                pass # See below

        expert_output = unscramble(expert_output, restore_indices)
        return shared_out + expert_output * weights_flat.unsqueeze(-1)

    def _dispatch_naive(self, x_flat, z, shared_out, expert_ids_flat, weights_flat):
        """Naive boolean-mask dispatch per expert (benchmark baseline)."""
        expert_output = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            mask = expert_ids_flat == e
            if not mask.any():
                continue
            
            if self.skip_expert_fc1:
                out = z[mask] # Use precomputed intermediate
            else:
                out = self.expert_fc1s[e](x_flat[mask])
                out = self.activation(out)
                
            out = self.expert_fc2s[e](out)
            expert_output[mask] = out * self.expert_scales[e]
        return shared_out + expert_output * weights_flat.unsqueeze(-1)

    def _dispatch_stream(self, x_flat, z, shared_out, expert_ids_flat, weights_flat):
        """CUDA stream parallel dispatch — all experts on dedicated streams."""
        if not torch.cuda.is_available():
            return self._dispatch_grouped(x_flat, z, shared_out, expert_ids_flat, weights_flat)

        device = x_flat.device
        sorted_tokens, _, restore_indices, boundaries = route_sort_vectorized(
            x_flat, expert_ids_flat, self.num_experts
        )
        sort_done = torch.cuda.Event()
        sort_done.record()

        if not hasattr(self, '_cuda_streams') or self._cuda_streams is None:
            self._cuda_streams = [torch.cuda.Stream(device=device) for _ in range(self.num_experts)]

        expert_output = torch.zeros_like(sorted_tokens)
        events = []

        for e in range(self.num_experts):
            with torch.cuda.stream(self._cuda_streams[e]):
                self._cuda_streams[e].wait_event(sort_done)
                start = boundaries[e].item()
                end = boundaries[e + 1].item()
                if start < end:
                    tokens_e = sorted_tokens[start:end]
                    if self.skip_expert_fc1:
                        out = self.shared_fc1(tokens_e)
                        out = self.activation(out)
                    else:
                        out = self.expert_fc1s[e](tokens_e)
                        out = self.activation(out)
                    out = self.expert_fc2s[e](out)
                    expert_output[start:end] = out * self.expert_scales[e]
                evt = torch.cuda.Event()
                evt.record(self._cuda_streams[e])
                events.append(evt)

        default = torch.cuda.current_stream(device)
        for evt in events:
            default.wait_event(evt)

        expert_output = unscramble(expert_output, restore_indices)
        return shared_out + expert_output * weights_flat.unsqueeze(-1)

    def _dispatch_fusion(self, x_flat, z, shared_out, expert_ids_flat, weights_flat, threshold=0.60):
        """Fuse dominant expert weights with shared basis when load is skewed."""
        N = x_flat.shape[0]
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=x_flat.device)
        for e in range(self.num_experts):
            counts[e] = (expert_ids_flat == e).sum()

        dom = counts.argmax().item()
        dom_frac = counts[dom].item() / max(N, 1)

        output = torch.zeros_like(x_flat)

        if dom_frac >= threshold:
            # Fused path for dominant expert
            dom_mask = expert_ids_flat == dom
            scale = self.expert_scales[dom].item()
            # Fused weights (Shared + Expert)
            # This only works efficiently if FC1 is shared.
            # If not shared, we'd need to fuse FC1 too.
            if self.skip_expert_fc1:
                w2f = self.shared_fc2.weight + self.expert_fc2s[dom].weight * scale
                b2f = self.shared_fc2.bias + self.expert_fc2s[dom].bias * scale
                if dom_mask.any():
                    # z is activation(shared_fc1(x))
                    fused = F.linear(z[dom_mask], w2f, b2f)
                    output[dom_mask] = fused * weights_flat[dom_mask].unsqueeze(-1)
            else:
                # Non-optimized fusion (just for correctness)
                w1f = self.shared_fc1.weight + self.expert_fc1s[dom].weight * scale
                b1f = self.shared_fc1.bias + self.expert_fc1s[dom].bias * scale
                w2f = self.shared_fc2.weight + self.expert_fc2s[dom].weight * scale
                b2f = self.shared_fc2.bias + self.expert_fc2s[dom].bias * scale
                if dom_mask.any():
                    t = x_flat[dom_mask]
                    fused = F.linear(t, w1f, b1f)
                    fused = self.activation(fused)
                    fused = F.linear(fused, w2f, b2f)
                    output[dom_mask] = fused * weights_flat[dom_mask].unsqueeze(-1)
            
            # Non-dominant experts: normal two-pass
            for e in range(self.num_experts):
                if e == dom:
                    continue
                mask = expert_ids_flat == e
                if not mask.any():
                    continue
                tokens_e = x_flat[mask]
                
                if self.skip_expert_fc1:
                    out = self.expert_fc2s[e](z[mask])
                else:
                    out = self.expert_fc1s[e](tokens_e)
                    out = self.activation(out)
                    out = self.expert_fc2s[e](out)
                    
                output[mask] = (shared_out[mask] + out * self.expert_scales[e]) * weights_flat[mask].unsqueeze(-1)
        else:
            output = self._dispatch_grouped(x_flat, z, shared_out, expert_ids_flat, weights_flat)

        return output

    def _dispatch_adaptive(self, x_flat, shared_out, expert_ids_flat, weights_flat):
        """Auto-select dispatch strategy based on runtime load distribution."""
        N = x_flat.shape[0]
        counts = torch.zeros(self.num_experts, dtype=torch.float, device=x_flat.device)
        for e in range(self.num_experts):
            counts[e] = (expert_ids_flat == e).sum()
        max_frac = (counts.max() / max(N, 1)).item()
        load_cv = (counts.std() / (counts.mean() + 1e-8)).item()

        if max_frac >= 0.60:
            return self._dispatch_fusion(x_flat, shared_out, expert_ids_flat, weights_flat)
        elif torch.cuda.is_available() and load_cv <= 0.30:
            return self._dispatch_stream(x_flat, shared_out, expert_ids_flat, weights_flat)
        else:
            return self._dispatch_grouped(x_flat, shared_out, expert_ids_flat, weights_flat)


class MoEFFNWrapper(nn.Module):
    """
    Wrapper that integrates a GroupedExpertMLP with a Router.

    This is the complete drop-in replacement for a standard FFN layer.
    It handles routing + expert execution in a single forward pass.
    """

    def __init__(
        self,
        moe_ffn: GroupedExpertMLP,
        router: nn.Module,
    ):
        super().__init__()
        self.moe_ffn = moe_ffn
        self.router = router

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Full MoE forward: route then execute.

        Args:
            x: (..., hidden_dim) input tensor

        Returns:
            (..., hidden_dim) output tensor
        """
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Route and execute on the hidden-state tensor only."""
        # Route
        expert_indices, expert_weights, _ = self.router(x)

        # For top-1 routing, squeeze the last dim
        if expert_indices.dim() > x.dim() - 1:
            expert_indices = expert_indices[..., 0]
            expert_weights = expert_weights[..., 0]

        # Execute
        output = self.moe_ffn(x, expert_indices, expert_weights)

        return output


class DenseFFNBaseline(nn.Module):
    """
    Simple dense FFN baseline for comparison.

    Wraps a standard fc1 -> activation -> fc2 pattern.
    """

    def __init__(self, fc1: nn.Linear, fc2: nn.Linear, activation: nn.Module = None):
        super().__init__()
        self.fc1 = fc1
        self.fc2 = fc2
        self.activation = activation or nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
