"""
CLEAR-MoE Parallel Expert Dispatch — Five dispatch strategies for MoE FFN execution.

Strategy taxonomy:
    1. NaiveSequential   — baseline: no sorting, loop over experts with boolean masks
    2. RouteSortedGrouped — existing: sort tokens → sequential grouped GEMMs
    3. CUDAStreamParallel — novel: sort tokens → launch all experts concurrently on
                            dedicated CUDA streams, synchronize after
    4. ExpertFusion      — novel: fuse dominant expert weights with shared basis for
                            single-pass computation when one expert holds >threshold
                            fraction of tokens
    5. AdaptiveLoadBalanced — novel: at runtime, measure load distribution and pick
                              the best of the above strategies for each forward pass

Theoretical analysis
--------------------
Let N = num tokens, E = num experts, t_linear = time per token per expert.

NaiveSequential:     latency ≈ E * t_linear     (all tokens see all experts via mask)
RouteSortedGrouped:  latency ≈ E * (N/E) * t_linear = N * t_linear  (skip empty experts)
CUDAStreamParallel:  latency ≈ max_e(N_e) * t_linear  (overlap expert kernels on GPU SMs)
ExpertFusion:        latency ≈ t_linear + (E-1)*(N_minority/N)*t_linear  (one pass dominant)
AdaptiveLoadBalanced: picks minimum-latency strategy based on measured N_e distribution

For balanced routing (N_e ≈ N/E for all e):
    RouteSortedGrouped gives ~E× improvement over NaiveSequential
    CUDAStreamParallel gives marginal gain over RouteSortedGrouped (already ~balanced)
For imbalanced routing (one expert gets 70%+ tokens):
    ExpertFusion outperforms both by fusing the dominant path
    AdaptiveLoadBalanced detects imbalance and selects ExpertFusion automatically
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from runtime.route_sort import route_sort_vectorized, unscramble

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dispatch result container
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    """Output + timing breakdown from a dispatch strategy."""
    output: torch.Tensor
    timing: Dict[str, float] = field(default_factory=dict)  # ms per phase
    expert_counts: Optional[torch.Tensor] = None            # (E,) token counts
    strategy_used: str = ""


# ---------------------------------------------------------------------------
# 1. Naive Sequential Dispatch (baseline)
# ---------------------------------------------------------------------------

class NaiveSequentialDispatch(nn.Module):
    """
    Baseline: no token sorting. For each expert, boolean-mask the tokens that
    belong to it, compute the expert, scatter results back.

    Pathological for GPU efficiency: creates E small, irregular GEMMs with
    variable-size inputs determined at runtime. Used only as a performance
    lower bound in benchmarks.
    """

    def __init__(self, expert_fc1s, expert_fc2s, expert_scales, shared_module, activation):
        super().__init__()
        self.expert_fc1s = expert_fc1s
        self.expert_fc2s = expert_fc2s
        self.expert_scales = expert_scales
        self.shared_module = shared_module
        self.activation = activation
        self.num_experts = len(expert_fc1s)

    def forward(
        self,
        x_flat: torch.Tensor,
        expert_ids: torch.Tensor,
        weights_flat: torch.Tensor,
    ) -> DispatchResult:
        t0 = _cuda_time()

        # Shared basis (all tokens)
        shared_out = self.shared_module(x_flat)
        t_shared = _cuda_time()

        # Expert residuals via boolean masking
        expert_out = torch.zeros_like(x_flat)

        t_dispatch_start = _cuda_time()
        for e in range(self.num_experts):
            mask = expert_ids == e
            if not mask.any():
                continue
            tokens_e = x_flat[mask]
            out = self.expert_fc1s[e](tokens_e)
            out = self.activation(out)
            out = self.expert_fc2s[e](out)
            expert_out[mask] = out * self.expert_scales[e]
        t_dispatch_end = _cuda_time()

        output = shared_out + expert_out * weights_flat.unsqueeze(-1)

        counts = torch.zeros(self.num_experts, dtype=torch.long, device=x_flat.device)
        for e in range(self.num_experts):
            counts[e] = (expert_ids == e).sum()

        return DispatchResult(
            output=output,
            timing={
                "shared_ms": t_shared - t0,
                "dispatch_ms": t_dispatch_end - t_dispatch_start,
                "total_ms": _cuda_time() - t0,
            },
            expert_counts=counts,
            strategy_used="naive_sequential",
        )


# ---------------------------------------------------------------------------
# 2. Route-Sorted Grouped Dispatch (existing, refactored as module)
# ---------------------------------------------------------------------------

class RouteSortedGroupedDispatch(nn.Module):
    """
    Sort tokens by expert assignment → execute grouped GEMMs sequentially.

    Sort converts E irregular GEMMs into E contiguous GEMMs. Memory-access
    pattern becomes sequential, cache-friendly, and cuBLAS can plan kernels
    on regular shapes. This is the existing CLEAR-MoE approach, refactored
    here for head-to-head benchmark comparison.
    """

    def __init__(self, expert_fc1s, expert_fc2s, expert_scales, shared_module, activation):
        super().__init__()
        self.expert_fc1s = expert_fc1s
        self.expert_fc2s = expert_fc2s
        self.expert_scales = expert_scales
        self.shared_module = shared_module
        self.activation = activation
        self.num_experts = len(expert_fc1s)

    def forward(
        self,
        x_flat: torch.Tensor,
        expert_ids: torch.Tensor,
        weights_flat: torch.Tensor,
    ) -> DispatchResult:
        t0 = _cuda_time()

        shared_out = self.shared_module(x_flat)
        t_shared = _cuda_time()

        sorted_tokens, sorted_ids, restore_indices, boundaries = route_sort_vectorized(
            x_flat, expert_ids, self.num_experts
        )
        t_sort = _cuda_time()

        expert_output = torch.zeros_like(sorted_tokens)

        for e in range(self.num_experts):
            start = boundaries[e].item()
            end = boundaries[e + 1].item()
            if start >= end:
                continue
            tokens_e = sorted_tokens[start:end]
            out = self.expert_fc1s[e](tokens_e)
            out = self.activation(out)
            out = self.expert_fc2s[e](out)
            expert_output[start:end] = out * self.expert_scales[e]

        t_compute = _cuda_time()

        expert_output = unscramble(expert_output, restore_indices)
        output = shared_out + expert_output * weights_flat.unsqueeze(-1)
        t_end = _cuda_time()

        counts = boundaries[1:] - boundaries[:-1]

        return DispatchResult(
            output=output,
            timing={
                "shared_ms": t_shared - t0,
                "sort_ms": t_sort - t_shared,
                "expert_compute_ms": t_compute - t_sort,
                "unscramble_ms": t_end - t_compute,
                "total_ms": t_end - t0,
            },
            expert_counts=counts,
            strategy_used="route_sorted_grouped",
        )


# ---------------------------------------------------------------------------
# 3. CUDA Stream Parallel Dispatch (novel contribution)
# ---------------------------------------------------------------------------

class CUDAStreamParallelDispatch(nn.Module):
    """
    Launch all expert GEMMs concurrently on dedicated CUDA streams.

    After route-sorting, each expert's token slice is dispatched to a separate
    CUDA stream. On GPUs with sufficient SM count, the GPU scheduler can
    execute multiple expert kernels simultaneously, converting sequential
    O(E * t_e) latency to O(max_e t_e) in the ideal parallel case.

    Stream overlap efficiency depends on:
    - Number of free SMs after shared basis computation
    - Size of each expert's GEMM (small GEMMs have less SM overlap potential)
    - Memory bandwidth contention between concurrent GEMMs

    Synchronization: all expert streams sync before the combine step.
    Expert output buffers are pre-allocated to avoid stream-to-stream races.
    """

    def __init__(
        self,
        expert_fc1s,
        expert_fc2s,
        expert_scales,
        shared_module,
        activation,
        num_streams: int = None,
    ):
        super().__init__()
        self.expert_fc1s = expert_fc1s
        self.expert_fc2s = expert_fc2s
        self.expert_scales = expert_scales
        self.shared_module = shared_module
        self.activation = activation
        self.num_experts = len(expert_fc1s)
        # One stream per expert; reuse if num_streams < num_experts
        n_streams = num_streams or self.num_experts
        self._streams = None  # Lazy init (needs CUDA device)
        self._n_streams = n_streams

    def _get_streams(self, device):
        if self._streams is None or len(self._streams) == 0:
            self._streams = [torch.cuda.Stream(device=device) for _ in range(self._n_streams)]
        return self._streams

    def _fallback_grouped(self, x_flat, expert_ids, weights_flat) -> DispatchResult:
        """CPU fallback: same as RouteSortedGroupedDispatch."""
        t0 = _cuda_time()
        shared_out = self.shared_module(x_flat)
        sorted_tokens, _, restore_indices, boundaries = route_sort_vectorized(
            x_flat, expert_ids, self.num_experts
        )
        expert_output = torch.zeros_like(sorted_tokens)
        for e in range(self.num_experts):
            start = boundaries[e].item()
            end = boundaries[e + 1].item()
            if start >= end:
                continue
            tokens_e = sorted_tokens[start:end]
            out = self.expert_fc1s[e](tokens_e)
            out = self.activation(out)
            out = self.expert_fc2s[e](out)
            expert_output[start:end] = out * self.expert_scales[e]
        expert_output = unscramble(expert_output, restore_indices)
        output = shared_out + expert_output * weights_flat.unsqueeze(-1)
        return DispatchResult(
            output=output,
            timing={"total_ms": _cuda_time() - t0},
            expert_counts=boundaries[1:] - boundaries[:-1],
            strategy_used="cuda_stream_parallel(cpu_fallback)",
        )

    def forward(
        self,
        x_flat: torch.Tensor,
        expert_ids: torch.Tensor,
        weights_flat: torch.Tensor,
    ) -> DispatchResult:
        device = x_flat.device

        # Fall back to grouped execution when CUDA unavailable
        if not torch.cuda.is_available() or str(device) == "cpu":
            return self._fallback_grouped(x_flat, expert_ids, weights_flat)

        t0 = _cuda_time()

        # Shared basis on default stream
        shared_out = self.shared_module(x_flat)
        t_shared = _cuda_time()

        # Sort tokens (still on default stream — must complete before expert streams read)
        sorted_tokens, sorted_ids, restore_indices, boundaries = route_sort_vectorized(
            x_flat, expert_ids, self.num_experts
        )
        # Record event: sort done; expert streams will wait on this
        sort_done = torch.cuda.Event()
        sort_done.record()
        t_sort = _cuda_time()

        streams = self._get_streams(device)
        # Pre-allocate output buffer (no race condition: each expert writes to its own slice)
        expert_output = torch.zeros_like(sorted_tokens)

        # Launch all experts concurrently on dedicated streams
        expert_events = []
        for e in range(self.num_experts):
            stream_e = streams[e % len(streams)]
            with torch.cuda.stream(stream_e):
                # Wait for sort to complete before reading sorted_tokens
                stream_e.wait_event(sort_done)

                start = boundaries[e].item()
                end = boundaries[e + 1].item()

                if start < end:
                    tokens_e = sorted_tokens[start:end]
                    out = self.expert_fc1s[e](tokens_e)
                    out = self.activation(out)
                    out = self.expert_fc2s[e](out)
                    expert_output[start:end] = out * self.expert_scales[e]

                # Record completion event for this expert
                evt = torch.cuda.Event()
                evt.record(stream_e)
                expert_events.append(evt)

        # Sync all expert streams back to default stream
        default_stream = torch.cuda.current_stream(device)
        for evt in expert_events:
            default_stream.wait_event(evt)

        t_compute = _cuda_time()

        expert_output = unscramble(expert_output, restore_indices)
        output = shared_out + expert_output * weights_flat.unsqueeze(-1)
        t_end = _cuda_time()

        counts = boundaries[1:] - boundaries[:-1]

        return DispatchResult(
            output=output,
            timing={
                "shared_ms": t_shared - t0,
                "sort_ms": t_sort - t_shared,
                "parallel_expert_ms": t_compute - t_sort,
                "unscramble_ms": t_end - t_compute,
                "total_ms": t_end - t0,
            },
            expert_counts=counts,
            strategy_used="cuda_stream_parallel",
        )


# ---------------------------------------------------------------------------
# 4. Expert Fusion Dispatch (novel contribution)
# ---------------------------------------------------------------------------

class ExpertFusionDispatch(nn.Module):
    """
    Fuse the dominant expert's weights with the shared basis for single-pass
    computation when one expert claims ≥ fusion_threshold fraction of tokens.

    Mechanism: if expert d captures fraction p_d ≥ threshold of tokens,
    we compute a fused linear layer:
        W_fused_fc1 = W_shared_fc1 + W_expert_d_fc1
        W_fused_fc2 = W_shared_fc2 + W_expert_d_fc2

    Then for majority tokens: output = fused_forward(x)  [one pass]
    For minority tokens: output = shared_forward(x) + expert_e_forward(x)  [two passes]

    Net effect: reduces two back-to-back GEMMs (shared + residual) to one fused GEMM
    for the dominant cluster, saving ~50% GEMM operations for those tokens.

    This is especially valuable for imbalanced routing (e.g., fine-grained texture
    expert dominates in uniform-background images).
    """

    def __init__(
        self,
        expert_fc1s,
        expert_fc2s,
        expert_scales,
        shared_module,
        activation,
        fusion_threshold: float = 0.6,
    ):
        super().__init__()
        self.expert_fc1s = expert_fc1s
        self.expert_fc2s = expert_fc2s
        self.expert_scales = expert_scales
        self.shared_module = shared_module
        self.activation = activation
        self.num_experts = len(expert_fc1s)
        self.fusion_threshold = fusion_threshold

        # Extract shared weights for on-the-fly fusion
        # shared_module must expose fc1/fc2 directly
        self._shared_fc1_weight = None
        self._shared_fc1_bias = None
        self._shared_fc2_weight = None
        self._shared_fc2_bias = None
        self._extract_shared_weights(shared_module)

    def _extract_shared_weights(self, shared_module):
        if hasattr(shared_module, 'fc1') and hasattr(shared_module, 'fc2'):
            self._shared_fc1_weight = shared_module.fc1.weight
            self._shared_fc1_bias = shared_module.fc1.bias
            self._shared_fc2_weight = shared_module.fc2.weight
            self._shared_fc2_bias = shared_module.fc2.bias
        elif hasattr(shared_module, 'shared_fc1'):
            self._shared_fc1_weight = shared_module.shared_fc1.weight
            self._shared_fc1_bias = shared_module.shared_fc1.bias
            self._shared_fc2_weight = shared_module.shared_fc2.weight
            self._shared_fc2_bias = shared_module.shared_fc2.bias

    def _fused_forward(
        self,
        tokens: torch.Tensor,
        expert_idx: int,
        scale: float,
    ) -> torch.Tensor:
        """Single-pass fused computation: shared + expert residual in one GEMM."""
        if self._shared_fc1_weight is None:
            # Fallback: two-pass if weights not extractable
            shared = self.shared_module(tokens) if hasattr(self.shared_module, '__call__') else tokens
            exp_out = self.expert_fc1s[expert_idx](tokens)
            exp_out = self.activation(exp_out)
            exp_out = self.expert_fc2s[expert_idx](exp_out)
            return shared + exp_out * scale

        # Fuse weights: W_fused = W_shared + scale * W_expert
        w1_fused = self._shared_fc1_weight + self.expert_fc1s[expert_idx].weight * scale
        b1_fused = self._shared_fc1_bias + self.expert_fc1s[expert_idx].bias * scale
        w2_fused = self._shared_fc2_weight + self.expert_fc2s[expert_idx].weight * scale
        b2_fused = self._shared_fc2_bias + self.expert_fc2s[expert_idx].bias * scale

        out = F.linear(tokens, w1_fused, b1_fused)
        out = self.activation(out)
        out = F.linear(out, w2_fused, b2_fused)
        return out

    def forward(
        self,
        x_flat: torch.Tensor,
        expert_ids: torch.Tensor,
        weights_flat: torch.Tensor,
    ) -> DispatchResult:
        t0 = _cuda_time()
        N = x_flat.shape[0]

        # Count expert loads
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=x_flat.device)
        for e in range(self.num_experts):
            counts[e] = (expert_ids == e).sum()

        fractions = counts.float() / max(N, 1)
        dominant_expert = fractions.argmax().item()
        dominant_frac = fractions[dominant_expert].item()

        t_routing = _cuda_time()

        output = torch.zeros_like(x_flat)

        if dominant_frac >= self.fusion_threshold:
            # Fuse dominant expert with shared basis (single GEMM pass)
            dom_mask = expert_ids == dominant_expert
            dom_tokens = x_flat[dom_mask]
            dom_scale = self.expert_scales[dominant_expert].item()

            if dom_tokens.numel() > 0:
                fused_out = self._fused_forward(dom_tokens, dominant_expert, dom_scale)
                output[dom_mask] = fused_out * weights_flat[dom_mask].unsqueeze(-1)

            t_fused = _cuda_time()

            # Remaining experts: standard shared + residual
            shared_out = self.shared_module(x_flat)
            for e in range(self.num_experts):
                if e == dominant_expert:
                    continue
                mask = expert_ids == e
                if not mask.any():
                    continue
                tokens_e = x_flat[mask]
                out = self.expert_fc1s[e](tokens_e)
                out = self.activation(out)
                out = self.expert_fc2s[e](out)
                output[mask] = (shared_out[mask] + out * self.expert_scales[e]) * weights_flat[mask].unsqueeze(-1)

            t_end = _cuda_time()
            strategy = f"expert_fusion(dom={dominant_expert},frac={dominant_frac:.2f})"
        else:
            # Fall back to route-sorted grouped (no dominant expert)
            sorted_tokens, _, restore_indices, boundaries = route_sort_vectorized(
                x_flat, expert_ids, self.num_experts
            )
            shared_out = self.shared_module(x_flat)
            expert_out_sorted = torch.zeros_like(sorted_tokens)

            for e in range(self.num_experts):
                start = boundaries[e].item()
                end = boundaries[e + 1].item()
                if start >= end:
                    continue
                tokens_e = sorted_tokens[start:end]
                out = self.expert_fc1s[e](tokens_e)
                out = self.activation(out)
                out = self.expert_fc2s[e](out)
                expert_out_sorted[start:end] = out * self.expert_scales[e]

            expert_out = unscramble(expert_out_sorted, restore_indices)
            output = (shared_out + expert_out) * weights_flat.unsqueeze(-1)
            t_end = _cuda_time()
            t_fused = t_routing
            strategy = "expert_fusion(fallback_grouped)"

        return DispatchResult(
            output=output,
            timing={
                "routing_analysis_ms": t_routing - t0,
                "fused_compute_ms": t_fused - t_routing,
                "residual_compute_ms": t_end - t_fused,
                "total_ms": t_end - t0,
            },
            expert_counts=counts,
            strategy_used=strategy,
        )


# ---------------------------------------------------------------------------
# 5. Adaptive Load-Balanced Dispatch (novel contribution)
# ---------------------------------------------------------------------------

class AdaptiveLoadBalancedDispatch(nn.Module):
    """
    Runtime-adaptive dispatch: measure expert load distribution each forward
    pass and select the optimal dispatch strategy automatically.

    Decision policy:
        load_cv = std(counts) / mean(counts)   (coefficient of variation)
        max_frac = max(counts) / N             (dominant expert fraction)

        if max_frac >= fusion_threshold:
            → ExpertFusionDispatch  (one expert dominates → fuse it)
        elif load_cv <= balance_threshold and cuda_available:
            → CUDAStreamParallelDispatch  (balanced load → parallel streams)
        else:
            → RouteSortedGroupedDispatch  (default efficient path)

    Also tracks running statistics to adapt thresholds over time.
    """

    def __init__(
        self,
        expert_fc1s,
        expert_fc2s,
        expert_scales,
        shared_module,
        activation,
        fusion_threshold: float = 0.60,
        balance_threshold: float = 0.30,
        warmup_steps: int = 10,
    ):
        super().__init__()
        self.num_experts = len(expert_fc1s)
        self.fusion_threshold = fusion_threshold
        self.balance_threshold = balance_threshold
        self.warmup_steps = warmup_steps
        self._step = 0

        # All three strategies share the same parameters
        self.grouped = RouteSortedGroupedDispatch(
            expert_fc1s, expert_fc2s, expert_scales, shared_module, activation
        )
        self.stream_parallel = CUDAStreamParallelDispatch(
            expert_fc1s, expert_fc2s, expert_scales, shared_module, activation
        )
        self.fusion = ExpertFusionDispatch(
            expert_fc1s, expert_fc2s, expert_scales, shared_module, activation,
            fusion_threshold=fusion_threshold,
        )

        # Rolling latency tracking per strategy (for feedback)
        self._latency_history: Dict[str, List[float]] = {
            "route_sorted_grouped": [],
            "cuda_stream_parallel": [],
            "expert_fusion": [],
        }

    def _select_strategy(self, expert_ids: torch.Tensor) -> str:
        N = expert_ids.numel()
        counts = torch.zeros(self.num_experts, dtype=torch.float, device=expert_ids.device)
        for e in range(self.num_experts):
            counts[e] = (expert_ids == e).sum()

        max_frac = (counts.max() / max(N, 1)).item()

        if max_frac >= self.fusion_threshold:
            return "fusion"

        if (torch.cuda.is_available()
                and str(expert_ids.device) != "cpu"
                and self._step >= self.warmup_steps):
            load_cv = (counts.std() / (counts.mean() + 1e-8)).item()
            if load_cv <= self.balance_threshold:
                return "stream"

        return "grouped"

    def forward(
        self,
        x_flat: torch.Tensor,
        expert_ids: torch.Tensor,
        weights_flat: torch.Tensor,
    ) -> DispatchResult:
        self._step += 1
        strategy = self._select_strategy(expert_ids)

        if strategy == "fusion":
            result = self.fusion(x_flat, expert_ids, weights_flat)
        elif strategy == "stream":
            result = self.stream_parallel(x_flat, expert_ids, weights_flat)
        else:
            result = self.grouped(x_flat, expert_ids, weights_flat)

        # Track latency for feedback adaptation
        key = result.strategy_used.split("(")[0]
        if key in self._latency_history:
            self._latency_history[key].append(result.timing.get("total_ms", 0.0))
            # Keep last 100 measurements
            if len(self._latency_history[key]) > 100:
                self._latency_history[key].pop(0)

        return result

    def get_latency_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for k, hist in self._latency_history.items():
            if hist:
                t = torch.tensor(hist)
                stats[k] = {
                    "mean_ms": t.mean().item(),
                    "p50_ms": t.median().item(),
                    "p90_ms": t.quantile(0.9).item(),
                    "count": len(hist),
                }
        return stats


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    num_tokens: int = 196         # ViT-S/16 tokens per image at 224x224
    batch_size: int = 8
    hidden_dim: int = 384         # DeiT-Small hidden dim
    intermediate_dim: int = 1536  # 4x hidden
    num_experts: int = 4
    num_warmup: int = 20
    num_iters: int = 100
    device: str = "cuda"


def create_synthetic_dispatch_modules(
    cfg: BenchmarkConfig,
    device: str = "cuda",
) -> Dict[str, nn.Module]:
    """
    Create one synthetic set of MLP weights and return all five dispatch
    strategies wrapping the same weights (fair comparison).
    """
    H, I, E = cfg.hidden_dim, cfg.intermediate_dim, cfg.num_experts

    # Shared basis module (simple two-linear wrapper)
    class SharedBasis(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(H, I).to(device)
            self.fc2 = nn.Linear(I, H).to(device)
            self.act = nn.GELU()
        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    shared = SharedBasis()
    activation = nn.GELU().to(device)

    expert_fc1s = nn.ModuleList([nn.Linear(H, I).to(device) for _ in range(E)])
    expert_fc2s = nn.ModuleList([nn.Linear(I, H).to(device) for _ in range(E)])
    expert_scales = nn.Parameter(torch.ones(E, device=device))

    kwargs = dict(
        expert_fc1s=expert_fc1s,
        expert_fc2s=expert_fc2s,
        expert_scales=expert_scales,
        shared_module=shared,
        activation=activation,
    )

    return {
        "naive_sequential": NaiveSequentialDispatch(**kwargs),
        "route_sorted_grouped": RouteSortedGroupedDispatch(**kwargs),
        "cuda_stream_parallel": CUDAStreamParallelDispatch(**kwargs),
        "expert_fusion": ExpertFusionDispatch(**kwargs),
        "adaptive": AdaptiveLoadBalancedDispatch(**kwargs),
    }


def run_dispatch_benchmark(
    cfg: BenchmarkConfig,
    load_imbalance: float = 0.0,
) -> Dict[str, Dict]:
    """
    Benchmark all dispatch strategies under a given load imbalance scenario.

    Args:
        cfg: benchmark configuration
        load_imbalance: fraction of tokens forced to expert 0.
            0.0 = uniform load; 0.8 = expert 0 gets 80% tokens.

    Returns:
        results dict keyed by strategy name
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA unavailable — benchmark runs on CPU, times will differ")

    device = cfg.device if torch.cuda.is_available() else "cpu"
    N = cfg.batch_size * cfg.num_tokens
    H = cfg.hidden_dim
    E = cfg.num_experts

    strategies = create_synthetic_dispatch_modules(cfg, device)

    def make_inputs():
        x = torch.randn(N, H, device=device)
        weights = torch.ones(N, device=device) / E

        if load_imbalance > 0:
            # Force expert 0 to get load_imbalance fraction of tokens
            ids = torch.randint(0, E, (N,), device=device)
            n_dominant = int(N * load_imbalance)
            ids[:n_dominant] = 0
            ids[n_dominant:] = torch.randint(1, E, (N - n_dominant,), device=device)
        else:
            ids = torch.randint(0, E, (N,), device=device)
        return x, ids, weights

    results = {}

    for name, strategy in strategies.items():
        strategy.eval()
        latencies = []

        with torch.no_grad():
            # Warmup
            for _ in range(cfg.num_warmup):
                x, ids, w = make_inputs()
                _ = strategy(x, ids, w)
            if device == "cuda":
                torch.cuda.synchronize()

            # Measure
            for _ in range(cfg.num_iters):
                x, ids, w = make_inputs()
                if device == "cuda":
                    torch.cuda.synchronize()
                t_start = time.perf_counter()
                result = strategy(x, ids, w)
                if device == "cuda":
                    torch.cuda.synchronize()
                t_end = time.perf_counter()
                latencies.append((t_end - t_start) * 1000)  # ms

        lat = torch.tensor(latencies)
        results[name] = {
            "mean_ms": lat.mean().item(),
            "p50_ms": lat.median().item(),
            "p90_ms": lat.quantile(0.9).item(),
            "std_ms": lat.std().item(),
            "min_ms": lat.min().item(),
            "max_ms": lat.max().item(),
            "throughput_tokens_per_sec": (N / (lat.mean().item() / 1000)),
            "speedup_vs_naive": None,  # filled below
        }

    # Compute speedup vs naive sequential
    naive_mean = results.get("naive_sequential", {}).get("mean_ms", 1.0)
    for name in results:
        results[name]["speedup_vs_naive"] = naive_mean / results[name]["mean_ms"]

    return results


def format_benchmark_table(
    results: Dict[str, Dict],
    load_imbalance: float,
) -> str:
    """Format benchmark results as a markdown table."""
    lines = [
        f"\n## Dispatch Strategy Benchmark (load_imbalance={load_imbalance:.0%})\n",
        "| Strategy | p50 (ms) | p90 (ms) | Throughput (tok/s) | Speedup vs Naive |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        speedup = r.get("speedup_vs_naive")
        if speedup is None:
            speedup = 1.0
        lines.append(
            f"| {name} "
            f"| {r['p50_ms']:.3f} "
            f"| {r['p90_ms']:.3f} "
            f"| {r['throughput_tokens_per_sec']:,.0f} "
            f"| {speedup:.2f}× |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: CUDA-synchronized wall-clock timestamp (ms)
# ---------------------------------------------------------------------------

def _cuda_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() * 1000
