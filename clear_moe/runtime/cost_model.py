"""CostModel — analytical latency model for CLEAR-MoE++ runtime.

Implements §8 formula from the research plan:
    T_total = T_router + T_bucket + T_dispatch + max_r(T_expert_r)
            + T_combine + T_sync + T_bubble

Also computes additional metrics for paper analysis:
    path_entropy, path_churn, expert_hotness_skew,
    dispatch_bytes_per_token, remote_routing_fraction,
    parallel_slack, straggler_ratio
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class CostModelOutput:
    """Complete latency and efficiency model output."""
    # Stage times (ms)
    t_router: float = 0.0
    t_bucket: float = 0.0
    t_dispatch: float = 0.0
    t_expert_max: float = 0.0
    t_combine: float = 0.0
    t_sync: float = 0.0
    t_bubble: float = 0.0

    # Derived
    t_total: float = 0.0

    # Routing quality metrics
    path_entropy: float = 0.0          # -Σ p(e) log p(e) over routing distribution
    path_churn: float = 0.0            # mean Hamming(r_l, r_{l+1})
    expert_hotness_skew: float = 0.0   # max_load / mean_load

    # Communication metrics
    dispatch_bytes_per_token: int = 0  # bytes sent per token in EP dispatch
    remote_routing_fraction: float = 0.0

    # Parallelism efficiency
    parallel_slack: float = 0.0        # (T_expert_max - T_expert_mean) / T_expert_max
    straggler_ratio: float = 0.0       # T_expert_max / T_expert_mean

    def to_dict(self) -> Dict[str, float]:
        return {
            't_router': self.t_router,
            't_bucket': self.t_bucket,
            't_dispatch': self.t_dispatch,
            't_expert_max': self.t_expert_max,
            't_combine': self.t_combine,
            't_sync': self.t_sync,
            't_bubble': self.t_bubble,
            't_total': self.t_total,
            'path_entropy': self.path_entropy,
            'path_churn': self.path_churn,
            'expert_hotness_skew': self.expert_hotness_skew,
            'dispatch_bytes_per_token': float(self.dispatch_bytes_per_token),
            'remote_routing_fraction': self.remote_routing_fraction,
            'parallel_slack': self.parallel_slack,
            'straggler_ratio': self.straggler_ratio,
        }


class CostModel:
    """Computes CLEAR-MoE++ cost model metrics from timing and routing data.

    Usage:
        model = CostModel(num_experts=4, hidden_dim=384, dtype_bytes=4)
        output = model.compute(
            timing_dict=profiler.get_timing().to_dict(),
            load_vec=router_aux['load_vec'],
            path_churn=router_aux.get('path_churn', 0.0),
            expert_times=[...],
            remote_routing_fraction=0.3,
        )
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        dtype_bytes: int = 4,  # float32 = 4 bytes
    ):
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.dtype_bytes = dtype_bytes

    def compute(
        self,
        timing_dict: Dict[str, float],
        load_vec: Optional[torch.Tensor] = None,
        path_churn: float = 0.0,
        expert_times: Optional[List[float]] = None,
        remote_routing_fraction: float = 0.0,
        bubble_ratio: float = 0.0,
    ) -> CostModelOutput:
        """Compute full cost model output.

        Args:
            timing_dict:             from DispatchTiming.to_dict()
            load_vec:                (E,) normalized routing distribution
            path_churn:              mean Hamming distance from PathConsistentRouter
            expert_times:            per-expert execution times in ms
            remote_routing_fraction: fraction of tokens to remote experts
            bubble_ratio:            PP bubble fraction

        Returns:
            CostModelOutput with all metrics
        """
        out = CostModelOutput()

        # Stage times from profiler
        out.t_router = timing_dict.get('t_router', 0.0)
        out.t_bucket = timing_dict.get('t_bucket', 0.0)
        out.t_dispatch = timing_dict.get('t_dispatch', 0.0)
        out.t_expert_max = timing_dict.get('t_expert_max', 0.0)
        out.t_combine = timing_dict.get('t_combine', 0.0)
        out.t_sync = timing_dict.get('t_sync', 0.0)
        out.t_bubble = timing_dict.get('t_bubble', 0.0)

        # T_total per plan §8 formula
        out.t_total = (
            out.t_router + out.t_bucket + out.t_dispatch
            + out.t_expert_max + out.t_combine + out.t_sync + out.t_bubble
        )

        # Path entropy: -Σ p(e) log p(e)
        if load_vec is not None and load_vec.numel() > 0:
            p = load_vec.float().clamp(min=1e-10)
            p = p / p.sum()
            out.path_entropy = float(-(p * p.log()).sum().item())
        else:
            out.path_entropy = 0.0

        out.path_churn = path_churn

        # Expert hotness skew: max_load / mean_load
        if load_vec is not None and load_vec.numel() > 0:
            mean_load = load_vec.float().mean().item()
            max_load = load_vec.float().max().item()
            out.expert_hotness_skew = max_load / max(mean_load, 1e-10)

        # Dispatch bytes per token in EP mode
        out.dispatch_bytes_per_token = self.hidden_dim * self.dtype_bytes

        out.remote_routing_fraction = remote_routing_fraction

        # Parallelism efficiency from per-expert times
        if expert_times and len(expert_times) > 0:
            t_max = max(expert_times)
            t_mean = sum(expert_times) / len(expert_times)
            out.parallel_slack = (t_max - t_mean) / max(t_max, 1e-10)
            out.straggler_ratio = t_max / max(t_mean, 1e-10)

        return out

    def compute_theoretical(
        self,
        N: int,
        D: int,
        E: int,
        top_k: int = 1,
        num_ranks: int = 1,
        load_vec: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute theoretical cost estimates (no profiling required).

        Useful for roofline analysis before running experiments.

        Returns:
            dict with theoretical FLOPs, bytes, and ideal times (arbitrary units)
        """
        # Expert GEMM: each token processes through top_k experts
        # Each expert FFN: 2 * D * D_ffn FLOPs (approx D_ffn = 4*D)
        d_ffn = 4 * D
        flops_per_token = top_k * 2 * D * d_ffn
        total_flops = N * flops_per_token

        # Router FLOPs: N * D * E (one linear projection)
        router_flops = N * D * E

        # Dispatch bytes in EP: N * D * bytes (all-to-all)
        dispatch_bytes = N * D * self.dtype_bytes * (1 - 1 / max(num_ranks, 1))

        return {
            'total_flops': total_flops,
            'router_flops': router_flops,
            'dispatch_bytes': dispatch_bytes,
            'flops_per_token': flops_per_token,
            'expert_compute_fraction': total_flops / max(total_flops + router_flops, 1),
        }
