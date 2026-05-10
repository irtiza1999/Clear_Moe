"""SimulatedEP — Expert Parallelism simulation on single GPU.

No real NCCL/distributed. Partitions experts across N virtual ranks via
CPU list partitioning. Simulates all-to-all latency via configurable delay.

Measures: remote_routing_fraction, simulated_comm_overhead_ms, expert_gemm_ms

Reference: CLEAR-MoE++ plan §7.2; DeepEP arXiv:2603.13606 (adapted for single GPU)
"""
import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SimulatedEP:
    """Simulates expert parallelism on a single GPU.

    Virtual ranks: experts partitioned into `num_ranks` groups.
    Tokens destined for remote experts incur `comm_latency_us` microsecond delay.

    Args:
        num_experts:     E
        num_ranks:       number of virtual expert-parallel ranks
        comm_latency_us: simulated all-to-all latency per transfer (microseconds)
        local_rank:      which virtual rank this instance "is" (default: 0)
    """

    def __init__(
        self,
        num_experts: int,
        num_ranks: int = 2,
        comm_latency_us: float = 100.0,
        local_rank: int = 0,
    ):
        if num_ranks > num_experts:
            raise ValueError(f"num_ranks ({num_ranks}) > num_experts ({num_experts})")
        self.num_experts = num_experts
        self.num_ranks = num_ranks
        self.comm_latency_us = comm_latency_us
        self.local_rank = local_rank

        # Assign each expert to a rank (round-robin)
        self.expert_to_rank: Dict[int, int] = {
            e: e % num_ranks for e in range(num_experts)
        }
        # Inverse: rank -> list of expert IDs owned by that rank
        self.rank_to_experts: Dict[int, List[int]] = {r: [] for r in range(num_ranks)}
        for e, r in self.expert_to_rank.items():
            self.rank_to_experts[r].append(e)

        self._last_stats: Dict = {}

    def build_device_map(self) -> dict:
        """Return device_map compatible with CommAwareRouter.

        Format: {expert_id: rank_id, 'local_rank': self.local_rank}
        """
        dm = dict(self.expert_to_rank)
        dm['local_rank'] = self.local_rank
        return dm

    def dispatch(
        self,
        tokens: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Simulated dispatch: sort tokens by owning rank, inject comm latency.

        Args:
            tokens:  (N, D) input tokens
            indices: (N,) primary expert assignment per token

        Returns:
            tokens_by_rank: same (N, D) but with simulated comm overhead measured
            stats: dict with remote_routing_fraction, comm_overhead_ms
        """
        N = tokens.shape[0]
        flat_idx = indices.reshape(-1).long()

        # Count remote tokens (those going to non-local ranks)
        remote_mask = torch.tensor(
            [self.expert_to_rank.get(int(e), 0) != self.local_rank
             for e in flat_idx.tolist()],
            dtype=torch.bool,
        )
        remote_count = remote_mask.sum().item()
        remote_fraction = remote_count / max(N, 1)

        # Simulate all-to-all latency for remote tokens
        if remote_count > 0:
            comm_start = time.perf_counter()
            # Simulate: each remote token incurs latency proportional to comm_latency_us
            # In real EP: this is an all-to-all collective. Here we sleep proportionally.
            simulated_us = self.comm_latency_us * (remote_count / N)
            time.sleep(simulated_us / 1e6)
            comm_ms = (time.perf_counter() - comm_start) * 1000
        else:
            comm_ms = 0.0

        stats = {
            'remote_routing_fraction': remote_fraction,
            'remote_token_count': remote_count,
            'simulated_comm_overhead_ms': comm_ms,
        }
        self._last_stats = stats
        return tokens, stats

    def combine(
        self,
        expert_outputs: torch.Tensor,
        original_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Simulated combine: gather expert outputs back, inject return comm latency.

        Args:
            expert_outputs: (N, D) outputs from expert execution
            original_indices: (N,) original token-to-expert assignments

        Returns:
            combined: (N, D) outputs
            combine_comm_ms: simulated combine communication time
        """
        N = expert_outputs.shape[0]
        flat_idx = original_indices.reshape(-1).long()
        remote_mask = torch.tensor(
            [self.expert_to_rank.get(int(e), 0) != self.local_rank
             for e in flat_idx.tolist()],
            dtype=torch.bool,
        )
        remote_count = remote_mask.sum().item()

        combine_ms = 0.0
        if remote_count > 0:
            t_start = time.perf_counter()
            simulated_us = self.comm_latency_us * (remote_count / N)
            time.sleep(simulated_us / 1e6)
            combine_ms = (time.perf_counter() - t_start) * 1000

        return expert_outputs, combine_ms

    def run_ep_forward(
        self,
        x: torch.Tensor,
        router,
        experts: List[nn.Module],
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Full EP-aware forward pass with timing.

        Returns:
            output:      (B, S, D) or (N, D) expert outputs
            stats:       timing and routing statistics
        """
        orig_shape = x.shape
        if x.dim() == 3:
            N = x.shape[0] * x.shape[1]
            x_flat = x.reshape(N, x.shape[2])
        else:
            x_flat = x
            N = x.shape[0]

        device_map = self.build_device_map()

        # Route
        t0 = time.perf_counter()
        indices, weights, logits, aux = router(
            x, layer_idx=layer_idx, device_map=device_map
        )
        t_router = (time.perf_counter() - t0) * 1000

        primary_idx = indices.reshape(N, -1)[:, 0]

        # Dispatch (with simulated comm)
        dispatched, dispatch_stats = self.dispatch(x_flat, primary_idx)

        # Expert GEMM
        t_expert_start = time.perf_counter()
        from clear_moe.dispatch.bucketize import bucketize, apply_bucket_plan, invert_bucket_plan
        plan = bucketize(dispatched, primary_idx, self.num_experts)
        sorted_x = apply_bucket_plan(dispatched, plan)
        sorted_out = torch.empty_like(sorted_x)
        for e, expert in enumerate(experts):
            offset = int(plan.bucket_offsets[e])
            size = int(plan.bucket_sizes[e])
            if size > 0:
                sorted_out[offset:offset + size] = expert(sorted_x[offset:offset + size])
        output = invert_bucket_plan(sorted_out, plan)
        t_expert = (time.perf_counter() - t_expert_start) * 1000

        # Combine (with simulated comm)
        output, combine_comm_ms = self.combine(output, primary_idx)

        stats = {
            't_router_ms': t_router,
            't_expert_ms': t_expert,
            't_dispatch_comm_ms': dispatch_stats['simulated_comm_overhead_ms'],
            't_combine_comm_ms': combine_comm_ms,
            't_total_ms': t_router + t_expert + dispatch_stats['simulated_comm_overhead_ms'] + combine_comm_ms,
            'remote_routing_fraction': dispatch_stats['remote_routing_fraction'],
            'num_ranks': self.num_ranks,
        }

        return output.reshape(orig_shape), stats
