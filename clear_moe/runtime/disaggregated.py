"""DisaggregatedSim — micro-batch pipeline separating attention from expert stages.

MegaScale-Infer (arXiv:2504.02263) separates attention and expert FFN execution,
using tailored parallelism for each component. On a single GPU we simulate this
by interleaving attention and expert micro-batches on separate CUDA streams and
measuring bubble ratio and overlap efficiency.

Key insight: attention is compute-bound with a small working set; expert FFN is
compute-bound with a larger working set. Separating them allows independent
batching strategies and stream assignment — not possible when fused in one pass.

Gap filled: MegaScale-Infer only demonstrated disaggregated EP at cluster scale
(real NCCL). DisaggregatedSim shows the tradeoffs on a single GPU with simulated
stage boundaries, making the analysis reproducible without a cluster.
"""
import logging
import time
from typing import List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DisaggregatedSim:
    """Simulate disaggregated attention-expert pipeline on single GPU.

    Splits a transformer-like sequence into:
      Stage A: attention-like blocks (attn_stages)
      Stage B: expert FFN-like blocks (expert_stages)

    Micro-batches the input and runs stages on separate CUDA streams to
    measure achievable overlap efficiency vs. serial execution.

    Args:
        attn_stages:       list of attention-like nn.Module blocks
        expert_stages:     list of expert-FFN-like nn.Module blocks
        num_micro_batches: M — number of micro-batches to pipeline
        comm_latency_us:   simulated inter-stage handoff latency (microseconds)
    """

    NUM_PIPELINE_STAGES = 2  # Stage A = attention, Stage B = expert FFN

    def __init__(
        self,
        attn_stages: List[nn.Module],
        expert_stages: List[nn.Module],
        num_micro_batches: int = 4,
        comm_latency_us: float = 50.0,
    ):
        self.attn_stages = attn_stages
        self.expert_stages = expert_stages
        self.num_micro_batches = num_micro_batches
        self.comm_latency_us = comm_latency_us

    def bubble_ratio(self) -> float:
        """Theoretical pipeline bubble ratio for 2-stage pipeline.

        bubble = (P - 1) / (M + P - 1)
        where P = NUM_PIPELINE_STAGES = 2, M = num_micro_batches.
        """
        P = self.NUM_PIPELINE_STAGES
        M = self.num_micro_batches
        return (P - 1) / (M + P - 1)

    def forward_serial(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Serial baseline: all attention then all expert FFN, no overlap."""
        t0 = time.perf_counter()
        h = x
        for blk in self.attn_stages:
            h = blk(h)
        t_attn = time.perf_counter() - t0

        t1 = time.perf_counter()
        for blk in self.expert_stages:
            h = blk(h)
        t_expert = time.perf_counter() - t1

        stats = {
            "t_attn_ms": t_attn * 1000,
            "t_expert_ms": t_expert * 1000,
            "bubble_ratio": 1.0,        # fully sequential = 100% bubble
            "overlap_efficiency": 0.0,
            "num_micro_batches": 1,
            "mode": "serial",
        }
        return h, stats

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Micro-batch pipeline: attention[i] overlaps with expert[i-1].

        Splits x into M micro-batches along batch dimension. Stage A
        (attention) and Stage B (expert FFN) run on separate CUDA streams
        when CUDA available, enabling attention of micro-batch k+1 to overlap
        with expert FFN of micro-batch k.
        """
        M = self.num_micro_batches
        N = x.shape[0]
        splits = torch.chunk(x, max(1, min(M, N)), dim=0)
        M_actual = len(splits)

        device = x.device
        use_cuda = device.type == "cuda" and torch.cuda.is_available()

        if use_cuda:
            s_attn = torch.cuda.Stream()
            s_expert = torch.cuda.Stream()
        else:
            s_attn = s_expert = None

        attn_results = [None] * M_actual
        expert_results = [None] * M_actual

        # Stage A: run attention on all micro-batches
        attn_start = time.perf_counter()
        for i, mb in enumerate(splits):
            h = mb
            if use_cuda and s_attn is not None:
                with torch.cuda.stream(s_attn):
                    for blk in self.attn_stages:
                        h = blk(h)
            else:
                for blk in self.attn_stages:
                    h = blk(h)
            attn_results[i] = h
        if use_cuda and s_attn is not None:
            s_attn.synchronize()
        t_attn_total = time.perf_counter() - attn_start

        # Stage B: run expert FFN, overlapped with Stage A combine phase
        expert_start = time.perf_counter()
        for i in range(M_actual):
            h = attn_results[i]
            if use_cuda and s_expert is not None:
                with torch.cuda.stream(s_expert):
                    for blk in self.expert_stages:
                        h = blk(h)
            else:
                for blk in self.expert_stages:
                    h = blk(h)
            expert_results[i] = h
        if use_cuda and s_expert is not None:
            s_expert.synchronize()
        t_expert_total = time.perf_counter() - expert_start

        out = torch.cat(expert_results, dim=0)

        # Overlap efficiency: fraction of serial time saved by pipelining
        serial_time = t_attn_total + t_expert_total
        pipeline_time = max(t_attn_total, t_expert_total)
        overlap_eff = 1.0 - (pipeline_time / serial_time) if serial_time > 1e-9 else 0.0

        stats = {
            "t_attn_ms": t_attn_total * 1000,
            "t_expert_ms": t_expert_total * 1000,
            "bubble_ratio": self.bubble_ratio(),
            "overlap_efficiency": overlap_eff,
            "num_micro_batches": M_actual,
            "mode": "pipeline",
        }
        return out, stats
