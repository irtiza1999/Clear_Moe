"""SimulatedPP — Pipeline Parallelism simulation on single GPU.

Splits transformer blocks into P stages. Runs micro-batch pipeline:
stage i processes micro-batch k while stage i+1 processes k-1.

Measures: actual pipeline utilization, bubble_ratio, throughput comparison.
bubble_ratio = (P-1) / (M + P-1) where M = num micro-batches

Reference: CLEAR-MoE++ plan §7.3; CLEAR-MoE++ plan §10 D10
"""
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SimulatedPP:
    """Simulates pipeline parallelism on a single GPU.

    Splits a list of transformer stages into P pipeline stages.
    Runs micro-batch pipeline with configurable M micro-batches.

    Args:
        stages:       list of P nn.Module stages (transformer block groups)
        num_stages:   P (must equal len(stages))
        num_microbatches: M — number of micro-batches to pipeline
    """

    def __init__(
        self,
        stages: List[nn.Module],
        num_microbatches: int = 4,
    ):
        self.stages = stages
        self.num_stages = len(stages)
        self.num_microbatches = num_microbatches

    @property
    def bubble_ratio(self) -> float:
        """Theoretical pipeline bubble ratio: (P-1) / (M + P-1)."""
        P = self.num_stages
        M = self.num_microbatches
        return (P - 1) / (M + P - 1)

    def forward_serial(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Non-pipeline baseline: run all stages sequentially."""
        t_start = time.perf_counter()
        output = x
        stage_times = []
        for stage in self.stages:
            t0 = time.perf_counter()
            output = stage(output)
            stage_times.append((time.perf_counter() - t0) * 1000)
        t_total = (time.perf_counter() - t_start) * 1000
        return output, {
            'mode': 'serial',
            't_total_ms': t_total,
            'stage_times_ms': stage_times,
            'bubble_ratio': 0.0,
            'num_stages': self.num_stages,
            'num_microbatches': 1,
        }

    def forward_pipeline(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Micro-batch pipeline forward. Simulates bubble via timing measurements.

        Note: On single GPU, true pipeline parallelism is not possible.
        This measures sequential micro-batch execution overhead to model
        bubble ratio and throughput vs. serial baseline.
        """
        P = self.num_stages
        M = self.num_microbatches

        # Split batch into M micro-batches
        N = x.shape[0]
        chunk_size = max(1, N // M)
        microbatches = [x[i:i + chunk_size] for i in range(0, N, chunk_size)]
        M_actual = len(microbatches)

        t_start = time.perf_counter()
        outputs = []
        stage_exec_times = []

        # Simulate 1F1B schedule: for each (stage, microbatch) pair
        # On single GPU: run sequentially but measure wall time
        for mb in microbatches:
            out = mb
            mb_times = []
            for stage in self.stages:
                t0 = time.perf_counter()
                out = stage(out)
                mb_times.append((time.perf_counter() - t0) * 1000)
            stage_exec_times.append(mb_times)
            outputs.append(out)

        t_total = (time.perf_counter() - t_start) * 1000
        output = torch.cat(outputs, dim=0)

        # Compute measured bubble ratio
        # In ideal pipeline: total_work_time / (total_work_time + bubble_time)
        # Here: approximate from measured stage times
        avg_stage_times = [
            sum(mb[s] for mb in stage_exec_times) / M_actual
            for s in range(P)
        ]
        t_max_stage = max(avg_stage_times)
        t_ideal = M_actual * t_max_stage * P / P  # M * max_stage for P stages
        bubble_ratio_measured = max(0.0, 1.0 - (t_ideal / t_total)) if t_total > 0 else 0.0

        return output, {
            'mode': 'pipeline',
            't_total_ms': t_total,
            'stage_times_ms': avg_stage_times,
            'bubble_ratio_theoretical': self.bubble_ratio,
            'bubble_ratio_measured': bubble_ratio_measured,
            'num_stages': P,
            'num_microbatches': M_actual,
        }


def split_blocks_into_stages(
    blocks: List[nn.Module],
    num_stages: int,
) -> List[nn.Sequential]:
    """Split transformer block list into num_stages groups.

    Args:
        blocks:     flat list of transformer blocks (e.g., model.blocks)
        num_stages: P — number of pipeline stages

    Returns:
        List of P nn.Sequential modules, each containing ~len(blocks)/P blocks
    """
    n = len(blocks)
    chunk = max(1, n // num_stages)
    stages = []
    for i in range(0, n, chunk):
        stage_blocks = blocks[i:i + chunk]
        stages.append(nn.Sequential(*stage_blocks))
    # Ensure exactly num_stages (merge last partial stage)
    while len(stages) > num_stages:
        last = stages.pop()
        stages[-1] = nn.Sequential(*list(stages[-1].children()), *list(last.children()))
    return stages
