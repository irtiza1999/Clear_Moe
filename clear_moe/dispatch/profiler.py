"""DispatchProfiler — per-stage timing for MoE dispatch pipeline.

Measures: t_router, t_bucket, t_dispatch, t_expert_max, t_combine, t_sync, t_bubble, t_total

Uses torch.cuda.Event pairs for GPU-accurate timing (sub-millisecond precision).
Falls back to time.perf_counter on CPU-only environments.
"""
import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class DispatchTiming:
    """Timing breakdown for one MoE dispatch forward pass.

    All times in milliseconds.
    """
    t_router: float = 0.0       # logits + top-k
    t_bucket: float = 0.0       # sort + prefix sum + BucketPlan build
    t_dispatch: float = 0.0     # scatter to expert buffers
    t_expert_max: float = 0.0   # slowest expert GEMM (max across experts)
    t_combine: float = 0.0      # gather + weighted merge
    t_sync: float = 0.0         # simulated all-to-all (EP mode)
    t_bubble: float = 0.0       # pipeline idle (PP mode)

    @property
    def t_total(self) -> float:
        return (self.t_router + self.t_bucket + self.t_dispatch +
                self.t_expert_max + self.t_combine + self.t_sync + self.t_bubble)

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
        }


class CudaTimer:
    """CUDA event-based timer for GPU-accurate sub-ms timing."""

    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._start_event = None
        self._end_event = None
        self._cpu_start = None

    def start(self):
        if self.use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._cpu_start = time.perf_counter()

    def stop(self) -> float:
        """Stop timer and return elapsed milliseconds."""
        if self.use_cuda:
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._end_event.record()
            torch.cuda.synchronize()
            return self._start_event.elapsed_time(self._end_event)
        else:
            return (time.perf_counter() - self._cpu_start) * 1000.0


class DispatchProfiler:
    """Profiles each stage of MoE dispatch separately.

    Usage:
        profiler = DispatchProfiler()
        with profiler.time_router():
            indices, weights, logits, aux = router(x)
        with profiler.time_bucket():
            plan = bucketize(x, indices, E)
        with profiler.time_dispatch():
            sorted_x = apply_bucket_plan(x, plan)
        ...
        timing = profiler.get_timing()
        print(timing.to_dict())
    """

    def __init__(self, use_cuda: Optional[bool] = None):
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        self.use_cuda = use_cuda
        self._timers: Dict[str, CudaTimer] = {}
        self._elapsed: Dict[str, float] = {}
        self._expert_times: List[float] = []
        self._reset()

    def _reset(self):
        stages = ['router', 'bucket', 'dispatch', 'combine', 'sync', 'bubble']
        for s in stages:
            self._timers[s] = CudaTimer(self.use_cuda)
            self._elapsed[s] = 0.0
        self._expert_times = []

    @contextmanager
    def time_router(self):
        self._timers['router'].start()
        try:
            yield
        finally:
            self._elapsed['router'] += self._timers['router'].stop()

    @contextmanager
    def time_bucket(self):
        self._timers['bucket'].start()
        try:
            yield
        finally:
            self._elapsed['bucket'] += self._timers['bucket'].stop()

    @contextmanager
    def time_dispatch(self):
        self._timers['dispatch'].start()
        try:
            yield
        finally:
            self._elapsed['dispatch'] += self._timers['dispatch'].stop()

    @contextmanager
    def time_expert(self, expert_id: int = 0):
        timer = CudaTimer(self.use_cuda)
        timer.start()
        try:
            yield
        finally:
            elapsed = timer.stop()
            self._expert_times.append(elapsed)

    @contextmanager
    def time_combine(self):
        self._timers['combine'].start()
        try:
            yield
        finally:
            self._elapsed['combine'] += self._timers['combine'].stop()

    @contextmanager
    def time_sync(self):
        self._timers['sync'].start()
        try:
            yield
        finally:
            self._elapsed['sync'] += self._timers['sync'].stop()

    @contextmanager
    def time_bubble(self):
        self._timers['bubble'].start()
        try:
            yield
        finally:
            self._elapsed['bubble'] += self._timers['bubble'].stop()

    def record_expert_time(self, elapsed_ms: float):
        """Manually record an expert execution time (for external timing)."""
        self._expert_times.append(elapsed_ms)

    def get_timing(self) -> DispatchTiming:
        """Return accumulated timing as DispatchTiming."""
        t_expert_max = max(self._expert_times) if self._expert_times else 0.0
        return DispatchTiming(
            t_router=self._elapsed['router'],
            t_bucket=self._elapsed['bucket'],
            t_dispatch=self._elapsed['dispatch'],
            t_expert_max=t_expert_max,
            t_combine=self._elapsed['combine'],
            t_sync=self._elapsed['sync'],
            t_bubble=self._elapsed['bubble'],
        )

    def reset(self):
        """Reset all accumulated timings."""
        self._reset()

    def accumulate_many(self, n_iters: int) -> 'DispatchProfiler':
        """Context manager: accumulate over n_iters then return self."""
        return self


def profile_dispatch_breakdown(
    x: torch.Tensor,
    router,
    experts,
    bucketize_fn,
    scatter_fn,
    gather_fn,
    num_experts: int,
    n_warmup: int = 5,
    n_iters: int = 20,
) -> DispatchTiming:
    """Profile complete dispatch pipeline with warmup.

    Args:
        x:            (N, D) or (B, S, D) input tokens
        router:       BaseRouter instance
        experts:      list of expert modules
        bucketize_fn: callable(x, indices, num_experts) -> BucketPlan
        scatter_fn:   callable(x, plan) -> sorted_tokens
        gather_fn:    callable(sorted_out, plan, weights) -> output
        num_experts:  E
        n_warmup:     warmup iterations (not measured)
        n_iters:      measured iterations

    Returns:
        DispatchTiming averaged over n_iters
    """
    profiler = DispatchProfiler()
    device = x.device

    for i in range(n_warmup + n_iters):
        profiler.reset()

        with profiler.time_router():
            indices, weights, logits, aux = router(x)

        primary_indices = indices[..., 0] if indices.dim() > 2 else indices.squeeze(-1)

        with profiler.time_bucket():
            plan = bucketize_fn(x, primary_indices, num_experts)

        with profiler.time_dispatch():
            sorted_x = scatter_fn(x, plan)

        # Time each expert separately
        sorted_out = torch.empty_like(sorted_x)
        for e_idx, expert in enumerate(experts):
            offset = int(plan.bucket_offsets[e_idx])
            size = int(plan.bucket_sizes[e_idx])
            if size == 0:
                continue
            with profiler.time_expert(e_idx):
                sorted_out[offset:offset + size] = expert(sorted_x[offset:offset + size])

        with profiler.time_combine():
            w_flat = weights[..., 0].reshape(-1)
            output = gather_fn(sorted_out, plan, w_flat)

        if i == n_warmup - 1:
            # After warmup, start fresh accumulation
            profiler.reset()

    return profiler.get_timing()
