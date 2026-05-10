"""MoEExecutor — central dispatch engine for CLEAR-MoE++.

Backends:
    naive        — loop over experts, padded GEMM (serial baseline)
    grouped      — expert-sorted contiguous dispatch (serial, better cache)
    stream       — 3 CUDA streams: dispatch / expert / combine overlap (parallel)
    stream_fine  — per-expert CUDA streams + dependency-ordered combine (Comet-inspired)
    triton       — Triton fused scatter+GEMM+gather kernel (parallel)
    cublas_gemm  — torch batched GEMM via manual grouping (parallel)

All backends accept the same inputs and return (output, timing_dict).
"""
import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from clear_moe.dispatch.bucketize import BucketPlan, bucketize, apply_bucket_plan, invert_bucket_plan
from clear_moe.dispatch.scatter_gather import padded_scatter, padding_free_scatter
from clear_moe.dispatch.profiler import DispatchProfiler, DispatchTiming
from clear_moe.dispatch.triton_kernels import triton_fused_dispatch, HAS_TRITON

logger = logging.getLogger(__name__)

BACKENDS = ['naive', 'grouped', 'stream', 'stream_fine', 'triton', 'cublas_gemm']


class MoEExecutor:
    """Pluggable MoE dispatch executor.

    Args:
        backend:      one of BACKENDS
        num_experts:  E
        bucket_mode:  token bucketing strategy ('none', 'expert_sort', 'group_sort', 'latent_bucket')
        capacity:     max tokens per expert for padded backends (None = dynamic)
        profile:      if True, track per-stage timing
        centroids:    (E, D) precomputed centroids for latent_bucket mode
    """

    def __init__(
        self,
        backend: str = 'grouped',
        num_experts: int = 4,
        bucket_mode: str = 'expert_sort',
        capacity: Optional[int] = None,
        profile: bool = False,
        centroids: Optional[torch.Tensor] = None,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. Choose from {BACKENDS}")
        self.backend = backend
        self.num_experts = num_experts
        self.bucket_mode = bucket_mode
        self.capacity = capacity
        self.profile = profile
        self.centroids = centroids
        self._profiler = DispatchProfiler() if profile else None

    def forward(
        self,
        x: torch.Tensor,
        router,
        experts: List[nn.Module],
        layer_idx: Optional[int] = None,
        device_map: Optional[dict] = None,
        load_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Run MoE dispatch.

        Args:
            x:          (..., D) input tokens
            router:     BaseRouter instance
            experts:    list of E expert modules
            layer_idx:  for path-aware routers
            device_map: for comm-aware routers
            load_stats: external load override

        Returns:
            output:      same shape as x
            timing_dict: stage timing in ms (empty if profile=False)
        """
        if self.profile:
            self._profiler.reset()

        orig_shape = x.shape
        if x.dim() == 3:
            B, S, D = x.shape
            x_flat = x.reshape(B * S, D)
        elif x.dim() == 2:
            x_flat = x
            B, S = x.shape[0], 1
            D = x.shape[1]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        N = x_flat.shape[0]

        # Route
        if self.profile:
            self._profiler._timers['router'].start()
        indices, weights, logits, aux = router(
            x, layer_idx=layer_idx, device_map=device_map, load_stats=load_stats
        )
        if self.profile:
            self._profiler._elapsed['router'] += self._profiler._timers['router'].stop()

        # Primary expert index per token for bucketing
        primary_idx = indices.reshape(N, -1)[:, 0]  # (N,)

        # Bucketize
        if self.profile:
            self._profiler._timers['bucket'].start()
        plan = bucketize(
            x_flat, primary_idx, self.num_experts,
            mode=self.bucket_mode, centroids=self.centroids
        )
        if self.profile:
            self._profiler._elapsed['bucket'] += self._profiler._timers['bucket'].stop()

        # Dispatch by backend
        if self.backend == 'naive':
            output = self._backend_naive(x_flat, plan, experts)
        elif self.backend == 'grouped':
            output = self._backend_grouped(x_flat, plan, experts)
        elif self.backend == 'stream':
            output = self._backend_stream(x_flat, plan, experts)
        elif self.backend == 'stream_fine':
            output, _ft = self._run_stream_fine(x, indices, weights, experts)
            output = output.reshape(orig_shape[0] * (orig_shape[1] if x.dim() == 3 else 1), -1)
        elif self.backend == 'triton':
            output = self._backend_triton(x_flat, plan, experts, weights)
        elif self.backend == 'cublas_gemm':
            output = self._backend_cublas(x_flat, plan, experts)

        # Apply routing weights (top-k weighted sum)
        top_k = indices.reshape(N, -1).shape[-1]
        if top_k > 1:
            # For top_k > 1: run each expert, combine weighted outputs
            # Simple approach: output already from primary expert, scale by weight
            w_primary = weights.reshape(N, -1)[:, 0].unsqueeze(-1)
            output = output * w_primary

        timing_dict = {}
        if self.profile:
            timing = self._profiler.get_timing()
            timing_dict = timing.to_dict()

        return output.reshape(orig_shape), timing_dict

    def _backend_naive(self, x, plan, experts):
        """Serial loop, padded — baseline."""
        prof = self._profiler
        if prof:
            prof._timers['dispatch'].start()
        sorted_x = apply_bucket_plan(x, plan)
        if prof:
            prof._elapsed['dispatch'] += prof._timers['dispatch'].stop()

        D = x.shape[-1]
        cap = self.capacity or int(plan.bucket_sizes.max().item())
        out = torch.zeros_like(x)
        for e, expert in enumerate(experts):
            offset = int(plan.bucket_offsets[e])
            size = int(plan.bucket_sizes[e])
            if size == 0:
                continue
            if prof:
                prof._timers['router'].start()  # reuse timer slot
            tokens_e = sorted_x[offset:offset + size]
            padded = tokens_e
            if size < cap:
                pad = torch.zeros(cap - size, D, device=x.device, dtype=x.dtype)
                padded = torch.cat([tokens_e, pad], dim=0)
            e_out = expert(padded)[:size]
            out[offset:offset + size] = e_out
            if prof:
                prof._expert_times.append(prof._timers['router'].stop())

        if prof:
            prof._timers['combine'].start()
        result = invert_bucket_plan(out, plan)
        if prof:
            prof._elapsed['combine'] += prof._timers['combine'].stop()
        return result

    def _backend_grouped(self, x, plan, experts):
        """No-padding contiguous grouped dispatch — serial."""
        if self._profiler:
            self._profiler._timers['dispatch'].start()
        sorted_x = apply_bucket_plan(x, plan)
        if self._profiler:
            self._profiler._elapsed['dispatch'] += self._profiler._timers['dispatch'].stop()

        sorted_out = torch.empty_like(sorted_x)
        for e, expert in enumerate(experts):
            offset = int(plan.bucket_offsets[e])
            size = int(plan.bucket_sizes[e])
            if size == 0:
                continue
            if self._profiler:
                t = time.perf_counter()
            sorted_out[offset:offset + size] = expert(sorted_x[offset:offset + size])
            if self._profiler:
                self._profiler._expert_times.append((time.perf_counter() - t) * 1000)

        if self._profiler:
            self._profiler._timers['combine'].start()
        result = invert_bucket_plan(sorted_out, plan)
        if self._profiler:
            self._profiler._elapsed['combine'] += self._profiler._timers['combine'].stop()
        return result

    def _backend_stream(self, x, plan, experts):
        """3-stream overlap: dispatch / expert / combine on separate CUDA streams."""
        if not torch.cuda.is_available():
            logger.warning("stream backend: CUDA not available, falling back to grouped")
            return self._backend_grouped(x, plan, experts)

        s_dispatch = torch.cuda.Stream()
        s_expert = torch.cuda.Stream()
        s_combine = torch.cuda.Stream()

        # Dispatch stream: scatter tokens
        with torch.cuda.stream(s_dispatch):
            if self._profiler:
                self._profiler._timers['dispatch'].start()
            sorted_x = apply_bucket_plan(x, plan)
            if self._profiler:
                self._profiler._elapsed['dispatch'] += self._profiler._timers['dispatch'].stop()

        # Expert stream: wait for dispatch, run experts
        with torch.cuda.stream(s_expert):
            s_expert.wait_stream(s_dispatch)
            sorted_out = torch.empty_like(sorted_x)
            for e, expert in enumerate(experts):
                offset = int(plan.bucket_offsets[e])
                size = int(plan.bucket_sizes[e])
                if size == 0:
                    continue
                if self._profiler:
                    evt_s = torch.cuda.Event(enable_timing=True)
                    evt_e = torch.cuda.Event(enable_timing=True)
                    evt_s.record(s_expert)
                sorted_out[offset:offset + size] = expert(sorted_x[offset:offset + size])
                if self._profiler:
                    evt_e.record(s_expert)
                    torch.cuda.synchronize()
                    self._profiler._expert_times.append(evt_s.elapsed_time(evt_e))

        # Combine stream: wait for expert, gather
        with torch.cuda.stream(s_combine):
            s_combine.wait_stream(s_expert)
            if self._profiler:
                self._profiler._timers['combine'].start()
            result = invert_bucket_plan(sorted_out, plan)
            if self._profiler:
                self._profiler._elapsed['combine'] += self._profiler._timers['combine'].stop()

        torch.cuda.synchronize()
        return result

    def _backend_triton(self, x, plan, experts, weights):
        """Triton fused scatter+GEMM+gather."""
        w_flat = weights.reshape(x.shape[0], -1)[:, 0]
        if self._profiler:
            self._profiler._timers['dispatch'].start()
        result = triton_fused_dispatch(x, plan, experts, w_flat)
        if self._profiler:
            self._profiler._elapsed['dispatch'] += self._profiler._timers['dispatch'].stop()
        return result

    def _backend_cublas(self, x, plan, experts):
        """Batched GEMM via torch.bmm over expert buckets."""
        if self._profiler:
            self._profiler._timers['dispatch'].start()
        sorted_x = apply_bucket_plan(x, plan)
        if self._profiler:
            self._profiler._elapsed['dispatch'] += self._profiler._timers['dispatch'].stop()

        sorted_out = torch.empty_like(sorted_x)

        # Check if experts are linear layers for batched GEMM
        if all(isinstance(e, nn.Linear) for e in experts):
            for e_idx, expert in enumerate(experts):
                offset = int(plan.bucket_offsets[e_idx])
                size = int(plan.bucket_sizes[e_idx])
                if size == 0:
                    continue
                # torch.mm for single expert (cuBLAS backend)
                tokens_e = sorted_x[offset:offset + size]
                sorted_out[offset:offset + size] = torch.mm(tokens_e, expert.weight.T)
                if expert.bias is not None:
                    sorted_out[offset:offset + size] += expert.bias
        else:
            for e_idx, expert in enumerate(experts):
                offset = int(plan.bucket_offsets[e_idx])
                size = int(plan.bucket_sizes[e_idx])
                if size == 0:
                    continue
                sorted_out[offset:offset + size] = expert(sorted_x[offset:offset + size])

        if self._profiler:
            self._profiler._timers['combine'].start()
        result = invert_bucket_plan(sorted_out, plan)
        if self._profiler:
            self._profiler._elapsed['combine'] += self._profiler._timers['combine'].stop()
        return result

    # ------------------------------------------------------------------
    # Convenience wrappers: take (x, indices, weights, experts) directly
    # bypassing the router, for direct testing and benchmarking.
    # ------------------------------------------------------------------

    def _run_grouped(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        experts: List[nn.Module],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Grouped dispatch on pre-computed indices (bypasses router call)."""
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        N = flat.shape[0]
        primary_idx = indices.reshape(N, -1)[:, 0]
        plan = bucketize(flat, primary_idx, self.num_experts, mode='expert_sort')
        t0 = time.perf_counter()
        out = self._backend_grouped(flat, plan, experts)
        t_ms = (time.perf_counter() - t0) * 1000
        timing: Dict[str, float] = {
            "t_total": t_ms, "t_router": 0.0, "t_bucket": 0.0,
            "t_dispatch": 0.0, "t_expert_max": t_ms, "t_combine": 0.0,
            "t_sync": 0.0, "t_bubble": 0.0,
        }
        return out.reshape(shape), timing

    def _run_stream_fine(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        experts: List[nn.Module],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Fine-grained compute-communication overlap (Comet arXiv:2502.19811 inspired).

        Each expert gets its own CUDA stream. Combine of expert[i] fires as
        soon as expert[i] records its done-event, overlapping with expert[i+1]
        compute. Finer-grained than the 3-stream backend which serialises all
        expert GEMMs on one stream before combining.

        On CPU: sequential fallback with the same timing interface.
        """
        t0 = time.perf_counter()
        shape = x.shape
        flat = x.reshape(-1, shape[-1])   # (N, D)
        N, D = flat.shape
        flat_idx = indices.reshape(N, -1)[:, 0]   # (N,) top-1 assignment
        flat_w = weights.reshape(N, -1)[:, 0]     # (N,)

        out = torch.zeros_like(flat)

        # Sort tokens by expert for contiguous memory access
        t_bucket_s = time.perf_counter()
        sorted_order = torch.argsort(flat_idx, stable=True)
        sorted_experts_idx = flat_idx[sorted_order]
        t_bucket = (time.perf_counter() - t_bucket_s) * 1000

        device = flat.device
        use_cuda = device.type == "cuda" and torch.cuda.is_available()

        if use_cuda:
            streams = [torch.cuda.Stream() for _ in range(self.num_experts)]
            done_events = [torch.cuda.Event() for _ in range(self.num_experts)]
        else:
            streams = [None] * self.num_experts
            done_events = [None] * self.num_experts

        expert_outputs: List[Optional[torch.Tensor]] = [None] * self.num_experts
        expert_tok_ids: List[Optional[torch.Tensor]] = [None] * self.num_experts
        expert_weights_list: List[Optional[torch.Tensor]] = [None] * self.num_experts
        t_expert_list: List[float] = []

        # Phase 1: launch all expert GEMMs, each on its own stream
        for e in range(self.num_experts):
            mask = (sorted_experts_idx == e).nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                expert_tok_ids[e] = torch.empty(0, dtype=torch.long, device=device)
                expert_weights_list[e] = torch.empty(0, device=device)
                continue
            tok_ids = sorted_order[mask]
            expert_tok_ids[e] = tok_ids
            expert_weights_list[e] = flat_w[tok_ids]
            tokens_e = flat[tok_ids]

            te0 = time.perf_counter()
            if use_cuda and streams[e] is not None:
                with torch.cuda.stream(streams[e]):
                    expert_outputs[e] = experts[e](tokens_e)
                    done_events[e].record(streams[e])
            else:
                expert_outputs[e] = experts[e](tokens_e)
            t_expert_list.append((time.perf_counter() - te0) * 1000)

        # Phase 2: fine-grained combine — wait per-expert event then scatter.
        # Overlaps expert[i+1] compute with combine of expert[i].
        t_combine_s = time.perf_counter()
        for e in range(self.num_experts):
            tok_ids = expert_tok_ids[e]
            if tok_ids is None or tok_ids.numel() == 0:
                continue
            if use_cuda and done_events[e] is not None:
                torch.cuda.current_stream().wait_event(done_events[e])
            w = expert_weights_list[e].unsqueeze(-1)
            out[tok_ids] += w * expert_outputs[e]
        t_combine = (time.perf_counter() - t_combine_s) * 1000

        if use_cuda:
            torch.cuda.synchronize()

        t_total = (time.perf_counter() - t0) * 1000
        timing: Dict[str, float] = {
            "t_router": 0.0,
            "t_bucket": t_bucket,
            "t_dispatch": 0.0,
            "t_expert_max": max(t_expert_list) if t_expert_list else 0.0,
            "t_combine": t_combine,
            "t_sync": 0.0,
            "t_bubble": 0.0,
            "t_total": t_total,
        }
        return out.reshape(shape), timing
