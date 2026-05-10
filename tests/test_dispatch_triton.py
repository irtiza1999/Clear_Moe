"""Tests for Triton kernels and DispatchProfiler.

Triton kernel tests use CPU fallbacks when CUDA is unavailable.
Correctness: Triton output must match PyTorch reference within 1e-4.
"""
import pytest
import torch
import torch.nn as nn
from clear_moe.dispatch import (
    BucketPlan, bucketize, apply_bucket_plan, invert_bucket_plan,
    HAS_TRITON,
    triton_scatter, triton_grouped_gemm, triton_gather, triton_fused_dispatch,
    DispatchProfiler, DispatchTiming,
)
from clear_moe.dispatch.scatter_gather import padding_free_scatter


TOL = 1e-4


class TestTritonScatter:
    def test_output_shape(self):
        N, D, E = 8, 16, 4
        x = torch.randn(N, D)
        indices = torch.randint(0, E, (N,))
        plan = bucketize(x, indices, E)
        out = triton_scatter(x, plan)
        assert out.shape == (N, D)

    def test_matches_pytorch_scatter(self):
        N, D, E = 8, 32, 4
        torch.manual_seed(0)
        x = torch.randn(N, D)
        indices = torch.randint(0, E, (N,))
        plan = bucketize(x, indices, E)
        out_triton = triton_scatter(x, plan)
        out_pytorch = apply_bucket_plan(x, plan)
        assert torch.allclose(out_triton, out_pytorch, atol=TOL)

    def test_round_trip_with_gather(self):
        N, D, E = 16, 32, 4
        torch.manual_seed(1)
        x = torch.randn(N, D)
        indices = torch.randint(0, E, (N,))
        plan = bucketize(x, indices, E)
        sorted_x = triton_scatter(x, plan)
        weights = torch.ones(N)
        out = triton_gather(sorted_x, plan, weights)
        assert torch.allclose(out, x, atol=TOL), "Round-trip failed (scatter+gather != identity)"


class TestTritonGroupedGemm:
    def test_output_shape(self):
        N, D_in, D_out, E = 16, 32, 64, 4
        torch.manual_seed(2)
        x = torch.randn(N, D_in)
        indices = torch.randint(0, E, (N,))
        plan = bucketize(x, indices, E)
        sorted_x = apply_bucket_plan(x, plan)
        W = torch.randn(E, D_in, D_out)
        out = triton_grouped_gemm(sorted_x, W, plan)
        assert out.shape == (N, D_out)

    def test_matches_pytorch_matmul(self):
        N, D_in, D_out, E = 16, 32, 32, 4
        torch.manual_seed(3)
        x = torch.randn(N, D_in)
        indices = torch.randint(0, E, (N,))
        plan = bucketize(x, indices, E)
        sorted_x = apply_bucket_plan(x, plan)
        W = torch.randn(E, D_in, D_out) * 0.1

        # PyTorch reference
        out_ref = torch.zeros(N, D_out)
        for e in range(E):
            offset = int(plan.bucket_offsets[e])
            size = int(plan.bucket_sizes[e])
            if size > 0:
                out_ref[offset:offset + size] = sorted_x[offset:offset + size] @ W[e]

        out_triton = triton_grouped_gemm(sorted_x, W, plan)
        assert torch.allclose(out_triton, out_ref, atol=TOL), \
            f"Max diff: {(out_triton - out_ref).abs().max()}"


class TestTritonFusedDispatch:
    def test_output_shape(self):
        N, D, E = 16, 32, 4
        torch.manual_seed(4)
        x = torch.randn(N, D)
        indices = torch.randint(0, E, (N,))
        experts = [nn.Linear(D, D, bias=False) for _ in range(E)]
        with torch.no_grad():
            for e in experts:
                nn.init.eye_(e.weight)
        plan = bucketize(x, indices, E)
        weights = torch.ones(N)
        out = triton_fused_dispatch(x, plan, experts, weights)
        assert out.shape == (N, D)

    def test_matches_sequential_dispatch(self):
        N, D, E = 8, 16, 4
        torch.manual_seed(5)
        x = torch.randn(N, D)
        indices = torch.randint(0, E, (N,))
        experts = [nn.Linear(D, D, bias=False) for _ in range(E)]
        with torch.no_grad():
            for e in experts:
                nn.init.eye_(e.weight)
        plan = bucketize(x, indices, E)
        weights = torch.ones(N)

        out_triton = triton_fused_dispatch(x, plan, experts, weights)
        out_ref = padding_free_scatter(x, plan, experts)
        assert torch.allclose(out_triton, out_ref, atol=TOL * 10), \
            f"Max diff: {(out_triton - out_ref).abs().max()}"


class TestDispatchProfiler:
    def test_timing_keys(self):
        profiler = DispatchProfiler(use_cuda=False)
        with profiler.time_router():
            _ = torch.randn(4, 4) @ torch.randn(4, 4)
        with profiler.time_bucket():
            _ = torch.randn(4, 4)
        timing = profiler.get_timing()
        d = timing.to_dict()
        for key in ['t_router', 't_bucket', 't_dispatch', 't_expert_max',
                    't_combine', 't_sync', 't_bubble', 't_total']:
            assert key in d, f"Missing key: {key}"

    def test_total_equals_sum(self):
        profiler = DispatchProfiler(use_cuda=False)
        with profiler.time_router():
            time_work = sum(range(1000))
        with profiler.time_bucket():
            time_work = sum(range(1000))
        timing = profiler.get_timing()
        expected = timing.t_router + timing.t_bucket
        assert abs(timing.t_total - expected) < 1e-9

    def test_expert_time_max(self):
        profiler = DispatchProfiler(use_cuda=False)
        profiler.record_expert_time(5.0)
        profiler.record_expert_time(10.0)
        profiler.record_expert_time(3.0)
        timing = profiler.get_timing()
        assert timing.t_expert_max == 10.0

    def test_reset(self):
        profiler = DispatchProfiler(use_cuda=False)
        with profiler.time_router():
            _ = torch.randn(4, 4)
        profiler.reset()
        timing = profiler.get_timing()
        assert timing.t_router == 0.0
