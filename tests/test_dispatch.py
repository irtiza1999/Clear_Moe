"""Unit tests for dispatch layer."""
import pytest
import torch
import torch.nn as nn
from clear_moe.dispatch import BucketPlan, bucketize, apply_bucket_plan, invert_bucket_plan
from clear_moe.dispatch import padded_scatter, padding_free_scatter


def _dummy_experts(E, D):
    """E identity-like experts for testing round-trips."""
    return [nn.Identity() for _ in range(E)]


class TestBucketize:
    def test_expert_sort_output_shapes(self):
        N, D, E = 8, 64, 4
        x = torch.randn(N, D)
        indices = torch.randint(0, E, (N,))
        plan = bucketize(x, indices, E, mode='expert_sort')
        assert plan.token_order.shape == (N,)
        assert plan.bucket_offsets.shape == (E,)
        assert plan.bucket_sizes.shape == (E,)
        assert plan.inverse_order.shape == (N,)

    def test_bucket_sizes_sum_to_N(self):
        N, D, E = 16, 32, 4
        x = torch.randn(N, D)
        indices = torch.randint(0, E, (N,))
        plan = bucketize(x, indices, E, mode='expert_sort')
        assert plan.bucket_sizes.sum().item() == N

    def test_round_trip_identity(self):
        N, D, E = 8, 16, 4
        x = torch.randn(N, D)
        indices = torch.randint(0, E, (N,))
        plan = bucketize(x, indices, E, mode='expert_sort')
        sorted_x = apply_bucket_plan(x, plan)
        restored = invert_bucket_plan(sorted_x, plan)
        assert torch.allclose(x, restored)

    def test_none_mode(self):
        N, D, E = 8, 16, 4
        x = torch.randn(N, D)
        indices = torch.randint(0, E, (N,))
        plan = bucketize(x, indices, E, mode='none')
        assert plan.token_order.tolist() == list(range(N))

    def test_group_sort(self):
        N, D, E = 8, 16, 4
        x = torch.randn(N, D)
        indices = torch.randint(0, E, (N,))
        plan = bucketize(x, indices, E, mode='group_sort', group_size=2)
        assert plan.bucket_sizes.sum().item() == N

    def test_latent_bucket(self):
        N, D, E = 8, 16, 4
        x = torch.randn(N, D)
        indices = torch.randint(0, E, (N,))
        centroids = torch.randn(E, D)
        plan = bucketize(x, indices, E, mode='latent_bucket', centroids=centroids)
        assert plan.bucket_sizes.sum().item() == N

    def test_3d_input(self):
        B, S, D, E = 2, 4, 32, 4
        x = torch.randn(B, S, D)
        indices = torch.randint(0, E, (B, S))
        plan = bucketize(x, indices, E, mode='expert_sort')
        assert plan.token_order.shape == (B * S,)

    def test_invalid_mode(self):
        x = torch.randn(8, 16)
        indices = torch.randint(0, 4, (8,))
        with pytest.raises(ValueError):
            bucketize(x, indices, 4, mode='invalid_mode')


class TestScatterGather:
    def test_padded_scatter_shape(self):
        N, D, E = 8, 16, 4
        x = torch.randn(N, D)
        indices = torch.randint(0, E, (N,))
        experts = [nn.Linear(D, D, bias=False) for _ in range(E)]
        with torch.no_grad():
            for e in experts:
                nn.init.eye_(e.weight)
        plan = bucketize(x, indices, E, mode='expert_sort')
        out = padded_scatter(x, plan, experts)
        assert out.shape == (N, D)

    def test_padding_free_scatter_shape(self):
        N, D, E = 8, 16, 4
        x = torch.randn(N, D)
        indices = torch.randint(0, E, (N,))
        experts = [nn.Linear(D, D, bias=False) for _ in range(E)]
        with torch.no_grad():
            for e in experts:
                nn.init.eye_(e.weight)
        plan = bucketize(x, indices, E, mode='expert_sort')
        out = padding_free_scatter(x, plan, experts)
        assert out.shape == (N, D)

    def test_padded_and_padding_free_same_output(self):
        N, D, E = 8, 16, 4
        torch.manual_seed(42)
        x = torch.randn(N, D)
        indices = torch.randint(0, E, (N,))
        experts = [nn.Linear(D, D, bias=False) for _ in range(E)]
        with torch.no_grad():
            for e in experts:
                nn.init.eye_(e.weight)
        plan = bucketize(x, indices, E, mode='expert_sort')
        out_padded = padded_scatter(x, plan, experts)
        out_free = padding_free_scatter(x, plan, experts)
        assert torch.allclose(out_padded, out_free, atol=1e-5)
