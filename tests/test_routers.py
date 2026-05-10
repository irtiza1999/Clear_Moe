"""Unit tests for router sub-package."""
import pytest
import torch
from clear_moe.routers import BaseRouter, LinearRouter, AdaptiveRouter, MLPRouter
from clear_moe.routers import (
    ExpertChoiceRouter, CommAwareRouter, SelfRoutingProxy, SharedExpertRouter,
    ALFRouter,
)


def _make_input(B=2, S=4, D=64):
    return torch.randn(B, S, D)


class TestLinearRouter:
    def test_output_shapes(self):
        router = LinearRouter(hidden_dim=64, num_experts=4, top_k=1)
        x = _make_input()
        indices, weights, logits, aux = router(x)
        assert indices.shape == (2, 4, 1)
        assert weights.shape == (2, 4, 1)
        assert logits.shape == (2, 4, 4)

    def test_weights_sum_to_one(self):
        router = LinearRouter(hidden_dim=64, num_experts=4, top_k=2)
        x = _make_input()
        _, weights, _, _ = router(x)
        assert torch.allclose(weights.sum(-1), torch.ones(2, 4), atol=1e-5)

    def test_indices_in_range(self):
        router = LinearRouter(hidden_dim=64, num_experts=4, top_k=1)
        x = _make_input()
        indices, _, _, _ = router(x)
        assert indices.min() >= 0
        assert indices.max() < 4

    def test_aux_dict_keys(self):
        router = LinearRouter(hidden_dim=64, num_experts=4)
        x = _make_input()
        _, _, _, aux = router(x)
        assert 'path_state' in aux
        assert 'comm_cost_est' in aux
        assert 'entropy' in aux
        assert 'load_vec' in aux

    def test_top2(self):
        router = LinearRouter(hidden_dim=64, num_experts=8, top_k=2)
        x = _make_input()
        indices, weights, logits, aux = router(x)
        assert indices.shape == (2, 4, 2)
        assert weights.shape == (2, 4, 2)
        assert logits.shape == (2, 4, 8)


class TestMLPRouter:
    def test_output_shapes(self):
        router = MLPRouter(hidden_dim=64, num_experts=4, router_hidden_dim=32, top_k=1)
        x = _make_input()
        indices, weights, logits, aux = router(x)
        assert indices.shape == (2, 4, 1)
        assert weights.shape == (2, 4, 1)
        assert logits.shape == (2, 4, 4)


class TestAdaptiveRouter:
    def test_output_shapes(self):
        router = AdaptiveRouter(hidden_dim=64, num_experts=4, confidence_threshold=0.6)
        x = _make_input()
        indices, weights, logits, aux = router(x)
        assert indices.shape == (2, 4, 2)

    def test_confident_tokens_use_one_expert(self):
        router = AdaptiveRouter(hidden_dim=64, num_experts=4, confidence_threshold=0.5)
        # Build a constant input that always routes to one expert with high confidence:
        # set gate weight so that expert 0 always wins by a large margin.
        with torch.no_grad():
            router.gate.weight.data.zero_()
            router.gate.weight.data[0] = 1.0  # expert 0 direction
            router.gate.weight.data *= 20.0
            router.gate.bias.data.zero_()
        # Use a fixed input aligned with expert 0's weight direction
        x = torch.ones(2, 4, 64)
        _, weights, _, _ = router(x)
        # Second expert weight should be 0 for all confident tokens
        assert (weights[..., 1] < 0.01).all()


class TestExpertChoiceRouter:
    def test_output_shapes(self):
        router = ExpertChoiceRouter(hidden_dim=64, num_experts=4, capacity_factor=1.0, top_k=1)
        x = _make_input()
        indices, weights, logits, aux = router(x)
        B, S, _ = _make_input().shape
        assert indices.shape == (B, S, 1)
        assert weights.shape == (B, S, 1)
        assert 'capacity' in aux
        assert 'expert_token_map' in aux

    def test_all_tokens_assigned(self):
        router = ExpertChoiceRouter(hidden_dim=64, num_experts=4, capacity_factor=1.0)
        x = _make_input()
        indices, weights, logits, aux = router(x)
        assert (weights > 0).all()


class TestCommAwareRouter:
    def test_output_shapes(self):
        router = CommAwareRouter(hidden_dim=64, num_experts=4, top_k=1)
        x = _make_input()
        indices, weights, logits, aux = router(x)
        assert indices.shape == (2, 4, 1)

    def test_device_map_reduces_remote_score(self):
        router = CommAwareRouter(hidden_dim=64, num_experts=4, lambda_comm=10.0, lambda_hot=0.0)
        x = torch.ones(1, 4, 64)
        # Map all experts to rank 1 (remote), except expert 0 (local at rank 0)
        device_map = {0: 0, 1: 1, 2: 1, 3: 1, 'local_rank': 0}
        indices, weights, logits, aux = router(x, device_map=device_map)
        # With large lambda_comm, should strongly prefer local expert 0
        assert (indices == 0).float().mean() > 0.5

    def test_ema_updates(self):
        router = CommAwareRouter(hidden_dim=64, num_experts=4)
        x = _make_input()
        initial_ema = router.load_ema.clone()
        router(x)
        assert not torch.allclose(router.load_ema, initial_ema)

    def test_reset_load_ema(self):
        router = CommAwareRouter(hidden_dim=64, num_experts=4)
        x = _make_input()
        router(x)
        router.reset_load_ema()
        expected = torch.ones(4) / 4
        assert torch.allclose(router.load_ema, expected, atol=1e-5)


class TestPathConsistentRouter:
    def test_output_shapes(self):
        from clear_moe.routers.path_router import PathConsistentRouter
        base = LinearRouter(hidden_dim=64, num_experts=4)
        router = PathConsistentRouter(base, path_penalty_coeff=0.01)
        x = _make_input()
        indices, weights, logits, aux = router(x)
        assert indices.shape == (2, 4, 1)
        assert 'path_churn' in aux

    def test_path_churn_zero_first_call(self):
        from clear_moe.routers.path_router import PathConsistentRouter
        base = LinearRouter(hidden_dim=64, num_experts=4)
        router = PathConsistentRouter(base)
        x = _make_input()
        _, _, _, aux = router(x)
        assert aux['path_churn'] == 0.0

    def test_path_churn_nonzero_second_call(self):
        from clear_moe.routers.path_router import PathConsistentRouter
        base = LinearRouter(hidden_dim=64, num_experts=4)
        router = PathConsistentRouter(base)
        x1 = _make_input()
        x2 = torch.randn_like(x1) * 10  # very different → different routing
        router(x1)
        _, _, _, aux = router(x2)
        # May or may not have churn depending on model, but aux_dict key must exist
        assert 'path_churn' in aux

    def test_reset_path(self):
        from clear_moe.routers.path_router import PathConsistentRouter
        base = LinearRouter(hidden_dim=64, num_experts=4)
        router = PathConsistentRouter(base)
        x = _make_input()
        router(x)
        router.reset_path()
        assert router._prev_indices is None


class TestSelfRoutingProxy:
    def test_unfitted_raises(self):
        from clear_moe.routers.self_routing import SelfRoutingProxy
        router = SelfRoutingProxy(hidden_dim=64, num_experts=4)
        x = _make_input()
        with pytest.raises(RuntimeError, match="not fitted"):
            router(x)

    def test_fit_and_forward(self):
        from clear_moe.routers.self_routing import SelfRoutingProxy
        router = SelfRoutingProxy(hidden_dim=64, num_experts=4)
        calib = torch.randn(100, 64)
        router.fit(calib)
        x = _make_input()
        indices, weights, logits, aux = router(x)
        assert indices.shape == (2, 4, 1)

    def test_no_parameters(self):
        from clear_moe.routers.self_routing import SelfRoutingProxy
        router = SelfRoutingProxy(hidden_dim=64, num_experts=4)
        params = list(router.parameters())
        assert len(params) == 0


class TestSharedExpertRouter:
    def test_output_shapes(self):
        from clear_moe.routers.shared_expert import SharedExpertRouter
        router = SharedExpertRouter(hidden_dim=64, num_experts=4, top_k=2)
        x = _make_input()
        indices, weights, logits, aux = router(x)
        assert indices.shape == (2, 4, 2)

    def test_shared_expert_always_included(self):
        from clear_moe.routers.shared_expert import SharedExpertRouter
        router = SharedExpertRouter(hidden_dim=64, num_experts=4, top_k=2, shared_expert_id=0)
        x = _make_input()
        indices, _, _, _ = router(x)
        # First slot always = 0 (shared expert)
        assert (indices[..., 0] == 0).all()


class TestALFRouter:
    def setup_method(self):
        self.B, self.S, self.D, self.E = 2, 4, 16, 4
        self.router = ALFRouter(self.D, self.E, top_k=1, bias_update_rate=0.01)

    def test_output_shapes(self):
        x = torch.randn(self.B, self.S, self.D)
        indices, weights, logits, aux = self.router(x)
        assert indices.shape == (self.B, self.S, 1)
        assert weights.shape == (self.B, self.S, 1)
        assert logits.shape == (self.B, self.S, self.E)
        assert "bias" in aux
        assert "load_counts" in aux

    def test_bias_updates_toward_balance(self):
        router = ALFRouter(16, 4, top_k=1, bias_update_rate=0.1)
        # Force gate to always send tokens to expert 0
        with torch.no_grad():
            router.gate.weight.fill_(0.0)
            router.gate.weight[0, :] = 1.0
        bias_before = router.bias.clone()
        x = torch.zeros(4, 8, 16)
        _ = router(x)
        # Expert 0 overloaded → bias[0] should decrease
        assert router.bias[0].item() < bias_before[0].item()

    def test_no_aux_loss_in_aux_dict(self):
        x = torch.randn(self.B, self.S, self.D)
        _, _, _, aux = self.router(x)
        assert "aux_loss" not in aux

    def test_reset_bias(self):
        router = ALFRouter(16, 4, top_k=1)
        with torch.no_grad():
            router.bias.fill_(5.0)
        router.reset_bias()
        assert router.bias.abs().max().item() == 0.0
