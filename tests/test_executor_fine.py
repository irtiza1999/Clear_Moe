"""Tests for stream_fine executor backend (fine-grained overlap, Comet-inspired)."""
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from clear_moe.runtime.executor import MoEExecutor, BACKENDS


def make_experts(num_experts, hidden_dim):
    return [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_experts)]


class TestStreamFineBackend:
    def setup_method(self):
        torch.manual_seed(42)
        self.E, self.D = 4, 32
        self.experts = make_experts(self.E, self.D)

    def test_backend_registered(self):
        assert "stream_fine" in BACKENDS

    def test_output_shape(self):
        exec_ = MoEExecutor(backend="stream_fine", num_experts=self.E)
        x = torch.randn(2, 8, self.D)
        indices = torch.randint(0, self.E, (2, 8, 1))
        weights = torch.ones(2, 8, 1)
        out, timing = exec_._run_stream_fine(x, indices, weights, self.experts)
        assert out.shape == x.shape

    def test_timing_keys_present(self):
        exec_ = MoEExecutor(backend="stream_fine", num_experts=self.E, profile=True)
        x = torch.randn(2, 8, self.D)
        indices = torch.randint(0, self.E, (2, 8, 1))
        weights = torch.ones(2, 8, 1)
        out, timing = exec_._run_stream_fine(x, indices, weights, self.experts)
        assert "t_bucket" in timing
        assert "t_expert_max" in timing
        assert "t_combine" in timing
        assert "t_total" in timing

    def test_stream_fine_same_output_as_grouped(self):
        torch.manual_seed(0)
        experts = make_experts(self.E, self.D)
        x = torch.randn(2, 8, self.D)
        # Deterministic balanced routing: round-robin
        indices = torch.arange(16).remainder(self.E).reshape(2, 8, 1)
        weights = torch.ones(2, 8, 1)

        exec_grouped = MoEExecutor(backend="grouped", num_experts=self.E)
        exec_fine = MoEExecutor(backend="stream_fine", num_experts=self.E)

        out_g, _ = exec_grouped._run_grouped(x, indices, weights, experts)
        out_f, _ = exec_fine._run_stream_fine(x, indices, weights, experts)
        assert torch.allclose(out_g, out_f, atol=1e-5), \
            f"max diff: {(out_g - out_f).abs().max().item()}"
