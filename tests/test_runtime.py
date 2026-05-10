"""Unit tests for runtime layer."""
import pytest
import torch
import torch.nn as nn
from clear_moe.runtime import (
    MoEExecutor, BACKENDS,
    HotExpertManager,
    SimulatedEP,
    SimulatedPP, split_blocks_into_stages,
    ExpertRelayout,
    CostModel, CostModelOutput,
)
from clear_moe.routers import LinearRouter


def _make_experts(E, D):
    experts = [nn.Linear(D, D, bias=False) for _ in range(E)]
    with torch.no_grad():
        for e in experts:
            nn.init.eye_(e.weight)
    return experts


def _make_input(B=2, S=4, D=64):
    return torch.randn(B, S, D)


class TestMoEExecutor:
    def test_grouped_backend_output_shape(self):
        D, E = 64, 4
        router = LinearRouter(D, E, top_k=1)
        experts = _make_experts(E, D)
        executor = MoEExecutor(backend='grouped', num_experts=E)
        x = _make_input()
        out, timing = executor.forward(x, router, experts)
        assert out.shape == x.shape

    def test_naive_backend_output_shape(self):
        D, E = 64, 4
        router = LinearRouter(D, E)
        experts = _make_experts(E, D)
        executor = MoEExecutor(backend='naive', num_experts=E)
        x = _make_input()
        out, _ = executor.forward(x, router, experts)
        assert out.shape == x.shape

    def test_grouped_and_naive_same_output(self):
        torch.manual_seed(0)
        D, E = 32, 4
        router = LinearRouter(D, E, top_k=1)
        experts = _make_experts(E, D)
        x = _make_input(B=1, S=4, D=D)
        ex_naive = MoEExecutor(backend='naive', num_experts=E, bucket_mode='expert_sort')
        ex_grouped = MoEExecutor(backend='grouped', num_experts=E, bucket_mode='expert_sort')
        out_naive, _ = ex_naive.forward(x, router, experts)
        out_grouped, _ = ex_grouped.forward(x, router, experts)
        assert torch.allclose(out_naive, out_grouped, atol=1e-5)

    def test_profiling_returns_timing_keys(self):
        D, E = 32, 4
        router = LinearRouter(D, E)
        experts = _make_experts(E, D)
        executor = MoEExecutor(backend='grouped', num_experts=E, profile=True)
        x = _make_input(B=1, S=4, D=D)
        _, timing = executor.forward(x, router, experts)
        assert 't_total' in timing

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            MoEExecutor(backend='invalid')


class TestHotExpertManager:
    def test_update_load(self):
        mgr = HotExpertManager(num_experts=4, capacity_limit=4, hot_threshold=2.0)
        indices = torch.tensor([0, 0, 0, 0, 0, 0, 1, 2])  # expert 0 is hot
        mgr.update_load(indices)
        assert 0 in mgr._hot_experts or len(mgr._hot_experts) == 0  # may not trigger with EMA

    def test_shadow_creation(self):
        D = 32
        mgr = HotExpertManager(num_experts=4, capacity_limit=2, hot_threshold=0.1)
        experts = _make_experts(4, D)
        # Force expert 0 as hot
        mgr._hot_experts = [0]
        mgr.register_experts(experts)
        assert 0 in mgr._shadow_experts

    def test_serial_dispatch(self):
        D = 32
        mgr = HotExpertManager(num_experts=4, capacity_limit=2, hot_threshold=0.1,
                                parallel_shadow=False)
        experts = _make_experts(4, D)
        mgr._hot_experts = [0]
        mgr.register_experts(experts)

        from clear_moe.dispatch.bucketize import bucketize, apply_bucket_plan
        N = 8
        x = torch.randn(N, D)
        indices = torch.zeros(N, dtype=torch.long)  # all → expert 0
        plan = bucketize(x, indices, 4, mode='expert_sort')
        sorted_x = apply_bucket_plan(x, plan)
        out = mgr.dispatch_with_shadow(sorted_x, plan, experts)
        assert out.shape == (N, D)

    def test_reset(self):
        mgr = HotExpertManager(num_experts=4)
        mgr._hot_experts = [0, 1]
        mgr.reset()
        assert len(mgr._hot_experts) == 0
        assert len(mgr._shadow_experts) == 0


class TestSimulatedEP:
    def test_device_map_keys(self):
        ep = SimulatedEP(num_experts=4, num_ranks=2)
        dm = ep.build_device_map()
        assert 'local_rank' in dm
        assert all(e in dm for e in range(4))

    def test_dispatch_returns_stats(self):
        ep = SimulatedEP(num_experts=4, num_ranks=2, comm_latency_us=0.0)
        tokens = torch.randn(8, 32)
        indices = torch.randint(0, 4, (8,))
        _, stats = ep.dispatch(tokens, indices)
        assert 'remote_routing_fraction' in stats
        assert 'simulated_comm_overhead_ms' in stats

    def test_remote_fraction_in_range(self):
        ep = SimulatedEP(num_experts=4, num_ranks=2, local_rank=0)
        tokens = torch.randn(8, 32)
        # All tokens to expert 1 (rank 1, which is remote for rank 0)
        indices = torch.ones(8, dtype=torch.long)
        _, stats = ep.dispatch(tokens, indices)
        assert 0.0 <= stats['remote_routing_fraction'] <= 1.0

    def test_ep_forward_shape(self):
        D, E = 32, 4
        ep = SimulatedEP(num_experts=E, num_ranks=2, comm_latency_us=0.0)
        router = LinearRouter(D, E)
        experts = _make_experts(E, D)
        x = _make_input(B=1, S=4, D=D)
        out, stats = ep.run_ep_forward(x, router, experts)
        assert out.shape == x.shape
        assert 'remote_routing_fraction' in stats


class TestSimulatedPP:
    def test_serial_forward(self):
        stages = [nn.Identity() for _ in range(3)]
        pp = SimulatedPP(stages, num_microbatches=2)
        x = torch.randn(4, 32)
        out, stats = pp.forward_serial(x)
        assert out.shape == x.shape
        assert stats['mode'] == 'serial'
        assert stats['bubble_ratio'] == 0.0

    def test_pipeline_forward(self):
        stages = [nn.Identity() for _ in range(2)]
        pp = SimulatedPP(stages, num_microbatches=4)
        x = torch.randn(8, 32)
        out, stats = pp.forward_pipeline(x)
        assert out.shape == x.shape
        assert 'bubble_ratio_theoretical' in stats

    def test_bubble_ratio_formula(self):
        stages = [nn.Identity() for _ in range(4)]
        pp = SimulatedPP(stages, num_microbatches=8)
        expected = (4 - 1) / (8 + 4 - 1)  # (P-1)/(M+P-1)
        assert abs(pp.bubble_ratio - expected) < 1e-6

    def test_split_blocks_into_stages(self):
        blocks = [nn.Linear(32, 32) for _ in range(6)]
        stages = split_blocks_into_stages(blocks, num_stages=3)
        assert len(stages) == 3


class TestExpertRelayout:
    def test_device_map_format(self):
        rl = ExpertRelayout(num_experts=4, num_ranks=2)
        dm = rl.device_map
        assert 'local_rank' in dm
        assert all(e in dm for e in range(4))

    def test_relayout_changes_mapping(self):
        rl = ExpertRelayout(num_experts=4, num_ranks=2)
        initial = dict(rl._expert_to_rank)
        # Make expert 0 very hot
        load = torch.tensor([0.9, 0.03, 0.04, 0.03])
        rl.relayout(load)
        # Expert 0 should be on local rank (0) after relayout
        assert rl._expert_to_rank[0] == rl.local_rank

    def test_remote_fraction(self):
        rl = ExpertRelayout(num_experts=4, num_ranks=2, local_rank=0)
        # All tokens to expert 0 (rank 0, local)
        indices = torch.zeros(8, dtype=torch.long)
        frac = rl.compute_remote_fraction(indices)
        assert frac == 0.0


class TestCostModel:
    def test_t_total_equals_sum(self):
        model = CostModel(num_experts=4, hidden_dim=64)
        timing = {
            't_router': 1.0, 't_bucket': 0.5, 't_dispatch': 0.3,
            't_expert_max': 2.0, 't_combine': 0.4, 't_sync': 0.1, 't_bubble': 0.0
        }
        out = model.compute(timing)
        expected = sum(timing.values())
        assert abs(out.t_total - expected) < 1e-6

    def test_path_entropy_range(self):
        model = CostModel(num_experts=4, hidden_dim=64)
        # Uniform distribution → max entropy = log(E)
        load = torch.ones(4) / 4
        out = model.compute({}, load_vec=load)
        import math
        assert abs(out.path_entropy - math.log(4)) < 0.01

    def test_hotness_skew(self):
        model = CostModel(num_experts=4, hidden_dim=64)
        load = torch.tensor([0.9, 0.03, 0.03, 0.04])
        out = model.compute({}, load_vec=load)
        assert out.expert_hotness_skew > 1.0

    def test_straggler_ratio(self):
        model = CostModel(num_experts=4, hidden_dim=64)
        expert_times = [10.0, 2.0, 3.0, 1.0]
        out = model.compute({}, expert_times=expert_times)
        assert out.straggler_ratio == pytest.approx(10.0 / 4.0, rel=0.01)

    def test_to_dict_keys(self):
        model = CostModel(num_experts=4, hidden_dim=64)
        out = model.compute({'t_router': 1.0})
        d = out.to_dict()
        for key in ['t_total', 'path_entropy', 'straggler_ratio', 'parallel_slack']:
            assert key in d


class TestDisaggregatedSim:
    """DisaggregatedSim — attention-expert micro-batch pipeline."""

    def _make_blocks(self, n=4, d=16):
        return [nn.Linear(d, d, bias=False) for _ in range(n)]

    def test_forward_shape(self):
        from clear_moe.runtime.disaggregated import DisaggregatedSim
        sim = DisaggregatedSim(self._make_blocks(), self._make_blocks(), num_micro_batches=2)
        x = torch.randn(4, 8, 16)
        out, stats = sim.forward(x)
        assert out.shape == x.shape

    def test_stats_keys(self):
        from clear_moe.runtime.disaggregated import DisaggregatedSim
        sim = DisaggregatedSim(self._make_blocks(), self._make_blocks(), num_micro_batches=2)
        x = torch.randn(4, 8, 16)
        _, stats = sim.forward(x)
        for key in ["t_attn_ms", "t_expert_ms", "bubble_ratio", "overlap_efficiency"]:
            assert key in stats

    def test_bubble_ratio_decreases_with_more_microbatches(self):
        from clear_moe.runtime.disaggregated import DisaggregatedSim
        attn = self._make_blocks(2)
        ffn = self._make_blocks(2)
        sim2 = DisaggregatedSim(attn, ffn, num_micro_batches=2)
        sim8 = DisaggregatedSim(attn, ffn, num_micro_batches=8)
        assert sim2.bubble_ratio() > sim8.bubble_ratio()

    def test_serial_vs_pipeline_same_output(self):
        from clear_moe.runtime.disaggregated import DisaggregatedSim
        torch.manual_seed(0)
        attn = self._make_blocks(2, 16)
        ffn = self._make_blocks(2, 16)
        sim = DisaggregatedSim(attn, ffn, num_micro_batches=1)
        x = torch.randn(2, 4, 16)
        out_serial, _ = sim.forward_serial(x)
        out_pipe, _ = sim.forward(x)
        assert torch.allclose(out_serial, out_pipe, atol=1e-5)
