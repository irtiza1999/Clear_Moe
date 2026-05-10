"""Script 22: Hybrid Parallel Runner — EP × PP × stream Pareto frontier.

Tests all parallel mode combinations:
    serial / ep2 / ep4 / pp2 / ep2_pp2 / ep2_stream

Measures: throughput (tokens/sec), latency (ms), remote_routing_fraction.
Produces data for Figure E (Pareto frontier: throughput vs accuracy retention).

Usage:
    python scripts/22_hybrid_parallel_runner.py --config configs/ablation_v3_full.yaml --dry-run
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from clear_moe.routers import LinearRouter
from clear_moe.runtime import (
    MoEExecutor, SimulatedEP, SimulatedPP,
    split_blocks_into_stages, CostModel
)
from clear_moe.dispatch.bucketize import bucketize

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


PARALLEL_MODES = ['serial', 'ep2', 'ep4', 'pp2', 'ep2_pp2', 'ep2_stream']


def run_serial(x, router, experts, E, n_iters) -> dict:
    executor = MoEExecutor(backend='grouped', num_experts=E)
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            out, _ = executor.forward(x, router, experts)
            times.append((time.perf_counter() - t0) * 1000)
    N = x.reshape(-1, x.shape[-1]).shape[0]
    mean_t = sum(times) / len(times)
    return {'mode': 'serial', 'mean_latency_ms': mean_t, 'throughput_tok_per_ms': N / mean_t,
            'remote_routing_fraction': 0.0, 'bubble_ratio': 0.0}


def run_ep(x, router, experts, E, num_ranks, backend, comm_latency_us, n_iters) -> dict:
    ep = SimulatedEP(num_experts=E, num_ranks=num_ranks, comm_latency_us=comm_latency_us)
    times = []
    remote_fracs = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            out, stats = ep.run_ep_forward(x, router, experts)
            times.append((time.perf_counter() - t0) * 1000)
            remote_fracs.append(stats['remote_routing_fraction'])
    N = x.reshape(-1, x.shape[-1]).shape[0]
    mean_t = sum(times) / len(times)
    return {
        'mode': f'ep{num_ranks}' if backend == 'grouped' else f'ep{num_ranks}_stream',
        'mean_latency_ms': mean_t,
        'throughput_tok_per_ms': N / mean_t,
        'remote_routing_fraction': sum(remote_fracs) / len(remote_fracs),
        'bubble_ratio': 0.0,
    }


def run_pp(x_batched: List, stages, num_microbatches, n_iters) -> dict:
    pp = SimulatedPP(stages, num_microbatches=num_microbatches)
    times = []
    bubble_ratios = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            out, stats = pp.forward_pipeline(x_batched)
            times.append((time.perf_counter() - t0) * 1000)
            bubble_ratios.append(stats['bubble_ratio_theoretical'])
    mean_t = sum(times) / len(times)
    N = x_batched.shape[0]
    return {
        'mode': f'pp{pp.num_stages}',
        'mean_latency_ms': mean_t,
        'throughput_tok_per_ms': N / mean_t,
        'remote_routing_fraction': 0.0,
        'bubble_ratio': bubble_ratios[0] if bubble_ratios else 0.0,
    }


def run_ep_pp(x, x_batched, router, experts, stages, E, ep_ranks, pp_stages, comm_latency_us, n_iters) -> dict:
    ep = SimulatedEP(num_experts=E, num_ranks=ep_ranks, comm_latency_us=comm_latency_us)
    pp = SimulatedPP(stages, num_microbatches=4)
    times = []
    remote_fracs = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            # EP dispatch + expert execution
            ep_out, ep_stats = ep.run_ep_forward(x, router, experts)
            # PP staging (simulated on top)
            _, pp_stats = pp.forward_pipeline(ep_out.reshape(x_batched.shape[0], -1, ep_out.shape[-1]) if ep_out.dim() < 3 else ep_out)
            times.append((time.perf_counter() - t0) * 1000)
            remote_fracs.append(ep_stats['remote_routing_fraction'])
    N = x.reshape(-1, x.shape[-1]).shape[0]
    mean_t = sum(times) / len(times)
    return {
        'mode': f'ep{ep_ranks}_pp{pp_stages}',
        'mean_latency_ms': mean_t,
        'throughput_tok_per_ms': N / mean_t,
        'remote_routing_fraction': sum(remote_fracs) / len(remote_fracs),
        'bubble_ratio': pp.bubble_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description='Hybrid parallel runner')
    parser.add_argument('--config', default='configs/ablation_v3_full.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'outputs/22_hybrid_parallel_runner_{timestamp}.json'

    if args.dry_run:
        B, S, D, E = 2, 4, 32, 4
        n_warmup, n_iters = 2, 5
        comm_latency_us = 0.0
        num_blocks = 4
    else:
        B, S = 2, 197
        D = 384
        E = config.get('extraction', {}).get('num_experts', 4)
        n_warmup, n_iters = 10, 30
        comm_latency_us = config.get('comm_latency_us', 100.0)
        num_blocks = 6

    x = torch.randn(B, S, D)
    router = LinearRouter(D, E, top_k=1)
    experts = [nn.Linear(D, D, bias=False) for _ in range(E)]
    with torch.no_grad():
        for e in experts:
            nn.init.eye_(e.weight)

    # Create stages for PP
    blocks = [nn.Identity() for _ in range(num_blocks)]
    stages_2 = split_blocks_into_stages(blocks, num_stages=2)
    stages_4 = split_blocks_into_stages(blocks[:4], num_stages=min(4, num_blocks))

    results = []

    # Warmup all
    executor = MoEExecutor(backend='grouped', num_experts=E)
    with torch.no_grad():
        for _ in range(n_warmup):
            executor.forward(x, router, experts)

    # Run all modes
    results.append(run_serial(x, router, experts, E, n_iters))

    for ep_ranks in [2, 4]:
        if ep_ranks <= E:
            res = run_ep(x, router, experts, E, ep_ranks, 'grouped', comm_latency_us, n_iters)
            results.append(res)

    x_batched = x.reshape(B * S, D)
    results.append(run_pp(x_batched, stages_2, num_microbatches=4, n_iters=n_iters))

    # EP2 + PP2
    if E >= 2:
        results.append(run_ep_pp(x, x_batched, router, experts, stages_2, E,
                                  ep_ranks=2, pp_stages=2,
                                  comm_latency_us=comm_latency_us, n_iters=n_iters))

    # EP2 + stream backend
    res_ep2_stream = run_ep(x, router, experts, E, 2, 'stream', comm_latency_us, n_iters)
    res_ep2_stream['mode'] = 'ep2_stream'
    results.append(res_ep2_stream)

    # Print Pareto table (Fig E data)
    print('\n' + '='*80)
    print(f'{"Mode":<18} {"Latency(ms)":>12} {"Tput(tok/ms)":>14} {"RemoteFrac":>11} {"Bubble":>8}')
    print('-'*80)
    for r in results:
        print(f'{r["mode"]:<18} {r["mean_latency_ms"]:>12.3f} {r["throughput_tok_per_ms"]:>14.2f} '
              f'{r["remote_routing_fraction"]:>11.3f} {r["bubble_ratio"]:>8.3f}')
    print('='*80)

    with open(out_path, 'w') as f:
        json.dump({'config': str(args.config), 'results': results}, f, indent=2)
    logger.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
