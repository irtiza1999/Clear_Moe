"""Script 18: Token Bucket Benchmark — compare 4 bucketization modes.

Measures T_bucket and T_dispatch for:
    none / expert_sort / group_sort / latent_bucket

at varying batch sizes and expert counts.

Usage:
    python scripts/18_token_bucket_benchmark.py --config configs/ablation_v3_full.yaml
    python scripts/18_token_bucket_benchmark.py --config configs/ablation_v3_full.yaml --dry-run
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from clear_moe.dispatch import bucketize, BucketPlan, apply_bucket_plan, invert_bucket_plan
from clear_moe.dispatch import DispatchProfiler

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


BUCKET_MODES = ['none', 'expert_sort', 'group_sort', 'latent_bucket']


def benchmark_bucket_mode(
    mode: str,
    x: torch.Tensor,
    indices: torch.Tensor,
    experts,
    num_experts: int,
    centroids=None,
    n_warmup: int = 5,
    n_iters: int = 20,
):
    """Benchmark one bucket mode: measure T_bucket + T_dispatch."""
    N, D = x.shape[0], x.shape[-1]

    # Warmup
    for _ in range(n_warmup):
        plan = bucketize(x.reshape(-1, D), indices.reshape(-1), num_experts,
                         mode=mode, centroids=centroids)
        _ = apply_bucket_plan(x.reshape(-1, D), plan)

    bucket_times = []
    dispatch_times = []

    for _ in range(n_iters):
        t0 = time.perf_counter()
        plan = bucketize(x.reshape(-1, D), indices.reshape(-1), num_experts,
                         mode=mode, centroids=centroids)
        bucket_times.append((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        sorted_x = apply_bucket_plan(x.reshape(-1, D), plan)
        # Run experts
        sorted_out = torch.empty_like(sorted_x)
        for e in range(num_experts):
            offset = int(plan.bucket_offsets[e])
            size = int(plan.bucket_sizes[e])
            if size > 0:
                sorted_out[offset:offset + size] = experts[e](sorted_x[offset:offset + size])
        _ = invert_bucket_plan(sorted_out, plan)
        dispatch_times.append((time.perf_counter() - t1) * 1000)

    return {
        'mode': mode,
        'N': N if x.dim() == 2 else x.shape[0] * x.shape[1],
        'D': D,
        'E': num_experts,
        't_bucket_mean_ms': sum(bucket_times) / len(bucket_times),
        't_dispatch_mean_ms': sum(dispatch_times) / len(dispatch_times),
        't_bucket_min_ms': min(bucket_times),
        't_dispatch_min_ms': min(dispatch_times),
    }


def main():
    parser = argparse.ArgumentParser(description='Token bucket benchmark')
    parser.add_argument('--config', default='configs/ablation_v3_full.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'outputs/18_token_bucket_benchmark_{timestamp}.json'

    if args.dry_run:
        batch_sizes = [16, 64]
        expert_counts = [4]
        D = 32
        n_warmup, n_iters = 2, 5
    else:
        batch_sizes = [32, 128, 512, 2048]
        expert_counts = config.get('expert_counts', [4, 8])
        D = 384
        n_warmup, n_iters = 10, 30

    results = []

    for E in expert_counts:
        for N in batch_sizes:
            x = torch.randn(N, D)
            indices = torch.randint(0, E, (N,))
            experts = [nn.Linear(D, D, bias=False) for _ in range(E)]

            # Precompute centroids for latent_bucket
            centroids = torch.randn(E, D)

            for mode in BUCKET_MODES:
                centroids_arg = centroids if mode == 'latent_bucket' else None
                result = benchmark_bucket_mode(
                    mode, x, indices, experts, E,
                    centroids=centroids_arg,
                    n_warmup=n_warmup, n_iters=n_iters,
                )
                results.append(result)
                logger.info(f'E={E} N={N} mode={mode:15s} '
                            f'bucket={result["t_bucket_mean_ms"]:.3f}ms '
                            f'dispatch={result["t_dispatch_mean_ms"]:.3f}ms')

    # Print table
    print('\n' + '='*80)
    print(f'{"Mode":<20} {"N":>6} {"E":>4} {"T_bucket(ms)":>14} {"T_dispatch(ms)":>16}')
    print('-'*80)
    for r in results:
        print(f'{r["mode"]:<20} {r["N"]:>6} {r["E"]:>4} '
              f'{r["t_bucket_mean_ms"]:>14.3f} {r["t_dispatch_mean_ms"]:>16.3f}')
    print('='*80)

    with open(out_path, 'w') as f:
        json.dump({'config': str(args.config), 'results': results}, f, indent=2)
    logger.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
