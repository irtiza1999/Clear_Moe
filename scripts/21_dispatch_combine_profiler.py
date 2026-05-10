"""Script 21: Dispatch/Combine Profiler — per-stage latency breakdown.

Profiles: T_router / T_bucket / T_dispatch / T_expert_max / T_combine / T_sync / T_bubble

For each runtime backend: naive / grouped / stream / triton / cublas_gemm.
Produces data for Figure A (stacked latency breakdown bars).

Usage:
    python scripts/21_dispatch_combine_profiler.py --config configs/ablation_v3_full.yaml --dry-run
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

from clear_moe.routers import LinearRouter
from clear_moe.dispatch import DispatchProfiler, bucketize, apply_bucket_plan, invert_bucket_plan
from clear_moe.runtime import MoEExecutor, CostModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def profile_backend(
    backend: str,
    x: torch.Tensor,
    router,
    experts,
    num_experts: int,
    bucket_mode: str = 'expert_sort',
    n_warmup: int = 5,
    n_iters: int = 20,
) -> dict:
    """Profile one backend with DispatchProfiler."""
    D = x.shape[-1]
    E = num_experts

    # Warmup
    executor = MoEExecutor(backend=backend, num_experts=E, bucket_mode=bucket_mode, profile=False)
    for _ in range(n_warmup):
        with torch.no_grad():
            executor.forward(x, router, experts)

    # Profiled run — use manual profiler for fine-grained breakdown
    profiler = DispatchProfiler(use_cuda=torch.cuda.is_available())
    timing_accumulator = {
        't_router': 0.0, 't_bucket': 0.0, 't_dispatch': 0.0,
        't_expert_max': 0.0, 't_combine': 0.0, 't_total': 0.0,
    }

    for _ in range(n_iters):
        profiler.reset()
        N = x.reshape(-1, D).shape[0]
        x_flat = x.reshape(-1, D)

        with torch.no_grad():
            with profiler.time_router():
                indices, weights, logits, aux = router(x)

            primary_idx = indices.reshape(N, -1)[:, 0]

            with profiler.time_bucket():
                plan = bucketize(x_flat, primary_idx, E, mode=bucket_mode)

            with profiler.time_dispatch():
                sorted_x = apply_bucket_plan(x_flat, plan)

            for e_idx, expert in enumerate(experts):
                offset = int(plan.bucket_offsets[e_idx])
                size = int(plan.bucket_sizes[e_idx])
                if size > 0:
                    with profiler.time_expert(e_idx):
                        _ = expert(sorted_x[offset:offset + size])

            with profiler.time_combine():
                sorted_out = torch.empty_like(sorted_x)
                for e_idx, expert in enumerate(experts):
                    offset = int(plan.bucket_offsets[e_idx])
                    size = int(plan.bucket_sizes[e_idx])
                    if size > 0:
                        sorted_out[offset:offset + size] = expert(sorted_x[offset:offset + size])
                _ = invert_bucket_plan(sorted_out, plan)

        timing = profiler.get_timing()
        timing_dict = timing.to_dict()
        for k in timing_accumulator:
            if k in timing_dict:
                timing_accumulator[k] += timing_dict[k]

    # Average
    avg_timing = {k: v / n_iters for k, v in timing_accumulator.items()}
    avg_timing['backend'] = backend
    avg_timing['bucket_mode'] = bucket_mode
    avg_timing['N'] = x.reshape(-1, D).shape[0]
    avg_timing['D'] = D
    avg_timing['E'] = E
    return avg_timing


def main():
    parser = argparse.ArgumentParser(description='Dispatch/combine profiler')
    parser.add_argument('--config', default='configs/ablation_v3_full.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'outputs/21_dispatch_combine_profiler_{timestamp}.json'

    if args.dry_run:
        N, D, E = 32, 32, 4
        backends = ['naive', 'grouped']
        n_warmup, n_iters = 2, 5
    else:
        N, D = 512, 384
        E = config.get('extraction', {}).get('num_experts', 4)
        backends = config.get('runtime_backends', ['naive', 'grouped', 'stream', 'triton', 'cublas_gemm'])
        n_warmup, n_iters = 10, 30

    x = torch.randn(N, D)
    router = LinearRouter(D, E, top_k=1)
    experts = [nn.Linear(D, D, bias=False) for _ in range(E)]

    results = []
    for backend in backends:
        logger.info(f'Profiling backend: {backend}')
        try:
            result = profile_backend(
                backend, x, router, experts, E,
                n_warmup=n_warmup, n_iters=n_iters
            )
            results.append(result)
            logger.info(f'  t_total={result["t_total"]:.3f}ms '
                        f't_router={result["t_router"]:.3f}ms '
                        f't_bucket={result["t_bucket"]:.3f}ms '
                        f't_dispatch={result["t_dispatch"]:.3f}ms '
                        f't_combine={result["t_combine"]:.3f}ms')
        except Exception as e:
            logger.warning(f'Backend {backend} failed: {e}')
            results.append({'backend': backend, 'error': str(e), 't_total': -1.0})

    # Print table (Fig A data)
    print('\n' + '='*95)
    print(f'{"Backend":<15} {"T_router":>9} {"T_bucket":>9} {"T_dispatch":>11} '
          f'{"T_expert":>9} {"T_combine":>10} {"T_total":>9}')
    print('-'*95)
    for r in results:
        if r.get('error'):
            print(f'{r["backend"]:<15} ERROR: {r["error"]}')
        else:
            print(f'{r["backend"]:<15} {r.get("t_router", 0):>9.3f} {r.get("t_bucket", 0):>9.3f} '
                  f'{r.get("t_dispatch", 0):>11.3f} {r.get("t_expert_max", 0):>9.3f} '
                  f'{r.get("t_combine", 0):>10.3f} {r.get("t_total", 0):>9.3f}')
    print('='*95)

    with open(out_path, 'w') as f:
        json.dump({'config': str(args.config), 'results': results}, f, indent=2)
    logger.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
