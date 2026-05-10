"""Script 19: Hot Expert Runtime — parallel vs serial shadow under load skew.

Sweeps:
    - hot_expert_policies: [none, overflow_dense, shadow_serial, shadow_parallel]
    - skew_levels: balanced, mild, moderate, severe

For each combination: measures throughput and quality (output MSE vs dense).

Usage:
    python scripts/19_hot_expert_runtime.py --config configs/ablation_v3_full.yaml --dry-run
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

from clear_moe.dispatch import bucketize, apply_bucket_plan, invert_bucket_plan
from clear_moe.runtime import HotExpertManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

POLICIES = ['none', 'overflow_dense', 'shadow_serial', 'shadow_parallel']

# Skew: (skew_name, hot_expert_fraction_of_tokens)
SKEW_LEVELS = [
    ('balanced',  0.25),
    ('mild',      0.50),
    ('moderate',  0.70),
    ('severe',    0.90),
]


def make_skewed_indices(N: int, E: int, hot_fraction: float) -> torch.Tensor:
    """Generate skewed expert assignment with hot_fraction going to expert 0."""
    n_hot = int(N * hot_fraction)
    n_rest = N - n_hot
    hot = torch.zeros(n_hot, dtype=torch.long)
    rest = torch.randint(1, E, (n_rest,), dtype=torch.long)
    idx = torch.cat([hot, rest])
    perm = torch.randperm(N)
    return idx[perm]


def run_policy(
    policy: str,
    tokens: torch.Tensor,
    indices: torch.Tensor,
    experts,
    E: int,
    capacity_limit: int,
    hot_threshold: float = 0.3,
) -> dict:
    """Run one dispatch policy and measure throughput."""
    N, D = tokens.shape

    t0 = time.perf_counter()

    if policy == 'none':
        # Drop overflow — route to expert as-is up to capacity_limit
        plan = bucketize(tokens, indices, E, mode='expert_sort')
        sorted_x = apply_bucket_plan(tokens, plan)
        sorted_out = torch.empty_like(sorted_x)
        for e in range(E):
            offset = int(plan.bucket_offsets[e])
            size = int(plan.bucket_sizes[e])
            if size == 0:
                continue
            actual = min(size, capacity_limit)
            sorted_out[offset:offset + actual] = experts[e](sorted_x[offset:offset + actual])
            if size > actual:
                # Copy input as-is for overflow (dropped)
                sorted_out[offset + actual:offset + size] = sorted_x[offset + actual:offset + size]
        output = invert_bucket_plan(sorted_out, plan)

    elif policy == 'overflow_dense':
        # Overflow tokens computed by dense fallback (all experts average)
        plan = bucketize(tokens, indices, E, mode='expert_sort')
        sorted_x = apply_bucket_plan(tokens, plan)
        sorted_out = torch.empty_like(sorted_x)
        avg_weights = sum(e.weight for e in experts) / E
        for e in range(E):
            offset = int(plan.bucket_offsets[e])
            size = int(plan.bucket_sizes[e])
            if size == 0:
                continue
            actual = min(size, capacity_limit)
            sorted_out[offset:offset + actual] = experts[e](sorted_x[offset:offset + actual])
            if size > actual:
                overflow = sorted_x[offset + actual:offset + size]
                sorted_out[offset + actual:offset + size] = torch.nn.functional.linear(
                    overflow, avg_weights)
        output = invert_bucket_plan(sorted_out, plan)

    elif policy in ('shadow_serial', 'shadow_parallel'):
        parallel = (policy == 'shadow_parallel') and torch.cuda.is_available()
        mgr = HotExpertManager(
            num_experts=E,
            capacity_limit=capacity_limit,
            hot_threshold=hot_threshold,
            parallel_shadow=parallel,
        )
        mgr.update_load(indices)
        mgr.register_experts(experts)

        plan = bucketize(tokens, indices, E, mode='expert_sort')
        sorted_x = apply_bucket_plan(tokens, plan)
        sorted_out = mgr.dispatch_with_shadow(sorted_x, plan, experts)
        output = invert_bucket_plan(sorted_out, plan)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    elapsed = (time.perf_counter() - t0) * 1000  # ms

    # Compute output MSE vs reference (all experts at full capacity)
    plan_ref = bucketize(tokens, indices, E, mode='expert_sort')
    sorted_ref = apply_bucket_plan(tokens, plan_ref)
    sorted_ref_out = torch.empty_like(sorted_ref)
    for e in range(E):
        offset = int(plan_ref.bucket_offsets[e])
        size = int(plan_ref.bucket_sizes[e])
        if size > 0:
            sorted_ref_out[offset:offset + size] = experts[e](sorted_ref[offset:offset + size])
    ref_output = invert_bucket_plan(sorted_ref_out, plan_ref)
    mse = float((output - ref_output).pow(2).mean().item())

    return {
        'policy': policy,
        'elapsed_ms': elapsed,
        'output_mse': mse,
        'throughput_tokens_per_ms': N / max(elapsed, 1e-6),
    }


def main():
    parser = argparse.ArgumentParser(description='Hot expert runtime benchmark')
    parser.add_argument('--config', default='configs/ablation_v3_full.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'outputs/19_hot_expert_runtime_{timestamp}.json'

    if args.dry_run:
        N, D, E = 32, 32, 4
        capacity_limit = 8
        hot_threshold = 0.3
        skew_levels = [('balanced', 0.25), ('severe', 0.90)]
        n_reps = 3
    else:
        N, D, E = 512, 384, config.get('extraction', {}).get('num_experts', 4)
        capacity_limit = config.get('hot_expert_capacity_limit', 16)
        hot_threshold = config.get('hot_expert_threshold', 2.0)
        skew_levels = SKEW_LEVELS
        n_reps = 20

    experts = [nn.Linear(D, D, bias=False) for _ in range(E)]
    with torch.no_grad():
        for e in experts:
            nn.init.eye_(e.weight)

    results = []

    for skew_name, hot_frac in skew_levels:
        logger.info(f'Skew: {skew_name} (hot_frac={hot_frac:.2f})')
        tokens = torch.randn(N, D)
        indices = make_skewed_indices(N, E, hot_frac)

        for policy in POLICIES:
            trial_results = []
            for _ in range(n_reps):
                r = run_policy(policy, tokens, indices, experts, E,
                               capacity_limit, hot_threshold)
                trial_results.append(r)
            mean_elapsed = sum(r['elapsed_ms'] for r in trial_results) / n_reps
            mean_mse = sum(r['output_mse'] for r in trial_results) / n_reps
            mean_tput = sum(r['throughput_tokens_per_ms'] for r in trial_results) / n_reps
            entry = {
                'skew': skew_name,
                'hot_fraction': hot_frac,
                'policy': policy,
                'mean_elapsed_ms': mean_elapsed,
                'mean_mse': mean_mse,
                'throughput_tokens_per_ms': mean_tput,
            }
            results.append(entry)
            logger.info(f'  {policy:20s} elapsed={mean_elapsed:.3f}ms mse={mean_mse:.4f}')

    # Print table
    print('\n' + '='*90)
    print(f'{"Skew":<12} {"Policy":<22} {"Elapsed(ms)":>12} {"MSE":>12} {"Tput(tok/ms)":>14}')
    print('-'*90)
    for r in results:
        print(f'{r["skew"]:<12} {r["policy"]:<22} {r["mean_elapsed_ms"]:>12.3f} '
              f'{r["mean_mse"]:>12.4f} {r["throughput_tokens_per_ms"]:>14.2f}')
    print('='*90)

    with open(out_path, 'w') as f:
        json.dump({'config': str(args.config), 'results': results}, f, indent=2)
    logger.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
