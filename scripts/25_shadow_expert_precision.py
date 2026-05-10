"""Script 25: Shadow Expert Precision Adaptation.

Overloaded expert runs BF16/FP16, lightly loaded experts stay FP32.
Measures: latency vs quality tradeoff.

Variants:
    fp32_uniform     — all experts FP32 (baseline)
    fp16_uniform     — all experts FP16
    adaptive_bf16    — hot expert BF16, cold FP32
    adaptive_parallel— BF16 shadow on s_shadow, FP32 primary on s_primary

Usage:
    python scripts/25_shadow_expert_precision.py --config configs/ablation_v3_full.yaml --dry-run
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

from clear_moe.dispatch.bucketize import bucketize, apply_bucket_plan, invert_bucket_plan
from clear_moe.runtime import HotExpertManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PRECISION_VARIANTS = ['fp32_uniform', 'fp16_uniform', 'adaptive_bf16', 'adaptive_parallel']


def dispatch_with_precision(
    variant: str,
    tokens: torch.Tensor,
    indices: torch.Tensor,
    experts_fp32: nn.ModuleList,
    hot_expert_id: int,
    capacity_limit: int,
    E: int,
) -> torch.Tensor:
    """Dispatch with given precision strategy."""
    N, D = tokens.shape

    if variant == 'fp32_uniform':
        dtype = torch.float32
        experts = experts_fp32
    elif variant == 'fp16_uniform':
        dtype = torch.float16
        experts = [nn.Linear(e.in_features, e.out_features, bias=False).to(dtype)
                   for e in experts_fp32]
        with torch.no_grad():
            for src, dst in zip(experts_fp32, experts):
                dst.weight.data.copy_(src.weight.data.to(dtype))
    else:
        dtype = torch.float32  # mixed precision handled inline
        experts = experts_fp32

    plan = bucketize(tokens.float(), indices, E, mode='expert_sort')
    sorted_x = apply_bucket_plan(tokens.float(), plan)
    sorted_out = torch.zeros_like(sorted_x)

    if variant in ('fp32_uniform', 'fp16_uniform'):
        for e_idx in range(E):
            offset = int(plan.bucket_offsets[e_idx])
            size = int(plan.bucket_sizes[e_idx])
            if size == 0:
                continue
            tok = sorted_x[offset:offset + size].to(dtype)
            out = experts[e_idx](tok)
            sorted_out[offset:offset + size] = out.float()

    elif variant == 'adaptive_bf16':
        # Hot expert uses BF16, others FP32
        for e_idx in range(E):
            offset = int(plan.bucket_offsets[e_idx])
            size = int(plan.bucket_sizes[e_idx])
            if size == 0:
                continue
            precision = torch.bfloat16 if e_idx == hot_expert_id else torch.float32
            try:
                tok = sorted_x[offset:offset + size].to(precision)
                out = experts[e_idx](tok)
                sorted_out[offset:offset + size] = out.float()
            except RuntimeError:
                # BF16 not supported on this hardware — fall back to FP16
                tok = sorted_x[offset:offset + size].to(torch.float16)
                # Create FP16 expert on-the-fly
                e_fp16 = nn.Linear(experts[e_idx].in_features, experts[e_idx].out_features, bias=False).to(torch.float16)
                with torch.no_grad():
                    e_fp16.weight.data.copy_(experts[e_idx].weight.data.to(torch.float16))
                out = e_fp16(tok)
                sorted_out[offset:offset + size] = out.float()

    elif variant == 'adaptive_parallel':
        # Hot expert: BF16 shadow on s_shadow; others: FP32 on s_primary
        s_primary = torch.cuda.Stream() if torch.cuda.is_available() else None
        s_shadow = torch.cuda.Stream() if torch.cuda.is_available() else None

        for e_idx in range(E):
            offset = int(plan.bucket_offsets[e_idx])
            size = int(plan.bucket_sizes[e_idx])
            if size == 0:
                continue
            tok = sorted_x[offset:offset + size]

            if e_idx == hot_expert_id and size > capacity_limit:
                primary_tok = tok[:capacity_limit]
                overflow_tok = tok[capacity_limit:]

                if s_primary is not None:
                    with torch.cuda.stream(s_primary):
                        sorted_out[offset:offset + capacity_limit] = experts[e_idx](primary_tok)
                    with torch.cuda.stream(s_shadow):
                        try:
                            overflow_bf16 = overflow_tok.to(torch.bfloat16)
                            e_bf16 = nn.Linear(experts[e_idx].in_features, experts[e_idx].out_features, bias=False).to(torch.bfloat16)
                            with torch.no_grad():
                                e_bf16.weight.data.copy_(experts[e_idx].weight.data.to(torch.bfloat16))
                            sorted_out[offset + capacity_limit:offset + size] = e_bf16(overflow_bf16).float()
                        except RuntimeError:
                            sorted_out[offset + capacity_limit:offset + size] = experts[e_idx](overflow_tok)
                    torch.cuda.synchronize()
                else:
                    sorted_out[offset:offset + capacity_limit] = experts[e_idx](primary_tok)
                    sorted_out[offset + capacity_limit:offset + size] = experts[e_idx](overflow_tok)
            else:
                sorted_out[offset:offset + size] = experts[e_idx](tok)

    return invert_bucket_plan(sorted_out, plan)


def main():
    parser = argparse.ArgumentParser(description='Shadow expert precision adaptation')
    parser.add_argument('--config', default='configs/ablation_v3_full.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'outputs/25_shadow_expert_precision_{timestamp}.json'

    if args.dry_run:
        N, D, E = 64, 32, 4
        capacity_limit = 8
        n_iters = 3
        hot_fraction = 0.7
    else:
        N, D = 512, 384
        E = config.get('extraction', {}).get('num_experts', 4)
        capacity_limit = config.get('hot_expert_capacity_limit', 16)
        n_iters = 20
        hot_fraction = 0.7

    # Make hot-skewed indices
    n_hot = int(N * hot_fraction)
    hot_expert_id = 0
    indices = torch.cat([
        torch.zeros(n_hot, dtype=torch.long),
        torch.randint(1, E, (N - n_hot,), dtype=torch.long),
    ])[torch.randperm(N)]

    tokens = torch.randn(N, D)
    experts_fp32 = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(E)])
    with torch.no_grad():
        for e in experts_fp32:
            nn.init.eye_(e.weight)

    # Reference output (FP32 uniform, full capacity)
    plan_ref = bucketize(tokens, indices, E, mode='expert_sort')
    sorted_ref = apply_bucket_plan(tokens, plan_ref)
    sorted_ref_out = torch.empty_like(sorted_ref)
    with torch.no_grad():
        for e_idx in range(E):
            offset = int(plan_ref.bucket_offsets[e_idx])
            size = int(plan_ref.bucket_sizes[e_idx])
            if size > 0:
                sorted_ref_out[offset:offset + size] = experts_fp32[e_idx](sorted_ref[offset:offset + size])
    ref_output = invert_bucket_plan(sorted_ref_out, plan_ref)

    results = []
    for variant in PRECISION_VARIANTS:
        trial_times = []
        trial_mses = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            with torch.no_grad():
                out = dispatch_with_precision(variant, tokens, indices, experts_fp32,
                                              hot_expert_id, capacity_limit, E)
            elapsed = (time.perf_counter() - t0) * 1000
            mse = float((out - ref_output).pow(2).mean().item())
            trial_times.append(elapsed)
            trial_mses.append(mse)

        entry = {
            'variant': variant,
            'mean_latency_ms': sum(trial_times) / len(trial_times),
            'mean_mse_vs_fp32': sum(trial_mses) / len(trial_mses),
            'min_latency_ms': min(trial_times),
            'throughput_tok_per_ms': N / (sum(trial_times) / len(trial_times)),
        }
        results.append(entry)
        logger.info(f'{variant:25s} latency={entry["mean_latency_ms"]:.3f}ms mse={entry["mean_mse_vs_fp32"]:.4f}')

    print('\n' + '='*75)
    print(f'{"Variant":<25} {"Latency(ms)":>12} {"MSE vs FP32":>13} {"Tput(tok/ms)":>14}')
    print('-'*75)
    for r in results:
        print(f'{r["variant"]:<25} {r["mean_latency_ms"]:>12.3f} '
              f'{r["mean_mse_vs_fp32"]:>13.4f} {r["throughput_tok_per_ms"]:>14.2f}')
    print('='*75)

    with open(out_path, 'w') as f:
        json.dump({'config': str(args.config), 'results': results}, f, indent=2)
    logger.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
