"""Script 24: Dynamic Expert Relayout — hotness-driven remapping.

Simulates multi-epoch training where expert load shifts.
After each epoch, relayout reassigns hot experts to local rank.
Measures: remote_routing_fraction before/after relayout per epoch.
Produces data for Figure F (expert hotness heatmap over time).

Usage:
    python scripts/24_dynamic_relayout.py --config configs/ablation_v3_full.yaml --dry-run
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

from clear_moe.routers import CommAwareRouter, LinearRouter
from clear_moe.runtime import SimulatedEP, ExpertRelayout, HotExpertManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def simulate_epoch(
    router,
    x: torch.Tensor,
    E: int,
    device_map: dict,
    n_batches: int = 10,
) -> torch.Tensor:
    """Simulate one epoch: collect routing statistics across batches.

    Returns: (E,) mean load vector across epoch.
    """
    load_accumulator = torch.zeros(E)
    N = x.reshape(-1, x.shape[-1]).shape[0]

    with torch.no_grad():
        for _ in range(n_batches):
            batch = x + torch.randn_like(x) * 0.1  # slight variation
            indices, weights, logits, aux = router(batch, device_map=device_map)
            flat = indices.reshape(-1).long()
            counts = torch.bincount(flat, minlength=E).float()
            load_accumulator += counts / (counts.sum() + 1e-8)

    return load_accumulator / n_batches


def main():
    parser = argparse.ArgumentParser(description='Dynamic expert relayout')
    parser.add_argument('--config', default='configs/ablation_v3_full.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'outputs/24_dynamic_relayout_{timestamp}.json'

    if args.dry_run:
        B, S, D, E = 2, 4, 32, 4
        num_epochs = 5
        n_batches = 5
        comm_latency_us = 0.0
        num_ranks = 2
    else:
        B, S, D = 2, 197, 384
        E = config.get('extraction', {}).get('num_experts', 4)
        num_epochs = 10
        n_batches = 20
        comm_latency_us = config.get('comm_latency_us', 100.0)
        num_ranks = 2

    x = torch.randn(B, S, D)

    # Static EP (no relayout)
    ep_static = SimulatedEP(num_experts=E, num_ranks=num_ranks, comm_latency_us=comm_latency_us)
    router_static = LinearRouter(D, E, top_k=1)

    # Dynamic EP (with relayout)
    ep_dynamic = SimulatedEP(num_experts=E, num_ranks=num_ranks, comm_latency_us=comm_latency_us)
    rl = ExpertRelayout(num_experts=E, num_ranks=num_ranks)
    router_dynamic = CommAwareRouter(D, E, top_k=1, lambda_comm=0.1, lambda_hot=0.1)

    epoch_results = []

    for epoch in range(num_epochs):
        # Introduce load shift: increasingly skewed toward expert 0
        skew_factor = min(0.3 + epoch * 0.05, 0.9)
        # Modify router to prefer expert 0 by biasing
        with torch.no_grad():
            if hasattr(router_static, 'gate'):
                router_static.gate.bias.data[0] = skew_factor * 3.0
            if hasattr(router_dynamic, 'gate'):
                router_dynamic.gate.bias.data[0] = skew_factor * 3.0

        device_map_static = ep_static.build_device_map()
        device_map_dynamic = rl.device_map

        # Static: no relayout
        load_static = simulate_epoch(router_static, x, E, device_map_static, n_batches)
        N_flat = x.reshape(-1, D).shape[0]
        indices_static, _, _, _ = router_static(x, device_map=device_map_static)
        remote_static = ep_static.build_device_map()
        frac_static = rl.compute_remote_fraction(indices_static.reshape(-1))

        # Dynamic: relayout after each epoch
        load_dynamic = simulate_epoch(router_dynamic, x, E, device_map_dynamic, n_batches)
        indices_dynamic, _, _, _ = router_dynamic(x, device_map=device_map_dynamic)
        frac_before = rl.compute_remote_fraction(indices_dynamic.reshape(-1))
        # Relayout
        rl.relayout(load_dynamic)
        device_map_dynamic_new = rl.device_map
        indices_dynamic_new, _, _, _ = router_dynamic(x, device_map=device_map_dynamic_new)
        frac_after = rl.compute_remote_fraction(indices_dynamic_new.reshape(-1))

        entry = {
            'epoch': epoch,
            'skew_factor': skew_factor,
            'load_static': load_static.tolist(),
            'load_dynamic': load_dynamic.tolist(),
            'remote_frac_static': frac_static,
            'remote_frac_dynamic_before': frac_before,
            'remote_frac_dynamic_after': frac_after,
        }
        epoch_results.append(entry)
        logger.info(f'Epoch {epoch}: skew={skew_factor:.2f} '
                    f'static_remote={frac_static:.3f} '
                    f'dynamic_before={frac_before:.3f} '
                    f'dynamic_after={frac_after:.3f}')

    # Print table (Fig F data)
    print('\n' + '='*75)
    print(f'{"Epoch":>6} {"Skew":>6} {"StaticRemote":>14} {"DynBefore":>11} {"DynAfter":>10}')
    print('-'*75)
    for r in epoch_results:
        print(f'{r["epoch"]:>6} {r["skew_factor"]:>6.2f} '
              f'{r["remote_frac_static"]:>14.3f} '
              f'{r["remote_frac_dynamic_before"]:>11.3f} '
              f'{r["remote_frac_dynamic_after"]:>10.3f}')
    print('='*75)

    with open(out_path, 'w') as f:
        json.dump({'config': str(args.config), 'epochs': epoch_results}, f, indent=2)
    logger.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
