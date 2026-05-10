"""Script 23: CommAwareRouter λ sweep — remote routing fraction vs throughput.

Sweeps lambda_comm and lambda_hot values.
Measures: remote_routing_fraction, mean_latency, load_imbalance.
Produces data for Figure C.

Usage:
    python scripts/23_comm_aware_router.py --config configs/ablation_v3_full.yaml --dry-run
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

from clear_moe.routers import CommAwareRouter
from clear_moe.runtime import SimulatedEP, CostModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def sweep_lambda(
    x: torch.Tensor,
    E: int,
    lambda_comm: float,
    lambda_hot: float,
    comm_latency_us: float,
    n_iters: int = 20,
) -> dict:
    """Run CommAwareRouter with given lambda values and measure routing behavior."""
    D = x.shape[-1]
    N = x.reshape(-1, D).shape[0]

    router = CommAwareRouter(D, E, top_k=1, lambda_comm=lambda_comm, lambda_hot=lambda_hot)
    ep = SimulatedEP(num_experts=E, num_ranks=2, comm_latency_us=comm_latency_us)
    device_map = ep.build_device_map()
    experts = [nn.Identity() for _ in range(E)]

    latencies = []
    remote_fracs = []
    entropies = []

    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            indices, weights, logits, aux = router(x, device_map=device_map)
            latencies.append((time.perf_counter() - t0) * 1000)
            entropies.append(aux.get('entropy', 0.0))
            # Compute remote fraction under EP-2 layout
            primary = indices.reshape(N, -1)[:, 0]
            _, stats = ep.dispatch(x.reshape(N, D), primary)
            remote_fracs.append(stats['remote_routing_fraction'])

    return {
        'lambda_comm': lambda_comm,
        'lambda_hot': lambda_hot,
        'mean_latency_ms': sum(latencies) / len(latencies),
        'mean_remote_fraction': sum(remote_fracs) / len(remote_fracs),
        'mean_entropy': sum(entropies) / len(entropies),
        'load_imbalance': float(router.load_ema.max() / router.load_ema.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description='CommAwareRouter lambda sweep')
    parser.add_argument('--config', default='configs/ablation_v3_full.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'outputs/23_comm_aware_router_{timestamp}.json'

    if args.dry_run:
        B, S, D, E = 2, 4, 32, 4
        lambda_comm_values = [0.0, 0.1, 1.0]
        lambda_hot_values = [0.0, 0.1]
        n_iters = 5
        comm_latency_us = 0.0
    else:
        B, S, D = 2, 197, 384
        E = config.get('extraction', {}).get('num_experts', 4)
        lambda_comm_values = config.get('lambda_comm_values', [0.0, 0.1, 0.3, 1.0])
        lambda_hot_values = config.get('lambda_hot_values', [0.0, 0.1, 0.3, 1.0])
        n_iters = 30
        comm_latency_us = config.get('comm_latency_us', 100.0)

    x = torch.randn(B, S, D)
    results = []

    for lc in lambda_comm_values:
        for lh in lambda_hot_values:
            result = sweep_lambda(x, E, lc, lh, comm_latency_us, n_iters)
            results.append(result)
            logger.info(f'lam_comm={lc:.2f} lam_hot={lh:.2f} -> '
                        f'remote={result["mean_remote_fraction"]:.3f} '
                        f'latency={result["mean_latency_ms"]:.3f}ms')

    print('\n' + '='*75)
    print(f'{"lam_comm":>8} {"lam_hot":>8} {"Remote%":>10} {"Entropy":>10} {"Imbalance":>11}')
    print('-'*75)
    for r in results:
        print(f'{r["lambda_comm"]:>8.2f} {r["lambda_hot"]:>8.2f} '
              f'{r["mean_remote_fraction"]*100:>9.1f}% {r["mean_entropy"]:>10.3f} '
              f'{r["load_imbalance"]:>11.3f}')
    print('='*75)

    with open(out_path, 'w') as f:
        json.dump({'config': str(args.config), 'results': results}, f, indent=2)
    logger.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
