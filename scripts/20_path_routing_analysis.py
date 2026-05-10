"""Script 20: Path Routing Analysis — path entropy, churn, consistency across layers.

Compares:
    - independent per-layer router (baseline)
    - PathConsistentRouter (path penalty applied)

Measures: path_entropy, path_churn per layer, cross-layer assignment consistency.

Usage:
    python scripts/20_path_routing_analysis.py --config configs/ablation_v3_full.yaml --dry-run
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
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from clear_moe.routers import LinearRouter, PathConsistentRouter
from clear_moe.runtime import CostModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def simulate_layer_routing(
    routers,
    tokens_per_layer,
    use_path_router: bool,
    num_experts: int,
) -> dict:
    """Simulate routing across L layers of a transformer.

    Args:
        routers:          list of L router instances (one per layer)
        tokens_per_layer: list of L tensors, each (B, S, D)
        use_path_router:  whether routers are PathConsistentRouter instances
        num_experts:      E

    Returns:
        dict with per-layer path metrics
    """
    all_indices = []
    all_churns = []
    all_entropies = []
    all_load_vecs = []

    if use_path_router:
        for r in routers:
            if isinstance(r, PathConsistentRouter):
                r.reset_path()

    with torch.no_grad():
        for l, (router, x) in enumerate(zip(routers, tokens_per_layer)):
            indices, weights, logits, aux = router(x, layer_idx=l)
            all_indices.append(indices)
            all_entropies.append(aux.get('entropy', 0.0))
            if aux.get('load_vec') is not None:
                all_load_vecs.append(aux['load_vec'])
            # Compute churn vs previous layer
            if len(all_indices) >= 2:
                prev = all_indices[-2]
                curr = all_indices[-1]
                if prev.shape == curr.shape:
                    churn = (prev != curr).float().mean().item()
                else:
                    churn = 0.0
                all_churns.append(churn)
            else:
                all_churns.append(0.0)

    # Aggregate path metrics
    mean_churn = sum(all_churns) / max(len(all_churns), 1)
    mean_entropy = sum(all_entropies) / max(len(all_entropies), 1)

    # Cross-layer consistency: fraction of tokens that kept same expert across all layers
    if len(all_indices) >= 2:
        flat_indices = [idx.reshape(-1) for idx in all_indices]
        # Compare each layer to first layer
        consistent = torch.ones(flat_indices[0].shape, dtype=torch.bool)
        for idx in flat_indices[1:]:
            if idx.shape == flat_indices[0].shape:
                consistent &= (idx == flat_indices[0])
        cross_layer_consistency = float(consistent.float().mean().item())
    else:
        cross_layer_consistency = 1.0

    load_imbalance = 0.0
    if all_load_vecs:
        mean_load = torch.stack(all_load_vecs).mean(0)
        load_imbalance = float(mean_load.max() / mean_load.mean())

    return {
        'use_path_router': use_path_router,
        'num_layers': len(routers),
        'mean_path_churn': mean_churn,
        'mean_entropy': mean_entropy,
        'cross_layer_consistency': cross_layer_consistency,
        'load_imbalance': load_imbalance,
        'per_layer_churn': all_churns,
        'per_layer_entropy': all_entropies,
    }


def main():
    parser = argparse.ArgumentParser(description='Path routing analysis')
    parser.add_argument('--config', default='configs/ablation_v3_full.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'outputs/20_path_routing_analysis_{timestamp}.json'

    if args.dry_run:
        D, E, B, S, L = 32, 4, 2, 4, 6
        n_runs = 5
    else:
        D = 384
        E = config.get('extraction', {}).get('num_experts', 4)
        B, S = 2, 197  # typical ViT (CLS + 14*14 patches)
        L = 6          # last half of DeiT-Small (12 layers total)
        n_runs = 20

    # Build routers: independent vs path-consistent
    independent_routers = [LinearRouter(D, E, top_k=1) for _ in range(L)]
    path_routers = [
        PathConsistentRouter(LinearRouter(D, E, top_k=1),
                             path_penalty_coeff=config.get('path_penalty_coeff', 0.01))
        for _ in range(L)
    ]

    all_results = []

    for n in range(n_runs):
        # Re-randomize tokens each run to get stable statistics
        tokens_per_layer = [torch.randn(B, S, D) for _ in range(L)]

        res_indep = simulate_layer_routing(
            independent_routers, tokens_per_layer,
            use_path_router=False, num_experts=E
        )
        res_path = simulate_layer_routing(
            path_routers, tokens_per_layer,
            use_path_router=True, num_experts=E
        )
        all_results.append({'independent': res_indep, 'path_consistent': res_path})

    # Average over runs
    def avg_key(results, variant, key):
        vals = [r[variant][key] for r in results if isinstance(r[variant][key], (int, float))]
        return sum(vals) / max(len(vals), 1)

    summary = {
        'independent': {
            'mean_path_churn': avg_key(all_results, 'independent', 'mean_path_churn'),
            'mean_entropy': avg_key(all_results, 'independent', 'mean_entropy'),
            'cross_layer_consistency': avg_key(all_results, 'independent', 'cross_layer_consistency'),
            'load_imbalance': avg_key(all_results, 'independent', 'load_imbalance'),
        },
        'path_consistent': {
            'mean_path_churn': avg_key(all_results, 'path_consistent', 'mean_path_churn'),
            'mean_entropy': avg_key(all_results, 'path_consistent', 'mean_entropy'),
            'cross_layer_consistency': avg_key(all_results, 'path_consistent', 'cross_layer_consistency'),
            'load_imbalance': avg_key(all_results, 'path_consistent', 'load_imbalance'),
        },
    }

    print('\n' + '='*70)
    print(f'{"Metric":<30} {"Independent":>18} {"PathConsistent":>18}')
    print('-'*70)
    for k in ['mean_path_churn', 'mean_entropy', 'cross_layer_consistency', 'load_imbalance']:
        vi = summary['independent'][k]
        vp = summary['path_consistent'][k]
        print(f'{k:<30} {vi:>18.4f} {vp:>18.4f}')
    print('='*70)

    with open(out_path, 'w') as f:
        json.dump({'config': str(args.config), 'summary': summary, 'runs': len(all_results)}, f, indent=2)
    logger.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
