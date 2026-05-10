"""Dynamic expert-device relayout based on hotness statistics.

After each evaluation epoch, reassigns expert-rank mapping to reduce
remote routing fraction. Hot experts move to local rank (rank 0).
Cold experts move to remote ranks.

Feeds back into CommAwareRouter.device_map for next epoch.

Reference: CLEAR-MoE++ plan §7.2; LAER-MoE arXiv:2602.11686
"""
import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class ExpertRelayout:
    """Load-adaptive expert-device remapping.

    Args:
        num_experts:  E
        num_ranks:    number of virtual ranks
        local_rank:   the "local" rank that we optimize for (default: 0)
    """

    def __init__(
        self,
        num_experts: int,
        num_ranks: int = 2,
        local_rank: int = 0,
    ):
        self.num_experts = num_experts
        self.num_ranks = num_ranks
        self.local_rank = local_rank
        # Initial assignment: round-robin
        self._expert_to_rank: Dict[int, int] = {
            e: e % num_ranks for e in range(num_experts)
        }
        self._relayout_history: List[Dict[int, int]] = []

    @property
    def device_map(self) -> dict:
        """Return CommAwareRouter-compatible device_map dict."""
        dm = dict(self._expert_to_rank)
        dm['local_rank'] = self.local_rank
        return dm

    def compute_hotness_scores(
        self, load_ema: torch.Tensor
    ) -> Tuple[List[int], List[int]]:
        """Rank experts by load. Returns (hot_experts, cold_experts) sorted by load."""
        loads = load_ema.tolist()
        ranked = sorted(range(self.num_experts), key=lambda e: loads[e], reverse=True)
        # Top half = hot, bottom half = cold
        split = self.num_experts // 2
        return ranked[:split], ranked[split:]

    def relayout(self, load_ema: torch.Tensor) -> Dict[int, int]:
        """Recompute expert-rank assignment based on current load EMA.

        Strategy: assign top-K hottest experts to local rank (rank 0).
        Remaining experts distributed across other ranks.
        K = floor(E / num_ranks) to maintain balance.

        Args:
            load_ema: (E,) normalized load per expert

        Returns:
            new expert_to_rank mapping
        """
        experts_per_rank = self.num_experts // self.num_ranks
        # Sort experts by load descending
        loads = load_ema.tolist()
        sorted_experts = sorted(range(self.num_experts), key=lambda e: loads[e], reverse=True)

        new_mapping: Dict[int, int] = {}
        # Assign top experts_per_rank to local rank
        for i, e in enumerate(sorted_experts):
            rank = i // max(experts_per_rank, 1)
            rank = min(rank, self.num_ranks - 1)
            # Remap: hottest goes to local_rank
            new_mapping[e] = self.local_rank if rank == 0 else (rank % self.num_ranks)

        # Store history
        self._relayout_history.append(dict(self._expert_to_rank))
        self._expert_to_rank = new_mapping

        hot, cold = self.compute_hotness_scores(load_ema)
        local_experts = [e for e, r in new_mapping.items() if r == self.local_rank]
        logger.info(
            f"[relayout] Reassigned. Hot experts: {hot[:4]} → local rank {self.local_rank}. "
            f"Local experts after relayout: {local_experts}"
        )
        return new_mapping

    def compute_remote_fraction(self, indices: torch.Tensor) -> float:
        """Compute fraction of tokens routing to remote experts under current layout.

        Args:
            indices: (N,) or (B, S) expert assignments
        Returns:
            fraction in [0, 1] of tokens assigned to non-local experts
        """
        flat = indices.reshape(-1).tolist()
        remote = sum(1 for e in flat if self._expert_to_rank.get(int(e), 0) != self.local_rank)
        return remote / max(len(flat), 1)

    def reset(self):
        """Reset to initial round-robin assignment."""
        self._expert_to_rank = {e: e % self.num_ranks for e in range(self.num_experts)}
        self._relayout_history.clear()
