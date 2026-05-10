"""HotExpertManager — N4 novelty: dynamic hot-expert shadow replication.

When one expert receives disproportionate tokens (load > threshold * mean),
creates a shadow copy of its weights and routes overflow tokens to the shadow.

Two modes:
    serial   — shadow expert runs sequentially after primary (baseline)
    parallel — shadow expert runs on separate CUDA stream concurrently (faster)

Reference: CLEAR-MoE++ plan §4.2 N4; LAER-MoE arXiv:2602.11686 (adapted)
"""
import copy
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HotExpertManager:
    """Tracks expert load and manages shadow copies for hot experts.

    Args:
        num_experts:        E — total expert count
        capacity_limit:     max tokens routed to primary expert before overflow
        hot_threshold:      trigger shadow when load(e) > threshold * mean_load
        ema_momentum:       EMA decay for load tracking
        parallel_shadow:    if True, run shadow on separate CUDA stream
    """

    def __init__(
        self,
        num_experts: int,
        capacity_limit: int = 16,
        hot_threshold: float = 2.0,
        ema_momentum: float = 0.9,
        parallel_shadow: bool = False,
    ):
        self.num_experts = num_experts
        self.capacity_limit = capacity_limit
        self.hot_threshold = hot_threshold
        self.ema_momentum = ema_momentum
        self.parallel_shadow = parallel_shadow

        self._load_ema = torch.ones(num_experts) / num_experts
        self._shadow_experts: Dict[int, nn.Module] = {}
        self._hot_experts: List[int] = []

    def update_load(self, indices: torch.Tensor):
        """Update EMA load from current batch routing indices.

        Args:
            indices: (N,) or (B, S) expert assignments for current batch
        """
        flat = indices.reshape(-1).long()
        counts = torch.bincount(flat, minlength=self.num_experts).float()
        counts = counts / (counts.sum() + 1e-8)
        self._load_ema = (
            self.ema_momentum * self._load_ema.to(counts.device)
            + (1 - self.ema_momentum) * counts
        )
        # Identify hot experts
        mean_load = self._load_ema.mean().item()
        self._hot_experts = [
            e for e in range(self.num_experts)
            if self._load_ema[e].item() > self.hot_threshold * mean_load
        ]

    def register_experts(self, experts: List[nn.Module]):
        """Create shadow copies for current hot experts."""
        for e in self._hot_experts:
            if e not in self._shadow_experts:
                logger.info(f"Creating shadow copy for hot expert {e} "
                            f"(load={self._load_ema[e]:.3f})")
                self._shadow_experts[e] = copy.deepcopy(experts[e])

    def is_hot(self, expert_id: int) -> bool:
        return expert_id in self._hot_experts

    def has_shadow(self, expert_id: int) -> bool:
        return expert_id in self._shadow_experts

    @property
    def load_ema(self) -> torch.Tensor:
        return self._load_ema.clone()

    @property
    def hot_experts(self) -> List[int]:
        return list(self._hot_experts)

    def dispatch_with_shadow(
        self,
        sorted_tokens: torch.Tensor,
        plan,
        experts: List[nn.Module],
    ) -> torch.Tensor:
        """Dispatch tokens to experts, routing overflow to shadow copies.

        For hot experts: primary handles up to capacity_limit tokens.
        Overflow tokens go to shadow copy.
        Non-hot experts: standard dispatch.

        Args:
            sorted_tokens: (N, D) tokens sorted by expert (from apply_bucket_plan)
            plan:          BucketPlan
            experts:       list of E expert modules

        Returns:
            (N, D) expert outputs in sorted order
        """
        sorted_out = torch.empty_like(sorted_tokens)

        if self.parallel_shadow and torch.cuda.is_available():
            return self._dispatch_parallel(sorted_tokens, plan, experts, sorted_out)
        else:
            return self._dispatch_serial(sorted_tokens, plan, experts, sorted_out)

    def _dispatch_serial(self, sorted_tokens, plan, experts, sorted_out):
        """Serial dispatch: shadow runs after primary on same stream."""
        for e, expert in enumerate(experts):
            offset = int(plan.bucket_offsets[e])
            size = int(plan.bucket_sizes[e])
            if size == 0:
                continue

            tokens_e = sorted_tokens[offset:offset + size]

            if self.is_hot(e) and self.has_shadow(e) and size > self.capacity_limit:
                # Primary handles first capacity_limit tokens
                primary_tokens = tokens_e[:self.capacity_limit]
                overflow_tokens = tokens_e[self.capacity_limit:]

                # Primary runs first (serial)
                sorted_out[offset:offset + self.capacity_limit] = expert(primary_tokens)
                # Shadow runs after (serial)
                shadow = self._shadow_experts[e]
                sorted_out[offset + self.capacity_limit:offset + size] = shadow(overflow_tokens)
            else:
                sorted_out[offset:offset + size] = expert(tokens_e)

        return sorted_out

    def _dispatch_parallel(self, sorted_tokens, plan, experts, sorted_out):
        """Parallel dispatch: shadow expert on s_shadow stream, primary on s_primary."""
        s_primary = torch.cuda.Stream()
        s_shadow = torch.cuda.Stream()

        for e, expert in enumerate(experts):
            offset = int(plan.bucket_offsets[e])
            size = int(plan.bucket_sizes[e])
            if size == 0:
                continue

            tokens_e = sorted_tokens[offset:offset + size]

            if self.is_hot(e) and self.has_shadow(e) and size > self.capacity_limit:
                primary_tokens = tokens_e[:self.capacity_limit]
                overflow_tokens = tokens_e[self.capacity_limit:]

                # Primary on s_primary, shadow on s_shadow — run concurrently
                with torch.cuda.stream(s_primary):
                    sorted_out[offset:offset + self.capacity_limit] = expert(primary_tokens)
                with torch.cuda.stream(s_shadow):
                    shadow = self._shadow_experts[e]
                    sorted_out[offset + self.capacity_limit:offset + size] = shadow(overflow_tokens)

                # Synchronize both before next expert
                torch.cuda.synchronize()
            else:
                with torch.cuda.stream(s_primary):
                    sorted_out[offset:offset + size] = expert(tokens_e)

        torch.cuda.synchronize()
        return sorted_out

    def reset(self):
        """Clear shadow copies and reset load tracking."""
        self._shadow_experts.clear()
        self._hot_experts = []
        self._load_ema = torch.ones(self.num_experts) / self.num_experts
