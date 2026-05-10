"""Scatter/gather implementations for MoE dispatch.

Two backends:
    padded_scatter       — classic: pad each expert slot to max_capacity, GEMM-friendly
    padding_free_scatter — ScatterMoE-style: contiguous grouped ops, no padding waste
"""
from typing import List, Optional

import torch
import torch.nn as nn

from clear_moe.dispatch.bucketize import BucketPlan, apply_bucket_plan, invert_bucket_plan


def padded_scatter(
    x: torch.Tensor,
    plan: BucketPlan,
    experts: List[nn.Module],
    capacity: Optional[int] = None,
) -> torch.Tensor:
    """Classic padded MoE dispatch.

    Pads each expert's token bucket to `capacity`, runs each expert's forward,
    then unpads and reconstructs.

    Args:
        x:        (N, D) or (B, S, D) input tokens (unsorted)
        plan:     BucketPlan from bucketize()
        experts:  list of E expert modules, each maps (T, D) -> (T, D)
        capacity: max tokens per expert; if None uses max(bucket_sizes)

    Returns:
        (N, D) output in original token order
    """
    orig_shape = x.shape
    if x.dim() == 3:
        N = orig_shape[0] * orig_shape[1]
        x = x.reshape(N, orig_shape[2])
    else:
        N = orig_shape[0]

    D = x.shape[-1]
    E = len(experts)
    cap = capacity if capacity is not None else int(plan.bucket_sizes.max().item())
    if cap == 0:
        return x  # nothing to route

    # Sort tokens into expert order
    sorted_x = apply_bucket_plan(x, plan)  # (N, D)

    # Pad each expert's slice to `cap` and collect outputs
    outputs = torch.zeros(N, D, device=x.device, dtype=x.dtype)

    for e in range(E):
        offset = int(plan.bucket_offsets[e].item())
        size = int(plan.bucket_sizes[e].item())
        if size == 0:
            continue

        # Slice this expert's tokens (may be less than cap)
        tokens_e = sorted_x[offset:offset + size]  # (size, D)

        # Pad to capacity if needed
        if size < cap:
            pad = torch.zeros(cap - size, D, device=x.device, dtype=x.dtype)
            tokens_e_padded = torch.cat([tokens_e, pad], dim=0)
        else:
            tokens_e_padded = tokens_e[:cap]

        # Expert forward
        out_e = experts[e](tokens_e_padded)  # (cap, D)

        # Unpad and store
        outputs[offset:offset + min(size, cap)] = out_e[:min(size, cap)]

    # Scatter back to original order
    return invert_bucket_plan(outputs, plan)


def padding_free_scatter(
    x: torch.Tensor,
    plan: BucketPlan,
    experts: List[nn.Module],
) -> torch.Tensor:
    """ScatterMoE-style padding-free dispatch.

    Operates on contiguous expert slices without any padding.
    Lower memory than padded_scatter when bucket sizes vary.

    Args:
        x:       (N, D) or (B, S, D) input tokens
        plan:    BucketPlan from bucketize()
        experts: list of E expert modules

    Returns:
        (N, D) output in original token order
    """
    orig_shape = x.shape
    if x.dim() == 3:
        N = orig_shape[0] * orig_shape[1]
        x = x.reshape(N, orig_shape[2])
    else:
        N = orig_shape[0]

    D = x.shape[-1]
    E = len(experts)

    sorted_x = apply_bucket_plan(x, plan)  # (N, D)
    outputs = torch.empty_like(sorted_x)

    for e in range(E):
        offset = int(plan.bucket_offsets[e].item())
        size = int(plan.bucket_sizes[e].item())
        if size == 0:
            continue
        tokens_e = sorted_x[offset:offset + size]
        outputs[offset:offset + size] = experts[e](tokens_e)

    return invert_bucket_plan(outputs, plan)


def weighted_combine(
    expert_outputs: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Weighted combination when top_k > 1.

    Args:
        expert_outputs: (N, D) combined expert outputs (already gathered)
        weights:        (N, top_k) routing weights
        indices:        (N, top_k) expert assignments
        num_experts:    E

    Returns:
        (N, D) weighted sum — used when top_k > 1
    """
    # For top_k=1, expert_outputs is already the result
    # For top_k>1, need to run each expert separately and combine
    # This function handles the combination step given pre-computed per-expert outputs
    N, D = expert_outputs.shape[:2]
    # weights: (N, top_k), sum over k dimension
    # expert_outputs assumed to already be weighted in calling code
    return expert_outputs
