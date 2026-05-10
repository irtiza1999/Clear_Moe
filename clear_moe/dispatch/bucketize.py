"""Token bucketization — organizes tokens by expert before dispatch.

Modes:
    none         — unsorted, naive gather (baseline)
    expert_sort  — argsort by primary expert ID → contiguous per-expert slices
    group_sort   — sort by expert // group_size → coarser locality
    latent_bucket — sort by nearest precomputed centroid (fitted on calibration set)
"""
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BucketPlan:
    """Token organization plan produced by bucketize().

    Attributes:
        token_order:    (N,) int64 permutation — new_buf[i] = x[token_order[i]]
        bucket_offsets: (E,) int64 start offset in sorted buffer per expert
        bucket_sizes:   (E,) int64 token count per expert
        inverse_order:  (N,) int64 to scatter expert outputs back to original positions
    """
    token_order: torch.Tensor
    bucket_offsets: torch.Tensor
    bucket_sizes: torch.Tensor
    inverse_order: torch.Tensor


def bucketize(
    x: torch.Tensor,
    indices: torch.Tensor,
    num_experts: int,
    mode: str = 'expert_sort',
    centroids: Optional[torch.Tensor] = None,
    group_size: int = 2,
) -> BucketPlan:
    """Organize tokens into expert buckets.

    Args:
        x:           (N, D) or (B, S, D) token representations
        indices:     (N,) or (B, S) or (B, S, top_k) primary expert assignment
                     Only the first expert per token is used for bucketing.
        num_experts: total number of experts E
        mode:        bucketing strategy
        centroids:   (E, D) precomputed calibration centroids for 'latent_bucket' mode.
                     Fitted on calibration activations via KMeans; fixed at inference.
        group_size:  expert group size for 'group_sort' mode

    Returns:
        BucketPlan with token_order, bucket_offsets, bucket_sizes, inverse_order
    """
    # Flatten to (N, D) and (N,) for uniform handling
    orig_shape = x.shape
    if x.dim() == 3:
        N = orig_shape[0] * orig_shape[1]
        x_flat = x.reshape(N, orig_shape[2])
    else:
        N = orig_shape[0]
        x_flat = x

    # Extract primary expert index per token
    if indices.dim() == 3:
        # (B, S, top_k) → take first expert
        primary = indices[..., 0].reshape(-1)
    elif indices.dim() == 2:
        primary = indices.reshape(-1)
    else:
        primary = indices

    primary = primary.long()
    device = x_flat.device

    if mode == 'none':
        order = torch.arange(N, device=device, dtype=torch.long)
        sort_key = primary
    elif mode == 'expert_sort':
        order = torch.argsort(primary, stable=True)
        sort_key = primary[order]
    elif mode == 'group_sort':
        group_key = primary // group_size
        order = torch.argsort(group_key, stable=True)
        sort_key = primary[order]
    elif mode == 'latent_bucket':
        if centroids is None:
            raise ValueError("latent_bucket mode requires precomputed centroids (E, D)")
        # Assign each token to nearest centroid
        # centroids: (E, D), x_flat: (N, D)
        dists = torch.cdist(x_flat.float(), centroids.float())  # (N, E)
        centroid_assign = dists.argmin(dim=-1)  # (N,)
        order = torch.argsort(centroid_assign, stable=True)
        sort_key = primary[order]
    else:
        raise ValueError(f"Unknown bucketize mode: {mode!r}. "
                         f"Choose from: none, expert_sort, group_sort, latent_bucket")

    # Compute bucket_offsets and bucket_sizes from sorted primary assignments
    counts = torch.bincount(sort_key, minlength=num_experts)  # (E,)
    offsets = torch.zeros(num_experts, dtype=torch.long, device=device)
    offsets[1:] = counts[:-1].cumsum(0)

    # inverse_order: undo the permutation
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(N, device=device, dtype=torch.long)

    return BucketPlan(
        token_order=order,
        bucket_offsets=offsets,
        bucket_sizes=counts,
        inverse_order=inverse,
    )


def apply_bucket_plan(x: torch.Tensor, plan: BucketPlan) -> torch.Tensor:
    """Permute tokens according to BucketPlan.token_order. Returns (N, D) sorted buffer."""
    if x.dim() == 3:
        N = x.shape[0] * x.shape[1]
        x = x.reshape(N, x.shape[2])
    return x[plan.token_order]


def invert_bucket_plan(expert_out: torch.Tensor, plan: BucketPlan) -> torch.Tensor:
    """Scatter expert outputs back to original token positions."""
    return expert_out[plan.inverse_order]
