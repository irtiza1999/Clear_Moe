"""Triton fused kernels for MoE dispatch.

Three kernels:
    fused_scatter_kernel  — permute tokens into contiguous expert buffers
    grouped_gemm_kernel   — block-sparse GEMM over expert buckets (fc1 pass)
    fused_gather_kernel   — weighted scatter outputs back to token positions

All kernels work on GTX 960 Maxwell (sm_50, no tensor cores).
BLOCK sizes tuned for 2GB VRAM.

Falls back to PyTorch grouped dispatch if triton is not available.
"""
import logging
from typing import List, Optional

import torch
import torch.nn as nn

from clear_moe.dispatch.bucketize import BucketPlan, apply_bucket_plan, invert_bucket_plan

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    logger.warning(
        "triton not available — TritonDispatch will fall back to grouped PyTorch dispatch. "
        "Install with: pip install triton"
    )


if HAS_TRITON:
    @triton.jit
    def _scatter_kernel(
        x_ptr,       # input (N, D) float32
        out_ptr,     # output (N, D) float32
        order_ptr,   # token_order (N,) int64
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Permute tokens according to token_order. Program axis 0 = token index."""
        tok = tl.program_id(0)
        if tok >= N:
            return
        # Load source token index from order permutation
        src = tl.load(order_ptr + tok)
        # Copy D-dimensional token from src position to tok position
        for d0 in tl.range(0, D, BLOCK_D):
            offs = d0 + tl.arange(0, BLOCK_D)
            mask = offs < D
            val = tl.load(x_ptr + src * D + offs, mask=mask, other=0.0)
            tl.store(out_ptr + tok * D + offs, val, mask=mask)

    @triton.jit
    def _grouped_gemm_kernel(
        # Inputs
        a_ptr,              # sorted token activations (N_total, D_in)
        b_ptr,              # expert weight matrices (E, D_in, D_out) — flattened
        c_ptr,              # output (N_total, D_out)
        # Expert metadata
        offsets_ptr,        # bucket_offsets (E,) int64
        sizes_ptr,          # bucket_sizes (E,) int64
        # Dims
        D_in: tl.constexpr,
        D_out: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Block-sparse GEMM over expert buckets.

        Grid: (E, ceil(max_C/BLOCK_M), ceil(D_out/BLOCK_N))
        Each program: one (BLOCK_M, D_out) tile for one expert.
        """
        expert_id = tl.program_id(0)
        tile_m = tl.program_id(1)
        tile_n = tl.program_id(2)

        # Load this expert's metadata
        bucket_offset = tl.load(offsets_ptr + expert_id)
        bucket_size = tl.load(sizes_ptr + expert_id)

        m_start = tile_m * BLOCK_M
        if m_start >= bucket_size:
            return

        n_start = tile_n * BLOCK_N

        # Row/col offsets for this tile
        m_offs = m_start + tl.arange(0, BLOCK_M)
        n_offs = n_start + tl.arange(0, BLOCK_N)

        # Pointers to this expert's token block in sorted buffer
        a_base = a_ptr + (bucket_offset + m_start) * D_in
        # Expert weight matrix: expert e occupies [e*D_in*D_out : (e+1)*D_in*D_out]
        b_base = b_ptr + expert_id * D_in * D_out

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, D_in, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)

            a_mask = (m_offs < bucket_size)[:, None] & (k_offs < D_in)[None, :]
            b_mask = (k_offs < D_in)[:, None] & (n_offs < D_out)[None, :]

            # Load A tile: (BLOCK_M, BLOCK_K)
            a = tl.load(
                a_base + m_offs[:, None] * D_in + k_offs[None, :],
                mask=a_mask, other=0.0
            )
            # Load B tile: (BLOCK_K, BLOCK_N)  [weight matrix is (D_in, D_out)]
            b = tl.load(
                b_base + k_offs[:, None] * D_out + n_offs[None, :],
                mask=b_mask, other=0.0
            )
            acc = tl.dot(a, b, acc)

        # Store output tile
        c_base = c_ptr + (bucket_offset + m_start) * D_out
        c_mask = (m_offs < bucket_size)[:, None] & (n_offs < D_out)[None, :]
        tl.store(
            c_base + m_offs[:, None] * D_out + n_offs[None, :],
            acc, mask=c_mask
        )

    @triton.jit
    def _gather_kernel(
        expert_out_ptr,  # (N, D) sorted expert outputs
        out_ptr,         # (N, D) final output in original token order
        inv_order_ptr,   # inverse_order (N,) int64
        weights_ptr,     # routing weights (N,) — first expert weight per token
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Weighted scatter: place expert outputs back to original token positions."""
        tok = tl.program_id(0)
        if tok >= N:
            return
        dst = tl.load(inv_order_ptr + tok)
        w = tl.load(weights_ptr + tok)
        for d0 in tl.range(0, D, BLOCK_D):
            offs = d0 + tl.arange(0, BLOCK_D)
            mask = offs < D
            val = tl.load(expert_out_ptr + tok * D + offs, mask=mask, other=0.0)
            tl.store(out_ptr + dst * D + offs, val * w, mask=mask)


def triton_scatter(x: torch.Tensor, plan: BucketPlan) -> torch.Tensor:
    """GPU-parallel token scatter using Triton kernel.

    Args:
        x:    (N, D) input tokens (contiguous float32)
        plan: BucketPlan with token_order

    Returns:
        (N, D) tokens permuted into expert order
    """
    if not HAS_TRITON:
        return apply_bucket_plan(x, plan)

    if x.dim() == 3:
        N = x.shape[0] * x.shape[1]
        x = x.reshape(N, x.shape[2])
    else:
        N = x.shape[0]

    D = x.shape[1]
    x = x.contiguous()
    out = torch.empty_like(x)
    order = plan.token_order.contiguous()

    BLOCK_D = min(64, D)
    # Ensure BLOCK_D is power-of-two (Triton requirement)
    BLOCK_D = 1 << (BLOCK_D - 1).bit_length()
    BLOCK_D = max(BLOCK_D, 1)

    grid = (N,)
    _scatter_kernel[grid](
        x, out, order,
        N=N, D=D, BLOCK_D=BLOCK_D,
    )
    return out


def triton_grouped_gemm(
    sorted_tokens: torch.Tensor,
    expert_weights: torch.Tensor,
    plan: BucketPlan,
) -> torch.Tensor:
    """Triton block-sparse GEMM over expert buckets.

    Args:
        sorted_tokens:  (N, D_in) contiguous float32 sorted by expert
        expert_weights: (E, D_in, D_out) expert weight matrices, contiguous
        plan:           BucketPlan with bucket_offsets and bucket_sizes

    Returns:
        (N, D_out) expert outputs in sorted order
    """
    if not HAS_TRITON:
        # Fallback: loop over experts
        N, D_in = sorted_tokens.shape
        E, _, D_out = expert_weights.shape
        out = torch.zeros(N, D_out, device=sorted_tokens.device, dtype=sorted_tokens.dtype)
        for e in range(E):
            offset = int(plan.bucket_offsets[e])
            size = int(plan.bucket_sizes[e])
            if size == 0:
                continue
            tokens_e = sorted_tokens[offset:offset + size]
            out[offset:offset + size] = tokens_e @ expert_weights[e]
        return out

    N, D_in = sorted_tokens.shape
    E, _, D_out = expert_weights.shape
    out = torch.zeros(N, D_out, device=sorted_tokens.device, dtype=sorted_tokens.dtype)

    sorted_tokens = sorted_tokens.contiguous()
    # Flatten expert weights to (E, D_in * D_out) then back to (E, D_in, D_out)
    expert_weights_flat = expert_weights.reshape(E, D_in * D_out).contiguous()

    max_bucket = max(int(plan.bucket_sizes.max().item()), 1)

    # Conservative block sizes for GTX 960 (sm_50, no tensor cores)
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    grid = (
        E,
        (max_bucket + BLOCK_M - 1) // BLOCK_M,
        (D_out + BLOCK_N - 1) // BLOCK_N,
    )

    _grouped_gemm_kernel[grid](
        sorted_tokens,
        expert_weights_flat,
        out,
        plan.bucket_offsets.contiguous(),
        plan.bucket_sizes.contiguous(),
        D_in=D_in, D_out=D_out,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


def triton_gather(
    sorted_expert_out: torch.Tensor,
    plan: BucketPlan,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted gather: scatter sorted expert outputs back to original positions.

    Args:
        sorted_expert_out: (N, D) expert outputs in bucket-sorted order
        plan:              BucketPlan with inverse_order
        weights:           (N,) or (N, 1) routing weight for each token

    Returns:
        (N, D) outputs in original token order, weighted
    """
    if not HAS_TRITON:
        out = sorted_expert_out * weights.reshape(-1, 1)
        return invert_bucket_plan(out, plan)

    N, D = sorted_expert_out.shape
    out = torch.zeros_like(sorted_expert_out)
    sorted_expert_out = sorted_expert_out.contiguous()
    inv_order = plan.inverse_order.contiguous()
    w = weights.reshape(-1).contiguous().float()

    BLOCK_D = min(64, D)
    BLOCK_D = 1 << (BLOCK_D - 1).bit_length()
    BLOCK_D = max(BLOCK_D, 1)

    grid = (N,)
    _gather_kernel[grid](
        sorted_expert_out, out, inv_order, w,
        N=N, D=D, BLOCK_D=BLOCK_D,
    )
    return out


def triton_fused_dispatch(
    x: torch.Tensor,
    plan: BucketPlan,
    experts: List[nn.Module],
    weights: torch.Tensor,
) -> torch.Tensor:
    """Full fused dispatch: scatter → per-expert GEMM → gather.

    Uses Triton kernels for scatter and gather steps.
    For GEMM, uses triton_grouped_gemm if expert weights are linear layers,
    otherwise falls back to sequential expert forward.

    Args:
        x:       (N, D) or (B, S, D) input tokens
        plan:    BucketPlan
        experts: list of E nn.Module experts
        weights: (N,) or (B*S,) first-expert routing weights for gather

    Returns:
        (N, D) output in original token order
    """
    if x.dim() == 3:
        N = x.shape[0] * x.shape[1]
        x = x.reshape(N, x.shape[2])

    # Step 1: Triton scatter
    sorted_x = triton_scatter(x, plan)

    # Step 2: Check if all experts are single nn.Linear → use grouped GEMM
    if HAS_TRITON and all(isinstance(e, nn.Linear) for e in experts):
        E = len(experts)
        D_in = experts[0].in_features
        D_out = experts[0].out_features
        # Stack expert weight matrices: (E, D_in, D_out)
        w_stack = torch.stack([e.weight.T for e in experts], dim=0)  # (E, D_in, D_out)
        sorted_out = triton_grouped_gemm(sorted_x, w_stack, plan)
        # Add bias if present
        for e_idx, e in enumerate(experts):
            if e.bias is not None:
                offset = int(plan.bucket_offsets[e_idx])
                size = int(plan.bucket_sizes[e_idx])
                if size > 0:
                    sorted_out[offset:offset + size] += e.bias
    else:
        # Fallback: sequential expert forward on sorted tokens
        sorted_out = torch.empty_like(sorted_x)
        for e_idx, expert in enumerate(experts):
            offset = int(plan.bucket_offsets[e_idx])
            size = int(plan.bucket_sizes[e_idx])
            if size == 0:
                continue
            sorted_out[offset:offset + size] = expert(sorted_x[offset:offset + size])

    # Step 3: Triton gather (weighted)
    w_flat = weights.reshape(-1).to(x.dtype)
    # Map weights from original order to sorted order
    sorted_weights = w_flat[plan.token_order]
    return triton_gather(sorted_out, plan, sorted_weights)
