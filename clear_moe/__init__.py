"""
CLEAR-MoE for Vision: Calibration-Driven Layer-Selective Expert Extraction
from Pretrained Dense Backbones.

Core library package providing:
- models: Model loading and FFN layer access
- calibration: Calibration data loading and activation capture
- scoring: Layer scoring for expertization selection
- extraction: Shared-basis + residual expert decomposition
- router: Router architecture and training (legacy scripts 01-16)
- routers: Extended router sub-package (CLEAR-MoE++ v3)
- dispatch: Token organization and kernel backends
- runtime: MoE executor and parallelism primitives
- metrics: Evaluation metrics (accuracy, latency, energy)
- utils: Configuration, seeding, logging utilities
"""

__version__ = "0.2.0"
__author__ = "CLEAR-MoE Team"

# ── routers ──────────────────────────────────────────────────────────────────
from clear_moe.routers import (
    BaseRouter,
    LinearRouter,
    AdaptiveRouter,
    MLPRouter,
    ExpertChoiceRouter,
    CommAwareRouter,
    PathConsistentRouter,
    SelfRoutingProxy,
    SharedExpertRouter,
    make_path_consistent,
)

# ── dispatch ──────────────────────────────────────────────────────────────────
from clear_moe.dispatch import (
    BucketPlan,
    bucketize,
    apply_bucket_plan,
    invert_bucket_plan,
    padded_scatter,
    padding_free_scatter,
    weighted_combine,
    triton_scatter,
    triton_grouped_gemm,
    triton_gather,
    triton_fused_dispatch,
    HAS_TRITON,
    CudaTimer,
    DispatchTiming,
    DispatchProfiler,
)

# ── runtime ───────────────────────────────────────────────────────────────────
from clear_moe.runtime import (
    MoEExecutor,
    HotExpertManager,
    SimulatedEP,
    SimulatedPP,
    ExpertRelayout,
    CostModel,
    CostModelOutput,
)

__all__ = [
    # routers
    "BaseRouter", "LinearRouter", "AdaptiveRouter", "MLPRouter",
    "ExpertChoiceRouter", "CommAwareRouter", "PathConsistentRouter",
    "SelfRoutingProxy", "SharedExpertRouter", "make_path_consistent",
    # dispatch
    "BucketPlan", "bucketize", "apply_bucket_plan", "invert_bucket_plan",
    "padded_scatter", "padding_free_scatter", "weighted_combine",
    "triton_scatter", "triton_grouped_gemm", "triton_gather", "triton_fused_dispatch",
    "HAS_TRITON", "CudaTimer", "DispatchTiming", "DispatchProfiler",
    # runtime
    "MoEExecutor", "HotExpertManager", "SimulatedEP", "SimulatedPP",
    "ExpertRelayout", "CostModel", "CostModelOutput",
]
