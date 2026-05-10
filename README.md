# CLEAR-MoE++: Calibration-Driven Layer-Selective Expert Extraction from Pretrained Vision Transformers

## Project Overview

CLEAR-MoE++ is a post-training expert extraction pipeline that converts pretrained vision transformer (ViT) backbones into sparse Mixture-of-Experts (MoE) models without full retraining. The pipeline achieves computational efficiency through selective layer expertization, shared-basis decomposition, and optimized runtime dispatch.

**Data Policy:** This repository does not include raw datasets. Download Imagenette and Cityscapes locally using the helper scripts in `datasets/` before running the pipeline.

**Key Results:**
- **Classification (DeiT-Small on Imagenette):** 86.51% Top-1 accuracy (99.8% of dense baseline 86.69%)
- **Active Parameter Reduction:** 42.18% fewer parameters with top-1 routing
- **Segmentation Transfer (SegFormer-B0 on Cityscapes):** 0.5757 mIoU (fully preserved)
- **Dispatch Performance:** 249,775 tokens/second with cuBLAS backend (18.63× over CPU)
- **PCIe Overhead Measurement:** 5.04 ms per batch for CPU-router + GPU-expert disaggregation

## Project Structure

```
706/
├── clear_moe/                    # Core implementation
│   ├── calibration.py           # Activation logging & statistics
│   ├── extraction.py            # SVD + k-means expert decomposition
│   ├── hierarchical_moe.py       # H-MoE variant
│   ├── metrics.py               # Evaluation metrics
│   ├── models.py                # MoE model definitions
│   ├── router.py                # Router architectures (7 variants)
│   ├── scoring.py               # Layer selection scoring
│   ├── soft_elastic_moe.py       # Soft/Elastic MoE variants
│   ├── utils.py                 # Utilities
│   ├── dispatch/                # Dispatch backends (6 strategies)
│   ├── routers/                 # Router implementations
│   └── runtime/                 # Runtime analysis & measurement
├── configs/                      # Configuration files
│   ├── deit_s_imagenet.yaml     # Default DeiT-Small config
│   ├── ablation_deit_s.yaml     # Ablation study config
│   ├── cityscapes_segformer_b0.yaml  # SegFormer-B0 config
│   └── ...
├── data/                        # Cached datasets (gitignored)
├── datasets/                    # Dataset downloaders
├── report_writing/              # Paper LaTeX & figures
├── outputs/                     # Experiment results (gitignored)
├── scripts/                     # Standalone execution scripts
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
├── run_ablation.py             # Full ablation runner
└── README.md                    # This file
```

## Installation & Setup

### 1. Python Environment

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
source .venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import torch; import torchvision; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python -c "import timm; print('timm OK')"
```

### 3. Download Datasets Locally

```bash
# Imagenette (download to data/imagenette/, ~1.4 GB archive)
python datasets/download_imagenette.py

# Cityscapes (requires manual registration at https://www.cityscapes-dataset.com/)
# or use the smaller HuggingFace fallback for ablation experiments
python datasets/download_cityscapes.py --output data/cityscapes
python datasets/download_cityscapes.py --output data/cityscapes --hf-fallback
```

If you already have the datasets elsewhere, set the corresponding paths in your config files instead of copying them into the repository.

## Quick Start: Full Pipeline

Run the complete CLEAR-MoE++ pipeline from dense baseline to dispatch benchmarking in one command:

```bash
python run_ablation.py --config configs/ablation_deit_s.yaml
```

This will:
1. Load DeiT-Small pretrained backbone
2. Run calibration pass on 200 Imagenette images
3. Score all 12 FFN layers
4. Extract 6 high-scoring late-stage layers
5. Fit routers and benchmark 7 router variants
6. Measure 6 dispatch backends across 4 imbalance scenarios
7. Report classification & segmentation accuracy
8. Save all results to `outputs/`

**Expected Runtime:** 2-3 hours on GTX 960

## Pipeline Stages (Detailed)

### Stage 1: Dense Baseline
Evaluate pretrained dense model to establish quality/latency ceiling.

```python
from clear_moe.models import load_deit_model
model = load_deit_model('deit_small_patch16_224')
# Results: Top-1 86.69%, Top-5 98.48%, latency 11.37 ms
```

### Stage 2: Calibration Pass
Capture FFN layer activations from 200 calibration images without weight updates.

```python
from clear_moe.calibration import capture_activations
activations = capture_activations(
    model=model,
    data_loader=calib_loader,
    layer_names=['layer_6', 'layer_7', ...],
    output_dir='outputs/activations/'
)
```

### Stage 3: Layer Scoring & Selection
Score each FFN layer by activation multimodality + sparsity - output sensitivity.

```python
from clear_moe.scoring import score_layers
scores = score_layers(
    activations=activations,
    policy='last_half',  # Select back 6 of 12 layers
    alpha=0.4, beta=0.4, gamma=0.2
)
# Result: Layers 6-11 selected for extraction
```

### Stage 4: Expert Extraction
Decompose selected layers into shared basis (SVD) + residual experts (k-means).

```python
from clear_moe.extraction import extract_experts
moe_model = extract_experts(
    dense_model=model,
    layer_scores=scores,
    num_experts=4,
    basis_rank=192,  # 99.2% variance retention
    clusters=4
)
# Result: 6 layers now have 4 experts each + shared basis
```

### Stage 5: Router Fitting
Assign tokens to expert clusters using learned lightweight routers.

```python
from clear_moe.router import fit_routers
routers = fit_routers(
    activations=activations,
    clusters=cluster_assignments,
    epochs=5,
    learning_rate=1e-3
)
```

### Stage 6: Inference & Evaluation
Run MoE model on validation set with different routing strategies.

```python
# Hard routing (top-1)
moe_result = evaluate_moe(
    model=moe_model,
    routers=routers,
    data_loader=val_loader,
    routing='hard_top_1'
)
# Result: 86.51% Top-1 (99.8% retention), 20.95 ms latency

# Soft routing (all experts active)
soft_result = evaluate_moe(
    model=moe_model,
    routers=routers,
    data_loader=val_loader,
    routing='soft_all'
)

# Elastic routing (confidence-based threshold)
elastic_result = evaluate_moe(
    model=moe_model,
    routers=routers,
    data_loader=val_loader,
    routing='elastic',
    threshold=0.2
)
```

### Stage 7: Dispatch Benchmarking
Compare 6 dispatch backends across 4 token-load imbalance levels.

```python
from clear_moe.dispatch import benchmark_backends
results = benchmark_backends(
    backends=['naive', 'grouped', 'stream', 'triton', 'cublas'],
    token_batch_size=1568,
    expert_count=4,
    imbalance_levels=[0, 40, 60, 80]  # %
)
# Result: cuBLAS peaks at 243k tok/s (0% imbalance)
#         Triton holds 176k tok/s across all imbalances
```

### Stage 8: Multi-Device Scaling (Simulated)
Estimate performance under data-parallel, expert-parallel, and pipeline-parallel execution.

```python
from clear_moe.runtime import simulate_parallelism
scaling = simulate_parallelism(
    single_device_throughput=225455,
    num_devices=2,
    strategies=['ep', 'dp', 'pp']
)
# Result: EP-2 achieves 174k tok/s (39% efficiency, 22% comm overhead)
#         DP-2 achieves 72.5k tok/s (16% efficiency, AllReduce bottleneck)
#         PP-2 achieves 7.6k tok/s (2% efficiency, pipeline bubble)
```

### Stage 9: Roofline Analysis
Identify whether operations are compute-bound or memory-bound.

```python
from clear_moe.runtime import roofline_analysis
roofline = roofline_analysis(
    operations=['dense_ffn', 'expert_gemm', 'router_gate', 'token_sort'],
    gpu_name='gtx960'
)
# Result: Dense FFN (AI=74.3) compute-bound
#         Expert GEMM (AI=22.7) just above ridge point
#         Router gate (AI=1.9) memory-bound (key bottleneck)
```

### Stage 10: Ablation Study
Sweep over 6 design axes (24 configurations total).

```bash
python run_ablation.py --config configs/ablation_deit_s.yaml
```

**Ablation axes:**
- Layer strategy: `last_half`, `score_top_k`, `all_layers`, `alternating`
- Expert count E: 2, 4, 8, 16
- Basis rank r: 8, 16, 32, 64, 128
- Router depth: 0 (linear), 1 (MLP-1), 2 (MLP-2)
- Calibration size: 50, 200, 500, 2000 images
- MoE architecture: flat_hard, H-MoE, Soft-MoE, Elastic-MoE

## Dataset Information

The datasets used by CLEAR-MoE++ are expected to live on your local machine and are intentionally not tracked in GitHub.

### Imagenette (Classification)
- **Source:** 10-class ImageNet subset (very distinct classes)
- **Size:** 13,276 clean images after preprocessing
- **Split:** 9,391 train / 3,885 validation
- **Resolution:** 224×224 (resized with Lanczos interpolation)
- **Preprocessing:** SHA-256 dedup + robust z-score + Isolation Forest outlier removal
- **Normalization:** ImageNet standard (μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225])
- **Label mapping:** Imagenette 10-class indices → ImageNet-1K indices (e.g., "tench" = 0)

### Cityscapes (Segmentation)
- **Source:** Urban driving scenes with pixel-level annotations
- **Size:** 5,000 total (200 calibration, 500 validation)
- **Resolution:** 512×512 (downsampled from 2048×1024)
- **Classes:** 19 semantic categories (road, car, pedestrian, etc.)
- **Metric:** mIoU (mean Intersection-over-Union across all classes)

## Key Metrics & Definitions

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **Top-1 Accuracy** | % of test images where highest confidence prediction = ground truth | Primary classification quality metric |
| **Top-5 Accuracy** | % of images where correct label in top-5 predictions | Robustness indicator |
| **mIoU** | Mean Intersection-over-Union across all segmentation classes | Dense prediction quality (0=bad, 1=perfect) |
| **Accuracy Retention** | MoE Top-1 / Dense Top-1 × 100% | How much classification quality preserved |
| **p50 Latency** | Median wall-clock time over 100 runs (ms) | Robust to outliers (not mean) |
| **Throughput** | Images/second (img/s) or tokens/second (tok/s) | Higher = faster system |
| **Active-param Reduction** | (1 - k/E) × expert_fraction × 100% | Percentage parameter savings |
| **Hotness Skew** | max_experts / mean_experts | Load imbalance (perfect=1.0, imbalanced>>2.0) |
| **Arithmetic Intensity** | FLOPs per byte of memory accessed | High AI = compute-bound, low = memory-bound |

## Router Architectures (7 Variants)

1. **Linear Router** - Simple matrix multiply (384→4), fastest, lowest capacity
2. **MLP Router** - 384→1536→4, adds capacity with modest overhead
3. **Expert-Choice Router** - Experts select tokens (not vice-versa), deterministic capacity
4. **Communication-Aware Router** - Penalizes costly expert assignments
5. **Path-Consistent Router** - Cross-layer coherence (same tokens to same experts)
6. **Self-Routing** - No learnable parameters, subspace projection
7. **ALFRouter (Novel)** - Auxiliary-Loss-Free, bias-based capacity control, best load balance

## Dispatch Backends (6 Strategies)

| Backend | Token Handling | Imbalance Robustness | Best Use Case |
|---------|----------------|----------------------|---|
| **Naive Sequential** | Loop over experts one at a time | Poor | Baseline only (slow) |
| **Route-Sorted Grouped** | Sort by assigned expert, batch by expert | Medium | Low-moderate imbalance |
| **CUDA Stream (3-stream)** | Temporal overlap via multiple CUDA streams | Medium | Balanced to moderate imbalance |
| **Expert Fusion** | Fuse expert kernels | Poor at high imbalance | Moderate imbalance only |
| **Triton Scatter-Gather** | Custom fused kernel (OpenAI Triton) | High (imbalance-agnostic) | Production (unpredictable load) |
| **cuBLAS Batched-GEMM** | Stack all experts into single 3D tensor | Very poor at high imbalance | Balanced load only (peak perf) |

**Performance Summary (GTX 960):**
- **0% imbalance:** cuBLAS wins (243k tok/s)
- **40% imbalance:** cuBLAS still fast (202k tok/s) but Triton competitive
- **60% imbalance:** Triton surpasses (180k tok/s vs 156k cuBLAS)
- **80% imbalance:** Triton dominates (179k tok/s vs 125k cuBLAS collapses from padding)

## Configuration Files

Each config is a YAML file specifying the full pipeline:

```yaml
# configs/deit_s_imagenet.yaml
model:
  name: 'deit_small_patch16_224'
  pretrained: true

task: 'classification'
dataset: 'imagenette'
calibration_size: 200

extraction:
  num_experts: 4
  basis_rank: 192  # 99.2% variance
  clusters: 4
  layer_strategy: 'last_half'

router:
  type: 'linear'
  depth: 0
  learning_rate: 1e-3
  epochs: 5

dispatch:
  backend: 'cublas'
  batch_size: 8

evaluation:
  val_split_size: 3885
  compute_latency: true
  compute_throughput: true
```

## Important Notes

### Label Mapping (Critical for ImageNet/Imagenette)
Imagenette classes must be mapped to ImageNet-1K indices during evaluation:
```python
IMAGENETTE_CLASSES = {
    0: 'tench',           # ImageNet class 0
    1: 'english springer', # ImageNet class 2
    2: 'cassette player',  # ImageNet class 482
    ...
}
# When evaluating, map predicted logits[0:10] to their ImageNet indices
```

### Shared Basis Initialization (Critical for Transfer)
Always initialize shared FFN weights from the original dense model:
```python
moe_layer.shared_fc1.weight.copy_(dense_layer.fc1.weight)
moe_layer.shared_fc1.bias.copy_(dense_layer.fc1.bias)
# NOT random initialization!
```

### PCIe Overhead (Heterogeneous Serving)
Disaggregated CPU-router + GPU-expert execution incurs 5.04 ms per batch:
- **At batch size 1:** 5 ms overhead per image (impractical)
- **At batch size 32:** ~0.15 ms overhead per image (acceptable)
- **Recommendation:** Keep model on single device for interactive inference

### GPU Memory Requirements
- **DeiT-Small MoE:** ~2 GB VRAM (fits on GTX 960 with batch size 8)
- **SegFormer-B0 MoE:** ~1.5 GB VRAM
- **Headroom:** Leave ~0.5 GB for PyTorch overhead and kernel buffers

## Output Artifacts

After running the pipeline, results are saved to `outputs/`:

```
outputs/
├── full_runs/logs/
│   ├── dense_baseline_results.json       # Dense model metrics
│   ├── activation_stats.json             # Calibration pass statistics
│   ├── layer_scores.json                 # Layer selection scores
│   ├── extraction_summary.json           # Expert decomposition results
│   ├── routing_stats.json                # Router assignment statistics
│   └── moe_cls_results.json              # Classification evaluation
├── ablations/
│   └── 20260423_032158/                 # Ablation timestamp
│       ├── layer_strategy_*.json
│       ├── expert_count_*.json
│       ├── basis_rank_*.json
│       └── ...
├── dispatch_benchmarks/
│   └── dispatch_results_*.json           # Backend performance comparison
└── roofline/
    └── roofline_analysis.json            # Compute vs memory-bound analysis
```

## Reproducing Results

### To reproduce exact paper results:

```bash
# Step 1: EDA and preprocessing (already done, skip)
python scripts/01_eda_full_pipeline.py

# Step 2: Dense baseline
python scripts/03_dense_baseline.py --config configs/deit_s_imagenet.yaml

# Step 3: Calibration
python scripts/04_calibration_pass.py --config configs/deit_s_imagenet.yaml

# Step 4: Layer scoring
python scripts/05_layer_scoring.py --config configs/deit_s_imagenet.yaml

# Step 5: Expert extraction
python scripts/06_expert_extraction.py --config configs/deit_s_imagenet.yaml

# Step 6: Router fitting
python scripts/07_router_fitting.py --config configs/deit_s_imagenet.yaml

# Step 7: Dispatch benchmarking
python scripts/08_dispatch_benchmark.py

# Step 8: Parallel scaling (simulated)
python scripts/09_parallel_scaling.py

# Step 9: Roofline analysis
python scripts/10_roofline_analysis.py

# Or run all at once:
python run_ablation.py --config configs/ablation_deit_s.yaml
```

## Testing

Run the test suite to verify correctness:

```bash
pytest tests/ -v
pytest tests/ -v -k "test_extraction"  # Specific test
pytest tests/ --cov=clear_moe          # With coverage report
```

## Troubleshooting

**Q: "CUDA out of memory" on GTX 960?**
- Reduce batch size: `--batch_size 4` (default 8)
- Clear GPU cache: `torch.cuda.empty_cache()` after each stage

**Q: Accuracy drops to ~9.5% after extraction?**
- Check shared basis initialization (must copy from dense model)
- Verify label mapping for Imagenette (critical!)
- Ensure calibration activations were saved correctly

**Q: Latency much slower than expected?**
- Verify dispatch backend (use `--dispatch cublas` for peak performance)
- Check GPU utilization: `nvidia-smi` (should see high usage)
- Reduce router depth for faster inference: `--router_depth 0`

**Q: ImportError for timm or transformers?**
- Reinstall: `pip install --upgrade timm transformers`
- Verify torch version matches CUDA: `python -c "import torch; print(torch.version.cuda)"`

## Citation

```bibtex
@article{hossain2026clearmoe,
  title={CLEAR-MoE++: Calibration-Driven Layer-Selective Expert Extraction from Pretrained Vision Transformers with Parallelism-Aware Routing and Runtime Co-Design},
  author={Hossain, Md. Irtiza},
  journal={arXiv preprint arXiv:2605.xxxxx},
  year={2026}
}
```

## Contact & Support

For questions, issues, or contributions:
- Check existing GitHub issues
- Review the `docs/` folder for detailed methodology
- Examine test cases in `tests/` for usage examples

## License

This project is provided for research and educational purposes.

---

**Last Updated:** May 10, 2026  
**Status:** Project Complete | Paper in Preparation
