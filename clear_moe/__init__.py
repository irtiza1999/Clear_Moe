"""
CLEAR-MoE: Calibration-Driven Post-Training Expert Extraction for Vision Transformers.

Core library:
- models: Model loading and FFN layer access
- calibration: Calibration data loading and activation capture
- scoring: Layer scoring for expertization selection
- extraction: Shared-basis + residual expert decomposition
- moe_builder: Assemble ClearMoEFFN modules from extracted experts
- router: Router architecture and training
- baselines: Ablation baseline FFN variants (SVD-only, disjoint, etc.)
- metrics: Evaluation metrics (accuracy, latency, energy)
- utils: Configuration, seeding, logging utilities
"""

__version__ = "0.2.0"
__author__ = "CLEAR-MoE Team"
