"""
CLEAR-MoE Models — Model loading, FFN layer enumeration, and MoE wrapping.

Supports:
- DeiT / ViT models via timm
- SegFormer models via HuggingFace transformers

Provides a unified interface to:
1. Load pretrained models
2. Enumerate all FFN/MLP layers
3. Replace selected FFN layers with MoE expert modules
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class FFNLayerInfo:
    """Metadata about a single FFN/MLP layer in the backbone."""
    name: str                   # Dot-separated name path
    module: nn.Module           # Reference to the actual module
    layer_index: int            # Position in the sequence of FFN layers
    parent_name: str            # Name of parent module
    hidden_dim: int             # Input/output dimension
    intermediate_dim: int       # FFN intermediate (expanded) dimension


def load_model(config: Dict) -> nn.Module:
    """
    Load a pretrained model based on config.

    Args:
        config: Configuration dict with model.name, model.source, model.pretrained

    Returns:
        Pretrained model in eval mode
    """
    model_cfg = config["model"]
    source = model_cfg["source"]
    name = model_cfg["name"]
    pretrained = model_cfg.get("pretrained", True)

    if source == "timm":
        model = _load_timm_model(name, pretrained)
    elif source == "transformers":
        model = _load_hf_model(name, pretrained, config)
    else:
        raise ValueError(f"Unknown model source: {source}")

    model.eval()
    logger.info(f"Loaded model '{name}' from {source} (pretrained={pretrained})")

    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")

    return model


def _load_timm_model(name: str, pretrained: bool) -> nn.Module:
    """Load a model from the timm library."""
    import timm
    model = timm.create_model(name, pretrained=pretrained)
    return model


def _load_hf_model(name: str, pretrained: bool, config: Dict) -> nn.Module:
    """Load a model from HuggingFace transformers."""
    from transformers import AutoModelForSemanticSegmentation, AutoModelForImageClassification

    task = config.get("data", {}).get("task", "classification")

    if task == "segmentation":
        if pretrained:
            model = AutoModelForSemanticSegmentation.from_pretrained(name)
        else:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(name)
            model = AutoModelForSemanticSegmentation.from_config(hf_config)
    else:
        if pretrained:
            model = AutoModelForImageClassification.from_pretrained(name)
        else:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(name)
            model = AutoModelForImageClassification.from_config(hf_config)

    return model


def get_ffn_layers(model: nn.Module, config: Dict) -> List[FFNLayerInfo]:
    """
    Enumerate all FFN/MLP layers in a model.

    This function identifies the feedforward network layers that are
    candidates for expertization. It supports both timm ViT/DeiT style
    (with .mlp or .fc1/fc2 submodules) and HuggingFace SegFormer style
    (with .output.dense layers in the MLP blocks).

    Args:
        model: The pretrained model
        config: Config dict with model source info

    Returns:
        List of FFNLayerInfo describing each FFN layer
    """
    source = config["model"]["source"]
    ffn_layers = []

    if source == "timm":
        ffn_layers = _get_timm_ffn_layers(model)
    elif source == "transformers":
        task = config.get("data", {}).get("task", "classification")
        if task == "segmentation":
            ffn_layers = _get_segformer_ffn_layers(model)
        else:
            ffn_layers = _get_hf_vit_ffn_layers(model)

    logger.info(f"Found {len(ffn_layers)} FFN layers")
    for info in ffn_layers:
        logger.debug(
            f"  [{info.layer_index}] {info.name}: "
            f"hidden={info.hidden_dim}, intermediate={info.intermediate_dim}"
        )

    return ffn_layers


def _get_timm_ffn_layers(model: nn.Module) -> List[FFNLayerInfo]:
    """Extract FFN layers from timm ViT/DeiT models."""
    ffn_layers = []
    idx = 0

    for name, module in model.named_modules():
        # timm ViT uses blocks[i].mlp which contains fc1 and fc2
        if hasattr(module, "fc1") and hasattr(module, "fc2"):
            if isinstance(module.fc1, nn.Linear) and isinstance(module.fc2, nn.Linear):
                hidden_dim = module.fc1.in_features
                intermediate_dim = module.fc1.out_features
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                ffn_layers.append(FFNLayerInfo(
                    name=name,
                    module=module,
                    layer_index=idx,
                    parent_name=parent_name,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                ))
                idx += 1

    return ffn_layers


def _get_segformer_ffn_layers(model: nn.Module) -> List[FFNLayerInfo]:
    """Extract FFN/MLP layers from HuggingFace SegFormer."""
    ffn_layers = []
    idx = 0

    for name, module in model.named_modules():
        # SegFormer encoder blocks use SegformerMixFFN, which matches the
        # shared-basis + residual MoE abstraction. Skip SegformerMLP decoder
        # projections because they are single linear embeddings and do not
        # fit the two-matrix FFN replacement used by CLEAR-MoE.
        if type(module).__name__ != "SegformerMixFFN":
            continue

        dense1 = getattr(module, "dense1", None)
        dense2 = getattr(module, "dense2", None)
        if not isinstance(dense1, nn.Linear) or not isinstance(dense2, nn.Linear):
            continue

        hidden_dim = dense1.in_features
        intermediate_dim = dense1.out_features

        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        ffn_layers.append(FFNLayerInfo(
            name=name,
            module=module,
            layer_index=idx,
            parent_name=parent_name,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        ))
        idx += 1

    return ffn_layers


def _get_hf_vit_ffn_layers(model: nn.Module) -> List[FFNLayerInfo]:
    """Extract FFN layers from HuggingFace ViT/DeiT models."""
    ffn_layers = []
    idx = 0

    for name, module in model.named_modules():
        module_type = type(module).__name__
        # HF ViT uses ViTIntermediate + ViTOutput pattern, or
        # sometimes a single MLP block
        if "intermediate" in name.lower() and isinstance(module, nn.Module):
            # Check if it has a dense linear layer
            if hasattr(module, "dense") and isinstance(module.dense, nn.Linear):
                hidden_dim = module.dense.in_features
                intermediate_dim = module.dense.out_features
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                ffn_layers.append(FFNLayerInfo(
                    name=name,
                    module=module,
                    layer_index=idx,
                    parent_name=parent_name,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                ))
                idx += 1

    return ffn_layers


def replace_ffn_with_moe(
    model: nn.Module,
    layer_names: List[str],
    moe_modules: Dict[str, nn.Module],
) -> nn.Module:
    """
    Replace selected FFN layers in the model with MoE expert modules.

    This modifies the model in-place by swapping FFN layers with
    GroupedExpertMLP modules that contain shared basis + routed residual experts.

    Args:
        model: The pretrained model to modify
        layer_names: Names of FFN layers to replace
        moe_modules: Dict mapping layer name -> MoE replacement module

    Returns:
        Modified model with MoE layers
    """
    for layer_name in layer_names:
        if layer_name not in moe_modules:
            logger.warning(f"No MoE module provided for layer {layer_name}, skipping")
            continue

        moe_module = moe_modules[layer_name]

        # Navigate to parent and replace the child
        parts = layer_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        setattr(parent, parts[-1], moe_module)
        logger.info(f"Replaced {layer_name} with MoE module")

    return model


def freeze_dense_params(model: nn.Module, moe_layer_names: List[str]):
    """
    Freeze all parameters except those in MoE layers (routers, expert scales).

    This is used during router fitting to keep the dense backbone frozen
    while only training the lightweight routing components.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze MoE module parameters
    for name, param in model.named_parameters():
        for moe_name in moe_layer_names:
            if name.startswith(moe_name):
                param.requires_grad = True
                break

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Frozen model: {trainable:,} trainable / {total:,} total params "
        f"({100 * trainable / total:.2f}%)"
    )


def get_model_input_transform(config: Dict):
    """
    Get the appropriate input transform/preprocessing for evaluation.

    Returns a torchvision transform or a HuggingFace feature extractor.
    """
    source = config["model"]["source"]

    if source == "timm":
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        model_name = config["model"]["name"]
        # Create a temporary model to get data config
        data_config = resolve_data_config({}, model=model_name)
        transform = create_transform(**data_config, is_training=False)
        return transform

    elif source == "transformers":
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(config["model"]["name"])
        return processor

    return None
