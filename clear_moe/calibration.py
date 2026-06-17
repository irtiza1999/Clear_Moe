"""
CLEAR-MoE Calibration — Data loading and activation capture for expertization.

Supports:
- ImageNet-1K / Imagenette (classification) via torchvision or HuggingFace
- Cityscapes (segmentation) via HuggingFace datasets
- Memory-efficient hook-based activation capture with disk caching

Designed for limited VRAM (GTX 960 with 2-4GB):
- Processes images one batch at a time
- Saves activations to disk incrementally
- Supports configurable calibration subset sizes
"""

import os
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────

def get_calibration_dataloader(config: Dict, transform=None, seed: int = 42) -> DataLoader:
    """
    Build a calibration DataLoader from config.

    Uses a subset of the training data for activation capture.
    Subset selection uses a fixed seed for reproducibility.

    Args:
        config: Config dict with data section
        transform: Image transform (from get_model_input_transform)
        seed: Random seed for reproducible subset selection

    Returns:
        DataLoader for calibration
    """
    data_cfg = config["data"]
    dataset_name = data_cfg["dataset"]
    calib_size = data_cfg.get("calib_size", 2000)
    batch_size = data_cfg.get("val_batch_size", 4)
    num_workers = data_cfg.get("num_workers", 2)
    seed = config.get("seed", seed)

    dataset = _load_dataset(dataset_name, split="train", config=config, transform=transform)

    # Take a reproducible random subset for calibration (training split only)
    if len(dataset) > calib_size:
        rng = torch.Generator()
        rng.manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=rng)[:calib_size].tolist()
        dataset = Subset(dataset, indices)
        logger.info(
            f"Using calibration subset of {calib_size} samples from {dataset_name} "
            f"(seed={seed})"
        )
    else:
        logger.info(f"Using full dataset of {len(dataset)} samples for calibration")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=False,
    )

    return loader


def get_eval_dataloader(config: Dict, transform=None) -> DataLoader:
    """Build evaluation DataLoader (full validation set)."""
    data_cfg = config["data"]
    dataset_name = data_cfg["dataset"]
    batch_size = data_cfg.get("val_batch_size", 4)
    num_workers = data_cfg.get("num_workers", 2)

    dataset = _load_dataset(dataset_name, split="val", config=config, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=False,
    )

    logger.info(f"Eval dataloader: {len(dataset)} samples, batch_size={batch_size}")
    return loader


def _load_dataset(
    dataset_name: str,
    split: str,
    config: Dict,
    transform=None,
) -> Dataset:
    """
    Load a dataset by name.

    Supports: imagenette, imagenet, cityscapes, cifar100 (fallback).
    """
    if dataset_name == "imagenette":
        return _load_imagenette(split, transform, config)
    elif dataset_name == "imagenet":
        return _load_imagenet(split, transform, config)
    elif dataset_name == "cityscapes":
        return _load_cityscapes(split, transform, config)
    elif dataset_name == "cifar100":
        return _load_cifar100(split, transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_imagenette(split: str, transform, config: Dict) -> Dataset:
    """
    Load Imagenette (10-class ImageNet subset) — auto-downloads.

    Tries multiple strategies:
    1. Direct URL tarball download + ImageFolder
    2. HuggingFace datasets with multiple dataset names
    3. CIFAR-100 fallback
    """
    import tarfile
    import urllib.request

    data_cfg = config.get("data", {})

    # Honor config first. Supports both:
    # - data_dir pointing directly to imagenette root with train/val
    # - data_dir parent that contains imagenette2-320
    configured_root = data_cfg.get("data_dir")
    if configured_root:
        configured_root = Path(configured_root)
        if (configured_root / "train").exists() and (configured_root / "val").exists():
            data_root = configured_root
        elif (configured_root / "imagenette2-320").exists():
            data_root = configured_root / "imagenette2-320"
        else:
            data_root = configured_root
    else:
        data_root = Path("./data/imagenette2-320")

    split_dir = data_root / ("train" if split == "train" else "val")

    # Strategy 1: Check if already downloaded
    if split_dir.exists() and any(split_dir.iterdir()):
        logger.info(f"Loading Imagenette from {split_dir}")
        from torchvision.datasets import ImageFolder
        return ImageFolder(str(split_dir), transform=transform)

    # Strategy 2: Download tarball directly
    tarball_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    tarball_path = Path("./data/imagenette2-320.tgz")

    try:
        tarball_path.parent.mkdir(parents=True, exist_ok=True)

        if not tarball_path.exists():
            logger.info(f"Downloading Imagenette from {tarball_url} ...")
            urllib.request.urlretrieve(tarball_url, str(tarball_path))
            logger.info("Download complete.")

        if not split_dir.exists():
            logger.info("Extracting Imagenette...")
            with tarfile.open(str(tarball_path), "r:gz") as tar:
                tar.extractall(path="./data")
            logger.info("Extraction complete.")

        if split_dir.exists():
            from torchvision.datasets import ImageFolder
            return ImageFolder(str(split_dir), transform=transform)

    except Exception as e:
        logger.warning(f"Failed to download/extract Imagenette: {e}")

    # Strategy 3: Try HuggingFace datasets with different names
    hf_names = ["imagenette", "iamkamleshwar/imagenette"]
    for hf_name in hf_names:
        try:
            from datasets import load_dataset as hf_load
            hf_split = "train" if split == "train" else "validation"
            hf_dataset = hf_load(hf_name, split=hf_split)
            logger.info(f"Loaded Imagenette from HuggingFace ({hf_name})")
            return HFImageDataset(hf_dataset, transform=transform, task="classification")
        except Exception:
            continue

    # Strategy 4: Fallback to CIFAR-100
    logger.warning("All Imagenette sources failed. Falling back to CIFAR-100")
    return _load_cifar100(split, transform)


def _load_imagenet(split: str, transform, config: Dict) -> Dataset:
    """
    Load ImageNet-1K.

    First tries local path, then HuggingFace datasets.
    """
    data_cfg = config["data"]
    local_dir = data_cfg.get("imagenet_dir")

    if local_dir and os.path.exists(local_dir):
        from torchvision.datasets import ImageFolder
        split_dir = os.path.join(local_dir, "val" if split == "val" else "train")
        return ImageFolder(split_dir, transform=transform)

    # Try HuggingFace (requires authentication)
    try:
        from datasets import load_dataset as hf_load
        hf_split = "train" if split == "train" else "validation"
        hf_dataset = hf_load("ILSVRC/imagenet-1k", split=hf_split, trust_remote_code=True)
        return HFImageDataset(hf_dataset, transform=transform, task="classification")
    except Exception as e:
        logger.warning(f"Failed to load ImageNet from HuggingFace: {e}")
        logger.info("Falling back to Imagenette")
        return _load_imagenette(split, transform, config)


def _load_cityscapes(split: str, transform, config: Dict) -> Dataset:
    """
    Load Cityscapes dataset.

    Tries (in order):
    1. Local path from config key 'data_dir' or 'cityscapes_dir'
       Note: torchvision.Cityscapes target_type="semantic" returns raw label
       IDs (0-33), NOT training IDs (0-18). We convert via CityscapesLabelMapper.
    2. Synthetic dataset (real Cityscapes requires manual download/license).

    For a fully self-contained segmentation evaluation use
    experiments/run_segmentation.py which evaluates ADE20K with the matching
    ADE20K-finetuned SegFormer model (no manual download required).
    """
    data_cfg = config["data"]
    # Support both 'data_dir' and legacy 'cityscapes_dir' config keys
    local_dir = data_cfg.get("data_dir") or data_cfg.get("cityscapes_dir")

    if local_dir and os.path.exists(local_dir):
        from torchvision.datasets import Cityscapes
        cs_split = "val" if split == "val" else "train"
        base_ds = Cityscapes(
            local_dir, split=cs_split, mode="fine",
            target_type="semantic", transform=transform,
        )
        logger.info(f"Loaded Cityscapes from {local_dir} ({cs_split}): {len(base_ds)} images")
        return CityscapesTrainIdDataset(base_ds)

    logger.warning(
        "No local Cityscapes data found. "
        "For real segmentation evaluation run experiments/run_segmentation.py "
        "(uses ADE20K, no license required). "
        "Falling back to synthetic data."
    )
    return SyntheticSegDataset(size=200, num_classes=19, image_size=512)


def _load_cifar100(split: str, transform) -> Dataset:
    """Load CIFAR-100 as a lightweight fallback."""
    from torchvision.datasets import CIFAR100

    is_train = split == "train"
    dataset = CIFAR100(
        root="./data", train=is_train, download=True, transform=transform
    )
    logger.info(f"Loaded CIFAR-100 ({split}): {len(dataset)} samples")
    return dataset


# ──────────────────────────────────────────────────────────────────────
# HuggingFace dataset wrappers
# ──────────────────────────────────────────────────────────────────────

class HFImageDataset(Dataset):
    """Wraps a HuggingFace dataset for classification with torchvision transforms."""

    def __init__(self, hf_dataset, transform=None, task="classification"):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.task = task

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        label = item.get("label", 0)

        if self.transform is not None:
            if callable(self.transform):
                # torchvision transform
                image = self.transform(image)
            else:
                # HuggingFace processor
                inputs = self.transform(images=image, return_tensors="pt")
                image = inputs["pixel_values"].squeeze(0)

        return image, label


class HFSegmentationDataset(Dataset):
    """Wraps a HuggingFace segmentation dataset."""

    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]
        annotation = item.get("annotation", item.get("label", None))

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform is not None:
            if callable(self.transform) and not hasattr(self.transform, "feature_extractor_type"):
                image = self.transform(image)
                if annotation is not None:
                    import torchvision.transforms.functional as F
                    annotation = torch.tensor(np.array(annotation), dtype=torch.long)
            else:
                # HuggingFace processor
                inputs = self.transform(images=image, return_tensors="pt")
                image = inputs["pixel_values"].squeeze(0)
                if annotation is not None:
                    annotation = torch.tensor(np.array(annotation), dtype=torch.long)

        if annotation is None:
            annotation = torch.zeros(1, dtype=torch.long)

        return image, annotation


class CityscapesTrainIdDataset(Dataset):
    """
    Wraps torchvision.Cityscapes to convert raw label IDs (0-33) → training IDs (0-18).

    torchvision.Cityscapes with target_type="semantic" returns the raw Cityscapes
    label pixel values (0-33 for 34 categories), NOT the training IDs (0-18 for the
    19 evaluation categories). The SegFormer and other models trained on Cityscapes
    expect training IDs. Without this conversion, mean accuracy ≈ 1/19 because only
    label IDs 0-18 pass the `target < num_classes` filter, and those IDs do not
    correspond to the model's output class space.

    Mapping source: https://github.com/mcordts/cityscapesScripts
    """

    # Maps Cityscapes label ID → training ID (255 = ignore)
    LABEL_TO_TRAINID = {
        0: 255, 1: 255, 2: 255, 3: 255, 4: 255,
        5: 255, 6: 255,
        7: 0,   # road
        8: 1,   # sidewalk
        9: 255, 10: 255,
        11: 2,  # building
        12: 3,  # wall
        13: 4,  # fence
        14: 255, 15: 255, 16: 255,
        17: 5,  # pole
        18: 255,
        19: 6,  # traffic light
        20: 7,  # traffic sign
        21: 8,  # vegetation
        22: 9,  # terrain
        23: 10, # sky
        24: 11, # person
        25: 12, # rider
        26: 13, # car
        27: 14, # truck
        28: 15, # bus
        29: 255, 30: 255,
        31: 16, # train
        32: 17, # motorcycle
        33: 18, # bicycle
    }
    _LUT = None

    @classmethod
    def _get_lut(cls):
        if cls._LUT is None:
            lut = torch.full((256,), 255, dtype=torch.long)
            for label_id, train_id in cls.LABEL_TO_TRAINID.items():
                lut[label_id] = train_id
            cls._LUT = lut
        return cls._LUT

    def __init__(self, base_dataset: Dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, target = self.base[idx]
        # target is a PIL Image with label IDs; convert to tensor then remap
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(np.array(target), dtype=torch.long)
        lut = self._get_lut()
        # Clamp to valid LUT range then index
        target_clamped = target.clamp(0, 255)
        target_train = lut[target_clamped]
        return image, target_train


class SyntheticSegDataset(Dataset):
    """Synthetic segmentation dataset for testing when real data is unavailable."""

    def __init__(self, size=200, num_classes=19, image_size=512):
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(3, self.image_size, self.image_size)
        label = torch.randint(0, self.num_classes, (self.image_size, self.image_size))
        return image, label


# ──────────────────────────────────────────────────────────────────────
# Activation capture
# ──────────────────────────────────────────────────────────────────────

class ActivationLogger:
    """
    Hook-based activation capture for FFN layers.

    Captures the input activations to each FFN layer during a forward pass.
    Designed for memory efficiency on limited VRAM GPUs:
    - Hooks capture activations and move them to CPU immediately
    - Activations are saved to disk incrementally
    - Only the current batch's activations are in GPU memory at any time

    Usage:
        logger = ActivationLogger(model, layer_names)
        logger.start()
        for batch in dataloader:
            model(batch)
        activations = logger.stop()
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        save_dir: Optional[str] = None,
    ):
        self.model = model
        self.layer_names = layer_names
        self.save_dir = Path(save_dir) if save_dir else None
        self.hooks: List[torch.utils.hooks.RemovableHook] = []
        self.activations: Dict[str, List[torch.Tensor]] = {
            name: [] for name in layer_names
        }
        self._batch_count = 0

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Register forward hooks on all target layers."""
        self.activations = {name: [] for name in self.layer_names}
        self._batch_count = 0

        for name in self.layer_names:
            module = self._get_module(name)
            if module is None:
                logger.warning(f"Could not find module: {name}")
                continue

            hook = module.register_forward_hook(
                self._make_hook(name)
            )
            self.hooks.append(hook)

        logger.info(f"Registered {len(self.hooks)} activation hooks")

    def stop(self) -> Dict[str, torch.Tensor]:
        """
        Remove hooks and return concatenated activations.

        Returns:
            Dict mapping layer name to concatenated activation tensor.
            Shape: (total_tokens, hidden_dim)
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        result = {}
        for name, act_list in self.activations.items():
            if act_list:
                result[name] = torch.cat(act_list, dim=0)
                logger.info(
                    f"  {name}: {result[name].shape} "
                    f"({result[name].nbytes / 1e6:.1f} MB)"
                )
            else:
                logger.warning(f"  {name}: no activations captured")

        logger.info(f"Captured activations from {self._batch_count} batches")
        return result

    def _make_hook(self, layer_name: str):
        """Create a forward hook that captures FFN input activations."""
        def hook_fn(module, input, output):
            # Input to FFN is typically a tuple; take the first element
            if isinstance(input, tuple):
                act = input[0]
            else:
                act = input

            # Move to CPU immediately to save GPU memory
            act_cpu = act.detach().cpu()

            # Flatten spatial dimensions: (B, ..., D) -> (B*N, D)
            if act_cpu.dim() > 2:
                act_cpu = act_cpu.reshape(-1, act_cpu.shape[-1])
            elif act_cpu.dim() == 2:
                pass  # Already (B, D) — unlikely for ViT but handle it

            self.activations[layer_name].append(act_cpu)
            self._batch_count += 1

        return hook_fn

    def _get_module(self, name: str) -> Optional[nn.Module]:
        """Get a module by its dot-separated name."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def save_all(self, output_dir: str):
        """Save all captured activations to disk."""
        save_dir = Path(output_dir) / "activations"
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, act_list in self.activations.items():
            if act_list:
                acts = torch.cat(act_list, dim=0)
                clean_name = name.replace(".", "_")
                save_path = save_dir / f"{clean_name}.pt"
                torch.save(acts, save_path)
                logger.info(f"Saved {name}: {acts.shape} to {save_path}")


def collect_activations(
    model: nn.Module,
    dataloader: DataLoader,
    layer_names: List[str],
    device: torch.device,
    save_dir: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to collect activations from a model.

    Runs the model on the calibration data and captures FFN input activations
    for all specified layers.

    Args:
        model: Pretrained model in eval mode
        dataloader: Calibration data loader
        layer_names: Names of FFN layers to capture
        device: Torch device
        save_dir: Optional directory to save activations
        max_batches: Optional limit on number of batches

    Returns:
        Dict mapping layer name to activation tensor (total_tokens, hidden_dim)
    """
    act_logger = ActivationLogger(model, layer_names, save_dir=save_dir)
    act_logger.start()

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Collecting activations")):
            if max_batches and batch_idx >= max_batches:
                break

            images = images.to(device)
            _ = model(images)

            # Free GPU memory
            del images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    activations = act_logger.stop()

    if save_dir:
        act_logger.save_all(save_dir)

    return activations
