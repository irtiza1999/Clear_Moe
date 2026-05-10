"""Download/setup Cityscapes for CLEAR-MoE segmentation experiments.

Cityscapes requires registration at https://www.cityscapes-dataset.com/
Full dataset: ~11GB. Provides gtFine annotations + leftImg8bit images.

Alternatives (no registration):
  - HuggingFace segments/sidewalk-semantic (subset, 1K images)
  - Local path if already downloaded
"""
import argparse
import os
import sys
from pathlib import Path


HF_FALLBACK_DATASET = "segments/sidewalk-semantic"
CITYSCAPES_REGISTER_URL = "https://www.cityscapes-dataset.com/register/"
CITYSCAPES_DOWNLOAD_URL = "https://www.cityscapes-dataset.com/file-handling/?packageID=1"


def setup_cityscapes(output_dir: str = "data/cityscapes", use_hf_fallback: bool = False):
    """Setup Cityscapes dataset. Prints instructions if manual download needed.

    Args:
        output_dir:       destination for dataset
        use_hf_fallback:  use HuggingFace segments/sidewalk-semantic instead of full Cityscapes
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if use_hf_fallback:
        return _setup_hf_fallback(output_path)

    # Check if already downloaded
    if _verify_cityscapes(output_path):
        print(f"[cityscapes] Found existing dataset at {output_path}")
        return output_path

    # Check CITYSCAPES_DATASET env var
    env_path = os.environ.get("CITYSCAPES_DATASET")
    if env_path and _verify_cityscapes(Path(env_path)):
        print(f"[cityscapes] Using CITYSCAPES_DATASET={env_path}")
        return Path(env_path)

    _print_download_instructions(output_path)
    return None


def _verify_cityscapes(root: Path) -> bool:
    """Check if Cityscapes directory structure is valid."""
    required = [
        root / "leftImg8bit" / "train",
        root / "leftImg8bit" / "val",
        root / "gtFine" / "train",
        root / "gtFine" / "val",
    ]
    return all(p.exists() for p in required)


def _print_download_instructions(output_path: Path):
    """Print manual download instructions."""
    print("\n" + "="*70)
    print("CITYSCAPES DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print(f"\n1. Register at: {CITYSCAPES_REGISTER_URL}")
    print(f"2. Download 'gtFine_trainvaltest.zip' and 'leftImg8bit_trainvaltest.zip'")
    print(f"3. Extract both to: {output_path}")
    print(f"   Expected structure:")
    print(f"     {output_path}/leftImg8bit/train/<city>/<city>_*.png")
    print(f"     {output_path}/gtFine/train/<city>/<city>_*_gtFine_labelIds.png")
    print(f"\nOR use HuggingFace fallback (no registration, smaller subset):")
    print(f"   python datasets/download_cityscapes.py --output {output_path} --hf-fallback")
    print(f"\nOR set environment variable:")
    print(f"   export CITYSCAPES_DATASET=/path/to/cityscapes")
    print("="*70 + "\n")


def _setup_hf_fallback(output_path: Path):
    """Download segments/sidewalk-semantic from HuggingFace as Cityscapes substitute."""
    print(f"[cityscapes] Using HuggingFace fallback: {HF_FALLBACK_DATASET}")
    print(f"[cityscapes] This is a subset (~1K images), sufficient for ablation studies.")
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_FALLBACK_DATASET)
        hf_dir = output_path / "sidewalk_semantic"
        hf_dir.mkdir(exist_ok=True)
        # Save as a simple directory dataset
        ds.save_to_disk(str(hf_dir))
        print(f"[cityscapes] HF fallback saved to {hf_dir}")
        print(f"[cityscapes] Set config: data.dataset: sidewalk_semantic")
        print(f"[cityscapes] Set config: data.data_dir: {hf_dir}")
        return hf_dir
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR downloading HF fallback: {e}")
        print(f"Get HuggingFace token at https://huggingface.co/settings/tokens")
        print(f"Then: huggingface-cli login")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup Cityscapes dataset")
    parser.add_argument("--output", default="data/cityscapes",
                        help="Output directory (default: data/cityscapes)")
    parser.add_argument("--hf-fallback", action="store_true",
                        help="Use HuggingFace segments/sidewalk-semantic instead of full Cityscapes")
    args = parser.parse_args()
    result = setup_cityscapes(args.output, use_hf_fallback=args.hf_fallback)
    if result:
        print(f"\nDataset ready at: {result}")
