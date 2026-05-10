"""Download ImageNette (10-class ImageNet subset) for CLEAR-MoE experiments.

ImageNette-320: 1.4GB, 10 classes, train/val split.
Source: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
"""
import argparse
import hashlib
import os
import sys
import tarfile
from pathlib import Path


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_MD5 = "9f2c6b88b0b5d2f30ed01d7d8d3b7bc9"  # approximate
IMAGENETTE_CLASSES = [
    'n01440764',  # tench
    'n02102040',  # English springer
    'n02979186',  # cassette player
    'n03000684',  # chain saw
    'n03028079',  # church
    'n03394916',  # French horn
    'n03417042',  # garbage truck
    'n03425413',  # gas pump
    'n03445777',  # golf ball
    'n03888257',  # parachute
]
CLASS_NAMES = [
    'tench', 'english_springer', 'cassette_player', 'chain_saw',
    'church', 'french_horn', 'garbage_truck', 'gas_pump',
    'golf_ball', 'parachute'
]


def download_imagenette(output_dir: str = "data/imagenette") -> Path:
    """Download and extract ImageNette-320 dataset.

    Args:
        output_dir: destination directory (creates if not exists)

    Returns:
        Path to extracted dataset root (contains train/ and val/ subdirs)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extracted_dir = output_path / "imagenette2-320"
    if extracted_dir.exists() and (extracted_dir / "train").exists():
        print(f"[download_imagenette] Already extracted at {extracted_dir}")
        return extracted_dir

    tgz_path = output_path / "imagenette2-320.tgz"

    if not tgz_path.exists():
        print(f"[download_imagenette] Downloading from {IMAGENETTE_URL}")
        print(f"[download_imagenette] Destination: {tgz_path}")
        print(f"[download_imagenette] Size: ~1.4 GB")
        _download_with_progress(IMAGENETTE_URL, tgz_path)
    else:
        print(f"[download_imagenette] Found cached archive at {tgz_path}")

    print(f"[download_imagenette] Extracting to {output_path} ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(output_path)

    if not extracted_dir.exists():
        raise RuntimeError(f"Extraction failed — expected {extracted_dir}")

    _verify_structure(extracted_dir)
    print(f"[download_imagenette] Done. Dataset at {extracted_dir}")
    return extracted_dir


def _download_with_progress(url: str, dest: Path):
    """Download url to dest with progress bar."""
    try:
        import urllib.request
        import shutil

        def reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, int(count * block_size * 100 / total_size))
                bar = '#' * (pct // 2)
                print(f"\r  [{bar:<50}] {pct}%", end='', flush=True)

        urllib.request.urlretrieve(url, str(dest), reporthook)
        print()  # newline after progress bar
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}\n"
                           f"Manually download from {url} and place at {dest}")


def _verify_structure(root: Path):
    """Verify train/val directory structure exists."""
    for split in ['train', 'val']:
        split_dir = root / split
        if not split_dir.exists():
            raise RuntimeError(f"Missing {split_dir} in extracted dataset")
        classes_found = [d.name for d in split_dir.iterdir() if d.is_dir()]
        if len(classes_found) < 10:
            raise RuntimeError(
                f"Expected 10 class dirs in {split_dir}, found {len(classes_found)}: {classes_found}"
            )
    print(f"[download_imagenette] Structure verified: train + val splits, 10 classes each")


def get_imagenette_path(data_root: str = "data/imagenette") -> Path:
    """Return path to imagenette2-320 root, downloading if needed."""
    root = Path(data_root) / "imagenette2-320"
    if not root.exists():
        download_imagenette(data_root)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ImageNette-320 dataset")
    parser.add_argument("--output", default="data/imagenette",
                        help="Output directory (default: data/imagenette)")
    args = parser.parse_args()
    path = download_imagenette(args.output)
    print(f"\nDataset ready at: {path}")
    print(f"Use in configs: data.imagenet_dir: {path}")
