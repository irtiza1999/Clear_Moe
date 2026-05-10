#!/usr/bin/env python3
"""
Script 00 - Detailed EDA + Multi-Method Preprocessing Selection

What this script does:
1. Scans an ImageFolder-style dataset (train/val or flat class folders)
2. Runs EDA with multiple methods per step
3. Compares candidate preprocessing strategies and selects one using explicit criteria
4. Exports detailed reports (JSON + Markdown)
5. Materializes a cleaned dataset (hardlinks/copy) for training

Example:
    python scripts/00_eda_preprocess.py \
        --dataset_root data/imagenette2-320 \
        --output_dir outputs/eda_preprocess \
        --image_size 224 \
        --materialize
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from scipy.stats import ks_2samp, ttest_ind, wasserstein_distance
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import silhouette_score


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)


@dataclass
class Sample:
    path: Path
    split: str
    cls: str


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _discover_dataset(dataset_root: Path) -> List[Sample]:
    samples: List[Sample] = []

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    if train_dir.exists() and val_dir.exists():
        for split_name, split_dir in (("train", train_dir), ("val", val_dir)):
            for cls_dir in split_dir.iterdir():
                if not cls_dir.is_dir():
                    continue
                cls = cls_dir.name
                for img_path in _iter_images(cls_dir):
                    samples.append(Sample(path=img_path, split=split_name, cls=cls))
        return samples

    class_dirs = [p for p in dataset_root.iterdir() if p.is_dir()]
    for cls_dir in class_dirs:
        cls = cls_dir.name
        for img_path in _iter_images(cls_dir):
            samples.append(Sample(path=img_path, split="all", cls=cls))
    return samples


def _pil_verify(path: Path) -> Tuple[bool, Optional[str]]:
    try:
        with Image.open(path) as img:
            img.verify()
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _pil_strict_decode(path: Path) -> Tuple[bool, Optional[str]]:
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            _ = np.asarray(rgb)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _dhash(path: Path, hash_size: int = 8) -> Optional[int]:
    try:
        with Image.open(path) as img:
            gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            arr = np.asarray(gray, dtype=np.int16)
            diff = arr[:, 1:] > arr[:, :-1]
            bits = diff.flatten()
            out = 0
            for b in bits:
                out = (out << 1) | int(b)
            return out
    except Exception:
        return None


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _image_features(path: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as img:
            return _image_features_from_pil(img)
    except Exception:
        return None


def _image_features_from_pil(img: Image.Image) -> Optional[np.ndarray]:
    try:
        rgb = img.convert("RGB")
        w, h = rgb.size

        # Resize for stable feature extraction speed.
        small = rgb.resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0

        means = arr.reshape(-1, 3).mean(axis=0)
        stds = arr.reshape(-1, 3).std(axis=0)

        gray = np.asarray(small.convert("L"), dtype=np.float32) / 255.0
        entropy = _hist_entropy(gray)

        # Blur proxy: variance of high-pass residual.
        blur_arr = np.asarray(small.filter(ImageFilter.GaussianBlur(radius=1.5)).convert("L"), dtype=np.float32) / 255.0
        edge_energy = float(np.var(gray - blur_arr))

        # Aspect and area (log-space for stability).
        aspect = float(w / max(h, 1))
        area_log = math.log(max(w * h, 1))

        feats = np.array([
            means[0], means[1], means[2],
            stds[0], stds[1], stds[2],
            entropy,
            edge_energy,
            aspect,
            area_log,
        ], dtype=np.float32)
        return feats
    except Exception:
        return None


def _hist_entropy(gray: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(gray, bins=bins, range=(0.0, 1.0), density=True)
    hist = hist + 1e-12
    hist = hist / hist.sum()
    return float(-(hist * np.log2(hist)).sum())


def _normalize_score(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return [0.0] * len(values)
    return [float((v - lo) / (hi - lo)) for v in arr]


def _summary_counter(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda x: x[0])}


def _sample_for_compute(items: List[Sample], max_n: int, seed: int) -> List[Sample]:
    if len(items) <= max_n:
        return items
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx[:max_n]]


def _compute_dataset_stats(samples: List[Sample], seed: int, max_n: int = 4000) -> Dict[str, List[float]]:
    sampled = _sample_for_compute(samples, max_n=max_n, seed=seed)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for s in sampled:
        with Image.open(s.path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        flat = arr.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_sq_sum += (flat ** 2).sum(axis=0)
        pixel_count += flat.shape[0]

    mean = channel_sum / max(pixel_count, 1)
    var = channel_sq_sum / max(pixel_count, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return {
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "sampled_images": [len(sampled)],
    }


def _normalization_quality(samples: List[Sample], mean: np.ndarray, std: np.ndarray, seed: int, max_n: int = 1500) -> Dict[str, float]:
    sampled = _sample_for_compute(samples, max_n=max_n, seed=seed)
    ch_mean = np.zeros(3, dtype=np.float64)
    ch_std = np.zeros(3, dtype=np.float64)

    for s in sampled:
        with Image.open(s.path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        z = (arr - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
        flat = z.reshape(-1, 3)
        ch_mean += flat.mean(axis=0)
        ch_std += flat.std(axis=0)

    ch_mean /= max(len(sampled), 1)
    ch_std /= max(len(sampled), 1)

    mean_abs = float(np.abs(ch_mean).mean())
    std_delta = float(np.abs(ch_std - 1.0).mean())
    score = mean_abs + std_delta
    return {
        "mean_abs_after_norm": mean_abs,
        "std_delta_after_norm": std_delta,
        "quality_score_lower_is_better": score,
    }


def _resize_quality(samples: List[Sample], size: int, method: Image.Resampling, seed: int, max_n: int = 800) -> Dict[str, float]:
    sampled = _sample_for_compute(samples, max_n=max_n, seed=seed)
    entropies: List[float] = []
    edge_energies: List[float] = []

    for s in sampled:
        with Image.open(s.path) as img:
            rgb = img.convert("RGB").resize((size, size), method)
            gray = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
            entropies.append(_hist_entropy(gray))
            blur_gray = np.asarray(rgb.filter(ImageFilter.GaussianBlur(radius=1.5)).convert("L"), dtype=np.float32) / 255.0
            edge_energies.append(float(np.var(gray - blur_gray)))

    return {
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_edge_energy": float(np.mean(edge_energies)) if edge_energies else 0.0,
    }


def _stratified_split(samples: List[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    labels = [s.cls for s in samples]
    idx = np.arange(len(samples))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(idx, labels))
    train_samples = [Sample(path=samples[i].path, split="train", cls=samples[i].cls) for i in train_idx]
    val_samples = [Sample(path=samples[i].path, split="val", cls=samples[i].cls) for i in val_idx]
    return train_samples, val_samples


def _kl_uniform(counter: Counter) -> float:
    counts = np.array(list(counter.values()), dtype=np.float64)
    probs = counts / max(counts.sum(), 1.0)
    u = np.full_like(probs, 1.0 / max(len(probs), 1))
    probs = np.clip(probs, 1e-12, 1.0)
    return float((probs * np.log(probs / u)).sum())


def _imbalance_ratio(counter: Counter) -> float:
    vals = list(counter.values())
    if not vals:
        return 1.0
    return float(max(vals) / max(min(vals), 1))


def _dedupe_phash_candidates(samples: List[Sample], max_pairs_per_bucket: int = 4000, threshold: int = 3) -> List[Tuple[str, str, int]]:
    by_prefix: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for s in samples:
        h = _dhash(s.path)
        if h is None:
            continue
        prefix = h >> 48
        by_prefix[prefix].append((str(s.path), h))

    near_dups: List[Tuple[str, str, int]] = []
    for _, bucket in by_prefix.items():
        if len(bucket) < 2:
            continue
        pair_count = 0
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                d = _hamming(bucket[i][1], bucket[j][1])
                if d <= threshold:
                    near_dups.append((bucket[i][0], bucket[j][0], d))
                pair_count += 1
                if pair_count >= max_pairs_per_bucket:
                    break
            if pair_count >= max_pairs_per_bucket:
                break
    return near_dups


def _write_manifest_csv(path: Path, rows: List[Tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class"])
        for rel_path, cls in rows:
            w.writerow([rel_path, cls])


def _materialize_dataset(
    src_root: Path,
    dst_root: Path,
    train_rows: List[Tuple[str, str]],
    val_rows: List[Tuple[str, str]],
    prefer_hardlink: bool = True,
) -> Dict[str, int]:
    copied = 0
    linked = 0

    def _put(rows: List[Tuple[str, str]], split: str) -> None:
        nonlocal copied, linked
        for rel_path, cls in rows:
            src = src_root / rel_path
            dst = dst_root / split / cls / Path(rel_path).name
            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists():
                continue

            if prefer_hardlink:
                try:
                    os.link(src, dst)
                    linked += 1
                    continue
                except Exception:
                    pass

            shutil.copy2(src, dst)
            copied += 1

    _put(train_rows, "train")
    _put(val_rows, "val")
    return {"linked": linked, "copied": copied}


def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a / (b + eps))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _jsd_from_probs(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m))


def _psi_from_probs(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, 1e-12, 1.0)
    a = np.clip(actual, 1e-12, 1.0)
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_hist_probs(a: np.ndarray, b: np.ndarray, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if hi - lo < 1e-12:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa.astype(np.float64) + 1e-12
    pb = pb.astype(np.float64) + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    return pa, pb


def _feature_shift_metrics(
    ref_features: np.ndarray,
    cmp_features: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(feature_names):
        a = ref_features[:, i]
        b = cmp_features[:, i]
        pa, pb = _numeric_hist_probs(a, b, bins=20)

        ks = ks_2samp(a, b)
        tt = ttest_ind(a, b, equal_var=False)
        w_dist = wasserstein_distance(a, b)

        out[name] = {
            "psi": _psi_from_probs(pa, pb),
            "jsd": _jsd_from_probs(pa, pb),
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "ttest_stat": float(tt.statistic),
            "ttest_pvalue": float(tt.pvalue),
            "wasserstein": float(w_dist),
            "ref_mean": float(np.mean(a)),
            "cmp_mean": float(np.mean(b)),
            "ref_std": float(np.std(a)),
            "cmp_std": float(np.std(b)),
        }
    return out


def _extract_feature_matrix(samples: List[Sample], root: Path, max_n: int, seed: int) -> Tuple[np.ndarray, List[str], List[str]]:
    sampled = _sample_for_compute(samples, max_n=max_n, seed=seed)
    feats: List[np.ndarray] = []
    labels: List[str] = []
    rels: List[str] = []
    for s in sampled:
        f = _image_features(s.path)
        if f is not None:
            feats.append(f)
            labels.append(s.cls)
            rels.append(_safe_rel(s.path, root))

    if not feats:
        return np.zeros((0, 10), dtype=np.float32), [], []
    return np.stack(feats, axis=0), labels, rels


def _save_class_distribution_svg(counter: Counter, title: str, out_path: Path) -> None:
    names = sorted(counter.keys())
    vals = [counter[k] for k in names]

    plt.figure(figsize=(11, 4.5))
    plt.bar(names, vals)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def _save_embedding_svgs(
    features: np.ndarray,
    labels: Sequence[str],
    out_dir: Path,
    seed: int,
    prefix: str,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if len(features) < 5:
        return out

    unique_labels = sorted(set(labels))
    label_to_int = {k: i for i, k in enumerate(unique_labels)}
    y = np.array([label_to_int[l] for l in labels], dtype=np.int32)

    # Linear dimensionality reduction: PCA.
    pca = PCA(n_components=2, random_state=seed)
    emb_pca = pca.fit_transform(features)
    pca_path = out_dir / f"{prefix}_pca.svg"
    plt.figure(figsize=(7, 6))
    plt.scatter(emb_pca[:, 0], emb_pca[:, 1], c=y, s=7, alpha=0.65, cmap="tab10")
    plt.title(f"{prefix.upper()} PCA (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(pca_path, format="svg")
    plt.close()
    out["pca"] = str(pca_path)

    # Non-linear dimensionality reduction: t-SNE.
    tsne_input = features
    tsne_labels = y
    if len(features) > 3000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(features), size=3000, replace=False)
        tsne_input = features[idx]
        tsne_labels = y[idx]

    perplexity = min(30, max(5, (len(tsne_input) - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca", learning_rate="auto")
    emb_tsne = tsne.fit_transform(tsne_input)
    tsne_path = out_dir / f"{prefix}_tsne.svg"
    plt.figure(figsize=(7, 6))
    plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=tsne_labels, s=7, alpha=0.65, cmap="tab10")
    plt.title(f"{prefix.upper()} t-SNE (2D)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(tsne_path, format="svg")
    plt.close()
    out["tsne"] = str(tsne_path)

    return out


def _cluster_metrics(features: np.ndarray, n_clusters: int, seed: int) -> Dict[str, float]:
    if len(features) < max(20, n_clusters * 2):
        return {"kmeans_inertia": 0.0, "silhouette": 0.0, "n_clusters": n_clusters}

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(features)
    sil = silhouette_score(features, labels) if len(set(labels)) > 1 else 0.0
    return {
        "kmeans_inertia": float(km.inertia_),
        "silhouette": float(sil),
        "n_clusters": int(n_clusters),
    }


def _load_external_features(
    external_dataset_root: Optional[str],
    seed: int,
    max_n: int,
) -> Tuple[Optional[np.ndarray], Optional[List[str]], str, Dict[str, int]]:
    # Option A: user-provided image folder path.
    if external_dataset_root:
        ext_root = Path(external_dataset_root).resolve()
        if ext_root.exists():
            ext_samples = _discover_dataset(ext_root)
            X_ext, y_ext, _ = _extract_feature_matrix(ext_samples, ext_root, max_n=max_n, seed=seed)
            return X_ext, y_ext, str(ext_root), dict(Counter([s.cls for s in ext_samples]))

    # Option B: lightweight external baseline (CIFAR-100 test split).
    try:
        from torchvision.datasets import CIFAR100

        ds = CIFAR100(root="./data", train=False, download=True)
        idx = np.arange(len(ds))
        if len(idx) > max_n:
            rng = np.random.default_rng(seed)
            idx = rng.choice(idx, size=max_n, replace=False)

        feats: List[np.ndarray] = []
        labels: List[str] = []
        for i in idx:
            img, y = ds[int(i)]
            f = _image_features_from_pil(img)
            if f is not None:
                feats.append(f)
                labels.append(str(y))

        if feats:
            return np.stack(feats, axis=0), labels, "CIFAR100:test", dict(Counter(labels))
    except Exception:
        pass

    return None, None, "none", {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed EDA and preprocessing before training")
    parser.add_argument("--dataset_root", type=str, required=True, help="ImageFolder root path")
    parser.add_argument("--output_dir", type=str, default="outputs/eda_preprocess", help="Output report directory")
    parser.add_argument("--image_size", type=int, default=224, help="Training input image size")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio for re-splitting")
    parser.add_argument("--max_feature_images", type=int, default=12000, help="Max images for expensive feature computation")
    parser.add_argument("--outlier_frac", type=float, default=0.01, help="Expected outlier fraction for IsolationForest")
    parser.add_argument("--external_dataset_root", type=str, default=None, help="Optional external dataset ImageFolder root for generalizability EDA")
    parser.add_argument("--viz_sample_size", type=int, default=3000, help="Max samples used for PCA/t-SNE visualizations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--materialize", action="store_true", help="Create cleaned train/val dataset folder")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.output_dir).resolve() / f"eda_{_now_tag()}"
    out_root.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    samples = _discover_dataset(dataset_root)
    if not samples:
        raise RuntimeError("No images found in dataset_root")

    total_images = len(samples)
    class_counts_all = Counter([s.cls for s in samples])
    split_counts = Counter([s.split for s in samples])

    # 1) File integrity check with two methods.
    integrity_fail_verify: List[Tuple[str, str]] = []
    integrity_fail_decode: List[Tuple[str, str]] = []
    valid_after_integrity: List[Sample] = []

    for s in samples:
        ok_v, err_v = _pil_verify(s.path)
        ok_d, err_d = _pil_strict_decode(s.path)

        if not ok_v:
            integrity_fail_verify.append((_safe_rel(s.path, dataset_root), err_v or "unknown"))
        if not ok_d:
            integrity_fail_decode.append((_safe_rel(s.path, dataset_root), err_d or "unknown"))

        if ok_v and ok_d:
            valid_after_integrity.append(s)

    # 2) Duplicate detection: exact hash and near-duplicate dHash.
    hash_to_paths: Dict[str, List[str]] = defaultdict(list)
    for s in valid_after_integrity:
        md5 = _file_md5(s.path)
        hash_to_paths[md5].append(_safe_rel(s.path, dataset_root))

    exact_dup_groups = [v for v in hash_to_paths.values() if len(v) > 1]
    exact_dup_members = set()
    for g in exact_dup_groups:
        for p in g[1:]:
            exact_dup_members.add(p)

    nd_input = _sample_for_compute(valid_after_integrity, max_n=min(args.max_feature_images, 8000), seed=args.seed)
    near_dups = _dedupe_phash_candidates(nd_input)
    near_dup_members = set()
    for a, b, _ in near_dups:
        rel_a = _safe_rel(Path(a), dataset_root)
        rel_b = _safe_rel(Path(b), dataset_root)
        near_dup_members.add(rel_b)
        near_dup_members.add(rel_a)

    # 3) Feature extraction + outlier detection (2 methods).
    feat_input = _sample_for_compute(valid_after_integrity, max_n=args.max_feature_images, seed=args.seed)
    feature_rows: List[Tuple[Sample, np.ndarray]] = []
    for s in feat_input:
        feats = _image_features(s.path)
        if feats is not None:
            feature_rows.append((s, feats))

    outliers_zscore: set[str] = set()
    outliers_iforest: set[str] = set()

    if feature_rows:
        X = np.stack([r[1] for r in feature_rows], axis=0)

        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        mad = np.where(mad < 1e-6, 1e-6, mad)
        robust_z = 0.6745 * (X - med) / mad
        robust_norm = np.abs(robust_z).max(axis=1)
        z_mask = robust_norm > 4.0
        for (s, _), is_out in zip(feature_rows, z_mask):
            if is_out:
                outliers_zscore.add(_safe_rel(s.path, dataset_root))

        iso = IsolationForest(
            contamination=max(min(args.outlier_frac, 0.2), 0.001),
            random_state=args.seed,
            n_estimators=200,
        )
        pred = iso.fit_predict(X)
        for (s, _), p in zip(feature_rows, pred):
            if p == -1:
                outliers_iforest.add(_safe_rel(s.path, dataset_root))

    # Compare outlier methods.
    overlap = outliers_zscore.intersection(outliers_iforest)
    outlier_method_table = {
        "robust_zscore": len(outliers_zscore),
        "isolation_forest": len(outliers_iforest),
        "intersection": len(overlap),
        "union": len(outliers_zscore.union(outliers_iforest)),
    }

    # Conservative choice: remove intersection only (lower false positives).
    chosen_outliers = overlap

    # 4) Build filtered set after integrity + exact dup removal + chosen outlier removal.
    remove_set = set(exact_dup_members).union(chosen_outliers)
    filtered = [s for s in valid_after_integrity if _safe_rel(s.path, dataset_root) not in remove_set]

    # 5) Split strategy comparison.
    has_official = split_counts.get("train", 0) > 0 and split_counts.get("val", 0) > 0
    official_train = [s for s in filtered if s.split == "train"]
    official_val = [s for s in filtered if s.split == "val"]

    if not has_official:
        official_train, official_val = _stratified_split(filtered, val_ratio=args.val_ratio, seed=args.seed)

    rs_train, rs_val = _stratified_split(filtered, val_ratio=args.val_ratio, seed=args.seed)

    official_train_counts = Counter([s.cls for s in official_train])
    official_val_counts = Counter([s.cls for s in official_val])
    rs_train_counts = Counter([s.cls for s in rs_train])
    rs_val_counts = Counter([s.cls for s in rs_val])

    split_eval = {
        "official": {
            "train_n": len(official_train),
            "val_n": len(official_val),
            "val_kl_to_uniform": _kl_uniform(official_val_counts) if official_val_counts else 0.0,
            "val_imbalance_ratio": _imbalance_ratio(official_val_counts) if official_val_counts else 1.0,
        },
        "stratified_shuffle": {
            "train_n": len(rs_train),
            "val_n": len(rs_val),
            "val_kl_to_uniform": _kl_uniform(rs_val_counts),
            "val_imbalance_ratio": _imbalance_ratio(rs_val_counts),
        },
    }

    # Prefer official split for benchmark comparability unless class balance is much worse.
    official_balance = split_eval["official"]["val_kl_to_uniform"] + split_eval["official"]["val_imbalance_ratio"]
    strat_balance = split_eval["stratified_shuffle"]["val_kl_to_uniform"] + split_eval["stratified_shuffle"]["val_imbalance_ratio"]
    if has_official and official_balance <= strat_balance * 1.15:
        chosen_split_name = "official"
        chosen_train, chosen_val = official_train, official_val
    else:
        chosen_split_name = "stratified_shuffle"
        chosen_train, chosen_val = rs_train, rs_val

    # 6) Normalization strategy comparison.
    ds_stats = _compute_dataset_stats(chosen_train, seed=args.seed)
    ds_mean = np.array(ds_stats["mean"], dtype=np.float64)
    ds_std = np.array(ds_stats["std"], dtype=np.float64)

    norm_eval_imagenet = _normalization_quality(chosen_train, IMAGENET_MEAN, IMAGENET_STD, seed=args.seed)
    norm_eval_dataset = _normalization_quality(chosen_train, ds_mean, ds_std, seed=args.seed)

    if norm_eval_dataset["quality_score_lower_is_better"] < norm_eval_imagenet["quality_score_lower_is_better"]:
        chosen_norm = {
            "name": "dataset_specific",
            "mean": ds_stats["mean"],
            "std": ds_stats["std"],
        }
    else:
        chosen_norm = {
            "name": "imagenet_default",
            "mean": [float(x) for x in IMAGENET_MEAN],
            "std": [float(x) for x in IMAGENET_STD],
        }

    # 7) Resize interpolation strategy comparison.
    resize_methods = {
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    resize_eval: Dict[str, Dict[str, float]] = {}
    for name, method in resize_methods.items():
        resize_eval[name] = _resize_quality(chosen_train, size=args.image_size, method=method, seed=args.seed)

    ent_scores = _normalize_score([resize_eval[k]["mean_entropy"] for k in resize_methods])
    edge_scores = _normalize_score([resize_eval[k]["mean_edge_energy"] for k in resize_methods])

    resize_composite = {}
    keys = list(resize_methods.keys())
    for i, k in enumerate(keys):
        # Joint score: preserve information (entropy) + detail (edge energy).
        resize_composite[k] = 0.5 * ent_scores[i] + 0.5 * edge_scores[i]
    chosen_resize = max(resize_composite.items(), key=lambda x: x[1])[0]

    # 8) Class imbalance remedies comparison.
    train_counts = Counter([s.cls for s in chosen_train])
    ir = _imbalance_ratio(train_counts)
    remedies = {
        "none": {"expected_stability": 1.0 / max(ir, 1.0), "complexity": 0.1},
        "class_weighted_loss": {"expected_stability": min(1.0, 1.3 / max(ir, 1.0)), "complexity": 0.3},
        "weighted_sampler": {"expected_stability": min(1.0, 1.6 / max(ir, 1.0)), "complexity": 0.5},
    }
    remedy_score = {
        k: 0.75 * v["expected_stability"] + 0.25 * (1.0 - v["complexity"])
        for k, v in remedies.items()
    }
    chosen_remedy = max(remedy_score.items(), key=lambda x: x[1])[0]

    # 9) Intra/inter-dataset EDA per publication checklist.
    feature_names = [
        "mean_r", "mean_g", "mean_b",
        "std_r", "std_g", "std_b",
        "entropy", "edge_energy", "aspect_ratio", "log_area",
    ]

    X_train, y_train, _ = _extract_feature_matrix(
        chosen_train,
        root=dataset_root,
        max_n=min(args.max_feature_images, 10000),
        seed=args.seed,
    )
    X_val, y_val, _ = _extract_feature_matrix(
        chosen_val,
        root=dataset_root,
        max_n=min(args.max_feature_images, 5000),
        seed=args.seed,
    )

    intra_shift = {}
    if len(X_train) > 10 and len(X_val) > 10:
        intra_shift = _feature_shift_metrics(X_train, X_val, feature_names)

    # External/generalizability dataset analysis (inter-dataset).
    X_ext, y_ext, ext_source, ext_class_counts = _load_external_features(
        external_dataset_root=args.external_dataset_root,
        seed=args.seed,
        max_n=min(args.max_feature_images, 5000),
    )

    inter_shift = {}
    if X_ext is not None and len(X_ext) > 10 and len(X_train) > 10:
        inter_shift = _feature_shift_metrics(X_train, X_ext, feature_names)

    # Clustering diagnostics (distribution + cluster structure).
    n_cls = max(2, len(set(y_train))) if y_train else 2
    cluster_train = _cluster_metrics(X_train, n_clusters=n_cls, seed=args.seed) if len(X_train) > 0 else {}
    cluster_val = _cluster_metrics(X_val, n_clusters=max(2, len(set(y_val))), seed=args.seed) if len(X_val) > 0 else {}
    cluster_ext = _cluster_metrics(X_ext, n_clusters=max(2, len(set(y_ext))), seed=args.seed) if X_ext is not None and y_ext else {}

    # Visualizations (vector graphics: SVG).
    viz_dir = out_root / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    _save_class_distribution_svg(Counter([s.cls for s in chosen_train]), "Train Class Distribution", viz_dir / "train_class_distribution.svg")
    _save_class_distribution_svg(Counter([s.cls for s in chosen_val]), "Validation Class Distribution", viz_dir / "val_class_distribution.svg")

    viz_embeddings = {
        "train": _save_embedding_svgs(
            features=X_train[: args.viz_sample_size],
            labels=y_train[: args.viz_sample_size],
            out_dir=viz_dir,
            seed=args.seed,
            prefix="train",
        ) if len(X_train) else {},
        "val": _save_embedding_svgs(
            features=X_val[: args.viz_sample_size],
            labels=y_val[: args.viz_sample_size],
            out_dir=viz_dir,
            seed=args.seed,
            prefix="val",
        ) if len(X_val) else {},
    }
    if X_ext is not None and y_ext:
        viz_embeddings["external"] = _save_embedding_svgs(
            features=X_ext[: args.viz_sample_size],
            labels=y_ext[: args.viz_sample_size],
            out_dir=viz_dir,
            seed=args.seed,
            prefix="external",
        )
    else:
        viz_embeddings["external"] = {}

    # 10) Export manifests and optionally materialize cleaned dataset.
    train_rows = [(_safe_rel(s.path, dataset_root), s.cls) for s in chosen_train]
    val_rows = [(_safe_rel(s.path, dataset_root), s.cls) for s in chosen_val]

    train_csv = out_root / "train_manifest.csv"
    val_csv = out_root / "val_manifest.csv"
    _write_manifest_csv(train_csv, train_rows)
    _write_manifest_csv(val_csv, val_rows)

    materialized_stats = {"linked": 0, "copied": 0}
    materialized_path = ""
    if args.materialize:
        materialized_root = out_root / "dataset_preprocessed"
        materialized_stats = _materialize_dataset(
            src_root=dataset_root,
            dst_root=materialized_root,
            train_rows=train_rows,
            val_rows=val_rows,
            prefer_hardlink=True,
        )
        materialized_path = str(materialized_root)

    # 11) Persist summary artifacts.
    summary = {
        "input": {
            "dataset_root": str(dataset_root),
            "total_images": total_images,
            "splits_detected": dict(split_counts),
            "classes": len(class_counts_all),
            "class_counts": _summary_counter(class_counts_all),
        },
        "integrity": {
            "method_a_pil_verify_failures": len(integrity_fail_verify),
            "method_b_strict_decode_failures": len(integrity_fail_decode),
            "chosen_valid_images": len(valid_after_integrity),
            "failed_examples": {
                "verify": integrity_fail_verify[:25],
                "strict_decode": integrity_fail_decode[:25],
            },
        },
        "duplicates": {
            "exact_duplicate_groups": len(exact_dup_groups),
            "exact_duplicate_members_removed": len(exact_dup_members),
            "near_duplicate_pairs_found": len(near_dups),
            "near_duplicate_members": len(near_dup_members),
            "near_duplicate_examples": near_dups[:30],
        },
        "outliers": {
            "method_counts": outlier_method_table,
            "selected_policy": "intersection_of_zscore_and_iforest",
            "removed_count": len(chosen_outliers),
            "removed_examples": sorted(list(chosen_outliers))[:50],
        },
        "split_strategy": {
            "candidates": split_eval,
            "selected": chosen_split_name,
            "selected_train_n": len(chosen_train),
            "selected_val_n": len(chosen_val),
            "selected_train_class_counts": _summary_counter(Counter([s.cls for s in chosen_train])),
            "selected_val_class_counts": _summary_counter(Counter([s.cls for s in chosen_val])),
        },
        "normalization": {
            "imagenet": norm_eval_imagenet,
            "dataset_specific": norm_eval_dataset,
            "dataset_stats": ds_stats,
            "selected": chosen_norm,
        },
        "resize_interpolation": {
            "candidates": resize_eval,
            "composite_scores": resize_composite,
            "selected": chosen_resize,
        },
        "imbalance_remedy": {
            "imbalance_ratio": ir,
            "candidates": remedies,
            "scores": remedy_score,
            "selected": chosen_remedy,
        },
        "intra_dataset_analysis": {
            "train_samples_for_features": int(len(X_train)),
            "val_samples_for_features": int(len(X_val)),
            "feature_shift_metrics": intra_shift,
            "cluster_metrics": {
                "train": cluster_train,
                "val": cluster_val,
            },
        },
        "inter_dataset_analysis": {
            "external_source": ext_source,
            "external_class_counts": ext_class_counts,
            "external_samples_for_features": int(len(X_ext)) if X_ext is not None else 0,
            "feature_shift_metrics": inter_shift,
            "cluster_metrics": {
                "external": cluster_ext,
            },
            "generalizability_note": (
                "Inter-dataset metrics available" if inter_shift else
                "No external dataset shift metrics were computed; provide --external_dataset_root for domain-relevant validation"
            ),
        },
        "visualizations": {
            "directory": str(viz_dir),
            "files": {
                "train_class_distribution": str(viz_dir / "train_class_distribution.svg"),
                "val_class_distribution": str(viz_dir / "val_class_distribution.svg"),
                "embeddings": viz_embeddings,
            },
        },
        "outputs": {
            "train_manifest": str(train_csv),
            "val_manifest": str(val_csv),
            "materialized_dataset": materialized_path,
            "materialized_stats": materialized_stats,
        },
    }

    summary_json = out_root / "eda_preprocessing_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plan_yaml = out_root / "preprocessing_plan.yaml"
    plan_yaml.write_text(
        "\n".join([
            "preprocessing:",
            f"  dataset_root: {dataset_root}",
            f"  selected_split: {chosen_split_name}",
            f"  train_manifest: {train_csv}",
            f"  val_manifest: {val_csv}",
            f"  remove_exact_duplicates: true",
            f"  remove_outliers: intersection_of_zscore_and_iforest",
            f"  resize_interpolation: {chosen_resize}",
            f"  input_size: {args.image_size}",
            f"  normalization:",
            f"    name: {chosen_norm['name']}",
            f"    mean: [{', '.join([f'{v:.6f}' for v in chosen_norm['mean']])}]",
            f"    std: [{', '.join([f'{v:.6f}' for v in chosen_norm['std']])}]",
            f"  imbalance_remedy: {chosen_remedy}",
            f"  materialized_dataset: {materialized_path}",
        ]) + "\n",
        encoding="utf-8",
    )

    md_report = out_root / "eda_preprocessing_report.md"
    md_lines = [
        "# Detailed EDA + Preprocessing Decision Report",
        "",
        "## Dataset Overview",
        f"- Root: {dataset_root}",
        f"- Total images: {total_images}",
        f"- Classes: {len(class_counts_all)}",
        f"- Split detection: {dict(split_counts)}",
        "",
        "## Integrity (2 Methods)",
        f"- PIL verify failures: {len(integrity_fail_verify)}",
        f"- Strict decode failures: {len(integrity_fail_decode)}",
        f"- Kept after integrity gate: {len(valid_after_integrity)}",
        "",
        "## Duplicates (2 Methods)",
        f"- Exact duplicate groups: {len(exact_dup_groups)}",
        f"- Exact duplicate members removed: {len(exact_dup_members)}",
        f"- Near duplicate pairs (dHash): {len(near_dups)}",
        "",
        "## Outliers (2 Methods)",
        f"- Robust z-score outliers: {len(outliers_zscore)}",
        f"- IsolationForest outliers: {len(outliers_iforest)}",
        f"- Intersection removed (chosen): {len(chosen_outliers)}",
        "",
        "## Split Strategy (2 Methods)",
        f"- Official split score: {official_balance:.6f}",
        f"- Stratified shuffle score: {strat_balance:.6f}",
        f"- Selected split strategy: {chosen_split_name}",
        f"- Selected train/val: {len(chosen_train)}/{len(chosen_val)}",
        "",
        "## Normalization (2 Methods)",
        f"- ImageNet quality score: {norm_eval_imagenet['quality_score_lower_is_better']:.6f}",
        f"- Dataset-specific quality score: {norm_eval_dataset['quality_score_lower_is_better']:.6f}",
        f"- Selected normalization: {chosen_norm['name']}",
        "",
        "## Resize Interpolation (3 Methods)",
        f"- Bilinear score: {resize_composite['bilinear']:.6f}",
        f"- Bicubic score: {resize_composite['bicubic']:.6f}",
        f"- Lanczos score: {resize_composite['lanczos']:.6f}",
        f"- Selected interpolation: {chosen_resize}",
        "",
        "## Imbalance Remedy (3 Methods)",
        f"- Imbalance ratio (max/min): {ir:.4f}",
        f"- Selected remedy: {chosen_remedy}",
        "",
        "## Intra-Dataset Analysis (Train vs Validation)",
        f"- Feature samples used (train/val): {len(X_train)}/{len(X_val)}",
        "- Metrics per feature include: PSI, JSD, KS test, Welch t-test, Wasserstein distance",
        f"- Train clustering silhouette: {cluster_train.get('silhouette', 0.0):.6f}",
        f"- Val clustering silhouette: {cluster_val.get('silhouette', 0.0):.6f}",
        "",
        "## Inter-Dataset Analysis (Generalizability)",
        f"- External source: {ext_source}",
        f"- External feature sample size: {len(X_ext) if X_ext is not None else 0}",
        "- Metrics per feature include: PSI, JSD, KS test, Welch t-test, Wasserstein distance",
        f"- External clustering silhouette: {cluster_ext.get('silhouette', 0.0):.6f}",
        f"- Note: {'inter-dataset shift computed' if inter_shift else 'no external shift computed (provide --external_dataset_root for domain-specific dataset)'}",
        "",
        "## Dimensionality Reduction + Visualization",
        "- Linear method: PCA (2D)",
        "- Non-linear method: t-SNE (2D)",
        f"- Visualization directory (SVG): {viz_dir}",
        f"- Train class distribution: {viz_dir / 'train_class_distribution.svg'}",
        f"- Val class distribution: {viz_dir / 'val_class_distribution.svg'}",
        "",
        "## Output Artifacts",
        f"- JSON summary: {summary_json}",
        f"- Train manifest: {train_csv}",
        f"- Val manifest: {val_csv}",
        f"- Preprocessing plan: {plan_yaml}",
        f"- Materialized dataset: {materialized_path if materialized_path else 'not requested'}",
        "",
        "## Notes",
        "- This workflow intentionally compares multiple methods at each stage instead of hard-coding one.",
        "- Recommended defaults were selected by explicit metric-based criteria in this run.",
    ]
    md_report.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("=" * 80)
    print("EDA + PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Input images: {total_images}")
    print(f"Kept images after filtering: {len(filtered)}")
    print(f"Selected split: {chosen_split_name} ({len(chosen_train)} train / {len(chosen_val)} val)")
    print(f"Selected normalization: {chosen_norm['name']}")
    print(f"Selected interpolation: {chosen_resize}")
    print(f"Selected imbalance remedy: {chosen_remedy}")
    print(f"External dataset source: {ext_source}")
    print(f"External shift computed: {'yes' if inter_shift else 'no'}")
    print(f"Visualization directory (SVG): {viz_dir}")
    print(f"Summary JSON: {summary_json}")
    print(f"Report Markdown: {md_report}")
    print(f"Plan YAML: {plan_yaml}")
    if materialized_path:
        print(f"Materialized dataset: {materialized_path}")
        print(f"Hardlinks: {materialized_stats['linked']}, Copies: {materialized_stats['copied']}")


if __name__ == "__main__":
    main()
