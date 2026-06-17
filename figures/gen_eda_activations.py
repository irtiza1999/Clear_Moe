#!/usr/bin/env python3
"""
Generate EDA activation figures for the paper (Fig A1).

Extracts DeiT-Small CLS-token features from Imagenette val split,
runs PCA (2D) and t-SNE (2D), and saves PDF files to the exact path
referenced by the paper's IfFileExists guard in appendix.tex:
  outputs/eda_full/eda_20260505_184035/visualizations/val_pca.pdf
  outputs/eda_full/eda_20260505_184035/visualizations/val_tsne.pdf

Usage:
  python figures/gen_eda_activations.py
  python figures/gen_eda_activations.py --data_dir data/imagenette2-320 --seed 42
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Hardcoded output path matching appendix.tex \IfFileExists reference
OUTPUT_DIR = Path("outputs/eda_full/eda_20260505_184035/visualizations")

IMAGENETTE_CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def extract_features(model, loader, device):
    model.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = model.forward_features(imgs)
            if feats.ndim == 3:
                cls = feats[:, 0, :]
            else:
                cls = feats
            all_feats.append(cls.cpu().float().numpy())
            all_labels.extend(labels.numpy().tolist())
    return np.concatenate(all_feats, axis=0), np.array(all_labels)


def plot_embedding(emb, labels, class_names, title, xlabel, ylabel, out_path):
    n_classes = len(class_names)
    cmap = plt.cm.get_cmap("tab10", n_classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        mask = labels == i
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[cmap(i)], s=8,
                   alpha=0.65, label=name, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=8)
    legend = ax.legend(fontsize=7, markerscale=2, loc="best",
                       framealpha=0.8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/imagenette2-320")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_dir = Path(args.data_dir) / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Val dir not found: {val_dir}")

    dataset = datasets.ImageFolder(str(val_dir), transform=transform)
    class_names = [IMAGENETTE_CLASSES.get(c, c) for c in dataset.classes]
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=2, pin_memory=False)

    print("Loading DeiT-Small ...")
    model = timm.create_model("deit_small_patch16_224", pretrained=True,
                               num_classes=0)
    model = model.to(device)
    model.eval()

    print(f"Extracting features for {len(dataset)} val images ...")
    feats, labels = extract_features(model, loader, device)
    print(f"Feature matrix: {feats.shape}")  # (N, 384)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # PCA
    print("Running PCA ...")
    pca = PCA(n_components=2, random_state=args.seed)
    emb_pca = pca.fit_transform(feats)
    var_explained = pca.explained_variance_ratio_.sum() * 100
    plot_embedding(
        emb_pca, labels, class_names,
        title=f"DeiT-Small val features — PCA (var expl. {var_explained:.1f}%)",
        xlabel="PC 1",
        ylabel="PC 2",
        out_path=OUTPUT_DIR / "val_pca.pdf",
    )

    # t-SNE (cap at 3000 to keep runtime manageable)
    print("Running t-SNE ...")
    if len(feats) > 3000:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(feats), size=3000, replace=False)
        tsne_feats = feats[idx]
        tsne_labels = labels[idx]
    else:
        tsne_feats = feats
        tsne_labels = labels

    # Use PCA init for faster convergence
    pca_init = PCA(n_components=2, random_state=args.seed).fit_transform(tsne_feats)
    pca_init /= (pca_init.std() + 1e-8)  # normalise to ~N(0,1) for t-SNE init

    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        n_iter=1000,
        init=pca_init,
        learning_rate="auto",
        random_state=args.seed,
        metric="euclidean",
    )
    emb_tsne = tsne.fit_transform(tsne_feats)
    plot_embedding(
        emb_tsne, tsne_labels, class_names,
        title="DeiT-Small val features — t-SNE",
        xlabel="Dim 1",
        ylabel="Dim 2",
        out_path=OUTPUT_DIR / "val_tsne.pdf",
    )

    print("Done. Both PDFs written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
