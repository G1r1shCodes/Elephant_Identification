"""
Embedding Diagnostic Script for Elephant Re-ID Pipeline
========================================================

Purpose: Measure whether the preprocessing pipeline preserves identity signal.
         Uses a pretrained ConvNeXt-Tiny backbone (ImageNet features) to compute
         embeddings, then measures same-ID vs different-ID distance distributions.

If separation is good → preprocessing preserves identity.
If separation is bad → crops are not identity-discriminative.

Usage:
    python embedding_diagnostic.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
import os
import random

# ==================== CONFIG ====================

# Test set: 10 identities, 4-6 images each
# These are picked from the processed dataset (diverse identities)
TEST_IDENTITIES = {
    "Makhna_1":  "data/processed_heads/Makhna/Makhna_1",
    "Makhna_2":  "data/processed_heads/Makhna/Makhna_2",
    "Makhna_5":  "data/processed_heads/Makhna/Makhna_5",
    "Makhna_9":  "data/processed_heads/Makhna/Makhna_9",
    "Makhna_4":  "data/processed_heads/Makhna/Makhna_4",
    "Herd2_AF1": "data/processed_heads/Herd/Herd_2/Adult_Female/AF_ELE_1",
    "Herd2_AF5": "data/processed_heads/Herd/Herd_2/Adult_Female/AF_ELE_5",
    "Herd3_AF1": "data/processed_heads/Herd/Herd_3/Adult_female/AF_ELE_1",
    "Herd4_AF5": "data/processed_heads/Herd/Herd_4/Adult_Female/AF_ELE_5",
    "Herd4_AF9": "data/processed_heads/Herd/Herd_4/Adult_Female/AF_ELE_9",
}

MAX_IMAGES_PER_ID = 6  # Cap per identity to keep balanced
PROJECT_ROOT = Path(__file__).parent.parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== DATASET ====================

class DiagnosticDataset(Dataset):
    def __init__(self, identity_map, max_per_id=6):
        self.samples = []  # (path, label_idx)
        self.labels = []
        self.label_names = []

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform = transform

        for idx, (name, rel_path) in enumerate(sorted(identity_map.items())):
            folder = PROJECT_ROOT / rel_path
            if not folder.exists():
                print(f"  [SKIP] {name}: {folder} not found")
                continue

            images = sorted([f for f in folder.iterdir()
                            if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])

            if len(images) < 2:
                print(f"  [SKIP] {name}: only {len(images)} images")
                continue

            # Cap and shuffle
            if len(images) > max_per_id:
                random.seed(42)
                images = random.sample(images, max_per_id)

            for img_path in images:
                self.samples.append((str(img_path), idx))

            self.label_names.append(name)
            print(f"  [OK] {name}: {len(images)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label

# ==================== MODEL ====================

class EmbeddingModel(nn.Module):
    """Pretrained ConvNeXt-Tiny — NO training, just feature extraction."""
    def __init__(self):
        super().__init__()
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat = self.backbone.features(x)
        feat = self.pool(feat).flatten(1)
        return F.normalize(feat, dim=1)

# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("ELEPHANT RE-ID — EMBEDDING DIAGNOSTIC")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"\nLoading test identities...")

    dataset = DiagnosticDataset(TEST_IDENTITIES, MAX_IMAGES_PER_ID)
    print(f"\nTotal samples: {len(dataset)}")
    print(f"Identities loaded: {len(dataset.label_names)}")

    if len(dataset.label_names) < 3:
        print("[FAIL] Too few identities. Check paths.")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Extract embeddings
    print("\nExtracting embeddings (pretrained ConvNeXt-Tiny)...")
    model = EmbeddingModel().to(device).eval()

    all_emb = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            emb = model(imgs)
            all_emb.append(emb.cpu())
            all_labels.append(labels)

    all_emb = torch.cat(all_emb).numpy()
    all_labels = torch.cat(all_labels).numpy()
    print(f"Embeddings shape: {all_emb.shape}")

    # Compute pairwise cosine distances
    # distance = 1 - cosine_similarity
    from sklearn.metrics.pairwise import cosine_distances
    dist_matrix = cosine_distances(all_emb)

    same_dists = []
    diff_dists = []
    n = len(all_labels)

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i][j]
            if all_labels[i] == all_labels[j]:
                same_dists.append(d)
            else:
                diff_dists.append(d)

    same_dists = np.array(same_dists)
    diff_dists = np.array(diff_dists)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Same-ID pairs:     {len(same_dists)}")
    print(f"Diff-ID pairs:     {len(diff_dists)}")
    print(f"\nSame-ID distance:  {np.mean(same_dists):.4f} (±{np.std(same_dists):.4f})")
    print(f"Diff-ID distance:  {np.mean(diff_dists):.4f} (±{np.std(diff_dists):.4f})")

    separation = np.mean(diff_dists) - np.mean(same_dists)
    print(f"\n{'='*40}")
    print(f"SEPARATION:        {separation:.4f}")
    print(f"{'='*40}")

    if separation > 0.25:
        print("[GOOD] preprocessing preserves identity signal")
    elif separation > 0.15:
        print("[OKAY] identity signal exists but weak")
    else:
        print("[BAD] preprocessing is destroying identity signal")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(same_dists, bins=40, alpha=0.6, color='green', label=f'Same ID (mean={np.mean(same_dists):.3f})', density=True)
    plt.hist(diff_dists, bins=40, alpha=0.6, color='red', label=f'Diff ID (mean={np.mean(diff_dists):.3f})', density=True)
    plt.axvline(np.mean(same_dists), color='darkgreen', linestyle='--', linewidth=2)
    plt.axvline(np.mean(diff_dists), color='darkred', linestyle='--', linewidth=2)
    plt.xlabel('Cosine Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Identity Separation (Gap = {separation:.4f})', fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()

    output_path = PROJECT_ROOT / "embedding_diagnostic.png"
    plt.savefig(str(output_path), dpi=150)
    print(f"\nGraph saved to: {output_path}")

    # Per-identity breakdown
    print("\n" + "=" * 60)
    print("PER-IDENTITY ANALYSIS")
    print("=" * 60)
    unique_labels = sorted(set(all_labels))
    for label in unique_labels:
        mask = all_labels == label
        if mask.sum() < 2:
            continue
        indices = np.where(mask)[0]
        intra_dists = []
        for i, j in combinations(indices, 2):
            intra_dists.append(dist_matrix[i][j])
        name = dataset.label_names[label] if label < len(dataset.label_names) else f"ID_{label}"
        print(f"  {name:20s}: intra-dist = {np.mean(intra_dists):.4f} (n={mask.sum()})")


if __name__ == "__main__":
    main()
