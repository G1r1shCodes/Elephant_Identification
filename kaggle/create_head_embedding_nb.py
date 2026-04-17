import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [text]
    })

def add_code(text):
    lines = [line + '\n' for line in text.split('\n')]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

add_md("""# 🐘 Elephant Head Re-ID — Embedding Training (Multi-Similarity Loss)

**Purpose:** Train a ConvNeXt-Tiny embedding model to produce identity-discriminative
embeddings from elephant head crops.

**Why this is needed:**
- Pretrained ImageNet features give separation gap = 0.028 (unusable)
- Need metric learning to force same-elephant embeddings closer, different-elephant farther

**Prerequisites:**
1. Upload `processed_heads/` as Kaggle dataset `elephant-head-crops`
2. Run with **GPU T4 x2** accelerator
3. Expect ~15 min training time""")

add_md("## Cell 1 — Install & Imports")
add_code("""import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")""")

add_md("## Cell 2 — Dataset (Head Crops)")
add_code("""class HeadCropDataset(Dataset):
    \"\"\"
    Loads pre-processed elephant head crops.
    Identity = deepest folder containing images.
    Skips identities with < min_images.
    \"\"\"
    def __init__(self, root_dir, transform=None, min_images_per_id=2):
        self.samples = []  # (path, label_idx)
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.label_to_indices = defaultdict(list)

        identity_folders = {}
        for dirpath, dirnames, filenames in os.walk(root_dir):
            images = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                rel = os.path.relpath(dirpath, root_dir)
                identity_folders[rel] = [os.path.join(dirpath, f) for f in images]

        idx = 0
        skipped = 0
        for name, paths in sorted(identity_folders.items()):
            if len(paths) < min_images_per_id:
                skipped += 1
                continue
            self.class_to_idx[name] = idx
            self.idx_to_class[idx] = name
            for p in paths:
                self.label_to_indices[idx].append(len(self.samples))
                self.samples.append((p, idx))
            idx += 1

        self.num_classes = idx
        print(f"Loaded {len(self.samples)} images across {idx} identities")
        if skipped:
            print(f"  Skipped {skipped} identities with < {min_images_per_id} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label""")

add_md("## Cell 3 — Transforms & Load Dataset")
add_code("""# CRITICAL: Resize directly to 224x224 — NO CenterCrop
# Head detector already aligned the ROI, CenterCrop would cut ears

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),  # reduced — ears are asymmetric
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Exact Kaggle dataset path
DATASET_PATH = Path("/kaggle/input/datasets/girishcodes/elephant-head-crops/processed_heads")

print(f"Dataset path: {DATASET_PATH}")

train_dataset = HeadCropDataset(str(DATASET_PATH), train_transform, min_images_per_id=2)
eval_dataset = HeadCropDataset(str(DATASET_PATH), eval_transform, min_images_per_id=2)

print(f"\\nTotal samples: {len(train_dataset)}")
print(f"Number of identities: {train_dataset.num_classes}")""")

add_md("## Cell 4 — Visual Inspection")
add_code("""fig, axes = plt.subplots(3, 6, figsize=(18, 9))
indices = random.sample(range(len(eval_dataset)), min(18, len(eval_dataset)))

for i, ax in enumerate(axes.flat):
    if i < len(indices):
        img, label = eval_dataset[indices[i]]
        # Denormalize
        img = img.clone()
        for t, m, s in zip(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        img = img.permute(1,2,0).numpy()
        img = np.clip(img, 0, 1)
        identity = eval_dataset.idx_to_class.get(label, f"ID_{label}")
        ax.imshow(img)
        ax.set_title(identity.split("/")[-1], fontsize=7)
    ax.axis('off')
plt.suptitle("Head Crop Samples", fontsize=14)
plt.tight_layout()
plt.show()""")

add_md("## Cell 5 — Embedding Model (ConvNeXt-Tiny)")
add_code("""class HeadEmbeddingModel(nn.Module):
    \"\"\"
    ConvNeXt-Tiny backbone → 768-D features → 256-D L2-normalized embeddings.
    Simple and effective for small datasets.
    \"\"\"
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        feat = self.backbone.features(x)
        feat = self.pool(feat).flatten(1)  # (B, 768)
        emb = self.embed(feat)             # (B, 256)
        return F.normalize(emb, p=2, dim=1)

model = HeadEmbeddingModel(embed_dim=256).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable:    {trainable:,}")""")

add_md("## Cell 6 — Multi-Similarity Loss")
add_code("""class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=40.0, base=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, embeddings, labels):
        sim = torch.matmul(embeddings, embeddings.t())
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.t())
        neg_mask = ~pos_mask

        loss = 0
        valid = 0
        for i in range(len(embeddings)):
            pos = sim[i][pos_mask[i]].clone()
            neg = sim[i][neg_mask[i]].clone()

            pos = pos[pos < 1 - 1e-6]  # remove self-similarity

            if len(pos) == 0 or len(neg) == 0:
                continue

            # Hard positive mining: focus on hardest positives
            pos_loss = (1.0 / self.alpha) * torch.log(
                1 + torch.sum(torch.exp(-self.alpha * (pos - self.base)))
            )

            # Semi-hard negative mining
            neg_filtered = neg[neg > pos.min() - 0.1]
            if len(neg_filtered) == 0:
                neg_filtered = neg

            neg_loss = (1.0 / self.beta) * torch.log(
                1 + torch.sum(torch.exp(self.beta * (neg_filtered - self.base)))
            )

            loss += pos_loss + neg_loss
            valid += 1

        return loss / max(valid, 1)

criterion = MultiSimilarityLoss(alpha=2.0, beta=40.0, base=0.5)
print("Loss: Multi-Similarity Loss (alpha=2.0, beta=40.0, base=0.5)")""")

add_md("## Cell 7 — Training Setup (Warmup + Differential LR)")
add_code("""# Phase 1 (epochs 1-3): Freeze backbone, train only embedding head
# Phase 2 (epochs 4-15): Unfreeze backbone with low LR

# Start frozen
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW([
    {'params': model.embed.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
], weight_decay=1e-4)

print("Phase 1: Backbone FROZEN, training embedding head only")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")""")

add_md("## Cell 8 — Training Loop (P×M Sampling)")
add_code("""TOTAL_EPOCHS = 15
WARMUP_EPOCHS = 3
P = 6   # identities per batch
M = 4   # images per identity
BATCH_SIZE = P * M

label_to_indices = train_dataset.label_to_indices
available_labels = [l for l, idx in label_to_indices.items() if len(idx) >= 2]
print(f"Labels with >= 2 images: {len(available_labels)}")

actual_P = min(P, len(available_labels))
steps_per_epoch = max(1, len(train_dataset) // BATCH_SIZE)

history = {'epoch': [], 'loss': [], 'phase': []}

for epoch in range(1, TOTAL_EPOCHS + 1):

    # Phase transition: unfreeze backbone at epoch WARMUP_EPOCHS+1
    if epoch == WARMUP_EPOCHS + 1:
        print(f"\\n{'='*60}")
        print(f"PHASE 2: Unfreezing backbone with differential LR")
        print(f"{'='*60}")
        for param in model.backbone.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': 5e-5, 'weight_decay': 1e-4},
            {'params': model.embed.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable:,}")

    model.train()
    total_loss = 0
    n_batches = 0

    for step in range(steps_per_epoch):
        # P×M sampling
        selected = random.sample(available_labels, actual_P)
        batch_indices = []
        for label in selected:
            indices = label_to_indices[label]
            if len(indices) >= M:
                batch_indices.extend(random.sample(indices, M))
            else:
                batch_indices.extend(random.choices(indices, k=M))

        batch = [train_dataset[i] for i in batch_indices]
        imgs = torch.stack([x[0] for x in batch]).to(device)
        labels = torch.tensor([x[1] for x in batch]).to(device)

        embeddings = model(imgs)
        loss = criterion(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if epoch > WARMUP_EPOCHS:
        scheduler.step()

    avg_loss = total_loss / max(1, n_batches)
    lr = optimizer.param_groups[0]['lr']
    phase = "WARMUP" if epoch <= WARMUP_EPOCHS else "FULL"
    history['epoch'].append(epoch)
    history['loss'].append(avg_loss)
    history['phase'].append(phase)

    print(f"Epoch {epoch:2d}/{TOTAL_EPOCHS} [{phase:6s}] Loss: {avg_loss:.4f}  LR: {lr:.6f}")

print("\\n✅ Training complete!")""")

add_md("## Cell 9 — Loss Curve")
add_code("""plt.figure(figsize=(10, 5))
colors = ['orange' if p == 'WARMUP' else 'blue' for p in history['phase']]
plt.bar(history['epoch'], history['loss'], color=colors, alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('MS-Loss')
plt.title('Training Loss (orange=warmup, blue=full)')
plt.tight_layout()
plt.show()""")

add_md("## Cell 10 — Evaluate: Same-ID vs Diff-ID Separation")
add_code("""model.eval()
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

all_emb = []
all_labels = []

with torch.no_grad():
    for imgs, labels in eval_loader:
        imgs = imgs.to(device)
        emb = model(imgs)
        all_emb.append(emb.cpu())
        all_labels.append(labels)

all_emb = torch.cat(all_emb).numpy()
all_labels = torch.cat(all_labels).numpy()
print(f"Embeddings: {all_emb.shape}")

# Cosine distance matrix
from sklearn.metrics.pairwise import cosine_distances
dist_matrix = cosine_distances(all_emb)

same_dists = []
diff_dists = []
n = len(all_labels)

for i in range(n):
    for j in range(i+1, n):
        d = dist_matrix[i][j]
        if all_labels[i] == all_labels[j]:
            same_dists.append(d)
        else:
            diff_dists.append(d)

same_dists = np.array(same_dists)
diff_dists = np.array(diff_dists)

print(f"\\nSame-ID pairs:  {len(same_dists)}")
print(f"Diff-ID pairs:  {len(diff_dists)}")
print(f"\\nSame-ID distance:  {np.mean(same_dists):.4f} (±{np.std(same_dists):.4f})")
print(f"Diff-ID distance:  {np.mean(diff_dists):.4f} (±{np.std(diff_dists):.4f})")

separation = np.mean(diff_dists) - np.mean(same_dists)
print(f"\\n{'='*40}")
print(f"SEPARATION GAP:    {separation:.4f}")
print(f"{'='*40}")

if separation > 0.25:
    print("🟢 GOOD — model learned identity features")
elif separation > 0.15:
    print("🟡 OKAY — identity signal exists, needs more data/training")
elif separation > 0.08:
    print("🟠 WEAK — some learning, but not enough")
else:
    print("🔴 BAD — model not learning identity")

print(f"\\nBaseline (pretrained, no training): 0.0285")
print(f"Improvement: {separation / 0.0285:.1f}x")""")

add_md("## Cell 11 — Separation Histogram")
add_code("""plt.figure(figsize=(10, 6))
plt.hist(same_dists, bins=40, alpha=0.6, color='green',
         label=f'Same ID (mean={np.mean(same_dists):.3f})', density=True)
plt.hist(diff_dists, bins=40, alpha=0.6, color='red',
         label=f'Diff ID (mean={np.mean(diff_dists):.3f})', density=True)
plt.axvline(np.mean(same_dists), color='darkgreen', linestyle='--', linewidth=2)
plt.axvline(np.mean(diff_dists), color='darkred', linestyle='--', linewidth=2)
plt.xlabel('Cosine Distance', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title(f'Trained Model — Identity Separation (Gap = {separation:.4f})', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()""")

add_md("## Cell 12 — Per-Identity Analysis")
add_code("""from itertools import combinations

print("PER-IDENTITY INTRA-DISTANCE")
print("=" * 50)
unique_labels = sorted(set(all_labels))
for label in unique_labels:
    mask = all_labels == label
    if mask.sum() < 2:
        continue
    indices = np.where(mask)[0]
    intra = [dist_matrix[i][j] for i, j in combinations(indices, 2)]
    name = eval_dataset.idx_to_class.get(label, f"ID_{label}")
    short = name.split("/")[-1]
    status = "✅" if np.mean(intra) < 0.15 else "⚠️" if np.mean(intra) < 0.25 else "❌"
    print(f"  {status} {short:25s}: {np.mean(intra):.4f} (n={mask.sum()})")""")

add_md("## Cell 13 — Rank-1 / Rank-5 Accuracy")
add_code("""from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(all_emb)
np.fill_diagonal(sim_matrix, -1)  # exclude self

rank1_correct = 0
rank5_correct = 0
total = 0

for i in range(n):
    query_label = all_labels[i]
    # Sort by similarity (descending)
    ranked = np.argsort(-sim_matrix[i])

    # Rank-1
    if all_labels[ranked[0]] == query_label:
        rank1_correct += 1

    # Rank-5
    if query_label in all_labels[ranked[:5]]:
        rank5_correct += 1

    total += 1

print(f"Rank-1 Accuracy: {rank1_correct/total:.4f} ({rank1_correct}/{total})")
print(f"Rank-5 Accuracy: {rank5_correct/total:.4f} ({rank5_correct}/{total})")""")

add_md("## Cell 14 — Save Model")
add_code("""import shutil

# Save full state
torch.save({
    'model_state_dict': model.state_dict(),
    'embed_dim': 256,
    'num_classes': train_dataset.num_classes,
    'separation_gap': float(separation),
}, "/kaggle/working/elephant_head_reid_v1.pth")

# Also copy to easy download location
shutil.copy2(
    "/kaggle/working/elephant_head_reid_v1.pth",
    "/kaggle/working/elephant_head_reid_v1_download.pth"
)

print("✅ Model saved!")
print(f"   Separation gap achieved: {separation:.4f}")
print(f"   Download from Output tab: elephant_head_reid_v1_download.pth")""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = r'd:\Elephant_ReIdentification\kaggle\elephant-head-embedding-training.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Notebook generated: {output_path}")
