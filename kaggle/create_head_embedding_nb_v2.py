"""
Generate Kaggle notebook: Elephant Head Re-ID v2 — Pose-Invariant Identity Learning

Key upgrades over v1:
  1. Embedding-diversity sampling (farthest within-identity selection)
  2. Top-K hard negative mining in Multi-Similarity Loss
  3. Identity center regularization loss
  4. Larger batch composition (P=8 × M=3)
  5. Aggressive pose-simulating augmentations
  6. min_images_per_id=3 for training (singletons kept for eval only)
  7. 30 epochs (5 warmup)
  8. Cross-identity pose confusion evaluation
"""
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

# ============================================================================
# CELL 1 — Title
# ============================================================================
add_md("""# 🐘 Elephant Head Re-ID v2 — Pose-Invariant Identity Learning

**Why v2?**
v1 achieved separation gap = 0.679 but learned **pose similarity instead of identity invariance**.
Result: different elephants in the same pose scored 0.907 similarity — higher than same elephant in different poses (-0.14).

**What's changed:**
1. **Embedding-diversity sampling** — picks farthest images within same identity as positives
2. **Top-K hard negative mining** — explicitly trains against "same pose, different elephant"
3. **Identity center regularization** — stabilizes per-identity clusters in embedding space
4. **Aggressive augmentations** — perspective warp, affine transforms, random erasing
5. **Larger batch diversity** — P=8 identities × M=3 images for richer negative landscape

**Expected outcome:** Different-elephant same-pose drops from 0.907 → ~0.5, making clustering usable.""")

# ============================================================================
# CELL 2 — Imports
# ============================================================================
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
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)""")

# ============================================================================
# CELL 3 — Dataset
# ============================================================================
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
        skipped_names = []
        for name, paths in sorted(identity_folders.items()):
            if len(paths) < min_images_per_id:
                skipped += 1
                skipped_names.append(f"{name}({len(paths)})")
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
            if skipped <= 20:
                print(f"  Skipped: {', '.join(skipped_names)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label""")

# ============================================================================
# CELL 4 — Transforms & Load
# ============================================================================
add_md("## Cell 3 — Transforms & Load Dataset")
add_code("""# === AGGRESSIVE augmentations to reduce texture/pose overfitting ===
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),      # reduced — ears are asymmetric
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.85, 1.15),
        shear=10,
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),            # force non-color features
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # simulate occlusion
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Exact Kaggle dataset path
DATASET_PATH = Path("/kaggle/input/datasets/girishcodes/elephant-head-crops/processed_heads")

print(f"Dataset path: {DATASET_PATH}")

# Training: min 3 images per identity (singletons/pairs can't form meaningful positives)
train_dataset = HeadCropDataset(str(DATASET_PATH), train_transform, min_images_per_id=3)

# Eval: keep everything with >= 2 for evaluation
eval_dataset = HeadCropDataset(str(DATASET_PATH), eval_transform, min_images_per_id=2)

print(f"\\nTraining: {len(train_dataset)} images, {train_dataset.num_classes} identities")
print(f"Evaluation: {len(eval_dataset)} images, {eval_dataset.num_classes} identities")""")

# ============================================================================
# CELL 5 — Visual Inspection
# ============================================================================
add_md("## Cell 4 — Visual Inspection")
add_code("""fig, axes = plt.subplots(3, 6, figsize=(18, 9))
indices = random.sample(range(len(eval_dataset)), min(18, len(eval_dataset)))

for i, ax in enumerate(axes.flat):
    if i < len(indices):
        img, label = eval_dataset[indices[i]]
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

# ============================================================================
# CELL 6 — Model
# ============================================================================
add_md("## Cell 5 — Embedding Model (ConvNeXt-Tiny)")
add_code("""class HeadEmbeddingModel(nn.Module):
    \"\"\"
    ConvNeXt-Tiny backbone → 768-D features → 256-D L2-normalized embeddings.
    Same architecture as v1 for drop-in replacement.
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

# ============================================================================
# CELL 7 — Loss Functions
# ============================================================================
add_md("""## Cell 6 — Loss Functions

**Two losses working together:**
1. **Multi-Similarity Loss** with top-K hard negative mining — pushes apart same-pose different elephants
2. **Identity Center Loss** — pulls all embeddings of one identity toward a stable center""")

add_code("""class MultiSimilarityLossV2(nn.Module):
    \"\"\"
    Multi-Similarity Loss with explicit Top-K Hard Negative Mining.

    Key difference from v1:
      - v1: semi-hard filter (neg > pos.min() - 0.1) → misses hardest cases
      - v2: takes top-K most similar negatives → directly attacks
            "same pose, different elephant" confusion
    \"\"\"
    def __init__(self, alpha=2.0, beta=50.0, base=0.5, hard_neg_k=10):
        super().__init__()
        self.alpha = alpha
        self.beta = beta          # increased from 40→50 for stronger negative repulsion
        self.base = base
        self.hard_neg_k = hard_neg_k

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

            # === Hard positive mining: focus on the most dissimilar positives ===
            # These are "same elephant, different appearance" — exactly what we
            # need the model to learn to handle
            pos_loss = (1.0 / self.alpha) * torch.log(
                1 + torch.sum(torch.exp(-self.alpha * (pos - self.base)))
            )

            # === Top-K Hard Negative Mining (KEY CHANGE) ===
            # Instead of semi-hard filter, take the K most similar negatives
            # These are "different elephant, same pose" — the exact failure case
            k = min(self.hard_neg_k, len(neg))
            hard_neg, _ = neg.topk(k)  # highest similarity negatives

            neg_loss = (1.0 / self.beta) * torch.log(
                1 + torch.sum(torch.exp(self.beta * (hard_neg - self.base)))
            )

            loss += pos_loss + neg_loss
            valid += 1

        return loss / max(valid, 1)


class IdentityCenterLoss(nn.Module):
    \"\"\"
    Regularization: force all embeddings of the same identity toward a center.

    For each identity in the batch, compute the centroid of its embeddings,
    then penalize deviation from it. This stabilizes identity clusters and
    prevents individual outlier embeddings from drifting.

    loss = (1/N) * Σ ||emb_i - center(identity_i)||²
    \"\"\"
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, labels):
        unique_labels = labels.unique()
        loss = 0.0
        count = 0
        for lbl in unique_labels:
            mask = labels == lbl
            group_emb = embeddings[mask]
            if len(group_emb) < 2:
                continue
            center = group_emb.mean(dim=0, keepdim=True)
            # L2 distance to center
            dists = ((group_emb - center) ** 2).sum(dim=1)
            loss += dists.mean()
            count += 1
        return loss / max(count, 1)


# Initialize losses
criterion_ms = MultiSimilarityLossV2(alpha=2.0, beta=50.0, base=0.5, hard_neg_k=10)
criterion_center = IdentityCenterLoss()
CENTER_LOSS_WEIGHT = 0.1  # λ — start gentle, center loss is a regularizer

print("Loss 1: Multi-Similarity Loss v2 (alpha=2.0, beta=50.0, top-K=10 hard negatives)")
print(f"Loss 2: Identity Center Loss (λ={CENTER_LOSS_WEIGHT})")
print(f"Total Loss = MS_Loss + {CENTER_LOSS_WEIGHT} × Center_Loss")""")

# ============================================================================
# CELL 8 — Embedding-Diversity Sampler
# ============================================================================
add_md("""## Cell 7 — Embedding-Diversity Sampler

**Key insight:** We can't rely on "pick a different pose" because most identities
only have ~3 images, likely all from the same session/pose. Instead, we use
**embedding-based diversity**: pick the most dissimilar images within each identity
as positives.

At the start of each epoch, we compute embeddings for all training images,
then for each identity, we sort images by pairwise distance to maximize diversity
in each batch.""")

add_code("""class DiversityPxMSampler:
    \"\"\"
    P × M sampler that selects diverse images within each identity.

    Every `refresh_interval` epochs, re-embeds the training set and computes
    within-identity pairwise distances. During sampling, picks M images per
    identity that maximize pairwise distance (greedy farthest-point selection).

    Falls back to random sampling if embeddings aren't computed yet.
    \"\"\"
    def __init__(self, dataset, model, device, P=8, M=3, refresh_interval=3):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.P = P
        self.M = M
        self.refresh_interval = refresh_interval
        self.label_to_indices = dataset.label_to_indices
        self.available_labels = [l for l, idx in self.label_to_indices.items()
                                 if len(idx) >= 2]
        self._identity_distances = {}  # label → (N_id, N_id) distance matrix
        self._embeddings = None

    def refresh_embeddings(self, epoch):
        \"\"\"Re-embed training set to update diversity distances.\"\"\"
        if epoch % self.refresh_interval != 0:
            return

        print(f"  [Sampler] Refreshing embeddings for diversity sampling...")
        self.model.eval()
        all_emb = []

        # Embed in batches
        with torch.no_grad():
            all_indices = list(range(len(self.dataset)))
            for start in range(0, len(all_indices), 32):
                batch_indices = all_indices[start:start+32]
                batch = [self.dataset[i] for i in batch_indices]
                imgs = torch.stack([x[0] for x in batch]).to(self.device)
                embs = self.model(imgs)
                all_emb.append(embs.cpu())

        self._embeddings = torch.cat(all_emb)  # (N, 256)

        # Build per-identity distance matrices
        for label in self.available_labels:
            indices = self.label_to_indices[label]
            if len(indices) < 2:
                continue
            embs = self._embeddings[indices]
            # Cosine distance = 1 - cosine_similarity
            sim = embs @ embs.t()
            dist = 1.0 - sim
            self._identity_distances[label] = dist.numpy()

        self.model.train()
        print(f"  [Sampler] Refreshed {len(self._identity_distances)} identity distance matrices")

    def _select_diverse_M(self, label):
        \"\"\"Greedy farthest-point selection of M images from one identity.\"\"\"
        indices = self.label_to_indices[label]

        if len(indices) <= self.M:
            # Not enough images — use all + repeat
            selected = list(indices)
            while len(selected) < self.M:
                selected.append(random.choice(indices))
            return selected

        if label not in self._identity_distances:
            # No distance matrix yet — random fallback
            return random.sample(indices, self.M)

        dist_mat = self._identity_distances[label]
        n = len(indices)

        # Start with a random image
        selected_local = [random.randint(0, n - 1)]

        for _ in range(self.M - 1):
            # Pick the image farthest from all currently selected
            best_idx = -1
            best_min_dist = -1.0
            for candidate in range(n):
                if candidate in selected_local:
                    continue
                min_dist = min(dist_mat[candidate][s] for s in selected_local)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = candidate
            if best_idx >= 0:
                selected_local.append(best_idx)
            else:
                break

        return [indices[i] for i in selected_local]

    def sample_batch(self):
        \"\"\"Sample one P×M batch with diversity-aware image selection.\"\"\"
        actual_P = min(self.P, len(self.available_labels))
        selected_labels = random.sample(self.available_labels, actual_P)

        batch_indices = []
        for label in selected_labels:
            batch_indices.extend(self._select_diverse_M(label))

        batch = [self.dataset[i] for i in batch_indices]
        imgs = torch.stack([x[0] for x in batch])
        labels = torch.tensor([x[1] for x in batch])
        return imgs, labels""")

# ============================================================================
# CELL 9 — Training Setup
# ============================================================================
add_md("## Cell 8 — Training Setup (Warmup + Differential LR)")
add_code("""TOTAL_EPOCHS = 30
WARMUP_EPOCHS = 5
P = 8   # identities per batch (up from 6 → more negative diversity)
M = 3   # images per identity
BATCH_SIZE = P * M

# Initialize diversity sampler
sampler = DiversityPxMSampler(train_dataset, model, device, P=P, M=M, refresh_interval=3)

steps_per_epoch = max(1, len(train_dataset) // BATCH_SIZE)

# Start frozen
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW([
    {'params': model.embed.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
], weight_decay=1e-4)

print(f"Phase 1 (epochs 1-{WARMUP_EPOCHS}): Backbone FROZEN, embedding head only")
print(f"Phase 2 (epochs {WARMUP_EPOCHS+1}-{TOTAL_EPOCHS}): Full fine-tune")
print(f"Batch: P={P} × M={M} = {BATCH_SIZE} images")
print(f"Steps/epoch: {steps_per_epoch}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")""")

# ============================================================================
# CELL 10 — Training Loop
# ============================================================================
add_md("## Cell 9 — Training Loop")
add_code("""history = {'epoch': [], 'loss': [], 'ms_loss': [], 'center_loss': [], 'phase': []}

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

    # Refresh diversity embeddings periodically
    sampler.refresh_embeddings(epoch)

    model.train()
    total_loss = 0
    total_ms = 0
    total_center = 0
    n_batches = 0

    for step in range(steps_per_epoch):
        # Diversity-aware P×M sampling
        imgs, labels = sampler.sample_batch()
        imgs = imgs.to(device)
        labels = labels.to(device)

        embeddings = model(imgs)

        # Combined loss
        ms_loss = criterion_ms(embeddings, labels)
        center_loss = criterion_center(embeddings, labels)
        loss = ms_loss + CENTER_LOSS_WEIGHT * center_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ms += ms_loss.item()
        total_center += center_loss.item()
        n_batches += 1

    if epoch > WARMUP_EPOCHS:
        scheduler.step()

    avg_loss = total_loss / max(1, n_batches)
    avg_ms = total_ms / max(1, n_batches)
    avg_center = total_center / max(1, n_batches)
    lr = optimizer.param_groups[0]['lr']
    phase = "WARMUP" if epoch <= WARMUP_EPOCHS else "FULL"
    history['epoch'].append(epoch)
    history['loss'].append(avg_loss)
    history['ms_loss'].append(avg_ms)
    history['center_loss'].append(avg_center)
    history['phase'].append(phase)

    print(f"Epoch {epoch:2d}/{TOTAL_EPOCHS} [{phase:6s}] "
          f"Loss: {avg_loss:.4f} (MS: {avg_ms:.4f} + Center: {avg_center:.4f})  LR: {lr:.6f}")

print("\\n✅ Training complete!")""")

# ============================================================================
# CELL 11 — Loss Curve
# ============================================================================
add_md("## Cell 10 — Loss Curves")
add_code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = ['orange' if p == 'WARMUP' else 'blue' for p in history['phase']]
ax1.bar(history['epoch'], history['loss'], color=colors, alpha=0.7)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Total Loss')
ax1.set_title('Total Loss (orange=warmup, blue=full)')

ax2.plot(history['epoch'], history['ms_loss'], 'b-o', label='MS Loss', markersize=3)
ax2.plot(history['epoch'], history['center_loss'], 'r-o', label='Center Loss', markersize=3)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Components')
ax2.legend()

plt.tight_layout()
plt.show()""")

# ============================================================================
# CELL 12 — Evaluation: Same-ID vs Diff-ID Separation
# ============================================================================
add_md("## Cell 11 — Evaluate: Same-ID vs Diff-ID Separation")
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

# Cosine similarity matrix (not distance — easier to compare with our 0.907 number)
sim_matrix = all_emb @ all_emb.T

same_sims = []
diff_sims = []
n = len(all_labels)

for i in range(n):
    for j in range(i+1, n):
        s = sim_matrix[i][j]
        if all_labels[i] == all_labels[j]:
            same_sims.append(s)
        else:
            diff_sims.append(s)

same_sims = np.array(same_sims)
diff_sims = np.array(diff_sims)

print(f"\\nSame-ID pairs:  {len(same_sims)}")
print(f"Diff-ID pairs:  {len(diff_sims)}")
print(f"\\nSame-ID similarity:  {np.mean(same_sims):.4f} (±{np.std(same_sims):.4f})")
print(f"  min: {np.min(same_sims):.4f}, max: {np.max(same_sims):.4f}")
print(f"Diff-ID similarity:  {np.mean(diff_sims):.4f} (±{np.std(diff_sims):.4f})")
print(f"  min: {np.min(diff_sims):.4f}, max: {np.max(diff_sims):.4f}")

separation = np.mean(same_sims) - np.mean(diff_sims)
print(f"\\n{'='*50}")
print(f"SEPARATION GAP (sim):    {separation:.4f}")
print(f"{'='*50}")

# Critical metric: what % of diff-ID pairs score above 0.65 (our clustering threshold)?
false_merge_rate = (diff_sims > 0.65).mean() * 100
print(f"\\n🚨 CRITICAL: {false_merge_rate:.1f}% of different-elephant pairs would false-merge at threshold=0.65")
print(f"   (Worst diff-ID sim: {np.max(diff_sims):.4f})")
print(f"   (v1 had 0.907 — this should be < 0.60)")

# Same-ID below threshold = fragmentation
fragmentation_rate = (same_sims < 0.65).mean() * 100
print(f"\\n⚠️  {fragmentation_rate:.1f}% of same-elephant pairs would fragment at threshold=0.65")

if separation > 0.30 and false_merge_rate < 5:
    print("\\n🟢 GOOD — model learned identity invariance")
elif separation > 0.20:
    print("\\n🟡 OKAY — identity signal improved, still some confusion")
elif separation > 0.10:
    print("\\n🟠 WEAK — partial learning, needs more data or training")
else:
    print("\\n🔴 BAD — model not learning identity")""")

# ============================================================================
# CELL 13 — Separation Histogram  
# ============================================================================
add_md("## Cell 12 — Separation Histogram")
add_code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: similarity distribution
ax1.hist(same_sims, bins=50, alpha=0.6, color='green',
         label=f'Same ID (mean={np.mean(same_sims):.3f})', density=True)
ax1.hist(diff_sims, bins=50, alpha=0.6, color='red',
         label=f'Diff ID (mean={np.mean(diff_sims):.3f})', density=True)
ax1.axvline(0.65, color='black', linestyle='--', linewidth=2, label='Threshold 0.65')
ax1.axvline(np.mean(same_sims), color='darkgreen', linestyle=':', linewidth=2)
ax1.axvline(np.mean(diff_sims), color='darkred', linestyle=':', linewidth=2)
ax1.set_xlabel('Cosine Similarity', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title(f'v2 Identity Separation (Gap = {separation:.4f})', fontsize=14)
ax1.legend(fontsize=10)

# Right: false merge zone detail
ax2.hist(diff_sims[diff_sims > 0.4], bins=30, alpha=0.6, color='red',
         label='Diff-ID (sim > 0.4)', density=True)
ax2.hist(same_sims[same_sims > 0.4], bins=30, alpha=0.6, color='green',
         label='Same-ID (sim > 0.4)', density=True)
ax2.axvline(0.65, color='black', linestyle='--', linewidth=2, label='Threshold')
ax2.set_xlabel('Cosine Similarity', fontsize=12)
ax2.set_title('Overlap Zone Detail (sim > 0.4)', fontsize=14)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.show()""")

# ============================================================================
# CELL 14 — Per-Identity Analysis
# ============================================================================
add_md("## Cell 13 — Per-Identity Analysis & Worst Cases")
add_code("""from itertools import combinations

print("PER-IDENTITY INTRA-SIMILARITY")
print("=" * 50)
unique_labels = sorted(set(all_labels))
identity_stats = []
for label in unique_labels:
    mask = all_labels == label
    if mask.sum() < 2:
        continue
    indices = np.where(mask)[0]
    intra = [sim_matrix[i][j] for i, j in combinations(indices, 2)]
    name = eval_dataset.idx_to_class.get(label, f"ID_{label}")
    short = name.split("/")[-1]
    mean_sim = np.mean(intra)
    min_sim = np.min(intra)
    status = "✅" if min_sim > 0.50 else "⚠️" if min_sim > 0.30 else "❌"
    identity_stats.append((short, mean_sim, min_sim, mask.sum()))
    print(f"  {status} {short:25s}: mean={mean_sim:.4f} min={min_sim:.4f} (n={mask.sum()})")

# Worst cross-identity false matches
print(f"\\n{'='*50}")
print("TOP-10 WORST CROSS-IDENTITY FALSE MATCHES")
print("=" * 50)
worst_pairs = []
for i in range(n):
    for j in range(i+1, n):
        if all_labels[i] != all_labels[j]:
            worst_pairs.append((sim_matrix[i][j], i, j))

worst_pairs.sort(reverse=True)
for sim, i, j in worst_pairs[:10]:
    name_i = eval_dataset.idx_to_class.get(all_labels[i], "?").split("/")[-1]
    name_j = eval_dataset.idx_to_class.get(all_labels[j], "?").split("/")[-1]
    print(f"  ❌ {name_i} ↔ {name_j}: {sim:.4f}")""")

# ============================================================================
# CELL 15 — Rank-1 / Rank-5
# ============================================================================
add_md("## Cell 14 — Rank-1 / Rank-5 Accuracy")
add_code("""np.fill_diagonal(sim_matrix, -1)  # exclude self

rank1_correct = 0
rank5_correct = 0
total = 0

for i in range(n):
    query_label = all_labels[i]
    ranked = np.argsort(-sim_matrix[i])

    if all_labels[ranked[0]] == query_label:
        rank1_correct += 1
    if query_label in all_labels[ranked[:5]]:
        rank5_correct += 1
    total += 1

print(f"Rank-1 Accuracy: {rank1_correct/total:.4f} ({rank1_correct}/{total})")
print(f"Rank-5 Accuracy: {rank5_correct/total:.4f} ({rank5_correct}/{total})")""")

# ============================================================================
# CELL 16 — Save Model
# ============================================================================
add_md("## Cell 15 — Save Model")
add_code("""import shutil

# Save full state
save_dict = {
    'model_state_dict': model.state_dict(),
    'embed_dim': 256,
    'num_classes': train_dataset.num_classes,
    'separation_gap': float(separation),
    'false_merge_rate_at_065': float(false_merge_rate),
    'fragmentation_rate_at_065': float(fragmentation_rate),
    'training_config': {
        'version': 'v2_pose_invariant',
        'loss': 'MultiSimilarityV2 + IdentityCenterLoss',
        'beta': 50.0,
        'hard_neg_k': 10,
        'center_loss_weight': CENTER_LOSS_WEIGHT,
        'P': P, 'M': M,
        'epochs': TOTAL_EPOCHS,
        'warmup_epochs': WARMUP_EPOCHS,
        'min_images_per_id': 3,
        'diversity_sampling': True,
    },
}

torch.save(save_dict, "/kaggle/working/elephant_head_reid_v2.pth")

shutil.copy2(
    "/kaggle/working/elephant_head_reid_v2.pth",
    "/kaggle/working/elephant_head_reid_v2_download.pth"
)

print("✅ Model saved!")
print(f"   Separation gap: {separation:.4f}")
print(f"   False merge rate @ 0.65: {false_merge_rate:.1f}%")
print(f"   Download: elephant_head_reid_v2_download.pth")""")

# ============================================================================
# Write the notebook
# ============================================================================
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

output_path = r'd:\Elephant_ReIdentification\kaggle\elephant-head-embedding-training-v2.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Notebook generated: {output_path}")
