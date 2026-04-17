"""
Generate Kaggle notebook: Elephant Head Re-ID v3 — Ambiguity-Aware Identity Learning

Key upgrades over v2:
  1. ArcFace as regularizer (margin=0.25, weight=0.25) — enforces angular boundary
  2. Temperature scaling at SIMILARITY level inside loss (not on embedding)
  3. Stronger MS-Loss negative pressure (beta=75, hard_neg_k=20)
  4. Worst-pair replay — explicitly forces hardest cross-identity pairs into batches
  5. Ambiguity-aware evaluation — gap score, entropy, per-pair confidence
  6. Saves as elephant_head_reid_v3.pth (drop-in replacement)
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
add_md("""# 🐘 Elephant Head Re-ID v3 — Ambiguity-Aware Identity Learning

**Why v3?**
v2 achieved separation gap = 0.8727 and same-ID mean = 0.914 — a massive improvement.
But the model became **overconfident**: pairs like `JM_ELE_84 ↔ JU_ELE_96` scored 0.9998
because the embedding space has no enforced angular margin between classes.

**What's changed (targeted fixes):**
1. **ArcFace as regularizer** (margin=0.25, weight=0.25) — enforces geometric boundaries between identity clusters without replacing the open-set MS ranking loss
2. **Temperature scaling at similarity level** (`sim / T` inside loss only) — prevents embedding collapse without disrupting cosine similarity at inference
3. **Stronger negative pressure** — beta=75, hard_neg_k=20 (from 50/10)
4. **Worst-pair replay** — known hardest cross-identity pairs (from v2 eval) are force-loaded into batches every 5 epochs
5. **Ambiguity-aware evaluation** — reports gap score and entropy to detect uncertain assignments

**Expected outcome:** Worst false match drops from 0.9998 → ~0.85. False merge rate @ 0.75 < 3%.""")

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
import math
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

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
    Loads pre-processed elephant head crops (cleaned dataset).
    Identity = deepest folder containing images.
    Skips identities with < min_images.
    Ignores _quarantined, _filtered, _no_head_in_crop directories.
    \"\"\"
    def __init__(self, root_dir, transform=None, min_images_per_id=2):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.label_to_indices = defaultdict(list)

        SKIP_DIRS = {'_quarantined', '_filtered', '_no_head_in_crop'}

        identity_folders = {}
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Skip utility/noise directories
            parts = set(Path(dirpath).parts)
            if parts & SKIP_DIRS:
                continue
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

# ============================================================================
# CELL 4 — Transforms & Load
# ============================================================================
add_md("## Cell 3 — Transforms & Load Dataset")
add_code("""train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.85, 1.15),
        shear=10,
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Updated path — uses the cleaned dataset (post-quarantine)
DATASET_PATH = Path("/kaggle/input/datasets/girishcodes/elephant-processed-heads/processed_heads")

print(f"Dataset path: {DATASET_PATH}")

train_dataset = HeadCropDataset(str(DATASET_PATH), train_transform, min_images_per_id=3)
eval_dataset  = HeadCropDataset(str(DATASET_PATH), eval_transform,  min_images_per_id=2)

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
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        identity = eval_dataset.idx_to_class.get(label, f"ID_{label}")
        ax.imshow(img)
        ax.set_title(identity.split("/")[-1], fontsize=7)
    ax.axis('off')
plt.suptitle("Head Crop Samples (Cleaned Dataset)", fontsize=14)
plt.tight_layout()
plt.show()""")

# ============================================================================
# CELL 6 — Model
# ============================================================================
add_md("""## Cell 5 — Embedding Model (ConvNeXt-Tiny)

Same architecture as v1/v2 — **drop-in replacement**.
ArcFace head is training-only; only the embedding backbone is used at inference.""")
add_code("""class HeadEmbeddingModel(nn.Module):
    \"\"\"
    ConvNeXt-Tiny backbone → 768-D → 256-D L2-normalized embeddings.
    Training adds an ArcFace head, but only this backbone is saved/used at inference.
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


class ArcFaceHead(nn.Module):
    \"\"\"
    ArcFace geometric constraint (training-only regularizer).

    Enforces cos(θ + margin) between identity directions in embedding space.
    This prevents the embedding collapse that caused 0.9998 cross-ID scores.

    IMPORTANT: This is NOT used at inference — embeddings remain open-set.
    Low margin (0.25) + low weight (0.25) = regularizer, not dominant objective.
    \"\"\"
    def __init__(self, embed_dim, num_classes, margin=0.25, scale=32.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        # Cosine similarity to class centers
        cosine = F.linear(embeddings, F.normalize(self.weight))
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))

        # cos(θ + margin) = cos θ · cos m - sin θ · sin m
        phi = cosine * self.cos_m - sine * self.sin_m

        # Only apply margin to the correct class, else use cosine directly
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine

        return F.cross_entropy(output * self.scale, labels)


model = HeadEmbeddingModel(embed_dim=256).to(device)
arcface_head = ArcFaceHead(
    embed_dim=256,
    num_classes=train_dataset.num_classes,
    margin=0.25,
    scale=32.0
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Embedding model params: {total_params:,}")
print(f"ArcFace head params:    {sum(p.numel() for p in arcface_head.parameters()):,}")
print(f"ArcFace head: {train_dataset.num_classes} classes, margin=0.25, scale=32.0")""")

# ============================================================================
# CELL 7 — Loss Functions
# ============================================================================
add_md("""## Cell 6 — Loss Functions (v3 Hybrid)

**Three losses working together:**
1. **MS-Loss v3** — temperature-scaled similarity + top-20 hard negatives (beta=75)
2. **Identity Center Loss** — stabilizes per-identity clusters
3. **ArcFace Loss** — enforces angular margin to prevent embedding collapse

```
Total Loss = MS_Loss + 0.1 × Center_Loss + 0.25 × ArcFace_Loss
```""")

add_code("""class MultiSimilarityLossV3(nn.Module):
    \"\"\"
    MS-Loss v3: Temperature scaling applied at SIMILARITY level (not embedding).

    Why temperature scaling matters:
      - Embeddings are L2-normalized → cosine similarity is the same regardless
        of embedding scale
      - Temperature T < 1 applied to SIM (not embedding) sharpens gradients
        on hard pairs in the loss, making the model more sensitive to small
        angular differences
      - This does NOT change inference cosine similarity at all

    Key changes from v2:
      - beta: 50 → 75 (stronger negative repulsion)
      - hard_neg_k: 10 → 20 (catch more hard pairs)
      - Temperature T=0.2 applied inside loss
    \"\"\"
    def __init__(self, alpha=2.0, beta=75.0, base=0.5, hard_neg_k=20, temperature=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.hard_neg_k = hard_neg_k
        self.T = temperature

    def forward(self, embeddings, labels):
        # Raw cosine similarity
        sim_raw = torch.matmul(embeddings, embeddings.t())
        # Temperature-scaled version for loss computation only
        sim = sim_raw / self.T

        labels_ = labels.unsqueeze(1)
        pos_mask = (labels_ == labels_.t())
        neg_mask = ~pos_mask

        # Init as tensor so .item() always works even if valid=0
        loss = torch.zeros(1, device=embeddings.device, requires_grad=True).squeeze()
        valid = 0
        base_scaled = self.base / self.T  # e.g. 0.5/0.2 = 2.5
        MAX_EXP = 80.0  # clamp to prevent exp() overflow (exp(80) ≈ 5e34, safe)

        for i in range(len(embeddings)):
            pos = sim[i][pos_mask[i]].clone()
            neg = sim[i][neg_mask[i]].clone()
            pos = pos[pos < (1.0 / self.T) - 1e-4]  # remove self-similarity (scaled)

            if len(pos) == 0 or len(neg) == 0:
                continue

            # Hard positive: most dissimilar same-identity (pose variation)
            # Clamp to avoid exp underflow/overflow
            pos_arg = torch.clamp(-self.alpha * (pos - base_scaled), max=MAX_EXP)
            pos_loss = (1.0 / self.alpha) * torch.log(
                1 + torch.sum(torch.exp(pos_arg))
            )

            # Top-K hardest negatives: highest-sim different-identity pairs
            k = min(self.hard_neg_k, len(neg))
            hard_neg, _ = neg.topk(k)
            neg_arg = torch.clamp(self.beta * (hard_neg - base_scaled), max=MAX_EXP)
            neg_loss = (1.0 / self.beta) * torch.log(
                1 + torch.sum(torch.exp(neg_arg))
            )

            loss = loss + pos_loss + neg_loss
            valid += 1

        return loss / max(valid, 1)


class IdentityCenterLoss(nn.Module):
    \"\"\"Pull all embeddings of same identity toward centroid.\"\"\"
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
            dists = ((group_emb - center) ** 2).sum(dim=1)
            loss += dists.mean()
            count += 1
        return loss / max(count, 1)


# Loss weights
CENTER_LOSS_WEIGHT  = 0.1
ARCFACE_LOSS_WEIGHT = 0.25

criterion_ms     = MultiSimilarityLossV3(alpha=2.0, beta=75.0, base=0.5, hard_neg_k=20, temperature=0.2)
criterion_center = IdentityCenterLoss()

print("Loss 1: MS-Loss v3 (beta=75, hard_neg_k=20, T=0.2)")
print(f"Loss 2: Identity Center Loss (λ={CENTER_LOSS_WEIGHT})")
print(f"Loss 3: ArcFace Regularizer  (λ={ARCFACE_LOSS_WEIGHT}, margin=0.25)")
print(f"Total  = MS_Loss + {CENTER_LOSS_WEIGHT}×Center + {ARCFACE_LOSS_WEIGHT}×ArcFace")""")

# ============================================================================
# CELL 8 — Diversity Sampler + Worst-Pair Replay
# ============================================================================
add_md("""## Cell 7 — Diversity Sampler + Worst-Pair Replay

**New in v3: Worst-pair replay**

From v2 evaluation, we know which specific identity pairs are the hardest:
```
JM_ELE_84 ↔ JU_ELE_96: 0.9998
Calf_68 ↔ JU_ELE_96: 0.9998
AF_ELE_22_R ↔ Calf_68: 0.9998
```

Every 5 epochs, we force these pairs into the same batch. This is more effective
than just increasing K, because these pairs may never appear together in random sampling.""")

add_code("""# Known worst pairs from v2 evaluation (identity name → dataset label idx)
# These are computed dynamically at runtime using the class-to-idx map
KNOWN_WORST_PAIRS_NAMES = [
    ("JM_ELE_84", "JU_ELE_96"),
    ("Calf_68",   "JU_ELE_96"),
    ("AF_ELE_22_R", "Calf_68"),
    ("Calf_56",   "JU_ELE_94"),
    ("Calf_57",   "JF_ELE_74"),
    ("AF_ELE_15", "AF_ELE_16_R"),
    ("JM_ELE_86", "SAF_ELE_113"),
]


def resolve_worst_pairs(dataset, worst_pairs_names):
    \"\"\"Map identity short-names to dataset label indices for worst-pair replay.\"\"\"
    # Build short_name → label_idx map (key = last path component)
    short_to_label = {}
    for full_name, label_idx in dataset.class_to_idx.items():
        short = full_name.split(os.sep)[-1]
        short_to_label[short] = label_idx

    resolved = []
    for a, b in worst_pairs_names:
        la = short_to_label.get(a)
        lb = short_to_label.get(b)
        if la is not None and lb is not None:
            resolved.append((la, lb))
            print(f"  ✓ Worst pair registered: {a}({la}) ↔ {b}({lb})")
        else:
            missing = [n for n, l in [(a, la), (b, lb)] if l is None]
            print(f"  ⚠ Could not resolve: {missing} (may have been cleaned/merged)")
    return resolved


resolved_worst_pairs = resolve_worst_pairs(train_dataset, KNOWN_WORST_PAIRS_NAMES)
print(f"Registered {len(resolved_worst_pairs)} worst pairs for replay")


class DiversityPxMSampler:
    \"\"\"
    P × M sampler with:
    - Greedy farthest-point positive selection (v2 feature)
    - Worst-pair replay: forces known hard-negative identity pairs into same batch
    \"\"\"
    def __init__(self, dataset, model, device, P=8, M=3,
                 refresh_interval=3, worst_pairs=None, replay_interval=5):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.P = P
        self.M = M
        self.refresh_interval = refresh_interval
        self.replay_interval = replay_interval
        self.worst_pairs = worst_pairs or []
        self.label_to_indices = dataset.label_to_indices
        self.available_labels = [l for l, idx in self.label_to_indices.items()
                                 if len(idx) >= 2]
        self._identity_distances = {}
        self._embeddings = None

    def refresh_embeddings(self, epoch):
        if epoch % self.refresh_interval != 0:
            return
        print(f"  [Sampler] Refreshing embeddings for diversity sampling...")
        self.model.eval()
        all_emb = []
        with torch.no_grad():
            all_indices = list(range(len(self.dataset)))
            for start in range(0, len(all_indices), 32):
                batch_idx = all_indices[start:start+32]
                batch = [self.dataset[i] for i in batch_idx]
                imgs = torch.stack([x[0] for x in batch]).to(self.device)
                embs = self.model(imgs)
                all_emb.append(embs.cpu())
        self._embeddings = torch.cat(all_emb)
        for label in self.available_labels:
            indices = self.label_to_indices[label]
            if len(indices) < 2:
                continue
            embs = self._embeddings[indices]
            sim = embs @ embs.t()
            dist = 1.0 - sim
            self._identity_distances[label] = dist.numpy()
        self.model.train()
        print(f"  [Sampler] Refreshed {len(self._identity_distances)} identity distance matrices")

    def _select_diverse_M(self, label):
        indices = self.label_to_indices[label]
        if len(indices) <= self.M:
            selected = list(indices)
            while len(selected) < self.M:
                selected.append(random.choice(indices))
            return selected
        if label not in self._identity_distances:
            return random.sample(indices, self.M)
        dist_mat = self._identity_distances[label]
        n = len(indices)
        selected_local = [random.randint(0, n - 1)]
        for _ in range(self.M - 1):
            best_idx, best_min_dist = -1, -1.0
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

    def sample_batch(self, epoch=None):
        \"\"\"
        Sample P×M batch. If worst-pair replay is active, inject worst-pair
        labels to guarantee they appear in the batch.
        \"\"\"
        # Worst-pair replay: force hardest pairs every replay_interval epochs
        forced_labels = set()
        if epoch is not None and epoch % self.replay_interval == 0 and self.worst_pairs:
            pair = random.choice(self.worst_pairs)
            forced_labels.update(pair)

        # Select P labels (respect forced labels)
        available = [l for l in self.available_labels if l not in forced_labels]
        remaining_P = max(0, self.P - len(forced_labels))
        actual_extra = min(remaining_P, len(available))
        selected_labels = list(forced_labels) + random.sample(available, actual_extra)

        batch_indices = []
        for label in selected_labels:
            batch_indices.extend(self._select_diverse_M(label))

        batch = [self.dataset[i] for i in batch_indices]
        imgs = torch.stack([x[0] for x in batch])
        labels = torch.tensor([x[1] for x in batch])
        return imgs, labels


sampler = DiversityPxMSampler(
    train_dataset, model, device,
    P=8, M=3, refresh_interval=3,
    worst_pairs=resolved_worst_pairs,
    replay_interval=5
)
print("Sampler initialized with worst-pair replay every 5 epochs")""")

# ============================================================================
# CELL 9 — Training Setup
# ============================================================================
add_md("## Cell 8 — Training Setup (Warmup + Differential LR)")
add_code("""TOTAL_EPOCHS = 30
WARMUP_EPOCHS = 5
P = 8
M = 3
BATCH_SIZE = P * M

steps_per_epoch = max(1, len(train_dataset) // BATCH_SIZE)

# Start frozen (embedding head only)
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW([
    {'params': model.embed.parameters(),    'lr': 1e-3,  'weight_decay': 1e-4},
    {'params': arcface_head.parameters(),   'lr': 1e-3,  'weight_decay': 1e-4},
], weight_decay=1e-4)

print(f"Phase 1 (epochs 1-{WARMUP_EPOCHS}): Backbone FROZEN — embedding head + ArcFace head")
print(f"Phase 2 (epochs {WARMUP_EPOCHS+1}-{TOTAL_EPOCHS}): Full fine-tune (differential LR)")
print(f"Batch: P={P} × M={M} = {BATCH_SIZE}")
print(f"Steps/epoch: {steps_per_epoch}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")""")

# ============================================================================
# CELL 10 — Training Loop
# ============================================================================
add_md("## Cell 9 — Training Loop")
add_code("""history = {
    'epoch': [], 'loss': [], 'ms_loss': [],
    'center_loss': [], 'arcface_loss': [], 'phase': []
}

for epoch in range(1, TOTAL_EPOCHS + 1):

    if epoch == WARMUP_EPOCHS + 1:
        print(f"\\n{'='*60}")
        print(f"PHASE 2: Unfreezing backbone with differential LR")
        print(f"{'='*60}")
        for param in model.backbone.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(),  'lr': 5e-5,  'weight_decay': 1e-4},
            {'params': model.embed.parameters(),     'lr': 5e-4,  'weight_decay': 1e-4},
            {'params': arcface_head.parameters(),    'lr': 5e-4,  'weight_decay': 1e-4},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS
        )
        print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    sampler.refresh_embeddings(epoch)
    model.train()
    arcface_head.train()

    total_loss = total_ms = total_center = total_arc = 0
    n_batches = 0

    for step in range(steps_per_epoch):
        imgs, labels = sampler.sample_batch(epoch=epoch)
        imgs   = imgs.to(device)
        labels = labels.to(device)

        embeddings = model(imgs)

        ms_loss     = criterion_ms(embeddings, labels)
        center_loss = criterion_center(embeddings, labels)
        arc_loss    = arcface_head(embeddings, labels)

        loss = ms_loss + CENTER_LOSS_WEIGHT * center_loss + ARCFACE_LOSS_WEIGHT * arc_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(arcface_head.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        total_loss   += loss.item()
        total_ms     += ms_loss.item()
        total_center += center_loss.item()
        total_arc    += arc_loss.item()
        n_batches    += 1

    if epoch > WARMUP_EPOCHS:
        scheduler.step()

    avg   = lambda x: x / max(1, n_batches)
    phase = "WARMUP" if epoch <= WARMUP_EPOCHS else "FULL"
    history['epoch'].append(epoch)
    history['loss'].append(avg(total_loss))
    history['ms_loss'].append(avg(total_ms))
    history['center_loss'].append(avg(total_center))
    history['arcface_loss'].append(avg(total_arc))
    history['phase'].append(phase)

    lr = optimizer.param_groups[0]['lr']
    replay = " [REPLAY]" if epoch % 5 == 0 else ""
    print(f"Epoch {epoch:2d}/{TOTAL_EPOCHS} [{phase:6s}]{replay} "
          f"Loss={avg(total_loss):.4f} "
          f"MS={avg(total_ms):.4f} "
          f"Center={avg(total_center):.4f} "
          f"ArcFace={avg(total_arc):.4f}  "
          f"LR={lr:.6f}")

print("\\n✅ Training complete!")""")

# ============================================================================
# CELL 11 — Loss Curves
# ============================================================================
add_md("## Cell 10 — Loss Curves")
add_code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

colors = ['orange' if p == 'WARMUP' else 'steelblue' for p in history['phase']]
axes[0].bar(history['epoch'], history['loss'], color=colors, alpha=0.8)
axes[0].set_title('Total Loss (orange=warmup)')
axes[0].set_xlabel('Epoch')

axes[1].plot(history['epoch'], history['ms_loss'],     'b-o', label='MS Loss', markersize=3)
axes[1].plot(history['epoch'], history['center_loss'], 'r-o', label='Center',  markersize=3)
axes[1].set_title('MS + Center Loss')
axes[1].legend()
axes[1].set_xlabel('Epoch')

axes[2].plot(history['epoch'], history['arcface_loss'], 'g-o', label='ArcFace', markersize=3)
axes[2].set_title('ArcFace Regularizer Loss')
axes[2].legend()
axes[2].set_xlabel('Epoch')

plt.tight_layout()
plt.show()""")

# ============================================================================
# CELL 12 — Evaluation
# ============================================================================
add_md("## Cell 11 — Evaluate: Separation + Ambiguity Metrics")
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

all_emb    = torch.cat(all_emb).numpy()
all_labels = torch.cat(all_labels).numpy()
print(f"Embeddings: {all_emb.shape}")

sim_matrix = all_emb @ all_emb.T

same_sims = []
diff_sims = []
n = len(all_labels)

for i in range(n):
    for j in range(i + 1, n):
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
print(f"\\n{'='*55}")
print(f"SEPARATION GAP (sim): {separation:.4f}")
print(f"{'='*55}")

# v3 uses a harder threshold: 0.75 (not 0.65)
false_merge_rate_065 = (diff_sims > 0.65).mean() * 100
false_merge_rate_075 = (diff_sims > 0.75).mean() * 100
fragmentation_rate   = (same_sims < 0.65).mean() * 100

print(f"\\n🚨 False-merge @ 0.65: {false_merge_rate_065:.1f}%   (target: < 5%)")
print(f"🚨 False-merge @ 0.75: {false_merge_rate_075:.1f}%   (target: < 3%)  <-- new threshold")
print(f"   Worst diff-ID sim:  {np.max(diff_sims):.4f}         (v2 was 0.9998 — target: < 0.90)")
print(f"\\n⚠️  Fragmentation @ 0.65: {fragmentation_rate:.1f}%")

# 📊 Ambiguity metric: for each query, compute gap between best and 2nd-best match
np.fill_diagonal(sim_matrix, -1)
gaps = []
for i in range(n):
    ranked_sims = np.sort(sim_matrix[i])[::-1]
    if len(ranked_sims) >= 2:
        gaps.append(ranked_sims[0] - ranked_sims[1])

gaps = np.array(gaps)
print(f"\\n📊 AMBIGUITY ANALYSIS")
print(f"  Mean gap (best vs 2nd-best): {np.mean(gaps):.4f}")
print(f"  Ambiguous queries (gap < 0.10): {(gaps < 0.10).mean()*100:.1f}%")
print(f"  High-confidence queries (gap > 0.20): {(gaps > 0.20).mean()*100:.1f}%")

print(f"\\n{'='*55}")
if separation > 0.50 and false_merge_rate_075 < 3:
    print("🟢 EXCELLENT — model ready for production")
elif separation > 0.40 and false_merge_rate_075 < 5:
    print("🟢 GOOD — suitable for conservative deployment")
elif separation > 0.30:
    print("🟡 OKAY — usable with strict thresholds")
else:
    print("🔴 BAD — model not separating identities")""")

# ============================================================================
# CELL 13 — Separation Histogram
# ============================================================================
add_md("## Cell 12 — Separation Histogram")
add_code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.hist(same_sims, bins=50, alpha=0.6, color='green',
         label=f'Same ID (mean={np.mean(same_sims):.3f})', density=True)
ax1.hist(diff_sims, bins=50, alpha=0.6, color='red',
         label=f'Diff ID (mean={np.mean(diff_sims):.3f})', density=True)
ax1.axvline(0.75, color='black',  linestyle='--', linewidth=2, label='Threshold 0.75')
ax1.axvline(0.65, color='gray',   linestyle='--', linewidth=1, label='Old Threshold 0.65', alpha=0.6)
ax1.axvline(np.mean(same_sims),   color='darkgreen', linestyle=':', linewidth=2)
ax1.axvline(np.mean(diff_sims),   color='darkred',   linestyle=':', linewidth=2)
ax1.set_xlabel('Cosine Similarity', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title(f'v3 Identity Separation (Gap = {separation:.4f})', fontsize=14)
ax1.legend(fontsize=10)

ax2.hist(diff_sims[diff_sims > 0.4], bins=30, alpha=0.6, color='red',
         label='Diff-ID (sim > 0.4)', density=True)
ax2.hist(same_sims[same_sims > 0.4], bins=30, alpha=0.6, color='green',
         label='Same-ID (sim > 0.4)', density=True)
ax2.axvline(0.75, color='black', linestyle='--', linewidth=2, label='New Threshold 0.75')
ax2.axvline(0.65, color='gray',  linestyle='--', linewidth=1, label='Old Threshold 0.65', alpha=0.6)
ax2.set_xlabel('Cosine Similarity', fontsize=12)
ax2.set_title('Overlap Zone Detail (sim > 0.4)', fontsize=14)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.show()""")

# ============================================================================
# CELL 14 — Per-Identity + Worst Pairs
# ============================================================================
add_md("## Cell 13 — Per-Identity Analysis & Worst Cross-ID Matches")
add_code("""from itertools import combinations

print("PER-IDENTITY INTRA-SIMILARITY")
print("=" * 55)
unique_labels = sorted(set(all_labels))
for label in unique_labels:
    mask = all_labels == label
    if mask.sum() < 2:
        continue
    indices = np.where(mask)[0]
    intra = [sim_matrix[i][j] for i, j in combinations(indices, 2)]
    name = eval_dataset.idx_to_class.get(label, f"ID_{label}")
    short = name.split(os.sep)[-1]
    mean_sim = np.mean(intra)
    min_sim  = np.min(intra)
    status = "✅" if min_sim > 0.50 else "⚠️" if min_sim > 0.30 else "❌"
    print(f"  {status} {short:25s}: mean={mean_sim:.4f} min={min_sim:.4f} (n={mask.sum()})")

# Worst cross-identity false matches
print(f"\\n{'='*55}")
print("TOP-15 WORST CROSS-IDENTITY FALSE MATCHES")
print("=" * 55)
worst_pairs_eval = []
for i in range(n):
    for j in range(i + 1, n):
        if all_labels[i] != all_labels[j]:
            worst_pairs_eval.append((sim_matrix[i][j], i, j))
worst_pairs_eval.sort(reverse=True)

for sim_val, i, j in worst_pairs_eval[:15]:
    name_i = eval_dataset.idx_to_class.get(all_labels[i], "?").split(os.sep)[-1]
    name_j = eval_dataset.idx_to_class.get(all_labels[j], "?").split(os.sep)[-1]
    flag = "🔴" if sim_val > 0.90 else "🟠" if sim_val > 0.80 else "🟡"
    print(f"  {flag} {name_i} ↔ {name_j}: {sim_val:.4f}")""")

# ============================================================================
# CELL 15 — Rank-1 / Rank-5
# ============================================================================
add_md("## Cell 14 — Rank-1 / Rank-5 Accuracy")
add_code("""rank1_correct = 0
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

save_dict = {
    'model_state_dict': model.state_dict(),
    'embed_dim': 256,
    'num_classes': train_dataset.num_classes,
    'separation_gap': float(separation),
    'false_merge_rate_at_065': float(false_merge_rate_065),
    'false_merge_rate_at_075': float(false_merge_rate_075),
    'fragmentation_rate_at_065': float(fragmentation_rate),
    'training_config': {
        'version': 'v3_ambiguity_aware',
        'loss': 'MSLossV3 + CenterLoss + ArcFaceRegularizer',
        'ms_beta': 75.0,
        'ms_hard_neg_k': 20,
        'ms_temperature': 0.2,
        'arcface_margin': 0.25,
        'arcface_scale': 32.0,
        'arcface_weight': ARCFACE_LOSS_WEIGHT,
        'center_loss_weight': CENTER_LOSS_WEIGHT,
        'P': P, 'M': M,
        'epochs': TOTAL_EPOCHS,
        'warmup_epochs': WARMUP_EPOCHS,
        'min_images_per_id_train': 3,
        'diversity_sampling': True,
        'worst_pair_replay_interval': 5,
        'data_version': 'cleaned_v1',
    },
}

torch.save(save_dict, "/kaggle/working/elephant_head_reid_v3.pth")
shutil.copy2(
    "/kaggle/working/elephant_head_reid_v3.pth",
    "/kaggle/working/elephant_head_reid_v3_download.pth"
)

print("✅ Model saved: elephant_head_reid_v3.pth")
print(f"   Version:            v3_ambiguity_aware")
print(f"   Separation gap:     {separation:.4f}")
print(f"   False merge @ 0.65: {false_merge_rate_065:.1f}%")
print(f"   False merge @ 0.75: {false_merge_rate_075:.1f}%  <-- new operating threshold")
print(f"   Download file:      elephant_head_reid_v3_download.pth")
print()
print("Place in models/elephant_head_reid_v3.pth — pipeline.py auto-detects it.")""")

# ============================================================================
# Write notebook
# ============================================================================
import os
out_path = os.path.join(os.path.dirname(__file__), "elephant-head-embedding-training-v3.ipynb")

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook generated: {out_path}")
print(f"Cells: {len(cells)}")
