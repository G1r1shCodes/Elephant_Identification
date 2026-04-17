"""
Generate Kaggle notebook: Elephant Head Re-ID v7.0 — Identity Cue Forcing

Forces the model to use fine-grained identity cues (ear tears, wrinkle patterns)
by changing the TRAINING SIGNAL, not the architecture.

Key v7 upgrades:
  1. Hard-positive mining — explicitly forces anchor <-> hardest positive alignment
  2. Region-disrupting augmentation — RandomResizedCrop breaks global shape reliance
  3. ArcFace weight reduced (0.25 -> 0.10) — lets MS-Loss drive fine-grained separation
  4. v3 augmentation + v6 clean data + honest zero-shot eval
"""

import json

cells = []


def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})


def add_code(text):
    lines = [line + "\n" for line in text.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines,
        }
    )


# ============================================================================
add_md("""# 🐘 Elephant Head Re-ID v7.0 — Identity Cue Forcing

**Why v7?**
Previous models learn "overall head shape + texture = identity" but ignore fine-grained cues like **ear tears, wrinkle patterns, vein maps**. This is why Makhna_6 vs Makhna_7 score 0.87 despite a clear ear notch difference.

**The feature exists in the image. The training signal was not forcing the model to USE it.**

**v7 fixes (training signal, NOT architecture):**
1. **Hard-Positive Alignment Loss** — explicitly forces anchor ↔ hardest-positive alignment, making the model learn that ear tear = decisive cue
2. **Region-disrupting augmentation** — `RandomResizedCrop(224, 0.7-1.0)` forces model to not rely only on global shape
3. **ArcFace weight reduced** (0.25 → 0.10) — less global class separation, more fine-grained cue learning
4. **v3's strong augmentation** + **v6's clean data** + **honest zero-shot eval**

**Expected result:** Makhna_6 vs 7 similarity: 0.87 → ~0.70-0.78""")

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
from itertools import combinations
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)""")

# ============================================================================
add_md("## Cell 2 — Dataset (Disjoint Identity Split)")
add_code("""class HeadCropDataset(Dataset):
    def __init__(self, root_dir, transform=None, min_images=3, split='train', split_ratio=0.8):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.label_to_indices = defaultdict(list)

        SKIP_DIRS = {'_quarantined', '_filtered', '_no_head_in_crop'}
        BAD_CLASSES = {'Herd_1_AF_1', 'Herd_3_AF_ELE_12', 'Herd_4_AF_ELE_1', 
                       'Herd_4_AF_ELE_31', 'Makhna_4', 'Makhna_2', 'Herd_4_JF_ELE_79_L',
                       'Herd_4_JF_ELE_80', 'Herd_4_AF_ELE_5', 'Herd_4_AF_ELE_38',
                       'Herd_2_JM_ELE_48', 'Herd_4_JU_ELE_98_L',
                       'Herd_4_AF_ELE_9', 'Makhna_3', 'Herd_4_AF_ELE_12', 'Herd_2_AF_ELE_8', 
                       'Makhna_1', 'Herd_2_AF_ELE_6', 'Herd_4_Calf_55', 'Herd_2_AF_ELE_7', 
                       'Herd_2_AF_ELE_15', 'Herd_3_AF_ELE_3'}

        all_folders = {}
        for dirpath, dirnames, filenames in os.walk(root_dir):
            parts = set(Path(dirpath).parts)
            if parts & SKIP_DIRS or Path(dirpath).name in BAD_CLASSES:
                continue
            images = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) >= min_images:
                rel = os.path.relpath(dirpath, root_dir)
                all_folders[rel] = sorted([os.path.join(dirpath, f) for f in images])

        valid_identities = sorted(all_folders.keys())
        random.seed(SEED)
        random.shuffle(valid_identities)
        
        n_train = int(len(valid_identities) * split_ratio)
        train_ids = set(valid_identities[:n_train])
        eval_ids = set(valid_identities[n_train:])

        target = train_ids if split == 'train' else (eval_ids if split == 'eval' else valid_identities)
        
        idx = 0
        for name in sorted(target):
            paths = all_folders[name]
            self.class_to_idx[name] = idx
            self.idx_to_class[idx] = name
            for p in paths:
                self.label_to_indices[idx].append(len(self.samples))
                self.samples.append((p, idx))
            idx += 1

        self.num_classes = idx
        print(f"[{split.upper()}] {len(self.samples)} images, {idx} identities")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label""")

# ============================================================================
add_md("""## Cell 3 — Transforms & Load Dataset

**v7 augmentation: minimal + safe.** The real fix is hard-positive mining, not augmentation.
Heavy perspective/erasing would destroy the very identity cues (ear tears) we need the model to learn.""")
add_code("""# v7: SAFE minimal augmentation — nudge, don't destroy
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),     # conservative crop jitter
    transforms.RandomHorizontalFlip(p=0.5),                  # elephants are symmetric enough
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.06)),     # tiny patches only — don't destroy ear
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sampler_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DATASET_PATH = Path("/kaggle/input/elephant-training-heads-v6/training_heads_v6")

print(f"Dataset path: {DATASET_PATH}")
train_dataset = HeadCropDataset(str(DATASET_PATH), train_transform, min_images=3, split='train')
eval_dataset  = HeadCropDataset(str(DATASET_PATH), eval_transform,  min_images=3, split='eval')
full_eval_dataset = HeadCropDataset(str(DATASET_PATH), eval_transform, min_images=3, split='all')""")

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
plt.suptitle("Head Crop Samples (Held-Out Eval Set)", fontsize=14)
plt.tight_layout()
plt.show()""")

# ============================================================================
add_md("## Cell 5 — Embedding Model (ConvNeXt-Tiny)")
add_code("""class HeadEmbeddingModel(nn.Module):
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
        feat = self.pool(feat).flatten(1)
        emb = self.embed(feat)
        return F.normalize(emb, p=2, dim=1)

class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim, num_classes, margin=0.35, scale=32.0):
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
        cosine = F.linear(embeddings, F.normalize(self.weight))
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return F.cross_entropy(output * self.scale, labels)

model = HeadEmbeddingModel(embed_dim=256).to(device)
arcface_head = ArcFaceHead(embed_dim=256, num_classes=train_dataset.num_classes).to(device)
print(f"Embedding model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"ArcFace head classes: {train_dataset.num_classes}")""")

# ============================================================================
add_md("""## Cell 6 — Loss Functions (v7: Identity Cue Forcing)

**Five losses working together:**
1. **MS-Loss v3** — alpha=2.0 (gentler), beta=75, hard_neg_k=20
2. **Hard-Positive Alignment** — ***NEW*** — explicitly forces anchor ↔ hardest-positive alignment
3. **Identity Center Loss** — stabilizes clusters
4. **ArcFace** — weight REDUCED 0.25 → 0.10 (less global push, more fine-grained room)
5. **BatchHard Triplet** — additional signal

```
Total = MS + 0.5×HardPosAlign + 0.1×Center + 0.10×ArcFace + 0.5×Triplet
```

**Why Hard-Positive Alignment matters:**
For Makhna_6, the hardest positive = different pose where ear tear is visible.
Forcing the model to align anchor ↔ hardest-positive teaches: `ear tear = identity cue`.""")
add_code("""class MultiSimilarityLossV3(nn.Module):
    def __init__(self, alpha=2.0, beta=75.0, base=0.5, hard_neg_k=20, temperature=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.hard_neg_k = hard_neg_k
        self.T = temperature

    def forward(self, embeddings, labels):
        sim = torch.matmul(embeddings, embeddings.t()) / self.T
        labels_ = labels.unsqueeze(1)
        pos_mask = (labels_ == labels_.t())
        loss = torch.zeros(1, device=embeddings.device, requires_grad=True).squeeze()
        valid = 0
        base_scaled = self.base / self.T
        MAX_EXP = 80.0

        for i in range(len(embeddings)):
            pos = sim[i][pos_mask[i]].clone()
            neg = sim[i][~pos_mask[i]].clone()
            pos = pos[pos < (1.0 / self.T) - 1e-4]
            if len(pos) == 0 or len(neg) == 0:
                continue
            pos_arg = torch.clamp(-self.alpha * (pos - base_scaled), max=MAX_EXP)
            pos_loss = (1.0 / self.alpha) * torch.log(1 + torch.sum(torch.exp(pos_arg)))
            hard_neg, _ = neg.topk(min(self.hard_neg_k, len(neg)))
            neg_arg = torch.clamp(self.beta * (hard_neg - base_scaled), max=MAX_EXP)
            neg_loss = (1.0 / self.beta) * torch.log(1 + torch.sum(torch.exp(neg_arg)))
            loss = loss + pos_loss + neg_loss
            valid += 1
        return loss / max(valid, 1)


class HardPositiveAlignmentLoss(nn.Module):
    \"\"\"Forces anchor to align with its HARDEST positive (most dissimilar same-ID image).
    
    This is the key missing signal: without it, the model learns 'average head shape'
    instead of 'this specific ear notch is decisive'.
    
    For each anchor:
      1. Find its hardest positive (lowest cosine sim, same identity)
      2. Penalize (1 - similarity)
    
    This forces the model to embed ALL views of the same elephant nearby,
    including the unusual pose where the ear tear is visible.
    \"\"\"
    def __init__(self):
        super().__init__()
    
    def forward(self, embeddings, labels):
        sim = embeddings @ embeddings.t()
        
        loss = torch.zeros(1, device=embeddings.device, requires_grad=True).squeeze()
        valid = 0
        total_hardest_pos = 0.0
        
        for i in range(len(embeddings)):
            pos_mask = (labels == labels[i])
            pos_mask[i] = False
            
            pos_sims = sim[i][pos_mask]
            if len(pos_sims) == 0:
                continue
            
            # Hardest positive = the one with LOWEST similarity
            hardest_pos_sim = pos_sims.min()
            total_hardest_pos += hardest_pos_sim.item()
            
            # Penalize if hardest positive is far away
            loss = loss + (1.0 - hardest_pos_sim)
            valid += 1
            
        mean_hardest_pos = total_hardest_pos / max(valid, 1)
        return loss / max(valid, 1), mean_hardest_pos


class IdentityCenterLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, embeddings, labels):
        unique_labels = labels.unique()
        loss = 0.0
        for lbl in unique_labels:
            mask = labels == lbl
            group_emb = embeddings[mask]
            if len(group_emb) < 2: continue
            center = group_emb.mean(dim=0, keepdim=True)
            loss += ((group_emb - center) ** 2).sum(dim=1).mean()
        return loss / max(len(unique_labels), 1)


class GlobalBatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.50, memory_size=2048, embed_dim=256):
        super().__init__()
        self.margin = margin
        self.memory_size = memory_size
        self.register_buffer('memory_emb', torch.zeros(memory_size, embed_dim))
        self.register_buffer('memory_labels', torch.zeros(memory_size, dtype=torch.long) - 1)
        self.ptr = 0

    def forward(self, embeddings, labels):
        sim = embeddings @ embeddings.t()
        
        valid_mask = self.memory_labels != -1
        global_sim = None
        if valid_mask.any():
            valid_mem_emb = self.memory_emb[valid_mask]
            valid_mem_labels = self.memory_labels[valid_mask]
            global_sim = embeddings @ valid_mem_emb.t()

        loss = torch.zeros(1, device=embeddings.device, requires_grad=True).squeeze()
        valid = 0
        
        for i in range(len(embeddings)):
            pos_mask = (labels == labels[i])
            pos_mask[i] = False
            pos_sim = sim[i][pos_mask]
            
            if len(pos_sim) == 0: continue
            hardest_pos = pos_sim.min()
            
            neg_mask = (labels != labels[i])
            local_neg_sim = sim[i][neg_mask]
            
            best_neg = torch.tensor(-1.0, device=embeddings.device)
            if len(local_neg_sim) > 0:
                best_neg = local_neg_sim.max()
                
            if global_sim is not None:
                g_neg_mask = (valid_mem_labels != labels[i])
                g_neg_sim = global_sim[i][g_neg_mask]
                if len(g_neg_sim) > 0:
                    best_g_neg = g_neg_sim.max()
                    if best_neg.item() == -1.0:
                        best_neg = best_g_neg
                    else:
                        best_neg = torch.max(best_neg, best_g_neg)
                    
            if best_neg.item() == -1.0: continue

            l = torch.relu(best_neg - hardest_pos + self.margin)
            loss = loss + l
            valid += 1
            
        batch_size = embeddings.size(0)
        with torch.no_grad():
            if self.ptr + batch_size <= self.memory_size:
                self.memory_emb[self.ptr:self.ptr+batch_size] = embeddings.detach()
                self.memory_labels[self.ptr:self.ptr+batch_size] = labels.detach()
                self.ptr = (self.ptr + batch_size) % self.memory_size
            else:
                overflow = (self.ptr + batch_size) - self.memory_size
                self.memory_emb[self.ptr:] = embeddings.detach()[:batch_size-overflow]
                self.memory_labels[self.ptr:] = labels.detach()[:batch_size-overflow]
                self.memory_emb[:overflow] = embeddings.detach()[batch_size-overflow:]
                self.memory_labels[:overflow] = labels.detach()[batch_size-overflow:]
                self.ptr = overflow

        return loss / max(valid, 1)

MS_LOSS_WEIGHT = 0.0
CENTER_LOSS_WEIGHT = 0.1
ARCFACE_LOSS_WEIGHT = 0.6     
TRIPLET_LOSS_WEIGHT = 1.0      
HARD_POS_WEIGHT = 0.5          

criterion_ms = MultiSimilarityLossV3()
criterion_hard_pos = HardPositiveAlignmentLoss()
criterion_center = IdentityCenterLoss()
criterion_triplet = GlobalBatchHardTripletLoss(margin=0.50, memory_size=2048, embed_dim=256).to(device)

print("Loss 1: MS-Loss v3 (alpha=2.0, beta=75, hard_neg_k=20, T=0.2)")
print(f"Loss 2: Hard-Positive Alignment (weight={HARD_POS_WEIGHT})")
print(f"Loss 3: Identity Center Loss (weight={CENTER_LOSS_WEIGHT})")
print(f"Loss 4: ArcFace Regularizer  (weight={ARCFACE_LOSS_WEIGHT}, margin=0.35)")
print(f"Loss 5: Global BatchHard Triplet (weight={TRIPLET_LOSS_WEIGHT}, margin=0.50, memory=2048)")""")

# ============================================================================
add_md("## Cell 7 — Diversity Sampler (P=8, M=3)")
add_code("""class DiversityPxMSampler:
    def __init__(self, dataset, model, device, P=8, M=3, refresh_interval=3):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.P = P
        self.M = M
        self.refresh_interval = refresh_interval
        self.label_to_indices = dataset.label_to_indices
        self.available_labels = [lbl for lbl, idxs in self.label_to_indices.items() if len(idxs) >= M]
        self._identity_distances = {}

    def refresh_embeddings(self, epoch):
        if epoch % self.refresh_interval != 0:
            return
        self.model.eval()
        old_transform = self.dataset.transform
        self.dataset.transform = sampler_transform
        
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
        )
        
        all_emb = []
        with torch.no_grad():
            for imgs, _ in loader:
                all_emb.append(self.model(imgs.to(self.device)).cpu())
                
        self._embeddings = torch.cat(all_emb)
        for label in self.available_labels:
            indices = self.label_to_indices[label]
            embs = self._embeddings[indices]
            self._identity_distances[label] = (1.0 - embs @ embs.t()).numpy()
        self.dataset.transform = old_transform
        self.model.train()

    def sample_batch(self, epoch=None):
        selected = random.sample(self.available_labels, min(self.P, len(self.available_labels)))
        batch_indices = []
        for l in selected:
            indices = self.label_to_indices[l]
            if l not in self._identity_distances or len(indices) <= self.M:
                batch_indices.extend(random.sample(indices, min(self.M, len(indices))))
                continue
            dist_mat = self._identity_distances[l]
            chosen_local = [random.randint(0, len(indices)-1)]
            for _ in range(self.M-1):
                best_idx, best_max = -1, -1.0
                for candidate in range(len(indices)):
                    if candidate in chosen_local: continue
                    min_dist = min(dist_mat[candidate][s] for s in chosen_local)
                    if min_dist > best_max:
                        best_max = min_dist; best_idx = candidate
                chosen_local.append(best_idx)
            batch_indices.extend([indices[i] for i in chosen_local])
            
        batch = [self.dataset[i] for i in batch_indices]
        return torch.stack([x[0] for x in batch]), torch.tensor([x[1] for x in batch])

sampler = DiversityPxMSampler(train_dataset, model, device, P=8, M=3)
print(f"Sampler: P=8, M=3 (batch={8*3}), {len(sampler.available_labels)} identities")""")

# ============================================================================
add_md("## Cell 8 — Training Setup")
add_code("""TOTAL_EPOCHS = 30
WARMUP_EPOCHS = 5
P, M = 8, 3
BATCH_SIZE = P * M
steps_per_epoch = max(1, len(train_dataset) // BATCH_SIZE)

for param in model.backbone.parameters(): param.requires_grad = False
optimizer = torch.optim.AdamW([
    {'params': model.embed.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
    {'params': arcface_head.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
])

print(f"Phase 1 (epochs 1-{WARMUP_EPOCHS}): Backbone FROZEN")
print(f"Phase 2 (epochs {WARMUP_EPOCHS+1}-{TOTAL_EPOCHS}): Full fine-tune")
print(f"Batch: P={P} x M={M} = {BATCH_SIZE}")
print(f"Steps/epoch: {steps_per_epoch}")""")

# ============================================================================
add_md("## Cell 9 — Training Loop")
add_code("""history = {
    'epoch': [], 'loss': [], 'ms_loss': [], 'hard_pos_loss': [],
    'center_loss': [], 'arcface_loss': [], 'triplet_loss': [], 'topk_loss': [], 'phase': [],
    'mean_hard_sim': [], 'mean_gap': []
}

history = {
    'epoch': [], 'loss': [], 'ms_loss': [], 'hard_pos_loss': [],
    'center_loss': [], 'arcface_loss': [], 'triplet_loss': [], 'phase': [],
    'mean_hard_sim': []
}

for epoch in range(1, TOTAL_EPOCHS + 1):
    if epoch == WARMUP_EPOCHS + 1:
        print(f"
{'='*60}")
        print(f"PHASE 2: Unfreezing backbone with differential LR")
        print(f"{'='*60}")
        for param in model.backbone.parameters(): param.requires_grad = True
        optimizer.add_param_group({'params': model.backbone.parameters(), 'lr': 5e-6, 'weight_decay': 1e-4})
        optimizer.param_groups[0]['lr'] = 5e-5  # embed
        optimizer.param_groups[1]['lr'] = 5e-5  # arcface
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS)

    sampler.refresh_embeddings(epoch)
    model.train()
    arcface_head.train()
    total_loss = total_ms = total_hp = total_center = total_arc = total_trip = total_hard_sim = 0
    n_batches = 0

    for step in range(steps_per_epoch):
        imgs, labels = sampler.sample_batch(epoch)
        imgs, labels = imgs.to(device), labels.to(device)
        
        embeddings = model(imgs)
        ms_loss = criterion_ms(embeddings, labels)
        hard_pos_loss, batch_hard_sim = criterion_hard_pos(embeddings, labels)
        center_loss = criterion_center(embeddings, labels)
        arc_loss = arcface_head(embeddings, labels)
        triplet_loss = criterion_triplet(embeddings, labels)
        
        loss = (MS_LOSS_WEIGHT * ms_loss
                + HARD_POS_WEIGHT * hard_pos_loss
                + CENTER_LOSS_WEIGHT * center_loss
                + ARCFACE_LOSS_WEIGHT * arc_loss
                + TRIPLET_LOSS_WEIGHT * triplet_loss)
            
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(arcface_head.parameters()), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_ms += ms_loss.item()
        total_hp += hard_pos_loss.item()
        total_center += center_loss.item()
        total_arc += arc_loss.item()
        total_trip += triplet_loss.item()
        total_hard_sim += batch_hard_sim
        n_batches += 1

    if epoch > WARMUP_EPOCHS: scheduler.step()
    
    avg = lambda x: x / max(1, n_batches)
    phase = "WARMUP" if epoch <= WARMUP_EPOCHS else "FULL"
    history['epoch'].append(epoch)
    history['loss'].append(avg(total_loss))
    history['ms_loss'].append(avg(total_ms))
    history['hard_pos_loss'].append(avg(total_hp))
    history['center_loss'].append(avg(total_center))
    history['arcface_loss'].append(avg(total_arc))
    history['triplet_loss'].append(avg(total_trip))
    history['mean_hard_sim'].append(avg(total_hard_sim))
    history['phase'].append(phase)
    
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:2d}/{TOTAL_EPOCHS} [{phase:6s}] "
          f"Loss={avg(total_loss):.4f} HardSim={avg(total_hard_sim):.3f} "
          f"Trip={avg(total_trip):.4f} Arc={avg(total_arc):.4f} LR={lr:.6f}")""")

# ============================================================================
add_md("## Cell 10 — Loss Curves")
add_code("""fig, axes = plt.subplots(2, 3, figsize=(18, 10))

colors = ['orange' if p == 'WARMUP' else 'steelblue' for p in history['phase']]
axes[0,0].bar(history['epoch'], history['loss'], color=colors, alpha=0.8)
axes[0,0].set_title('Total Loss (orange=warmup)')

axes[0,1].plot(history['epoch'], history['ms_loss'], 'b-o', markersize=3, label='MS Loss')
axes[0,1].set_title('MS Loss')

axes[0,2].plot(history['epoch'], history['hard_pos_loss'], 'r-o', markersize=3, label='Hard-Pos')
axes[0,2].set_title('Hard-Positive Alignment')

axes[1,0].plot(history['epoch'], history['center_loss'], 'g-o', markersize=3, label='Center')
axes[1,0].set_title('Center Loss')

axes[1,1].plot(history['epoch'], history['arcface_loss'], 'm-o', markersize=3, label='ArcFace')
axes[1,1].set_title('ArcFace Loss')

axes[1,2].plot(history['epoch'], history['triplet_loss'], 'c-o', markersize=3, label='Triplet')
axes[1,2].set_title('Global Triplet Loss')

for ax in axes.flat:
    ax.set_xlabel('Epoch')
    ax.legend()
plt.tight_layout()
plt.show()""")

# ============================================================================
add_md("""## Cell 11 — ZERO-SHOT EVALUATION & FULL DATASET AUDIT

**v7.1: Flip TTA + Centroid matching added.**
Flip TTA averages embeddings of original + horizontally flipped image = reduces noise.
Centroid matching compares per-identity centroids instead of individual images = much more robust.""")
add_code("""# --- FLIP TTA helper ---
def embed_with_tta(model, dataloader, device):
    \"\"\"Embed all images with flip TTA: emb = avg(original, h-flipped).\"\"\"
    model.eval()
    all_emb, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            emb_orig = model(imgs)
            emb_flip = model(torch.flip(imgs, dims=[3]))  # horizontal flip
            emb = F.normalize((emb_orig + emb_flip) / 2, p=2, dim=1)  # average + renormalize
            all_emb.append(emb.cpu())
            all_labels.append(labels)
    return torch.cat(all_emb).numpy(), torch.cat(all_labels).numpy()

# --- CLUSTERING UTILS ---
def assign_to_centroids(query_embs, centroid_matrix, threshold=0.70):
    sims = query_embs @ centroid_matrix.T
    best_sim, best_idx = sims.max(axis=1) # query_embs/matrices are numpy

    assignments = []
    for i in range(len(query_embs)):
        if best_sim[i] >= threshold:
            assignments.append(best_idx[i])
        else:
            assignments.append(-1)  # unknown / new identity

    return assignments, best_sim

def filter_outliers(embeddings, labels, threshold=0.5):
    # Pass tensors to this function, returns list of valid indices
    cleaned = []
    for lbl in set(labels.tolist()):
        idxs = [i for i, l in enumerate(labels) if l.item() == lbl]
        group = embeddings[idxs]
        centroid = F.normalize(group.mean(dim=0), p=2, dim=0)

        sims = group @ centroid
        for i, sim in zip(idxs, sims):
            if sim >= threshold:
                cleaned.append(i)

    return cleaned

# --- PASS 1: ZERO-SHOT EVALUATION with Flip TTA ---
model.eval()
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
all_emb, all_labels = embed_with_tta(model, eval_loader, device)
sim_matrix = all_emb @ all_emb.T
n = len(all_labels)

same_sims, diff_sims = [], []
for i in range(n):
    for j in range(i + 1, n):
        sim = sim_matrix[i][j]
        if all_labels[i] == all_labels[j]: same_sims.append(sim)
        else: diff_sims.append(sim)
same_sims, diff_sims = np.array(same_sims), np.array(diff_sims)
separation = np.mean(same_sims) - np.mean(diff_sims)
fm_065 = (diff_sims > 0.65).mean() * 100
fm_075 = (diff_sims > 0.75).mean() * 100
fragmentation = (same_sims < 0.65).mean() * 100

print(f"=======================================================")
print(f"ZERO-SHOT EVALUATION ({len(eval_dataset.samples)} imgs, {eval_dataset.num_classes} identities)")
print(f"=======================================================")
print(f"Same-ID: {np.mean(same_sims):.4f} (+-{np.std(same_sims):.4f}) min: {np.min(same_sims):.4f}")
print(f"Diff-ID: {np.mean(diff_sims):.4f} (+-{np.std(diff_sims):.4f}) max: {np.max(diff_sims):.4f}")
print(f"\\nSEPARATION GAP: {separation:.4f}")
print(f"False-merge @ 0.65: {fm_065:.1f}%")
print(f"False-merge @ 0.75: {fm_075:.2f}%")
print(f"Fragmentation @ 0.65: {fragmentation:.1f}%")

np.fill_diagonal(sim_matrix, -1)
gaps = []
for i in range(n):
    ranked = np.sort(sim_matrix[i])[::-1]
    if len(ranked) >= 2:
        gaps.append(ranked[0] - ranked[1])
gaps = np.array(gaps)
print(f"\\nAMBIGUITY ANALYSIS")
print(f"  Mean gap (best vs 2nd): {np.mean(gaps):.4f}")
print(f"  Ambiguous (gap < 0.10): {(gaps < 0.10).mean()*100:.1f}%")
print(f"  High-confidence (gap > 0.20): {(gaps > 0.20).mean()*100:.1f}%")

if separation > 0.50 and fm_075 < 3: print("\\nVERDICT: EXCELLENT")
elif separation > 0.40 and fm_075 < 5: print("\\nVERDICT: GOOD")
elif separation > 0.30: print("\\nVERDICT: OKAY")
else: print("\\nVERDICT: NEEDS WORK")

# --- CENTROID-BASED EVALUATION ---
print(f"\\n{'='*55}")
print(f"CENTROID-BASED EVALUATION (more robust than instance-level)")
print(f"{'='*55}")
centroids = {}
for lbl in sorted(set(all_labels)):
    mask = all_labels == lbl
    embs = torch.tensor(all_emb[mask]) # Convert back to tensor just for clean norm wrapper if desired, np is fine
    centroid_tensor = F.normalize(embs.mean(dim=0), p=2, dim=0)
    centroids[lbl] = centroid_tensor.numpy()

centroid_labels = sorted(centroids.keys())
centroid_matrix = np.array([centroids[l] for l in centroid_labels])
centroid_sim = centroid_matrix @ centroid_matrix.T
np.fill_diagonal(centroid_sim, -1)

# Per-query centroid matching
centroid_rank1 = 0
for i in range(n):
    query_label = all_labels[i]
    query_emb = all_emb[i]
    # Compare to centroids
    sims_to_centroids = query_emb @ centroid_matrix.T
    # Find best matching centroid
    best_idx = np.argmax(sims_to_centroids)
    if centroid_labels[best_idx] == query_label:
        centroid_rank1 += 1

print(f"  Centroid Rank-1: {centroid_rank1/n:.4f} ({centroid_rank1}/{n})")

# Centroid ambiguity
centroid_gaps = []
for i in range(n):
    query_emb = all_emb[i]
    sims_to_centroids = query_emb @ centroid_matrix.T
    ranked = np.sort(sims_to_centroids)[::-1]
    if len(ranked) >= 2:
        centroid_gaps.append(ranked[0] - ranked[1])
centroid_gaps = np.array(centroid_gaps)
print(f"  Centroid ambiguity (gap < 0.10): {(centroid_gaps < 0.10).mean()*100:.1f}%")
print(f"  Centroid high-confidence (gap > 0.20): {(centroid_gaps > 0.20).mean()*100:.1f}%")
print(f"  Centroid mean gap: {np.mean(centroid_gaps):.4f}")

# --- PASS 2: FULL DATASET AUDIT ---
full_eval_loader = DataLoader(full_eval_dataset, batch_size=32, shuffle=False)
all_emb, all_labels = embed_with_tta(model, full_eval_loader, device)
sim_matrix = all_emb @ all_emb.T
n = len(all_labels)

same_sims_full, diff_sims_full = [], []
worst_pairs = []
id_to_same_sims = defaultdict(list)
worst_cross_pairs = []

for i in range(n):
    for j in range(i + 1, n):
        sim = sim_matrix[i][j]
        if all_labels[i] == all_labels[j]: 
            same_sims_full.append(sim)
            worst_pairs.append((sim, i, j))
            id_to_same_sims[all_labels[i]].append(sim)
        else: 
            diff_sims_full.append(sim)
            worst_cross_pairs.append((sim, i, j))

print(f"\\n=======================================================")
print(f"FULL DATASET AUDIT ({len(full_eval_dataset.samples)} images)")
print(f"=======================================================")

print("\\nPER-IDENTITY INTRA-SIMILARITY")
print("=" * 55)
unique_labels = sorted(set(all_labels))
for label in unique_labels:
    mask = all_labels == label
    if mask.sum() < 2: continue
    indices_arr = np.where(mask)[0]
    intra = [sim_matrix[i][j] for i, j in combinations(indices_arr, 2)]
    name = full_eval_dataset.idx_to_class.get(label, f"ID_{label}").split("/")[-1]
    mean_sim = np.mean(intra)
    min_sim = np.min(intra)
    status = "OK" if min_sim > 0.50 else "WARN" if min_sim > 0.30 else "BAD"
    print(f"  [{status:4s}] {name:25s}: mean={mean_sim:.4f} min={min_sim:.4f} (n={mask.sum()})")

id_means = [(np.mean(sims), lbl) for lbl, sims in id_to_same_sims.items() if len(sims) > 0]
id_means.sort(key=lambda x: x[0])
print(f"\\nWORST 10 IDENTITIES (Same-ID Mean):")
for mean_sim, lbl in id_means[:10]:
    identity = full_eval_dataset.idx_to_class.get(lbl, f"ID_{lbl}").split("/")[-1]
    sims_arr = id_to_same_sims[lbl]
    worst = np.min(sims_arr)
    print(f"  [{len(sims_arr)} pairs] {identity:20s}: Mean={mean_sim:.4f} | Worst={worst:.4f}")

print(f"\\nTOP-25 WORST CROSS-IDENTITY FALSE MATCHES")
worst_cross_pairs.sort(reverse=True)
for sim_val, i, j in worst_cross_pairs[:25]:
    path_i = full_eval_dataset.samples[i][0]
    path_j = full_eval_dataset.samples[j][0]
    rel_i = str(Path(path_i).parent.name) + "/" + str(Path(path_i).name)
    rel_j = str(Path(path_j).parent.name) + "/" + str(Path(path_j).name)
    flag = "CRIT" if sim_val > 0.90 else "HIGH" if sim_val > 0.80 else "MED"
    print(f"  [{flag}] {rel_i} <-> {rel_j}  [{sim_val:.4f}]")""")

# ============================================================================
add_md("## Cell 12 — Separation Histogram")
add_code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.hist(same_sims, bins=50, alpha=0.6, color='green',
         label=f'Same ID (mean={np.mean(same_sims):.3f})', density=True)
ax1.hist(diff_sims, bins=50, alpha=0.6, color='red',
         label=f'Diff ID (mean={np.mean(diff_sims):.3f})', density=True)
ax1.axvline(0.75, color='black',  linestyle='--', linewidth=2, label='Threshold 0.75')
ax1.axvline(0.65, color='gray',   linestyle='--', linewidth=1, label='Old Threshold 0.65', alpha=0.6)
ax1.set_xlabel('Cosine Similarity', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title(f'v7 Zero-Shot Separation (Gap = {separation:.4f})', fontsize=14)
ax1.legend(fontsize=10)

overlap_diff = diff_sims[diff_sims > 0.4]
overlap_same = same_sims[same_sims > 0.4]
if len(overlap_diff) > 0:
    ax2.hist(overlap_diff, bins=30, alpha=0.6, color='red', label='Diff-ID (sim > 0.4)', density=True)
if len(overlap_same) > 0:
    ax2.hist(overlap_same, bins=30, alpha=0.6, color='green', label='Same-ID (sim > 0.4)', density=True)
ax2.axvline(0.75, color='black', linestyle='--', linewidth=2, label='Threshold 0.75')
ax2.set_xlabel('Cosine Similarity', fontsize=12)
ax2.set_title('Overlap Zone Detail (sim > 0.4)', fontsize=14)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.show()""")

# ============================================================================
add_md("## Cell 13 — Rank-1 / Rank-5 Accuracy (Zero-Shot with Flip TTA)")
add_code("""# Re-embed eval set with flip TTA
eval_loader2 = DataLoader(eval_dataset, batch_size=32, shuffle=False)
all_emb2, all_labels2 = embed_with_tta(model, eval_loader2, device)
sim_eval = all_emb2 @ all_emb2.T
np.fill_diagonal(sim_eval, -1)
n_eval = len(all_labels2)

rank1_correct = rank5_correct = 0
for i in range(n_eval):
    query_label = all_labels2[i]
    ranked = np.argsort(-sim_eval[i])
    if all_labels2[ranked[0]] == query_label:
        rank1_correct += 1
    if query_label in all_labels2[ranked[:5]]:
        rank5_correct += 1

print(f"Rank-1 Accuracy (TTA): {rank1_correct/n_eval:.4f} ({rank1_correct}/{n_eval})")
print(f"Rank-5 Accuracy (TTA): {rank5_correct/n_eval:.4f} ({rank5_correct}/{n_eval})")""")

# ============================================================================
add_md("## Cell 14 — Audit Worst Same-ID Pairs")
add_code("""worst_pairs.sort(key=lambda x: x[0])
worst_k = worst_pairs[:12]

fig, axes = plt.subplots(4, 6, figsize=(18, 12))
curr_idx = 0
for sim_val, i, j in worst_k:
    if curr_idx + 1 >= len(axes.flat): break
    img_i, lbl_i = full_eval_dataset[i]
    img_j, lbl_j = full_eval_dataset[j]
    
    def to_np(t):
        t = t.clone()
        for c, m, s in zip(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
            c.mul_(s).add_(m)
        return np.clip(t.permute(1, 2, 0).numpy(), 0, 1)

    ax1, ax2 = axes.flat[curr_idx], axes.flat[curr_idx+1]
    identity = full_eval_dataset.idx_to_class.get(all_labels[i], "?").split("/")[-1]
    
    ax1.imshow(to_np(img_i))
    ax1.set_title(f"{identity}\\nSim: {sim_val:.3f}", fontsize=8, color='coral')
    ax1.axis('off')
    
    ax2.imshow(to_np(img_j))
    ax2.set_title(f"{identity}\\nSim: {sim_val:.3f}", fontsize=8, color='coral')
    ax2.axis('off')
    curr_idx += 2

for ax in axes.flat[curr_idx:]: ax.axis('off')
plt.suptitle("Worst Same-ID Pairs (Hard-Positive Auditing)", fontsize=16)
plt.tight_layout()
plt.show()""")

# ============================================================================
add_md("## Cell 15 — Save Model")
add_code("""import shutil

save_dict = {
    'model_state_dict': model.state_dict(),
    'embed_dim': 256,
    'num_classes': train_dataset.num_classes,
    'separation_gap': float(separation),
    'false_merge_rate_at_065': float(fm_065),
    'false_merge_rate_at_075': float(fm_075),
    'fragmentation_rate_at_065': float(fragmentation),
    'training_config': {
        'version': 'v7.4_global_hard_negative_memory',
        'loss': 'HardPosAlign + CenterLoss + ArcFace + GlobalBatchHardTriplet',
        'ms_alpha': 2.0, 'ms_beta': 75.0, 'ms_hard_neg_k': 20, 'ms_temperature': 0.2,
        'hard_pos_weight': HARD_POS_WEIGHT,
        'arcface_margin': 0.25, 'arcface_scale': 32.0,
        'arcface_weight': ARCFACE_LOSS_WEIGHT,
        'center_loss_weight': CENTER_LOSS_WEIGHT,
        'triplet_weight': TRIPLET_LOSS_WEIGHT, 'triplet_margin': 0.3,
        'P': P, 'M': M, 'epochs': TOTAL_EPOCHS, 'warmup_epochs': WARMUP_EPOCHS,
        'augmentation': 'v7_region_disrupting (RandomResizedCrop+perspective+erasing)',
        'eval_split': 'disjoint_80_20_zero_shot',
        'data_version': 'training_heads_v6_cleaned',
    },
}

torch.save(save_dict, "/kaggle/working/elephant_head_reid_v7.0.pth")
shutil.copy2("/kaggle/working/elephant_head_reid_v7.0.pth", "/kaggle/working/elephant_head_reid_v7.0_download.pth")

print("Model saved: elephant_head_reid_v7.0.pth")
print(f"  Version:            v7.0_identity_cue_forcing")
print(f"  Separation gap:     {separation:.4f}")
print(f"  False merge @ 0.75: {fm_075:.2f}%")
print(f"  Download:           elephant_head_reid_v7.0_download.pth")""")

# ============================================================================
import os

out_path = os.path.join(
    os.path.dirname(__file__), "elephant-head-embedding-training-v7.0.ipynb"
)

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook generated: {out_path}")
print(f"Cells: {len(cells)}")
