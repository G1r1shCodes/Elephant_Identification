"""
Generate Kaggle notebook: Elephant Head Re-ID v5 — Held-Out Validation & Strict Filtering

Key upgrades over v4:
  1. Strict filtering (min_images_per_id = 4)
  2. Disjoint Identity Split (80% train, 20% val) — True zero-shot open-set evaluation!
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

add_md("""# 🐘 Elephant Head Re-ID v5 — True Held-Out Evaluation

**Why v5?**
The v4 model successfully stabilized the embedding space and hit a 0.75 separation gap. However, its evaluation was slightly optimistic because the training and evaluation sets utilized the same identities (and images!).

**What's changed (targeted fixes):**
1. **Strict Identity Filtering** — any identity with `< 4` images is completely discarded from training to ensure stable anchors.
2. **Disjoint Identity Split (80/20)** — the evaluation dataset is now composed of **entirely held-out identities**. This provides a true "Zero-Shot / Open-Set" metric!
3. **Data Dependency** — Relies on `training_heads_v5_clean` (where the exact cross-identity collisions and poison identities have been strictly pruned).""")

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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)""")

add_md("## Cell 2 — Dataset (Train / Val Split)")
add_code("""class HeadCropDataset(Dataset):
    \"\"\"
    Loads pre-processed elephant head crops.
    For v5, we do a proper identity-level split to ensure zero-shot evaluation!
    \"\"\"
    def __init__(self, root_dir, transform=None, min_images=4, split='train', split_ratio=0.8):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.label_to_indices = defaultdict(list)

        SKIP_DIRS = {'_quarantined', '_filtered', '_no_head_in_crop'}

        # Gather all valid folders
        all_folders = {}
        for dirpath, dirnames, filenames in os.walk(root_dir):
            parts = set(Path(dirpath).parts)
            if parts & SKIP_DIRS:
                continue
            images = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) >= min_images:
                rel = os.path.relpath(dirpath, root_dir)
                all_folders[rel] = sorted([os.path.join(dirpath, f) for f in images])

        # Force the known hard-negative pairs to be in the training set so replay works
        KNOWN_HARD_PAIRS = {
            "JM_ELE_84", "JU_ELE_96", "Calf_68", "AF_ELE_22_R", 
            "Calf_56", "JU_ELE_94", "Calf_57", "JF_ELE_74", 
            "AF_ELE_15", "AF_ELE_16_R", "JM_ELE_86", "SAF_ELE_113"
        }
        
        valid_identities = sorted(all_folders.keys())
        
        # Split logic
        random.seed(SEED)
        random.shuffle(valid_identities)
        
        train_identities = set()
        for ident in valid_identities:
            if any(hard in ident for hard in KNOWN_HARD_PAIRS):
                train_identities.add(ident)
                
        remaining = [i for i in valid_identities if i not in train_identities]
        needed_train = int(len(valid_identities) * split_ratio) - len(train_identities)
        if needed_train > 0:
            train_identities.update(remaining[:needed_train])
            eval_identities = set(remaining[needed_train:])
        else:
            eval_identities = set(remaining)

        target_identities = train_identities if split == 'train' else eval_identities
        
        idx = 0
        for name in sorted(target_identities):
            paths = all_folders[name]
            self.class_to_idx[name] = idx
            self.idx_to_class[idx] = name
            for p in paths:
                self.label_to_indices[idx].append(len(self.samples))
                self.samples.append((p, idx))
            idx += 1

        self.num_classes = idx
        print(f"[{split.upper()}] Loaded {len(self.samples)} images across {idx} identities")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label""")

add_md("## Cell 3 — Transforms & Load Dataset")
add_code("""train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
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

# IMPORTANT: Ensure kaggle dataset is your pruned V5
DATASET_PATH = Path("/kaggle/input/elephant-training-heads-v5/training_heads_v5_clean")

print(f"Dataset path: {DATASET_PATH}")
train_dataset = HeadCropDataset(str(DATASET_PATH), train_transform, min_images=4, split='train')
eval_dataset  = HeadCropDataset(str(DATASET_PATH), eval_transform,  min_images=4, split='eval')""")

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
plt.suptitle("Head Crop Samples (Cleaned Held-Out Eval Dataset)", fontsize=14)
plt.tight_layout()
plt.show()""")

add_md("## Cell 5 — Embedding Model")
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
        cosine = F.linear(embeddings, F.normalize(self.weight))
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return F.cross_entropy(output * self.scale, labels)

model = HeadEmbeddingModel(embed_dim=256).to(device)
arcface_head = ArcFaceHead(embed_dim=256, num_classes=train_dataset.num_classes).to(device)""")

add_md("## Cell 6 — Loss Functions")
add_code("""class MultiSimilarityLossV4(nn.Module):
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

CENTER_LOSS_WEIGHT = 0.1
ARCFACE_LOSS_WEIGHT = 0.25
criterion_ms = MultiSimilarityLossV4()
criterion_center = IdentityCenterLoss()""")

add_md("## Cell 7 — Sampler")
add_code("""KNOWN_WORST_PAIRS_NAMES = [
    ("JM_ELE_84", "JU_ELE_96"), ("Calf_68", "JU_ELE_96"),
    ("AF_ELE_22_R", "Calf_68"), ("Calf_56", "JU_ELE_94"),
    ("Calf_57", "JF_ELE_74"), ("AF_ELE_15", "AF_ELE_16_R"),
    ("JM_ELE_86", "SAF_ELE_113")
]

class DiversityPxMSampler:
    def __init__(self, dataset, model, device, P=8, M=3, refresh_interval=3, worst_pairs=None, replay_interval=5):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.P = P
        self.M = M
        self.refresh_interval = refresh_interval
        self.replay_interval = replay_interval
        self.worst_pairs = worst_pairs or []
        self.label_to_indices = dataset.label_to_indices
        self.available_labels = [lbl for lbl, idxs in self.label_to_indices.items() if len(idxs) >= M]
        self._identity_distances = {}

    def refresh_embeddings(self, epoch):
        if epoch % self.refresh_interval != 1 and epoch != 1: return
        self.model.eval()
        all_emb = []
        with torch.no_grad():
            for start in range(0, len(self.dataset), 32):
                batch = [self.dataset[i] for i in range(start, min(start+32, len(self.dataset)))]
                imgs = torch.stack([x[0] for x in batch]).to(self.device)
                all_emb.append(self.model(imgs).cpu())
        self._embeddings = torch.cat(all_emb)
        for label in self.available_labels:
            indices = self.label_to_indices[label]
            embs = self._embeddings[indices]
            self._identity_distances[label] = (1.0 - embs @ embs.t()).numpy()
        self.model.train()

    def sample_batch(self, epoch=None):
        forced = set()
        if epoch and epoch % self.replay_interval == 0 and self.worst_pairs:
            forced.update(random.choice(self.worst_pairs))
            
        avail = [l for l in self.available_labels if l not in forced]
        selected = list(forced) + random.sample(avail, max(0, min(self.P - len(forced), len(avail))))
        
        batch_indices = []
        for l in selected:
            indices = self.label_to_indices[l]
            if l not in self._identity_distances or len(indices) <= self.M:
                batch_indices.extend(random.sample(indices, self.M) if len(indices) >= self.M else indices)
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

# Resolve shortnames to label IDs for Train split
resolved_worst_pairs = []
short_to_label = {name.split("/")[-1]: idx for name, idx in train_dataset.class_to_idx.items()}
for a, b in KNOWN_WORST_PAIRS_NAMES:
    if a in short_to_label and b in short_to_label:
        resolved_worst_pairs.append((short_to_label[a], short_to_label[b]))

sampler = DiversityPxMSampler(train_dataset, model, device, P=8, M=3, worst_pairs=resolved_worst_pairs)""")

add_md("## Cell 8 — Training Setup")
add_code("""TOTAL_EPOCHS = 35
WARMUP_EPOCHS = 5
P, M = 8, 3
BATCH_SIZE = P * M
steps_per_epoch = max(1, len(train_dataset) // BATCH_SIZE)

for param in model.backbone.parameters(): param.requires_grad = False
optimizer = torch.optim.AdamW([
    {'params': model.embed.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
    {'params': arcface_head.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
])""")

add_md("## Cell 9 — Training Loop")
add_code("""history = {'epoch': [], 'loss': [], 'ms_loss': [], 'center_loss': [], 'arcface_loss': [], 'phase': []}

for epoch in range(1, TOTAL_EPOCHS + 1):
    if epoch == WARMUP_EPOCHS + 1:
        for param in model.backbone.parameters(): param.requires_grad = True
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': 5e-5, 'weight_decay': 1e-4},
            {'params': model.embed.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4},
            {'params': arcface_head.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS)

    sampler.refresh_embeddings(epoch)
    model.train()
    arcface_head.train()
    total_loss = total_ms = total_center = total_arc = 0

    for step in range(steps_per_epoch):
        imgs, labels = sampler.sample_batch(epoch)
        imgs, labels = imgs.to(device), labels.to(device)
        
        embeddings = model(imgs)
        ms_loss = criterion_ms(embeddings, labels)
        center_loss = criterion_center(embeddings, labels)
        arc_loss = arcface_head(embeddings, labels)
        
        loss = ms_loss + CENTER_LOSS_WEIGHT * center_loss + ARCFACE_LOSS_WEIGHT * arc_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(arcface_head.parameters()), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_ms += ms_loss.item()
        total_center += center_loss.item()
        total_arc += arc_loss.item()

    if epoch > WARMUP_EPOCHS: scheduler.step()
    
    avg = lambda x: x / steps_per_epoch
    history['loss'].append(avg(total_loss))
    history['epoch'].append(epoch)
    print(f"Epoch {epoch:2d}/{TOTAL_EPOCHS} Loss={avg(total_loss):.4f} MS={avg(total_ms):.4f} ArcFace={avg(total_arc):.4f}")""")

add_md("## Cell 10 — ZERO-SHOT EVALUATION")
add_code("""model.eval()
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

all_emb, all_labels = [], []
with torch.no_grad():
    for imgs, labels in eval_loader:
        all_emb.append(model(imgs.to(device)).cpu())
        all_labels.append(labels)

all_emb = torch.cat(all_emb).numpy()
all_labels = torch.cat(all_labels).numpy()

sim_matrix = all_emb @ all_emb.T
n = len(all_labels)

same_sims, diff_sims = [], []
for i in range(n):
    for j in range(i + 1, n):
        if all_labels[i] == all_labels[j]: same_sims.append(sim_matrix[i][j])
        else: diff_sims.append(sim_matrix[i][j])

same_sims, diff_sims = np.array(same_sims), np.array(diff_sims)
separation = np.mean(same_sims) - np.mean(diff_sims)

print(f"ZERO-SHOT EVALUATION (Held-out Identities)")
print(f"Same-ID: {np.mean(same_sims):.4f} (±{np.std(same_sims):.4f}) min: {np.min(same_sims):.4f}")
print(f"Diff-ID: {np.mean(diff_sims):.4f} (±{np.std(diff_sims):.4f}) max: {np.max(diff_sims):.4f}")
print(f"\\nSEPARATION GAP: {separation:.4f}")

fm_075 = (diff_sims > 0.75).mean() * 100
print(f"🚨 False-merge @ 0.75: {fm_075:.2f}%")
print(f"   Worst diff-ID sim:  {np.max(diff_sims):.4f}")

np.fill_diagonal(sim_matrix, -1)
gaps = [np.sort(sim_matrix[i])[::-1][0] - np.sort(sim_matrix[i])[::-1][1] for i in range(n) if len(sim_matrix[i]) >= 2]
gaps = np.array(gaps)
print(f"\\n📊 AMBIGUITY ANALYSIS")
print(f"  Ambiguous queries (gap < 0.10): {(gaps < 0.10).mean()*100:.1f}%")""")

add_md("## Cell 11 — Save")
add_code("""torch.save({
    'model_state_dict': model.state_dict(),
    'version': 'v5_zero_shot',
    'separation_gap': float(separation),
    'false_merge_rate_at_075': float(fm_075),
}, "/kaggle/working/elephant_head_reid_v5.pth")
import shutil
shutil.copy2("/kaggle/working/elephant_head_reid_v5.pth", "/kaggle/working/elephant_head_reid_v5_download.pth")
print("Saved v5!")""")

with open('kaggle/elephant-head-embedding-training-v5.ipynb', 'w') as f:
    json.dump({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, f, indent=2)
print("Generated kaggle/elephant-head-embedding-training-v5.ipynb")
