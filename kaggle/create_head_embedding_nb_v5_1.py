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

add_md("""# 🐘 Elephant Head Re-ID v6.0 — Herd Collision Fix & Triplet Mining

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

        if split == 'train':
            target_identities = train_identities
        elif split == 'eval':
            target_identities = eval_identities
        else:
            target_identities = valid_identities
        
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
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

# IMPORTANT: Ensure kaggle dataset is your newly generated V6
DATASET_PATH = Path("/kaggle/input/elephant-training-heads-v6/training_heads_v6")

print(f"Dataset path: {DATASET_PATH}")
train_dataset = HeadCropDataset(str(DATASET_PATH), train_transform, min_images=4, split='train')
eval_dataset  = HeadCropDataset(str(DATASET_PATH), eval_transform,  min_images=4, split='eval')
full_eval_dataset = HeadCropDataset(str(DATASET_PATH), eval_transform, min_images=4, split='all')""")


add_md("## Cell 3.5 — Exact Duplicate Audit")
add_code("""import hashlib
from collections import defaultdict

print("Scanning for exact duplicate images across differing identities...")
file_hashes = defaultdict(list)
for path, label in full_eval_dataset.samples:
    with open(path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    file_hashes[file_hash].append((path, label))

duplicates_found = 0
for fhash, paths_and_labels in file_hashes.items():
    if len(paths_and_labels) > 1:
        unique_labels = set(lbl for _, lbl in paths_and_labels)
        if len(unique_labels) > 1:  # Only flag if duplicates span DIFFERENT identities
            print("\n🔴 CROSS-IDENTITY EXACT DUPLICATE DETECTED:")
            for p, lbl in paths_and_labels:
                name = full_eval_dataset.idx_to_class.get(lbl, '?').split('/')[-1]
                print(f"   - [{name}] {p}")
            duplicates_found += 1

if duplicates_found == 0:
    print("✅ No cross-identity exact duplicates found!")
else:
    print(f"\nTotal duplicated cross-identity groups: {duplicates_found}")""")

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
    def __init__(self, alpha=3.0, beta=75.0, base=0.5, hard_neg_k=20, temperature=0.2):
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

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        sim = embeddings @ embeddings.t()
        loss = torch.zeros(1, device=embeddings.device, requires_grad=True).squeeze()
        valid = 0
        for i in range(len(embeddings)):
            mask = labels == labels[i]
            pos_sim = sim[i][mask].clone()
            neg_sim = sim[i][~mask].clone()
            
            pos_sim = pos_sim[pos_sim < 0.999]
            if len(pos_sim) == 0 or len(neg_sim) == 0: continue
            
            hardest_pos = pos_sim.min()
            hardest_neg = neg_sim.max()
            
            l = torch.relu(hardest_neg - hardest_pos + self.margin)
            loss = loss + l
            valid += 1
            
        return loss / max(valid, 1)

CENTER_LOSS_WEIGHT = 0.1
ARCFACE_LOSS_WEIGHT = 0.25
TRIPLET_LOSS_WEIGHT = 0.5
criterion_ms = MultiSimilarityLossV4()
criterion_center = IdentityCenterLoss()
criterion_triplet = BatchHardTripletLoss(margin=0.3)""")

add_md("## Cell 7 — Sampler")
add_code("""KNOWN_WORST_PAIRS_NAMES = []

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
        
        # temporarily load images with sampler_transform
        old_transform = self.dataset.transform
        self.dataset.transform = sampler_transform
        
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
            
        # restore
        self.dataset.transform = old_transform
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

sampler = DiversityPxMSampler(train_dataset, model, device, P=6, M=4, worst_pairs=resolved_worst_pairs)""")

add_md("## Cell 8 — Training Setup")
add_code("""TOTAL_EPOCHS = 35
WARMUP_EPOCHS = 5
P, M = 6, 4
BATCH_SIZE = P * M
steps_per_epoch = max(1, len(train_dataset) // BATCH_SIZE)

for param in model.backbone.parameters(): param.requires_grad = False
optimizer = torch.optim.AdamW([
    {'params': model.embed.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
    {'params': arcface_head.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
])""")

add_md("## Cell 9 — Training Loop")
add_code("""history = {'epoch': [], 'loss': [], 'ms_loss': [], 'center_loss': [], 'arcface_loss': [], 'triplet_loss': []}

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
    total_loss = total_ms = total_center = total_arc = total_trip = 0

    for step in range(steps_per_epoch):
        imgs, labels = sampler.sample_batch(epoch)
        imgs, labels = imgs.to(device), labels.to(device)
        
        embeddings = model(imgs)
        ms_loss = criterion_ms(embeddings, labels)
        center_loss = criterion_center(embeddings, labels)
        arc_loss = arcface_head(embeddings, labels)
        triplet_loss = criterion_triplet(embeddings, labels)
        
        loss = ms_loss + CENTER_LOSS_WEIGHT * center_loss + ARCFACE_LOSS_WEIGHT * arc_loss + TRIPLET_LOSS_WEIGHT * triplet_loss
            
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(arcface_head.parameters()), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_ms += ms_loss.item()
        total_center += center_loss.item()
        total_arc += arc_loss.item()
        total_trip += triplet_loss.item()

    if epoch > WARMUP_EPOCHS: scheduler.step()
    
    avg = lambda x: x / steps_per_epoch
    history['loss'].append(avg(total_loss))
    history['epoch'].append(epoch)
    print(f"Epoch {epoch:2d}/{TOTAL_EPOCHS} Loss={avg(total_loss):.4f} MS={avg(total_ms):.4f} ArcFace={avg(total_arc):.4f} Trip={avg(total_trip):.4f}")""")

add_md("## Cell 10 — ZERO-SHOT EVALUATION & FULL DATASET AUDIT")
add_code("""# --- PASS 1: ZERO-SHOT EVALUATION (Held-out Identities Only) ---
model.eval()
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
        sim = sim_matrix[i][j]
        if all_labels[i] == all_labels[j]: same_sims.append(sim)
        else: diff_sims.append(sim)
same_sims, diff_sims = np.array(same_sims), np.array(diff_sims)
separation = np.mean(same_sims) - np.mean(diff_sims)
fm_075 = (diff_sims > 0.75).mean() * 100

print(f"=======================================================")
print(f"ZERO-SHOT EVALUATION ({len(eval_dataset.samples)} images)")
print(f"=======================================================")
print(f"Same-ID: {np.mean(same_sims):.4f} (±{np.std(same_sims):.4f}) min: {np.min(same_sims):.4f}")
print(f"Diff-ID: {np.mean(diff_sims):.4f} (±{np.std(diff_sims):.4f}) max: {np.max(diff_sims):.4f}")
print(f"\\nSEPARATION GAP: {separation:.4f}")
print(f"🚨 False-merge @ 0.75: {fm_075:.2f}%")

# --- PASS 2: FULL DATASET AUDIT (All Identities) ---
from pathlib import Path
full_eval_loader = DataLoader(full_eval_dataset, batch_size=32, shuffle=False)
all_emb, all_labels = [], []
with torch.no_grad():
    for imgs, labels in full_eval_loader:
        all_emb.append(model(imgs.to(device)).cpu())
        all_labels.append(labels)
all_emb = torch.cat(all_emb).numpy()
all_labels = torch.cat(all_labels).numpy()
sim_matrix = all_emb @ all_emb.T
n = len(all_labels)

same_sims, diff_sims = [], []
worst_pairs = []
id_to_same_sims = defaultdict(list)
worst_cross_pairs = []

for i in range(n):
    for j in range(i + 1, n):
        sim = sim_matrix[i][j]
        if all_labels[i] == all_labels[j]: 
            same_sims.append(sim)
            worst_pairs.append((sim, i, j))
            id_to_same_sims[all_labels[i]].append(sim)
        else: 
            diff_sims.append(sim)
            worst_cross_pairs.append((sim, i, j))

print(f"\\n=======================================================")
print(f"FULL DATASET AUDIT ({len(full_eval_dataset.samples)} images)")
print(f"=======================================================")

id_means = [(np.mean(sims), lbl) for lbl, sims in id_to_same_sims.items() if len(sims) > 0]
id_means.sort(key=lambda x: x[0])

# NEW: Identity Cleanliness Report
print("\\n🧹 IDENTITY CLEANLINESS REPORT (Bottom 10 Identities):")
for mean_sim, lbl in id_means[:10]:
    identity = full_eval_dataset.idx_to_class.get(lbl, f"ID_{lbl}")
    name = identity.split("/")[-1]
    sims = id_to_same_sims[lbl]
    worst = np.min(sims)
    print(f"  [{len(sims)} pairs] {name:20s}: Mean={mean_sim:.4f} | Worst={worst:.4f}")

print(f"\\nTOP-25 WORST CROSS-IDENTITY FALSE MATCHES")
worst_cross_pairs.sort(reverse=True)
for sim_val, i, j in worst_cross_pairs[:25]:
    lbl_i, lbl_j = all_labels[i], all_labels[j]
    path_i = full_eval_dataset.samples[i][0]
    path_j = full_eval_dataset.samples[j][0]
    # Extract last two components for readable path
    rel_i = str(Path(path_i).parent.name) + "/" + str(Path(path_i).name)
    rel_j = str(Path(path_j).parent.name) + "/" + str(Path(path_j).name)
    flag = "🔴" if sim_val > 0.90 else "🟠" if sim_val > 0.80 else "🟡"
    print(f"  {flag} {rel_i} ↔ {rel_j}  [{sim_val:.4f}]")""")

add_md("## Cell 11 — Audit Worst Same-ID Pairs")
add_code("""worst_pairs.sort(key=lambda x: x[0])
worst_k = worst_pairs[:12] # Plot 12 worst same-ID pairs

fig, axes = plt.subplots(4, 6, figsize=(18, 12))
curr_idx = 0
for sim, i, j in worst_k:
    img_i, lbl_i = full_eval_dataset[i]
    img_j, lbl_j = full_eval_dataset[j]
    
    def to_np(t):
        t = t.clone()
        for c, m, s in zip(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
            c.mul_(s).add_(m)
        return np.clip(t.permute(1, 2, 0).numpy(), 0, 1)

    ax1, ax2 = axes.flat[curr_idx], axes.flat[curr_idx+1]
    
    identity = full_eval_dataset.idx_to_class.get(all_labels[i], f"ID_{all_labels[i]}")
    name = identity.split("/")[-1]
    
    ax1.imshow(to_np(img_i))
    ax1.set_title(f"{name}\\nSim: {sim:.3f}", fontsize=8, color='coral')
    ax1.axis('off')
    
    ax2.imshow(to_np(img_j))
    ax2.set_title(f"{name}\\nSim: {sim:.3f}", fontsize=8, color='coral')
    ax2.axis('off')
    curr_idx += 2

for ax in axes.flat[curr_idx:]: ax.axis('off')

plt.suptitle("Worst Same-ID Pairs (Hard-Positive Auditing)", fontsize=16)
plt.tight_layout()
plt.show()""")

add_md("## Cell 12 — Fair Model Comparison (v4 vs v6 on Same Zero-Shot Split)")
add_code("""# Load the v4 model weights and evaluate on the SAME eval_dataset split.
# This is the only fair apples-to-apples comparison:
# v4 was likely evaluated on identities it had seen during training (leakage).
# Here we force it to face the same held-out identities as v6.

V4_PATH = "/kaggle/input/elephant-reid-v4/elephant_head_reid_v4.pth"

def eval_model_on_split(model_instance, dataset, device):
    model_instance.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_emb, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            all_emb.append(model_instance(imgs.to(device)).cpu())
            all_labels.append(labels)
    all_emb = torch.cat(all_emb).numpy()
    all_labels = torch.cat(all_labels).numpy()
    sim_mat = all_emb @ all_emb.T
    n = len(all_labels)
    same, diff = [], []
    for i in range(n):
        for j in range(i + 1, n):
            s = sim_mat[i][j]
            if all_labels[i] == all_labels[j]: same.append(s)
            else: diff.append(s)
    same, diff = np.array(same), np.array(diff)
    gap = np.mean(same) - np.mean(diff)
    fm075 = (diff > 0.75).mean() * 100
    return {
        "same_mean": float(np.mean(same)),
        "same_min": float(np.min(same)),
        "diff_mean": float(np.mean(diff)),
        "diff_max": float(np.max(diff)),
        "gap": float(gap),
        "fm075": float(fm075),
    }

print("Evaluating v6.0 model on held-out zero-shot split...")
v6_metrics = eval_model_on_split(model, eval_dataset, device)

if os.path.exists(V4_PATH):
    print("Loading v4 weights for comparison...")
    v4_model = HeadEmbeddingModel(embed_dim=256).to(device)
    ckpt = torch.load(V4_PATH, map_location=device, weights_only=False)
    v4_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    v4_metrics = eval_model_on_split(v4_model, eval_dataset, device)
    has_v4 = True
else:
    print(f"⚠️  v4 checkpoint not found at {V4_PATH} — skipping v4 comparison.")
    print("   Upload elephant_head_reid_v4.pth as a Kaggle dataset input to enable comparison.")
    has_v4 = False

print()
print("=" * 62)
print(f"{'Metric':<28} {'v4 (same split)':>15} {'v6.0':>15}")
print("=" * 62)
rows = [
    ("Same-ID mean",  "same_mean"),
    ("Same-ID min",   "same_min"),
    ("Diff-ID mean",  "diff_mean"),
    ("Diff-ID max",   "diff_max"),
    ("Separation gap","gap"),
    ("False-merge @0.75 (%)", "fm075"),
]
for label, key in rows:
    v4_val = f"{v4_metrics[key]:.4f}" if has_v4 else "  N/A"
    v6_val = f"{v6_metrics[key]:.4f}"
    print(f"  {label:<26} {v4_val:>15} {v6_val:>15}")
print("=" * 62)
if has_v4:
    delta = v6_metrics["gap"] - v4_metrics["gap"]
    verdict = "✅ v6.0 WINS" if delta > 0.01 else ("➡️  DRAW" if abs(delta) <= 0.01 else "⚠️  v4 wins — investigate v6 data quality")
    print(f"  Gap delta (v6 - v4): {delta:+.4f}  →  {verdict}")
""")

add_md("## Cell 13 — Save")
add_code("""torch.save({
    'model_state_dict': model.state_dict(),
    'version': 'v6.0_zero_shot',
    'separation_gap': float(separation),
    'false_merge_rate_at_075': float(fm_075),
}, "/kaggle/working/elephant_head_reid_v6.0.pth")
import shutil
shutil.copy2("/kaggle/working/elephant_head_reid_v6.0.pth", "/kaggle/working/elephant_head_reid_v6.0_download.pth")
print("Saved v6.0!")""")

with open('kaggle/elephant-head-embedding-training-v6.0.ipynb', 'w') as f:
    json.dump({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, f, indent=2)
print("Generated kaggle/elephant-head-embedding-training-v6.0.ipynb")
