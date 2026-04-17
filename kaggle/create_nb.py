import json
import os

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

add_md('## 🔹 **Cell 1 — Imports & Setup**')
code1 = """import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import convnext_tiny
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)"""
add_code(code1)

add_md('## 🔹 **Cell 2 — Dataset (Pre-Processed Crops)**')
code2 = """class ElephantDataset(Dataset):
    \"\"\"
    Dataset for pre-processed elephant crops from the identity-aware pipeline.
    
    Expects directory structure:
        root/
            Identity_1/
                image1.jpg
                image2.jpg
            Identity_2/
                ...
    
    NO additional filtering needed — preprocessing already handles:
    - YOLO detection + padding
    - Blur rejection
    - Identity score gating
    \"\"\"
    def __init__(self, root_dir, transform=None, min_images_per_id=2):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Walk the directory tree recursively to find identity folders
        # Identity = deepest folder containing images
        identity_folders = {}
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Skip _rejected and _weak folders
            if '_rejected' in dirpath or '_weak' in dirpath:
                continue
            
            images = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                # Use relative path as identity key
                rel = os.path.relpath(dirpath, root_dir)
                identity_folders[rel] = [(os.path.join(dirpath, f)) for f in images]
        
        # Assign class indices, filtering identities with too few samples
        idx = 0
        skipped_ids = 0
        for identity_name, img_paths in sorted(identity_folders.items()):
            if len(img_paths) < min_images_per_id:
                skipped_ids += 1
                continue
            
            self.class_to_idx[identity_name] = idx
            self.idx_to_class[idx] = identity_name
            
            for path in img_paths:
                self.samples.append((path, idx))
            idx += 1
        
        print(f"Loaded {len(self.samples)} images across {idx} identities")
        if skipped_ids:
            print(f"  Skipped {skipped_ids} identities with < {min_images_per_id} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        
        # No center crop hack needed — crops are already identity-focused
        if self.transform:
            image = self.transform(image)
        
        return image, label"""
add_code(code2)

add_md('## 🔹 **Cell 3 — Transforms & Loader**')
code3 = """# Proper transforms for rectangular wildlife crops
# Resize to 256 on short side, then center-crop to 224x224
# This handles any aspect ratio without distortion

train_transform = transforms.Compose([
    transforms.Resize(256),  # resize shortest side to 256
    transforms.CenterCrop(224),  # consistent square input
    transforms.RandomHorizontalFlip(p=0.5),  # elephant ID is NOT flip-invariant for ears
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

from pathlib import Path

# Update this path to your Kaggle dataset
DATASET_PATH = Path('/kaggle/input/elephant-reid-processed')

dataset = ElephantDataset(str(DATASET_PATH), train_transform, min_images_per_id=2)
eval_dataset = ElephantDataset(str(DATASET_PATH), eval_transform, min_images_per_id=2)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Total samples: {len(dataset)}")
print(f"Number of identities: {len(dataset.class_to_idx)}")"""
add_code(code3)

add_md('## 🔹 **Cell 3.5 — Visual Inspection**')
code3_5 = """import matplotlib.pyplot as plt
import random

def show_samples(dataset, n=12):
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        # Denormalize for display
        img = img.clone()
        for t, m, s in zip(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        img = img.permute(1,2,0).numpy()
        img = np.clip(img, 0, 1)
        
        identity = dataset.idx_to_class.get(label, f"ID_{label}")
        
        plt.subplot(3, 4, i+1)
        plt.imshow(img)
        plt.title(f"{identity}", fontsize=8)
        plt.axis('off')
    
    plt.suptitle("Sample Crops from Preprocessed Dataset", fontsize=14)
    plt.tight_layout()
    plt.show()

show_samples(dataset)"""
add_code(code3_5)

add_md('## 🔹 **Cell 4 — ConvNeXt-Tiny Embedding Model**')
code4 = """class ElephantReIDModel(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = convnext_tiny(weights="DEFAULT")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        feat = self.backbone.features(x)
        feat = self.pool(feat).flatten(1)
        emb = self.embed(feat)
        return F.normalize(emb, dim=1)

model = ElephantReIDModel(embed_dim=256).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")"""
add_code(code4)

add_md('## 🔹 **Cell 5 — Multi-Similarity Loss & Optimizer**')
code5 = """class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=50.0, base=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, embeddings, labels):
        sim = torch.matmul(embeddings, embeddings.t())

        labels = labels.unsqueeze(1)
        mask = labels == labels.t()

        loss = 0
        for i in range(len(embeddings)):
            pos = sim[i][mask[i]].clone()
            neg = sim[i][~mask[i]].clone()

            pos = pos[pos < 1]  # remove self
            
            # Semi-hard negative mining
            neg = neg[neg > 0.2]   # lower bound
            neg = neg[neg < 0.8]   # upper bound

            if len(pos) == 0 or len(neg) == 0:
                continue

            pos_loss = (1/self.alpha) * torch.log(
                1 + torch.sum(torch.exp(-self.alpha * (pos - self.base)))
            )

            neg_loss = (1/self.beta) * torch.log(
                1 + torch.sum(torch.exp(self.beta * (neg - self.base)))
            )

            loss += pos_loss + neg_loss

        return loss / len(embeddings)

criterion = MultiSimilarityLoss()

# Differential LR: lower for pretrained backbone, higher for embedding head
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 5e-5},
    {'params': model.embed.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

# Cosine annealing scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)"""
add_code(code5)

add_md('## 🔹 **Cell 6 — Training Loop (P×M Sampling)**')
code6 = """from collections import defaultdict
import random

label_to_indices = defaultdict(list)

for idx, (_, label) in enumerate(dataset.samples):
    label_to_indices[label].append(idx)

EPOCHS = 10
P = 8   # identities per batch
M = 4   # images per identity

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    n_batches = 0

    steps = max(1, len(dataset) // (P * M))
    
    available_labels = [l for l, indices in label_to_indices.items() if len(indices) >= 2]
    actual_P = min(P, len(available_labels))

    for _ in range(steps):
        selected_labels = random.sample(available_labels, actual_P)

        batch_indices = []
        for label in selected_labels:
            indices = label_to_indices[label]
            if len(indices) >= M:
                batch_indices.extend(random.sample(indices, M))
            else:
                batch_indices.extend(random.choices(indices, k=M))

        batch = [dataset[i] for i in batch_indices]

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

    scheduler.step()
    avg_loss = total_loss / max(1, n_batches)
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, LR: {lr:.6f}")"""
add_code(code6)

add_md('## 🔹 **Cell 7 — Compute Embeddings**')
code7 = """model.eval()

# Use eval transform (no augmentation)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

all_embeddings = []
all_labels = []

with torch.no_grad():
    for imgs, labels in eval_loader:
        imgs = imgs.to(device)
        emb = model(imgs)

        all_embeddings.append(emb.cpu())
        all_labels.append(labels)

all_embeddings = torch.cat(all_embeddings)
all_labels = torch.cat(all_labels)

print("Embeddings shape:", all_embeddings.shape)"""
add_code(code7)

add_md('## 🔹 **Cell 8 — Pairwise Similarity Analysis**')
code8 = """from sklearn.metrics.pairwise import cosine_similarity

emb_np = all_embeddings.numpy()
labels_np = all_labels.numpy()

sim_matrix = cosine_similarity(emb_np)

same_sim = []
diff_sim = []

n = len(labels_np)

for i in range(n):
    for j in range(i+1, n):
        if labels_np[i] == labels_np[j]:
            same_sim.append(sim_matrix[i][j])
        else:
            diff_sim.append(sim_matrix[i][j])

print("=== Same Elephant Similarity ===")
print(f"  Mean: {np.mean(same_sim):.4f}")
print(f"  Min:  {np.min(same_sim):.4f}")
print(f"  Max:  {np.max(same_sim):.4f}")
print(f"  Std:  {np.std(same_sim):.4f}")

print("\\n=== Different Elephant Similarity ===")
print(f"  Mean: {np.mean(diff_sim):.4f}")
print(f"  Min:  {np.min(diff_sim):.4f}")
print(f"  Max:  {np.max(diff_sim):.4f}")
print(f"  Std:  {np.std(diff_sim):.4f}")

# Separation gap
gap = np.mean(same_sim) - np.mean(diff_sim)
print(f"\\nMean separation gap: {gap:.4f}")
if gap > 0.3:
    print("✅ Good separation!")
elif gap > 0.15:
    print("⚠️ Moderate separation — may need more training or data")
else:
    print("❌ Poor separation — check data quality")

# Histogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.hist(same_sim, bins=50, alpha=0.6, label='Same ID', color='green')
plt.hist(diff_sim, bins=50, alpha=0.6, label='Different ID', color='red')
plt.xlabel('Cosine Similarity')
plt.ylabel('Count')
plt.title('Pairwise Similarity Distribution')
plt.legend()
plt.tight_layout()
plt.show()"""
add_code(code8)

add_md('## 🔹 **Cell 9 — Conservative Clustering**')
code9 = """THRESH_HIGH = 0.75
THRESH_LOW = 0.5

n = len(sim_matrix)
unassigned = set(range(n))

clusters = []
uncertain = []

while unassigned:
    i = unassigned.pop()
    group = [i]

    candidates = []
    for j in unassigned:
        if sim_matrix[i][j] > THRESH_HIGH:
            candidates.append(j)

    for j in candidates:
        valid = True
        for g in group:
            if sim_matrix[j][g] < THRESH_LOW:
                valid = False
                break
        
        if valid:
            group.append(j)

    for g in group:
        if g in unassigned:
            unassigned.remove(g)

    if len(group) == 1:
        uncertain.append(group[0])
    else:
        clusters.append(group)

print(f"Clusters formed: {len(clusters)}")
print(f"Uncertain samples: {len(uncertain)}")

for idx, c in enumerate(clusters[:5]):
    labels_in_cluster = [labels_np[i] for i in c]
    unique = set(labels_in_cluster)
    print(f"Cluster {idx+1}: size={len(c)}, unique_ids={len(unique)}, ids={unique}")"""
add_code(code9)

add_md('## 🔹 **Cell 10 — Confidence-Classified Suggestions**')
code10 = """TOP_K = 5
HIGH_CONF = 0.70
MED_CONF  = 0.65
LOW_CONF  = 0.60
GAP_MIN   = 0.08

suggestions = {}

for i in range(n):
    sims = sim_matrix[i]
    
    top_indices = np.argsort(-sims)
    top_indices = [idx for idx in top_indices if idx != i][:TOP_K]

    classified = []
    for idx in top_indices:
        score = sims[idx]
        if score >= HIGH_CONF:
            level = "HIGH"
        elif score >= MED_CONF:
            level = "MEDIUM"
        elif score >= LOW_CONF:
            level = "LOW"
        else:
            continue
        classified.append((idx, score, level))

    suggestions[i] = classified

# Gap-validated summary
accepted = 0
ambiguous = 0
no_match = 0

for i in range(n):
    s = suggestions[i]
    if len(s) == 0:
        no_match += 1
    elif len(s) >= 2:
        top1_score = s[0][1]
        top2_score = s[1][1]
        gap = top1_score - top2_score
        if gap > GAP_MIN and top1_score >= HIGH_CONF:
            accepted += 1
        elif top1_score >= LOW_CONF:
            ambiguous += 1
        else:
            no_match += 1
    else:
        if s[0][1] >= HIGH_CONF:
            accepted += 1
        else:
            ambiguous += 1

print(f"Accepted (STRONG MATCH): {accepted}")
print(f"Ambiguous (POSSIBLE/WEAK): {ambiguous}")
print(f"No match (NEW IDENTITY): {no_match}")"""
add_code(code10)

add_md('## 🔹 **Cell 11 — Save Model**')
code11 = """torch.save(model.state_dict(), "/kaggle/working/elephant_reid_v5.pth")
print("Model saved!")"""
add_code(code11)

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
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r'd:\Elephant_ReIdentification\kaggle\elephant-reid-training-v5.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook v5 generated!")
