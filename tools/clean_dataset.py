"""
Data Cleaning: Find and quarantine mislabeled crops in processed_heads.

Strategy:
1. Embed all crops using the v2 model (or v1 if v2 not available)
2. For each identity, find crops whose embedding is closer to ANOTHER
   identity's centroid than to their own → likely mislabeled
3. Move flagged crops to a quarantine folder for manual review
4. Report statistics

Also flags source images that produced multi-detections (>1 head) so they
can be skipped during future preprocessing.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
from torchvision import transforms

# ── Config ──────────────────────────────────────────────────────────────────
PROCESSED_HEADS = Path("data/processed_heads")
QUARANTINE_DIR  = Path("data/processed_heads/_quarantined")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Similarity threshold: if a crop is more similar to another identity's
# centroid than to its own centroid by this margin, it's flagged
MISLABEL_MARGIN = 0.05  

# Also flag if a crop's similarity to its own centroid is below this
MIN_SELF_SIM = 0.30

# ── Load model ──────────────────────────────────────────────────────────────
from pipeline import HeadEmbeddingModel, EMBED_DIM, REID_MODEL_PATH

print(f"Loading model: {REID_MODEL_PATH}")
model = HeadEmbeddingModel(embed_dim=EMBED_DIM).to(DEVICE)
checkpoint = torch.load(str(REID_MODEL_PATH), map_location=DEVICE, weights_only=True)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Scan dataset ────────────────────────────────────────────────────────────
print(f"\nScanning: {PROCESSED_HEADS}")

identity_images = defaultdict(list)  # identity_name → [(path, embedding)]
skipped_dirs = {"_filtered", "_no_head_in_crop", "_quarantined"}

for dirpath, dirnames, filenames in os.walk(PROCESSED_HEADS):
    # Skip utility directories
    dirname = os.path.basename(dirpath)
    if dirname.startswith("_"):
        continue
    
    images = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        continue
    
    rel = os.path.relpath(dirpath, PROCESSED_HEADS)
    
    for img_file in images:
        img_path = os.path.join(dirpath, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = model(tensor).squeeze().cpu()
            identity_images[rel].append((img_path, emb))
        except Exception as e:
            print(f"  ⚠️ Failed to process {img_path}: {e}")

print(f"\nFound {sum(len(v) for v in identity_images.values())} images "
      f"across {len(identity_images)} identities")

# ── Compute centroids ───────────────────────────────────────────────────────
centroids = {}
for identity, items in identity_images.items():
    embs = torch.stack([e for _, e in items])
    centroids[identity] = F.normalize(embs.mean(0), p=2, dim=0)

identity_names = sorted(centroids.keys())
centroid_matrix = torch.stack([centroids[n] for n in identity_names])  # (N_ids, D)

# ── Find mislabeled crops ──────────────────────────────────────────────────
flagged = []  # (path, true_identity, best_other_identity, self_sim, best_other_sim)
clean_count = 0

for identity, items in identity_images.items():
    own_idx = identity_names.index(identity)
    
    for img_path, emb in items:
        # Similarity to all centroids
        sims = (emb.unsqueeze(0) @ centroid_matrix.T).squeeze()  # (N_ids,)
        
        self_sim = float(sims[own_idx])
        
        # Best OTHER identity
        sims_copy = sims.clone()
        sims_copy[own_idx] = -999  # mask self
        best_other_idx = int(sims_copy.argmax())
        best_other_sim = float(sims_copy[best_other_idx])
        best_other_name = identity_names[best_other_idx]
        
        # Flag conditions:
        # 1. More similar to another identity than own (with margin)
        # 2. OR very low similarity to own centroid
        is_mislabel = (best_other_sim > self_sim + MISLABEL_MARGIN)
        is_outlier = (self_sim < MIN_SELF_SIM)
        
        if is_mislabel or is_outlier:
            reason = "MISLABEL" if is_mislabel else "OUTLIER"
            flagged.append((img_path, identity, best_other_name, 
                          self_sim, best_other_sim, reason))
        else:
            clean_count += 1

# ── Report ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"RESULTS: {len(flagged)} flagged, {clean_count} clean")
print(f"{'='*70}")

if flagged:
    # Group by identity
    by_identity = defaultdict(list)
    for item in flagged:
        by_identity[item[1]].append(item)
    
    print(f"\nFlagged crops by identity:")
    for identity in sorted(by_identity.keys()):
        items = by_identity[identity]
        print(f"\n  📁 {identity} ({len(items)} flagged):")
        for path, own_id, other_id, self_sim, other_sim, reason in items:
            fname = os.path.basename(path)
            print(f"    ❌ [{reason}] {fname}")
            print(f"       own={self_sim:.3f}  best_other={other_id}({other_sim:.3f})")

# ── Quarantine ──────────────────────────────────────────────────────────────
if flagged:
    print(f"\n{'='*70}")
    response = input(f"Move {len(flagged)} flagged crops to {QUARANTINE_DIR}? [y/N]: ")
    
    if response.lower() == 'y':
        moved = 0
        for path, identity, other_id, self_sim, other_sim, reason in flagged:
            dest_dir = QUARANTINE_DIR / identity
            os.makedirs(dest_dir, exist_ok=True)
            fname = os.path.basename(path)
            dest = dest_dir / fname
            shutil.move(path, dest)
            moved += 1
        
        print(f"✅ Moved {moved} crops to {QUARANTINE_DIR}")
        print(f"\nRemaining clean dataset:")
        
        # Recount
        total = 0
        for dirpath, dirnames, filenames in os.walk(PROCESSED_HEADS):
            dirname = os.path.basename(dirpath)
            if dirname.startswith("_"):
                continue
            images = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total += len(images)
        print(f"  {total} images remaining")
    else:
        print("Skipped quarantine. Review the flagged list above manually.")
else:
    print("\n✅ No mislabeled crops detected! Dataset looks clean.")
