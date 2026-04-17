"""
Targeted cleanup for v7 training dataset.
Fixes the worst identities and cross-identity collisions from the v7.0 eval report.

Two passes:
1. INTRA-IDENTITY CLEANUP: For the worst 10 identities, remove outlier images
   (images whose embedding is far from the folder centroid).
2. CROSS-IDENTITY COLLISION FIX: Merge Makhna_6 into Makhna_7 (nearly identical),
   and quarantine specific bad images causing false matches in other pairs.
"""
import sys, os, shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from pipeline import HeadEmbeddingModel, EMBED_DIM

DATASET_DIR = Path("data/training_heads_v6")
QUARANTINE_DIR = DATASET_DIR / "_quarantined" / "v7_targeted"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
print("Loading model...")
MODEL_PATH = Path("models")
model_file = None
for name in ("elephant_head_reid_v5.pth", "elephant_head_reid_v4.pth", 
             "elephant_head_reid_v3.pth", "elephant_head_reid_v2.pth"):
    p = MODEL_PATH / name
    if p.exists():
        model_file = p
        break
if model_file is None:
    print("ERROR: No model found in models/")
    sys.exit(1)

print(f"Using model: {model_file}")
model = HeadEmbeddingModel(embed_dim=EMBED_DIM).to(DEVICE)
checkpoint = torch.load(str(model_file), map_location=DEVICE, weights_only=False)
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

def embed_folder(folder_path):
    """Embed all images in a folder, return list of (path, embedding)."""
    items = []
    for f in sorted(os.listdir(folder_path)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(folder_path, f)
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = model(tensor).squeeze().cpu()
            items.append((path, emb))
        except Exception as e:
            print(f"  [ERROR] {path}: {e}")
    return items

def quarantine(path, reason, identity):
    """Move a file to quarantine."""
    dest_dir = QUARANTINE_DIR / reason / identity
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / os.path.basename(path)
    shutil.move(str(path), str(dest))
    print(f"    -> Quarantined: {os.path.basename(path)} ({reason})")

def merge_folders(src_name, dst_name):
    """Merge src identity folder into dst (they're the same elephant)."""
    src = DATASET_DIR / src_name
    dst = DATASET_DIR / dst_name
    if not src.exists():
        print(f"  [SKIP] {src_name} not found")
        return
    if not dst.exists():
        print(f"  [SKIP] {dst_name} not found")
        return
    
    count = 0
    for f in os.listdir(src):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_path = src / f
            dst_path = dst / f
            if dst_path.exists():
                # Rename to avoid collision
                stem = Path(f).stem
                ext = Path(f).suffix
                dst_path = dst / f"{stem}_from_{src_name}{ext}"
            shutil.move(str(src_path), str(dst_path))
            count += 1
    
    # Remove empty source directory
    try:
        if src.exists() and not any(src.iterdir()):
            src.rmdir()
    except OSError:
        pass
    
    print(f"  Merged {count} images from {src_name} -> {dst_name}")

# ============================================================================
# PASS 1: INTRA-IDENTITY CLEANUP
# ============================================================================
print("\n" + "=" * 70)
print("PASS 1: INTRA-IDENTITY CLEANUP (Worst Identities)")
print("=" * 70)

# These identities have very low intra-class similarity
WORST_IDENTITIES = [
    "Herd_2_SAF_ELE_58_R",       # Mean=0.32
    "Herd_4_Calf_55",             # Mean=0.32
    "Herd_2_SAU_ELE_70",          # Mean=0.36
    "Herd_2_AF_ELE_15",           # Mean=0.42
    "Makhna_4",                   # Mean=0.42
    "Herd_4_AF_ELE_34_L",         # Mean=0.47
    "Herd_2_SAF_ELE_59_d_check_adult",  # Mean=0.50
    "Makhna_1",                   # Mean=0.51 (136 pairs = large folder)
    "Herd_2_AF_ELE_4_R",          # Mean=0.55
    "Makhna_12",                  # Mean=0.64
]

# Threshold: any image with sim < this to its own centroid gets quarantined
OUTLIER_THRESHOLD = 0.35
# For folders with very low mean, be even more aggressive
AGGRESSIVE_THRESHOLD = 0.45

total_quarantined = 0

for identity in WORST_IDENTITIES:
    folder = DATASET_DIR / identity
    if not folder.exists():
        print(f"\n  [SKIP] {identity} - folder not found")
        continue
    
    items = embed_folder(str(folder))
    if len(items) < 2:
        print(f"\n  [SKIP] {identity} - only {len(items)} image(s)")
        continue
    
    # Compute centroid
    embs = torch.stack([e for _, e in items])
    centroid = F.normalize(embs.mean(0), p=2, dim=0)
    
    # Compute self-similarity for each image
    sims = (embs @ centroid.unsqueeze(1)).squeeze()
    mean_sim = float(sims.mean())
    
    print(f"\n  {identity} ({len(items)} images, mean_sim_to_centroid={mean_sim:.3f})")
    
    # Use aggressive threshold for the worst folders
    threshold = AGGRESSIVE_THRESHOLD if mean_sim < 0.45 else OUTLIER_THRESHOLD
    
    removed = 0
    remaining = len(items)
    for idx in range(len(items)):
        path, emb = items[idx]
        sim = float(sims[idx])
        # Don't remove so many that the folder becomes unusable (keep at least 3)
        if sim < threshold and (remaining - removed) > 3:
            print(f"    [OUTLIER] {os.path.basename(path)}: sim={sim:.3f}")
            quarantine(path, "outlier", identity)
            removed += 1
    
    if removed == 0:
        print(f"    All images OK (above threshold {threshold:.2f})")
    else:
        print(f"    Removed {removed}/{len(items)} outliers")
        total_quarantined += removed

# ============================================================================
# PASS 2: CROSS-IDENTITY COLLISION FIXES
# ============================================================================
print("\n" + "=" * 70)
print("PASS 2: CROSS-IDENTITY COLLISION FIXES")
print("=" * 70)

# --- Fix 1: Makhna_6 and Makhna_7 ---
# These are almost certainly the same elephant (similarity 0.86-0.88 across ALL images).
# The DSCN numbers are consecutive (4659-4671), suggesting same photo session.
# Action: MERGE Makhna_6 into Makhna_7
print("\n[FIX 1] Makhna_6 <-> Makhna_7 (sim=0.87+, consecutive DSCNs => same elephant)")
merge_folders("Makhna_6", "Makhna_7")

# --- Fix 2: Makhna_1 <-> Makhna_2 and Makhna_1 <-> Makhna_3 ---
# High cross-similarity (0.82-0.85). Need to check specific bad images.
# The collision is driven by specific images, not all images.
# Quarantine the specific images that are closest to the wrong identity.
print("\n[FIX 2] Makhna collision check (Makhna_1/2/3)")
makhna_folders = {}
for name in ["Makhna_1", "Makhna_2", "Makhna_3"]:
    folder = DATASET_DIR / name
    if folder.exists():
        makhna_folders[name] = embed_folder(str(folder))

if len(makhna_folders) >= 2:
    # Compute centroids
    makhna_centroids = {}
    for name, items in makhna_folders.items():
        if items:
            embs = torch.stack([e for _, e in items])
            makhna_centroids[name] = F.normalize(embs.mean(0), p=2, dim=0)
    
    # For each image in each folder, check if it's closer to another Makhna centroid
    for name, items in makhna_folders.items():
        own_centroid = makhna_centroids.get(name)
        if own_centroid is None:
            continue
        other_names = [n for n in makhna_centroids if n != name]
        
        for path, emb in items:
            self_sim = float(F.cosine_similarity(emb.unsqueeze(0), own_centroid.unsqueeze(0)))
            for other_name in other_names:
                other_sim = float(F.cosine_similarity(emb.unsqueeze(0), makhna_centroids[other_name].unsqueeze(0)))
                if other_sim > self_sim + 0.10:
                    print(f"  [MISLABEL] {name}/{os.path.basename(path)}: "
                          f"self={self_sim:.3f}, {other_name}={other_sim:.3f}")
                    quarantine(path, "mislabel_makhna", name)
                    total_quarantined += 1
                    break  # Only quarantine once per image

# --- Fix 3: Other specific cross-identity problem pairs ---
print("\n[FIX 3] Checking other high-similarity cross-identity pairs")
CROSS_PAIRS_TO_CHECK = [
    ("Herd_4_JF_ELE_72", "Herd_4_JF_ELE_74"),
    ("Herd_2_AF_ELE_27", "Herd_2_SAF_ELE_63"),
    ("Herd_2_AF_ELE_1", "Herd_2_AF_ELE_8"),
    ("Herd_4_JM_ELE_82", "Herd_4_JM_ELE_86"),
]

for name_a, name_b in CROSS_PAIRS_TO_CHECK:
    folder_a = DATASET_DIR / name_a
    folder_b = DATASET_DIR / name_b
    if not folder_a.exists() or not folder_b.exists():
        continue
    
    items_a = embed_folder(str(folder_a))
    items_b = embed_folder(str(folder_b))
    
    if not items_a or not items_b:
        continue
    
    centroid_a = F.normalize(torch.stack([e for _, e in items_a]).mean(0), p=2, dim=0)
    centroid_b = F.normalize(torch.stack([e for _, e in items_b]).mean(0), p=2, dim=0)
    
    cross_sim = float(F.cosine_similarity(centroid_a.unsqueeze(0), centroid_b.unsqueeze(0)))
    print(f"\n  {name_a} <-> {name_b}: centroid_sim={cross_sim:.3f}")
    
    # Check each image in A: is it closer to B's centroid?
    for path, emb in items_a:
        self_sim = float(F.cosine_similarity(emb.unsqueeze(0), centroid_a.unsqueeze(0)))
        other_sim = float(F.cosine_similarity(emb.unsqueeze(0), centroid_b.unsqueeze(0)))
        if other_sim > self_sim + 0.08:
            print(f"    [MISLABEL] {name_a}/{os.path.basename(path)}: "
                  f"self={self_sim:.3f}, {name_b}={other_sim:.3f}")
            quarantine(path, "cross_identity", name_a)
            total_quarantined += 1
    
    # Check each image in B: is it closer to A's centroid?
    for path, emb in items_b:
        self_sim = float(F.cosine_similarity(emb.unsqueeze(0), centroid_b.unsqueeze(0)))
        other_sim = float(F.cosine_similarity(emb.unsqueeze(0), centroid_a.unsqueeze(0)))
        if other_sim > self_sim + 0.08:
            print(f"    [MISLABEL] {name_b}/{os.path.basename(path)}: "
                  f"self={self_sim:.3f}, {name_a}={other_sim:.3f}")
            quarantine(path, "cross_identity", name_b)
            total_quarantined += 1

# ============================================================================
# PASS 3: Clean up empty directories
# ============================================================================
print("\n" + "=" * 70)
print("CLEANUP")
print("=" * 70)

empty_removed = 0
for root, dirs, files in os.walk(str(DATASET_DIR), topdown=False):
    if "_quarantined" in root or "_filtered" in root:
        continue
    imgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not imgs and root != str(DATASET_DIR):
        try:
            os.rmdir(root)
            print(f"  Removed empty directory: {os.path.basename(root)}")
            empty_removed += 1
        except OSError:
            pass

# Final count
total_remaining = 0
total_identities = 0
for d in os.listdir(DATASET_DIR):
    full = DATASET_DIR / d
    if full.is_dir() and not d.startswith("_"):
        imgs = [f for f in os.listdir(full) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            total_identities += 1
            total_remaining += len(imgs)

print(f"\n{'=' * 70}")
print(f"SUMMARY")
print(f"{'=' * 70}")
print(f"  Total quarantined this run: {total_quarantined}")
print(f"  Empty folders removed: {empty_removed}")
print(f"  Remaining: {total_remaining} images across {total_identities} identities")
print(f"  Quarantine location: {QUARANTINE_DIR}")
