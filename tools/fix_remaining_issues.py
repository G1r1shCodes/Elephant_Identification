"""
Fix remaining worst identities and cross-identity issues in training_heads_v6.
Uses the embedding model to identify which specific images are outliers.
"""
import sys, os, shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from pipeline import HeadEmbeddingModel, EMBED_DIM

DATASET_DIR = Path("data/training_heads_v6")
QUARANTINE = DATASET_DIR / "_quarantined" / "v7_identity_fix"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
MODEL_PATH = Path("models")
model_file = None
for name in ("elephant_head_reid_v5.pth", "elephant_head_reid_v4.pth", "elephant_head_reid_v3.pth"):
    p = MODEL_PATH / name
    if p.exists():
        model_file = p; break

print(f"Model: {model_file}")
model = HeadEmbeddingModel(embed_dim=EMBED_DIM).to(DEVICE)
ckpt = torch.load(str(model_file), map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def embed_folder(folder):
    items = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        path = os.path.join(folder, f)
        try:
            img = Image.open(path).convert("RGB")
            t = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = model(t).squeeze().cpu()
            items.append((path, f, emb))
        except: pass
    return items

def quarantine(path, reason):
    dest = QUARANTINE / reason
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest / os.path.basename(path)))

total_moved = 0

# ============================================================================
# PASS 1: Clean worst identities — remove images far from centroid
# ============================================================================
print("\n" + "="*60)
print("PASS 1: WORST IDENTITY CLEANUP")
print("="*60)

WORST_IDS = {
    "Herd_2_SAF_ELE_58_R": 0.32,   # only 2 imgs, skip if too few
    "Herd_4_Calf_55": 0.32,
    "Herd_2_SAU_ELE_70": 0.36,
    "Herd_2_AF_ELE_15": 0.42,
    "Makhna_4": 0.42,
    "Herd_4_AF_ELE_34_L": 0.47,
    "Herd_2_SAF_ELE_59_d_check_adult": 0.50,
    "Makhna_1": 0.51,
    "Herd_2_AF_ELE_4_R": 0.55,
    "Makhna_12": 0.64,
}

for identity, mean_sim in WORST_IDS.items():
    folder = DATASET_DIR / identity
    if not folder.exists(): continue
    items = embed_folder(str(folder))
    if len(items) < 3:
        print(f"\n  {identity}: only {len(items)} images — skipping (too few to clean)")
        continue

    embs = torch.stack([e for _, _, e in items])
    # Compute pairwise similarities
    sim_mat = (embs @ embs.t()).numpy()
    np.fill_diagonal(sim_mat, 0)
    
    # Mean similarity of each image to all others in the folder
    per_img_mean = sim_mat.sum(axis=1) / (len(items) - 1)
    
    # Find outliers: images whose mean sim to others is much lower than the group mean
    group_mean = per_img_mean.mean()
    group_std = per_img_mean.std()
    
    print(f"\n  {identity} ({len(items)} imgs, group_mean={group_mean:.3f})")
    
    removed = 0
    for idx in range(len(items)):
        path, fname, _ = items[idx]
        img_mean = per_img_mean[idx]
        # Remove if: (a) below 0.4 absolute, OR (b) more than 1.5 std below the group mean
        # But keep at least 3 images
        is_outlier = (img_mean < 0.40) or (img_mean < group_mean - 1.5 * group_std and group_std > 0.05)
        remaining = len(items) - removed
        
        if is_outlier and remaining > 3:
            print(f"    [REMOVE] {fname}: mean_sim={img_mean:.3f} (group={group_mean:.3f})")
            quarantine(path, identity)
            removed += 1
            total_moved += 1
    
    if removed == 0:
        print(f"    All OK (lowest={per_img_mean.min():.3f})")

# ============================================================================
# PASS 2: Handle Herd_2_SAF_ELE_58_R (only 2 images with mean 0.32 — likely wrong ID)
# ============================================================================
print("\n" + "="*60)
print("PASS 2: TINY NOISY IDENTITIES")
print("="*60)

folder = DATASET_DIR / "Herd_2_SAF_ELE_58_R"
if folder.exists():
    items = embed_folder(str(folder))
    if len(items) == 2:
        embs = torch.stack([e for _, _, e in items])
        sim = float((embs[0] @ embs[1]).item())
        print(f"\n  Herd_2_SAF_ELE_58_R: 2 images, internal sim={sim:.3f}")
        if sim < 0.50:
            print(f"    QUARANTINING entire folder (sim {sim:.3f} << 0.50 — these are likely not the same elephant)")
            for path, fname, _ in items:
                quarantine(path, "Herd_2_SAF_ELE_58_R")
                total_moved += 1
            try: folder.rmdir()
            except: pass

# ============================================================================
# PASS 3: Check specific cross-identity collision images 
# These are pairs where the model sees high similarity between different IDs.
# We need to check if specific images are actually mislabeled.
# ============================================================================
print("\n" + "="*60)
print("PASS 3: CROSS-IDENTITY COLLISION CHECK")
print("="*60)

# For each collision pair: embed both folders, check if any image is closer to wrong folder
COLLISION_PAIRS = [
    ("Herd_4_JF_ELE_72", "Herd_4_JF_ELE_74"),
    ("Herd_2_AF_ELE_27", "Herd_2_SAF_ELE_63"),
    ("Herd_2_AF_ELE_1", "Herd_2_AF_ELE_8"),
    ("Herd_4_JM_ELE_82", "Herd_4_JM_ELE_86"),
    ("Makhna_1", "Makhna_3"),
]

for name_a, name_b in COLLISION_PAIRS:
    fa, fb = DATASET_DIR / name_a, DATASET_DIR / name_b
    if not fa.exists() or not fb.exists(): continue
    
    items_a = embed_folder(str(fa))
    items_b = embed_folder(str(fb))
    if not items_a or not items_b: continue
    
    embs_a = torch.stack([e for _, _, e in items_a])
    embs_b = torch.stack([e for _, _, e in items_b])
    cent_a = F.normalize(embs_a.mean(0), p=2, dim=0)
    cent_b = F.normalize(embs_b.mean(0), p=2, dim=0)
    
    cross = float(F.cosine_similarity(cent_a.unsqueeze(0), cent_b.unsqueeze(0)))
    print(f"\n  {name_a} <-> {name_b} (centroid_sim={cross:.3f})")
    
    # Check A images
    for path, fname, emb in items_a:
        self_sim = float(F.cosine_similarity(emb.unsqueeze(0), cent_a.unsqueeze(0)))
        other_sim = float(F.cosine_similarity(emb.unsqueeze(0), cent_b.unsqueeze(0)))
        if other_sim > self_sim + 0.05 and len(items_a) > 3:
            print(f"    [MISLABEL] {name_a}/{fname}: self={self_sim:.3f} vs {name_b}={other_sim:.3f}")
            quarantine(path, f"cross_{name_a}")
            total_moved += 1
    
    # Check B images
    for path, fname, emb in items_b:
        self_sim = float(F.cosine_similarity(emb.unsqueeze(0), cent_b.unsqueeze(0)))
        other_sim = float(F.cosine_similarity(emb.unsqueeze(0), cent_a.unsqueeze(0)))
        if other_sim > self_sim + 0.05 and len(items_b) > 3:
            print(f"    [MISLABEL] {name_b}/{fname}: self={self_sim:.3f} vs {name_a}={other_sim:.3f}")
            quarantine(path, f"cross_{name_b}")
            total_moved += 1

# ============================================================================
# Final stats
# ============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

total_imgs = 0
total_ids = 0
for d in sorted(os.listdir(DATASET_DIR)):
    full = DATASET_DIR / d
    if full.is_dir() and not d.startswith("_"):
        imgs = [f for f in os.listdir(full) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if imgs:
            total_ids += 1
            total_imgs += len(imgs)

print(f"  Quarantined this run: {total_moved}")
print(f"  Remaining: {total_imgs} images across {total_ids} identities")
