"""
Model-Assisted Dataset Cleaning Pipeline
Target: data/training_heads_v4

1. Outlier Check: Removes garbage crops (self_sim < 0.30)
2. Mislabel Check: Removes cross-identity contamination (other_sim > self_sim + 0.10)
3. Collision Check: Identifies identical IDs and merges/quarantines duplicates.
"""
import sys
import os
import shutil
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from collections import defaultdict
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline import HeadEmbeddingModel, EMBED_DIM, REID_MODEL_PATH

DATASET_DIR = Path("data/training_heads_v6")
QUARANTINE_DIR = DATASET_DIR / "_quarantined"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MISLABEL_MARGIN = 0.10
MIN_SELF_SIM = 0.30

# The user explicitly called out some exact collisions/bad pairs
# We'll use the script to figure out exactly what to do, but we
# will also definitively detect all 0.90+ centroid collisions.

def main():
    print("=" * 70)
    print("🐘 V4 MODEL-ASSISTED DATASET CLEANUP")
    print("=" * 70)

    print(f"Loading V3 model: {REID_MODEL_PATH}")
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

    identity_images = defaultdict(list)
    skipped_dirs = {"_quarantined", "_filtered"}

    # 1. Embed everything
    print(f"Scanning {DATASET_DIR}...")
    for root, dirs, files in os.walk(DATASET_DIR):
        parts = set(Path(root).parts)
        if parts & skipped_dirs:
            continue
            
        images = [f for f in files if f.lower().endswith(('.jpg', '.png'))]
        if not images:
            continue
            
        identity = os.path.basename(root)
        for img in images:
            path = os.path.join(root, img)
            try:
                pil_img = Image.open(path).convert("RGB")
                tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    emb = model(tensor).squeeze().cpu()
                identity_images[identity].append((path, emb))
            except Exception as e:
                print(f"  [ERROR] Failed reading {path}: {e}")

    total_imgs = sum(len(v) for v in identity_images.values())
    print(f"Loaded {total_imgs} images across {len(identity_images)} identities.")

    # 2. Compute Centroids
    centroids = {}
    for identity, items in identity_images.items():
        embs = torch.stack([e for _, e in items])
        centroids[identity] = F.normalize(embs.mean(0), p=2, dim=0)

    identities = sorted(centroids.keys())
    centroid_matrix = torch.stack([centroids[n] for n in identities])

    # 3. Collision Detection (Same elephant in multiple directories)
    print("\n" + "=" * 70)
    print("🔍 CENTROID COLLISION REPORT (Sim > 0.90)")
    print("=" * 70)
    collisions_found = 0
    # We will automatically quarantine the smaller directory if sim > 0.96 (exact duplicate cluster)
    auto_quarantine_dirs = set()
    
    for i in range(len(identities)):
        for j in range(i + 1, len(identities)):
            sim = float(F.cosine_similarity(centroid_matrix[i].unsqueeze(0), centroid_matrix[j].unsqueeze(0)))
            if sim > 0.90:
                id1 = identities[i]
                id2 = identities[j]
                n1 = len(identity_images[id1])
                n2 = len(identity_images[id2])
                print(f"  [COLLISION] {id1} (n={n1}) ↔ {id2} (n={n2}): {sim:.4f}")
                collisions_found += 1
                
                # If they are practically identical clusters, we MUST remove one or they break MS-Loss.
                if sim > 0.95:
                    to_remove = id2 if n1 >= n2 else id1
                    auto_quarantine_dirs.add(to_remove)
                    print(f"      -> AUTO-QUARANTINING duplicate cluster: {to_remove}")

    # 4. Outlier & Mislabel Check
    print("\n" + "=" * 70)
    print("🧹 INTRA-CLUSTER CLEANING")
    print("=" * 70)
    
    removal_count = defaultdict(int)
    total_count = {id_name: len(items) for id_name, items in identity_images.items()}
    flagged = []

    for identity, items in identity_images.items():
        if identity in auto_quarantine_dirs:
            # Whole dir is quarantined, don't individual-clean it
            for path, _ in items:
                flagged.append((path, identity, "DUPE_CLUSTER", 1.0, 1.0, "DUPLICATE_CLUSTER"))
                removal_count[identity] += 1
            continue

        own_idx = identities.index(identity)
        
        for img_path, emb in items:
            sims = (emb.unsqueeze(0) @ centroid_matrix.T).squeeze()
            self_sim = float(sims[own_idx])
            
            sims_copy = sims.clone()
            sims_copy[own_idx] = -999
            best_other_idx = int(sims_copy.argmax())
            best_other_sim = float(sims_copy[best_other_idx])
            best_other_name = identities[best_other_idx]
            
            is_mislabel = best_other_sim > (self_sim + MISLABEL_MARGIN)
            is_outlier = self_sim < MIN_SELF_SIM
            
            if is_mislabel:
                print(f"  [MISLABEL] {os.path.basename(img_path)} | self={self_sim:.3f}, other={best_other_name}({best_other_sim:.3f})")
                flagged.append((img_path, identity, best_other_name, self_sim, best_other_sim, "MISLABEL"))
                removal_count[identity] += 1
            elif is_outlier:
                print(f"  [OUTLIER]  {os.path.basename(img_path)} | self={self_sim:.3f}")
                flagged.append((img_path, identity, "NONE", self_sim, 0.0, "OUTLIER"))
                removal_count[identity] += 1

    # 5. Review Metrics & Execute Quarantine
    print("\n" + "=" * 70)
    print("📊 CLEANING SUMMARY")
    print("=" * 70)
    for identity, removed in sorted(removal_count.items()):
        total = total_count[identity]
        pct = (removed / total) * 100
        warn = "⚠️ HIGHRISK" if pct > 35 else ""
        print(f"  {identity:20s}: Removed {removed}/{total} ({pct:.1f}%) {warn}")

    print(f"\nTotal crops to quarantine: {len(flagged)}")
    
    if flagged:
        print("\nMoving files to _quarantined...")
        for path, identity, _, _, _, reason in flagged:
            dest_dir = QUARANTINE_DIR / reason / identity
            dest_dir.mkdir(parents=True, exist_ok=True)
            new_path = dest_dir / os.path.basename(path)
            # Use copy then remove, or move. Since it's local disk, move is fast
            try:
                shutil.move(str(path), str(new_path))
            except Exception as e:
                print(f"Failed to move {path}: {e}")
                
        # Clean up empty directories
        for root, dirs, files in os.walk(DATASET_DIR, topdown=False):
            if root == str(DATASET_DIR) or "_quarantined" in root:
                continue
            if not os.listdir(root):
                os.rmdir(root)
        
        print(f"✅ Successfully quarantined {len(flagged)} bad crops.")

if __name__ == "__main__":
    main()
