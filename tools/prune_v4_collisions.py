"""
V4 Smart Collision Pruning Script

1. Purge "Poison" identities (mean < 0.50) from Kaggle evaluation.
2. Embed the rest with V4 model.
3. Identify cross-identity image pairs with similarity > 0.92.
4. Quarantine the specific image whose sim_to_own_centroid < 0.50.
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
from pipeline import HeadEmbeddingModel, EMBED_DIM

DATASET_DIR = Path("data/training_heads_v4")
QUARANTINE_DIR = DATASET_DIR / "_quarantined"
POISON_DIR = QUARANTINE_DIR / "WEAK_IDENTITY"
COLLISION_DIR = QUARANTINE_DIR / "POISON_CROP"

V4_MODEL_PATH = Path("models/elephant_head_reid_v4.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CROSS_SIM_THRESHOLD = 0.92
OUTLIER_THRESHOLD = 0.50

POISON_IDENTITIES = {
    "AF_ELE_14_R", "AF_ELE_24_R", "AF_ELE_28_L", "AF_ELE_41_L", 
    "Calf_61", "JF_ELE_75", "JU_ELE_37", "JU_ELE_95"
}

def main():
    print("=" * 70)
    print("🐘 V4 SMART PRUNING SCRIPT")
    print("=" * 70)
    
    POISON_DIR.mkdir(parents=True, exist_ok=True)
    COLLISION_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nPhase 1: Quarantining Poison Identities")
    for poison_id in POISON_IDENTITIES:
        target_dir = DATASET_DIR / poison_id
        if target_dir.exists():
            print(f"  [QUARANTINE] Moving poison identity folder: {poison_id}")
            try:
                shutil.move(str(target_dir), str(POISON_DIR / poison_id))
            except Exception as e:
                print(f"  Failed to move {poison_id}: {e}")

    print(f"\nPhase 2: Loading V4 Model from {V4_MODEL_PATH}")
    model = HeadEmbeddingModel(embed_dim=EMBED_DIM).to(DEVICE)
    try:
        checkpoint = torch.load(str(V4_MODEL_PATH), map_location=DEVICE, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    identity_images = defaultdict(list)
    skipped_dirs = {"_quarantined", "_filtered"}

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
                identity_images[identity].append({"path": path, "emb": emb})
            except Exception as e:
                print(f"  [ERROR] Failed reading {path}: {e}")

    total_imgs = sum(len(v) for v in identity_images.items())
    print(f"Loaded {sum(len(v) for v in identity_images.values())} images across {len(identity_images)} identities.")

    print("\nPhase 3: Computing Centroids")
    centroids = {}
    for identity, items in identity_images.items():
        embs = torch.stack([e["emb"] for e in items])
        centroids[identity] = F.normalize(embs.mean(0), p=2, dim=0)

    print("\nPhase 4: Smart Collision Pruning (Sim > 0.92)")
    identities = sorted(identity_images.keys())
    
    quarantine_list = set()
    collisions_found = 0
    
    # We will build a flat list of all images to do cross-identity checking
    flat_images = []
    for identity, items in identity_images.items():
        for item in items:
            flat_images.append({
                "identity": identity,
                "path": item["path"],
                "emb": item["emb"],
                "centroid": centroids[identity] # own centroid
            })
            
    print("Computing NxN image similarities...")
    # Chunking to avoid massive memory usage for 1000x1000
    all_embs = torch.stack([item["emb"] for item in flat_images])
    sim_matrix = all_embs @ all_embs.T

    n = len(flat_images)
    for i in range(n):
        for j in range(i + 1, n):
            if flat_images[i]["identity"] == flat_images[j]["identity"]:
                continue
                
            sim = float(sim_matrix[i][j])
            if sim > CROSS_SIM_THRESHOLD:
                # We have a collision!
                collisions_found += 1
                imgA = flat_images[i]
                imgB = flat_images[j]
                
                simA_centroid = float(F.cosine_similarity(imgA["emb"].unsqueeze(0), imgA["centroid"].unsqueeze(0)))
                simB_centroid = float(F.cosine_similarity(imgB["emb"].unsqueeze(0), imgB["centroid"].unsqueeze(0)))
                
                print(f"  [COLLISION] {imgA['identity']} ↔ {imgB['identity']}: {sim:.4f}")
                
                # Smart Pruning
                if simA_centroid < OUTLIER_THRESHOLD:
                    print(f"      -> ❌ Bad crop in {imgA['identity']} (self_sim {simA_centroid:.3f}). Quarantining {os.path.basename(imgA['path'])}")
                    quarantine_list.add(imgA["path"])
                if simB_centroid < OUTLIER_THRESHOLD:
                    print(f"      -> ❌ Bad crop in {imgB['identity']} (self_sim {simB_centroid:.3f}). Quarantining {os.path.basename(imgB['path'])}")
                    quarantine_list.add(imgB["path"])
                    
                if simA_centroid >= OUTLIER_THRESHOLD and simB_centroid >= OUTLIER_THRESHOLD:
                    print(f"      -> ⚠️ Genuine similarity (A_self={simA_centroid:.3f}, B_self={simB_centroid:.3f}). Leaving both.")

    print(f"\nTotal bad collision crops to quarantine: {len(quarantine_list)}")
    
    if quarantine_list:
        removed_count = defaultdict(int)
        for path in quarantine_list:
            identity = os.path.basename(os.path.dirname(path))
            dest_dir = COLLISION_DIR / identity
            dest_dir.mkdir(parents=True, exist_ok=True)
            new_path = dest_dir / os.path.basename(path)
            try:
                shutil.move(str(path), str(new_path))
                removed_count[identity] += 1
            except Exception as e:
                print(f"Failed to move {path}: {e}")
                
        print("\n📊 SMART PRUNING IMPACT:")
        for identity, count in sorted(removed_count.items()):
            remaining = len(identity_images[identity]) - count
            warn = "⚠️ HIGHRISK" if remaining < 2 else ""
            print(f"  {identity:20s}: Removed {count}. Remaining: {remaining} {warn}")
    
    # Cleanup empty
    for root, dirs, files in os.walk(DATASET_DIR, topdown=False):
        if root == str(DATASET_DIR) or "_quarantined" in root:
            continue
        if not os.listdir(root):
            os.rmdir(root)

    print("\n✅ Smart Pruning Complete!")

if __name__ == "__main__":
    main()
