"""
Script to list all individual crops that collide across IDs (sim > 0.90) for manual review.
"""
import sys
import os
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline import HeadEmbeddingModel, EMBED_DIM

DATASET_DIR = Path("data/training_heads_v4")
V4_MODEL_PATH = Path("models/elephant_head_reid_v4.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CROSS_SIM_THRESHOLD = 0.92

def main():
    print("=" * 70)
    print("🐘 V4 COLLIDING CROP REPORT (> 0.92)")
    print("=" * 70)
    
    model = HeadEmbeddingModel(embed_dim=EMBED_DIM).to(DEVICE)
    checkpoint = torch.load(str(V4_MODEL_PATH), map_location=DEVICE, weights_only=True)
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

    skipped_dirs = {"_quarantined", "_filtered"}
    flat_images = []

    for root, dirs, files in os.walk(DATASET_DIR):
        parts = set(Path(root).parts)
        if parts & skipped_dirs:
            continue
            
        images = [f for f in files if f.lower().endswith(('.jpg', '.png'))]
        identity = os.path.basename(root)
        
        for img in images:
            path = os.path.join(root, img)
            try:
                pil_img = Image.open(path).convert("RGB")
                tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    emb = model(tensor).squeeze().cpu()
                flat_images.append({
                    "identity": identity,
                    "filename": img,
                    "path": path,
                    "emb": emb
                })
            except Exception as e:
                pass

    if not flat_images:
        print("No images found.")
        return

    all_embs = torch.stack([item["emb"] for item in flat_images])
    sim_matrix = all_embs @ all_embs.T

    n = len(flat_images)
    collisions = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if flat_images[i]["identity"] == flat_images[j]["identity"]:
                continue
                
            sim = float(sim_matrix[i][j])
            if sim > CROSS_SIM_THRESHOLD:
                collisions.append((sim, flat_images[i], flat_images[j]))

    collisions.sort(key=lambda x: x[0], reverse=True)
    
    for sim, A, B in collisions:
        print(f"[{sim:.4f}] {A['identity']}/{A['filename']} ↔ {B['identity']}/{B['filename']}")

if __name__ == "__main__":
    main()
