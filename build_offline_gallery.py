import os
import shutil

def build_offline_bundle():
    base_dir = "d:/Elephant_ReIdentification"
    src_dir = os.path.join(base_dir, "data/training_heads_v6")
    out_dir = os.path.join(base_dir, "gallery_references")
    
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    def process_identity(identity_name, src_id_path):
        target_folder = os.path.join(out_dir, identity_name)
        
        imgs = [f for f in os.listdir(src_id_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not imgs:
            return
            
        os.makedirs(target_folder, exist_ok=True)
        # Sort to ensure consistent images
        imgs.sort()
        # Take max 3 reference images per identity
        for img in imgs[:3]:
            shutil.copy(os.path.join(src_id_path, img), os.path.join(target_folder, img))

    if os.path.exists(src_dir):
        for idx in os.listdir(src_dir):
            id_path = os.path.join(src_dir, idx)
            if os.path.isdir(id_path):
                process_identity(idx, id_path)

    print(f"Offline Gallery Bundle successfully built from curated v6 dataset at: {out_dir}")

if __name__ == "__main__":
    build_offline_bundle()
