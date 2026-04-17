"""
Generates pristine training dataset from raw directory.

Reads raw images from data/raw/Herd... and data/raw/Makhna...
Runs pipeline.detect_and_crop_head to get the single BEST head crop.
Discards any fallbacks or failures.
Saves to data/training_heads_v4/
"""
import sys
import os
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline import detect_and_crop_head

DATA_RAW = Path("data/raw")
OUT_DIR = Path("data/training_heads_v6")

def main():
    if not DATA_RAW.exists():
        print(f"Directory {DATA_RAW} not found. Cannot proceed.")
        return

    print("=" * 60)
    print("Generating Pristine Re-ID Dataset from RAW Detections")
    print("=" * 60)
    
    # 1. Gather all raw image paths by recursively finding directories with images
    identity_images = {}
    for source_dir in ["Herd_ID_Udalguri_24", "Makhna_id_udalguri_24"]:
        src_path = DATA_RAW / source_dir
        if not src_path.exists():
            continue
            
        for root, dirs, files in os.walk(src_path):
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                parts = Path(root).relative_to(DATA_RAW).parts
                if parts[0] == "Herd_ID_Udalguri_24":
                    identity_dir = f"{parts[1]}_{parts[-1]}"
                else:
                    identity_dir = parts[-1]
                    
                if identity_dir not in identity_images:
                    identity_images[identity_dir] = []
                identity_images[identity_dir].extend([(Path(root) / f) for f in images])

    total_images = sum(len(imgs) for imgs in identity_images.values())
    print(f"Found {total_images} raw images across {len(identity_images)} identities.")

    # 2. Process
    success = 0
    fallbacks_skipped = 0
    failures = 0

    for identity, filepaths in tqdm(identity_images.items(), desc="Processing Identities"):
        out_sys = OUT_DIR / identity
        out_sys.mkdir(parents=True, exist_ok=True)

        for img_path in filepaths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                failures += 1
                continue

            # Run strictly without fallback to ensure only pure YOLO bounding boxes survive
            res = detect_and_crop_head(img_bgr, allow_fallback=False)
            
            if res is None or (isinstance(res, tuple) and res[0] is None):
                failures += 1
                continue
                
            crop_rgb, is_fallback = res[:2]
            if is_fallback:
                # Should not happen because allow_fallback=False, but just in case
                fallbacks_skipped += 1
                continue
                
            # Valid pristine crop found
            out_file = out_sys / img_path.name
            crop_rgb.save(str(out_file))
            success += 1

    print("\n" + "=" * 60)
    print("DONE:")
    print(f"  Pristine head crops saved: {success}  ({(success/max(total_images,1))*100:.1f}%)")
    print(f"  Rejected (No valid head): {failures}")
    print(f"  Saved to: {OUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
