import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "restructured"

def is_blurry(img_np, threshold=50):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def analyze_dataset():
    if not DATA_DIR.exists():
        print(f"Directory not found: {DATA_DIR}")
        return

    total_images = 0
    rejected_brightness = 0
    rejected_size = 0
    rejected_blur = 0
    rejected_low_contrast = 0
    rejected_load_error = 0
    
    print("Starting comprehensive crop analysis...")

    for folder in DATA_DIR.iterdir():
        if not folder.is_dir():
            continue
            
        for img_path in folder.iterdir():
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            total_images += 1
            
            try:
                # Load image
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img_np = np.array(img)
                    
                # 1. Size Check
                h, w, c = img_np.shape
                if h < 100 or w < 50:
                    rejected_size += 1
                    continue
                    
                # 2. Brightness Check
                if img_np.mean() < 40:
                    rejected_brightness += 1
                    continue
                    
                # 3. Contrast Check
                if img_np.std() < 20:
                    rejected_low_contrast += 1
                    continue
                    
                # 4. Blur Check (Laplacian Variance)
                if is_blurry(img_np, threshold=50):
                    rejected_blur += 1
                    continue
                    
            except Exception as e:
                rejected_load_error += 1
                continue

    total_rejected = rejected_brightness + rejected_size + rejected_blur + rejected_low_contrast + rejected_load_error
    reject_ratio = (total_rejected / total_images) * 100 if total_images > 0 else 0

    print("\n--- ANALYSIS RESULTS ---")
    print(f"Total Images Scanned: {total_images}")
    print(f"Total Rejected: {total_rejected} ({reject_ratio:.2f}%)")
    print(f"  -> Too Small (h<100 or w<50): {rejected_size}")
    print(f"  -> Too Dark (mean<40): {rejected_brightness}")
    print(f"  -> Low Contrast (std<20): {rejected_low_contrast}")
    print(f"  -> Blurry (Laplacian Var < 50): {rejected_blur}")
    print(f"  -> Load Errors/Corrupt: {rejected_load_error}")
    
    if reject_ratio > 25:
        print("\n⚠️ WARNING: Filter is too aggressive! Removing >25% of your dataset.")
    elif reject_ratio < 5:
        print("\n✅ Filter is very light (<5%). We may need to increase the Blur Threshold.")
    else:
        print("\n✅ Filter is optimal (between 5% and 20%).")

if __name__ == "__main__":
    analyze_dataset()
