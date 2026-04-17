"""
Generate a montage of 20 random head crops for visual consistency validation.
"""
import os
import random
import cv2
import numpy as np
from pathlib import Path

def create_montage(input_dir, output_path, num_images=20, grid_cols=5, target_size=(224, 224)):
    input_path = Path(input_dir)
    images_list = list(input_path.rglob("*.jpg"))
    
    if not images_list:
        print("No images found for montage.")
        return
        
    random.seed(42)  # For reproducible montages
    samples = random.sample(images_list, min(num_images, len(images_list)))
    
    # Read and resize images
    imgs = []
    for p in samples:
        img = cv2.imread(str(p))
        if img is not None:
            # Resize
            img = cv2.resize(img, target_size)
            # Add filename text
            cv2.putText(img, p.name, (10, target_size[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            imgs.append(img)
            
    if not imgs:
        print("Failed to read any images.")
        return
        
    # Pad if not enough images
    while len(imgs) < num_images:
        imgs.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
        
    # Create grid
    grid_rows = int(np.ceil(num_images / grid_cols))
    
    row_images = []
    for r in range(grid_rows):
        start = r * grid_cols
        end = min((r + 1) * grid_cols, len(imgs))
        row = np.hstack(imgs[start:end])
        row_images.append(row)
        
    montage = np.vstack(row_images)
    
    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), montage)
    print(f"Montage saved to {out_path}")

if __name__ == "__main__":
    create_montage(
        "data/processed_heads", 
        "debug/head_crops_montage.jpg", 
        num_images=20, 
        grid_cols=5
    )
