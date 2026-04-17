"""
STEP 3-5: Split labeled data + Train YOLO head detector
=========================================================

Run this AFTER you finish labeling in labelImg.
It will:
  1. Split 80/20 into train/val
  2. Create data.yaml
  3. Launch YOLO training

Usage:
    python train_head_detector.py
"""

import os
import shutil
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LABEL_DIR = PROJECT_ROOT / "head_labeling" / "images"
DATASET_DIR = PROJECT_ROOT / "head_detector_dataset"

def setup_dataset():
    """Split labeled images into train/val (80/20)."""
    
    # Find all labeled images (those with corresponding .txt files)
    all_labels = sorted(LABEL_DIR.glob("*.txt"))
    # Filter out classes.txt if present
    all_labels = [l for l in all_labels if l.name != "classes.txt"]
    
    labeled_images = []
    for label_path in all_labels:
        # Find matching image
        stem = label_path.stem
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            img_path = LABEL_DIR / (stem + ext)
            if img_path.exists():
                labeled_images.append((img_path, label_path))
                break
    
    print(f"Found {len(labeled_images)} labeled images")
    
    if len(labeled_images) < 10:
        print("❌ Too few labeled images. Need at least 10.")
        print("   Open labelImg and label more images first!")
        return False
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(labeled_images)
    
    split = int(0.8 * len(labeled_images))
    train_set = labeled_images[:split]
    val_set = labeled_images[split:]
    
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")
    
    # Create directory structure
    for subset in ['train', 'val']:
        (DATASET_DIR / "images" / subset).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / subset).mkdir(parents=True, exist_ok=True)
    
    # Copy files
    for img_path, label_path in train_set:
        shutil.copy2(img_path, DATASET_DIR / "images" / "train" / img_path.name)
        shutil.copy2(label_path, DATASET_DIR / "labels" / "train" / label_path.name)
    
    for img_path, label_path in val_set:
        shutil.copy2(img_path, DATASET_DIR / "images" / "val" / img_path.name)
        shutil.copy2(label_path, DATASET_DIR / "labels" / "val" / label_path.name)
    
    # Create data.yaml
    yaml_content = f"""path: {DATASET_DIR}
train: images/train
val: images/val

names:
  0: elephant_head
"""
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\n✅ Dataset created at: {DATASET_DIR}")
    print(f"✅ data.yaml saved to: {yaml_path}")
    
    return True


def train():
    """Train YOLOv8n head detector."""
    from ultralytics import YOLO
    
    yaml_path = DATASET_DIR / "data.yaml"
    
    print("\n" + "="*60)
    print("TRAINING YOLO HEAD DETECTOR")
    print("="*60)
    
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=8,
        patience=10,
        save=True,
        project=str(PROJECT_ROOT / "runs" / "head_detect"),
        name="train",
        exist_ok=True,
    )
    
    best_weights = PROJECT_ROOT / "runs" / "head_detect" / "train" / "weights" / "best.pt"
    print(f"\n✅ Training complete!")
    print(f"Best weights: {best_weights}")
    
    return best_weights


def sanity_check(weights_path):
    """Run prediction on a few val images to visually verify."""
    from ultralytics import YOLO
    
    model = YOLO(str(weights_path))
    
    val_images = list((DATASET_DIR / "images" / "val").glob("*.jpg")) + \
                 list((DATASET_DIR / "images" / "val").glob("*.JPG"))
    
    test_images = val_images[:5]  # Just 5 for sanity check
    
    print("\n" + "="*60)
    print("SANITY CHECK — Predictions on val images")
    print("="*60)
    
    results = model.predict(
        source=[str(p) for p in test_images],
        save=True,
        project=str(PROJECT_ROOT / "runs" / "head_detect"),
        name="sanity_check",
        exist_ok=True,
        conf=0.3,
    )
    
    for r in results:
        boxes = r.boxes
        print(f"  {Path(r.path).name}: {len(boxes)} head(s) detected")
    
    output_dir = PROJECT_ROOT / "runs" / "head_detect" / "sanity_check"
    print(f"\n✅ Check predictions at: {output_dir}")


if __name__ == "__main__":
    print("="*60)
    print("ELEPHANT HEAD DETECTOR — Setup & Training")
    print("="*60)
    
    # Step 3: Split dataset
    if not setup_dataset():
        exit(1)
    
    # Step 5: Train
    weights = train()
    
    # Step 6: Sanity check
    if weights and weights.exists():
        sanity_check(weights)
