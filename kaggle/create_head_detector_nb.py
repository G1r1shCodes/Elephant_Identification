import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [text]
    })

def add_code(text):
    lines = [line + '\n' for line in text.split('\n')]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

add_md("""# 🐘 Elephant Head Detector — YOLO Training

**Purpose:** Train a YOLOv8n model to detect elephant heads (head + ears region only).

**Prerequisites:**
1. Label ~100 images using labelImg (YOLO format) locally
2. Upload the labeled folder as a Kaggle dataset (images + .txt labels together)
3. Run this notebook with **GPU T4 x2** accelerator

**Dataset structure expected:**
```
/kaggle/input/elephant-head-labels/
    DSCN3044.JPG
    DSCN3044.txt
    DSCN3058.JPG
    DSCN3058.txt
    ...
    classes.txt
```""")

add_md("## Cell 1 — Install & Imports")
add_code("""!pip install ultralytics -q

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

print("Setup complete!")""")

add_md("## Cell 2 — Find & Split Labeled Data (80/20)")
add_code("""# Update this path to match your uploaded Kaggle dataset name
INPUT_BASE = Path("/kaggle/input")
DATASET_DIR = Path("/kaggle/working/dataset")

# Use recursive glob to find labels regardless of nesting
all_txt_files = sorted(INPUT_BASE.rglob("*.txt"))

# Filter out non-label files
ignored_names = ['classes.txt', 'predefined_classes.txt', 'notes.txt']
label_files = [f for f in all_txt_files if f.name.lower() not in ignored_names]

labeled_pairs = []
for label_path in label_files:
    stem = label_path.stem
    parent = label_path.parent
    # Check for image in the same folder as the label
    for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
        img_path = parent / (stem + ext)
        if img_path.exists():
            labeled_pairs.append((img_path, label_path))
            break

print(f"Found {len(labeled_pairs)} labeled image-label pairs")

if len(labeled_pairs) == 0:
    print("❌ ERROR: No labeled images found!")
    print("Listing all files in /kaggle/input to debug:")
    for f in list(INPUT_BASE.rglob("*"))[:20]:
        print(f"  {f}")
else:
    # Shuffle and split
    random.seed(42)
    random.shuffle(labeled_pairs)

    split = int(0.8 * len(labeled_pairs))
    train_set = labeled_pairs[:split]
    val_set = labeled_pairs[split:]

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

    print("✅ Dataset split complete!")""")

add_md("## Cell 3 — Create data.yaml")
add_code("""yaml_content = f\"\"\"path: {DATASET_DIR}
train: images/train
val: images/val

names:
  0: elephant_head
\"\"\"

yaml_path = DATASET_DIR / "data.yaml"
yaml_path.write_text(yaml_content)
print(f"✅ data.yaml created at: {yaml_path}")
print(yaml_content)""")

add_md("## Cell 4 — Train YOLOv8n Head Detector")
add_code("""model = YOLO("yolov8n.pt")

results = model.train(
    data=str(yaml_path),
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    save=True,
    project="/kaggle/working/runs",
    name="head_detect",
    exist_ok=True,
    workers=2,
)

print("\\n✅ Training complete!")
best_weights = Path("/kaggle/working/runs/head_detect/weights/best.pt")
print(f"Best weights: {best_weights}")
print(f"Exists: {best_weights.exists()}")""")

add_md("## Cell 5 — Sanity Check (Predict on Val Images)")
add_code("""best_weights = Path("/kaggle/working/runs/head_detect/weights/best.pt")
model = YOLO(str(best_weights))

val_images = list((DATASET_DIR / "images" / "val").glob("*.[jJ][pP][gG]")) + \\
             list((DATASET_DIR / "images" / "val").glob("*.[pP][nN][gG]"))

# Predict on first 10 val images
test_images = val_images[:10]

results = model.predict(
    source=[str(p) for p in test_images],
    save=True,
    project="/kaggle/working/runs",
    name="sanity_check",
    exist_ok=True,
    conf=0.25,
)

for r in results:
    boxes = r.boxes
    print(f"  {Path(r.path).name}: {len(boxes)} head(s) detected, conf={[f'{b.conf.item():.2f}' for b in boxes]}")

print(f"\\n✅ Check /kaggle/working/runs/sanity_check/ for visual results")""")

add_md("## Cell 6 — Visualize Predictions")
add_code("""import matplotlib.pyplot as plt
from PIL import Image

pred_dir = Path("/kaggle/working/runs/sanity_check")
pred_images = sorted(pred_dir.glob("*.[jJ][pP][gG]"))[:8]

if pred_images:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(pred_images):
            img = Image.open(pred_images[i])
            ax.imshow(img)
            ax.set_title(pred_images[i].name, fontsize=9)
        ax.axis('off')
    plt.suptitle("Head Detector — Sanity Check Predictions", fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("No prediction images found")""")

add_md("## Cell 7 — Download Best Weights")
add_code("""# Copy best weights to /kaggle/working for easy download
shutil.copy2(
    "/kaggle/working/runs/head_detect/weights/best.pt",
    "/kaggle/working/elephant_head_yolov8n_best.pt"
)

print("✅ Download: /kaggle/working/elephant_head_yolov8n_best.pt")
print("   → Click the 'Output' tab on the right to download")""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = r'd:\Elephant_ReIdentification\kaggle\elephant-head-detector-training.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Notebook generated: {output_path}")
