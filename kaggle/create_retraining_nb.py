import json

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def add_code(text):
    lines = [line + '\n' for line in text.split('\n')]
    if lines: lines[-1] = lines[-1].rstrip('\n')
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines})

add_md("""# 🐘 Elephant Head Detector — Retraining (v2)

**Based on recent visual analysis, the model failed into 3 categories:**
1. **Missed Occluded Heads (`a3 (8)`):** Head behind a tree was missed.
2. **Hallucinated Heads (`a3_1 (10)`):** Saw an elephant body without a head and hallucinated one, or missed the tiny head.
3. **Multiple Background Heads (`a2 (66)`):** Detected multiple heads, causing the Re-ID pipeline to pick the wrong one.

### 🎯 Labeling Suggestions BEFORE Running This:
*   **Occlusions:** Deliberately label heads even if a tree trunk or branch covers part of them.
*   **Negative Samples:** Add 10-20 images of pure background or elephant bodies with **no head visible**. Create an empty `.txt` file for these images in labelImg. This heavily reduces hallucinations!
*   **Background Elephants:** Label ALL visible elephant heads in an image, not just the foreground one.

### 🛠️ YOLO Hyperparameter Upgrades (Included Below):
*   `imgsz=1024` (increased from 640) to detect smaller features better.
*   `epochs=100` for longer convergence instead of 50.
*   `mosaic=1.0` (default) and `mixup=0.1` to help with occlusions.
""")

add_md("## Cell 1 — Install & Imports")
add_code("""!pip install ultralytics -q

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

print("Setup complete!")""")

add_md("## Cell 2 — Find & Split Labeled Data (80/20)")
add_code("""INPUT_BASE = Path("/kaggle/input")
DATASET_DIR = Path("/kaggle/working/dataset")

all_txt_files = sorted(INPUT_BASE.rglob("*.txt"))
ignored_names = ['classes.txt', 'predefined_classes.txt', 'notes.txt']
label_files = [f for f in all_txt_files if f.name.lower() not in ignored_names]

# Include empty text files! These are critical negative samples.
labeled_pairs = []
for label_path in label_files:
    stem = label_path.stem
    parent = label_path.parent
    for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
        img_path = parent / (stem + ext)
        if img_path.exists():
            labeled_pairs.append((img_path, label_path))
            break

print(f"Found {len(labeled_pairs)} labeled image-label pairs (including negative background samples)")

random.seed(42)
random.shuffle(labeled_pairs)

split = int(0.8 * len(labeled_pairs))
train_set = labeled_pairs[:split]
val_set = labeled_pairs[split:]

for subset in ['train', 'val']:
    (DATASET_DIR / "images" / subset).mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / subset).mkdir(parents=True, exist_ok=True)

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
print(yaml_content)""")

add_md("## Cell 4 — Train YOLOv8n Head Detector (Upgraded)")
add_code("""model = YOLO("yolov8n.pt")

results = model.train(
    data=str(yaml_path),
    epochs=100,             # Increased from 50
    imgsz=1024,             # High resolution for small/occluded heads
    batch=8,                # Lower batch size since imgsz is higher
    patience=15,            # Wait longer for improvements
    save=True,
    project="/kaggle/working/runs",
    name="head_detect_v2",
    exist_ok=True,
    workers=2,
    mixup=0.1,              # Added augmentation for occlusion robustness
)

print("\\n✅ Training complete!")""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {"accelerator": "gpu", "dataSources": [], "isGpuEnabled": True, "isInternetEnabled": True}
    },
    "nbformat": 4, "nbformat_minor": 4
}

output_path = r'd:\\Elephant_ReIdentification\\kaggle\\elephant-head-detector-retraining-v2.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f"Notebook generated: {output_path}")
