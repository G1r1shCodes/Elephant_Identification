"""
Elephant Re-Identification — Direct Head Detection Preprocessing
==============================================================

This script serves as the sole ROI extractor, replacing MegaDetector.
It runs the custom YOLOv8n Head Detector directly on the raw image,
uses an arrow-based target selection logic to pick the correct elephant,
and saves the correctly padded head crops for the embedding model.

Key Improvements:
- MegaDetector completely removed (pure head detection pipeline)
- Proper arrow priority logic (direct hit -> closest -> fallback)
- Strict padding boundaries (8-12%) so ears are preserved without noise
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# YOLO imports
from ultralytics import YOLO

# RAW image support
try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False
    print("⚠️ rawpy not installed - .NRW files will be skipped")

# ==================== CONFIGURATION ==================== #

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Update output to a new directory
PROCESSED_ROOT = DATA_ROOT / "processed_heads"
MAKHNA_RAW = DATA_ROOT / "raw" / "Makhna_id_udalguri_24"
HERD_RAW = DATA_ROOT / "raw" / "Herd_ID_Udalguri_24"

# Custom YOLO model replacing MegaDetector
MODEL_PATH = PROJECT_ROOT / "models" / "elephant_head_yolov8n_best.pt"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please download it from Kaggle!")
print(f"Loading custom head detector: {MODEL_PATH.name}...")
yolo_model = YOLO(str(MODEL_PATH))

# Detection Parameters
CONF_THRESHOLD = 0.4
PADDING_RATIO = 0.10  # 10% padding (range 0.08-0.12)

# Arrow Parameters
MIN_ARROW_AREA = 4000
ARROW_HSV_LOWER1 = np.array([0, 100, 100])
ARROW_HSV_UPPER1 = np.array([10, 255, 255])
ARROW_HSV_LOWER2 = np.array([160, 100, 100])
ARROW_HSV_UPPER2 = np.array([180, 255, 255])

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

# ==================== IMAGE LOADING ==================== #

def load_image_with_raw_support(image_path):
    _, ext = os.path.splitext(image_path)
    if ext.lower() in ['.nrw']:
        if not HAS_RAWPY:
            return None
        try:
            with rawpy.imread(image_path) as raw:
                rgb = raw.postprocess()
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return cv2.imread(image_path)

# ==================== ARROW DETECTION ==================== #

def detect_arrow_tip(image):
    """Detect red arrow tip. Returns (x, y) or None."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, ARROW_HSV_LOWER1, ARROW_HSV_UPPER1)
    mask2 = cv2.inRange(hsv, ARROW_HSV_LOWER2, ARROW_HSV_UPPER2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < MIN_ARROW_AREA:
        return None
    
    moments = cv2.moments(largest_contour)
    if moments["m00"] == 0:
        return None
    
    return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

# ==================== DETECTION & SELECTION ==================== #

def point_in_bbox(point: Tuple[int, int], bbox: List[int]) -> bool:
    """Check if (x,y) point is inside bounding box [x1, y1, x2, y2]."""
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2

def box_center_distance(point: Tuple[int, int], bbox: List[int]) -> float:
    """Calculate Euclidean distance from point to bbox center."""
    px, py = point
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return np.sqrt((px - cx)**2 + (py - cy)**2)

def detect_head(image):
    """Run head detector and return best bounding box based on arrow logic."""
    results = yolo_model(image, conf=CONF_THRESHOLD, verbose=False)[0]
    
    if len(results.boxes) == 0:
        return None
        
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    scores = results.boxes.conf.cpu().numpy()
    
    # Store as list of dicts for easy sorting
    detections = [{"bbox": list(b), "conf": float(s)} for b, s in zip(boxes, scores)]
    
    # Pre-filter logic: Sort by confidence descending
    detections = sorted(detections, key=lambda d: d['conf'], reverse=True)
    
    arrow_tip = detect_arrow_tip(image)
    
    # 1. Direct hit (best)
    if arrow_tip is not None:
        for det in detections:
            if point_in_bbox(arrow_tip, det['bbox']):
                return det['bbox']
                
        # 2. Closest center if arrow exists but misses the box
        closest = min(detections, key=lambda d: box_center_distance(arrow_tip, d['bbox']))
        return closest['bbox']
        
    # 3. Fallback (no arrow) -> highest confidence
    return detections[0]['bbox']

# ==================== CROPPING ==================== #

def crop_head(image, bbox):
    """Extract crop with 10% padding (maintaining ears)."""
    x1, y1, x2, y2 = bbox
    h_img, w_img = image.shape[:2]
    
    w = x2 - x1
    h = y2 - y1
    
    pad = int(PADDING_RATIO * max(w, h))
    
    x1_pad = max(0, x1 - pad)
    y1_pad = max(0, y1 - pad)
    x2_pad = min(w_img, x2 + pad)
    y2_pad = min(h_img, y2 + pad)
    
    crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
    return crop

# ==================== PROCESSING PIPELINE ==================== #

def process_dataset(input_root, output_base_name):
    input_root = str(input_root)
    output_dir = PROCESSED_ROOT / output_base_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {output_base_name} Dataset...")
    processed_count = 0
    
    for root, _, files in os.walk(input_root):
        for filename in files:
            input_path = os.path.join(root, filename)
            _, ext = os.path.splitext(filename)
            if ext not in VALID_EXTENSIONS:
                continue
                
            image = load_image_with_raw_support(input_path)
            if image is None:
                continue
                
            bbox = detect_head(image)
            
            if bbox is None:
                print(f"[SKIP] {filename} (No heads detected)")
                continue
                
            crop = crop_head(image, bbox)
            
            # Save crop
            rel_path = os.path.relpath(input_path, input_root)
            out_path = output_dir / rel_path
            out_path = out_path.with_suffix(".jpg")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(out_path), crop)
            processed_count += 1
            print(f"[OK] {filename} -> {crop.shape[1]}x{crop.shape[0]}")
            
    return processed_count

def main():
    print(f"Direct Head Detection Pipeline Initialized")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Conf Threshold: {CONF_THRESHOLD}")
    print(f"Padding: {int(PADDING_RATIO*100)}%")
    print("="*60)
    
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    
    makhna_count = process_dataset(MAKHNA_RAW, "Makhna")
    herd_count = process_dataset(HERD_RAW, "Herd")
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Makhna processed: {makhna_count}")
    print(f"Herd processed:   {herd_count}")
    print(f"Total processed:  {makhna_count + herd_count}")

if __name__ == "__main__":
    main()
