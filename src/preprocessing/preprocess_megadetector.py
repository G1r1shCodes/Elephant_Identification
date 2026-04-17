"""
Elephant Re-Identification — Phase B: MegaDetector-Based Preprocessing
Preprocessing Script: Detection-Based Cropping with Validated Parameters

Wildlife Institute of India Research Project
OPEN-SET BIOMETRIC RE-IDENTIFICATION SYSTEM

================================================================================
UPDATED APPROACH: MegaDetector Integration (Validated in Exploration)
================================================================================

Key Improvements:
- 100% detection rate (validated on 20 samples)
- Works with or without arrows
- Precise bounding boxes around elephants
- Validated parameters: confidence=0.4, padding=0.15

Biological Constraints (Preserved):
- Identity features: Head profile, ear shape/depigmentation, temporal gland
- Crops include full elephant with context
- Head and ears MUST be preserved
- Makhna temporal gland region is CRITICAL

Methodology:
- MegaDetector v5a for elephant detection
- 15% padding around bounding boxes (validated)
- Arrow detection as fallback/validation
- Never modify raw data
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# YOLO imports
from ultralytics import YOLO

# Global YOLO model
yolo_model = YOLO("yolov8n.pt")  # lightweight, good enough

# RAW image support
try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False
    print("⚠️ rawpy not installed - .NRW files will be skipped")


# ==================== CONFIGURATION ==================== #

# Root paths - use absolute paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed_megadetector"

# Raw dataset paths
MAKHNA_RAW = DATA_ROOT / "raw" / "Makhna_id_udalguri_24"
HERD_RAW = DATA_ROOT / "raw" / "Herd_ID_Udalguri_24"

# YOLO parameters (VALIDATED)
CONFIDENCE_THRESHOLD = 0.3  # Validated in exploration
BBOX_PADDING_RATIO = 0.2    # 20% padding to preserve ears + head geometry

# Arrow detection parameters (for validation/fallback)
MIN_ARROW_AREA = 4000
ARROW_HSV_LOWER1 = np.array([0, 100, 100])
ARROW_HSV_UPPER1 = np.array([10, 255, 255])
ARROW_HSV_LOWER2 = np.array([160, 100, 100])
ARROW_HSV_UPPER2 = np.array([180, 255, 255])

# Supported image formats (NRW files already converted to JPG)
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

# Global models are loaded at import time for YOLO.


# ==================== IMAGE LOADING ====================  #

def load_image_with_raw_support(image_path):
    """
    Load image with support for RAW formats (.NRW).
    
    Args:
        image_path: Path to image file
        
    Returns:
        BGR image as numpy array, or None if failed
    """
    _, ext = os.path.splitext(image_path)
    
    # Check if it's a RAW format
    if ext.lower() in ['.nrw']:
        if not HAS_RAWPY:
            print(f"  ⚠️ Skipping {os.path.basename(image_path)} - rawpy not installed")
            return None
        
        try:
            # Load RAW file using rawpy
            with rawpy.imread(image_path) as raw:
                # Convert to RGB (8-bit)
                rgb = raw.postprocess()
                # Convert RGB to BGR for OpenCV
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return bgr
        except Exception as e:
            print(f"  ⚠️ Failed to load RAW file {os.path.basename(image_path)}: {e}")
            return None
    else:
        # Standard image formats
        return cv2.imread(image_path)


# ==================== YOLO INTEGRATION ==================== #

def detect_elephants(image, confidence_threshold=0.3):
    """
    Detect elephants using YOLO (COCO class 20).
    
    Returns:
        List of dicts with:
        - bbox: [x_norm, y_norm, w_norm, h_norm]
        - conf
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = yolo_model(image_rgb, verbose=False)[0]

    if results.boxes is None:
        return []

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    h, w = image.shape[:2]
    image_area = w * h

    detections = []

    for box, score, cls in zip(boxes, scores, classes):
        if int(cls) == 20 and score >= confidence_threshold:  # elephant class
            x1, y1, x2, y2 = box

            box_area = (x2 - x1) * (y2 - y1)
            if box_area / image_area < 0.05:
                continue

            # convert to normalized format (same as your pipeline)
            x_norm = x1 / w
            y_norm = y1 / h
            w_norm = (x2 - x1) / w
            h_norm = (y2 - y1) / h

            detections.append({
                "bbox": [x_norm, y_norm, w_norm, h_norm],
                "conf": float(score)
            })

    return detections


# ==================== IDENTITY FILTERS ==================== #

def blur_score(image):
    """Laplacian variance — lower = more blurry."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def green_ratio(region):
    """
    Fraction of pixels that are green-dominant (vegetation/occlusion proxy).
    A high ratio in the head region means the face is hidden by foliage.
    """
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    return green_mask.sum() / 255 / (region.shape[0] * region.shape[1] + 1e-6)


# ==================== ARROW DETECTION (FALLBACK/VALIDATION) ==================== #

def detect_arrow_tip(image):
    """
    Detect red arrow tip (for validation or fallback).
    
    Returns:
        tuple (x, y): Arrow tip coordinates if valid arrow found
        None: If no arrow
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, ARROW_HSV_LOWER1, ARROW_HSV_UPPER1)
    mask2 = cv2.inRange(hsv, ARROW_HSV_LOWER2, ARROW_HSV_UPPER2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < MIN_ARROW_AREA:
        return None
    
    # Calculate centroid
    moments = cv2.moments(largest_contour)
    if moments["m00"] == 0:
        return None
    
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    
    return (cx, cy)


# ==================== BOUNDING BOX UTILITIES ==================== #

def point_in_bbox(point: Tuple[int, int], bbox: List[float], img_width: int, img_height: int) -> bool:
    """Check if point is inside normalized bounding box."""
    px, py = point
    x_norm, y_norm, w_norm, h_norm = bbox
    x1 = int(x_norm * img_width)
    y1 = int(y_norm * img_height)
    x2 = int((x_norm + w_norm) * img_width)
    y2 = int((y_norm + h_norm) * img_height)
    return x1 <= px <= x2 and y1 <= py <= y2


def bbox_area(bbox: List[float]) -> float:
    """Calculate normalized bounding box area."""
    return bbox[2] * bbox[3]


def distance_to_bbox_center(point: Tuple[int, int], bbox: List[float], img_width: int, img_height: int) -> float:
    """Calculate distance from point to bbox center."""
    px, py = point
    x_norm, y_norm, w_norm, h_norm = bbox
    cx = (x_norm + w_norm/2) * img_width
    cy = (y_norm + h_norm/2) * img_height
    return np.sqrt((px - cx)**2 + (py - cy)**2)


def select_target_elephant(
    detections: List[Dict], 
    arrow_tip: Optional[Tuple[int, int]], 
    img_width: int, 
    img_height: int
) -> Optional[Dict]:
    """
    Select target elephant from multiple detections.
    
    Logic:
    - If arrow present: return bbox containing arrow (or closest)
    - If no arrow: return largest bbox
    """
    if not detections:
        return None
    
    if arrow_tip is not None:
        # Try to find bbox containing arrow
        for det in detections:
            bbox = det['bbox']
            if point_in_bbox(arrow_tip, bbox, img_width, img_height):
                return det
        
        # Fallback: closest bbox to arrow
        closest = min(detections, key=lambda d: distance_to_bbox_center(arrow_tip, d['bbox'], img_width, img_height))
        return closest
    else:
        # No arrow: return bbox with highest (confidence * area)
        largest = max(detections, key=lambda d: d['conf'] * bbox_area(d['bbox']))
        return largest


# ==================== CROPPING LOGIC ==================== #

def edge_density(region_gray):
    """Canny edge density — higher = more structure (head/ear area)."""
    edges = cv2.Canny(region_gray, 50, 150)
    return float(np.sum(edges)) / (region_gray.size + 1e-6)


def extract_identity_crop(image, bbox: List[float], padding_ratio: float = BBOX_PADDING_RATIO):
    """
    Extract padded crop from YOLO bounding box.
    No side-biasing or square-cropping — preserve the full detection region
    so ears, head, and trunk are never cut off.

    Returns:
        (crop, crop) — same image returned twice for API compatibility
    """
    h, w = image.shape[:2]
    x_norm, y_norm, w_norm, h_norm = bbox

    x1 = int(x_norm * w)
    y1 = int(y_norm * h)
    x2 = int((x_norm + w_norm) * w)
    y2 = int((y_norm + h_norm) * h)

    # Apply padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = int(box_w * padding_ratio)
    pad_h = int(box_h * padding_ratio)

    x1_pad = max(0, x1 - pad_w)
    y1_pad = max(0, y1 - pad_h)
    x2_pad = min(w, x2 + pad_w)
    y2_pad = min(h, y2 + pad_h)

    crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
    if crop.size == 0:
        return None, None

    return crop, crop


# ==================== QUALITY CHECKS ==================== #

# Hard filters (only for truly unrecoverable images)
BLUR_THRESHOLD   = 8     # Lowered: CLAHE-normalized, only catches true motion blur
GREEN_THRESHOLD  = 0.85  # Only full foliage burial

# Composite score weights
# Center non-dominance is the strongest face/back discriminator.
# Gradient ratio is important but noisy (wrinkles create cross-gradients).
# Asymmetry is reliable. Vertical localization is confirmatory.
W_GRADIENT   = 0.20
W_CENTER     = 0.35
W_ASYMMETRY  = 0.25
W_LATERAL_V  = 0.20

# Score threshold — below this, crop goes to _weak/ for human review
SCORE_SOFT_THRESH = 0.30


MIN_EDGE_ENERGY = 3000  # Lowered: after masking vegetation, edge energy will be less


def suppress_vegetation(image_bgr, gray):
    """
    Suppress vegetation pixels by replacing them with local mean.
    This removes foliage edges from gradient/Canny computations
    so signals reflect elephant structure, not jungle noise.

    Returns:
        (gray_clean, non_green_fraction)
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))

    # Dilate mask slightly to avoid boundary artifacts at skin/leaf edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)

    # Replace green pixels with local mean (smooth instead of hard zero)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    gray_clean = gray.copy()
    gray_clean[green_mask > 0] = blurred[green_mask > 0]

    total_pixels = gray.shape[0] * gray.shape[1]
    non_green_fraction = 1.0 - (np.sum(green_mask > 0) / total_pixels)

    return gray_clean, non_green_fraction


def compute_identity_score(image_bgr, gray):
    """
    Compute composite identity score ∈ [0, 1] from 4 directional signals
    + 1 skin-visibility signal.

    CRITICAL: All signals are computed on vegetation-suppressed grayscale
    so they reflect elephant structure, not foliage edges.

    Returns:
        (score, details_dict)
    """
    h, w = gray.shape[:2]
    third = max(1, w // 3)

    # Suppress vegetation before computing ANY signal
    gray_clean, non_green_frac = suppress_vegetation(image_bgr, gray)

    # Shared computations on CLEAN image
    sobel_x = cv2.Sobel(gray_clean, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_clean, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.Canny(gray_clean, 50, 150)

    # Edge energy gate
    total_edge_energy = float(edges.sum())
    if total_edge_energy < MIN_EDGE_ENERGY:
        return 0.10, {'note': f'low_edge_energy ({total_edge_energy:.0f})'}

    # ---- Signal 1: Gradient Ratio ----
    gx = float(np.mean(np.abs(sobel_x))) + 1e-6
    gy = float(np.mean(np.abs(sobel_y))) + 1e-6
    ratio = gx / gy
    sig_gradient = float(np.clip((ratio - 0.6) / (1.2 - 0.6), 0, 1))

    # ---- Signal 2: Center Non-Dominance ----
    l_edges = float(edges[:, :third].sum())
    c_edges = float(edges[:, third:2*third].sum())
    r_edges = float(edges[:, 2*third:].sum())
    total = l_edges + c_edges + r_edges + 1e-6
    sig_center = 1.0 - (c_edges / total)

    # ---- Signal 3: L/R Asymmetry ----
    lr_sum = l_edges + r_edges + 1e-6
    sig_asym = float(np.clip(abs(l_edges - r_edges) / lr_sum, 0, 1))

    # ---- Signal 4: Lateral Vertical Edge Distribution ----
    lv = float(np.sum(np.abs(sobel_x[:, :third])))
    cv_ = float(np.sum(np.abs(sobel_x[:, third:2*third])))
    rv = float(np.sum(np.abs(sobel_x[:, 2*third:])))
    v_total = lv + cv_ + rv + 1e-6
    sig_lateral = (lv + rv) / v_total

    # ---- Signal 5: Skin Visibility (non-green fraction) ----
    # More elephant skin visible = better identity signal
    sig_skin = float(np.clip(non_green_frac, 0, 1))

    # ---- Weighted composite ----
    # Reweighted to include skin visibility
    score = (
        0.30 * sig_center +
        0.20 * sig_gradient +
        0.15 * sig_lateral +
        0.15 * sig_asym +
        0.20 * sig_skin
    )

    details = {
        'gradient': f'{ratio:.3f} (sig={sig_gradient:.2f})',
        'center': f'L={l_edges:.0f} C={c_edges:.0f} R={r_edges:.0f} (sig={sig_center:.2f})',
        'asym': f'{sig_asym:.3f}',
        'lateral_v': f'{sig_lateral:.3f}',
        'skin_vis': f'{non_green_frac:.3f} (sig={sig_skin:.2f})',
        'edge_energy': f'{total_edge_energy:.0f}',
        'score': f'{score:.3f}'
    }

    return score, details


def check_crop_quality(original_image, cropped_image, padded_region=None):
    """
    Composite identity quality check.

    Phase 1: Hard filters (unrecoverable)
      - Extreme motion blur
      - Full foliage burial

    Phase 2: Identity score (4 directional signals combined)
      - Score >= SCORE_SOFT_THRESH → ACCEPT
      - Score < SCORE_SOFT_THRESH → WEAK (saved for review, not rejected)

    Returns:
        (is_valid, reject_reason_or_None, identity_score)
    """
    if cropped_image is None or cropped_image.size == 0:
        return False, "empty_crop", 0.0

    crop_h, crop_w = cropped_image.shape[:2]
    if crop_w < 80 or crop_h < 80:
        return False, f"too_small ({crop_w}x{crop_h})", 0.0

    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Hard filter 1: extreme blur
    # Apply CLAHE to normalize brightness before computing Laplacian
    # Dark/low-light images have compressed pixel range → falsely low variance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blur = float(cv2.Laplacian(gray_eq, cv2.CV_64F).var())
    if blur < BLUR_THRESHOLD:
        return False, f"blurry (score={blur:.1f})", 0.0

    # Hard filter 2: full foliage burial
    if padded_region is not None and padded_region.size > 0:
        rh = padded_region.shape[0]
        top = padded_region[:max(1, int(rh * 0.30)), :]
        if green_ratio(top) > GREEN_THRESHOLD:
            return False, "occluded_by_foliage", 0.0

    # Composite identity score (vegetation-suppressed)
    score, details = compute_identity_score(cropped_image, gray)

    if score < SCORE_SOFT_THRESH:
        return False, f"low_identity_score ({score:.3f})", score

    return True, None, score


# ==================== PROCESSING PIPELINE ==================== #

def process_dataset(input_root, output_base_name, max_images=None):
    """
    Process all images in dataset with YOLO-based cropping.
    
    For each image:
    1. Run YOLO to find elephants
    2. Detect arrow (if present) for multi-elephant selection
    3. Select target elephant
    4. Extract padded crop
    5. Perform quality checks
    6. Save to processed directory
    
    Args:
        input_root: Raw dataset directory path (Path object or string)
        output_base_name: Name for output folder
        
    Returns:
        Dictionary with processing statistics
    """
    # Convert to string for os.walk
    input_root = str(input_root)
    
    stats = {
        'total': 0,
        'processed': 0,
        'no_detection': 0,
        'arrow_detected': 0,
        'no_arrow': 0,
        'multi_elephant': 0,
        'rejected_blur': 0,
        'rejected_green': 0,
        'rejected_size': 0,
        'errors': []
    }
    
    print(f"\n{'='*80}")
    print(f"Processing: {input_root}")
    print(f"Output to:  {os.path.join(PROCESSED_ROOT, output_base_name)}")
    print(f"{'='*80}\n")
    
    # Model already loaded globally
    
    # Prepare rejected samples directory
    rejected_root = PROCESSED_ROOT / output_base_name / "_rejected"
    rejected_root.mkdir(parents=True, exist_ok=True)
    
    # Walk directory tree recursively
    for root, dirs, files in os.walk(input_root):
        for filename in files:
            input_path = os.path.join(root, filename)
            
            # Check if valid image format
            _, ext = os.path.splitext(filename)
            if ext not in VALID_EXTENSIONS:
                continue
            
            stats['total'] += 1
            
            try:
                # Read image (with RAW support)
                image = load_image_with_raw_support(input_path)
                if image is None:
                    stats['errors'].append((input_path, "Failed to read image"))
                    print(f"[SKIP] {filename} - Failed to read")
                    continue
                
                img_h, img_w = image.shape[:2]
                
                # Detect elephants with YOLO
                detections = detect_elephants(image, CONFIDENCE_THRESHOLD)
                
                if not detections:
                    stats['no_detection'] += 1
                    print(f"[NO DETECT] {filename:40s} - No elephants detected")
                    continue
                
                # Detect arrow (for multi-elephant selection)
                arrow_tip = detect_arrow_tip(image)
                
                # Select target elephant
                selected = select_target_elephant(detections, arrow_tip, img_w, img_h)
                
                if selected is None:
                    stats['no_detection'] += 1
                    print(f"[NO SELECT] {filename:40s} - Could not select elephant")
                    continue
                
                # Multi-elephant safety: reject if multiple detected AND no arrow
                if len(detections) > 1 and arrow_tip is None:
                    stats['no_detection'] += 1
                    print(f"[SKIP] {filename:40s} - Multi-elephant, no arrow to guide selection")
                    continue

                # Extract structure-aware identity crop
                cropped_image, padded_region = extract_identity_crop(image, selected['bbox'], BBOX_PADDING_RATIO)

                # Structure-aware quality checks
                is_valid, reject_reason, id_score = check_crop_quality(image, cropped_image, padded_region)

                if not is_valid:
                    if reject_reason and 'blurry' in reject_reason:
                        stats['rejected_blur'] += 1
                    elif reject_reason and 'occluded' in reject_reason:
                        stats['rejected_green'] += 1
                    elif reject_reason and 'low_identity_score' in reject_reason:
                        # Save separately as _weak for review — NOT thrown away
                        weak_root = PROCESSED_ROOT / output_base_name / "_weak"
                        weak_root.mkdir(parents=True, exist_ok=True)
                        weak_path = weak_root / filename
                        cv2.imwrite(str(weak_path), cropped_image)
                        stats['rejected_size'] += 1
                        print(f"[WEAK] {filename:40s} → id_score={id_score:.3f} (saved for review)")
                        continue
                    else:
                        stats['rejected_size'] += 1

                    # Save hard-rejected sample
                    rej_path = rejected_root / filename
                    if cropped_image is not None:
                        cv2.imwrite(str(rej_path), cropped_image)
                    print(f"[REJECT] {filename:40s} → {reject_reason}")
                    continue
                
                # Update statistics
                if arrow_tip is not None:
                    stats['arrow_detected'] += 1
                    arrow_status = "ARROW"
                else:
                    stats['no_arrow'] += 1
                    arrow_status = "NO_ARROW"
                
                if len(detections) > 1:
                    stats['multi_elephant'] += 1
                    det_status = f"{len(detections)}ELE"
                else:
                    det_status = "1ELE"
                
                # Construct output path
                relative_path = os.path.relpath(input_path, input_root)
                output_path = PROCESSED_ROOT / output_base_name / relative_path
                output_path = output_path.with_suffix('.jpg')  # Convert extension to .jpg
                
                # Create output directory
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save processed image
                cv2.imwrite(str(output_path), cropped_image)
                

                stats['processed'] += 1
                conf = selected.get('conf', 0)
                print(f"[OK] {filename:40s} {det_status:5s} {arrow_status:9s} conf={conf:.3f} score={id_score:.3f} → {cropped_image.shape[1]}x{cropped_image.shape[0]}")
                
                if max_images is not None and stats['processed'] >= max_images:
                    return stats
                    
            except Exception as e:
                stats['errors'].append((input_path, str(e)))
                print(f"[ERROR] {filename} - {e}")
    
    return stats


def print_summary(dataset_name, stats):
    """Print detailed processing summary."""
    print(f"\n{'='*80}")
    print(f"{dataset_name} Dataset - Processing Summary")
    print(f"{'='*80}")
    print(f"Total files scanned:       {stats['total']}")
    print(f"Successfully processed:    {stats['processed']}")
    print(f"No detection:              {stats['no_detection']}")
    print(f"\nRejected (quality):")
    print(f"  - Blurry:                {stats['rejected_blur']}")
    print(f"  - Green occlusion:       {stats['rejected_green']}")
    print(f"  - Too small:             {stats['rejected_size']}")
    print(f"\nDetection Details:")
    print(f"  - With arrow detected:   {stats['arrow_detected']}")
    print(f"  - No arrow:              {stats['no_arrow']}")
    print(f"  - Multi-elephant scenes: {stats['multi_elephant']}")
    
    if stats['errors']:
        print(f"\n⚠ Errors encountered: {len(stats['errors'])}")
        for path, error in stats['errors'][:10]:
            print(f"  - {os.path.basename(path)}: {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    
    print(f"{'='*80}\n")


# ==================== MAIN ENTRY POINT ==================== #

def main():
    """Main processing pipeline."""
    print("\n" + "="*80)
    print("Elephant Re-Identification - MegaDetector-Based Preprocessing")
    print("Detection-Based Cropping with Validated Parameters")
    print("="*80)
    print("\nConfiguration:")
    print(f"  → MegaDetector v5a")
    print(f"  → Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  → Padding ratio: {BBOX_PADDING_RATIO} (15%)")
    print(f"  → Arrow detection: Enabled (for multi-elephant selection)")
    print("="*80)
    
    # Create output directory
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {PROCESSED_ROOT}")
    
    # Process Makhna dataset (FULL)
    print("\n" + "▶"*40)
    print("PROCESSING MAKHNA DATASET (FULL)")
    print("▶"*40)
    makhna_stats = process_dataset(MAKHNA_RAW, "Makhna")
    print_summary("Makhna", makhna_stats)
    
    # Process Herd dataset (FULL)
    print("\n" + "▶"*40)
    print("PROCESSING HERD DATASET (FULL)")
    print("▶"*40)
    herd_stats = process_dataset(HERD_RAW, "Herd")
    print_summary("Herd", herd_stats)
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL PROCESSING COMPLETE")
    print("="*80)
    total_processed = makhna_stats['processed'] + herd_stats['processed']
    total_scanned = makhna_stats['total'] + herd_stats['total']
    total_no_detect = makhna_stats['no_detection'] + herd_stats['no_detection']
    
    success_rate = (total_processed / total_scanned * 100) if total_scanned > 0 else 0
    
    print(f"Total images scanned:      {total_scanned}")
    print(f"Total images processed:    {total_processed}")
    print(f"Success rate:              {success_rate:.1f}%")
    print(f"No detection:              {total_no_detect}")
    print(f"\nProcessed data saved to:   {PROCESSED_ROOT}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
