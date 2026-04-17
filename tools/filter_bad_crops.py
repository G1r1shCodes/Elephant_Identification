"""
Quality Filter for Head Crops
=============================
Scans processed_heads/ and removes obviously bad crops:
1. Too dark (mostly shadow/night)
2. Too blurry (motion blur, out of focus)
3. Wrong aspect ratio (tall/narrow = body sliver, not head)
4. Too much green (foliage, not elephant)
5. Too much red (arrow artifact dominating the crop)
6. Low edge energy (featureless — sky, blur, dark background)

Moves bad crops to processed_heads/_filtered/ for review.
Generates a montage of removed images so you can verify.
"""

import cv2
import numpy as np
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "training_heads_v6"
FILTERED_DIR = PROCESSED_DIR / "_filtered"
DEBUG_DIR = PROJECT_ROOT / "debug"

# Thresholds (conservative — only catch truly bad crops)
MIN_BRIGHTNESS = 35        # mean pixel value (0-255)
MIN_BLUR_SCORE = 8         # Laplacian variance (lowered — distant elephants are softer)
MAX_GREEN_RATIO = 0.55     # fraction of green pixels
MAX_RED_RATIO = 0.06       # fraction of red pixels (arrow artifacts)
MIN_EDGE_DENSITY = 0.005   # Canny edge fraction (lowered — elephant skin is smooth)
MAX_ASPECT_RATIO = 2.5     # height/width or width/height
MIN_SIZE = 100             # minimum dimension in pixels
MIN_ELEPHANT_GRAY_RATIO = 0.20  # minimum fraction of gray/brown elephant-like pixels

def analyze_crop(img_path):
    """Analyze a crop and return (is_good, reason)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False, "unreadable"
    
    h, w = img.shape[:2]
    
    # 1. Too small
    if min(h, w) < MIN_SIZE:
        return False, f"too_small ({w}x{h})"
    
    # 2. Bad aspect ratio (tall slivers = body, not head)
    aspect = max(h/w, w/h)
    if aspect > MAX_ASPECT_RATIO:
        return False, f"bad_aspect ({aspect:.1f})"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Too dark
    mean_brightness = float(gray.mean())
    if mean_brightness < MIN_BRIGHTNESS:
        return False, f"too_dark (brightness={mean_brightness:.0f})"
    
    # 4. Too blurry
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < MIN_BLUR_SCORE:
        return False, f"too_blurry (score={blur_score:.1f})"
    
    # 5. Too much green (foliage crop)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    green_ratio = float(green_mask.sum() / 255) / (h * w)
    if green_ratio > MAX_GREEN_RATIO:
        return False, f"too_green ({green_ratio:.2f})"
    
    # 6. Too much red (arrow artifacts dominating)
    red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_ratio = float(red_mask.sum() / 255) / (h * w)
    if red_ratio > MAX_RED_RATIO:
        return False, f"red_artifact ({red_ratio:.2f})"
    
    # 7. Low edge energy (featureless — sky, dark bg, etc)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.sum() / 255) / (h * w)
    if edge_density < MIN_EDGE_DENSITY:
        return False, f"low_edges ({edge_density:.3f})"
    
    return True, "ok"


def create_reject_montage(rejected_paths, output_path, max_images=40):
    """Create a montage of rejected images for visual review."""
    size = (150, 150)
    cols = 8
    
    samples = rejected_paths[:max_images]
    imgs = []
    for path, reason in samples:
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, size)
        # Add reason text
        cv2.putText(img, reason[:20], (5, size[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        imgs.append(img)
    
    if not imgs:
        return
    
    # Pad to fill row
    while len(imgs) % cols != 0:
        imgs.append(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    
    rows = []
    for r in range(0, len(imgs), cols):
        rows.append(np.hstack(imgs[r:r+cols]))
    montage = np.vstack(rows)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), montage)
    print(f"Reject montage saved: {output_path}")


def main():
    print("=" * 60)
    print("HEAD CROP QUALITY FILTER")
    print("=" * 60)
    
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    
    all_crops = list(PROCESSED_DIR.rglob("*.jpg"))
    # Exclude already-filtered and quarantined
    all_crops = [p for p in all_crops if "_filtered" not in str(p) and "_quarantined" not in str(p)]
    
    print(f"Total crops to check: {len(all_crops)}")
    
    good = 0
    bad = 0
    rejected = []  # (path, reason)
    reason_counts = {}
    
    for crop_path in sorted(all_crops):
        is_good, reason = analyze_crop(crop_path)
        
        if is_good:
            good += 1
        else:
            bad += 1
            rejected.append((crop_path, reason))
            
            # Move to filtered directory (preserving structure)
            rel = crop_path.relative_to(PROCESSED_DIR)
            dest = FILTERED_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(crop_path), str(dest))
            
            category = reason.split(" ")[0].split("(")[0]
            reason_counts[category] = reason_counts.get(category, 0) + 1
            
            print(f"  [REMOVE] {crop_path.name} -> {reason}")
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Good crops:    {good}")
    print(f"Removed:       {bad}")
    print(f"Survival rate: {good/(good+bad)*100:.1f}%")
    print(f"\nRemoval reasons:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:20s}: {count}")
    
    # Create montage of rejected
    if rejected:
        create_reject_montage(rejected, str(DEBUG_DIR / "rejected_crops_montage.jpg"))
    
    print(f"\nFiltered crops moved to: {FILTERED_DIR}")
    print(f"Review them — if any are actually good, move them back!")


if __name__ == "__main__":
    main()
