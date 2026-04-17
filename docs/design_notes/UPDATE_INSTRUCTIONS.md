# Instructions: Update Your Notebook to Save Visualizations

## Step 1: Add Output Directory Setup
**In Cell 2 (Configuration & Paths)**, add these lines at the end:

```python
# Visualization output directory
VIZ_OUTPUT = PROJECT_ROOT / "visualization_outputs"
VIZ_OUTPUT.mkdir(exist_ok=True)

print(f"✓ Visualizations will be saved to: {VIZ_OUTPUT.absolute()}")
```

---

## Step 2: Update Cell 4.4 (Visualization Functions)

**Replace the entire visualization cell with:**

```python
def visualize_detection_result(
    image: np.ndarray,
    detections: List[Dict],
    arrow_tip: Optional[Tuple[int, int]],
    selected_detection: Optional[Dict],
    image_path: str
):
    """Visualize detection results - SAVES to file"""
    img_display = image.copy()
    h, w = img_display.shape[:2]
    
    # Draw all detections in yellow
    for det in detections:
        bbox = det['bbox']
        conf = det.get('conf', 0)
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int((bbox[0] + bbox[2]) * w)
        y2 = int((bbox[1] + bbox[3]) * h)
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img_display, f"{conf:.2f}", (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Draw selected detection in green
    if selected_detection:
        bbox = selected_detection['bbox']
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int((bbox[0] + bbox[2]) * w)
        y2 = int((bbox[1] + bbox[3]) * h)
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(img_display, "SELECTED", (x1, y1-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw arrow tip in red
    if arrow_tip:
        cv2.circle(img_display, arrow_tip, 10, (0, 0, 255), -1)
        cv2.putText(img_display, "ARROW", (arrow_tip[0]+15, arrow_tip[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Create and SAVE figure
    plt.figure(figsize=(18, 12))
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title(f"{Path(image_path).name} | Detections: {len(detections)} | Arrow: {arrow_tip is not None}", 
             fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    # Save instead of show
    save_path = VIZ_OUTPUT / f"detection_{Path(image_path).stem}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {save_path.name}")

def visualize_crop_comparison(
    image: np.ndarray,
    selected_detection: Dict,
    padding_ratios: List[float]
):
    """Compare crops - SAVES to file"""
    h, w = image.shape[:2]
    bbox = selected_detection['bbox']
    
    fig, axes = plt.subplots(1, len(padding_ratios), figsize=(18, 6))
    if len(padding_ratios) == 1:
        axes = [axes]
    
    for ax, padding in zip(axes, padding_ratios):
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int((bbox[0] + bbox[2]) * w)
        y2 = int((bbox[1] + bbox[3]) * h)
        
        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)
        
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(w, x2 + pad_w)
        y2_pad = min(h, y2 + pad_h)
        
        crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Padding: {padding:.0%}", fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    save_path = VIZ_OUTPUT / f"crop_comparison_{Path(image_path).stem}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {save_path.name}")

print("✓ Visualization functions defined (save mode)")
```

---

## Step 3: Update Cell 7 (Threshold Tuning)

**Replace the plot section with:**

```python
# Visualize threshold impact
plt.figure(figsize=(12, 6))
for threshold in CONFIDENCE_THRESHOLDS:
    counts = threshold_stats[threshold]
    plt.plot(range(len(counts)), counts, marker='o', label=f"Threshold {threshold}")

plt.xlabel("Image Index", fontsize=12)
plt.ylabel("Number of Detections", fontsize=12)
plt.title("Detection Count vs Confidence Threshold", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# SAVE instead of show
save_path = VIZ_OUTPUT / "threshold_comparison.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"💾 Saved threshold plot to: {save_path.name}")
```

---

## Step 4: Run the Notebook

After making these changes:
1. **Restart the kernel** (Kernel → Restart & Run All)
2. **Check the folder**: `Elephant_ReIdentification/visualization_outputs/`
3. **Open the PNG files** to view your visualizations!

---

## Expected Output Files:
- `detection_DSCN3544.png` (and other image names)
- `crop_comparison_*.png`
- `threshold_comparison.png`

All images will be saved as high-quality PNG files that you can open with any image viewer.
