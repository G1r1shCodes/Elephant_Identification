import sys
import cv2
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline import get_head_detector

def main():
    if len(sys.argv) < 2:
        return

    path = sys.argv[1]
    detector = get_head_detector()
    
    if os.path.isdir(path):
        images = glob.glob(os.path.join(path, "*.JPG"))[:6]
        fig, axes = plt.subplots(1, min(len(images), 6), figsize=(20, 5))
        if len(images) == 1: axes = [axes]
        
        for idx, img_path in enumerate(images):
            img_bgr = cv2.imread(img_path)
            results = detector(img_bgr, conf=0.15, imgsz=1280, iou=0.45, verbose=False)[0]
            plotted_img = results.plot()
            plotted_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
            
            axes[idx].imshow(plotted_rgb)
            axes[idx].set_title(os.path.basename(img_path))
            axes[idx].axis('off')
            
        plt.tight_layout()
        out_path = "yolo_folder_preview.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        img_bgr = cv2.imread(path)
        results = detector(img_bgr, conf=0.15, imgsz=1280, iou=0.45, verbose=False)[0]
        plotted_img = results.plot()
        cv2.imwrite("yolo_box_preview.jpg", plotted_img)
        print("Saved yolo_box_preview.jpg")

if __name__ == "__main__":
    main()
