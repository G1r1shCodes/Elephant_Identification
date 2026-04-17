import os
from ultralytics import YOLO
from pathlib import Path

def main():
    model_path = "models/elephant_head_yolov8n_best.pt"
    if not os.path.exists(model_path):
        model_path = "elephant_head_yolov8n_best.pt"

    model = YOLO(model_path)
    
    source_dir = r"C:\Users\giris\Desktop\Elephant Images"
    output_dir = r"C:\Users\giris\.gemini\antigravity\brain\1b2b72f7-750c-4585-9fa4-deb2a0c01cd0\scratch\yolo_visuals"
    os.makedirs(output_dir, exist_ok=True)
    
    images = list(Path(source_dir).glob("*.jpg")) + list(Path(source_dir).glob("*.jpeg")) + list(Path(source_dir).glob("*.JPG"))
    
    print(f"Running on {len(images)} images -> {output_dir}")
    
    results = model.predict(source=[str(p) for p in images], save=True, project=output_dir, name="predict", exist_ok=True, conf=0.15)
    print("Done")

if __name__ == "__main__":
    main()
