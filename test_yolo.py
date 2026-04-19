import cv2
from pipeline import detect_and_crop_head

img_path = r"d:\Elephant_ReIdentification\Elephant_Image_Folder_1\Makhna_17_DSCN6949.JPG"
img_bgr = cv2.imread(img_path)

if img_bgr is None:
    print("Could not load image.")
else:
    h, w = img_bgr.shape[:2]
    print(f"Original image size: {w}x{h}")
    crop_rgb, is_fallback = detect_and_crop_head(img_bgr)
    print(f"Crop returned. Fallback? {is_fallback}")
    if crop_rgb:
        print(f"Crop size: {crop_rgb.size}")
