"""
Elephant Re-ID — End-to-End Pipeline
====================================

Image -> Head Detect -> Crop -> Embed -> Match/Unknown

Uses:
  - YOLOv8n head detector (trained)
  - ConvNeXt-Tiny embedding model (trained with MS-Loss)
  - Gallery of known identity embeddings
  - Threshold-based matching with logging

Usage:
    # Build gallery from processed_heads
    python pipeline.py --build-gallery

    # Identify a single image
    python pipeline.py --identify path/to/image.jpg

    # Batch test a folder
    python pipeline.py --test-folder path/to/folder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import math
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
import json
import argparse
import os

# ==================== CONFIG ==================== #

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"

HEAD_DETECTOR_PATH = MODELS_DIR / "elephant_head_yolov8nv2_best.pt"
GALLERY_PATH = MODELS_DIR / "gallery_embeddings.pt"
HEAD_REFERENCE_BANK_PATH = MODELS_DIR / "head_crop_reference_bank_v2.pt"


# Model preference: env override → v3 → v2 → v1
def _resolve_reid_model() -> Path:
    override = os.environ.get("ELEPHANT_REID_MODEL")
    if override:
        p = Path(override)
        if p.exists():
            return p
        raise FileNotFoundError(f"ELEPHANT_REID_MODEL override not found: {p}")
    for name in (
        "elephant_head_reid_v8.2.pth",
        "elephant_head_reid_v8.1.pth",
        "elephant_head_reid_v8_download.pth",
        "elephant_head_reid_v8.pth",
        "elephant_head_reid_v7.0 (1).pth",
        "elephant_head_reid_v3.pth",
        "elephant_head_reid_v2.pth",
        "elephant_head_reid_v1.pth",
    ):
        p = MODELS_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "No Re-ID model found. Place elephant_head_reid_v3.pth (or v2/v1) in models/"
    )


REID_MODEL_PATH = _resolve_reid_model()

PROCESSED_HEADS_DIR = PROJECT_ROOT / "data" / "training_heads_v6"

# Detection
DETECT_CONF = 0.4
PADDING_RATIO = 0.10
MIN_VALID_HEAD_CONF = 0.20
MIN_VALID_HEAD_AREA_RATIO = 0.001
MAX_VALID_HEAD_AREA_RATIO = 0.90  # raised from 0.55 to allow extreme close-ups (occupying almost the whole frame)
MIN_VALID_HEAD_ASPECT = 0.35
MAX_VALID_HEAD_ASPECT = 2.80
TILE_OVERLAP_RATIO = 0.20
MIN_CROP_EDGE = 80
CROP_BLUR_THRESHOLD = 50.0
CROP_CONTRAST_THRESHOLD = 30.0
CROP_CENTER_SAT_THRESHOLD = 95.0
HEAD_REFERENCE_TOPK = 5
HEAD_REFERENCE_MAX_IMAGES = 512
HEAD_REFERENCE_MAX_NEG_IMAGES = 256
HEAD_REFERENCE_MIN_MAX_SIM = 0.20
HEAD_REFERENCE_MIN_TOPK_SIM = 0.14
HEAD_REFERENCE_CAP_MAX_SIM = 0.66
HEAD_REFERENCE_CAP_TOPK_SIM = 0.57
HEAD_REFERENCE_MIN_MARGIN = 0.05

# Matching thresholds (open-set re-ID calibrated to hybrid score)
DIST_STRICT = 0.035  # strict distance for HIGH confidence
DIST_LOOSE = 0.075  # looser distance for MEDIUM confidence
GAP_STRICT = 0.080  # required gap for HIGH confidence (maintained strict)
GAP_LOOSE = 0.040  # minimum gap for MEDIUM confidence

EMBED_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Arrow detection
MIN_ARROW_AREA = 4000
ARROW_HSV_LOWER1 = np.array([0, 100, 100])
ARROW_HSV_UPPER1 = np.array([10, 255, 255])
ARROW_HSV_LOWER2 = np.array([160, 100, 100])
ARROW_HSV_UPPER2 = np.array([180, 255, 255])

# ==================== EMBEDDING MODEL ==================== #


class HeadEmbeddingModel(nn.Module):
    """Same architecture as training notebook."""

    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = convnext_tiny(weights=None)  # no pretrained, we load our own
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        feat = self.backbone.features(x)
        feat = self.pool(feat).flatten(1)
        emb = self.embed(feat)
        return F.normalize(emb, p=2, dim=1)


INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ==================== MODEL LOADING ==================== #

_head_detector = None
_reid_model = None
_head_reference_bank = None


def get_head_detector():
    global _head_detector
    if _head_detector is None:
        print(f"Loading head detector: {HEAD_DETECTOR_PATH.name}")
        _head_detector = YOLO(str(HEAD_DETECTOR_PATH))
    return _head_detector


def get_reid_model():
    global _reid_model, EMBED_DIM
    if _reid_model is None:
        print(f"Loading Re-ID model: {REID_MODEL_PATH.name}")
        checkpoint = torch.load(
            str(REID_MODEL_PATH), map_location=DEVICE, weights_only=False
        )

        # Determine embed_dim from checkpoint if available
        embed_dim = EMBED_DIM
        if isinstance(checkpoint, dict):
            if (
                "model_state_dict" in checkpoint
                and "embed.4.weight" in checkpoint["model_state_dict"]
            ):
                embed_dim = checkpoint["model_state_dict"]["embed.4.weight"].shape[0]
            else:
                embed_dim = checkpoint.get("embed_dim", EMBED_DIM)

        EMBED_DIM = embed_dim

        _reid_model = HeadEmbeddingModel(embed_dim=embed_dim).to(DEVICE)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            _reid_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            _reid_model.load_state_dict(checkpoint)

        _reid_model.eval()
        if isinstance(checkpoint, dict):
            print(
                f"  Separation gap from training: {checkpoint.get('separation_gap', 'N/A')}"
            )
            print(f"  Embed dim: {embed_dim}")
    return _reid_model


def _collect_head_reference_paths(max_images=HEAD_REFERENCE_MAX_IMAGES):
    image_paths = []
    if PROCESSED_HEADS_DIR.exists():
        for dirpath, dirnames, filenames in os.walk(str(PROCESSED_HEADS_DIR)):
            if any(tag in dirpath for tag in ("_filtered", "_no_head", "_quarantined")):
                continue
            dirnames[:] = [d for d in dirnames if not d.startswith("_")]
            images = sorted(
                os.path.join(dirpath, f)
                for f in filenames
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            image_paths.extend(images)

    if len(image_paths) > max_images:
        step = len(image_paths) / float(max_images)
        image_paths = [image_paths[int(i * step)] for i in range(max_images)]
    return image_paths


def _collect_head_negative_paths(max_images=HEAD_REFERENCE_MAX_NEG_IMAGES):
    image_paths = []
    for folder_name in ("_no_head_in_crop",):
        folder = PROCESSED_HEADS_DIR / folder_name
        if not folder.exists():
            continue
        for dirpath, _, filenames in os.walk(str(folder)):
            images = sorted(
                os.path.join(dirpath, f)
                for f in filenames
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            image_paths.extend(images)

    if len(image_paths) > max_images:
        step = len(image_paths) / float(max_images)
        image_paths = [image_paths[int(i * step)] for i in range(max_images)]
    return image_paths


def get_head_reference_bank(use_cache=True):
    """Lazy-load a bank of valid head embeddings for crop validation."""
    global _head_reference_bank
    if _head_reference_bank is not None:
        return _head_reference_bank

    if use_cache and HEAD_REFERENCE_BANK_PATH.exists():
        try:
            cached = torch.load(
                str(HEAD_REFERENCE_BANK_PATH), map_location="cpu", weights_only=False
            )
            if isinstance(cached, dict) and "embeddings" in cached:
                cached["embeddings"] = cached["embeddings"].float()
                _head_reference_bank = cached
                return _head_reference_bank
        except Exception:
            pass

    ref_paths = _collect_head_reference_paths()
    neg_paths = _collect_head_negative_paths()
    if not ref_paths:
        _head_reference_bank = {
            "embeddings": torch.empty((0, EMBED_DIM), dtype=torch.float32),
            "negative_embeddings": torch.empty((0, EMBED_DIM), dtype=torch.float32),
            "threshold_max": HEAD_REFERENCE_MIN_MAX_SIM,
            "threshold_topk": HEAD_REFERENCE_MIN_TOPK_SIM,
            "threshold_margin": HEAD_REFERENCE_MIN_MARGIN,
        }
        return _head_reference_bank

    print(f"Building head reference bank from {len(ref_paths)} crop(s)...")
    model = get_reid_model()
    embeddings = []
    negative_embeddings = []
    with torch.no_grad():
        for path in ref_paths:
            try:
                pil_img = Image.open(path).convert("RGB")
                tensor = INFERENCE_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
                emb = model(tensor).squeeze(0).cpu()
                embeddings.append(emb)
            except Exception:
                continue
        for path in neg_paths:
            try:
                pil_img = Image.open(path).convert("RGB")
                tensor = INFERENCE_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
                emb = model(tensor).squeeze(0).cpu()
                negative_embeddings.append(emb)
            except Exception:
                continue

    if not embeddings:
        _head_reference_bank = {
            "embeddings": torch.empty((0, EMBED_DIM), dtype=torch.float32),
            "negative_embeddings": torch.empty((0, EMBED_DIM), dtype=torch.float32),
            "threshold_max": HEAD_REFERENCE_MIN_MAX_SIM,
            "threshold_topk": HEAD_REFERENCE_MIN_TOPK_SIM,
            "threshold_margin": HEAD_REFERENCE_MIN_MARGIN,
        }
        return _head_reference_bank

    bank = torch.stack(embeddings).float()
    negative_bank = (
        torch.stack(negative_embeddings).float()
        if negative_embeddings
        else torch.empty((0, EMBED_DIM), dtype=torch.float32)
    )
    if len(bank) <= 1:
        threshold_max = HEAD_REFERENCE_MIN_MAX_SIM
        threshold_topk = HEAD_REFERENCE_MIN_TOPK_SIM
        threshold_margin = HEAD_REFERENCE_MIN_MARGIN
    else:
        sims = bank @ bank.T
        sims.fill_diagonal_(-1.0)
        max_sims = sims.max(dim=1).values.numpy()
        topk = min(HEAD_REFERENCE_TOPK, sims.shape[1] - 1)
        topk_vals = torch.topk(sims, k=topk, dim=1).values
        topk_means = topk_vals.mean(dim=1).numpy()
        threshold_max = max(
            float(np.percentile(max_sims, 5)) - 0.03,
            HEAD_REFERENCE_MIN_MAX_SIM,
        )
        threshold_topk = max(
            float(np.percentile(topk_means, 5)) - 0.03,
            HEAD_REFERENCE_MIN_TOPK_SIM,
        )
        threshold_max = min(threshold_max, HEAD_REFERENCE_CAP_MAX_SIM)
        threshold_topk = min(threshold_topk, HEAD_REFERENCE_CAP_TOPK_SIM)
        if len(negative_bank) > 0:
            pos_to_neg = torch.mm(bank, negative_bank.T).max(dim=1).values.numpy()
            margins = max_sims - pos_to_neg
            threshold_margin = max(
                float(np.percentile(margins, 5)) - 0.03,
                HEAD_REFERENCE_MIN_MARGIN,
            )
        else:
            threshold_margin = HEAD_REFERENCE_MIN_MARGIN

    _head_reference_bank = {
        "embeddings": bank,
        "negative_embeddings": negative_bank,
        "threshold_max": float(threshold_max),
        "threshold_topk": float(threshold_topk),
        "threshold_margin": float(threshold_margin),
        "count": int(len(bank)),
        "negative_count": int(len(negative_bank)),
    }
    try:
        torch.save(_head_reference_bank, str(HEAD_REFERENCE_BANK_PATH))
    except Exception:
        pass
    return _head_reference_bank


def score_head_crop_reference(crop):
    """Compare a candidate crop to the valid-head embedding manifold."""
    bank = get_head_reference_bank()
    refs = bank.get("embeddings")
    if refs is None or len(refs) == 0:
        return {
            "head_ref_max_sim": 0.0,
            "head_ref_topk_mean": 0.0,
            "head_ref_negative_max": 0.0,
            "head_ref_margin": 0.0,
            "head_ref_valid": False,
            "head_ref_hard_negative": False,
            "head_ref_threshold_max": HEAD_REFERENCE_MIN_MAX_SIM,
            "head_ref_threshold_topk": HEAD_REFERENCE_MIN_TOPK_SIM,
            "head_ref_threshold_margin": HEAD_REFERENCE_MIN_MARGIN,
        }

    model = get_reid_model()
    with torch.no_grad():
        tensor = INFERENCE_TRANSFORM(crop).unsqueeze(0).to(DEVICE)
        emb = model(tensor).squeeze(0).cpu()

    sims = torch.mv(refs, emb)
    max_sim = float(sims.max().item())
    topk = min(HEAD_REFERENCE_TOPK, len(refs))
    topk_mean = float(torch.topk(sims, k=topk).values.mean().item())
    neg_refs = bank.get("negative_embeddings")
    neg_max = 0.0
    if neg_refs is not None and len(neg_refs) > 0:
        neg_max = float(torch.mv(neg_refs, emb).max().item())
    thr_max = float(bank.get("threshold_max", HEAD_REFERENCE_MIN_MAX_SIM))
    thr_topk = float(bank.get("threshold_topk", HEAD_REFERENCE_MIN_TOPK_SIM))
    thr_margin = float(bank.get("threshold_margin", HEAD_REFERENCE_MIN_MARGIN))
    margin = max_sim - neg_max
    hard_floor_max = max(thr_max - 0.18, 0.08)
    hard_floor_topk = max(thr_topk - 0.12, 0.05)

    return {
        "head_ref_max_sim": round(max_sim, 4),
        "head_ref_topk_mean": round(topk_mean, 4),
        "head_ref_negative_max": round(neg_max, 4),
        "head_ref_margin": round(margin, 4),
        "head_ref_valid": bool(
            (max_sim >= thr_max or topk_mean >= thr_topk) and margin >= thr_margin
        ),
        "head_ref_hard_negative": bool(
            (max_sim < hard_floor_max and topk_mean < hard_floor_topk)
            or (neg_max >= max_sim and margin < 0.02)
        ),
        "head_ref_threshold_max": round(thr_max, 4),
        "head_ref_threshold_topk": round(thr_topk, 4),
        "head_ref_threshold_margin": round(thr_margin, 4),
    }


def build_head_reference_bank(force_rebuild=False):
    """Precompute and cache the valid-head reference bank used at inference."""
    global _head_reference_bank
    if force_rebuild and HEAD_REFERENCE_BANK_PATH.exists():
        try:
            HEAD_REFERENCE_BANK_PATH.unlink()
        except OSError:
            pass
        _head_reference_bank = None
    bank = get_head_reference_bank(use_cache=not force_rebuild)
    print(
        f"Head reference bank ready: {bank.get('count', 0)} crop(s) | "
        f"threshold_max={bank.get('threshold_max', 0.0):.3f} | "
        f"threshold_topk={bank.get('threshold_topk', 0.0):.3f}"
    )
    return bank


# ==================== HEAD DETECTION ==================== #


def detect_arrow(image_bgr):
    """Detect red arrow tip and direction. Returns dict or None."""
    import math
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, ARROW_HSV_LOWER1, ARROW_HSV_UPPER1)
    mask2 = cv2.inRange(hsv, ARROW_HSV_LOWER2, ARROW_HSV_UPPER2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_ARROW_AREA:
        return None

    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])

    eL = tuple(largest[largest[:, :, 0].argmin()][0])
    eR = tuple(largest[largest[:, :, 0].argmax()][0])
    eT = tuple(largest[largest[:, :, 1].argmin()][0])
    eB = tuple(largest[largest[:, :, 1].argmax()][0])

    dists = [math.hypot(x - cx, y - cy) for x, y in [eL, eR, eT, eB]]
    dirs = ["LEFT", "RIGHT", "UP", "DOWN"]
    pts = [eL, eR, eT, eB]
    idx = dists.index(max(dists))

    return {
        "tip": pts[idx],
        "direction": dirs[idx]
    }


def _is_valid_head_detection(det, w_img, h_img, arrow_tip=None):
    x1, y1, x2, y2 = det["bbox"]
    conf = float(det["conf"])
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    area_ratio = (box_w * box_h) / float(max(1, w_img * h_img))
    aspect_ratio = box_w / float(box_h)

    # If an arrow is present, be a little more permissive because the user
    # explicitly pointed to the target head.
    min_conf = 0.15 if arrow_tip is not None else MIN_VALID_HEAD_CONF

    return (
        conf >= min_conf
        and MIN_VALID_HEAD_AREA_RATIO <= area_ratio <= MAX_VALID_HEAD_AREA_RATIO
        and MIN_VALID_HEAD_ASPECT <= aspect_ratio <= MAX_VALID_HEAD_ASPECT
    )


def _score_detection_candidate(det, w_img, h_img, arrow_tip=None):
    """Rank plausible boxes instead of trusting raw confidence alone."""
    x1, y1, x2, y2 = det["bbox"]
    conf = float(det["conf"])
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    area_ratio = (box_w * box_h) / float(max(1, w_img * h_img))
    aspect_ratio = box_w / float(box_h)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    center_dx = abs(cx - (w_img / 2.0)) / max(1.0, w_img / 2.0)
    center_dy = abs(cy - (h_img / 2.0)) / max(1.0, h_img / 2.0)
    center_penalty = center_dx + center_dy

    score = 0.0
    score += conf * 2.5
    score += min(area_ratio / 0.02, 1.0) * 2.0
    score += max(0.0, 1.0 - center_penalty) * 1.5
    score += max(0.0, 1.0 - abs(aspect_ratio - 1.0)) * 0.5

    if arrow_tip is not None:
        px, py = arrow_tip
        if x1 <= px <= x2 and y1 <= py <= y2:
            score += 3.0

    return score


def crop_quality_score(crop, detection_meta=None):
    """Cheap quality heuristic to catch obviously bad crops before embedding."""
    if isinstance(crop, Image.Image):
        crop_rgb = np.array(crop.convert("RGB"))
    else:
        crop_rgb = np.asarray(crop)

    h, w = crop_rgb.shape[:2]
    if h == 0 or w == 0:
        return {"score": 0, "weak": True, "reason": "empty_crop"}

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    aspect = w / float(max(1, h))

    y0 = int(h * 0.2)
    y1 = max(y0 + 1, int(h * 0.8))
    x0 = int(w * 0.2)
    x1 = max(x0 + 1, int(w * 0.8))
    center_sat = float(hsv[y0:y1, x0:x1, 1].mean())

    score = 0
    size_ok = h >= MIN_CROP_EDGE and w >= MIN_CROP_EDGE
    if size_ok:
        score += 1
    if blur > CROP_BLUR_THRESHOLD:
        score += 1
    if contrast > CROP_CONTRAST_THRESHOLD:
        score += 1
    if 0.5 < aspect < 2.0:
        score += 1
    if center_sat < CROP_CENTER_SAT_THRESHOLD:
        score += 1

    weak = score <= 1
    low_confidence = score == 2

    if detection_meta:
        conf = float(detection_meta.get("conf", 1.0))
        area_ratio = float(detection_meta.get("area_ratio", 1.0))
        center_offset = float(detection_meta.get("center_offset", 0.0))
        selection_score = float(detection_meta.get("selection_score", 999.0))
        head_ref_max = float(detection_meta.get("head_ref_max_sim", 1.0))
        head_ref_topk = float(detection_meta.get("head_ref_topk_mean", 1.0))
        head_ref_margin = float(detection_meta.get("head_ref_margin", 1.0))
        head_ref_valid = bool(detection_meta.get("head_ref_valid", True))
        head_ref_hard_negative = bool(
            detection_meta.get("head_ref_hard_negative", False)
        )
        source = str(detection_meta.get("source", ""))
        tile_source = source.startswith("tile-")

        # Hard reject only when the detection itself is structurally suspect.
        weak = weak or (
            selection_score < 2.2
            or area_ratio < 0.0012
            or (area_ratio < 0.03 and center_offset > 0.70 and conf < 0.27)
            or (tile_source and head_ref_hard_negative)
            or (
                tile_source
                and area_ratio < 0.02
                and selection_score < 4.0
                and conf < 0.40
            )
        )

        # Keep low-confidence as a softer flag so good-but-weak boxes do not
        # get thrown away just because YOLO confidence is modest.
        low_confidence = low_confidence or (
            conf < 0.25
            or selection_score < 3.1
            or center_offset > 0.62
            or (
                tile_source
                and not head_ref_valid
                and (head_ref_max < 0.24 or head_ref_margin < 0.04)
            )
        )

    return {
        "score": int(score),
        "weak": bool(weak),
        "low_confidence": bool(low_confidence and not weak),
        "blur": round(blur, 2),
        "contrast": round(contrast, 2),
        "aspect": round(aspect, 3),
        "center_saturation": round(center_sat, 2),
        "size": [int(w), int(h)],
    }


def _detect_on_tiles(image_bgr, detector, imgsz=1280, grid_size=2, conf=0.12):
    """Run YOLO on overlapping 2x2 tiles to recover small distant heads."""
    h_img, w_img = image_bgr.shape[:2]
    tile_w = max(int(math.ceil(w_img / float(grid_size))), 1)
    tile_h = max(int(math.ceil(h_img / float(grid_size))), 1)
    step_x = max(int(tile_w * (1.0 - TILE_OVERLAP_RATIO)), 1)
    step_y = max(int(tile_h * (1.0 - TILE_OVERLAP_RATIO)), 1)

    detections = []
    for y0 in range(0, max(h_img - tile_h + 1, 1), step_y):
        for x0 in range(0, max(w_img - tile_w + 1, 1), step_x):
            y1 = min(y0 + tile_h, h_img)
            x1 = min(x0 + tile_w, w_img)
            tile = image_bgr[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            r = detector(tile, conf=conf, imgsz=imgsz, iou=0.30, verbose=False)[0]
            if len(r.boxes) == 0:
                continue
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores):
                bx1, by1, bx2, by2 = box.tolist()
                detections.append(
                    {
                        "bbox": [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0],
                        "conf": float(score),
                    }
                )

    return sorted(detections, key=lambda d: d["conf"], reverse=True)


def _bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _nms_detections(detections, iou_threshold=0.45):
    kept = []
    for det in sorted(detections, key=lambda d: d["conf"], reverse=True):
        if any(_bbox_iou(det["bbox"], prev["bbox"]) >= iou_threshold for prev in kept):
            continue
        kept.append(det)
    return kept


def _recover_tile_detections(image_bgr, detector, w_img, h_img, arrow_tip=None):
    recovered = []
    for grid_size, conf, imgsz in ((2, 0.12, 1280), (3, 0.08, 1280)):
        recovered.extend(
            _detect_on_tiles(
                image_bgr,
                detector,
                imgsz=imgsz,
                grid_size=grid_size,
                conf=conf,
            )
        )
    recovered = _nms_detections(recovered)
    return [
        det
        for det in recovered
        if _is_valid_head_detection(det, w_img, h_img, arrow_tip=arrow_tip)
    ]


def _candidate_padding_ratios(area_ratio):
    if area_ratio < 0.01:
        return [0.12, 0.20, 0.28]
    if area_ratio < 0.03:
        return [0.10, 0.16, 0.22]
    return [0.10, 0.14]


def _crop_with_padding(image_bgr, bbox, pad_ratio):
    x1, y1, x2, y2 = bbox
    h_img, w_img = image_bgr.shape[:2]

    # Enforce square crop centered on head
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    size = max(x2 - x1, y2 - y1)

    x1 = cx - size // 2
    y1 = cy - size // 2
    x2 = cx + size // 2
    y2 = cy + size // 2

    # padding based on the new size
    pad = int(pad_ratio * size)

    x1_p = max(0, x1 - pad)
    y1_p = max(0, y1 - pad)
    x2_p = min(w_img, x2 + pad)
    y2_p = min(h_img, y2 + pad)
    crop_bgr = image_bgr[y1_p:y2_p, x1_p:x2_p]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def _evaluate_detection_candidate(
    image_bgr, det, w_img, h_img, arrow_tip=None, source_tag=""
):
    x1, y1, x2, y2 = det["bbox"]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    area_ratio = (box_w * box_h) / float(max(1, w_img * h_img))
    aspect_ratio = box_w / float(box_h)
    box_cx = (x1 + x2) / 2.0
    box_cy = (y1 + y2) / 2.0
    base_selection = _score_detection_candidate(det, w_img, h_img, arrow_tip=arrow_tip)

    best_variant = None
    for pad_ratio in _candidate_padding_ratios(area_ratio):
        crop_img = _crop_with_padding(image_bgr, det["bbox"], pad_ratio)
        detection_meta = {
            "conf": float(det["conf"]),
            "area_ratio": round(area_ratio, 6),
            "aspect_ratio": round(aspect_ratio, 4),
            "center_offset": round(
                (abs(box_cx - (w_img / 2.0)) / max(1.0, w_img / 2.0))
                + (abs(box_cy - (h_img / 2.0)) / max(1.0, h_img / 2.0)),
                4,
            ),
            "selection_score": round(base_selection, 4),
            "candidate_count": 0,
            "padding_ratio": round(pad_ratio, 3),
        }
        detection_meta.update(score_head_crop_reference(crop_img))
        quality_meta = crop_quality_score(crop_img, detection_meta=detection_meta)

        total_score = base_selection
        total_score += 4.0 * float(detection_meta["head_ref_max_sim"])
        total_score += 2.0 * float(detection_meta["head_ref_topk_mean"])
        if str(source_tag).startswith("tile-"):
            total_score -= 3.5 * float(detection_meta["head_ref_negative_max"])
            total_score += 2.5 * float(detection_meta["head_ref_margin"])
        total_score += 0.25 * float(quality_meta["score"])
        if detection_meta.get("head_ref_valid"):
            total_score += 1.0
        if quality_meta.get("low_confidence"):
            total_score -= 0.5
        if quality_meta.get("weak"):
            total_score -= 3.0

        variant = {
            "bbox": det["bbox"],
            "conf": float(det["conf"]),
            "crop_img": crop_img,
            "detection_meta": detection_meta,
            "quality_meta": quality_meta,
            "total_score": float(total_score),
        }
        if best_variant is None or variant["total_score"] > best_variant["total_score"]:
            best_variant = variant

    return best_variant


def detect_and_crop_head(image_bgr, allow_fallback=True):
    """
    Run head detector on image, apply arrow selection logic, crop with padding.
    Returns (PIL Image (RGB) of crop, is_fallback).

    Uses multi-scale cascade [640, 1024, 1280] to handle full wildlife images
    where heads may occupy only 5-15% of the frame.
    """
    detector = get_head_detector()

    # ── Input validation ─────────────────────────────────────────────────────
    assert image_bgr.dtype == np.uint8, f"Expected uint8, got {image_bgr.dtype}"
    assert image_bgr.ndim == 3, f"Expected 3-channel image, got shape {image_bgr.shape}"
    h_orig, w_orig = image_bgr.shape[:2]

    # ── Multi-scale detection cascade ────────────────────────────────────────
    # YOLO was trained on tight head crops, so small heads in full images are
    # invisible at imgsz=640. Escalate resolution until we find a hit.
    SCALE_CASCADE = [640, 1024, 1280]
    results = None
    used_imgsz = None
    for imgsz in SCALE_CASCADE:
        r = detector(image_bgr, conf=0.15, imgsz=imgsz, iou=0.30, verbose=False)[0]
        if len(r.boxes) > 0:
            results = r
            used_imgsz = imgsz
            break

    if results is None or len(results.boxes) == 0:
        # Only save one debug image per failure to avoid flooding disk
        debug_path = "debug_yolo_failure.jpg"
        if not os.path.exists(debug_path):
            debug_img = detector(image_bgr, conf=0.05, imgsz=1280, verbose=False)[
                0
            ].plot()
            cv2.imwrite(debug_path, debug_img)
        print(
            f"[YOLO] No head found at full-image scales ({SCALE_CASCADE}) | img={w_orig}x{h_orig}"
        )
        arrow_info = detect_arrow(image_bgr)
        arrow_tip = arrow_info["tip"] if arrow_info else None
        detections = _recover_tile_detections(
            image_bgr,
            detector,
            w_orig,
            h_orig,
            arrow_tip=arrow_tip,
        )
        if not detections:
            if not allow_fallback:
                return None, False
            # ── 75% Fallback Center Crop ──────────────────────────────────────
            scale = 0.75
            ch, cw = int(h_orig * scale), int(w_orig * scale)
            y = (h_orig - ch) // 2
            x = (w_orig - cw) // 2
            crop_bgr = image_bgr[y : y + ch, x : x + cw]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(crop_rgb), True
        print(
            f"[YOLO] Recovered {len(detections)} tile detection(s) | img={w_orig}x{h_orig}"
        )
        results = None
        used_imgsz = f"tile-{1280}"
    else:
        max_conf = float(results.boxes.conf.max())
        print(
            f"[YOLO] {len(results.boxes)} detection(s) | max_conf={max_conf:.2f} | imgsz={used_imgsz} | img={w_orig}x{h_orig}"
        )

        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        scores = results.boxes.conf.cpu().numpy()
        detections = sorted(
            [{"bbox": list(b), "conf": float(s)} for b, s in zip(boxes, scores)],
            key=lambda d: d["conf"],
            reverse=True,
        )

        arrow_info = detect_arrow(image_bgr)
        arrow_tip = arrow_info["tip"] if arrow_info else None
        detections = [
            det
            for det in detections
            if _is_valid_head_detection(det, w_orig, h_orig, arrow_tip=arrow_tip)
        ]

    if not detections:
        print(
            f"[YOLO] Rejected all detections as invalid head crops "
            f"(conf/size/aspect) | img={w_orig}x{h_orig}"
        )
        tile_detections = _recover_tile_detections(
            image_bgr,
            detector,
            w_orig,
            h_orig,
            arrow_tip=arrow_tip,
        )
        if tile_detections:
            detections = tile_detections
            used_imgsz = f"tile-{1280}"
            print(
                f"[YOLO] Recovered {len(detections)} tile detection(s) after full-image rejection | img={w_orig}x{h_orig}"
            )
        else:
            if not allow_fallback:
                return None, False
            scale = 0.75
            ch, cw = int(h_orig * scale), int(w_orig * scale)
            y = (h_orig - ch) // 2
            x = (w_orig - cw) // 2
            crop_bgr = image_bgr[y : y + ch, x : x + cw]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(crop_rgb), True

    # Evaluate a few strong candidates using the head-reference bank so we do
    # not rely on detector confidence alone.
    ranked_detections = sorted(
        detections,
        key=lambda det: _score_detection_candidate(
            det, w_orig, h_orig, arrow_tip=arrow_tip
        ),
        reverse=True,
    )[:5]
    evaluated = [
        _evaluate_detection_candidate(
            image_bgr,
            det,
            w_orig,
            h_orig,
            arrow_tip=arrow_tip,
            source_tag=str(used_imgsz),
        )
        for det in ranked_detections
    ]
    evaluated = [item for item in evaluated if item is not None]
    if not evaluated:
        if not allow_fallback:
            return None, False
        scale = 0.75
        ch, cw = int(h_orig * scale), int(w_orig * scale)
        y = (h_orig - ch) // 2
        x = (w_orig - cw) // 2
        crop_bgr = image_bgr[y : y + ch, x : x + cw]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(crop_rgb), True

    # Arrow-guided direction-aware selection
    selected_bbox = None
    selected_variant = None
    if arrow_info is not None:
        px, py = arrow_info["tip"]
        direction = arrow_info["direction"]
        
        dir_candidates = []
        for variant in evaluated:
            x1, y1, x2, y2 = variant["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            
            valid_dir = False
            if direction == "DOWN" and cy > py:
                valid_dir = True
            elif direction == "LEFT" and cx < px:
                valid_dir = True
            elif direction == "RIGHT" and cx > px:
                valid_dir = True
            elif direction == "UP" and cy < py:
                valid_dir = True
                
            direct_hit = (x1 <= px <= x2 and y1 <= py <= y2)
            
            if valid_dir or direct_hit:
                import math
                dist = math.hypot(cx - px, cy - py)
                MIN_DIST = 80
                
                # Reject heads too close to the arrow tip (proximity bias fix)
                if direct_hit or dist > MIN_DIST:
                    dir_candidates.append((variant, dist))

        if dir_candidates:
            # Pick the candidate closest to the arrow tip (Option A) 
            # instead of highest YOLO score, to prevent adult-bias in overlapping scenes
            selected_variant = min(dir_candidates, key=lambda item: item[1])[0]
            selected_bbox = selected_variant["bbox"]

    if selected_variant is None:
        # Fallback if no arrow or no candidates survived the direction filter
        selected_variant = max(evaluated, key=lambda item: item["total_score"])
        selected_bbox = selected_variant["bbox"]

    crop_img = selected_variant["crop_img"]
    detection_meta = dict(selected_variant["detection_meta"])
    detection_meta["candidate_count"] = len(detections)
    detection_meta["source"] = str(used_imgsz)
    detection_meta["total_score"] = round(float(selected_variant["total_score"]), 4)
    crop_img.info["detection_meta"] = detection_meta
    return crop_img, False


# ==================== EMBEDDING ==================== #


def extract_embedding(pil_image):
    """Extract 256-D L2-normalized embedding from a PIL Image using TTA + Crop."""
    model = get_reid_model()

    # 1. Original
    t_orig = INFERENCE_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

    # 2. Horizontal Flip
    t_flip = torch.flip(t_orig, dims=[3])

    # 3. Zoom Crop
    w, h = pil_image.size
    zoom_box = (int(0.05 * w), int(0.05 * h), int(0.95 * w), int(0.95 * h))
    img_zoom = pil_image.crop(zoom_box)
    t_zoom = INFERENCE_TRANSFORM(img_zoom).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb_orig = model(t_orig)
        return emb_orig.squeeze().cpu()


# ==================== GALLERY ==================== #


def build_gallery():
    """Build gallery embeddings from processed_heads dataset."""
    print("=" * 60)
    print("BUILDING GALLERY FROM PROCESSED HEADS")
    print("=" * 60)

    gallery = {}  # identity_name -> list of embeddings*
    identity_folders = {}

    for dirpath, dirnames, filenames in os.walk(str(PROCESSED_HEADS_DIR)):
        if "_filtered" in dirpath or "_no_head" in dirpath or "_quarantined" in dirpath:
            continue
        images = [f for f in filenames if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if images:
            rel = os.path.relpath(dirpath, str(PROCESSED_HEADS_DIR))
            identity_folders[rel] = [os.path.join(dirpath, f) for f in images]

    total = 0
    for identity_name, img_paths in sorted(identity_folders.items()):
        embeddings = []
        for p in img_paths:
            try:
                pil_img = Image.open(p).convert("RGB")
                emb = extract_embedding(pil_img)
                embeddings.append(emb)
                total += 1
            except Exception as e:
                print(f"  [ERR] {p}: {e}")

        if embeddings:
            import torch.nn.functional as F

            # Store mean embedding + all individual embeddings
            stacked = torch.stack(embeddings)
            centroid = stacked.mean(dim=0)
            norm = torch.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm

            if len(embeddings) > 1:
                sim_matrix = stacked @ stacked.T
                idx = torch.triu_indices(
                    sim_matrix.size(0), sim_matrix.size(1), offset=1
                )
                sims_flat = sim_matrix[idx[0], idx[1]]
                intra_std = float(torch.std(sims_flat))
                min_sim = float(torch.min(sims_flat))
                mean_sim = float(torch.mean(sims_flat))
            else:
                intra_std = 0.0
                min_sim = 1.0
                mean_sim = 1.0

            # v8 Hard Pruning
            if min_sim < 0.20 or mean_sim < 0.50 or intra_std > 0.30:
                print(
                    f"  [DROP] {identity_name} (min={min_sim:.2f}, mean={mean_sim:.2f}, std={intra_std:.2f})"
                )
                continue

            if min_sim < 0.30 or intra_std > 0.20:
                status = "UNSTABLE"
            elif mean_sim < 0.60:
                status = "WEAK"
            else:
                status = "STRONG"

            # Multi-Centroid Construction
            centroids_list = []
            if len(embeddings) >= 2:
                sim_mat_np = sim_matrix.numpy()
                dist_mat = np.clip(1.0 - sim_mat_np, 0.0, 2.0)

                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=0.40,
                    metric="precomputed",
                    linkage="average",
                )
                labels = clustering.fit_predict(dist_mat)

                clusters = defaultdict(list)
                for i, lbl in enumerate(labels):
                    clusters[lbl].append(i)

                valid_clusters = []
                for lbl, idxs in clusters.items():
                    if len(idxs) >= 2:
                        c_embs = stacked[idxs]
                        c_sims = c_embs @ c_embs.T
                        idx_triu = torch.triu_indices(len(idxs), len(idxs), offset=1)
                        c_mean_sim = float(torch.mean(c_sims[idx_triu[0], idx_triu[1]]))
                        if c_mean_sim >= 0.55:
                            valid_clusters.append((len(idxs), idxs))

                # Sort by size descending, keep top 3
                valid_clusters.sort(key=lambda x: x[0], reverse=True)
                for _, idxs in valid_clusters[:3]:
                    c_emb = stacked[idxs].mean(dim=0)
                    c_emb = F.normalize(c_emb, p=2, dim=0)
                    centroids_list.append(c_emb)

            # Fallback to global mean if no valid clusters found
            if not centroids_list:
                centroids_list.append(centroid)

            centroids_tensor = torch.stack(centroids_list)

            gallery[identity_name] = {
                "mean": centroid,
                "centroids": centroids_tensor,
                "embeddings": stacked,
                "count": len(embeddings),
                "intra_std": intra_std,
                "min_sim": min_sim,
                "mean_sim": mean_sim,
                "status": status,
            }
            short = (
                identity_name.split("/")[-1] if "/" in identity_name else identity_name
            )
            print(
                f"  [OK] {short}: {len(embeddings)} images, {len(centroids_list)} centroid(s)"
            )

    # Save
    torch.save(gallery, str(GALLERY_PATH))
    print(f"\nGallery saved: {GALLERY_PATH}")
    print(f"  Identities: {len(gallery)}")
    print(f"  Total embeddings: {total}")
    return gallery


def load_gallery():
    """Load gallery from disk and append adaptive thresholds."""
    if not GALLERY_PATH.exists():
        print("No gallery found. Building...")
        return build_gallery()
    gallery = torch.load(str(GALLERY_PATH), map_location="cpu", weights_only=False)
    return gallery


# ==================== MATCHING ==================== #


def identify(image_path, gallery=None):
    """
    Full pipeline: image -> head crop -> embedding -> match.

    Returns dict:
    {
        "image": str,
        "head_found": bool,
        "identity": str or None,
        "confidence": str,  # "HIGH" / "MEDIUM" / "LOW" / "UNKNOWN"
        "distance": float,
        "top5": [(identity, distance), ...]
    }
    """
    if gallery is None:
        gallery = load_gallery()

    result = {
        "image": str(image_path),
        "head_found": False,
        "identity": None,
        "confidence": "UNKNOWN",
        "distance": float("inf"),
        "top5": [],
    }

    # 1. Load image
    import cv2

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"  [ERR] Cannot read: {image_path}")
        return result

    # 2. Detect and crop head
    head_crop = detect_and_crop_head(image_bgr)
    if head_crop is None or (isinstance(head_crop, tuple) and head_crop[0] is None):
        print(f"  [SKIP] No head detected: {Path(image_path).name}")
        return result
    if isinstance(head_crop, tuple):
        head_crop = head_crop[0]
    result["head_found"] = True

    # 3. Extract embedding
    query_emb = extract_embedding(head_crop)

    # 4. Compare against gallery (Multi-Centroid Math)
    import torch.nn.functional as F

    query_emb = F.normalize(query_emb.unsqueeze(0), p=2, dim=1).squeeze(0)

    scores_list = []
    for identity_name, data in gallery.items():
        if "centroids" in data:
            centroids = data["centroids"]
            centroids = F.normalize(centroids, p=2, dim=1)
            sims_to_centroids = torch.mv(centroids, query_emb)
            score = float(sims_to_centroids.max())
        else:
            centroid = data.get("mean")
            centroid = F.normalize(centroid, p=2, dim=0)
            score = float(torch.dot(query_emb, centroid))

        # Group Consistency Gate
        all_embs = data.get("embeddings")
        if all_embs is not None and len(all_embs) > 0:
            # limit embeddings per identity to prevent large pools from dominating
            if all_embs.shape[0] > 8:
                idx = torch.randperm(all_embs.shape[0])[:8]
                all_embs = all_embs[idx]

            sims = torch.mv(all_embs, query_emb)
            top3 = torch.sort(sims, descending=True).values[:3]

            if len(top3) > 1 and top3[0] > 0.65 and top3[1] < 0.45:
                score *= 0.7  # Single strong match rejection penalty
            elif len(top3) < 2 or float(top3.mean()) < 0.52:
                score *= 0.6  # Consistency rejection penalty

        short = identity_name.split("/")[-1] if "/" in identity_name else identity_name
        scores_list.append({"name": short, "score": score})

    # Sort by hybrid effective score descending
    scores_list.sort(key=lambda x: x["score"], reverse=True)
    result["top5"] = [(d["name"], d["score"]) for d in scores_list[:5]]

    if not scores_list:
        return result

    top = scores_list[0]
    result["distance"] = top["score"]

    # Gap between top-1 and top-2 (ignore near-zeros to prevent fake high gaps)
    valid_scores = [s["score"] for s in scores_list if s["score"] > 0.1]
    if len(valid_scores) >= 2:
        gap = valid_scores[0] - valid_scores[1]
    else:
        gap = 0.0
    result["gap"] = gap

    # 5. Gap / Decision Logic
    best_score = top["score"]

    if best_score > 0.65 and gap > 0.10:
        result["identity"] = top["name"]
        result["confidence"] = "HIGH"
    elif best_score > 0.55 and gap > 0.08:
        result["identity"] = top["name"]
        result["confidence"] = "MEDIUM"
    else:
        result["identity"] = None
        result["confidence"] = "UNKNOWN"

    result["embedding"] = query_emb.cpu().numpy()
    return result


def print_result(result):
    """Pretty-print an identification result."""
    fname = Path(result["image"]).name

    if not result["head_found"]:
        print(f"  {fname}: NO HEAD DETECTED")
        return

    conf = result["confidence"]
    dist = result["distance"]

    if conf == "HIGH":
        tag = "[MATCH]"
    elif conf == "MEDIUM":
        tag = "[REVIEW]"
    else:
        tag = "[UNKNOWN]"

    identity = result["identity"] or "---"
    gap = result.get("gap", 0.0)
    print(f"  {tag} {fname} -> {identity} (dist={dist:.4f}, gap={gap:.4f})")

    # Top 5
    for rank, (name, d) in enumerate(result["top5"], 1):
        marker = "<-" if rank == 1 else ""
        print(f"         #{rank}: {name:25s} dist={d:.4f} {marker}")


# ==================== MAIN ==================== #


def main():
    parser = argparse.ArgumentParser(description="Elephant Re-ID Pipeline")
    parser.add_argument(
        "--build-gallery",
        action="store_true",
        help="Build gallery from processed_heads",
    )
    parser.add_argument(
        "--build-head-bank",
        action="store_true",
        help="Build cached valid-head reference bank",
    )
    parser.add_argument(
        "--identify", type=str, help="Identify elephant in a single image"
    )
    parser.add_argument("--test-folder", type=str, help="Test all images in a folder")
    args = parser.parse_args()

    if args.build_gallery:
        build_gallery()
    elif args.build_head_bank:
        build_head_reference_bank(force_rebuild=True)

    elif args.identify:
        gallery = load_gallery()
        result = identify(args.identify, gallery)
        print_result(result)

    elif args.test_folder:
        gallery = load_gallery()
        folder = Path(args.test_folder)
        images = sorted(
            [
                f
                for f in folder.rglob("*")
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )

        print(f"\nTesting {len(images)} images from {folder}")
        print("=" * 60)

        stats = {"HIGH": 0, "MEDIUM": 0, "UNKNOWN": 0, "NO_HEAD": 0}
        results = []

        for img_path in images:
            result = identify(img_path, gallery)
            print_result(result)
            results.append(result)

            if not result["head_found"]:
                stats["NO_HEAD"] += 1
            else:
                stats[result["confidence"]] += 1

        # Summary
        total = len(images)
        print(f"\n{'=' * 60}")
        print(f"SUMMARY ({total} images)")
        print(f"{'=' * 60}")
        print(
            f"  HIGH confidence match:  {stats['HIGH']:3d} ({stats['HIGH'] / total * 100:.0f}%)"
        )
        print(
            f"  MEDIUM (needs review):  {stats['MEDIUM']:3d} ({stats['MEDIUM'] / total * 100:.0f}%)"
        )
        print(
            f"  UNKNOWN (new elephant): {stats['UNKNOWN']:3d} ({stats['UNKNOWN'] / total * 100:.0f}%)"
        )
        print(
            f"  No head detected:       {stats['NO_HEAD']:3d} ({stats['NO_HEAD'] / total * 100:.0f}%)"
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
