"""
Elephant Re-ID -- Open-Set Evaluation
=====================================

Tests the FULL pipeline behavior including UNKNOWN detection.

Two test modes:
1. KNOWN test: identities IN gallery -> should be matched correctly
2. UNKNOWN test: identities EXCLUDED from gallery -> should be rejected

This is the REAL test of a re-ID system.
"""
import sys, random
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from pipeline import (
    get_reid_model, get_head_detector, extract_embedding,
    detect_and_crop_head, load_gallery, identify,
    GALLERY_PATH, PROCESSED_HEADS_DIR,
    DIST_STRICT, DIST_LOOSE, GAP_STRICT, GAP_LOOSE
)
from pathlib import Path
from collections import defaultdict
from PIL import Image
import os
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
RAW_MAKHNA = PROJECT_ROOT / "data" / "raw" / "Makhna_id_udalguri_24"
RAW_HERD = PROJECT_ROOT / "data" / "raw" / "Herd_ID_Udalguri_24"


def find_all_raw_identities():
    """Find all identity folders in raw data."""
    identities = []

    if RAW_MAKHNA.exists():
        for d in sorted(RAW_MAKHNA.iterdir()):
            if d.is_dir():
                imgs = [f for f in d.rglob("*") if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
                if len(imgs) >= 2:
                    identities.append((d.name, d, imgs))

    if RAW_HERD.exists():
        for herd_dir in sorted(RAW_HERD.iterdir()):
            if not herd_dir.is_dir():
                continue
            for cat_dir in sorted(herd_dir.iterdir()):
                if not cat_dir.is_dir():
                    continue
                for id_dir in sorted(cat_dir.iterdir()):
                    if not id_dir.is_dir():
                        continue
                    imgs = [f for f in id_dir.rglob("*") if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
                    if len(imgs) >= 2:
                        identities.append((id_dir.name, id_dir, imgs))

    return identities


def build_partial_gallery(exclude_ids):
    """Build gallery excluding certain identities (for UNKNOWN testing)."""
    full_gallery = load_gallery()

    partial = {}
    excluded_names = set()
    for key, data in full_gallery.items():
        leaf = key.replace("\\", "/").split("/")[-1]
        if leaf in exclude_ids:
            excluded_names.add(leaf)
            continue
        partial[key] = data

    print(f"Partial gallery: {len(partial)} identities (excluded {len(excluded_names)})")
    return partial


def run():
    # Load full gallery first
    full_gallery = load_gallery()

    all_ids = find_all_raw_identities()
    print(f"Found {len(all_ids)} raw identities with >= 2 images")

    # Pick 15 identities to EXCLUDE (treat as unknown)
    # Select a mix: some Makhna, some Herd, some easy, some hard
    random.seed(42)
    exclude_candidates = [name for name, _, imgs in all_ids if len(imgs) >= 3]
    exclude_set = set(random.sample(exclude_candidates, min(15, len(exclude_candidates))))

    print(f"\nHELD-OUT (UNKNOWN) identities: {sorted(exclude_set)}")

    # Build partial gallery
    partial_gallery = build_partial_gallery(exclude_set)

    # Split test data
    known_ids = [(n, d, imgs) for n, d, imgs in all_ids if n not in exclude_set]
    unknown_ids = [(n, d, imgs) for n, d, imgs in all_ids if n in exclude_set]

    print(f"\nKnown identities to test: {len(known_ids)}")
    print(f"Unknown identities to test: {len(unknown_ids)}")

    # ==================== TEST KNOWN ====================
    print("\n" + "=" * 70)
    print("TEST 1: KNOWN IDENTITIES (should match correctly)")
    print("=" * 70)

    known_total = 0
    known_detected = 0
    known_high_correct = 0
    known_high_wrong = 0
    known_med_correct = 0
    known_med_wrong = 0
    known_unknown = 0
    known_no_head = 0

    for idx, (true_id, folder, images) in enumerate(known_ids):
        for img_path in images:
            known_total += 1
            r = identify(str(img_path), partial_gallery)

            if not r["head_found"]:
                known_no_head += 1
                continue

            known_detected += 1
            predicted = r["identity"]
            conf = r["confidence"]

            is_correct = False
            if predicted:
                pred_leaf = predicted.split("\\")[-1].split("/")[-1]
                if pred_leaf == true_id:
                    is_correct = True

            if conf == "UNKNOWN":
                known_unknown += 1
            elif conf == "HIGH":
                if is_correct:
                    known_high_correct += 1
                else:
                    known_high_wrong += 1
            elif conf == "MEDIUM":
                if is_correct:
                    known_med_correct += 1
                else:
                    known_med_wrong += 1

        if (idx + 1) % 20 == 0:
            print(f"  ... processed {idx+1}/{len(known_ids)} known identities")

    # ==================== TEST UNKNOWN ====================
    print("\n" + "=" * 70)
    print("TEST 2: UNKNOWN IDENTITIES (should be rejected)")
    print("=" * 70)

    unk_total = 0
    unk_detected = 0
    unk_correctly_rejected = 0  # classified as UNKNOWN (good!)
    unk_false_high = 0           # classified as HIGH (very bad!)
    unk_false_medium = 0         # classified as MEDIUM (bad)
    unk_no_head = 0

    for idx, (true_id, folder, images) in enumerate(unknown_ids):
        for img_path in images:
            unk_total += 1
            r = identify(str(img_path), partial_gallery)

            if not r["head_found"]:
                unk_no_head += 1
                continue

            unk_detected += 1
            conf = r["confidence"]

            if conf == "UNKNOWN":
                unk_correctly_rejected += 1
            elif conf == "HIGH":
                unk_false_high += 1
                print(f"  [FALSE HIGH] {img_path.name} -> {r['identity']} (dist={r['distance']:.4f})")
            elif conf == "MEDIUM":
                unk_false_medium += 1

    # ==================== REPORT ====================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- KNOWN IDENTITIES ---")
    print(f"Total images:        {known_total}")
    print(f"Head detected:       {known_detected} ({known_detected*100//max(1,known_total)}%)")
    print(f"HIGH correct:        {known_high_correct}")
    print(f"HIGH wrong:          {known_high_wrong}")
    print(f"MEDIUM correct:      {known_med_correct}")
    print(f"MEDIUM wrong:        {known_med_wrong}")
    print(f"Rejected (UNKNOWN):  {known_unknown}  (false reject)")
    known_correct = known_high_correct + known_med_correct
    known_match_acc = known_correct * 100 / max(1, known_detected)
    false_reject = known_unknown * 100 / max(1, known_detected)
    print(f"Match accuracy:      {known_match_acc:.1f}%")
    print(f"False reject rate:   {false_reject:.1f}%")

    print("\n--- UNKNOWN IDENTITIES ---")
    print(f"Total images:        {unk_total}")
    print(f"Head detected:       {unk_detected} ({unk_detected*100//max(1,unk_total)}%)")
    print(f"Correctly rejected:  {unk_correctly_rejected}")
    print(f"False HIGH:          {unk_false_high}  <-- DANGEROUS")
    print(f"False MEDIUM:        {unk_false_medium}")
    rejection_rate = unk_correctly_rejected * 100 / max(1, unk_detected)
    print(f"Rejection rate:      {rejection_rate:.1f}%")

    print("\n--- SYSTEM QUALITY ---")
    high_precision = known_high_correct * 100 / max(1, known_high_correct + known_high_wrong + unk_false_high)
    print(f"HIGH precision:      {high_precision:.1f}%  (of all HIGH predictions, how many correct)")
    print(f"Known match acc:     {known_match_acc:.1f}%")
    print(f"Unknown rejection:   {rejection_rate:.1f}%")

    # Ideal targets
    print("\n--- vs TARGETS ---")
    print(f"HIGH precision:      {high_precision:.1f}% / target 95%+  {'PASS' if high_precision >= 95 else 'FAIL'}")
    print(f"Unknown rejection:   {rejection_rate:.1f}% / target 70%+  {'PASS' if rejection_rate >= 70 else 'FAIL'}")
    print(f"False reject:        {false_reject:.1f}% / target <15%   {'PASS' if false_reject < 15 else 'FAIL'}")


if __name__ == "__main__":
    run()
