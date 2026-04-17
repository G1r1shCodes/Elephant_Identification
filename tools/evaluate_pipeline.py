"""
Fast re-evaluation: recomputes metrics using the new gap-based thresholds
by re-running only on a representative subset (5 identities from each group).
Runs much faster than full evaluation.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline import load_gallery, identify
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
RAW_MAKHNA = PROJECT_ROOT / "data" / "raw" / "Makhna_id_udalguri_24"
RAW_HERD = PROJECT_ROOT / "data" / "raw" / "Herd_ID_Udalguri_24"

def find_test_identities():
    """Find all identity folders in raw data."""
    identities = []
    
    if RAW_MAKHNA.exists():
        for d in sorted(RAW_MAKHNA.iterdir()):
            if d.is_dir():
                imgs = [f for f in d.rglob("*") if f.suffix.lower() in {'.jpg','.jpeg','.png'}]
                if imgs:
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
                    imgs = [f for f in id_dir.rglob("*") if f.suffix.lower() in {'.jpg','.jpeg','.png'}]
                    if imgs:
                        name = f"{herd_dir.name}/{cat_dir.name}/{id_dir.name}"
                        identities.append((name, id_dir, imgs))
    
    return identities


def run():
    gallery = load_gallery()
    identities = find_test_identities()
    
    print("=" * 70)
    print("ELEPHANT RE-ID PIPELINE -- FULL EVALUATION (gap-based thresholds)")
    print("=" * 70)
    print(f"Gallery: {len(gallery)} identities")
    print(f"Test set: {len(identities)} identity folders")
    print(f"Thresholds: DIST < 0.30 AND GAP > 0.10 for HIGH")
    print()
    
    total_images = 0
    total_detected = 0
    total_correct_high = 0
    total_correct_medium = 0
    total_wrong_high = 0
    total_wrong_medium = 0
    total_unknown = 0
    total_no_head = 0
    
    confusion_pairs = defaultdict(int)
    
    for idx, (true_id, folder, images) in enumerate(identities):
        leaf_id = folder.name
        
        for img_path in images:
            total_images += 1
            r = identify(str(img_path), gallery)
            
            if not r["head_found"]:
                total_no_head += 1
                continue
            
            total_detected += 1
            predicted = r["identity"]
            dist = r["distance"]
            conf = r["confidence"]
            
            is_correct = False
            if predicted:
                pred_leaf = predicted.split("\\")[-1].split("/")[-1]
                if pred_leaf == leaf_id:
                    is_correct = True
            
            if conf == "UNKNOWN":
                total_unknown += 1
            elif conf == "HIGH":
                if is_correct:
                    total_correct_high += 1
                else:
                    total_wrong_high += 1
                    pred_leaf = predicted.split("\\")[-1].split("/")[-1] if predicted else "???"
                    confusion_pairs[(leaf_id, pred_leaf)] += 1
            elif conf == "MEDIUM":
                if is_correct:
                    total_correct_medium += 1
                else:
                    total_wrong_medium += 1
                    pred_leaf = predicted.split("\\")[-1].split("/")[-1] if predicted else "???"
                    confusion_pairs[(leaf_id, pred_leaf)] += 1
        
        # Progress
        if (idx + 1) % 10 == 0:
            print(f"  ... processed {idx+1}/{len(identities)} identities ({total_images} images so far)")
    
    # Report
    total_correct = total_correct_high + total_correct_medium
    total_wrong = total_wrong_high + total_wrong_medium
    
    print()
    print("=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    print(f"Total images:            {total_images}")
    print(f"Head detected:           {total_detected} ({total_detected*100//max(1,total_images)}%)")
    print(f"No head:                 {total_no_head} ({total_no_head*100//max(1,total_images)}%)")
    print()
    print(f"Of detected images:")
    print(f"  HIGH correct match:    {total_correct_high}")
    print(f"  HIGH wrong match:      {total_wrong_high}  <-- DANGEROUS")
    print(f"  MEDIUM correct:        {total_correct_medium}")
    print(f"  MEDIUM wrong:          {total_wrong_medium}")
    print(f"  UNKNOWN (rejected):    {total_unknown}")
    print()
    
    pipeline_acc = total_correct * 100 / max(1, total_images)
    match_acc = total_correct * 100 / max(1, total_detected)
    false_high = total_wrong_high * 100 / max(1, total_correct_high + total_wrong_high)
    reject_rate = total_unknown * 100 / max(1, total_detected)
    
    print(f"Pipeline accuracy:       {pipeline_acc:.1f}%")
    print(f"Match accuracy (det):    {match_acc:.1f}%")
    print(f"FALSE HIGH rate:         {false_high:.1f}%  (wrong but confident)")
    print(f"Rejection (unknown) rate:{reject_rate:.1f}%")
    
    if confusion_pairs:
        print()
        print("=" * 70)
        print("TOP CONFUSION PAIRS (wrong matches)")
        print("=" * 70)
        sorted_conf = sorted(confusion_pairs.items(), key=lambda x: -x[1])
        for (true_id, pred_id), count in sorted_conf[:15]:
            print(f"  {true_id:25s} -> {pred_id:25s} ({count}x)")
    
    # Save
    with open("debug/evaluation_report.txt", "w") as f:
        f.write(f"Pipeline accuracy: {pipeline_acc:.1f}%\n")
        f.write(f"Match accuracy: {match_acc:.1f}%\n")
        f.write(f"False HIGH rate: {false_high:.1f}%\n")
        f.write(f"Rejection rate: {reject_rate:.1f}%\n")
        f.write(f"Total: {total_images}, Detected: {total_detected}\n")
        f.write(f"Correct HIGH: {total_correct_high}, Wrong HIGH: {total_wrong_high}\n")
        f.write(f"Correct MEDIUM: {total_correct_medium}, Wrong MEDIUM: {total_wrong_medium}\n")
        f.write(f"Unknown: {total_unknown}, No head: {total_no_head}\n")
    
    print(f"\nReport saved to debug/evaluation_report.txt")


if __name__ == "__main__":
    run()
