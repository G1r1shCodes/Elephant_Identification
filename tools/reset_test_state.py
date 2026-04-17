"""
tools/reset_test_state.py
─────────────────────────
One-shot helper to wipe Unknown_* entries from the gallery and clear
an output folder before a fresh classification test.

Usage:
    python tools/reset_test_state.py --output "C:\\Users\\giris\\Desktop\\Elephant Output"
"""

import argparse, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_engine import ElephantEngine


def main():
    parser = argparse.ArgumentParser(description="Reset test state for a clean classification run")
    parser.add_argument("--output", required=False,
                        help="Output folder to clear (removes Unknown_* subfolders + cluster JSON)")
    args = parser.parse_args()

    # 1. Clean gallery Unknown_* entries
    engine = ElephantEngine()
    removed = [k for k in list(engine.gallery.keys()) if k.startswith("Unknown_")]
    if removed:
        for k in removed:
            del engine.gallery[k]
        engine._save_gallery_with_backup()
        print(f"Gallery: removed {removed}")
    else:
        print("Gallery: no Unknown_* entries found, already clean.")

    print(f"Gallery: {len(engine.gallery)} named entries remain.")

    # 2. Optionally clear the output folder
    if args.output and os.path.isdir(args.output):
        deleted = []
        for name in os.listdir(args.output):
            p = os.path.join(args.output, name)
            if os.path.isdir(p) and name.startswith("Unknown_"):
                shutil.rmtree(p)
                deleted.append(name)
        # Always remove cluster JSON so fresh run starts from scratch
        cjson = os.path.join(args.output, "unknown_clusters.json")
        if os.path.exists(cjson):
            os.remove(cjson)
            deleted.append("unknown_clusters.json")
        if deleted:
            print(f"Output folder: removed {deleted}")
        else:
            print("Output folder: nothing to remove.")
    else:
        print("Output folder: not specified or doesn't exist — skipping.")

    print("\nDone. Ready for a clean classification run.")


if __name__ == "__main__":
    main()
