import json
import os
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_pairs(title, pairs, limit):
    print(title)
    if not pairs:
        print("  (none)")
        return
    for idx, pair in enumerate(pairs[:limit], start=1):
        left, right = pair
        print(f"  {idx}. {left}  <->  {right}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_train_pairs.py <output_folder> [limit]")
        raise SystemExit(1)

    base_dir = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    pairs_path = os.path.join(base_dir, "train_pairs_v4.json")
    stats_path = os.path.join(base_dir, "train_pairs_v4_stats.json")

    if not os.path.exists(pairs_path) or not os.path.exists(stats_path):
        print("Missing exported pair files. Run export_feedback_pairs.py first.")
        raise SystemExit(1)

    pairs = load_json(pairs_path)
    stats = load_json(stats_path)

    print("Stats")
    print(json.dumps(stats, indent=2))
    print()
    print_pairs("Positive Pairs", pairs.get("positives", []), limit)
    print()
    print_pairs("Negative Pairs", pairs.get("negatives", []), limit)


if __name__ == "__main__":
    main()
