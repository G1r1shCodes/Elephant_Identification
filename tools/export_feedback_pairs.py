import json
import os
import sys

from review_store import ReviewStore


POSITIVE_VARIANCE_MAX = 0.15
POSITIVE_MIN_CLUSTER_SIZE = 4
POSITIVE_MIN_AVG_PAIRWISE = 0.75
HARD_NEGATIVE_MIN_SIM = 0.65


def unique_pairs(pairs):
    seen = set()
    result = []
    for left, right, _sim in pairs:
        if not left or not right or left == right:
            continue
        key = tuple(sorted((left, right)))
        if key in seen:
            continue
        seen.add(key)
        result.append([left, right])
    return result


def add_negative_pairs(pairs, query_paths, candidate):
    for query_path in query_paths:
        for sample in candidate.get("sample_pairs", []):
            sample_path = sample.get("path")
            sample_sim = sample.get("sim")
            if (
                sample_path
                and sample_path != query_path
                and sample_sim is not None
                and sample_sim > HARD_NEGATIVE_MIN_SIM
            ):
                pairs.append((query_path, sample_path, float(sample_sim)))


def add_positive_pairs(pairs, query_paths, candidate):
    if not positive_candidate_eligible(candidate):
        return
    for query_path in query_paths:
        for sample in candidate.get("sample_pairs", []):
            sample_path = sample.get("path")
            sample_sim = sample.get("sim")
            if sample_path and sample_path != query_path:
                pairs.append((query_path, sample_path, float(sample_sim or 0.0)))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def positive_candidate_eligible(candidate):
    variance = candidate.get("variance")
    cluster_size = candidate.get("cluster_size", 0)
    avg_pairwise_sim = candidate.get("avg_pairwise_sim")
    return (
        variance is not None
        and variance < POSITIVE_VARIANCE_MAX
        and cluster_size >= POSITIVE_MIN_CLUSTER_SIZE
        and avg_pairwise_sim is not None
        and avg_pairwise_sim > POSITIVE_MIN_AVG_PAIRWISE
    )


def export_pairs(base_dir):
    store = ReviewStore(base_dir)
    feedback_items = store.list_feedback_pairs()

    positives = []
    negatives = []
    used_cluster_sizes = []
    feedback_total = len(feedback_items)
    high_conf_total = 0
    risky_skipped = 0
    low_conf_skipped = 0

    for item in feedback_items:
        if item.get("risk") == "high":
            risky_skipped += 1
            continue
        if item.get("confidence") != "high":
            low_conf_skipped += 1
            continue
        high_conf_total += 1

        action = item.get("action")
        query_paths = [p for p in item.get("query_paths", []) if p]
        cand_a = item.get("candidate_a", {})
        cand_b = item.get("candidate_b", {})

        if action == "ASSIGN_A":
            add_positive_pairs(positives, query_paths, cand_a)
            add_negative_pairs(negatives, query_paths, cand_b)
            if positive_candidate_eligible(cand_a):
                used_cluster_sizes.append(cand_a.get("cluster_size", 0))
        elif action == "ASSIGN_B":
            add_positive_pairs(positives, query_paths, cand_b)
            add_negative_pairs(negatives, query_paths, cand_a)
            if positive_candidate_eligible(cand_b):
                used_cluster_sizes.append(cand_b.get("cluster_size", 0))
        elif action == "KEEP_NEW":
            add_negative_pairs(negatives, query_paths, cand_a)
            add_negative_pairs(negatives, query_paths, cand_b)
        elif action == "DISCARD":
            continue

    positive_sims = [sim for _left, _right, sim in positives]
    negative_sims = [sim for _left, _right, sim in negatives]
    payload = {
        "positives": unique_pairs(positives),
        "negatives": unique_pairs(negatives),
    }
    stats = {
        "total_feedback": feedback_total,
        "high_confidence_feedback": high_conf_total,
        "low_confidence_skipped": low_conf_skipped,
        "risky_skipped": risky_skipped,
        "used_positives": len(payload["positives"]),
        "used_negatives": len(payload["negatives"]),
        "avg_cluster_size_used": round(mean(used_cluster_sizes), 3),
        "avg_positive_sim": round(mean(positive_sims), 3),
        "avg_negative_sim": round(mean(negative_sims), 3),
        "positive_constraints": {
            "variance_max": POSITIVE_VARIANCE_MAX,
            "cluster_size_min": POSITIVE_MIN_CLUSTER_SIZE,
            "avg_pairwise_min": POSITIVE_MIN_AVG_PAIRWISE,
        },
        "negative_constraints": {
            "hard_negative_min_sim": HARD_NEGATIVE_MIN_SIM,
        },
    }

    output_path = os.path.join(base_dir, "train_pairs_v4.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    stats_path = os.path.join(base_dir, "train_pairs_v4_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return output_path, stats_path, stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_feedback_pairs.py <output_folder>")
        raise SystemExit(1)

    base_dir = sys.argv[1]
    if not os.path.isdir(base_dir):
        print(f"Output folder does not exist: {base_dir}")
        raise SystemExit(1)

    output_path, stats_path, stats = export_pairs(base_dir)
    print(f"Wrote {output_path}")
    print(f"Wrote {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
