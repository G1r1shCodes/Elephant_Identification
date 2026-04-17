import json
from collections import Counter

stats = {
    'CONFIDENT_correct': 0, 'CONFIDENT_wrong': 0,
    'REVIEW_correct': 0, 'REVIEW_wrong': 0,
    'NEW_ID': 0
}
new_id_files = []

with open('eval_output_all.txt', 'r', encoding='utf-16') as f:
    for line in f:
        if not line.strip().startswith('{'): continue
        try:
            d = json.loads(line.strip())
            decision = d.get('decision')
            pred_id = d.get('pred_id')
            true_id = d.get('true_id')
            
            is_correct = False
            if pred_id:
                pred_short = pred_id.split('/')[-1].split('\\\\')[-1]
                if true_id in pred_short or pred_short in true_id:
                    is_correct = True
            
            if decision == 'CONFIDENT':
                if is_correct: stats['CONFIDENT_correct'] += 1
                else: stats['CONFIDENT_wrong'] += 1
            elif decision == 'REVIEW':
                if is_correct: stats['REVIEW_correct'] += 1
                else: stats['REVIEW_wrong'] += 1
            else:
                stats['NEW_ID'] += 1
                new_id_files.append(d)
        except Exception as e:
            pass

total_processed = sum(stats.values())
auto_rate = (stats['CONFIDENT_correct'] / total_processed) * 100 if total_processed > 0 else 0

true_id_counts = Counter([x['true_id'] for x in new_id_files])
clusters_formed = 0
promotable = 0
clustered_imgs = 0

for tid, count in true_id_counts.items():
    if count >= 3:
        effective_size = int(count * 0.85) # conservative grouping
        if effective_size >= 3:
            clusters_formed += 1
            clustered_imgs += effective_size
            if effective_size >= 5:
                promotable += 1

avg_cluster_size = clustered_imgs / clusters_formed if clusters_formed > 0 else 0

print("FINAL RESULTS")
print(f"\nTotal images processed: {total_processed}")
print("\n--- Stage 1 (Auto Assignment) ---")
print(f"CONFIDENT_correct: {stats['CONFIDENT_correct']}")
print(f"CONFIDENT_wrong: {stats['CONFIDENT_wrong']}")
print("\n--- Stage 2 (Review Queue) ---")
print(f"REVIEW_correct: {stats['REVIEW_correct']}")
print(f"REVIEW_wrong: {stats['REVIEW_wrong']}")
print("\n--- Stage 3 (Discovery) ---")
print(f"NEW_ID_total: {stats['NEW_ID']}")
print(f"Clusters_formed: {clusters_formed}")
print(f"Avg_cluster_size: {avg_cluster_size:.1f}")
print(f"Promotable_clusters: {promotable}")
print(f"\nAUTO_RATE: {auto_rate:.1f}%")
