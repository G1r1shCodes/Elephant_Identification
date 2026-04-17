"""
tools/umap_visualizer.py — Elephant ReID Cluster Embedding Visualizer

Plots a UMAP projection of:
  - Cluster sample embeddings (Unknown clusters)
  - Known gallery identity centroids (optional)

Usage:
    python tools/umap_visualizer.py [output_folder] [--no-gallery]

Output:
    output/umap_<timestamp>.png

Requirements:
    pip install umap-learn matplotlib
"""

import sys, os, argparse, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import umap
except ImportError:
    print("ERROR: umap-learn not installed. Run: pip install umap-learn")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

import torch
import torch.nn.functional as F
import numpy as np


# ── Argument parsing ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="UMAP cluster visualizer")
parser.add_argument(
    "output_folder",
    nargs="?",
    default=None,
    help="Path to the classification output folder containing unknown_clusters.json"
)
parser.add_argument(
    "--no-gallery", action="store_true",
    help="Skip plotting known gallery centroids"
)
parser.add_argument(
    "--save-dir", default=None,
    help="Directory to save the plot (default: <output_folder>/umap_plots/)"
)
args = parser.parse_args()

# ── Locate files ────────────────────────────────────────────────────────────────
if args.output_folder:
    output_folder = args.output_folder
else:
    output_folder = input("Enter classification output folder path: ").strip()

cluster_file  = os.path.join(output_folder, "unknown_clusters.json")
gallery_path  = os.path.join(
    os.path.dirname(os.path.dirname(cluster_file)),
    "models", "gallery_embeddings.pt"
)

if not os.path.exists(cluster_file):
    print(f"ERROR: No cluster file found at {cluster_file}")
    sys.exit(1)

# ── Load cluster samples ─────────────────────────────────────────────────────────
with open(cluster_file) as f:
    cluster_data = json.load(f)

if not cluster_data:
    print("No clusters found in cluster file.")
    sys.exit(0)

all_embeddings = []
all_labels     = []
all_stability  = []  # True = flagged unstable

for name, info in cluster_data.items():
    samples = info.get("samples", [])
    flagged = info.get("stability_flag", False)
    for s in samples:
        v = torch.tensor(s, dtype=torch.float32)
        v = F.normalize(v, p=2, dim=0)
        all_embeddings.append(v.numpy())
        all_labels.append(name)
        all_stability.append(flagged)

print(f"Loaded {len(all_embeddings)} sample embeddings from {len(cluster_data)} cluster(s).")

# ── Load gallery centroids (optional) ─────────────────────────────────────────
gallery_embeddings = []
gallery_labels     = []
if not args.no_gallery and os.path.exists(gallery_path):
    try:
        gallery = torch.load(gallery_path, map_location="cpu", weights_only=False)
        for name, emb in gallery.items():
            centroid = F.normalize(emb.mean(dim=0), p=2, dim=0)
            gallery_embeddings.append(centroid.numpy())
            gallery_labels.append(f"[Known] {name}")
        print(f"Loaded {len(gallery_embeddings)} known gallery identities.")
    except Exception as e:
        print(f"Could not load gallery: {e}")
else:
    print("Skipping gallery (--no-gallery or file not found).")

# ── Combine and run UMAP ───────────────────────────────────────────────────────
combined_embs   = np.array(all_embeddings + gallery_embeddings, dtype=np.float32)
combined_labels = all_labels + gallery_labels

n = len(combined_embs)
n_neighbors = max(2, min(15, n - 1))
print(f"Running UMAP on {n} embeddings (n_neighbors={n_neighbors})...")

reducer = umap.UMAP(
    n_neighbors  = n_neighbors,
    min_dist     = 0.1,
    metric       = "cosine",
    random_state = 42,
    n_components = 2,
)
proj = reducer.fit_transform(combined_embs)

# ── Plot ───────────────────────────────────────────────────────────────────────
unique_clusters = list(cluster_data.keys())
colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_clusters), 1)))
cluster_color_map = {name: colors[i % len(colors)] for i, name in enumerate(unique_clusters)}

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_facecolor("#F4F5F7")
fig.patch.set_facecolor("#FFFFFF")

# Plot cluster samples
for i, (x, y) in enumerate(proj[:len(all_embeddings)]):
    name     = all_labels[i]
    flagged  = all_stability[i]
    color    = cluster_color_map.get(name, "gray")
    marker   = "X" if flagged else "o"
    edgec    = "red" if flagged else "white"
    ax.scatter(x, y, c=[color], marker=marker, edgecolors=edgec,
               linewidths=0.8, s=80, alpha=0.85, zorder=2)

# Plot gallery centroids as stars
if gallery_embeddings:
    for i, (x, y) in enumerate(proj[len(all_embeddings):]):
        ax.scatter(x, y, c="black", marker="*", s=160, zorder=3, alpha=0.7)
        ax.annotate(
            gallery_labels[i].replace("[Known] ", ""),
            (x, y), fontsize=6, ha="center", va="bottom",
            color="#0C2340", alpha=0.8
        )

# Cluster labels at centroid of each group's projected points
for name in unique_clusters:
    idxs  = [i for i, l in enumerate(all_labels) if l == name]
    if idxs:
        cx = np.mean([proj[i, 0] for i in idxs])
        cy = np.mean([proj[i, 1] for i in idxs])
        flagged_clust = cluster_data[name].get("stability_flag", False)
        suffix = " ⚠" if flagged_clust else ""
        ax.annotate(
            f"{name}{suffix}",
            (cx, cy), fontsize=9, fontweight="bold", ha="center",
            color=cluster_color_map.get(name, "gray"),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none")
        )

# Legend
legend_patches = [
    mpatches.Patch(color=cluster_color_map.get(n, "gray"), label=n)
    for n in unique_clusters
]
if gallery_embeddings:
    legend_patches.append(
        mpatches.Patch(color="black", label="Known gallery (★)")
    )
legend_patches.append(
    mpatches.Patch(color="red", label="⚠ Unstable (X marker)")
)
ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
          framealpha=0.9, edgecolor="#D0D5DD")

ax.set_title(
    f"Elephant ReID — UMAP Embedding Space\n"
    f"{len(cluster_data)} cluster(s)  |  {len(all_embeddings)} sample points  |  "
    f"{len(gallery_embeddings)} gallery identities",
    fontsize=11, fontweight="bold", color="#0C2340", pad=12
)
ax.set_xlabel("UMAP Dimension 1", fontsize=9)
ax.set_ylabel("UMAP Dimension 2", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4, color="#D0D5DD")

# Save
save_dir = args.save_dir or os.path.join(output_folder, "umap_plots")
os.makedirs(save_dir, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(save_dir, f"umap_{ts}.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {save_path}")
plt.close()
