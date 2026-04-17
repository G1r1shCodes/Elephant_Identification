"""
Evaluate best_model v4.6 using embeddings.pt (all 259 identities).
Leave-one-out protocol: each image is the query; rest of same-ID images are gallery.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── Load embeddings ──────────────────────────────────────────────────────────
EMB_PATH = ROOT / "models" / "embeddings.pt"
if not EMB_PATH.exists():
    EMB_PATH = ROOT / "models" / "gallery_embeddings.pt"

print(f"Loading: {EMB_PATH}")
ckpt = torch.load(str(EMB_PATH), map_location="cpu", weights_only=False)

if isinstance(ckpt, dict) and "embeddings" in ckpt:
    embs   = ckpt["embeddings"]                # (N, 384) — already L2-normalized
    labels  = ckpt["labels"]                    # (N,) int
    id_map = ckpt.get("id_map") or ckpt.get("idx_to_identity", {})
else:
    # Flat format from Tab 3: {identity: tensor}
    emb_list = []
    lbl_list = []
    id_map = {}
    for idx, (ident, emb) in enumerate(ckpt.items()):
        id_map[idx] = ident
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        emb_list.append(emb)
        lbl_list.extend([idx] * emb.shape[0])
    
    embs = torch.cat(emb_list, dim=0)
    labels = torch.tensor(lbl_list)


embs   = F.normalize(embs.float(), dim=1)
lbls   = labels.numpy()
N      = len(lbls)

print(f"  Embeddings: {embs.shape}")
print(f"  Identities: {len(set(lbls.tolist()))}")
print()

# ── Leave-one-out evaluation ─────────────────────────────────────────────────
# For each image i: query = embs[i], gallery = all other images (same + different IDs)
# Exclude self from ranking.

sim_mat = torch.mm(embs, embs.T).numpy()   # (N, N)  cosine similarity
np.fill_diagonal(sim_mat, -2.0)            # exclude self

counts = Counter(lbls.tolist())
valid  = np.array([counts[l] >= 2 for l in lbls])   # need ≥2 images per ID
idx    = np.where(valid)[0]

r1 = r5 = r10 = total = 0
aps = []

for i in idx:
    order  = np.argsort(-sim_mat[i])        # descending similarity
    m      = (lbls[order] == lbls[i])       # match mask
    if m[:1].any():  r1  += 1
    if m[:5].any():  r5  += 1
    if m[:10].any(): r10 += 1
    total  += 1
    nt = m.sum()
    if nt > 0:
        cs  = np.cumsum(m)
        prec = cs / (np.arange(len(m)) + 1)
        aps.append((prec * m).sum() / nt)

mAP   = float(np.mean(aps)) if aps else 0.0
r1_pct  = r1  / total * 100
r5_pct  = r5  / total * 100
r10_pct = r10 / total * 100
mAP_pct = mAP * 100

print("=" * 55)
print("  RESULTS — best_model v4.6  (leave-one-out)")
print("=" * 55)
print(f"  Rank-1   : {r1_pct:.2f}%")
print(f"  Rank-5   : {r5_pct:.2f}%")
print(f"  Rank-10  : {r10_pct:.2f}%")
print(f"  mAP      : {mAP_pct:.2f}%")
print(f"  Queries  : {total}  (of {N} images, {len(set(lbls.tolist()))} IDs)")
print("=" * 55)

# ── Centroid matching ────────────────────────────────────────────────────────
# For each query image, compare to per-ID centroids (excluding own image from centroid).
print("\nComputing centroid metrics...")
cr1 = cr5 = cr10 = ctotal = 0
uid_list = sorted(set(lbls.tolist()))

for i in idx:
    dists = {}
    for uid in uid_list:
        mask = np.where(lbls == uid)[0]
        if uid == lbls[i]:
            mask = mask[mask != i]
            if len(mask) == 0:
                continue
        centroid = F.normalize(embs[mask].mean(dim=0), dim=0)
        dists[uid] = float(1.0 - torch.dot(embs[i], centroid).item())   # cosine dist
    if not dists:
        continue
    ranked = sorted(dists, key=dists.get)
    if ranked[0]    == lbls[i]: cr1  += 1
    if lbls[i] in ranked[:5]:  cr5  += 1
    if lbls[i] in ranked[:10]: cr10 += 1
    ctotal += 1

print(f"  Centroid Rank-1  : {cr1/ctotal*100:.2f}%")
print(f"  Centroid Rank-5  : {cr5/ctotal*100:.2f}%")
print(f"  Centroid Rank-10 : {cr10/ctotal*100:.2f}%")
print(f"  Queries          : {ctotal}")
print("=" * 55)
