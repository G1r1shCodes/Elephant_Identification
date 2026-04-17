# Kaggle Training — v4.4 / v4.6

## 📁 Files

| File | Description |
|------|-------------|
| `elephant-reid-training-v4.4.ipynb` | Full training notebook (v4.4 architecture, v4.6 augmentations) |
| `dataset-metadata.json` | Kaggle dataset metadata |
| `dataset/` | Dataset configuration |

## 🚀 Quick Start

1. Upload `elephant-reid-training-v4.4.ipynb` to [kaggle.com](https://kaggle.com)
2. Enable **GPU P100 or T4** + **Internet**
3. Add dataset: `restructured-elephant-dataset`
4. Run all cells (~40 epochs, ~30–50 min on P100)

## ✅ What to Watch

Metrics are printed every **5 epochs**:

```
R1=XX.XX%, R5=XX.XX%, R10=XX.XX%, mAP=XX.XX%  (standard)
Centroid: R1=XX.XX%, R5=XX.XX%                 (centroid matching)
CropRobustness: avg_intra_crop_sim=X.XXXX       (TTA stability)
```

**Diagnostics every epoch:**
```
DIAG ep05 | grad: early=... mid=... late=... proj=... tap=... arc=...
           | arc_W_norm=... tgt_logit=... gate=... queue=...
```

**Healthy signs:**
- `gate` value climbing: tap is contributing
- `tgt_logit` > 10 by epoch 20: ArcFace confident
- `CropRobustness` > 0.70: stable partial-view embeddings

## 📦 Outputs

After training, download from `/kaggle/working/`:

| File | Contents |
|------|----------|
| `best_model.pth` | Best checkpoint (highest Rank-1) |
| `last_model.pth` | Final epoch checkpoint |
| `embeddings.pt` | Per-image embeddings + centroids + id_map for all 259 identities |

> When using `embeddings.pt` with `app.py`, the key is `idx_to_identity` (as loaded by `gallery_embeddings.pt`).
> The notebook saves under `id_map` — rename the key if loading directly.

## 🏗️ Architecture Summary

- **Model**: `DualBranchModel` (ConvNeXt-Tiny, 3-part stripe pooling, Stage-2 tap)
- **Embedding**: 384-D, BN-Neck, L2-normalized
- **Losses**: SemiHard Triplet + ArcFace (w=0.07) + Multi-crop Consistency + Part Dropout
- **Classes**: 259 elephant identities
- **Best checkpoint**: saved whenever `max(R1_standard, R1_centroid)` improves
