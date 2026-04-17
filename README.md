# 🐘 Elephant Re-Identification System

> **Wildlife Institute of India (WII) — Unique Elephant Identification**  
> **Current Model:** V8.2 · ConvNeXt-Tiny · 128-D embeddings  
> **Stack:** Python · PyTorch · YOLOv8 · PyQt6

**Built a graph-based, human-in-the-loop elephant re-identification system, currently in the calibration phase to make similarity-based decisions reliable and scalable.**

---

## 🎯 Core Objective

Build a robust Elephant Re-Identification System that can:

- **Identify** the same elephant across different images
- **Handle** pose variation, lighting changes, partial views
- **Avoid** incorrect automatic merges
- **Use** a human-in-the-loop system to ensure correctness

> **Key idea: AI suggests, Human verifies.**

---

## 🧠 System Overview

### Pipeline Flow

```text
[ 📷 Raw Field Image ]
         │
         ▼
[ YOLOv8n Head Detection ]          ← Multi-scale cascade (640→1024→1280)
         │
         ▼
[ Crop & Quality Gate ]             ← Blur, contrast, head reference bank
         │
         ▼
[ ConvNeXt Embedding (128-D) ]      ← L2-normalized, Flip-TTA
         │
         ▼
[ Gallery Matching & Graph Clustering ] ──(Ambiguous)──> [ ⚠ Human Review Queue ]
         │                                                      │
         ▼                                                      │
[ ✅ Named Identities & Unknown Clusters ] <────────────────────┘
```

### Key Stages

| Stage | Description |
|-------|-------------|
| **Detection & Cropping** | Extract elephant head crops (not full body). Focus on identity features: ears, tusks, wrinkles. |
| **Feature Extraction** | Each image → 128-D vector embedding. Similarity computed using cosine similarity. |
| **Graph-Based Clustering** | Images connected based on similarity. Clusters formed as connected components. Avoids greedy/centroid-only failures. |
| **Review & Merge (UI)** | Shows clusters (Unknown_1, Unknown_2…). Suggests possible merges. Human confirms or rejects. |

---

## 🧩 Project Phases

### ✅ Phase 1 — Feature Extraction (DONE)
- Elephant detection + head cropping
- Embedding generation
- Shift from full-body → head-based features (major improvement)

### ✅ Phase 2 — Clustering System (DONE)
- Initial clustering (centroid-based → replaced)
- Upgraded to graph-based clustering
- Reduced fragmentation issues

### ✅ Phase 3A — Review UI (DONE)
- Built Review & Merge interface
- Cluster visualization
- Manual merge / split / promote actions

### ✅ Phase 3B — Intelligent Suggestions (DONE)
- Ranking of candidate clusters
- Max similarity + centroid similarity
- Bridge reasoning (indirect connections)
- Relative similarity (rank within cluster)
- Improved suggestion quality

### ⚠️ Phase 3C — Calibration & Reliability (ONGOING)

**Goal:** Make the system trustworthy and data-driven.

**Implemented:**
- Decision logging (`merge_decisions.csv`)
- Scores: direct similarity, bridge strength, cluster cohesion
- Bounded scoring (no overconfidence)
- Confidence hints (Safe / Review / Weak)

**In Progress:**
- Collecting real user decisions
- Running analyzer on score distributions
- Deriving safe threshold and reject threshold
- Validating precision (how reliable "safe merge" is)

### 🔵 Phase 4 — Semi-Automation (NEXT)

**Goal:** Reduce manual effort.

- Auto-merge high-confidence clusters
- Priority-based review queue
- Confidence tiers (Safe / Review / Weak)
- Human focuses only on uncertain cases

### 🔵 Phase 5 — System Maturity (FUTURE)

**Goal:** Make system reliable and scalable.

- Better embeddings (handle pose variation)
- Adaptive thresholds (data-driven)
- Robust pipeline (noise, edge cases)
- Evaluation metrics (precision, recall)
- Works across datasets, not just one

---

## 📂 Repository Structure

```text
├── app.py                      # PyQt6 Desktop Application (4-tab UI)
├── pipeline.py                 # Detection + Embedding inference engine
├── core_engine.py              # Clustering, matching & merge logic
├── cluster_health.py           # Cluster safety checks & health monitoring
├── review_store.py             # Ambiguity queue persistence
├── requirements.txt            # Python dependencies
├── elephant_reid.spec          # PyInstaller build spec
├── app_config.json             # Runtime state persistence
├── PROJECT_GUIDE.md            # Detailed 34KB engineering deep-dive
├── README.md                   # This file
│
├── models/                     # YOLO & Re-ID model weights (gitignored)
├── data/                       # Datasets, DBs, archives (gitignored)
├── tools/                      # Diagnostic, evaluation & dataset tools
├── kaggle/                     # Training notebooks (V1–V8)
├── docs/                       # WII methodology & design documentation
├── tests/                      # Unit tests (cluster manager, health checks)
├── logs/                       # Runtime logs (gitignored)
├── src/                        # Legacy model code & preprocessing
└── notebooks/                  # Exploration notebooks
```

---

## ⚠️ Key Challenges

| Challenge | Description |
|-----------|-------------|
| **Intra-class variation** | Same elephant looks very different across images |
| **Precision vs Recall** | Strict → fragmentation; Loose → wrong merges |
| **Embedding limitations** | Some visually similar images still get low similarity |
| **Decision-support design** | Showing useful suggestions without overwhelming or misleading the user |

---

## 🏗️ Architecture Details

| Property | Value |
|----------|-------|
| **Backbone** | ConvNeXt-Tiny (28.6M params, ImageNet pretrained) |
| **Input size** | 224 × 224 RGB |
| **Embedding dim** | 128-D (L2-normalized) |
| **Training loss** | Hard-Positive Alignment (1.0) + ArcFace (0.6) + Triplet (0.5) + Center (0.2) |
| **Clustering** | Graph-based connected components with triple-condition merge guards |
| **Head detector** | YOLOv8n with multi-scale cascade + tiled recovery |

> For the complete technical deep-dive into loss functions, thresholds, clustering algorithms, and UI architecture, see **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)**.

---

**Wildlife Institute of India Research Project**
