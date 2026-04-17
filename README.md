# 🐘 Elephant Re-Identification System

Open-set biometric elephant re-identification engine built for the **Wildlife Institute of India (WII)**.

This system guarantees human-validated tracking of wild elephants across disparate camera trap locations. It utilizes a conservative deep learning pipeline driven by a **YOLOv8** head detector, a **ConvNeXt-Tiny (V8.2)** embedding extractor, and aggressive, non-drifting graph-connected clustering logic.

---

## 🚀 Core Components

The system has been heavily optimized and pruned. The runtime is driven entirely by three core files:

| Core File | Purpose |
|------|---------|
| **`app.py`** | The Desktop Application (PyQt6). Provides a 4-tab interface for Mass Processing, Gallery Browsing, Identity Registration, and Human-in-the-loop Merge Queues. |
| **`pipeline.py`** | The Deep Learning Inference layer. Executes the multi-scale Head Detection cascade, applies heuristic crop-quality gating, and generates strict 128-D L2-normalized metric embeddings. |
| **`core_engine.py`** | The Clustering Logic. Implements K-Reciprocal nearest neighbor re-ranking, cross-session Unknown Cluster Management, and dynamic graph-cluster merging governed by conservative thresholds. |

---

## 🏗️ Architecture Stack

- **Detection:** YOLOv8n (trained aggressively for diverse field scales and partial occlusions)
- **Feature Extractor:** ConvNeXt-Tiny (trained using Hard-Positive Alignment and ArcFace loss)
- **Dimensionality:** L2-normalized 128-D embeddings
- **Clustering:** Transitive graph-based clustering with triple-condition merge guards (to heavily penalize and block False Positives).
- **GUI Framework:** PyQt6

---

## 📂 Repository Structure

```text
├── app.py                      # Main PyQt6 App
├── pipeline.py                 # Core AI evaluation engine
├── core_engine.py              # Clustering and match logic
├── ruff_cache/                 # Pre-commit configurations
├── app_config.json             # State serialization
├── PROJECT_GUIDE.md            # Highly detailed engineering manifesto!
├── README.md                   # This file
├── tools/                      # Diagnostic scripts, evaluating tools, & dataset pruning modules
├── models/                     # Holds yolov8n.pt and elephant_head_reid checkpoints
├── data/                       # Contains active runtime images and SQlite tracking DBs
├── docs/                       # Historical design notes and WII documentation
├── logs/                       # Application runtime and debug crash logs
└── tests/                      # System integrity unit tests
```

---

## 📘 Deep Dive Guide

The application logic, exact hyper-parameter threshold charts, tuning methodologies, and complete design rationale are extremely well documented.

**Please refer to the `PROJECT_GUIDE.md` contained at the root of the project for a complete 34 KB deep-dive into the backend systems.**
