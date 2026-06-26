# 🐘 Elephant Re-Identification System

![Patent Pending](https://img.shields.io/badge/Patent-Pending-red)
![Status](https://img.shields.io/badge/Status-Proprietary-blue)
![Domain](https://img.shields.io/badge/Domain-Computer_Vision-success)
![Focus](https://img.shields.io/badge/Focus-Wildlife_Conservation-green)

> **⚠️ Notice:** This project and its underlying methodology are currently under **patent protection**. All source code, model weights, and proprietary training datasets have been removed from this public repository to protect intellectual property. This repository serves as an architectural overview and demonstration of the system's capabilities.

**A graph-based, human-in-the-loop elephant re-identification system designed as a risk-aware, decision-support pipeline for field researchers.**

---

## 🎯 Core Objective

The system addresses the critical challenge of tracking individual elephants across diverse field images by providing a robust Re-Identification (Re-ID) pipeline that can:

- **Identify** the same elephant across different images despite pose variation, lighting changes, and partial views.
- **Isolate** identity features by shifting from full-body analysis to precise head/ear cropping.
- **Prevent** incorrect automatic merges by employing a graph-based clustering algorithm rather than naive centroid matching.
- **Empower** researchers through a Human-in-the-Loop (HITL) interface—operating on the principle: *AI suggests, Human verifies*.

---

## 🏗️ System Architecture

The pipeline processes raw field images through a multi-stage cascade, extracting robust features and grouping them safely before presenting ambiguous cases to a human reviewer.

```mermaid
graph TD
    %% Styling
    classDef input fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#bbf,stroke:#333,stroke-width:2px;
    classDef model fill:#fbb,stroke:#333,stroke-width:2px;
    classDef decision fill:#ff9,stroke:#333,stroke-width:2px;
    classDef output fill:#bfb,stroke:#333,stroke-width:2px;
    classDef human fill:#fbf,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;

    A([📷 Raw Field Image]) ::: input
    
    subgraph Detection Phase
        B[YOLOv8n Head Detection<br/>Multi-scale cascade 640→1024→1280] ::: process
        C[Crop & Quality Gate<br/>Blur, contrast, head reference checks] ::: process
    end
    
    subgraph Feature Extraction
        D[ConvNeXt-Tiny Backbone<br/>28.6M params, ImageNet Pretrained] ::: model
        E[128-D Embedding<br/>L2-normalized, Flip-TTA] ::: process
    end
    
    subgraph Clustering Engine
        F{Graph-Based Clustering &<br/>Gallery Matching} ::: decision
        F -- "High Confidence<br/>(Clear Match)" --> G([✅ Named Identities &<br/>Stable Clusters]) ::: output
        F -- "Ambiguous Match<br/>(Below Threshold)" --> H([⚠️ Human Review Queue]) ::: human
    end
    
    subgraph Review UI
        H --> I{Human-in-the-Loop<br/>Review Interface} ::: human
        I -- "Confirm / Reject / Split" --> G
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

---

## 🧩 Architectural Highlights

### 1. Detection & Quality Gating
- **Multi-Scale Head Detection:** Utilizes YOLOv8n specifically fine-tuned for elephant head detection (ignoring body mass which introduces noise).
- **Quality Gates:** Automatically filters out blurry or low-contrast crops to prevent poisoning the gallery.

### 2. Feature Extraction
- **ConvNeXt-Tiny Backbone:** Chosen for its balance of efficiency and modern convolutional performance.
- **Robust Loss Function Strategy:** Model trained using a composite loss function including Hard-Positive Alignment, ArcFace, Triplet, and Center loss to maximize inter-class variance and minimize intra-class variance.

### 3. Graph-Based Clustering
- **Beyond Centroids:** Replaced traditional centroid-based clustering with a graph-based connected components approach, significantly reducing fragmentation issues caused by extreme pose variations.
- **Triple-Condition Merge Guards:** Strict thresholds prevent overconfident merges, prioritizing precision over recall to maintain gallery purity.

### 4. Human-in-the-Loop (HITL) Ecosystem
- **Intelligent Suggestions:** The system ranks candidate clusters based on direct similarity, bridge strength (indirect connections), and relative cluster cohesion.
- **Safety-First UI:** Provides safety blocks, single-image warnings, and gap-analysis confidence filters. The ultimate authority always defers to the human reviewer.

---

## 🏷️ Tags & Topics

`#ComputerVision` `#WildlifeConservation` `#DeepLearning` `#PyTorch` `#YOLOv8` `#ConvNeXt` `#HumanInTheLoop` `#MetricLearning` `#ImageRetrieval` `#GraphClustering`

---

*Note: For inquiries regarding the patent, licensing, or academic collaboration, please contact the repository owner.*
