## Open-Set Biometric Algorithm (Complete Methodology)
## 1. Problem Definition

The objective of this project is to design an **Elephant Re-Identification (Re-ID) system** capable of recognizing **individual elephants** from field images under **open-set conditions**.
Unlike closed-set classification, the system must:

- Correctly identify **known elephants**
- Explicitly detect **unknown / new individuals**
- Support **incremental enrolment without retraining**
This problem is therefore formulated as an **open-set biometric recognition task**, analogous to human face recognition systems (e.g., Face-ID).

---

## 2. Why Open-Set Biometric Recognition (Not Classification)
### Closed-Set Classification (Rejected)
A closed-set classifier answers:
> “Is this elephant ID-1, ID-2, or ID-N?”

This approach fails because:
- The number of identities is **not fixed**
- Only **2–3 images per elephant** are available (few-shot)
- New elephants appear continuously
- Adding a new elephant requires **retraining the entire network**
- Retraining risks **catastrophic forgetting**
---
### Open-Set Biometric Recognition (Adopted)
In the biometric paradigm:
- The model **does not learn identity labels**
- It learns a **similarity function**
- Each image is mapped to a **fixed-length embedding vector**
- Similar elephants → embeddings close together
- Different elephants → embeddings far apart
New elephants are enrolled by **storing embeddings**, not retraining the model.

---
## 3. Dataset Overview
### 3.1 Dataset Composition
The dataset consists of **high-resolution field images** and is divided into two main groups:
### 🟤 Makhna Dataset
- Adult male elephants (tuskless bulls)
- Each folder corresponds to **one individual Makhna**
- **Highly imbalanced number of images per individual**:
    - Some individuals have **very few images** (e.g., 2–3)
    - Others have **moderate to large collections** (e.g., 20–30+ images)
- Example:
    - _Makhna_9_ → ~36 images
    - _Makhna_10_ → ~5 images
This natural imbalance reflects **uneven field sightings** and must be handled explicitly during training.
---
### 🟢 Herd Dataset
- Elephants observed in **social (herd) settings**
- Organized hierarchically by:
    - **Herd number**
    - **Demographic group**:
        - Adult_Female
        - **Sub_Adult**
        - Juvenile
        - Calf
- Each lowest-level folder corresponds to **one individual elephant**
This structure preserves both **social context** and **individual identity**.
---
### 3.2 Image Characteristics
- Resolution:
    - Mostly ~4608 × 3456
    - Some ~5184 × 3456
- High variability in:
    - Pose
    - Lighting
    - Occlusion
    - Background
This is **uncontrolled, real-world field data**.
---
## 4. Key Dataset Challenges

1. **Few-shot learning** (2–3 images per ID)
2. **No bounding box annotations**
3. **Multi-elephant scenes**
4. **High inter-individual similarity**
5. **Large biological variation across age and sex classes**
---
## 5. Weak Supervision via Red Arrow
### What the Arrow Is
- Digitally painted
- Solid red
- Same shape across images
- Appears **only in multi-elephant images**
- Points to the **target elephant**
### What the Arrow Is NOT
- Not a bounding box
- Not a head locator
- Not a biometric region indicator
### Correct Interpretation

> **The arrow is an identity selector, not a localization signal.**

It answers:
> “Which elephant in this image is the one of interest?”
## 6. Phase A — Data Understanding (Frozen Assumptions)

|ID|Assumption|Status|
|---|---|---|
|A1|Arrow is digitally painted and consistent|✅|
|A2|Arrow always points to target elephant|✅|
|A3|Arrow position varies (head/back/side)|✅|
|A4|Images are high resolution|✅|
|A5|Open-set recognition required|✅|
|A6|Identity unit = individual elephant|✅|
|A7|Robustness > speed|✅|

---
## 7. Phase B — Preprocessing & MegaDetector Integration
### 7.1 Objective
Convert raw images into **elephant-centric inputs** suitable for biometric learning using automated detection.
### 7.2 MegaDetector Integration (UPDATED)

**Key Breakthrough:** Integrated MegaDetector v5a for automated elephant detection
- **100% detection rate** (validated on sample set)
- Works with or without arrows
- Provides precise bounding boxes around elephants
- Eliminates manual cropping heuristics

**Validated Parameters:**
- MegaDetector Model: **v5a**
- Confidence Threshold: **0.4** (validated through exploration)
- Padding Ratio: **15%** around bounding box (validated)
- Arrow Detection: Retained for **multi-elephant selection**

### 7.3 Detection-Based Preprocessing Strategy

#### Step 1: Elephant Detection
1. Load MegaDetector v5a model (once at startup)
2. Run detection on each image
3. Filter for animal category (category '1')
4. Apply confidence threshold (≥0.4)
5. Obtain normalized bounding boxes [x, y, width, height]

#### Step 2: Target Elephant Selection

**For Multi-Elephant Scenes (Arrow Present):**
1. Detect red arrow via HSV segmentation
2. Find arrow tip coordinates
3. Select detection containing arrow point
4. Fallback: Select detection closest to arrow if none contain it

**For Single-Elephant Scenes (No Arrow):**
- Select largest bounding box (by area)

#### Step 3: Context-Preserving Crop Extraction
1. Take selected bounding box
2. Add **15% padding** on all sides:
   - Horizontal padding: 15% of box width
   - Vertical padding: 15% of box height
3. Clamp to image boundaries
4. Extract crop region

### 7.4 Why MegaDetector-Based Cropping Works
- **Precise localization** instead of heuristic anchoring
- **Consistent crop quality** across all images
- Padding preserves biometric context:
    - Head profile and dome shape
    - Ear shape, tears, and depigmentation
    - Temporal gland region (Makhnas)
    - Upper torso texture
- **Robust to pose variation** and multi-elephant scenes
- **Scalable**: no manual annotation required
---
## 8. Phase C — Biologically-Aware Feature Extraction
### 8.1 Motivation
Elephant identity cues differ significantly across:
- Sex (Makhnas vs Females)
- Age (Adults vs Calves)
A single-stream CNN cannot model this heterogeneity effectively.
---
## 8.2 Dual-Branch Feature Extractor
### Branch 1: Texture Branch
**Purpose:** Capture fine-grained local details
Targets:
- Ear depigmentation (pink spots)
- Ear tears and notches
- Skin and trunk texture
Characteristics:
- Shallow
- High spatial resolution
- Small receptive field
Dominant for:
- Adult females
- Some adult males
---
### Branch 2: Semantic Shape Branch
**Purpose:** Capture global geometric structure
Targets:
- Body bulk (Makhnas)
- Head dome shape (Calves)
- Ear curvature
- Overall proportions
Characteristics:
- Deep
- Low spatial resolution
- Large receptive field
Dominant for:
- Calves / Juveniles
- Makhnas
---
## 8.3 Biological Attention Map (BAM)
### Objective
Learn **where to look** based on biologically meaningful regions.
### Expected Attention Behavior
#### Makhnas
- Temporal gland / cheek region
- Eye-adjacent areas
- Body bulk
(Captures musth secretion and gland morphology)
---
#### Adult Females
- Ear pinna
- Ear edges and tears
- Facial texture
---
#### Calves / Juveniles
- Head shape
- Ear curvature
- Global proportions
(Compensates for lack of texture cues)

---
### Key Property
- Attention is **learned implicitly**
- No explicit sex/age labels required
- Metric learning drives specialization
---
## 9. Feature Fusion & Embedding Projection
1. Texture branch features
2. Shape branch features
3. Apply Biological Attention Map to each
4. Pool features independently
5. Fuse via late fusion (concatenation / weighted sum)
6. Project to **128-dimensional embedding**
7. Apply L2 normalization
---
## 10. Phase D — Metric Learning (Few-Shot Optimization)
### Objective
Learn a **discriminative embedding space** instead of memorizing identities.
### Loss Function
**Triplet Margin Loss with Online Hard Negative Mining**
- Optimizes relative distances
- Encourages:
    - Intra-ID compactness
    - Inter-ID separation
- Well-suited for few-shot learning
---
### Why Metric Learning Solves Few-Shot
- Learns _how to compare_, not _what to classify_
- Generalizes naturally to unseen identities
- Embeddings are reusable
---
## 11. Artifact Handling (Arrow Bias Prevention)

Although arrows remain visible in some crops:

- Same elephant appears with and without arrows
- Arrow position varies
To prevent shortcut learning
- Apply **Random Erasing**
- Apply strong spatial augmentations
This forces reliance on **biological features**, not artificial cues.
---
## 12. Phase E — Open-Set Inference & Enrollment
### Enrollment
- Compute embeddings for known elephants
- Store in gallery database
- Model remains frozen
---
### Inference
1. Compute embedding for query image
2. Compute cosine similarity with gallery
3. Apply confidence threshold
---
### Decision Logic
- Similarity ≥ threshold → **Known Elephant**
- Similarity < threshold → **New / Unknown Individual**
This explicitly supports **open-set recognition**.
---

## 13. Final Algorithm Summary

> The proposed system integrates contextual anchoring, biologically-aware feature extraction, and metric learning to enable robust open-set elephant re-identification under real-world field conditions.

---
## 14. Project Status

- ✔ Dataset fully understood
- ✔ Weak supervision correctly interpreted
- ✔ **MegaDetector integration validated (100% detection rate)**
- ✔ **Detection-based preprocessing implemented**
- ✔ Feature extractor biologically grounded
- ✔ Metric learning strategy defined
- ✔ Ready for training & evaluation
---

### One-Line Summary (PI-Level)

> _We designed an open-set biometric elephant re-identification system that integrates MegaDetector v5a for automated detection with arrow-based selection in multi-elephant scenes, preserves biological context through validated padded crops (15%), and learns discriminative embeddings using biologically informed attention and metric learning._