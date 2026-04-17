import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny
import torch.nn.functional as F
import os
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- UI CONFIG & STYLING ----------------
st.set_page_config(page_title="Elephant Identity Manager", page_icon="🐘", layout="wide")

st.markdown("""
<style>
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 60, 114, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(100, 100, 100, 0.2);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .metric-card h3 {
        margin-bottom: 5px !important;
        padding-bottom: 0px !important;
    }
    .metric-card p {
        margin-top: 0px !important;
        font-weight: 500;
        color: #555;
    }
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    /* Headers */
    h1, h2, h3, h4 {
        color: #f8f9fa !important;
        font-family: 'Inter', sans-serif;
    }
    /* Dynamic pill badges */
    .decision-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 15px;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
    }
    .decision-high { background: #2ecc71; }
    .decision-med { background: #f1c40f; color: black; }
    .decision-weak { background: #e67e22; color: white; }
    .decision-new { background: #e74c3c; }
    /* Unsquish Images */
    [data-testid="stImage"] > img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 5px;
        background: rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
class SimpleReIDModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = convnext_tiny(weights="DEFAULT")
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(768, 256)

    def forward(self, x):
        feat = self.backbone.features(x)
        feat = self.pool(feat).flatten(1)
        emb = self.fc(feat)
        return F.normalize(emb, dim=1)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(model_path):
    model = SimpleReIDModel()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor()
])

# ---------------- EMBEDDING & CROPPING ----------------
def center_crop_elephant(img):
    w, h = img.size
    # crop central region (elephant usually centered)
    return img.crop((
        int(w*0.2), 
        int(h*0.1), 
        int(w*0.8), 
        int(h*0.9)
    ))

def get_embedding(model, image):
    img = center_crop_elephant(image)
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(img)
    emb = emb.squeeze().numpy()
    # Explicit L2 Norm safeguard for stable similarities
    emb = emb / np.linalg.norm(emb)
    return emb

# ---------------- MAIN UI ----------------
st.title("🐘 Elephant Identity Manager")
st.markdown("### AI-assisted grouping system with confidence-aware decision support")

model_path = st.text_input("Enter model path (.pth)", r"models\ms_loss_hardmining_filtered_model_v2.pth")
model = load_model(model_path)

uploaded_files = st.file_uploader("Upload Elephant Images", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    images = []
    embeddings = []

    for i, file in enumerate(uploaded_files):
        img = Image.open(file).convert("RGB")
        images.append((file.name, img))
        embeddings.append(get_embedding(model, img))

    embeddings = np.array(embeddings)
    sim_matrix = cosine_similarity(embeddings)

    # ---------------- GROUPING LOGIC ----------------
    THRESH_HIGH = 0.75
    THRESH_LOW = 0.5
    n = len(images)
    unassigned = set(range(n))
    clusters = []
    uncertain = []

    while unassigned:
        i = unassigned.pop()
        group = [i]
        candidates = [j for j in unassigned if sim_matrix[i][j] > THRESH_HIGH]
        for j in candidates:
            if all(sim_matrix[j][g] >= THRESH_LOW for g in group):
                group.append(j)
        for g in group:
            if g in unassigned:
                unassigned.remove(g)
        if len(group) == 1:
            uncertain.append(group[0])
        else:
            clusters.append(group)

    # Calculate overall metrics
    accepted_matches = 0
    possible_matches = 0
    weak_matches = 0
    new_ids = 0
    
    for i in range(n):
        sims = sim_matrix[i]
        top_indices = np.argsort(-sims)
        top_indices = [idx for idx in top_indices if idx != i][:5]
        
        if len(top_indices) >= 2:
            top1 = sims[top_indices[0]]
            top2 = sims[top_indices[1]]
            gap = top1 - top2
        else:
            top1 = sims[top_indices[0]] if top_indices else 0
            gap = 0
            
        if top1 >= 0.70 and gap > 0.08:
            accepted_matches += 1
        elif top1 >= 0.65 and gap > 0.05:
            possible_matches += 1
        elif top1 >= 0.60:
            weak_matches += 1
        else:
            new_ids += 1

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(f"<div class='metric-card'><h3>{len(clusters)}</h3><p>Safe Groups</p></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><h3>{accepted_matches}</h3><p>Strong Links</p></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><h3>{possible_matches}</h3><p>Possible</p></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card'><h3>{weak_matches}</h3><p>Weak (Review)</p></div>", unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='metric-card'><h3>{new_ids}</h3><p>New IDs</p></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- DISPLAY CLUSTERS ----------------
    if clusters:
        st.subheader("🧠 Detected Elephant Groups (Confirmed)")
        for idx, cluster in enumerate(clusters):
            st.markdown(f"**Cluster {idx+1} | Size: {len(cluster)}**")
            cols = st.columns(min(len(cluster), 5))
            for i, c in enumerate(cluster):
                if i < 5:
                    cols[i].image(images[c][1], caption=images[c][0], use_column_width=True)
            if len(cluster) > 5:
                st.caption(f"... and {len(cluster) - 5} more images")
            st.divider()

    # ---------------- SUGGESTIONS / UNCERTAIN ----------------
    st.subheader("💡 AI Decision Support (Uncertain Pool & Suggestions)")

    for i in range(n):
        sims = sim_matrix[i]
        top_indices = np.argsort(-sims)
        top_indices = [idx for idx in top_indices if idx != i][:5]

        # Get top1, top2
        if len(top_indices) >= 2:
            top1 = sims[top_indices[0]]
            top2 = sims[top_indices[1]]
            gap = top1 - top2
        else:
            top1 = sims[top_indices[0]] if top_indices else 0
            gap = 0

        # Decision logic (TIGHTENED 4-Tier Separation)
        if top1 >= 0.70 and gap > 0.08:
            decision = "STRONG MATCH"
            badge_class = "decision-high"
            icon = "🟢"
        elif top1 >= 0.65 and gap > 0.05:
            decision = "POSSIBLE MATCH"
            badge_class = "decision-med"
            icon = "🟡"
        elif top1 >= 0.60:
            decision = "WEAK MATCH (REVIEW)"
            badge_class = "decision-weak"
            icon = "🟠"
        else:
            decision = "NEW / UNKNOWN"
            badge_class = "decision-new"
            icon = "🔴"

        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(images[i][1], caption=f"Query: {images[i][0]}")
                st.markdown(f"<div class='decision-badge {badge_class}'>{icon} {decision}</div>", unsafe_allow_html=True)
                st.write(f"Top Sim: **{top1:.3f}** | Gap: **{gap:.3f}**")
            
            with col2:
                # Show matches
                valid_matches = [idx for idx in top_indices if sims[idx] >= 0.5]
                if valid_matches:
                    match_cols = st.columns(min(len(valid_matches), 5))
                    for m_idx, match_id in enumerate(valid_matches):
                        if m_idx < 5:
                            score = sims[match_id]
                            match_cols[m_idx].image(images[match_id][1], caption=f"{images[match_id][0]} | Sim: {score:.3f}")
                else:
                    st.info("No confident similarities found. Flagged as new individual isolate.")
            st.markdown("---")

st.success("System Ready 🚀")
