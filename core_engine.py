import os
import sys
import json
import shutil
from datetime import datetime

# 🚨 CRITICAL WINDOWS FIX: Prevents DLL conflicts between PyTorch OpenMP and PyQt6
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F

DEBUG_CLUSTERING = True

from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import logging
import numpy as np
import math
from review_store import ReviewStore

# ── Logging Configuration ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("elephant_reid_runtime.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ElephantEngine")

from pipeline import (
    get_reid_model,
    INFERENCE_TRANSFORM,
    DIST_STRICT,
    DIST_LOOSE,
    GAP_STRICT,
    GAP_LOOSE,
)
import pipeline as _pipeline  # for dynamic EMBED_DIM access after model load

AUTO_ENROLL_PROVISIONAL = False


# ══════════════════════════════════════════════════════════════════════════════
# Unknown Elephant Cluster Manager — Agglomerative Clustering
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.metrics.pairwise import cosine_distances
from cluster_health import ClusterHealthMonitor


def k_reciprocal_rerank(embeddings, k1=None, k2=None, lambda_value=0.3):
    """K-reciprocal re-ranking for ReID (Zhong et al., CVPR 2017).

    Rewards pairs that are mutual nearest neighbors — i.e., belong to the same
    identity even across backgrounds — and penalises coincidental high similarity
    driven by shared background or lighting.

    Args:
        embeddings : (N, D) float32 tensor of L2-normalised embeddings.
        k1         : neighbourhood size for reciprocal encoding  (default: min(⌊N/2⌋, 8))
        k2         : neighbourhood size for query expansion      (default: min(3, k1))
        lambda_value: weight for original cosine distance term   (default: 0.3)

    Returns:
        dist (N, N) numpy array — re-ranked distances (lower = more similar).
    """
    N = len(embeddings)
    if N < 2:
        return np.zeros((N, N), dtype=np.float32)

    if k1 is None:
        k1 = max(2, min(N // 2, 8))
    if k2 is None:
        k2 = max(1, min(3, k1))

    emb_np = (
        embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    )
    # Cosine distance matrix [0, 2]
    dist = cosine_distances(emb_np).astype(np.float32)

    # k-reciprocal nearest neighbours for each sample
    def k_reciprocal_neigh(dist_mat, q, k):
        order = np.argsort(dist_mat[q])  # ascending distance
        nn = order[1 : k + 1]  # exclude self
        reciprocal = np.where(np.argsort(dist_mat[nn])[:, 0] < k)[
            0
        ]  # those that rank q in top-k back
        return set(nn[reciprocal]) | {q}

    # Build Jaccard-based feature vectors (sparse)
    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        Rk = k_reciprocal_neigh(dist, i, k1)
        # Query expansion: extend with secondary reciprocals
        Rk_exp = set(Rk)
        for j in Rk:
            Rk2 = k_reciprocal_neigh(dist, j, max(1, k1 // 2))
            if len(Rk2 & Rk) / len(Rk2) >= 0.5:
                Rk_exp |= Rk2
        Rk_list = sorted(Rk_exp)
        w = np.exp(-dist[i, Rk_list])
        w /= w.sum() + 1e-8
        V[i, Rk_list] = w

    # Jaccard distance
    jacc = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            num = np.minimum(V[i], V[j]).sum()
            den = np.maximum(V[i], V[j]).sum() + 1e-8
            jacc[i, j] = jacc[j, i] = 1.0 - num / den

    # Query expansion over k2 neighbours for Jaccard
    dist_qe = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        nn_k2 = np.argsort(dist[i])[1 : k2 + 1]
        dist_qe[i] = (V[i] + V[nn_k2].sum(0)) / (k2 + 1)

    jacc_qe = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            num = np.minimum(dist_qe[i], dist_qe[j]).sum()
            den = np.maximum(dist_qe[i], dist_qe[j]).sum() + 1e-8
            jacc_qe[i, j] = jacc_qe[j, i] = 1.0 - num / den

    # Final distance = convex combination
    final = (1 - lambda_value) * jacc_qe + lambda_value * dist
    np.fill_diagonal(final, 0.0)
    return final


def seed_and_grow_batch(
    embeddings,
    max_sim_threshold: float = 0.65,  # rigorous: prevent false merges of different elephants
    top_k_threshold: float = 0.60,  # required top-k consistency
    consistency_threshold: float = 0.60,  # minimum intra-cluster similarity after freeze
):
    """Within-batch seed-and-grow clustering for unknown elephant embeddings.

    Builds clusters purely from sample-to-sample similarity (no centroid during
    the growth phase) so centroid drift cannot cause false merges.

    Guardrails
    ----------
    1. Seed selection   — pick the most isolated image (lowest max-sim to others)
                          prevents chain errors from ambiguous starting points.
    2. Per-step recompute — c_embs is rebuilt at every candidate check so newly
                             added members immediately influence subsequent decisions.
    3. No centroid      — growth uses raw sample-to-sample similarity only.
    4. Singletons kept  — size == 1 is honest uncertainty, not a bug.
    5. Consistency gate — after freeze, mean intra-cluster sim is checked;
                          FAILED clusters are recursively bisected (not exploded).

    Parameters
    ----------
    embeddings            : list of 1-D float32 tensors (L2-normalised)
    max_sim_threshold     : maximum sample similarity required to join cluster
    top_k_threshold       : mean of top-3 sample similarities required to join
    consistency_threshold : minimum mean intra-cluster similarity after freeze

    Returns
    -------
    List of lists of int indices into `embeddings`.
    Each sub-list is a cluster; singletons ([i]) are valid.
    """
    N = len(embeddings)
    if N == 0:
        return []
    if N == 1:
        return [[0]]

    # Pre-compute full pairwise cosine similarity (L2-normalised → dot product)
    emb_np = torch.stack(embeddings).cpu().numpy().astype(np.float32)  # (N, D)
    sim_mat = emb_np @ emb_np.T
    np.fill_diagonal(sim_mat, 0.0)

    def _second_best_excluding(anchor_idx, exclude_idx):
        row = sim_mat[anchor_idx].copy()
        row[anchor_idx] = -1.0
        row[exclude_idx] = -1.0
        valid = row[row >= 0.0]
        if valid.size == 0:
            return 0.0
        return float(valid.max())

    def _pair_context(anchor_idx, peer_idx):
        """Return neighborhood support around a candidate pair.

        True same-elephant pairs usually have at least one extra nearby sample.
        Spurious crops often form isolated pairs with no third-image support.
        """
        return max(
            _second_best_excluding(anchor_idx, peer_idx),
            _second_best_excluding(peer_idx, anchor_idx),
        )

    def _check_consistency(indices):
        """Return mean intra-cluster similarity for a list of indices."""
        if len(indices) <= 1:
            return 1.0
        c_embs = emb_np[indices]
        intra = [
            float(c_embs[a] @ c_embs[b])
            for a in range(len(c_embs))
            for b in range(a + 1, len(c_embs))
        ]
        return float(np.mean(intra))

    def _bisect(indices):
        """
        Recursively split a failing cluster into two sub-clusters.
        Finds the two most dissimilar members, uses them as seeds for two
        halves, assigns every other member to the closer seed.
        Returns a list of clusters (each a list of indices).
        """
        if len(indices) <= 1:
            return [indices]

        # Find the most dissimilar pair as seeds for the two halves
        sub_sim = sim_mat[np.ix_(indices, indices)]
        np.fill_diagonal(sub_sim, 1.0)  # exclude self-sim from argmin
        flat_idx = int(np.argmin(sub_sim))
        seed_a_local = flat_idx // len(indices)
        seed_b_local = flat_idx % len(indices)
        seed_a = indices[seed_a_local]
        seed_b = indices[seed_b_local]

        if seed_a == seed_b:
            return [indices]

        # Assign each member to whichever seed is closer
        half_a, half_b = [seed_a], [seed_b]
        for idx in indices:
            if idx == seed_a or idx == seed_b:
                continue
            sim_to_a = float(emb_np[idx] @ emb_np[seed_a])
            sim_to_b = float(emb_np[idx] @ emb_np[seed_b])
            if sim_to_a >= sim_to_b:
                half_a.append(idx)
            else:
                half_b.append(idx)

        # Recursively check each half
        result = []
        for half in [half_a, half_b]:
            if len(half) <= 1:
                result.append(half)
            elif _check_consistency(half) >= consistency_threshold:
                result.append(half)
            else:
                result.extend(_bisect(half))
        return result

    unassigned = list(range(N))
    clusters = []

    while unassigned:
        if len(unassigned) == 1:
            clusters.append(list(unassigned))
            break

        # ── Seed: most isolated unassigned image ────────────────────────────
        sub = np.array(unassigned)
        sub_sims = sim_mat[np.ix_(sub, sub)].copy()
        np.fill_diagonal(sub_sims, 0.0)
        seed_local = int(np.argmin(sub_sims.max(axis=1)))
        seed = unassigned[seed_local]

        cluster = [seed]
        unassigned.remove(seed)

        # ── Grow: iterate until no new members are added ─────────────────────
        changed = True
        while changed:
            changed = False
            for cand in list(unassigned):  # snapshot — safe to mutate unassigned
                c_embs = emb_np[cluster]  # (K, D) — rebuilt each iteration
                sims = c_embs @ emb_np[cand]  # (K,)
                max_sim = float(sims.max())
                min_sim = float(sims.min())

                # Dynamic Node-level Gap Filtering against entire batch
                row = sim_mat[cand].copy()
                if len(row) > 1:
                    order = np.argsort(row)[::-1]
                    cand_gap = row[order[0]] - row[order[1]]
                else:
                    cand_gap = 1.0

                pair_context = (
                    max(_pair_context(cand, member_idx) for member_idx in cluster)
                    if cluster
                    else 0.0
                )

                # Strong rule: keep the original conservative path for very
                # confident joins.
                if cand_gap > 0.10 and min_sim >= 0.15 and max_sim >= 0.75:
                    cluster.append(cand)
                    unassigned.remove(cand)
                    changed = True  # restart pass — new member matters
                    continue

                # Recovery rule: allow slightly weaker singleton-pair joins
                # only when the pair has local neighborhood support.
                if (
                    len(cluster) == 1
                    and cand_gap > 0.04
                    and max_sim >= max_sim_threshold
                    and pair_context >= top_k_threshold
                ):
                    cluster.append(cand)
                    unassigned.remove(cand)
                    changed = True

        # ── Consistency gate: bisect instead of explode ──────────────────────
        if len(cluster) > 1:
            mean_intra = _check_consistency(cluster)
            if mean_intra < consistency_threshold:
                logger.warning(
                    f"seed-and-grow: cluster of {len(cluster)} failed "
                    f"consistency ({mean_intra:.3f} < {consistency_threshold}) "
                    f"— bisecting."
                )
                sub_clusters = _bisect(cluster)
                clusters.extend(sub_clusters)
                continue

        clusters.append(cluster)

    return clusters


class UnknownClusterManager:
    """Incremental centroid clustering for unknown elephants.

    Each new embedding is assigned to the best-matching existing cluster
    (argmax cosine similarity) using a dual-threshold strategy:

        sim >= STRONG_THRESHOLD (0.70)  -> assign directly
        WEAK_THRESHOLD (0.60) <= sim < 0.70 -> sample verification:
                max sample similarity >= SAMPLE_VERIFY_THRESHOLD (0.65)
                    -> assign; else -> new cluster
        sim < WEAK_THRESHOLD            -> new cluster

    Centroid is always recomputed from the stored sample set (not a
    running mean), so one bad image cannot permanently drift the centroid.

    Sample cap uses diversity replacement: the sample most similar to the
    current centroid (most redundant) is dropped first, keeping edge poses.

    Post-batch merging: any two clusters whose centroids exceed
    MERGE_THRESHOLD (0.78) cosine similarity are merged.
    """

    STRONG_THRESHOLD = 0.80  # direct assign  (prioritize purity)
    WEAK_THRESHOLD = 0.70  # enter sample-verify zone
    SAMPLE_VERIFY_THRESHOLD = 0.75  # max sample sim required
    MERGE_THRESHOLD = 0.82  # post-batch centroid merge check
    MAX_SAMPLES = 10  # stored representatives per cluster

    def __init__(
        self,
        unknown_dir,
        cluster_file,
        strong_threshold=STRONG_THRESHOLD,
        weak_threshold=WEAK_THRESHOLD,
        sample_verify_threshold=SAMPLE_VERIFY_THRESHOLD,
        merge_threshold=MERGE_THRESHOLD,
        max_samples=MAX_SAMPLES,
    ):
        self.unknown_dir = unknown_dir
        self.cluster_file = cluster_file
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
        self.sample_verify_threshold = sample_verify_threshold
        self.merge_threshold = merge_threshold
        self.max_samples = max_samples
        self.clusters = {}  # name -> {centroid, samples, count, created_at}
        self.last_assignment = {}

        os.makedirs(self.unknown_dir, exist_ok=True)
        self._load()

    def _load(self):
        """Load clusters from JSON. Missing, corrupt, or legacy feature-space file -> start fresh."""
        if not os.path.exists(self.cluster_file):
            return
        try:
            with open(self.cluster_file, "r") as f:
                data = json.load(f)

            # Check feature space dimensionality on first item
            if data and "centroid" in list(data.values())[0]:
                loaded_dim = len(list(data.values())[0]["centroid"])
                if loaded_dim != _pipeline.EMBED_DIM:
                    logger.warning(
                        f"Embedding dim mismatch (loaded={loaded_dim}, model={_pipeline.EMBED_DIM}). "
                        f"Resetting cluster cache to prevent feature space collision."
                    )
                    return

            for name, info in data.items():
                cluster_info = {
                    "centroid": torch.tensor(info["centroid"], dtype=torch.float32),
                    "samples": [
                        torch.tensor(s, dtype=torch.float32) for s in info["samples"]
                    ],
                    "count": info.get("count", len(info["samples"])),
                    "created_at": info.get("created_at", ""),
                    "variance": info.get("variance", 0.0),
                }
                for key, value in info.items():
                    if key not in {
                        "centroid",
                        "samples",
                        "count",
                        "created_at",
                        "variance",
                    }:
                        cluster_info[key] = value
                self.clusters[name] = cluster_info
        except Exception:
            self.clusters = {}

    def save(self):
        """Persist cluster state to JSON."""
        data = {}
        for name, info in self.clusters.items():
            payload = {
                "centroid": info["centroid"].tolist(),
                "samples": [s.tolist() for s in info["samples"]],
                "count": info["count"],
                "created_at": info.get("created_at", ""),
                "variance": info.get("variance", 0.0),
            }
            for key, value in info.items():
                if key in payload or key in {"centroid", "samples"}:
                    continue
                if isinstance(value, torch.Tensor):
                    continue
                payload[key] = value
            data[name] = payload
        with open(self.cluster_file, "w") as f:
            json.dump(data, f, indent=2)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _recompute_centroid(self, name):
        """Recompute centroid from the stored sample set (not running mean).
        Safe to call after every sample addition or removal.
        """
        import torch.nn.functional as F

        samples = self.clusters[name]["samples"]
        if not samples:
            return
        stacked = torch.stack(samples)
        centroid = stacked.mean(dim=0)
        norm = torch.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        self.clusters[name]["centroid"] = centroid

        # Calculate cluster variance/spread
        if len(samples) > 1:
            sims = F.cosine_similarity(stacked, centroid.unsqueeze(0))
            variance = float(1.0 - sims.mean())
        else:
            variance = 0.0
        self.clusters[name]["variance"] = variance

    def _add_to_cluster(self, name, embedding):
        """Add embedding to cluster, applying diversity-based sample cap."""
        cluster = self.clusters[name]
        cluster["samples"].append(embedding.clone())
        cluster["count"] += 1

        if len(cluster["samples"]) > self.max_samples:
            # Diversity removal: drop the sample closest to the current centroid
            # (most redundant) so edge-case poses are preserved.
            sims = [
                float(torch.dot(s, cluster["centroid"])) for s in cluster["samples"]
            ]
            most_redundant = sims.index(max(sims))
            cluster["samples"].pop(most_redundant)

        self._recompute_centroid(name)

    def _create_cluster(self, embedding):
        """Create a brand-new Unknown_N cluster from a single embedding."""
        new_id = len(self.clusters) + 1
        name = f"Unknown_{new_id}"
        self.clusters[name] = {
            "centroid": embedding.clone(),
            "samples": [embedding.clone()],
            "count": 1,
            "created_at": datetime.now().isoformat(),
        }
        os.makedirs(os.path.join(self.unknown_dir, name), exist_ok=True)
        return name

    # ── Public API ─────────────────────────────────────────────────────────────

    def assign_group(self, group_embs):
        """Assign a multi-image group to the best existing cluster, or create new.

        Uses centroid similarity and flip TTA embeddings instead of pairwise
        instance similarities. Prevents cluster drift.

        Args:
            group_embs: list of 1-D float32 tensors (L2-normalised TTA embeddings)

        Returns:
            (cluster_name: str, score: float)
        """
        if not group_embs:
            raise ValueError("group_embs must be non-empty")

        # 1. Compute group centroid
        group_centroid = F.normalize(torch.stack(group_embs).mean(0), p=2, dim=0)

        if not self.clusters:
            name = self._create_cluster_from_samples(group_embs)
            self.last_assignment = {
                "decision": "UNKNOWN",
                "current_cluster": name,
                "score": 1.0,
                "gap": 1.0,
                "candidates": [],
                "ambiguous": False,
            }
            return name, 1.0

        candidates = []
        for cname, cluster in self.clusters.items():
            if not cluster["samples"]:
                continue

            centroid = F.normalize(cluster["centroid"], p=2, dim=0)
            score = float(torch.dot(group_centroid, centroid))

            # Allow singleton merges only if signal is very strong
            if len(cluster["samples"]) < 2 and score <= 0.70:
                continue

            candidates.append((score, cname))

        candidates.sort(reverse=True, key=lambda x: x[0])

        if not candidates:
            # Fallback if no valid targets exist (e.g. all existing clusters are singletons)
            name = self._create_cluster_from_samples(group_embs)
            self.last_assignment = {
                "decision": "UNKNOWN",
                "current_cluster": name,
                "score": 1.0,
                "gap": 1.0,
                "candidates": [],
                "ambiguous": False,
            }
            return name, 1.0

        top_sim, top_name = candidates[0]
        second_sim = candidates[1][0] if len(candidates) >= 2 else 0.0
        gap = top_sim - second_sim

        variance = self.clusters[top_name].get("variance", 0.0)
        status = "UNSTABLE" if variance > 0.25 else "STRONG"

        if status == "STRONG":
            required_gap = 0.20
        elif status == "WEAK":
            required_gap = 0.22
        else:
            required_gap = 0.25

        ranked_candidates = [
            {"name": cname, "score": float(score)} for score, cname in candidates[:4]
        ]

        if top_sim >= 0.72:
            decision = "HIGH"
        elif top_sim > 0.70 and gap > required_gap * 0.8:
            decision = "HIGH"
        elif top_sim > 0.60 and gap > 0.12:
            decision = "MEDIUM"
        else:
            decision = "UNKNOWN"

        if decision == "UNKNOWN":
            name = self._create_cluster_from_samples(group_embs)
            self.clusters[name]["ambiguous"] = False
            self.last_assignment = {
                "decision": decision,
                "current_cluster": name,
                "score": top_sim,
                "gap": gap,
                "candidates": ranked_candidates,
                "ambiguous": False,
            }
            return name, top_sim

        elif decision == "MEDIUM":
            logger.info(
                f"assign_group: Outlier/Ambiguous. top_sim={top_sim:.3f}, gap={gap:.3f}. "
                f"Creating new cluster instead of false merge."
            )
            name = self._create_cluster_from_samples(group_embs)
            self.clusters[name]["ambiguous"] = True
            self.last_assignment = {
                "decision": decision,
                "current_cluster": name,
                "score": top_sim,
                "gap": gap,
                "candidates": ranked_candidates,
                "ambiguous": True,
            }
            return name, top_sim

        else:
            # HIGH - Confident Merge
            for emb in group_embs:
                self._add_to_cluster(top_name, emb)

            self.last_assignment = {
                "decision": decision,
                "current_cluster": top_name,
                "score": top_sim,
                "gap": gap,
                "candidates": ranked_candidates,
                "ambiguous": False,
            }
        return top_name, top_sim

    def _create_cluster_from_samples(self, group_embs):
        """Create a new Unknown_N cluster seeded with all group embeddings."""
        new_id = len(self.clusters) + 1
        name = f"Unknown_{new_id}"
        rep = F.normalize(torch.stack(group_embs).mean(0), p=2, dim=0)
        self.clusters[name] = {
            "centroid": rep.clone(),
            "samples": [e.clone() for e in group_embs],
            "count": len(group_embs),
            "created_at": datetime.now().isoformat(),
        }
        os.makedirs(os.path.join(self.unknown_dir, name), exist_ok=True)
        return name

    def assign(self, embedding, input_id="unknown"):
        """Assign one L2-normalised embedding using strict dual-scoring and gap logic.

        Returns:
            (cluster_name: str, score: float, confidence: str)
        """
        import math

        if not self.clusters:
            name = self._create_cluster(embedding)
            self.last_assignment = {
                "decision": "UNKNOWN",
                "current_cluster": name,
                "score": 1.0,
                "gap": 1.0,
                "candidates": [],
                "ambiguous": False,
            }
            return name, 1.0, "UNKNOWN"

        # Step 1: Compute Dual Score against existing clusters
        scores = {}
        for name, cluster in self.clusters.items():
            cluster_size = len(cluster["samples"])

            # Cluster affinity: collective strength
            sims = []
            for sample in cluster["samples"]:
                sims.append(float(torch.dot(embedding, sample)))

            if not sims:
                continue

            mean_sim = sum(sims) / len(sims)
            min_sim = min(sims)

            # Count strong edges
            strong_edges = sum(1 for s in sims if s > 0.52)

            sim_to_cen = float(torch.dot(embedding, cluster["centroid"]))

            if DEBUG_CLUSTERING:
                logger.info(f"\n[DEBUG] Evaluating node against cluster {name}")
                logger.info(f"  size={cluster_size}")
                logger.info(f"  sims={['%.3f' % s for s in sims]}")
                logger.info(
                    f"  mean={mean_sim:.3f} | min={min_sim:.3f} | strong_edges={strong_edges} | cen={sim_to_cen:.3f}"
                )

            reasons = []

            # ---- CLUSTER PROTECTION ----
            if cluster_size >= 2:
                if mean_sim < 0.50 and strong_edges == 0 and sim_to_cen < 0.60:
                    reasons.append(
                        "structural reject (low mean, no strong edges, low centroid)"
                    )

                if reasons:
                    if DEBUG_CLUSTERING:
                        logger.info(f"  -> \u274c REJECTED ({', '.join(reasons)})")
                    continue

            # ---- CENTROID / STRUCTURE ----
            if cluster_size >= 2:
                centroid_ok = (
                    mean_sim > 0.52
                    or strong_edges >= 2
                    or (strong_edges == 1 and max(sims) > 0.60)
                )
            else:
                centroid_ok = (
                    mean_sim > 0.52
                    or strong_edges >= 1
                    or (strong_edges == 1 and max(sims) > 0.60)
                )

            if not centroid_ok:
                if DEBUG_CLUSTERING:
                    logger.info("  -> \u274c REJECTED (low centroid)")
                continue

            if DEBUG_CLUSTERING:
                logger.info("  -> \u2705 ACCEPTED (cluster affinity)")

            variance = cluster.get("variance", 0.0)

            # --- CLUSTER REASONING SCORE ---
            import numpy as np

            std_sim = float(np.std(sims)) if cluster_size > 1 else 0.0

            # Tie-breaker: penalize looser clusters using internal variance
            cluster_penalty = 0.03 * variance
            effective_score = (
                mean_sim + (0.05 * strong_edges) - (0.03 * std_sim) - cluster_penalty
            )

            scores[name] = {
                "effective": effective_score,
                "variance": variance,
                "cluster_size": cluster_size,
                "status": "UNSTABLE" if variance > 0.25 else "STRONG",
            }

        # Find best and second best
        sorted_scores = sorted(
            scores.items(), key=lambda x: x[1]["effective"], reverse=True
        )

        if not sorted_scores:
            # Fallback if no valid targets exist (e.g. all existing are singletons)
            name = self._create_cluster(embedding)
            self.last_assignment = {
                "decision": "UNKNOWN",
                "current_cluster": name,
                "score": 1.0,
                "gap": 1.0,
                "candidates": [],
                "ambiguous": False,
            }
            return name, 1.0, "UNKNOWN"

        best_name = sorted_scores[0][0]
        best_score = sorted_scores[0][1]["effective"]

        second_score = (
            sorted_scores[1][1]["effective"] if len(sorted_scores) > 1 else 0.0
        )
        gap = best_score - second_score

        # Step 2: Structural Gap-based decision tree
        if len(sorted_scores) > 1 and gap < 0.03:
            # Ambiguous between two valid structural matches -> Review
            cluster_name = None
            decision = "AMBIGUOUS"
            required_gap = 0.03
        else:
            # Inherently worthy of assignment since it passed structural filters
            self._add_to_cluster(best_name, embedding)
            decision = "HIGH"
            cluster_name = best_name
            required_gap = 0.03

        candidates = [
            {"name": name, "score": float(info["effective"])}
            for name, info in sorted_scores[:4]
        ]
        self.last_assignment = {
            "decision": decision,
            "current_cluster": cluster_name,
            "score": best_score,
            "gap": gap,
            "required_gap": required_gap,
            "candidates": candidates,
            "ambiguous": decision == "AMBIGUOUS",
        }

        return cluster_name, best_score, decision

    def merge_clusters(self):
        """Post-batch: merge clusters only when ALL THREE conditions hold:

        1. centroid_sim  >= MERGE_THRESHOLD
        2. sample_matches >= 2   (at least 2 cross-cluster sample pairs above T)
        3. mean_sample_sim >= MERGE_THRESHOLD - 0.03

        This triple guard prevents "two lucky outlier samples" triggering an
        irreversible merge.  Returns set of cluster names absorbed (deleted).
        """
        names = list(self.clusters.keys())
        merged = set()

        for i in range(len(names)):
            if names[i] in merged:
                continue
            for j in range(i + 1, len(names)):
                if names[j] in merged:
                    continue

                ci = self.clusters[names[i]]
                cj = self.clusters[names[j]]

                # Condition 1: centroid similarity
                centroid_sim = float(torch.dot(ci["centroid"], cj["centroid"]))
                if centroid_sim < self.merge_threshold:
                    continue

                # Conditions 2 & 3: pairwise sample similarity
                cross_sims = [
                    float(torch.dot(si, sj))
                    for si in ci["samples"]
                    for sj in cj["samples"]
                ]
                if not cross_sims:
                    continue

                # Anti-chain-link
                merged_samples = ci["samples"] + cj["samples"]
                sims = []
                for idx_s in range(len(merged_samples)):
                    for jdx_s in range(idx_s + 1, len(merged_samples)):
                        sims.append(
                            float(
                                torch.dot(merged_samples[idx_s], merged_samples[jdx_s])
                            )
                        )

                if not sims:
                    continue

                import numpy as np

                avg_sim = float(np.mean(sims))
                std_sim = float(np.std(sims))
                min_sim = float(np.min(sims))

                MIN_INTERNAL_SIM = 0.65
                MAX_INTERNAL_STD = 0.12
                MIN_PAIRWISE_SIM = 0.50

                if (
                    avg_sim <= MIN_INTERNAL_SIM
                    or std_sim >= MAX_INTERNAL_STD
                    or min_sim <= MIN_PAIRWISE_SIM
                ):
                    logger.info(
                        f"Merge blocked {names[j]}→{names[i]}: validation failed (avg={avg_sim:.3f}, min={min_sim:.3f}, std={std_sim:.3f})"
                    )
                    continue

                # All conditions satisfied — absorb j into i
                ci["samples"].extend(cj["samples"])
                ci["count"] += cj["count"]

                while len(ci["samples"]) > self.max_samples:
                    sims_to_c = [
                        float(torch.dot(s, ci["centroid"])) for s in ci["samples"]
                    ]
                    ci["samples"].pop(sims_to_c.index(max(sims_to_c)))

                self._recompute_centroid(names[i])
                merged.add(names[j])
                logger.info(
                    f"Merged {names[j]} -> {names[i]} "
                    f"(centroid={centroid_sim:.3f}, "
                    f"avg_sim={avg_sim:.3f}, "
                    f"min_sim={min_sim:.3f})"
                )

        for name in merged:
            del self.clusters[name]

        return merged

    def get_merge_suggestions(self):
        """Phase 2.6: Structural Cluster-to-Cluster Merge Suggestion."""
        names = list(self.clusters.keys())
        suggestions = []

        import numpy as np

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ci = self.clusters[names[i]]
                cj = self.clusters[names[j]]

                samples_i = ci["samples"]
                samples_j = cj["samples"]
                if not samples_i or not samples_j:
                    continue

                stacked_i = torch.stack(samples_i)
                stacked_j = torch.stack(samples_j)

                # Cross-similarity matrix between samples of i and j
                sims = torch.mm(stacked_i, stacked_j.t()).cpu().numpy().flatten()

                mean_sim = float(sims.mean())
                strong_edges = int((sims > 0.55).sum())

                if mean_sim > 0.58 and strong_edges >= 2:
                    suggestions.append(
                        {
                            "cluster_a": names[i],
                            "cluster_b": names[j],
                            "mean_sim": mean_sim,
                            "strong_edges": strong_edges,
                            "confidence": "HIGH" if mean_sim > 0.62 else "MEDIUM",
                        }
                    )

        # Sort by strongest mean similarity
        suggestions.sort(key=lambda x: x["mean_sim"], reverse=True)
        return suggestions

    @property
    def cluster_summary(self):
        """Return {name: count} dict for all clusters."""
        return {name: info["count"] for name, info in self.clusters.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Graph Intelligence Utilities
# ══════════════════════════════════════════════════════════════════════════════


def build_adjacency(sim_matrix, edge_thr=0.50, strong_thr=0.65):
    N = sim_matrix.shape[0]
    adjacency = {i: set() for i in range(N)}

    # Precompute neighbors above base threshold
    base_neighbors = [
        set(torch.where(sim_matrix[i] > edge_thr)[0].tolist()) - {i} for i in range(N)
    ]

    def is_hub(node, thr=0.55, max_links=3):
        links = sum(
            1 for j in range(N) if j != node and sim_matrix[node, j].item() > thr
        )
        return links > max_links

    def is_consistent_edge(i, j):
        shared = base_neighbors[i].intersection(base_neighbors[j])
        valid = 0
        for k in shared:
            if sim_matrix[i, k].item() > 0.55 and sim_matrix[j, k].item() > 0.55:
                if sim_matrix[i, j].item() > 0.55:
                    valid += 1
        return valid >= 1

    for i in range(N):
        for j in base_neighbors[i]:
            if j <= i:
                continue

            if not is_hub(i) and not is_hub(j):
                if is_consistent_edge(i, j):
                    adjacency[i].add(j)
                    adjacency[j].add(i)

    return adjacency


def get_components(adjacency):
    visited = set()
    components = []

    for node in adjacency:
        if node in visited:
            continue

        stack = [node]
        comp = []

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            stack.extend(adjacency[cur] - visited)

        components.append(comp)

    return components


def safe_cleanup(component, sim_matrix, weak_thr=0.40):
    # Build subgraph
    sub_adj = {i: set() for i in component}

    for i in component:
        for j in component:
            if i == j:
                continue

            sim = sim_matrix[i, j].item()

            # Keep edge if strong OR needed for connectivity
            if sim >= weak_thr:
                sub_adj[i].add(j)

    # Recompute components AFTER cleanup
    return get_components(sub_adj)


def is_valid_cluster(component, sim_matrix):
    if len(component) == 1:
        return True

    sims = []
    strong_edges = 0

    for i in component:
        for j in component:
            if i >= j:
                continue
            s = sim_matrix[i, j].item()
            sims.append(s)
            if s > 0.50:
                strong_edges += 1

    mean_sim = sum(sims) / len(sims)
    consistency = strong_edges / len(sims)

    if len(component) == 2:
        return mean_sim > 0.55

    return (mean_sim > 0.48) and (consistency > 0.6)


def would_break_cluster(cluster, node, sim_matrix):
    sims = [sim_matrix[node, m].item() for m in cluster]
    mean_sim = sum(sims) / len(sims)
    return mean_sim < 0.45


def expand_clusters(components, sim_matrix, expand_thr=0.48):
    N = sim_matrix.shape[0]

    # Track ownership
    assigned = {}
    for idx, comp in enumerate(components):
        for node in comp:
            assigned[node] = idx

    for node in range(N):
        if node in assigned:
            continue

        best_cluster = None
        best_links = 0

        for idx, comp in enumerate(components):
            links = sum(
                1 for member in comp if sim_matrix[node, member].item() > expand_thr
            )

            if links > best_links:
                best_links = links
                best_cluster = idx

        # Only assign if strong support
        if best_cluster is not None and best_links >= 2:
            if not would_break_cluster(components[best_cluster], node, sim_matrix):
                components[best_cluster].append(node)
                assigned[node] = best_cluster

    return components


def deduplicate(components):
    seen = set()
    clean = []

    for comp in components:
        unique = []
        for node in comp:
            if node not in seen:
                unique.append(node)
                seen.add(node)
        if unique:
            clean.append(unique)

    return clean


class ElephantEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models", "best_model v4.6.pth")
        self.gallery_path = os.path.join(base_dir, "models", "gallery_embeddings.pt")

        # Initialize Model (PHASE 1: New Embedding logic)
        self.model = get_reid_model()

        # Smart loader that handles ALL database formats
        self.gallery = {}
        if os.path.exists(self.gallery_path):
            data = torch.load(
                self.gallery_path, map_location=self.device, weights_only=False
            )

            if not data:
                pass
            # 1. New Pipeline Format: {"Identity": {"embeddings": Tensor, "adaptive_thr": float, ...}}
            elif (
                isinstance(list(data.values())[0], dict)
                and "embeddings" in list(data.values())[0]
            ):
                for identity, info in data.items():
                    if self._is_provisional_identity(identity):
                        continue
                    if isinstance(info, dict) and "embeddings" in info:
                        embs = info["embeddings"].to(self.device)
                    else:
                        # Fallback for mixed arrays (e.g. broken Auto-Enrollments)
                        embs = info.to(self.device)
                        if embs.dim() == 1:
                            embs = embs.unsqueeze(0)

                    # Use internal adder to reconstruct thresholds uniformly
                    self._add_to_gallery_internal(identity, embs)

            # 2. Complex Kaggle/Streamlit Format: {"embeddings": Tensor, "labels": Tensor, "idx_to_identity": dict}
            elif (
                isinstance(data, dict)
                and "embeddings" in data
                and not isinstance(list(data.values())[0], dict)
            ):
                embs = data["embeddings"].to(self.device)
                labels = data["labels"].to(self.device)
                id_map = data["idx_to_identity"]

                for idx, identity in id_map.items():
                    if self._is_provisional_identity(identity):
                        continue
                    mask = labels == idx
                    self._add_to_gallery_internal(identity, embs[mask])

            # 3. Simple Flat Format: {"Identity": Tensor}
            else:
                for identity, emb in data.items():
                    if self._is_provisional_identity(identity):
                        continue
                    if emb.dim() == 1:
                        emb = emb.unsqueeze(0)
                    self._add_to_gallery_internal(identity, emb.to(self.device))

        # Re-ID Model uses INFERENCE_TRANSFORM (224x224)
        self.transform = INFERENCE_TRANSFORM

        self.confidence_threshold = (
            0.40  # WII within-elephant max ~0.42; cross-elephant max ~0.33
        )
        self.MAX_BACKUPS = 5  # Rolling backup limit — change to keep more/fewer backups

    @staticmethod
    def _is_provisional_identity(identity):
        return isinstance(identity, str) and identity.startswith("Unknown_")

    def extract_embedding(self, image_path, return_crop=False, allow_fallback=True):
        """Phase 2: Extracts 256-D embedding using YOLO head detector and ConvNeXt-Tiny.
        If no head is detected, returns None (UI handles this natively).
        """
        try:
            # We must use OpenCV to load the image because detect_and_crop_head expects BGR
            import cv2
            from pipeline import detect_and_crop_head, crop_quality_score

            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                logger.error(f"Failed to read image: {image_path}")
                return None

            # Phase 2: New Head Detector Crop
            crop_rgb, is_fallback = detect_and_crop_head(
                img_bgr,
                allow_fallback=allow_fallback,
            )

            if crop_rgb is None:
                logger.warning(f"No head detected (fallback failed) for {image_path}")
                return None, False

            detection_meta = dict(crop_rgb.info.get("detection_meta", {}))
            quality_meta = crop_quality_score(crop_rgb, detection_meta=detection_meta)

            if quality_meta.get("weak"):
                logger.warning(
                    f"Weak crop rejected for {image_path}: "
                    f"score={quality_meta.get('score')} "
                    f"blur={quality_meta.get('blur')} "
                    f"contrast={quality_meta.get('contrast')}"
                )
            tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.model(tensor)
                # Verify L2 norm (model output should already be L2 normalized)
                norm_val = float(torch.norm(emb, p=2, dim=1).item())
                if abs(norm_val - 1.0) > 0.05:
                    logger.warning(f"Embedding norm is not 1! Norm: {norm_val}")
                # Ensure it's perfectly L2 normalized to 1.0
                emb = F.normalize(emb, p=2, dim=1)

            del tensor

            if return_crop:
                return emb, is_fallback, crop_rgb, quality_meta
            return emb, is_fallback
        except Exception as e:
            logger.error(f"Failed to extract embedding for {image_path}: {e}")
            raise

    def extract_embedding_from_saved_crop(self, image_path):
        """Embed an already-cropped review/output image without re-running detection.

        This is used when rebuilding clusters from files inside `Unknown_*` or
        review folders. Re-detecting on saved crops causes false rejections and
        inconsistent cluster rebuilds.
        """
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model(tensor)
            return emb, False
        except Exception as e:
            logger.error(f"Failed to extract crop embedding for {image_path}: {e}")
            return None, False

    def _add_to_gallery_internal(self, identity, embs):
        """Helper to store embeddings and compute adaptive identity thresholds dynamically."""
        thr = DIST_STRICT
        if len(embs) > 1:
            try:
                import numpy as np

                sims = F.cosine_similarity(embs.unsqueeze(1), embs.unsqueeze(0), dim=-1)
                triu_indices = torch.triu_indices(len(embs), len(embs), offset=1)
                sims_flat = sims[triu_indices[0], triu_indices[1]]
                dists = 1.0 - sims_flat.cpu().numpy()
                thr = np.percentile(dists, 90)
                thr = float(min(thr, DIST_LOOSE))
            except Exception:
                pass
        self.gallery[identity] = {"embeddings": embs, "adaptive_thr": thr}

    def _predict_from_emb(self, query_emb, is_fallback=False):
        """Phase 3: Core gallery lookup using multi-centroid open-set hybrid scoring."""
        if query_emb is not None:
            query_emb = query_emb.squeeze()
        if query_emb is None:
            return {"label": "Unknown", "score_pct": 0.0, "top_matches": [], "gap": 0.0}

        distances = []
        for identity, data in self.gallery.items():
            if "centroids" in data:
                centroids = data["centroids"]
                centroids = F.normalize(centroids, p=2, dim=1)
                sims_to_centroids = torch.mv(centroids, query_emb)
                k = min(3, len(sims_to_centroids))
                score = float(torch.topk(sims_to_centroids, k).values.mean())
            else:
                # Fallback to single mean or all embeddings
                all_db_embs = data["embeddings"]
                sims = F.cosine_similarity(query_emb.unsqueeze(0), all_db_embs)
                k = min(3, len(sims))
                score = float(torch.topk(sims, k).values.mean())

            # Extract Score and Cohesion directly
            cohesion = score
            all_embs = data.get("embeddings")
            if all_embs is not None and len(all_embs) > 0:
                sims = torch.mv(all_embs, query_emb)
                top3 = torch.sort(sims, descending=True).values[:3]
                score = float(top3[0])
                cohesion = float(top3.mean()) if len(top3) >= 2 else float(top3[0])

            distances.append(
                {
                    "name": identity,
                    "score": score,
                    "cohesion": cohesion,
                    "adaptive_thr": data.get("adaptive_thr", DIST_STRICT),
                }
            )

        if not distances:
            return {"label": "Unknown", "score_pct": 0.0, "top_matches": [], "gap": 0.0}

        distances.sort(key=lambda x: x["score"], reverse=True)
        top = distances[0]
        best_score = top["score"]

        valid_scores = [d["score"] for d in distances if d["score"] > 0.1]
        gap = valid_scores[0] - valid_scores[1] if len(valid_scores) >= 2 else 0.0

        # Same herd ambiguity check
        same_herd_ambiguity = False
        if len(distances) >= 2 and gap < 0.10:
            name1_parts = distances[0]["name"].split("_")
            name2_parts = distances[1]["name"].split("_")
            if len(name1_parts) >= 2 and len(name2_parts) >= 2:
                herd1 = f"{name1_parts[0]}_{name1_parts[1]}"
                herd2 = f"{name2_parts[0]}_{name2_parts[1]}"
                if herd1 == herd2 and distances[0]["name"] != distances[1]["name"]:
                    same_herd_ambiguity = True

        score_pct = max(0, min(100, int(best_score * 100)))

        HIGH_CONF = 0.88
        CANDIDATE = 0.75
        MARGIN = 0.05
        MIN_COHESION = 0.50

        cohesion = top.get("cohesion", best_score)
        
        if best_score >= HIGH_CONF:
            label = top["name"]
        elif best_score >= CANDIDATE:
            if gap >= MARGIN and cohesion >= MIN_COHESION and not same_herd_ambiguity:
                label = top["name"]
            else:
                label = "Unknown"  # Push to review
        else:
            label = "Unknown"  # Push to review

        return {
            "label": label,
            "score_pct": score_pct,
            "top_matches": distances[:5],
            "gap": gap,
        }

    def predict_image(self, image_path):
        """Returns (label, similarity_pct) for the best gallery match."""
        emb_res = self.extract_embedding(image_path)
        if emb_res is None or (isinstance(emb_res, tuple) and emb_res[0] is None):
            return "Unknown", 0.0
        if isinstance(emb_res, tuple):
            query_emb, is_fallback = emb_res[:2]
        else:
            query_emb, is_fallback = emb_res, False
        res = self._predict_from_emb(query_emb, is_fallback)
        return res["label"], res["score_pct"]

    def _stamp_watermark_image(self, img, dst_path, label, score_pct):
        """Saves a PIL image to dst_path with a small watermark pill at the bottom-right."""
        img = img.copy().convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        W, H = img.size

        line1 = label
        line2 = f"{score_pct:.1f}% match"

        # Font size scales with image height — no upper cap
        font_size = max(24, H // 40)
        try:
            font_bold = ImageFont.truetype("arialbd.ttf", font_size)
            font_light = ImageFont.truetype("arial.ttf", font_size - 2)
        except IOError:
            font_bold = ImageFont.load_default()
            font_light = font_bold

        # Measure each line using absolute bbox coords (left, top, right, bottom)
        b1 = draw.textbbox((0, 0), line1, font=font_bold)
        b2 = draw.textbbox((0, 0), line2, font=font_light)

        # Proportional padding/gap based on font size
        pad_x = font_size // 2
        pad_y = font_size // 2
        gap = font_size // 4
        margin = font_size // 2  # distance from image edge

        pill_w = max(b1[2] - b1[0], b2[2] - b2[0]) + pad_x * 2

        # True draw heights
        h1 = b1[3] - b1[1]
        h2 = b2[3] - b2[1]
        pill_h = pad_y + h1 + gap + h2 + pad_y

        # Pin y1 to image bottom; derive y0 upward (always in-bounds)
        x1 = W - margin
        y1 = H - margin
        x0 = x1 - pill_w
        y0 = y1 - pill_h

        # ── Semi-transparent navy pill ─────────────────────────────────────
        draw.rounded_rectangle(
            [x0, y0, x1, y1], radius=font_size // 4, fill=(10, 35, 64, 210)
        )

        # ── Draw text (subtract top-bearing so text sits correctly inside pill) ──
        tx = x0 + pad_x

        # Line 1 — white bold
        ty1 = y0 + pad_y - b1[1]
        draw.text((tx, ty1), line1, font=font_bold, fill=(255, 255, 255, 255))

        # Line 2 — gold (advance by h1 + gap, then correct for bearing)
        ty2 = ty1 + h1 + gap - b2[1]
        draw.text((tx, ty2), line2, font=font_light, fill=(197, 164, 78, 255))

        img.save(dst_path, quality=92)
        img.close()  # Force close just in case

    def _stamp_watermark(self, src_path, dst_path, label, score_pct):
        """Copies the image at src_path to dst_path with a small watermark pill."""
        img = Image.open(src_path).convert("RGB")
        self._stamp_watermark_image(img, dst_path, label, score_pct)
        img.close()

    def _remove_existing_file_routes(
        self, base_output_folder, filename, review_store=None
    ):
        """Remove stale copies of a reprocessed file from previous runs.

        Reusing the same output folder across batches can leave the same image in
        multiple destinations (Unknown_*, weak crop review, rejected). This
        helper removes those stale copies before the new routing decision is made.
        Returns the affected Unknown_* cluster names so they can be rebuilt from
        their remaining on-disk files.
        """
        affected_unknowns = set()
        if review_store is not None:
            review_store.remove_open_ambiguities_for_filenames([filename])

        if not os.path.isdir(base_output_folder):
            return affected_unknowns

        for category in os.listdir(base_output_folder):
            cat_path = os.path.join(base_output_folder, category)
            if not os.path.isdir(cat_path):
                continue
            candidate = os.path.join(cat_path, filename)
            if not os.path.exists(candidate):
                continue
            try:
                os.remove(candidate)
                if category.startswith("Unknown_"):
                    affected_unknowns.add(category)
            except OSError:
                continue
        return affected_unknowns

    def _rank_review_candidates(self, embedding, cluster_mgr, exclude_cluster=None,
                                  source_samples=None):
        """Rank all other clusters by similarity to the source cluster.

        Uses full sample-to-sample comparison (not just centroid) so that
        pose-variant singletons still find their correct match.  Also computes
        bridge paths — if source→X and X→target are both strong, the target
        gets a boost even when direct similarity is low.

        Args:
            embedding:       source cluster centroid (1-D tensor)
            cluster_mgr:     UnknownClusterManager instance
            exclude_cluster: name of source cluster to skip
            source_samples:  list of sample tensors from the source cluster
                             (if None, falls back to centroid-only)

        Returns:
            Top 5 candidates sorted by effective score (max of direct and bridge).
        """
        if source_samples is None:
            source_samples = [embedding]

        # ── Phase 1: Direct sample-to-sample scoring ──────────────────────
        raw_candidates = {}
        for name, cluster in cluster_mgr.clusters.items():
            if name == exclude_cluster:
                continue

            target_samples = cluster.get("samples", [])
            if not target_samples:
                target_samples = [cluster["centroid"]]

            # Max similarity across ALL source×target sample pairs
            # Initialize to -1.0 (not 0.0!) so negative-similarity clusters
            # are still tracked — extreme pose variance can produce negative
            # dot products between genuine same-identity images.
            max_sim = -1.0
            for s_emb in source_samples:
                for t_emb in target_samples:
                    sim = float(torch.dot(s_emb, t_emb))
                    if sim > max_sim:
                        max_sim = sim

            # Also check centroid-to-centroid as a baseline
            centroid_sim = float(torch.dot(embedding, cluster["centroid"]))
            best_direct = max(max_sim, centroid_sim)

            raw_candidates[name] = {
                "name": name,
                "score": best_direct,
                "max_member": best_direct,
                "centroid_sim": centroid_sim,
                "bridge_path": "",
                "bridge_score": 0.0,
                "effective": best_direct,
                "cohesion": 1.0,
                "count": int(cluster.get("count", len(cluster.get("samples", [])))),
            }

        # ── Phase 1.5: Encompass Gallery Identities for Merges ────────────
        for identity_name, data in self.gallery.items():
            if identity_name == exclude_cluster:
                continue

            target_samples = data.get("embeddings", [])
            if not isinstance(target_samples, list):
                if hasattr(target_samples, 'shape'):
                    target_samples = [emb for emb in target_samples]

            max_sim = -1.0
            for s_emb in source_samples:
                for t_emb in target_samples:
                    if hasattr(t_emb, "to"):
                        t_emb = t_emb.to(s_emb.device)
                    sim = float(torch.dot(s_emb, t_emb))
                    if sim > max_sim:
                        max_sim = sim

            # Base centroid approx (use best sample as proxy if no real centroid)
            centroid_sim = max_sim
            raw_candidates[identity_name] = {
                "name": identity_name,
                "score": max_sim,
                "max_member": max_sim,
                "centroid_sim": centroid_sim,
                "bridge_path": "",
                "bridge_score": 0.0,
                "effective": max_sim,
                "cohesion": 1.0, # Known gallery items are treated as perfectly cohesive
                "count": len(target_samples),
            }

        # ── Phase 2: Bridge path detection ────────────────────────────────
        # If source→X is strong and X→target is strong, boost target's score
        # even when source→target is weak (pose variance bypass)
        candidate_names = list(raw_candidates.keys())
        for bridge_name in candidate_names:
            bridge_cluster = cluster_mgr.clusters.get(bridge_name)
            if not bridge_cluster:
                continue
            bridge_samples = bridge_cluster.get("samples", [bridge_cluster["centroid"]])

            # How well does the source match the bridge?
            source_to_bridge = raw_candidates[bridge_name]["max_member"]
            if source_to_bridge < 0.25:
                continue  # bridge too weak to be useful

            for target_name in candidate_names:
                if target_name == bridge_name:
                    continue
                target_cluster = cluster_mgr.clusters.get(target_name)
                if not target_cluster:
                    continue
                target_samples = target_cluster.get("samples", [target_cluster["centroid"]])

                # How well does the bridge match the target?
                bridge_to_target = 0.0
                for b_emb in bridge_samples:
                    for t_emb in target_samples:
                        sim = float(torch.dot(b_emb, t_emb))
                        if sim > bridge_to_target:
                            bridge_to_target = sim

                # Bridge score = geometric mean of two legs (penalises weak legs)
                bridge_score = (source_to_bridge * bridge_to_target) ** 0.5

                if bridge_score > raw_candidates[target_name]["bridge_score"]:
                    raw_candidates[target_name]["bridge_score"] = bridge_score
                    raw_candidates[target_name]["bridge_path"] = (
                        f"{exclude_cluster} → {bridge_name} → {target_name} "
                        f"({source_to_bridge:.3f} × {bridge_to_target:.3f})"
                    )

        # ── Phase 3: Effective score = max(direct, bridge) ────────────────
        for cand in raw_candidates.values():
            if cand["max_member"] < 0:
                cand["effective"] = cand["max_member"]  # Preserve negative scores instead of flattening to 0 via bridge max
            else:
                cand["effective"] = max(cand["max_member"], cand["bridge_score"])

        all_cands = sorted(
            raw_candidates.values(), key=lambda x: x["effective"], reverse=True
        )
        
        # Guarantee all Unknown clusters are passed to the UI, regardless of rank
        unknown_cands = [c for c in all_cands if str(c["name"]).startswith("Unknown_")]
        gallery_cands = [c for c in all_cands if not str(c["name"]).startswith("Unknown_")]
        
        candidates = unknown_cands + gallery_cands[:30]

        # Debug: print all candidate scores for diagnosis
        if candidates:
            print(f"\n[DEBUG] _rank_review_candidates for exclude={exclude_cluster}")
            for c in candidates[:15]:  # limit terminal spam
                tag = ""
                if c["bridge_score"] > c["max_member"]:
                    tag = f" [BRIDGE BOOST: {c['bridge_path']}]"
                print(
                    f"  {c['name']:20s} | direct={c['max_member']:.4f} "
                    f"| bridge={c['bridge_score']:.4f} "
                    f"| effective={c['effective']:.4f} "
                    f"| count={c['count']}{tag}"
                )
            print(f"  Returning {len(unknown_cands)} unknowns + top gallery matches.")

        return candidates

    def _rebuild_unknown_cluster_from_folder(
        self, cluster_mgr, base_output_folder, cluster_name
    ):
        """Sync one unknown cluster's state back to the files currently on disk."""
        cluster_dir = os.path.join(base_output_folder, cluster_name)
        old_info = cluster_mgr.clusters.get(cluster_name, {})
        image_paths = []
        if os.path.isdir(cluster_dir):
            image_paths = [
                os.path.join(cluster_dir, name)
                for name in sorted(os.listdir(cluster_dir))
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

        if not image_paths:
            cluster_mgr.clusters.pop(cluster_name, None)
            return

        embeddings = []
        for path in image_paths:
            emb_res = self.extract_embedding_from_saved_crop(path)
            emb = emb_res[0] if isinstance(emb_res, tuple) else emb_res
            if emb is not None:
                embeddings.append(emb.squeeze(0).cpu())

        if not embeddings:
            cluster_mgr.clusters.pop(cluster_name, None)
            return

        cluster_mgr.clusters[cluster_name] = {
            "centroid": embeddings[0].clone(),
            "samples": [embeddings[0].clone()],
            "count": 1,
            "created_at": old_info.get("created_at", ""),
            "variance": 0.0,
        }
        for key, value in old_info.items():
            if key not in {"centroid", "samples", "count", "variance"}:
                cluster_mgr.clusters[cluster_name][key] = value
        cluster_mgr._recompute_centroid(cluster_name)
        for emb in embeddings[1:]:
            cluster_mgr._add_to_cluster(cluster_name, emb)

    def process_batch(self, input_folder, base_output_folder, progress_callback=None):
        """Classify a folder of images using a hybrid clustering strategy:

        1. Gallery matching   — known elephants matched first via confidence_threshold.
        2. Within-batch re-ranking — all unknown embeddings collected, then k-reciprocal
           re-ranking applied to refine pairwise distances.  Same elephant in different
           backgrounds score closer; background-confused pairs score further apart.
        3. Cross-session centroid matching — re-ranked groups matched against existing
           Unknown cluster centroids (incremental centroid clustering).
        4. Auto-enrollment — new clusters enrolled in gallery as provisional identities.

        Returns:
            auto_enrolled (set[str]): new cluster names added to gallery this session.
        """
        os.makedirs(base_output_folder, exist_ok=True)
        rejected_dir = os.path.join(base_output_folder, "_rejected_no_head")
        weak_crop_dir = os.path.join(base_output_folder, "_weak_crop_review")
        os.makedirs(rejected_dir, exist_ok=True)
        os.makedirs(weak_crop_dir, exist_ok=True)

        cluster_mgr = UnknownClusterManager(
            unknown_dir=base_output_folder,
            cluster_file=os.path.join(base_output_folder, "unknown_clusters.json"),
        )
        review_store = ReviewStore(base_output_folder)
        health = ClusterHealthMonitor()

        valid_exts = (".jpg", ".jpeg", ".png")
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]

        # If a file is reprocessed into the same output folder, clear its old
        # routed copies first so one image cannot appear in multiple buckets.
        touched_unknowns = set()
        queued_singletons = set()
        for filename in files:
            touched_unknowns.update(
                self._remove_existing_file_routes(
                    base_output_folder, filename, review_store=review_store
                )
            )
        for cluster_name in sorted(touched_unknowns):
            self._rebuild_unknown_cluster_from_folder(
                cluster_mgr, base_output_folder, cluster_name
            )

        pre_clusters = set(cluster_mgr.clusters.keys())

        # Snapshot centroid directions BEFORE this batch touches the clusters
        pre_centroids = {
            name: info["centroid"].clone()
            for name, info in cluster_mgr.clusters.items()
        }
        # Track how many images each cluster receives this batch
        batch_additions = {}

        # ── Phase 1: Gallery matching — sort known / unknown ─────────────────
        unknown_embs = []  # 1-D tensors
        unknown_files = []  # filenames
        unknown_paths = []  # full source paths
        unknown_crops = []  # PIL crops used for embedding
        batch_singletons = []  # review candidates after all assignments
        unknown_candidates = []  # rich dicts from _predict_from_emb

        for i, filename in enumerate(files):
            filepath = os.path.join(input_folder, filename)
            logger.info(f"Processing image {i + 1}/{len(files)}: {filename}")
            try:
                emb_res = self.extract_embedding(
                    filepath,
                    return_crop=True,
                    allow_fallback=False,
                )
                # Check for None explicitly taking into account the tuple return
                if emb_res is None or (
                    isinstance(emb_res, tuple) and emb_res[0] is None
                ):
                    # Distinguish no-head from weak-crop rejection so bad crops
                    # do not silently poison the unknown pool.
                    if (
                        isinstance(emb_res, tuple)
                        and len(emb_res) >= 4
                        and emb_res[2] is not None
                    ):
                        crop_rgb = emb_res[2]
                        quality_meta = emb_res[3] or {}
                        logger.info(
                            f"  -> Skipped: Weak crop review "
                            f"(score={quality_meta.get('score', 'n/a')})."
                        )
                        try:
                            self._stamp_watermark_image(
                                crop_rgb,
                                os.path.join(weak_crop_dir, filename),
                                "WeakCrop",
                                float(quality_meta.get("score", 0)) * 20.0,
                            )
                        except Exception:
                            try:
                                crop_rgb.save(os.path.join(weak_crop_dir, filename))
                            except Exception:
                                pass
                    else:
                        logger.info(f"  -> Skipped: No elephant head detected.")
                        try:
                            shutil.copy(filepath, os.path.join(rejected_dir, filename))
                        except Exception:
                            pass
                    if progress_callback:
                        progress_callback(int(((i + 1) / len(files)) * 70))
                    continue

                if isinstance(emb_res, tuple):
                    if len(emb_res) >= 4:
                        query_emb, is_fallback, crop_rgb, quality_meta = emb_res[:4]
                    elif len(emb_res) >= 3:
                        query_emb, is_fallback, crop_rgb = emb_res[:3]
                        quality_meta = {}
                    else:
                        query_emb, is_fallback = emb_res
                        crop_rgb = None
                        quality_meta = {}
                else:
                    query_emb, is_fallback, crop_rgb, quality_meta = (
                        emb_res,
                        False,
                        None,
                        {},
                    )

                res = self._predict_from_emb(query_emb, is_fallback)
                label, score_pct = res["label"], res["score_pct"]

                if label != "Unknown":
                    logger.info(f"  -> Match found: {label} ({score_pct}%)")
                    cat_folder = os.path.join(base_output_folder, label)
                    os.makedirs(cat_folder, exist_ok=True)
                    if crop_rgb is not None:
                        self._stamp_watermark_image(
                            crop_rgb,
                            os.path.join(cat_folder, filename),
                            label,
                            score_pct,
                        )
                    else:
                        self._stamp_watermark(
                            filepath,
                            os.path.join(cat_folder, filename),
                            label,
                            score_pct,
                        )
                else:
                    unknown_embs.append(query_emb.squeeze(0).cpu())
                    unknown_files.append(filename)
                    unknown_paths.append(filepath)
                    unknown_crops.append(crop_rgb)
                    unknown_candidates.append(res)

                if progress_callback:
                    progress_callback(int(((i + 1) / len(files)) * 70))

                del query_emb
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                raise e
                continue

        # ── Phase 2: Graph-Based Connected-Component Clustering ───────────
        #
        # Instead of greedy centroid assignment, let cluster structure emerge
        # from a similarity graph.  Same elephant in different poses stays
        # connected via transitive edges; different elephants stay separated.
        #
        # Tuned thresholds for 128-D v8.2 embedding space:
        GRAPH_EDGE_THRESHOLD = 0.47
        GRAPH_CLUSTER_MEAN_MIN = 0.48
        GRAPH_CLUSTER_MIN_SIM = 0.40

        proposed_merges = {}
        if unknown_embs:
            N = len(unknown_embs)
            logger.info(f"Graph Clustering: processing {N} unknown image(s)...")

            # STEP 0 - Stack and L2-normalise
            embs_tensor = torch.stack(unknown_embs)
            embs_tensor = embs_tensor / embs_tensor.norm(dim=1, keepdim=True)

            # STEP 1 - Full pairwise cosine similarity matrix
            sim_matrix = embs_tensor @ embs_tensor.T  # (N, N)

            if DEBUG_CLUSTERING:
                sims = sim_matrix[
                    ~torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
                ]
                logger.info("\n[DEBUG] Similarity Stats:")
                if sims.numel() > 0:
                    logger.info(
                        f"  mean={sims.mean():.3f} | std={sims.std():.3f} | min={sims.min():.3f} | max={sims.max():.3f}"
                    )
                else:
                    logger.info("  Not enough images for pairwise similarity stats")
                logger.info("\n[DEBUG] Top-3 neighbors per image:")
                for i in range(N):
                    row = sim_matrix[i].clone()
                    row[i] = -1
                    topk = torch.topk(row, k=min(3, max(1, N - 1)))
                    neighbors = [
                        (int(idx), float(val))
                        for val, idx in zip(topk.values, topk.indices)
                    ]
                    logger.info(
                        f"  Node {i} ({unknown_files[i]}): {[(unknown_files[idx], round(v, 3)) for idx, v in neighbors]}"
                    )

            # STEP 1.5 - Force Strong Merges (Pre-Graph Anchors)
            STRONG_MATCH_THRESHOLD = 0.72
            parent = list(range(N))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                pa, pb = find(a), find(b)
                if pa != pb:
                    parent[pb] = pa

            for i in range(N):
                for j in range(i + 1, N):
                    if sim_matrix[i, j].item() > STRONG_MATCH_THRESHOLD:
                        union(i, j)

            strong_groups = {}
            for i in range(N):
                root = find(i)
                strong_groups.setdefault(root, []).append(i)

            valid_clusters = list(strong_groups.values())

            if DEBUG_CLUSTERING:
                logger.info("\n[DEBUG] Strong Anchor Groups (Union-Find):")
                for root, members in strong_groups.items():
                    if len(members) > 1:
                        logger.info(
                            f"  Root {root}: {[unknown_files[m] for m in members]}"
                        )

            # Now expand the strong anchors using the looser graph threshold
            final_groups = expand_clusters(valid_clusters, sim_matrix, expand_thr=0.47)

            # Deduplicate and ensure all nodes are assigned
            final_groups = deduplicate(final_groups)

            assigned_nodes = set()
            for group in final_groups:
                assigned_nodes.update(group)

            if DEBUG_CLUSTERING:
                logger.info("\n[DEBUG] Final Batch Cluster Summary (Pre-Gallery):")
                for idx, members in enumerate(final_groups):
                    if len(members) <= 1:
                        logger.info(
                            f"  Cluster {idx}: {[unknown_files[m] for m in members]} (singleton)"
                        )
                        continue
                    sub = sim_matrix[members][:, members]
                    mask = ~torch.eye(len(members), dtype=torch.bool, device=sub.device)
                    vals = sub[mask]
                    logger.info(
                        f"  Cluster {idx}: {[unknown_files[m] for m in members]}"
                    )
                    logger.info(
                        f"    mean={vals.mean():.3f} | min={vals.min():.3f} | std={vals.std():.3f}"
                    )

                logger.info(
                    "\n[DEBUG] Missed Strong Pairs (>0.70 but different clusters):"
                )

                def find_cluster(idx):
                    for i, g in enumerate(final_groups):
                        if idx in g:
                            return i
                    return -1

                for i in range(N):
                    for j in range(i + 1, N):
                        if sim_matrix[i, j].item() > 0.70:
                            if find_cluster(i) != find_cluster(j):
                                logger.info(
                                    f"  \u274c {unknown_files[i]} - {unknown_files[j]}: {sim_matrix[i, j].item():.3f}"
                                )

            for i in range(N):
                if i not in assigned_nodes:
                    final_groups.append([i])
            final_groups = deduplicate(final_groups)

            def is_ambiguous_cluster(component, sim_matrix):
                if len(component) <= 1:
                    return False
                sims = []
                for i in component:
                    for j in component:
                        if i >= j:
                            continue
                        sims.append(sim_matrix[i, j].item())
                if not sims:
                    return False
                mean_sim = sum(sims) / len(sims)
                min_sim = min(sims)
                import numpy as np

                std_sim = float(np.std(sims))
                return std_sim > 0.12 or (mean_sim < 0.60 and min_sim < 0.45)

            def refine_ambiguous_cluster(comp, sim_matrix):
                if len(comp) <= 2:
                    return [[x] for x in comp]

                # Structural integrity check: A node must have at least one relatively strong 
                # neighbor (>= 0.65) to formally belong to a group, preventing bridging artifacts.
                refined = []
                leftovers = []
                
                for node in comp:
                    peer_sims = [sim_matrix[node, other].item() for other in comp if other != node]
                    max_peer_sim = max(peer_sims) if peer_sims else 0.0
                    
                    if max_peer_sim >= 0.52:
                        refined.append(node)
                    else:
                        leftovers.append(node)

                if len(refined) > 1:
                    return [refined] + [[x] for x in leftovers]
                else:
                    return [[x] for x in comp]

            reviewed_groups = []
            for group in final_groups:
                if is_ambiguous_cluster(group, sim_matrix):
                    logger.info(
                        f"  [Ambiguity] Cluster of size {len(group)} flagged as ambiguous. Refining..."
                    )
                    refined_subgroups = refine_ambiguous_cluster(group, sim_matrix)
                    reviewed_groups.extend(refined_subgroups)
                else:
                    reviewed_groups.append(group)
            final_groups = reviewed_groups

            # System-level safety layer: cluster-to-gallery conflict check
            group_candidate_matches = {}
            for group in final_groups:
                if len(group) >= 2:
                    g_embs = torch.stack([unknown_embs[i] for i in group])
                    g_centroid = F.normalize(g_embs.mean(dim=0), p=2, dim=0)

                    # Test against gallery
                    max_sim = 0.0
                    conflict_id = None
                    for identity, data in self.gallery.items():
                        c_sim = 0.0
                        if "centroids" in data:
                            centroids = F.normalize(data["centroids"], p=2, dim=1)
                            c_sim = float(
                                torch.mv(
                                    centroids, g_centroid.to(centroids.device)
                                ).max()
                            )
                        else:
                            all_db = data["embeddings"]
                            c_sim = float(
                                F.cosine_similarity(
                                    g_centroid.unsqueeze(0).to(all_db.device), all_db
                                ).max()
                            )
                        if c_sim > max_sim:
                            max_sim = c_sim
                            conflict_id = identity

                    if max_sim > 0.75:
                        # Convert to tuple for safety hashable usage
                        group_candidate_matches[tuple(group)] = conflict_id

            grouped_items = [(g, False, 0.0) for g in final_groups]
            grouped_items.sort(key=lambda x: len(x[0]), reverse=True)

            for all_img_idx, is_auto, avg_w in grouped_items:
                member_embs = [unknown_embs[i] for i in all_img_idx]
                member_files = [unknown_files[i] for i in all_img_idx]
                member_paths = [unknown_paths[i] for i in all_img_idx]
                member_crops = [unknown_crops[i] for i in all_img_idx]

                # Legacy Assignment logic restored for singletons
                # Provides aggressive grouping (0.52 threshold) vs V8.2 strict threshold
                if len(member_embs) == 1:
                    cluster_name, score, _ = cluster_mgr.assign(member_embs[0])
                else:
                    cluster_name, score = cluster_mgr.assign_group(member_embs)
                decision = cluster_mgr.last_assignment.get("decision", "GROUP")
                assignment_meta = cluster_mgr.last_assignment or {}

                if decision == "AMBIGUOUS" and cluster_name is None:
                    ranked = assignment_meta.get("candidates", [])
                    c1 = ranked[0] if len(ranked) > 0 else {}
                    c2 = ranked[1] if len(ranked) > 1 else {}

                    # Assign to the top candidate cluster so the image is visible
                    # in the UI grid. The human review entry will still flag it.
                    target_cluster_name = c1.get("name")
                    if not target_cluster_name:
                        # Fallback: create a new cluster if no candidate
                        target_cluster_name = cluster_mgr._create_cluster(member_embs[0])

                    cluster_folder = os.path.join(
                        base_output_folder, target_cluster_name
                    )
                    os.makedirs(cluster_folder, exist_ok=True)

                    for fpath, fname, crop_rgb, emb in zip(
                        member_paths, member_files, member_crops, member_embs
                    ):
                        target_path = os.path.join(cluster_folder, fname)
                        import shutil
                        try:
                            shutil.copy(fpath, target_path)
                            if crop_rgb is not None:
                                crop_dir = os.path.join(cluster_folder, ".crops")
                                os.makedirs(crop_dir, exist_ok=True)
                                crop_rgb.save(os.path.join(crop_dir, fname))
                        except Exception as e:
                            logger.error(f"Failed to copy original image {fname}: {e}")

                        # Add embedding to cluster so suggestions work
                        cluster_mgr._add_to_cluster(target_cluster_name, emb)

                    # Queue for human review with both candidates visible
                    for emb, fname in zip(member_embs, member_files):
                        review_store.add_ambiguity(
                            {
                                "source_type": "single",
                                "review_reason": "AMBIGUOUS_TIE",
                                "current_cluster": target_cluster_name,
                                "file_paths": [os.path.join(cluster_folder, fname)],
                                "source_filenames": [fname],
                                "image_embedding": emb.tolist(),
                                "candidate_a": {"name": c1.get("name"), "score": c1.get("score")},
                                "candidate_b": {"name": c2.get("name"), "score": c2.get("score")},
                                "top_candidates": ranked,
                                "best_score": float(c1.get("score", 0.0)),
                                "gap": float(c1.get("score", 0.0) - c2.get("score", 0.0)),
                                "decision": "AMBIGUOUS",
                            }
                        )

                    logger.info(
                        f"  [ambiguous] {member_files[0]} tied between {c1.get('name')} and {c2.get('name')}"
                        f" → assigned to {target_cluster_name} for review"
                    )
                    continue

                cluster_score_pct = round((score + 1) / 2 * 100, 1)
                cluster_folder = os.path.join(base_output_folder, cluster_name)
                os.makedirs(cluster_folder, exist_ok=True)

                for fpath, fname, crop_rgb in zip(
                    member_paths, member_files, member_crops
                ):
                    import shutil
                    try:
                        shutil.copy(fpath, os.path.join(cluster_folder, fname))
                        if crop_rgb is not None:
                            crop_dir = os.path.join(cluster_folder, ".crops")
                            os.makedirs(crop_dir, exist_ok=True)
                            crop_rgb.save(os.path.join(crop_dir, fname))
                    except Exception as e:
                        logger.error(f"Failed to copy original image {fname}: {e}")

                batch_additions[cluster_name] = batch_additions.get(
                    cluster_name, 0
                ) + len(all_img_idx)

                # Cluster-to-gallery conflict logic
                if len(member_embs) >= 2:
                    conflict_id = group_candidate_matches.get(tuple(all_img_idx))
                    if conflict_id:
                        review_store.add_ambiguity(
                            {
                                "source_type": "cluster_conflict",
                                "review_reason": "REVIEW_CANDIDATE_MATCH",
                                "current_cluster": cluster_name,
                                "file_paths": [
                                    os.path.join(cluster_folder, f)
                                    for f in member_files
                                ],
                                "source_filenames": member_files,
                                "image_embedding": member_embs[0].tolist(),
                                "candidate_a": {
                                    "name": conflict_id,
                                    "score": 0.75,
                                },  # Approximate
                                "decision": "REVIEW_CANDIDATE_MATCH",
                            }
                        )

                if len(member_embs) == 1:
                    orig_res = unknown_candidates[all_img_idx[0]]
                    batch_singletons.append(
                        {
                            "filename": member_files[0],
                            "embedding": member_embs[0].clone(),
                            "cluster_name": cluster_name,
                            "file_path": os.path.join(cluster_folder, member_files[0]),
                            "gallery_matches": orig_res.get("top_matches", []),
                        }
                    )

                if assignment_meta.get("ambiguous"):
                    ranked = assignment_meta.get("candidates", [])
                    candidate_a = dict(ranked[0]) if len(ranked) > 0 else {}
                    candidate_b = dict(ranked[1]) if len(ranked) > 1 else {}
                    if candidate_a.get("name") in cluster_mgr.clusters:
                        candidate_a["variance"] = float(
                            cluster_mgr.clusters[candidate_a["name"]].get(
                                "variance", 0.0
                            )
                        )
                    if candidate_b.get("name") in cluster_mgr.clusters:
                        candidate_b["variance"] = float(
                            cluster_mgr.clusters[candidate_b["name"]].get(
                                "variance", 0.0
                            )
                        )

                    for emb, fname in zip(member_embs, member_files):
                        review_store.add_ambiguity(
                            {
                                "source_type": "single",
                                "current_cluster": cluster_name,
                                "file_paths": [os.path.join(cluster_folder, fname)],
                                "source_filenames": [fname],
                                "image_embedding": emb.tolist(),
                                "candidate_a": candidate_a,
                                "candidate_b": candidate_b,
                                "top_candidates": ranked,
                                "best_score": float(
                                    assignment_meta.get("score", score)
                                ),
                                "gap": float(assignment_meta.get("gap", 0.0)),
                                "decision": assignment_meta.get("decision", decision),
                            }
                        )
                elif (
                    len(member_embs) == 1
                    and assignment_meta.get("decision") == "UNKNOWN"
                ):
                    ranked = assignment_meta.get("candidates", [])
                    # Soft-review path: unresolved near-matches should still
                    # reach the human inbox even when they are not a true top-2
                    # tie. This is where pairs like a2(20)/a2(44) live.
                    ranked_for_review = sorted(
                        ranked,
                        key=lambda cand: (
                            -(
                                float(cand.get("score", 0.0))
                                + 0.06
                                * min(
                                    max(
                                        int(
                                            cluster_mgr.clusters.get(
                                                cand.get("name", ""), {}
                                            ).get("count", 0)
                                        )
                                        - 1,
                                        0,
                                    ),
                                    3,
                                )
                            ),
                            -int(
                                cluster_mgr.clusters.get(cand.get("name", ""), {}).get(
                                    "count", 0
                                )
                            ),
                        ),
                    )
                    candidate_a = (
                        dict(ranked_for_review[0]) if len(ranked_for_review) > 0 else {}
                    )
                    candidate_b = (
                        dict(ranked_for_review[1]) if len(ranked_for_review) > 1 else {}
                    )
                    top_score = float(candidate_a.get("score", 0.0))
                    second_score = float(candidate_b.get("score", 0.0))
                    candidate_cluster_size = 0
                    if candidate_a.get("name") in cluster_mgr.clusters:
                        candidate_cluster_size = int(
                            cluster_mgr.clusters[candidate_a["name"]].get("count", 0)
                        )
                    if top_score > 0.35 or top_score >= 0.55 or (
                        candidate_cluster_size >= 2 and top_score >= 0.50
                    ):
                        if candidate_a.get("name") in cluster_mgr.clusters:
                            candidate_a["variance"] = float(
                                cluster_mgr.clusters[candidate_a["name"]].get(
                                    "variance", 0.0
                                )
                            )
                        if candidate_b.get("name") in cluster_mgr.clusters:
                            candidate_b["variance"] = float(
                                cluster_mgr.clusters[candidate_b["name"]].get(
                                    "variance", 0.0
                                )
                            )

                        queued_singletons.add(cluster_name)
                        review_store.add_ambiguity(
                            {
                                "source_type": "soft_match",
                                "review_reason": "low_confidence_candidate",
                                "current_cluster": cluster_name,
                                "file_paths": [
                                    os.path.join(cluster_folder, member_files[0])
                                ],
                                "source_filenames": [member_files[0]],
                                "image_embedding": member_embs[0].tolist(),
                                "candidate_a": candidate_a,
                                "candidate_b": candidate_b,
                                "top_candidates": ranked,
                                "best_score": top_score,
                                "gap": float(top_score - second_score),
                                "decision": "REVIEW",
                            }
                        )
                        logger.info(
                            f"  [review] queued soft-match for {member_files[0]} "
                            f"against {candidate_a.get('name', 'n/a')} ({top_score:.3f})"
                        )

                size = len(all_img_idx)
                logger.info(
                    f"  [centroid-grouped] "
                    f"{size} image(s) -> {cluster_name} "
                    f"(score={cluster_score_pct}%)"
                )

            # After the full batch is assigned, queue review items for any
            # singletons that still have plausible alternative clusters. This
            # catches symmetric same-batch pairs like a2(20)/a2(44) that are
            # invisible during the first singleton's assignment because the
            # partner cluster does not exist yet.
            for record in batch_singletons:
                current_name = record["cluster_name"]
                current_info = cluster_mgr.clusters.get(current_name)
                if not current_info or int(current_info.get("count", 0)) != 1:
                    continue

                ranked = self._rank_review_candidates(
                    record["embedding"],
                    cluster_mgr,
                    exclude_cluster=current_name,
                )
                if not ranked:
                    continue

                candidate_a = dict(ranked[0])
                candidate_b = dict(ranked[1]) if len(ranked) > 1 else {}
                top_score = float(candidate_a.get("score", 0.0))
                second_score = float(candidate_b.get("score", 0.0))
                candidate_count = int(candidate_a.get("count", 0))

                should_review = top_score > 0.35
                if not should_review or current_name in queued_singletons:
                    continue

                if top_score >= 0.72:
                    # Force merge instead of reviewing
                    target_name = candidate_a["name"]
                    target_info = cluster_mgr.clusters.get(target_name)
                    if target_info:
                        # Direct assignment bypassing review
                        # Move file and update cluster
                        target_dir = os.path.join(base_output_folder, target_name)
                        source_dir = os.path.join(base_output_folder, current_name)
                        fname = record["filename"]
                        src_path = record["file_path"]
                        dst_path = os.path.join(target_dir, fname)
                        if os.path.exists(src_path):
                            import shutil

                            shutil.move(src_path, dst_path)
                            cluster_mgr._add_to_cluster(
                                target_name, record["embedding"]
                            )

                            # Clean up old singleton
                            del cluster_mgr.clusters[current_name]
                            try:
                                os.rmdir(source_dir)
                            except OSError:
                                pass
                            cluster_mgr._save_clusters()
                            logger.info(
                                f"  [auto-merge] Postpass merged {fname} ({current_name}) -> {target_name} ({top_score:.3f})"
                            )
                        continue

                review_store.add_ambiguity(
                    {
                        "source_type": "singleton_postpass",
                        "review_reason": "post_batch_singleton_candidate",
                        "current_cluster": current_name,
                        "file_paths": [record["file_path"]],
                        "source_filenames": [record["filename"]],
                        "image_embedding": record["embedding"].tolist(),
                        "candidate_a": candidate_a,
                        "candidate_b": candidate_b,
                        "top_candidates": ranked[:4],
                        "best_score": top_score,
                        "gap": float(top_score - second_score),
                        "decision": "REVIEW",
                    }
                )
                logger.info(
                    f"  [review-post] queued singleton review for {record['filename']} "
                    f"against {candidate_a.get('name', 'n/a')} ({top_score:.3f})"
                )

        if progress_callback:
            progress_callback(90)

        # Post-batch: merge clusters that converged to the same identity
        merged = cluster_mgr.merge_clusters()
        if merged:
            logger.info(f"Post-batch merge: absorbed {merged}")

        # ── Health checks (growth + stability) ───────────────────────────────
        for name, cluster in cluster_mgr.clusters.items():
            # Growth guard
            additions = batch_additions.get(name, 0)
            if additions > 0 and name in pre_centroids:
                growth_ok = health.check_growth(
                    name,
                    additions,
                    pre_centroids[name],
                    cluster["centroid"],
                )
                cluster["growth_warning"] = growth_ok
            else:
                cluster.setdefault("growth_warning", False)

            # Stability score
            if len(cluster["samples"]) >= 2:
                stats = health.compute_stability(name, cluster["samples"])
                cluster["stability_flag"] = stats["flagged"]
                cluster["stability_ratio"] = stats["ratio"]
                cluster["stability_min"] = stats["min_sim"]
            else:
                cluster.setdefault("stability_flag", False)
                cluster.setdefault("stability_ratio", 0.0)
                cluster.setdefault("stability_min", 1.0)

        cluster_mgr.save()
        logger.info("Batch processing complete. Cluster state saved.")

        # --- Phase 2.6: Generate Structural Merge Suggestions ---
        structural_merges = cluster_mgr.get_merge_suggestions()
        if structural_merges:
            logger.info("\n[DEBUG] \U0001f517 Suggested Structural Merges:")
            for m in structural_merges:
                logger.info(f"  {m['cluster_a']} <-> {m['cluster_b']}")
                logger.info(f"    Confidence: {m['confidence']}")
                logger.info(
                    f"    Reason: mean_sim={m['mean_sim']:.3f}, strong_edges={m['strong_edges']}"
                )

                key = f"{m['cluster_a']}_{m['cluster_b']}"
                if key not in proposed_merges:
                    proposed_merges[key] = []
                
                proposed_merges[key].append({
                    "type": "structural_merge",
                    "cluster_a": m["cluster_a"],
                    "cluster_b": m["cluster_b"],
                    "avg_weight": m["mean_sim"], # For app.py UI
                    "min_weight": m.get("min_sim", 0.0), # For app.py UI
                    "n_merged": 2,
                    "mean_sim": m["mean_sim"],
                    "strong_edges": m["strong_edges"],
                    "confidence": m["confidence"],
                })

        if progress_callback:
            progress_callback(95)

        # Keep new unknown clusters provisional until a human explicitly
        # promotes them. Auto-enrolling unknowns into the gallery causes future
        # runs to "confirm" unreviewed clusters as if they were known IDs.
        new_clusters = set(cluster_mgr.clusters.keys()) - pre_clusters
        auto_enrolled = set()
        if AUTO_ENROLL_PROVISIONAL:
            for name in new_clusters:
                centroid = cluster_mgr.clusters[name]["centroid"]
                self._add_to_gallery_internal(
                    name, centroid.unsqueeze(0).to(self.device)
                )
                auto_enrolled.add(name)
                logger.info(f"Auto-enrolled provisional identity: {name}")

            if auto_enrolled:
                self._save_gallery_with_backup()
                logger.info(
                    f"Gallery updated: {len(auto_enrolled)} provisional identit(ies)."
                )
        elif new_clusters:
            logger.info(
                f"Kept {len(new_clusters)} cluster(s) provisional; awaiting manual promotion."
            )

        if progress_callback:
            progress_callback(100)

        return auto_enrolled, proposed_merges

    def _save_gallery_with_backup(self):
        """Saves gallery_embeddings.pt with a rolling timestamped backup.

        Backups are stored in a 'backups/' subfolder next to the database file.
        Only the most recent MAX_BACKUPS files are retained; older ones are deleted.
        """
        # ── 1. Create backup of existing file ──────────────────────────────────
        if os.path.exists(self.gallery_path):
            backup_dir = os.path.join(os.path.dirname(self.gallery_path), "backups")
            os.makedirs(backup_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"gallery_embeddings_backup_{timestamp}.pt"
            shutil.copy(self.gallery_path, os.path.join(backup_dir, backup_name))

            # ── 2. Prune old backups, keep only the newest MAX_BACKUPS ──────────
            backups = sorted(
                [
                    f
                    for f in os.listdir(backup_dir)
                    if f.startswith("gallery_embeddings_backup_")
                ],
                reverse=True,  # newest first (lexicographic == chronological for YYYYMMDD_HHMMSS)
            )
            for old_backup in backups[self.MAX_BACKUPS :]:
                os.remove(os.path.join(backup_dir, old_backup))

        # ── 3. Save the updated database ───────────────────────────────────────
        torch.save(self.gallery, self.gallery_path)

    def find_similar_elephants(self, folder_path, threshold=0.80):
        """Checks if images in a folder visually match any elephant already in the database.

        Returns a list of (elephant_id, similarity_percentage) for all matches
        above the given threshold, sorted by similarity descending.
        """
        embeddings = []
        valid_exts = (".jpg", ".jpeg", ".png")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_exts):
                filepath = os.path.join(folder_path, filename)
                emb_res = self.extract_embedding(filepath)
                emb = emb_res[0] if isinstance(emb_res, tuple) else emb_res
                if emb is not None:
                    embeddings.append(emb)

        if not embeddings:
            return []

        # Average the new images into a single query vector
        query = torch.mean(torch.stack(embeddings), dim=0)
        query = F.normalize(query, p=2, dim=1)

        matches = []
        for elephant_id, data in self.gallery.items():
            db_embs = data["embeddings"]
            scores = torch.matmul(query, db_embs.T)
            max_score = scores.max().item()
            similarity_pct = (max_score + 1) / 2 * 100  # map [-1,1] → [0%,100%]
            if similarity_pct >= threshold * 100:
                matches.append((elephant_id, similarity_pct))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def update_database(self, new_folder_path, elephant_id):
        """Enrol or enrich an elephant in the gallery.

        New elephant ID  → averages the new images into a fresh gallery entry.
        Existing ID      → appends the new image embeddings to the existing
                           gallery so the max() lookup keeps full coverage.

        Returns (success: bool, is_update: bool).
        """
        embeddings = []
        valid_exts = (".jpg", ".jpeg", ".png")
        for filename in os.listdir(new_folder_path):
            if filename.lower().endswith(valid_exts):
                filepath = os.path.join(new_folder_path, filename)
                emb_res = self.extract_embedding(filepath)
                emb = emb_res[0] if isinstance(emb_res, tuple) else emb_res
                if emb is not None:
                    embeddings.append(emb)

        if not embeddings:
            return False, False

        is_update = elephant_id in self.gallery

        if is_update:
            new_embs = torch.stack([e.squeeze(0) for e in embeddings])  # (n_new, dim)
            new_embs = F.normalize(new_embs, p=2, dim=1).to(self.device)
            existing = self.gallery[elephant_id]["embeddings"]
            merged = torch.cat([existing, new_embs], dim=0)
            self._add_to_gallery_internal(elephant_id, merged)
        else:
            new_embs = torch.stack([e.squeeze(0) for e in embeddings])
            self._add_to_gallery_internal(
                elephant_id, F.normalize(new_embs, p=2, dim=1).to(self.device)
            )

        self._save_gallery_with_backup()
        return True, is_update
