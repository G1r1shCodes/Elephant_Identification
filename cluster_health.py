"""
cluster_health.py — ClusterHealthMonitor

Provides four safety checks for the elephant re-identification system:

    check_growth()       — flag clusters that grew too fast (possible contamination)
    compute_stability()  — flag clusters whose internal similarity is too dispersed
    evaluate_merge()     — triple-condition merge guard (separate from manager)
    detect_duplicates()  — cross-check confirmed gallery identities for possible dupes

Designed to be used both by core_engine.py (per-batch) and by
app.py (startup / periodic checks).
"""

import logging
from itertools import combinations

import torch
import torch.nn.functional as F

logger = logging.getLogger("ElephantEngine")

# ── Calibrated thresholds for WII wildlife images ─────────────────────────────
# Within-elephant similarity range: 0.29 – 0.42
# Cross-elephant max:               0.31 – 0.33

STABILITY_RATIO_THRESHOLD = 0.18   # std/mean > 0.18  → ⚠ unstable
STABILITY_MIN_SIM         = 0.25   # min pairwise sim < 0.25 → force-flag
GROWTH_MIN_ADDITIONS      = 5      # batch additions needed to trigger check
GROWTH_CENTROID_SHIFT     = 0.04   # 1 - cosine(old, new) >= this → flag
DUPLICATE_THRESHOLD       = 0.40   # centroid sim >= this → possible duplicate


class ClusterHealthMonitor:
    """Encapsulates all cluster safety checks in one place.

    Usage:
        monitor = ClusterHealthMonitor()

        # After batch assignment
        monitor.check_growth(cluster_name, additions, old_centroid, new_centroid)
        monitor.compute_stability(cluster_name, samples)

        # After gallery update
        flags = monitor.detect_duplicates(gallery)

        # Before any merge
        ok = monitor.evaluate_merge(cluster_i, cluster_j, merge_threshold)
    """

    def __init__(
        self,
        stability_ratio   = STABILITY_RATIO_THRESHOLD,
        stability_min_sim = STABILITY_MIN_SIM,
        growth_additions  = GROWTH_MIN_ADDITIONS,
        growth_shift      = GROWTH_CENTROID_SHIFT,
        duplicate_thresh  = DUPLICATE_THRESHOLD,
    ):
        self.stability_ratio   = stability_ratio
        self.stability_min_sim = stability_min_sim
        self.growth_additions  = growth_additions
        self.growth_shift      = growth_shift
        self.duplicate_thresh  = duplicate_thresh

    # ── Growth Guard ───────────────────────────────────────────────────────────

    def check_growth(
        self,
        cluster_name: str,
        additions: int,
        old_centroid: torch.Tensor,
        new_centroid: torch.Tensor,
    ) -> bool:
        """Return True if this cluster's growth looks suspicious.

        Conditions (both must hold):
            additions >= GROWTH_MIN_ADDITIONS
            centroid shift = 1 - cosine(old, new) >= GROWTH_CENTROID_SHIFT

        A large batch of images from the *same* elephant barely moves the
        centroid. Mixed-identity contamination moves it significantly.
        """
        if additions < self.growth_additions:
            return False

        shift = 1.0 - float(torch.dot(
            F.normalize(old_centroid, p=2, dim=0),
            F.normalize(new_centroid, p=2, dim=0),
        ))
        flagged = shift >= self.growth_shift
        if flagged:
            logger.warning(
                f"[HealthMonitor] Growth warning on '{cluster_name}': "
                f"{additions} additions, centroid shift={shift:.4f}"
            )
        return flagged

    # ── Stability Score ────────────────────────────────────────────────────────

    def compute_stability(
        self,
        cluster_name: str,
        samples: list,
    ) -> dict:
        """Compute pairwise similarity statistics over a cluster's sample set.

        Returns a dict with keys:
            mean_sim   (float)
            std_sim    (float)
            min_sim    (float)
            ratio      (float)  std / mean
            flagged    (bool)   True if cluster looks unstable

        Flags if:
            ratio > STABILITY_RATIO_THRESHOLD  OR  min_sim < STABILITY_MIN_SIM
        """
        n = len(samples)
        if n < 2:
            return {
                "mean_sim": 1.0, "std_sim": 0.0,
                "min_sim": 1.0, "ratio": 0.0, "flagged": False,
            }

        sims = [
            float(torch.dot(samples[i], samples[j]))
            for i in range(n)
            for j in range(i + 1, n)
        ]

        mean_sim = sum(sims) / len(sims)
        variance = sum((s - mean_sim) ** 2 for s in sims) / len(sims)
        std_sim  = variance ** 0.5
        min_sim  = min(sims)
        ratio    = std_sim / max(mean_sim, 1e-8)

        flagged = (ratio > self.stability_ratio) or (min_sim < self.stability_min_sim)
        if flagged:
            logger.warning(
                f"[HealthMonitor] Unstable cluster '{cluster_name}': "
                f"mean={mean_sim:.3f}, std={std_sim:.3f}, "
                f"ratio={ratio:.3f}, min={min_sim:.3f}"
            )

        return {
            "mean_sim": round(mean_sim, 4),
            "std_sim":  round(std_sim,  4),
            "min_sim":  round(min_sim,  4),
            "ratio":    round(ratio,    4),
            "flagged":  flagged,
        }

    # ── Duplicate Identity Detection ───────────────────────────────────────────

    def detect_duplicates(self, gallery: dict) -> list:
        """Cross-check all confirmed gallery identity centroids.

        gallery: dict of {identity_name: tensor(N, D)} as stored in self.gallery

        Returns list of (name_i, name_j, similarity) tuples for pairs that
        may be the same individual.  Empty list = no issues found.
        """
        # Average each identity's stored embeddings to get one centroid
        centroids = {}
        for name, emb_tensor in gallery.items():
            centroids[name] = F.normalize(emb_tensor.mean(dim=0), p=2, dim=0)

        duplicates = []
        names = list(centroids.keys())
        for i, j in combinations(range(len(names)), 2):
            n1, n2 = names[i], names[j]
            sim = float(torch.dot(centroids[n1], centroids[n2]))
            if sim >= self.duplicate_thresh:
                logger.warning(
                    f"[HealthMonitor] Possible duplicate identities: "
                    f"'{n1}' ↔ '{n2}'  (sim={sim:.3f})"
                )
                duplicates.append((n1, n2, round(sim, 3)))

        return duplicates

    # ── Merge Guard (convenience wrapper) ─────────────────────────────────────

    def evaluate_merge(
        self,
        ci: dict,
        cj: dict,
        merge_threshold: float,
    ) -> tuple:
        """Triple-condition merge guard.

        Args:
            ci, cj          : cluster dicts with 'centroid' and 'samples' keys
            merge_threshold : float

        Returns:
            (approved: bool, reason: str)
        """
        centroid_sim = float(torch.dot(ci["centroid"], cj["centroid"]))
        if centroid_sim < merge_threshold:
            return False, f"centroid_sim={centroid_sim:.3f} < {merge_threshold}"

        cross_sims = [
            float(torch.dot(si, sj))
            for si in ci["samples"]
            for sj in cj["samples"]
        ]
        if not cross_sims:
            return False, "no samples to compare"

        sample_matches  = sum(1 for s in cross_sims if s >= merge_threshold)
        mean_sample_sim = sum(cross_sims) / len(cross_sims)

        if sample_matches < 2:
            return False, (
                f"only {sample_matches} sample pair(s) above threshold "
                f"(need ≥ 2)"
            )

        if mean_sample_sim < merge_threshold - 0.03:
            return False, (
                f"mean_sample_sim={mean_sample_sim:.3f} < "
                f"{merge_threshold - 0.03:.3f}"
            )

        return True, (
            f"centroid={centroid_sim:.3f}, "
            f"sample_matches={sample_matches}, "
            f"mean_sample_sim={mean_sample_sim:.3f}"
        )
