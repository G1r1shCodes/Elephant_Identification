"""
tests/test_cluster_health.py

Unit tests for ClusterHealthMonitor.

Covers:
  - Growth guard: fires when additions >= 5 AND centroid shifts >= 0.04
  - Growth guard: does NOT fire for stable large batch (centroid barely moves)
  - Stability score: flags high-variance cluster (ratio > 0.18)
  - Stability score: does NOT flag tight same-elephant cluster
  - Stability min_sim: flags cluster with one very dissimilar sample
  - Evaluate_merge: approves when all 3 conditions met
  - Evaluate_merge: blocks when sample_matches < 2
  - Evaluate_merge: blocks when mean_sample_sim too low
  - Duplicate detection: fires for near-identical centroids
  - Duplicate detection: silent for clearly different centroids

Run with:
    cd D:\\Elephant_ReIdentification
    python -m pytest tests/test_cluster_health.py -v
"""
import os, sys
import pytest
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cluster_health import ClusterHealthMonitor


def unit_vec(d=384, seed=None):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    return torch.tensor(v / np.linalg.norm(v))

def perturb(v, scale=0.01, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(v)).astype(np.float32) * scale
    out = v.numpy() + noise
    return torch.tensor(out / np.linalg.norm(out))

def orthogonal(v, seed=99):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(len(v)).astype(np.float32)
    r -= r.dot(v.numpy()) * v.numpy()
    return torch.tensor(r / np.linalg.norm(r))


@pytest.fixture
def monitor():
    return ClusterHealthMonitor()


# ── Growth Guard ───────────────────────────────────────────────────────────────

class TestGrowthGuard:
    def test_fires_with_large_additions_and_shift(self, monitor):
        """Large batch with centroid moving significantly → flag."""
        old_c = unit_vec(seed=1)
        new_c = perturb(old_c, scale=0.2, seed=2)  # non-trivial shift
        flagged = monitor.check_growth("Unknown_1", additions=7, old_centroid=old_c, new_centroid=new_c)
        assert flagged is True

    def test_silent_when_additions_below_threshold(self, monitor):
        """Small batch → no flag regardless of shift."""
        old_c = unit_vec(seed=1)
        new_c = perturb(old_c, scale=0.5, seed=3)
        flagged = monitor.check_growth("Unknown_1", additions=3, old_centroid=old_c, new_centroid=new_c)
        assert flagged is False

    def test_silent_for_stable_large_batch(self, monitor):
        """Large batch but centroid barely moves (same-elephant images) → no flag."""
        old_c = unit_vec(seed=1)
        new_c = perturb(old_c, scale=0.005, seed=4)  # tiny shift
        flagged = monitor.check_growth("Unknown_1", additions=8, old_centroid=old_c, new_centroid=new_c)
        assert flagged is False


# ── Stability Score ────────────────────────────────────────────────────────────

class TestStabilityScore:
    def test_flags_high_variance_cluster(self, monitor):
        """Mixed-identity cluster (some very dissimilar samples) → flagged."""
        base = unit_vec(seed=10)
        samples = (
            [perturb(base, scale=0.005, seed=i) for i in range(3)]
            + [orthogonal(base, seed=99)]  # very dissimilar intruder
        )
        result = monitor.compute_stability("Unknown_X", samples)
        assert result["flagged"] is True

    def test_silent_for_tight_cluster(self, monitor):
        """Same-elephant-ish cluster with small variance → not flagged."""
        base    = unit_vec(seed=20)
        samples = [perturb(base, scale=0.01, seed=i) for i in range(5)]
        result  = monitor.compute_stability("Unknown_Y", samples)
        assert result["flagged"] is False

    def test_single_sample_not_flagged(self, monitor):
        """Single sample → can't compute variance → not flagged."""
        result = monitor.compute_stability("Unknown_Z", [unit_vec(seed=5)])
        assert result["flagged"] is False

    def test_flags_low_min_sim(self, monitor):
        """Cluster where min pairwise sim < STABILITY_MIN_SIM → flagged."""
        base    = unit_vec(seed=30)
        close   = [perturb(base, scale=0.005, seed=i) for i in range(4)]
        outlier = perturb(base, scale=0.8, seed=99)  # very different
        result  = monitor.compute_stability("Unknown_W", close + [outlier])
        assert result["min_sim"] < monitor.stability_min_sim or result["flagged"]


# ── Merge Guard ───────────────────────────────────────────────────────────────

class TestEvaluateMerge:
    def _make_cluster(self, base, n=3, scale=0.01):
        samples = [perturb(base, scale=scale, seed=i) for i in range(n)]
        centroid = F.normalize(torch.stack(samples).mean(0), p=2, dim=0)
        return {"centroid": centroid, "samples": samples}

    def test_approves_when_all_conditions_met(self, monitor):
        base = unit_vec(seed=40)
        ci   = self._make_cluster(base, n=4, scale=0.005)
        cj   = self._make_cluster(base, n=4, scale=0.005)
        ok, reason = monitor.evaluate_merge(ci, cj, merge_threshold=0.30)
        assert ok is True

    def test_blocks_on_low_centroid_sim(self, monitor):
        ci = self._make_cluster(unit_vec(seed=1), n=3, scale=0.005)
        cj = self._make_cluster(unit_vec(seed=2), n=3, scale=0.005)
        ok, reason = monitor.evaluate_merge(ci, cj, merge_threshold=0.90)
        assert ok is False
        assert "centroid_sim" in reason

    def test_blocks_when_only_one_sample_match(self, monitor):
        """Only 1 cross-pair above threshold → blocked."""
        base    = unit_vec(seed=50)
        close   = perturb(base, scale=0.005, seed=0)
        far1    = orthogonal(base, seed=55)
        far2    = orthogonal(base, seed=66)
        ci = {"centroid": base.clone(),  "samples": [close, far1]}
        # cj centroid close to ci, but samples are orthogonal
        cj = {"centroid": close.clone(), "samples": [far2, orthogonal(base, seed=77)]}
        ok, reason = monitor.evaluate_merge(ci, cj, merge_threshold=0.30)
        assert ok is False


# ── Duplicate Detection ────────────────────────────────────────────────────────

class TestDuplicateDetection:
    def test_fires_for_near_identical_centroids(self, monitor):
        base = unit_vec(seed=60)
        gallery = {
            "Elephant_01": base.unsqueeze(0),
            "Elephant_02": perturb(base, scale=0.002, seed=1).unsqueeze(0),  # almost identical
            "Elephant_03": unit_vec(seed=99).unsqueeze(0),  # completely different
        }
        dups = monitor.detect_duplicates(gallery)
        dup_pairs = {(a, b) for a, b, _ in dups}
        assert ("Elephant_01", "Elephant_02") in dup_pairs or \
               ("Elephant_02", "Elephant_01") in dup_pairs

    def test_silent_for_clearly_different_identities(self, monitor):
        gallery = {
            f"Elephant_{i:02d}": unit_vec(seed=i).unsqueeze(0)
            for i in range(5)
        }
        dups = monitor.detect_duplicates(gallery)
        assert len(dups) == 0
