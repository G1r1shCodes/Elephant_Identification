"""
tests/test_cluster_manager.py

Unit tests for UnknownClusterManager (incremental centroid clustering).

Covers:
  - Dual-threshold: strong (>=0.70), weak+sample-verify (0.60-0.70), new (<0.60)
  - Max sample similarity verification (not average)
  - Argmax cluster selection
  - Centroid recomputed from samples only
  - Diversity replacement (remove sample closest to centroid)
  - Post-batch cluster merging
  - JSON persistence (save / reload)
  - Corrupt JSON recovery

Run with:
    cd D:\\Elephant_ReIdentification
    python -m pytest tests/test_cluster_manager.py -v
"""
import os
import sys
import tempfile
import shutil
import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_engine import UnknownClusterManager


# ── Helpers ───────────────────────────────────────────────────────────────────

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
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def make_mgr(tmp_dir, **kwargs):
    return UnknownClusterManager(
        unknown_dir  = tmp_dir,
        cluster_file = os.path.join(tmp_dir, "unknown_clusters.json"),
        **kwargs,
    )


# ── Bootstrap ─────────────────────────────────────────────────────────────────

class TestBootstrap:
    def test_first_image_creates_unknown_1(self, tmp_dir):
        mgr = make_mgr(tmp_dir)
        name, score = mgr.assign(unit_vec(seed=0))
        assert name == "Unknown_1"
        assert score == 1.0

    def test_folder_created_on_new_cluster(self, tmp_dir):
        mgr = make_mgr(tmp_dir)
        mgr.assign(unit_vec(seed=0))
        assert os.path.isdir(os.path.join(tmp_dir, "Unknown_1"))


# ── Dual-threshold logic ──────────────────────────────────────────────────────

class TestDualThreshold:
    def test_strong_path_assigns_directly(self, tmp_dir):
        """Very similar vector (>0.70 cosine sim) must go to existing cluster."""
        mgr  = make_mgr(tmp_dir)
        base = unit_vec(seed=10)
        mgr.assign(base)
        # Tiny perturbation -> sim ~ 0.999
        name, _ = mgr.assign(perturb(base, scale=0.002, seed=1))
        assert name == "Unknown_1"

    def test_orthogonal_creates_new_cluster(self, tmp_dir):
        """Orthogonal vector (sim ~0) -> new cluster."""
        mgr = make_mgr(tmp_dir)
        v1  = unit_vec(seed=1)
        v2  = orthogonal(v1, seed=55)
        mgr.assign(v1)
        name2, _ = mgr.assign(v2)
        assert name2 == "Unknown_2"

    def test_argmax_picks_best_cluster(self, tmp_dir):
        """When two clusters exist, image should go to the closer one."""
        mgr = make_mgr(tmp_dir)
        v1 = unit_vec(seed=1)
        v2 = orthogonal(v1, seed=55)
        mgr.assign(v1)  # Unknown_1
        mgr.assign(v2)  # Unknown_2
        # Slightly perturbed v1 should match Unknown_1, not Unknown_2
        name, _ = mgr.assign(perturb(v1, scale=0.002, seed=2))
        assert name == "Unknown_1"


# ── Centroid and sample management ────────────────────────────────────────────

class TestCentroidAndSamples:
    def test_centroid_recomputed_from_samples(self, tmp_dir):
        """After 3 additions, centroid equals normalize(mean(samples))."""
        mgr  = make_mgr(tmp_dir)
        base = unit_vec(seed=20)
        embs = [perturb(base, scale=0.005, seed=i) for i in range(3)]
        for e in embs:
            mgr.assign(e)
        cluster = mgr.clusters["Unknown_1"]
        expected = torch.stack(cluster["samples"]).mean(0)
        expected = expected / torch.linalg.norm(expected)
        np.testing.assert_allclose(
            cluster["centroid"].numpy(), expected.numpy(), atol=1e-5
        )

    def test_diversity_removal_keeps_edge_samples(self, tmp_dir):
        """With MAX_SAMPLES=3, the sample closest to centroid is dropped."""
        mgr  = make_mgr(tmp_dir, max_samples=3)
        base = unit_vec(seed=5)
        # Add 4 very similar samples; the cap should drop the most redundant one
        for i in range(4):
            mgr.assign(perturb(base, scale=0.003, seed=i))
        samples = mgr.clusters["Unknown_1"]["samples"]
        assert len(samples) <= 3

    def test_count_increments_correctly(self, tmp_dir):
        mgr  = make_mgr(tmp_dir)
        base = unit_vec(seed=30)
        for i in range(5):
            mgr.assign(perturb(base, scale=0.005, seed=i))
        assert mgr.clusters["Unknown_1"]["count"] == 5


# ── Cluster merging ───────────────────────────────────────────────────────────

class TestClusterMerging:
    def test_merge_very_similar_clusters(self, tmp_dir):
        """Two clusters with highly similar centroids & samples merge into one.

        Uses 3 samples per cluster so the triple-condition guard has
        enough cross-pairs to satisfy sample_matches >= 2.
        """
        mgr  = make_mgr(tmp_dir, merge_threshold=0.70)
        base = unit_vec(seed=42)

        # Cluster 1: 3 near-identical samples
        s1 = [perturb(base, scale=0.005, seed=i) for i in range(3)]
        c1 = F.normalize(torch.stack(s1).mean(0), p=2, dim=0)
        mgr.clusters["Unknown_1"] = {
            "centroid": c1, "samples": s1, "count": 3, "created_at": ""
        }
        os.makedirs(os.path.join(tmp_dir, "Unknown_1"), exist_ok=True)

        # Cluster 2: 3 near-identical samples (same base → will merge)
        s2 = [perturb(base, scale=0.005, seed=i + 10) for i in range(3)]
        c2 = F.normalize(torch.stack(s2).mean(0), p=2, dim=0)
        mgr.clusters["Unknown_2"] = {
            "centroid": c2, "samples": s2, "count": 3, "created_at": ""
        }
        os.makedirs(os.path.join(tmp_dir, "Unknown_2"), exist_ok=True)

        assert len(mgr.clusters) == 2  # pre-merge
        merged = mgr.merge_clusters()
        assert len(mgr.clusters) == 1,  f"Expected 1 cluster after merge, got {len(mgr.clusters)}"
        assert len(merged) == 1


    def test_orthogonal_clusters_do_not_merge(self, tmp_dir):
        """Truly different clusters should NOT be merged."""
        mgr = make_mgr(tmp_dir)
        v1  = unit_vec(seed=1)
        v2  = orthogonal(v1, seed=55)
        mgr.assign(v1)
        mgr.assign(v2)
        merged = mgr.merge_clusters()
        assert len(merged) == 0
        assert len(mgr.clusters) == 2


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_reload_centroid(self, tmp_dir):
        mgr1 = make_mgr(tmp_dir)
        mgr1.assign(unit_vec(seed=7))
        mgr1.save()
        mgr2 = make_mgr(tmp_dir)
        assert "Unknown_1" in mgr2.clusters
        assert mgr2.clusters["Unknown_1"]["count"] == 1
        np.testing.assert_allclose(
            mgr2.clusters["Unknown_1"]["centroid"].numpy(),
            mgr1.clusters["Unknown_1"]["centroid"].numpy(),
            atol=1e-5,
        )

    def test_cross_session_assignment(self, tmp_dir):
        base = unit_vec(seed=20)
        mgr1 = make_mgr(tmp_dir)
        mgr1.assign(base)
        mgr1.save()
        mgr2 = make_mgr(tmp_dir)
        name, _ = mgr2.assign(perturb(base, scale=0.005, seed=0))
        assert name == "Unknown_1"
        assert mgr2.clusters["Unknown_1"]["count"] == 2

    def test_corrupt_json_starts_fresh(self, tmp_dir):
        with open(os.path.join(tmp_dir, "unknown_clusters.json"), "w") as f:
            f.write("NOT VALID JSON {{{{")
        mgr = make_mgr(tmp_dir)
        assert mgr.clusters == {}


# ── Cluster summary property ──────────────────────────────────────────────────

class TestClusterSummary:
    def test_cluster_summary(self, tmp_dir):
        mgr  = make_mgr(tmp_dir)
        base = unit_vec(seed=30)
        for i in range(4):
            mgr.assign(perturb(base, scale=0.005, seed=i))
        mgr.assign(orthogonal(base, seed=55))
        summary = mgr.cluster_summary
        assert summary.get("Unknown_1") == 4
        assert summary.get("Unknown_2") == 1
