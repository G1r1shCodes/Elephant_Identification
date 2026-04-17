import json
import os
from datetime import datetime
from uuid import uuid4


class ReviewStore:
    """JSON-backed persistence for ambiguity review items and feedback pairs."""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.ambiguity_file = os.path.join(base_dir, "ambiguities.json")
        self.feedback_file = os.path.join(base_dir, "feedback_pairs.json")

    def _load_list(self, path):
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_list(self, path, items):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)

    def list_ambiguities(self, unresolved_only=False):
        items = self._load_list(self.ambiguity_file)
        if unresolved_only:
            items = [item for item in items if item.get("status", "OPEN") == "OPEN"]
        return sorted(
            items,
            key=lambda item: (
                item.get("status", "OPEN") != "OPEN",
                item.get("created_at", ""),
            ),
        )

    def add_ambiguous_match(self, node_path, candidates, current_cluster=None):
        items = self._load_list(self.ambiguity_file)

        # Deduplication
        for item in items:
            if (
                item.get("status", "OPEN") == "OPEN"
                and item.get("type") == "ambiguous_match"
                and node_path in item.get("file_paths", [])
            ):
                return item["id"]

        record = {
            "id": str(uuid4()),
            "type": "ambiguous_match",
            "file_paths": [node_path],
            "source_filenames": [os.path.basename(node_path)],
            "candidates": candidates,
            "current_cluster": current_cluster,
            "status": "OPEN",
            "created_at": datetime.now().isoformat(),
        }

        items.append(record)
        self._save_list(self.ambiguity_file, items)
        return record["id"]

    def add_ambiguity(self, record):
        items = self._load_list(self.ambiguity_file)
        current_cluster = record.get("current_cluster")
        file_paths = sorted(record.get("file_paths", []))

        for item in items:
            if (
                item.get("status", "OPEN") == "OPEN"
                and item.get("current_cluster") == current_cluster
                and sorted(item.get("file_paths", [])) == file_paths
            ):
                item.update(record)
                item.setdefault("id", item.get("id") or str(uuid4()))
                item.setdefault("created_at", datetime.now().isoformat())
                self._save_list(self.ambiguity_file, items)
                return item["id"]

        record = dict(record)
        record.setdefault("id", str(uuid4()))
        record.setdefault("status", "OPEN")
        record.setdefault("created_at", datetime.now().isoformat())
        items.append(record)
        self._save_list(self.ambiguity_file, items)
        return record["id"]

    def resolve_ambiguity(self, ambiguity_id, resolution):
        items = self._load_list(self.ambiguity_file)
        for item in items:
            if item.get("id") == ambiguity_id:
                item["status"] = "RESOLVED"
                item["resolution"] = resolution
                item["resolved_at"] = datetime.now().isoformat()
                self._save_list(self.ambiguity_file, items)
                return True
        return False

    def remove_open_ambiguities_for_filenames(self, filenames):
        filename_set = {name for name in filenames if name}
        if not filename_set:
            return 0

        items = self._load_list(self.ambiguity_file)
        kept = []
        removed = 0
        for item in items:
            if item.get("status", "OPEN") != "OPEN":
                kept.append(item)
                continue
            item_names = set(item.get("source_filenames", []))
            if item_names & filename_set:
                removed += 1
                continue
            kept.append(item)

        if removed:
            self._save_list(self.ambiguity_file, kept)
        return removed

    def resolve_open_ambiguities(
        self,
        filenames=None,
        cluster_names=None,
        exclude_ids=None,
        resolution=None,
    ):
        filename_set = {name for name in (filenames or []) if name}
        cluster_set = {name for name in (cluster_names or []) if name}
        exclude_set = {item_id for item_id in (exclude_ids or []) if item_id}
        if not filename_set and not cluster_set:
            return 0

        items = self._load_list(self.ambiguity_file)
        changed = 0
        default_resolution = {
            "action": "AUTO_CLOSE",
            "status": "related_state_changed",
        }
        merged_resolution = dict(default_resolution)
        if resolution:
            merged_resolution.update(resolution)

        for item in items:
            if item.get("status", "OPEN") != "OPEN":
                continue
            if item.get("id") in exclude_set:
                continue

            item_filenames = set(item.get("source_filenames", []))
            if not item_filenames:
                item_filenames = {
                    os.path.basename(path)
                    for path in item.get("file_paths", [])
                    if path
                }

            item_clusters = {
                item.get("current_cluster"),
                item.get("candidate_a", {}).get("name"),
                item.get("candidate_b", {}).get("name"),
            }
            item_clusters.discard(None)
            item_clusters.discard("")

            if (filename_set and item_filenames & filename_set) or (
                cluster_set and item_clusters & cluster_set
            ):
                item["status"] = "RESOLVED"
                item["resolution"] = dict(merged_resolution)
                item["resolved_at"] = datetime.now().isoformat()
                changed += 1

        if changed:
            self._save_list(self.ambiguity_file, items)
        return changed

    def add_feedback_pair(self, pair):
        items = self._load_list(self.feedback_file)
        record = dict(pair)
        record.setdefault("created_at", datetime.now().isoformat())
        items.append(record)
        self._save_list(self.feedback_file, items)

    def list_feedback_pairs(self):
        return self._load_list(self.feedback_file)
