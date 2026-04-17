# Changelog — 07 Mar 2026

## Unknown Elephant Clustering

Previously all unrecognised images went into a single flat `Unknown/` folder.

**New behaviour:** unknown images are automatically grouped into similarity-based clusters — `Unknown_1`, `Unknown_2`, etc. — so researchers can track potential new individuals across sightings.

### What was added

**`core_engine.py` — `UnknownClusterManager` class**
- Greedy clustering with two-stage verification: centroid match → sample verification → new cluster
- Running-mean centroid updates (re-normalised after each image)
- Clusters persist across sessions via `unknown_clusters.json` in the output folder
- Thresholds: `CLUSTER_THRESHOLD = 0.35`, `SECONDARY_THRESHOLD = 0.20`

**`core_engine.py` — `process_batch()` refactor**
- Embedding computed once per image (reused for gallery lookup + clustering)
- Known elephants → labelled folder + watermark (unchanged)
- Unknown images → `UnknownClusterManager.assign()` → `Unknown_N/` folder + watermark showing cluster name and similarity %

**`core_engine.py` — `update_database()` fix**
- Previously **overwrote** the gallery entry when an existing ID was used
- Now **merges** new embeddings with existing ones (weighted average) — original knowledge is preserved
- Returns `(success, is_update)` tuple so the UI shows the right message

### UI changes

**Tab 1 — Classification Report**
- Two visual sections: `IDENTIFIED ELEPHANTS` (gold badge) and `POTENTIAL NEW ELEPHANTS` (amber badge, italic)
- Title line shows: `N Known Elephant(s) | M Unknown Cluster(s) | X Image(s) Processed`
- Status bar shows known and cluster counts separately

**Tab 3 — Database Management** *(was "Database Enrolment")*
- Renamed to reflect dual purpose
- Gold hint panel explains: new ID → fresh entry, existing ID → gallery enrichment
- Button updated to `Select Folder & Enroll / Update`
- Placeholder text guides the user on both modes

### Tests

`tests/test_cluster_manager.py` — 10 unit tests covering:
- Bootstrap, similar/dissimilar grouping, count increment
- Sample cap, JSON save/reload, cross-session persistence
- Corrupt JSON recovery, `cluster_summary` property
