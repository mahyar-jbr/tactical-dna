# Tactical DNA — Coaches, Styles, and Player Roles from Football Passing Networks

**EECS 4414 Information Networks — Final Project**
Mahyar Jaberi · `mhyrjbr@my.yorku.ca` · York University

This project treats football **passing networks** — directed weighted graphs whose
edges count completed passes between players — as *tactical fingerprints*, and
uses their structural signatures for three inference tasks on StatsBomb open data.
From a single team-match network we ask: who coached it, what tactical archetype
it belongs to, and what structural role each player plays. The pipeline builds
per-match passing networks over a feature embedding spanning **centrality, global
structure, and network motifs**, across FC Barcelona's coaching eras, the 2018 and
2022 World Cups, and Bayern Munich.

## The three problems

- **P1 — Coach identification (supervised).** Given a network's structural
  features φ(G), predict the team's head coach. Evaluated on 517 FC Barcelona
  networks across eight coaches with logistic regression, random forest, and
  XGBoost under stratified and leakage-free season-grouped cross-validation, plus
  a cross-context transfer test (does a coach's signature follow them to a new
  club / national team?).
- **P2 — Tactical archetype discovery (unsupervised).** Cluster the merged
  775-network corpus on φ(G) with k-means and Ward, choosing *k* by silhouette and
  the elbow, then interpret the clusters against known tactical schools
  (possession vs. direct).
- **P3 — Player role embedding (unsupervised).** For each player, aggregate an
  ego-network signature ψ(p) across all their appearances, project it to 2-D with
  PCA and UMAP, and measure **position purity** (do a player's nearest neighbours
  share their on-pitch position?) plus qualitative playmaker-analogue queries.

---

## 1. Repository structure

```
.
├── README.md                  This file.
├── requirements.txt           Pinned dependencies (exact versions).
├── src/                       All source code (14 Python files).
│   ├── sb_cache.py            Disk-caching wrapper over statsbombpy. (helper, not run directly)
│   ├── phase0_scope.py        P0: list free competitions, lock the Barça La Liga scope.
│   ├── build_networks.py      P1 data: build the 517 Barça 11-node passing networks. (also exposes load_network)
│   ├── features.py            Feature extraction φ(G): 8 global + 20 centrality + 16 motif. (exposes extract)
│   ├── summarize_phase2.py    Corpus + summary-stats tables, the Clásico figure.
│   ├── phase3_p1.py           P1 models: LR / RF / XGBoost, both CV schemes, ablation, confusion matrix.
│   ├── build_bayern.py        Build Bayern 2015/16 networks (same pipeline) for the transfer test.
│   ├── transfer_test.py       Original Guardiola Barça→Bayern transfer (n = 2).
│   ├── download_worldcups.py  Pre-download World Cup 2018 + 2022 event JSON.
│   ├── build_worldcup.py      Build the 256 World Cup networks (both teams per match).
│   ├── merge_corpus.py        Merge Barça + Bayern + WC into the 775-network corpus.
│   ├── transfer_test_multi.py P1 transfer: Guardiola / L. Enrique / Martino at other teams.
│   ├── problem2_cluster.py    P2: k-means / Ward, silhouette + elbow, UMAP.
│   └── problem3_roles.py      P3: ego ψ(p), PCA / UMAP, position purity, nearest neighbours.
├── outputs/                   Result artifacts (CSV / JSON / TXT / PNG) — see the results map below.
│   └── p3_appearances.jsonl   P3 resume checkpoint (all 775 networks' per-player features).
└── cache/                     Cached StatsBomb data (see "Data" below).
    ├── competitions/          competitions.json
    ├── matches/               Per (competition, season) match lists.
    ├── networks/              517 built Barça networks (node/edge JSON).
    ├── networks_bayern/       2 built Bayern networks.
    └── networks_wc/           256 built World Cup networks.
```

> The bundled `cache/` contains the **built networks** (and match lists), not the
> raw 5.3 GB event archive — see the two reproduction paths in §4.

---

## 2. Dependencies

Developed and tested on **Python 3.13.7**; nearby 3.x versions (3.11+) should
work, as the code uses no version-specific features. Exact pinned versions are in
[`requirements.txt`](requirements.txt):

```
statsbombpy==1.18.0
networkx==3.6.1
pandas==3.0.3
numpy==2.4.6
scipy==1.17.1
scikit-learn==1.8.0
matplotlib==3.10.9
xgboost==3.2.0
umap-learn==0.5.12
```

Install into a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All scripts are run from the project root with `src/` on the path, e.g.
`PYTHONPATH=src python src/features.py` (or `cd src && python features.py`).

---

## 3. Data

This project uses the **StatsBomb Open Data** (free tier), consumed through the
`statsbombpy` client and cached to `cache/` on first download.

- Source: <https://github.com/statsbomb/open-data>
- The code consumes three resource types per competition/season — *competitions*,
  *matches*, *lineups*, and *events* — via `src/sb_cache.py`, which writes each
  downloaded JSON under `cache/` so every later run is offline and deterministic.
- **Attribution (required by the StatsBomb user agreement):** data provided by
  StatsBomb. See <https://github.com/statsbomb/open-data> for the data and its
  licence. This project is a non-commercial academic submission.

**What is bundled vs. downloaded.** The raw event archive is ~5.3 GB and is *not*
included. The zip instead bundles the **already-built passing networks**
(`cache/networks*/`, ~6 MB) and the **processed feature matrices**
(`outputs/features*.csv`) plus the **P3 checkpoint**
(`outputs/p3_appearances.jsonl`). This is enough to reproduce all reported numbers
and figures **offline** (§4, Path A). Regenerating the networks from raw events
requires re-downloading from StatsBomb (§4, Path B).

---

## 4. Reproduce, step by step

There are two paths. **Path A** reproduces every reported result offline from the
bundled built-networks + checkpoint (no internet). **Path B** rebuilds everything
from the raw StatsBomb events (needs internet, ~5 GB download).

### Path A — offline, from the bundled artifacts (recommended for marking)

Each command is standalone and reads the previous step's outputs. From the project
root, with the venv active:

```bash
# --- Feature matrices (read the bundled built networks; no events needed) ---
PYTHONPATH=src python src/features.py            # -> outputs/features.csv (517 × 50)
PYTHONPATH=src python src/summarize_phase2.py    # -> Table 1, Table 2, Figure 1

# --- Problem 1: coach identification ---
PYTHONPATH=src python src/phase3_p1.py           # -> Tables 4, 5, 6 ; Figure 2
PYTHONPATH=src python src/transfer_test_multi.py # -> Table 3 (cross-context transfer)

# --- Problem 2: tactical archetypes (uses the bundled features_all.csv) ---
PYTHONPATH=src python src/problem2_cluster.py    # -> P2 metrics ; Figure 3

# --- Problem 3: player roles (resumes from the bundled checkpoint) ---
PYTHONPATH=src python src/problem3_roles.py      # -> Table 7 ; Figure 4
```

Notes for Path A:
- `features_all.csv`, `features_worldcup.csv`, and `features_bayern.csv` are
  bundled, so P2 and the transfer test run without rebuilding the WC/Bayern
  networks.
- `problem3_roles.py` detects `outputs/p3_appearances.jsonl` and prints
  `Resuming: 775 networks already logged`, skipping the event/lineup walk and
  going straight to the embedding, purity, and neighbour queries. This is why P3
  reproduces **without** the raw events or lineups on disk.

This exact path was verified in a clean-room run: a fresh venv from
`requirements.txt`, only the bundled files present, and the events/lineups caches
absent — features, P1, P2, and P3 all reproduced the report's numbers.

### Path B — full rebuild from raw StatsBomb data (needs internet)

To regenerate the cache from scratch (≈5 GB of event JSON, ~10–15 min on a warm
connection, one-time; everything is cached afterwards):

```bash
# 0. Scope (downloads competitions + match lists)
PYTHONPATH=src python src/phase0_scope.py        # -> outputs/phase0_scope.json

# 1. Build the 517 Barça networks (downloads ~517 Barça event files + lineups)
PYTHONPATH=src python src/build_networks.py      # -> cache/networks/ , phase1_build_log.csv

# 2. Bayern 2015/16 (downloads 2 Leverkusen–Bayern matches)
PYTHONPATH=src python src/build_bayern.py        # -> cache/networks_bayern/ , features_bayern.csv

# 3. World Cups 2018 + 2022 (downloads 128 matches' events + lineups, then builds)
PYTHONPATH=src python src/download_worldcups.py  # -> cache/events/ (download only)
PYTHONPATH=src python src/build_worldcup.py      # -> cache/networks_wc/ , features_worldcup.csv

# 4. Feature matrices + merged corpus
PYTHONPATH=src python src/features.py            # -> outputs/features.csv
PYTHONPATH=src python src/merge_corpus.py        # -> outputs/features_all.csv (775 networks)

# 5. Then run the analysis exactly as in Path A:
PYTHONPATH=src python src/summarize_phase2.py    # Tables 1,2 ; Figure 1
PYTHONPATH=src python src/phase3_p1.py           # Tables 4,5,6 ; Figure 2
PYTHONPATH=src python src/transfer_test_multi.py # Table 3
PYTHONPATH=src python src/problem2_cluster.py    # Figure 3
# For a from-scratch P3, delete the checkpoint first so the full walk runs:
rm -f outputs/p3_appearances.jsonl
PYTHONPATH=src python src/problem3_roles.py      # Table 7 ; Figure 4 (rebuilds ψ(p) from events+lineups)
```

`download_worldcups.py` is idempotent and resumable; all builders skip already
cached files, so Path B can be re-run safely.

---

## 5. Results-to-code map

Every table, figure, and headline number in the final report traces to one script
and its output file.

| Report item | Produced by | Output artifact |
|---|---|---|
| **Table 1** (per-season corpus, 517) | `summarize_phase2.py` | `outputs/corpus_table.csv` |
| **Table 2** (summary stats) | `summarize_phase2.py` | `outputs/summary_stats.csv` |
| **Table 3** (cross-context transfer) | `transfer_test_multi.py` | `outputs/transfer_multi.csv`, `transfer_multi.png` |
| **Table 4** (coach label distribution) | `phase3_p1.py` | printed; labels in `features.csv` via `label_coach` |
| **Table 5** (coach ID, both CVs) | `phase3_p1.py` | `outputs/phase3_p1_results.csv` |
| **Table 6** (feature-family ablation) | `phase3_p1.py` | `outputs/phase3_p1_results.csv` |
| **Table 7** (position purity) | `problem3_roles.py` | `outputs/p3_position_purity.csv` |
| **Figure 1** (Clásico network) | `summarize_phase2.py` | `outputs/example_network.png` |
| **Figure 2** (confusion matrix) | `phase3_p1.py` | `outputs/confusion_matrix.png` |
| **Figure 3** (cluster UMAP) | `problem2_cluster.py` | `outputs/p2_umap.png` (+ `p2_elbow_silhouette.png`) |
| **Figure 4** (role UMAP) | `problem3_roles.py` | `outputs/p3_umap.png` (+ `p3_pca.png`) |

Headline numbers and their source:

| Number | Value | Source |
|---|---|---|
| Barça corpus | 517 networks | `features.csv` (`phase1_build_log.csv`) |
| Full corpus | 775 networks / 42 teams | `features_all.csv` (`merge_corpus.py`) |
| P1 majority baseline | 0.263 | `phase3_p1_results.csv` |
| P1 random-CV acc | RF 0.358 / XGB 0.298 / LR 0.244 | `phase3_p1_results.csv` |
| P1 season-grouped acc | RF 0.250 / XGB 0.256 / LR 0.187 | `phase3_p1_results.csv` |
| P1 ablation (RF) | centrality 0.321 / motifs 0.242 / combined 0.358 | `phase3_p1_results.csv` |
| Guardiola recall | 0.71 | `phase3_p1.py` per-class report (`confusion_matrix.png`) |
| Transfer recovery | Guardiola 0.00 / Enrique 0.42 / Martino 0.11 | `transfer_multi.csv` |
| P2 partition | k = 2, 547 / 227 on 774 nets | `p2_cluster_quality.csv`, `p2_cluster_samples.txt` |
| P3 players | 271 (≥ 5 appearances) | `player_roles.csv` |
| P3 purity (k=5) | raw 0.883 / UMAP 0.865 / PCA 0.680 vs 0.295 | `p3_position_purity.csv` |

---

## 6. Hardware, runtime, and the P3 checkpoint

- **Path A (offline):** a few minutes total on a laptop. `features.py` ~1 min,
  `phase3_p1.py` ~1–2 min (XGBoost + 6 CV passes), `problem2_cluster.py` ~1 min,
  `problem3_roles.py` from checkpoint ~1 min.
- **Path B (full rebuild):** dominated by the one-time ~5 GB StatsBomb download
  (~10–15 min). A from-scratch P3 walk over all 775 networks reads the raw event
  JSON per network and takes longer.
- **Reproducibility:** all randomness is seeded (`RANDOM_STATE = 42`) across CV
  splits, k-means restarts (`n_init = 10`), random forest, XGBoost, PCA, and UMAP.
- **Memory note and the checkpoint.** Building ψ(p) in P3 streams the raw event
  JSON for every network and is memory-sensitive on machines under load. To make
  the walk robust and resumable, `problem3_roles.py` appends each finished
  network's per-player features to `outputs/p3_appearances.jsonl` and, on a later
  run, resumes from that file (skipping completed networks by filename). The
  bundled checkpoint already contains all 775 networks, so P3 reproduces the
  embedding and Table 7 directly — and a rebuild that is interrupted can simply be
  re-run to continue.
