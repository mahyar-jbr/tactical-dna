"""
Problem 3 — Player Role Embedding.

For every player p in the full corpus (Barca + Bayern + World Cups), build an
aggregated ego-network feature vector psi(p) summarising the player's local
network role across all matches they started.

Per appearance (one network, the player is one of the 11 nodes) we compute:
  - ego_degree         : # distinct neighbours (in OR out) in the 11-node graph
  - ego_density        : edge density of the ego subgraph (player + neighbours)
  - ego_clustering     : the node's local (weighted) clustering coefficient
  - win_deg / wout_deg : weighted in- and out-degree (passes received / made)
  - deg_ratio          : out/(in+out) — distributor vs receiver tilt
  - triad role frequencies (directed 2-path positions, normalised):
        relay_mid  : how often the node is the MIDDLE of a directed 2-path
                     u->p->w  (a transit / chain player)
        source_2   : how often it is the SOURCE  p->*->*
        sink_2     : how often it is the SINK    *->*->p
  - spatial: mean pass-origin x (advancement), mean y (lateral),
             disp_x, disp_y (std of pass-origin spread), disp_radial
These per-appearance vectors are averaged over all of the player's appearances
to give psi(p). Players with >= MIN_APPEARANCES starts are kept.

Each player's PRIMARY POSITION is the modal StatsBomb starting position across
their appearances, mapped to a coarse line (GK / DEF / MID / FWD) for colouring
and position purity.

Outputs:
  outputs/player_roles.csv         (psi(p) + meta)
  outputs/p3_pca.png, outputs/p3_umap.png
  outputs/p3_position_purity.csv
  outputs/p3_neighbors.txt
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# NOTE: matplotlib / sklearn / umap are imported lazily inside the functions
# that need them (only at the end). The heavy 775-network accumulation loop
# therefore runs with a minimal footprint, which matters on this low-memory,
# swap-pressured machine.

from sb_cache import PROJECT_ROOT
from build_networks import load_network

OUT = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42
MIN_APPEARANCES = 5

NET_DIRS = [
    PROJECT_ROOT / "cache" / "networks",          # Barca (match<MID>.json)
    PROJECT_ROOT / "cache" / "networks_bayern",    # Bayern
    PROJECT_ROOT / "cache" / "networks_wc",        # WC (match<MID>_<TEAM>.json)
]

# coarse position line from StatsBomb position name
POS_LINE = {
    "Goalkeeper": "GK",
    "Right Back": "DEF", "Left Back": "DEF",
    "Right Center Back": "DEF", "Left Center Back": "DEF", "Center Back": "DEF",
    "Right Wing Back": "DEF", "Left Wing Back": "DEF",
    "Right Defensive Midfield": "MID", "Left Defensive Midfield": "MID",
    "Center Defensive Midfield": "MID",
    "Right Center Midfield": "MID", "Left Center Midfield": "MID",
    "Center Midfield": "MID",
    "Right Midfield": "MID", "Left Midfield": "MID",
    "Right Attacking Midfield": "MID", "Left Attacking Midfield": "MID",
    "Center Attacking Midfield": "MID",
    "Right Wing": "FWD", "Left Wing": "FWD",
    "Right Center Forward": "FWD", "Left Center Forward": "FWD",
    "Center Forward": "FWD", "Secondary Striker": "FWD",
}
LINE_COLOR = {"GK": "#888888", "DEF": "#1f77b4", "MID": "#2ca02c",
              "FWD": "#d62728"}


def network_files():
    """All cached network JSON paths across the Barca, Bayern, and WC dirs."""
    files = []
    for d in NET_DIRS:
        files.extend(sorted(glob.glob(str(d / "match*.json"))))
    return files


def match_id_of(path):
    """Parse the integer match id from a network filename (handles the
    `match<MID>.json` and `match<MID>_<TEAM>.json` forms)."""
    base = os.path.basename(path)
    # match<MID>.json  or  match<MID>_<TEAM>.json
    core = base[len("match"):-len(".json")]
    return int(core.split("_")[0])


# ---- per-node ego features on one network -------------------------------
def ego_features(G: nx.DiGraph, node: int) -> dict:
    """
    Per-node ego-network features for one player in one passing network.

    Returns ego degree/density/clustering, weighted in/out-degree, a
    distributor ratio, and three directed 2-path role frequencies (how often
    the node is the relay/middle, the source, or the sink of a u->p->w chain),
    each normalized by the node's total 2-path count.
    """
    succ = set(G.successors(node))
    pred = set(G.predecessors(node))
    nbrs = succ | pred

    ego_nodes = nbrs | {node}
    ego_sub = G.subgraph(ego_nodes)
    ego_density = nx.density(ego_sub) if len(ego_nodes) > 1 else 0.0
    clustering = nx.clustering(G, node, weight="weight")

    win = G.in_degree(node, weight="weight")
    wout = G.out_degree(node, weight="weight")
    tot = win + wout
    deg_ratio = (wout / tot) if tot > 0 else 0.5

    # directed 2-path roles (counts over the whole graph), normalised by the
    # total number of directed 2-paths so they are comparable across networks.
    relay = sum(1 for u in pred for w in succ if u != w and u != node and w != node)
    src2 = sum(1 for a in succ for b in G.successors(a)
               if b != node and a != node)
    snk2 = sum(1 for a in pred for b in G.predecessors(a)
               if b != node and a != node)
    Z = max(relay + src2 + snk2, 1)

    return {
        "ego_degree": len(nbrs),
        "ego_density": ego_density,
        "ego_clustering": clustering,
        "win_deg": float(win),
        "wout_deg": float(wout),
        "deg_ratio": deg_ratio,
        "relay_mid": relay / Z,
        "source_2": src2 / Z,
        "sink_2": snk2 / Z,
    }


def _empty_spatial():
    return {"mean_x": np.nan, "mean_y": np.nan, "disp_x": np.nan,
            "disp_y": np.nan, "disp_radial": np.nan}


# Slim, memory-light reader: parse the cached event JSON directly and keep only
# the pass-origin coordinates per player. Avoids building the full ~80-column
# statsbombpy DataFrame (which causes swapping on this low-memory machine).
EVENT_CACHE_DIR = PROJECT_ROOT / "cache" / "events"


def spatial_features_slim(match_id: int, team: str) -> dict[int, dict]:
    """{player_id: spatial dict} read straight from the cached event JSON."""
    path = EVENT_CACHE_DIR / f"match{match_id}.json"
    with open(path) as fh:
        records = json.load(fh)
    by_player: dict[int, list] = defaultdict(list)
    for e in records:
        if e.get("type") != "Pass" or e.get("team") != team:
            continue
        loc = e.get("location")
        pid = e.get("player_id")
        if pid is None or not isinstance(loc, list) or len(loc) != 2:
            continue
        by_player[int(pid)].append(loc)
    out: dict[int, dict] = {}
    for pid, locs in by_player.items():
        xy = np.asarray(locs, dtype=float)
        mx, my = xy[:, 0].mean(), xy[:, 1].mean()
        dx, dy = xy[:, 0].std(), xy[:, 1].std()
        radial = np.sqrt(((xy - [mx, my]) ** 2).sum(axis=1)).mean()
        out[pid] = {"mean_x": mx, "mean_y": my, "disp_x": dx, "disp_y": dy,
                    "disp_radial": radial}
    return out


# ---- player starting position from lineups (slim JSON read) -------------
LINEUP_CACHE_DIR = PROJECT_ROOT / "cache" / "lineups"


def starting_positions_slim(match_id: int, team: str) -> dict[int, str]:
    """{player_id: starting position} read straight from the cached lineups JSON."""
    path = LINEUP_CACHE_DIR / f"match{match_id}.json"
    with open(path) as fh:
        raw = json.load(fh)  # {team: [player records]}
    if team not in raw:
        return {}
    out = {}
    for row in raw[team]:
        positions = row.get("positions")
        if not isinstance(positions, list) or not positions:
            continue
        pos = None
        for p in positions:
            if p.get("start_reason") == "Starting XI":
                pos = p.get("position")
                break
        if pos is None:
            pos = positions[0].get("position")
        if pos:
            out[int(row["player_id"])] = pos
    return out


FEATURE_KEYS = ["ego_degree", "ego_density", "ego_clustering", "win_deg",
                "wout_deg", "deg_ratio", "relay_mid", "source_2", "sink_2",
                "mean_x", "mean_y", "disp_x", "disp_y", "disp_radial"]


def main():
    files = network_files()
    print("=" * 74)
    print(f"PROBLEM 3 — player role embedding over {len(files)} networks")
    print("=" * 74)

    # accumulate per-player lists of per-appearance feature dicts
    acc = defaultdict(list)
    names = {}
    positions_seen = defaultdict(Counter)

    # Resume support: a per-network appearance log is appended to disk so the
    # heavy walk survives interruption on this swap-pressured machine.
    apprec_path = OUT / "p3_appearances.jsonl"
    done_files = set()
    if apprec_path.exists():
        with open(apprec_path) as fh:
            for ln in fh:
                try:
                    done_files.add(json.loads(ln)["f"])
                except Exception:  # noqa: BLE001
                    pass
        print(f"Resuming: {len(done_files)} networks already logged.", flush=True)

    fh_app = open(apprec_path, "a")
    for i, f in enumerate(files, 1):
        fname = os.path.basename(f)
        if fname in done_files:
            continue
        mid = match_id_of(f)
        # Barca networks live in cache/networks/ (load by id); Bayern and WC
        # networks have custom filenames/dirs, so read them by full path.
        if "networks_bayern" in f or "networks_wc" in f:
            G = _load_wc(f)
        else:
            G = load_network(mid)
        team = G.graph.get("team")
        # memory-light: read pass coords + positions straight from cached JSON
        spat_all = spatial_features_slim(mid, team)
        pos_all = starting_positions_slim(mid, team)
        net_rows = []
        for node in G.nodes:
            ego = ego_features(G, node)
            spat = spat_all.get(int(node), _empty_spatial())
            row = {**ego, **spat}
            label = G.nodes[node].get("label") or G.nodes[node].get("name") \
                or str(node)
            pos = pos_all.get(int(node))
            net_rows.append({"pid": int(node), "team": team, "label": label,
                             "pos": pos, "feat": row})
        # append this network's rows to the resume log, then flush
        fh_app.write(json.dumps({"f": fname, "rows": net_rows}) + "\n")
        fh_app.flush()
        if i % 50 == 0:
            print(f"  ... processed {i}/{len(files)} networks", flush=True)
    fh_app.close()

    # ---- read the full appearance log back and aggregate ----
    with open(apprec_path) as fh:
        for ln in fh:
            rec = json.loads(ln)
            for r in rec["rows"]:
                key = (r["pid"], r["team"])
                acc[key].append(r["feat"])
                names[key] = r["label"]
                if r["pos"]:
                    positions_seen[key][r["pos"]] += 1

    # aggregate to psi(p); keep players with >= MIN_APPEARANCES
    rows = []
    for key, app_list in acc.items():
        n_app = len(app_list)
        if n_app < MIN_APPEARANCES:
            continue
        pid, team = key
        agg = {k: float(np.nanmean([a[k] for a in app_list]))
               for k in FEATURE_KEYS}
        pos_counter = positions_seen[key]
        prim_pos = pos_counter.most_common(1)[0][0] if pos_counter else "Unknown"
        line = POS_LINE.get(prim_pos, "Unknown")
        agg.update({"player_id": pid, "team": team, "name": names[key],
                    "appearances": n_app, "primary_position": prim_pos,
                    "line": line})
        rows.append(agg)

    df = pd.DataFrame(rows)
    # drop any residual NaN (players who never had a valid spatial sample)
    before = len(df)
    df = df.dropna(subset=FEATURE_KEYS).reset_index(drop=True)
    df.to_csv(OUT / "player_roles.csv", index=False)

    print(f"\nPlayers with >= {MIN_APPEARANCES} appearances: {before} "
          f"({len(df)} after dropping incomplete).")
    print(f"  by line: {dict(df['line'].value_counts())}")
    print(f"Wrote outputs/player_roles.csv ({df.shape[0]} x {df.shape[1]})")

    # ---- standardise + project (heavy libs imported only now) ----
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X = StandardScaler().fit_transform(df[FEATURE_KEYS].values)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    emb_pca = pca.fit_transform(X)
    print(f"\nPCA explained variance (2D): "
          f"{pca.explained_variance_ratio_.sum():.2%}")
    _scatter(emb_pca, df, "PCA", OUT / "p3_pca.png")

    emb_umap = None
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                            random_state=RANDOM_STATE)
        emb_umap = reducer.fit_transform(X)
        _scatter(emb_umap, df, "UMAP", OUT / "p3_umap.png")
    except Exception as e:  # noqa: BLE001
        print(f"UMAP skipped: {e}")

    # ---- export the exact 2D coordinates (same seeded embeddings used for the
    # PNGs above) so the interactive HTML can reuse them verbatim. This writes
    # coordinates only; it does not alter any computation or result. ----
    export = df[["name", "team", "primary_position", "line",
                 "appearances", "player_id"]].copy()
    export["pca_x"] = emb_pca[:, 0]
    export["pca_y"] = emb_pca[:, 1]
    if emb_umap is not None:
        export["umap_x"] = emb_umap[:, 0]
        export["umap_y"] = emb_umap[:, 1]
    export.to_csv(OUT / "p3_embedding_coords.csv", index=False)
    print(f"Wrote outputs/p3_embedding_coords.csv "
          f"({len(export)} players x {export.shape[1]} cols)")

    # ---- position purity ----
    print("\n" + "-" * 74)
    print("POSITION PURITY (fraction of k-NN sharing the player's line)")
    print("-" * 74)
    purity_rows = []
    for space_name, emb in [("feature(raw)", X), ("PCA-2D", emb_pca)] + \
            ([("UMAP-2D", emb_umap)] if emb_umap is not None else []):
        for k in (5, 10):
            pur = position_purity(emb, df["line"].values, k)
            base = random_baseline(df["line"].values, k)
            purity_rows.append({"space": space_name, "k": k,
                                "purity": pur, "random_baseline": base,
                                "lift": pur - base})
            print(f"  {space_name:<13} k={k:<3} purity={pur:.3f}  "
                  f"random={base:.3f}  lift=+{pur-base:.3f}")
    pd.DataFrame(purity_rows).to_csv(OUT / "p3_position_purity.csv", index=False)

    # ---- nearest neighbours for landmark playmakers ----
    landmark_neighbors(df, X)

    # ---- copy-paste summary ----
    print("\n" + "=" * 74)
    print("PROBLEM 3 SUMMARY (copy-paste)")
    print("=" * 74)
    print(f"- psi(p) for {len(df)} players (>= {MIN_APPEARANCES} starts), "
          f"14 ego/spatial features.")
    best = max(purity_rows, key=lambda r: r["lift"])
    print(f"- Best position purity: {best['space']} k={best['k']} "
          f"purity {best['purity']:.3f} vs random {best['random_baseline']:.3f} "
          f"(lift +{best['lift']:.3f}).")
    print(f"- Figures: p3_pca.png ({pca.explained_variance_ratio_.sum():.0%} "
          f"var), p3_umap.png. Neighbours -> p3_neighbors.txt.")


def _load_wc(path):
    """Load a WC network (custom filename) by reading the JSON directly."""
    with open(path) as fh:
        data = json.load(fh)
    G = nx.DiGraph()
    G.graph.update(data["graph"])
    for n in data["nodes"]:
        nid = n.pop("id")
        G.add_node(int(nid), **n)
    for e in data["edges"]:
        G.add_edge(int(e["source"]), int(e["target"]), weight=int(e["weight"]))
    return G


def position_purity(emb, lines, k):
    """Mean fraction of each point's k nearest neighbours sharing its line."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nn.kneighbors(emb)
    fracs = []
    for i in range(len(emb)):
        neigh = idx[i][1:]  # drop self
        fracs.append(np.mean(lines[neigh] == lines[i]))
    return float(np.mean(fracs))


def random_baseline(lines, k):
    """Expected purity under random placement = sum of squared class shares."""
    counts = pd.Series(lines).value_counts(normalize=True)
    return float((counts ** 2).sum())


def _scatter(emb, df, method, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 8))
    for line, color in LINE_COLOR.items():
        m = df["line"].values == line
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1], c=color, label=line, s=28,
                       alpha=0.75, edgecolors="none")
    # annotate a few landmark playmakers if present
    for target in ["Xavi", "Andrés Iniesta", "Sergio Busquets", "Luka Modrić",
                   "Toni Kroos", "Lionel Messi"]:
        hit = df[df["name"] == target]
        if len(hit):
            ridx = hit.index[0]
            ax.annotate(target, (emb[ridx, 0], emb[ridx, 1]), fontsize=8,
                        fontweight="bold", xytext=(5, 5),
                        textcoords="offset points")
    ax.set_title(f"{method} of player role space psi(p) "
                 f"(coloured by primary position line)\n"
                 f"{len(df)} players, Barca + Bayern + World Cups")
    ax.set_xlabel(f"{method}-1"); ax.set_ylabel(f"{method}-2")
    ax.legend(title="position line")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def landmark_neighbors(df, X):
    from sklearn.neighbors import NearestNeighbors
    targets = ["Xavi", "Luka Modrić", "Andrés Iniesta", "Toni Kroos",
               "Sergio Busquets"]
    print("\n" + "-" * 74)
    print("LANDMARK PLAYMAKERS — availability + nearest neighbours in role space")
    print("-" * 74)
    present = {}
    for t in targets:
        hit = df[df["name"] == t]
        present[t] = len(hit) > 0
    print("In corpus: " + ", ".join(
        f"{t}={'YES' if present[t] else 'NO'}" for t in targets))

    nn = NearestNeighbors(n_neighbors=11).fit(X)
    lines = []
    for t in targets:
        hit = df[df["name"] == t]
        if hit.empty:
            lines.append(f"\n[{t}] NOT in corpus (>= {MIN_APPEARANCES} starts).")
            continue
        i = hit.index[0]
        _, idx = nn.kneighbors(X[i].reshape(1, -1))
        neigh = [j for j in idx[0] if j != i][:10]
        block = [f"\n[{t}]  ({df.loc[i,'team']}, {df.loc[i,'primary_position']}, "
                 f"{int(df.loc[i,'appearances'])} apps) — 10 nearest in role space:"]
        for rank, j in enumerate(neigh, 1):
            block.append(
                f"   {rank:>2}. {df.loc[j,'name']:<24} "
                f"({df.loc[j,'team']:<12} {df.loc[j,'primary_position']:<22} "
                f"{df.loc[j,'line']})")
        lines.append("\n".join(block))
        print("\n".join(block))
    (OUT / "p3_neighbors.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
