"""
Phase 2 — per-network feature extraction.

For each cached Barca network we compute the feature embedding phi(G_m):

  GLOBAL STRUCTURAL
    n_edges, density, total_passes,
    weighted clustering coefficient (Fagiolo directed-weighted),
    average shortest-path length (on the largest strongly-connected
      component, using distance = 1/weight; unweighted-hop ASPL too),
    algebraic connectivity (Fiedler value of the UNDIRECTED weighted
      Laplacian, weights of reciprocal arcs summed),
    spectral radius (largest |eigenvalue| of the weighted adjacency).

  CENTRALITY SUMMARIES  (mean, variance, skewness, Gini per distribution)
    weighted in-degree, weighted out-degree,
    betweenness (distance = 1/weight),
    eigenvector centrality (weighted),
    PageRank (weighted).

  MOTIFS
    nx.triadic_census -> 16 directed-triad counts, normalized by C(11,3)=165.

Important modelling choices (documented for the report):
  * Betweenness & weighted ASPL invert edge weight to a DISTANCE (1/weight):
    in a passing network more passes = stronger tie = shorter path. Using the
    raw weight as distance would be semantically backwards.
  * ASPL is computed on the largest strongly-connected component because a
    handful of very sparse networks (short pre-substitution windows) are not
    strongly connected; we also record scc_size so these cases are auditable.
  * Eigenvector centrality uses the numpy solver (robust; the power-iteration
    variant can fail to converge on some weighted graphs).

Writes outputs/features.csv (one row per network).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import skew

from build_networks import load_network
from sb_cache import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs"
N_NODES = 11
N_TRIADS = 165  # C(11, 3)

TRIAD_TYPES = ['003', '012', '102', '021D', '021U', '021C', '111D', '111U',
               '030T', '030C', '201', '120D', '120U', '120C', '210', '300']


# ----------------------------------------------------------------------------
# summary-statistic helpers for a per-node distribution
# ----------------------------------------------------------------------------
def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative distribution (0 = equal)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    s = x.sum()
    if n == 0 or s == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * s) - (n + 1.0) / n)


def summarize(values, prefix: str) -> dict:
    """mean / variance / skewness / Gini of a distribution."""
    v = np.asarray(list(values), dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(v)),
        f"{prefix}_var": float(np.var(v)),
        f"{prefix}_skew": float(skew(v)) if np.std(v) > 1e-12 else 0.0,
        f"{prefix}_gini": gini(v),
    }


# ----------------------------------------------------------------------------
# graph helpers
# ----------------------------------------------------------------------------
def undirected_weighted(G: nx.DiGraph) -> nx.Graph:
    """Collapse to an undirected graph summing reciprocal arc weights."""
    U = nx.Graph()
    U.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        if U.has_edge(u, v):
            U[u][v]["weight"] += w
        else:
            U.add_edge(u, v, weight=w)
    return U


def add_distance(G: nx.DiGraph) -> nx.DiGraph:
    """Return a copy with edge attribute distance = 1 / weight."""
    H = G.copy()
    for _, _, d in H.edges(data=True):
        d["distance"] = 1.0 / d["weight"]
    return H


def aspl_on_lscc(G: nx.DiGraph, weight: str | None):
    """
    Average shortest-path length on the largest strongly-connected component.
    Returns (aspl, scc_size). NaN if the LSCC has <2 nodes.
    """
    sccs = list(nx.strongly_connected_components(G))
    lscc = max(sccs, key=len)
    H = G.subgraph(lscc)
    if H.number_of_nodes() < 2:
        return float("nan"), H.number_of_nodes()
    return nx.average_shortest_path_length(H, weight=weight), H.number_of_nodes()


# ----------------------------------------------------------------------------
# main feature extraction for one network
# ----------------------------------------------------------------------------
def extract(G: nx.DiGraph) -> dict:
    """
    Compute the full feature embedding phi(G) for one passing network.

    Returns a flat dict with the network's ids/metadata plus 8 global metrics,
    20 centrality summary statistics (mean/var/skew/Gini of weighted in-degree,
    weighted out-degree, betweenness, eigenvector, PageRank), and 16 normalized
    triadic-census motif fractions. Reused by every phase that needs features.
    """
    feats: dict = {
        "match_id": G.graph["match_id"],
        "season": G.graph["season"],
        "date": G.graph["date"],
    }

    # ---- global structural ----
    feats["n_nodes"] = G.number_of_nodes()
    feats["n_edges"] = G.number_of_edges()
    feats["total_passes"] = int(sum(d["weight"] for _, _, d in G.edges(data=True)))
    feats["density"] = nx.density(G)

    # weighted clustering (Fagiolo directed-weighted); weights normalized
    # internally by networkx using max weight.
    feats["weighted_clustering"] = nx.average_clustering(G, weight="weight")

    # average shortest-path length on LSCC
    Gd = add_distance(G)
    aspl_w, scc = aspl_on_lscc(Gd, weight="distance")
    aspl_u, _ = aspl_on_lscc(G, weight=None)
    feats["aspl_weighted"] = aspl_w        # distance=1/weight
    feats["aspl_unweighted"] = aspl_u      # hop count
    feats["scc_size"] = scc
    feats["strongly_connected"] = int(scc == N_NODES)

    # algebraic connectivity (Fiedler) of undirected weighted Laplacian
    U = undirected_weighted(G)
    L = nx.laplacian_matrix(U, weight="weight").toarray().astype(float)
    lap_eigs = np.sort(np.linalg.eigvalsh(L))
    feats["algebraic_connectivity"] = float(lap_eigs[1])

    # spectral radius: largest |eigenvalue| of weighted adjacency
    A = nx.to_numpy_array(G, weight="weight")
    adj_eigs = np.linalg.eigvals(A)
    feats["spectral_radius"] = float(np.max(np.abs(adj_eigs)))

    # ---- centrality summaries ----
    win = dict(G.in_degree(weight="weight"))
    wout = dict(G.out_degree(weight="weight"))
    btw = nx.betweenness_centrality(Gd, weight="distance", normalized=True)
    # Eigenvector centrality on the UNDIRECTED weighted graph: for directed
    # graphs that are not strongly connected (13 sparse networks here) the
    # numpy solver returns a non-real / NaN dominant eigenvector. The
    # undirected adjacency is symmetric, so its leading eigenvector is always
    # real and well-defined; this is the convention used in most passing-network
    # studies. Falls back to NaN only if the (rare) weakly-disconnected graph
    # still fails.
    try:
        eig = nx.eigenvector_centrality_numpy(U, weight="weight")
    except Exception:  # noqa: BLE001
        # last-resort: per-component leading eigenvector, padded
        eig = {n: float("nan") for n in G.nodes()}
        for comp in nx.connected_components(U):
            sub = U.subgraph(comp)
            if sub.number_of_nodes() == 1:
                eig[next(iter(comp))] = 0.0
                continue
            try:
                ec = nx.eigenvector_centrality_numpy(sub, weight="weight")
                eig.update(ec)
            except Exception:  # noqa: BLE001
                for n in comp:
                    eig[n] = 0.0
    pr = nx.pagerank(G, weight="weight")

    feats.update(summarize(win.values(), "windeg"))
    feats.update(summarize(wout.values(), "woutdeg"))
    feats.update(summarize(btw.values(), "betw"))
    feats.update(summarize(eig.values(), "eig"))
    feats.update(summarize(pr.values(), "pagerank"))

    # ---- motifs: triadic census, normalized ----
    tc = nx.triadic_census(G)
    assert sum(tc.values()) == N_TRIADS, "triad counts must sum to C(11,3)=165"
    for t in TRIAD_TYPES:
        feats[f"triad_{t}"] = tc.get(t, 0) / N_TRIADS

    return feats


def main():
    files = sorted(glob.glob(str(PROJECT_ROOT / "cache" / "networks" / "match*.json")))
    print("=" * 78)
    print(f"PHASE 2 — extracting features from {len(files)} networks")
    print("=" * 78)

    rows = []
    for i, f in enumerate(files, 1):
        mid = int(os.path.basename(f)[5:-5])
        G = load_network(mid)
        rows.append(extract(G))
        if i % 100 == 0:
            print(f"  ... {i}/{len(files)}")

    df = pd.DataFrame(rows)
    # season as ordered category for nice grouping
    df = df.sort_values(["season", "date"]).reset_index(drop=True)
    df.to_csv(OUT / "features.csv", index=False)
    print(f"\nWrote outputs/features.csv  ({df.shape[0]} rows x {df.shape[1]} cols)")

    # quick sanity prints
    print(f"\nnot strongly-connected networks: "
          f"{int((df['strongly_connected'] == 0).sum())}")
    print(f"triad rows sum to 1.0: "
          f"{np.allclose(df[[c for c in df.columns if c.startswith('triad_')]].sum(axis=1), 1.0)}")
    print("\nGlobal-metric column ranges:")
    for c in ["n_edges", "density", "total_passes", "weighted_clustering",
              "aspl_weighted", "aspl_unweighted", "algebraic_connectivity",
              "spectral_radius"]:
        print(f"  {c:<24} min {df[c].min():.3f}  median {df[c].median():.3f}  "
              f"max {df[c].max():.3f}")


if __name__ == "__main__":
    main()
