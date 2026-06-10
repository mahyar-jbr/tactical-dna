"""
Phase 2 reporting — corpus-size table, summary-stats table, example figure.

Reads outputs/features.csv and produces:
  (a) corpus-size table   : match count per season  -> printed + corpus_table.csv
  (b) summary-stats table : mean +/- std of each GLOBAL metric, overall and
                            per season               -> printed + summary_stats.csv
  (c) example_network.png : the 2010-11-29 Barca 5-0 Real Madrid "Manita"
                            (Guardiola), nodes laid out & sized by PageRank,
                            edges by completed-pass weight.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from build_networks import load_network
from sb_cache import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs"

# Global metrics reported in the summary-stats table.
GLOBAL_METRICS = [
    "n_edges", "density", "total_passes", "weighted_clustering",
    "aspl_weighted", "aspl_unweighted", "algebraic_connectivity",
    "spectral_radius",
]
PRETTY = {
    "n_edges": "Edges",
    "density": "Density",
    "total_passes": "Total passes",
    "weighted_clustering": "Wgt. clustering",
    "aspl_weighted": "ASPL (1/w dist)",
    "aspl_unweighted": "ASPL (hops)",
    "algebraic_connectivity": "Algebraic conn. (Fiedler)",
    "spectral_radius": "Spectral radius",
}

EXAMPLE_MATCH = 69299  # 2010-11-29 Barca 5-0 Real Madrid, Guardiola "Manita"


def corpus_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-season match count + mean passes/edges (the report's Table 1)."""
    g = (df.groupby("season")
         .agg(matches=("match_id", "count"),
              mean_passes=("total_passes", "mean"),
              mean_edges=("n_edges", "mean"))
         .reset_index())
    g["mean_passes"] = g["mean_passes"].round(1)
    g["mean_edges"] = g["mean_edges"].round(1)
    return g


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """mean +/- std of each global metric, overall and per season."""
    records = []
    # overall
    row = {"season": "ALL", "n": len(df)}
    for m in GLOBAL_METRICS:
        row[f"{m}_mean"] = df[m].mean()
        row[f"{m}_std"] = df[m].std()
    records.append(row)
    # per season
    for season, sub in df.groupby("season"):
        row = {"season": season, "n": len(sub)}
        for m in GLOBAL_METRICS:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std()
        records.append(row)
    return pd.DataFrame(records)


def print_corpus(g: pd.DataFrame):
    print("=" * 70)
    print("(a) CORPUS-SIZE TABLE  — matches per season")
    print("=" * 70)
    print(f"{'season':<12}{'matches':>9}{'mean passes':>14}{'mean edges':>13}")
    print("-" * 48)
    for _, r in g.iterrows():
        print(f"{r['season']:<12}{int(r['matches']):>9}"
              f"{r['mean_passes']:>14.1f}{r['mean_edges']:>13.1f}")
    print("-" * 48)
    print(f"{'TOTAL':<12}{int(g['matches'].sum()):>9}")


def print_summary(ss: pd.DataFrame):
    print("\n" + "=" * 70)
    print("(b) SUMMARY-STATS TABLE — mean +/- std of each global metric")
    print("=" * 70)
    # overall block (transposed for readability)
    overall = ss[ss["season"] == "ALL"].iloc[0]
    print(f"\nOVERALL (n = {int(overall['n'])} networks):")
    print(f"  {'metric':<28}{'mean':>12}{'std':>12}")
    print("  " + "-" * 52)
    for m in GLOBAL_METRICS:
        print(f"  {PRETTY[m]:<28}{overall[f'{m}_mean']:>12.3f}"
              f"{overall[f'{m}_std']:>12.3f}")

    # per-season: show mean+/-std for a few headline metrics in a compact grid
    print("\nPER SEASON (mean +/- std):")
    cols = ["density", "weighted_clustering", "algebraic_connectivity",
            "spectral_radius"]
    hdr = f"  {'season':<11}{'n':>4}  " + "".join(
        f"{PRETTY[c]:>26}" for c in cols)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in ss[ss["season"] != "ALL"].iterrows():
        cells = "".join(
            f"{r[c+'_mean']:>13.3f}+/-{r[c+'_std']:<10.3f}" for c in cols)
        print(f"  {r['season']:<11}{int(r['n']):>4}  {cells}")


def make_figure(match_id: int):
    G = load_network(match_id)
    pr = nx.pagerank(G, weight="weight")

    # layout: spring layout weighted by pass volume gives a tactical shape
    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.9, iterations=200)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    # node sizes by PageRank
    pr_vals = np.array([pr[n] for n in G.nodes()])
    sizes = 800 + 9000 * (pr_vals - pr_vals.min()) / (np.ptp(pr_vals) + 1e-9)

    # edge widths by weight
    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    widths = 0.4 + 4.5 * (weights / weights.max())

    nx.draw_networkx_edges(
        G, pos, ax=ax, width=widths, edge_color="#888888",
        alpha=0.55, arrows=True, arrowsize=9,
        connectionstyle="arc3,rad=0.08", node_size=sizes)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=sizes, node_color=pr_vals,
        cmap="YlOrRd", edgecolors="#7a0a0a", linewidths=1.5)

    # short labels: StatsBomb nickname (e.g. "Xavi", "Lionel Messi"), stored
    # on the node at build time; fall back to the full name's last token.
    labels = {}
    for n in G.nodes():
        lab = G.nodes[n].get("label")
        if not lab:
            name = G.nodes[n].get("name", str(n))
            lab = name.split()[-1] if name else str(n)
        labels[n] = lab
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=9, font_weight="bold")

    season = G.graph["season"]
    date = G.graph["date"]
    tp = G.graph["total_passes"]
    ax.set_title(
        f"FC Barcelona passing network — El Clasico, {date} ({season})\n"
        f"Barca 5-0 Real Madrid (Guardiola). "
        f"Nodes sized/coloured by PageRank, edges by completed-pass count "
        f"(window: pre-first-sub, {tp} passes).",
        fontsize=11)
    ax.axis("off")

    sm = plt.cm.ScalarMappable(
        cmap="YlOrRd",
        norm=plt.Normalize(vmin=pr_vals.min(), vmax=pr_vals.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("PageRank", fontsize=9)

    fig.tight_layout()
    out_path = OUT / "example_network.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path.relative_to(PROJECT_ROOT)} "
          f"(top PageRank: "
          f"{G.nodes[max(pr, key=pr.get)]['name']} "
          f"= {max(pr.values()):.3f})")


def main():
    df = pd.read_csv(OUT / "features.csv")

    g = corpus_table(df)
    g.to_csv(OUT / "corpus_table.csv", index=False)
    print_corpus(g)

    ss = summary_stats(df)
    ss.to_csv(OUT / "summary_stats.csv", index=False)
    print_summary(ss)

    make_figure(EXAMPLE_MATCH)

    # copy-paste block
    overall = ss[ss["season"] == "ALL"].iloc[0]
    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY (copy-paste)")
    print("=" * 70)
    print(f"- features.csv: {df.shape[0]} networks x {df.shape[1]} features "
          f"(8 global + 20 centrality summaries + 16 motif fractions + ids).")
    print(f"- Corpus: {int(g['matches'].sum())} networks across "
          f"{len(g)} seasons ({g['matches'].min()}-{g['matches'].max()} per season).")
    print(f"- Overall density {overall['density_mean']:.3f}+/-{overall['density_std']:.3f}, "
          f"weighted clustering {overall['weighted_clustering_mean']:.3f}"
          f"+/-{overall['weighted_clustering_std']:.3f}, "
          f"spectral radius {overall['spectral_radius_mean']:.1f}"
          f"+/-{overall['spectral_radius_std']:.1f}.")
    print(f"- Tables -> corpus_table.csv, summary_stats.csv ; "
          f"figure -> example_network.png (Manita Clasico).")


if __name__ == "__main__":
    main()
