"""
Problem 2 — Tactical Archetype Discovery (clustering).

On the merged corpus (Barca + Bayern + World Cup), using the 44 structural
features (8 global + 20 centrality summaries + 16 motifs; ids/metadata excluded):

  4. Standardise features. Run k-means and agglomerative (Ward) clustering;
     report silhouette across k=2..10 and the elbow (k-means inertia) curve,
     and pick k.
  5. For the chosen k, report silhouette, modularity (of the cluster partition
     on a k-NN similarity graph over the feature space), and intra- vs
     inter-cluster Euclidean distance. List a sample of matches per cluster
     (team, competition, coach) for interpretation.
  6. UMAP projection coloured by cluster -> PNG, with a couple of landmark
     matches annotated (a Pep-era Clasico, the 2022 WC Final) if present.

Outputs:
  outputs/p2_kselection.csv, outputs/p2_elbow_silhouette.png,
  outputs/p2_cluster_quality.csv, outputs/p2_cluster_samples.txt,
  outputs/p2_umap.png
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from sb_cache import PROJECT_ROOT
from phase3_p1 import feature_columns

OUT = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42
K_RANGE = list(range(2, 11))

LANDMARKS = [
    # (match_id, team, label for annotation)
    (69299, "Barcelona", "Clasico 5-0 (Pep, 2010)"),
]
WC2022_FINAL_TEAMS = {"Argentina", "France"}  # 2022 final


def cluster_features(df):
    """Return the 44 structural feature columns used for clustering."""
    cent, motif, glob, combined = feature_columns(df)
    return combined


def k_selection(X):
    """
    Sweep k = 2..10 for both k-means and Ward clustering, returning a per-k
    DataFrame plus the k-means inertia (elbow) and silhouette curves used to
    choose the number of clusters.
    """
    rows = []
    inertias, sils_km, sils_ag = [], [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
        s_km = silhouette_score(X, km.labels_)
        s_ag = silhouette_score(X, ag.labels_)
        inertias.append(km.inertia_)
        sils_km.append(s_km)
        sils_ag.append(s_ag)
        rows.append({"k": k, "kmeans_inertia": km.inertia_,
                     "kmeans_silhouette": s_km, "ward_silhouette": s_ag})
        print(f"  k={k:>2}  inertia={km.inertia_:>10.1f}  "
              f"sil(kmeans)={s_km:.3f}  sil(ward)={s_ag:.3f}")
    return pd.DataFrame(rows), inertias, sils_km, sils_ag


def plot_elbow_sil(ks, inertias, sils_km, sils_ag, chosen_k):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(ks, inertias, "o-", color="steelblue")
    a1.axvline(chosen_k, ls="--", color="grey")
    a1.set_xlabel("k"); a1.set_ylabel("k-means inertia (SSE)")
    a1.set_title("Elbow curve")
    a2.plot(ks, sils_km, "o-", label="k-means")
    a2.plot(ks, sils_ag, "s-", label="Ward")
    a2.axvline(chosen_k, ls="--", color="grey", label=f"chosen k={chosen_k}")
    a2.set_xlabel("k"); a2.set_ylabel("silhouette score")
    a2.set_title("Silhouette vs k"); a2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "p2_elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def modularity_of_partition(X, labels, n_neighbors=15):
    """Modularity of the cluster partition on a kNN similarity graph."""
    A = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity",
                         include_self=False)
    A = 0.5 * (A + A.T)  # symmetrise
    G = nx.from_scipy_sparse_array(A)
    communities = {}
    for node, lab in enumerate(labels):
        communities.setdefault(lab, set()).add(node)
    return nx.community.modularity(G, list(communities.values()))


def intra_inter_distance(X, labels):
    """Mean intra-cluster and inter-cluster Euclidean distance (centroid-based)."""
    uniq = np.unique(labels)
    centroids = {c: X[labels == c].mean(axis=0) for c in uniq}
    intra = []
    for c in uniq:
        pts = X[labels == c]
        intra.append(np.linalg.norm(pts - centroids[c], axis=1).mean())
    inter = []
    cs = list(uniq)
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            inter.append(np.linalg.norm(centroids[cs[i]] - centroids[cs[j]]))
    return float(np.mean(intra)), float(np.mean(inter))


def main():
    df = pd.read_csv(OUT / "features_all.csv")
    cols = cluster_features(df)

    # Drop degenerate networks: a few World Cup team-matches have an extremely
    # short pre-substitution window (e.g. Iran WC2022 with 2 passes) leaving a
    # disconnected graph whose ASPL is undefined (NaN). These are not
    # meaningful passing networks; we exclude them from clustering and report
    # the count.
    n0 = len(df)
    finite = df[cols].notna().all(axis=1)
    dropped = df[~finite]
    df = df[finite].reset_index(drop=True)
    if len(dropped):
        print(f"Dropped {len(dropped)} degenerate network(s) with undefined "
              f"features (e.g. "
              f"{dropped.iloc[0]['team']} {dropped.iloc[0]['competition']}, "
              f"{int(dropped.iloc[0]['total_passes'])} passes).")

    X = StandardScaler().fit_transform(df[cols].values)
    print("=" * 74)
    print(f"PROBLEM 2 — clustering {len(df)} networks on {len(cols)} features "
          f"(from {n0}; -{len(dropped)} degenerate)")
    print("=" * 74)

    # ---- k selection ----
    print("\nk-selection (k=2..10):")
    ksel, inertias, sils_km, sils_ag = k_selection(X)
    ksel.to_csv(OUT / "p2_kselection.csv", index=False)

    # choose k = argmax mean silhouette across the two methods
    ksel["mean_sil"] = ksel[["kmeans_silhouette", "ward_silhouette"]].mean(axis=1)
    chosen_k = int(ksel.loc[ksel["mean_sil"].idxmax(), "k"])
    print(f"\nChosen k = {chosen_k} (max mean silhouette = "
          f"{ksel['mean_sil'].max():.3f})")
    plot_elbow_sil(K_RANGE, inertias, sils_km, sils_ag, chosen_k)

    # ---- fit chosen k with both methods; pick the better-silhouette one ----
    km = KMeans(n_clusters=chosen_k, n_init=10,
                random_state=RANDOM_STATE).fit(X)
    ag = AgglomerativeClustering(n_clusters=chosen_k, linkage="ward").fit(X)
    s_km, s_ag = silhouette_score(X, km.labels_), silhouette_score(X, ag.labels_)
    if s_km >= s_ag:
        labels, method, sil = km.labels_, "k-means", s_km
    else:
        labels, method, sil = ag.labels_, "ward", s_ag
    df["cluster"] = labels
    print(f"\nFinal partition: {method} (silhouette {sil:.3f})")

    # ---- quality metrics ----
    mod = modularity_of_partition(X, labels)
    intra, inter = intra_inter_distance(X, labels)
    print("-" * 74)
    print(f"CLUSTER QUALITY (k={chosen_k}, {method}):")
    print(f"  silhouette            : {sil:.3f}")
    print(f"  modularity (kNN graph): {mod:.3f}")
    print(f"  mean intra-cluster dist: {intra:.3f}")
    print(f"  mean inter-cluster dist: {inter:.3f}")
    print(f"  inter/intra ratio      : {inter/intra:.2f}")
    pd.DataFrame([{"k": chosen_k, "method": method, "silhouette": sil,
                   "modularity": mod, "intra_dist": intra,
                   "inter_dist": inter,
                   "inter_intra_ratio": inter/intra}]
                 ).to_csv(OUT / "p2_cluster_quality.csv", index=False)

    # ---- per-cluster composition + samples ----
    lines = []
    print("\n" + "-" * 74)
    print("CLUSTER COMPOSITION & SAMPLES")
    print("-" * 74)
    rng = np.random.default_rng(RANDOM_STATE)
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        comp_mix = dict(sub["competition"].value_counts())
        team_mix = dict(sub["team"].value_counts().head(5))
        coach_mix = dict(sub[sub["coach"] != ""]["coach"].value_counts().head(5))
        head = (f"\nCLUSTER {c}: {len(sub)} networks | "
                f"competitions={comp_mix}")
        print(head)
        print(f"  top teams: {team_mix}")
        print(f"  labelled coaches: {coach_mix}")
        lines.append(head)
        lines.append(f"  top teams: {team_mix}")
        lines.append(f"  labelled coaches: {coach_mix}")
        # sample up to 8 matches
        samp = sub.sample(min(8, len(sub)), random_state=RANDOM_STATE)
        print("  sample matches (team | competition | coach | date):")
        lines.append("  sample matches:")
        for _, r in samp.iterrows():
            coach = r["coach"] if r["coach"] else "-"
            s = (f"    {str(r['team'])[:22]:<22} | {str(r['competition']):<11} "
                 f"| {coach:<10} | {str(r['date'])[:10]}")
            print(s)
            lines.append(s)
    (OUT / "p2_cluster_samples.txt").write_text("\n".join(lines), encoding="utf-8")

    # ---- UMAP ----
    try:
        make_umap(df, X, labels, chosen_k)
        print("\nSaved outputs/p2_umap.png")
    except Exception as e:  # noqa: BLE001
        print(f"\nUMAP skipped: {e}")

    # ---- copy-paste summary ----
    print("\n" + "=" * 74)
    print("PROBLEM 2 SUMMARY (copy-paste)")
    print("=" * 74)
    print(f"- Clustered {len(df)} networks (Barca+Bayern+WC) on {len(cols)} "
          f"standardised features.")
    print(f"- k-selection by silhouette over k=2..10 -> chose k={chosen_k} "
          f"({method}).")
    print(f"- Quality: silhouette {sil:.3f}, modularity {mod:.3f}, "
          f"intra {intra:.2f} / inter {inter:.2f} "
          f"(ratio {inter/intra:.2f}).")
    sizes = dict(df['cluster'].value_counts().sort_index())
    print(f"- Cluster sizes: {sizes}")
    print(f"- Samples -> p2_cluster_samples.txt ; UMAP -> p2_umap.png ; "
          f"k-curves -> p2_elbow_silhouette.png")


def make_umap(df, X, labels, k):
    import umap  # noqa
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=RANDOM_STATE)
    emb = reducer.fit_transform(X)

    # Export the exact 2D coordinates + cluster labels (same seeded embedding
    # used for the PNG) so the interactive HTML can reuse them verbatim. This
    # writes coordinates only and does not alter any computation or result.
    export = df[["match_id", "team", "competition", "date", "coach",
                 "cluster"]].copy()
    export["umap_x"] = emb[:, 0]
    export["umap_y"] = emb[:, 1]
    # flag the landmark matches so the HTML annotates the same points
    export["landmark"] = ""
    for mid, team, text in LANDMARKS:
        m = (export["match_id"] == mid) & (export["team"] == team)
        export.loc[m, "landmark"] = text
    fin = (export["competition"] == "WC2022") & \
        (export["team"].isin(WC2022_FINAL_TEAMS))
    if fin.any():
        fin_df = export[fin]
        last = fin_df[fin_df["date"] == fin_df["date"].max()]
        if len(last):
            export.loc[last.index[0], "landmark"] = "WC2022 Final"
    export.to_csv(OUT / "p2_embedding_coords.csv", index=False)
    print(f"Wrote outputs/p2_embedding_coords.csv "
          f"({len(export)} networks x {export.shape[1]} cols)")

    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=18,
                    alpha=0.75)
    ax.set_title(f"UMAP of passing-network feature space "
                 f"(coloured by cluster, k={k})\n"
                 f"Barca (La Liga) + Bayern + World Cup 2018/2022")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    legend = ax.legend(*sc.legend_elements(), title="cluster",
                       loc="best", fontsize=8)
    ax.add_artist(legend)

    # annotate landmarks
    def annotate(mask, text):
        if mask.any():
            idx = np.where(mask.values)[0][0]
            ax.annotate(text, (emb[idx, 0], emb[idx, 1]),
                        fontsize=9, fontweight="bold",
                        xytext=(10, 10), textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color="black"))

    for mid, team, text in LANDMARKS:
        annotate((df["match_id"] == mid) & (df["team"] == team), text)
    # 2022 WC Final (Argentina or France in WC2022)
    fin = (df["competition"] == "WC2022") & (df["team"].isin(WC2022_FINAL_TEAMS))
    # the final is the last WC2022 date; pick Argentina row on max date
    if fin.any():
        fin_df = df[fin]
        last = fin_df[fin_df["date"] == fin_df["date"].max()]
        if len(last):
            ridx = df.index.get_loc(last.index[0])
            ax.annotate("WC2022 Final", (emb[ridx, 0], emb[ridx, 1]),
                        fontsize=9, fontweight="bold",
                        xytext=(10, -15), textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color="darkred"))

    fig.tight_layout()
    fig.savefig(OUT / "p2_umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
