"""
Build self-contained interactive HTML visualizations + matching hi-res PNGs
for the live presentation.

Reuses the EXACT coordinates / labels / colors / cluster & position
assignments already produced by the analysis (no embeddings are recomputed):
  - P3 player-role UMAP  <- outputs/p3_embedding_coords.csv (+ player_roles.csv
                            for the raw-feature kNN that drives the
                            "structural successor" highlight)
  - P2 cluster UMAP      <- outputs/p2_embedding_coords.csv
  - Clasico passing net  <- cache/networks/match69299.json (PageRank recomputed
                            only for node sizing; the graph itself is the cached
                            built network)

Every HTML file is fully OFFLINE: the Plotly JS library is inlined
(include_plotlyjs=True) and the data is embedded in the file. No CDN, no server.

Outputs (to outputs/viz/):
  p3_umap_interactive.html, p2_umap_interactive.html, clasico_network.html
  p3_umap_hires.png, p2_umap_hires.png  (~1600 px wide)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from sb_cache import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs"
VIZ = OUT / "viz"
VIZ.mkdir(parents=True, exist_ok=True)

YORK_RED = "#E11837"

# Exact colors used in the static PNGs.
LINE_COLOR = {"GK": "#888888", "DEF": "#1f77b4", "MID": "#2ca02c",
              "FWD": "#d62728"}
# P2 clusters use matplotlib tab10: 0 -> blue, 1 -> orange.
CLUSTER_COLOR = {0: "#1f77b4", 1: "#ff7f0e"}

# These 14 features define the role space; kNN for the highlight is computed in
# this standardized space (identical to problem3_roles.landmark_neighbors).
FEATURE_KEYS = ["ego_degree", "ego_density", "ego_clustering", "win_deg",
                "wout_deg", "deg_ratio", "relay_mid", "source_2", "sink_2",
                "mean_x", "mean_y", "disp_x", "disp_y", "disp_radial"]

PLAYMAKERS = ["Xavi", "Andrés Iniesta", "Sergio Busquets", "Luka Modrić"]

BASE_FONT = dict(family="Helvetica, Arial, sans-serif", size=16, color="#222")


def _layout(title, subtitle, xaxis, yaxis):
    return go.Layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:14px'>"
                         f"{subtitle}</span>", x=0.5, xanchor="center",
                   font=dict(size=22)),
        font=BASE_FONT,
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(title=xaxis, showgrid=True, gridcolor="#eee",
                   zeroline=False),
        yaxis=dict(title=yaxis, showgrid=True, gridcolor="#eee",
                   zeroline=False, scaleanchor="x", scaleratio=1),
        legend=dict(font=dict(size=16), bordercolor="#ccc", borderwidth=1,
                    bgcolor="rgba(255,255,255,0.9)"),
        margin=dict(l=60, r=40, t=90, b=60),
        hoverlabel=dict(font_size=15),
    )


# ---------------------------------------------------------------------------
# 1. P3 player-role UMAP (highest priority)
# ---------------------------------------------------------------------------
def build_p3():
    coords = pd.read_csv(OUT / "p3_embedding_coords.csv")
    feats = pd.read_csv(OUT / "player_roles.csv")
    # align feats to coords order via player_id+team (unique per role profile)
    feats = feats.set_index(["player_id", "team"])
    coords = coords.set_index(["player_id", "team"])
    df = coords.join(feats[FEATURE_KEYS], how="left").reset_index()

    # kNN (k=10) in standardized raw-feature space -> the SAME neighbours the
    # report lists. Embed neighbour index lists for the interactive highlight.
    X = StandardScaler().fit_transform(df[FEATURE_KEYS].values)
    nn = NearestNeighbors(n_neighbors=11).fit(X)
    _, idx = nn.kneighbors(X)
    neighbors = [[int(j) for j in row[1:]] for row in idx]  # drop self

    xy = df[["umap_x", "umap_y"]].values
    names = df["name"].tolist()
    lines = df["line"].tolist()
    teams = df["team"].tolist()
    positions = df["primary_position"].tolist()
    apps = df["appearances"].tolist()

    fig = go.Figure(layout=_layout(
        "Player Role Space — UMAP of ego-network signatures",
        "271 players (Barça + Bayern + World Cups), coloured by position line · "
        "hover for player · click a name to highlight structural neighbours",
        "UMAP-1", "UMAP-2"))

    # one scatter trace per position line (for the legend + colors)
    for line in ["GK", "DEF", "MID", "FWD"]:
        m = df["line"] == line
        fig.add_trace(go.Scatter(
            x=df.loc[m, "umap_x"], y=df.loc[m, "umap_y"], mode="markers",
            name=line, marker=dict(size=10, color=LINE_COLOR[line],
                                   line=dict(width=0.5, color="white")),
            customdata=np.stack([df.loc[m, "name"], df.loc[m, "primary_position"],
                                 df.loc[m, "team"], df.loc[m, "appearances"]],
                                axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} "
                          "(%{customdata[2]})<br>%{customdata[3]} appearances"
                          "<extra></extra>"))

    # label the key playmakers directly on the plot; fan the labels out (they
    # cluster tightly in the deep-midfield region) so all four stay legible.
    pm_offsets = {"Xavi": (55, 45), "Andrés Iniesta": (60, 0),
                  "Sergio Busquets": (55, -45), "Luka Modrić": (-70, 55)}
    for pm in PLAYMAKERS:
        hit = df[df["name"] == pm]
        if len(hit):
            r = hit.iloc[0]
            ax_, ay_ = pm_offsets.get(pm, (20, -28))
            fig.add_annotation(x=r["umap_x"], y=r["umap_y"], text=f"<b>{pm}</b>",
                               showarrow=True, arrowhead=2, arrowcolor=YORK_RED,
                               arrowwidth=2, ax=ax_, ay=ay_,
                               font=dict(size=15, color=YORK_RED),
                               bgcolor="rgba(255,255,255,0.9)",
                               bordercolor=YORK_RED, borderwidth=1)

    # highlight trace (initially empty) for kNN lines + emphasised points
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="neighbour links",
                             line=dict(color=YORK_RED, width=1.5),
                             hoverinfo="skip", showlegend=False,
                             visible=True))
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="neighbours",
                             marker=dict(size=15, color="rgba(225,24,55,0)",
                                         line=dict(color=YORK_RED, width=3)),
                             hoverinfo="skip", showlegend=False))

    # Build the offline HTML with a custom search box + click handler.
    _write_p3_html(fig, names, xy.tolist(), neighbors, teams, positions, apps)

    # hi-res static PNG (matplotlib, matches the interactive styling)
    _png_p3(df)
    return df


def _write_p3_html(fig, names, xy, neighbors, teams, positions, apps):
    import plotly.io as pio
    # the two highlight traces are the last two; capture their indices
    n_traces = len(fig.data)
    link_idx, pts_idx = n_traces - 2, n_traces - 1

    div = pio.to_html(fig, include_plotlyjs=True, full_html=False,
                      div_id="p3plot",
                      config={"displaylogo": False, "responsive": True})

    data_js = json.dumps({"names": names, "xy": xy, "neighbors": neighbors,
                          "teams": teams, "positions": positions,
                          "apps": apps, "linkIdx": link_idx,
                          "ptsIdx": pts_idx})

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Player Role Space — Tactical DNA</title>
<style>
  body {{ font-family: Helvetica, Arial, sans-serif; margin: 0; background:#fff; }}
  #bar {{ padding: 12px 20px; background:#fff; border-bottom:2px solid {YORK_RED};
          display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}
  #bar label {{ font-size:16px; font-weight:bold; }}
  #player {{ font-size:16px; padding:6px 10px; min-width:260px; }}
  #clear {{ font-size:15px; padding:6px 14px; cursor:pointer;
            background:{YORK_RED}; color:#fff; border:none; border-radius:4px; }}
  #info {{ font-size:15px; color:#444; margin-left:8px; }}
  #p3plot {{ width:100vw; height:calc(100vh - 64px); }}
</style></head>
<body>
<div id="bar">
  <label for="player">Find a player:</label>
  <input list="players" id="player" placeholder="type a name… e.g. Xavi">
  <datalist id="players"></datalist>
  <button id="clear">Clear</button>
  <span id="info">Select a player to highlight its 10 nearest structural neighbours.</span>
</div>
{div}
<script>
const D = {data_js};
const gd = document.getElementById('p3plot');
const dl = document.getElementById('players');
D.names.forEach(n => {{ const o=document.createElement('option'); o.value=n; dl.appendChild(o); }});

function highlight(i) {{
  const nb = D.neighbors[i];
  const lx=[], ly=[], px=[], py=[];
  nb.forEach(j => {{
    lx.push(D.xy[i][0], D.xy[j][0], null);
    ly.push(D.xy[i][1], D.xy[j][1], null);
    px.push(D.xy[j][0]); py.push(D.xy[j][1]);
  }});
  px.push(D.xy[i][0]); py.push(D.xy[i][1]);
  Plotly.restyle(gd, {{x:[lx], y:[ly]}}, [D.linkIdx]);
  Plotly.restyle(gd, {{x:[px], y:[py]}}, [D.ptsIdx]);
  const names = nb.map(j => D.names[j]).join(', ');
  document.getElementById('info').innerHTML =
    '<b>'+D.names[i]+'</b> ('+D.positions[i]+', '+D.teams[i]+') → nearest: '+names;
}}
function clearHi() {{
  Plotly.restyle(gd, {{x:[[]], y:[[]]}}, [D.linkIdx, D.ptsIdx]);
  document.getElementById('info').textContent =
    'Select a player to highlight its 10 nearest structural neighbours.';
}}
document.getElementById('player').addEventListener('change', e => {{
  const i = D.names.indexOf(e.target.value);
  if (i >= 0) highlight(i);
}});
document.getElementById('clear').addEventListener('click', () => {{
  document.getElementById('player').value=''; clearHi();
}});
// also allow clicking a point to highlight it
gd.on('plotly_click', ev => {{
  const name = ev.points[0].customdata && ev.points[0].customdata[0];
  const i = D.names.indexOf(name);
  if (i >= 0) {{ document.getElementById('player').value=name; highlight(i); }}
}});
</script>
</body></html>"""
    path = VIZ / "p3_umap_interactive.html"
    path.write_text(html, encoding="utf-8")
    print(f"Wrote {path.relative_to(PROJECT_ROOT)} "
          f"({path.stat().st_size/1e6:.2f} MB)")


def _png_p3(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 11))
    for line in ["GK", "DEF", "MID", "FWD"]:
        m = df["line"] == line
        ax.scatter(df.loc[m, "umap_x"], df.loc[m, "umap_y"], s=70,
                   c=LINE_COLOR[line], label=line, alpha=0.8,
                   edgecolors="white", linewidths=0.6)
    # The four playmakers sit almost on top of each other in the deep-midfield
    # region, so fan their labels out in different directions with leader lines
    # to keep all four legible on a projected slide.
    pm_offsets = {"Xavi": (45, 35), "Andrés Iniesta": (45, -5),
                  "Sergio Busquets": (45, -45), "Luka Modrić": (-90, 40)}
    for pm in PLAYMAKERS:
        hit = df[df["name"] == pm]
        if len(hit):
            r = hit.iloc[0]
            dx, dy = pm_offsets.get(pm, (8, 8))
            ax.annotate(pm, (r["umap_x"], r["umap_y"]), fontsize=15,
                        fontweight="bold", color=YORK_RED,
                        xytext=(dx, dy), textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color=YORK_RED, lw=1.5),
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec=YORK_RED, alpha=0.9))
    ax.set_title("Player Role Space — UMAP of ego-network signatures (271 players)",
                 fontsize=20, fontweight="bold")
    ax.set_xlabel("UMAP-1", fontsize=16); ax.set_ylabel("UMAP-2", fontsize=16)
    ax.legend(title="position line", fontsize=15, title_fontsize=15)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#eee")
    fig.tight_layout()
    p = OUT / "p3_umap_hires.png"
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p.relative_to(PROJECT_ROOT)} "
          f"({p.stat().st_size/1e6:.2f} MB)")


# ---------------------------------------------------------------------------
# 2. P2 cluster UMAP
# ---------------------------------------------------------------------------
def build_p2():
    df = pd.read_csv(OUT / "p2_embedding_coords.csv")

    fig = go.Figure(layout=_layout(
        "Tactical Archetypes — UMAP of passing-network features",
        "774 team-match networks (Barça + Bayern + World Cups), k=2 clusters · "
        "hover for match",
        "UMAP-1", "UMAP-2"))

    cluster_name = {0: "Cluster 0 — more direct", 1: "Cluster 1 — possession"}
    for c in [0, 1]:
        m = df["cluster"] == c
        fig.add_trace(go.Scatter(
            x=df.loc[m, "umap_x"], y=df.loc[m, "umap_y"], mode="markers",
            name=cluster_name[c],
            marker=dict(size=8, color=CLUSTER_COLOR[c],
                        line=dict(width=0.3, color="white")),
            customdata=np.stack([df.loc[m, "team"], df.loc[m, "competition"],
                                 df.loc[m, "date"].astype(str),
                                 df.loc[m, "coach"].fillna("")], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · "
                          "%{customdata[2]}<br>coach: %{customdata[3]}"
                          "<extra></extra>"))

    # annotate landmarks (flagged in the export)
    for _, r in df[df["landmark"].fillna("") != ""].iterrows():
        fig.add_annotation(x=r["umap_x"], y=r["umap_y"], text=f"<b>{r['landmark']}</b>",
                           showarrow=True, arrowhead=2, arrowcolor=YORK_RED,
                           arrowwidth=2, ax=30, ay=-30,
                           font=dict(size=15, color=YORK_RED),
                           bgcolor="rgba(255,255,255,0.9)",
                           bordercolor=YORK_RED, borderwidth=1)

    import plotly.io as pio
    html = pio.to_html(fig, include_plotlyjs=True, full_html=True,
                       config={"displaylogo": False, "responsive": True})
    path = VIZ / "p2_umap_interactive.html"
    path.write_text(html, encoding="utf-8")
    print(f"Wrote {path.relative_to(PROJECT_ROOT)} "
          f"({path.stat().st_size/1e6:.2f} MB)")

    _png_p2(df, cluster_name)


def _png_p2(df, cluster_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 11))
    for c in [0, 1]:
        m = df["cluster"] == c
        ax.scatter(df.loc[m, "umap_x"], df.loc[m, "umap_y"], s=45,
                   c=CLUSTER_COLOR[c], label=cluster_name[c], alpha=0.75,
                   edgecolors="white", linewidths=0.3)
    for _, r in df[df["landmark"].fillna("") != ""].iterrows():
        ax.annotate(r["landmark"], (r["umap_x"], r["umap_y"]), fontsize=14,
                    fontweight="bold", color=YORK_RED,
                    xytext=(10, 10), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color=YORK_RED, lw=2),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=YORK_RED, alpha=0.9))
    ax.set_title("Tactical Archetypes — UMAP of passing-network features "
                 "(774 networks, k=2)", fontsize=20, fontweight="bold")
    ax.set_xlabel("UMAP-1", fontsize=16); ax.set_ylabel("UMAP-2", fontsize=16)
    ax.legend(fontsize=15)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#eee")
    fig.tight_layout()
    p = OUT / "p2_umap_hires.png"
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p.relative_to(PROJECT_ROOT)} "
          f"({p.stat().st_size/1e6:.2f} MB)")


# ---------------------------------------------------------------------------
# 3. Clasico passing network (optional)
# ---------------------------------------------------------------------------
def build_clasico():
    import networkx as nx
    path = PROJECT_ROOT / "cache" / "networks" / "match69299.json"
    if not path.exists():
        print("Clasico network not cached; skipping.")
        return
    data = json.loads(path.read_text())
    G = nx.DiGraph()
    for n in data["nodes"]:
        G.add_node(int(n["id"]), label=n.get("label", str(n["id"])))
    for e in data["edges"]:
        G.add_edge(int(e["source"]), int(e["target"]), weight=int(e["weight"]))

    pr = nx.pagerank(G, weight="weight")
    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.9, iterations=200)

    edge_x, edge_y, ew = [], [], []
    for u, v, d in G.edges(data=True):
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
        ew.append(d["weight"])
    wmax = max(ew) if ew else 1

    fig = go.Figure(layout=_layout(
        "El Clásico passing network — Barça 5–0 Real Madrid (2010, Guardiola)",
        "nodes sized & coloured by PageRank · edges weighted by completed passes",
        "", ""))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)

    # edges as one grey trace (width can't vary within a trace; use mean look)
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color="rgba(150,150,150,0.45)", width=1.2),
                             hoverinfo="skip", showlegend=False))

    nx_, ny_, sizes, colors, texts = [], [], [], [], []
    prv = np.array([pr[n] for n in G.nodes()])
    for n in G.nodes():
        nx_.append(pos[n][0]); ny_.append(pos[n][1])
        sizes.append(20 + 70 * (pr[n] - prv.min()) / (np.ptp(prv) + 1e-9))
        colors.append(pr[n])
        texts.append(f"<b>{G.nodes[n]['label']}</b><br>PageRank: {pr[n]:.3f}")
    fig.add_trace(go.Scatter(
        x=nx_, y=ny_, mode="markers+text",
        text=[G.nodes[n]["label"] for n in G.nodes()],
        textposition="top center", textfont=dict(size=13),
        marker=dict(size=sizes, color=colors, colorscale="YlOrRd",
                    line=dict(color="#7a0a0a", width=1.5),
                    colorbar=dict(title="PageRank"), showscale=True),
        hovertext=texts, hoverinfo="text", showlegend=False))

    import plotly.io as pio
    html = pio.to_html(fig, include_plotlyjs=True, full_html=True,
                       config={"displaylogo": False, "responsive": True})
    p = VIZ / "clasico_network.html"
    p.write_text(html, encoding="utf-8")
    print(f"Wrote {p.relative_to(PROJECT_ROOT)} "
          f"({p.stat().st_size/1e6:.2f} MB)")


def main():
    print("=" * 70)
    print("BUILDING INTERACTIVE VISUALIZATIONS (offline, self-contained)")
    print("=" * 70)
    build_p3()
    build_p2()
    build_clasico()
    print("\nAll visualizations written to outputs/viz/ and hi-res PNGs to outputs/.")


if __name__ == "__main__":
    main()
