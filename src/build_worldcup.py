"""
Build World Cup 2018 + 2022 passing networks (BOTH teams per match) and the
50-feature matrix, using the IDENTICAL pipeline as Barca/Bayern:
11-starter pre-first-substitution window, completed open-play passes, 11 nodes.

For Problem 2 (tactical archetype discovery) we want as many diverse
team-match networks as possible, so we build a network for *each* team in
*every* World Cup match (not just one team).

Each row is tagged with team, competition (WC2018/WC2022), and date.
Coach is left blank here (national-team coaches are added only where needed
for the transfer test, in transfer_test_multi.py).

Outputs:
  cache/networks_wc/match<MID>_<TEAMSLUG>.json
  outputs/features_worldcup.csv   (same 50 feature columns + team/competition)
"""
from __future__ import annotations

import re
from pathlib import Path

import networkx as nx
import pandas as pd

from sb_cache import matches, events, lineups, PROJECT_ROOT
from build_networks import (starting_xi, first_sub_index, _nickname_map,
                            SETPIECE_PASS_TYPES, _write_network, load_network)
from features import extract

WC = [(43, 3, "WC2018"), (43, 106, "WC2022")]
WC_NET_CACHE = PROJECT_ROOT / "cache" / "networks_wc"
WC_NET_CACHE.mkdir(parents=True, exist_ok=True)
OUT = PROJECT_ROOT / "outputs"


def _slug(team: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", team)[:24]


def build_team_network(match_id: int, team: str, comp_tag: str,
                       date: str) -> dict | None:
    """Build one team's network for a match. Returns feature row or None."""
    ev = events(match_id)
    starters = starting_xi(match_id, team, ev=ev)
    if len(starters) != 11:
        return None

    cut = first_sub_index(ev)  # first sub by EITHER team
    window = ev if cut is None else ev[ev["index"] < cut]
    first_sub_min = (None if cut is None
                     else int(ev[ev["index"] == cut].iloc[0]["minute"]))

    passes = window[(window["type"] == "Pass") & (window["team"] == team)].copy()
    passes = passes[passes["pass_outcome"].isna()]
    passes = passes[~passes["pass_type"].isin(SETPIECE_PASS_TYPES)]
    sids = set(starters)
    passes = passes[passes["player_id"].isin(sids)
                    & passes["pass_recipient_id"].isin(sids)]

    nicks = _nickname_map(match_id, team)
    G = nx.DiGraph()
    for pid, pname in starters.items():
        G.add_node(int(pid), name=pname, label=nicks.get(int(pid), pname))
    agg = (passes.groupby([passes["player_id"].astype(int),
                           passes["pass_recipient_id"].astype(int)]).size())
    total = 0
    for (u, v), w in agg.items():
        if u == v:
            continue
        G.add_edge(int(u), int(v), weight=int(w))
        total += int(w)

    if G.number_of_nodes() != 11:
        return None
    assert G.number_of_nodes() == 11

    season_tag = comp_tag  # use the competition tag as the "season" slot
    G.graph.update({"match_id": match_id, "season": season_tag, "date": date,
                    "team": team, "competition_id": 43,
                    "first_sub_minute": first_sub_min, "total_passes": total})
    path = WC_NET_CACHE / f"match{match_id}_{_slug(team)}.json"
    # reuse the canonical writer by temporarily pointing it at our file
    _write_network_to(path, G)

    row = extract(G)
    row["team"] = team
    row["competition"] = comp_tag
    return row


def _write_network_to(path: Path, G: nx.DiGraph) -> None:
    import json
    data = {"graph": G.graph,
            "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes],
            "edges": [{"source": u, "target": v, "weight": d["weight"]}
                      for u, v, d in G.edges(data=True)]}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)


def main():
    print("=" * 74)
    print("BUILD — World Cup 2018 + 2022 passing networks (both teams/match)")
    print("=" * 74)

    rows, skipped = [], 0
    teams_seen = set()
    for comp, season, tag in WC:
        md = matches(comp, season).sort_values("match_date")
        n_built = 0
        for _, m in md.iterrows():
            mid = int(m["match_id"])
            date = str(m["match_date"])
            for team in (m["home_team"], m["away_team"]):
                r = build_team_network(mid, team, tag, date)
                if r is None:
                    skipped += 1
                else:
                    rows.append(r)
                    teams_seen.add(team)
                    n_built += 1
        print(f"  {tag}: {n_built} team-match networks built "
              f"({len(md)} matches)")

    df = pd.DataFrame(rows).sort_values(["competition", "date", "team"])
    df = df.reset_index(drop=True)
    df.to_csv(OUT / "features_worldcup.csv", index=False)

    print("-" * 74)
    print(f"Total WC team-match networks: {len(df)}  (skipped {skipped})")
    print(f"Distinct national teams: {df['team'].nunique()}")
    print(f"  per competition: "
          f"{dict(df.groupby('competition')['team'].nunique())}")
    print(f"Wrote outputs/features_worldcup.csv  ({df.shape[0]} x {df.shape[1]})")
    print("\nTeams:")
    for t in sorted(teams_seen):
        print(f"  {t}")


if __name__ == "__main__":
    main()
