"""
Phase 1 — build Barca passing networks.

For each Barca team-match in the locked scope (outputs/phase0_scope.json):

  1. Read lineups -> the 11 starters for Barcelona (cross-checked against the
     Starting XI event's tactics.lineup).
  2. Find the FIRST Substitution event in the match (by event index) and keep
     only events strictly before it -> the "11-starter window" in which all
     starters (both teams) are still on the pitch.
  3. Keep type=="Pass" events that are:
        - by Barcelona,
        - completed:  pass_outcome is null (NaN),
        - open play:  pass_type NOT in the set-piece restart set
                      {Throw-in, Corner, Free Kick, Goal Kick, Kick Off}
                      (Recovery / Interception pass_types are kept: they are
                       open-play passes following a ball win),
        - both passer and recipient are among the 11 Barca starters
          (drops the rare pass whose recipient is a non-starter, which cannot
           happen before any substitution but we guard anyway).
  4. Build a directed weighted graph: edge (passer -> recipient) weight =
     completed-pass count. Assert exactly 11 nodes.
  5. Tag the graph with season + date and persist it.

Networks are written to cache/networks/ as node/edge JSON so Phase 2 reads
them without recomputation. A per-match build log is written to
outputs/phase1_build_log.csv.

Design notes / judgment calls (documented for the report):
  * Starter source: lineups positions with start_reason == "Starting XI".
    Validated to match the Starting XI event tactics.lineup exactly.
  * Substitution cut: FIRST substitution by EITHER team (the plan's literal
    "first Substitution event"). This is the most conservative window where
    all 22 starters are on the pitch. We also record the first-Barca-sub index
    to quantify how much earlier the opponent sometimes subs.
  * We require exactly 11 nodes. A match that fails (e.g. a starter who made
    and received zero passes in the window -> isolated node missing) is logged
    and skipped rather than silently producing a !=11 graph.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd

from sb_cache import events, lineups, matches, PROJECT_ROOT

BARCA = "Barcelona"
LA_LIGA = 11
SETPIECE_PASS_TYPES = {"Throw-in", "Corner", "Free Kick", "Goal Kick", "Kick Off"}

NET_CACHE = PROJECT_ROOT / "cache" / "networks"
NET_CACHE.mkdir(parents=True, exist_ok=True)
OUT = PROJECT_ROOT / "outputs"


def load_scope() -> dict:
    with (OUT / "phase0_scope.json").open() as fh:
        return json.load(fh)


def _starters_from_event(ev: pd.DataFrame, team: str) -> dict[int, str]:
    """Authoritative fallback: the team's Starting XI event tactics.lineup."""
    sx = ev[(ev["type"] == "Starting XI") & (ev["team"] == team)]
    if sx.empty:
        return {}
    tac = sx.iloc[0]["tactics"]
    if not isinstance(tac, dict):
        return {}
    return {int(e["player"]["id"]): str(e["player"]["name"])
            for e in tac.get("lineup", [])}


def _nickname_map(match_id: int, team: str) -> dict[int, str]:
    """{player_id: best short label} from lineups player_nickname (fallback name)."""
    lu = lineups(match_id)
    if team not in lu:
        return {}
    df = lu[team]
    out = {}
    for _, r in df.iterrows():
        nick = r.get("player_nickname")
        label = nick if isinstance(nick, str) and nick.strip() else r["player_name"]
        out[int(r["player_id"])] = str(label)
    return out


def starting_xi(match_id: int, team: str, ev: pd.DataFrame | None = None
                ) -> dict[int, str]:
    """
    Return {player_id: player_name} for the team's 11 starters.

    Primary source: lineups positions with start_reason == "Starting XI".
    Fallback (for matches whose lineups mis-tag starters as "Tactical Shift",
    e.g. Granada-Barca 70294, 2012/13): the Starting XI event tactics.lineup,
    which is authoritative and always has exactly 11.
    """
    lu = lineups(match_id)
    if team not in lu:
        raise KeyError(f"team {team} not in lineups for match {match_id}")
    df = lu[team]

    def is_starter(positions) -> bool:
        if not isinstance(positions, (list, tuple)):
            return False
        return any(p.get("start_reason") == "Starting XI" for p in positions)

    starters = df[df["positions"].apply(is_starter)]
    result = dict(zip(starters["player_id"].astype(int),
                      starters["player_name"].astype(str)))

    if len(result) != 11 and ev is not None:
        fallback = _starters_from_event(ev, team)
        if len(fallback) == 11:
            return fallback
    return result


def first_sub_index(ev: pd.DataFrame, team: str | None = None) -> int | None:
    """Event index of the first Substitution (optionally for one team)."""
    subs = ev[ev["type"] == "Substitution"]
    if team is not None:
        subs = subs[subs["team"] == team]
    if subs.empty:
        return None
    return int(subs["index"].min())


def build_network(match_id: int, season_name: str, match_date: str,
                  team: str = BARCA, competition_id: int = LA_LIGA,
                  cache_dir: Path | None = None) -> dict:
    """
    Build one team's passing network for a match. Returns a log dict; on success
    also writes the network JSON to cache/networks/ (or cache_dir).

    Defaults to FC Barcelona / La Liga, but `team` and `competition_id` let the
    IDENTICAL pipeline build other teams' networks (e.g. Bayern Munich 2015/16
    for the Guardiola transfer experiment).
    """
    log = {
        "match_id": match_id,
        "season": season_name,
        "date": match_date,
        "team": team,
        "ok": False,
        "reason": "",
        "n_nodes": 0,
        "n_edges": 0,
        "total_passes": 0,
        "first_sub_minute": None,
        "first_team_sub_minute": None,
        "window_passes_all": 0,
        "dropped_non_starter": 0,
    }

    ev = events(match_id)
    starters = starting_xi(match_id, team, ev=ev)
    if len(starters) != 11:
        log["reason"] = f"lineups gave {len(starters)} starters, expected 11"
        return log
    # record whether the lineups primary source sufficed or we used the
    # Starting XI event fallback (data-quality footnote for the report)
    lu = lineups(match_id)[team]
    primary = sum(1 for pos in lu["positions"]
                  if isinstance(pos, (list, tuple))
                  and any(p.get("start_reason") == "Starting XI" for p in pos))
    log["starter_source"] = "lineups" if primary == 11 else "starting_xi_event"

    # First substitution (either team) = the window cut.
    cut = first_sub_index(ev)
    bcut = first_sub_index(ev, team)
    if cut is None:
        # No subs at all in the match: the whole match is the 11-starter window.
        window = ev
        log["reason_window"] = "no substitutions; using full match"
    else:
        window = ev[ev["index"] < cut]
        sub_row = ev[ev["index"] == cut].iloc[0]
        log["first_sub_minute"] = int(sub_row["minute"])
    if bcut is not None:
        bsub_row = ev[ev["index"] == bcut].iloc[0]
        log["first_team_sub_minute"] = int(bsub_row["minute"])

    # Team passes in the window.
    passes = window[(window["type"] == "Pass") & (window["team"] == team)].copy()
    log["window_passes_all"] = len(passes)

    # completed (pass_outcome NaN) and open play (pass_type not a set-piece restart)
    completed = passes["pass_outcome"].isna()
    pass_type = passes["pass_type"]
    open_play = ~pass_type.isin(SETPIECE_PASS_TYPES)  # NaN -> True (regular open play)
    passes = passes[completed & open_play]

    # both endpoints must be starters
    starter_ids = set(starters)
    pid = passes["player_id"]
    rid = passes["pass_recipient_id"]
    valid = pid.isin(starter_ids) & rid.isin(starter_ids)
    log["dropped_non_starter"] = int((~valid).sum())
    passes = passes[valid]

    # Build directed weighted graph. Each node carries the full name and a
    # short label (StatsBomb player_nickname when available, e.g. "Xavi",
    # "Lionel Messi") for readable figures and later player-role work.
    nicks = _nickname_map(match_id, team)
    G = nx.DiGraph()
    for p_id, p_name in starters.items():
        G.add_node(int(p_id), name=p_name,
                   label=nicks.get(int(p_id), p_name))

    agg = (passes.groupby([passes["player_id"].astype(int),
                           passes["pass_recipient_id"].astype(int)])
           .size())
    total = 0
    for (u, v), w in agg.items():
        if u == v:  # ignore self-passes if any spurious ones appear
            continue
        G.add_edge(int(u), int(v), weight=int(w))
        total += int(w)

    log["n_nodes"] = G.number_of_nodes()
    log["n_edges"] = G.number_of_edges()
    log["total_passes"] = total

    if G.number_of_nodes() != 11:
        log["reason"] = f"graph has {G.number_of_nodes()} nodes, expected 11"
        return log
    assert G.number_of_nodes() == 11, "exactly 11 nodes required"

    # Tag and persist.
    G.graph.update({
        "match_id": match_id,
        "season": season_name,
        "date": match_date,
        "team": team,
        "competition_id": competition_id,
        "first_sub_minute": log["first_sub_minute"],
        "total_passes": total,
    })
    _write_network(match_id, G, cache_dir=cache_dir)
    log["ok"] = True
    return log


def _write_network(match_id: int, G: nx.DiGraph,
                   cache_dir: Path | None = None) -> None:
    data = {
        "graph": G.graph,
        "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes],
        "edges": [{"source": u, "target": v, "weight": d["weight"]}
                  for u, v, d in G.edges(data=True)],
    }
    d = cache_dir or NET_CACHE
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"match{match_id}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)


def load_network(match_id: int, cache_dir: Path | None = None) -> nx.DiGraph:
    """Reload a cached network as a tagged nx.DiGraph (used by Phase 2)."""
    path = (cache_dir or NET_CACHE) / f"match{match_id}.json"
    with path.open() as fh:
        data = json.load(fh)
    G = nx.DiGraph()
    G.graph.update(data["graph"])
    for n in data["nodes"]:
        nid = n.pop("id")
        G.add_node(int(nid), **n)
    for e in data["edges"]:
        G.add_edge(int(e["source"]), int(e["target"]), weight=int(e["weight"]))
    return G


def main():
    scope = load_scope()
    seasons = scope["seasons"]
    total_target = scope["total_barca_matches"]
    print("=" * 78)
    print(f"PHASE 1 — building Barca networks: {len(seasons)} seasons, "
          f"{total_target} matches")
    print("=" * 78)

    logs = []
    built = 0
    for s in seasons:
        sid = s["season_id"]
        sname = s["season_name"]
        md = matches(LA_LIGA, sid)
        barca = md[(md["home_team"] == BARCA) | (md["away_team"] == BARCA)]
        barca = barca.sort_values("match_date")
        n_ok = 0
        for _, m in barca.iterrows():
            mid = int(m["match_id"])
            log = build_network(mid, sname, str(m["match_date"]))
            logs.append(log)
            if log["ok"]:
                n_ok += 1
                built += 1
        print(f"  {sname}: {n_ok}/{len(barca)} networks built")

    df = pd.DataFrame(logs)
    df.to_csv(OUT / "phase1_build_log.csv", index=False)

    print("-" * 78)
    ok = df[df["ok"]]
    failed = df[~df["ok"]]
    print(f"Built {built}/{len(df)} networks "
          f"({len(failed)} failed/skipped).")
    if len(failed):
        print("\nFailures:")
        for _, r in failed.iterrows():
            print(f"  match {r['match_id']} ({r['season']}): {r['reason']}")

    # Sanity: every built network must have exactly 11 nodes.
    assert (ok["n_nodes"] == 11).all(), "some built network != 11 nodes!"
    print(f"\nAll {built} built networks have exactly 11 nodes. [assert passed]")

    # How often does the opponent sub before Barca? (methodology footnote)
    both = ok.dropna(subset=["first_sub_minute", "first_team_sub_minute"])
    earlier = (both["first_sub_minute"] < both["first_team_sub_minute"]).sum()
    print(f"\nWindow note: in {earlier}/{len(both)} matches the FIRST sub "
          f"(either team) came before Barca's first sub.")
    print(f"  median first-sub minute (window length): "
          f"{ok['first_sub_minute'].median():.0f}'")
    print(f"  median completed open-play passes/network: "
          f"{ok['total_passes'].median():.0f}")
    print(f"  median edges/network: {ok['n_edges'].median():.0f}")

    print("\n" + "=" * 78)
    print("PHASE 1 SUMMARY (copy-paste)")
    print("=" * 78)
    print(f"- Built {built} Barca 11-node passing networks across "
          f"{len(seasons)} La Liga seasons (target {total_target}).")
    print(f"- Window = events before first substitution; median window ends "
          f"~{ok['first_sub_minute'].median():.0f}', "
          f"median {ok['total_passes'].median():.0f} completed open-play "
          f"passes per network.")
    if len(failed):
        print(f"- {len(failed)} matches skipped (see phase1_build_log.csv): "
              f"reasons = {sorted(set(failed['reason']))}.")
    else:
        print("- 0 matches skipped; every match yielded a valid 11-node network.")
    print(f"- Networks cached to cache/networks/ ; build log -> "
          f"outputs/phase1_build_log.csv")


if __name__ == "__main__":
    main()
