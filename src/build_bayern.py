"""
Build Bayern Munich Bundesliga 2015/16 passing networks (Guardiola's final
season at Bayern) using the IDENTICAL pipeline as the Barca networks, then
extract the same feature embedding.

  - team = "Bayern Munich", competition_id = 9, season_id = 27
  - 11-starter pre-first-substitution window, completed open-play passes,
    exactly 11 nodes (asserted) — same build_network() code path as Barca.
  - same features.extract() -> feature columns identical to features.csv,
    so the Barca-trained P1 model can score these networks directly.
  - coach label = "Guardiola" (his last season at Bayern).

Outputs:
  cache/networks_bayern/match*.json
  outputs/features_bayern.csv   (same columns as features.csv + coach)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sb_cache import matches, PROJECT_ROOT
from build_networks import build_network, load_network
from features import extract

TEAM = "Bayern Munich"
COMP = 9
SEASON_ID = 27
SEASON_NAME = "2015/2016"
COACH = "Guardiola"

BAYERN_NET_CACHE = PROJECT_ROOT / "cache" / "networks_bayern"
BAYERN_NET_CACHE.mkdir(parents=True, exist_ok=True)
OUT = PROJECT_ROOT / "outputs"


def main():
    """Build Bayern's 2015/16 networks and write features_bayern.csv."""
    print("=" * 78)
    print(f"BUILD — {TEAM} Bundesliga {SEASON_NAME} (Guardiola)")
    print("=" * 78)

    md = matches(COMP, SEASON_ID)
    games = md[(md["home_team"] == TEAM) | (md["away_team"] == TEAM)]
    games = games.sort_values("match_date")
    print(f"{TEAM} matches in {SEASON_NAME}: {len(games)}")

    logs, feats = [], []
    for _, m in games.iterrows():
        mid = int(m["match_id"])
        log = build_network(mid, SEASON_NAME, str(m["match_date"]),
                            team=TEAM, competition_id=COMP,
                            cache_dir=BAYERN_NET_CACHE)
        logs.append(log)
        if log["ok"]:
            G = load_network(mid, cache_dir=BAYERN_NET_CACHE)
            row = extract(G)
            row["coach"] = COACH
            row["team"] = TEAM
            feats.append(row)

    ldf = pd.DataFrame(logs)
    built = int(ldf["ok"].sum())
    print(f"Built {built}/{len(ldf)} Bayern networks "
          f"({len(ldf) - built} failed/skipped).")
    if built != len(ldf):
        for _, r in ldf[~ldf["ok"]].iterrows():
            print(f"  skipped match {r['match_id']}: {r['reason']}")

    fdf = pd.DataFrame(feats)
    # all built networks must have exactly 11 nodes
    assert (fdf["n_nodes"] == 11).all(), "some Bayern network != 11 nodes"
    fdf = fdf.sort_values("date").reset_index(drop=True)
    fdf.to_csv(OUT / "features_bayern.csv", index=False)

    print(f"\nAll {built} Bayern networks have exactly 11 nodes. [assert passed]")
    print(f"median passes/net: {fdf['total_passes'].median():.0f}  "
          f"median edges: {fdf['n_edges'].median():.0f}  "
          f"median density: {fdf['density'].median():.3f}")
    print(f"Wrote outputs/features_bayern.csv  ({fdf.shape[0]} x {fdf.shape[1]})")

    # quick comparison to Barca Guardiola era
    print("\n(For reference — Barca density median is ~0.74 in the Pep peak;")
    print(f" Bayern 2015/16 density median = {fdf['density'].median():.3f}.)")


if __name__ == "__main__":
    main()
