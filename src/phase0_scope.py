"""
Phase 0 — setup & scope.

1. Fetch StatsBomb free competitions.json and print every available
   competition (id, name, season).
2. Report which of the proposal's targets exist in the FREE data:
     - La Liga (competition_id == 11) seasons
     - World Cup 2018 & 2022 (competition_id == 43)
     - any Bayern Munich Bundesliga season
3. Lock scope to FC Barcelona's own passing networks across ALL available
   La Liga seasons, and print the chosen season list (with Barca match
   counts per season).

Writes outputs/phase0_scope.json so later phases consume the locked scope
instead of re-deciding it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sb_cache import competitions, matches, PROJECT_ROOT

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)

LA_LIGA = 11
WORLD_CUP = 43
BARCA = "Barcelona"

# Decision (SYNC 1): drop seasons with too few Barca matches to give reliable
# per-season summary stats or non-degenerate P1 classes. This removes the
# 1973/74 (1 match) and 2004/05 (7 match) outliers.
MIN_MATCHES_PER_SEASON = 10

# Bayern Munich's Bundesliga 2015/16 is FREE (competition_id=9, season_id=27).
# Tonight stays Barca-only, but we record this so the Guardiola Barca->Bayern
# transfer experiment (proposal headline) can be added later without re-scoping.
BAYERN_BUNDESLIGA = {"competition_id": 9, "season_id": 27,
                     "season_name": "2015/2016", "team": "Bayern Munich"}


def line(char="=", n=78):
    print(char * n)


def main():
    line()
    print("PHASE 0 — STATSBOMB FREE COMPETITIONS")
    line()

    comp = competitions()
    # Canonical columns in statsbombpy competitions():
    # competition_id, season_id, country_name, competition_name, competition_gender,
    # season_name, ... (plus update timestamps)
    cols = ["competition_id", "season_id", "country_name",
            "competition_name", "competition_gender", "season_name"]
    cols = [c for c in cols if c in comp.columns]
    view = comp[cols].sort_values(
        ["competition_name", "season_name"]).reset_index(drop=True)

    print(f"\nTotal free competition-seasons: {len(view)}\n")
    print(view.to_string(index=False))

    # ---- Targeted availability check ---------------------------------------
    line()
    print("TARGETED AVAILABILITY (vs. proposal scope)")
    line()

    # La Liga (comp 11)
    laliga = comp[comp["competition_id"] == LA_LIGA].sort_values("season_name")
    print(f"\n[La Liga | competition_id={LA_LIGA}] "
          f"{len(laliga)} free season(s):")
    for _, r in laliga.iterrows():
        print(f"    season_id={r['season_id']:>4}  {r['season_name']}")

    # World Cup 2018 & 2022 (comp 43)
    wc = comp[comp["competition_id"] == WORLD_CUP].sort_values("season_name")
    print(f"\n[FIFA World Cup | competition_id={WORLD_CUP}] "
          f"{len(wc)} free season(s):")
    for _, r in wc.iterrows():
        flag = ""
        if str(r["season_name"]) in ("2018", "2022"):
            flag = "   <-- proposal target"
        print(f"    season_id={r['season_id']:>4}  {r['season_name']}{flag}")
    for target in ("2018", "2022"):
        present = (wc["season_name"].astype(str) == target).any()
        print(f"    World Cup {target}: "
              f"{'AVAILABLE' if present else 'NOT in free data'}")

    # Bundesliga / Bayern Munich
    bund = comp[comp["competition_name"].str.contains(
        "Bundesliga", case=False, na=False)]
    print(f"\n[Bundesliga] {len(bund)} free season(s):")
    if len(bund):
        for _, r in bund.iterrows():
            print(f"    competition_id={r['competition_id']:>4} "
                  f"season_id={r['season_id']:>4}  {r['season_name']}")
    else:
        print("    none in free data")

    # Does ANY free competition contain a Bayern Munich team? Scan matches of
    # every Bundesliga season (cheap; usually 0-1 seasons).
    bayern_found = []
    for _, r in bund.iterrows():
        try:
            md = matches(int(r["competition_id"]), int(r["season_id"]))
        except Exception as e:  # noqa: BLE001
            print(f"    (could not load matches for season "
                  f"{r['season_name']}: {e})")
            continue
        teams = set(md.get("home_team", pd.Series(dtype=str))) | \
            set(md.get("away_team", pd.Series(dtype=str)))
        bayern = [t for t in teams if "Bayern" in str(t)]
        if bayern:
            bayern_found.append((r["season_name"], bayern))
    print("\n[Bayern Munich check]")
    if bayern_found:
        for sn, teams in bayern_found:
            print(f"    Bundesliga {sn}: {teams}")
    else:
        print("    No Bayern Munich season in the FREE StatsBomb data.")
        print("    => Guardiola Barca->Bayern transfer experiment is NOT")
        print("       supported by free data (final-report caveat).")

    # ---- Lock scope: Barca across all La Liga seasons ----------------------
    line()
    print("SCOPE LOCK — FC BARCELONA, ALL AVAILABLE LA LIGA SEASONS")
    line()

    all_seasons = []
    for _, r in laliga.iterrows():
        sid = int(r["season_id"])
        sname = str(r["season_name"])
        md = matches(LA_LIGA, sid)
        barca = md[(md["home_team"] == BARCA) | (md["away_team"] == BARCA)]
        all_seasons.append({
            "competition_id": LA_LIGA,
            "season_id": sid,
            "season_name": sname,
            "barca_matches": len(barca),
        })
    all_seasons.sort(key=lambda d: d["season_name"])

    # Apply the min-match filter (SYNC 1 decision).
    chosen = [c for c in all_seasons if c["barca_matches"] >= MIN_MATCHES_PER_SEASON]
    dropped = [c for c in all_seasons if c["barca_matches"] < MIN_MATCHES_PER_SEASON]
    grand_total = sum(c["barca_matches"] for c in chosen)

    print(f"\n{'season':<12}{'season_id':<12}{'Barca matches':<16}{'status':<10}")
    print("-" * 48)
    for c in all_seasons:
        status = "DROPPED" if c["barca_matches"] < MIN_MATCHES_PER_SEASON else "kept"
        print(f"{c['season_name']:<12}{c['season_id']:<12}"
              f"{c['barca_matches']:<16}{status:<10}")
    print("-" * 48)
    print(f"(filter: >= {MIN_MATCHES_PER_SEASON} Barca matches/season)")
    if dropped:
        print(f"DROPPED: {[(c['season_name'], c['barca_matches']) for c in dropped]}")
    print(f"\n{'KEPT TOTAL':<24}{grand_total:<14}")
    print(f"Seasons in scope: {[c['season_name'] for c in chosen]}")
    print(f"Total Barca team-match passing networks to build: {grand_total}")

    # ---- Persist locked scope ---------------------------------------------
    out = {
        "team": BARCA,
        "competition_id": LA_LIGA,
        "competition_name": "La Liga",
        "min_matches_per_season": MIN_MATCHES_PER_SEASON,
        "seasons": chosen,                 # kept seasons only
        "dropped_seasons": dropped,        # below the min-match threshold
        "all_la_liga_seasons": all_seasons,
        "total_barca_matches": grand_total,
        "availability": {
            "la_liga_seasons": laliga["season_name"].astype(str).tolist(),
            "world_cup_2018": bool((wc["season_name"].astype(str) == "2018").any()),
            "world_cup_2022": bool((wc["season_name"].astype(str) == "2022").any()),
            "bayern_bundesliga_2015_16": True,  # comp 9 season 27, verified above
        },
        # Recorded for the later Guardiola Barca->Bayern transfer experiment.
        "bayern_transfer_target": BAYERN_BUNDESLIGA,
    }
    out_path = PROJECT_ROOT / "outputs" / "phase0_scope.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nLocked scope written to {out_path.relative_to(PROJECT_ROOT)}")

    line()
    print("PHASE 0 SUMMARY (copy-paste)")
    line()
    print(f"- Free La Liga seasons: {laliga['season_name'].astype(str).tolist()}")
    print(f"- World Cup 2018 free: {out['availability']['world_cup_2018']}; "
          f"2022 free: {out['availability']['world_cup_2022']}")
    print("- Bayern Munich Bundesliga 2015/16 free: True "
          "(comp 9, season 27) -> Guardiola transfer experiment feasible.")
    if dropped:
        print(f"- Dropped thin seasons (<{MIN_MATCHES_PER_SEASON} matches): "
              f"{[c['season_name'] for c in dropped]}")
    print(f"- Locked scope: FC Barcelona, La Liga, "
          f"{len(chosen)} season(s), {grand_total} matches.")


if __name__ == "__main__":
    main()
