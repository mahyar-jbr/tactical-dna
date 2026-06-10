"""
Merge all team-match feature matrices into one corpus for Problem 2.

Sources (all share the same 50 structural feature columns from features.extract):
  * outputs/features.csv          FC Barcelona, La Liga 2005/06-2020/21 (coach labelled)
  * outputs/features_bayern.csv   Bayern Munich 2015/16 (Guardiola)
  * outputs/features_worldcup.csv World Cup 2018 + 2022, both teams per match

Adds uniform metadata columns: team, competition, coach (coach known only for
the Barca/Bayern rows and the Spain/Mexico WC2022 rows; blank otherwise).

Writes outputs/features_all.csv.
"""
from __future__ import annotations

import pandas as pd

from sb_cache import PROJECT_ROOT
from phase3_p1 import label_coach

OUT = PROJECT_ROOT / "outputs"

# National-team coaches we can label (subset; used for transfer interpretation)
WC_COACH = {
    ("Spain", "WC2022"): "L.Enrique",
    ("Mexico", "WC2022"): "Martino",
}

FEATURE_META = ["match_id", "season", "date", "team", "competition", "coach"]


def main():
    barca = pd.read_csv(OUT / "features.csv")
    barca["date"] = pd.to_datetime(barca["date"])
    barca["coach"] = barca.apply(label_coach, axis=1)
    barca["team"] = "Barcelona"
    barca["competition"] = "LaLiga"

    bayern = pd.read_csv(OUT / "features_bayern.csv")
    bayern["competition"] = "Bundesliga"
    bayern["coach"] = "Guardiola"

    wc = pd.read_csv(OUT / "features_worldcup.csv")
    wc["coach"] = [WC_COACH.get((t, c), "") for t, c in
                   zip(wc["team"], wc["competition"])]
    if "season" not in wc.columns:
        wc["season"] = wc["competition"]

    frames = []
    for df, src in [(barca, "barca"), (bayern, "bayern"), (wc, "wc")]:
        d = df.copy()
        d["source"] = src
        frames.append(d)
    alldf = pd.concat(frames, ignore_index=True, sort=False)

    # ensure metadata cols exist
    for c in FEATURE_META + ["source"]:
        if c not in alldf.columns:
            alldf[c] = ""
    alldf["coach"] = alldf["coach"].fillna("")

    alldf.to_csv(OUT / "features_all.csv", index=False)
    print(f"Merged corpus: {len(alldf)} networks")
    print(alldf.groupby("source").size().to_string())
    print(f"competitions: {dict(alldf['competition'].value_counts())}")
    print(f"distinct teams: {alldf['team'].nunique()}")
    print(f"labelled-coach rows: {(alldf['coach'] != '').sum()}")
    print(f"Wrote outputs/features_all.csv ({alldf.shape[0]} x {alldf.shape[1]})")


if __name__ == "__main__":
    main()
