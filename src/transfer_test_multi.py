"""
Multi-coach cross-context transfer test.

Three coaches who were in the Barca training set later coached national teams
present in the StatsBomb free data:

  * Pep Guardiola   -> Bayern Munich 2015/16        (2 networks)
  * Luis Enrique    -> Spain, World Cup 2022        (national team)
  * Tata Martino    -> Mexico, World Cup 2022        (national team)

We train the Problem-1 coach classifiers on the FULL Barca corpus and predict
the coach for each of these out-of-club networks. For each coach we report the
RECOVERY RATE: how often the Barca-trained model recovers the correct coach,
versus labelling the network as some other Barca coach.

This turns the n=2 Bayern test into a 3-coach, 3-context test of whether a
coach's passing-network signature is coach-bound (transfers across clubs and
even to national teams) or club/roster-bound.

Reads: outputs/features.csv (Barca train), outputs/features_bayern.csv,
       outputs/features_worldcup.csv (Spain/Mexico test).
Writes: outputs/transfer_multi.csv, outputs/transfer_multi.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sb_cache import PROJECT_ROOT
from phase3_p1 import (label_coach, feature_columns, make_models, COACH_ORDER,
                       MIN_CLASS)

OUT = PROJECT_ROOT / "outputs"

# (display name, source csv, filter, true coach)
TEST_GROUPS = [
    ("Guardiola @ Bayern 2015/16", "bayern", None, "Guardiola"),
    ("L.Enrique @ Spain WC2022", "wc", ("Spain", "WC2022"), "L.Enrique"),
    ("Martino @ Mexico WC2022", "wc", ("Mexico", "WC2022"), "Martino"),
]


def main():
    barca = pd.read_csv(OUT / "features.csv")
    barca["date"] = pd.to_datetime(barca["date"])
    barca["coach"] = barca.apply(label_coach, axis=1)
    counts = barca["coach"].value_counts()
    keep = counts[counts >= MIN_CLASS].index.tolist()
    barca = barca[barca["coach"].isin(keep)].copy()
    classes = [c for c in COACH_ORDER if c in keep]
    _, _, _, combined = feature_columns(barca)

    bayern = pd.read_csv(OUT / "features_bayern.csv")
    wc = pd.read_csv(OUT / "features_worldcup.csv")

    def subset(src, flt):
        if src == "bayern":
            return bayern
        team, comp = flt
        return wc[(wc["team"] == team) & (wc["competition"] == comp)]

    X_tr = barca[combined].values
    y_tr = barca["coach"].values

    models = make_models()
    for name, model in models.items():
        model.fit(X_tr, y_tr)

    print("=" * 78)
    print("MULTI-COACH CROSS-CONTEXT TRANSFER TEST")
    print("=" * 78)
    print(f"Train: {len(barca)} Barca networks, {len(classes)} coaches.")
    print(f"Models: {list(models.keys())}\n")

    rows = []
    for disp, src, flt, true_coach in TEST_GROUPS:
        sub = subset(src, flt)
        n = len(sub)
        if n == 0:
            print(f"[{disp}] NO networks found — skipping")
            continue
        X_te = sub[combined].values
        print("-" * 78)
        print(f"[{disp}]  true coach = {true_coach},  n = {n} networks")
        for name, model in models.items():
            preds = model.predict(X_te)
            recovered = int((preds == true_coach).sum())
            rate = recovered / n
            vc = pd.Series(preds).value_counts()
            breakdown = ", ".join(f"{k}:{v}" for k, v in vc.items())
            print(f"    {name:<20} recovery {recovered}/{n} = {rate:.2f}   "
                  f"[{breakdown}]")
            rows.append({"group": disp, "true_coach": true_coach, "n": n,
                         "model": name, "recovered": recovered,
                         "recovery_rate": rate, "breakdown": breakdown})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "transfer_multi.csv", index=False)

    # aggregate recovery per coach (mean over models)
    print("\n" + "=" * 78)
    print("RECOVERY RATE PER COACH (mean across models)")
    print("=" * 78)
    agg = (res.groupby(["group", "true_coach", "n"])["recovery_rate"]
           .mean().reset_index())
    for _, r in agg.iterrows():
        print(f"  {r['group']:<32} {r['recovery_rate']:.2f}  (n={int(r['n'])})")

    _plot(res)

    # overall conclusion
    overall = res["recovery_rate"].mean()
    chance = 1 / len(classes)
    print("\n" + "=" * 78)
    print("TRANSFER SUMMARY (copy-paste)")
    print("=" * 78)
    for _, r in agg.iterrows():
        print(f"- {r['true_coach']} ({r['group'].split('@')[1].strip()}): "
              f"mean recovery {r['recovery_rate']:.2f} across "
              f"{res['model'].nunique()} models (n={int(r['n'])}).")
    print(f"- Overall mean recovery {overall:.2f} vs uniform chance "
          f"{chance:.2f} ({len(classes)} classes).")
    print(f"- Total cross-context test networks: {res.groupby('group')['n'].first().sum()}")


def _plot(res):
    groups = res["group"].unique()
    models = res["model"].unique()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(groups))
    w = 0.8 / len(models)
    for i, mdl in enumerate(models):
        sub = res[res["model"] == mdl].set_index("group").reindex(groups)
        ax.bar(x + i * w, sub["recovery_rate"].values, w, label=mdl)
    ax.axhline(1 / 8, ls="--", color="grey", lw=1, label="uniform chance (1/8)")
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels([g.replace(" @ ", "\n@ ") for g in groups], fontsize=9)
    ax.set_ylabel("recovery rate (correct coach)")
    ax.set_ylim(0, 1)
    ax.set_title("Cross-context coach-signature recovery\n"
                 "Barca-trained models on the same coaches at other clubs / "
                 "national teams")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "transfer_multi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
