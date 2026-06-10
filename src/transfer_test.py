"""
Marquee experiment — Guardiola Barca -> Bayern coach-signature transfer.

Train the Problem-1 coach classifiers on the FULL Barca corpus (8 coaches,
2005/06-2020/21), then predict on Guardiola's Bayern Munich 2015/16 networks
(a different club, league, and squad). Report, per model, the fraction of
Bayern networks classified as 'Guardiola' vs each other Barca coach.

Interpretation:
  HIGH Guardiola-rate  -> the coach signature is COACH-BOUND: it transfers
                          across clubs/leagues/rosters (supports the proposal's
                          central hypothesis).
  LOW  Guardiola-rate  -> the signal is largely CLUB/ROSTER-BOUND, consistent
                          with the season-grouped-CV leakage finding.

Both outcomes are reported as a real conclusion. We also report a chance
reference: the class prior P(Guardiola) in the Barca training set, and the
1/K uniform rate, so "high" and "low" are judged against a baseline rather
than absolutely.

Reads outputs/features.csv (Barca) and outputs/features_bayern.csv (Bayern).
Writes outputs/transfer_test.csv and outputs/transfer_test.png.
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


def main():
    # --- Barca training corpus (same labelling as Phase 3) ---
    barca = pd.read_csv(OUT / "features.csv")
    barca["date"] = pd.to_datetime(barca["date"])
    barca["coach"] = barca.apply(label_coach, axis=1)
    counts = barca["coach"].value_counts()
    keep = counts[counts >= MIN_CLASS].index.tolist()
    barca = barca[barca["coach"].isin(keep)].copy()
    classes = [c for c in COACH_ORDER if c in keep]

    # --- Bayern test set (Guardiola) ---
    bayern = pd.read_csv(OUT / "features_bayern.csv")

    _, _, _, combined = feature_columns(barca)
    # ensure Bayern has the same feature columns
    missing = [c for c in combined if c not in bayern.columns]
    assert not missing, f"Bayern features missing columns: {missing}"

    X_tr = barca[combined].values
    y_tr = barca["coach"].values
    X_te = bayern[combined].values
    n_te = len(bayern)

    print("=" * 74)
    print("TRANSFER TEST — Guardiola Barca -> Bayern 2015/16")
    print("=" * 74)
    print(f"Train: {len(barca)} Barca networks, {len(classes)} coaches.")
    print(f"Test : {n_te} Bayern-Guardiola networks (unseen club/league).")
    prior_guardiola = (y_tr == "Guardiola").mean()
    print(f"\nChance references for 'classified as Guardiola':")
    print(f"  uniform 1/K           = {1/len(classes):.3f}")
    print(f"  Barca class prior P(G) = {prior_guardiola:.3f} "
          f"(Guardiola is the largest Barca class)")

    rows = []
    print("\n" + "-" * 74)
    print(f"{'model':<22}{'Guardiola rate':>16}{'most common other':>22}")
    print("-" * 74)
    per_model_preds = {}
    for name, model in make_models().items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        per_model_preds[name] = preds
        vc = pd.Series(preds).value_counts()
        g_rate = (preds == "Guardiola").mean()
        others = vc.drop("Guardiola", errors="ignore")
        top_other = (f"{others.index[0]} ({others.iloc[0]})"
                     if len(others) else "-")
        print(f"{name:<22}{g_rate:>15.3f} {top_other:>22}")
        # full breakdown row
        row = {"model": name, "n_bayern": n_te,
               "guardiola_rate": g_rate,
               "guardiola_count": int((preds == "Guardiola").sum())}
        for c in classes:
            row[f"pred_{c}"] = int((preds == c).sum())
        rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "transfer_test.csv", index=False)

    # --- full per-model breakdown ---
    print("\n" + "-" * 74)
    print("FULL PREDICTION BREAKDOWN (count of Bayern networks per predicted coach)")
    print("-" * 74)
    hdr = f"{'model':<22}" + "".join(f"{c[:9]:>11}" for c in classes)
    print(hdr)
    for _, r in res.iterrows():
        cells = "".join(f"{int(r[f'pred_{c}']):>11}" for c in classes)
        print(f"{r['model']:<22}{cells}")

    _plot(res, classes, n_te, prior_guardiola)

    # --- conclusion ---
    mean_g = res["guardiola_rate"].mean()
    print("\n" + "=" * 74)
    print("TRANSFER-TEST SUMMARY (copy-paste)")
    print("=" * 74)
    for _, r in res.iterrows():
        print(f"- {r['model']}: {int(r['guardiola_count'])}/{n_te} Bayern "
              f"networks classified as Guardiola "
              f"({r['guardiola_rate']:.3f}).")
    print(f"- Mean Guardiola-rate across models: {mean_g:.3f} "
          f"(vs uniform {1/len(classes):.3f}, Barca prior "
          f"{prior_guardiola:.3f}).")
    if mean_g > max(prior_guardiola, 0.5):
        verdict = ("ABOVE both chance baselines -> evidence the Guardiola "
                   "signature is COACH-BOUND and transfers across clubs.")
    elif mean_g > 1/len(classes):
        verdict = ("above uniform chance but not dominant -> PARTIAL transfer; "
                   "the signature is detectable across clubs but diluted, "
                   "consistent with the season-grouped leakage finding.")
    else:
        verdict = ("at/below chance -> the signal is largely CLUB/ROSTER-BOUND, "
                   "not a transferable coach fingerprint.")
    print(f"- Conclusion: {verdict}")


def _plot(res, classes, n_te, prior):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    models = res["model"].tolist()
    x = np.arange(len(classes))
    width = 0.8 / len(models)
    for i, (_, r) in enumerate(res.iterrows()):
        vals = [r[f"pred_{c}"] / n_te for c in classes]
        ax.bar(x + i * width, vals, width, label=r["model"])
    ax.axhline(prior, ls="--", color="grey", lw=1,
               label=f"Barca prior P(Guardiola)={prior:.2f}")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("fraction of Bayern networks predicted")
    ax.set_title("Guardiola Barca->Bayern transfer test\n"
                 f"How {n_te} Bayern-Guardiola networks are classified by the "
                 f"Barca-trained models")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "transfer_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
