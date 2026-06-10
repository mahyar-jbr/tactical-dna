"""
Phase 3 — Problem 1 (Coach Identification), first cut.

Builds a date-aware Barca coach label for every network, then trains two
class-weighted classifiers (logistic regression, random forest) with
stratified 5-fold cross-validation. Reports:

  * mean +/- std accuracy and macro-F1 for each model,
  * a majority-class baseline accuracy,
  * a confusion matrix (out-of-fold predictions) -> confusion_matrix.png,
  * an ablation: accuracy for three feature sets
        centrality-only | motifs-only | combined.

Coach mapping (verified FC Barcelona head-coach tenures; SYNC 2 decisions:
keep all 8 coaches, include Rijkaard):
    2005/06-2007/08  Frank Rijkaard
    2008/09-2011/12  Pep Guardiola
    2012/13          Tito Vilanova
    2013/14          Tata Martino
    2014/15-2016/17  Luis Enrique
    2017/18, 2018/19 Ernesto Valverde
    2019/20          Valverde (before 2020-01-13) / Quique Setien (on/after)
    2020/21          Ronald Koeman

The 2019/20 season is split by match DATE at Valverde's sacking
(2020-01-13): a per-season label would mis-assign 19 Setien matches, so we
label per match instead -> a cleaner, more correct ground truth.
"""
from __future__ import annotations

import os
# Keep numeric thread pools small and predictable before numpy/sklearn/xgboost
# import. Prevents OpenMP/BLAS oversubscription deadlocks on macOS when many
# CV fits run in sequence (XGBoost + RandomForest each spawning thread pools).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sb_cache import PROJECT_ROOT

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception as _xgb_err:  # noqa: BLE001
    _HAS_XGB = False
    import warnings
    warnings.warn(
        f"XGBoost unavailable, will be skipped: {_xgb_err}. "
        "On macOS this usually means libomp is missing (`brew install libomp`).")

OUT = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42
N_FOLDS = 5

VALVERDE_OUT = pd.Timestamp("2020-01-13")  # Valverde sacked; Setien in

SEASON_COACH = {
    "2005/2006": "Rijkaard", "2006/2007": "Rijkaard", "2007/2008": "Rijkaard",
    "2008/2009": "Guardiola", "2009/2010": "Guardiola",
    "2010/2011": "Guardiola", "2011/2012": "Guardiola",
    "2012/2013": "Vilanova", "2013/2014": "Martino",
    "2014/2015": "L.Enrique", "2015/2016": "L.Enrique", "2016/2017": "L.Enrique",
    "2017/2018": "Valverde", "2018/2019": "Valverde",
    "2020/2021": "Koeman",
    # 2019/2020 handled by date below
}

# chronological order for nice confusion-matrix axes
COACH_ORDER = ["Rijkaard", "Guardiola", "Vilanova", "Martino",
               "L.Enrique", "Valverde", "Setien", "Koeman"]

MIN_CLASS = 5  # need >= N_FOLDS for stratified CV


def label_coach(row) -> str:
    """
    Map one network's (season, date) to its FC Barcelona head coach.

    Most seasons map directly via SEASON_COACH. The 2019/20 season is split by
    match DATE at Valverde's dismissal (2020-01-13) into Valverde / Setien, a
    more accurate ground truth than a single season-level label.
    """
    s, d = row["season"], row["date"]
    if s in SEASON_COACH:
        return SEASON_COACH[s]
    if s == "2019/2020":
        return "Valverde" if d < VALVERDE_OUT else "Setien"
    return "UNKNOWN"


def feature_columns(df: pd.DataFrame):
    """Return (centrality_cols, motif_cols, global_cols, combined_cols)."""
    cent = [c for c in df.columns
            if any(c.startswith(p) for p in
                   ("windeg_", "woutdeg_", "betw_", "eig_", "pagerank_"))]
    motif = [c for c in df.columns if c.startswith("triad_")]
    glob = ["n_edges", "density", "total_passes", "weighted_clustering",
            "aspl_weighted", "aspl_unweighted", "algebraic_connectivity",
            "spectral_radius"]
    glob = [c for c in glob if c in df.columns]
    combined = cent + motif + glob
    return cent, motif, glob, combined


class XGBStringWrapper:
    """
    Thin wrapper so XGBClassifier fits the same string-label interface as the
    other models. XGBoost requires y encoded 0..K-1; we label-encode on fit and
    decode on predict. Class imbalance is handled with per-sample weights
    (inverse class frequency), the gradient-boosting analogue of
    class_weight='balanced'.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._le = None
        self._clf = None
        self._scaler = None

    def fit(self, X, y):
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._le = LabelEncoder().fit(y)
        yi = self._le.transform(y)
        # inverse-frequency sample weights (balanced)
        counts = np.bincount(yi)
        w = (len(yi) / (len(counts) * counts))[yi]
        self._clf = XGBClassifier(**self.kwargs)
        self._clf.fit(Xs, yi, sample_weight=w)
        return self

    def predict(self, X):
        Xs = self._scaler.transform(X)
        return self._le.inverse_transform(self._clf.predict(Xs))


def make_models(include_xgb: bool = True):
    """Class-weighted LR, RF (and XGBoost if available), each behind a scaling
    pipeline (scaling is a no-op for the tree models but keeps the interface
    uniform)."""
    lr = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
    rf = Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=500,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    models = {"Logistic Regression": lr, "Random Forest": rf}
    if include_xgb and _HAS_XGB:
        # n_jobs=1: single-threaded XGBoost. On macOS, XGBoost's OpenMP pool
        # combined with already-initialised numpy/BLAS threads can deadlock
        # under repeated CV fits; n_jobs=1 is fast enough for 517 samples and
        # avoids the fork/thread contention entirely.
        models["XGBoost"] = XGBStringWrapper(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, n_jobs=1, nthread=1,
            tree_method="hist", eval_metric="mlogloss")
    return models


def cv_scores(model, X, y, cv, groups=None):
    """Per-fold accuracy and macro-F1. groups -> grouped split (no leakage)."""
    accs, f1s = [], []
    splitter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for tr, te in splitter:
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return np.array(accs), np.array(f1s)


def main():
    df = pd.read_csv(OUT / "features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["coach"] = df.apply(label_coach, axis=1)

    print("=" * 72)
    print("PHASE 3 — Problem 1: Coach Identification (first cut)")
    print("=" * 72)

    # drop coaches with too few matches (none expected; min class = 19)
    counts = df["coach"].value_counts()
    keep = counts[counts >= MIN_CLASS].index.tolist()
    dropped = counts[counts < MIN_CLASS]
    df = df[df["coach"].isin(keep)].copy()
    classes = [c for c in COACH_ORDER if c in keep]

    print("\nClass distribution (per-match coach labels):")
    for c in classes:
        print(f"  {c:<11} {int(counts[c]):>4}")
    print(f"  {'TOTAL':<11} {len(df):>4}  ({len(classes)} coaches)")
    if len(dropped):
        print(f"  dropped (<{MIN_CLASS}): {dict(dropped)}")

    y = df["coach"].values
    cent, motif, glob, combined = feature_columns(df)
    print(f"\nFeature sets: centrality={len(cent)}, motifs={len(motif)}, "
          f"global={len(glob)}, combined={len(combined)}")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    models = make_models()

    # ---- majority-class baseline -----------------------------------------
    Xc = df[combined].values
    dummy = DummyClassifier(strategy="most_frequent")
    base_acc, base_f1 = cv_scores(dummy, Xc, y, cv)
    maj = counts[classes].max() / counts[classes].sum()
    print("\n" + "-" * 72)
    print("BASELINE")
    print("-" * 72)
    print(f"  Majority-class baseline accuracy: {maj:.3f} "
          f"(predict '{counts[classes].idxmax()}'); "
          f"CV check {base_acc.mean():.3f}, macro-F1 {base_f1.mean():.3f}")

    # ---- main models on combined features --------------------------------
    print("\n" + "-" * 72)
    print(f"MAIN MODELS (combined features, stratified {N_FOLDS}-fold CV, "
          f"class-weighted)")
    print("-" * 72)
    results = {}
    for name, model in models.items():
        accs, f1s = cv_scores(model, Xc, y, cv)
        results[name] = (accs, f1s)
        print(f"  {name:<22} accuracy {accs.mean():.3f} +/- {accs.std():.3f}   "
              f"macro-F1 {f1s.mean():.3f} +/- {f1s.std():.3f}")

    # ---- rigorous companion: SEASON-GROUPED CV (no same-season leakage) ---
    # Because each coach occupies contiguous seasons and rosters change yearly,
    # random folds let same-season matches leak across train/test, partly
    # measuring "which season" instead of "which coach". Season-grouped CV
    # keeps every season intact in one fold -> a leakage-free lower bound.
    print("\n" + "-" * 72)
    print(f"SEASON-GROUPED CV (combined features; no same-season leakage)")
    print("-" * 72)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    groups = df["season"].values
    grouped = {}
    for name, model in make_models().items():
        accs, f1s = cv_scores(model, Xc, y, sgkf, groups=groups)
        grouped[name] = (accs, f1s)
        print(f"  {name:<22} accuracy {accs.mean():.3f} +/- {accs.std():.3f}   "
              f"macro-F1 {f1s.mean():.3f} +/- {f1s.std():.3f}")
    print("  (lower than random-fold CV: the gap = same-season memorisation.)")

    # ---- ablation: centrality / motifs / combined ------------------------
    print("\n" + "-" * 72)
    print("ABLATION — accuracy by feature set (mean +/- std over folds)")
    print("-" * 72)
    sets = {"centrality-only": cent, "motifs-only": motif,
            "combined": combined}
    print(f"  {'feature set':<18}" +
          "".join(f"{n:>26}" for n in models))
    abl = {}
    for sname, cols in sets.items():
        Xs = df[cols].values
        row = {}
        cells = ""
        for mname, model in make_models().items():
            accs, _ = cv_scores(model, Xs, y, cv)
            row[mname] = (accs.mean(), accs.std())
            cells += f"{accs.mean():>15.3f} +/- {accs.std():<6.3f}"
        abl[sname] = row
        print(f"  {sname:<18}{cells}")

    # ---- confusion matrix (out-of-fold preds, best model) ----------------
    # Manual OOF loop so it works for any model (incl. the XGB wrapper, which
    # is not a sklearn estimator and can't go through cross_val_predict).
    best_name = max(results, key=lambda k: results[k][0].mean())
    best_model = make_models()[best_name]
    oof = np.empty_like(y)
    for tr, te in cv.split(Xc, y):
        best_model.fit(Xc[tr], y[tr])
        oof[te] = best_model.predict(Xc[te])
    cm = confusion_matrix(y, oof, labels=classes)
    _plot_confusion(cm, classes, best_name)

    # per-class F1 from OOF for the best model
    from sklearn.metrics import classification_report
    print("\n" + "-" * 72)
    print(f"PER-CLASS REPORT (out-of-fold, {best_name})")
    print("-" * 72)
    rep = classification_report(y, oof, labels=classes, digits=3,
                                zero_division=0)
    print(rep)

    # ---- save results CSV -------------------------------------------------
    rows = []
    for name, (accs, f1s) in results.items():
        rows.append({"model": name, "feature_set": "combined",
                     "cv": "stratified-random",
                     "acc_mean": accs.mean(), "acc_std": accs.std(),
                     "f1_mean": f1s.mean(), "f1_std": f1s.std()})
    for name, (accs, f1s) in grouped.items():
        rows.append({"model": name, "feature_set": "combined",
                     "cv": "season-grouped",
                     "acc_mean": accs.mean(), "acc_std": accs.std(),
                     "f1_mean": f1s.mean(), "f1_std": f1s.std()})
    for sname, row in abl.items():
        for mname, (am, asd) in row.items():
            rows.append({"model": mname, "feature_set": sname,
                         "acc_mean": am, "acc_std": asd,
                         "f1_mean": np.nan, "f1_std": np.nan})
    rows.append({"model": "Majority baseline", "feature_set": "n/a",
                 "acc_mean": maj, "acc_std": 0.0,
                 "f1_mean": base_f1.mean(), "f1_std": base_f1.std()})
    pd.DataFrame(rows).to_csv(OUT / "phase3_p1_results.csv", index=False)

    # ---- copy-paste summary ----------------------------------------------
    print("\n" + "=" * 72)
    print("PHASE 3 SUMMARY (copy-paste)")
    print("=" * 72)
    print(f"- P1 coach ID over {len(df)} Barca networks, {len(classes)} "
          f"coaches (2005/06-2020/21).")
    print(f"- Majority-class baseline accuracy: {maj:.3f}.")
    for name, (accs, f1s) in results.items():
        print(f"- {name} (random CV): accuracy {accs.mean():.3f}+/-"
              f"{accs.std():.3f}, macro-F1 {f1s.mean():.3f}+/-{f1s.std():.3f} "
              f"(+{accs.mean()-maj:.3f} over baseline).")
    for name, (accs, f1s) in grouped.items():
        print(f"- {name} (season-grouped, leakage-free): accuracy "
              f"{accs.mean():.3f}+/-{accs.std():.3f}, macro-F1 "
              f"{f1s.mean():.3f}+/-{f1s.std():.3f}.")
    print("- Ablation (best model, accuracy): " +
          ", ".join(f"{s}={abl[s][best_name][0]:.3f}" for s in sets))
    print(f"- Confusion matrix ({best_name}) -> confusion_matrix.png ; "
          f"results -> phase3_p1_results.csv")


def _plot_confusion(cm, classes, model_name):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted coach")
    ax.set_ylabel("True coach")
    ax.set_title(f"Coach-identification confusion matrix ({model_name})\n"
                 f"5-fold out-of-fold predictions; cell = row-normalised rate "
                 f"(count)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            txt = f"{cm_norm[i, j]:.2f}\n({cm[i, j]})"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="row-normalised rate")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
