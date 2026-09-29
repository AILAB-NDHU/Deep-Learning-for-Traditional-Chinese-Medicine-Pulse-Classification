"""
TCM Pulse — SMOTE Variants Comparison  (3-class: 弦脈 / 滑脈 / 緊脈)
======================================================================
Compares four oversampling strategies on the ORIGINAL 9-Gaussian-parameter
features (no waveform augmentation) using XGBoost + Subject-level CV.
Choppy (澀脈) is excluded.  Tight (緊脈) is the minority class.

Strategies
----------
  baseline        : no oversampling
  SMOTE           : traditional SMOTE (imblearn)
  BorderlineSMOTE : borderline variant — only borderline minority samples
  ADASYN          : Adaptive Synthetic Sampling (density-adaptive)

CV Protocol
-----------
  StratifiedGroupKFold(n_splits=5), groups = subject_id
  Resampling applied to TRAINING fold ONLY.
  StandardScaler fit on training fold only; transforms test fold with same params.
  compute_sample_weight('balanced') applied to all strategies for XGBoost.

Outputs (outputs/)
------------------
  smote_variants_comparison.csv   per-fold metrics for all 4 strategies
  smote_variants_comparison.png   grouped bar chart (Macro-F1 / BalAcc / PR-AUC)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from xgboost import XGBClassifier

# ── constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = ["A1", "mu1", "sigma1", "A2", "mu2", "sigma2",
                "A3", "mu3", "sigma3"]
N_SPLITS     = 5
RANDOM_STATE = 42

# 3-class experiment — Choppy excluded; Tight is the minority class
CLASS_ORDER = ["Wiry", "Slippery", "Tight"]

STRATEGIES  = ["baseline", "SMOTE", "BorderlineSMOTE", "ADASYN"]


# ── model factory ─────────────────────────────────────────────────────────────
def make_xgb():
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss",
        n_jobs=-1, random_state=RANDOM_STATE,
        verbosity=0,
    )


# ── oversampling ──────────────────────────────────────────────────────────────
def resample(strategy: str, X_tr: np.ndarray, y_tr: np.ndarray):
    """Apply the requested strategy to (X_tr, y_tr). Returns (X_res, y_res).

    k is clipped to the minimum class count − 1 so all strategies work even
    when one class has very few samples (e.g. Tight ≈ 163 total, ~130 in train).
    """
    min_cnt = int(np.bincount(y_tr).min())
    k = max(1, min(5, min_cnt - 1))

    if strategy == "baseline":
        return X_tr, y_tr
    elif strategy == "SMOTE":
        sampler = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
    elif strategy == "BorderlineSMOTE":
        sampler = BorderlineSMOTE(k_neighbors=k, random_state=RANDOM_STATE)
    elif strategy == "ADASYN":
        sampler = ADASYN(n_neighbors=k, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return sampler.fit_resample(X_tr, y_tr)


# ── metric helper ─────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, y_prob, n_classes):
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    y_bin    = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])
    roc_auc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    pr_auc  = average_precision_score(y_bin, y_prob, average="macro")
    return {
        "macro_f1": round(macro_f1, 4),
        "bal_acc":  round(bal_acc,  4),
        "roc_auc":  round(roc_auc,  4),
        "pr_auc":   round(pr_auc,   4),
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # ── 1. Load original dataset (no augmented rows) ──────────────────────────
    df = pd.read_csv(OUT / "dataset.csv")
    df = df[df["converged"] == True].reset_index(drop=True)

    # Filter to 3 classes only (safety check)
    df = df[df["pulse_type_en"].isin(CLASS_ORDER)].reset_index(drop=True)

    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    true_order = list(le.classes_)          # alphabetical: Slippery/Tight/Wiry
    df["label"] = le.transform(df["pulse_type_en"])

    X      = df[FEATURE_COLS].values.astype(np.float64)
    y      = df["label"].values
    groups = df["subject_id"].values
    n_cls  = len(true_order)

    print(f"Dataset : {X.shape[0]} beats  |  {n_cls} classes  |  "
          f"{df['subject_id'].nunique()} subjects")
    print("Class   :", {c: int((y == i).sum()) for i, c in enumerate(true_order)})
    print()

    # ── 2. Cross-validation ───────────────────────────────────────────────────
    skf  = StratifiedGroupKFold(n_splits=N_SPLITS)
    rows = []          # one row per (fold, strategy)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y, groups), 1):
        X_tr_raw, X_te_raw = X[tr_idx], X[te_idx]
        y_tr,     y_te     = y[tr_idx], y[te_idx]

        # Scale: fit on train, apply to test — same scaler for all strategies
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_raw)
        X_te_sc = scaler.transform(X_te_raw)

        print(f"Fold {fold}  train_orig={len(y_tr)}  test={len(y_te)}")

        for strat in STRATEGIES:
            try:
                X_res, y_res = resample(strat, X_tr_sc, y_tr)
            except Exception as exc:
                print(f"  [{strat:<16}] WARN: {exc} — falling back to original")
                X_res, y_res = X_tr_sc, y_tr

            sw  = compute_sample_weight("balanced", y_res)
            clf = make_xgb()
            clf.fit(X_res, y_res, sample_weight=sw)

            y_pred = clf.predict(X_te_sc)
            y_prob = clf.predict_proba(X_te_sc)
            m = evaluate(y_te, y_pred, y_prob, n_cls)

            rows.append({"fold": fold, "strategy": strat, **m})
            print(f"  {strat:<18}  n_train={len(y_res):5d}  "
                  f"F1={m['macro_f1']:.3f}  "
                  f"BalAcc={m['bal_acc']:.3f}  "
                  f"PR={m['pr_auc']:.3f}")
        print()

    # ── 3. Aggregate ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    csv_path   = OUT / "smote_variants_comparison.csv"
    results_df.to_csv(csv_path, index=False)

    summary = (results_df
               .groupby("strategy")[["macro_f1", "bal_acc", "pr_auc"]]
               .agg(["mean", "std"])
               .loc[STRATEGIES])          # preserve display order

    # ── 4. Print summary table ────────────────────────────────────────────────
    COL_W   = 20
    STRAT_W = 18
    line_len = STRAT_W + COL_W * 3 + 2
    print("\n" + "═" * line_len)
    print("SMOTE VARIANTS  |  XGBoost  |  3-class: Wiry / Slippery / Tight  "
          "(Tight as minority)")
    print("═" * line_len)
    print(f"  {'Strategy':<{STRAT_W}}"
          f"{'Macro-F1':>{COL_W}}"
          f"{'BalAcc':>{COL_W}}"
          f"{'PR-AUC':>{COL_W}}")
    print("─" * line_len)
    for strat in STRATEGIES:
        r = summary.loc[strat]
        f1_s = f"{r[('macro_f1','mean')]:.4f} ± {r[('macro_f1','std')]:.4f}"
        ba_s = f"{r[('bal_acc', 'mean')]:.4f} ± {r[('bal_acc', 'std')]:.4f}"
        pr_s = f"{r[('pr_auc',  'mean')]:.4f} ± {r[('pr_auc',  'std')]:.4f}"
        print(f"  {strat:<{STRAT_W}}{f1_s:>{COL_W}}{ba_s:>{COL_W}}{pr_s:>{COL_W}}")
    print("═" * line_len)

    # ── 5. Bar chart ──────────────────────────────────────────────────────────
    metric_keys   = ["macro_f1", "bal_acc", "pr_auc"]
    metric_labels = ["Macro-F1", "Balanced Acc", "PR-AUC"]
    palette       = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    x             = np.arange(len(STRATEGIES))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=False)
    fig.suptitle(
        "XGBoost + SMOTE Variants — 3-Class (Tight as minority)\n"
        "5-fold Subject-level CV  |  original 9-Gaussian features",
        fontsize=11, fontweight="bold",
    )

    for ax, mkey, mlabel in zip(axes, metric_keys, metric_labels):
        means = [summary.loc[s][(mkey, "mean")] for s in STRATEGIES]
        stds  = [summary.loc[s][(mkey, "std")]  for s in STRATEGIES]
        bars  = ax.bar(x, means, yerr=stds, capsize=5,
                       color=palette, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGIES, rotation=22, ha="right", fontsize=9)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=9)
        ymax = max(m + s for m, s in zip(means, stds))
        ax.set_ylim(0, ymax * 1.28)
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mean + std + 0.006,
                    f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    chart_path = OUT / "smote_variants_comparison.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Saved → {csv_path.relative_to(ROOT)}")
    print(f"  Saved → {chart_path.relative_to(ROOT)}")
    print("DONE")


if __name__ == "__main__":
    main()
