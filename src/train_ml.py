"""
TCM Pulse — 3-Class ML Classification (弦脈 / 滑脈 / 緊脈)
============================================================
Excludes 澀脈 (Choppy).  Augmentation targets 緊脈 (Tight) instead.

Design
------
* Features  : 9 Gaussian parameters (A1,μ1,σ1, A2,μ2,σ2, A3,μ3,σ3)
* Labels    : Wiry / Slippery / Tight
* CV        : StratifiedGroupKFold(n_splits=5), groups = subject_id
* Imbalance : Tight waveform augmentation injected into training fold only
              + compute_sample_weight for XGBoost
* Models    : SVM (RBF), Random Forest, XGBoost
* Metrics   : Macro-F1, Balanced Accuracy, Macro ROC-AUC, Macro PR-AUC

Outputs (outputs/)
------------------
  ml_results_aug.csv            per-fold metrics (with augmentation)
  confusion_matrices_aug.png    3-panel normalised confusion matrices
  classification_report_aug.txt aggregate per-class report

Data Leakage Prevention
-----------------------
  GroupKFold is performed on the ORIGINAL dataset only.
  Synthetic Tight beats (aug_mode != 'original') are injected into the
  TRAINING fold ONLY, filtered to subject_ids present in that training fold.
  The test fold is always 100% original data.
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ── constants ──────────────────────────────────────────────────────────────────
FEATURE_COLS = ["A1", "mu1", "sigma1", "A2", "mu2", "sigma2",
                "A3", "mu3", "sigma3"]
N_SPLITS     = 5
RANDOM_STATE = 42

# 3-class experiment — Choppy excluded
CLASS_ORDER = ["Wiry", "Slippery", "Tight"]


def make_models():
    return {
        "SVM": SVC(
            kernel="rbf", C=10.0, gamma="scale",
            class_weight="balanced",
            probability=True, random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss",
            n_jobs=-1, random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }


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


def main():
    # ── 1. Load & prepare data ─────────────────────────────────────────────────
    df = pd.read_csv(OUT / "dataset.csv")
    df = df[df["converged"] == True].reset_index(drop=True)

    # Filter to 3 classes only (safety check — build_dataset already excludes Choppy)
    df = df[df["pulse_type_en"].isin(CLASS_ORDER)].reset_index(drop=True)

    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    true_order = list(le.classes_)           # alphabetical: ['Slippery','Tight','Wiry']
    df["label"] = le.transform(df["pulse_type_en"])

    X      = df[FEATURE_COLS].values.astype(np.float64)
    y      = df["label"].values
    groups = df["subject_id"].values
    n_cls  = len(true_order)

    print(f"Dataset: {X.shape[0]} beats, {n_cls} classes, "
          f"{df['subject_id'].nunique()} subjects")
    print("Class dist:", {c: int((y == i).sum()) for i, c in enumerate(true_order)})
    print()

    # ── 1b. Load augmented Tight (synthetic rows only) ─────────────────────────
    aug_df    = pd.read_csv(OUT / "tight_augmented_meta.csv")
    aug_df    = aug_df[
        (aug_df["converged"] == True) &
        (aug_df["aug_mode"] != "original")
    ].reset_index(drop=True)
    aug_label = int(le.transform(["Tight"])[0])
    aug_X_all = aug_df[FEATURE_COLS].values.astype(np.float64)
    aug_y_all = np.full(len(aug_df), aug_label, dtype=int)
    print(f"Augmented Tight : {len(aug_df)} synthetic beats (no originals)")
    print(f"  mode dist     : {aug_df['aug_mode'].value_counts().to_dict()}")
    print()

    # ── 2. Cross-validation ────────────────────────────────────────────────────
    skf = StratifiedGroupKFold(n_splits=N_SPLITS)

    fold_rows = []
    all_preds = {m: {"y_true": [], "y_pred": []}
                 for m in ["SVM", "RandomForest", "XGBoost"]}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y, groups), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        # Inject augmented Tight (train-fold subject_ids only)
        train_sids = set(groups[tr_idx])
        fold_mask  = aug_df["subject_id"].isin(train_sids).values
        X_aug_fold = scaler.transform(aug_X_all[fold_mask])
        y_aug_fold = aug_y_all[fold_mask]

        X_res = np.vstack([X_tr, X_aug_fold])
        y_res = np.concatenate([y_tr, y_aug_fold])

        print(f"Fold {fold}  orig_train={len(y_tr)}  +aug={len(y_aug_fold)}"
              f"  total_train={len(y_res)}  test={len(y_te)}")

        models = make_models()
        sw_res = compute_sample_weight("balanced", y_res)

        for name, clf in models.items():
            if name == "XGBoost":
                clf.fit(X_res, y_res, sample_weight=sw_res)
            else:
                clf.fit(X_res, y_res)

            y_pred   = clf.predict(X_te)
            y_prob   = clf.predict_proba(X_te)
            metrics  = evaluate(y_te, y_pred, y_prob, n_cls)
            fold_rows.append({"fold": fold, "model": name, **metrics})
            all_preds[name]["y_true"].extend(y_te.tolist())
            all_preds[name]["y_pred"].extend(y_pred.tolist())

            print(f"  {name:14s}  "
                  f"F1={metrics['macro_f1']:.3f}  "
                  f"BalAcc={metrics['bal_acc']:.3f}  "
                  f"ROC={metrics['roc_auc']:.3f}  "
                  f"PR={metrics['pr_auc']:.3f}")
        print()

    # ── 3. Aggregate ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(fold_rows)
    results_df.to_csv(OUT / "ml_results_aug.csv", index=False)

    summary = (results_df.groupby("model")[["macro_f1", "bal_acc", "roc_auc", "pr_auc"]]
               .agg(["mean", "std"]))
    print("\n" + "=" * 72)
    print("SUMMARY  (mean ± std across 5 folds)  — 3-class: Wiry/Slippery/Tight")
    print("=" * 72)
    for model in ["SVM", "RandomForest", "XGBoost"]:
        row = summary.loc[model]
        print(f"\n  {model}")
        for m in ["macro_f1", "bal_acc", "roc_auc", "pr_auc"]:
            mu  = row[(m, "mean")]
            std = row[(m, "std")]
            print(f"    {m:12s}: {mu:.4f} ± {std:.4f}")

    # ── 4. Confusion matrices ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        "Normalised Confusion Matrices — With Tight Augmentation\n"
        "(5-Fold Subject-level CV, 3-class: Wiry / Slippery / Tight)",
        fontsize=11, fontweight="bold", y=1.04,
    )
    for ax, model in zip(axes, ["SVM", "RandomForest", "XGBoost"]):
        yt = np.array(all_preds[model]["y_true"])
        yp = np.array(all_preds[model]["y_pred"])
        cm = confusion_matrix(yt, yp, normalize="true")

        im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(n_cls))
        ax.set_xticklabels(true_order, rotation=25, ha="right", fontsize=9)
        ax.set_yticks(range(n_cls))
        ax.set_yticklabels(true_order, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True",      fontsize=10)
        ax.set_title(model, fontsize=11, fontweight="bold")
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                        color="white" if cm[i,j] > 0.5 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices_aug.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 5. Classification report (XGBoost) ────────────────────────────────────
    yt_xgb = np.array(all_preds["XGBoost"]["y_true"])
    yp_xgb = np.array(all_preds["XGBoost"]["y_pred"])
    report = classification_report(yt_xgb, yp_xgb, target_names=true_order, zero_division=0)
    print("\n=== XGBoost classification report ===")
    print(report)
    (OUT / "classification_report_aug.txt").write_text(
        "XGBoost with Tight augmentation\n"
        "3-class (Wiry / Slippery / Tight), 5-fold subject-level CV\n\n"
        + report
    )

    print(f"\n  Saved → outputs/ml_results_aug.csv")
    print(f"  Saved → outputs/confusion_matrices_aug.png")
    print(f"  Saved → outputs/classification_report_aug.txt")
    print("=" * 72)
    print("ML 3-class + Tight augmentation DONE")


if __name__ == "__main__":
    main()
