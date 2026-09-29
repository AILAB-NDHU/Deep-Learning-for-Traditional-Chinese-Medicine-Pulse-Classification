"""
Revision R1 — re-run the classical route (SVM / RF / XGBoost) with the three
Gaussian components sorted by mu (mu1 < mu2 < mu3) to remove label-switching
ambiguity in the 9-D feature space (Reviewer 1, Major 4).

Faithful re-implementation of `_working_archive/stat_rerun_harness.py::run_ml`
(the harness that produced the paper's 50-run numbers), minus the torch imports
so it runs on a CPU-only machine.  Everything that defines a run is preserved:

  * StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=repeat), repeat = 0..9
  * only converged beats; StandardScaler fit on the training fold
  * converged synthetic Tight beats (Gaussian features) injected into the
    training fold for training-fold subjects only  (== paper's classical route)
  * identical model hyper-parameters and seeds
  * identical metric code (macro-F1 primary; bal-acc, ROC-AUC, PR-AUC, Tight-F1)

`--no-sort` reproduces the original feature ordering and is the sanity check:
its XGBoost mean should match the archive (0.7415 over 50 runs).

Outputs one CSV in revision_r1/outputs/, one row per (model, repeat, fold),
same columns as the archive raw CSV plus `mu_sorted`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "TCM" / "outputs" / "_intermediate"
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(exist_ok=True)

CLASS_ORDER = ["Wiry", "Slippery", "Tight"]
N_SPLITS = 5
N_CLASSES = 3
FEATURE_COLS = ["A1", "mu1", "sigma1", "A2", "mu2", "sigma2", "A3", "mu3", "sigma3"]
RAW_COLS = ["model", "repeat", "fold", "seed", "macro_f1", "bal_acc", "roc_auc",
            "pr_auc", "tight_f1", "protocol", "use_aug", "use_sampler", "n_test", "mu_sorted"]


# ── mu-sorting ───────────────────────────────────────────────────────────────
def sort_components(F: np.ndarray) -> tuple[np.ndarray, int]:
    """Reorder (A_i, mu_i, sigma_i) triplets so that mu1 < mu2 < mu3 per row.

    Returns the sorted matrix and the number of rows whose ordering changed.
    """
    P = F.reshape(-1, 3, 3)                      # (N, component, [A, mu, sigma])
    order = np.argsort(P[:, :, 1], axis=1, kind="stable")
    changed = int((order != np.arange(3)).any(axis=1).sum())
    Ps = np.take_along_axis(P, order[:, :, None], axis=1)
    return Ps.reshape(-1, 9), changed


# ── data ─────────────────────────────────────────────────────────────────────
def load_data(sort: bool) -> dict:
    df = pd.read_csv(DATA / "dataset.csv")
    df = df[df["pulse_type_en"].isin(CLASS_ORDER)].reset_index(drop=True)
    le = LabelEncoder().fit(sorted(CLASS_ORDER))          # Slippery=0, Tight=1, Wiry=2
    y = le.transform(df["pulse_type_en"].values)
    groups = df["subject_id"].values
    feats = df[FEATURE_COLS].values.astype(np.float64)
    converged = df["converged"].values.astype(bool)

    aug_meta_full = pd.read_csv(DATA / "tight_augmented_meta.csv")
    synth = (aug_meta_full["aug_mode"] != "original").values
    aug_meta = aug_meta_full[synth].reset_index(drop=True)
    tight_lbl = int(le.transform(["Tight"])[0])
    aug_y = np.full(len(aug_meta), tight_lbl, dtype=int)
    aug_subj = aug_meta["subject_id"].values
    aug_feats = aug_meta[FEATURE_COLS].values.astype(np.float64)
    aug_conv = aug_meta["converged"].values.astype(bool)

    n_changed = n_changed_aug = 0
    if sort:
        feats, n_changed = sort_components(feats)
        aug_feats, n_changed_aug = sort_components(aug_feats)

    print(f"beats={len(df)}  subjects={df['subject_id'].nunique()}  "
          f"synthetic Tight={len(aug_meta)} (converged {aug_conv.sum()})")
    print(f"mu-sorted={sort}  rows reordered: original={n_changed}  synthetic={n_changed_aug}")
    return dict(y=y, groups=groups, feats=feats, converged=converged,
                aug_y=aug_y, aug_subj=aug_subj, aug_feats=aug_feats, aug_conv=aug_conv,
                tight_idx=tight_lbl)


# ── metrics (verbatim from harness) ──────────────────────────────────────────
def compute_metrics(y_true, y_prob, tight_idx):
    y_pred = y_prob.argmax(axis=1)
    labels = list(range(N_CLASSES))
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        roc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        roc = float("nan")
    pr_list = [average_precision_score((y_true == c).astype(int), y_prob[:, c])
               for c in labels if (y_true == c).sum() > 0]
    pr = float(np.mean(pr_list)) if pr_list else float("nan")
    return dict(macro_f1=float(macro_f1), bal_acc=float(bal_acc), roc_auc=float(roc),
                pr_auc=pr, tight_f1=float(per_class_f1[tight_idx]))


# ── one fold, all three models (verbatim logic from harness.run_ml) ──────────
def run_ml(split, data):
    train_idx, test_idx = split
    feats, y, groups, conv = data["feats"], data["y"], data["groups"], data["converged"]
    tr_i = train_idx[conv[train_idx]]
    te_i = test_idx[conv[test_idx]]
    X_tr, y_tr = feats[tr_i], y[tr_i]
    X_te, y_te = feats[te_i], y[te_i]
    tr_subj = groups[tr_i]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    amask = data["aug_conv"] & np.isin(data["aug_subj"], list(set(tr_subj)))
    X_aug = scaler.transform(data["aug_feats"][amask])
    X_res = np.vstack([X_tr, X_aug])
    y_res = np.concatenate([y_tr, data["aug_y"][amask]])
    sw = compute_sample_weight("balanced", y_res)

    models = {
        "SVM": SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                   probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=None,
                                               class_weight="balanced", n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                                 n_jobs=-1, random_state=42, verbosity=0),
    }
    out = {}
    for name, clf in models.items():
        if name == "XGBoost":
            clf.fit(X_res, y_res, sample_weight=sw)
        else:
            clf.fit(X_res, y_res)
        m = compute_metrics(y_te, clf.predict_proba(X_te), data["tight_idx"])
        m["n_test"] = len(y_te)
        out[name] = m
    return out


# ── runner ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", dest="sort", action="store_true", default=True)
    ap.add_argument("--no-sort", dest="sort", action="store_false")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    tag = "musorted" if args.sort else "unsorted_check"
    raw_csv = OUTDIR / f"classical_{tag}_raw.csv"
    done = set()
    if raw_csv.exists():
        d = pd.read_csv(raw_csv)
        done = set(zip(d["model"], d["repeat"], d["fold"]))

    data = load_data(args.sort)
    t0 = time.time()
    for repeat in range(args.repeats):
        skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=repeat)
        for fold, split in enumerate(skf.split(data["feats"], data["y"], data["groups"])):
            if all((m, repeat, fold) in done for m in ("SVM", "RandomForest", "XGBoost")):
                continue
            res = run_ml(split, data)
            rows = [dict(model=name, repeat=repeat, fold=fold, seed=repeat, **m,
                         protocol="leakfree", use_aug=True, use_sampler=False,
                         mu_sorted=args.sort) for name, m in res.items()]
            pd.DataFrame(rows)[RAW_COLS].to_csv(raw_csv, mode="a",
                                                header=not raw_csv.exists(), index=False)
            print(f"[{time.time()-t0:6.0f}s] r{repeat} f{fold}  " +
                  "  ".join(f"{n}={m['macro_f1']:.4f}" for n, m in res.items()), flush=True)

    d = pd.read_csv(raw_csv)
    print("\n== summary (mean ± SD over runs) ==")
    print(d.groupby("model")[["macro_f1", "bal_acc", "roc_auc", "pr_auc", "tight_f1"]]
           .agg(["mean", "std", "count"]).round(4).to_string())
    print(f"\nwritten: {raw_csv}")


if __name__ == "__main__":
    main()
