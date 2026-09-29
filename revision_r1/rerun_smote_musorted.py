"""
SMOTE-variant ablation under the unified repeated-CV harness.
=============================================================

Purpose
-------
Re-run ONLY the feature-space imbalance-strategy ablation (paper Table `tab:smote`)
under the SAME repeated, leakage-free harness used by `stat_rerun_harness.py`, so
that the SMOTE table is consistent with the rest of the statistical rerun.

What is PRESERVED (unchanged from stat_rerun_harness.run_ml)
-----------------------------------------------------------
* Feature set      : the nine three-Gaussian parameters (FEATURE_COLS), converged
                     beats only, StandardScaler fit on the training fold.
* Model            : XGBoost with hyperparameters IDENTICAL to the main run
                     (n_estimators=300, max_depth=6, lr=0.05, subsample=0.8,
                     colsample_bytree=0.8, eval_metric=mlogloss, random_state=42).
* Leakage control  : subject-level StratifiedGroupKFold grouped by subject; the
                     SAME outer-partition seeds as the main run (outer seed =
                     repeat index 0..9); synthetic Tight beats injected from
                     TRAIN-fold subjects only; the test fold is scaled with the
                     train scaler, never oversampled, and read exactly once.
* Baseline row     : reproduces the main-run XGBoost (`run_ml`) leakage-free number
                     byte-for-byte (consistency check / pairing anchor).

What is CHANGED (the point of this job)
---------------------------------------
* Four minority-class strategies are compared on the assembled training matrix:
    baseline        : no feature-space oversampling (class-weighted XGBoost).
    SMOTE           : imblearn.over_sampling.SMOTE            (Chawla et al. 2002).
    BorderlineSMOTE : imblearn.over_sampling.BorderlineSMOTE  (Han et al. 2005).
    ADASYN          : imblearn.over_sampling.ADASYN           (He et al. 2008).
  Each sampler is fit_resample'd ONLY on the training-fold feature matrix; the
  synthetic points are interpolations of TRAINING-fold minority features and
  introduce NO new subjects, so they cannot leak test identities.

Design
------
* Repeated subject-level CV: N_REPEATS repeats x 5 folds, identical partitions to
  the main run (StratifiedGroupKFold(shuffle=True, random_state=repeat)).
* Resumable: appends one row per (model, repeat, fold) to the raw CSV; an
  interrupted run re-reads it and skips finished cells.
* ADASYN degeneration is logged PER FOLD to a sidecar CSV (it may fail / fall back
  to the baseline when the minority density criterion cannot be met).

Determinism
-----------
XGBoost random_state=42; sampler random_state=42 (== the original single-run
RANDOM_STATE). Per-run torch/numpy seed logged for provenance. imbalanced-learn
and xgboost versions logged at startup.
"""

import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.resolve()
import os
RES = Path(os.environ.get("RES_DIR", ROOT / "outputs")).resolve()
RES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import imblearn
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import xgboost

# Reuse the main harness's data loader, metrics and constants verbatim so the
# feature extraction / leakage logic is provably the same code path.
from rerun_deep_with_preds import (   # revision copy: mu-sorted features, env-configurable paths
    load_data, compute_metrics, N_SPLITS, N_CLASSES, TIGHT_IDX,
    RUN_SEED_BASE, MODEL_IDS, RAW_COLS,
)

SAMPLER_SEED = 42          # == compare_smote_variants.py RANDOM_STATE
XGB_SEED = 42              # == run_ml XGBoost random_state

# strategy key -> (raw-CSV model name, sampler factory or None for baseline)
STRATEGIES = ["baseline", "SMOTE", "BorderlineSMOTE", "ADASYN"]
MODEL_NAME = {
    "baseline": "xgb_baseline",
    "SMOTE": "xgb_smote",
    "BorderlineSMOTE": "xgb_borderline",
    "ADASYN": "xgb_adasyn",
}

RAW_CSV = RES / "smote_musorted_raw.csv"
ADASYN_LOG = RES / "smote_musorted_adasyn_log.csv"


def make_xgb():
    """XGBoost with hyperparameters IDENTICAL to stat_rerun_harness.run_ml."""
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
        n_jobs=-1, random_state=XGB_SEED, verbosity=0,
    )


def make_sampler(strategy, k):
    if strategy == "SMOTE":
        return SMOTE(k_neighbors=k, random_state=SAMPLER_SEED)
    if strategy == "BorderlineSMOTE":
        return BorderlineSMOTE(k_neighbors=k, random_state=SAMPLER_SEED)
    if strategy == "ADASYN":
        return ADASYN(n_neighbors=k, random_state=SAMPLER_SEED)
    raise ValueError(strategy)


def assemble_training_matrix(split, data):
    """Replicate run_ml's XGBoost data assembly EXACTLY (scaled, aug-injected).

    Returns (X_res, y_res, X_te, y_te, minority_count_before_resample).
    """
    train_idx, test_idx = split
    feats, y, groups = data["feats"], data["y"], data["groups"]
    conv = data["converged"]
    aug_feats, aug_y, aug_subj, aug_conv = (
        data["aug_feats"], data["aug_y"], data["aug_subj"], data["aug_conv"])

    tr_mask = conv[train_idx]
    te_mask = conv[test_idx]
    tr_i = train_idx[tr_mask]
    te_i = test_idx[te_mask]
    X_tr, y_tr = feats[tr_i], y[tr_i]
    X_te, y_te = feats[te_i], y[te_i]
    tr_subj = groups[tr_i]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # inject aug (converged synthetic, train-fold subjects only) — identical to run_ml
    amask = aug_conv & np.isin(aug_subj, list(set(tr_subj)))
    X_aug = scaler.transform(aug_feats[amask])
    y_aug = aug_y[amask]
    X_res = np.vstack([X_tr, X_aug])
    y_res = np.concatenate([y_tr, y_aug])
    return X_res, y_res, X_te, y_te


def run_strategy(strategy, X_res, y_res, X_te, y_te):
    """Apply one imbalance strategy to the assembled training matrix, fit XGBoost,
    score the (never-oversampled) test fold. Returns (metrics, info)."""
    counts_before = np.bincount(y_res, minlength=N_CLASSES)
    min_before = int(counts_before.min())
    info = dict(degenerated=False, n_before=int(len(y_res)),
                min_before=min_before, n_after=int(len(y_res)),
                min_after=min_before, error="")

    if strategy == "baseline":
        X_s, y_s = X_res, y_res
    else:
        k = max(1, min(5, min_before - 1))
        sampler = make_sampler(strategy, k)
        try:
            X_s, y_s = sampler.fit_resample(X_res, y_res)
            counts_after = np.bincount(y_s, minlength=N_CLASSES)
            info["n_after"] = int(len(y_s))
            info["min_after"] = int(counts_after.min())
            # ADASYN can "succeed" yet add ~no synthetic points (density degenerate)
            if len(y_s) == len(y_res):
                info["degenerated"] = True
        except Exception as exc:  # noqa: BLE001 — mirror original silent fallback, but LOG it
            X_s, y_s = X_res, y_res
            info["degenerated"] = True
            info["error"] = f"{type(exc).__name__}: {exc}"

    sw = compute_sample_weight("balanced", y_s)
    clf = make_xgb()
    clf.fit(X_s, y_s, sample_weight=sw)
    y_prob = clf.predict_proba(X_te)
    m = compute_metrics(y_te, y_prob)
    m["epochs"] = 0
    m["n_test"] = len(y_te)
    m["params"] = 0
    return m, info


def load_done(raw_csv):
    if raw_csv.exists():
        d = pd.read_csv(raw_csv)
        return set(zip(d["model"], d["repeat"], d["fold"]))
    return set()


def append_row(raw_csv, row):
    header = not raw_csv.exists()
    pd.DataFrame([row])[RAW_COLS].to_csv(raw_csv, mode="a", header=header, index=False)


def append_adasyn_log(row):
    header = not ADASYN_LOG.exists()
    cols = ["repeat", "fold", "strategy", "degenerated", "n_before", "min_before",
            "n_after", "min_after", "error"]
    pd.DataFrame([row])[cols].to_csv(ADASYN_LOG, mode="a", header=header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    print(f"imbalanced-learn {imblearn.__version__} | xgboost {xgboost.__version__} | "
          f"repeats={args.repeats} | sampler_seed={SAMPLER_SEED} | xgb_seed={XGB_SEED}")
    data = load_data()
    print(f"Data: feats={data['feats'].shape} Tight_idx={TIGHT_IDX} "
          f"subjects={len(np.unique(data['groups']))} aug_feats={data['aug_feats'].shape}")

    done = load_done(RAW_CSV)
    print(f"Already done cells: {len(done)}")

    t_start = time.time()
    adasyn_degen = 0
    adasyn_total = 0
    for repeat in range(args.repeats):
        skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=repeat)
        splits = list(skf.split(data["feats"], data["y"], data["groups"]))
        for fold, split in enumerate(splits):
            X_res, y_res, X_te, y_te = assemble_training_matrix(split, data)
            # per-run seed for provenance (XGBoost itself is fixed-seed and deterministic)
            run_seed = RUN_SEED_BASE + repeat * 100 + fold * 10 + MODEL_IDS["XGBoost"]
            for strategy in STRATEGIES:
                mname = MODEL_NAME[strategy]
                if (mname, repeat, fold) in done:
                    continue
                t0 = time.time()
                m, info = run_strategy(strategy, X_res, y_res, X_te, y_te)
                append_row(RAW_CSV, dict(
                    model=mname, repeat=repeat, fold=fold, seed=repeat,
                    protocol="leakfree", use_aug=True, use_sampler=(strategy != "baseline"),
                    run_seed=run_seed, **m))
                if strategy == "ADASYN":
                    adasyn_total += 1
                    adasyn_degen += int(info["degenerated"])
                    append_adasyn_log(dict(repeat=repeat, fold=fold, strategy=strategy, **{
                        k: info[k] for k in ["degenerated", "n_before", "min_before",
                                             "n_after", "min_after", "error"]}))
                flag = "  <ADASYN-FELLBACK>" if (strategy == "ADASYN" and info["degenerated"]) else ""
                print(f"[{time.time()-t_start:6.0f}s] r{repeat} f{fold} {mname:15s} "
                      f"F1={m['macro_f1']:.3f} tight={m['tight_f1']:.3f} "
                      f"n_tr {info['n_before']}->{info['n_after']} "
                      f"minTight {info['min_before']}->{info['min_after']}"
                      f"{flag} ({time.time()-t0:.1f}s)", flush=True)

    print(f"\nDONE in {time.time()-t_start:.0f}s. Rows -> {RAW_CSV.name}")
    print(f"ADASYN degeneration: {adasyn_degen}/{adasyn_total} folds fell back to baseline.")
    summary = dict(adasyn_degen_folds=adasyn_degen, adasyn_total_folds=adasyn_total,
                   imblearn=imblearn.__version__, xgboost=xgboost.__version__,
                   sampler_seed=SAMPLER_SEED, xgb_seed=XGB_SEED)
    (RES / "smote_musorted_run_meta.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
