"""
Revision R1 — subject-level analysis from the per-beat prediction dumps
(outputs/preds/<model>__<protocol>__r<repeat>_f<fold>.npz).

Answers Reviewer 2 Major 3 ("unit of analysis and uncertainty estimates") and
Reviewer 1 minor (aggregated confusion matrices with per-class P/R):

  A. Beat-level vs subject-weighted vs subject-level macro-F1 per model
     - subject-weighted: every beat weighted 1 / (#beats of its subject-label
       group in that test fold)  -> each subject counts once
     - subject-level: mean probability over a subject-label group's beats ->
       one prediction per group (a subject with two pulse-type labels forms two
       groups; that is how the labels were assigned in the data)
  B. Subject-clustered bootstrap 95% CI of macro-F1 and Tight-F1 on each
     repeat's full out-of-fold prediction set (resample subjects, not beats),
     summarised over the 10 repeats
  C. Confusion matrices aggregated over the 50 runs: beat-level, and
     subject-level (groups), each with per-class precision / recall / F1
  D. Per-fold subject and class counts; beats-per-subject distribution
  E. Tight-class F1: distribution over the 50 runs + clustered CI

Usage:  python analyze_subject_level.py [--protocol leakfree] [--models resnet,XGBoost,...]
Writes outputs/subject_level_report.md  (+ csv side tables)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

import os
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("RES_DIR", HERE / "outputs")).resolve()
PRED_DIR = OUT / "preds"
DATA = Path(os.environ.get("TCM_DATA", HERE.parent / "TCM" / "outputs" / "_intermediate")).resolve()

CLASS_NAMES = ["Slippery", "Tight", "Wiry"]        # LabelEncoder(sorted) order used by harness
TIGHT = 1
N_BOOT = 2000
RNG = np.random.default_rng(0)

DISPLAY = {"resnet": "ResNet1D (plain)", "resnet_gated": "ResNet1D + Gated",
           "resnet_gc": "ResNet1D + GC", "resnet_gc_gated": "ResNet1D + GC + Gated",
           "protonet": "Prototypical Network", "inception": "Inception1D",
           "cnn_aug": "1D-CNN + 6-mode aug", "cnn_noaug": "1D-CNN (no aug)",
           "cnn_aug_nosampler": "1D-CNN + aug, no sampler", "cnn_noaug_nosampler": "1D-CNN no aug, no sampler",
           "transformer": "Patch Transformer", "XGBoost": "XGBoost",
           "RandomForest": "Random Forest", "SVM": "SVM"}


# ── helpers ──────────────────────────────────────────────────────────────────
def macro_f1(y, yhat, w=None):
    return f1_score(y, yhat, average="macro", labels=[0, 1, 2], zero_division=0, sample_weight=w)


def tight_f1(y, yhat, w=None):
    return f1_score(y, yhat, average=None, labels=[0, 1, 2], zero_division=0, sample_weight=w)[TIGHT]


def group_weights(groups):
    """1 / group size for every element -> each group sums to weight 1."""
    _, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def subject_level(y, prob, groups):
    """Mean probability per subject-label group -> one label & prediction per group."""
    df = pd.DataFrame(prob, columns=CLASS_NAMES)
    df["g"], df["y"] = groups, y
    agg = df.groupby("g").agg({**{c: "mean" for c in CLASS_NAMES}, "y": "first"})
    return agg["y"].values, agg[CLASS_NAMES].values.argmax(1)


def cluster_boot(y, yhat, groups, n_boot=N_BOOT):
    """Bootstrap over subject groups (clusters); returns CIs for macro-F1 and Tight-F1."""
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    mf, tf = [], []
    for _ in range(n_boot):
        pick = RNG.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        mf.append(macro_f1(y[idx], yhat[idx])); tf.append(tight_f1(y[idx], yhat[idx]))
    return np.percentile(mf, [2.5, 97.5]), np.percentile(tf, [2.5, 97.5])


def load_preds(protocol, models):
    """-> {model: {repeat: {fold: (test_idx, y_true, y_prob)}}}"""
    store = defaultdict(lambda: defaultdict(dict))
    for f in sorted(PRED_DIR.glob(f"*__{protocol}__r*_f*.npz")):
        name, _, rf = f.stem.split("__")
        if models and name not in models:
            continue
        r, k = rf[1:].split("_f")
        z = np.load(f)
        store[name][int(r)][int(k)] = (z["test_idx"], z["y_true"], z["y_prob"])
    return store


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="leakfree")
    ap.add_argument("--models", default=None)
    args = ap.parse_args()
    models = set(args.models.split(",")) if args.models else None

    ds = pd.read_csv(DATA / "dataset.csv")
    y_all = pd.Categorical(ds["pulse_type_en"], categories=CLASS_NAMES).codes
    subj = ds["subject_id"].values
    # subject-label group id: subject + label (25 subjects carry two labels)
    grp_all = (ds["subject_id"].astype(str) + "|" + ds["pulse_type_en"]).values

    store = load_preds(args.protocol, models)
    if not store:
        sys.exit(f"no prediction files for protocol={args.protocol} in {PRED_DIR}")
    L = [f"# Subject-level analysis ({args.protocol})\n"]

    # ── D. dataset structure (model-independent) ─────────────────────────────
    L.append("## D. Unit-of-analysis structure\n")
    bps = ds.groupby(["subject_id", "pulse_type_en"]).size()
    L.append(f"- beats = {len(ds)}, subjects = {ds.subject_id.nunique()}, "
             f"subject-label groups = {len(bps)} (25 subjects carry two labels)")
    L.append(f"- beats per subject-label group: median {bps.median():.0f}, IQR "
             f"[{bps.quantile(.25):.0f}, {bps.quantile(.75):.0f}], range [{bps.min()}, {bps.max()}]")
    L.append("\n| Class | beats | groups | beats/group median [min, max] |\n|---|---|---|---|")
    for c in CLASS_NAMES:
        b = bps.xs(c, level="pulse_type_en")
        L.append(f"| {c} | {int(b.sum())} | {len(b)} | {b.median():.0f} [{b.min()}, {b.max()}] |")
    # per-fold subject / class counts (splits are deterministic: seed = repeat)
    rows = []
    for rep in range(10):
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
        for k, (_, te) in enumerate(skf.split(np.zeros(len(ds)), y_all, subj)):
            r = dict(repeat=rep, fold=k, test_beats=len(te), test_subjects=len(np.unique(subj[te])))
            for ci, c in enumerate(CLASS_NAMES):
                m = y_all[te] == ci
                r[f"{c}_beats"] = int(m.sum()); r[f"{c}_subjects"] = len(np.unique(subj[te][m]))
            rows.append(r)
    folds = pd.DataFrame(rows); folds.to_csv(OUT / "per_fold_counts.csv", index=False)
    L.append("\nPer-test-fold counts over the 50 folds (full table: `per_fold_counts.csv`):\n")
    L.append("| | beats | subjects | Tight beats | **Tight subjects** | Slippery subj. | Wiry subj. |\n|---|---|---|---|---|---|---|")
    for stat in ["min", "median", "max"]:
        f = getattr(folds, stat)(numeric_only=True)
        L.append(f"| {stat} | {f.test_beats:.0f} | {f.test_subjects:.0f} | {f.Tight_beats:.0f} | "
                 f"**{f.Tight_subjects:.0f}** | {f.Slippery_subjects:.0f} | {f.Wiry_subjects:.0f} |")

    # ── A/B/C/E per model ────────────────────────────────────────────────────
    summary = []
    cm_beat_all, cm_subj_all = {}, {}
    for name in sorted(store, key=lambda n: -np.mean([macro_f1(v[1], v[2].argmax(1)) for r in store[n].values() for v in r.values()])):
        reps = store[name]
        per_run = defaultdict(list)          # metric -> list over 50 runs
        oof_ci_m, oof_ci_t = [], []
        cm_b = np.zeros((3, 3), int); cm_s = np.zeros((3, 3), int)
        for rep, folds_d in sorted(reps.items()):
            oof_idx, oof_y, oof_p = [], [], []
            for k, (ti, yt, pr) in sorted(folds_d.items()):
                yhat = pr.argmax(1)
                g = grp_all[ti]
                per_run["beat_macro_f1"].append(macro_f1(yt, yhat))
                per_run["beat_tight_f1"].append(tight_f1(yt, yhat))
                w = group_weights(g)
                per_run["subjw_macro_f1"].append(macro_f1(yt, yhat, w))
                per_run["subjw_tight_f1"].append(tight_f1(yt, yhat, w))
                ys, ps = subject_level(yt, pr, g)
                per_run["subj_macro_f1"].append(macro_f1(ys, ps))
                per_run["subj_tight_f1"].append(tight_f1(ys, ps))
                cm_b += confusion_matrix(yt, yhat, labels=[0, 1, 2])
                cm_s += confusion_matrix(ys, ps, labels=[0, 1, 2])
                oof_idx.append(ti); oof_y.append(yt); oof_p.append(pr)
            # full OOF set for this repeat -> clustered bootstrap
            oi = np.concatenate(oof_idx); oy = np.concatenate(oof_y); op = np.concatenate(oof_p)
            if len(oi) == len(ds):                    # complete repeat only
                cm_ci, tf_ci = cluster_boot(oy, op.argmax(1), subj[oi])
                oof_ci_m.append(cm_ci); oof_ci_t.append(tf_ci)
        n_runs = len(per_run["beat_macro_f1"])
        s = {k: (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else np.nan) for k, v in per_run.items()}
        row = dict(model=name, n_runs=n_runs, **{f"{k}_mean": v[0] for k, v in s.items()},
                   **{f"{k}_sd": v[1] for k, v in s.items()})
        if oof_ci_m:
            cm_arr = np.array(oof_ci_m); tf_arr = np.array(oof_ci_t)
            row.update(clu_ci_macro_lo=cm_arr[:, 0].mean(), clu_ci_macro_hi=cm_arr[:, 1].mean(),
                       clu_ci_tight_lo=tf_arr[:, 0].mean(), clu_ci_tight_hi=tf_arr[:, 1].mean(),
                       n_complete_repeats=len(oof_ci_m))
        summary.append(row); cm_beat_all[name] = cm_b; cm_subj_all[name] = cm_s

    summ = pd.DataFrame(summary); summ.to_csv(OUT / "subject_level_summary.csv", index=False)

    L.append("\n## A. Beat-level vs subject-weighted vs subject-level macro-F1 (mean ± SD over runs)\n")
    L.append("| Model | runs | beat-level | subject-weighted | subject-level (mean-prob) | Tight-F1 beat | Tight-F1 subj-level |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in summ.iterrows():
        L.append(f"| {DISPLAY.get(r.model, r.model)} | {r.n_runs} | {r.beat_macro_f1_mean:.3f} ± {r.beat_macro_f1_sd:.3f} | "
                 f"{r.subjw_macro_f1_mean:.3f} ± {r.subjw_macro_f1_sd:.3f} | {r.subj_macro_f1_mean:.3f} ± {r.subj_macro_f1_sd:.3f} | "
                 f"{r.beat_tight_f1_mean:.2f} | {r.subj_tight_f1_mean:.2f} |")

    if "clu_ci_macro_lo" in summ:
        L.append("\n## B. Subject-clustered bootstrap 95% CI (per-repeat OOF set, resampling subjects; averaged over repeats)\n")
        L.append("| Model | complete repeats | macro-F1 CI | Tight-F1 CI |\n|---|---|---|---|")
        for _, r in summ.dropna(subset=["clu_ci_macro_lo"]).iterrows():
            L.append(f"| {DISPLAY.get(r.model, r.model)} | {int(r.n_complete_repeats)} | "
                     f"[{r.clu_ci_macro_lo:.3f}, {r.clu_ci_macro_hi:.3f}] | [{r.clu_ci_tight_lo:.3f}, {r.clu_ci_tight_hi:.3f}] |")

    L.append("\n## C. Confusion matrices aggregated over all runs (rows = true, cols = predicted; per-class P / R / F1)\n")
    for name in summ.model:
        for tag, cm in [("beat-level", cm_beat_all[name]), ("subject-level (groups)", cm_subj_all[name])]:
            p, rc, f, _ = precision_recall_fscore_support(
                np.repeat(np.arange(3), cm.sum(1)), np.concatenate([np.repeat(np.arange(3), cm[i]) for i in range(3)]),
                labels=[0, 1, 2], zero_division=0)
            L.append(f"\n**{DISPLAY.get(name, name)} — {tag}** (n = {cm.sum()})\n")
            L.append("| true \\ pred | " + " | ".join(CLASS_NAMES) + " | recall |\n|---|---|---|---|---|")
            for i, c in enumerate(CLASS_NAMES):
                L.append(f"| {c} | " + " | ".join(f"{cm[i, j]} ({cm[i, j] / max(1, cm[i].sum()) * 100:.1f}%)" for j in range(3)) + f" | {rc[i]:.3f} |")
            L.append("| precision | " + " | ".join(f"{p[j]:.3f}" for j in range(3)) + " | |")
            L.append("| F1 | " + " | ".join(f"{f[j]:.3f}" for j in range(3)) + " | |")

    L.append("\n## E. Tight-class F1 instability (beat-level, over runs)\n")
    L.append("| Model | mean | SD | min | max | runs with Tight-F1 = 0 |\n|---|---|---|---|---|---|")
    for name in summ.model:
        v = np.array([tight_f1(yt, pr.argmax(1)) for r in store[name].values() for (_, yt, pr) in r.values()])
        L.append(f"| {DISPLAY.get(name, name)} | {v.mean():.3f} | {v.std(ddof=1):.3f} | {v.min():.2f} | {v.max():.2f} | {(v == 0).sum()} |")

    rep = OUT / f"subject_level_report_{args.protocol}.md"
    rep.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)); print(f"\nwritten: {rep}")


if __name__ == "__main__":
    main()
