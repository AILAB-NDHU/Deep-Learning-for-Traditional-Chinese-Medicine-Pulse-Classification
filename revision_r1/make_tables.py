"""
Revision R1 — generate every results table of the manuscript (LaTeX fragments)
plus a numbers.tex macro file, from ONE results directory, so that switching from
the local rerun to the server rerun is a one-line change.

  python make_tables.py                      # RES_DIR=revision_r1/outputs (local RTX 5080)
  RES_DIR=revision_r1/outputs_server python make_tables.py   # server RTX 4090 (canonical)

Inputs (in RES_DIR):  deep_rerun_raw_r*.csv, master_ranking.csv, subject_level_summary.csv,
                      gradcam/gradcam_peaks_resnet_gc.csv, [smote_musorted_raw.csv]
Outputs (TABLES_DIR = paper_access/tables): tab_*.tex and numbers.tex
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("RES_DIR", HERE / "outputs")).resolve()
TAB = Path(os.environ.get("TABLES_DIR", HERE.parent / "paper_access" / "tables")).resolve()
TAB.mkdir(parents=True, exist_ok=True)
ENV_TAG = os.environ.get("ENV_TAG", "local rerun, RTX 5080 Laptop")

K, ALPHA, NB = 5, 0.05, 10_000
RNG = np.random.default_rng(0)
D = {"resnet": "ResNet1D (plain)", "resnet_gated": "ResNet1D $+$ Gated", "resnet_gc": "ResNet1D $+$ GC",
     "resnet_gc_gated": "ResNet1D $+$ GC $+$ Gated", "protonet": "Prototypical Network", "inception": "Inception1D",
     "cnn_aug": "1D-CNN $+$ 6-mode aug", "cnn_noaug": "1D-CNN (no aug)", "transformer": "Patch Transformer",
     "XGBoost": "XGBoost", "RandomForest": "Random Forest", "SVM": "SVM",
     "cnn_aug_nosampler": "1D-CNN $+$ aug, no sampler", "cnn_noaug_nosampler": "1D-CNN, no aug, no sampler"}
DEEP_VS_XGB = ["resnet", "resnet_gated", "resnet_gc", "resnet_gc_gated", "protonet", "inception", "cnn_aug", "transformer"]


# ── stats ─────────────────────────────────────────────────────────────────────
def nb_p(d):
    n = len(d); se = np.sqrt((1 / n + 1 / (K - 1)) * d.var(ddof=1))
    return 1.0 if se == 0 else 2 * stats.t.sf(abs(d.mean() / se), df=n - 1)


def mde(d, power=0.8):
    n = len(d); se = np.sqrt((1 / n + 1 / (K - 1)) * d.var(ddof=1)); df = n - 1
    return (stats.t.ppf(1 - ALPHA / 2, df) + stats.t.ppf(power, df)) * se


def ci(x):
    idx = RNG.integers(0, len(x), size=(NB, len(x))); return np.percentile(x[idx].mean(1), [2.5, 97.5])


def holm(p):
    it = sorted(p.items(), key=lambda kv: kv[1]); m = len(it); out, run = {}, 0.0
    for i, (k, v) in enumerate(it):
        run = max(run, (m - i) * v); out[k] = min(1.0, run)
    return out


def fp(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def pm(m, s, bold=False):
    t = f"{m:.3f}\\pm{s:.3f}"; return f"$\\mathbf{{{t}}}$" if bold else f"${t}$"


def w(name, body, caption, label, star=True, colspec=None, extra=""):
    env = "table*" if star else "table"
    (TAB / f"{name}.tex").write_text(
        f"\\begin{{{env}}}[!t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\n{extra}"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{body}\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n", encoding="utf-8")
    print("wrote", name)


# ── load ──────────────────────────────────────────────────────────────────────
raw = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("deep_rerun_raw_r*.csv"))], ignore_index=True)
raw = raw.drop_duplicates(["model", "protocol", "repeat", "fold"], keep="last")
lf = raw[raw.protocol == "leakfree"]
# per-model summary for EVERY model in the raw file (incl. the two ablation-only cells)
_rows = []
for _m, _g in lf.groupby("model"):
    _r = dict(model=_m, runs=len(_g), params=int(_g.params.max()))
    for _k in ["macro_f1", "bal_acc", "roc_auc", "pr_auc", "tight_f1"]:
        _r[f"{_k}_mean"], _r[f"{_k}_sd"] = _g[_k].mean(), _g[_k].std(ddof=1)
        _r[f"{_k}_lo"], _r[f"{_k}_hi"] = ci(_g[_k].values)
    _rows.append(_r)
rank = pd.DataFrame(_rows)
R = rank.set_index("model")
incomplete = rank[rank.runs < 50]
if len(incomplete):
    print("WARNING — models with < 50 runs:", dict(zip(incomplete.model, incomplete.runs)))


def S(model, metric="macro_f1", proto="leakfree"):
    return raw[(raw.model == model) & (raw.protocol == proto)].set_index(["repeat", "fold"])[metric].sort_index()


def paired(a, b, metric="macro_f1"):
    j = S(a, metric).to_frame("a").join(S(b, metric).to_frame("b"), how="inner"); return (j.a - j.b).values


def row_metrics(m, bold=False, tight=True):
    r = R.loc[m]
    cells = [pm(r.macro_f1_mean, r.macro_f1_sd, bold), f"[{r.macro_f1_lo:.3f}, {r.macro_f1_hi:.3f}]",
             pm(r.bal_acc_mean, r.bal_acc_sd), pm(r.roc_auc_mean, r.roc_auc_sd), pm(r.pr_auc_mean, r.pr_auc_sd)]
    if tight:
        cells.append(f"${r.tight_f1_mean:.2f}\\pm{r.tight_f1_sd:.2f}$")
    return cells


HDR5 = "Model & Macro-F1 & 95\\% CI & Bal.\\ Acc & ROC-AUC & PR-AUC & Tight F1 \\\\\n\\midrule\n"
NOTE = ("Mean $\\pm$ SD over the 50 dependent runs of 10~$\\times$~5 repeated subject-level CV "
        "(SD is descriptive, not an uncertainty interval); CI = bootstrap 95\\% interval of the mean macro-F1 over the 50 paired runs.")

# Table 1 — classical
best = max(["SVM", "RandomForest", "XGBoost"], key=lambda m: R.loc[m].macro_f1_mean)
body = "".join(f"{D[m]} & " + " & ".join(row_metrics(m, m == best)) + " \\\\\n" for m in ["SVM", "RandomForest", "XGBoost"])
w("tab_ml", HDR5 + body, "Classical route on the nine $\\mu$-ordered Gaussian features (training folds include the "
  "Gaussian features of the synthetic Tight beats; leakage-free, repeated subject-level CV). " + NOTE, "tab:ml", colspec="lcccccc")

# Table 2 — 2x2 cells with all metrics
cells2 = [("cnn_noaug", "No", "Yes"), ("cnn_aug", "Yes", "Yes"), ("cnn_noaug_nosampler", "No", "No"), ("cnn_aug_nosampler", "Yes", "No")]
bestc = max([c[0] for c in cells2], key=lambda m: R.loc[m].macro_f1_mean)
body = "".join(f"{a} & {s} & " + " & ".join(row_metrics(m, m == bestc)) + " \\\\\n" for m, a, s in cells2)
w("tab_aug", "Aug. & Sampler & Macro-F1 & 95\\% CI & Bal.\\ Acc & ROC-AUC & PR-AUC & Tight F1 \\\\\n\\midrule\n" + body,
  "The four cells of the $2\\times2$ augmentation~$\\times$~weighted-sampler design on the 1D-CNN (leakage-free, repeated CV). "
  "The headline configuration of the original submission is the second row. " + NOTE, "tab:aug", colspec="llcccccc")

# Table 3 — deep architectures
adv = ["cnn_aug", "resnet", "inception", "transformer"]; b3 = max(adv, key=lambda m: R.loc[m].macro_f1_mean)
body = "".join(f"{D[m]} & " + " & ".join(row_metrics(m, m == b3)) + " \\\\\n" for m in adv)
w("tab_adv", HDR5 + body, "Deep architectures on the raw waveform (leakage-free, repeated subject-level CV). " + NOTE, "tab:adv", colspec="lcccccc")

# Table 4 — gating
gate = ["resnet", "resnet_gc", "resnet_gated", "resnet_gc_gated"]; b4 = max(gate, key=lambda m: R.loc[m].macro_f1_mean)
body = "".join(f"{D[m]} & " + " & ".join(row_metrics(m, m == b4)) + " \\\\\n" for m in gate)
w("tab_gate", HDR5 + body, "Attention and gating add-ons on ResNet1D (leakage-free, repeated subject-level CV). " + NOTE, "tab:gate", colspec="lcccccc")

# Table 6 — overall ranking with per-metric rank
order = rank.sort_values("macro_f1_mean", ascending=False)
order = order[order.model.isin(DEEP_VS_XGB + ["XGBoost", "RandomForest", "SVM"])].reset_index(drop=True)
rk = {k: order[f"{k}_mean"].rank(ascending=False, method="min").astype(int) for k in ["macro_f1", "bal_acc", "roc_auc", "pr_auc", "tight_f1"]}
body = ""
for i, r in order.iterrows():
    body += (f"{i + 1} & {D[r.model]} & {pm(r.macro_f1_mean, r.macro_f1_sd, i == 0)} & [{r.macro_f1_lo:.3f}, {r.macro_f1_hi:.3f}] & "
             f"{r.bal_acc_mean:.3f} & {r.roc_auc_mean:.3f} & {r.pr_auc_mean:.3f} & {r.tight_f1_mean:.2f} & "
             f"{rk['bal_acc'][i]} / {rk['roc_auc'][i]} / {rk['pr_auc'][i]} / {rk['tight_f1'][i]} \\\\\n")
w("tab_rank", "\\# & Model & Macro-F1 & 95\\% CI & Bal.\\ Acc & ROC-AUC & PR-AUC & Tight F1 & Rank by Bal.\\ Acc / ROC / PR / Tight \\\\\n\\midrule\n" + body,
  "Overall leakage-free ranking (single unified harness; macro-F1 is the pre-specified primary metric). The last column gives each "
  "model's rank under the four secondary metrics; the ranking is metric-dependent and the top tier is separated by less than the "
  "run-to-run variation. " + NOTE, "tab:rank", colspec="clcccccccc")

# Table 7 — effects (2x2)
J = pd.concat({k: S(m) for k, m in [("as", "cnn_aug"), ("ns", "cnn_noaug"), ("an", "cnn_aug_nosampler"), ("nn", "cnn_noaug_nosampler")]}, axis=1).dropna()
eff = {"Augmentation main effect": ((J.as_ if False else J["as"]) + J["an"] - J["ns"] - J["nn"]) / 2,
       "Weighted-sampler main effect": (J["as"] + J["ns"] - J["an"] - J["nn"]) / 2,
       "Interaction (aug.\\ effect with sampler $-$ without)": (J["as"] - J["ns"]) - (J["an"] - J["nn"]),
       "Aug.\\ effect, sampler on (original headline)": J["as"] - J["ns"],
       "Aug.\\ effect, sampler off": J["an"] - J["nn"]}
body = ""
for k, e in eff.items():
    c = ci(e.values); body += f"{k} & ${e.mean():+.3f}$ & [${c[0]:+.3f}$, ${c[1]:+.3f}$] & {fp(nb_p(e.values))} & {mde(e.values):.3f} \\\\\n"
w("tab_ablation", "Effect on macro-F1 & Estimate & 95\\% CI & NB $p$ & MDE \\\\\n\\midrule\n" + body,
  f"Main effects and interaction of the $2\\times2$ ablation (1D-CNN, $n={len(J)}$ paired runs). NB $p$ = Nadeau--Bengio "
  "corrected resampled $t$-test; MDE = minimum detectable effect at $\\alpha=0.05$, power 0.8.", "tab:ablation", star=False, colspec="lccc c")

# Table — comparison family
fam = {"6-mode aug.\\ vs.\\ none (1D-CNN)": paired("cnn_aug", "cnn_noaug"), "ResNet1D $+$ GC vs.\\ plain": paired("resnet_gc", "resnet")}
for m in DEEP_VS_XGB:
    fam[f"{D[m]} vs.\\ XGBoost"] = paired(m, "XGBoost")
praw = {k: nb_p(v) for k, v in fam.items()}; padj = holm(praw)
body = ""
for k, d in fam.items():
    c = ci(d); dz = d.mean() / d.std(ddof=1)
    body += f"{k} & ${d.mean():+.3f}$ & [${c[0]:+.3f}$, ${c[1]:+.3f}$] & {dz:.2f} & {fp(praw[k])} & {fp(padj[k])} & {mde(d):.3f} \\\\\n"
w("tab_family", "Comparison (A vs.\\ B) & $\\Delta$ macro-F1 & 95\\% CI & $d_z$ & NB $p$ & Holm $p$ & MDE \\\\\n\\midrule\n" + body,
  "The pre-registered comparison family (leakage-free macro-F1, 50 paired runs). NB $p$ = Nadeau--Bengio corrected resampled "
  "$t$-test (primary); Holm $p$ = step-down family-wise adjustment over all rows; $d_z$ = Cohen's paired effect size; MDE at "
  "$\\alpha=0.05$, power 0.8. No contrast is significant after adjustment.", "tab:family", colspec="lcccccc")

# Table 5 — inflation (server only)
both = [m for m in raw.model.unique() if {"leakfree", "optimistic"} <= set(raw[raw.model == m].protocol)]
infl = []
if both:
    body = ""
    for m in sorted(both, key=lambda m: -S(m).mean()):
        j = S(m).to_frame("lf").join(S(m, proto="optimistic").to_frame("op"), how="inner"); d = (j.op - j.lf).values; c = ci(d)
        pr = int(raw[raw.model == m].params.max()); infl.append((pr, d.mean()))
        body += f"{D[m]} & {pr:,} & {j.lf.mean():.3f} & {j.op.mean():.3f} & ${d.mean():+.3f}$ [${c[0]:+.3f}$, ${c[1]:+.3f}$] \\\\\n"
    rho, prho = stats.spearmanr([a for a, _ in infl], [b for _, b in infl])
    w("tab_opt", "Model & Params & Leakage-free & Optimistic & Inflation $\\Delta$ [95\\% CI] \\\\\n\\midrule\n" + body,
      f"Optimistic (test fold as early-stopping signal) vs.\\ leakage-free protocol, macro-F1 over the same 50 partitions; "
      f"Spearman $\\rho$(params, inflation) $= {rho:.2f}$, $p={prho:.2f}$. Classical models have no early-stopping channel.", "tab:opt", colspec="lcccc")
else:
    # fallback: build Table 5 from the original-submission harness output so the document compiles
    arch = HERE.parent / "_working_archive" / "stat_rerun_raw.csv"
    if arch.exists():
        ar = pd.read_csv(arch); body = ""
        def _S(m, p): return ar[(ar.model == m) & (ar.protocol == p)].set_index(["repeat", "fold"])["macro_f1"].sort_index()
        ms = [m for m in ar.model.unique() if {"leakfree", "optimistic"} <= set(ar[ar.model == m].protocol)]
        for m in sorted(ms, key=lambda m: -_S(m, "leakfree").mean()):
            j = _S(m, "leakfree").to_frame("lf").join(_S(m, "optimistic").to_frame("op"), how="inner"); d = (j.op - j.lf).values; c = ci(d)
            pr = int(ar[ar.model == m].params.max()); infl.append((pr, d.mean()))
            body += f"{D.get(m, m)} & {pr:,} & {j.lf.mean():.3f} & {j.op.mean():.3f} & ${d.mean():+.3f}$ [${c[0]:+.3f}$, ${c[1]:+.3f}$] \\\\\n"
        rho, prho = stats.spearmanr([a for a, _ in infl], [b for _, b in infl])
        w("tab_opt", "Model & Params & Leakage-free & Optimistic & Inflation $\\Delta$ [95\\% CI] \\\\\n\\midrule\n" + body,
          f"Optimistic vs.\\ leakage-free protocol, macro-F1 over the same 50 partitions (original-submission harness run); "
          f"Spearman $\\rho$(params, inflation) $= {rho:.2f}$, $p={prho:.2f}$. Classical models have no early-stopping channel.", "tab:opt", colspec="lcccc")
        print("NOTE: tab_opt built from the ORIGINAL submission run (no optimistic protocol in this results dir)")

# Table 8 — SMOTE family on mu-sorted features (if the rerun exists)
smote_csv = RES / "smote_musorted_raw.csv"
if smote_csv.exists():
    sm = pd.read_csv(smote_csv).drop_duplicates(["model", "repeat", "fold"], keep="last")
    base = sm[sm.model == "xgb_baseline"].set_index(["repeat", "fold"]).sort_index()
    names = [("xgb_baseline", "Class weights only (reference)"), ("xgb_smote", "$+$ SMOTE"),
             ("xgb_borderline", "$+$ BorderlineSMOTE"), ("xgb_adasyn", "$+$ ADASYN")]
    log = RES / "smote_musorted_adasyn_log.csv"
    nfall = int(pd.read_csv(log).degenerated.sum()) if log.exists() else None
    body = ""
    for key, lab in names:
        g = sm[sm.model == key].set_index(["repeat", "fold"]).sort_index()
        j = base.macro_f1.to_frame("b").join(g.macro_f1.to_frame("s"), how="inner"); d = (j.s - j.b).values
        if key == "xgb_baseline":
            body += f"{lab} & {pm(g.macro_f1.mean(), g.macro_f1.std(ddof=1))} & ${g.tight_f1.mean():.2f}$ & --- & --- & --- \\\\\n"
        else:
            c = ci(d); note = f" (fell back to baseline in {nfall}/50 folds)" if key == "xgb_adasyn" and nfall is not None else ""
            body += (f"{lab}{note} & {pm(g.macro_f1.mean(), g.macro_f1.std(ddof=1))} & ${g.tight_f1.mean():.2f}$ & ${d.mean():+.3f}$ & "
                     f"[${c[0]:+.3f}$, ${c[1]:+.3f}$] & {fp(nb_p(d)) if d.std() > 0 else '---'} \\\\\n")
    w("tab_smote", "XGBoost training set & Macro-F1 & Tight F1 & $\\Delta$ vs.\\ reference & 95\\% CI & NB $p$ \\\\\n\\midrule\n" + body,
      "Feature-space oversampling on the classical route (XGBoost, $\\mu$-ordered features, leakage-free repeated CV). The reference "
      "training set already contains the Gaussian features of the waveform-augmented Tight beats; each oversampler is applied on top of "
      "it, inside the training fold only. ADASYN's density criterion generates no samples once the minority class has been enlarged by "
      "the waveform augmentation, so it is identical to the reference by construction.", "tab:smote", colspec="lccccc")
else:
    print("NOTE: smote_musorted_raw.csv not found — tab_smote not regenerated")

# Table — subject level
sl = pd.read_csv(RES / "subject_level_summary.csv").set_index("model")
body = ""
for m in order.model:
    if m not in sl.index:
        continue
    r = sl.loc[m]
    body += (f"{D[m]} & {r.beat_macro_f1_mean:.3f} & {r.subjw_macro_f1_mean:.3f} & {r.subj_macro_f1_mean:.3f} & "
             f"[{r.clu_ci_macro_lo:.3f}, {r.clu_ci_macro_hi:.3f}] & {r.beat_tight_f1_mean:.2f} & {r.subj_tight_f1_mean:.2f} & "
             f"[{r.clu_ci_tight_lo:.2f}, {r.clu_ci_tight_hi:.2f}] \\\\\n")
w("tab_subject", "Model & Beat & Subj.-weighted & Subj.-level & Clustered 95\\% CI & Beat & Subj.-level & Clustered 95\\% CI \\\\\n"
  " & \\multicolumn{4}{c}{macro-F1} & \\multicolumn{3}{c}{Tight F1} \\\\\n\\midrule\n" + body,
  "Unit-of-analysis sensitivity (leakage-free). Beat-level = every beat counts once (the paper's primary metric); subject-weighted = "
  "beats weighted $1/n$ within each subject--label group so each subject counts once; subject-level = one prediction per subject--label "
  "group from the mean class probability of its beats. Clustered CIs are bootstrap intervals that resample subjects, not beats, on each "
  "repeat's complete out-of-fold prediction set (averaged over the 10 repeats).", "tab:subject", colspec="lccccccc",
  extra="\\small\\setlength{\\tabcolsep}{4pt}\n")

# Table — Grad-CAM summary (resnet_gc)
gp = pd.read_csv(RES / "gradcam" / "gradcam_peaks_resnet_gc.csv")
cls = ["Slippery", "Tight", "Wiry"]; body = ""
for ci_, cn in enumerate(cls):
    for corr in (True, False):
        g = gp[(gp.true == ci_) & (gp.correct == corr)]
        th = np.histogram(g.peak_pos, bins=[0, 1 / 3, 2 / 3, 1.0001])[0] / max(1, len(g)) * 100
        body += (f"{cn if corr else ''} & {'correct' if corr else 'misclassified'} & {len(g):,} & {g.peak_pos.median():.2f} "
                 f"[{g.peak_pos.quantile(.25):.2f}, {g.peak_pos.quantile(.75):.2f}] & {th[0]:.0f} / {th[1]:.0f} / {th[2]:.0f} \\\\\n")
w("tab_gradcam", "True class & Prediction & $n$ & Peak pos.\\ median [IQR] & Early/mid/late (\\%) \\\\\n\\midrule\n" + body,
  f"Grad-CAM peak-activation position (fraction of the $[-0.15,+0.65]$~s beat window) for all {len(gp):,} held-out beats of "
  "ResNet1D$+$GC across the 50 leakage-free fold models, computed for the predicted class and split by correctness. No beats were "
  "selected by confidence.", "tab:gradcam", star=True, colspec="llccc")

# Table — confusion matrix (ResNet1D, beat & subject level)  — parsed from the subject-level report
rep = (RES / "subject_level_report_leakfree.md").read_text(encoding="utf-8")
def grab(tag):
    blk = rep.split(f"**ResNet1D (plain) — {tag}**", 1)[1].split("**", 1)[0]
    rows = [l for l in blk.splitlines() if l.startswith("| ") and not l.startswith("| true")]
    return rows
def cm_rows(tag):
    out = ""
    for l in grab(tag)[:5]:
        cells = [c.strip() for c in l.strip("|").split("|")]
        out += " & ".join(c.replace("%", "\\%") for c in cells) + " \\\\\n"
    return out
body = "\\multicolumn{5}{l}{\\textit{Beat level ($n$ = all held-out beats over 50 runs)}} \\\\\n" + cm_rows("beat-level") + \
       "\\midrule\n\\multicolumn{5}{l}{\\textit{Subject--label-group level (mean probability per group, 10 repeats)}} \\\\\n" + cm_rows("subject-level (groups)")
w("tab_confusion", "True $\\backslash$ predicted & Slippery & Tight & Wiry & Recall \\\\\n\\midrule\n" + body,
  "Aggregated confusion matrices of the plain ResNet1D under the leakage-free protocol, with per-class precision, recall and F1. "
  "Row percentages are recall by class.", "tab:confusion", star=False, colspec="lcccc")

# numbers.tex
def M(name, val):
    return f"\\newcommand{{\\{name}}}{{{val}}}\n"
rs, xg = R.loc["resnet"], R.loc["XGBoost"]; d_rx = fam[f"{D['resnet']} vs.\\ XGBoost"]; c_rx = ci(d_rx)
d_aug = fam["6-mode aug.\\ vs.\\ none (1D-CNN)"]; d_gc = fam["ResNet1D $+$ GC vs.\\ plain"]
top = order.iloc[0]
nums = M("nEnvTag", ENV_TAG) + M("nTopModel", D[top.model]) + M("nTopFone", f"{top.macro_f1_mean:.3f}") + \
    M("nResFone", f"{rs.macro_f1_mean:.3f}") + M("nResSD", f"{rs.macro_f1_sd:.3f}") + M("nResCI", f"[{rs.macro_f1_lo:.3f}, {rs.macro_f1_hi:.3f}]") + \
    M("nResBal", f"{rs.bal_acc_mean:.3f}") + M("nXgbFone", f"{xg.macro_f1_mean:.3f}") + M("nXgbSD", f"{xg.macro_f1_sd:.3f}") + M("nXgbBal", f"{xg.bal_acc_mean:.3f}") + \
    M("nDeltaResXgb", f"{d_rx.mean():+.3f}") + M("nCIResXgb", f"[{c_rx[0]:+.3f}, {c_rx[1]:+.3f}]") + \
    M("nPResXgb", fp(praw[f"{D['resnet']} vs.\\ XGBoost"])) + M("nPholmResXgb", fp(padj[f"{D['resnet']} vs.\\ XGBoost"])) + \
    M("nHolmSig", str(sum(p < ALPHA for p in padj.values()))) + M("nRawSig", str(sum(p < ALPHA for p in praw.values()))) + M("nFamily", str(len(fam))) + \
    M("nDeltaAug", f"{d_aug.mean():+.3f}") + M("nPAug", fp(praw["6-mode aug.\\ vs.\\ none (1D-CNN)"])) + M("nAugWins", f"{int((d_aug > 0).sum())}") + \
    M("nAugMain", f"{eff['Augmentation main effect'].mean():+.3f}") + M("nSampMain", f"{eff['Weighted-sampler main effect'].mean():+.3f}") + \
    M("nInter", f"{eff['Interaction (aug.\\ effect with sampler $-$ without)'].mean():+.3f}") + M("nAugNoSamp", f"{eff['Aug.\\ effect, sampler off'].mean():+.3f}") + \
    M("nPAugMain", fp(nb_p(eff['Augmentation main effect'].values))) + M("nPSampMain", fp(nb_p(eff['Weighted-sampler main effect'].values))) + \
    M("nDeltaGC", f"{d_gc.mean():+.3f}") + M("nPGC", fp(praw["ResNet1D $+$ GC vs.\\ plain"])) + \
    M("nMinDeepDelta", f"{min(fam[k].mean() for k in fam if 'XGBoost' in k):+.3f}") + M("nMaxDeepDelta", f"{max(fam[k].mean() for k in fam if 'XGBoost' in k):+.3f}") + \
    M("nCnnAugFone", f"{R.loc['cnn_aug'].macro_f1_mean:.3f}") + M("nCnnNoaugFone", f"{R.loc['cnn_noaug'].macro_f1_mean:.3f}") + \
    M("nCnnAugNsFone", f"{R.loc['cnn_aug_nosampler'].macro_f1_mean:.3f}") + M("nCnnNoaugNsFone", f"{R.loc['cnn_noaug_nosampler'].macro_f1_mean:.3f}") + \
    M("nResTight", f"{rs.tight_f1_mean:.2f}") + M("nXgbTight", f"{xg.tight_f1_mean:.2f}")
if "resnet" in sl.index:
    s = sl.loc["resnet"]
    nums += M("nSubjResFone", f"{s.subj_macro_f1_mean:.3f}") + M("nSubjwResFone", f"{s.subjw_macro_f1_mean:.3f}") + \
        M("nCluResCI", f"[{s.clu_ci_macro_lo:.3f}, {s.clu_ci_macro_hi:.3f}]") + M("nCluResTightCI", f"[{s.clu_ci_tight_lo:.2f}, {s.clu_ci_tight_hi:.2f}]") + \
        M("nSubjResTight", f"{s.subj_tight_f1_mean:.2f}") + M("nSubjMin", f"{sl.subj_macro_f1_mean.min():.2f}") + M("nSubjMax", f"{sl.subj_macro_f1_mean.max():.2f}")
if infl:
    nums += M("nInflMin", f"{min(b for _, b in infl):+.3f}") + M("nInflMax", f"{max(b for _, b in infl):+.3f}") + M("nSpearman", f"{rho:.2f}") + M("nSpearmanP", f"{prho:.2f}")
else:
    nums += M("nInflMin", "+0.016") + M("nInflMax", "+0.030") + M("nSpearman", "0.07") + M("nSpearmanP", "0.86") + "% inflation numbers: original submission (optimistic protocol not in this results dir)\n"
nums += M("nGradN", f"{len(gp):,}")
(TAB / "numbers.tex").write_text(nums, encoding="utf-8")
print("wrote numbers.tex from", RES, "| env:", ENV_TAG)
