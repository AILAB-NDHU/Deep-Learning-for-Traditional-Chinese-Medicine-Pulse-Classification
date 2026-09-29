"""
Revision R1 — master statistics from the local rerun (outputs/deep_rerun_raw_r*.csv).

Produces every number the revised tables need, in one place, from ONE consistent
environment (the local RTX-5080 rerun), so that Tables 1, 3, 4, 5, 6, 7, 8 and the
Abstract share the same runs:

  1. Master ranking (Table 6 + R1-M2): all four metrics + Tight-F1, mean, SD,
     bootstrap 95% CI over the 50 paired runs, per model (leak-free)
  2. Pre-registered comparison family: Δ, 95% CI, Cohen d_z, Nadeau–Bengio p,
     Holm-adjusted p, MDE  (R1-M3, R2-M2, R2-m6, R2-m8)
  3. 2×2 augmentation × sampler ablation with BOTH main effects AND the
     interaction, each with NB p (R1-M1a)
  4. Optimistic vs leak-free inflation with bootstrap CI + Spearman vs params
     (Table 5) — only for models that have both protocols in the rerun
  5. Metric-wise rank table: where does XGBoost stand on each metric? (R1-M2)

Writes outputs/master_report.md and outputs/master_ranking.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
K_FOLDS, ALPHA, N_BOOT = 5, 0.05, 10_000
RNG = np.random.default_rng(0)
METRICS = ["macro_f1", "bal_acc", "roc_auc", "pr_auc", "tight_f1"]
DISPLAY = {"resnet": "ResNet1D (plain)", "resnet_gated": "ResNet1D + Gated",
           "resnet_gc": "ResNet1D + GC", "resnet_gc_gated": "ResNet1D + GC + Gated",
           "protonet": "Prototypical Network", "inception": "Inception1D",
           "cnn_aug": "1D-CNN + 6-mode aug", "cnn_noaug": "1D-CNN (no aug)",
           "cnn_aug_nosampler": "1D-CNN + aug, no sampler", "cnn_noaug_nosampler": "1D-CNN no aug, no sampler",
           "transformer": "Patch Transformer", "XGBoost": "XGBoost", "RandomForest": "Random Forest", "SVM": "SVM"}
RANK_MODELS = ["resnet", "resnet_gated", "resnet_gc", "resnet_gc_gated", "protonet", "inception",
               "cnn_aug", "transformer", "XGBoost", "RandomForest", "SVM", "cnn_noaug"]
DEEP_VS_XGB = ["resnet", "resnet_gated", "resnet_gc", "resnet_gc_gated", "protonet", "inception", "cnn_aug", "transformer"]


# ── statistics (identical formulae to _working_archive/stat_rerun_analysis.py) ──
def nb_ttest(d):
    n = len(d); var_d = d.var(ddof=1); rho = 1.0 / (K_FOLDS - 1)
    se = np.sqrt((1.0 / n + rho) * var_d)
    if se == 0:
        return np.nan, 1.0, se
    t = d.mean() / se
    return t, 2 * stats.t.sf(abs(t), df=n - 1), se


def mde(d, power=0.8):
    n = len(d); var_d = d.var(ddof=1); rho = 1.0 / (K_FOLDS - 1)
    se = np.sqrt((1.0 / n + rho) * var_d); df = n - 1
    return (stats.t.ppf(1 - ALPHA / 2, df) + stats.t.ppf(power, df)) * se


def boot_ci(x, stat=np.mean):
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    return np.percentile(stat(x[idx], axis=1), [2.5, 97.5])


def holm(p: dict):
    items = sorted(p.items(), key=lambda kv: kv[1]); m = len(items); adj, run = {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, (m - i) * v); adj[k] = min(1.0, run)
    return adj


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def series(df, model, proto, metric="macro_f1"):
    s = df[(df.model == model) & (df.protocol == proto)].set_index(["repeat", "fold"])[metric].sort_index()
    return s


def paired(df, a, b, proto="leakfree", metric="macro_f1"):
    j = series(df, a, proto, metric).to_frame("a").join(series(df, b, proto, metric).to_frame("b"), how="inner")
    return (j.a - j.b).values, len(j)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    shards = sorted(OUT.glob("deep_rerun_raw_r*.csv"))
    df = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
    df = df.drop_duplicates(subset=["model", "protocol", "repeat", "fold"], keep="last")
    L = ["# Master statistics — local rerun (leak-free unless stated)\n",
         f"{len(shards)} shards, {len(df)} rows; models: {sorted(df.model.unique())}\n"]
    counts = df[df.protocol == "leakfree"].groupby("model").size()
    incomplete = counts[counts < 50]
    if len(incomplete):
        L.append(f"> **Incomplete models (runs < 50):** {incomplete.to_dict()} — numbers below are provisional.\n")

    # 1. master ranking
    L.append("## 1. Master ranking (leak-free, 10 × 5 repeated subject-level CV; mean ± SD, bootstrap 95 % CI)\n")
    L.append("| # | Model | runs | macro-F1 | 95 % CI | Bal. Acc | ROC-AUC | PR-AUC | Tight-F1 | params |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for m in RANK_MODELS:
        g = df[(df.model == m) & (df.protocol == "leakfree")]
        if len(g) == 0:
            continue
        r = dict(model=m, runs=len(g), params=int(g.params.max()))
        for k in METRICS:
            r[f"{k}_mean"] = g[k].mean(); r[f"{k}_sd"] = g[k].std(ddof=1)
            ci = boot_ci(g[k].values); r[f"{k}_lo"], r[f"{k}_hi"] = ci
        rows.append(r)
    rank = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    rank.to_csv(OUT / "master_ranking.csv", index=False)
    for i, r in rank.iterrows():
        L.append(f"| {i + 1} | {DISPLAY[r.model]} | {r.runs} | **{r.macro_f1_mean:.3f}** ± {r.macro_f1_sd:.3f} | "
                 f"[{r.macro_f1_lo:.3f}, {r.macro_f1_hi:.3f}] | {r.bal_acc_mean:.3f} | {r.roc_auc_mean:.3f} | "
                 f"{r.pr_auc_mean:.3f} | {r.tight_f1_mean:.2f} | {r.params:,} |")

    # 5. metric-wise ranks
    L.append("\n## 5. Rank of each model under each metric (1 = best) — R1-M2\n")
    L.append("| Model | macro-F1 | Bal. Acc | ROC-AUC | PR-AUC | Tight-F1 |\n|---|---|---|---|---|---|")
    ranks = {k: rank[f"{k}_mean"].rank(ascending=False, method="min").astype(int) for k in METRICS}
    for i, r in rank.iterrows():
        L.append(f"| {DISPLAY[r.model]} | " + " | ".join(str(ranks[k][i]) for k in METRICS) + " |")
    xg = rank[rank.model == "XGBoost"].index
    if len(xg):
        i = xg[0]
        beaten = {k: int((rank[f"{k}_mean"] < rank.loc[i, f"{k}_mean"]).sum()) for k in METRICS}
        L.append(f"\nXGBoost out-ranks this many of the other {len(rank) - 1} models per metric: {beaten}")

    # 2. comparison family
    L.append("\n## 2. Pre-registered comparison family (leak-free macro-F1; paired by repeat × fold)\n")
    L.append("Nadeau–Bengio corrected resampled t-test: se = √((1/n + ρ)·s²_d), ρ = 1/(k−1) = 0.25, n = 50, k = 5. "
             "Holm step-down over the whole family. MDE at α = 0.05, power 0.8.\n")
    fam, labels = {}, {}
    for a, b, lab in [("cnn_aug", "cnn_noaug", "6-mode aug vs no-aug (1D-CNN)"),
                      ("resnet_gc", "resnet", "ResNet1D+GC vs plain ResNet1D")] + \
                     [(m, "XGBoost", f"{DISPLAY[m]} vs XGBoost") for m in DEEP_VS_XGB]:
        d, n = paired(df, a, b)
        if n:
            fam[lab] = d
    raw_p = {k: nb_ttest(v)[1] for k, v in fam.items()}; adj_p = holm(raw_p)
    L.append("| Comparison (A vs B) | n | Δ | 95 % CI(Δ) | d_z | NB p | Holm p | MDE | |Δ| ≥ MDE? |\n|---|---|---|---|---|---|---|---|---|")
    for k, d in fam.items():
        ci = boot_ci(d); dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
        L.append(f"| {k} | {len(d)} | {d.mean():+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {dz:.2f} | "
                 f"{'**' if raw_p[k] < ALPHA else ''}{fmt_p(raw_p[k])}{'**' if raw_p[k] < ALPHA else ''} | "
                 f"{'**' if adj_p[k] < ALPHA else ''}{fmt_p(adj_p[k])}{'**' if adj_p[k] < ALPHA else ''} | "
                 f"{mde(d):.3f} | {'yes' if abs(d.mean()) >= mde(d) else 'no'} |")
    L.append(f"\nRaw α = 0.05: {sum(p < ALPHA for p in raw_p.values())}/{len(fam)} significant; "
             f"**Holm: {sum(p < ALPHA for p in adj_p.values())}/{len(fam)}**. "
             f"Deep-vs-XGBoost deltas all positive: {all(fam[k].mean() > 0 for k in fam if 'XGBoost' in k)}.")
    # sign / Wilcoxon as descriptive sensitivity (R2-M1b)
    if "6-mode aug vs no-aug (1D-CNN)" in fam:
        d = fam["6-mode aug vs no-aug (1D-CNN)"]; pos = int((d > 0).sum()); neg = int((d < 0).sum())
        pw = stats.wilcoxon(d, zero_method="wilcox").pvalue if (d != 0).any() else np.nan
        ps = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else np.nan
        L.append(f"\nDescriptive sensitivity for the augmentation contrast (treats the 50 dependent runs as independent — "
                 f"NOT a valid test of the null): augmentation ahead in {pos}/{pos + neg} runs, sign p = {fmt_p(ps)}, "
                 f"Wilcoxon p = {fmt_p(pw)}.")

    # 3. 2x2 ablation with interaction
    cells = {"aug+samp": "cnn_aug", "noaug+samp": "cnn_noaug", "aug+nosamp": "cnn_aug_nosampler", "noaug+nosamp": "cnn_noaug_nosampler"}
    if all(len(series(df, m, "leakfree")) for m in cells.values()):
        S = {k: series(df, m, "leakfree") for k, m in cells.items()}
        J = pd.concat(S, axis=1).dropna()
        L.append(f"\n## 3. 2 × 2 augmentation × sampler ablation on the 1D-CNN (n = {len(J)} paired runs) — R1-M1a\n")
        L.append("| | sampler ON | sampler OFF |\n|---|---|---|")
        L.append(f"| aug ON | {J['aug+samp'].mean():.3f} ± {J['aug+samp'].std():.3f} | {J['aug+nosamp'].mean():.3f} ± {J['aug+nosamp'].std():.3f} |")
        L.append(f"| aug OFF | {J['noaug+samp'].mean():.3f} ± {J['noaug+samp'].std():.3f} | {J['noaug+nosamp'].mean():.3f} ± {J['noaug+nosamp'].std():.3f} |")
        aug_me = ((J['aug+samp'] + J['aug+nosamp']) - (J['noaug+samp'] + J['noaug+nosamp'])) / 2
        samp_me = ((J['aug+samp'] + J['noaug+samp']) - (J['aug+nosamp'] + J['noaug+nosamp'])) / 2
        inter = (J['aug+samp'] - J['noaug+samp']) - (J['aug+nosamp'] - J['noaug+nosamp'])
        L.append("\n| Effect (macro-F1) | estimate | 95 % CI | NB p |\n|---|---|---|---|")
        for lab, e in [("Augmentation main effect", aug_me), ("Sampler main effect", samp_me),
                       ("Interaction (aug effect with sampler − without)", inter),
                       ("Aug effect | sampler ON (headline, Table 2)", J['aug+samp'] - J['noaug+samp']),
                       ("Aug effect | sampler OFF", J['aug+nosamp'] - J['noaug+nosamp'])]:
            ci = boot_ci(e.values); p = nb_ttest(e.values)[1]
            L.append(f"| {lab} | {e.mean():+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {fmt_p(p)} |")

    # 4. inflation
    both = [m for m in df.model.unique() if {"leakfree", "optimistic"} <= set(df[df.model == m].protocol.unique())]
    if both:
        L.append("\n## 4. Optimistic vs leak-free inflation (Table 5)\n")
        L.append("| Model | leak-free | optimistic | Δ inflation | 95 % CI | params |\n|---|---|---|---|---|---|")
        infl = []
        for m in sorted(both, key=lambda m: -series(df, m, "leakfree").mean()):
            d, n = paired(df, m, m, "leakfree") if False else (None, 0)
            lf = series(df, m, "leakfree"); op = series(df, m, "optimistic")
            j = lf.to_frame("lf").join(op.to_frame("op"), how="inner"); dd = (j.op - j.lf).values
            ci = boot_ci(dd); pr = int(df[df.model == m].params.max())
            infl.append((pr, dd.mean()))
            L.append(f"| {DISPLAY.get(m, m)} | {j.lf.mean():.3f} | {j.op.mean():.3f} | {dd.mean():+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {pr:,} |")
        if len(infl) >= 4:
            rho, p = stats.spearmanr([a for a, _ in infl], [b for _, b in infl])
            L.append(f"\nSpearman ρ(params, inflation) = {rho:.2f}, p = {fmt_p(p)}; inflation range "
                     f"{min(b for _, b in infl):+.3f} … {max(b for _, b in infl):+.3f}.")
    else:
        L.append("\n## 4. Inflation — optimistic protocol not yet rerun locally (Table 5 pending).")

    (OUT / "master_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)); print(f"\nwritten: {OUT / 'master_report.md'}")


if __name__ == "__main__":
    main()
