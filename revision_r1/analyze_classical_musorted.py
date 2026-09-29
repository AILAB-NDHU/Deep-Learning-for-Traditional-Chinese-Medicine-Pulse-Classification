"""
Revision R1 — analysis of the mu-sorted classical rerun.

1. Reproduction check: unsorted rerun vs archive, per (model, repeat, fold).
2. Sorted vs unsorted classical results: mean, SD, bootstrap 95% CI, paired delta.
3. Deep-vs-classical comparisons re-done against the *sorted* XGBoost, with the
   Nadeau–Bengio corrected resampled t-test (paper's primary test) — and the
   Holm step-down correction over the full pre-registered comparison family
   (aug vs no-aug, GC vs plain, every deep model vs XGBoost).  Reported for the
   ORIGINAL family too, so Reviewer 1 Major 3 / Reviewer 2 Major 2 can be
   answered with both numbers.

Writes revision_r1/outputs/classical_musorted_report.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
ARCH = HERE.parent / "_working_archive"

K_FOLDS = 5
ALPHA = 0.05
N_BOOT = 10_000
RNG = np.random.default_rng(0)

DISPLAY = {
    "resnet": "ResNet1D (plain)", "resnet_gated": "ResNet1D + Gated",
    "resnet_gc": "ResNet1D + GC", "resnet_gc_gated": "ResNet1D + GC + Gated",
    "protonet": "Prototypical Network", "inception": "Inception1D",
    "cnn_aug": "1D-CNN + 6-mode aug", "transformer": "Patch Transformer",
    "cnn_noaug": "1D-CNN (no aug)", "XGBoost": "XGBoost", "RandomForest": "Random Forest",
    "SVM": "SVM",
}
DEEP_VS_XGB = ["resnet", "resnet_gated", "resnet_gc", "resnet_gc_gated",
               "protonet", "inception", "cnn_aug", "transformer"]


# ── statistics (same formulae as _working_archive/stat_rerun_analysis.py) ────
def corrected_resampled_ttest(d: np.ndarray, k_folds: int = K_FOLDS):
    """Nadeau & Bengio (2003): var inflated by (1/n + rho), rho = 1/(k-1)."""
    n = len(d)
    mean_d = d.mean()
    var_d = d.var(ddof=1)
    rho = 1.0 / (k_folds - 1)
    se = np.sqrt((1.0 / n + rho) * var_d)
    if se == 0:
        return np.nan, 1.0, n - 1, 0.0
    t = mean_d / se
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return t, p, n - 1, se


def boot_ci(x: np.ndarray, n_boot: int = N_BOOT):
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = min(1.0, running)
    return adj


def paired(df: pd.DataFrame, a: str, b: str, metric="macro_f1") -> np.ndarray:
    A = df[(df.model == a) & (df.protocol == "leakfree")].set_index(["repeat", "fold"])[metric]
    B = df[(df.model == b) & (df.protocol == "leakfree")].set_index(["repeat", "fold"])[metric]
    j = A.to_frame("a").join(B.to_frame("b"), how="inner")
    assert len(j) == 50, f"{a} vs {b}: {len(j)} paired runs"
    return (j.a - j.b).values


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows cp950 console
    arch = pd.read_csv(ARCH / "stat_rerun_raw.csv")
    uns = pd.read_csv(OUT / "classical_unsorted_check_raw.csv")
    srt = pd.read_csv(OUT / "classical_musorted_raw.csv")
    L = ["# Classical route with mu-sorted Gaussian components — analysis\n"]

    # 1. reproduction check
    L.append("## 1. Reproduction check (unsorted rerun vs archive, paired by model/repeat/fold)\n")
    m = uns.merge(arch[arch.model.isin(["SVM", "RandomForest", "XGBoost"])],
                  on=["model", "repeat", "fold"], suffixes=("_new", "_old"))
    m["d"] = (m.macro_f1_new - m.macro_f1_old).abs()
    L.append(f"{len(m)} paired runs; max |Δ macro-F1| = {m.d.max():.2e}, mean = {m.d.mean():.2e}\n")
    L.append("| Model | max abs diff | archive mean | rerun mean |\n|---|---|---|---|")
    for mod, g in m.groupby("model"):
        L.append(f"| {mod} | {g.d.max():.2e} | {g.macro_f1_old.mean():.4f} | {g.macro_f1_new.mean():.4f} |")

    # 2. sorted vs unsorted
    L.append("\n## 2. Effect of mu-sorting on the classical models (50 runs, leakage-free)\n")
    L.append("| Model | unsorted macro-F1 | sorted macro-F1 | 95% CI (sorted) | paired Δ (sorted−unsorted) | 95% CI(Δ) | NB p | sorted Bal.Acc | sorted ROC-AUC | sorted PR-AUC | sorted Tight-F1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    both = pd.concat([uns.assign(protocol="leakfree"), srt.assign(protocol="leakfree")])
    for mod in ["SVM", "RandomForest", "XGBoost"]:
        u = uns[uns.model == mod].set_index(["repeat", "fold"]).sort_index()
        s = srt[srt.model == mod].set_index(["repeat", "fold"]).sort_index()
        d = (s.macro_f1 - u.macro_f1).values
        _, p, _, _ = corrected_resampled_ttest(d)
        ci_s = boot_ci(s.macro_f1.values); ci_d = boot_ci(d)
        L.append(f"| {mod} | {u.macro_f1.mean():.3f} ± {u.macro_f1.std():.3f} | "
                 f"**{s.macro_f1.mean():.3f} ± {s.macro_f1.std():.3f}** | [{ci_s[0]:.3f}, {ci_s[1]:.3f}] | "
                 f"{d.mean():+.3f} | [{ci_d[0]:+.3f}, {ci_d[1]:+.3f}] | {fmt_p(p)} | "
                 f"{s.bal_acc.mean():.3f} | {s.roc_auc.mean():.3f} | {s.pr_auc.mean():.3f} | {s.tight_f1.mean():.2f} |")

    # 3. deep vs classical, original vs sorted XGBoost, with Holm
    L.append("\n## 3. Pre-registered comparison family — Nadeau–Bengio p, raw and Holm-adjusted\n")
    L.append("Family (from `stat_rerun_analysis.py` L266-274): 6-mode aug vs no-aug (1D-CNN); "
             "ResNet1D+GC vs plain; each deep model vs XGBoost. ρ = 1/(k−1) = 0.25, n = 50.\n")
    for label, xgb_df in [("ORIGINAL (unsorted) XGBoost", arch),
                          ("**mu-SORTED** XGBoost", pd.concat([arch[arch.model != "XGBoost"], srt[srt.model == "XGBoost"]]))]:
        fam = {"cnn_aug vs cnn_noaug": paired(arch, "cnn_aug", "cnn_noaug"),
               "resnet_gc vs resnet": paired(arch, "resnet_gc", "resnet")}
        for mdl in DEEP_VS_XGB:
            fam[f"{mdl} vs XGBoost"] = paired(xgb_df, mdl, "XGBoost")
        raw_p = {k: corrected_resampled_ttest(v)[1] for k, v in fam.items()}
        adj_p = holm(raw_p)
        L.append(f"\n### 3.{1 if label.startswith('ORIG') else 2} {label}\n")
        L.append("| Comparison (A vs B) | Δ macro-F1 | 95% CI(Δ) | NB p (raw) | Holm-adjusted p | sig. (raw) | sig. (Holm) |")
        L.append("|---|---|---|---|---|---|---|")
        for k, d in fam.items():
            ci = boot_ci(d)
            L.append(f"| {k} | {d.mean():+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {fmt_p(raw_p[k])} | "
                     f"{fmt_p(adj_p[k])} | {'**yes**' if raw_p[k] < ALPHA else 'no'} | "
                     f"{'**yes**' if adj_p[k] < ALPHA else 'no'} |")
        n_raw = sum(p < ALPHA for p in raw_p.values()); n_adj = sum(p < ALPHA for p in adj_p.values())
        L.append(f"\n{n_raw}/{len(fam)} significant at raw α=0.05; **{n_adj}/{len(fam)} after Holm**.")

    # 4. class-wise ordering-violation summary for the methods text
    ds = pd.read_csv(HERE.parent / "TCM" / "outputs" / "_intermediate" / "dataset.csv")
    viol = ~((ds.mu1 < ds.mu2) & (ds.mu2 < ds.mu3))
    L.append("\n## 4. Ordering-violation rate in the original fits (for Section III-E)\n")
    L.append("| Class | beats | violations | rate |\n|---|---|---|---|")
    for c, g in ds.groupby("pulse_type_en"):
        L.append(f"| {c} | {len(g)} | {int(viol[g.index].sum())} | {viol[g.index].mean()*100:.1f}% |")
    L.append(f"| **All** | {len(ds)} | {int(viol.sum())} | {viol.mean()*100:.1f}% |")

    rep = OUT / "classical_musorted_report.md"
    rep.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten: {rep}")


if __name__ == "__main__":
    main()
