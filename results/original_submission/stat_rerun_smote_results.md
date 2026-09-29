# SMOTE-Variant Ablation — Statistical Rerun (TCM 3-Class Pulse)

Feature-space imbalance-strategy ablation (paper Table `tab:smote`), re-run under the **same repeated, leakage-free harness** as the main statistical rerun so the table is consistent with the rest of the paper. Design: **10 repeats x 5 folds = 50 runs**, XGBoost on the nine three-Gaussian features. Primary metric: macro-F1. α = 0.05 (two-sided).

**Harness / determinism.** Classical route reused verbatim from `stat_rerun_harness.run_ml` (converged beats only; StandardScaler fit on the training fold; synthetic Tight beats injected from train-fold subjects only; test fold scaled with the train scaler, never oversampled, read once). Same outer partition seeds as the main run (StratifiedGroupKFold, shuffle=True, random_state = repeat index 0..9), so each strategy is **paired by (repeat, fold)** with the main-run XGBoost baseline. XGBoost hyperparameters and random_state=42 are identical to the main run; sampler random_state=42. Versions: imbalanced-learn 0.14.2, xgboost 3.3.0.

**Leakage note.** SMOTE/BorderlineSMOTE/ADASYN operate in the 9-D feature space and are `fit_resample`'d **only on the training-fold matrix**. Synthetic points are interpolations of training-fold minority features — they introduce **no new subjects** and therefore cannot leak test-fold identities; the test fold is never oversampled and never seen during fitting.

**Consistency check.** The baseline reproduces the main-run XGBoost leakage-free number byte-for-byte — per-run |Δ| vs main-run XGBoost: max=0.00e+00, mean=0.00e+00 over 50 paired runs (main-run XGBoost mean macro-F1 = 0.7415, this baseline = 0.7415).


## 1. Strategy comparison (leakage-free, macro-F1)

| Strategy | Mean macro-F1 | SD | 95% CI (bootstrap) | Tight-F1 | Δ vs baseline | d_z | corrected-t p | Wilcoxon p | sign k/N |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (class weights) | 0.741 | 0.030 | [0.733, 0.750] | 0.462 | — (reference) | — | — | — | — |
| SMOTE | 0.740 | 0.030 | [0.732, 0.749] | 0.460 | -0.001 [-0.004, +0.001] | -0.15 | 0.7683 | 0.5463 | 25/50 |
| BorderlineSMOTE | 0.732 | 0.029 | [0.725, 0.740] | 0.442 | -0.009 [-0.012, -0.006] | -0.82 | 0.1215 | 1.86e-06 | 38/50 |
| ADASYN | 0.741 | 0.030 | [0.733, 0.750] | 0.462 | +0.000 [+0.000, +0.000] | — | 1.0000 | 1.0000 | 0/0 |

*n = 50 paired runs per strategy. CI = percentile bootstrap (10,000 resamples) over per-run values. Δ = mean(strategy − baseline), paired by (repeat, fold). Corrected-t = Nadeau-Bengio dependence-corrected resampled t-test (**primary p**). Bold Δ = significant at α=0.05 by the corrected-t. sign k/N = larger of wins/losses over N non-tied runs.*


## 2. ADASYN per-fold degeneration

ADASYN fell back to the class-weighted baseline in **50/50 folds** (every fold raised `ValueError: No samples will be generated with the provided ratio settings.`).

With the minority (Tight) class already enlarged by the harness's synthetic-beat injection, ADASYN's density-ratio criterion computes zero samples to generate and the sampler raises before producing any synthetic points; the fold therefore trains on the identical class-weighted baseline data. This reproduces the fallback seen in the original single-partition run — the `xgb_adasyn` and `xgb_baseline` rows are consequently identical by construction on every fold.


## 3. Plain-language summary (drop-in for the paper)

Under repeated subject-level cross-validation (10 repeats x 5 folds = 50 runs) with the unified leakage-free harness, none of the three feature-space oversampling variants (SMOTE, BorderlineSMOTE, ADASYN) differed significantly from the class-weighted XGBoost baseline on macro-F1 under the dependence-corrected resampled t-test (all |Δ| ≤ 0.009 macro-F1; smallest corrected-t p = 0.1215 ≥ 0.05); ADASYN was in fact identical to the baseline because its density criterion generated no synthetic samples and it fell back in 50/50 folds. The SMOTE-family oversamplers are therefore **statistically indistinguishable from the class-weighted baseline**, matching the paper's original single-partition conclusion. (BorderlineSMOTE was directionally the weakest (Δ = -0.009 macro-F1, consistently below baseline: Wilcoxon p = 1.86e-06, sign 38/50), but this small degradation does not survive the dependence-corrected primary test (p = 0.1215) and sits within ±0.01 macro-F1 — i.e. a reproducible-in-direction but negligible-in-magnitude effect, not evidence that any oversampler helps).
