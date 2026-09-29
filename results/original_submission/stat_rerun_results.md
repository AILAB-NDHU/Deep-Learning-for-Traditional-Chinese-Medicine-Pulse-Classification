# Statistical Rerun Results — TCM 3-Class Pulse Classification

Repeated subject-level cross-validation, unified harness. Design: **10 repeats x 5 folds = 50 runs** per model per protocol. Primary metric: macro-F1. alpha = 0.05 (a priori, two-sided).


All models share identical partitions, seeds and samplers each repeat, so every pairwise comparison is **paired by (repeat, fold)**. Leakage controls (subject-level grouping, synthetic beats only in inner-train subjects, test fold read once) are preserved every repeat. Seeds: outer partition seed = repeat index (0..9); per-run torch/numpy seed = 1000 + repeat*100 + fold*10 + model_id (logged in the raw CSV).


## 1. Master table (leakage-free protocol)

| Model | Mean macro-F1 | SD | 95% CI (bootstrap) | Bal.Acc | ROC-AUC | PR-AUC | Tight-F1 | Params |
|---|---|---|---|---|---|---|---|---|
| SVM (RBF) | 0.720 | 0.033 | [0.711, 0.730] | 0.779 | 0.932 | 0.804 | 0.419 | — |
| Random Forest | 0.729 | 0.035 | [0.719, 0.738] | 0.778 | 0.929 | 0.798 | 0.438 | — |
| XGBoost | 0.741 | 0.030 | [0.733, 0.750] | 0.795 | 0.934 | 0.810 | 0.462 | — |
| 1D-CNN (no aug) | 0.733 | 0.044 | [0.721, 0.745] | 0.791 | 0.936 | 0.802 | 0.441 | 44,163 |
| 1D-CNN (+6-mode aug) | 0.766 | 0.064 | [0.748, 0.783] | 0.770 | 0.959 | 0.825 | 0.492 | 44,163 |
| ResNet1D | 0.791 | 0.048 | [0.778, 0.804] | 0.791 | 0.956 | 0.836 | 0.556 | 1,168,707 |
| Inception1D | 0.769 | 0.056 | [0.752, 0.784] | 0.761 | 0.956 | 0.830 | 0.502 | 506,019 |
| Patch Transformer | 0.763 | 0.036 | [0.753, 0.773] | 0.810 | 0.949 | 0.830 | 0.499 | 154,371 |
| ResNet1D+GC | 0.781 | 0.043 | [0.770, 0.793] | 0.797 | 0.954 | 0.821 | 0.535 | 1,213,619 |
| ResNet1D+Gated | 0.789 | 0.048 | [0.775, 0.801] | 0.795 | 0.957 | 0.832 | 0.553 | 1,255,843 |
| ResNet1D+GC+Gated | 0.780 | 0.041 | [0.769, 0.792] | 0.797 | 0.955 | 0.821 | 0.530 | 1,300,755 |
| Prototypical Network | 0.770 | 0.040 | [0.759, 0.781] | 0.815 | 0.952 | 0.822 | 0.521 | 1,234,368 |

*n = 50 runs per model. CI = percentile bootstrap (10,000 resamples) over per-run macro-F1.*


## 2. Headline pairwise comparisons (leakage-free, macro-F1)

Delta = mean(A - B), paired by (repeat, fold). Corrected-t = Nadeau-Bengio corrected resampled t-test (**primary p**). Sign test counts runs favouring A.

> **How to read the three p-values.** The corrected-t is deliberately conservative: with 5-fold CV its variance is inflated by (1/n + 1/(k-1)) = (1/50 + 1/4), i.e. its standard error is ~3.7x the naive paired-t SE, because the 50 runs share heavily overlapping training sets and are *not* independent. The Wilcoxon and sign tests treat the 50 runs as independent and therefore reward a *consistent direction* even for small margins. When the sign/Wilcoxon tests are highly significant but the corrected-t is not (e.g. augmentation below), read it as: **the effect's direction is highly reproducible across partitions, but its magnitude is too small to survive the dependence-corrected test at this sample size** — report it as a consistent-but-small effect, not a confirmed one.

| Comparison (A vs B) | Delta | 95% CI(Delta) | d_z | corrected-t p | Wilcoxon p | sign k/N | sign p |
|---|---|---|---|---|---|---|---|
| 6-mode aug vs no-aug (1D-CNN) | +0.032 | [+0.018, +0.047] | +0.61 | 0.2466 | 5.12e-05 | 36/50 | 0.0026 |
| ResNet1D+GC vs plain ResNet1D | -0.010 | [-0.019, -0.001] | -0.29 | 0.5755 | 0.0932 | 28/50 | 0.4799 |
| 1D-CNN (+6-mode aug) vs XGBoost | +0.024 | [+0.007, +0.041] | +0.40 | 0.4479 | 0.0072 | 33/50 | 0.0328 |
| **ResNet1D vs XGBoost** | +0.050 | [+0.038, +0.062] | +1.13 | 0.0350 | 2.33e-09 | 43/50 | 2.10e-07 |
| Inception1D vs XGBoost | +0.027 | [+0.010, +0.043] | +0.46 | 0.3783 | 9.92e-05 | 38/50 | 0.0003 |
| Patch Transformer vs XGBoost | +0.022 | [+0.014, +0.029] | +0.78 | 0.1420 | 9.36e-07 | 41/50 | 5.61e-06 |
| ResNet1D+GC vs XGBoost | +0.040 | [+0.028, +0.051] | +0.93 | 0.0797 | 7.24e-08 | 41/50 | 5.61e-06 |
| ResNet1D+Gated vs XGBoost | +0.047 | [+0.033, +0.061] | +0.91 | 0.0871 | 3.26e-07 | 43/50 | 2.10e-07 |
| ResNet1D+GC+Gated vs XGBoost | +0.039 | [+0.028, +0.050] | +0.97 | 0.0667 | 2.07e-08 | 41/50 | 5.61e-06 |
| Prototypical Network vs XGBoost | +0.028 | [+0.018, +0.039] | +0.76 | 0.1525 | 6.73e-06 | 36/50 | 0.0026 |

*Bold comparison = significant at alpha=0.05 by the corrected-t (primary) test. sign k/N = larger of wins/losses over N non-tied runs.*


**Minimum detectable effect (MDE)** at alpha=0.05, power=0.8, given the observed repeated-CV variance: for the augmentation contrast the smallest macro-F1 difference detectable is **~0.079**. Per-comparison MDEs are listed in the effect-size table below; null results should be read as *'no difference detectable at this power'*, not *'no difference'*.


## 3. Effect sizes, confidence intervals & MDE (leakage-free)

| Comparison | Delta | 95% CI(Delta) | Cohen d_z | corrected-t sig? | MDE (a=.05,power=.8) | |Delta| >= MDE? |
|---|---|---|---|---|---|---|
| 6-mode aug vs no-aug (1D-CNN) | +0.032 | [+0.018, +0.047] | +0.61 | no | 0.079 | no |
| ResNet1D+GC vs plain ResNet1D | -0.010 | [-0.019, -0.001] | -0.29 | no | 0.051 | no |
| 1D-CNN (+6-mode aug) vs XGBoost | +0.024 | [+0.007, +0.041] | +0.40 | no | 0.090 | no |
| ResNet1D vs XGBoost | +0.050 | [+0.038, +0.062] | +1.13 | yes | 0.066 | no |
| Inception1D vs XGBoost | +0.027 | [+0.010, +0.043] | +0.46 | no | 0.088 | no |
| Patch Transformer vs XGBoost | +0.022 | [+0.014, +0.029] | +0.78 | no | 0.041 | no |
| ResNet1D+GC vs XGBoost | +0.040 | [+0.028, +0.051] | +0.93 | no | 0.064 | no |
| ResNet1D+Gated vs XGBoost | +0.047 | [+0.033, +0.061] | +0.91 | no | 0.077 | no |
| ResNet1D+GC+Gated vs XGBoost | +0.039 | [+0.028, +0.050] | +0.97 | no | 0.059 | no |
| Prototypical Network vs XGBoost | +0.028 | [+0.018, +0.039] | +0.76 | no | 0.056 | no |

*The MDE is the smallest true difference this repeated-CV design could detect with 80% power under the corrected test; it is a **power** property, not a significance verdict. A result can be significant (corrected-t p<0.05) yet have |Delta| just below the MDE -- that means the finding is real but the design was under-powered for an effect that small, so treat it as borderline and prioritise replication. 'corrected-t sig? = yes AND |Delta| >= MDE = yes' is the strongest tier of evidence.*


## 4. Optimistic-vs-leakage-free inflation & complexity relationship

Inflation Delta = mean(optimistic macro-F1 - leakage-free macro-F1), paired by (repeat, fold). ML models have no early-stop channel so their inflation is exactly 0 (excluded from the Spearman).

| Model | Params | Leakfree F1 | Optimistic F1 | Inflation Delta | 95% CI(Delta) | corrected-t p |
|---|---|---|---|---|---|---|
| 1D-CNN (no aug) | 44,163 | 0.733 | 0.749 | +0.016 | [+0.007, +0.024] | 0.3386 |
| 1D-CNN (+6-mode aug) | 44,163 | 0.766 | 0.795 | +0.030 | [+0.018, +0.042] | 0.2001 |
| ResNet1D | 1,168,707 | 0.791 | 0.810 | +0.019 | [+0.010, +0.027] | 0.2467 |
| Inception1D | 506,019 | 0.769 | 0.796 | +0.027 | [+0.011, +0.044] | 0.3799 |
| Patch Transformer | 154,371 | 0.763 | 0.783 | +0.020 | [+0.013, +0.028] | 0.1654 |
| ResNet1D+GC | 1,213,619 | 0.781 | 0.806 | +0.024 | [+0.016, +0.033] | 0.1197 |
| ResNet1D+Gated | 1,255,843 | 0.789 | 0.807 | +0.018 | [+0.008, +0.029] | 0.3600 |
| ResNet1D+GC+Gated | 1,300,755 | 0.780 | 0.808 | +0.028 | [+0.016, +0.040] | 0.2154 |
| Prototypical Network | 1,234,368 | 0.770 | 0.794 | +0.024 | [+0.013, +0.035] | 0.2524 |
| SVM (RBF) | — | 0.720 | 0.720 | +0.000 | [+0.000, +0.000] | 1.0000 (no early-stop channel) |
| Random Forest | — | 0.729 | 0.729 | +0.000 | [+0.000, +0.000] | 1.0000 (no early-stop channel) |
| XGBoost | — | 0.741 | 0.741 | +0.000 | [+0.000, +0.000] | 1.0000 (no early-stop channel) |

**Spearman(parameter count, inflation Delta)** over the 9 deep models: **rho = +0.067**, p = 0.8641, bootstrap 95% CI [-0.825, +0.862].

Interpretation: the complexity–inflation relationship is **weak / non-significant** (p >= 0.05). If non-significant, the 'leakage inflates the most complex model most' claim should be restricted to the specific GC-vs-plain reversal rather than stated as a general monotone trend.


## 5. 2x2 augmentation x sampler ablation (1D-CNN, leakage-free)

| Augmentation | Weighted sampler | Mean macro-F1 | 95% CI | Tight-F1 |
|---|---|---|---|---|
| aug | sampler | 0.766 | [0.748, 0.783] | 0.492 |
| no-aug | sampler | 0.733 | [0.721, 0.745] | 0.441 |
| aug | no-sampler | 0.774 | [0.757, 0.790] | 0.508 |
| no-aug | no-sampler | 0.766 | [0.752, 0.780] | 0.498 |

**Main effects (paired, corrected-t):**

- **Augmentation** main effect: Delta = +0.020 [+0.011, +0.029], d_z = +0.59, corrected-t p = 0.2645 (n.s.)
- **Weighted sampler** main effect: Delta = -0.020 [-0.027, -0.013], d_z = -0.78, corrected-t p = 0.1374 (n.s.)

## 6. Plain-language summary (drop-in for Results/Discussion)

Under repeated subject-level cross-validation (10 repeats x 5 folds = 50 runs) with a single unified, leakage-free harness, 6-mode Tight augmentation produced a small but highly reproducible improvement in 1D-CNN macro-F1: its direction is consistent across partitions (sign test 36/50, p = 0.0026; Wilcoxon p = 5.12e-05), but the dependence-corrected primary test does not reach significance and the effect sits below the minimum detectable effect (Delta = +0.032, 95% CI [+0.018, +0.047], d_z = +0.61, corrected-t p = 0.2466; MDE = 0.079). This tempers the paper's headline '+0.070' single-partition claim: the gain is real in direction but roughly half that size and not confirmable under the corrected test.  The proposed Global-Context module did NOT significantly change macro-F1 versus plain ResNet1D (Delta = -0.010 [-0.019, -0.001], corrected-t p = 0.5755), and its |Delta| is below the minimum detectable effect (0.051); we therefore report it as **no detectable difference at this power** rather than an improvement -- confirming the paper's negative claim for GC while quantifying it honestly.  All 8 deep models out-ranked XGBoost in direction (positive Delta, 8/8 significant by the sign test), but under the conservative corrected-t only 1 (ResNet1D) cleared alpha=0.05; the remaining ~0.02-0.04 macro-F1 margins fall below their MDEs. So 'deep beats classical' holds as a consistent trend and is statistically confirmed for the strongest model (ResNet1D, Delta approx +0.05) but should not be over-stated for the smaller gaps.  Across the 9 deep models, leakage inflation (optimistic - leakage-free) correlated with parameter count at Spearman rho = +0.067 (p = 0.8641), i.e. no statistically reliable monotone complexity-inflation trend; the inflation-with-complexity claim should be narrowed to the GC-vs-plain reversal only.  The 2x2 ablation disentangles the confound and attributes what gain there is to augmentation rather than the sampler: augmentation's main effect was +0.020 (corrected-t p = 0.2645), whereas the weighted random sampler's main effect was -0.020 and actually slightly hurt performance (p = 0.1374). Neither main effect clears the corrected test, but the direction is unambiguous -- the +0.070 the paper credited to augmentation is not a sampler artefact (the sampler, if anything, is marginally detrimental).  In short: differences with corrected-t p < 0.05 AND |Delta| above the MDE are the statistically supported claims; all others are reported as 'no detectable difference at this power' rather than evidence of equivalence.
