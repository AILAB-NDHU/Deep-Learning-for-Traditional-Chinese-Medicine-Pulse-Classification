# Replication across environments — leak-free macro-F1


## Protocol: leakfree

| Model | A: original (server, Jul-2026) mean ± SD | B: local rerun (RTX 5080) mean ± SD | C: server rerun (RTX 4090) mean ± SD | mean paired abs diff A→B | mean paired abs diff A→C |
|---|---|---|---|---|---|
| ResNet1D (plain) | 0.791 ± 0.048 | 0.789 ± 0.049 | 0.791 ± 0.044 | 0.024 (n=50) | 0.027 (n=50) |
| ResNet1D + Gated | 0.789 ± 0.048 | 0.786 ± 0.047 | 0.779 ± 0.045 | 0.027 (n=50) | 0.032 (n=50) |
| ResNet1D + GC | 0.781 ± 0.043 | 0.775 ± 0.038 | 0.779 ± 0.049 | 0.026 (n=50) | 0.024 (n=50) |
| ResNet1D + GC + Gated | 0.780 ± 0.041 | 0.781 ± 0.047 | 0.774 ± 0.042 | 0.028 (n=50) | 0.029 (n=50) |
| 1D-CNN + aug, no sampler | 0.774 ± 0.060 | 0.771 ± 0.056 | 0.767 ± 0.067 | 0.024 (n=50) | 0.028 (n=50) |
| Prototypical Network | 0.770 ± 0.040 | 0.773 ± 0.049 | 0.774 ± 0.048 | 0.028 (n=50) | 0.030 (n=50) |
| Inception1D | 0.769 ± 0.056 | 0.783 ± 0.052 | 0.787 ± 0.049 | 0.037 (n=50) | 0.035 (n=50) |
| 1D-CNN no aug, no sampler | 0.766 ± 0.052 | 0.766 ± 0.052 | 0.766 ± 0.056 | 0.017 (n=50) | 0.017 (n=50) |
| 1D-CNN + 6-mode aug | 0.766 ± 0.064 | 0.766 ± 0.058 | 0.771 ± 0.064 | 0.033 (n=50) | 0.028 (n=50) |
| Patch Transformer | 0.763 ± 0.036 | 0.765 ± 0.039 | 0.763 ± 0.036 | 0.021 (n=50) | 0.000 (n=50) |
| XGBoost | 0.741 ± 0.030 | 0.743 ± 0.032 | 0.742 ± 0.032 | 0.009 (n=50) | 0.008 (n=50) |
| 1D-CNN (no aug) | 0.733 ± 0.044 | 0.736 ± 0.050 | 0.732 ± 0.047 | 0.021 (n=50) | 0.021 (n=50) |
| Random Forest | 0.729 ± 0.035 | 0.728 ± 0.035 | 0.728 ± 0.035 | 0.008 (n=50) | 0.008 (n=50) |
| SVM | 0.720 ± 0.033 | 0.719 ± 0.030 | 0.719 ± 0.030 | 0.008 (n=50) | 0.008 (n=50) |

Ranking by mean macro-F1:

- **A: original (server, Jul-2026)**: ResNet1D (plain) > ResNet1D + Gated > ResNet1D + GC > ResNet1D + GC + Gated > 1D-CNN + aug, no sampler > Prototypical Network > Inception1D > 1D-CNN no aug, no sampler > 1D-CNN + 6-mode aug > Patch Transformer > XGBoost > 1D-CNN (no aug) > Random Forest > SVM
- **B: local rerun (RTX 5080)**: ResNet1D (plain) > ResNet1D + Gated > Inception1D > ResNet1D + GC + Gated > ResNet1D + GC > Prototypical Network > 1D-CNN + aug, no sampler > 1D-CNN + 6-mode aug > 1D-CNN no aug, no sampler > Patch Transformer > XGBoost > 1D-CNN (no aug) > Random Forest > SVM
- **C: server rerun (RTX 4090)**: ResNet1D (plain) > Inception1D > ResNet1D + Gated > ResNet1D + GC > ResNet1D + GC + Gated > Prototypical Network > 1D-CNN + 6-mode aug > 1D-CNN + aug, no sampler > 1D-CNN no aug, no sampler > Patch Transformer > XGBoost > 1D-CNN (no aug) > Random Forest > SVM

Largest change in a model's mean macro-F1, A→B: **0.015**

Largest change in a model's mean macro-F1, A→C: **0.018**

## Protocol: optimistic

| Model | A: original (server, Jul-2026) mean ± SD | C: server rerun (RTX 4090) mean ± SD | mean paired abs diff A→C |
|---|---|---|---|
| ResNet1D (plain) | 0.810 ± 0.039 | 0.810 ± 0.040 | 0.019 (n=50) |
| ResNet1D + GC + Gated | 0.808 ± 0.044 | 0.811 ± 0.037 | 0.022 (n=50) |
| ResNet1D + Gated | 0.807 ± 0.036 | 0.810 ± 0.044 | 0.028 (n=50) |
| ResNet1D + GC | 0.806 ± 0.042 | 0.801 ± 0.043 | 0.023 (n=50) |
| Inception1D | 0.796 ± 0.045 | 0.801 ± 0.050 | 0.036 (n=50) |
| 1D-CNN + 6-mode aug | 0.795 ± 0.062 | 0.798 ± 0.055 | 0.023 (n=50) |
| Prototypical Network | 0.794 ± 0.042 | 0.799 ± 0.037 | 0.024 (n=50) |
| Patch Transformer | 0.783 ± 0.040 | 0.783 ± 0.040 | 0.000 (n=50) |
| 1D-CNN (no aug) | 0.749 ± 0.047 | 0.754 ± 0.044 | 0.016 (n=50) |

Ranking by mean macro-F1:

- **A: original (server, Jul-2026)**: ResNet1D (plain) > ResNet1D + GC + Gated > ResNet1D + Gated > ResNet1D + GC > Inception1D > 1D-CNN + 6-mode aug > Prototypical Network > Patch Transformer > 1D-CNN (no aug)
- **C: server rerun (RTX 4090)**: ResNet1D + GC + Gated > ResNet1D (plain) > ResNet1D + Gated > ResNet1D + GC > Inception1D > Prototypical Network > 1D-CNN + 6-mode aug > Patch Transformer > 1D-CNN (no aug)

Largest change in a model's mean macro-F1, A→C: **0.006**