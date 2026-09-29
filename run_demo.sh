#!/usr/bin/env bash
# End-to-end demonstration on SYNTHETIC data (no study data are included in this repository).
#   1. fabricate chart PNGs          -> data/synthetic_demo/pluse/
#   2. image-to-signal + segmentation + three-Gaussian features -> outputs/dataset.csv, outputs/beats_matrix.npy
#   3. waveform-level Tight augmentation                          -> outputs/tight_augmented_*
#   4. one repeat of the unified leakage-free harness (fast subset of models)
#   5. statistics report
# Full paper runs: python stat_rerun_harness.py --repeats 10   (all models, both protocols; hours on a GPU)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTHONIOENCODING=utf-8
export PULSE_ROOT="$PWD/data/synthetic_demo/pluse"

echo "== 1. synthetic charts ==";        $PY make_synthetic_demo.py
echo "== 2. build dataset ==";           $PY src/build_dataset.py | tail -8
echo "== 3. Tight augmentation ==";      $PY -m src.augmentation.signal_warping | tail -3
echo "== 4. harness (1 repeat, subset) =="; $PY stat_rerun_harness.py --repeats 1 --models cnn_noaug,cnn_aug,resnet,SVM,RandomForest,XGBoost
echo "== 5. statistics ==";              $PY stat_rerun_analysis.py | tail -20
echo "done. Raw per-run scores: stat_rerun_raw.csv ; report: stat_rerun_results.md"
