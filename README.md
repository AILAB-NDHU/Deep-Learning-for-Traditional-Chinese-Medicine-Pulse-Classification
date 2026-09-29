# Deep Learning for Traditional Chinese Medicine Pulse Classification from Reconstructed Waveform Images — A Leakage-Free Evaluation

Code for the manuscript *"Deep Learning for Traditional Chinese Medicine Pulse Classification from Reconstructed Waveform Images: A Leakage-Free Evaluation"* (IEEE Access, manuscript Access-2026-37634, under revision), AI Lab, National Dong Hwa University.

The repository contains **every script that produced a number in the paper**: the image-to-signal pipeline, the three-Gaussian feature extractor, the waveform-level augmentation, all model definitions, the unified repeated-CV leakage-free harness with its dependence-corrected statistics, and the revision scripts (per-beat prediction dumps, subject-level analysis, Grad-CAM over all held-out beats, table generation).

## ⚠️ Data

**The study data are not included and cannot be shared.** The pulse charts were collected under Hualien Tzu Chi Hospital IRB107-08-B from 179 volunteers; neither the chart images nor the beat-level matrices may be redistributed.

`make_synthetic_demo.py` fabricates **synthetic stand-in charts** from the same parametric three-Gaussian pulse model that the paper uses for feature extraction, rendered in the device's chart style, so that the whole pipeline can be executed. These synthetic data are not measurements of any person and have no clinical meaning; results on them only demonstrate that the code runs.

## Layout

```
src/preprocessing/   image_reader.py (per-column centroid extraction), signal_filter.py, segmentation.py
src/features/        gaussian_fit.py  (three-Gaussian fit; components are mu-ordered downstream, see revision_r1/)
src/augmentation/    signal_warping.py (six-mode waveform-level Tight augmentation)
src/train_*.py       model definitions and the original single-partition training scripts
src/cv_utils.py      subject-grouped inner validation split
src/build_dataset.py PNG -> filtered signal -> beats -> nine Gaussian features (outputs/dataset.csv, beats_matrix.npy)
src/validate_roundtrip.py   synthetic round-trip re-rendering consistency test (Section III-C)

stat_rerun_harness.py        unified repeated-CV harness (10 x 5 subject-grouped folds, leak-free + optimistic protocols)
stat_rerun_analysis.py       Nadeau-Bengio corrected t-tests, bootstrap CIs, MDE, inflation table
stat_rerun_smote_*.py        feature-space oversampling ablation under the same harness

revision_r1/                 scripts added for the revision
  rerun_deep_with_preds.py     harness copy that saves per-beat probabilities and fold weights; mu-ordered features
  rerun_smote_musorted.py      SMOTE family on mu-ordered features
  analyze_master.py            ranking, Holm-adjusted comparison family, 2x2 ablation with interaction
  analyze_subject_level.py     beat vs subject-weighted vs subject-level metrics, subject-clustered CIs, confusion matrices
  analyze_gradcam_all_heldout.py  Grad-CAM for every held-out beat (correct and incorrect)
  make_tables.py               generates every LaTeX table of the manuscript from a results directory
  compare_environments.py      replication across environments (original run / local rerun / server rerun)
  count_discards.py            reader-rejection and boundary-truncation statistics
  remote/                      shell launchers used on the lab server

results/original_submission/  per-run scores of the submitted version (July 2026)
results/server_rerun_2026-09/ per-run scores of the revision rerun (both protocols)
```

## Quick start (synthetic demo, CPU is fine)

```bash
pip install -r requirements.txt
bash run_demo.sh
```

`run_demo.sh` fabricates ~80 synthetic charts, runs image-to-signal reconstruction, segmentation and Gaussian fitting, builds the augmented Tight set, executes one repeat of the harness on a fast subset of models, and prints the statistics report.

## Reproducing the paper's tables (requires the study data)

```bash
# 1. place dataset.csv, beats_matrix.npy, tight_augmented_beats.npy, tight_augmented_meta.csv in outputs/
# 2. full harness, both protocols, per-beat predictions saved (hours on one GPU; shard by --only-repeat)
cd revision_r1
TCM_ROOT=.. TCM_DATA=../outputs RES_DIR=./outputs python rerun_deep_with_preds.py --protocols leakfree,optimistic
python rerun_smote_musorted.py
python analyze_master.py && python analyze_subject_level.py && python analyze_gradcam_all_heldout.py --model resnet_gc
python make_tables.py          # -> LaTeX fragments + numbers.tex
```

Statistical protocol (Section III-H of the paper): the sole primary test is the Nadeau–Bengio corrected resampled *t*-test with ρ = 1/(k−1) = 0.25, n = 50 paired runs; Holm step-down adjustment over the pre-registered comparison family; bootstrap 95 % CIs; minimum detectable effects at α = 0.05, power 0.8.

## Environment

Original run: Python 3.12.3, PyTorch 2.12.0 (CUDA 13.0), scikit-learn 1.9.0, xgboost 3.3.0, imbalanced-learn 0.14.2, matplotlib 3.11.0, NVIDIA RTX 4090. GPU kernels are not bit-reproducible across hardware; `results/` documents two independent reruns whose per-model means agree with the original within 0.018 macro-F1.

## Citation

Chiu X.-D., Liao C.-T., Kavak S., Yen S.-J. *Deep Learning for Traditional Chinese Medicine Pulse Classification from Reconstructed Waveform Images: A Leakage-Free Evaluation.* IEEE Access (under revision), 2026.

## License

MIT (see `LICENSE`).
