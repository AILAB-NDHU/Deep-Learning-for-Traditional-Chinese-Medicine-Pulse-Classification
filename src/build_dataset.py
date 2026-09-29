"""
TCM Pulse — 3-Class Dataset Builder (弦脈 / 滑脈 / 緊脈)
==========================================================
Identical to the original build_dataset.py but excludes 澀脈 (Choppy).
Data source: /home/ailab_server/tcm/pluse/  (shared, read-only)

Outputs (outputs/)
------------------
  dataset.csv      — one row per beat (metadata + Gaussian params)
  beats_matrix.npy — shape (N, 200), row-aligned with dataset.csv

Usage
-----
    conda run -n tcm python src/build_dataset.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

# ── project paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent          # /home/.../tcm_3class
SRC  = ROOT / "src"
sys.path.insert(0, str(ROOT))

from src.preprocessing.image_reader import read_waveform_from_png
from src.preprocessing.signal_filter import bandpass_filter
from src.preprocessing.segmentation  import segment_and_normalize
from src.features.gaussian_fit        import fit_three_gaussians

# ── constants ──────────────────────────────────────────────────────────────────
import os
DATA_ROOT = Path(os.environ.get("PULSE_ROOT", ROOT / "data" / "synthetic_demo" / "pluse"))  # chart PNGs; the study data cannot be shared
OUT_DIR   = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# 澀脈 (Choppy) intentionally excluded — 3-class experiment
PULSE_TYPE_MAP = {
    "弦脈": "Wiry",
    "滑脈": "Slippery",
    "緊脈": "Tight",
}

COLUMNS = [
    "subject_id", "recording_id",
    "pulse_type", "pulse_type_en",
    "beat_idx",
    "A1", "mu1", "sigma1",
    "A2", "mu2", "sigma2",
    "A3", "mu3", "sigma3",
    "fit_r2", "converged",
]


def process_file(
    png_path: Path, pulse_type: str, pulse_type_en: str
) -> tuple[list[dict], np.ndarray]:
    """Return (row_dicts, beat_vectors) where beat_vectors has shape (n_beats, 200)."""
    stem  = png_path.stem
    parts = stem.split("_", 1)
    subject_id   = int(parts[0]) if parts[0].isdigit() else parts[0]
    recording_id = parts[1] if len(parts) > 1 else "1"

    waveform, fs = read_waveform_from_png(png_path, assumed_duration_s=6.0)
    filtered = bandpass_filter(waveform, fs=fs, lowcut=0.5, highcut=10.0, order=4)
    beats, _ = segment_and_normalize(
        filtered, fs=fs, target_length=200,
        max_hr_bpm=220.0, height_k=0.5, prominence_ratio=0.20,
    )

    rows: list[dict] = []
    beat_vectors: list[np.ndarray] = []
    for beat_idx, beat in enumerate(beats):
        params, converged, r2 = fit_three_gaussians(beat, n_restarts=5)
        if params is not None:
            A1, mu1, s1, A2, mu2, s2, A3, mu3, s3 = params
        else:
            A1=mu1=s1=A2=mu2=s2=A3=mu3=s3 = np.nan

        rows.append({
            "subject_id":    subject_id,
            "recording_id":  recording_id,
            "pulse_type":    pulse_type,
            "pulse_type_en": pulse_type_en,
            "beat_idx":      beat_idx,
            "A1": A1, "mu1": mu1, "sigma1": s1,
            "A2": A2, "mu2": mu2, "sigma2": s2,
            "A3": A3, "mu3": mu3, "sigma3": s3,
            "fit_r2":    round(r2, 6) if not np.isnan(r2) else np.nan,
            "converged": bool(converged),
        })
        beat_vectors.append(beat.astype(np.float32))

    return rows, np.stack(beat_vectors, axis=0) if beat_vectors else np.empty((0, 200), dtype=np.float32)


def main() -> None:
    all_rows:  list[dict]      = []
    all_beats: list[np.ndarray] = []
    n_files_ok    = 0
    n_files_fail  = 0
    n_beats_fail  = 0
    n_beats_total = 0

    for folder_cn, folder_en in sorted(PULSE_TYPE_MAP.items()):
        folder = DATA_ROOT / folder_cn
        if not folder.exists():
            print(f"[WARN] folder not found: {folder}")
            continue

        png_files = sorted(folder.glob("*.png"))
        print(f"\n[{folder_en}] ({folder_cn})  — {len(png_files)} files")

        for png_path in png_files:
            try:
                rows, bvecs = process_file(png_path, folder_cn, folder_en)
                n_files_ok    += 1
                n_beats_total += len(rows)
                fail = sum(1 for r in rows if not r["converged"])
                n_beats_fail  += fail
                all_rows.extend(rows)
                if bvecs.shape[0] > 0:
                    all_beats.append(bvecs)
                print(
                    f"  {png_path.name:20s}  "
                    f"beats={len(rows):3d}  "
                    f"fit_fail={fail:2d}  "
                    f"mean_R2={np.nanmean([r['fit_r2'] for r in rows]):.3f}"
                )
            except Exception as exc:
                n_files_fail += 1
                print(f"  [ERROR] {png_path.name}: {exc}")

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    out_path = OUT_DIR / "dataset.csv"
    df.to_csv(out_path, index=False, float_format="%.6f")

    beats_matrix = np.concatenate(all_beats, axis=0) if all_beats else np.empty((0, 200), dtype=np.float32)
    npy_path = OUT_DIR / "beats_matrix.npy"
    np.save(npy_path, beats_matrix)

    print("\n" + "=" * 60)
    print("DATASET SUMMARY  (3-class: Wiry / Slippery / Tight)")
    print("=" * 60)
    print(f"  Files processed OK   : {n_files_ok}")
    print(f"  Files failed         : {n_files_fail}")
    print(f"  Total beats          : {n_beats_total}")
    pct = 100 * n_beats_fail / max(n_beats_total, 1)
    print(f"  Gaussian fit failures: {n_beats_fail} ({pct:.1f}%)")
    print()
    print(df.groupby("pulse_type_en")[["beat_idx", "fit_r2"]].agg(
        beat_count=("beat_idx", "count"),
        mean_R2=("fit_r2", "mean"),
    ).to_string())
    print()
    print(f"  Saved → {out_path.relative_to(ROOT)}")
    print(f"  Saved → {npy_path.relative_to(ROOT)}")
    print(f"  CSV shape        : {df.shape}")
    print(f"  beats_matrix     : {beats_matrix.shape}  dtype={beats_matrix.dtype}")
    print(f"  Subjects: {df['subject_id'].nunique()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
