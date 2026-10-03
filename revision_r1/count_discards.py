"""
Revision R1 — segmentation / discard statistics (Reviewer 2, Major 5: "numbers of
discarded images and boundary-truncated beats").

Read-only pass over the original chart PNGs with exactly the build_dataset.py
pipeline (reader -> band-pass -> peak detection -> fixed-window segmentation);
nothing in the canonical outputs/ directory is touched.  For every class folder:
  files total / files rejected by the reader (and why) / systolic peaks detected /
  beats kept / beats discarded because the [-0.15, +0.65] s window crossed a
  recording boundary.
Cross-checks the kept-beat count against dataset.csv.

Env: TCM_ROOT (code), TCM_DATA (dataset.csv), RES_DIR (output), PULSE_ROOT (PNG folders;
default /home/ailab_server/tcm/pluse).
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TCM = Path(os.environ.get("TCM_ROOT", HERE.parent / "TCM")).resolve()
sys.path.insert(0, str(TCM))
DATA = Path(os.environ.get("TCM_DATA", TCM / "outputs")).resolve()
RES = Path(os.environ.get("RES_DIR", HERE / "outputs")).resolve()
PULSE = Path(os.environ.get("PULSE_ROOT", "/home/ailab_server/tcm/pluse"))

from src.preprocessing.image_reader import read_waveform_from_png          # noqa: E402
from src.preprocessing.signal_filter import bandpass_filter                # noqa: E402
from src.preprocessing.segmentation import detect_peaks_dynamic, segment_and_normalize  # noqa: E402

CLASSES = {"弦脈": "Wiry", "滑脈": "Slippery", "緊脈": "Tight", "澀脈": "Choppy (excluded)"}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows, reasons = [], defaultdict(Counter)
    for zh, en in CLASSES.items():
        folder = PULSE / zh
        if not folder.exists():
            print(f"[WARN] {folder} missing"); continue
        pngs = sorted(folder.glob("*.png"))
        st = Counter(files=len(pngs), files_rejected=0, files_ok=0, peaks=0,
                     beats_kept=0, beats_truncated=0, files_with_zero_beats=0)
        for p in pngs:
            try:
                wf, fs = read_waveform_from_png(p, assumed_duration_s=6.0)
            except Exception as exc:                       # reader rejects the image
                st["files_rejected"] += 1
                reasons[en][type(exc).__name__ + ": " + str(exc).split(" in ")[0][:60]] += 1
                continue
            filt = bandpass_filter(wf, fs=fs, lowcut=0.5, highcut=10.0, order=4)
            peaks = detect_peaks_dynamic(filt, fs=fs, max_hr_bpm=220.0, height_k=0.5, prominence_ratio=0.20)
            beats, kept_peaks = segment_and_normalize(filt, fs=fs, target_length=200,
                                                      max_hr_bpm=220.0, height_k=0.5, prominence_ratio=0.20)
            st["files_ok"] += 1
            st["peaks"] += int(len(peaks))
            st["beats_kept"] += int(len(beats))
            st["beats_truncated"] += int(len(peaks) - len(beats))
            if len(beats) == 0:
                st["files_with_zero_beats"] += 1
        rows.append(dict(cls=en, **st))

    df = pd.DataFrame(rows).fillna(0).astype({c: int for c in ["files", "files_rejected", "files_ok", "peaks",
                                                                   "beats_kept", "beats_truncated", "files_with_zero_beats"]})
    ds = pd.read_csv(DATA / "dataset.csv")
    in_ds = ds.pulse_type_en.value_counts()
    df["beats_in_dataset_csv"] = df.cls.map(in_ds).fillna(0).astype(int)

    L = ["# Segmentation and discard statistics (read-only rerun of the build_dataset pipeline)\n",
         f"PNG root: `{PULSE}`  |  dataset.csv: `{DATA / 'dataset.csv'}`\n",
         "| Class | PNG files | rejected by reader | processed | peaks detected | beats kept | boundary-truncated (discarded) | files with 0 beats | beats in dataset.csv |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.cls} | {r.files} | {r.files_rejected} | {r.files_ok} | {r.peaks} | {r.beats_kept} | "
                 f"{r.beats_truncated} ({r.beats_truncated / max(1, r.peaks) * 100:.1f} %) | {r.files_with_zero_beats} | {r.beats_in_dataset_csv} |")
    t = df[df.cls != "Choppy (excluded)"].sum(numeric_only=True)
    L.append(f"| **3-class total** | {t.files} | {t.files_rejected} | {t.files_ok} | {t.peaks} | {t.beats_kept} | "
             f"{t.beats_truncated} ({t.beats_truncated / max(1, t.peaks) * 100:.1f} %) | {t.files_with_zero_beats} | {t.beats_in_dataset_csv} |")
    if reasons:
        L.append("\nReader rejection reasons:\n")
        for cls, c in reasons.items():
            for k, v in c.items():
                L.append(f"- {cls}: {v} × {k}")
    match = int(t.beats_kept) == int(t.beats_in_dataset_csv)
    L.append(f"\nKept-beat count {'MATCHES' if match else 'DOES NOT MATCH'} dataset.csv "
             f"({int(t.beats_kept)} vs {int(t.beats_in_dataset_csv)}).")
    (RES / "discard_counts.md").write_text("\n".join(L), encoding="utf-8")
    df.to_csv(RES / "discard_counts.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
