"""
TCM Pulse — PNG Waveform Extractor
====================================
Reads a matplotlib-rendered pulse waveform PNG and returns a 1-D numpy array.

All images in the dataset share the same format:
  - Background  : white (255, 255, 255)
  - Waveform    : matplotlib default blue  #1f77b4  (R=31, G=119, B=180)
  - Grid lines  : light gray/blue  (~195, 210, 225)
  - X-axis      : 0 – 6 seconds  (uniform across all recordings)

Algorithm
---------
1. Load image → RGB numpy array.
2. Apply blue-dominance mask  (B − R > 30, B − G > 10, B > 100, R < 180).
   This selects the waveform line while rejecting the grid and background.
3. For each pixel column in the detected plot area, compute the centroid
   row of waveform pixels.  Linear interpolation fills any gaps.
4. Invert the y-axis (image y increases downward; signal peaks upward).
5. Z-score normalise.
6. Estimate effective fs = plot_width_px / assumed_duration_s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import signal
from scipy.interpolate import interp1d


def read_waveform_from_png(
    image_path: str | Path,
    assumed_duration_s: float = 6.0,
) -> tuple[np.ndarray, float]:
    """Extract pulse waveform from a matplotlib-rendered PNG.

    Parameters
    ----------
    image_path : str or Path
        Path to the PNG file.
    assumed_duration_s : float
        Total recording duration shown on the x-axis (seconds).
        All dataset images use 6.0 s.

    Returns
    -------
    waveform : np.ndarray
        1-D z-score-normalised signal, length = plot_width_pixels.
    fs : float
        Estimated effective sampling rate  (plot_width_px / duration_s).

    Raises
    ------
    ValueError
        If the waveform colour mask detects < 20 valid columns.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.int32)   # shape (H, W, 3)

    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # ── Blue-dominance mask ──────────────────────────────────────────────────
    # Targets #1f77b4 (R=31, G=119, B=180) and its anti-aliased neighbours.
    # Grid lines (~190,210,225) fail the R < 180 criterion.
    # White background fails B − R > 30.
    mask = (
        (B - R > 30) &   # blue dominates red
        (B - G > 10) &   # blue dominates green
        (B > 100) &      # exclude very dark pixels (axis borders)
        (R < 180)        # exclude near-white / near-gray pixels
    )

    # ── Find horizontal extent of the plot area ──────────────────────────────
    col_hits = mask.any(axis=0)
    valid_cols = np.where(col_hits)[0]
    if len(valid_cols) < 20:
        raise ValueError(
            f"Only {len(valid_cols)} columns with waveform pixels found in "
            f"'{image_path}'.  Expected the standard matplotlib blue waveform."
        )

    x_min, x_max = int(valid_cols[0]), int(valid_cols[-1])
    plot_width = x_max - x_min + 1

    # ── Per-column centroid y-position ───────────────────────────────────────
    y_positions = np.full(plot_width, np.nan)
    for i, x in enumerate(range(x_min, x_max + 1)):
        col = mask[:, x]
        if col.any():
            y_positions[i] = float(np.where(col)[0].mean())

    # ── Interpolate over columns with no waveform pixel ─────────────────────
    x_idx = np.arange(plot_width)
    valid = ~np.isnan(y_positions)
    if valid.sum() < 20:
        raise ValueError(
            f"Insufficient waveform columns in '{image_path}' "
            f"({valid.sum()} / {plot_width})."
        )
    interp_fn = interp1d(
        x_idx[valid], y_positions[valid],
        kind="linear", bounds_error=False, fill_value="extrapolate",
    )
    y_positions = interp_fn(x_idx)

    # ── Invert y-axis & z-score normalise ────────────────────────────────────
    # Image row 0 = top; signal peaks point upward → negate
    raw = -(y_positions - y_positions.max())
    std = raw.std()
    if std < 1e-10:
        raise ValueError(f"Near-zero variance signal in '{image_path}'.")
    waveform = (raw - raw.mean()) / std

    fs = plot_width / assumed_duration_s
    return waveform, float(fs)
