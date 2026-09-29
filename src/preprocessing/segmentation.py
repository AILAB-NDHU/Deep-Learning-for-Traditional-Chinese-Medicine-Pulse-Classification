"""
TCM Pulse Beat Segmentation & Length Normalization
===================================================
Phase 1 (continued) / Response to Reviewer 6

Pipeline
--------
Filtered signal
    → detect_peaks_dynamic   (adaptive threshold, prominence-based)
    → segment_beats          (fixed window around each systolic peak)
    → normalize_length       (FFT-based resample to target_length points)
    → np.ndarray  shape (n_beats, target_length)

Design decisions
----------------
* Peak detection uses scipy.signal.find_peaks with THREE robustness guards:
    1. height  : adaptive threshold = median + k * MAD  (resistant to outliers)
    2. distance: minimum inter-peak gap derived from max physiological HR (220 BPM)
    3. prominence: must stand out from local background by ≥ 0.2 * signal range
* Segmentation uses a fixed window [−pre, +post] around each peak,
  not beat-to-beat, so segment length is deterministic before resampling.
* Length normalization uses scipy.signal.resample (FFT-based),
  which preserves spectral content better than linear interpolation.

Usage
-----
>>> from src.preprocessing.segmentation import segment_and_normalize
>>> beats = segment_and_normalize(filtered_signal, fs=500.0, target_length=200)
>>> beats.shape   # (n_beats, 200)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Dynamic threshold peak detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_peaks_dynamic(
    sig: np.ndarray,
    fs: float,
    max_hr_bpm: float = 220.0,
    height_k: float = 0.5,
    prominence_ratio: float = 0.20,
) -> np.ndarray:
    """Detect systolic peaks with an adaptive threshold.

    Three robustness guards are applied simultaneously:

    1. **Adaptive height threshold** — ``median + height_k × MAD``.
       Uses Median Absolute Deviation (MAD) instead of std, making the
       threshold resistant to large-amplitude artefacts.

    2. **Minimum inter-peak distance** — derived from ``max_hr_bpm``.
       Prevents double-detection of the same beat.

    3. **Prominence filter** — peak must protrude above its local
       neighbourhood by at least ``prominence_ratio × peak-to-peak``.
       Suppresses small ripples riding on the main waveform.

    Parameters
    ----------
    sig : np.ndarray
        1-D filtered pulse signal.
    fs : float
        Sampling frequency (Hz).
    max_hr_bpm : float
        Maximum expected heart rate (BPM); sets the minimum inter-peak
        distance. Default 220 BPM covers all physiological cases.
    height_k : float
        Multiplier for MAD in the adaptive height threshold.
        Increase to require taller peaks; decrease to be more permissive.
    prominence_ratio : float
        Required prominence as a fraction of the signal peak-to-peak range.

    Returns
    -------
    peak_indices : np.ndarray
        Sample indices of detected peaks (sorted ascending).

    Raises
    ------
    ValueError
        If fewer than 2 peaks are detected (cannot form any beat segment).
    """
    sig = np.asarray(sig, dtype=np.float64)

    # Guard 1: adaptive height (median + k * MAD)
    median = np.median(sig)
    mad = np.median(np.abs(sig - median))
    height_threshold = median + height_k * mad

    # Guard 2: minimum distance (samples) from max physiological HR
    min_distance_samples = int(fs * 60.0 / max_hr_bpm)
    min_distance_samples = max(min_distance_samples, 1)

    # Guard 3: prominence relative to full signal range
    prominence_threshold = prominence_ratio * float(np.ptp(sig))

    peaks, _ = signal.find_peaks(
        sig,
        height=height_threshold,
        distance=min_distance_samples,
        prominence=prominence_threshold,
    )

    if len(peaks) < 2:
        raise ValueError(
            f"Only {len(peaks)} peak(s) detected. "
            "Check that the signal is filtered and not too short. "
            f"Tried: height>{height_threshold:.4f}, "
            f"distance>{min_distance_samples} samples, "
            f"prominence>{prominence_threshold:.4f}."
        )

    return peaks


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Beat segmentation — fixed window around each peak
# ═══════════════════════════════════════════════════════════════════════════════

def segment_beats(
    sig: np.ndarray,
    peak_indices: np.ndarray,
    fs: float,
    window_pre_s: float = 0.15,
    window_post_s: float = 0.65,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Extract a fixed-duration window around each detected peak.

    Window: ``[peak − window_pre_s, peak + window_post_s]``

    A fixed window (not beat-to-beat) is used so that every raw segment
    has the same number of samples before resampling, and the systolic
    peak is always at a consistent relative position (useful for
    morphological feature extraction).

    Parameters
    ----------
    sig : np.ndarray
        1-D signal (should be bandpass-filtered).
    peak_indices : np.ndarray
        Peak sample indices from ``detect_peaks_dynamic``.
    fs : float
        Sampling frequency (Hz).
    window_pre_s : float
        Seconds before the peak to include. Default 0.15 s.
    window_post_s : float
        Seconds after the peak to include. Default 0.65 s.
        Total default window = 0.80 s, sufficient for one full TCM beat.

    Returns
    -------
    segments : list of np.ndarray
        Each element is one beat waveform (variable length ≈
        ``(window_pre_s + window_post_s) × fs`` samples).
    valid_peaks : np.ndarray
        Subset of ``peak_indices`` for which a complete window was available
        (peaks near the signal boundaries are dropped).
    """
    sig = np.asarray(sig, dtype=np.float64)
    n = len(sig)
    pre  = int(round(window_pre_s  * fs))
    post = int(round(window_post_s * fs))

    segments: List[np.ndarray] = []
    valid_peaks: List[int] = []

    for p in peak_indices:
        start = p - pre
        end   = p + post + 1          # +1 so the window is inclusive
        if start < 0 or end > n:      # drop boundary peaks
            continue
        segments.append(sig[start:end].copy())
        valid_peaks.append(int(p))

    if len(segments) == 0:
        raise ValueError(
            "No complete beat segments could be extracted. "
            "The signal may be too short relative to the window size, "
            f"or peaks are too close to the boundaries. "
            f"window = [{window_pre_s}s, {window_post_s}s], "
            f"signal duration = {n / fs:.2f}s."
        )

    return segments, np.asarray(valid_peaks, dtype=int)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Length normalisation — FFT-based resample
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_length(
    segments: List[np.ndarray],
    target_length: int = 200,
) -> np.ndarray:
    """Resample every beat segment to ``target_length`` points.

    Uses ``scipy.signal.resample`` (FFT-based), which preserves spectral
    content better than linear or spline interpolation for quasi-periodic
    waveforms.

    Parameters
    ----------
    segments : list of np.ndarray
        Raw beat segments (may have slightly different lengths due to
        floating-point rounding in ``segment_beats``).
    target_length : int
        Number of samples in every output beat. Default 200.

    Returns
    -------
    np.ndarray, shape (n_beats, target_length)
        Resampled beats matrix, ready for feature extraction or 1D-CNN.
    """
    resampled = np.stack(
        [signal.resample(seg, target_length) for seg in segments],
        axis=0,
    )
    return resampled


# ═══════════════════════════════════════════════════════════════════════════════
# 4. End-to-end convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def segment_and_normalize(
    filtered_signal: np.ndarray,
    fs: float,
    target_length: int = 200,
    window_pre_s: float = 0.15,
    window_post_s: float = 0.65,
    max_hr_bpm: float = 220.0,
    height_k: float = 0.5,
    prominence_ratio: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full pipeline: detect peaks → segment → resample.

    Parameters
    ----------
    filtered_signal : np.ndarray
        Bandpass-filtered 1-D signal.
    fs : float
        Sampling frequency (Hz).
    target_length : int
        Output samples per beat.
    window_pre_s, window_post_s : float
        Beat window around each peak (seconds).
    max_hr_bpm, height_k, prominence_ratio :
        Forwarded to ``detect_peaks_dynamic``.

    Returns
    -------
    beats : np.ndarray, shape (n_beats, target_length)
        Normalised beat matrix.
    valid_peaks : np.ndarray, shape (n_beats,)
        Sample indices of the peaks used (relative to ``filtered_signal``).
    """
    peaks = detect_peaks_dynamic(
        filtered_signal, fs,
        max_hr_bpm=max_hr_bpm,
        height_k=height_k,
        prominence_ratio=prominence_ratio,
    )
    segments, valid_peaks = segment_beats(
        filtered_signal, peaks, fs,
        window_pre_s=window_pre_s,
        window_post_s=window_post_s,
    )
    beats = normalize_length(segments, target_length=target_length)
    return beats, valid_peaks


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_segmentation(
    sig: np.ndarray,
    fs: float,
    peak_indices: np.ndarray,
    beats: np.ndarray,
    target_length: int = 200,
    pulse_type: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Three-panel segmentation quality plot.

    Panel layout:
    ┌──────────────────────────────────────────────┐
    │  Filtered signal with detected peaks marked  │
    ├───────────────────────┬──────────────────────┤
    │  All beats overlaid   │  Mean ± std envelope │
    └───────────────────────┴──────────────────────┘

    Parameters
    ----------
    sig : np.ndarray
        Filtered signal used for detection.
    fs : float
        Sampling frequency (Hz).
    peak_indices : np.ndarray
        Detected peak positions (sample indices).
    beats : np.ndarray, shape (n_beats, target_length)
        Normalised beats from ``segment_and_normalize``.
    target_length : int
        Expected number of columns in ``beats``.
    pulse_type : str
        Label shown in the figure title.
    save_path : str, optional
        Output file path (.png / .pdf).
    show : bool
        Call ``plt.show()`` when True.

    Returns
    -------
    matplotlib.figure.Figure
    """
    t_sig = np.arange(len(sig)) / fs
    t_beat = np.linspace(0, 1, target_length)   # normalised beat axis [0, 1]
    n_beats = beats.shape[0]

    fig = plt.figure(figsize=(14, 8))
    suffix = f"  [{pulse_type}]" if pulse_type else ""
    fig.suptitle(
        f"TCM Pulse — Peak Detection & Beat Segmentation{suffix}",
        fontsize=13, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.50, wspace=0.35,
        top=0.93, bottom=0.08,
    )

    # ── Panel 1 (top, full width): signal + peak markers ─────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_sig, sig, color="#546E7A", linewidth=0.7, alpha=0.85,
             label="Filtered signal")
    ax1.scatter(
        t_sig[peak_indices], sig[peak_indices],
        color="#E53935", s=40, zorder=5, label=f"Peaks (n={len(peak_indices)})",
    )
    # shade each extracted beat window
    window_pre  = int(round(0.15 * fs))
    window_post = int(round(0.65 * fs))
    for p in peak_indices:
        s0 = (p - window_pre)  / fs
        s1 = (p + window_post) / fs
        if 0 <= s0 and s1 <= t_sig[-1]:
            ax1.axvspan(s0, s1, alpha=0.08, color="#42A5F5")
    ax1.set_title(
        f"Adaptive Peak Detection  "
        f"(height: median+0.5×MAD, distance: {int(fs*60/220)} samples, "
        f"prominence: 20% range)",
        fontsize=9, pad=4,
    )
    ax1.set_xlabel("Time (s)", fontsize=9)
    ax1.set_ylabel("Amplitude (a.u.)", fontsize=9)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.2, linewidth=0.5)

    # ── Panel 2 (bottom-left): all beats overlaid ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for beat in beats:
        ax2.plot(t_beat, beat, color="#42A5F5", linewidth=0.6, alpha=0.35)
    ax2.plot(t_beat, np.mean(beats, axis=0),
             color="#1565C0", linewidth=1.8, label="Mean beat")
    ax2.set_title(
        f"All Beats Overlay  (n={n_beats}, resampled to {target_length} pts)",
        fontsize=10, pad=4,
    )
    ax2.set_xlabel("Normalised time (0–1)", fontsize=9)
    ax2.set_ylabel("Amplitude (a.u.)", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, linewidth=0.5)

    # ── Panel 3 (bottom-right): mean ± 1 std envelope ────────────────────────
    ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)
    mean_beat = np.mean(beats, axis=0)
    std_beat  = np.std(beats,  axis=0)
    ax3.plot(t_beat, mean_beat, color="#1565C0", linewidth=1.8, label="Mean")
    ax3.fill_between(
        t_beat,
        mean_beat - std_beat,
        mean_beat + std_beat,
        alpha=0.25, color="#42A5F5", label="±1 SD",
    )
    ax3.set_title("Mean ± 1 SD Envelope", fontsize=10, pad=4)
    ax3.set_xlabel("Normalised time (0–1)", fontsize=9)
    ax3.set_ylabel("Amplitude (a.u.)", fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2, linewidth=0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig
