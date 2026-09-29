"""
TCM Pulse — 3-Gaussian Feature Extraction
==========================================
Phase 2 / Feature Extraction

Model
-----
Each normalised beat is fitted by a sum of three Gaussians:

    f(t) = Σ_{i=1}^{3}  A_i · exp( −(t − μ_i)² / (2 σ_i²) )

Parameters (9 total per beat):
    θ = (A1, μ1, σ1,  A2, μ2, σ2,  A3, μ3, σ3)

Physical interpretation (TCM pulse morphology):
    G1 — percussion wave   (main systolic upstroke,   μ ≈ 0.15–0.25)
    G2 — tidal wave        (systolic shoulder/notch,   μ ≈ 0.35–0.55)
    G3 — diastolic wave    (diastolic decay,            μ ≈ 0.55–0.80)

Fitting strategy
----------------
* Input beat is min-max normalised to [0, 1] before fitting
  for numerical stability.
* curve_fit is called with ``method='trf'`` (Trust Region Reflective)
  and explicit bounds to prevent unphysical solutions.
* Five initial conditions with different μ placements are tried;
  the fit with the highest R² is kept.
* If ALL trials fail (RuntimeError), returns (None, False, nan).
* Convergence failure rate is reported in build_dataset.py.

Usage
-----
>>> from src.features.gaussian_fit import fit_three_gaussians
>>> params, ok, r2 = fit_three_gaussians(beat_200pts)
>>> # params: np.ndarray shape (9,) or None
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Model function
# ═══════════════════════════════════════════════════════════════════════════════

def three_gaussian(
    t: np.ndarray,
    A1: float, mu1: float, s1: float,
    A2: float, mu2: float, s2: float,
    A3: float, mu3: float, s3: float,
) -> np.ndarray:
    """Sum of three Gaussian functions evaluated at array *t*.

    All parameters must be non-negative; *s* values must be > 0.
    """
    def _g(A: float, mu: float, s: float) -> np.ndarray:
        return A * np.exp(-0.5 * ((t - mu) / s) ** 2)
    return _g(A1, mu1, s1) + _g(A2, mu2, s2) + _g(A3, mu3, s3)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. R² helper
# ═══════════════════════════════════════════════════════════════════════════════

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Fitting with multiple restarts
# ═══════════════════════════════════════════════════════════════════════════════

# Five initial conditions — varied μ placements to escape local minima.
# Rows: [mu1, mu2, mu3], [A1, A2, A3], [s1, s2, s3]
_INIT_CONDITIONS = [
    ([0.20, 0.45, 0.72], [0.85, 0.40, 0.25], [0.07, 0.09, 0.11]),
    ([0.15, 0.38, 0.65], [0.90, 0.50, 0.30], [0.05, 0.08, 0.12]),
    ([0.22, 0.50, 0.75], [0.80, 0.35, 0.20], [0.08, 0.10, 0.13]),
    ([0.18, 0.42, 0.70], [0.85, 0.45, 0.25], [0.06, 0.09, 0.12]),
    ([0.12, 0.35, 0.60], [0.95, 0.55, 0.35], [0.07, 0.11, 0.14]),
]

_BOUNDS_LOW  = [0.0, 0.0, 0.01,  0.0, 0.0, 0.01,  0.0, 0.0, 0.01]
_BOUNDS_HIGH = [2.0, 1.0, 0.50,  2.0, 1.0, 0.50,  2.0, 1.0, 0.50]


def fit_three_gaussians(
    beat: np.ndarray,
    n_restarts: int = 5,
) -> Tuple[Optional[np.ndarray], bool, float]:
    """Fit a 3-Gaussian model to one normalised beat waveform.

    Parameters
    ----------
    beat : np.ndarray, shape (N,)
        A single beat waveform (output of ``normalize_length``).
        Does **not** need to be pre-normalised; the function normalises
        internally to [0, 1] for stable optimisation.
    n_restarts : int
        Number of initial conditions to try (1–5). Default 5.

    Returns
    -------
    params : np.ndarray, shape (9,) or None
        Best-fit parameters  (A1, μ1, σ1, A2, μ2, σ2, A3, μ3, σ3).
        ``None`` if all restarts failed to converge.
    converged : bool
        True if at least one restart succeeded.
    r2 : float
        R² of the best fit.  ``np.nan`` if convergence failed.

    Notes
    -----
    Parameters are fitted on the amplitude-normalised beat (min-max [0,1])
    and on the time axis t ∈ [0, 1].  Downstream users should be aware
    that *A* and the signal scale are decoupled from absolute amplitude.
    """
    beat = np.asarray(beat, dtype=np.float64)
    t = np.linspace(0.0, 1.0, len(beat))

    # Min-max normalise amplitude to [0, 1] for numerical stability
    b_min, b_max = beat.min(), beat.max()
    beat_norm = (beat - b_min) / max(b_max - b_min, 1e-10)

    best_params: Optional[np.ndarray] = None
    best_r2 = -np.inf

    for mus, As, sigmas in _INIT_CONDITIONS[:n_restarts]:
        p0 = [
            As[0], mus[0], sigmas[0],
            As[1], mus[1], sigmas[1],
            As[2], mus[2], sigmas[2],
        ]
        try:
            popt, _ = curve_fit(
                three_gaussian, t, beat_norm,
                p0=p0,
                bounds=(_BOUNDS_LOW, _BOUNDS_HIGH),
                maxfev=5000,
                method="trf",
            )
            r2 = compute_r2(beat_norm, three_gaussian(t, *popt))
            if r2 > best_r2:
                best_r2 = r2
                best_params = popt
        except (RuntimeError, ValueError):
            continue

    if best_params is None:
        return None, False, np.nan

    return np.asarray(best_params, dtype=np.float64), True, best_r2
