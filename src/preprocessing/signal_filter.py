"""
TCM Pulse Signal Preprocessing — 訊號去噪與基線飄移消除
=========================================================
Phase 1 / 回應 Reviewer 6：以 Butterworth 帶通濾波器取代 DWT

設計原則
--------
* 使用 ``scipy.signal.sosfiltfilt`` 零相位雙向濾波，不引入相位延遲。
* 帶通範圍 0.5–10 Hz：
    - 低截止 0.5 Hz：消除呼吸頻率 (~0.2 Hz) 引起的基線飄移。
    - 高截止 10.0 Hz：移除高頻電氣雜訊，保留脈波主要診斷諧波。
* SOS 格式設計濾波器：比傳統 BA 格式數值更穩定，適合 4 階以上。
* 輸入支援 ``np.ndarray`` / ``pd.Series``；輸出統一為 ``np.ndarray``。

快速使用
--------
>>> from src.preprocessing.signal_filter import bandpass_filter, plot_filter_comparison
>>> filtered = bandpass_filter(raw_signal, fs=500.0)
>>> fig = plot_filter_comparison(raw_signal, filtered, fs=500.0, pulse_type="弦脈")
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal


# ── Type alias ────────────────────────────────────────────────────────────────
SignalLike = Union[np.ndarray, pd.Series]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 核心濾波函式
# ═══════════════════════════════════════════════════════════════════════════════

def bandpass_filter(
    raw_signal: SignalLike,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 10.0,
    order: int = 4,
) -> np.ndarray:
    """Butterworth 帶通濾波器 — 消除基線飄移與高頻雜訊。

    使用 second-order sections (SOS) 格式設計濾波器，並以 ``sosfiltfilt``
    進行零相位雙向濾波，確保輸出訊號不含相位延遲，診斷時序特徵得以保全。

    Parameters
    ----------
    raw_signal : np.ndarray or pd.Series
        原始脈波時序訊號（一維）。
    fs : float
        取樣頻率 (Hz)。
    lowcut : float, optional
        低截止頻率 (Hz)，預設 0.5 Hz。
        消除呼吸頻率 (~0.2 Hz) 引起的基線飄移。
    highcut : float, optional
        高截止頻率 (Hz)，預設 10.0 Hz。
        移除高頻電氣雜訊，保留脈波主要診斷諧波。
    order : int, optional
        Butterworth 濾波器階數，預設 4。

    Returns
    -------
    np.ndarray
        與輸入等長的濾波後訊號（float64）。

    Raises
    ------
    ValueError
        若截止頻率超出有效範圍，或訊號長度不足以執行雙向濾波。

    Examples
    --------
    >>> import numpy as np
    >>> from src.preprocessing.signal_filter import bandpass_filter
    >>> rng = np.random.default_rng(0)
    >>> raw = np.sin(2 * np.pi * 1.2 * np.arange(1000) / 500) + 0.1 * rng.standard_normal(1000)
    >>> filtered = bandpass_filter(raw, fs=500.0)
    >>> filtered.shape
    (1000,)
    """
    sig = np.asarray(raw_signal, dtype=np.float64)

    if sig.ndim != 1:
        raise ValueError(
            f"僅支援一維訊號，目前輸入維度：{sig.ndim}。"
            "若為多通道資料，請逐通道呼叫此函式。"
        )

    nyquist = fs / 2.0
    if not (0.0 < lowcut < highcut < nyquist):
        raise ValueError(
            f"截止頻率需滿足 0 < lowcut < highcut < Nyquist ({nyquist:.2f} Hz)，"
            f"目前：lowcut={lowcut}, highcut={highcut}"
        )

    # sosfiltfilt 最短訊號需求：padlen ≈ 3 × (2×order + 1)
    min_len = 3 * (2 * order + 1) + 1
    if len(sig) < min_len:
        raise ValueError(
            f"訊號長度 ({len(sig)}) 過短，order={order} 時至少需要 {min_len} 個樣本。"
        )

    low = lowcut / nyquist
    high = highcut / nyquist

    # SOS 格式：相比傳統 (b, a) 係數數值更穩定，適合高階濾波器
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")

    # 零相位雙向濾波：forward pass + backward pass，相位誤差為零
    filtered = signal.sosfiltfilt(sos, sig)

    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 頻率響應工具
# ═══════════════════════════════════════════════════════════════════════════════

def get_filter_frequency_response(
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 10.0,
    order: int = 4,
    n_points: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """計算 Butterworth 帶通濾波器的頻率響應。

    Parameters
    ----------
    fs : float
        取樣頻率 (Hz)。
    lowcut, highcut : float
        截止頻率 (Hz)。
    order : int
        濾波器階數。
    n_points : int
        頻率軸解析度，預設 4096。

    Returns
    -------
    freqs : np.ndarray
        頻率軸 (Hz)，範圍 [0, fs/2]。
    magnitude_db : np.ndarray
        對應頻率響應增益 (dB)。
    """
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    w, h = signal.sosfreqz(sos, worN=n_points)
    freqs = w * nyquist / np.pi
    magnitude_db = 20.0 * np.log10(np.abs(h) + 1e-12)
    return freqs, magnitude_db


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 視覺化函式
# ═══════════════════════════════════════════════════════════════════════════════

def plot_filter_comparison(
    raw_signal: SignalLike,
    filtered_signal: SignalLike,
    fs: float,
    title: str = "TCM Pulse — Bandpass Filter Verification",
    pulse_type: str = "",
    lowcut: float = 0.5,
    highcut: float = 10.0,
    order: int = 4,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """並排視覺化「原始波形」與「濾波後波形」，含 PSD 頻域佐證。

    圖表佈局（2 × 2 grid）：

    ┌───────────────────────┬───────────────────────┐
    │  原始波形（時域）      │  濾波後波形（時域）    │
    ├───────────────────────┼───────────────────────┤
    │  時域疊加（形態驗證）  │  PSD 頻域對照         │
    └───────────────────────┴───────────────────────┘

    此函式專為回應審稿人「濾波是否破壞診斷形態學特徵」設計：
    - 子圖 1/2 side-by-side 直觀展示濾波效果。
    - 子圖 3 疊加對照，確認波峰位置與形態完整保留。
    - 子圖 4 PSD 顯示基線飄移（< 0.5 Hz）與高頻雜訊（> 10 Hz）
      已被移除，脈波主頻率成分未受影響。

    Parameters
    ----------
    raw_signal : np.ndarray or pd.Series
        原始脈波訊號。
    filtered_signal : np.ndarray or pd.Series
        濾波後訊號（由 ``bandpass_filter`` 產出）。
    fs : float
        取樣頻率 (Hz)。
    title : str
        圖表主標題。
    pulse_type : str
        脈象類型標籤，例如 ``"弦脈"``，顯示於主標題。
    lowcut, highcut : float
        截止頻率 (Hz)，用於 PSD 子圖標示截止線。
    order : int
        濾波器階數，顯示於子圖副標。
    save_path : str, optional
        輸出圖檔路徑（支援 .png / .pdf）。``None`` 表示不儲存。
    show : bool
        是否呼叫 ``plt.show()``；無顯示環境請設為 ``False``。

    Returns
    -------
    matplotlib.figure.Figure
        可進一步操作或儲存的圖表物件。
    """
    raw = np.asarray(raw_signal, dtype=np.float64)
    filt = np.asarray(filtered_signal, dtype=np.float64)
    n = len(raw)
    t = np.arange(n) / fs

    COLOR_RAW = "#E57373"   # 暖紅：原始
    COLOR_FILT = "#42A5F5"  # 藍：濾波後

    fig = plt.figure(figsize=(14, 9))
    full_title = title + (f"  [{pulse_type}]" if pulse_type else "")
    fig.suptitle(full_title, fontsize=13, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.55, wspace=0.35,
        top=0.93, bottom=0.08,
    )

    # ── Panel 1: Raw waveform (time domain) ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, raw, color=COLOR_RAW, linewidth=0.9, alpha=0.85)
    ax1.set_title("Raw Signal", fontsize=10, pad=4)
    ax1.set_xlabel("Time (s)", fontsize=9)
    ax1.set_ylabel("Amplitude (a.u.)", fontsize=9)
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    _annotate_stats(ax1, raw)

    # ── Panel 2: Filtered waveform (time domain) ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.plot(t, filt, color=COLOR_FILT, linewidth=0.9, alpha=0.85)
    ax2.set_title(
        f"Filtered Signal\n"
        f"Butterworth BP {lowcut}–{highcut} Hz, order={order}, zero-phase",
        fontsize=9, pad=4,
    )
    ax2.set_xlabel("Time (s)", fontsize=9)
    ax2.set_ylabel("Amplitude (a.u.)", fontsize=9)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    _annotate_stats(ax2, filt)

    # ── Panel 3: Overlay — morphology verification ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, raw,  color=COLOR_RAW,  linewidth=0.9, alpha=0.50, label="Raw")
    ax3.plot(t, filt, color=COLOR_FILT, linewidth=1.3, alpha=0.90, label="Filtered")
    ax3.set_title("Overlay — Morphology Verification", fontsize=10, pad=4)
    ax3.set_xlabel("Time (s)", fontsize=9)
    ax3.set_ylabel("Amplitude (a.u.)", fontsize=9)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.25, linewidth=0.5)

    # ── Panel 4: Power Spectral Density comparison ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    nperseg = min(256, max(n // 4, 32))
    f_r, p_r = signal.welch(raw,  fs=fs, nperseg=nperseg)
    f_f, p_f = signal.welch(filt, fs=fs, nperseg=nperseg)
    ax4.semilogy(f_r, p_r, color=COLOR_RAW,  linewidth=0.9, alpha=0.65, label="Raw PSD")
    ax4.semilogy(f_f, p_f, color=COLOR_FILT, linewidth=1.1, alpha=0.90, label="Filtered PSD")
    ax4.axvspan(0,       lowcut,  alpha=0.08, color="gray")
    ax4.axvspan(highcut, fs / 2,  alpha=0.08, color="gray")
    ax4.axvline(x=lowcut,  color="gray", linestyle="--", linewidth=0.8)
    ax4.axvline(
        x=highcut, color="gray", linestyle="--", linewidth=0.8,
        label=f"Cutoff ({lowcut}/{highcut} Hz)",
    )
    ax4.set_title("Power Spectral Density (PSD)", fontsize=10, pad=4)
    ax4.set_xlabel("Frequency (Hz)", fontsize=9)
    ax4.set_ylabel("PSD (V²/Hz, log)", fontsize=9)
    ax4.set_xlim(0, min(fs / 2, 30))
    ax4.legend(fontsize=7, loc="upper right")
    ax4.grid(True, alpha=0.25, linewidth=0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_filter_response(
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 10.0,
    order: int = 4,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """繪製 Butterworth 帶通濾波器的頻率響應曲線（Bode Plot）。

    可附於論文附錄，作為濾波器設計合理性的佐證。

    Parameters
    ----------
    fs : float
        取樣頻率 (Hz)。
    lowcut, highcut : float
        截止頻率 (Hz)。
    order : int
        濾波器階數。
    save_path : str, optional
        輸出圖檔路徑。
    show : bool
        是否顯示互動視窗。

    Returns
    -------
    matplotlib.figure.Figure
    """
    freqs, mag_db = get_filter_frequency_response(fs, lowcut, highcut, order)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, mag_db, color="#1565C0", linewidth=1.5, label="|H(f)|")
    ax.axvline(
        x=lowcut,  color="#E53935", linestyle="--", linewidth=1.0,
        label=f"Low cutoff {lowcut} Hz",
    )
    ax.axvline(
        x=highcut, color="#E53935", linestyle=":",  linewidth=1.0,
        label=f"High cutoff {highcut} Hz",
    )
    ax.axhline(y=-3, color="#9E9E9E", linestyle="-.", linewidth=0.8, label="-3 dB")
    ax.axhspan(-3, 5, alpha=0.06, color="#42A5F5", label="Passband")
    ax.set_title(
        f"Butterworth Bandpass Filter — Frequency Response\n"
        f"(order={order}, {lowcut}–{highcut} Hz, fs={fs} Hz)",
        fontsize=11,
    )
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Gain (dB)", fontsize=10)
    ax.set_xlim(0, min(fs / 2, 30))
    ax.set_ylim(-80, 5)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 私有輔助函式
# ═══════════════════════════════════════════════════════════════════════════════

def _annotate_stats(ax: plt.Axes, sig: np.ndarray) -> None:
    """Annotate peak-to-peak (PP) and RMS in the top-right corner."""
    ptp = float(np.ptp(sig))
    rms = float(np.sqrt(np.mean(sig ** 2)))
    ax.text(
        0.97, 0.95,
        f"PP: {ptp:.3f}\nRMS: {rms:.4f}",
        transform=ax.transAxes,
        fontsize=7, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.65),
    )
