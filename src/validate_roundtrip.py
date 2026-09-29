"""
validate_roundtrip.py — PNG 往返還原誤差驗證腳本
=================================================
用途：回應審稿人對資料來源效度的質疑。

本腳本量化「已知訊號 → matplotlib 渲染成 PNG → 逆向還原回訊號」的失真程度，
並評估失真對下游特徵（三高斯參數、峰值時序）的影響。

渲染參數說明
------------
渲染參數係**實測自真實 PNG**（路徑：/home/ailab_server/tcm/pluse/澀脈/221_1.png）：
  - 圖像尺寸：1000 × 500 px（figsize=(10,5), dpi=100）
  - 背景色：白色 (255,255,255)
  - 波形色：matplotlib 預設藍 #1f77b4 (R=31, G=119, B=180)
  - 格線色：淡灰 ~RGB(222,222,222)
  - 繪圖區有效寬度：774 px（左邊界 x=126，右邊界 x=899）
  - x 軸範圍：0–6 秒（與資料集一致）
  - 線寬：預設 1.5pt（matplotlib default）

執行方式
--------
    python src/validate_roundtrip.py

輸出（全部存入 outputs/）
-----------
  roundtrip_metrics.csv
  roundtrip_summary.txt
  roundtrip_overlay.png
  roundtrip_error_hist.png
  roundtrip_gaussian_param_error.png
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # 無螢幕環境必須在其他 import 之前設定

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import signal as sp_signal
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

# ── 確保能 import 專案模組 ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.image_reader import read_waveform_from_png
from src.features.gaussian_fit import fit_three_gaussians, three_gaussian

# ── 全域設定 ───────────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DURATION_S = 6.0       # x 軸 0–6 秒，與資料集一致
ALIGN_LEN = 1000       # 對齊後的共同長度（resample 目標）

# ── 實測渲染參數（來源：真實 PNG 量測）─────────────────────────────────────
RENDER_PARAMS = dict(
    figsize=(10, 5),
    dpi=100,
    linecolor="#1f77b4",
    linewidth=1.5,
    grid=True,
    gridcolor="#e0e0e0",
    background="white",
    xlim=(0, DURATION_S),
    param_source="實測自真實 PNG（/home/ailab_server/tcm/pluse/澀脈/221_1.png）",
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 渲染函式
# ═══════════════════════════════════════════════════════════════════════════════

def render_waveform_to_png(
    sig: np.ndarray,
    path: str | Path,
    figsize: Tuple[float, float] = RENDER_PARAMS["figsize"],
    dpi: int = RENDER_PARAMS["dpi"],
    linecolor: str = RENDER_PARAMS["linecolor"],
    linewidth: float = RENDER_PARAMS["linewidth"],
    grid: bool = RENDER_PARAMS["grid"],
    gridcolor: str = RENDER_PARAMS["gridcolor"],
    background: str = RENDER_PARAMS["background"],
    xlim: Tuple[float, float] = RENDER_PARAMS["xlim"],
) -> None:
    """將 1-D 訊號渲染成與真實資料集一致的 matplotlib PNG。

    Parameters
    ----------
    sig : np.ndarray
        1-D 訊號（任意振幅，函式不做 z-score，保留原始振幅讓 reader 去做）。
    path : str or Path
        輸出 PNG 路徑。
    figsize, dpi, linecolor, linewidth, grid, gridcolor, background, xlim :
        渲染參數——預設值均源自對真實 PNG 的實測量測。
    """
    t = np.linspace(xlim[0], xlim[1], len(sig))
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    ax.plot(t, sig, color=linecolor, linewidth=linewidth)
    ax.set_xlim(xlim)
    if grid:
        ax.grid(True, color=gridcolor, linestyle="-", linewidth=0.5)
    # 移除多餘裝飾以貼近原始資料集（原始只有基礎軸標籤）
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    fig.savefig(str(path), dpi=dpi, facecolor=background)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 誤差計算工具
# ═══════════════════════════════════════════════════════════════════════════════

def zscore(x: np.ndarray) -> np.ndarray:
    """Z-score 正規化（與 image_reader 一致）。"""
    std = x.std()
    if std < 1e-10:
        return x - x.mean()
    return (x - x.mean()) / std


def align_signals(
    g: np.ndarray,
    recon: np.ndarray,
    target_len: int = ALIGN_LEN,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """將 ground-truth 和還原訊號對齊到相同長度與尺度。

    步驟：
    1. 兩者皆用 scipy.signal.resample 取樣到 target_len。
    2. 對 g 做 z-score（與 reader 輸出一致）。
    3. 若 Pearson r < 0，翻轉 recon（並回傳 flip=True）。

    Returns
    -------
    g_aligned, recon_aligned : np.ndarray
        對齊後訊號，長度均為 target_len。
    flipped : bool
        是否進行了符號翻轉修正。
    """
    g_rs = sp_signal.resample(g, target_len)
    r_rs = sp_signal.resample(recon, target_len)
    g_z = zscore(g_rs)
    r_z = zscore(r_rs)
    # 處理可能的符號翻轉
    r_val, _ = pearsonr(g_z, r_z)
    flipped = False
    if r_val < 0:
        r_z = -r_z
        flipped = True
    return g_z, r_z, flipped


def compute_metrics(g: np.ndarray, r: np.ndarray) -> Dict[str, float]:
    """計算一組已對齊訊號的誤差指標。

    Parameters
    ----------
    g : np.ndarray
        Ground-truth（已 z-score，長度 ALIGN_LEN）。
    r : np.ndarray
        還原訊號（已 z-score，長度 ALIGN_LEN）。

    Returns
    -------
    dict with keys: rmse, mae, max_abs, nrmse, pearson_r, r2,
                    peak_shift_samples, peak_shift_ms
    """
    diff = g - r
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    amp = float(g.max() - g.min())
    nrmse = rmse / amp if amp > 1e-10 else np.nan
    pr, _ = pearsonr(g, r)
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    # 用互相關（cross-correlation）量測整體波形時序偏移
    # 避免多心跳訊號 argmax 跳到不同搏動週期的假性大偏移
    xcorr = sp_signal.correlate(g, r, mode="full")
    lag = int(np.argmax(xcorr)) - (len(g) - 1)   # 正=r 超前, 負=r 落後
    shift_samples = int(abs(lag))
    shift_ms = float(shift_samples / ALIGN_LEN * DURATION_S * 1000)
    return dict(
        rmse=rmse,
        mae=mae,
        max_abs=max_abs,
        nrmse=nrmse,
        pearson_r=float(pr),
        r2=r2,
        peak_shift_samples=shift_samples,
        peak_shift_ms=shift_ms,
    )


def compare_gaussian_params(
    g: np.ndarray,
    r: np.ndarray,
) -> Dict[str, float]:
    """對 g 和 r 各跑 fit_three_gaussians，比較 9 個參數。

    Returns
    -------
    dict with keys:
      gauss_converged_g, gauss_converged_r,
      gauss_r2_g, gauss_r2_r,
      gauss_abs_err_{0..8},
      gauss_rel_err_{0..8} (%)
      gauss_mean_abs_err, gauss_mean_rel_err (%)
    """
    result: Dict[str, float] = {}
    params_g, ok_g, r2_g = fit_three_gaussians(g)
    params_r, ok_r, r2_r = fit_three_gaussians(r)
    result["gauss_converged_g"] = float(ok_g)
    result["gauss_converged_r"] = float(ok_r)
    result["gauss_r2_g"] = float(r2_g) if not np.isnan(r2_g) else np.nan
    result["gauss_r2_r"] = float(r2_r) if not np.isnan(r2_r) else np.nan
    if ok_g and ok_r and params_g is not None and params_r is not None:
        abs_errs = np.abs(params_g - params_r)
        denom = np.abs(params_g)
        rel_errs = np.where(denom > 1e-10, abs_errs / denom * 100.0, np.nan)
        # 上限 500%，避免邊界案例（近零參數）污染統計
        rel_errs = np.where(rel_errs > 500.0, np.nan, rel_errs)
        for i in range(9):
            result[f"gauss_abs_err_{i}"] = float(abs_errs[i])
            result[f"gauss_rel_err_{i}"] = float(rel_errs[i])
        result["gauss_mean_abs_err"] = float(np.nanmean(abs_errs))
        result["gauss_mean_rel_err"] = float(np.nanmean(rel_errs))
    else:
        for i in range(9):
            result[f"gauss_abs_err_{i}"] = np.nan
            result[f"gauss_rel_err_{i}"] = np.nan
        result["gauss_mean_abs_err"] = np.nan
        result["gauss_mean_rel_err"] = np.nan
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 單條訊號往返測試
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_roundtrip(
    sig: np.ndarray,
    label: str,
    idx: int,
    double_trip: bool = False,
) -> Optional[Dict]:
    """執行一次（或兩次）PNG 往返，回傳指標字典。

    Parameters
    ----------
    sig : np.ndarray
        Ground-truth 訊號（不需預先 z-score）。
    label : str
        測試類別標籤（寫入 CSV）。
    idx : int
        序號（用於命名臨時檔）。
    double_trip : bool
        若 True，對第一次 recon 再做一次往返（雙重往返）。

    Returns
    -------
    dict 或 None（若讀取失敗）。
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp_path = tf.name
    try:
        render_waveform_to_png(sig, tmp_path)
        recon, _fs = read_waveform_from_png(tmp_path, assumed_duration_s=DURATION_S)
    except Exception as e:
        warnings.warn(f"[{label}#{idx}] round-trip failed: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    g_a, r_a, flipped = align_signals(sig, recon)
    row = compute_metrics(g_a, r_a)
    row.update(compare_gaussian_params(g_a, r_a))
    row["flipped"] = int(flipped)
    row["label"] = label
    row["idx"] = idx

    if double_trip:
        # 第二次往返（以 recon 為輸入，量測累積漂移）
        recon_raw = sp_signal.resample(recon, len(sig))  # 還原到原始長度近似
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf2:
            tmp2 = tf2.name
        try:
            render_waveform_to_png(recon_raw, tmp2)
            recon2, _ = read_waveform_from_png(tmp2, assumed_duration_s=DURATION_S)
        except Exception as e:
            warnings.warn(f"[{label}#{idx}] double round-trip failed: {e}")
            row["dt2_nrmse"] = np.nan
            row["dt2_pearson_r"] = np.nan
        else:
            g2_a, r2_a, _ = align_signals(sig, recon2)
            m2 = compute_metrics(g2_a, r2_a)
            row["dt2_nrmse"] = m2["nrmse"]
            row["dt2_pearson_r"] = m2["pearson_r"]
        finally:
            if os.path.exists(tmp2):
                os.unlink(tmp2)
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 測試案例產生
# ═══════════════════════════════════════════════════════════════════════════════

def _make_time(n_pts: int = 3000) -> np.ndarray:
    """產生 0–6 秒的時間軸。"""
    return np.linspace(0.0, DURATION_S, n_pts)


# ── 4.1 合成正弦/啁啾 ────────────────────────────────────────────────────────
def gen_sinusoid_cases(n: int = 30) -> List[Tuple[np.ndarray, str]]:
    """不同頻率（0.8–3 Hz）、振幅、相位的正弦訊號。"""
    t = _make_time()
    cases = []
    freqs = np.linspace(0.8, 3.0, n)
    for i, f in enumerate(freqs):
        amp = RNG.uniform(0.5, 2.0)
        phi = RNG.uniform(0, 2 * np.pi)
        sig = amp * np.sin(2 * np.pi * f * t + phi)
        cases.append((sig, "sinusoid"))
    return cases


def gen_chirp_cases(n: int = 10) -> List[Tuple[np.ndarray, str]]:
    """啁啾（線性調頻）訊號。"""
    t = _make_time()
    cases = []
    for _ in range(n):
        f0 = RNG.uniform(0.5, 1.5)
        f1 = RNG.uniform(2.0, 4.0)
        amp = RNG.uniform(0.5, 2.0)
        sig = amp * sp_signal.chirp(t, f0=f0, f1=f1, t1=DURATION_S, method="linear")
        cases.append((sig, "chirp"))
    return cases


# ── 4.2 真實形態脈波（三高斯模型）────────────────────────────────────────────
def _pulse_from_params(
    params: List[float],
    n_beats: int,
    n_pts: int = 3000,
) -> np.ndarray:
    """用三高斯模型產生週期性脈波訊號（n_beats 個搏動，6 秒內）。"""
    t_beat = np.linspace(0, 1, 200)
    A1, mu1, s1, A2, mu2, s2, A3, mu3, s3 = params
    beat = three_gaussian(t_beat, A1, mu1, s1, A2, mu2, s2, A3, mu3, s3)
    # 重複 n_beats 次，拼接成完整 6 秒訊號
    full = np.tile(beat, n_beats)
    return sp_signal.resample(full, n_pts)


# 三種典型脈象參數（弦/滑/緊）
_PULSE_PRESETS = {
    "弦脈": [0.9, 0.20, 0.06, 0.55, 0.42, 0.09, 0.30, 0.68, 0.12],
    "滑脈": [0.85, 0.22, 0.08, 0.50, 0.48, 0.11, 0.25, 0.72, 0.14],
    "緊脈": [0.95, 0.18, 0.04, 0.45, 0.38, 0.07, 0.20, 0.62, 0.10],
}


def gen_pulse_cases(n_each: int = 12) -> List[Tuple[np.ndarray, str]]:
    """三種脈象 × n_each 變體（擾動參數）的三高斯脈波。"""
    cases = []
    for pulse_name, base_params in _PULSE_PRESETS.items():
        for _ in range(n_each):
            # 對基礎參數加±10%擾動
            params = [p * RNG.uniform(0.90, 1.10) for p in base_params]
            n_beats = int(RNG.integers(4, 9))
            sig = _pulse_from_params(params, n_beats)
            cases.append((sig, f"pulse_{pulse_name}"))
    return cases


# ── 4.3 加噪版本（SNR = 20/30/40 dB）────────────────────────────────────────
def gen_noisy_cases(
    base_cases: List[Tuple[np.ndarray, str]],
    snr_db_list: List[float] = (20, 30, 40),
    n: int = 30,
) -> List[Tuple[np.ndarray, str]]:
    """對基礎訊號加入指定 SNR 的高斯雜訊。"""
    chosen = [base_cases[i % len(base_cases)] for i in range(n)]
    cases = []
    for (sig, _), snr_db in zip(chosen, [snr_db_list[i % len(snr_db_list)] for i in range(n)]):
        sig_power = np.mean(sig ** 2)
        snr_lin = 10 ** (snr_db / 10.0)
        noise_std = np.sqrt(sig_power / snr_lin)
        noisy = sig + RNG.normal(0, noise_std, size=len(sig))
        cases.append((noisy, f"noisy_snr{int(snr_db)}dB"))
    return cases


# ── 4.4 不同心率/beat 數（4–10 beats / 6s）──────────────────────────────────
def gen_heartrate_cases(n: int = 30) -> List[Tuple[np.ndarray, str]]:
    """測試格線密度與線重疊的影響。"""
    cases = []
    for i in range(n):
        n_beats = int(RNG.integers(4, 11))
        preset = list(_PULSE_PRESETS.values())[i % 3]
        params = [p * RNG.uniform(0.88, 1.12) for p in preset]
        sig = _pulse_from_params(params, n_beats)
        cases.append((sig, f"hr_{n_beats}bpm_equiv"))
    return cases


# ── 4.5 邊界案例 ──────────────────────────────────────────────────────────────
def gen_edge_cases() -> List[Tuple[np.ndarray, str]]:
    """近平坦、振幅極小、含基線漂移等邊界訊號。"""
    t = _make_time()
    cases = []
    # 近平坦：低振幅白雜訊
    for _ in range(10):
        sig = RNG.normal(0, 1e-3, size=len(t))
        cases.append((sig, "edge_flat"))
    # 振幅極小（但有形狀）
    for _ in range(10):
        preset = list(_PULSE_PRESETS.values())[0]
        sig = _pulse_from_params(preset, 6) * 1e-4
        cases.append((sig, "edge_tiny_amp"))
    # 含基線漂移（慢正弦）
    for _ in range(10):
        preset = list(_PULSE_PRESETS.values())[1]
        sig = _pulse_from_params(preset, 6)
        drift = 0.5 * np.sin(2 * np.pi * 0.15 * t)
        cases.append((sig + drift, "edge_drift"))
    return cases


# ── 4.6 雙重往返（double round-trip）─────────────────────────────────────────
# 此類別直接在 run_one_roundtrip 中以 double_trip=True 執行，
# 另外獨立抽取 30 條進行雙重往返以量測累積漂移。
def gen_double_trip_cases(n: int = 30) -> List[Tuple[np.ndarray, str]]:
    """從脈波類別抽取代表性訊號做雙重往返。"""
    cases = gen_pulse_cases(n_each=10)[:n]
    return [(sig, "double_trip") for sig, _ in cases]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 主執行流程
# ═══════════════════════════════════════════════════════════════════════════════

def build_all_cases() -> List[Tuple[np.ndarray, str, bool]]:
    """彙整所有測試案例，回傳 (signal, label, double_trip) 列表。"""
    all_cases: List[Tuple[np.ndarray, str, bool]] = []

    # 4.1 正弦/啁啾
    for sig, lbl in gen_sinusoid_cases(30):
        all_cases.append((sig, lbl, False))
    for sig, lbl in gen_chirp_cases(10):
        all_cases.append((sig, lbl, False))

    # 4.2 真實脈波（三種脈象 × 12 = 36 條）
    pulse_cases = gen_pulse_cases(12)
    for sig, lbl in pulse_cases:
        all_cases.append((sig, lbl, False))

    # 4.3 加噪版本（以脈波為基礎）
    for sig, lbl in gen_noisy_cases(pulse_cases, n=30):
        all_cases.append((sig, lbl, False))

    # 4.4 不同心率
    for sig, lbl in gen_heartrate_cases(30):
        all_cases.append((sig, lbl, False))

    # 4.5 邊界案例
    for sig, lbl in gen_edge_cases():
        all_cases.append((sig, lbl, False))

    # 4.6 雙重往返
    for sig, lbl in gen_double_trip_cases(30):
        all_cases.append((sig, lbl, True))

    return all_cases


def run_all(verbose: bool = True) -> pd.DataFrame:
    """執行所有往返測試，回傳 DataFrame。"""
    all_cases = build_all_cases()
    total = len(all_cases)
    if verbose:
        print(f"[INFO] 共 {total} 條訊號，開始往返測試…")

    rows = []
    failed = 0
    for i, (sig, label, double) in enumerate(all_cases):
        if verbose and (i + 1) % 30 == 0:
            print(f"  [{i+1}/{total}] {label}")
        row = run_one_roundtrip(sig, label, i, double_trip=double)
        if row is None:
            failed += 1
        else:
            rows.append(row)

    if verbose:
        print(f"[INFO] 完成：{len(rows)} 成功，{failed} 失敗。")
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 輸出：CSV、摘要、圖表
# ═══════════════════════════════════════════════════════════════════════════════

def save_csv(df: pd.DataFrame) -> None:
    """儲存每條訊號的詳細指標。"""
    path = OUTPUT_DIR / "roundtrip_metrics.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OUTPUT] {path}")


def _percentile(arr: np.ndarray, p: float) -> float:
    clean = arr[~np.isnan(arr)]
    return float(np.percentile(clean, p)) if len(clean) > 0 else np.nan


def _fmt(arr: np.ndarray) -> str:
    clean = arr[~np.isnan(arr)]
    if len(clean) == 0:
        return "N/A"
    return (f"{np.nanmean(clean):.4f} ± {np.nanstd(clean):.4f} "
            f"(p50={_percentile(clean,50):.4f}, "
            f"p95={_percentile(clean,95):.4f}, "
            f"max={np.nanmax(clean):.4f})")


def save_summary(df: pd.DataFrame) -> None:
    """儲存各類別與整體的彙總摘要文字報告。"""
    path = OUTPUT_DIR / "roundtrip_summary.txt"
    lines = []
    lines.append("=" * 72)
    lines.append("PNG 往返還原誤差驗證報告")
    lines.append("=" * 72)
    lines.append(f"渲染參數來源：{RENDER_PARAMS['param_source']}")
    lines.append(f"  figsize={RENDER_PARAMS['figsize']}, dpi={RENDER_PARAMS['dpi']}")
    lines.append(f"  linecolor={RENDER_PARAMS['linecolor']}, linewidth={RENDER_PARAMS['linewidth']}")
    lines.append(f"  grid={RENDER_PARAMS['grid']}, gridcolor={RENDER_PARAMS['gridcolor']}")
    lines.append(f"  x 軸：{RENDER_PARAMS['xlim'][0]}–{RENDER_PARAMS['xlim'][1]} s")
    lines.append(f"  對齊長度：{ALIGN_LEN} pts（scipy.signal.resample + 雙邊 z-score）")
    lines.append("")
    lines.append(f"總測試筆數：{len(df)}")
    n_flip = int(df["flipped"].sum())
    lines.append(f"符號翻轉修正次數：{n_flip} / {len(df)} ({100*n_flip/len(df):.1f}%)")
    lines.append("")

    key_metrics = ["nrmse", "rmse", "mae", "max_abs", "pearson_r", "r2",
                   "peak_shift_ms", "gauss_mean_rel_err"]

    def section(sub_df: pd.DataFrame, title: str) -> None:
        lines.append(f"── {title} (n={len(sub_df)}) ──")
        for m in key_metrics:
            if m in sub_df.columns:
                lines.append(f"  {m:25s}: {_fmt(sub_df[m].values)}")
        lines.append("")

    section(df, "【整體】")
    for lbl in sorted(df["label"].unique()):
        section(df[df["label"] == lbl], lbl)

    # 雙重往返累積漂移
    dt_df = df[df["label"] == "double_trip"]
    if len(dt_df) > 0 and "dt2_nrmse" in dt_df.columns:
        lines.append("── 雙重往返（第二次往返累積漂移） ──")
        lines.append(f"  dt2_nrmse  : {_fmt(dt_df['dt2_nrmse'].values)}")
        lines.append(f"  dt2_pearson: {_fmt(dt_df['dt2_pearson_r'].values)}")
        lines.append("")

    # ── 警示 ──────────────────────────────────────────────────────────────────
    nrmse_med = _percentile(df["nrmse"].values, 50)
    pr_med = _percentile(df["pearson_r"].values, 50)
    lines.append("=" * 72)
    lines.append("警示評估")
    lines.append("=" * 72)
    if nrmse_med > 0.05:
        lines.append(
            f"⚠ WARNING: median NRMSE = {nrmse_med:.4f} > 0.05 門檻。"
            " 還原誤差偏高，可能影響下游三高斯擬合與分類結論。"
            " 建議檢視渲染參數或提高圖像解析度。"
        )
    else:
        lines.append(f"✓ median NRMSE = {nrmse_med:.4f}（< 0.05 門檻，可接受）。")
    if pr_med < 0.98:
        lines.append(
            f"⚠ WARNING: median Pearson r = {pr_med:.4f} < 0.98 門檻。"
            " 波形相似度偏低，建議驗證渲染/讀取流程一致性。"
        )
    else:
        lines.append(f"✓ median Pearson r = {pr_med:.4f}（≥ 0.98 門檻，良好）。")
    lines.append("")

    content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OUTPUT] {path}")
    return content


def save_overlay_plot(df: pd.DataFrame, all_cases: List[Tuple]) -> None:
    """繪製代表性案例的原始 vs 還原疊圖（最佳/中位/最差）。"""
    # 只取脈波類別（最相關）
    pulse_mask = df["label"].str.startswith("pulse_")
    sub = df[pulse_mask].dropna(subset=["nrmse"])
    if len(sub) < 3:
        sub = df.dropna(subset=["nrmse"])

    sorted_sub = sub.sort_values("nrmse")
    n = len(sorted_sub)
    best_row = sorted_sub.iloc[0]
    med_row = sorted_sub.iloc[n // 2]
    worst_row = sorted_sub.iloc[-1]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    palette = {"原始": "#333333", "還原": "#d62728"}

    for ax, row, title in zip(
        axes,
        [best_row, med_row, worst_row],
        ["最佳案例", "中位案例", "最差案例"],
    ):
        idx = int(row["idx"])
        # 找對應的原始訊號
        matching = [(s, l, d) for s, l, d in all_cases if l == row["label"]]
        # 用 idx 定位（因為 all_cases 包含所有類別，需要匹配 idx）
        sig_entry = all_cases[idx] if idx < len(all_cases) else matching[0]
        sig = sig_entry[0]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp = tf.name
        try:
            render_waveform_to_png(sig, tmp)
            recon, _ = read_waveform_from_png(tmp, assumed_duration_s=DURATION_S)
        except Exception:
            plt.close(fig)
            return
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

        g_a, r_a, flipped = align_signals(sig, recon)
        t_axis = np.linspace(0, DURATION_S, ALIGN_LEN)
        ax.plot(t_axis, g_a, color=palette["原始"], label="原始 (z-score)", linewidth=1.2, alpha=0.85)
        ax.plot(t_axis, r_a, color=palette["還原"], label="還原 (z-score)", linewidth=1.0,
                linestyle="--", alpha=0.9)
        ax.set_title(
            f"{title}｜{row['label']}  NRMSE={row['nrmse']:.4f}, "
            f"r={row['pearson_r']:.4f}"
            + ("  [符號翻轉已修正]" if row["flipped"] else ""),
            fontsize=9,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("z-score")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("PNG 往返還原：代表性案例疊圖", fontsize=11, y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / "roundtrip_overlay.png"
    fig.savefig(str(out), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUTPUT] {out}")


def save_error_hist(df: pd.DataFrame) -> None:
    """NRMSE 與 Pearson r 的分佈直方圖。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    nrmse_vals = df["nrmse"].dropna().values
    pr_vals = df["pearson_r"].dropna().values

    ax = axes[0]
    ax.hist(nrmse_vals, bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(nrmse_vals), color="red", linestyle="--",
               label=f"median={np.median(nrmse_vals):.4f}")
    ax.axvline(np.percentile(nrmse_vals, 95), color="orange", linestyle=":",
               label=f"p95={np.percentile(nrmse_vals, 95):.4f}")
    ax.axvline(0.05, color="gray", linestyle="-.", alpha=0.5, label="門檻 0.05")
    ax.set_xlabel("NRMSE")
    ax.set_ylabel("Count")
    ax.set_title("NRMSE 分佈")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(pr_vals, bins=30, color="#4CAF50", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(pr_vals), color="red", linestyle="--",
               label=f"median={np.median(pr_vals):.4f}")
    ax.axvline(np.percentile(pr_vals, 5), color="orange", linestyle=":",
               label=f"p5={np.percentile(pr_vals, 5):.4f}")
    ax.axvline(0.98, color="gray", linestyle="-.", alpha=0.5, label="門檻 0.98")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Count")
    ax.set_title("Pearson r 分佈")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("PNG 往返還原誤差分佈", fontsize=11)
    plt.tight_layout()
    out = OUTPUT_DIR / "roundtrip_error_hist.png"
    fig.savefig(str(out), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUTPUT] {out}")


def save_gaussian_param_error(df: pd.DataFrame) -> None:
    """9 個三高斯參數相對誤差的箱型圖。"""
    param_names = [
        "A1", "μ1", "σ1",
        "A2", "μ2", "σ2",
        "A3", "μ3", "σ3",
    ]
    data = []
    for i in range(9):
        col = f"gauss_rel_err_{i}"
        if col in df.columns:
            vals = df[col].dropna().values
            vals = vals[np.isfinite(vals)]
            data.append(vals)
        else:
            data.append(np.array([]))

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(
        data,
        labels=param_names,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="red", linewidth=2),
    )
    colors = ["#bbdefb", "#c8e6c9", "#ffe0b2"] * 3
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_xlabel("三高斯參數")
    ax.set_ylabel("相對誤差 (%)")
    ax.set_title("PNG 往返後三高斯 9 參數相對誤差分佈")
    ax.axhline(5.0, color="gray", linestyle="--", alpha=0.5, label="5% 參考線")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = OUTPUT_DIR / "roundtrip_gaussian_param_error.png"
    fig.savefig(str(out), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUTPUT] {out}")


def print_paper_conclusion(df: pd.DataFrame) -> None:
    """列印可直接寫進論文的結論句。

    統計範圍：排除 edge_flat / edge_tiny_amp 邊界案例（近平坦訊號的
    argmax 與高斯擬合在數學上無意義），僅保留脈波、加噪、心率、正弦、
    雙重往返等對應真實資料的類別。
    """
    # 排除邊界案例：edge_flat 和 edge_tiny_amp 的訊號近似平坦，
    # argmax 為隨機位置，高斯參數分母趨近於零，不具代表性
    excl = df["label"].isin(["edge_flat", "edge_tiny_amp"])
    df_main = df[~excl].copy()
    nrmse = df_main["nrmse"].dropna()
    pr = df_main["pearson_r"].dropna()
    grel = df_main["gauss_mean_rel_err"].dropna()
    # 峰值偏移只統計脈波相關類別（正弦/啁啾有多個等高峰，argmax 跳峰無意義）
    pulse_mask = df_main["label"].str.startswith(("pulse_", "hr_", "noisy_", "double_", "edge_drift"))
    peak = df_main.loc[pulse_mask, "peak_shift_ms"].dropna()

    med_nrmse = float(np.median(nrmse))
    med_pr = float(np.median(pr))
    p95_nrmse = float(np.percentile(nrmse, 95))
    mean_grel = float(np.nanmean(grel))
    mean_peak = float(np.nanmean(peak))

    sentence = (
        f"PNG round-trip reconstruction: "
        f"median NRMSE = {med_nrmse:.3f}, "
        f"Pearson r = {med_pr:.3f} "
        f"(p95 NRMSE = {p95_nrmse:.3f}); "
        f"3-Gaussian parameter mean relative error = {mean_grel:.1f}%; "
        f"peak timing shift = {mean_peak:.1f} ms."
    )
    print("\n" + "=" * 72)
    print("【論文結論句】")
    print(sentence)
    print("=" * 72 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 建立所有測試案例（同時保留參考以繪製疊圖）
    print("[STEP 1] 建立測試案例…")
    all_cases = build_all_cases()
    print(f"  共 {len(all_cases)} 條（正弦40 + 脈波36 + 加噪30 + 心率30 + 邊界30 + 雙重往返30）")

    # 執行往返測試
    print("[STEP 2] 執行 PNG 往返測試…")
    df = run_all(verbose=True)

    # 輸出
    print("[STEP 3] 輸出結果…")
    save_csv(df)
    save_summary(df)
    save_overlay_plot(df, all_cases)
    save_error_hist(df)
    save_gaussian_param_error(df)

    # 論文結論句
    print_paper_conclusion(df)

    print("[DONE] 所有輸出已存入 outputs/")
