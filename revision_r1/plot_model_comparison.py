"""
Revision R1 — Fig. 7 (fig5_model_comparison): leak-free macro-F1 per model as a
dot-and-interval plot (mean with bootstrap 95 % CI over the 50 paired runs),
sorted, coloured by route (deep waveform vs classical Gaussian-feature).

Design (dataviz skill): the data's job is *magnitude with uncertainty* -> dot +
interval, one axis; thin marks; recessive grid; identity by colour AND by a
direct text label (never colour alone); text in ink tokens, not series colour;
palette = validated categorical slots 1 (blue) and 2 (orange) of the reference
palette; the SD is shown as a faint whisker so readers can see it is wider than
the CI (Reviewer 2: mean ± SD is not an uncertainty interval).

Input : outputs/master_ranking.csv   (from analyze_master.py)
Output: outputs/fig5_model_comparison.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

DISPLAY = {"resnet": "ResNet1D (plain)", "resnet_gated": "ResNet1D + Gated",
           "resnet_gc": "ResNet1D + GC", "resnet_gc_gated": "ResNet1D + GC + Gated",
           "protonet": "Prototypical Network", "inception": "Inception1D",
           "cnn_aug": "1D-CNN + 6-mode aug", "cnn_noaug": "1D-CNN (no aug)",
           "transformer": "Patch Transformer", "XGBoost": "XGBoost",
           "RandomForest": "Random Forest", "SVM": "SVM"}
CLASSICAL = {"XGBoost", "RandomForest", "SVM"}
BLUE, ORANGE = "#2a78d6", "#eb6834"          # reference palette slots 1, 2 (light surface)
INK, INK2, GRID = "#1a1a19", "#5f5e5a", "#e3e2dc"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    r = pd.read_csv(OUT / "master_ranking.csv").sort_values("macro_f1_mean").reset_index(drop=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, ax = plt.subplots(figsize=(7.16 * 0.82, 0.32 * len(r) + 0.9))
    y = range(len(r))
    for i, row in r.iterrows():
        c = ORANGE if row.model in CLASSICAL else BLUE
        # faint SD whisker behind the CI (orientation only)
        ax.plot([row.macro_f1_mean - row.macro_f1_sd, row.macro_f1_mean + row.macro_f1_sd], [i, i],
                color=c, alpha=0.22, lw=4, solid_capstyle="butt", zorder=1)
        ax.plot([row.macro_f1_lo, row.macro_f1_hi], [i, i], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(row.macro_f1_mean, i, "o", ms=6, mfc=c, mec="white", mew=1.2, zorder=3)
    xmax = float((r.macro_f1_mean + r.macro_f1_sd).max())
    for i, row in r.iterrows():
        ax.text(xmax + 0.006, i, f"{row.macro_f1_mean:.3f}", va="center", ha="left",
                fontsize=7.5, color=INK2)
    ax.set_yticks(list(y)); ax.set_yticklabels([DISPLAY.get(m, m) for m in r.model], color=INK)
    ax.set_xlabel("Leak-free macro-F1 (mean; bootstrap 95 % CI; ±1 SD band)")
    ax.set_xlim(round(float((r.macro_f1_mean - r.macro_f1_sd).min()), 2) - 0.01, xmax + 0.045)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", length=0)
    # legend: identity by colour + text (always present for >= 2 series)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, lw=2, label="deep model (raw waveform)"),
                       Line2D([], [], marker="o", color=ORANGE, lw=2, label="classical model (9 Gaussian features)")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5_model_comparison.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    print("written:", OUT / "fig5_model_comparison.pdf")


if __name__ == "__main__":
    main()
