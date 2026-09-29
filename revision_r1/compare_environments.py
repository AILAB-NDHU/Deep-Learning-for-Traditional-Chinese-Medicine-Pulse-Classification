"""
Revision R1 — replication across environments.

Compares per-model leak-free macro-F1 (mean over the 50 paired runs) between
  A. original submission   (_working_archive/stat_rerun_raw.csv; server, Jul 2026)
  B. local rerun           (revision_r1/outputs/deep_rerun_raw_r*.csv; RTX 5080 laptop)
  C. server rerun          (revision_r1/outputs_server/deep_rerun_raw_r*.csv; RTX 4090)  [if present]
and reports, per model: means, mean paired |Δ| per run, and whether the ranking
changed.  Used for Appendix A.7 and the Response to Reviewers ("what changed and why").
Note: B and C use mu-sorted Gaussian features for the classical models; A did not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARCH = HERE.parent / "_working_archive" / "stat_rerun_raw.csv"
LOCAL = sorted((HERE / "outputs").glob("deep_rerun_raw_r*.csv"))
SERVER = sorted((HERE / "outputs_server").glob("deep_rerun_raw_r*.csv"))
DISPLAY = {"resnet": "ResNet1D (plain)", "resnet_gated": "ResNet1D + Gated", "resnet_gc": "ResNet1D + GC",
           "resnet_gc_gated": "ResNet1D + GC + Gated", "protonet": "Prototypical Network", "inception": "Inception1D",
           "cnn_aug": "1D-CNN + 6-mode aug", "cnn_noaug": "1D-CNN (no aug)", "transformer": "Patch Transformer",
           "XGBoost": "XGBoost", "RandomForest": "Random Forest", "SVM": "SVM",
           "cnn_aug_nosampler": "1D-CNN + aug, no sampler", "cnn_noaug_nosampler": "1D-CNN no aug, no sampler"}


def load(files):
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.drop_duplicates(["model", "protocol", "repeat", "fold"], keep="last")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    envs = {"A: original (server, Jul-2026)": pd.read_csv(ARCH), "B: local rerun (RTX 5080)": load(LOCAL)}
    if SERVER:
        envs["C: server rerun (RTX 4090)"] = load(SERVER)
    L = ["# Replication across environments — leak-free macro-F1\n"]
    for proto in ["leakfree", "optimistic"]:
        tabs = {}
        for name, df in envs.items():
            s = df[df.protocol == proto].set_index(["model", "repeat", "fold"])["macro_f1"]
            if len(s):
                tabs[name] = s
        if len(tabs) < 2:
            continue
        L.append(f"\n## Protocol: {proto}\n")
        names = list(tabs)
        models = sorted(set.intersection(*[set(t.index.get_level_values(0)) for t in tabs.values()]),
                        key=lambda m: -tabs[names[0]].xs(m).mean())
        hdr = "| Model | " + " | ".join(f"{n} mean ± SD" for n in names) + " | " + \
              " | ".join(f"mean paired abs diff {names[0][0]}→{n[0]}" for n in names[1:]) + " |"
        L.append(hdr); L.append("|" + "---|" * (1 + len(names) + len(names) - 1))
        for m in models:
            cells = []
            for n in names:
                x = tabs[n].xs(m); cells.append(f"{x.mean():.3f} ± {x.std(ddof=1):.3f}")
            diffs = []
            for n in names[1:]:
                j = tabs[names[0]].xs(m).to_frame("a").join(tabs[n].xs(m).to_frame("b"), how="inner")
                diffs.append(f"{(j.a - j.b).abs().mean():.3f} (n={len(j)})")
            L.append(f"| {DISPLAY.get(m, m)} | " + " | ".join(cells) + " | " + " | ".join(diffs) + " |")
        # ranking comparison
        L.append("\nRanking by mean macro-F1:\n")
        for n in names:
            order = sorted(models, key=lambda m: -tabs[n].xs(m).mean())
            L.append(f"- **{n}**: " + " > ".join(DISPLAY.get(m, m) for m in order))
        # max |Δ mean| across models
        for n in names[1:]:
            d = max(abs(tabs[names[0]].xs(m).mean() - tabs[n].xs(m).mean()) for m in models)
            L.append(f"\nLargest change in a model's mean macro-F1, {names[0][0]}→{n[0]}: **{d:.3f}**")
    out = HERE / "outputs" / "environment_replication.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)); print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
