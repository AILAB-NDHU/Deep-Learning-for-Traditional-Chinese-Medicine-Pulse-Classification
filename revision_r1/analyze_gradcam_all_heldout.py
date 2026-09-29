"""
Revision R1 — Grad-CAM on ALL held-out beats (Reviewer 2 Major 6; Reviewer 1 minor).

For every saved fold model (outputs/weights/<model>__leakfree__r<rep>_f<fold>.pt)
compute the 1D Grad-CAM (same GradCAM1D as TCM/gradcam_analysis.py) for every
beat of that fold's TEST set — correct and incorrect predictions alike — and
record the peak-activation location.  Nothing is selected by confidence.

Outputs (outputs/gradcam/):
  gradcam_peaks_<model>.csv        one row per (repeat, fold, beat): true, pred, correct,
                                   max-prob, peak position (0..1 of beat), CAM mass in
                                   early/mid/late thirds
  gradcam_summary_<model>.md       per-class peak-location distribution (correct vs
                                   incorrect), n per cell, mean-CAM figure
  fig_gradcam_mean_<model>.pdf/png mean CAM per true class, correct vs incorrect,
                                   with n in the panel titles
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
HERE = Path(__file__).resolve().parent
TCM = Path(os.environ.get("TCM_ROOT", HERE.parent / "TCM")).resolve()
sys.path.insert(0, str(TCM)); sys.path.insert(0, str(TCM / "src"))
DATA = Path(os.environ.get("TCM_DATA", TCM / "outputs" / "_intermediate")).resolve()
RES = Path(os.environ.get("RES_DIR", HERE / "outputs")).resolve()
W_DIR = RES / "weights"
OUT = RES / "gradcam"
OUT.mkdir(parents=True, exist_ok=True)

from train_advanced_models import PulseResNet            # noqa: E402
from train_gated_models import PulseResNetGC             # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Slippery", "Tight", "Wiry"]
MODELS = {"resnet_gc": (PulseResNetGC, "gc3"),           # last GC module (as in the paper)
          "resnet": (PulseResNet, "stage3")}             # last residual stage


class GradCAM1D:                                          # verbatim from TCM/gradcam_analysis.py
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model, self._acts, self._grads = model, None, None
        target_layer.register_forward_hook(self._save_acts)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, m, inp, out):
        self._acts = out.detach()

    def _save_grads(self, m, gin, gout):
        self._grads = gout[0].detach()

    def compute(self, x, class_indices):
        self.model.eval(); self.model.zero_grad()
        with torch.enable_grad():
            out = self.model(x)
            cls = torch.tensor(class_indices, device=x.device, dtype=torch.long)
            out[torch.arange(len(out)), cls].sum().backward()
        weights = self._grads.mean(dim=-1)
        cam = F.relu(torch.einsum("bc,bcl->bl", weights, self._acts))
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-1], mode="linear",
                            align_corners=False).squeeze(1)
        lo, hi = cam.min(-1, keepdim=True)[0], cam.max(-1, keepdim=True)[0]
        return ((cam - lo) / (hi - lo + 1e-8)).cpu().numpy(), out.softmax(1).detach().cpu().numpy()


def resolve_layer(model, name):
    layer = model
    for part in name.split("."):
        layer = getattr(layer, part)
    return layer


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet_gc", choices=list(MODELS))
    ap.add_argument("--protocol", default="leakfree")
    args = ap.parse_args()
    cls, layer_name = MODELS[args.model]

    beats = np.load(DATA / "beats_matrix.npy").astype(np.float32)
    ds = pd.read_csv(DATA / "dataset.csv")
    y_all = pd.Categorical(ds["pulse_type_en"], categories=CLASS_NAMES).codes
    subj = ds["subject_id"].values

    files = sorted(W_DIR.glob(f"{args.model}__{args.protocol}__r*_f*.pt"))
    if not files:
        sys.exit(f"no weights for {args.model}/{args.protocol} in {W_DIR}")
    print(f"{len(files)} fold models  |  device={DEVICE}")

    rows, cams = [], []
    for wf in files:
        rf = wf.stem.split("__")[-1]; rep, fold = (int(v) for v in rf[1:].split("_f"))
        pz = np.load(HERE / "outputs" / "preds" / f"{args.model}__{args.protocol}__{rf}.npz")
        te = pz["test_idx"]
        model = cls(len(CLASS_NAMES)).to(DEVICE)
        model.load_state_dict(torch.load(wf, map_location=DEVICE)); model.eval()
        gc = GradCAM1D(model, resolve_layer(model, layer_name))
        X = torch.from_numpy(beats[te][:, None, :]).to(DEVICE)
        for s in range(0, len(te), 256):
            xb = X[s:s + 256]
            with torch.no_grad():
                pred = model(xb).argmax(1).cpu().numpy()
            cam, prob = gc.compute(xb, pred)          # CAM for the PREDICTED class (standard)
            idx = te[s:s + 256]
            for i in range(len(idx)):
                c = cam[i]; n = len(c)
                rows.append(dict(repeat=rep, fold=fold, beat=int(idx[i]), subject=subj[idx[i]],
                                 true=int(y_all[idx[i]]), pred=int(pred[i]),
                                 correct=bool(pred[i] == y_all[idx[i]]),
                                 max_prob=float(prob[i].max()),
                                 peak_pos=float(np.argmax(c) / (n - 1)),
                                 mass_early=float(c[: n // 3].sum() / c.sum()),
                                 mass_mid=float(c[n // 3: 2 * n // 3].sum() / c.sum()),
                                 mass_late=float(c[2 * n // 3:].sum() / c.sum())))
                cams.append(c.astype(np.float32))
        print(f"  r{rep} f{fold}: {len(te)} beats", flush=True)

    df = pd.DataFrame(rows); C = np.stack(cams)
    df.to_csv(OUT / f"gradcam_peaks_{args.model}.csv", index=False)
    np.save(OUT / f"gradcam_cams_{args.model}.npy", C)

    # ── summary ──────────────────────────────────────────────────────────────
    L = [f"# Grad-CAM over all held-out beats — {args.model} ({args.protocol})\n",
         f"{len(df)} beat-level CAMs from {len(files)} fold models "
         f"({df.repeat.nunique()} repeats × {df.fold.nunique()} folds); CAM taken w.r.t. the predicted class; "
         "no confidence-based selection.\n"]
    L.append("## Peak-activation position (fraction of beat window, 0 = window start) by TRUE class and correctness\n")
    L.append("| true class | correct? | n | peak median [IQR] | mass early | mass mid | mass late | share with peak in early / mid / late third |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ci, cn in enumerate(CLASS_NAMES):
        for corr in (True, False):
            g = df[(df.true == ci) & (df.correct == corr)]
            if len(g) == 0:
                L.append(f"| {cn} | {'correct' if corr else 'incorrect'} | 0 | – | – | – | – | – |"); continue
            thirds = np.histogram(g.peak_pos, bins=[0, 1 / 3, 2 / 3, 1.0001])[0] / len(g) * 100
            L.append(f"| {cn} | {'correct' if corr else 'incorrect'} | {len(g)} | "
                     f"{g.peak_pos.median():.2f} [{g.peak_pos.quantile(.25):.2f}, {g.peak_pos.quantile(.75):.2f}] | "
                     f"{g.mass_early.mean():.2f} | {g.mass_mid.mean():.2f} | {g.mass_late.mean():.2f} | "
                     f"{thirds[0]:.0f}% / {thirds[1]:.0f}% / {thirds[2]:.0f}% |")
    L.append("\n## Does the class-specific pattern hold on misclassified beats?\n")
    L.append("| true class | peak median (correct) | peak median (incorrect) | Δ | Kolmogorov–Smirnov p |\n|---|---|---|---|---|")
    from scipy import stats
    for ci, cn in enumerate(CLASS_NAMES):
        a = df[(df.true == ci) & df.correct].peak_pos; b = df[(df.true == ci) & ~df.correct].peak_pos
        if len(a) and len(b):
            p = stats.ks_2samp(a, b).pvalue
            L.append(f"| {cn} | {a.median():.2f} | {b.median():.2f} | {b.median() - a.median():+.2f} | {p:.3g} |")
    L.append("\n## Is the peak location driven by the predicted class rather than the true class?\n")
    L.append("| predicted class | n | peak median [IQR] |\n|---|---|---|")
    for ci, cn in enumerate(CLASS_NAMES):
        g = df[df.pred == ci]
        L.append(f"| {cn} | {len(g)} | {g.peak_pos.median():.2f} [{g.peak_pos.quantile(.25):.2f}, {g.peak_pos.quantile(.75):.2f}] |")

    # ── figure: mean CAM per true class, correct vs incorrect ────────────────
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})
    t = np.linspace(-0.15, 0.65, C.shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ci, (ax, cn) in enumerate(zip(axes, CLASS_NAMES)):
        for corr, colour, ls in ((True, "#1f4e79", "-"), (False, "#c0392b", "--")):
            m = (df.true == ci) & (df.correct == corr)
            if m.sum():
                mu = C[m.values].mean(0); sd = C[m.values].std(0)
                ax.plot(t, mu, colour, ls=ls, lw=1.4, label=f"{'correct' if corr else 'incorrect'} (n={m.sum()})")
                ax.fill_between(t, mu - sd, mu + sd, color=colour, alpha=0.12, lw=0)
        ax.set_title(cn); ax.set_xlabel("time from systolic peak (s)"); ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    axes[0].set_ylabel("mean Grad-CAM (0–1)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_gradcam_mean_{args.model}.{ext}", dpi=300)
    L.append(f"\nFigure: `fig_gradcam_mean_{args.model}.pdf` — mean ± SD CAM per true class, correct (solid) vs incorrect (dashed), n in legend.")

    (OUT / f"gradcam_summary_{args.model}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
