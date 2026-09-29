"""
GradCAM Analysis — PulseResNetGC (1D Pulse Waveform Classification)
====================================================================
Trains PulseResNetGC (best model so far, F1≈0.834) on folds 1–4 of the
subject-level CV and holds out fold 0 as the test set.  GradCAM hooks
are registered on `model.gc3` (last Global Context module, output B×256×25)
to produce class activation maps that highlight which waveform regions
drive each prediction.

GradCAM formula
---------------
  w_k = mean_t( ∂y^c / ∂A^k_t )     [global-avg-pool of gradients]
  CAM = ReLU( Σ_k  w_k · A^k )       [weighted sum of feature maps]

The 25-point CAM is upsampled to 200 points and overlaid on the waveform.

Output files (outputs/)
-----------------------
  gradcam_examples.png  — 3 classes × 4 individual examples (colored line)
  gradcam_mean.png      — 3-class side-by-side mean CAM (publication figure)
  resnet_gc_weights.pt  — saved model weights for this fold
"""

import math, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"

# Re-use model and training helpers
from src.train_gated_models import (
    PulseResNetGC,
    DEVICE, SEED, BATCH_SIZE, CLASS_ORDER,
    make_weighted_sampler, class_weights_tensor,
    get_scheduler, train_epoch, predict, evaluate,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

CMAP = "RdYlBu_r"   # red=high activation, blue=low
CLASS_ZH = {"Slippery": "Slippery", "Tight": "Tight", "Wiry": "Wiry"}


# ═════════════════════════════════════════════════════════════════════════════
# GradCAM for 1D convolutional models
# ═════════════════════════════════════════════════════════════════════════════

class GradCAM1D:
    """
    Class Activation Map via gradient weighting.

    Usage
    -----
        gc = GradCAM1D(model, model.gc3)
        cam = gc.compute(x_batch, class_indices)   # (B, L)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model  = model
        self._acts  = None
        self._grads = None
        target_layer.register_forward_hook(self._save_acts)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, m, inp, out):
        self._acts = out.detach()           # (B, C, L_feat)

    def _save_grads(self, m, gin, gout):
        self._grads = gout[0].detach()      # (B, C, L_feat)

    def compute(self, x: torch.Tensor, class_indices: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x             : (B, 1, 200) float tensor on DEVICE
        class_indices : (B,) integer array — target class per sample
                        (use predicted class for standard GradCAM)

        Returns
        -------
        cam : (B, 200) float32 array, values normalised to [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        with torch.enable_grad():
            out  = self.model(x)
            cls  = torch.tensor(class_indices, device=x.device, dtype=torch.long)
            # Sum each sample's target-class score — cross-sample gradients are zero
            score = out[torch.arange(len(out)), cls].sum()
            score.backward()

        # w_k = mean_t( grad ) → (B, C)
        weights = self._grads.mean(dim=-1)
        # CAM = ReLU( Σ_k w_k A_k ) → (B, L_feat)
        cam = torch.einsum("bc,bcl->bl", weights, self._acts)
        cam = F.relu(cam)

        # Upsample to original signal length
        cam = F.interpolate(
            cam.unsqueeze(1), size=x.shape[-1],
            mode="linear", align_corners=False,
        ).squeeze(1)                                    # (B, 200)

        # Per-sample min-max normalisation
        lo = cam.min(dim=-1, keepdim=True)[0]
        hi = cam.max(dim=-1, keepdim=True)[0]
        cam = (cam - lo) / (hi - lo + 1e-8)

        return cam.cpu().numpy()


# ═════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

def colored_line(ax, wave: np.ndarray, cam: np.ndarray,
                 cmap=CMAP, lw: float = 2.0, vmin=0.0, vmax=1.0):
    """Draw a waveform as a single line whose colour encodes the CAM value."""
    t = np.arange(len(wave), dtype=float)
    pts  = np.stack([t, wave], axis=1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap=cmap, norm=Normalize(vmin, vmax),
                          linewidth=lw, capstyle="round")
    lc.set_array(cam[:-1])
    ax.add_collection(lc)
    margin = max((wave.max() - wave.min()) * 0.15, 0.05)
    ax.set_xlim(0, len(wave) - 1)
    ax.set_ylim(wave.min() - margin, wave.max() + margin)
    ax.set_xticks([]); ax.set_yticks([])
    return lc


def add_colorbar(fig, axes, cmap=CMAP, label="GradCAM activation"):
    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.015, shrink=0.7,
                        aspect=18, label=label)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])
    return cbar


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Device : {DEVICE}\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    beats = np.load(OUT / "beats_matrix.npy").astype(np.float32)
    df    = pd.read_csv(OUT / "dataset.csv")
    mask  = df["pulse_type_en"].isin(CLASS_ORDER)
    beats = beats[mask.values]
    df    = df[mask].reset_index(drop=True)

    le          = LabelEncoder().fit(sorted(CLASS_ORDER))
    CLASS_NAMES = list(le.classes_)          # ['Slippery', 'Tight', 'Wiry']
    n_classes   = len(CLASS_NAMES)

    y      = le.transform(df["pulse_type_en"].values)
    groups = df["subject_id"].values
    X      = beats[:, np.newaxis, :]

    # ── Augmented Tight ───────────────────────────────────────────────────────
    aug_full  = np.load(OUT / "tight_augmented_beats.npy").astype(np.float32)
    aug_meta  = pd.read_csv(OUT / "tight_augmented_meta.csv")
    synth     = (aug_meta["aug_mode"] != "original").values
    aug_X     = aug_full[synth][:, np.newaxis, :]
    aug_meta  = aug_meta[synth].reset_index(drop=True)
    tight_lbl = int(le.transform(["Tight"])[0])
    aug_y     = np.full(len(aug_meta), tight_lbl, dtype=int)

    # ── Fold-0 train / test split ─────────────────────────────────────────────
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(iter(skf.split(X, y, groups)))

    X_te, y_te = X[test_idx],  y[test_idx]
    X_tr_orig  = X[train_idx]
    y_tr_orig  = y[train_idx]

    train_sids = set(groups[train_idx])
    fm  = aug_meta["subject_id"].isin(train_sids).values
    X_tr = np.concatenate([X_tr_orig, aug_X[fm]])
    y_tr = np.concatenate([y_tr_orig, aug_y[fm]])

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()),
        batch_size=BATCH_SIZE, sampler=make_weighted_sampler(y_tr),
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te).long()),
        batch_size=256, shuffle=False,
    )

    # ── Train or reload ────────────────────────────────────────────────────────
    weight_path = OUT / "resnet_gc_weights.pt"
    model = PulseResNetGC(n_classes).to(DEVICE)

    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print(f"Loaded saved weights from {weight_path.name}")
    else:
        print("Training PulseResNetGC on folds 1-4 (fold-0 held out)…")
        cfg = dict(epochs=150, patience=30, lr=1e-3, wd=1e-4,
                   smooth=0.10, clip=None, warmup=0)
        cw        = class_weights_tensor(y_tr, n_classes)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg["smooth"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]
        )
        scheduler = get_scheduler(optimizer, cfg, cfg["epochs"])

        best_loss, patience_cnt, best_state = float("inf"), 0, None
        for ep in range(1, cfg["epochs"] + 1):
            train_epoch(model, train_loader, optimizer, criterion, cfg["clip"])
            scheduler.step()
            model.eval()
            val_loss = sum(
                F.cross_entropy(model(xb.to(DEVICE)), yb.to(DEVICE),
                                reduction="sum").item()
                for xb, yb in val_loader
            ) / len(y_te)
            if val_loss < best_loss - 1e-5:
                best_loss, patience_cnt = val_loss, 0
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= cfg["patience"]:
                    print(f"  Early stop at ep={ep}")
                    break
            if ep % 30 == 0:
                print(f"  ep={ep:3d}  val_loss={val_loss:.4f}")

        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        torch.save(model.state_dict(), weight_path)
        print(f"  Saved weights → {weight_path.name}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_prob = predict(model, val_loader)
    y_pred = y_prob.argmax(1)
    m      = evaluate(y_te, y_prob, n_classes)
    print(f"\n  F1={m['macro_f1']:.3f}  BalAcc={m['bal_acc']:.3f}"
          f"  ROC={m['roc_auc']:.3f}  PR={m['pr_auc']:.3f}")
    print(classification_report(y_te, y_pred,
                                 target_names=CLASS_NAMES, zero_division=0))

    # ── Register GradCAM on gc3 ────────────────────────────────────────────────
    gradcam = GradCAM1D(model, model.gc3)

    # ── Batch-compute CAMs ─────────────────────────────────────────────────────
    all_waves, all_cams, all_true, all_pred = [], [], [], []

    for xb, yb in val_loader:
        xb_dev = xb.to(DEVICE)
        with torch.no_grad():
            preds = model(xb_dev).argmax(1).cpu().numpy()
        cam = gradcam.compute(xb_dev, preds)     # (B, 200)
        all_waves.append(xb[:, 0].numpy())
        all_cams.append(cam)
        all_true.append(yb.numpy())
        all_pred.append(preds)

    all_waves = np.concatenate(all_waves)   # (N_test, 200)
    all_cams  = np.concatenate(all_cams)
    all_true  = np.concatenate(all_true)
    all_pred  = np.concatenate(all_pred)

    print(f"\nTest set : {len(all_waves)} beats\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Individual examples (3 classes × 4 columns)
    # ═══════════════════════════════════════════════════════════════════════════
    N_EXAMPLES = 4
    fig1, axes1 = plt.subplots(
        n_classes, 1 + N_EXAMPLES,
        figsize=(3.0 * (1 + N_EXAMPLES), 2.8 * n_classes),
        gridspec_kw={"wspace": 0.04, "hspace": 0.50},
    )
    fig1.suptitle(
        "GradCAM — PulseResNetGC  ·  1D Global Context ResNet\n"
        "Colour: red = high activation  ·  blue = low activation",
        fontsize=11, fontweight="bold", y=0.995,
    )

    for row, (cls_idx, cls_name) in enumerate(zip(range(n_classes), CLASS_NAMES)):
        # Correctly classified beats of this class
        correct = np.where((all_true == cls_idx) & (all_pred == cls_idx))[0]
        label   = CLASS_ZH.get(cls_name, cls_name)

        # ── Col 0: mean ──────────────────────────────────────────────────────
        ax0 = axes1[row, 0]
        if len(correct) == 0:
            ax0.axis("off")
            ax0.text(0.5, 0.5, "no correct pred.",
                     ha="center", va="center", transform=ax0.transAxes)
        else:
            m_wave = all_waves[correct].mean(axis=0)
            m_cam  = all_cams[correct].mean(axis=0)
            m_cam  = (m_cam - m_cam.min()) / (m_cam.max() - m_cam.min() + 1e-8)
            colored_line(ax0, m_wave, m_cam)
            ax0.set_title(f"Mean  (n={len(correct)})", fontsize=8, pad=3)

        ax0.set_ylabel(label, fontsize=9)

        # ── Col 1-4: 4 highest-confidence correct examples ───────────────────
        top_k = correct[np.argsort(y_prob[correct, cls_idx])[::-1][:N_EXAMPLES]]
        for col, idx in enumerate(top_k, start=1):
            ax = axes1[row, col]
            colored_line(ax, all_waves[idx], all_cams[idx])
            ax.set_title(f"p={y_prob[idx, cls_idx]:.2f}", fontsize=8, pad=3)

        for col in range(len(top_k) + 1, 1 + N_EXAMPLES):
            axes1[row, col].axis("off")

    add_colorbar(fig1, axes1[:, -1].tolist())
    fig1.savefig(OUT / "gradcam_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("  Saved → outputs/gradcam_examples.png")

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Class mean CAM  (publication-quality, 1 × 3)
    # ═══════════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(
        1, n_classes, figsize=(5.2 * n_classes, 4.2),
        gridspec_kw={"wspace": 0.08},
    )
    fig2.suptitle(
        "Mean GradCAM per Pulse Type — PulseResNetGC\n"
        "(averaged over all correctly classified test beats)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    t = np.arange(200)

    for col, (cls_idx, cls_name) in enumerate(zip(range(n_classes), CLASS_NAMES)):
        ax    = axes2[col]
        label = CLASS_ZH.get(cls_name, cls_name)
        correct = np.where((all_true == cls_idx) & (all_pred == cls_idx))[0]

        if len(correct) == 0:
            ax.text(0.5, 0.5, "no correct predictions",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=12, fontweight="bold")
            continue

        m_wave = all_waves[correct].mean(axis=0)
        std_wave = all_waves[correct].std(axis=0)
        m_cam  = all_cams[correct].mean(axis=0)
        m_cam  = (m_cam - m_cam.min()) / (m_cam.max() - m_cam.min() + 1e-8)

        # Shaded ±1 std band (gray)
        ax.fill_between(t, m_wave - std_wave, m_wave + std_wave,
                        color="gray", alpha=0.18, label="±1 std")

        # Fill under curve, colored by GradCAM
        baseline = m_wave.min() - (m_wave.max() - m_wave.min()) * 0.12
        for i in range(len(t) - 1):
            ax.fill_between(
                [t[i], t[i + 1]],
                [baseline, baseline],
                [m_wave[i], m_wave[i + 1]],
                color=matplotlib.colormaps[CMAP](m_cam[i]),
                alpha=0.55,
            )

        # Colored waveform line on top
        colored_line(ax, m_wave, m_cam, lw=2.2)

        # Mark the highest-activation time point
        peak = int(np.argmax(m_cam))
        ax.axvline(peak, color="crimson", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.text(peak + 4, m_wave.max() * 0.92,
                f"peak @ t={peak}\n({peak / 2:.0f}% beat)",
                fontsize=8, color="crimson", va="top")

        ax.set_xlim(0, 199)
        margin = (m_wave.max() - m_wave.min()) * 0.18
        ax.set_ylim(baseline, m_wave.max() + margin)
        ax.set_xticks([0, 50, 100, 150, 199])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
        ax.set_xlabel("Relative position within beat", fontsize=9)
        ax.set_yticks([])
        ax.set_title(f"{label}\n(n={len(correct)} correct)", fontsize=11,
                     fontweight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)

    add_colorbar(fig2, axes2.tolist(), label="Mean GradCAM activation (normalised)")
    fig2.savefig(OUT / "gradcam_mean.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved → outputs/gradcam_mean.png")

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 3 — Misclassified beats (where model went wrong)
    # ═══════════════════════════════════════════════════════════════════════════
    wrong = np.where(all_true != all_pred)[0]
    if len(wrong) > 0:
        n_show  = min(len(wrong), 8)
        n_cols2 = 4
        n_rows2 = math.ceil(n_show / n_cols2)
        fig3, axes3 = plt.subplots(
            n_rows2, n_cols2,
            figsize=(3.2 * n_cols2, 2.8 * n_rows2),
            gridspec_kw={"wspace": 0.05, "hspace": 0.50},
        )
        axes3 = axes3.flatten()
        fig3.suptitle(
            "GradCAM on Misclassified Beats\n"
            "(True class → Predicted class)",
            fontsize=11, fontweight="bold",
        )
        for i, idx in enumerate(wrong[:n_show]):
            ax   = axes3[i]
            wave = all_waves[idx]
            cam  = all_cams[idx]
            true_name = CLASS_NAMES[all_true[idx]]
            pred_name = CLASS_NAMES[all_pred[idx]]
            colored_line(ax, wave, cam)
            ax.set_title(
                f"True: {true_name}\n→ Pred: {pred_name}",
                fontsize=8, color="saddlebrown", pad=3,
            )
        for i in range(n_show, len(axes3)):
            axes3[i].axis("off")
        add_colorbar(fig3, axes3[:n_show].tolist())
        fig3.savefig(OUT / "gradcam_errors.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print("  Saved → outputs/gradcam_errors.png")

    print("\nDONE")


if __name__ == "__main__":
    main()
