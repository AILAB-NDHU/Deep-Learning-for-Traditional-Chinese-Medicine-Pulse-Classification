"""
TCM Pulse — Gated / Attention-Enhanced ResNet Comparison
=========================================================
Implements and compares two architectural improvements on ResNet1D:

  resnet      : Plain ResNet1D (baseline from train_advanced_models.py)

  resnet_gc   : ResNet1D + Global Context Attention (方案 A spirit)
    - After each residual stage, a Global Context (GC) module injects
      a sequence-wide context vector back as an additive residual.
    - Context = softmax(W_a · x) over positions → weighted sum → MLP → broadcast.
    - O(N·C) complexity; for N=200 this is equivalent to Linear Attention
      in terms of computation but simpler to implement correctly.
    - Captures dicrotic notch / inter-beat relationships without heavy params.

  resnet_gated : ResNet1D + Dynamic Gated Residual (方案 B)
    - Replaces x + F(x) with  alpha * x + (1-alpha) * F(x)
    - alpha ∈ (0,1) is computed per-channel via squeeze-excitation on skip(x).
    - When a beat has baseline drift or motion noise, alpha → 0 (trust F(x)).
    - When the beat is clean, alpha → 1 (preserve identity).
    - Inspired by Highway Networks + SENet; very few extra parameters.

  resnet_gc_gated : Combines both improvements.

Training (shared with train_advanced_models.py)
-----------------------------------------------
  * 6-mode Tight augmentation, subject-level 5-fold CV
  * AdamW + CosineAnnealingLR, label smoothing=0.10
  * WeightedRandomSampler + class-weighted CE
  * Early stopping, reproducible seed=42

Outputs (outputs/)
------------------
  gated_models_results.csv
  gated_models_comparison.png
  gated_models_report.txt
"""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    classification_report, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
N_SPLITS   = 5
CLASS_ORDER = ["Wiry", "Slippery", "Tight"]


# ═════════════════════════════════════════════════════════════════════════════
# Building blocks
# ═════════════════════════════════════════════════════════════════════════════

class GlobalContext1D(nn.Module):
    """
    Global Context (GC) module for 1D signals — O(N·C) complexity.

    Computes a single context vector via soft attention over the sequence,
    transforms it through a small MLP, then adds it back as a broadcast
    residual.  Equivalent in spirit to Linear Attention but simpler.

    Architecture:
      attn   = softmax( W_q · x )            # (B, 1, L)  position weights
      ctx    = sum_L( attn * x )             # (B, C)     global context
      out    = x + LayerNorm(FC(ctx))[:,:,None]   broadcasted residual
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(channels // reduction, 16)
        self.attn_proj  = nn.Conv1d(channels, 1, 1, bias=False)
        self.transform  = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, inner),
            nn.ReLU(inplace=True),
            nn.Linear(inner, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        attn = torch.softmax(self.attn_proj(x), dim=-1)   # (B, 1, L)
        ctx  = (x * attn).sum(dim=-1)                     # (B, C)
        ctx  = self.transform(ctx)                        # (B, C)
        return x + ctx.unsqueeze(-1)                      # broadcast


class PlainResBlock(nn.Module):
    """Standard pre-activation residual block (used in all variants)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.skip  = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if (stride != 1 or in_ch != out_ch) else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.skip(x))


class GatedResBlock(nn.Module):
    """
    Dynamic Gated Residual Block (方案 B).

    Replaces the standard  output = ReLU(F(x) + skip(x))  with:

        alpha = Sigmoid( SE( skip(x) ) )        # per-channel gate ∈ (0,1)
        output = ReLU( alpha * skip(x) + (1-alpha) * F(x) )

    alpha is computed via Squeeze-Excitation on the skip-connected features:
    - Global avg pool  →  FC(C → C//4) → ReLU → FC(C//4 → C) → Sigmoid

    Intuition: noisy / drifting input → alpha → 0 (trust filtered F(x));
               clean input           → alpha → 1 (preserve identity).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.skip  = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if (stride != 1 or in_ch != out_ch) else nn.Identity()
        )
        # Gate network: squeeze-excitation on skip features
        mid = max(out_ch // 4, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, out_ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        fx   = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        # Dynamic gate from skip features
        alpha = self.gate(skip).unsqueeze(-1)            # (B, C, 1)
        return self.relu(alpha * skip + (1.0 - alpha) * fx)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Plain ResNet1D  (reference from train_advanced_models.py)
# ═════════════════════════════════════════════════════════════════════════════
class PulseResNet(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.stage1 = nn.Sequential(PlainResBlock(64,  64,  7), PlainResBlock(64,  64,  7))
        self.stage2 = nn.Sequential(PlainResBlock(64,  128, 5, stride=2), PlainResBlock(128, 128, 5))
        self.stage3 = nn.Sequential(PlainResBlock(128, 256, 3, stride=2), PlainResBlock(256, 256, 3))
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(self.gap(x))


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 2 — ResNet1D + Global Context  (方案 A)
# Inserts a GC module after each stage to capture global waveform context.
# ═════════════════════════════════════════════════════════════════════════════
class PulseResNetGC(nn.Module):
    """
    ResNet1D with Global Context attention after each stage.

    The GC module after stage-k captures the global context of features at
    that resolution (e.g. after stage1: L=100; stage3: L=25).
    Helps detect dicrotic notch position relative to the whole beat.
    """

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.stage1 = nn.Sequential(PlainResBlock(64,  64,  7), PlainResBlock(64,  64,  7))
        self.gc1    = GlobalContext1D(64)

        self.stage2 = nn.Sequential(PlainResBlock(64,  128, 5, stride=2), PlainResBlock(128, 128, 5))
        self.gc2    = GlobalContext1D(128)

        self.stage3 = nn.Sequential(PlainResBlock(128, 256, 3, stride=2), PlainResBlock(256, 256, 3))
        self.gc3    = GlobalContext1D(256)

        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.gc1(self.stage1(x))
        x = self.gc2(self.stage2(x))
        x = self.gc3(self.stage3(x))
        return self.head(self.gap(x))


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 3 — ResNet1D with Dynamic Gated Residuals  (方案 B)
# ═════════════════════════════════════════════════════════════════════════════
class PulseResNetGated(nn.Module):
    """
    ResNet1D where every residual block uses dynamic gating instead of
    simple identity skip.  Output = alpha*skip + (1-alpha)*F(x).
    """

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.stage1 = nn.Sequential(GatedResBlock(64,  64,  7), GatedResBlock(64,  64,  7))
        self.stage2 = nn.Sequential(GatedResBlock(64,  128, 5, stride=2), GatedResBlock(128, 128, 5))
        self.stage3 = nn.Sequential(GatedResBlock(128, 256, 3, stride=2), GatedResBlock(256, 256, 3))
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(self.gap(x))


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 4 — ResNet1D + GC + Gated  (combined)
# ═════════════════════════════════════════════════════════════════════════════
class PulseResNetGCGated(nn.Module):
    """Combines both improvements: gated residuals + global context."""

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.stage1 = nn.Sequential(GatedResBlock(64,  64,  7), GatedResBlock(64,  64,  7))
        self.gc1    = GlobalContext1D(64)

        self.stage2 = nn.Sequential(GatedResBlock(64,  128, 5, stride=2), GatedResBlock(128, 128, 5))
        self.gc2    = GlobalContext1D(128)

        self.stage3 = nn.Sequential(GatedResBlock(128, 256, 3, stride=2), GatedResBlock(256, 256, 3))
        self.gc3    = GlobalContext1D(256)

        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.gc1(self.stage1(x))
        x = self.gc2(self.stage2(x))
        x = self.gc3(self.stage3(x))
        return self.head(self.gap(x))


# ═════════════════════════════════════════════════════════════════════════════
# Per-model configs
# ═════════════════════════════════════════════════════════════════════════════
MODEL_CONFIGS: dict = {
    "resnet":        dict(cls=PulseResNet,        epochs=150, patience=30, lr=1e-3, wd=1e-4, smooth=0.10, clip=None, warmup=0),
    "resnet_gc":     dict(cls=PulseResNetGC,      epochs=150, patience=30, lr=1e-3, wd=1e-4, smooth=0.10, clip=None, warmup=0),
    "resnet_gated":  dict(cls=PulseResNetGated,   epochs=150, patience=30, lr=1e-3, wd=1e-4, smooth=0.10, clip=None, warmup=0),
    "resnet_gc_gated": dict(cls=PulseResNetGCGated, epochs=150, patience=35, lr=8e-4, wd=2e-4, smooth=0.10, clip=None, warmup=5),
}


# ═════════════════════════════════════════════════════════════════════════════
# Training helpers  (identical to train_advanced_models.py)
# ═════════════════════════════════════════════════════════════════════════════
def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    _, counts      = np.unique(labels, return_counts=True)
    sample_weights = (1.0 / counts)[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(labels), replacement=True,
    )


def class_weights_tensor(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts  = np.bincount(labels, minlength=n_classes).astype(float)
    counts  = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def get_scheduler(optimizer, cfg: dict, epochs: int):
    if cfg["warmup"] > 0:
        def lr_fn(ep: int) -> float:
            if ep < cfg["warmup"]:
                return (ep + 1) / cfg["warmup"]
            progress = (ep - cfg["warmup"]) / max(1, epochs - cfg["warmup"])
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def train_epoch(model, loader, optimizer, criterion, clip) -> None:
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()


@torch.no_grad()
def predict(model, loader) -> np.ndarray:
    model.eval()
    return np.concatenate([
        torch.softmax(model(xb.to(DEVICE)), dim=1).cpu().numpy()
        for xb, _ in loader
    ])


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> dict:
    y_pred   = y_prob.argmax(axis=1)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    try:
        roc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        roc = float("nan")
    pr = float(np.mean([
        average_precision_score((y_true == c).astype(int), y_prob[:, c])
        for c in range(n_classes) if (y_true == c).sum() > 0
    ]))
    return {"macro_f1": macro_f1, "bal_acc": bal_acc, "roc_auc": roc, "pr_auc": pr}


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(f"Device : {DEVICE}")

    # Print parameter counts
    print("\nModel parameter counts:")
    for name, cfg in MODEL_CONFIGS.items():
        n = count_params(cfg["cls"]())
        print(f"  {name:20s}: {n:,} params")

    # ── Load data ──────────────────────────────────────────────────────────────
    beats = np.load(OUT / "beats_matrix.npy")
    df    = pd.read_csv(OUT / "dataset.csv")
    mask  = df["pulse_type_en"].isin(CLASS_ORDER)
    beats = beats[mask.values]
    df    = df[mask].reset_index(drop=True)

    le          = LabelEncoder().fit(sorted(CLASS_ORDER))
    CLASS_NAMES = list(le.classes_)
    n_classes   = len(CLASS_NAMES)

    y      = le.transform(df["pulse_type_en"].values)
    groups = df["subject_id"].values
    X      = beats[:, np.newaxis, :]

    print(f"\nDataset : {len(X)} beats, {n_classes} classes, "
          f"{len(np.unique(groups))} subjects")

    # ── Load augmented Tight ───────────────────────────────────────────────────
    aug_beats_full = np.load(OUT / "tight_augmented_beats.npy")
    aug_meta_full  = pd.read_csv(OUT / "tight_augmented_meta.csv")
    synth_mask     = (aug_meta_full["aug_mode"] != "original").values
    aug_meta       = aug_meta_full[synth_mask].reset_index(drop=True)
    aug_X          = aug_beats_full[synth_mask][:, np.newaxis, :]
    tight_label    = int(le.transform(["Tight"])[0])
    aug_y          = np.full(len(aug_meta), tight_label, dtype=int)
    print(f"Augmented Tight : {len(aug_meta)} synthetic beats\n")

    # ── CV splits ─────────────────────────────────────────────────────────────
    skf    = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X, y, groups))

    records: list[dict] = []
    pred_store = {name: {"y_true": [], "y_pred": [], "y_prob": []} for name in MODEL_CONFIGS}

    # ── Train all models ───────────────────────────────────────────────────────
    for model_name, cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*64}")
        print(f"  MODEL : {model_name.upper()}"
              f"  ({count_params(cfg['cls']()):,} params, "
              f"ep={cfg['epochs']}, smooth={cfg['smooth']}, "
              f"warmup={cfg['warmup']})")
        print(f"{'='*64}")

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            X_tr_orig, y_tr_orig = X[train_idx], y[train_idx]
            X_te, y_te           = X[test_idx],  y[test_idx]

            train_sids = set(groups[train_idx])
            fm         = aug_meta["subject_id"].isin(train_sids).values
            X_tr = np.concatenate([X_tr_orig, aug_X[fm]])
            y_tr = np.concatenate([y_tr_orig, aug_y[fm]])

            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te).long()),
                batch_size=256, shuffle=False,
            )
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()),
                batch_size=BATCH_SIZE, sampler=make_weighted_sampler(y_tr),
            )

            model     = cfg["cls"](n_classes).to(DEVICE)
            cw        = class_weights_tensor(y_tr, n_classes)
            criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg["smooth"])
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]
            )
            scheduler = get_scheduler(optimizer, cfg, cfg["epochs"])

            best_loss, patience_cnt, best_state, last_ep = float("inf"), 0, None, 0
            for ep in range(1, cfg["epochs"] + 1):
                last_ep = ep
                train_epoch(model, train_loader, optimizer, criterion, cfg["clip"])
                scheduler.step()

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        val_loss += F.cross_entropy(
                            model(xb.to(DEVICE)), yb.to(DEVICE), reduction="sum"
                        ).item()
                val_loss /= len(y_te)

                if val_loss < best_loss - 1e-5:
                    best_loss, patience_cnt = val_loss, 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt >= cfg["patience"]:
                        break

            if best_state:
                model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

            y_prob  = predict(model, val_loader)
            y_pred  = y_prob.argmax(axis=1)
            metrics = evaluate(y_te, y_prob, n_classes)

            print(f"  Fold {fold}  ep={last_ep:3d}  "
                  f"F1={metrics['macro_f1']:.3f}  "
                  f"BalAcc={metrics['bal_acc']:.3f}  "
                  f"ROC={metrics['roc_auc']:.3f}  "
                  f"PR={metrics['pr_auc']:.3f}")

            records.append({"model": model_name, "fold": fold, **metrics})
            pred_store[model_name]["y_true"].append(y_te)
            pred_store[model_name]["y_pred"].append(y_pred)
            pred_store[model_name]["y_prob"].append(y_prob)

    # ── Summary ───────────────────────────────────────────────────────────────
    results_df  = pd.DataFrame(records)
    metric_cols = ["macro_f1", "bal_acc", "roc_auc", "pr_auc"]

    print("\n" + "=" * 72)
    print("SUMMARY  (mean ± std)  — 3-class: Wiry / Slippery / Tight")
    print("=" * 72)
    best_f1, best_name = 0.0, ""
    for name in MODEL_CONFIGS:
        sub = results_df[results_df["model"] == name]
        mean_f1 = sub["macro_f1"].mean()
        marker = " ◀ best" if mean_f1 > best_f1 else ""
        if mean_f1 > best_f1:
            best_f1, best_name = mean_f1, name
        print(f"\n  [{name.upper()}]{marker}")
        for col in metric_cols:
            print(f"    {col:12s}: {sub[col].mean():.4f} ± {sub[col].std():.4f}")

    # ── Classification reports ────────────────────────────────────────────────
    report_lines = [
        "Gated/GC ResNet Comparison — 3-class (Wiry / Slippery / Tight)\n"
        "5-fold subject-level CV + 6-mode Tight augmentation\n\n"
    ]
    print(f"\n=== Per-model classification reports ===")
    for name in MODEL_CONFIGS:
        yt     = np.concatenate(pred_store[name]["y_true"])
        yp     = np.concatenate(pred_store[name]["y_pred"])
        report = classification_report(yt, yp, target_names=CLASS_NAMES, zero_division=0)
        header = f"=== {name.upper()} ===\n"
        print(header + report)
        report_lines.extend([header, report, "\n"])
    (OUT / "gated_models_report.txt").write_text("".join(report_lines))

    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        "GC + Gated ResNet Comparison — 3-class Pulse, 5-fold Subject-level CV\n"
        "(6-mode Tight augmentation)",
        fontsize=11, fontweight="bold",
    )
    metric_labels = ["Macro-F1", "Balanced Acc", "ROC-AUC", "PR-AUC"]
    colors  = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    names   = list(MODEL_CONFIGS.keys())

    for ax, col, label in zip(axes, metric_cols, metric_labels):
        means = [results_df[results_df["model"] == n][col].mean() for n in names]
        stds  = [results_df[results_df["model"] == n][col].std()  for n in names]
        bars  = ax.bar(range(len(names)), means, yerr=stds, capsize=5,
                       color=colors, alpha=0.85, tick_label=names)
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(max(0.0, min(means) - 0.12), 1.01)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + s + 0.005, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "gated_models_comparison.png", dpi=150)
    plt.close(fig)

    results_df.to_csv(OUT / "gated_models_results.csv", index=False, float_format="%.6f")
    print(f"\n  Saved → outputs/gated_models_results.csv")
    print(f"  Saved → outputs/gated_models_comparison.png")
    print(f"  Saved → outputs/gated_models_report.txt")
    print("=" * 72)
    print(f"DONE — best model: {best_name} (Macro-F1 = {best_f1:.4f})")


if __name__ == "__main__":
    main()
