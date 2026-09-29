"""
TCM Pulse — Advanced Models Comparison
=======================================
Trains and compares four 1D architectures on the 3-class pulse dataset
(弦脈/滑脈/緊脈) using 6-mode Tight augmentation and subject-level 5-fold CV.

Models
------
  baseline    : 3-layer 1D-CNN (reproduced from train_cnn.py)
  resnet      : Residual 1D-CNN  (3 stages: 64→128→256 channels, skip connections)
  inception   : Multi-scale 1D-CNN  (parallel k=7/15/31 branches per block)
  transformer : Patch-based Transformer Encoder  (patch_size=10, d_model=64, 3 layers)

Training Strategies (per-model optimised)
------------------------------------------
  * All use aug=True (6-mode Tight synthetic beats injected per fold)
  * StratifiedGroupKFold(n=5), groups = subject_id  (no leakage)
  * WeightedRandomSampler + label-smoothed CrossEntropyLoss + class weights
  * AdamW with CosineAnnealingLR; Transformer uses linear warmup first
  * Early stopping; gradient clipping for Transformer
  * Reproducible: SEED=42

Outputs (outputs/)
------------------
  advanced_models_results.csv       per-fold metrics for all 4 models
  advanced_models_comparison.png    bar-chart comparison
  advanced_models_report.txt        per-class classification reports
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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

# ── reproducibility / speed ───────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True      # faster for fixed input shapes

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
N_SPLITS   = 5
CLASS_ORDER = ["Wiry", "Slippery", "Tight"]


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Baseline CNN  (3-block, from train_cnn.py)
# ═════════════════════════════════════════════════════════════════════════════
class PulseCNN(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 2 — ResNet1D  (3 stages with residual blocks, stride downsampling)
# ═════════════════════════════════════════════════════════════════════════════
class ResBlock1D(nn.Module):
    """Pre-activation residual block with optional channel / stride projection."""

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
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.skip(x))


class PulseResNet(nn.Module):
    """
    Architecture (input 1×200):
      Stem  : Conv1d(1→64, k=7, s=2) → (64, 100)
      Stage1: ResBlock(64,  64,  k=7) × 2           → (64,  100)
      Stage2: ResBlock(64,  128, k=5, s=2) × 2      → (128,  50)
      Stage3: ResBlock(128, 256, k=3, s=2) × 2      → (256,  25)
      GAP   → FC(256→128)→ReLU→Dropout→FC(128→n)
    """

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.stage1 = nn.Sequential(ResBlock1D(64,  64,  7), ResBlock1D(64,  64,  7))
        self.stage2 = nn.Sequential(ResBlock1D(64,  128, 5, stride=2), ResBlock1D(128, 128, 5))
        self.stage3 = nn.Sequential(ResBlock1D(128, 256, 3, stride=2), ResBlock1D(256, 256, 3))
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(self.gap(x))


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 3 — Inception1D  (parallel k=7/15/31 branches + max-pool branch)
# ═════════════════════════════════════════════════════════════════════════════
class InceptionBlock1D(nn.Module):
    """
    Four parallel branches per block (each preceded by a 1×1 bottleneck):
      branch1 : k=7   (fine-grained detail)
      branch2 : k=15  (medium rhythm)
      branch3 : k=31  (long-range envelope)
      branch4 : MaxPool(3) + 1×1 conv  (pooling path)
    Output channels = 4 × base_ch.
    """

    def __init__(self, in_ch: int, base_ch: int):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, 1, bias=False),
            nn.Conv1d(base_ch, base_ch, 7,  padding=3,  bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, 1, bias=False),
            nn.Conv1d(base_ch, base_ch, 15, padding=7,  bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(),
        )
        self.b3 = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, 1, bias=False),
            nn.Conv1d(base_ch, base_ch, 31, padding=15, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, base_ch, 1, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class PulseInception(nn.Module):
    """
    Architecture (input 1×200):
      Stem     : Conv1d(1→32, k=7) → MaxPool(2)      → (32,  100)
      Inception1: InceptionBlock(32,  base=32)         → (128, 100)
      MaxPool  : → (128, 50)
      Inception2: InceptionBlock(128, base=48)         → (192,  50)
      MaxPool  : → (192, 25)
      Inception3: InceptionBlock(192, base=64)         → (256,  25)
      GAP      → FC(256→128)→ReLU→Dropout→FC(128→n)
    """

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.stem  = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.inc1  = InceptionBlock1D(32,  base_ch=32)   # out: 128
        self.pool1 = nn.MaxPool1d(2)
        self.inc2  = InceptionBlock1D(128, base_ch=48)   # out: 192
        self.pool2 = nn.MaxPool1d(2)
        self.inc3  = InceptionBlock1D(192, base_ch=64)   # out: 256
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.head  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool1(self.inc1(x))
        x = self.pool2(self.inc2(x))
        x = self.gap(self.inc3(x))
        return self.head(x)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 4 — Patch Transformer  (ViT-style for 1D, pre-LN, warmup LR)
# ═════════════════════════════════════════════════════════════════════════════
class PulseTransformer(nn.Module):
    """
    Divides the 200-point waveform into non-overlapping patches (patch_size=10
    → 20 tokens), prepends a [CLS] token, adds learnable positional embeddings,
    then applies a stack of Transformer Encoder layers.  Uses Pre-LayerNorm
    (norm_first=True) for more stable training on small datasets.

    Architecture:
      Patch embed : Linear(10 → d_model=64)
      CLS token   : learnable (1, 1, 64)
      Pos embed   : learnable (1, 21, 64)
      Encoder     : 3 × TransformerEncoderLayer(d_model=64, nhead=4,
                                                 dim_ff=256, dropout=0.2)
      Head        : CLS output → Linear(64→32) → ReLU → Dropout → Linear(32→n)
    """

    def __init__(
        self,
        n_classes:  int = 3,
        patch_size: int = 10,
        d_model:    int = 64,
        nhead:      int = 4,
        n_layers:   int = 3,
        d_ff:       int = 256,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.patch_size  = patch_size
        n_patches        = 200 // patch_size      # = 20

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self.drop        = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True,
            norm_first=True,          # pre-LN: more stable on small data
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # (B,1,200) → (B,200) → (B,20,10)
        x   = x.squeeze(1).reshape(B, -1, self.patch_size)
        x   = self.patch_embed(x)                           # (B,20,d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)                    # (B,21,d_model)
        x   = self.drop(x + self.pos_embed)
        x   = self.norm(self.transformer(x))
        return self.head(x[:, 0])                           # CLS token → logits


# ═════════════════════════════════════════════════════════════════════════════
# Per-model hyper-parameter configs
# ═════════════════════════════════════════════════════════════════════════════
MODEL_CONFIGS: dict = {
    # name         model class        ep    pat   lr     wd     smooth  clip   warmup
    "baseline":    dict(cls=PulseCNN,          epochs=120, patience=25, lr=1e-3, wd=1e-4, smooth=0.05, clip=None, warmup=0),
    "resnet":      dict(cls=PulseResNet,        epochs=150, patience=30, lr=1e-3, wd=1e-4, smooth=0.10, clip=None, warmup=0),
    "inception":   dict(cls=PulseInception,     epochs=150, patience=30, lr=1e-3, wd=1e-4, smooth=0.10, clip=None, warmup=0),
    "transformer": dict(cls=PulseTransformer,   epochs=200, patience=40, lr=3e-4, wd=1e-3, smooth=0.10, clip=1.0,  warmup=10),
}


# ═════════════════════════════════════════════════════════════════════════════
# Training helpers
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
        loss = criterion(model(xb), yb)
        loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()


@torch.no_grad()
def predict(model, loader) -> np.ndarray:
    model.eval()
    parts = []
    for xb, _ in loader:
        parts.append(torch.softmax(model(xb.to(DEVICE)), dim=1).cpu().numpy())
    return np.concatenate(parts)


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


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(f"Device : {DEVICE}")

    # ── Load original data ───────────────────────────────────────────────────
    beats = np.load(OUT / "beats_matrix.npy")          # (N, 200) float32
    df    = pd.read_csv(OUT / "dataset.csv")
    mask  = df["pulse_type_en"].isin(CLASS_ORDER)
    beats = beats[mask.values]
    df    = df[mask].reset_index(drop=True)

    le          = LabelEncoder().fit(sorted(CLASS_ORDER))
    CLASS_NAMES = list(le.classes_)                    # ['Slippery','Tight','Wiry']
    n_classes   = len(CLASS_NAMES)

    y      = le.transform(df["pulse_type_en"].values)
    groups = df["subject_id"].values
    X      = beats[:, np.newaxis, :]                   # (N, 1, 200)

    print(f"Dataset : {len(X)} beats, {n_classes} classes, "
          f"{len(np.unique(groups))} subjects")
    for name, code in zip(CLASS_NAMES, range(n_classes)):
        print(f"  {name}: {(y == code).sum()}")

    # ── Load augmented Tight beats (synthetic only) ──────────────────────────
    aug_beats_full = np.load(OUT / "tight_augmented_beats.npy")
    aug_meta_full  = pd.read_csv(OUT / "tight_augmented_meta.csv")
    synth_mask     = (aug_meta_full["aug_mode"] != "original").values
    aug_meta       = aug_meta_full[synth_mask].reset_index(drop=True)
    aug_X          = aug_beats_full[synth_mask][:, np.newaxis, :]
    tight_label    = int(le.transform(["Tight"])[0])
    aug_y          = np.full(len(aug_meta), tight_label, dtype=int)
    print(f"\nAugmented Tight : {len(aug_meta)} synthetic beats "
          f"(subjects: {aug_meta['subject_id'].nunique()})\n")

    # ── CV splits (shared across all models for fair comparison) ─────────────
    skf    = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X, y, groups))

    records: list[dict] = []
    pred_store = {
        name: {"y_true": [], "y_pred": [], "y_prob": []}
        for name in MODEL_CONFIGS
    }

    # ── Loop over models ──────────────────────────────────────────────────────
    for model_name, cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*64}")
        print(f"  MODEL : {model_name.upper()}"
              f"  (ep={cfg['epochs']}, pat={cfg['patience']}, "
              f"lr={cfg['lr']}, wd={cfg['wd']}, "
              f"smooth={cfg['smooth']}, warmup={cfg['warmup']})")
        print(f"{'='*64}")

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            X_tr_orig, y_tr_orig = X[train_idx], y[train_idx]
            X_te, y_te           = X[test_idx],  y[test_idx]

            # Inject aug beats restricted to train-fold subjects
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
                batch_size=BATCH_SIZE,
                sampler=make_weighted_sampler(y_tr),
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

                # Validation loss for early stopping
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        val_loss += nn.functional.cross_entropy(
                            model(xb.to(DEVICE)), yb.to(DEVICE),
                            reduction="sum",
                        ).item()
                val_loss /= len(y_te)

                if val_loss < best_loss - 1e-5:
                    best_loss, patience_cnt = val_loss, 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt >= cfg["patience"]:
                        break

            if best_state is not None:
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
    results_df = pd.DataFrame(records)
    metric_cols = ["macro_f1", "bal_acc", "roc_auc", "pr_auc"]

    print("\n" + "=" * 72)
    print("SUMMARY  (mean ± std)  — 3-class: Wiry / Slippery / Tight")
    print("=" * 72)
    for name in MODEL_CONFIGS:
        sub = results_df[results_df["model"] == name]
        print(f"\n  [{name.upper()}]")
        for col in metric_cols:
            print(f"    {col:12s}: {sub[col].mean():.4f} ± {sub[col].std():.4f}")

    # ── Classification reports ────────────────────────────────────────────────
    report_lines: list[str] = [
        "Advanced Models Comparison — 3-class (Wiry / Slippery / Tight)\n"
        "5-fold subject-level CV + 6-mode Tight augmentation\n\n"
    ]
    print("\n=== Per-model classification reports ===")
    for name in MODEL_CONFIGS:
        yt     = np.concatenate(pred_store[name]["y_true"])
        yp     = np.concatenate(pred_store[name]["y_pred"])
        report = classification_report(yt, yp, target_names=CLASS_NAMES, zero_division=0)
        header = f"=== {name.upper()} ===\n"
        print(header + report)
        report_lines.extend([header, report, "\n"])
    (OUT / "advanced_models_report.txt").write_text("".join(report_lines))

    # ── Comparison bar chart ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        "Advanced Models Comparison — 3-class Pulse, 5-fold Subject-level CV\n"
        "(all variants use 6-mode Tight augmentation)",
        fontsize=11, fontweight="bold",
    )
    metric_labels = ["Macro-F1", "Balanced Acc", "ROC-AUC", "PR-AUC"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    names  = list(MODEL_CONFIGS.keys())

    for ax, col, label in zip(axes, metric_cols, metric_labels):
        means = [results_df[results_df["model"] == n][col].mean() for n in names]
        stds  = [results_df[results_df["model"] == n][col].std()  for n in names]
        bars  = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(max(0.0, min(means) - 0.12), 1.01)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(stds) * 0.5) + 0.003,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8.5,
            )
    fig.tight_layout()
    fig.savefig(OUT / "advanced_models_comparison.png", dpi=150)
    plt.close(fig)

    results_df.to_csv(OUT / "advanced_models_results.csv", index=False, float_format="%.6f")

    print(f"\n  Saved → outputs/advanced_models_results.csv")
    print(f"  Saved → outputs/advanced_models_comparison.png")
    print(f"  Saved → outputs/advanced_models_report.txt")
    print("=" * 72)
    print("Advanced models comparison DONE")


if __name__ == "__main__":
    main()
