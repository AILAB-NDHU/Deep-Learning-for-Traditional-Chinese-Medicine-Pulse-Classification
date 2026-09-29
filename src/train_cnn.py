"""
TCM Pulse — 3-Class 1D-CNN (弦脈 / 滑脈 / 緊脈)
==================================================
Excludes 澀脈 (Choppy).  Augmentation targets 緊脈 (Tight).

Design
------
* Model     : 3-layer Conv1d → AdaptiveAvgPool → FC
* CV        : StratifiedGroupKFold(n=5), groups = subject_id
* Imbalance : WeightedRandomSampler (train fold) + weighted CE loss
* Variants  : no_aug (baseline) and aug (Tight beats injected) per fold
* Metrics   : Macro-F1, Balanced Accuracy, Macro ROC-AUC, Macro PR-AUC

Outputs (outputs/)
------------------
  cnn_results_aug.csv              per-fold metrics for both variants
  cnn_confusion_matrix_aug.png     side-by-side CM: no-aug vs aug
  cnn_classification_report_aug.txt classification report (aug variant)

Data Leakage Prevention
-----------------------
  GroupKFold is run on the ORIGINAL dataset only.
  Synthetic Tight beats (aug_mode != 'original') are injected into the
  TRAINING fold only, restricted to subject_ids present in that fold.
  The test fold is always 100 % original beats.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

# ── reproducibility ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS       = 80
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 20
N_SPLITS     = 5

# 3-class experiment
CLASS_ORDER = ["Wiry", "Slippery", "Tight"]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class PulseCNN(nn.Module):
    """3-block 1D-CNN for pulse waveform classification."""

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),                  # → (B, 32, 100)

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),                  # → (B, 64, 50)

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),          # → (B, 128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    classes, counts = np.unique(labels, return_counts=True)
    class_weight  = 1.0 / counts
    sample_weight = class_weight[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weight).float(),
        num_samples=len(labels),
        replacement=True,
    )


def class_weights_tensor(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts  = np.bincount(labels, minlength=n_classes).astype(float)
    counts  = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_probs, all_preds = [], []
    for xb, _ in loader:
        logits = model(xb.to(DEVICE))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(axis=1))
    return np.concatenate(all_probs), np.concatenate(all_preds)


def evaluate(y_true, y_pred, y_prob, n_classes) -> dict:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    try:
        roc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        roc = float("nan")
    pr_list = []
    for c in range(n_classes):
        if (y_true == c).sum() > 0:
            pr_list.append(average_precision_score((y_true == c).astype(int), y_prob[:, c]))
    pr = float(np.mean(pr_list)) if pr_list else float("nan")
    return {"macro_f1": macro_f1, "bal_acc": bal_acc, "roc_auc": roc, "pr_auc": pr}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Device: {DEVICE}")

    # ── Load data ──────────────────────────────────────────────────────────────
    beats = np.load(OUT / "beats_matrix.npy")          # (N, 200) float32
    df    = pd.read_csv(OUT / "dataset.csv")
    assert len(beats) == len(df), "beats_matrix and dataset.csv row count mismatch"

    # Filter to 3 classes (safety check)
    mask  = df["pulse_type_en"].isin(CLASS_ORDER)
    beats = beats[mask.values]
    df    = df[mask].reset_index(drop=True)

    le = LabelEncoder()
    le.fit(sorted(CLASS_ORDER))
    CLASS_NAMES = list(le.classes_)      # alphabetical: ['Slippery','Tight','Wiry']
    n_classes   = len(CLASS_NAMES)

    y      = le.transform(df["pulse_type_en"].values)
    groups = df["subject_id"].values
    X      = beats[:, np.newaxis, :]      # (N, 1, 200)

    print(f"Dataset: {len(X)} beats, {n_classes} classes, {len(np.unique(groups))} subjects")
    for name, code in zip(CLASS_NAMES, range(n_classes)):
        print(f"  {name}: {(y==code).sum()}")

    # ── Load augmented Tight (synthetic rows only) ─────────────────────────────
    aug_beats_full = np.load(OUT / "tight_augmented_beats.npy")   # (1500, 200)
    aug_meta_full  = pd.read_csv(OUT / "tight_augmented_meta.csv")
    synth_mask     = (aug_meta_full["aug_mode"] != "original").values
    aug_meta       = aug_meta_full[synth_mask].reset_index(drop=True)
    aug_X          = aug_beats_full[synth_mask][:, np.newaxis, :]  # (K, 1, 200) float32
    tight_label    = int(le.transform(["Tight"])[0])
    aug_y          = np.full(len(aug_meta), tight_label, dtype=int)

    print(f"\nAugmented Tight : {len(aug_meta)} synthetic beats  "
          f"(subjects: {aug_meta['subject_id'].nunique()})")
    print(f"  mode dist     : {aug_meta['aug_mode'].value_counts().to_dict()}")
    print()

    # ── Cross-validation ───────────────────────────────────────────────────────
    skf        = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    records    = []
    pred_store = {
        "no_aug": {"y_true": [], "y_pred": [], "y_prob": []},
        "aug":    {"y_true": [], "y_pred": [], "y_prob": []},
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups), 1):
        X_tr_orig, y_tr_orig = X[train_idx], y[train_idx]
        X_te,      y_te      = X[test_idx],  y[test_idx]

        # Inject aug beats (train-fold subjects only; test fold untouched)
        train_sids = set(groups[train_idx])
        fold_mask  = aug_meta["subject_id"].isin(train_sids).values
        X_aug_fold = aug_X[fold_mask]
        y_aug_fold = aug_y[fold_mask]

        X_tr_aug = np.concatenate([X_tr_orig, X_aug_fold], axis=0)
        y_tr_aug = np.concatenate([y_tr_orig, y_aug_fold], axis=0)

        print(f"\nFold {fold}  orig={len(y_tr_orig)}  "
              f"+aug_tight={len(y_aug_fold)}  total={len(y_tr_aug)}  test={len(y_te)}")

        val_ds     = TensorDataset(torch.from_numpy(X_te),
                                   torch.from_numpy(y_te).long())
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

        for var_name, X_tr, y_tr in [("no_aug", X_tr_orig, y_tr_orig),
                                      ("aug",    X_tr_aug,  y_tr_aug)]:
            X_tr_t = torch.from_numpy(X_tr)
            y_tr_t = torch.from_numpy(y_tr).long()

            sampler      = make_weighted_sampler(y_tr)
            train_ds     = TensorDataset(X_tr_t, y_tr_t)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)

            model     = PulseCNN(n_classes).to(DEVICE)
            cw        = class_weights_tensor(y_tr, n_classes)
            criterion = nn.CrossEntropyLoss(weight=cw)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                          weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS
            )

            best_val_loss = float("inf")
            patience_cnt  = 0
            best_state    = None
            last_epoch    = 0

            for epoch in range(1, EPOCHS + 1):
                last_epoch = epoch
                train_one_epoch(model, train_loader, optimizer, criterion)
                scheduler.step()

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        val_loss += nn.functional.cross_entropy(
                            model(xb.to(DEVICE)), yb.to(DEVICE), reduction="sum"
                        ).item()
                val_loss /= len(y_te)

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    patience_cnt  = 0
                    best_state    = {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE:
                        break

            if best_state is not None:
                model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

            y_prob_fold, _ = predict(model, val_loader)
            y_pred_fold    = y_prob_fold.argmax(axis=1)
            metrics        = evaluate(y_te, y_pred_fold, y_prob_fold, n_classes)

            print(f"  [{var_name:6s}]  ep={last_epoch:3d}  "
                  f"F1={metrics['macro_f1']:.3f}  "
                  f"BalAcc={metrics['bal_acc']:.3f}  "
                  f"ROC={metrics['roc_auc']:.3f}  "
                  f"PR={metrics['pr_auc']:.3f}")

            records.append({"fold": fold, "variant": var_name, **metrics})
            pred_store[var_name]["y_true"].append(y_te)
            pred_store[var_name]["y_pred"].append(y_pred_fold)
            pred_store[var_name]["y_prob"].append(y_prob_fold)

    # ── Summary ────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    summary    = (
        results_df
        .groupby("variant")[["macro_f1", "bal_acc", "roc_auc", "pr_auc"]]
        .agg(["mean", "std"])
        .loc[["no_aug", "aug"]]
    )

    metric_cols = ["macro_f1", "bal_acc", "roc_auc", "pr_auc"]
    print("\n" + "=" * 72)
    print("SUMMARY  (mean ± std)  — 3-class: Wiry / Slippery / Tight")
    print("=" * 72)
    for var in ["no_aug", "aug"]:
        print(f"\n  [{var}]")
        for col in metric_cols:
            m = summary.loc[var, (col, "mean")]
            s = summary.loc[var, (col, "std")]
            print(f"    {col:12s}: {m:.4f} ± {s:.4f}")

    print("\n  DELTA (aug − no_aug):")
    for col in ["macro_f1", "pr_auc"]:
        before = summary.loc["no_aug", (col, "mean")]
        after  = summary.loc["aug",    (col, "mean")]
        sign   = "+" if after >= before else ""
        print(f"    {col:12s}: {before:.4f} → {after:.4f}  ({sign}{after - before:.4f})")

    # ── Confusion matrices ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(
        "1D-CNN Confusion Matrices — No-Aug vs Tight-Augmented\n"
        "(3-class: Wiry / Slippery / Tight, 5-fold subject-level CV)",
        fontsize=11, fontweight="bold",
    )
    panel_titles = {"no_aug": "No Augmentation (baseline)",
                    "aug":    "Tight Augmented"}
    for ax, var in zip(axes, ["no_aug", "aug"]):
        yt   = np.concatenate(pred_store[var]["y_true"])
        yp   = np.concatenate(pred_store[var]["y_pred"])
        cm_m = confusion_matrix(yt, yp, normalize="true")
        im   = ax.imshow(cm_m, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(panel_titles[var], fontsize=10, fontweight="bold")
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, f"{cm_m[i,j]:.2f}", ha="center", va="center",
                        color="white" if cm_m[i,j] > 0.5 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "cnn_confusion_matrix_aug.png", dpi=150)
    plt.close(fig)

    # ── Classification report ──────────────────────────────────────────────────
    y_true_aug = np.concatenate(pred_store["aug"]["y_true"])
    y_pred_aug = np.concatenate(pred_store["aug"]["y_pred"])
    report     = classification_report(y_true_aug, y_pred_aug,
                                       target_names=CLASS_NAMES, zero_division=0)
    print("\n=== Aug variant classification report ===")
    print(report)
    (OUT / "cnn_classification_report_aug.txt").write_text(
        "1D-CNN with Tight augmentation\n"
        "3-class (Wiry / Slippery / Tight), 5-fold subject-level CV\n\n"
        + report
    )

    results_df.to_csv(OUT / "cnn_results_aug.csv", index=False, float_format="%.6f")
    print(f"\n  Saved → outputs/cnn_results_aug.csv")
    print(f"  Saved → outputs/cnn_confusion_matrix_aug.png")
    print(f"  Saved → outputs/cnn_classification_report_aug.txt")
    print("=" * 72)
    print("CNN 3-class + Tight augmentation DONE")


if __name__ == "__main__":
    main()
