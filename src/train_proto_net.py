"""
TCM Pulse — 1D Prototypical Network
=====================================
Meta-learning approach for few-shot style episodic training on the
imbalanced 3-class pulse diagnosis dataset.

Key ideas
---------
* Standard cross-entropy trains a model to memorise class separations
  for exactly the classes seen during training.  Prototypical Networks
  (Snell et al. 2017) instead train the model to learn an *embedding space*
  where same-class beats cluster tightly and different-class beats are far
  apart — this generalises better under extreme class imbalance and could
  help at deployment time with new subjects.

* Each training iteration samples a random "episode":
    - N classes are chosen (here always 3, since we have exactly 3)
    - For each class, K beats become the "support set" → the class
      prototype is their mean embedding
    - Q beats per class become the "query set"
    - Loss = cross-entropy over negative squared Euclidean distances
      (this IS differentiable and is the canonical ProtoNet loss)

Episode Sampler design
-----------------------
  EpisodeSampler.sample() →
      support  (N, K, 1, 200) float32 tensor
      s_labels (N, K)          int64   tensor
      query    (N, Q, 1, 200)  float32 tensor
      q_labels (N, Q)          int64   tensor

  Balanced within episode: each class contributes exactly K support
  and Q query beats.  For Tight (163 beats) this means K+Q must not
  exceed the fold-specific count; we set K=5, Q=10 by default.

Backbone
---------
  PulseEncoder: ResNet1D stem + 3 stages → GAP → L2-normalised
  128-dim embedding (the Softmax head from train_gated_models.py is
  replaced by a small projection MLP → L2 norm).

Prototype computation
----------------------
  proto_c  = mean_k( embed(support_c) )       # (N, D)
  dist_qc  = ||embed(q_i) - proto_c||²        # (N*Q, N)
  logits   = -dist_qc
  loss     = CrossEntropy(logits, q_labels)

Training strategy
------------------
  * Pre-train episodes use the augmented dataset (Tight boosted to 1500)
  * Subject-level split: train subjects vs test subjects, no leakage
  * 5-fold evaluation: for each fold, train on folds 1-4, test on fold 0
  * Each "epoch" = 100 episodes  (standard ProtoNet practice)
  * AdamW + CosineAnnealingLR, early stopping on val proto-accuracy

Outputs (outputs/)
------------------
  proto_net_results.csv
  proto_net_report.txt
  proto_net_comparison.png
  proto_net_encoder.pt     (trained encoder weights, fold-0)
"""

import math
import random
import sys
from pathlib import Path
from collections import defaultdict

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
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    classification_report, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

# Reuse shared building blocks
from src.train_gated_models import (
    PlainResBlock,
    DEVICE, SEED, CLASS_ORDER, predict, evaluate,
)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
N_WAY       = 3          # always 3 (we have exactly 3 classes)
K_SHOT      = 5          # support beats per class per episode
Q_QUERY     = 15         # query beats per class per episode
EMBED_DIM   = 128        # embedding vector dimension
EPISODES_PER_EPOCH = 100 # number of episodes per "epoch"
MAX_EPOCHS  = 80
PATIENCE    = 15
LR          = 5e-4
WD          = 1e-4
N_SPLITS    = 5
TEMPERATURE = 0.1        # distance scale: logits = -dist² / T  (lower → sharper)
GRAD_CLIP   = 5.0        # max-norm gradient clipping
WARMUP_EPOCHS = 5        # linear LR warm-up before cosine annealing


# ═════════════════════════════════════════════════════════════════════════════
# 1. Episode Sampler
# ═════════════════════════════════════════════════════════════════════════════

class EpisodeSampler:
    """
    N-way K-shot episode sampler for 1D pulse waveforms.

    Parameters
    ----------
    beats  : (N_total, 200) float32 array — normalised waveforms
    labels : (N_total,)     int   array — class indices 0/1/2
    n_way  : int — how many classes per episode
    k_shot : int — support beats per class
    q_query: int — query beats per class
    rng    : np.random.Generator | None

    Usage
    -----
        sampler = EpisodeSampler(beats_train, y_train)
        sup, s_lbl, qry, q_lbl = sampler.sample()

    Returns
    -------
    support  : (n_way, k_shot, 1, 200)  torch.FloatTensor on DEVICE
    s_labels : (n_way, k_shot)           torch.LongTensor
    query    : (n_way, q_query, 1, 200) torch.FloatTensor on DEVICE
    q_labels : (n_way * q_query,)       torch.LongTensor  (flat, contiguous)

    The query labels are *episode-local* indices (0 … n_way-1) so that
    the cross-entropy loss can directly index the n_way prototypes.
    """

    def __init__(
        self,
        beats:  np.ndarray,
        labels: np.ndarray,
        n_way:  int = N_WAY,
        k_shot: int = K_SHOT,
        q_query: int = Q_QUERY,
        rng: np.random.Generator | None = None,
    ):
        self.n_way   = n_way
        self.k_shot  = k_shot
        self.q_query = q_query
        self.rng     = rng or np.random.default_rng(SEED)

        # Index beats per class for fast sampling
        unique_classes = sorted(np.unique(labels))
        assert len(unique_classes) >= n_way, (
            f"Need at least {n_way} classes, got {len(unique_classes)}"
        )
        self.class_indices: dict[int, np.ndarray] = {
            c: np.where(labels == c)[0] for c in unique_classes
        }
        self.all_classes = unique_classes
        self.beats  = beats      # reference to the full array (no copy)

        # Validate: each class must have enough beats
        required = k_shot + q_query
        for c, idx in self.class_indices.items():
            if len(idx) < required:
                raise ValueError(
                    f"Class {c} has only {len(idx)} beats but "
                    f"k_shot+q_query={required} are needed per episode."
                )

    def sample(self) -> tuple[torch.Tensor, torch.Tensor,
                               torch.Tensor, torch.Tensor]:
        """Draw one episode at random."""
        # Choose n_way classes (for 3-class data this is all of them)
        chosen_classes = self.rng.choice(
            self.all_classes, size=self.n_way, replace=False
        ).tolist()

        support_beats, query_beats = [], []
        support_labels, query_labels = [], []

        for local_idx, cls in enumerate(chosen_classes):
            pool = self.class_indices[cls]
            # Sample k_shot + q_query without replacement
            sampled = self.rng.choice(pool, size=self.k_shot + self.q_query,
                                      replace=False)
            sup_idx = sampled[:self.k_shot]
            qry_idx = sampled[self.k_shot:]

            support_beats.append(self.beats[sup_idx])     # (K, 200)
            query_beats.append(self.beats[qry_idx])        # (Q, 200)
            support_labels.append(np.full(self.k_shot,  local_idx))
            query_labels.append(np.full(self.q_query, local_idx))

        # Stack into tensors: add channel dim for Conv1d
        # (N_way, K, 1, 200) and (N_way, Q, 1, 200)
        sup  = torch.from_numpy(
            np.stack(support_beats)[:, :, np.newaxis, :]   # (N, K, 1, 200)
        ).float().to(DEVICE)
        qry  = torch.from_numpy(
            np.stack(query_beats)[:, :, np.newaxis, :]     # (N, Q, 1, 200)
        ).float().to(DEVICE)
        s_lbl = torch.from_numpy(np.stack(support_labels)).long().to(DEVICE)
        q_lbl = torch.from_numpy(
            np.concatenate(query_labels)                    # (N*Q,)
        ).long().to(DEVICE)

        return sup, s_lbl, qry, q_lbl


# ═════════════════════════════════════════════════════════════════════════════
# 2. Encoder: ResNet1D backbone → L2-normalised embedding
# ═════════════════════════════════════════════════════════════════════════════

class PulseEncoder(nn.Module):
    """
    ResNet1D backbone with the classification head replaced by a
    projection MLP that outputs L2-normalised EMBED_DIM-dimensional vectors.

    Architecture
    ------------
      Stem   : Conv1d(1 → 64, k=7, stride=2)  → BN → ReLU   → (B, 64, 100)
      Stage1 : PlainResBlock(64, 64,  k=7) × 2               → (B, 64, 100)
      Stage2 : PlainResBlock(64→128, k=5, s=2) × 2           → (B,128,  50)
      Stage3 : PlainResBlock(128→256, k=3, s=2) × 2          → (B,256,  25)
      GAP    : AdaptiveAvgPool1d(1) → Flatten                 → (B, 256)
      Proj   : Linear(256→256) → BN → ReLU → Linear(256→D)   → (B, D)
      L2-norm: ‖z‖₂ = 1                                      → (B, D)

    The L2 normalisation puts all embeddings on a unit hypersphere, which
    stabilises training and makes Euclidean distance equivalent to cosine
    distance (up to a constant factor).
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            PlainResBlock(64,  64,  7),
            PlainResBlock(64,  64,  7),
        )
        self.stage2 = nn.Sequential(
            PlainResBlock(64,  128, 5, stride=2),
            PlainResBlock(128, 128, 5),
        )
        self.stage3 = nn.Sequential(
            PlainResBlock(128, 256, 3, stride=2),
            PlainResBlock(256, 256, 3),
        )
        self.gap  = nn.AdaptiveAvgPool1d(1)
        # Projection MLP: 256 → 256 → embed_dim
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, 200) — raw waveform batch
        Returns L2-normalised embedding (B, embed_dim)
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        z = self.proj(x)
        return F.normalize(z, p=2, dim=-1)   # L2-norm → unit sphere


# ═════════════════════════════════════════════════════════════════════════════
# 3. Prototype computation + ProtoNet episode loss
# ═════════════════════════════════════════════════════════════════════════════

def compute_prototypes(
    encoder: PulseEncoder,
    support: torch.Tensor,          # (N_way, K_shot, 1, 200)
) -> torch.Tensor:
    """
    Compute one class prototype per class.

    Steps
    -----
    1. Reshape support to (N_way * K_shot, 1, 200)
    2. Run through encoder → (N_way * K_shot, D)
    3. Reshape to (N_way, K_shot, D)
    4. Mean over K_shot → (N_way, D)
       This mean is taken *in embedding space*, which is the key insight
       of Prototypical Networks: the prototype is the mean feature vector,
       not a mean of the raw inputs.

    Returns
    -------
    prototypes : (N_way, D) float tensor
    """
    n_way, k_shot, c, l = support.shape
    # Flatten class × shot dimensions for a single forward pass
    x_flat   = support.view(n_way * k_shot, c, l)          # (N*K, 1, 200)
    emb_flat = encoder(x_flat)                              # (N*K, D)
    emb      = emb_flat.view(n_way, k_shot, -1)             # (N, K, D)
    return emb.mean(dim=1)                                  # (N, D)  ← prototype


def proto_episode_loss(
    encoder:    PulseEncoder,
    support:    torch.Tensor,           # (N_way, K_shot, 1, 200)
    query:      torch.Tensor,           # (N_way, Q_query, 1, 200)
    q_labels:   torch.Tensor,           # (N_way * Q_query,)  local 0…N-1
    temperature: float = TEMPERATURE,
) -> tuple[torch.Tensor, float]:
    """
    Run one episode and return the ProtoNet loss + accuracy.

    Algorithm
    ---------
    1. Compute prototypes from support set:  P ∈ ℝ^{N × D}
    2. L2-normalise prototypes → unit hypersphere (consistent with encoder output)
    3. Encode all query beats:               Z ∈ ℝ^{(N·Q) × D}  (already L2-normed)
    4. Compute squared Euclidean distances:
           ‖z_i − p_c‖² = 2 − 2 · z_i · p_cᵀ   (valid since both are unit-normed)
           dist_mat  ∈ ℝ^{(N·Q) × N}
    5. Temperature-scaled logits:  logits = −dist² / T
       (lower T → sharper distribution; T=0.1 works well empirically)
    6. Loss = CrossEntropyLoss(logits, q_labels)

    Why normalise prototypes?
    -------------------------
    The encoder outputs unit-norm embeddings.  Their mean (= prototype) lies
    *inside* the sphere (‖mean‖ ≤ 1).  Re-normalising brings prototypes back
    to the unit sphere so the formula  ‖a−b‖² = 2 − 2 a·bᵀ  holds exactly,
    and training is consistent with nearest-centroid inference.

    Returns
    -------
    loss     : scalar tensor (with grad)
    accuracy : float (0.0 – 1.0)
    """
    n_way, q_per_class = query.shape[0], query.shape[1]
    n_query_total      = n_way * q_per_class

    # ── Prototypes (L2-normalised → unit sphere) ──────────────────────────────
    prototypes = compute_prototypes(encoder, support)   # (N, D)  may have ‖·‖<1
    prototypes = F.normalize(prototypes, p=2, dim=-1)   # (N, D)  unit sphere ✓

    # ── Query embeddings (encoder already L2-normalises) ──────────────────────
    x_flat = query.view(n_query_total, query.shape[2], query.shape[3])
    z_q    = encoder(x_flat)                            # (N*Q, D)  unit sphere

    # ── Squared Euclidean distances on unit sphere ────────────────────────────
    # ‖a − b‖² = ‖a‖² + ‖b‖² − 2 a·bᵀ = 2 − 2 a·bᵀ   (both unit-normed)
    dot   = z_q @ prototypes.t()                        # (N*Q, N)  cosine sim
    dists = 2.0 - 2.0 * dot                             # (N*Q, N)  ∈ [0, 4]

    # ── Temperature-scaled logits ─────────────────────────────────────────────
    # Dividing by T sharpens the softmax; T=0.1 ≈ "concentrate near prototype"
    logits = -dists / temperature                       # (N*Q, N)
    loss   = F.cross_entropy(logits, q_labels)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    preds    = logits.argmax(dim=1)
    accuracy = (preds == q_labels).float().mean().item()

    return loss, accuracy


# ═════════════════════════════════════════════════════════════════════════════
# 4. Full training loop
# ═════════════════════════════════════════════════════════════════════════════

def run_epoch(
    encoder:  PulseEncoder,
    sampler:  EpisodeSampler,
    optimizer: torch.optim.Optimizer | None,
    n_episodes: int,
    train: bool,
) -> tuple[float, float]:
    """
    Run `n_episodes` episodes and return mean loss + mean accuracy.

    Gradient clipping (GRAD_CLIP) is applied before each optimizer step to
    prevent exploding gradients that can occur in early meta-learning episodes
    when the embedding space is still disorganised.
    """
    encoder.train(train)
    total_loss, total_acc = 0.0, 0.0

    ctx = torch.enable_grad if train else torch.no_grad
    with ctx():
        for _ in range(n_episodes):
            sup, s_lbl, qry, q_lbl = sampler.sample()
            loss, acc = proto_episode_loss(encoder, sup, qry, q_lbl)

            if train:
                optimizer.zero_grad()
                loss.backward()
                # ── Gradient clipping: prevents spikes in early training ────
                nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
            total_acc  += acc

    return total_loss / n_episodes, total_acc / n_episodes


def nearest_centroid_predict(
    encoder:  PulseEncoder,
    X_train:  np.ndarray,    # (N_train, 200)
    y_train:  np.ndarray,    # (N_train,)
    X_test:   np.ndarray,    # (N_test, 200)
    n_classes: int,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    After episodic training, evaluate using Nearest-Centroid classification:
    1. Encode all training beats → compute per-class mean embeddings
       (prototypes) over the *entire* training set.
    2. For each test beat, assign the class whose prototype is closest.

    This is the standard ProtoNet evaluation protocol at test time.

    Returns
    -------
    y_pred : (N_test,) int array
    y_prob : (N_test, n_classes) float array (softmax of -distances)
    """
    encoder.eval()

    def _encode(X: np.ndarray) -> np.ndarray:
        chunks = []
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(
                X[start:start+batch_size, np.newaxis, :]
            ).float().to(DEVICE)
            with torch.no_grad():
                chunks.append(encoder(xb).cpu().numpy())
        return np.concatenate(chunks)

    z_train = _encode(X_train)         # (N_train, D)  unit-normed
    z_test  = _encode(X_test)          # (N_test,  D)  unit-normed

    # Compute per-class prototype from full training split, then re-normalise
    # (mean of unit vectors has ‖·‖ ≤ 1; re-norm → consistent with training)
    prototypes = np.stack([
        z_train[y_train == c].mean(axis=0) for c in range(n_classes)
    ])                                  # (n_classes, D)
    norms      = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
    prototypes = prototypes / norms     # (n_classes, D)  unit-normed

    # Squared Euclidean distances (unit sphere: ‖a−b‖² = 2 − 2 a·bᵀ)
    dot   = z_test @ prototypes.T      # (N_test, n_classes)
    dists = 2.0 - 2.0 * dot           # (N_test, n_classes)

    y_prob = F.softmax(
        torch.from_numpy(-dists).float(), dim=-1
    ).numpy()                          # (N_test, n_classes)
    y_pred = y_prob.argmax(axis=1)    # (N_test,)

    return y_pred, y_prob


# ═════════════════════════════════════════════════════════════════════════════
# Main: 5-fold subject-level CV
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"Device : {DEVICE}")
    print(f"Config : {N_WAY}-way {K_SHOT}-shot {Q_QUERY}-query  "
          f"embed={EMBED_DIM}  eps/epoch={EPISODES_PER_EPOCH}  "
          f"max_epochs={MAX_EPOCHS}\n")

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
    X      = beats                           # (N, 200) — no channel dim yet

    # ── Augmented Tight ───────────────────────────────────────────────────────
    aug_full = np.load(OUT / "tight_augmented_beats.npy").astype(np.float32)
    aug_meta = pd.read_csv(OUT / "tight_augmented_meta.csv")
    synth    = (aug_meta["aug_mode"] != "original").values
    aug_X    = aug_full[synth]
    aug_meta = aug_meta[synth].reset_index(drop=True)
    tight_lbl = int(le.transform(["Tight"])[0])
    aug_y     = np.full(len(aug_meta), tight_lbl, dtype=int)
    print(f"Dataset : {len(X)} real beats + {len(aug_X)} synthetic Tight")

    # ── 5-fold CV ──────────────────────────────────────────────────────────────
    skf     = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                   random_state=SEED)
    splits  = list(skf.split(X, y, groups))
    records = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_te, y_te = X[test_idx],  y[test_idx]
        X_tr_orig  = X[train_idx]
        y_tr_orig  = y[train_idx]

        # Append synthetic Tight beats from train subjects only
        train_sids = set(groups[train_idx])
        fm  = aug_meta["subject_id"].isin(train_sids).values
        X_tr = np.concatenate([X_tr_orig, aug_X[fm]])
        y_tr = np.concatenate([y_tr_orig, aug_y[fm]])

        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{N_SPLITS}  "
              f"train={len(X_tr)} (Tight×{(y_tr==tight_lbl).sum()})  "
              f"test={len(X_te)}")
        print(f"{'='*60}")

        # ── Episode samplers ──────────────────────────────────────────────────
        # Split train into 80/20 for train-episodes / val-episodes
        rng_split = np.random.default_rng(SEED + fold)
        unique_sids = list(train_sids)
        rng_split.shuffle(unique_sids)
        val_sids  = set(unique_sids[:max(1, len(unique_sids)//5)])
        val_mask  = np.array([
            groups[i] in val_sids for i in train_idx
        ])
        # Augmented beats: source subject still in train_sids; split by subject
        aug_val_mask = aug_meta.loc[fm, "subject_id"].isin(val_sids).values

        # Full mask arrays (original + aug)
        full_val_mask = np.concatenate([val_mask, aug_val_mask])
        X_ep_tr  = X_tr[~full_val_mask]
        y_ep_tr  = y_tr[~full_val_mask]
        X_ep_val = X_tr[full_val_mask]
        y_ep_val = y_tr[full_val_mask]

        # Check minimum class count for val sampler
        min_val_count = min(
            (y_ep_val == c).sum() for c in range(n_classes)
        )
        q_val = min(Q_QUERY, max(1, min_val_count - K_SHOT))

        train_sampler = EpisodeSampler(X_ep_tr, y_ep_tr,
                                       n_way=N_WAY, k_shot=K_SHOT,
                                       q_query=Q_QUERY,
                                       rng=np.random.default_rng(SEED + fold))
        val_sampler   = EpisodeSampler(X_ep_val, y_ep_val,
                                       n_way=N_WAY, k_shot=K_SHOT,
                                       q_query=q_val,
                                       rng=np.random.default_rng(SEED + fold + 1000))

        # ── Model + optimiser ─────────────────────────────────────────────────
        encoder   = PulseEncoder(EMBED_DIM).to(DEVICE)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=WD)
        # Warm-up for WARMUP_EPOCHS epochs (linear 0→LR), then cosine decay
        warmup_sched  = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,          # starts at LR * 1e-3
            end_factor=1.0,
            total_iters=WARMUP_EPOCHS,
        )
        cosine_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, MAX_EPOCHS - WARMUP_EPOCHS)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[WARMUP_EPOCHS],
        )

        # ── Episodic training ─────────────────────────────────────────────────
        best_val_acc, patience_cnt, best_state = 0.0, 0, None

        for epoch in range(1, MAX_EPOCHS + 1):
            tr_loss, tr_acc = run_epoch(
                encoder, train_sampler, optimizer,
                EPISODES_PER_EPOCH, train=True,
            )
            val_loss, val_acc = run_epoch(
                encoder, val_sampler, None,
                max(20, EPISODES_PER_EPOCH // 5), train=False,
            )
            scheduler.step()

            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                patience_cnt = 0
                best_state = {k: v.cpu().clone()
                              for k, v in encoder.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"  Early stop at epoch {epoch}")
                    break

            if epoch % 10 == 0 or epoch == 1:
                print(f"  ep={epoch:3d}  "
                      f"tr_loss={tr_loss:.3f}  tr_acc={tr_acc:.3f}  "
                      f"val_loss={val_loss:.3f}  val_acc={val_acc:.3f}  "
                      f"best_val={best_val_acc:.3f}")

        # Reload best encoder
        if best_state:
            encoder.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_state.items()}
            )

        # ── Nearest-centroid evaluation on held-out fold ───────────────────────
        y_pred, y_prob = nearest_centroid_predict(
            encoder, X_tr, y_tr, X_te, n_classes
        )
        m = evaluate(y_te, y_prob, n_classes)

        print(f"\n  [Nearest Centroid on test fold]")
        print(f"  F1={m['macro_f1']:.3f}  BalAcc={m['bal_acc']:.3f}  "
              f"ROC={m['roc_auc']:.3f}  PR={m['pr_auc']:.3f}")
        print(classification_report(y_te, y_pred,
                                     target_names=CLASS_NAMES, zero_division=0))
        records.append({"fold": fold + 1, **m})

        # Save encoder from fold 0 for potential embedding visualisation
        if fold == 0:
            torch.save(encoder.state_dict(), OUT / "proto_net_encoder.pt")
            print("  Saved → outputs/proto_net_encoder.pt")

    # ── Summary ───────────────────────────────────────────────────────────────
    df_res     = pd.DataFrame(records)
    metric_cols = ["macro_f1", "bal_acc", "roc_auc", "pr_auc"]

    print("\n" + "=" * 60)
    print("ProtoNet 5-fold Summary")
    print("=" * 60)
    for col in metric_cols:
        print(f"  {col:12s}: {df_res[col].mean():.4f} ± {df_res[col].std():.4f}")

    df_res.to_csv(OUT / "proto_net_results.csv", index=False, float_format="%.6f")

    # ── t-SNE embedding visualisation (fold-0 encoder) ─────────────────────────
    print("\nGenerating t-SNE embedding visualisation…")
    try:
        from sklearn.manifold import TSNE
        # Reload fold-0 encoder
        enc_vis = PulseEncoder(EMBED_DIM).to(DEVICE)
        enc_vis.load_state_dict(
            torch.load(OUT / "proto_net_encoder.pt", map_location=DEVICE)
        )
        enc_vis.eval()

        # Encode a stratified random subset (max 800 beats per class)
        rng_vis   = np.random.default_rng(SEED)
        vis_idx   = np.concatenate([
            rng_vis.choice(np.where(y == c)[0],
                           size=min(800, (y == c).sum()), replace=False)
            for c in range(len(CLASS_NAMES))
        ])
        X_vis = X[vis_idx]
        y_vis = y[vis_idx]

        chunks = []
        for start in range(0, len(X_vis), 256):
            xb = torch.from_numpy(
                X_vis[start:start+256, np.newaxis, :]
            ).float().to(DEVICE)
            with torch.no_grad():
                chunks.append(enc_vis(xb).cpu().numpy())
        Z_vis = np.concatenate(chunks)            # (N_vis, D)

        tsne   = TSNE(n_components=2, perplexity=40, random_state=SEED,
                      max_iter=1000, init="pca")
        Z_2d   = tsne.fit_transform(Z_vis)         # (N_vis, 2)

        palette = ["#4C72B0", "#DD8452", "#55A868"]
        fig, ax = plt.subplots(figsize=(7, 6))
        for c, name in enumerate(CLASS_NAMES):
            mask_c = y_vis == c
            ax.scatter(Z_2d[mask_c, 0], Z_2d[mask_c, 1],
                       c=palette[c], label=name, alpha=0.55,
                       s=18, linewidths=0)
        ax.set_title("t-SNE of ProtoNet Embeddings (fold-0 encoder)",
                     fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        fig.tight_layout()
        fig.savefig(OUT / "proto_net_tsne.png", dpi=150)
        plt.close(fig)
        print("  Saved → outputs/proto_net_tsne.png")
    except Exception as e:
        print(f"  t-SNE visualisation skipped: {e}")

    # ── Comparison bar chart (ProtoNet vs best supervised models) ─────────────
    supervised = {
        "CNN aug":       {"macro_f1": 0.820, "bal_acc": 0.824, "roc_auc": 0.969, "pr_auc": 0.863},
        "ResNet1D+GC":   {"macro_f1": 0.834, "bal_acc": 0.844, "roc_auc": 0.968, "pr_auc": 0.857},
    }
    proto_mean = {col: df_res[col].mean() for col in metric_cols}
    proto_std  = {col: df_res[col].std()  for col in metric_cols}

    all_models = {**supervised, "ProtoNet (ours)": proto_mean}
    colors     = ["#4C72B0", "#55A868", "#C44E52"]
    metric_labels = ["Macro-F1", "Balanced Acc", "ROC-AUC", "PR-AUC"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        "Prototypical Network vs Supervised Baselines\n"
        "3-class Pulse  ·  5-fold Subject-level CV",
        fontsize=11, fontweight="bold",
    )
    names = list(all_models.keys())
    for ax, col, label in zip(axes, metric_cols, metric_labels):
        means = [all_models[n][col] for n in names]
        errs  = [proto_std[col] if n == "ProtoNet (ours)" else 0.0 for n in names]
        bars  = ax.bar(range(len(names)), means, yerr=errs, capsize=5,
                       color=colors, alpha=0.85, tick_label=names)
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(max(0, min(means) - 0.12), 1.01)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + 0.005, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "proto_net_comparison.png", dpi=150)
    plt.close(fig)

    # ── Text report ───────────────────────────────────────────────────────────
    report_lines = [
        "Prototypical Network — 3-class Pulse Diagnosis\n"
        f"{N_WAY}-way {K_SHOT}-shot {Q_QUERY}-query  "
        f"embed_dim={EMBED_DIM}  episodes/epoch={EPISODES_PER_EPOCH}\n"
        f"5-fold subject-level CV + 6-mode Tight augmentation\n\n",
        "Fold results:\n",
        df_res.to_string(index=False, float_format="%.4f"),
        "\n\nSummary (mean ± std):\n",
    ]
    for col in metric_cols:
        report_lines.append(
            f"  {col:12s}: {df_res[col].mean():.4f} ± {df_res[col].std():.4f}\n"
        )
    (OUT / "proto_net_report.txt").write_text("".join(report_lines))

    print(f"\n  Saved → outputs/proto_net_results.csv")
    print(f"  Saved → outputs/proto_net_comparison.png")
    print(f"  Saved → outputs/proto_net_report.txt")
    print("=" * 60)
    print(f"DONE — ProtoNet mean Macro-F1 = {df_res['macro_f1'].mean():.4f}")


if __name__ == "__main__":
    main()
