"""
Unified repeated-CV statistical-rerun harness for the TCM 3-class pulse paper.
==============================================================================

Purpose
-------
Re-evaluate every model under a single, apples-to-apples harness with
repeated subject-level cross-validation, emitting per-run scores so that the
downstream statistics (stat_rerun_analysis.py) can be recomputed at will.

What is PRESERVED (unchanged from the original scripts)
-------------------------------------------------------
* Model architectures      : imported verbatim from src/train_*.py
* Preprocessing / features : same beats_matrix.npy / dataset.csv / Gaussian feats
* Leakage-free split logic  : subject-level StratifiedGroupKFold + isolated
                              inner-validation split (cv_utils.make_inner_split),
                              synthetic beats injected into inner-train subjects
                              ONLY, test fold read exactly once.

What is CHANGED (the point of this rerun)
-----------------------------------------
* Evaluation replication   : repeated CV — N_REPEATS partition seeds x 5 folds,
                              all models share the SAME partitions/seeds/samplers.
* Two protocols per model  :
    - 'leakfree'   : early-stop on the isolated inner-validation fold
                     (the correct protocol; == the *_innerval.csv scripts).
    - 'optimistic' : the test fold doubles as the early-stop signal
                     (the original "optimistic upper bound" leak; RESULTS.md).
  ML models (SVM/RF/XGBoost) have no early-stopping channel, so the two
  protocols are identical by construction (inflation Delta == 0); they are run
  once under 'leakfree' and mirrored to 'optimistic' by the analysis step.

Determinism
-----------
Every run seeds torch/np with a deterministic function of (repeat, fold) and
logs the outer partition seed. See RUN_SEED_BASE.

Output
------
Appends one row per (model, protocol, repeat, fold) to RAW_CSV as it goes, so
the job is resumable: an interrupted run re-reads RAW_CSV and skips finished
cells. Columns:
    model, repeat, fold, seed, macro_f1, bal_acc, roc_auc, pr_auc,
    tight_f1, protocol, use_aug, use_sampler, params, n_test, run_seed, epochs
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── Revision-R1 copy of _working_archive/stat_rerun_harness.py ───────────────
# Identical training / splitting / metric logic.  Additions only:
#   * paths point at TCM/ (model code) and TCM/outputs/_intermediate (data)
#   * per-run test-fold probabilities are dumped to outputs/preds/*.npz
#     (needed for subject-level metrics, aggregated confusion matrices,
#      equal-subject-weight sensitivity — Reviewer 2 Major 3 / Reviewer 1 minor)
#   * best-epoch weights of selected models are saved (Grad-CAM on all held-out beats)
#   * --protocols lets us run leakfree only;  --mu-sort orders Gaussian
#     components by mu before the ML models (Reviewer 1 Major 4)
HERE = Path(__file__).parent.resolve()
# Local layout: paper/TCM (code) + paper/TCM/outputs/_intermediate (data).
# Server layout (~/tcm_3class): code in ROOT and ROOT/src, data in ROOT/outputs.
# Override with TCM_ROOT / TCM_DATA / RES_DIR environment variables.
ROOT = Path(os.environ.get("TCM_ROOT", HERE.parent / "TCM")).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(os.environ.get("TCM_DATA", ROOT / "outputs" / "_intermediate")).resolve()
RES = Path(os.environ.get("RES_DIR", HERE / "outputs")).resolve()
PRED_DIR = RES / "preds"
W_DIR = RES / "weights"
for _d in (RES, PRED_DIR, W_DIR):
    _d.mkdir(parents=True, exist_ok=True)
MU_SORT = True                                   # set from CLI in main()
SAVE_WEIGHTS: set = set()                        # model names whose weights to keep

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

# ── Model architectures (imported verbatim — NOT modified) ───────────────────
from train_cnn import PulseCNN
from train_advanced_models import PulseResNet, PulseInception, PulseTransformer
from train_gated_models import (
    PulseResNetGC, PulseResNetGated, PulseResNetGCGated, count_params,
)
from train_proto_net import (
    PulseEncoder, EpisodeSampler, nearest_centroid_predict, run_epoch,
    N_WAY, K_SHOT, Q_QUERY, EMBED_DIM, EPISODES_PER_EPOCH, MAX_EPOCHS as PROTO_MAX_EPOCHS,
    PATIENCE as PROTO_PATIENCE, LR as PROTO_LR, WD as PROTO_WD, WARMUP_EPOCHS as PROTO_WARMUP,
)
from cv_utils import make_inner_split, validate_inner_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
# Cap intra-op threads so many sharded workers can share the CPU (GPU is idle for
# these tiny models — the bottleneck is CPU/Python, so parallel workers scale well).
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

# ── Constants ────────────────────────────────────────────────────────────────
CLASS_ORDER = ["Wiry", "Slippery", "Tight"]
N_SPLITS = 5
INNER_SEED = 42            # inner-validation split seed (unchanged from originals)
RUN_SEED_BASE = 1000       # per-run torch/np seed = BASE + repeat*100 + fold*10 + model_id
FEATURE_COLS = ["A1", "mu1", "sigma1", "A2", "mu2", "sigma2", "A3", "mu3", "sigma3"]
BATCH_SIZE = 64

RAW_COLS = ["model", "repeat", "fold", "seed", "macro_f1", "bal_acc", "roc_auc",
            "pr_auc", "tight_f1", "protocol", "use_aug", "use_sampler",
            "params", "n_test", "run_seed", "epochs"]

# TIGHT class index under LabelEncoder().fit(sorted(CLASS_ORDER)) ->
# sorted = ['Slippery','Tight','Wiry'] -> Tight == index 1
CLASS_NAMES = sorted(CLASS_ORDER)
TIGHT_IDX = CLASS_NAMES.index("Tight")
N_CLASSES = len(CLASS_NAMES)


# ── Deep-model registry ──────────────────────────────────────────────────────
# builder: callable -> nn.Module ;  cfg mirrors the original per-model configs.
DEEP_CONFIGS = {
    # 1D-CNN (train_cnn.py config): the augmentation-headline model.
    "cnn_noaug":  dict(cls=PulseCNN,          epochs=80,  patience=20, lr=1e-3, wd=1e-4, smooth=0.00, clip=None, warmup=0, use_aug=False, use_sampler=True),
    "cnn_aug":    dict(cls=PulseCNN,          epochs=80,  patience=20, lr=1e-3, wd=1e-4, smooth=0.00, clip=None, warmup=0, use_aug=True,  use_sampler=True),
    # Advanced models (train_advanced_models.py configs; all aug=True).
    "resnet":     dict(cls=PulseResNet,       epochs=150, patience=30, lr=1e-3, wd=1e-4, smooth=0.10, clip=None, warmup=0, use_aug=True,  use_sampler=True),
    "inception":  dict(cls=PulseInception,    epochs=150, patience=30, lr=1e-3, wd=1e-4, smooth=0.10, clip=None, warmup=0, use_aug=True,  use_sampler=True),
    "transformer":dict(cls=PulseTransformer,  epochs=200, patience=40, lr=3e-4, wd=1e-3, smooth=0.10, clip=1.0,  warmup=10,use_aug=True,  use_sampler=True),
    # Gated / GC ResNet variants (train_gated_models.py configs; all aug=True).
    "resnet_gc":  dict(cls=PulseResNetGC,     epochs=150, patience=30, lr=1e-3, wd=1e-4, smooth=0.10, clip=None, warmup=0, use_aug=True,  use_sampler=True),
    "resnet_gated":dict(cls=PulseResNetGated, epochs=150, patience=30, lr=1e-3, wd=1e-4, smooth=0.10, clip=None, warmup=0, use_aug=True,  use_sampler=True),
    "resnet_gc_gated":dict(cls=PulseResNetGCGated, epochs=150, patience=35, lr=8e-4, wd=2e-4, smooth=0.10, clip=None, warmup=5, use_aug=True, use_sampler=True),
    # 2x2 ablation extra cells on the 1D-CNN (leakfree only).
    "cnn_aug_nosampler":   dict(cls=PulseCNN, epochs=80, patience=20, lr=1e-3, wd=1e-4, smooth=0.00, clip=None, warmup=0, use_aug=True,  use_sampler=False),
    "cnn_noaug_nosampler": dict(cls=PulseCNN, epochs=80, patience=20, lr=1e-3, wd=1e-4, smooth=0.00, clip=None, warmup=0, use_aug=False, use_sampler=False),
}
# stable integer id per model for seed derivation
MODEL_IDS = {name: i for i, name in enumerate(
    list(DEEP_CONFIGS.keys()) + ["protonet", "SVM", "RandomForest", "XGBoost"])}

# Which models get BOTH protocols vs leakfree only.
ABLATION_ONLY = {"cnn_aug_nosampler", "cnn_noaug_nosampler"}   # leakfree only
ML_MODELS = ["SVM", "RandomForest", "XGBoost"]                  # leakfree only (== optimistic)


# ── Data loading (once) ──────────────────────────────────────────────────────
def sort_components(Fm):
    """Reorder (A_i, mu_i, sigma_i) triplets so mu1 < mu2 < mu3 (label-switching fix)."""
    P = Fm.reshape(-1, 3, 3)
    order = np.argsort(P[:, :, 1], axis=1, kind="stable")
    changed = int((order != np.arange(3)).any(axis=1).sum())
    return np.take_along_axis(P, order[:, :, None], axis=1).reshape(-1, 9), changed


def load_data():
    beats = np.load(OUT / "beats_matrix.npy").astype(np.float32)  # (N,200)
    df = pd.read_csv(OUT / "dataset.csv")
    assert len(beats) == len(df)
    mask = df["pulse_type_en"].isin(CLASS_ORDER)
    beats = beats[mask.values]
    df = df[mask].reset_index(drop=True)

    le = LabelEncoder().fit(sorted(CLASS_ORDER))
    y = le.transform(df["pulse_type_en"].values)
    groups = df["subject_id"].values
    X = beats[:, np.newaxis, :]           # (N,1,200) for deep
    Xflat = beats                          # (N,200)   for protonet
    feats = df[FEATURE_COLS].values.astype(np.float64)  # ML features
    converged = df["converged"].values.astype(bool)

    # Augmented Tight (synthetic only)
    aug_beats_full = np.load(OUT / "tight_augmented_beats.npy").astype(np.float32)
    aug_meta_full = pd.read_csv(OUT / "tight_augmented_meta.csv")
    synth = (aug_meta_full["aug_mode"] != "original").values
    aug_meta = aug_meta_full[synth].reset_index(drop=True)
    aug_X = aug_beats_full[synth][:, np.newaxis, :]       # (K,1,200)
    aug_Xflat = aug_beats_full[synth]                      # (K,200)
    tight_lbl = int(le.transform(["Tight"])[0])
    aug_y = np.full(len(aug_meta), tight_lbl, dtype=int)
    aug_subj = aug_meta["subject_id"].values
    aug_feats = aug_meta[FEATURE_COLS].values.astype(np.float64)
    aug_conv = aug_meta["converged"].values.astype(bool)

    if MU_SORT:
        feats, n1 = sort_components(feats)
        aug_feats, n2 = sort_components(aug_feats)
        print(f"mu-sorted Gaussian components: reordered {n1} original / {n2} synthetic rows")

    return dict(X=X, Xflat=Xflat, y=y, groups=groups, feats=feats, converged=converged,
                aug_X=aug_X, aug_Xflat=aug_Xflat, aug_y=aug_y, aug_subj=aug_subj,
                aug_feats=aug_feats, aug_conv=aug_conv, tight_lbl=tight_lbl)


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_prob):
    y_pred = y_prob.argmax(axis=1)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=list(range(N_CLASSES)), zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    tight_f1 = float(per_class_f1[TIGHT_IDX])
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        roc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                            labels=list(range(N_CLASSES)))
    except ValueError:
        roc = float("nan")
    pr_list = [average_precision_score((y_true == c).astype(int), y_prob[:, c])
               for c in range(N_CLASSES) if (y_true == c).sum() > 0]
    pr = float(np.mean(pr_list)) if pr_list else float("nan")
    return dict(macro_f1=float(macro_f1), bal_acc=float(bal_acc), roc_auc=float(roc),
                pr_auc=pr, tight_f1=tight_f1)


def class_weights_tensor(labels):
    counts = np.bincount(labels, minlength=N_CLASSES).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (N_CLASSES * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def make_weighted_sampler(labels):
    _, counts = np.unique(labels, return_counts=True)
    sw = (1.0 / counts)[labels]
    return WeightedRandomSampler(torch.from_numpy(sw).float(), num_samples=len(labels),
                                 replacement=True)


def get_scheduler(optimizer, cfg, epochs):
    import math
    if cfg["warmup"] > 0:
        def lr_fn(ep):
            if ep < cfg["warmup"]:
                return (ep + 1) / cfg["warmup"]
            progress = (ep - cfg["warmup"]) / max(1, epochs - cfg["warmup"])
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# ── One deep training run ────────────────────────────────────────────────────
def run_deep(model_name, cfg, split, data, protocol, run_seed):
    """Train one deep model on one fold under `protocol`. Returns metrics dict."""
    train_idx, test_idx = split
    X, y, groups = data["X"], data["y"], data["groups"]
    aug_X, aug_y, aug_subj = data["aug_X"], data["aug_y"], data["aug_subj"]

    torch.manual_seed(run_seed)
    np.random.seed(run_seed)

    X_tr_full, y_tr_full = X[train_idx], y[train_idx]
    groups_tr_full = groups[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    use_aug = cfg["use_aug"]
    use_sampler = cfg["use_sampler"]

    if protocol == "leakfree":
        inner_tr_idx, inner_val_idx = make_inner_split(
            y_tr_full, groups_tr_full, n_inner=5, seed=INNER_SEED)
        vr = validate_inner_split(inner_tr_idx, inner_val_idx, groups_tr_full, y_tr_full)
        assert vr["no_subject_overlap"], f"{model_name}: inner split subject overlap!"
        X_in_tr, y_in_tr = X_tr_full[inner_tr_idx], y_tr_full[inner_tr_idx]
        X_val, y_val = X_tr_full[inner_val_idx], y_tr_full[inner_val_idx]
        aug_ok_subj = set(groups_tr_full[inner_tr_idx])
    else:  # optimistic: whole train fold trains; TEST fold is the early-stop signal
        X_in_tr, y_in_tr = X_tr_full, y_tr_full
        X_val, y_val = X_te, y_te
        aug_ok_subj = set(groups_tr_full)

    if use_aug:
        fm = np.isin(aug_subj, list(aug_ok_subj))
        X_in_tr = np.concatenate([X_in_tr, aug_X[fm]], axis=0)
        y_in_tr = np.concatenate([y_in_tr, aug_y[fm]], axis=0)

    # loaders
    tr_ds = TensorDataset(torch.from_numpy(X_in_tr), torch.from_numpy(y_in_tr).long())
    if use_sampler:
        train_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE,
                                  sampler=make_weighted_sampler(y_in_tr))
    else:
        train_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long()),
                            batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te).long()),
                             batch_size=256, shuffle=False)

    model = cfg["cls"](N_CLASSES).to(DEVICE)
    cw = class_weights_tensor(y_in_tr)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg["smooth"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = get_scheduler(optimizer, cfg, cfg["epochs"])

    best_loss, patience_cnt, best_state, last_ep = float("inf"), 0, None, 0
    for ep in range(1, cfg["epochs"] + 1):
        last_ep = ep
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            if cfg["clip"] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += F.cross_entropy(model(xb.to(DEVICE)), yb.to(DEVICE),
                                            reduction="sum").item()
        val_loss /= len(y_val)
        if val_loss < best_loss - 1e-5:
            best_loss, patience_cnt = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in test_loader:
            probs.append(torch.softmax(model(xb.to(DEVICE)), dim=1).cpu().numpy())
    y_prob = np.concatenate(probs)
    m = compute_metrics(y_te, y_prob)
    m["epochs"] = last_ep
    m["n_test"] = len(y_te)
    m["params"] = count_params(model)
    m["_y_prob"] = y_prob                         # popped & saved by main()
    if model_name in SAVE_WEIGHTS:
        m["_state"] = {k: v.cpu() for k, v in model.state_dict().items()}
    return m


# ── One ProtoNet run ─────────────────────────────────────────────────────────
def run_proto(split, data, protocol, run_seed):
    import random
    train_idx, test_idx = split
    Xf, y, groups = data["Xflat"], data["y"], data["groups"]
    aug_Xf, aug_y, aug_subj = data["aug_Xflat"], data["aug_y"], data["aug_subj"]
    tight_lbl = data["tight_lbl"]

    torch.manual_seed(run_seed); np.random.seed(run_seed); random.seed(run_seed)

    X_tr_full, y_tr_full = Xf[train_idx], y[train_idx]
    groups_tr_full = groups[train_idx]
    X_te, y_te = Xf[test_idx], y[test_idx]

    if protocol == "leakfree":
        inner_tr_idx, inner_val_idx = make_inner_split(
            y_tr_full, groups_tr_full, n_inner=5, seed=INNER_SEED)
        X_in_tr, y_in_tr = X_tr_full[inner_tr_idx], y_tr_full[inner_tr_idx]
        X_val, y_val = X_tr_full[inner_val_idx], y_tr_full[inner_val_idx]
        aug_ok_subj = set(groups_tr_full[inner_tr_idx])
    else:  # optimistic: test fold is the val signal
        X_in_tr, y_in_tr = X_tr_full, y_tr_full
        X_val, y_val = X_te, y_te
        aug_ok_subj = set(groups_tr_full)

    fm = np.isin(aug_subj, list(aug_ok_subj))
    X_in_tr = np.concatenate([X_in_tr, aug_Xf[fm]])
    y_in_tr = np.concatenate([y_in_tr, aug_y[fm]])

    min_val = min((y_val == c).sum() for c in range(N_CLASSES))
    q_val = min(Q_QUERY, max(1, min_val - K_SHOT))
    if min_val < K_SHOT + 1:
        # too few in some class to form an episode; fall back to accuracy-free training
        q_val = max(1, min_val - 1) if min_val > 1 else 1

    train_sampler = EpisodeSampler(X_in_tr, y_in_tr, n_way=N_WAY, k_shot=K_SHOT,
                                   q_query=Q_QUERY, rng=np.random.default_rng(run_seed))
    val_sampler = EpisodeSampler(X_val, y_val, n_way=N_WAY, k_shot=K_SHOT,
                                 q_query=q_val, rng=np.random.default_rng(run_seed + 1000))

    encoder = PulseEncoder(EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=PROTO_LR, weight_decay=PROTO_WD)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                                               total_iters=PROTO_WARMUP)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, PROTO_MAX_EPOCHS - PROTO_WARMUP))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[PROTO_WARMUP])

    best_val_acc, patience_cnt, best_state, last_ep = 0.0, 0, None, 0
    for ep in range(1, PROTO_MAX_EPOCHS + 1):
        last_ep = ep
        run_epoch(encoder, train_sampler, optimizer, EPISODES_PER_EPOCH, train=True)
        _, val_acc = run_epoch(encoder, val_sampler, None, max(20, EPISODES_PER_EPOCH // 5), train=False)
        scheduler.step()
        if val_acc > best_val_acc + 1e-4:
            best_val_acc, patience_cnt = val_acc, 0
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PROTO_PATIENCE:
                break
    if best_state is not None:
        encoder.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    y_pred, y_prob = nearest_centroid_predict(encoder, X_in_tr, y_in_tr, X_te, N_CLASSES)
    m = compute_metrics(y_te, y_prob)
    m["epochs"] = last_ep
    m["n_test"] = len(y_te)
    m["params"] = count_params(encoder)
    m["_y_prob"] = np.asarray(y_prob)
    return m


# ── One ML run (SVM / RF / XGBoost) — all three at once per fold ─────────────
def run_ml(split, data, run_seed):
    from xgboost import XGBClassifier
    train_idx, test_idx = split
    feats, y, groups = data["feats"], data["y"], data["groups"]
    conv = data["converged"]
    aug_feats, aug_y, aug_subj, aug_conv = data["aug_feats"], data["aug_y"], data["aug_subj"], data["aug_conv"]

    # ML pipeline uses only converged beats (matches train_ml.py)
    tr_mask = conv[train_idx]
    te_mask = conv[test_idx]
    tr_i = train_idx[tr_mask]
    te_i = test_idx[te_mask]
    X_tr, y_tr = feats[tr_i], y[tr_i]
    X_te, y_te = feats[te_i], y[te_i]
    tr_subj = groups[tr_i]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # inject aug (converged synthetic, train-fold subjects only)
    amask = aug_conv & np.isin(aug_subj, list(set(tr_subj)))
    X_aug = scaler.transform(aug_feats[amask])
    y_aug = aug_y[amask]
    X_res = np.vstack([X_tr, X_aug])
    y_res = np.concatenate([y_tr, y_aug])
    sw = compute_sample_weight("balanced", y_res)

    models = {
        "SVM": SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                   probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=None,
                                                class_weight="balanced", n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                                 n_jobs=-1, random_state=42, verbosity=0),
    }
    out = {}
    for name, clf in models.items():
        if name == "XGBoost":
            clf.fit(X_res, y_res, sample_weight=sw)
        else:
            clf.fit(X_res, y_res)
        y_prob = clf.predict_proba(X_te)
        # align columns to 0..N-1 in case a class is missing (shouldn't be)
        m = compute_metrics(y_te, y_prob)
        m["epochs"] = 0
        m["n_test"] = len(y_te)
        m["params"] = 0
        m["_y_prob"] = y_prob
        m["_te_i"] = te_i                         # converged-only test indices
        out[name] = m
    return out


def save_preds(name, proto, repeat, fold, test_idx, y_true, m):
    """Pop the per-beat outputs from a metrics dict and persist them."""
    y_prob = m.pop("_y_prob")
    te_i = m.pop("_te_i", test_idx)
    np.savez_compressed(PRED_DIR / f"{name}__{proto}__r{repeat}_f{fold}.npz",
                        test_idx=np.asarray(te_i), y_true=np.asarray(y_true), y_prob=y_prob)
    state = m.pop("_state", None)
    if state is not None:
        torch.save(state, W_DIR / f"{name}__{proto}__r{repeat}_f{fold}.pt")


# ── Runner ───────────────────────────────────────────────────────────────────
def load_done(raw_csv):
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
        return set(zip(df["model"], df["protocol"], df["repeat"], df["fold"]))
    return set()


def append_row(raw_csv, row):
    header = not raw_csv.exists()
    pd.DataFrame([row])[RAW_COLS].to_csv(raw_csv, mode="a", header=header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--only-repeat", type=int, default=None, help="run a single repeat seed")
    ap.add_argument("--models", type=str, default=None, help="comma list to restrict")
    ap.add_argument("--protocols", type=str, default="leakfree",
                    help="comma list: leakfree,optimistic (default leakfree only)")
    ap.add_argument("--save-weights", type=str, default="resnet,resnet_gc",
                    help="comma list of deep models whose best-epoch weights are saved")
    ap.add_argument("--no-mu-sort", action="store_true", help="keep original Gaussian component order")
    args = ap.parse_args()

    global MU_SORT, SAVE_WEIGHTS
    MU_SORT = not args.no_mu_sort
    SAVE_WEIGHTS = set(args.save_weights.split(",")) if args.save_weights else set()
    wanted_protocols = args.protocols.split(",")

    if args.only_repeat is not None:
        raw_csv = RES / f"deep_rerun_raw_r{args.only_repeat}.csv"
    else:
        raw_csv = RES / "deep_rerun_raw.csv"

    print(f"Device: {DEVICE}  |  repeats={args.repeats}  |  threads={torch.get_num_threads()}  |  csv={raw_csv.name}")
    data = load_data()
    print(f"Data: X={data['X'].shape}  Tight_idx={TIGHT_IDX}  "
          f"aug={data['aug_X'].shape}  subjects={len(np.unique(data['groups']))}")

    done = load_done(raw_csv)
    print(f"Already done cells: {len(done)}")

    repeats = [args.only_repeat] if args.only_repeat is not None else list(range(args.repeats))
    restrict = set(args.models.split(",")) if args.models else None

    deep_names = [n for n in DEEP_CONFIGS]
    t_start = time.time()

    for repeat in repeats:
        skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=repeat)
        splits = list(skf.split(data["X"], data["y"], data["groups"]))
        for fold, split in enumerate(splits):
            # ── Deep models ──
            for name in deep_names:
                if restrict and name not in restrict:
                    continue
                cfg = DEEP_CONFIGS[name]
                protocols = ["leakfree"] if name in ABLATION_ONLY else ["leakfree", "optimistic"]
                protocols = [p for p in protocols if p in wanted_protocols]
                for proto in protocols:
                    key = (name, proto, repeat, fold)
                    if key in done:
                        continue
                    run_seed = RUN_SEED_BASE + repeat * 100 + fold * 10 + MODEL_IDS[name]
                    t0 = time.time()
                    m = run_deep(name, cfg, split, data, proto, run_seed)
                    save_preds(name, proto, repeat, fold, split[1], data["y"][split[1]], m)
                    append_row(raw_csv, dict(model=name, repeat=repeat, fold=fold, seed=repeat,
                                    protocol=proto, use_aug=cfg["use_aug"],
                                    use_sampler=cfg["use_sampler"], run_seed=run_seed, **m))
                    print(f"[{time.time()-t_start:7.0f}s] r{repeat} f{fold} {name:20s} {proto:10s} "
                          f"F1={m['macro_f1']:.3f} tight={m['tight_f1']:.3f} ep={m['epochs']:3d} "
                          f"({time.time()-t0:.1f}s)", flush=True)

            # ── ProtoNet ──
            if not restrict or "protonet" in restrict:
                for proto in [p for p in ["leakfree", "optimistic"] if p in wanted_protocols]:
                    key = ("protonet", proto, repeat, fold)
                    if key in done:
                        continue
                    run_seed = RUN_SEED_BASE + repeat * 100 + fold * 10 + MODEL_IDS["protonet"]
                    t0 = time.time()
                    try:
                        m = run_proto(split, data, proto, run_seed)
                        save_preds("protonet", proto, repeat, fold, split[1], data["y"][split[1]], m)
                        append_row(raw_csv, dict(model="protonet", repeat=repeat, fold=fold, seed=repeat,
                                        protocol=proto, use_aug=True, use_sampler=False,
                                        run_seed=run_seed, **m))
                        print(f"[{time.time()-t_start:7.0f}s] r{repeat} f{fold} {'protonet':20s} {proto:10s} "
                              f"F1={m['macro_f1']:.3f} tight={m['tight_f1']:.3f} ep={m['epochs']:3d} "
                              f"({time.time()-t0:.1f}s)", flush=True)
                    except Exception as e:
                        print(f"  protonet r{repeat} f{fold} {proto} FAILED: {e}", flush=True)

            # ── ML models (leakfree only; == optimistic) ──
            if not restrict or any(m in (restrict or set()) for m in ML_MODELS):
                ml_key = ("SVM", "leakfree", repeat, fold)
                if ml_key not in done:
                    run_seed = RUN_SEED_BASE + repeat * 100 + fold * 10 + MODEL_IDS["SVM"]
                    t0 = time.time()
                    ml = run_ml(split, data, run_seed)
                    for name, m in ml.items():
                        save_preds(name, "leakfree", repeat, fold, split[1], data["y"][m["_te_i"]], m)
                        append_row(raw_csv, dict(model=name, repeat=repeat, fold=fold, seed=repeat,
                                        protocol="leakfree", use_aug=True, use_sampler=False,
                                        run_seed=run_seed, **m))
                    print(f"[{time.time()-t_start:7.0f}s] r{repeat} f{fold} {'ML(3)':20s} {'leakfree':10s} "
                          f"SVM={ml['SVM']['macro_f1']:.3f} RF={ml['RandomForest']['macro_f1']:.3f} "
                          f"XGB={ml['XGBoost']['macro_f1']:.3f} ({time.time()-t0:.1f}s)", flush=True)

    print(f"\nDONE in {time.time()-t_start:.0f}s. Rows in {raw_csv}.")


if __name__ == "__main__":
    main()
