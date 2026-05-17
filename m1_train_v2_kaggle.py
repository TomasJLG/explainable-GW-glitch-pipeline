"""
m1_train_v2_kaggle.py
Self-contained Kaggle script — no external project imports.

Trains M1 v2:
  - All 4 run02 scale_1p0 NPZs (H1/O3a, H1/O3b, L1/O3a, L1/O3b) — ~8000 nominales
  - latent_dim=32  (more compressed bottleneck than v1's 128)
  - Temporal split (P80 of t0)
  - Early stopping, patience=15

Then evaluates with run03 Gravity Spy labels (if dataset uploaded to Kaggle),
or falls back to inferred labels (P95 log_energy) on the val set.
"""

import glob
import json
import os
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        roc_curve,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn not available — AUROC metrics will be skipped")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NPZ_BASE   = "/kaggle/input/datasets/tomsjacobolegal/qdataset-npz-v2-output/mi_dataset/run02"
RUN03_PATH = "/kaggle/input/datasets/tomsjacobolegal/run03-h1-o3a/dataset_H1_O3a_scale_1p0s_run03.npz"
OUT_DIR    = "/kaggle/working"

LATENT_DIM   = 32
BATCH_SIZE   = 64
EPOCHS       = 80
LR           = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE     = 15
ENERGY_PCT   = 95.0  # percentile threshold for nominal filter

# v1 baseline for comparison table
V1_AUROC_AE   = 0.5265
V1_AUROC_COMB = 0.5269

NPZ_CONFIGS = [
    ("H1", "O3a", f"{NPZ_BASE}/H1/O3a/scale_1p0/dataset_H1_O3a_scale_1p0_run02.npz"),
    ("H1", "O3b", f"{NPZ_BASE}/H1/O3b/scale_1p0/dataset_H1_O3b_scale_1p0_run02.npz"),
    ("L1", "O3a", f"{NPZ_BASE}/L1/O3a/scale_1p0/dataset_L1_O3a_scale_1p0_run02.npz"),
    ("L1", "O3b", f"{NPZ_BASE}/L1/O3b/scale_1p0/dataset_L1_O3b_scale_1p0_run02.npz"),
]

# ---------------------------------------------------------------------------
# GlitchAE — inline, latent_dim=32
# ---------------------------------------------------------------------------

class GlitchAE(nn.Module):
    """
    Convolutional autoencoder for glitch detection.
    Input/output: (N, 1, 128, 128).
    """
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1,   32,  3, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32,  64,  3, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512), nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256 * 8 * 8), nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 8, 8)),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  1,   4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder_conv(self.decoder_fc(self.encoder_fc(self.encoder_conv(x))))

    def encode(self, x):
        return self.encoder_fc(self.encoder_conv(x))


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def compute_anomaly_scores(model, X_tensor, device, batch_size=256):
    model.eval()
    scores = []
    dl = DataLoader(TensorDataset(X_tensor), batch_size=batch_size)
    with torch.no_grad():
        for (x,) in dl:
            x = x.to(device)
            mse = ((x - model(x)) ** 2).mean(dim=[1, 2, 3])
            scores.append(mse.cpu().numpy())
    return np.concatenate(scores)


def compute_combined_score(ae_scores, log_energy, alpha=0.7):
    ae_n = (ae_scores   - ae_scores.min())   / (ae_scores.max()   - ae_scores.min()   + 1e-8)
    en_n = (log_energy  - log_energy.min())  / (log_energy.max()  - log_energy.min()  + 1e-8)
    return alpha * ae_n + (1 - alpha) * en_n


def apply_normalization(X, p1, p99):
    return np.clip((X - p1) / (p99 - p1 + 1e-8), 0.0, 1.0).astype(np.float32)


def make_tensor(X_norm):
    return torch.tensor(X_norm[:, None, :, :], dtype=torch.float32)


def tpr_at_fpr(fpr_arr, tpr_arr, target=0.01):
    idx = np.searchsorted(fpr_arr, target)
    if idx == 0:          return float(tpr_arr[0])
    if idx >= len(fpr_arr): return float(tpr_arr[-1])
    x0, x1 = fpr_arr[idx - 1], fpr_arr[idx]
    y0, y1 = tpr_arr[idx - 1], tpr_arr[idx]
    return float(y0 + (y1 - y0) * (target - x0) / (x1 - x0 + 1e-12))


def safe_auroc(binary, scores):
    if not HAS_SKLEARN:
        return None
    if binary.sum() == 0 or (binary == 0).sum() == 0:
        return None
    return float(roc_auc_score(binary, scores))


# ---------------------------------------------------------------------------
# Phase 1 — Load run02 data
# ---------------------------------------------------------------------------

def load_run02_npz(path):
    npz = np.load(path, allow_pickle=True)
    X          = npz["X"].astype(np.float32)
    t0         = npz["t0"].astype(np.float64) if "t0" in npz else np.arange(len(X), dtype=np.float64)
    log_energy = npz["log_energy"].astype(np.float32) if "log_energy" in npz else np.zeros(len(X), dtype=np.float32)
    return X, t0, log_energy


def phase1_load():
    print("\n" + "=" * 60)
    print("Phase 1 — Loading run02 (scale_1p0, all IFO×epoch)")
    print("=" * 60)

    X_parts, t0_parts, loge_parts = [], [], []
    counts = {}

    for ifo, epoch, path in NPZ_CONFIGS:
        if not Path(path).exists():
            print(f"  [SKIP] Not found: {path}")
            continue
        X, t0, loge = load_run02_npz(path)
        thr  = float(np.percentile(loge, ENERGY_PCT))
        mask = loge <= thr
        counts[f"{ifo}/{epoch}"] = int(mask.sum())
        print(f"  {ifo}/{epoch}: {len(X)} total  →  {mask.sum()} nominal  (P{ENERGY_PCT:.0f}={thr:.4f})")
        X_parts.append(X[mask])
        t0_parts.append(t0[mask])
        loge_parts.append(loge[mask])

    if not X_parts:
        print("[ERROR] No NPZ files loaded — check NPZ_BASE path.")
        return None, None, None, None, None, None, {}

    X_all    = np.concatenate(X_parts,    axis=0)
    t0_all   = np.concatenate(t0_parts,   axis=0)
    loge_all = np.concatenate(loge_parts, axis=0)

    p80       = float(np.percentile(t0_all, 80))
    tr_mask   = t0_all <= p80
    va_mask   = t0_all >  p80

    print(f"\n  Total nominal : {len(X_all)}")
    print(f"  Temporal split: P80 = GPS {p80:.0f}")
    print(f"    Train : {tr_mask.sum()}")
    print(f"    Val   : {va_mask.sum()}")

    return (
        X_all[tr_mask], t0_all[tr_mask], loge_all[tr_mask],
        X_all[va_mask], t0_all[va_mask], loge_all[va_mask],
        counts,
    )


# ---------------------------------------------------------------------------
# Phase 2 — Normalization
# ---------------------------------------------------------------------------

def phase2_normalize(X_train, X_val, out_dir):
    print("\n" + "=" * 60)
    print("Phase 2 — Normalization (p1/p99 from train set)")
    print("=" * 60)

    p1  = float(np.percentile(X_train, 1))
    p99 = float(np.percentile(X_train, 99))
    print(f"  p1={p1:.6f}   p99={p99:.6f}")

    norm_path = Path(out_dir) / "normalization_v2.json"
    with open(norm_path, "w") as f:
        json.dump({"p1": p1, "p99": p99}, f, indent=2)
    print(f"  Saved: {norm_path}")

    return apply_normalization(X_train, p1, p99), apply_normalization(X_val, p1, p99), p1, p99


# ---------------------------------------------------------------------------
# Phase 3 + 4 — Model definition + training
# ---------------------------------------------------------------------------

def phase34_train(X_train_n, X_val_n, device, out_dir):
    print("\n" + "=" * 60)
    print(f"Phase 3+4 — Training  latent_dim={LATENT_DIM}  epochs={EPOCHS}  bs={BATCH_SIZE}")
    print("=" * 60)

    model = GlitchAE(latent_dim=LATENT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_dl = DataLoader(
        TensorDataset(make_tensor(X_train_n)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_dl = DataLoader(
        TensorDataset(make_tensor(X_val_n)),
        batch_size=BATCH_SIZE * 2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5, verbose=False)
    criterion = nn.MSELoss()
    weights_path = Path(out_dir) / "best_m1_ae_v2.pt"

    best_val      = float("inf")
    patience_cnt  = 0
    log_rows      = []
    t_start       = time.time()

    print(f"\n  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'Status'}")
    print(f"  {'------':>6}  {'----------':>10}  {'----------':>10}  {'------'}")

    for epoch in range(1, EPOCHS + 1):
        # -- train --
        model.train()
        train_loss = 0.0
        for (x,) in train_dl:
            x = x.to(device)
            loss = criterion(model(x), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(X_train_n)

        # -- val --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x,) in val_dl:
                x = x.to(device)
                val_loss += criterion(model(x), x).item() * len(x)
        val_loss /= len(X_val_n)

        scheduler.step(val_loss)
        log_rows.append({"epoch": epoch, "train_loss": round(train_loss, 8), "val_loss": round(val_loss, 8)})

        status = ""
        if val_loss < best_val:
            best_val     = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), str(weights_path))
            status = "best *"
        else:
            patience_cnt += 1

        if epoch % 5 == 0 or status or patience_cnt == PATIENCE:
            elapsed = time.time() - t_start
            print(f"  {epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  "
                  f"{status or f'{patience_cnt}/{PATIENCE}'}  ({elapsed:.0f}s)")

        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    # reload best
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    model.eval()
    print(f"\n  Best val loss : {best_val:.6f}")
    print(f"  Saved weights : {weights_path}")

    with open(Path(out_dir) / "training_log_v2.json", "w") as f:
        json.dump({"latent_dim": LATENT_DIM, "best_val_loss": best_val, "epochs": log_rows}, f, indent=2)

    # training curve
    ep = [r["epoch"]      for r in log_rows]
    tr = [r["train_loss"] for r in log_rows]
    va = [r["val_loss"]   for r in log_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, tr, label="Train")
    ax.plot(ep, va, label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title(f"M1 v2 training  (latent_dim={LATENT_DIM})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "training_curve_v2.png"), dpi=150)
    plt.close(fig)

    return model


# ---------------------------------------------------------------------------
# Shared eval helper
# ---------------------------------------------------------------------------

def eval_metrics_and_plots(ae_scores, combined, binary, labels_str, loge,
                            out_dir, tag, p1, p99, is_real_labels):
    """
    Computes metrics + generates roc / distribution / boxplot plots.
    Returns dict of metrics (empty if sklearn missing or class imbalance).
    """
    out = {}
    n_nom  = int((binary == 0).sum())
    n_glit = int((binary == 1).sum())

    if HAS_SKLEARN and n_nom > 0 and n_glit > 0:
        auroc_ae  = roc_auc_score(binary, ae_scores)
        auprc_ae  = average_precision_score(binary, ae_scores)
        fpr_ae, tpr_ae, _ = roc_curve(binary, ae_scores)
        tpr1_ae   = tpr_at_fpr(fpr_ae, tpr_ae)

        auroc_c   = roc_auc_score(binary, combined)
        auprc_c   = average_precision_score(binary, combined)
        fpr_c,  tpr_c,  _ = roc_curve(binary, combined)
        tpr1_c    = tpr_at_fpr(fpr_c, tpr_c)

        out = dict(auroc_ae=auroc_ae, auprc_ae=auprc_ae, tpr1_ae=tpr1_ae,
                   auroc_c=auroc_c,   auprc_c=auprc_c,   tpr1_c=tpr1_c)

        label_type = "Gravity Spy real" if is_real_labels else "inferred (P95)"
        print(f"\n  Labels: {label_type}  |  nominal={n_nom}  glitch={n_glit}")
        print(f"  {'Score':<22} {'AUROC':>8} {'AUPRC':>8} {'TPR@FPR1%':>10}")
        print(f"  {'v2 AE only':<22} {auroc_ae:>8.4f} {auprc_ae:>8.4f} {tpr1_ae:>10.4f}")
        print(f"  {'v2 Combined(α=0.7)':<22} {auroc_c:>8.4f} {auprc_c:>8.4f} {tpr1_c:>10.4f}")

        if is_real_labels:
            print(f"\n  {'Metric':<24} {'v1':>8} {'v2':>8} {'Δ':>8}")
            print(f"  {'AUROC AE':<24} {V1_AUROC_AE:>8.4f} {auroc_ae:>8.4f} {auroc_ae - V1_AUROC_AE:>+8.4f}")
            print(f"  {'AUROC Combined':<24} {V1_AUROC_COMB:>8.4f} {auroc_c:>8.4f} {auroc_c - V1_AUROC_COMB:>+8.4f}")

        # ROC (only when run03 available — real labels)
        if is_real_labels:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr_ae, tpr_ae, lw=2,       label=f"AE only    AUROC={auroc_ae:.3f}")
            ax.plot(fpr_c,  tpr_c,  lw=2, ls="--", label=f"Combined   AUROC={auroc_c:.3f}")
            ax.plot([0, 1], [0, 1], "k:", lw=1)
            ax.axvline(0.01, color="gray", lw=1, ls=":")
            ax.scatter([0.01], [tpr1_ae], color="C0", zorder=5, label=f"TPR@FPR1%(AE)={tpr1_ae:.3f}")
            ax.scatter([0.01], [tpr1_c],  color="C1", marker="s", zorder=5, label=f"TPR@FPR1%(C)={tpr1_c:.3f}")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title(f"ROC — M1 v2 vs {tag}")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(Path(out_dir) / "roc_curve_v2.png"), dpi=150)
            plt.close(fig)

    # Score distribution — always generated
    fig, ax = plt.subplots(figsize=(7, 4))
    label_type_short = "Gravity Spy" if is_real_labels else "inferred P95"
    ax.hist(ae_scores[binary == 0], bins=60, alpha=0.6, density=True, label=f"Nominal n={n_nom}")
    ax.hist(ae_scores[binary == 1], bins=60, alpha=0.6, density=True, label=f"Anomalous n={n_glit}")
    ax.set_xlabel("AE Score (MSE)"); ax.set_ylabel("Density")
    ax.set_title(f"Score distribution — M1 v2  ({label_type_short} labels)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "score_distribution_v2.png"), dpi=150)
    plt.close(fig)

    # Boxplot + per-class table (only when real string labels available)
    if is_real_labels and labels_str is not None:
        all_cls = sorted(set(labels_str.tolist()))
        cls_stats = []
        for cls in all_cls:
            mask = labels_str == cls
            s = ae_scores[mask]
            cls_stats.append((cls, int(mask.sum()), float(s.mean()), float(s.std())))
        cls_stats.sort(key=lambda x: -x[2])

        print(f"\n  {'Clase':<34} {'N':>5} {'Media':>9} {'Std':>9}")
        for cls, cnt, mean, std in cls_stats:
            print(f"  {cls:<32} {cnt:>5} {mean:>9.5f} {std:>9.5f}")

        cls_ok = [c for c, n, _, _ in cls_stats if n >= 5]
        if cls_ok:
            order_ok = [c for c in [x[0] for x in cls_stats] if c in cls_ok]
            data_box = [ae_scores[labels_str == c] for c in order_ok]
            fig, ax = plt.subplots(figsize=(max(8, len(order_ok) * 0.9), 5))
            bp = ax.boxplot(data_box, patch_artist=True)
            colors = plt.cm.tab20(np.linspace(0, 1, len(order_ok)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.8)
            ax.set_xticks(range(1, len(order_ok) + 1))
            ax.set_xticklabels([c.replace("_", "\n") for c in order_ok], fontsize=7)
            ax.set_ylabel("AE Score (MSE)")
            ax.set_title("AE Score por clase Gravity Spy — M1 v2 (n≥5)")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(Path(out_dir) / "score_by_class_v2.png"), dpi=150)
            plt.close(fig)

    return out


# ---------------------------------------------------------------------------
# Phase 5 — Evaluation with run03 (or fallback)
# ---------------------------------------------------------------------------

def find_run03_npz():
    candidates = [
        RUN03_PATH,
        "/kaggle/input/run03-h1-o3a/H1/O3a/scale_1p0s/dataset_H1_O3a_scale_1p0s_run03.npz",
        "/kaggle/input/run03-h1-o3a/dataset_H1_O3a_scale_1p0s_run03.npz",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    for p in glob.glob("/kaggle/input/**/*.npz", recursive=True):
        if "run03" in Path(p).name.lower():
            return p
    return None


def phase5_eval(model, p1, p99, X_val_n, loge_val, device, out_dir):
    print("\n" + "=" * 60)
    print("Phase 5 — Evaluation with real labels")
    print("=" * 60)

    run03_path = find_run03_npz()

    if run03_path is not None:
        print(f"  run03 found: {run03_path}")
        npz       = np.load(run03_path, allow_pickle=True)
        X_r3      = npz["X"].astype(np.float32)
        loge_r3   = npz["log_energy"].astype(np.float32)
        labels_r3 = npz["labels"].astype(str)

        counts = Counter(labels_r3.tolist())
        print(f"  {len(labels_r3)} windows  |  {len(counts)} classes")
        for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:<32} {cnt:>4}")

        X_n      = apply_normalization(X_r3, p1, p99)
        ae_sc    = compute_anomaly_scores(model, make_tensor(X_n), device, BATCH_SIZE)
        combined = compute_combined_score(ae_sc, loge_r3, alpha=0.7)
        binary   = (labels_r3 != "No_Glitch").astype(int)

        metrics = eval_metrics_and_plots(
            ae_sc, combined, binary, labels_r3, loge_r3,
            out_dir, "Gravity Spy O3a", p1, p99, is_real_labels=True,
        )

        if "auroc_ae" in metrics:
            auroc = metrics["auroc_ae"]
            print("\n--- Conclusión automática ---")
            if auroc > 0.85:
                print("  M1 v2 detecta glitches reales efectivamente (AUROC > 0.85)")
            elif auroc > 0.7:
                print("  M1 v2 tiene capacidad parcial de detección (0.7 < AUROC < 0.85)")
            else:
                print("  M1 v2 no separa bien — considerar más datos o ajustar arquitectura (AUROC < 0.7)")

    else:
        print("  [INFO] run03 not found — using val set + inferred labels (P95 log_energy).")
        print("  To enable real-label evaluation:")
        print("    1. Run run03_bulk_generator.py locally")
        print("    2. Upload the NPZ as a Kaggle dataset named 'run03-h1-o3a'")

        ae_sc    = compute_anomaly_scores(model, make_tensor(X_val_n), device, BATCH_SIZE)
        combined = compute_combined_score(ae_sc, loge_val, alpha=0.7)
        thr      = float(np.percentile(loge_val, ENERGY_PCT))
        binary   = (loge_val > thr).astype(int)

        eval_metrics_and_plots(
            ae_sc, combined, binary, None, loge_val,
            out_dir, "val inferred", p1, p99, is_real_labels=False,
        )


# ---------------------------------------------------------------------------
# Phase 6 — Per-NPZ transfer evaluation
# ---------------------------------------------------------------------------

def phase6_transfer(model, p1, p99, device, out_dir):
    print("\n" + "=" * 60)
    print("Phase 6 — Transfer evaluation on individual run02 NPZs")
    print("=" * 60)

    rows = []
    for ifo, epoch, path in NPZ_CONFIGS:
        if not Path(path).exists():
            print(f"  [SKIP] {ifo}/{epoch}: not found")
            continue
        X, t0, loge = load_run02_npz(path)
        X_n  = apply_normalization(X, p1, p99)
        ae_sc = compute_anomaly_scores(model, make_tensor(X_n), device, BATCH_SIZE)

        thr        = float(np.percentile(loge, 95.0))
        binary_inf = (loge > thr).astype(int)
        auroc_inf  = safe_auroc(binary_inf, ae_sc)

        auroc_str = f"{auroc_inf:.4f}" if auroc_inf is not None else "N/A"
        print(f"  {ifo}/{epoch:<5}  n={len(X)}  "
              f"AE {ae_sc.mean():.5f}±{ae_sc.std():.5f}  "
              f"AUROC(inf)={auroc_str}")
        rows.append({
            "label":    f"{ifo}/{epoch}",
            "ae_mean":  float(ae_sc.mean()),
            "ae_std":   float(ae_sc.std()),
            "auroc_inf": auroc_inf if auroc_inf is not None else 0.0,
        })

    if len(rows) < 2:
        return

    labels_plot = [r["label"]    for r in rows]
    means       = [r["ae_mean"]  for r in rows]
    stds        = [r["ae_std"]   for r in rows]
    aurocs      = [r["auroc_inf"] for r in rows]
    x           = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_plot)
    axes[0].set_ylabel("AE Score (MSE mean ± std)")
    axes[0].set_title("Reconstruction error per IFO/epoch")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, aurocs, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_plot)
    axes[1].set_ylim(0, 1)
    axes[1].axhline(0.5, color="red", ls=":", lw=1, label="Random")
    axes[1].set_ylabel("AUROC (inferred P95 labels)")
    axes[1].set_title("Detection AUROC per IFO/epoch")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "transfer_comparison_v2.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: transfer_comparison_v2.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device       : {device}")
    print(f"LATENT_DIM   : {LATENT_DIM}")
    print(f"BATCH_SIZE   : {BATCH_SIZE}")
    print(f"EPOCHS (max) : {EPOCHS}  patience={PATIENCE}")
    print(f"Output dir   : {OUT_DIR}")

    # Phase 1
    (X_train, t0_train, loge_train,
     X_val,   t0_val,   loge_val,
     ifo_counts) = phase1_load()
    if X_train is None:
        return

    # Phase 2
    X_train_n, X_val_n, p1, p99 = phase2_normalize(X_train, X_val, OUT_DIR)
    del X_train, t0_train  # free memory

    # Phase 3+4
    model = phase34_train(X_train_n, X_val_n, device, OUT_DIR)
    del X_train_n  # free memory after training

    # Phase 5
    phase5_eval(model, p1, p99, X_val_n, loge_val, device, OUT_DIR)

    # Phase 6
    phase6_transfer(model, p1, p99, device, OUT_DIR)

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {OUT_DIR}/")
    print("Files:")
    for f in sorted(Path(OUT_DIR).glob("*_v2*")):
        size = f.stat().st_size
        print(f"  {f.name:<45} {size/1024:.1f} KB")


if __name__ == "__main__":
    main()
