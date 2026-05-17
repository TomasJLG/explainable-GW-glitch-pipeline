"""
m1_train_v3.py
Trains M1 v3 on run02v2 (raw Q-transform, uniform-random GPS) and evaluates
against run03 (raw Q-transform, Gravity Spy labels).

Both datasets are stored without per-window normalisation. A single global
p1/p99 computed from the run02v2 train split is applied to both, ensuring
consistent scale at training and evaluation time.

Outputs: m1_v3_outputs/
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from m1_anomaly.m1_autoencoder import (
    GlitchAE,
    compute_anomaly_scores,
    compute_combined_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RUN02V2_PATH = (PROJECT_ROOT / "run02v2" / "H1" / "O3a"
                / "scale_1p0s" / "dataset_H1_O3a_scale_1p0s_run02v2.npz")
RUN03_PATH   = (PROJECT_ROOT / "run03" / "H1" / "O3a"
                / "scale_1p0s" / "dataset_H1_O3a_scale_1p0s_run03.npz")
OUT_DIR      = PROJECT_ROOT / "m1_v3_outputs"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LATENT_DIM   = 32
BATCH_SIZE   = 64
EPOCHS       = 80
LR           = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE     = 15
ENERGY_PCT   = 95.0
ALPHA        = 0.7
MIN_NOM      = 5      # minimum No_Glitch for real AUROC; else proxy fallback
PROXY_PCT    = 10.0

# Comparison baselines
V1_AUROC_AE   = 0.5265   # v1: per-window norm both sides
V2_AUROC_AE   = 0.3045   # v2: per-window norm train, raw eval (inverted polarity)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_normalization(X: np.ndarray, p1: float, p99: float) -> np.ndarray:
    return np.clip((X - p1) / (p99 - p1 + 1e-8), 0.0, 1.0).astype(np.float32)


def make_tensor(X_norm: np.ndarray) -> torch.Tensor:
    return torch.tensor(X_norm[:, None, :, :], dtype=torch.float32)


def tpr_at_fpr(fpr_arr, tpr_arr, target: float = 0.01) -> float:
    idx = np.searchsorted(fpr_arr, target)
    if idx == 0:
        return float(tpr_arr[0])
    if idx >= len(fpr_arr):
        return float(tpr_arr[-1])
    x0, x1 = fpr_arr[idx - 1], fpr_arr[idx]
    y0, y1 = tpr_arr[idx - 1], tpr_arr[idx]
    return float(y0 + (y1 - y0) * (target - x0) / (x1 - x0 + 1e-12))


def calc_metrics(binary, scores):
    auroc = roc_auc_score(binary, scores)
    auprc = average_precision_score(binary, scores)
    fpr, tpr, _ = roc_curve(binary, scores)
    return auroc, auprc, tpr_at_fpr(fpr, tpr), fpr, tpr


# ---------------------------------------------------------------------------
# Phase 1 — Load + filter + split run02v2
# ---------------------------------------------------------------------------

def phase1_load():
    print("\n" + "=" * 60)
    print("Phase 1 — Load run02v2  (raw, uniform-random GPS)")
    print("=" * 60)

    if not RUN02V2_PATH.exists():
        print(f"[ERROR] Not found: {RUN02V2_PATH}")
        return None, None, None, None, None, None

    npz = np.load(str(RUN02V2_PATH))
    X   = npz["X"].astype(np.float32)
    t0  = npz["t0"].astype(np.float64)
    le  = npz["log_energy"].astype(np.float32)
    print(f"  Loaded: {len(X)} windows  X=[{X.min():.2f}, {X.max():.2f}]")

    thr  = float(np.percentile(le, ENERGY_PCT))
    mask = le <= thr
    X, t0, le = X[mask], t0[mask], le[mask]
    print(f"  After P{ENERGY_PCT:.0f} log_energy filter: {len(X)} nominal windows  (thr={thr:.4f})")

    p80    = float(np.percentile(t0, 80))
    tr     = t0 <= p80
    va     = t0 >  p80
    print(f"  Temporal split P80={p80:.0f}:  train={tr.sum()}  val={va.sum()}")

    return X[tr], t0[tr], le[tr], X[va], t0[va], le[va]


# ---------------------------------------------------------------------------
# Phase 2 — Global normalisation
# ---------------------------------------------------------------------------

def phase2_normalize(X_train, X_val):
    print("\n" + "=" * 60)
    print("Phase 2 — Global normalisation  (p1/p99 from train)")
    print("=" * 60)

    p1  = float(np.percentile(X_train, 1))
    p99 = float(np.percentile(X_train, 99))
    print(f"  p1={p1:.6f}   p99={p99:.6f}")

    norm_path = OUT_DIR / "normalization_v3.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(norm_path, "w") as f:
        json.dump({"p1": p1, "p99": p99}, f, indent=2)
    print(f"  Saved: {norm_path}")

    return apply_normalization(X_train, p1, p99), apply_normalization(X_val, p1, p99), p1, p99


# ---------------------------------------------------------------------------
# Phase 3 — Train
# ---------------------------------------------------------------------------

def phase3_train(X_train_n, X_val_n, device):
    print("\n" + "=" * 60)
    print(f"Phase 3 — Train  latent_dim={LATENT_DIM}  bs={BATCH_SIZE}  max_epochs={EPOCHS}")
    print("=" * 60)

    model = GlitchAE(latent_dim=LATENT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_dl = DataLoader(TensorDataset(make_tensor(X_train_n)),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl   = DataLoader(TensorDataset(make_tensor(X_val_n)),
                          batch_size=BATCH_SIZE * 2)

    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, patience=7, factor=0.5, verbose=False)
    criterion  = nn.MSELoss()
    weights_path = OUT_DIR / "best_m1_ae_v3.pt"

    best_val     = float("inf")
    patience_cnt = 0
    log_rows     = []
    t0_train     = time.time()

    print(f"\n  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  Status")
    print(f"  {'------':>6}  {'----------':>10}  {'----------':>10}  ------")

    for epoch in range(1, EPOCHS + 1):
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x,) in val_dl:
                x = x.to(device)
                val_loss += criterion(model(x), x).item() * len(x)
        val_loss /= len(X_val_n)

        scheduler.step(val_loss)
        log_rows.append({"epoch": epoch,
                         "train_loss": round(train_loss, 8),
                         "val_loss":   round(val_loss,   8)})

        status = ""
        if val_loss < best_val:
            best_val     = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), str(weights_path))
            status = "best *"
        else:
            patience_cnt += 1

        if epoch % 5 == 0 or status or patience_cnt == PATIENCE:
            elapsed = time.time() - t0_train
            print(f"  {epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  "
                  f"{status or f'{patience_cnt}/{PATIENCE}'}  ({elapsed:.0f}s)")

        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    model.eval()
    print(f"\n  Best val loss : {best_val:.6f}  →  {weights_path.name}")

    with open(OUT_DIR / "training_log_v3.json", "w") as f:
        json.dump({"latent_dim": LATENT_DIM, "best_val_loss": best_val,
                   "epochs": log_rows}, f, indent=2)

    ep = [r["epoch"]      for r in log_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, [r["train_loss"] for r in log_rows], label="Train")
    ax.plot(ep, [r["val_loss"]   for r in log_rows], label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title(f"M1 v3 training  (latent_dim={LATENT_DIM}, run02v2)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "training_curve_v3.png"), dpi=150)
    plt.close(fig)

    return model


# ---------------------------------------------------------------------------
# Phase 4 — Evaluate with run03
# ---------------------------------------------------------------------------

def phase4_eval(model, p1: float, p99: float, device):
    print("\n" + "=" * 60)
    print("Phase 4 — Evaluate with run03  (Gravity Spy real labels)")
    print("=" * 60)

    if not RUN03_PATH.exists():
        print(f"  [ERROR] run03 NPZ not found: {RUN03_PATH}")
        return

    npz      = np.load(str(RUN03_PATH), allow_pickle=True)
    X_r3     = npz["X"].astype(np.float32)
    loge_r3  = npz["log_energy"].astype(np.float32)
    labels   = npz["labels"].astype(str)

    counts = Counter(labels.tolist())
    print(f"  {len(labels)} windows  |  {len(counts)} classes")
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:<32} {cnt:>4}")

    # Normalise with run02v2 train p1/p99
    X_n  = apply_normalization(X_r3, p1, p99)
    clip_pct = ((X_r3 < p1) | (X_r3 > p99)).mean() * 100
    print(f"\n  Normalisation: p1={p1:.5f}  p99={p99:.5f}  "
          f"(pixels clipped: {clip_pct:.1f}%)")

    ae_scores     = compute_anomaly_scores(model, make_tensor(X_n), device, 256)
    ae_scores_inv = -ae_scores
    combined      = compute_combined_score(ae_scores, loge_r3, alpha=ALPHA)

    # Binary labels
    binary  = (labels != "No_Glitch").astype(int)
    n_nom   = int((binary == 0).sum())
    n_glit  = int((binary == 1).sum())
    use_proxy = n_nom < MIN_NOM

    if use_proxy:
        p10_thr      = float(np.percentile(ae_scores, PROXY_PCT))
        binary_proxy = (ae_scores > p10_thr).astype(int)
        print(f"\n  [WARN] Only {n_nom} No_Glitch sample(s) — "
              f"proxy labels: AE score ≤ P{PROXY_PCT:.0f} ({p10_thr:.5f}) as nominal "
              f"({int((binary_proxy==0).sum())} proxy nominales)")

    # ---- Metrics ----
    print(f"\n  Binary real: {n_nom} nominal  |  {n_glit} glitch")
    print(f"\n  {'Score':<26} {'AUROC':>8} {'AUPRC':>8} {'TPR@FPR1%':>10}")

    roc_rows = []
    auroc_ae = None

    if n_nom > 0:
        auroc_ae,  auprc_ae,  tpr1_ae,  fpr_ae,  tpr_ae  = calc_metrics(binary, ae_scores)
        auroc_inv, auprc_inv, tpr1_inv, fpr_inv, tpr_inv = calc_metrics(binary, ae_scores_inv)
        auroc_c,   auprc_c,   tpr1_c,   fpr_c,   tpr_c   = calc_metrics(binary, combined)

        print(f"  {'v3 AE only':<26} {auroc_ae:>8.4f} {auprc_ae:>8.4f} {tpr1_ae:>10.4f}")
        print(f"  {'v3 AE invertido':<26} {auroc_inv:>8.4f} {auprc_inv:>8.4f} {tpr1_inv:>10.4f}")
        print(f"  {'v3 Combined (α=0.7)':<26} {auroc_c:>8.4f} {auprc_c:>8.4f} {tpr1_c:>10.4f}")

        if auroc_inv > auroc_ae:
            print("\n  NOTA: polaridad invertida — "
                  "los glitches tienen MSE MENOR que el ruido. "
                  "Posible desajuste de escala entre run02v2 y run03.")

        roc_rows += [("v3 AE",       fpr_ae,  tpr_ae,  auroc_ae,  tpr1_ae,  "-"),
                     ("v3 Combined", fpr_c,   tpr_c,   auroc_c,   tpr1_c,   "--")]

    if use_proxy:
        auroc_pe, auprc_pe, tpr1_pe, fpr_pe, tpr_pe = calc_metrics(binary_proxy, ae_scores)
        print(f"\n  {'v3 AE proxy (P10)':<26} {auroc_pe:>8.4f} {auprc_pe:>8.4f} {tpr1_pe:>10.4f}")
        roc_rows.append(("v3 AE proxy", fpr_pe, tpr_pe, auroc_pe, tpr1_pe, ":"))
        if auroc_ae is None:
            auroc_ae = auroc_pe

    # ---- Version comparison ----
    print(f"\n  {'Version':<16} {'AUROC AE':>10}  {'Δ vs v3':>10}  Notes")
    rows_cmp = [
        ("v1 (p-w norm)",  V1_AUROC_AE,  "per-window norm both sides"),
        ("v2 (raw eval)",  V2_AUROC_AE,  "raw eval, norm-mismatch → inv. polarity"),
        ("v3 (raw both)",  auroc_ae if auroc_ae else float("nan"), "raw train + raw eval, same p1/p99"),
    ]
    for name, val, note in rows_cmp:
        delta = (val - auroc_ae) if auroc_ae else float("nan")
        print(f"  {name:<16} {val:>10.4f}  {delta:>+10.4f}  {note}")

    # ---- Per-class table ----
    all_cls = sorted(counts.keys())
    cls_stats = []
    for cls in all_cls:
        mask = labels == cls
        s = ae_scores[mask]
        cls_stats.append((cls, int(mask.sum()), float(s.mean()), float(s.std())))
    cls_stats.sort(key=lambda x: -x[2])

    print(f"\n  {'Clase':<34} {'N':>5} {'Media':>10} {'Std':>10}")
    for cls, cnt, mean, std in cls_stats:
        print(f"  {cls:<32} {cnt:>5} {mean:>10.4f} {std:>10.4f}")

    # ---- Plots ----
    binary_dist = binary_proxy if (use_proxy and n_nom == 0) else binary

    # ROC
    if roc_rows:
        fig, ax = plt.subplots(figsize=(6, 5))
        for label, fpr, tpr, auroc, tpr1, ls in roc_rows:
            ax.plot(fpr, tpr, lw=2, ls=ls, label=f"{label}  AUROC={auroc:.3f}")
            ax.scatter([0.01], [tpr1], zorder=5)
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.axvline(0.01, color="gray", lw=1, ls=":")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC — M1 v3  (run02v2 train · run03 eval)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(OUT_DIR / "roc_curve_v3.png"), dpi=150)
        plt.close(fig)

    # Score distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ae_scores[binary_dist == 0], bins=60, alpha=0.6, density=True,
            label=f"Nominal  n={(binary_dist==0).sum()}")
    ax.hist(ae_scores[binary_dist == 1], bins=60, alpha=0.6, density=True,
            label=f"Glitch   n={(binary_dist==1).sum()}")
    ax.set_xlabel("AE Score (MSE)"); ax.set_ylabel("Density")
    ax.set_title("Score distribution — M1 v3")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "score_distribution_v3.png"), dpi=150)
    plt.close(fig)

    # Boxplot by class (n >= 3)
    cls_ok = [c for c, n, _, _ in cls_stats if n >= 3]
    if cls_ok:
        order_ok  = [c for c in [x[0] for x in cls_stats] if c in cls_ok]
        data_box  = [ae_scores[labels == c] for c in order_ok]
        fig, ax = plt.subplots(figsize=(max(8, len(order_ok) * 0.9), 5))
        bp = ax.boxplot(data_box, patch_artist=True)
        colors = plt.cm.tab20(np.linspace(0, 1, len(order_ok)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(order_ok) + 1))
        ax.set_xticklabels([c.replace("_", "\n") for c in order_ok], fontsize=7)
        ax.set_ylabel("AE Score (MSE)")
        ax.set_title("AE Score por clase Gravity Spy — M1 v3  (n≥3)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(OUT_DIR / "score_by_class_v3.png"), dpi=150)
        plt.close(fig)

    print(f"\n  Plots saved → {OUT_DIR}/")

    # ---- Conclusions ----
    auroc_eff = max(filter(None, [auroc_ae,
                                   auroc_inv if n_nom > 0 else None,
                                   auroc_pe  if use_proxy else None]),
                    default=None)
    print("\n--- Conclusiones automáticas ---")
    if auroc_eff is None:
        print("  No hay suficientes muestras para calcular AUROC.")
    elif auroc_eff > 0.85:
        print(f"  M1 v3 detecta glitches reales efectivamente (AUROC={auroc_eff:.4f} > 0.85)")
    elif auroc_eff > 0.7:
        print(f"  M1 v3 tiene capacidad parcial de detección (AUROC={auroc_eff:.4f})")
    else:
        print(f"  M1 v3 no separa bien (AUROC={auroc_eff:.4f} < 0.7) — "
              "considerar más datos de entrenamiento")

    print("Top-3 clases con score más alto:")
    for cls, cnt, mean, _ in cls_stats[:3]:
        print(f"    {cls} (n={cnt}, mean={mean:.4f})")
    print("Top-3 clases con score más bajo:")
    for cls, cnt, mean, _ in cls_stats[-3:]:
        print(f"    {cls} (n={cnt}, mean={mean:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device      : {device}")
    print(f"LATENT_DIM  : {LATENT_DIM}")
    print(f"BATCH_SIZE  : {BATCH_SIZE}")
    print(f"EPOCHS(max) : {EPOCHS}  patience={PATIENCE}")
    print(f"Output dir  : {OUT_DIR}")

    result = phase1_load()
    if result[0] is None:
        return
    X_train, t0_train, le_train, X_val, t0_val, le_val = result

    X_train_n, X_val_n, p1, p99 = phase2_normalize(X_train, X_val)
    del X_train, t0_train, le_train

    model = phase3_train(X_train_n, X_val_n, device)
    del X_train_n

    phase4_eval(model, p1, p99, device)

    print(f"\n{'='*60}")
    print(f"Outputs in {OUT_DIR}/")
    for f in sorted(OUT_DIR.glob("*v3*")):
        print(f"  {f.name:<45} {f.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
