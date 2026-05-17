"""
m1_eval_run03.py
Evaluates the M1 autoencoder using real Gravity Spy labels from the run03 dataset.

Usage:
    python m1_eval_run03.py

Outputs in eval_run03/:
    roc_curve_real.png
    score_by_class.png
    score_distribution_real.png
    reconstructions_by_class.png
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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
# Paths & config
# ---------------------------------------------------------------------------

WEIGHTS_PATH = PROJECT_ROOT / "m1_anomaly" / "best_m1_ae.pt"
NORM_PATH    = PROJECT_ROOT / "m1_anomaly" / "normalization.json"
NPZ_PATH     = (PROJECT_ROOT / "run03" / "H1" / "O3a"
                / "scale_1p0s" / "dataset_H1_O3a_scale_1p0s_run03.npz")
OUT_DIR      = PROJECT_ROOT / "eval_run03"

LATENT_DIM   = 128
ALPHA        = 0.7
BATCH_SIZE   = 256

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_normalization(path: Path):
    with open(path) as f:
        d = json.load(f)
    return float(d["p1"]), float(d["p99"])


def load_model(path: Path, latent_dim: int, device: str) -> GlitchAE:
    model = GlitchAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(str(path), map_location=device))
    model.eval()
    return model


def load_run03(path: Path) -> dict:
    npz = np.load(str(path), allow_pickle=True)
    return {
        "X":          npz["X"].astype(np.float32),
        "t0":         npz["t0"].astype(np.float64),
        "log_energy": npz["log_energy"].astype(np.float32),
        "labels":     npz["labels"].astype(str),
        "peak_time":  npz["peak_time"].astype(np.float64),
        "snr":        npz["snr"].astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_normalization(X: np.ndarray, p1: float, p99: float) -> np.ndarray:
    return np.clip((X - p1) / (p99 - p1 + 1e-8), 0.0, 1.0).astype(np.float32)


def make_tensor(X_norm: np.ndarray) -> torch.Tensor:
    """(N, H, W) → (N, 1, H, W) float32 tensor."""
    return torch.tensor(X_norm[:, None, :, :], dtype=torch.float32)


def tpr_at_fpr(fpr_arr: np.ndarray, tpr_arr: np.ndarray, target: float = 0.01) -> float:
    idx = np.searchsorted(fpr_arr, target)
    if idx == 0:
        return float(tpr_arr[0])
    if idx >= len(fpr_arr):
        return float(tpr_arr[-1])
    x0, x1 = fpr_arr[idx - 1], fpr_arr[idx]
    y0, y1 = tpr_arr[idx - 1], tpr_arr[idx]
    return float(y0 + (y1 - y0) * (target - x0) / (x1 - x0 + 1e-12))


def reconstruct_all(model: GlitchAE, X_norm: np.ndarray, device: str) -> np.ndarray:
    """Returns (N, 128, 128) float32 reconstructions."""
    model.eval()
    parts = []
    dl = DataLoader(TensorDataset(make_tensor(X_norm)), batch_size=BATCH_SIZE)
    with torch.no_grad():
        for (x,) in dl:
            x_hat = model(x.to(device)).cpu().numpy()
            parts.append(x_hat[:, 0])
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc(fpr_ae, tpr_ae, auroc_ae, fpr_c, tpr_c, auroc_c,
             tpr1_ae, tpr1_c, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_ae, tpr_ae, lw=2,      label=f"AE only    AUROC={auroc_ae:.3f}")
    ax.plot(fpr_c,  tpr_c,  lw=2, ls="--", label=f"Combined   AUROC={auroc_c:.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.axvline(0.01, color="gray", lw=1, ls=":")
    ax.scatter([0.01], [tpr1_ae], color="C0", zorder=5,
               label=f"TPR@FPR1% (AE)={tpr1_ae:.3f}")
    ax.scatter([0.01], [tpr1_c], color="C1", marker="s", zorder=5,
               label=f"TPR@FPR1% (C)={tpr1_c:.3f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — M1 vs Gravity Spy labels reales")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "roc_curve_real.png"), dpi=150)
    plt.close(fig)


def plot_score_by_class(ae_scores: np.ndarray, labels: np.ndarray,
                        out_dir: Path, min_samples: int = 5):
    counts = Counter(labels.tolist())
    cls_list = sorted(
        [c for c, n in counts.items() if n >= min_samples],
        key=lambda c: -float(np.mean(ae_scores[labels == c])),
    )
    if not cls_list:
        print("  [WARN] No classes with >= 5 samples for boxplot.")
        return cls_list

    data = [ae_scores[labels == c] for c in cls_list]
    fig, ax = plt.subplots(figsize=(max(8, len(cls_list) * 0.9), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    colors = plt.cm.tab20(np.linspace(0, 1, len(cls_list)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(cls_list) + 1))
    ax.set_xticklabels([c.replace("_", "\n") for c in cls_list], fontsize=7)
    ax.set_ylabel("AE Score (MSE)")
    ax.set_title("AE Score por clase Gravity Spy — media descendente (n≥5)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "score_by_class.png"), dpi=150)
    plt.close(fig)
    return cls_list


def plot_score_distribution(ae_scores: np.ndarray, binary: np.ndarray, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 60
    ax.hist(ae_scores[binary == 0], bins=bins, alpha=0.6, density=True,
            label=f"Nominal (No_Glitch)  n={(binary==0).sum()}")
    ax.hist(ae_scores[binary == 1], bins=bins, alpha=0.6, density=True,
            label=f"Glitch real          n={(binary==1).sum()}")
    ax.set_xlabel("AE Score (MSE reconstrucción)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución scores AE — Gravity Spy labels reales")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "score_distribution_real.png"), dpi=150)
    plt.close(fig)


def plot_reconstructions_by_class(X_norm: np.ndarray, X_recon: np.ndarray,
                                  ae_scores: np.ndarray, labels: np.ndarray,
                                  out_dir: Path, top_n: int = 4):
    counts = Counter(labels.tolist())
    top_classes = [c for c, _ in counts.most_common(top_n)]

    fig, axes = plt.subplots(top_n, 3, figsize=(9, top_n * 2.5))
    if top_n == 1:
        axes = axes[None, :]

    for col, title in enumerate(["Original", "Reconstrucción", "Error |orig−recon|"]):
        axes[0, col].set_title(title, fontsize=10, pad=6)

    for row, cls in enumerate(top_classes):
        mask = labels == cls
        idx_in_class = np.where(mask)[0]
        best_local = int(np.argmax(ae_scores[mask]))
        idx = idx_in_class[best_local]

        orig  = X_norm[idx]
        recon = X_recon[idx]
        err   = np.abs(orig - recon)

        for col, (img, vmax) in enumerate([(orig, 1.0), (recon, 1.0), (err, float(err.max()) or 1.0)]):
            ax = axes[row, col]
            ax.imshow(img, origin="lower", aspect="auto", cmap="viridis",
                      vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        # Class label on the left
        axes[row, 0].set_ylabel(
            f"{cls.replace('_', ' ')}\n(n={int(mask.sum())})",
            fontsize=7, rotation=0, labelpad=68, ha="right", va="center",
        )

    fig.suptitle("Reconstrucciones por clase — ejemplo con mayor score AE", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_dir / "reconstructions_by_class.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    for p, name in [
        (WEIGHTS_PATH, "pesos del modelo"),
        (NORM_PATH,    "normalization.json"),
        (NPZ_PATH,     "dataset run03 NPZ"),
    ]:
        if not p.exists():
            print(f"[ERROR] No encontrado ({name}): {p}")
            sys.exit(1)

    # --- Load model + normalization ---
    p1, p99 = load_normalization(NORM_PATH)
    print(f"Normalización train: p1={p1:.5f}  p99={p99:.5f}")

    model = load_model(WEIGHTS_PATH, LATENT_DIM, device)
    print(f"Modelo cargado: {WEIGHTS_PATH.name}  latent_dim={LATENT_DIM}")

    # --- Load run03 ---
    d = load_run03(NPZ_PATH)
    labels = d["labels"]
    n = len(labels)
    counts = Counter(labels.tolist())
    print(f"\nDataset run03: {n} ventanas")
    print("Distribución de clases:")
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<32} {cnt:>4}")

    # --- Normalize & score ---
    X_norm   = apply_normalization(d["X"], p1, p99)
    X_tensor = make_tensor(X_norm)

    print("\nCalculando scores AE...")
    ae_scores = compute_anomaly_scores(model, X_tensor, device, BATCH_SIZE)
    combined  = compute_combined_score(ae_scores, d["log_energy"], alpha=ALPHA)

    # --- Binary labels: No_Glitch=0, everything else=1 ---
    binary = (labels != "No_Glitch").astype(int)
    n_nom  = int((binary == 0).sum())
    n_glit = int((binary == 1).sum())
    print(f"\nBinario: {n_nom} nominales (No_Glitch) | {n_glit} glitches")

    if n_nom == 0 or n_glit == 0:
        print("[ERROR] Necesito al menos una clase de cada tipo para calcular AUROC.")
        sys.exit(1)

    # --- Binary metrics (normal + inverted scores) ---
    ae_scores_inv = -ae_scores

    auroc_ae  = roc_auc_score(binary, ae_scores)
    auprc_ae  = average_precision_score(binary, ae_scores)
    fpr_ae, tpr_ae, _ = roc_curve(binary, ae_scores)
    tpr1_ae   = tpr_at_fpr(fpr_ae, tpr_ae)

    auroc_inv = roc_auc_score(binary, ae_scores_inv)
    auprc_inv = average_precision_score(binary, ae_scores_inv)
    fpr_inv, tpr_inv, _ = roc_curve(binary, ae_scores_inv)
    tpr1_inv  = tpr_at_fpr(fpr_inv, tpr_inv)

    auroc_c  = roc_auc_score(binary, combined)
    auprc_c  = average_precision_score(binary, combined)
    fpr_c, tpr_c, _ = roc_curve(binary, combined)
    tpr1_c   = tpr_at_fpr(fpr_c, tpr_c)

    print("\n--- Métricas detección binaria (No_Glitch vs resto) ---")
    print(f"{'Score':<22} {'AUROC':>8} {'AUPRC':>8} {'TPR@FPR1%':>10}")
    print(f"{'AE only':<22} {auroc_ae:>8.4f} {auprc_ae:>8.4f} {tpr1_ae:>10.4f}")
    print(f"{'AE invertido (-score)':<22} {auroc_inv:>8.4f} {auprc_inv:>8.4f} {tpr1_inv:>10.4f}")
    print(f"{'Combined (α=0.7)':<22} {auroc_c:>8.4f} {auprc_c:>8.4f} {tpr1_c:>10.4f}")

    if auroc_inv > auroc_ae:
        print(
            "\nNOTA: El modelo separa glitches pero con polaridad invertida — "
            "los glitches tienen error de reconstrucción MENOR que el ruido. "
            "Esto indica un problema de normalización entre run02 y run03."
        )

    # --- Per-class table ---
    all_classes = sorted(counts.keys())
    class_stats = []
    for cls in all_classes:
        mask = labels == cls
        s = ae_scores[mask]
        class_stats.append((cls, int(mask.sum()), float(s.mean()), float(s.std())))
    class_stats.sort(key=lambda x: -x[2])

    print("\n--- AE Score por clase (media descendente) ---")
    print(f"{'Clase':<34} {'N':>5} {'Media':>9} {'Std':>9}")
    for cls, cnt, mean, std in class_stats:
        print(f"  {cls:<32} {cnt:>5} {mean:>9.5f} {std:>9.5f}")

    # --- Plots ---
    print("\nGenerando plots...")
    plot_roc(fpr_ae, tpr_ae, auroc_ae, fpr_c, tpr_c, auroc_c, tpr1_ae, tpr1_c, OUT_DIR)
    plot_score_distribution(ae_scores, binary, OUT_DIR)
    plot_score_by_class(ae_scores, labels, OUT_DIR)

    X_recon = reconstruct_all(model, X_norm, device)
    plot_reconstructions_by_class(X_norm, X_recon, ae_scores, labels, OUT_DIR)

    print(f"Plots guardados en {OUT_DIR}/")

    # --- Conclusions ---
    auroc_effective = max(auroc_ae, auroc_inv)
    print("\n--- Conclusiones automáticas ---")
    if auroc_effective > 0.85:
        print("M1 detecta glitches reales efectivamente (AUROC > 0.85)")
    elif auroc_effective > 0.7:
        print("M1 tiene capacidad parcial de detección (0.7 < AUROC < 0.85)")
    else:
        print("M1 no separa bien — considerar más datos de entrenamiento "
              "o ajustar arquitectura (AUROC < 0.7)")

    top3_high = class_stats[:3]
    top3_low  = class_stats[-3:]
    print("Top-3 clases con score más alto (M1 las detecta mejor):")
    for cls, cnt, mean, _ in top3_high:
        print(f"  {cls} (n={cnt}, mean={mean:.5f})")
    print("Top-3 clases con score más bajo (M1 las reconstruye bien):")
    for cls, cnt, mean, _ in top3_low:
        print(f"  {cls} (n={cnt}, mean={mean:.5f})")


if __name__ == "__main__":
    main()
