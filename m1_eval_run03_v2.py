"""
m1_eval_run03_v2.py
Evaluates M1 v1 weights against run03 v2 (raw, un-normalised Q-transform images).

Key difference from m1_eval_run03.py:
  run03 v2 stores raw Q-transform values (no per-window P99 normalisation).
  Global normalisation is applied here using p1/p99 from the run02 train set.

Inputs:
  m1_anomaly/best_m1_ae.pt         — M1 v1 weights  (latent_dim=128)
  m1_anomaly/normalization.json    — p1/p99 from run02 train set
  run03/H1/O3a/scale_1p0s/...npz  — run03 v2 (raw)
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
OUT_DIR      = PROJECT_ROOT / "eval_run03_v2"

LATENT_DIM  = 128
ALPHA       = 0.7
BATCH_SIZE  = 256
MIN_NOM     = 5      # minimum No_Glitch samples before proxy fallback kicks in
PROXY_PCT   = 10.0   # bottom P10 AE score used as proxy nominales

# v1 baseline values for comparison
V1_AUROC_AE   = 0.5265
V1_AUROC_COMB = 0.5269

# ---------------------------------------------------------------------------
# Helpers
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
        "log_energy": npz["log_energy"].astype(np.float32),
        "labels":     npz["labels"].astype(str),
        "snr":        npz["snr"].astype(np.float32),
    }


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


def metrics_row(binary, scores):
    """Returns (auroc, auprc, tpr1, fpr_arr, tpr_arr)."""
    auroc = roc_auc_score(binary, scores)
    auprc = average_precision_score(binary, scores)
    fpr, tpr, _ = roc_curve(binary, scores)
    return auroc, auprc, tpr_at_fpr(fpr, tpr), fpr, tpr


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc(rows, out_dir: Path):
    """rows: list of (label, fpr, tpr, auroc, tpr1, style)"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, fpr, tpr, auroc, tpr1, style in rows:
        ax.plot(fpr, tpr, lw=2, ls=style, label=f"{label}  AUROC={auroc:.3f}")
        ax.scatter([0.01], [tpr1], zorder=5)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.axvline(0.01, color="gray", lw=1, ls=":")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — M1 v1 weights · run03 v2 (raw Q-transform)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "roc_curve_v2data.png"), dpi=150)
    plt.close(fig)


def plot_score_distribution(ae_scores, binary, label_tag, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ae_scores[binary == 0], bins=60, alpha=0.6, density=True,
            label=f"Nominal  n={(binary==0).sum()}")
    ax.hist(ae_scores[binary == 1], bins=60, alpha=0.6, density=True,
            label=f"Glitch   n={(binary==1).sum()}")
    ax.set_xlabel("AE Score (MSE)")
    ax.set_ylabel("Density")
    ax.set_title(f"Score distribution — {label_tag}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "score_distribution_v2data.png"), dpi=150)
    plt.close(fig)


def plot_score_by_class(ae_scores, labels, out_dir: Path, min_n: int = 3):
    counts = Counter(labels.tolist())
    cls_list = sorted(
        [c for c, n in counts.items() if n >= min_n],
        key=lambda c: -float(np.mean(ae_scores[labels == c])),
    )
    if not cls_list:
        return
    data = [ae_scores[labels == c] for c in cls_list]
    fig, ax = plt.subplots(figsize=(max(8, len(cls_list) * 0.9), 5))
    bp = ax.boxplot(data, patch_artist=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(cls_list)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(cls_list) + 1))
    ax.set_xticklabels([c.replace("_", "\n") for c in cls_list], fontsize=7)
    ax.set_ylabel("AE Score (MSE)")
    ax.set_title("AE Score por clase Gravity Spy — M1 v1 · run03 v2 (n≥3)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "score_by_class_v2data.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo : {device}")

    for p, name in [
        (WEIGHTS_PATH, "pesos del modelo"),
        (NORM_PATH,    "normalization.json"),
        (NPZ_PATH,     "run03 v2 NPZ"),
    ]:
        if not p.exists():
            print(f"[ERROR] No encontrado ({name}): {p}")
            return

    # --- Load ---
    p1, p99 = load_normalization(NORM_PATH)
    print(f"Normalización run02 train: p1={p1:.5f}  p99={p99:.5f}")

    model = load_model(WEIGHTS_PATH, LATENT_DIM, device)
    print(f"Modelo: {WEIGHTS_PATH.name}  latent_dim={LATENT_DIM}")

    d = load_run03(NPZ_PATH)
    labels = d["labels"]
    n = len(labels)
    counts = Counter(labels.tolist())
    print(f"\nRun03 v2: {n} ventanas  (raw Q-transform, no per-window normalisation)")
    print(f"X range: [{d['X'].min():.1f}, {d['X'].max():.1f}]  mean={d['X'].mean():.1f}")
    print("Distribución de clases:")
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<32} {cnt:>4}")

    # --- Apply global normalisation ---
    X_norm   = apply_normalization(d["X"], p1, p99)
    print(f"\nX normalizado: [{X_norm.min():.3f}, {X_norm.max():.3f}]  "
          f"mean={X_norm.mean():.3f}  (clip fuera de [0,1]: "
          f"{((d['X'] < p1) | (d['X'] > p99)).mean()*100:.1f}% pixels)")

    # --- Scores ---
    print("\nCalculando scores AE...")
    ae_scores    = compute_anomaly_scores(model, make_tensor(X_norm), device, BATCH_SIZE)
    ae_scores_inv = -ae_scores
    combined     = compute_combined_score(ae_scores, d["log_energy"], alpha=ALPHA)

    # --- Binary labels + proxy fallback ---
    binary = (labels != "No_Glitch").astype(int)
    n_nom  = int((binary == 0).sum())
    n_glit = int((binary == 1).sum())
    use_proxy = n_nom < MIN_NOM

    if use_proxy:
        p10_thr = float(np.percentile(ae_scores, PROXY_PCT))
        binary_proxy = (ae_scores > p10_thr).astype(int)
        n_nom_proxy  = int((binary_proxy == 0).sum())
        print(f"\n[WARN] Solo {n_nom} No_Glitch — usando proxy: "
              f"P{PROXY_PCT:.0f} AE score ({p10_thr:.5f}) como umbral nominal. "
              f"Proxy nominales: {n_nom_proxy}.")

    print(f"\nBinario real: {n_nom} nominales (No_Glitch) | {n_glit} glitches")

    # --- Metrics (real labels) ---
    print("\n--- Métricas detección binaria — labels reales ---")
    print(f"{'Score':<24} {'AUROC':>8} {'AUPRC':>8} {'TPR@FPR1%':>10}")

    roc_rows = []
    if n_nom > 0:
        auroc_ae,  auprc_ae,  tpr1_ae,  fpr_ae,  tpr_ae  = metrics_row(binary, ae_scores)
        auroc_inv, auprc_inv, tpr1_inv, fpr_inv, tpr_inv = metrics_row(binary, ae_scores_inv)
        auroc_c,   auprc_c,   tpr1_c,   fpr_c,   tpr_c   = metrics_row(binary, combined)
        print(f"{'AE only':<24} {auroc_ae:>8.4f} {auprc_ae:>8.4f} {tpr1_ae:>10.4f}")
        print(f"{'AE invertido (-score)':<24} {auroc_inv:>8.4f} {auprc_inv:>8.4f} {tpr1_inv:>10.4f}")
        print(f"{'Combined (α=0.7)':<24} {auroc_c:>8.4f} {auprc_c:>8.4f} {tpr1_c:>10.4f}")

        if auroc_inv > auroc_ae:
            print(
                "\nNOTA: El modelo separa glitches con polaridad invertida — "
                "los glitches tienen MSE MENOR que el ruido. "
                "Esto indica un problema de normalización entre run02 y run03."
            )

        roc_rows.append(("AE only (real)",     fpr_ae,  tpr_ae,  auroc_ae,  tpr1_ae,  "-"))
        roc_rows.append(("Combined (real)",    fpr_c,   tpr_c,   auroc_c,   tpr1_c,   "--"))
    else:
        print("  (sin No_Glitch — métricas reales no disponibles)")
        auroc_ae = auroc_inv = auroc_c = None

    # --- Metrics (proxy labels, if applicable) ---
    if use_proxy:
        auroc_pe, auprc_pe, tpr1_pe, fpr_pe, tpr_pe = metrics_row(binary_proxy, ae_scores)
        auroc_pc, auprc_pc, tpr1_pc, fpr_pc, tpr_pc = metrics_row(binary_proxy, combined)
        print(f"\n--- Métricas proxy (P{PROXY_PCT:.0f} AE score como nominal) ---")
        print(f"{'Score':<24} {'AUROC':>8} {'AUPRC':>8} {'TPR@FPR1%':>10}")
        print(f"{'AE proxy':<24} {auroc_pe:>8.4f} {auprc_pe:>8.4f} {tpr1_pe:>10.4f}")
        print(f"{'Combined proxy':<24} {auroc_pc:>8.4f} {auprc_pc:>8.4f} {tpr1_pc:>10.4f}")
        roc_rows.append(("AE proxy",     fpr_pe, tpr_pe, auroc_pe, tpr1_pe, ":"))
        roc_rows.append(("Combined proxy", fpr_pc, tpr_pc, auroc_pc, tpr1_pc, "-."))

    # --- v1 comparison ---
    print(f"\n--- Comparación v1 vs v2-data (AE only) ---")
    print(f"{'Metric':<22} {'v1 (norm/window)':>18} {'v2 (raw→global)':>16} {'Δ':>8}")
    if auroc_ae is not None:
        delta_ae   = auroc_ae - V1_AUROC_AE
        delta_comb = auroc_c  - V1_AUROC_COMB
        print(f"{'AUROC AE':<22} {V1_AUROC_AE:>18.4f} {auroc_ae:>16.4f} {delta_ae:>+8.4f}")
        print(f"{'AUROC Combined':<22} {V1_AUROC_COMB:>18.4f} {auroc_c:>16.4f} {delta_comb:>+8.4f}")
    else:
        print("  (AUROC no disponible sin No_Glitch reales)")

    # --- Per-class table ---
    all_classes = sorted(counts.keys())
    class_stats = []
    for cls in all_classes:
        mask = labels == cls
        s = ae_scores[mask]
        class_stats.append((cls, int(mask.sum()), float(s.mean()), float(s.std())))
    class_stats.sort(key=lambda x: -x[2])

    print("\n--- AE Score por clase (media descendente) ---")
    print(f"{'Clase':<34} {'N':>5} {'Media':>10} {'Std':>10}")
    for cls, cnt, mean, std in class_stats:
        print(f"  {cls:<32} {cnt:>5} {mean:>10.3f} {std:>10.3f}")

    # --- Plots ---
    print("\nGenerando plots...")
    if roc_rows:
        plot_roc(roc_rows, OUT_DIR)
    binary_for_dist = binary_proxy if (use_proxy and n_nom == 0) else binary
    plot_score_distribution(ae_scores, binary_for_dist,
                            "M1 v1 · run03 v2 raw", OUT_DIR)
    plot_score_by_class(ae_scores, labels, OUT_DIR)
    print(f"Plots guardados en {OUT_DIR}/")

    # --- Conclusions ---
    auroc_eff = max(filter(lambda x: x is not None, [auroc_ae, auroc_inv,
                                                      auroc_pe if use_proxy else None]),
                    default=None)
    print("\n--- Conclusiones automáticas ---")
    if auroc_eff is None:
        print("No hay suficientes muestras para calcular AUROC.")
    elif auroc_eff > 0.85:
        print("M1 detecta glitches reales efectivamente (AUROC > 0.85)")
    elif auroc_eff > 0.7:
        print("M1 tiene capacidad parcial de detección (0.7 < AUROC < 0.85)")
    else:
        print("M1 no separa bien — considerar más datos de entrenamiento "
              "o ajustar arquitectura (AUROC < 0.7)")

    print("Top-3 clases con score más alto (M1 las detecta mejor):")
    for cls, cnt, mean, _ in class_stats[:3]:
        print(f"  {cls} (n={cnt}, mean={mean:.3f})")
    print("Top-3 clases con score más bajo (M1 las reconstruye mejor):")
    for cls, cnt, mean, _ in class_stats[-3:]:
        print(f"  {cls} (n={cnt}, mean={mean:.3f})")


if __name__ == "__main__":
    main()
