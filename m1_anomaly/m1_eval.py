"""
m1_eval.py
==========
Script de evaluación del módulo M1 — genera métricas y plots requeridos.

Uso:
    python -m m1_anomaly.m1_eval --npz dataset_H1_O3a_scale_1p0_run02.npz \
                                 --weights best_m1_ae.pt \
                                 --norm normalization.json

Genera:
    eval_report.json       AUROC, AUPRC, TPR@FPR1% por configuración
    roc_curve.png          Curva ROC
    score_distribution.png Histograma scores nominales vs anómalos
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from m1_autoencoder import GlitchAE, compute_anomaly_scores, compute_combined_score
from m1_dataloader import apply_global_normalization, load_npz


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluación del autoencoder M1.")
    p.add_argument("--npz",        required=True,  help="Dataset de evaluación (.npz).")
    p.add_argument("--weights",    default="best_m1_ae.pt",    help="Pesos del modelo.")
    p.add_argument("--norm",       default="normalization.json", help="p1/p99 de normalización.")
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--alpha",      type=float, default=0.7,
                   help="Peso del AE en el score combinado (default 0.7).")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--report_out", default="eval_report.json")
    p.add_argument("--plots_dir",  default=".",    help="Directorio donde guardar los plots.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def load_model(weights_path: str, latent_dim: int, device: str) -> GlitchAE:
    model = GlitchAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def load_normalization(norm_path: str):
    with open(norm_path) as f:
        d = json.load(f)
    return d["p1"], d["p99"]


def prepare_tensor(X: np.ndarray) -> torch.Tensor:
    """(N, H, W) float32 → (N, 1, H, W) tensor."""
    return torch.tensor(X[:, None, :, :], dtype=torch.float32)


def tpr_at_fpr(fpr_arr: np.ndarray, tpr_arr: np.ndarray, target_fpr: float = 0.01) -> float:
    """Interpolación lineal del TPR en un FPR objetivo."""
    idx = np.searchsorted(fpr_arr, target_fpr)
    if idx == 0:
        return float(tpr_arr[0])
    if idx >= len(fpr_arr):
        return float(tpr_arr[-1])
    # Interpolación lineal entre los dos puntos más cercanos
    x0, x1 = fpr_arr[idx - 1], fpr_arr[idx]
    y0, y1 = tpr_arr[idx - 1], tpr_arr[idx]
    return float(y0 + (y1 - y0) * (target_fpr - x0) / (x1 - x0 + 1e-12))


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def run_evaluation(
    model: GlitchAE,
    d: dict,
    p1: float,
    p99: float,
    alpha: float,
    batch_size: int,
    device: str,
    tag: str,
    plots_dir: str,
) -> dict:
    """
    Evalúa el modelo sobre un dataset cargado y retorna el dict de métricas.

    Asume que d contiene:
        X          : (N, 128, 128) imágenes Q-transform
        log_energy : (N,) feature de energía
        labels     : (N,) binario  0=nominal  1=glitch
                     (si no existe, se infiere con P95 de log_energy)
    """
    # Normalización con los mismos p1/p99 del train
    X_norm = apply_global_normalization(d["X"], p1, p99)
    X_t    = prepare_tensor(X_norm).to(device)  # keep on CPU initially

    # Scores del AE
    ae_scores = compute_anomaly_scores(model, prepare_tensor(X_norm), device, batch_size)

    # Scores combinados
    log_energy    = d["log_energy"].astype(np.float32)
    combined      = compute_combined_score(ae_scores, log_energy, alpha=alpha)

    # Labels: usar las del dataset o inferir con P95 de log_energy
    if "labels" in d:
        labels = d["labels"].astype(int)
    else:
        thr    = np.percentile(log_energy, 95)
        labels = (log_energy > thr).astype(int)
        print(f"  [{tag}] Labels inferidas: {labels.sum()} anómalas de {len(labels)} ventanas.")

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------
    auroc = roc_auc_score(labels, combined)
    auprc = average_precision_score(labels, combined)
    fpr_arr, tpr_arr, _ = roc_curve(labels, combined)
    tpr1  = tpr_at_fpr(fpr_arr, tpr_arr, target_fpr=0.01)

    print(f"  [{tag}]  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  TPR@FPR1%={tpr1:.4f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    os.makedirs(plots_dir, exist_ok=True)

    # — Curva ROC —
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, lw=2, label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.axvline(0.01, color="red", lw=1, ls=":", label="FPR = 1%")
    ax.scatter([0.01], [tpr1], color="red", zorder=5, label=f"TPR@FPR1% = {tpr1:.3f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Curva ROC — {tag}")
    ax.legend()
    ax.grid(alpha=0.3)
    roc_path = os.path.join(plots_dir, f"roc_curve_{tag}.png")
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # — Distribución de scores —
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 60
    ax.hist(combined[labels == 0], bins=bins, alpha=0.6, label="Nominal",  density=True)
    ax.hist(combined[labels == 1], bins=bins, alpha=0.6, label="Anómalo",  density=True)
    ax.set_xlabel("Score combinado")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución de scores — {tag}")
    ax.legend()
    ax.grid(alpha=0.3)
    dist_path = os.path.join(plots_dir, f"score_distribution_{tag}.png")
    fig.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "tag":           tag,
        "n_samples":     int(len(labels)),
        "n_anomalous":   int(labels.sum()),
        "auroc":         round(auroc, 5),
        "auprc":         round(auprc, 5),
        "tpr_at_fpr1pct": round(tpr1, 5),
        "alpha":         alpha,
        "plots": {
            "roc":              roc_path,
            "score_distribution": dist_path,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    # Cargar modelo y normalización
    p1, p99 = load_normalization(args.norm)
    model   = load_model(args.weights, args.latent_dim, device)
    print(f"Modelo cargado desde '{args.weights}'  (latent_dim={args.latent_dim})")
    print(f"Normalización: p1={p1:.4f}  p99={p99:.4f}\n")

    # ------------------------------------------------------------------
    # Configuraciones de validación cruzada requeridas por el spec
    # Adaptar las rutas según los datasets disponibles.
    # ------------------------------------------------------------------
    configs = [
        # (tag,               npz_path)
        ("H1_O3a_baseline",   args.npz),   # Train H1 O3a → Test H1 O3a
        # Descomentar y ajustar rutas para las otras configuraciones:
        # ("L1_O3a_transfer",  "dataset_L1_O3a.npz"),
        # ("H1_O3b_transfer",  "dataset_H1_O3b.npz"),
    ]

    results = []
    for tag, npz_path in configs:
        if not os.path.isfile(npz_path):
            print(f"  [{tag}] Dataset no encontrado: '{npz_path}' — saltando.")
            continue
        print(f"Evaluando '{tag}'…")
        d      = load_npz(npz_path)
        result = run_evaluation(
            model, d, p1, p99,
            alpha=args.alpha,
            batch_size=args.batch_size,
            device=device,
            tag=tag,
            plots_dir=args.plots_dir,
        )
        results.append(result)

    # Guardar reporte
    report = {"configurations": results}
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReporte guardado en '{args.report_out}'")

    # ------------------------------------------------------------------
    # Señales de alarma automáticas (spec §5.2)
    # ------------------------------------------------------------------
    for r in results:
        if r["auroc"] < 0.7:
            print(f"  ⚠ [{r['tag']}] AUROC < 0.7 — el modelo no separa bien. "
                  "Aumentar latent_dim o añadir capas al encoder.")
        if r["auroc"] > 0.98:
            print(f"  ⚠ [{r['tag']}] AUROC > 0.98 — sospechar leakage. "
                  "Verificar split temporal y ausencia de glitches en train.")

    # Alerta de transferencia entre detectores
    baseline = next((r for r in results if "baseline" in r["tag"]), None)
    transfer = next((r for r in results if "L1" in r["tag"]), None)
    if baseline and transfer:
        drop = baseline["auroc"] - transfer["auroc"]
        if drop > 0.15:
            print(f"  ⚠ Caída de AUROC entre baseline y L1: {drop:.3f}. "
                  "El modelo puede estar sobreajustado a H1. "
                  "Reducir complejidad o añadir data augmentation (flip horizontal).")


if __name__ == "__main__":
    main()
