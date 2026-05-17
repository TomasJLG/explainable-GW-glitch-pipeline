"""
m2_pipeline_eval.py
Full M1 -> M2 pipeline evaluation: raw run03 windows -> M1 encoder ->
feature engineering -> M2 ANFIS -> macro-class prediction.

Generates:
  m2_outputs/pipeline_eval/
    pipeline_confusion_matrix.png
    feature_importance.png
    decision_boundaries_2d.png
    rule_firing_heatmap.png

Usage:
    python m2_pipeline_eval.py
"""

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import ANFIS from m2_anfis.py via importlib (avoids conflict with m2_anfis/ package)
_spec = importlib.util.spec_from_file_location("m2s", PROJECT_ROOT / "m2_anfis.py")
_m2   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m2)
ANFIS = _m2.ANFIS

from m1_anomaly.m1_autoencoder import GlitchAE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

M1_WEIGHTS   = PROJECT_ROOT / "m1_v3_outputs" / "best_m1_ae_v3.pt"
M1_NORM      = PROJECT_ROOT / "m1_v3_outputs" / "normalization_v3.json"
M2_CKPT      = PROJECT_ROOT / "m2_outputs" / "best_m2_anfis.pt"
FEATURES_NPZ = PROJECT_ROOT / "m2_data" / "m2_features.npz"
RUN03_NPZ    = (PROJECT_ROOT / "run03" / "H1" / "O3a"
                / "scale_1p0s" / "dataset_H1_O3a_scale_1p0s_run03.npz")
CSV_PATH     = PROJECT_ROOT / "gravityspy_o3" / "H1_O3a.csv"
OUT_DIR      = PROJECT_ROOT / "m2_outputs" / "pipeline_eval"

# ---------------------------------------------------------------------------
# Constants (must match m2_anfis.py)
# ---------------------------------------------------------------------------

CLASS_NAMES      = ["Loud", "Burst", "Scatter", "Other"]
MERGE_MAP        = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}   # Line(3)->Other(3)
ANFIS_FEAT_ORDER = ["peak_frequency", "ae_score", "log_energy",
                    "snr", "duration", "bandwidth"]
M1_LATENT_DIM    = 32
M1_BATCH         = 256
CSV_MATCH_TOL    = 0.1

# Feature index mapping: ANFIS feature -> NPZ column index
# NPZ order: pca_0(0), pca_1(1), ae_score(2), log_energy(3), snr(4),
#            peak_frequency(5), bandwidth(6), duration(7)
_NPZ_NAMES     = ["pca_0", "pca_1", "ae_score", "log_energy", "snr",
                  "peak_frequency", "bandwidth", "duration"]
ANFIS_TO_NPZ   = [_NPZ_NAMES.index(f) for f in ANFIS_FEAT_ORDER]
# = [5, 2, 3, 4, 7, 6]

CLASS_COLORS = ["#9b59b6", "#2980b9", "#e67e22", "#27ae60"]  # Loud/Burst/Scatter/Other

# ---------------------------------------------------------------------------
# 1. Load models and data
# ---------------------------------------------------------------------------

def load_m1(device):
    model = GlitchAE(latent_dim=M1_LATENT_DIM).to(device)
    state = torch.load(str(M1_WEIGHTS), map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    with open(str(M1_NORM)) as f:
        norm = json.load(f)
    p1, p99 = float(norm["p1"]), float(norm["p99"])
    print(f"  M1 loaded  (p1={p1:.4f}, p99={p99:.4f})")
    return model, p1, p99


def load_m2(device):
    ckpt  = torch.load(str(M2_CKPT), map_location=device, weights_only=False)
    model = ANFIS(ckpt["n_features"], ckpt["n_rules"], ckpt["n_classes"],
                  ckpt["centers"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"  M2 loaded  (rules={ckpt['n_rules']}, "
          f"features={ckpt['n_features']}, classes={ckpt['n_classes']})")
    return model, ckpt


def load_run03():
    npz = np.load(str(RUN03_NPZ), allow_pickle=True)
    return (npz["X"].astype(np.float32),           # (500, 128, 128)
            npz["log_energy"].astype(np.float32),  # (500,)
            npz["peak_time"].astype(np.float64),   # (500,)
            npz["snr"].astype(np.float32),         # (500,)
            npz["labels"].astype(str))             # (500,)


def load_csv():
    rows = {}
    with open(str(CSV_PATH), newline="") as f:
        for row in csv.DictReader(f):
            try:
                pt = float(row["peak_time"]) + float(row.get("peak_time_ns") or 0) / 1e9
                rows[pt] = {
                    "snr":            float(row["snr"]),
                    "peak_frequency": float(row["peak_frequency"]),
                    "duration":       float(row["duration"]),
                    "bandwidth":      float(row["bandwidth"]),
                }
            except (KeyError, ValueError):
                continue
    keys = np.array(sorted(rows.keys()), dtype=np.float64)
    print(f"  CSV loaded  ({len(rows)} triggers)")
    return rows, keys


def load_feature_norms():
    """Return feat_min (8,) and feat_max (8,) from the NPZ."""
    npz = np.load(str(FEATURES_NPZ), allow_pickle=True)
    return npz["feature_min"].astype(np.float32), npz["feature_max"].astype(np.float32)


def load_npz_features():
    """Load pre-computed normalized features + 5-class labels from NPZ."""
    npz = np.load(str(FEATURES_NPZ), allow_pickle=True)
    return (npz["features"].astype(np.float32),       # (500, 8)
            npz["labels_macro"].astype(np.int64))     # (500,) 0-4


# ---------------------------------------------------------------------------
# 2. Pipeline inference
# ---------------------------------------------------------------------------

def run_m1_inference(m1, X_raw, p1, p99, device):
    X_norm = np.clip((X_raw - p1) / (p99 - p1 + 1e-8), 0., 1.)
    X_t    = torch.tensor(X_norm[:, None], dtype=torch.float32)
    ae_list = []
    with torch.no_grad():
        for i in range(0, len(X_t), M1_BATCH):
            xb  = X_t[i:i + M1_BATCH].to(device)
            xr  = m1(xb)
            mse = ((xb - xr) ** 2).mean(dim=[1, 2, 3])
            ae_list.append(mse.cpu().numpy())
    return np.concatenate(ae_list).astype(np.float32)


def match_csv(peak_times, csv_dict, csv_keys, log_energy, snr_npz):
    N = len(peak_times)
    snr = np.zeros(N, np.float32)
    pf  = np.zeros(N, np.float32)
    dur = np.zeros(N, np.float32)
    bw  = np.zeros(N, np.float32)
    n_match = 0
    for i, pt in enumerate(peak_times):
        idx = int(np.searchsorted(csv_keys, pt))
        best_j, best_d = None, np.inf
        for j in (idx - 1, idx):
            if 0 <= j < len(csv_keys):
                d = abs(csv_keys[j] - pt)
                if d < best_d:
                    best_d, best_j = d, j
        if best_j is not None and best_d <= CSV_MATCH_TOL:
            r = csv_dict[csv_keys[best_j]]
            snr[i] = r["snr"]
            pf[i]  = r["peak_frequency"]
            dur[i] = r["duration"]
            bw[i]  = r["bandwidth"]
            n_match += 1
        else:
            snr[i] = snr_npz[i]
            pf[i]  = float(np.expm1(log_energy[i]))
            dur[i] = 1.0
    print(f"  CSV match: {n_match}/{N} ({N - n_match} fallbacks)")
    return snr, pf, dur, bw


def build_features(ae_scores, log_energy, snr, pf, dur, bw,
                   feat_min, feat_max):
    """
    Build raw (N, 6) matrix in ANFIS feature order, then MinMax-normalize
    using the min/max values from m2_features.npz.
    """
    raw   = np.stack([pf, ae_scores, log_energy, snr, dur, bw], axis=1)
    f_min = feat_min[ANFIS_TO_NPZ]   # (6,)
    f_max = feat_max[ANFIS_TO_NPZ]   # (6,)
    denom = f_max - f_min
    denom[denom < 1e-12] = 1.0
    norm  = np.clip((raw - f_min) / denom, 0., 1.).astype(np.float32)
    return raw, norm


def anfis_infer(model, X_norm, device):
    X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(X_t)
        preds  = logits.argmax(dim=1).cpu().numpy()
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        w_raw  = model.firing_strengths(X_t)
        w_bar  = (w_raw / w_raw.sum(1, keepdim=True).clamp(1e-12)).cpu().numpy()
    return preds, probs, w_bar


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    n  = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    per_class, f1s = {}, []
    for c, name in enumerate(CLASS_NAMES):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        sup = tp + fn
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class[name] = {"prec": prec, "rec": rec, "f1": f1, "support": sup}
        if sup > 0:
            f1s.append(f1)
    acc      = float(np.mean(y_true == y_pred))
    macro_f1 = float(np.mean(f1s)) if f1s else 0.
    return acc, macro_f1, per_class, cm


def print_metrics(acc, macro_f1, per_class, cm, title=""):
    if title:
        print(f"\n  --- {title} ---")
    print(f"  Accuracy : {acc:.4f}   Macro-F1 : {macro_f1:.4f}")
    hdr = f"  {'Class':10s}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in CLASS_NAMES:
        m = per_class[name]
        print(f"  {name:10s}  {m['prec']:6.3f}  {m['rec']:6.3f}"
              f"  {m['f1']:6.3f}  {m['support']:8d}")
    w  = max(len(n) for n in CLASS_NAMES) + 1
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print("  " + " " * (w + 2) + "  ".join(f"{n:>{w}}" for n in CLASS_NAMES))
    print("  " + "-" * (w + 2 + (w + 2) * len(CLASS_NAMES)))
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cm[i,j]:>{w}d}" for j in range(len(CLASS_NAMES)))
        print(f"  {name:>{w}}  {row}")


# ---------------------------------------------------------------------------
# 4. Full rule extraction
# ---------------------------------------------------------------------------

def _linguistic(val):
    return "low" if val < 0.33 else ("medium" if val < 0.67 else "high")


def print_all_rules(model, w_bar, feat_names, class_names, n_samples):
    """Print every rule with MF params, avg firing strength, and consequent."""
    a_np, b_np = [v.detach().cpu().numpy() for v in model.get_ab()]
    c_np   = model.c.detach().cpu().numpy()            # (R, F)
    con_np = model.consequent.detach().cpu().numpy()   # (R, F+1, C)
    avg_w  = w_bar.mean(axis=0)                        # (R,)
    order  = np.argsort(avg_w)[::-1]

    print(f"\n  All {len(order)} rules (sorted by avg firing strength on "
          f"{n_samples} samples):")

    for rank, r in enumerate(order):
        print(f"\n  Rule R{r:03d} (avg_strength={avg_w[r]:.4f}):")
        for fi, fname in enumerate(feat_names):
            cv = float(c_np[r, fi])
            av = float(a_np[r, fi])
            bv = float(b_np[r, fi])
            lv = _linguistic(cv)
            prefix = "    IF " if fi == 0 else "    AND"
            print(f"  {prefix} {fname:<18s} IS {lv:<7s}"
                  f"  [c={cv:.3f}, a={av:.3f}, b={bv:.3f}]")

        # Consequent: linear output at rule center
        x_aug   = np.append(c_np[r], 1.0)             # (F+1,)
        f_r     = con_np[r].T @ x_aug                 # (C,)
        dom_cls = class_names[int(np.argmax(f_r))]
        w_str   = ", ".join(f"{class_names[c]}={f_r[c]:+.3f}"
                            for c in range(len(class_names)))
        print(f"  THEN class={dom_cls}")
        print(f"       weights: {w_str}")


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    n = len(CLASS_NAMES)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    thr = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_feature_importance(X_norm, y_4class, path):
    try:
        from sklearn.feature_selection import f_classif
    except ImportError:
        print("  [SKIP] feature_importance.png (sklearn not available)")
        return
    F_scores, _ = f_classif(X_norm, y_4class)
    order = np.argsort(F_scores)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(ANFIS_FEAT_ORDER)))
    ax.barh([ANFIS_FEAT_ORDER[i] for i in order],
            [F_scores[i] for i in order],
            color=[colors[i] for i in order])
    ax.set_xlabel("ANOVA F-score")
    ax.set_title("Feature importance (ANOVA F vs 4-class label)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_decision_boundaries(model, X_norm, y_4class, device, path):
    """
    2D decision boundary: peak_frequency (col 0) vs ae_score (col 1).
    Other 4 features fixed at training-set medians.
    """
    GRID = 120
    pf_grid = np.linspace(0, 1, GRID)
    ae_grid = np.linspace(0, 1, GRID)
    PF, AE  = np.meshgrid(pf_grid, ae_grid)

    medians = np.median(X_norm, axis=0)             # (6,)
    grid    = np.tile(medians, (GRID * GRID, 1))     # (N_grid, 6)
    grid[:, 0] = PF.ravel()                          # peak_frequency
    grid[:, 1] = AE.ravel()                          # ae_score

    preds_grid, _, _ = anfis_infer(model, grid.astype(np.float32), device)
    Z = preds_grid.reshape(GRID, GRID)

    cmap_bg = plt.cm.colors.ListedColormap(
        [c + "55" for c in CLASS_COLORS]  # 33% alpha hex
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(PF, AE, Z, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                cmap=cmap_bg)

    for c_idx, (cname, col) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        mask = (y_4class == c_idx)
        ax.scatter(X_norm[mask, 0], X_norm[mask, 1],
                   c=col, label=cname, edgecolors="k", linewidths=0.4,
                   s=25, zorder=3)

    ax.set_xlabel("peak_frequency (normalized)")
    ax.set_ylabel("ae_score (normalized)")
    ax.set_title("Decision boundary: peak_frequency vs ae_score\n"
                 "(other features fixed at median)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_rule_firing_heatmap(w_bar, y_4class, path):
    """
    Heatmap of normalized firing strengths (rules x samples).
    Samples sorted by true class for readability.
    """
    sort_idx = np.argsort(y_4class)
    W_sorted = w_bar[sort_idx].T          # (R, N)
    y_sorted = y_4class[sort_idx]

    fig, ax = plt.subplots(figsize=(12, max(3, W_sorted.shape[0] * 0.6)))
    im = ax.imshow(W_sorted, aspect="auto", cmap="hot_r",
                   vmin=0, vmax=W_sorted.max())
    plt.colorbar(im, ax=ax, label="Normalised firing strength")

    # Class boundary lines
    boundaries = np.where(np.diff(y_sorted))[0] + 0.5
    for b in boundaries:
        ax.axvline(b, color="cyan", lw=1.2)

    ax.set_yticks(range(W_sorted.shape[0]))
    ax.set_yticklabels([f"R{r:03d}" for r in range(W_sorted.shape[0])])
    ax.set_xlabel("Sample (sorted by true class)")
    ax.set_ylabel("Rule")
    ax.set_title("Rule firing strength heatmap")

    # Class labels along the top
    prev = 0
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        idxs = np.where(y_sorted == cls_idx)[0]
        if len(idxs) == 0:
            continue
        mid = (idxs[0] + idxs[-1]) / 2
        ax.text(mid, -0.8, cls_name, ha="center", va="top", fontsize=8,
                color=CLASS_COLORS[cls_idx], transform=ax.get_xaxis_transform())

    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# 6. Final report
# ---------------------------------------------------------------------------

def print_final_report(model, ckpt, acc_all, mf1_all, per_class_all,
                       acc_te, mf1_te, per_class_te, w_bar, feat_names,
                       class_names, n_epochs_info):

    a_np, b_np = [v.detach().cpu().numpy() for v in model.get_ab()]
    c_np   = model.c.detach().cpu().numpy()
    con_np = model.consequent.detach().cpu().numpy()
    avg_w  = w_bar.mean(axis=0)
    top3   = np.argsort(avg_w)[::-1][:3]

    print("\n" + "=" * 60)
    print("  === M2 ANFIS Final Report ===")
    print("=" * 60)
    print(f"  Architecture    : Takagi-Sugeno 1st order, GBELLMF")
    print(f"  Features ({len(feat_names)})   : {feat_names}")
    print(f"  Rules           : {model.n_rules}")
    n_prem = 3 * model.n_rules * model.n_features
    n_cons = model.n_rules * (model.n_features + 1) * model.n_classes
    print(f"  Parameters      : {n_prem} (premise) + {n_cons} "
          f"(consequent) = {n_prem + n_cons} total")
    print(f"  Training        : Hybrid LSE + Adam  {n_epochs_info}")
    print(f"  Classes         : {class_names}")
    print()
    print(f"  Pipeline (all 500 samples):")
    print(f"    Accuracy  : {acc_all:.4f}")
    print(f"    Macro-F1  : {mf1_all:.4f}")
    f1_str = ", ".join(f"{n}={per_class_all[n]['f1']:.3f}" for n in class_names)
    print(f"    Per-class : {f1_str}")
    print()
    print(f"  Test split (80/20, seed=42, 102 samples):")
    print(f"    Accuracy  : {acc_te:.4f}")
    print(f"    Macro-F1  : {mf1_te:.4f}")
    f1_str2 = ", ".join(f"{n}={per_class_te[n]['f1']:.3f}" for n in class_names)
    print(f"    Per-class : {f1_str2}")
    print()
    print(f"  Key rules (top 3 by avg firing strength):")
    for rank, r in enumerate(top3):
        x_aug  = np.append(c_np[r], 1.0)
        f_r    = con_np[r].T @ x_aug
        dom    = class_names[int(np.argmax(f_r))]
        # Highest-firing feature (largest c value relative to median)
        c_vals = c_np[r]
        fi     = int(np.argmax(np.abs(c_vals - 0.5)))
        lv     = _linguistic(float(c_vals[fi]))
        print(f"  {rank+1}. R{r:03d}: {feat_names[fi]} IS {lv} "
              f"-> {dom}  (avg_strength={avg_w[r]:.4f})")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("  M2 Pipeline Evaluation")
    print("=" * 60)
    print(f"  Device: {device}")

    # ---- Load resources ----
    print("\n[1] Loading models and data ...")
    m1_model, p1, p99 = load_m1(device)
    m2_model, ckpt    = load_m2(device)
    feat_names         = ckpt["feat_names"]
    class_names        = ckpt["class_names"]

    X_raw, log_energy, peak_times, snr_npz, labels_orig = load_run03()
    csv_dict, csv_keys = load_csv()
    feat_min, feat_max = load_feature_norms()
    X_npz_all, y5_npz  = load_npz_features()   # 8-feature, 5-class

    # Ground-truth 4-class labels
    y_true = np.array([MERGE_MAP[int(v)] for v in y5_npz], dtype=np.int64)

    # ---- End-to-end pipeline ----
    print("\n[2] Running M1 encoder ...")
    ae_scores = run_m1_inference(m1_model, X_raw, p1, p99, device)
    print(f"  ae_scores: [{ae_scores.min():.4f}, {ae_scores.max():.4f}]")

    print("\n[3] Building features from M1 + CSV ...")
    snr, pf, dur, bw = match_csv(peak_times, csv_dict, csv_keys,
                                  log_energy, snr_npz)
    feat_raw, feat_norm = build_features(ae_scores, log_energy, snr, pf,
                                          dur, bw, feat_min, feat_max)
    print(f"  feat_norm shape: {feat_norm.shape}  "
          f"range [{feat_norm.min():.3f}, {feat_norm.max():.3f}]")

    print("\n[4] Running M2 ANFIS ...")
    preds_pipeline, probs, w_bar = anfis_infer(m2_model, feat_norm, device)

    # ---- Consistency check vs NPZ features ----
    print("\n[5] Consistency check: pipeline vs NPZ features ...")
    X_npz_6 = X_npz_all[:, ANFIS_TO_NPZ]  # reorder to ANFIS feature order
    preds_npz, _, _ = anfis_infer(m2_model, X_npz_6, device)
    n_agree = int(np.sum(preds_pipeline == preds_npz))
    print(f"  Predictions agree: {n_agree}/500  "
          f"({'PASS' if n_agree == 500 else 'MISMATCH -- check feature norms'})")

    # ---- Metrics (all 500) ----
    print("\n[6] Metrics on all 500 samples ...")
    acc_all, mf1_all, pc_all, cm_all = compute_metrics(y_true, preds_pipeline)
    print_metrics(acc_all, mf1_all, pc_all, cm_all, "All 500 samples (pipeline)")

    # ---- Metrics on test split (same as m2_anfis.py) ----
    print("\n[7] Metrics on test split (80/20 stratified, seed=42) ...")
    (_, _), (X_te, y_te) = _m2.stratified_split(
        X_npz_6, y_true, _m2.TRAIN_FRAC, _m2.SEED
    )
    preds_te, _, w_bar_te = anfis_infer(m2_model, X_te, device)
    acc_te, mf1_te, pc_te, cm_te = compute_metrics(y_te, preds_te)
    print_metrics(acc_te, mf1_te, pc_te, cm_te, "Test split (102 samples)")
    print(f"\n  Stored result from m2_anfis.py: "
          f"acc={ckpt['acc']:.4f}  macro_f1={ckpt['macro_f1']:.4f}")

    # ---- Full rule extraction ----
    print("\n[8] Full rule extraction ...")
    print_all_rules(m2_model, w_bar, feat_names, class_names, 500)

    # ---- Plots ----
    print("\n[9] Generating plots ...")
    plot_confusion_matrix(cm_all,
                          "M2 ANFIS -- Confusion matrix (all 500 samples)",
                          OUT_DIR / "pipeline_confusion_matrix.png")
    plot_feature_importance(feat_norm, y_true,
                            OUT_DIR / "feature_importance.png")
    plot_decision_boundaries(m2_model, feat_norm, y_true, device,
                             OUT_DIR / "decision_boundaries_2d.png")
    plot_rule_firing_heatmap(w_bar, y_true,
                             OUT_DIR / "rule_firing_heatmap.png")

    # ---- Final report ----
    print_final_report(m2_model, ckpt, acc_all, mf1_all, pc_all,
                       acc_te, mf1_te, pc_te, w_bar, feat_names,
                       class_names, "max 200 epochs, early stop patience=20")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
