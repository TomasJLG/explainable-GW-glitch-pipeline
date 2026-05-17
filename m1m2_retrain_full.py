"""
m1m2_retrain_full.py
Complete M1+M2 pipeline retraining with all available data.

Phase 1 -- Train M1 v4 on all run02v2 nominals (~6465 windows)
Phase 2 -- Extract M2 features from all labeled run03 windows (~1700)
Phase 3 -- Train M2 v2 ANFIS on the new feature set
Phase 4 -- Evaluate and compare with v1

Outputs:
    m1_v4_outputs/normalization_v4.json
    m1_v4_outputs/best_m1_ae_v4.pt
    m2_data/m2_features_v2.npz
    m2_outputs_v2/best_m2_anfis_v2.pt
    m2_outputs_v2/*.png

Usage:
    python m1m2_retrain_full.py
"""

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from m1_anomaly.m1_autoencoder import GlitchAE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_DIR      = PROJECT_ROOT / "gravityspy_o3"
RUN02_ROOT   = PROJECT_ROOT / "run02v2"
RUN03_ROOTS  = [PROJECT_ROOT / "run03", PROJECT_ROOT / "run03_minority"]
M1_V4_OUT    = PROJECT_ROOT / "m1_v4_outputs"
M2_DATA_DIR  = PROJECT_ROOT / "m2_data"
M2_V2_OUT    = PROJECT_ROOT / "m2_outputs_v2"

for d in (M1_V4_OUT, M2_DATA_DIR, M2_V2_OUT):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# M1 v4 config
# ---------------------------------------------------------------------------
M1_LATENT_DIM   = 32
M1_BATCH_SIZE   = 64
M1_LR           = 1e-3
M1_WEIGHT_DECAY = 1e-5
M1_MAX_EPOCHS   = 80
M1_PATIENCE     = 15
M1_LOGE_PCTILE  = 95    # percentile for nominal filter

# ---------------------------------------------------------------------------
# M2 v2 config
# ---------------------------------------------------------------------------
FEATURE_NAMES  = ["peak_frequency", "ae_score", "log_energy", "snr",
                  "duration", "bandwidth"]
CLASS_NAMES    = ["Loud", "Burst", "Scatter", "Other"]
N_FEATURES     = 6
N_CLASSES      = 4
CLUSTER_RA     = 0.15
M2_SPREAD      = 0.3
M2_LR          = 1e-2
M2_MAX_EPOCHS  = 200
M2_PATIENCE    = 20
M2_BATCH_SIZE  = 64
TRAIN_FRAC     = 0.80
SEED           = 42

# v1 baseline (all-500 metrics from pipeline_eval)
V1 = {"accuracy": 0.7460, "macro_f1": 0.6677,
      "Loud": 0.800, "Burst": 0.784, "Scatter": 0.728, "Other": 0.359}

# Fine->macro-4 mapping (Line folded into Other)
FINE_TO_MACRO4 = {
    "Extremely_Loud": 0, "Koi_Fish": 0, "Chirp": 0,
    "Low_Frequency_Burst": 1, "Blip": 1, "Blip_Low_Frequency": 1,
    "Scattered_Light": 2, "Fast_Scattering": 2,
}  # default -> 3 (Other, includes Line)

CSV_MAP = {
    ("H1", "O3a"): "H1_O3a.csv",
    ("H1", "O3b"): "H1_O3b.csv",
    ("L1", "O3a"): "L1_O3a.csv",
    ("L1", "O3b"): "L1_O3b.csv",
}

# ===========================================================================
# ANFIS (embedded, no import from m2_anfis package)
# ===========================================================================

class GBellMF(nn.Module):
    def __init__(self, n_rules: int, n_features: int):
        super().__init__()
        self.a_raw = nn.Parameter(torch.zeros(n_rules, n_features))
        self.b_raw = nn.Parameter(torch.zeros(n_rules, n_features))
        self.c     = nn.Parameter(torch.zeros(n_rules, n_features))

    def get_params(self):
        a = F.softplus(self.a_raw) + 1e-6
        b = F.softplus(self.b_raw) + 1e-6
        return a, b, self.c

    def forward(self, x):
        a, b, c = self.get_params()
        x_e = x.unsqueeze(1)            # (B, 1, F)
        mu  = 1.0 / (1.0 + ((x_e - c) / a).abs().pow(2 * b))  # (B, R, F)
        return mu.prod(dim=2)           # (B, R)

    def init_from_centers(self, centers, spread=0.3):
        if not isinstance(centers, torch.Tensor):
            centers = torch.tensor(centers, dtype=torch.float32)
        with torch.no_grad():
            self.c.data.copy_(centers)
            a_init = float(np.log(np.exp(max(spread - 1e-6, 1e-3)) - 1 + 1e-9))
            b_init = float(np.log(np.exp(2.0 - 1e-6) - 1 + 1e-9))
            self.a_raw.data.fill_(a_init)
            self.b_raw.data.fill_(b_init)


class ANFIS(nn.Module):
    def __init__(self, n_features: int, n_rules: int, n_classes: int):
        super().__init__()
        self.n_features = n_features
        self.n_rules    = n_rules
        self.n_classes  = n_classes
        self.mf         = GBellMF(n_rules, n_features)
        self.consequent = nn.Parameter(
            torch.zeros(n_rules, n_features + 1, n_classes))

    def get_w_bar(self, x):
        w = self.mf(x)
        return w / (w.sum(dim=1, keepdim=True) + 1e-8)

    def forward(self, x):
        w_bar = self.get_w_bar(x)                              # (B, R)
        x_aug = torch.cat([x, x.new_ones(x.shape[0], 1)], 1)  # (B, F+1)
        f     = torch.einsum("bf,rfc->brc", x_aug, self.consequent)  # (B, R, C)
        return (w_bar.unsqueeze(2) * f).sum(dim=1)             # (B, C)

    def init_from_centers(self, centers, spread=0.3):
        self.mf.init_from_centers(centers, spread)


# ===========================================================================
# Helpers
# ===========================================================================

def normalize_raw(X: np.ndarray, p1: float, p99: float) -> np.ndarray:
    X = np.clip(X, p1, p99)
    return ((X - p1) / (p99 - p1 + 1e-8)).astype(np.float32)


def subtractive_clustering(data: np.ndarray, ra: float = 0.5,
                            rb_factor: float = 1.5,
                            reject_ratio: float = 0.15) -> np.ndarray:
    X  = np.asarray(data, dtype=np.float64)
    rb = ra * rb_factor
    ra2, rb2 = (ra / 2) ** 2, (rb / 2) ** 2
    D2  = cdist(X, X, "sqeuclidean")
    P   = np.exp(-D2 / ra2).sum(axis=1)
    P0  = P.max()
    centers = []
    while True:
        i = int(np.argmax(P))
        if P[i] < reject_ratio * P0:
            break
        centers.append(X[i].copy())
        P -= P[i] * np.exp(-D2[i] / rb2)
        P  = np.maximum(P, 0.0)
    return np.array(centers)


def stratified_split(X: np.ndarray, y: np.ndarray,
                     train_frac: float, seed: int):
    rng   = np.random.default_rng(seed)
    tr_i, te_i = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * train_frac))
        tr_i.extend(idx[:cut])
        te_i.extend(idx[cut:])
    tr_i = np.array(tr_i); te_i = np.array(te_i)
    return (X[tr_i], y[tr_i]), (X[te_i], y[te_i])


def compute_metrics(y_true, y_pred, class_names):
    acc  = float(np.mean(y_true == y_pred))
    f1s  = {}
    for i, name in enumerate(class_names):
        tp = int(np.sum((y_pred == i) & (y_true == i)))
        fp = int(np.sum((y_pred == i) & (y_true != i)))
        fn = int(np.sum((y_pred != i) & (y_true == i)))
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1s[name] = 2 * p * r / max(p + r, 1e-9)
    macro_f1 = float(np.mean(list(f1s.values())))
    return acc, macro_f1, f1s


def anova_f_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    grand_mean = X.mean(axis=0)
    ss_between = sum(
        np.sum(y == c) * (X[y == c].mean(axis=0) - grand_mean) ** 2
        for c in classes
    )
    ss_within = sum(
        ((X[y == c] - X[y == c].mean(axis=0)) ** 2).sum(axis=0)
        for c in classes
    )
    k, N = len(classes), len(y)
    f = (ss_between / (k - 1)) / (ss_within / (N - k) + 1e-12)
    return f


def build_csv_lookups() -> dict:
    lookups = {}
    for (ifo, epoch), fname in CSV_MAP.items():
        path = CSV_DIR / fname
        if not path.exists():
            print(f"  [WARN] CSV not found: {path}")
            continue
        lk = {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    pt = float(row["peak_time"])
                    if row.get("peak_time_ns", "").strip():
                        pt += float(row["peak_time_ns"]) / 1e9
                    lk[pt] = {
                        "snr":            float(row["snr"]),
                        "peak_frequency": float(row["peak_frequency"]),
                        "duration":       float(row["duration"]),
                        "bandwidth":      float(row["bandwidth"]),
                    }
                except (KeyError, ValueError):
                    continue
        lookups[(ifo, epoch)] = lk
        print(f"  CSV lookup built: {fname}  ({len(lk)} triggers)")
    return lookups


# ===========================================================================
# Phase 1 -- M1 v4 Training
# ===========================================================================

def load_run02v2_all():
    X_list, t0_list, le_list = [], [], []
    npzs = sorted(RUN02_ROOT.rglob("*.npz"))
    if not npzs:
        print("[WARN] No run02v2 NPZ files found.")
        return None, None, None
    for p in npzs:
        d = np.load(str(p), allow_pickle=True)
        X_list.append(d["X"])
        t0_list.append(d["t0"])
        le_list.append(d["log_energy"])
        print(f"  Loaded {p.name}: {d['X'].shape[0]} windows")
    X  = np.concatenate(X_list,  axis=0)
    t0 = np.concatenate(t0_list, axis=0)
    le = np.concatenate(le_list, axis=0)
    return X, t0, le


def train_m1_v4(device: torch.device):
    print("\n" + "=" * 60)
    print("  Phase 1 -- M1 v4 Training")
    print("=" * 60)

    print("\n[1.1] Loading run02v2 data ...")
    X_raw, t0, log_e = load_run02v2_all()
    if X_raw is None:
        print("[ERROR] No data found. Aborting Phase 1.")
        return None, None, None

    print(f"  Total windows: {len(X_raw)}")

    # Nominal filter: remove high-energy outliers
    le_thresh = np.percentile(log_e, M1_LOGE_PCTILE)
    mask = log_e <= le_thresh
    X_raw = X_raw[mask]; t0 = t0[mask]; log_e = log_e[mask]
    print(f"  After nominal filter (log_e <= P{M1_LOGE_PCTILE}={le_thresh:.3f}): {len(X_raw)}")

    # Temporal split: P80 of t0
    t0_cut  = np.percentile(t0, 80)
    tr_mask = t0 <= t0_cut
    va_mask = ~tr_mask
    print(f"  Temporal split at t0={t0_cut:.0f}: train={tr_mask.sum()}  val={va_mask.sum()}")

    # Global P1/P99 from train
    p1  = float(np.percentile(X_raw[tr_mask], 1))
    p99 = float(np.percentile(X_raw[tr_mask], 99))
    print(f"  Normalization: p1={p1:.4f}  p99={p99:.4f}")

    norm_path = M1_V4_OUT / "normalization_v4.json"
    with open(norm_path, "w") as f:
        json.dump({"p1": p1, "p99": p99}, f)

    X_tr_n = normalize_raw(X_raw[tr_mask], p1, p99)
    X_va_n = normalize_raw(X_raw[va_mask], p1, p99)

    X_tr_t = torch.tensor(X_tr_n[:, None, :, :])
    X_va_t = torch.tensor(X_va_n[:, None, :, :])

    print(f"\n[1.2] Training GlitchAE (latent_dim={M1_LATENT_DIM}) ...")
    torch.manual_seed(SEED)
    model     = GlitchAE(latent_dim=M1_LATENT_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=M1_LR, weight_decay=M1_WEIGHT_DECAY)
    loader    = DataLoader(TensorDataset(X_tr_t), batch_size=M1_BATCH_SIZE, shuffle=True)

    best_val  = np.inf
    patience  = 0
    log_epochs = []
    ckpt_path  = M1_V4_OUT / "best_m1_ae_v4.pt"

    t0_train = time.time()
    for epoch in range(1, M1_MAX_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(loader)

        model.eval()
        with torch.no_grad():
            va_out  = model(X_va_t.to(device))
            va_loss = criterion(va_out, X_va_t.to(device)).item()

        log_epochs.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train={tr_loss:.5f}  val={va_loss:.5f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            patience = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_loss": va_loss,
                        "p1": p1, "p99": p99}, str(ckpt_path))
        else:
            patience += 1
            if patience >= M1_PATIENCE:
                print(f"  Early stop at epoch {epoch} (patience={M1_PATIENCE})")
                break

    elapsed = time.time() - t0_train
    print(f"  Training done in {elapsed:.1f}s  best_val_loss={best_val:.5f}")

    # Reload best
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    return model, p1, p99, log_epochs


# ===========================================================================
# Phase 2 -- Feature Extraction
# ===========================================================================

def load_run03_all():
    X_list, le_list, labels_list, pt_list, meta_list = [], [], [], [], []
    for root in RUN03_ROOTS:
        for p in sorted(root.rglob("*.npz")):
            d = np.load(str(p), allow_pickle=True)
            meta = json.loads(str(d["meta_json"]))
            X_list.append(d["X"])
            le_list.append(d["log_energy"].astype(np.float32))
            labels_list.extend(list(d["labels"]))
            pt_list.extend(list(d["peak_time"].astype(np.float64)))
            meta_list.extend([meta] * len(d["X"]))
            print(f"  Loaded {p.name}: {len(d['X'])} windows  "
                  f"({meta.get('ifo')}/{meta.get('epoch')})")
    if not X_list:
        return None, None, None, None, None
    X      = np.concatenate(X_list, axis=0)
    log_e  = np.concatenate(le_list, axis=0)
    labels = np.array(labels_list)
    pts    = np.array(pt_list, dtype=np.float64)
    return X, log_e, labels, pts, meta_list


def extract_features_v2(model: GlitchAE, p1: float, p99: float,
                         device: torch.device):
    print("\n" + "=" * 60)
    print("  Phase 2 -- Feature Extraction")
    print("=" * 60)

    print("\n[2.1] Loading labeled run03 data ...")
    X_raw, log_e, labels_fine, peak_times, meta_list = load_run03_all()
    if X_raw is None:
        print("[ERROR] No labeled data found.")
        return None
    print(f"  Total labeled windows: {len(X_raw)}")

    # Deduplication by peak_time
    seen = set()
    keep = []
    for i, pt in enumerate(peak_times):
        key = round(float(pt), 6)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    if len(keep) < len(peak_times):
        print(f"  Deduplication: {len(peak_times)} -> {len(keep)} unique windows")
        X_raw     = X_raw[keep]
        log_e     = log_e[keep]
        labels_fine = labels_fine[keep]
        peak_times  = peak_times[keep]
        meta_list   = [meta_list[i] for i in keep]
    N = len(X_raw)

    print(f"\n[2.2] Building CSV lookups ...")
    lookups = build_csv_lookups()

    print(f"\n[2.3] Matching CSV features ({N} windows) ...")
    feat_snr  = np.zeros(N, dtype=np.float32)
    feat_pf   = np.zeros(N, dtype=np.float32)
    feat_dur  = np.zeros(N, dtype=np.float32)
    feat_bw   = np.zeros(N, dtype=np.float32)
    n_matched = 0
    for i, (pt, meta) in enumerate(zip(peak_times, meta_list)):
        key    = (meta.get("ifo"), meta.get("epoch"))
        lookup = lookups.get(key, {})
        row    = lookup.get(float(pt))
        if row is None:
            # 1ms tolerance fallback
            pt_r  = round(float(pt), 3)
            for k_pt, v in lookup.items():
                if abs(k_pt - float(pt)) < 1e-3:
                    row = v; break
        if row:
            feat_snr[i]  = row["snr"]
            feat_pf[i]   = row["peak_frequency"]
            feat_dur[i]  = row["duration"]
            feat_bw[i]   = row["bandwidth"]
            n_matched   += 1
        else:
            feat_snr[i]  = float(np.nan)
            feat_pf[i]   = float(np.nan)
            feat_dur[i]  = float(np.nan)
            feat_bw[i]   = float(np.nan)
    print(f"  CSV match: {n_matched}/{N}")
    if n_matched < N:
        print(f"  [WARN] {N - n_matched} windows unmatched -- filled with NaN")

    print(f"\n[2.4] Computing M1 latents and AE scores ...")
    model.eval()
    X_norm = normalize_raw(X_raw, p1, p99)
    X_t    = torch.tensor(X_norm[:, None, :, :])
    latents, ae_scores = [], []
    bs = 128
    with torch.no_grad():
        for start in range(0, N, bs):
            xb   = X_t[start:start + bs].to(device)
            z    = model.encode(xb).cpu().numpy()
            xhat = model(xb).cpu()
            mse  = ((X_t[start:start + bs] - xhat) ** 2).mean(dim=[1, 2, 3]).numpy()
            latents.append(z)
            ae_scores.append(mse)
    latents   = np.concatenate(latents,   axis=0)   # (N, 32)
    ae_scores = np.concatenate(ae_scores, axis=0)   # (N,)

    # PCA on latents (info only)
    lat_centered = latents - latents.mean(axis=0)
    U, S, Vt = np.linalg.svd(lat_centered, full_matrices=False)
    var_ratio = (S ** 2) / (S ** 2).sum()
    k95 = int(np.searchsorted(np.cumsum(var_ratio), 0.95)) + 1
    print(f"  PCA: {k95} components explain 95% variance  "
          f"(top-2 explain {var_ratio[:2].sum()*100:.1f}%)")
    pca_components = Vt[:k95]
    pca_mean       = latents.mean(axis=0)

    print(f"\n[2.5] Assembling 6-feature matrix ...")
    # Features: [peak_frequency, ae_score, log_energy, snr, duration, bandwidth]
    feat_raw = np.column_stack([
        feat_pf, ae_scores, log_e.astype(np.float32),
        feat_snr, feat_dur, feat_bw,
    ]).astype(np.float32)  # (N, 6)

    # Drop rows with NaN
    valid = np.all(np.isfinite(feat_raw), axis=1)
    if valid.sum() < N:
        print(f"  Dropping {N - valid.sum()} rows with NaN features")
        feat_raw    = feat_raw[valid]
        labels_fine = labels_fine[valid]
        log_e       = log_e[valid]
        ae_scores   = ae_scores[valid]
        peak_times  = peak_times[valid]
        latents     = latents[valid]
        meta_list   = [meta_list[i] for i, v in enumerate(valid) if v]
        N = int(valid.sum())

    # Min-max normalization
    fmin = feat_raw.min(axis=0)
    fmax = feat_raw.max(axis=0)
    feat_norm = ((feat_raw - fmin) / (fmax - fmin + 1e-8)).astype(np.float32)

    # Macro labels (4 classes)
    labels_macro = np.array([FINE_TO_MACRO4.get(str(lb), 3)
                              for lb in labels_fine], dtype=np.int64)

    print(f"\n  Final dataset: {N} windows, 6 features, 4 classes")
    print(f"  Class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        cnt = int((labels_macro == i).sum())
        print(f"    {name:10s}: {cnt}")

    print(f"\n  Feature statistics (raw):")
    for j, fn in enumerate(FEATURE_NAMES):
        print(f"    {fn:20s}  "
              f"min={feat_raw[:, j].min():.3f}  "
              f"max={feat_raw[:, j].max():.3f}  "
              f"mean={feat_raw[:, j].mean():.3f}")

    out_path = M2_DATA_DIR / "m2_features_v2.npz"
    np.savez_compressed(
        str(out_path),
        features=feat_norm,
        features_raw=feat_raw,
        labels_macro=labels_macro,
        labels_original=labels_fine,
        feature_names=np.array(FEATURE_NAMES),
        macro_class_names=np.array(CLASS_NAMES),
        feature_min=fmin,
        feature_max=fmax,
        ae_scores_raw=ae_scores,
        peak_times=peak_times,
        pca_components=pca_components,
        pca_mean=pca_mean,
        pca_explained_variance=var_ratio[:k95],
    )
    print(f"\n  Saved: {out_path}")
    return feat_norm, labels_macro


# ===========================================================================
# Phase 3 -- M2 v2 Training
# ===========================================================================

def lse_update(model: ANFIS, X_t: torch.Tensor, y_t: torch.Tensor):
    model.eval()
    with torch.no_grad():
        w_bar = model.get_w_bar(X_t)                              # (N, R)
    N  = X_t.shape[0]
    F  = model.n_features
    R  = model.n_rules
    C  = model.n_classes
    x_aug = torch.cat([X_t, X_t.new_ones(N, 1)], dim=1)          # (N, F+1)
    Phi   = (w_bar.unsqueeze(2) * x_aug.unsqueeze(1)).reshape(N, R * (F + 1))
    Y     = np.zeros((N, C), dtype=np.float32)
    for i, yi in enumerate(y_t.cpu().numpy()):
        Y[i, int(yi)] = 1.0
    theta, _, _, _ = np.linalg.lstsq(Phi.cpu().numpy(), Y, rcond=None)
    model.consequent.data.copy_(
        torch.tensor(theta, dtype=torch.float32).reshape(R, F + 1, C))


def train_m2_v2(X_all: np.ndarray, y_all: np.ndarray,
                device: torch.device):
    print("\n" + "=" * 60)
    print("  Phase 3 -- M2 v2 ANFIS Training")
    print("=" * 60)

    (X_tr, y_tr), (X_te, y_te) = stratified_split(X_all, y_all, TRAIN_FRAC, SEED)
    print(f"\n  Split: train={len(X_tr)}  test={len(X_te)}  seed={SEED}")

    print(f"\n  Subtractive clustering (ra={CLUSTER_RA}) on {len(X_tr)} train samples ...")
    centers = subtractive_clustering(X_tr, ra=CLUSTER_RA)
    n_rules = len(centers)
    print(f"  Rules: {n_rules}")

    torch.manual_seed(SEED)
    model     = ANFIS(N_FEATURES, n_rules, N_CLASSES).to(device)
    centers_t = torch.tensor(centers, dtype=torch.float32, device=device)
    model.init_from_centers(centers_t, spread=M2_SPREAD)

    # Class weights
    weights = np.array([1.0 / max((y_tr == c).sum(), 1) for c in range(N_CLASSES)],
                       dtype=np.float32)
    weights = weights / weights.sum() * N_CLASSES
    crit    = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device))

    mf_params = [p for name, p in model.named_parameters() if "consequent" not in name]
    optimizer = optim.Adam(mf_params, lr=M2_LR)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.long, device=device)

    rng        = np.random.default_rng(SEED)
    best_val   = np.inf
    patience   = 0
    log_m2     = {"train_loss": [], "val_loss": [], "val_acc": []}
    ckpt_path  = M2_V2_OUT / "best_m2_anfis_v2.pt"

    t0_train = time.time()
    for epoch in range(1, M2_MAX_EPOCHS + 1):
        # LSE update on full train batch
        lse_update(model, X_tr_t, y_tr_t)

        # Adam update on MF params
        model.train()
        model.consequent.requires_grad_(False)
        idx = rng.permutation(len(X_tr))
        ep_loss = 0.0; nb = 0
        for s in range(0, len(X_tr), M2_BATCH_SIZE):
            bi   = idx[s:s + M2_BATCH_SIZE]
            xb   = X_tr_t[bi]; yb = y_tr_t[bi]
            optimizer.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item(); nb += 1
        model.consequent.requires_grad_(True)

        # Validation
        model.eval()
        with torch.no_grad():
            val_out  = model(X_te_t)
            val_loss = crit(val_out, y_te_t).item()
            val_acc  = (val_out.argmax(dim=1) == y_te_t).float().mean().item()

        log_m2["train_loss"].append(ep_loss / max(nb, 1))
        log_m2["val_loss"].append(val_loss)
        log_m2["val_acc"].append(val_acc)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}  train={ep_loss/max(nb,1):.4f}  "
                  f"val={val_loss:.4f}  val_acc={val_acc:.3f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            patience = 0
            torch.save({"state_dict": model.state_dict(),
                        "n_features": N_FEATURES, "n_rules": n_rules,
                        "n_classes": N_CLASSES, "centers": centers,
                        "feat_names": FEATURE_NAMES,
                        "class_names": CLASS_NAMES}, str(ckpt_path))
        else:
            patience += 1
            if patience >= M2_PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    elapsed = time.time() - t0_train
    print(f"  Done in {elapsed:.1f}s")

    # Reload best
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    return model, (X_tr, y_tr), (X_te, y_te), log_m2


# ===========================================================================
# Phase 4 -- Evaluation
# ===========================================================================

def evaluate_v2(model: ANFIS, X_all: np.ndarray, y_all: np.ndarray,
                X_te: np.ndarray, y_te: np.ndarray,
                device: torch.device, log_m1: list, log_m2: dict):
    print("\n" + "=" * 60)
    print("  Phase 4 -- Evaluation")
    print("=" * 60)

    model.eval()

    def infer(X):
        with torch.no_grad():
            return model(torch.tensor(X, dtype=torch.float32, device=device)
                         ).argmax(dim=1).cpu().numpy()

    y_pred_all = infer(X_all)
    y_pred_te  = infer(X_te)

    acc_all, mf1_all, f1s_all = compute_metrics(y_all, y_pred_all, CLASS_NAMES)
    acc_te,  mf1_te,  f1s_te  = compute_metrics(y_te,  y_pred_te,  CLASS_NAMES)

    print(f"\n  All {len(X_all)} samples:")
    print(f"    Accuracy  : {acc_all:.4f}")
    print(f"    Macro-F1  : {mf1_all:.4f}")
    for name in CLASS_NAMES:
        print(f"    F1({name:7s}) : {f1s_all[name]:.3f}")

    print(f"\n  Test set ({len(X_te)} samples):")
    print(f"    Accuracy  : {acc_te:.4f}")
    print(f"    Macro-F1  : {mf1_te:.4f}")
    for name in CLASS_NAMES:
        print(f"    F1({name:7s}) : {f1s_te[name]:.3f}")

    # Confusion matrix
    n = N_CLASSES
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_all, y_pred_all):
        cm[int(t), int(p)] += 1
    print(f"\n  Confusion matrix (all, rows=true, cols=pred):")
    header = "  " + " " * 10 + "  ".join(f"{c:>8}" for c in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cm[i, j]:>8d}" for j in range(n))
        print(f"  {name:>10}  {row}")

    # Rule extraction
    print(f"\n  Rule extraction ({model.n_rules} rules):")
    a, b, c = model.mf.get_params()
    c_np  = c.detach().cpu().numpy()
    a_np  = a.detach().cpu().numpy()
    b_np  = b.detach().cpu().numpy()
    cons  = model.consequent.detach().cpu().numpy()   # (R, F+1, C)
    X_t   = torch.tensor(X_all, dtype=torch.float32, device=device)
    with torch.no_grad():
        wb = model.get_w_bar(X_t).cpu().numpy()       # (N, R)
    avg_strength = wb.mean(axis=0)
    order = np.argsort(-avg_strength)
    col_q33 = np.percentile(c_np, 33, axis=0)
    col_q67 = np.percentile(c_np, 67, axis=0)
    def ling(v, q33, q67): return "low" if v <= q33 else ("medium" if v <= q67 else "high")
    for rank, r in enumerate(order):
        dom_cls = int(np.argmax(cons[r].sum(axis=0)))
        cond = "  AND  ".join(
            f"{FEATURE_NAMES[f]} IS {ling(c_np[r,f], col_q33[f], col_q67[f])}"
            for f in range(N_FEATURES))
        print(f"  R{r:02d} (str={avg_strength[r]:.3f}): IF {cond} => {CLASS_NAMES[dom_cls]}")
        if rank >= 4:
            break

    # Comparison vs v1
    print(f"\n  Comparison v1 (all-500) vs v2 (all-{len(X_all)}):")
    print(f"  {'Metric':20s}  {'v1':>8}  {'v2':>8}  {'delta':>8}")
    print(f"  {'-'*50}")
    def row(label, v1, v2):
        print(f"  {label:20s}  {v1:8.4f}  {v2:8.4f}  {v2-v1:+8.4f}")
    row("Accuracy",    V1["accuracy"],   acc_all)
    row("Macro-F1",    V1["macro_f1"],   mf1_all)
    for name in CLASS_NAMES:
        row(f"F1({name})",  V1.get(name, 0.0), f1s_all[name])

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _make_plots(model, X_all, y_all, cm, log_m1, log_m2, a_np, b_np, c_np)
    except Exception as e:
        print(f"\n  [WARN] Plotting failed: {e}")

    return acc_all, mf1_all, f1s_all


def _make_plots(model, X_all, y_all, cm, log_m1, log_m2, a_np, b_np, c_np):
    import matplotlib.pyplot as plt

    # 1. M1 v4 training curve
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs  = [d["epoch"] for d in log_m1]
    ax.plot(epochs, [d["train_loss"] for d in log_m1], label="Train")
    ax.plot(epochs, [d["val_loss"]   for d in log_m1], label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("M1 v4 -- Training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(M2_V2_OUT / "m1_v4_training_curve.png"), dpi=120)
    plt.close(fig)
    print("  Saved: m1_v4_training_curve.png")

    # 2. M2 v2 training curve
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep2 = range(1, len(log_m2["train_loss"]) + 1)
    axes[0].plot(ep2, log_m2["train_loss"], label="Train loss")
    axes[0].plot(ep2, log_m2["val_loss"],   label="Val loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("CE Loss")
    axes[0].set_title("M2 v2 -- Loss"); axes[0].legend()
    axes[1].plot(ep2, log_m2["val_acc"])
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("M2 v2 -- Val accuracy")
    fig.tight_layout()
    fig.savefig(str(M2_V2_OUT / "m2_v2_training_curve.png"), dpi=120)
    plt.close(fig)
    print("  Saved: m2_v2_training_curve.png")

    # 3. Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("M2 v2 -- Confusion matrix (all samples)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(str(M2_V2_OUT / "confusion_matrix_v2.png"), dpi=120)
    plt.close(fig)
    print("  Saved: confusion_matrix_v2.png")

    # 4. Feature importance (ANOVA F-score)
    scores = anova_f_scores(X_all, y_all)
    order  = np.argsort(scores)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([FEATURE_NAMES[i] for i in order][::-1],
            scores[order][::-1], color="steelblue")
    ax.set_xlabel("ANOVA F-score")
    ax.set_title("M2 v2 -- Feature importance")
    fig.tight_layout()
    fig.savefig(str(M2_V2_OUT / "feature_importance_v2.png"), dpi=120)
    plt.close(fig)
    print("  Saved: feature_importance_v2.png")

    # 5. Membership functions (bandwidth and peak_frequency -- most discriminating)
    n_rules = c_np.shape[0]
    colors  = plt.cm.tab10(np.linspace(0, 1, n_rules))
    feat_plot = [0, 1, 5]  # peak_frequency, ae_score, bandwidth
    fig, axes = plt.subplots(1, len(feat_plot), figsize=(13, 4))
    x_lin = np.linspace(0, 1, 200)
    for ax_i, fi in enumerate(feat_plot):
        ax = axes[ax_i]
        for r in range(n_rules):
            mu = 1.0 / (1.0 + (np.abs((x_lin - c_np[r, fi]) / a_np[r, fi])
                                ) ** (2 * b_np[r, fi]))
            ax.plot(x_lin, mu, color=colors[r], label=f"R{r:02d}", linewidth=1.2)
        ax.set_title(FEATURE_NAMES[fi])
        ax.set_xlabel("Normalized value")
        ax.set_ylabel("Membership degree")
        ax.set_ylim(-0.05, 1.05)
        if ax_i == len(feat_plot) - 1:
            ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("M2 v2 -- Membership functions (GBellMF)")
    fig.tight_layout()
    fig.savefig(str(M2_V2_OUT / "membership_functions_v2.png"), dpi=120)
    plt.close(fig)
    print("  Saved: membership_functions_v2.png")


# ===========================================================================
# Main
# ===========================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    t_total = time.time()

    # Phase 1
    result = train_m1_v4(device)
    if result[0] is None:
        sys.exit(1)
    model_m1, p1, p99, log_m1 = result

    # Phase 2
    feat_result = extract_features_v2(model_m1, p1, p99, device)
    if feat_result is None:
        sys.exit(1)
    X_all, y_all = feat_result

    # Phase 3
    model_m2, (X_tr, y_tr), (X_te, y_te), log_m2 = train_m2_v2(X_all, y_all, device)

    # Phase 4
    evaluate_v2(model_m2, X_all, y_all, X_te, y_te, device, log_m1, log_m2)

    print(f"\n{'='*60}")
    print(f"  Total time: {(time.time() - t_total)/60:.1f} min")
    print(f"  Outputs:")
    print(f"    {M1_V4_OUT}/best_m1_ae_v4.pt")
    print(f"    {M2_DATA_DIR}/m2_features_v2.npz")
    print(f"    {M2_V2_OUT}/best_m2_anfis_v2.pt")
    print(f"    {M2_V2_OUT}/*.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
