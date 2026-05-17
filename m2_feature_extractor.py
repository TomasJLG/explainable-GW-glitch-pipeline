"""
m2_feature_extractor.py
Extracts features for M2 ANFIS combining M1 encoder latents with
Gravity Spy trigger metadata.

Outputs: m2_data/m2_features.npz + m2_data/m2_features_meta.json

Feature vector per window (K + 6 dimensions):
    [pca_0 ... pca_{K-1}, ae_score, log_energy, snr, peak_frequency, bandwidth, duration]

Usage:
    python m2_feature_extractor.py
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

M1_WEIGHTS = PROJECT_ROOT / "m1_v3_outputs" / "best_m1_ae_v3.pt"
M1_NORM    = PROJECT_ROOT / "m1_v3_outputs" / "normalization_v3.json"
RUN03_NPZ  = (PROJECT_ROOT / "run03" / "H1" / "O3a"
              / "scale_1p0s" / "dataset_H1_O3a_scale_1p0s_run03.npz")
CSV_PATH   = PROJECT_ROOT / "gravityspy_o3" / "H1_O3a.csv"
OUT_DIR    = PROJECT_ROOT / "m2_data"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

LATENT_DIM       = 32
PCA_VARIANCE_THR = 0.95   # keep enough components to explain this fraction
CSV_MATCH_TOL    = 0.1    # seconds tolerance for peak_time matching
BATCH_SIZE       = 256

# ---------------------------------------------------------------------------
# Macro-class mapping (23 fine labels -> 5 classes)
# ---------------------------------------------------------------------------

MACRO_NAMES: list = ["Loud", "Burst", "Scatter", "Line", "Other"]
MACRO_TO_IDX: dict = {n: i for i, n in enumerate(MACRO_NAMES)}

FINE_TO_MACRO: dict = {
    "Extremely_Loud":      "Loud",
    "Koi_Fish":            "Loud",
    "Low_Frequency_Burst": "Burst",
    "Blip":                "Burst",
    "Blip_Low_Frequency":  "Burst",
    "Scattered_Light":     "Scatter",
    "Fast_Scattering":     "Scatter",
    "Low_Frequency_Lines": "Line",
    "Power_Line":          "Line",
    "Violin_Mode":         "Line",
    "Wandering_Line":      "Line",
    # Everything else -> Other
    "Whistle":             "Other",
    "Tomte":               "Other",
    "Scratchy":            "Other",
    "Repeating_Blips":     "Other",
    "Chirp":               "Other",
    "Air_Compressor":      "Other",
    "Paired_Doves":        "Other",
    "No_Glitch":           "Other",
    "Helix":               "Other",
    "Light_Modulation":    "Other",
    "1080Lines":           "Other",
    "1400Ripples":         "Other",
    "None_of_the_Above":   "Other",
}


def to_macro_idx(fine_label: str) -> int:
    macro = FINE_TO_MACRO.get(fine_label, "Other")
    return MACRO_TO_IDX[macro]


# ---------------------------------------------------------------------------
# Step 1: Load M1 model
# ---------------------------------------------------------------------------

def load_m1(device: torch.device):
    from m1_anomaly.m1_autoencoder import GlitchAE
    model = GlitchAE(latent_dim=LATENT_DIM).to(device)
    state = torch.load(str(M1_WEIGHTS), map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"  M1 loaded: {M1_WEIGHTS.name}  (latent_dim={LATENT_DIM})")
    return model


def load_norm():
    with open(str(M1_NORM)) as f:
        d = json.load(f)
    return float(d["p1"]), float(d["p99"])


# ---------------------------------------------------------------------------
# Step 2: Load run03, normalise, run M1 encoder
# ---------------------------------------------------------------------------

def load_run03():
    npz        = np.load(str(RUN03_NPZ), allow_pickle=True)
    X          = npz["X"].astype(np.float32)
    log_energy = npz["log_energy"].astype(np.float32)
    labels_raw = npz["labels"].astype(str)
    peak_time  = npz["peak_time"].astype(np.float64)
    snr_npz    = npz["snr"].astype(np.float32)
    print(f"  run03: {len(X)} windows  X=[{X.min():.3f}, {X.max():.3f}]")
    return X, log_energy, labels_raw, peak_time, snr_npz


def run_m1_inference(model, X_raw: np.ndarray, p1: float, p99: float,
                     device: torch.device):
    """Normalise X, extract latents (N, 32) and ae_scores (N,)."""
    X_norm = np.clip((X_raw - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    X_t    = torch.tensor(X_norm[:, None, :, :], dtype=torch.float32)

    latents_list, scores_list = [], []
    N = len(X_t)

    with torch.no_grad():
        for start in range(0, N, BATCH_SIZE):
            xb  = X_t[start:start + BATCH_SIZE].to(device)
            z   = model.encode(xb)
            xr  = model(xb)
            mse = ((xb - xr) ** 2).mean(dim=[1, 2, 3])
            latents_list.append(z.cpu().numpy())
            scores_list.append(mse.cpu().numpy())

    latents   = np.concatenate(latents_list, axis=0).astype(np.float32)
    ae_scores = np.concatenate(scores_list,  axis=0).astype(np.float32)
    print(f"  Latents: {latents.shape}  ae_score=[{ae_scores.min():.4f}, {ae_scores.max():.4f}]")
    return latents, ae_scores


# ---------------------------------------------------------------------------
# Step 3: Load CSV, match each window to its trigger row
# ---------------------------------------------------------------------------

def load_csv_triggers():
    """
    Returns dict {float_peak_time: {snr, peak_frequency, duration, bandwidth}}
    and a sorted numpy array of keys.
    """
    rows = {}
    with open(str(CSV_PATH), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pt  = float(row["peak_time"])
                ns  = float(row.get("peak_time_ns") or 0)
                key = pt + ns / 1e9
                rows[key] = {
                    "snr":            float(row["snr"]),
                    "peak_frequency": float(row["peak_frequency"]),
                    "duration":       float(row["duration"]),
                    "bandwidth":      float(row["bandwidth"]),
                    "ml_confidence":  float(row.get("ml_confidence") or 0.0),
                }
            except (KeyError, ValueError):
                continue
    sorted_keys = np.array(sorted(rows.keys()), dtype=np.float64)
    print(f"  CSV: {len(rows)} triggers loaded")
    return rows, sorted_keys


def match_triggers(peak_times: np.ndarray, csv_dict: dict,
                   csv_keys: np.ndarray, log_energy: np.ndarray,
                   snr_npz: np.ndarray):
    """
    For each NPZ window, find the closest CSV row within CSV_MATCH_TOL.
    Falls back to NPZ values if no match.

    Returns: snr, peak_frequency, duration, bandwidth  (each shape N,).
    """
    N = len(peak_times)
    snr_out = np.zeros(N, dtype=np.float32)
    pfreq   = np.zeros(N, dtype=np.float32)
    dur_out = np.zeros(N, dtype=np.float32)
    bw_out  = np.zeros(N, dtype=np.float32)
    n_matched = 0

    for i, pt in enumerate(peak_times):
        idx = int(np.searchsorted(csv_keys, pt))
        candidates = [j for j in (idx - 1, idx) if 0 <= j < len(csv_keys)]
        best_j, best_diff = None, np.inf
        for j in candidates:
            diff = abs(csv_keys[j] - pt)
            if diff < best_diff:
                best_diff, best_j = diff, j

        if best_j is not None and best_diff <= CSV_MATCH_TOL:
            r = csv_dict[csv_keys[best_j]]
            snr_out[i] = r["snr"]
            pfreq[i]   = r["peak_frequency"]
            dur_out[i] = r["duration"]
            bw_out[i]  = r["bandwidth"]
            n_matched += 1
        else:
            snr_out[i] = snr_npz[i]
            pfreq[i]   = float(np.expm1(log_energy[i]))  # log1p proxy inversion
            dur_out[i] = 1.0
            bw_out[i]  = 0.0

    print(f"  CSV match: {n_matched}/{N} matched  ({N - n_matched} fallbacks)")
    return snr_out, pfreq, dur_out, bw_out


# ---------------------------------------------------------------------------
# Step 4: PCA on latents (numpy SVD)
# ---------------------------------------------------------------------------

def fit_pca(latents: np.ndarray, variance_thr: float = PCA_VARIANCE_THR):
    """
    Fit PCA via economy SVD.
    Returns (mean (32,), Vt (32,32), evr (32,), K int).
    """
    mean = latents.mean(axis=0)
    Z    = latents - mean
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var  = (S ** 2) / (len(latents) - 1)
    evr  = var / var.sum()
    cumvar = np.cumsum(evr)
    K = int(np.searchsorted(cumvar, variance_thr)) + 1
    K = min(K, len(evr))
    print(f"  PCA: keeping {K} components  "
          f"(cumulative variance = {cumvar[K-1]*100:.1f}%)")
    return mean, Vt, evr, K


def apply_pca(latents: np.ndarray, mean: np.ndarray, Vt: np.ndarray,
              K: int) -> np.ndarray:
    Z = latents - mean
    return (Z @ Vt[:K].T).astype(np.float32)  # (N, K)


# ---------------------------------------------------------------------------
# Step 5: Assemble and normalise feature matrix
# ---------------------------------------------------------------------------

def build_features(pca_proj, ae_scores, log_energy, snr, pfreq, dur,
                   bw) -> np.ndarray:
    cols = [pca_proj, ae_scores[:, None], log_energy[:, None],
            snr[:, None], pfreq[:, None], bw[:, None], dur[:, None]]
    return np.concatenate(cols, axis=1).astype(np.float32)


def minmax_normalize(X_raw: np.ndarray):
    """Scale each feature column to [0, 1]. Returns (X_norm, feat_min, feat_max)."""
    feat_min = X_raw.min(axis=0)
    feat_max = X_raw.max(axis=0)
    denom    = feat_max - feat_min
    denom[denom < 1e-12] = 1.0
    X_norm = (X_raw - feat_min) / denom
    return (X_norm.astype(np.float32),
            feat_min.astype(np.float32),
            feat_max.astype(np.float32))


# ---------------------------------------------------------------------------
# Step 6: Summary
# ---------------------------------------------------------------------------

def print_summary(feat_names, features_raw, features_norm, labels_macro,
                  pca_evr, K):
    from collections import Counter

    N, F = features_raw.shape
    print(f"\n{'='*60}")
    print(f"  Feature matrix: {N} windows x {F} features")
    print(f"  PCA components kept: {K}  (>={PCA_VARIANCE_THR*100:.0f}% variance)")
    print("  Explained variance per PCA component:")
    for k in range(K):
        print(f"    pca_{k}: {pca_evr[k]*100:.1f}%")

    print("\n  Macro-class distribution:")
    dist = Counter(labels_macro.tolist())
    for idx, name in enumerate(MACRO_NAMES):
        cnt = dist.get(idx, 0)
        print(f"    {name:10s} ({idx}): {cnt:4d}  ({cnt/N*100:.1f}%)")

    print("\n  Feature statistics (raw):")
    header = f"  {'Feature':25s}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for j, name in enumerate(feat_names):
        col = features_raw[:, j]
        print(f"  {name:25s}  {col.min():10.4f}  {col.max():10.4f}"
              f"  {col.mean():10.4f}  {col.std():10.4f}")

    try:
        from sklearn.feature_selection import f_classif
        F_scores, p_values = f_classif(features_norm, labels_macro)
        print("\n  ANOVA F-scores (feature vs macro-class):")
        ranked = sorted(zip(feat_names, F_scores, p_values),
                        key=lambda x: -x[1])
        for name, fs, pv in ranked:
            print(f"    {name:25s}  F={fs:8.2f}  p={pv:.3e}")
    except ImportError:
        print("  (sklearn not available -- skipping F-score analysis)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0_wall = time.time()
    print("\n" + "=" * 60)
    print("  M2 Feature Extractor")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading M1 model ...")
    p1, p99 = load_norm()
    print(f"  Normalization: p1={p1:.4f}  p99={p99:.4f}")
    model = load_m1(device)

    print("\n[2] Loading run03 dataset ...")
    X_raw, log_energy, labels_raw, peak_times, snr_npz = load_run03()

    print("\n[3] Running M1 encoder ...")
    latents, ae_scores = run_m1_inference(model, X_raw, p1, p99, device)

    print("\n[4] Loading CSV triggers ...")
    csv_dict, csv_keys = load_csv_triggers()
    print("  Matching NPZ windows to CSV rows ...")
    snr_csv, pfreq, dur, bw = match_triggers(
        peak_times, csv_dict, csv_keys, log_energy, snr_npz
    )

    print("\n[5] Fitting PCA on latents ...")
    pca_mean, pca_Vt, pca_evr, K = fit_pca(latents)
    pca_proj = apply_pca(latents, pca_mean, pca_Vt, K)

    feat_names = [f"pca_{k}" for k in range(K)] + [
        "ae_score", "log_energy", "snr", "peak_frequency", "bandwidth", "duration"
    ]
    print(f"  Feature names ({len(feat_names)}): {feat_names}")

    print("\n[6] Building feature matrix ...")
    features_raw = build_features(pca_proj, ae_scores, log_energy,
                                  snr_csv, pfreq, dur, bw)
    features_norm, feat_min, feat_max = minmax_normalize(features_raw)
    print(f"  features_raw  shape: {features_raw.shape}  "
          f"range [{features_raw.min():.3f}, {features_raw.max():.3f}]")
    print(f"  features_norm shape: {features_norm.shape}  "
          f"range [{features_norm.min():.3f}, {features_norm.max():.3f}]")

    labels_macro      = np.array([to_macro_idx(l) for l in labels_raw],
                                 dtype=np.int64)
    macro_class_names = np.array(MACRO_NAMES, dtype=object)

    out_path = OUT_DIR / "m2_features.npz"
    np.savez_compressed(
        str(out_path),
        features               = features_norm,
        features_raw           = features_raw,
        labels_macro           = labels_macro,
        labels_original        = np.array(labels_raw, dtype=object),
        feature_names          = np.array(feat_names, dtype=object),
        macro_class_names      = macro_class_names,
        pca_explained_variance = pca_evr[:K].astype(np.float32),
        pca_components         = pca_Vt[:K].astype(np.float32),   # (K, 32)
        pca_mean               = pca_mean.astype(np.float32),      # (32,)
        feature_min            = feat_min,
        feature_max            = feat_max,
        ae_scores_raw          = ae_scores,
        peak_times             = peak_times,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n  Saved: {out_path}  ({size_mb:.2f} MB)")

    meta = {
        "n_windows":               int(len(features_raw)),
        "n_features":              int(len(feat_names)),
        "n_pca_components":        int(K),
        "pca_variance_thr":        PCA_VARIANCE_THR,
        "pca_cumulative_variance": float(np.cumsum(pca_evr)[K - 1]),
        "feature_names":           feat_names,
        "macro_class_names":       MACRO_NAMES,
        "macro_class_counts": {
            MACRO_NAMES[i]: int(np.sum(labels_macro == i))
            for i in range(len(MACRO_NAMES))
        },
        "norm_p1":               p1,
        "norm_p99":              p99,
        "csv_match_tolerance_s": CSV_MATCH_TOL,
        "generated_at":          time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = OUT_DIR / "m2_features_meta.json"
    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta:  {meta_path}")

    print_summary(feat_names, features_raw, features_norm, labels_macro,
                  pca_evr, K)

    elapsed = time.time() - t0_wall
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
