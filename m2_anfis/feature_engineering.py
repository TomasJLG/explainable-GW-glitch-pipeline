"""
feature_engineering.py
Loads pre-computed m2_data/m2_features.npz and prepares train/val/test splits
for both ANFIS modes.

Binary POC  : 3 features (peak_frequency, ae_score, log_energy), binary label
              Scatter(2) = 0,  everything else = 1
Full 5-class: all N features from the NPZ, 5-class label (0-4)

Split: 70 / 15 / 15 stratified. Falls back to random if a class is too small.
"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_NPZ = PROJECT_ROOT / "m2_data" / "m2_features.npz"

# Feature indices for binary POC (top-3 by ANOVA F-score: peak_freq, ae_score, log_energy)
# These are *name-based* so they adapt if K (PCA components) changes.
BINARY_POC_FEATURES = ["peak_frequency", "ae_score", "log_energy"]

# Minimum samples per class to attempt stratified split
MIN_STRATIFY = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stratified_split(X, y, train_frac=0.70, val_frac=0.15, seed=42):
    """
    Stratified split into train / val / test.
    Falls back to random if any class has fewer than MIN_STRATIFY samples.
    Returns (X_tr, y_tr), (X_va, y_va), (X_te, y_te).
    """
    rng    = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)

    if counts.min() < MIN_STRATIFY:
        # Random fallback
        idx = rng.permutation(len(X))
        n_tr = int(len(X) * train_frac)
        n_va = int(len(X) * val_frac)
        tr, va, te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
        return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])

    tr_idx, va_idx, te_idx = [], [], []
    for cls in classes:
        ci  = np.where(y == cls)[0]
        ci  = rng.permutation(ci)
        n   = len(ci)
        n_tr = max(1, int(n * train_frac))
        n_va = max(1, int(n * val_frac))
        tr_idx.extend(ci[:n_tr])
        va_idx.extend(ci[n_tr:n_tr+n_va])
        te_idx.extend(ci[n_tr+n_va:])

    tr_idx = np.array(tr_idx)
    va_idx = np.array(va_idx)
    te_idx = np.array(te_idx)
    return (X[tr_idx], y[tr_idx]), (X[va_idx], y[va_idx]), (X[te_idx], y[te_idx])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_features_npz(npz_path: Path = FEATURES_NPZ):
    """Return the raw NPZ object (caller selects what they need)."""
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Features NPZ not found: {npz_path}\n"
            "Run  python m2_feature_extractor.py  first."
        )
    return np.load(str(npz_path), allow_pickle=True)


def load_binary_poc(seed: int = 42):
    """
    Returns train/val/test splits for binary POC mode.

    Feature columns: peak_frequency, ae_score, log_energy  (n_features=3)
    Label: 0 = Scatter, 1 = non-Scatter

    Returns:
        splits   : ((X_tr, y_tr), (X_va, y_va), (X_te, y_te))
        feat_idx : list[int] -- column indices into features matrix
        feat_names: list[str]
    """
    npz = load_features_npz()
    features   = npz["features"].astype(np.float32)        # (N, F) normalised
    feat_names = list(npz["feature_names"].astype(str))
    labels_macro = npz["labels_macro"].astype(np.int64)    # 0-4

    # Select columns
    feat_idx = [feat_names.index(f) for f in BINARY_POC_FEATURES]
    X = features[:, feat_idx]

    # Binary label: Scatter(2) = 0, everything else = 1
    y = (labels_macro != 2).astype(np.int64)

    splits = _stratified_split(X, y, seed=seed)
    return splits, feat_idx, BINARY_POC_FEATURES


def load_full_5class(seed: int = 42):
    """
    Returns train/val/test splits for full 5-class mode.

    Returns:
        splits          : ((X_tr, y_tr), (X_va, y_va), (X_te, y_te))
        feat_names      : list[str]
        macro_class_names: list[str]
        class_weights   : np.ndarray (5,) -- balanced class weights for loss
    """
    npz = load_features_npz()
    features          = npz["features"].astype(np.float32)
    feat_names        = list(npz["feature_names"].astype(str))
    labels_macro      = npz["labels_macro"].astype(np.int64)
    macro_class_names = list(npz["macro_class_names"].astype(str))

    splits = _stratified_split(features, labels_macro, seed=seed)

    # Balanced class weights from training split
    y_tr = splits[0][1]
    n_classes = len(macro_class_names)
    class_weights = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        n_c = np.sum(y_tr == c)
        class_weights[c] = len(y_tr) / (n_classes * max(n_c, 1))

    return splits, feat_names, macro_class_names, class_weights


def print_split_summary(splits, label_names=None):
    """Print class distribution in each split."""
    split_names = ["Train", "Val  ", "Test "]
    for (X, y), name in zip(splits, split_names):
        counts = {int(c): int(np.sum(y == c)) for c in np.unique(y)}
        parts  = []
        for idx, cnt in sorted(counts.items()):
            tag = label_names[idx] if label_names else str(idx)
            parts.append(f"{tag}={cnt}")
        print(f"  {name}: {len(X):4d} samples  [{', '.join(parts)}]")
