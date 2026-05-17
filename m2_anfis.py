"""
m2_anfis.py
Standalone ANFIS Takagi-Sugeno classifier (4 macro-classes) trained on
features from m2_data/m2_features.npz produced by m2_feature_extractor.py.

Classes: Loud(0), Burst(1), Scatter(2), Other(3)  [Line merged into Other]
Features: peak_frequency, ae_score, log_energy, snr

Hybrid learning:
  - Forward pass: freeze MF params, update consequents via LSE
  - Backward pass: freeze consequents, update MF params via Adam
  Alternates each epoch.

Usage:
    python m2_anfis.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

PROJECT_ROOT   = Path(__file__).resolve().parent
FEATURES_NPZ   = PROJECT_ROOT / "m2_data" / "m2_features.npz"
FEATURES_META  = PROJECT_ROOT / "m2_data" / "m2_features_meta.json"
OUT_DIR        = PROJECT_ROOT / "m2_outputs"

TRAIN_FRAC     = 0.80
SEED           = 42
MAX_EPOCHS     = 200
PATIENCE       = 20
LR_MF          = 0.01          # Adam lr for MF premise params
CLUSTER_RA     = 0.15          # subtractive clustering radius (6D data caps at ~5-6 natural rules)
CLUSTER_RB_F   = 1.5           # rb = ra * factor
MIN_RULES      = 5             # fallback if clustering returns fewer
MAX_RULES      = 25
BATCH_SIZE     = None          # None = full-batch (<=500 samples)
TOP_K_RULES    = 5             # rules to print in IF-THEN format

# Feature and class selection
USE_FEATURES   = ["peak_frequency", "ae_score", "log_energy", "snr", "duration", "bandwidth"]
# Line(3) and Other(4) are merged into a single Other class (index 3)
MERGE_MAP      = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}   # old -> new label
CLASS_NAMES    = ["Loud", "Burst", "Scatter", "Other"]


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data():
    if not FEATURES_NPZ.exists():
        print(f"[ERROR] Not found: {FEATURES_NPZ}")
        print("  Run:  python m2_feature_extractor.py  first.")
        sys.exit(1)

    npz = np.load(str(FEATURES_NPZ), allow_pickle=True)
    X_all        = npz["features"].astype(np.float32)
    y_orig       = npz["labels_macro"].astype(np.int64)
    feat_names_all = list(npz["feature_names"].astype(str))
    labels_orig  = npz["labels_original"].astype(str)

    # Select 4 features by name
    col_idx  = [feat_names_all.index(f) for f in USE_FEATURES]
    X        = X_all[:, col_idx]
    feat_names = USE_FEATURES

    # Merge Line(3) + Other(4) -> Other(3)
    y = np.array([MERGE_MAP[int(v)] for v in y_orig], dtype=np.int64)
    class_names = CLASS_NAMES

    print(f"  Loaded: {X.shape[0]} samples  {X.shape[1]} features")
    print(f"  Features: {feat_names}")
    print(f"  Classes:  {class_names}")
    return X, y, feat_names, class_names, labels_orig


def stratified_split(X, y, train_frac, seed):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    tr_idx, te_idx = [], []
    for cls in classes:
        ci  = np.where(y == cls)[0]
        ci  = rng.permutation(ci)
        n_tr = max(1, int(len(ci) * train_frac))
        tr_idx.extend(ci[:n_tr])
        te_idx.extend(ci[n_tr:])
    return (X[tr_idx], y[tr_idx]), (X[te_idx], y[te_idx])


# ---------------------------------------------------------------------------
# 2. Subtractive clustering (Chiu 1994)
# ---------------------------------------------------------------------------

def subtractive_clustering(data: np.ndarray, ra: float = 0.2,
                            rb_factor: float = 1.5,
                            accept_ratio: float = 0.5,
                            reject_ratio: float = 0.15) -> np.ndarray:
    data  = np.asarray(data, dtype=np.float64)
    N, F  = data.shape
    rb    = ra * rb_factor
    sq_ra = (ra / 2.0) ** 2
    sq_rb = (rb / 2.0) ** 2

    # Initial potentials: O(N^2) -- fine for N <= 500
    potentials = np.zeros(N)
    for i in range(N):
        d2 = np.sum((data - data[i]) ** 2, axis=1)
        potentials[i] = np.sum(np.exp(-d2 / sq_ra))

    p_max_init = potentials.max()
    centers    = []

    for _ in range(N):
        best_idx = int(np.argmax(potentials))
        best_p   = potentials[best_idx]

        if best_p >= accept_ratio * p_max_init:
            centers.append(data[best_idx].copy())
        elif best_p < reject_ratio * p_max_init:
            break
        else:
            c_arr = np.array(centers)
            d_min = np.min(np.linalg.norm(c_arr - data[best_idx], axis=1))
            if (d_min / ra) + (best_p / p_max_init) >= 1.0:
                centers.append(data[best_idx].copy())
            else:
                potentials[best_idx] = 0.0
                continue

        d2 = np.sum((data - centers[-1]) ** 2, axis=1)
        potentials -= best_p * np.exp(-d2 / sq_rb)

    if not centers:
        centers = [data.mean(axis=0)]
    return np.array(centers, dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. ANFIS model
# ---------------------------------------------------------------------------

class ANFIS(nn.Module):
    """
    Takagi-Sugeno ANFIS with 5 layers:
      L1: GBellMF per (rule, feature) -> firing degrees
      L2: Product T-norm -> raw firing strength w (B, R)
      L3: Normalise -> w_bar (B, R)
      L4: Linear consequent per rule per class
      L5: Weighted sum -> logits (B, C)

    Premise params (a_raw, b_raw, c): updated via Adam.
    Consequent param: updated via LSE (data assignment each epoch).
    """

    def __init__(self, n_features: int, n_rules: int, n_classes: int,
                 centers: np.ndarray, a_init: float = 0.3, b_init: float = 2.0):
        super().__init__()
        self.n_features = n_features
        self.n_rules    = n_rules
        self.n_classes  = n_classes

        # Premise parameters (constrained via softplus + eps)
        a_raw_init = float(np.log(np.exp(a_init) - 1.0))
        b_raw_init = float(np.log(np.exp(b_init) - 1.0))
        self.a_raw = nn.Parameter(
            torch.full((n_rules, n_features), a_raw_init))
        self.b_raw = nn.Parameter(
            torch.full((n_rules, n_features), b_raw_init))
        c_tensor = torch.tensor(centers, dtype=torch.float32)  # (R, F)
        # Clamp centers to valid [0,1] range
        self.c = nn.Parameter(c_tensor.clamp(0.0, 1.0))

        # Consequent: (R, F+1, C) -- updated by LSE, no Adam gradient
        self.consequent = nn.Parameter(
            torch.zeros(n_rules, n_features + 1, n_classes),
            requires_grad=True   # needed for loss; excluded from Adam
        )

    def get_ab(self):
        a = F.softplus(self.a_raw) + 1e-6
        b = F.softplus(self.b_raw) + 1e-6
        return a, b

    def firing_strengths(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, F) -> w: (B, R)"""
        a, b = self.get_ab()
        x_e  = x.unsqueeze(1)                     # (B, 1, F)
        ratio = (x_e - self.c) / a                # (B, R, F)
        mu    = 1.0 / (1.0 + torch.abs(ratio) ** (2.0 * b))
        return mu.prod(dim=-1)                     # (B, R)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, F) -> logits: (B, C)"""
        B    = x.size(0)
        w    = self.firing_strengths(x)            # (B, R)
        w_bar = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)

        x_aug  = torch.cat([x, torch.ones(B, 1, device=x.device)], dim=1)  # (B, F+1)
        rule_out = torch.einsum("bf,rfc->brc", x_aug, self.consequent)     # (B, R, C)
        return (w_bar.unsqueeze(-1) * rule_out).sum(dim=1)                 # (B, C)

    def get_phi(self, x: torch.Tensor) -> torch.Tensor:
        """Returns design matrix Phi: (B, R*(F+1)) for LSE update."""
        B = x.size(0)
        with torch.no_grad():
            w     = self.firing_strengths(x)
            w_bar = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)       # (B, R)
            x_aug = torch.cat([x, torch.ones(B, 1, device=x.device)], dim=1)
            phi   = w_bar.unsqueeze(-1) * x_aug.unsqueeze(1)              # (B, R, F+1)
        return phi.view(B, -1)                                             # (B, R*(F+1))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 4. Hybrid training helpers
# ---------------------------------------------------------------------------

def lse_update(model: ANFIS, X_t: torch.Tensor, y_onehot: np.ndarray):
    """
    Update consequent parameters via least squares on the full training set.
    y_onehot: (N, C) float64 one-hot array.
    """
    phi_np = model.get_phi(X_t).cpu().numpy()          # (N, R*(F+1))
    theta, _, _, _ = np.linalg.lstsq(phi_np, y_onehot, rcond=None)
    # theta: (R*(F+1), C)
    R  = model.n_rules
    F1 = model.n_features + 1
    with torch.no_grad():
        model.consequent.data = torch.tensor(
            theta.reshape(R, F1, model.n_classes).astype(np.float32),
            device=model.consequent.device,
        )


def compute_class_weights(y_train: np.ndarray, n_classes: int,
                           device: torch.device) -> torch.Tensor:
    weights = np.zeros(n_classes, dtype=np.float32)
    N = len(y_train)
    for c in range(n_classes):
        n_c = np.sum(y_train == c)
        weights[c] = N / (n_classes * max(n_c, 1))
    return torch.tensor(weights, device=device)


# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------

def train(model: ANFIS, X_tr: np.ndarray, y_tr: np.ndarray,
          X_te: np.ndarray, y_te: np.ndarray,
          device: torch.device):

    n_classes  = model.n_classes
    X_tr_t     = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t     = torch.tensor(y_tr, dtype=torch.long,    device=device)
    X_te_t     = torch.tensor(X_te, dtype=torch.float32, device=device)
    y_te_t     = torch.tensor(y_te, dtype=torch.long,    device=device)

    # One-hot targets for LSE
    y_onehot   = np.eye(n_classes, dtype=np.float64)[y_tr]

    cw         = compute_class_weights(y_tr, n_classes, device)
    criterion  = nn.CrossEntropyLoss(weight=cw)

    # Adam optimises only premise parameters (NOT consequent)
    mf_optimizer = optim.Adam(
        [model.a_raw, model.b_raw, model.c],
        lr=LR_MF, weight_decay=1e-4
    )

    best_loss     = np.inf
    patience_ctr  = 0
    best_state    = None
    tr_losses, te_losses = [], []

    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):

        # -- Forward (LSE update for consequents) --
        lse_update(model, X_tr_t, y_onehot)

        # -- Backward (Adam update for MF params) --
        model.consequent.requires_grad_(False)
        model.train()
        logits = model(X_tr_t)
        loss   = criterion(logits, y_tr_t)
        mf_optimizer.zero_grad()
        loss.backward()
        mf_optimizer.step()
        model.consequent.requires_grad_(True)

        # -- Record losses --
        model.eval()
        with torch.no_grad():
            tr_loss = criterion(model(X_tr_t), y_tr_t).item()
            te_loss = criterion(model(X_te_t), y_te_t).item()
        tr_losses.append(tr_loss)
        te_losses.append(te_loss)

        if epoch % 20 == 0 or epoch == 1:
            tr_acc = (model(X_tr_t).argmax(1) == y_tr_t).float().mean().item()
            te_acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item()
            print(f"  Epoch {epoch:4d}  "
                  f"tr_loss={tr_loss:.4f}  te_loss={te_loss:.4f}  "
                  f"tr_acc={tr_acc:.3f}  te_acc={te_acc:.3f}")

        # -- Early stopping on training loss --
        if tr_loss < best_loss - 1e-5:
            best_loss    = tr_loss
            patience_ctr = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {epoch} "
                      f"(best_tr_loss={best_loss:.4f})")
                break

    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed:.1f}s  ({epoch} epochs)")

    # Restore best
    model.load_state_dict(best_state)
    return tr_losses, te_losses


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: ANFIS, X: np.ndarray, y: np.ndarray,
             class_names: list, device: torch.device):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(X_t)
        preds  = logits.argmax(dim=1).cpu().numpy()

    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y, preds):
        cm[int(t), int(p)] += 1

    per_class = {}
    f1_vals   = []
    for c, name in enumerate(class_names):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        sup = tp + fn
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class[name] = {"prec": prec, "rec": rec, "f1": f1, "support": sup}
        if sup > 0:
            f1_vals.append(f1)

    acc      = float(np.mean(preds == y))
    macro_f1 = float(np.mean(f1_vals)) if f1_vals else 0.0
    return acc, macro_f1, per_class, cm, preds


def print_metrics(acc, macro_f1, per_class, class_names):
    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Macro-F1  : {macro_f1:.4f}")
    print(f"\n  Per-class metrics:")
    hdr = f"  {'Class':10s}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in class_names:
        m = per_class[name]
        print(f"  {name:10s}  {m['prec']:6.3f}  {m['rec']:6.3f}"
              f"  {m['f1']:6.3f}  {m['support']:8d}")


def print_confusion_matrix(cm, class_names):
    w = max(max(len(n) for n in class_names), 5)
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    header = "  " + " " * (w + 2) + "  ".join(f"{n:>{w}}" for n in class_names)
    print(header)
    sep    = "  " + "-" * (len(header) - 2)
    print(sep)
    for i, name in enumerate(class_names):
        row = "  ".join(f"{cm[i, j]:>{w}d}" for j in range(len(class_names)))
        print(f"  {name:>{w}}  {row}")


# ---------------------------------------------------------------------------
# 7. Rule extraction
# ---------------------------------------------------------------------------

def linguistic(val: float) -> str:
    if val < 0.33:
        return "low"
    if val < 0.67:
        return "medium"
    return "high"


def extract_top_rules(model: ANFIS, X_te: np.ndarray, y_te: np.ndarray,
                      feat_names: list, class_names: list,
                      device: torch.device, top_k: int = TOP_K_RULES):
    model.eval()
    X_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    with torch.no_grad():
        w  = model.firing_strengths(X_t)
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-12)
        w_bar = (w / w_sum).cpu().numpy()          # (N, R)

    avg_strength = w_bar.mean(axis=0)              # (R,) mean over test samples
    top_rules    = np.argsort(avg_strength)[::-1][:top_k]

    c_np = model.c.detach().cpu().numpy()          # (R, F)

    print(f"\n  Top {top_k} rules by average firing strength (test set):")
    for rank, r in enumerate(top_rules):
        strength = avg_strength[r]
        conds    = []
        for f, fname in enumerate(feat_names):
            level = linguistic(float(c_np[r, f]))
            conds.append(f"{fname} IS {level}")

        # Dominant output class at rule center
        center_t = torch.tensor(c_np[r:r+1], dtype=torch.float32,
                                device=device)
        with torch.no_grad():
            logit_r = model(center_t).squeeze()
            cls_idx = int(logit_r.argmax().item())
        cls_name = class_names[cls_idx]

        cond_str = " AND ".join(conds)
        print(f"  R{r:03d}: IF {cond_str}")
        print(f"        THEN class={cls_name}  (avg_strength={strength:.4f})")

    return top_rules, avg_strength


# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------

def plot_training_curve(tr_losses, te_losses, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tr_losses, label="Train loss")
    ax.plot(te_losses, label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CrossEntropy loss")
    ax.set_title("M2 ANFIS -- Training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "training_curve_m2.png"
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(cm, class_names, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, aspect="auto", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("M2 ANFIS -- Confusion matrix (test)")
    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=11)
    fig.tight_layout()
    path = out_dir / "confusion_matrix_m2.png"
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_membership_functions(model: ANFIS, feat_names: list, out_dir: Path):
    n_feats = model.n_features
    n_rules = model.n_rules
    ncols   = 4
    nrows   = (n_feats + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    x_vals = np.linspace(0, 1, 300)
    a_np, b_np = [v.detach().cpu().numpy() for v in model.get_ab()]
    c_np       = model.c.detach().cpu().numpy()

    for f_idx in range(n_feats):
        ax = axes_flat[f_idx]
        for r in range(n_rules):
            a_v = float(a_np[r, f_idx])
            b_v = float(b_np[r, f_idx])
            c_v = float(c_np[r, f_idx])
            mu  = 1.0 / (1.0 + np.abs((x_vals - c_v) / max(a_v, 1e-9))
                         ** (2.0 * b_v))
            ax.plot(x_vals, mu, lw=1.0, alpha=0.7)
        ax.set_title(feat_names[f_idx], fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Norm. value", fontsize=7)
        ax.set_ylabel("Membership", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for idx in range(n_feats, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"M2 ANFIS -- Membership functions ({n_rules} rules)",
                 fontsize=11)
    fig.tight_layout()
    path = out_dir / "membership_functions_m2.png"
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("  M2 ANFIS Classifier  (4 classes, 4 features)")
    print("=" * 60)
    print(f"  Device: {device}")

    # ---- Load data ----
    print("\n[1] Loading features ...")
    X, y, feat_names, class_names, _ = load_data()
    n_features = X.shape[1]
    n_classes  = len(class_names)

    (X_tr, y_tr), (X_te, y_te) = stratified_split(X, y, TRAIN_FRAC, SEED)
    print(f"  Train: {len(X_tr)}  Test: {len(X_te)}")

    # Class distribution
    from collections import Counter
    for split_name, y_s in [("Train", y_tr), ("Test", y_te)]:
        dist = Counter(y_s.tolist())
        parts = [f"{class_names[c]}={dist.get(c,0)}" for c in range(n_classes)]
        print(f"  {split_name}: [{', '.join(parts)}]")

    # ---- Subtractive clustering ----
    print(f"\n[2] Subtractive clustering (ra={CLUSTER_RA}) ...")
    centers = subtractive_clustering(X_tr, ra=CLUSTER_RA,
                                     rb_factor=CLUSTER_RB_F)
    n_rules = len(centers)
    if n_rules < MIN_RULES:
        # Augment with uniformly-spaced centers
        extra = np.random.default_rng(SEED).uniform(
            0, 1, (MIN_RULES - n_rules, n_features)).astype(np.float32)
        centers = np.vstack([centers, extra])
        n_rules = len(centers)
        print(f"  Augmented to {n_rules} rules (below MIN_RULES={MIN_RULES})")
    n_rules = min(n_rules, MAX_RULES)
    centers = centers[:n_rules]
    print(f"  Rules: {n_rules}")

    # ---- Build model ----
    print(f"\n[3] Building ANFIS model ...")
    torch.manual_seed(SEED)
    model = ANFIS(n_features, n_rules, n_classes, centers).to(device)
    total_params = model.n_params()
    print(f"  n_rules={n_rules}  n_features={n_features}  n_classes={n_classes}")
    print(f"  Total parameters: {total_params}")
    premise_params = 3 * n_rules * n_features   # a, b, c
    conseq_params  = n_rules * (n_features + 1) * n_classes
    print(f"  Premise (MF) params: {premise_params}  "
          f"Consequent params: {conseq_params}")

    # ---- Train ----
    print(f"\n[4] Hybrid training (LSE + Adam, max {MAX_EPOCHS} epochs) ...")
    tr_losses, te_losses = train(model, X_tr, y_tr, X_te, y_te, device)

    # ---- Evaluate ----
    print(f"\n[5] Evaluation on test set ...")
    acc, macro_f1, per_class, cm, preds = evaluate(
        model, X_te, y_te, class_names, device
    )
    print_metrics(acc, macro_f1, per_class, class_names)
    print_confusion_matrix(cm, class_names)

    # ---- Rule extraction ----
    print(f"\n[6] Rule extraction ...")
    extract_top_rules(model, X_te, y_te, feat_names, class_names, device)

    # ---- Plots ----
    print(f"\n[7] Generating plots ...")
    try:
        plot_training_curve(tr_losses, te_losses, OUT_DIR)
        plot_confusion_matrix(cm, class_names, OUT_DIR)
        plot_membership_functions(model, feat_names, OUT_DIR)
    except Exception as e:
        print(f"  [WARN] Plot failed: {e}")

    # ---- Save model ----
    out_model = OUT_DIR / "best_m2_anfis.pt"
    torch.save({
        "state_dict":  model.state_dict(),
        "n_features":  n_features,
        "n_rules":     n_rules,
        "n_classes":   n_classes,
        "centers":     centers,
        "feat_names":  feat_names,
        "class_names": class_names,
        "acc":         acc,
        "macro_f1":    macro_f1,
        "per_class":   per_class,
    }, str(out_model))
    print(f"\n  Model saved: {out_model}")

    # ---- Summary ----
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Final results")
    print(f"{'='*60}")
    print(f"  Rules          : {n_rules}")
    print(f"  Total params   : {total_params}")
    print(f"  Test accuracy  : {acc:.4f}")
    print(f"  Test macro-F1  : {macro_f1:.4f}")
    print(f"  Outputs        : {OUT_DIR}/")
    print(f"  Total time     : {total_time:.1f}s")


if __name__ == "__main__":
    main()
