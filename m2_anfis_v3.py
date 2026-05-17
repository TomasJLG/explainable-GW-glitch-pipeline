"""
m2_anfis_v3.py
Two-stage hierarchical ANFIS classifier with per-IFO-epoch normalization.

Stage 1: Loud vs Rest  (binary, ra=0.15)
Stage 2: Burst vs Scatter vs Other  (3-class, ra=0.10, on Rest samples)

Input : m2_data/m2_features_v2.npz  (1637 samples, 6 features, 4 classes)
Output: m2_outputs_v3/  (checkpoints + plots)

Usage:
    python m2_anfis_v3.py
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_DIR      = PROJECT_ROOT / "gravityspy_o3"
NPZ_PATH     = PROJECT_ROOT / "m2_data" / "m2_features_v2.npz"
OUT_DIR      = PROJECT_ROOT / "m2_outputs_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_NAMES  = ["peak_frequency", "ae_score", "log_energy",
                  "snr", "duration", "bandwidth"]
CLASS_NAMES    = ["Loud", "Burst", "Scatter", "Other"]
STAGE2_NAMES   = ["Burst", "Scatter", "Other"]
N_FEATURES     = 6

SEED           = 42
TRAIN_FRAC     = 0.80
LR             = 1e-2
MAX_EPOCHS     = 200
PATIENCE       = 20
BATCH_SIZE     = 64
RA_S1          = 0.15
RA_S2          = 0.10
SPREAD         = 0.3
NORM_P_LO      = 2
NORM_P_HI      = 98

# GPS epoch boundaries
O3A_START, O3A_END = 1238166018, 1253977218
O3B_START, O3B_END = 1256655618, 1269363618

V1 = {"acc": 0.7460, "mf1": 0.6677,
      "Loud": 0.800, "Burst": 0.784, "Scatter": 0.728, "Other": 0.359}
V2 = {"acc": 0.5180, "mf1": 0.5427,
      "Loud": 0.856, "Burst": 0.488, "Scatter": 0.445, "Other": 0.382}

# ===========================================================================
# ANFIS (inline)
# ===========================================================================

class GBellMF(nn.Module):
    def __init__(self, n_rules: int, n_features: int):
        super().__init__()
        self.a_raw = nn.Parameter(torch.zeros(n_rules, n_features))
        self.b_raw = nn.Parameter(torch.zeros(n_rules, n_features))
        self.c     = nn.Parameter(torch.zeros(n_rules, n_features))

    def get_params(self):
        return (F.softplus(self.a_raw) + 1e-6,
                F.softplus(self.b_raw) + 1e-6,
                self.c)

    def forward(self, x):
        a, b, c = self.get_params()
        mu = 1.0 / (1.0 + ((x.unsqueeze(1) - c) / a).abs().pow(2 * b))
        return mu.prod(dim=2)   # (B, R)

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
        w_bar = self.get_w_bar(x)
        x_aug = torch.cat([x, x.new_ones(x.shape[0], 1)], dim=1)
        f = torch.einsum("bf,rfc->brc", x_aug, self.consequent)
        return (w_bar.unsqueeze(2) * f).sum(dim=1)

    def init_from_centers(self, centers, spread=0.3):
        self.mf.init_from_centers(centers, spread)


# ===========================================================================
# Helpers
# ===========================================================================

def subtractive_clustering(data: np.ndarray, ra: float,
                            rb_factor: float = 1.5,
                            reject_ratio: float = 0.15) -> np.ndarray:
    from scipy.spatial.distance import cdist
    X   = np.asarray(data, dtype=np.float64)
    rb  = ra * rb_factor
    D2  = cdist(X, X, "sqeuclidean")
    P   = np.exp(-D2 / (ra / 2) ** 2).sum(axis=1)
    P0  = P.max()
    centers = []
    while True:
        i = int(np.argmax(P))
        if P[i] < reject_ratio * P0:
            break
        centers.append(X[i].copy())
        P -= P[i] * np.exp(-D2[i] / (rb / 2) ** 2)
        P  = np.maximum(P, 0.0)
    return np.array(centers)


def stratified_split(X, y, train_frac, seed):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * train_frac))
        tr.extend(idx[:cut]); te.extend(idx[cut:])
    tr = np.array(tr); te = np.array(te)
    return tr, te


def lse_update(model: ANFIS, X_t: torch.Tensor, y_t: torch.Tensor):
    model.eval()
    with torch.no_grad():
        w_bar = model.get_w_bar(X_t)
    N, F, R, C = X_t.shape[0], model.n_features, model.n_rules, model.n_classes
    x_aug = torch.cat([X_t, X_t.new_ones(N, 1)], dim=1)
    Phi   = (w_bar.unsqueeze(2) * x_aug.unsqueeze(1)).reshape(N, R * (F + 1))
    Y     = np.zeros((N, C), dtype=np.float32)
    for i, yi in enumerate(y_t.cpu().numpy()):
        Y[i, int(yi)] = 1.0
    theta, _, _, _ = np.linalg.lstsq(Phi.cpu().numpy(), Y, rcond=None)
    model.consequent.data.copy_(
        torch.tensor(theta, dtype=torch.float32).reshape(R, F + 1, C))


def compute_metrics(y_true, y_pred, names):
    acc = float(np.mean(y_true == y_pred))
    f1s = {}
    for i, nm in enumerate(names):
        tp = int(np.sum((y_pred == i) & (y_true == i)))
        fp = int(np.sum((y_pred == i) & (y_true != i)))
        fn = int(np.sum((y_pred != i) & (y_true == i)))
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1s[nm] = 2 * p * r / max(p + r, 1e-9)
    return acc, float(np.mean(list(f1s.values()))), f1s


def class_weights_tensor(y, n_classes, device):
    w = np.array([1.0 / max((y == c).sum(), 1) for c in range(n_classes)],
                 dtype=np.float32)
    w = w / w.sum() * n_classes
    return torch.tensor(w, device=device)


# ===========================================================================
# IFO x epoch recovery
# ===========================================================================

def recover_ifo_epoch(peak_times: np.ndarray) -> np.ndarray:
    """
    Return array of strings like 'H1_O3a', 'L1_O3b' for each peak_time.
    Strategy: assign epoch from GPS range, assign IFO by checking against H1 CSVs.
    """
    print("  Building H1 trigger sets (for H1 vs L1 assignment) ...")
    h1_sets = {}
    for fname in ("H1_O3a.csv", "H1_O3b.csv"):
        p = CSV_DIR / fname
        if not p.exists():
            print(f"  [WARN] {fname} not found -- all {fname[:6]} assigned to L1")
            h1_sets[fname] = set()
            continue
        s = set()
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    pt = float(row["peak_time"])
                    if row.get("peak_time_ns", "").strip():
                        pt += float(row["peak_time_ns"]) / 1e9
                    s.add(round(pt, 3))
                except (KeyError, ValueError):
                    pass
        h1_sets[fname] = s
        print(f"    {fname}: {len(s)} triggers loaded")

    result = []
    for pt in peak_times:
        if O3A_START <= pt <= O3A_END:
            epoch = "O3a"
        else:
            epoch = "O3b"
        key   = round(float(pt), 3)
        fname = f"H1_{epoch}.csv"
        ifo   = "H1" if key in h1_sets.get(fname, set()) else "L1"
        result.append(f"{ifo}_{epoch}")
    return np.array(result)


# ===========================================================================
# Per-IFO-epoch normalization
# ===========================================================================

def fit_group_norms(X_raw: np.ndarray, ifo_epoch: np.ndarray,
                    tr_idx: np.ndarray):
    """
    Compute P{NORM_P_LO}/P{NORM_P_HI} per group from training indices.
    Returns dict: group -> (lo: np.ndarray[F], hi: np.ndarray[F])
    """
    groups = np.unique(ifo_epoch)
    norms  = {}
    for g in groups:
        mask_g_tr = (ifo_epoch[tr_idx] == g)
        X_g = X_raw[tr_idx][mask_g_tr]
        if len(X_g) == 0:
            lo = X_raw.min(axis=0)
            hi = X_raw.max(axis=0)
        else:
            lo = np.percentile(X_g, NORM_P_LO, axis=0).astype(np.float32)
            hi = np.percentile(X_g, NORM_P_HI, axis=0).astype(np.float32)
        norms[g] = (lo, hi)
        print(f"    {g}: {mask_g_tr.sum()} train samples  "
              f"snr=[{lo[3]:.2f}, {hi[3]:.2f}]  "
              f"pf=[{lo[0]:.2f}, {hi[0]:.2f}]")
    return norms


def apply_group_norms(X_raw: np.ndarray, ifo_epoch: np.ndarray,
                      norms: dict) -> np.ndarray:
    X_out = np.empty_like(X_raw)
    for g, (lo, hi) in norms.items():
        mask = ifo_epoch == g
        rng  = hi - lo + 1e-8
        X_out[mask] = np.clip((X_raw[mask] - lo) / rng, 0.0, 1.0)
    return X_out.astype(np.float32)


# ===========================================================================
# Training loop (shared by both stages)
# ===========================================================================

def train_anfis(model: ANFIS, X_tr: np.ndarray, y_tr: np.ndarray,
                X_te: np.ndarray, y_te: np.ndarray,
                device: torch.device, label: str):
    n_classes = model.n_classes
    crit = nn.CrossEntropyLoss(
        weight=class_weights_tensor(y_tr, n_classes, device))
    mf_params = [p for name, p in model.named_parameters()
                 if "consequent" not in name]
    optimizer = optim.Adam(mf_params, lr=LR)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long,    device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.long,    device=device)

    rng       = np.random.default_rng(SEED)
    best_val  = np.inf
    patience  = 0
    log       = {"train_loss": [], "val_loss": [], "val_acc": []}
    ckpt_path = OUT_DIR / f"best_anfis_{label}.pt"

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        lse_update(model, X_tr_t, y_tr_t)

        model.train()
        model.consequent.requires_grad_(False)
        idx    = rng.permutation(len(X_tr))
        ep_loss, nb = 0.0, 0
        for s in range(0, len(X_tr), BATCH_SIZE):
            bi = idx[s: s + BATCH_SIZE]
            optimizer.zero_grad()
            loss = crit(model(X_tr_t[bi]), y_tr_t[bi])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item(); nb += 1
        model.consequent.requires_grad_(True)

        model.eval()
        with torch.no_grad():
            val_out  = model(X_te_t)
            val_loss = crit(val_out, y_te_t).item()
            val_acc  = (val_out.argmax(1) == y_te_t).float().mean().item()

        log["train_loss"].append(ep_loss / max(nb, 1))
        log["val_loss"].append(val_loss)
        log["val_acc"].append(val_acc)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:4d}  "
                  f"train={ep_loss/max(nb,1):.4f}  "
                  f"val={val_loss:.4f}  acc={val_acc:.3f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss; patience = 0
            torch.save({"state_dict": model.state_dict(),
                        "n_rules": model.n_rules}, str(ckpt_path))
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s  (best_val={best_val:.4f})")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return log


# ===========================================================================
# Hierarchical prediction
# ===========================================================================

def predict_hierarchical(model_s1: ANFIS, model_s2: ANFIS,
                         X: np.ndarray, device: torch.device) -> np.ndarray:
    model_s1.eval(); model_s2.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        probs_s1   = model_s1(X_t).softmax(dim=1)[:, 1].cpu().numpy()
        loud_mask  = probs_s1 > 0.5
    y_pred = np.zeros(len(X), dtype=np.int64)
    # Stage 1 Loud -> class 0
    # y_pred[loud_mask] already = 0

    rest_idx = np.where(~loud_mask)[0]
    if len(rest_idx) > 0:
        X_rest = X[rest_idx]
        X_r_t  = torch.tensor(X_rest, dtype=torch.float32, device=device)
        with torch.no_grad():
            preds_s2 = model_s2(X_r_t).argmax(dim=1).cpu().numpy()
        # Stage 2: 0=Burst->1, 1=Scatter->2, 2=Other->3
        y_pred[rest_idx] = preds_s2 + 1
    return y_pred


# ===========================================================================
# Rule extraction (pretty-print)
# ===========================================================================

def print_rules(model: ANFIS, feat_names: list, class_names: list,
                label: str):
    a_np, b_np, c_np = [p.detach().cpu().numpy()
                        for p in model.mf.get_params()]
    cons = model.consequent.detach().cpu().numpy()   # (R, F+1, C)
    q33  = np.percentile(c_np, 33, axis=0)
    q67  = np.percentile(c_np, 67, axis=0)

    def ling(v, q33, q67):
        return "low" if v <= q33 else ("medium" if v <= q67 else "high")

    print(f"\n  Rules for {label} ({model.n_rules} total):")
    for r in range(min(model.n_rules, 8)):
        dom = int(np.argmax(cons[r].sum(axis=0)))
        conds = "  AND  ".join(
            f"{feat_names[f]} IS {ling(c_np[r,f], q33[f], q67[f])}"
            for f in range(len(feat_names)))
        print(f"  R{r:02d}: IF {conds}")
        print(f"       THEN {class_names[dom]}")


# ===========================================================================
# Plots
# ===========================================================================

def make_plots(log_s1, log_s2, cm, model_s1, model_s2):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available -- skipping plots")
        return

    # 1. Training curves (2 subplots)
    for log, label in [(log_s1, "Stage1_Loud-vs-Rest"),
                       (log_s2, "Stage2_Burst-Scatter-Other")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ep = range(1, len(log["train_loss"]) + 1)
        axes[0].plot(ep, log["train_loss"], label="Train")
        axes[0].plot(ep, log["val_loss"],   label="Val")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("CE Loss")
        axes[0].set_title(f"{label} -- Loss"); axes[0].legend()
        axes[1].plot(ep, log["val_acc"])
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"{label} -- Val accuracy")
        fig.tight_layout()
        fname = f"training_curves_{label.split('_')[0].lower()}.png"
        fig.savefig(str(OUT_DIR / fname), dpi=120)
        plt.close(fig)
        print(f"  Saved: {fname}")

    # 2. Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("M2 v3 -- Confusion matrix (hierarchical)")
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "confusion_matrix_v3.png"), dpi=120)
    plt.close(fig)
    print("  Saved: confusion_matrix_v3.png")

    # 3. Membership functions
    colors = plt.cm.tab10(np.linspace(0, 1, max(model_s1.n_rules, model_s2.n_rules)))
    feat_ids = [0, 5]   # peak_frequency, bandwidth
    for model, tag, n_rules in [
        (model_s1, "Stage1", model_s1.n_rules),
        (model_s2, "Stage2", model_s2.n_rules),
    ]:
        a_np, b_np, c_np = [p.detach().cpu().numpy()
                            for p in model.mf.get_params()]
        fig, axes = plt.subplots(1, len(feat_ids), figsize=(9, 4))
        x_lin = np.linspace(0, 1, 200)
        for ai, fi in enumerate(feat_ids):
            ax = axes[ai]
            for r in range(n_rules):
                mu = 1.0 / (1.0 + np.abs(
                    (x_lin - c_np[r, fi]) / a_np[r, fi]) ** (2 * b_np[r, fi]))
                ax.plot(x_lin, mu, color=colors[r],
                        label=f"R{r:02d}", linewidth=1.2)
            ax.set_title(FEATURE_NAMES[fi])
            ax.set_xlabel("Normalized"); ax.set_ylabel("Membership")
            ax.set_ylim(-0.05, 1.05)
        axes[-1].legend(fontsize=7, ncol=2)
        fig.suptitle(f"M2 v3 -- MF {tag} ({n_rules} rules)")
        fig.tight_layout()
        fname = f"membership_functions_{tag.lower()}.png"
        fig.savefig(str(OUT_DIR / fname), dpi=120)
        plt.close(fig)
        print(f"  Saved: {fname}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Input  : {NPZ_PATH}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if not NPZ_PATH.exists():
        print(f"[ERROR] {NPZ_PATH} not found. Run m1m2_retrain_full.py first.")
        return

    d         = np.load(str(NPZ_PATH), allow_pickle=True)
    X_raw     = d["features_raw"].astype(np.float32)   # (1637, 6) unnormalized
    y_all     = d["labels_macro"].astype(np.int64)      # 0=Loud,1=Burst,2=Scatter,3=Other
    peak_times = d["peak_times"]

    N = len(X_raw)
    print(f"\nLoaded {N} samples, {N_FEATURES} features, 4 classes")
    from collections import Counter
    print("Class dist:", {CLASS_NAMES[k]: v
                          for k, v in sorted(Counter(y_all.tolist()).items())})

    # ------------------------------------------------------------------
    # IFO x epoch recovery
    # ------------------------------------------------------------------
    print("\n[1] Recovering IFO x epoch labels ...")
    ifo_epoch = recover_ifo_epoch(peak_times)
    ie_counts  = Counter(ifo_epoch.tolist())
    print("  IFO x epoch dist:", dict(ie_counts))

    # ------------------------------------------------------------------
    # Stratified split (on full 4-class labels)
    # ------------------------------------------------------------------
    print("\n[2] Stratified 80/20 split ...")
    tr_idx, te_idx = stratified_split(X_raw, y_all, TRAIN_FRAC, SEED)
    print(f"  Train={len(tr_idx)}  Test={len(te_idx)}")

    # ------------------------------------------------------------------
    # Per-IFO-epoch normalization (fit on train)
    # ------------------------------------------------------------------
    print("\n[3] Per-IFO-epoch normalization ...")
    norms    = fit_group_norms(X_raw, ifo_epoch, tr_idx)
    X_norm   = apply_group_norms(X_raw, ifo_epoch, norms)
    X_tr     = X_norm[tr_idx];  y_tr = y_all[tr_idx]
    X_te     = X_norm[te_idx];  y_te = y_all[te_idx]

    # Save normalization params
    norm_out = {g: {"lo": lo.tolist(), "hi": hi.tolist()}
                for g, (lo, hi) in norms.items()}
    with open(OUT_DIR / "group_norms.json", "w") as f:
        json.dump(norm_out, f, indent=2)

    # ------------------------------------------------------------------
    # Stage 1 -- Loud vs Rest
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Stage 1 -- Loud vs Rest")
    print("=" * 60)

    # Binary labels: 1=Loud, 0=Rest
    y_tr_s1 = (y_tr == 0).astype(np.int64)
    y_te_s1 = (y_te == 0).astype(np.int64)
    print(f"  Train: Loud={y_tr_s1.sum()}  Rest={(y_tr_s1==0).sum()}")
    print(f"  Test : Loud={y_te_s1.sum()}  Rest={(y_te_s1==0).sum()}")

    print(f"\n  Subtractive clustering (ra={RA_S1}) on {len(X_tr)} samples ...")
    torch.manual_seed(SEED)
    centers_s1 = subtractive_clustering(X_tr, ra=RA_S1)
    n_rules_s1 = len(centers_s1)
    print(f"  Rules: {n_rules_s1}")

    model_s1 = ANFIS(N_FEATURES, n_rules_s1, 2).to(device)
    model_s1.init_from_centers(
        torch.tensor(centers_s1, dtype=torch.float32, device=device), SPREAD)

    print("\n  Training ...")
    log_s1 = train_anfis(model_s1, X_tr, y_tr_s1, X_te, y_te_s1, device, "s1")

    # Stage 1 metrics
    model_s1.eval()
    with torch.no_grad():
        probs_s1_tr = model_s1(torch.tensor(X_tr, device=device)).softmax(1)[:, 1].cpu().numpy()
        probs_s1_te = model_s1(torch.tensor(X_te, device=device)).softmax(1)[:, 1].cpu().numpy()
    y_pred_s1_tr = (probs_s1_tr > 0.5).astype(np.int64)
    y_pred_s1_te = (probs_s1_te > 0.5).astype(np.int64)
    acc_s1, _, f1s_s1 = compute_metrics(y_te_s1, y_pred_s1_te, ["Rest", "Loud"])
    print(f"\n  Stage 1 test -- acc={acc_s1:.4f}  "
          f"F1(Loud)={f1s_s1['Loud']:.3f}  F1(Rest)={f1s_s1['Rest']:.3f}")

    # ------------------------------------------------------------------
    # Stage 2 -- Burst vs Scatter vs Other
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Stage 2 -- Burst vs Scatter vs Other")
    print("=" * 60)

    # Select NON-Loud train samples (true labels)
    nl_tr = y_tr != 0
    X_tr_s2 = X_tr[nl_tr]
    y_tr_s2 = y_tr[nl_tr] - 1   # 1->0(Burst), 2->1(Scatter), 3->2(Other)

    nl_te = y_te != 0
    X_te_s2 = X_te[nl_te]
    y_te_s2 = y_te[nl_te] - 1

    print(f"  Train: {len(X_tr_s2)} samples  "
          f"Burst={(y_tr_s2==0).sum()}  "
          f"Scatter={(y_tr_s2==1).sum()}  "
          f"Other={(y_tr_s2==2).sum()}")
    print(f"  Test : {len(X_te_s2)} samples  "
          f"Burst={(y_te_s2==0).sum()}  "
          f"Scatter={(y_te_s2==1).sum()}  "
          f"Other={(y_te_s2==2).sum()}")

    print(f"\n  Subtractive clustering (ra={RA_S2}) on {len(X_tr_s2)} samples ...")
    torch.manual_seed(SEED)
    centers_s2 = subtractive_clustering(X_tr_s2, ra=RA_S2)
    n_rules_s2 = len(centers_s2)
    print(f"  Rules: {n_rules_s2}")

    model_s2 = ANFIS(N_FEATURES, n_rules_s2, 3).to(device)
    model_s2.init_from_centers(
        torch.tensor(centers_s2, dtype=torch.float32, device=device), SPREAD)

    print("\n  Training ...")
    log_s2 = train_anfis(model_s2, X_tr_s2, y_tr_s2, X_te_s2, y_te_s2, device, "s2")

    # Stage 2 metrics (on true non-Loud test samples)
    model_s2.eval()
    with torch.no_grad():
        p_s2 = model_s2(torch.tensor(X_te_s2, device=device)).argmax(1).cpu().numpy()
    acc_s2, mf1_s2, f1s_s2 = compute_metrics(y_te_s2, p_s2, STAGE2_NAMES)
    print(f"\n  Stage 2 test (true non-Loud) -- acc={acc_s2:.4f}  macro-F1={mf1_s2:.4f}")
    for nm in STAGE2_NAMES:
        print(f"    F1({nm})={f1s_s2[nm]:.3f}")

    # ------------------------------------------------------------------
    # Combined hierarchical evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Combined Evaluation (4 classes)")
    print("=" * 60)

    y_pred_all = predict_hierarchical(model_s1, model_s2, X_norm, device)
    y_pred_te  = predict_hierarchical(model_s1, model_s2, X_te, device)

    acc_all, mf1_all, f1s_all = compute_metrics(y_all,  y_pred_all, CLASS_NAMES)
    acc_te,  mf1_te,  f1s_te  = compute_metrics(y_te,   y_pred_te,  CLASS_NAMES)

    print(f"\n  All {N} samples:")
    print(f"    Accuracy  : {acc_all:.4f}")
    print(f"    Macro-F1  : {mf1_all:.4f}")
    for nm in CLASS_NAMES:
        print(f"    F1({nm:7s}) : {f1s_all[nm]:.3f}")

    print(f"\n  Test set ({len(X_te)} samples):")
    print(f"    Accuracy  : {acc_te:.4f}")
    print(f"    Macro-F1  : {mf1_te:.4f}")
    for nm in CLASS_NAMES:
        print(f"    F1({nm:7s}) : {f1s_te[nm]:.3f}")

    # Confusion matrix (all samples)
    cm = np.zeros((4, 4), dtype=np.int64)
    for t, p in zip(y_all, y_pred_all):
        cm[int(t), int(p)] += 1
    print(f"\n  Confusion matrix (all, rows=true, cols=pred):")
    hdr = "  " + " " * 10 + "  ".join(f"{c:>8}" for c in CLASS_NAMES)
    print(hdr)
    for i, nm in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cm[i, j]:>8d}" for j in range(4))
        print(f"  {nm:>10}  {row}")

    # Comparison table
    print(f"\n  Comparison v1 / v2 / v3 (all-samples):")
    print(f"  {'Metric':12s}  {'v1':>8}  {'v2':>8}  {'v3':>8}  {'v3-v1':>8}")
    print(f"  {'-'*55}")
    def cmp(label, k_acc_or_f1, is_acc=False):
        if is_acc:
            v3 = acc_all
        elif k_acc_or_f1 == "mf1":
            v3 = mf1_all
        else:
            v3 = f1s_all.get(k_acc_or_f1, 0.0)
        ref = {"acc": V1["acc"], "mf1": V1["mf1"],
               "Loud": V1["Loud"], "Burst": V1["Burst"],
               "Scatter": V1["Scatter"], "Other": V1["Other"]}
        ref2= {"acc": V2["acc"], "mf1": V2["mf1"],
               "Loud": V2["Loud"], "Burst": V2["Burst"],
               "Scatter": V2["Scatter"], "Other": V2["Other"]}
        key = "acc" if is_acc else k_acc_or_f1
        print(f"  {label:12s}  {ref[key]:8.4f}  {ref2[key]:8.4f}  {v3:8.4f}  {v3-ref[key]:+8.4f}")
    cmp("Accuracy",  "acc",     is_acc=True)
    cmp("Macro-F1",  "mf1")
    for nm in CLASS_NAMES:
        cmp(f"F1({nm})",   nm)

    # Rule extraction
    print_rules(model_s1, FEATURE_NAMES, ["Rest", "Loud"], "Stage-1 (Loud vs Rest)")
    print_rules(model_s2, FEATURE_NAMES, STAGE2_NAMES,     "Stage-2 (Burst/Scatter/Other)")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n[Plots]")
    make_plots(log_s1, log_s2, cm, model_s1, model_s2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  M2 v3 Summary")
    print(f"{'='*60}")
    print(f"  Architecture   : 2-stage hierarchical ANFIS")
    print(f"  Normalization  : per-IFO-epoch P{NORM_P_LO}/P{NORM_P_HI}")
    print(f"  Stage 1        : {n_rules_s1} rules (ra={RA_S1})  Loud vs Rest")
    print(f"  Stage 2        : {n_rules_s2} rules (ra={RA_S2})  Burst/Scatter/Other")
    print(f"  Accuracy (all) : {acc_all:.4f}  (v1={V1['acc']:.4f}  delta={acc_all-V1['acc']:+.4f})")
    print(f"  Macro-F1 (all) : {mf1_all:.4f}  (v1={V1['mf1']:.4f}  delta={mf1_all-V1['mf1']:+.4f})")
    print(f"  Output dir     : {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
