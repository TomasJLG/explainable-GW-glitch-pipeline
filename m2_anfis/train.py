"""
train.py
Training loops for M2 ANFIS.

Modes:
    binary_poc  : 3 features, 27 rules (3x3x3 grid), binary cross-entropy
    full_5class : all features, subtractive-clustering rules, cross-entropy
                  with balanced class weights

Both modes use early stopping on validation loss and save the best checkpoint.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from m2_anfis.anfis_model import ANFIS
from m2_anfis.clustering import grid_centers, subtractive_clustering
from m2_anfis.feature_engineering import (
    load_binary_poc,
    load_full_5class,
    print_split_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR     = PROJECT_ROOT / "m2_anfis" / "checkpoints"
LOG_DIR      = PROJECT_ROOT / "m2_anfis" / "logs"

# ---------------------------------------------------------------------------
# Shared training config
# ---------------------------------------------------------------------------

LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 200
PATIENCE     = 20
BATCH_SIZE   = 64
SEED         = 42

# Binary POC config
BINARY_N_FEATURES = 3
BINARY_N_LEVELS   = 3   # 3^3 = 27 rules
BINARY_SPREAD     = 0.3

# Full 5-class config
FULL_RA    = 0.2
FULL_SPREAD = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(X, y, device, binary=False):
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    if binary:
        yt = torch.tensor(y, dtype=torch.float32, device=device)
    else:
        yt = torch.tensor(y, dtype=torch.long, device=device)
    return Xt, yt


def _batch_iter(X, y, batch_size, rng):
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        bi = idx[start:start + batch_size]
        yield X[bi], y[bi]


def _accuracy(logits, y, binary=False):
    if binary:
        preds = (torch.sigmoid(logits.squeeze()) >= 0.5).long()
        return (preds == y.long()).float().mean().item()
    else:
        return (logits.argmax(dim=1) == y).float().mean().item()


def _save_checkpoint(model, path: Path, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, str(path))


def _save_log(log: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        json.dump(log, f, indent=2)


# ---------------------------------------------------------------------------
# Binary POC
# ---------------------------------------------------------------------------

def train_binary_poc(device: torch.device = None, seed: int = SEED):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  M2 ANFIS -- Binary POC")
    print("=" * 60)

    splits, feat_idx, feat_names = load_binary_poc(seed=seed)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits

    print(f"  Features ({len(feat_names)}): {feat_names}")
    print_split_summary(splits, label_names=["Scatter", "non-Scatter"])

    n_rules = BINARY_N_LEVELS ** BINARY_N_FEATURES   # 27
    print(f"\n  Rules: {n_rules}  (grid {BINARY_N_LEVELS}^{BINARY_N_FEATURES})")

    # Init model
    torch.manual_seed(seed)
    model = ANFIS(n_features=BINARY_N_FEATURES, n_rules=n_rules,
                  n_classes=1).to(device)
    centers = grid_centers(BINARY_N_FEATURES, BINARY_N_LEVELS).to(device)
    model.init_from_centers(centers, spread=BINARY_SPREAD)

    # Positive class weight to counter class imbalance (Scatter=0 is minority)
    n_neg = int(np.sum(y_tr == 0))   # Scatter
    n_pos = int(np.sum(y_tr == 1))   # non-Scatter
    pos_w = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32,
                         device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)

    rng       = np.random.default_rng(seed)
    Xtr_t, ytr_t = _to_tensor(X_tr, y_tr, device, binary=True)
    Xva_t, yva_t = _to_tensor(X_va, y_va, device, binary=True)

    best_val_loss = np.inf
    patience_ctr  = 0
    log           = {"mode": "binary_poc", "epochs": [], "train_loss": [],
                     "val_loss": [], "val_acc": []}

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for Xb, yb in _batch_iter(X_tr, y_tr, BATCH_SIZE, rng):
            Xbt, ybt = _to_tensor(Xb, yb, device, binary=True)
            optimizer.zero_grad()
            out  = model(Xbt).squeeze()
            loss = criterion(out, ybt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        model.eval()
        with torch.no_grad():
            val_out  = model(Xva_t).squeeze()
            val_loss = criterion(val_out, yva_t).item()
            val_acc  = _accuracy(val_out, yva_t, binary=True)

        log["epochs"].append(epoch)
        log["train_loss"].append(epoch_loss / max(n_batches, 1))
        log["val_loss"].append(val_loss)
        log["val_acc"].append(val_acc)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}  "
                  f"train_loss={epoch_loss/max(n_batches,1):.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_ctr  = 0
            _save_checkpoint(model, CKPT_DIR / "binary_poc_best.pt",
                             {"epoch": epoch, "val_loss": val_loss,
                              "val_acc": val_acc, "feat_names": feat_names,
                              "n_rules": n_rules})
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
                break

    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed:.1f}s")

    # Final test evaluation
    ckpt = torch.load(str(CKPT_DIR / "binary_poc_best.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    Xte_t, yte_t = _to_tensor(X_te, y_te, device, binary=True)
    with torch.no_grad():
        te_out  = model(Xte_t).squeeze()
        te_loss = criterion(te_out, yte_t).item()
        te_acc  = _accuracy(te_out, yte_t, binary=True)

    print(f"  Test  loss={te_loss:.4f}  acc={te_acc:.3f}")
    log["test_loss"] = te_loss
    log["test_acc"]  = te_acc
    _save_log(log, LOG_DIR / "binary_poc_log.json")

    return model, splits, feat_names


# ---------------------------------------------------------------------------
# Full 5-class
# ---------------------------------------------------------------------------

def train_full_5class(device: torch.device = None, seed: int = SEED):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  M2 ANFIS -- Full 5-class")
    print("=" * 60)

    splits, feat_names, macro_class_names, class_weights = load_full_5class(
        seed=seed
    )
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits
    n_features = X_tr.shape[1]
    n_classes  = len(macro_class_names)

    print(f"  Features ({n_features}): {feat_names}")
    print(f"  Classes  ({n_classes}): {macro_class_names}")
    print_split_summary(splits, label_names=macro_class_names)

    # Subtractive clustering on training data
    print(f"\n  Running subtractive clustering (ra={FULL_RA}) ...")
    centers_np = subtractive_clustering(X_tr, ra=FULL_RA)
    n_rules    = len(centers_np)
    print(f"  Rules from clustering: {n_rules}")

    # Init model
    torch.manual_seed(seed)
    model = ANFIS(n_features=n_features, n_rules=n_rules,
                  n_classes=n_classes).to(device)
    centers_t = torch.tensor(centers_np, dtype=torch.float32, device=device)
    model.init_from_centers(centers_t, spread=FULL_SPREAD)

    weights_t = torch.tensor(class_weights, dtype=torch.float32,
                              device=device)
    criterion = nn.CrossEntropyLoss(weight=weights_t)
    optimizer = optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)

    rng = np.random.default_rng(seed)
    Xva_t, yva_t = _to_tensor(X_va, y_va, device, binary=False)

    best_val_loss = np.inf
    patience_ctr  = 0
    log = {"mode": "full_5class", "n_rules": n_rules, "feat_names": feat_names,
           "class_names": macro_class_names,
           "epochs": [], "train_loss": [], "val_loss": [], "val_acc": []}

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for Xb, yb in _batch_iter(X_tr, y_tr, BATCH_SIZE, rng):
            Xbt, ybt = _to_tensor(Xb, yb, device, binary=False)
            optimizer.zero_grad()
            out  = model(Xbt)
            loss = criterion(out, ybt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        model.eval()
        with torch.no_grad():
            val_out  = model(Xva_t)
            val_loss = criterion(val_out, yva_t).item()
            val_acc  = _accuracy(val_out, yva_t, binary=False)

        log["epochs"].append(epoch)
        log["train_loss"].append(epoch_loss / max(n_batches, 1))
        log["val_loss"].append(val_loss)
        log["val_acc"].append(val_acc)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}  "
                  f"train_loss={epoch_loss/max(n_batches,1):.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_ctr  = 0
            _save_checkpoint(model, CKPT_DIR / "full_5class_best.pt",
                             {"epoch": epoch, "val_loss": val_loss,
                              "val_acc": val_acc, "n_rules": n_rules,
                              "n_features": n_features,
                              "feat_names": feat_names,
                              "class_names": macro_class_names})
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
                break

    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed:.1f}s")

    # Final test evaluation
    ckpt = torch.load(str(CKPT_DIR / "full_5class_best.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    Xte_t, yte_t = _to_tensor(X_te, y_te, device, binary=False)
    with torch.no_grad():
        te_out  = model(Xte_t)
        te_loss = criterion(te_out, yte_t).item()
        te_acc  = _accuracy(te_out, yte_t, binary=False)

    print(f"  Test  loss={te_loss:.4f}  acc={te_acc:.3f}")
    log["test_loss"] = te_loss
    log["test_acc"]  = te_acc
    _save_log(log, LOG_DIR / "full_5class_log.json")

    return model, splits, feat_names, macro_class_names
