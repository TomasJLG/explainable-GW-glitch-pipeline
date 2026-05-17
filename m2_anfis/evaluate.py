"""
evaluate.py
Evaluation, metrics, confusion matrix, and human-readable rule extraction
for M2 ANFIS models.
"""

import numpy as np
import torch

from m2_anfis.anfis_model import ANFIS


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def predict(model: ANFIS, X: np.ndarray, device: torch.device,
            binary: bool = False) -> np.ndarray:
    """Return predicted class indices (N,)."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(Xt)
        if binary:
            preds = (torch.sigmoid(logits.squeeze()) >= 0.5).long()
        else:
            preds = logits.argmax(dim=1)
    return preds.cpu().numpy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: list):
    """
    Returns dict with:
        accuracy, per_class (precision, recall, f1, support), macro_f1
    """
    n_classes = len(class_names)
    metrics   = {"accuracy": float(np.mean(y_true == y_pred))}

    per_class = {}
    f1_vals   = []
    for c, name in enumerate(class_names):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        support = tp + fn
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class[name] = {"precision": prec, "recall": rec,
                            "f1": f1, "support": support}
        if support > 0:
            f1_vals.append(f1)

    metrics["per_class"] = per_class
    metrics["macro_f1"]  = float(np.mean(f1_vals)) if f1_vals else 0.0
    return metrics


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def print_metrics(metrics: dict, class_names: list):
    print(f"\n  Overall accuracy : {metrics['accuracy']:.3f}")
    print(f"  Macro F1         : {metrics['macro_f1']:.3f}")
    print(f"\n  Per-class metrics:")
    header = f"  {'Class':12s}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, m in metrics["per_class"].items():
        print(f"  {name:12s}  {m['precision']:6.3f}  {m['recall']:6.3f}"
              f"  {m['f1']:6.3f}  {m['support']:8d}")


def print_confusion_matrix(cm: np.ndarray, class_names: list):
    print("\n  Confusion matrix (rows=true, cols=pred):")
    width = max(len(n) for n in class_names) + 2
    header = "  " + " " * width + "  ".join(f"{n:>{width}}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = "  ".join(f"{cm[i, j]:>{width}d}" for j in range(len(class_names)))
        print(f"  {name:>{width}}  {row}")


# ---------------------------------------------------------------------------
# Rule extraction
# ---------------------------------------------------------------------------

def extract_rules(model: ANFIS, feat_names: list,
                  class_names: list) -> list[dict]:
    """
    Interpret each fuzzy rule linguistically.

    For each rule r:
      - For each feature f, assign "low" / "medium" / "high" based on the
        center c[r,f] relative to the distribution of centers across rules.
        Bottom third -> "low", middle -> "medium", top third -> "high".
      - Report the dominant output class (argmax of the consequent weight
        vector at the rule center).

    Returns list of dicts, one per rule, sorted by firing strength variance
    (most discriminative rules first).
    """
    a, b, c = model.mf.get_params()
    c_np    = c.detach().cpu().numpy()          # (R, F)
    n_rules, n_feats = c_np.shape

    # Consequent weight: run the model at each rule center to get the
    # implied output. We use the center as a synthetic input.
    model.eval()
    with torch.no_grad():
        c_t   = c.clamp(0.0, 1.0)              # clip to [0,1] for valid input
        logits = model(c_t)                     # (R, C)
        if logits.shape[1] == 1:
            # Binary
            dominant_cls = (torch.sigmoid(logits.squeeze()) >= 0.5).long()
            dominant_cls = dominant_cls.cpu().numpy()
        else:
            dominant_cls = logits.argmax(dim=1).cpu().numpy()  # (R,)

    # Linguistic labels for centers
    col_q33 = np.percentile(c_np, 33, axis=0)
    col_q67 = np.percentile(c_np, 67, axis=0)

    def linguistic(val, q33, q67):
        if val <= q33:
            return "low"
        elif val <= q67:
            return "medium"
        return "high"

    rules = []
    for r in range(n_rules):
        conditions = {}
        for f, fname in enumerate(feat_names):
            conditions[fname] = linguistic(c_np[r, f], col_q33[f], col_q67[f])
        cls_idx = int(dominant_cls[r])
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        rules.append({
            "rule_id":    r,
            "conditions": conditions,
            "output":     cls_name,
            "center":     c_np[r].tolist(),
        })

    return rules


def print_rules(rules: list, max_rules: int = 20):
    print(f"\n  Fuzzy rules (top {min(max_rules, len(rules))} of {len(rules)}):")
    for rule in rules[:max_rules]:
        cond_str = "  AND  ".join(
            f"{k} is {v}" for k, v in rule["conditions"].items()
        )
        print(f"  R{rule['rule_id']:03d}: IF {cond_str}  =>  {rule['output']}")


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_model(model: ANFIS, splits, feat_names: list,
                   class_names: list, device: torch.device,
                   binary: bool = False):
    """Run full evaluation on all splits and print report."""
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits
    split_data = [("Train", X_tr, y_tr),
                  ("Val  ", X_va, y_va),
                  ("Test ", X_te, y_te)]

    print("\n" + "=" * 60)
    print("  Evaluation Report")
    print("=" * 60)

    for split_name, X, y in split_data:
        y_pred   = predict(model, X, device, binary=binary)
        metrics  = compute_metrics(y, y_pred, class_names)
        cm       = confusion_matrix(y, y_pred, len(class_names))

        print(f"\n-- {split_name} --")
        print_metrics(metrics, class_names)
        print_confusion_matrix(cm, class_names)

    # Rule extraction (test split)
    print("\n-- Rule Extraction --")
    rules = extract_rules(model, feat_names, class_names)
    print_rules(rules)
    return rules
