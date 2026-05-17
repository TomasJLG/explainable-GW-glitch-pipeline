"""
main.py
CLI entry point for M2 ANFIS training and evaluation.

Usage:
    python -m m2_anfis.main --mode binary_poc
    python -m m2_anfis.main --mode full_5class
    python -m m2_anfis.main --mode binary_poc --eval-only
    python -m m2_anfis.main --mode full_5class --eval-only
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CKPT_DIR = PROJECT_ROOT / "m2_anfis" / "checkpoints"


def run_binary_poc(eval_only: bool, device: torch.device, seed: int):
    from m2_anfis.anfis_model import ANFIS
    from m2_anfis.evaluate import evaluate_model
    from m2_anfis.feature_engineering import load_binary_poc
    from m2_anfis.train import (
        BINARY_N_FEATURES, BINARY_N_LEVELS, BINARY_SPREAD,
        train_binary_poc,
    )
    from m2_anfis.clustering import grid_centers

    ckpt_path = CKPT_DIR / "binary_poc_best.pt"

    if eval_only:
        if not ckpt_path.exists():
            print(f"[ERROR] No checkpoint found: {ckpt_path}")
            print("  Run without --eval-only first.")
            return

        print("\n[INFO] Loading checkpoint for eval-only ...")
        ckpt      = torch.load(str(ckpt_path), map_location=device,
                               weights_only=False)
        meta      = ckpt["meta"]
        n_rules   = meta["n_rules"]
        feat_names = meta["feat_names"]

        model = ANFIS(n_features=BINARY_N_FEATURES, n_rules=n_rules,
                      n_classes=1).to(device)
        centers = grid_centers(BINARY_N_FEATURES, BINARY_N_LEVELS).to(device)
        model.init_from_centers(centers, spread=BINARY_SPREAD)
        model.load_state_dict(ckpt["state_dict"])

        splits, feat_idx, feat_names = load_binary_poc(seed=seed)
    else:
        model, splits, feat_names = train_binary_poc(device=device, seed=seed)

    evaluate_model(model, splits, feat_names,
                   class_names=["Scatter", "non-Scatter"],
                   device=device, binary=True)


def run_full_5class(eval_only: bool, device: torch.device, seed: int):
    from m2_anfis.anfis_model import ANFIS
    from m2_anfis.evaluate import evaluate_model
    from m2_anfis.feature_engineering import load_full_5class
    from m2_anfis.train import train_full_5class

    ckpt_path = CKPT_DIR / "full_5class_best.pt"

    if eval_only:
        if not ckpt_path.exists():
            print(f"[ERROR] No checkpoint found: {ckpt_path}")
            print("  Run without --eval-only first.")
            return

        print("\n[INFO] Loading checkpoint for eval-only ...")
        ckpt       = torch.load(str(ckpt_path), map_location=device,
                                weights_only=False)
        meta       = ckpt["meta"]
        n_rules    = meta["n_rules"]
        n_features = meta["n_features"]
        feat_names  = meta["feat_names"]
        class_names = meta["class_names"]

        model = ANFIS(n_features=n_features, n_rules=n_rules,
                      n_classes=len(class_names)).to(device)
        model.load_state_dict(ckpt["state_dict"])

        splits, _, macro_class_names, _ = load_full_5class(seed=seed)
        class_names = macro_class_names
    else:
        model, splits, feat_names, class_names = train_full_5class(
            device=device, seed=seed
        )

    evaluate_model(model, splits, feat_names, class_names,
                   device=device, binary=False)


def main():
    parser = argparse.ArgumentParser(
        description="M2 ANFIS: train and evaluate glitch classifier"
    )
    parser.add_argument(
        "--mode", choices=["binary_poc", "full_5class"],
        default="full_5class",
        help="Training/evaluation mode (default: full_5class)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; load existing checkpoint and evaluate"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda' or 'cpu' (default: auto-detect)"
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice : {device}")
    print(f"Mode   : {args.mode}")
    print(f"Seed   : {args.seed}")
    print(f"Eval-only: {args.eval_only}")

    if args.mode == "binary_poc":
        run_binary_poc(args.eval_only, device, args.seed)
    else:
        run_full_5class(args.eval_only, device, args.seed)


if __name__ == "__main__":
    main()
