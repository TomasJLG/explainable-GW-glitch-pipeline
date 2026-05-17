"""
m1_train.py
===========
Script de entrenamiento del módulo M1 — Autoencoder de detección de anomalías.

Uso:
    python -m m1_anomaly.m1_train --npz dataset_H1_O3a_scale_1p0_run02.npz

Genera:
    best_m1_ae.pt        Pesos del mejor modelo (val_loss mínimo)
    normalization.json   Valores p1 y p99 usados para la normalización global
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from m1_autoencoder import GlitchAE
from m1_dataloader import (
    apply_global_normalization,
    build_temporal_split,
    compute_global_normalization,
    load_npz,
)

# ---------------------------------------------------------------------------
# Hiperparámetros por defecto (ajustar vía argumentos CLI)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    batch_size=64,
    epochs=50,
    lr=1e-3,
    latent_dim=128,
    weight_decay=1e-5,
    patience=10,
    energy_threshold_pct=95.0,
    model_out="best_m1_ae.pt",
    norm_out="normalization.json",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena el autoencoder M1.")
    p.add_argument("--npz",            required=True,  help="Ruta al archivo .npz del dataset.")
    p.add_argument("--batch_size",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    p.add_argument("--latent_dim",     type=int,   default=DEFAULTS["latent_dim"])
    p.add_argument("--weight_decay",   type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--patience",       type=int,   default=DEFAULTS["patience"])
    p.add_argument("--energy_pct",     type=float, default=DEFAULTS["energy_threshold_pct"],
                   help="Percentil de energía para filtrar nominales en train (default 95).")
    p.add_argument("--model_out",      default=DEFAULTS["model_out"])
    p.add_argument("--norm_out",       default=DEFAULTS["norm_out"])
    return p.parse_args()


def build_dataloaders(npz_path: str, energy_pct: float, batch_size: int):
    """Carga, normaliza y empaqueta los datos en DataLoaders de PyTorch."""
    # 1. Cargar NPZ
    d = load_npz(npz_path)

    # 2. Split temporal (NO aleatorio — evita leakage)
    split = build_temporal_split(d, energy_threshold_percentile=energy_pct)

    # 3. Normalización global derivada SOLO del train set
    p1, p99 = compute_global_normalization(split["X_train"])
    X_train = apply_global_normalization(split["X_train"], p1, p99)
    X_val   = apply_global_normalization(split["X_val"],   p1, p99)

    # 4. Tensores PyTorch con dimensión de canal: (N, 1, H, W)
    X_train_t = torch.tensor(X_train[:, None, :, :], dtype=torch.float32)
    X_val_t   = torch.tensor(X_val[:, None, :, :],   dtype=torch.float32)

    train_dl = DataLoader(TensorDataset(X_train_t), batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(TensorDataset(X_val_t),   batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    return train_dl, val_dl, len(X_train_t), len(X_val_t), p1, p99


def train_one_epoch(model, loader, criterion, optimizer, device, n_samples):
    model.train()
    running_loss = 0.0
    for (x,) in loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(x)
    return running_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_samples):
    model.eval()
    running_loss = 0.0
    for (x,) in loader:
        x = x.to(device)
        running_loss += criterion(model(x), x).item() * len(x)
    return running_loss / n_samples


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    # ------------------------------------------------------------------
    # Datos
    # ------------------------------------------------------------------
    print("Cargando y preparando datos…")
    train_dl, val_dl, n_train, n_val, p1, p99 = build_dataloaders(
        args.npz, args.energy_pct, args.batch_size
    )
    print(f"  Train: {n_train} ventanas  |  Val: {n_val} ventanas")

    # Guardar normalización — imprescindible para inferencia posterior
    json.dump({"p1": float(p1), "p99": float(p99)}, open(args.norm_out, "w"), indent=2)
    print(f"Normalización guardada en '{args.norm_out}'")

    # ------------------------------------------------------------------
    # Modelo, optimizador y función de pérdida
    # ------------------------------------------------------------------
    model     = GlitchAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {total_params:,}  |  latent_dim={args.latent_dim}")

    # ------------------------------------------------------------------
    # Bucle de entrenamiento con early stopping
    # ------------------------------------------------------------------
    best_val_loss    = float("inf")
    patience_counter = 0

    print(f"\n{'Epoch':>6}  {'Train loss':>12}  {'Val loss':>12}  {'Estado':>10}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device, n_train)
        val_loss   = evaluate(model, val_dl, criterion, device, n_val)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.model_out)
            status = "✓ guardado"
        else:
            patience_counter += 1
            status = f"paciencia {patience_counter}/{args.patience}"

        print(f"{epoch:>6d}  {train_loss:>12.6f}  {val_loss:>12.6f}  {status:>10}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping en epoch {epoch}.")
            break

    print(f"\nEntrenamiento finalizado. Mejor val_loss = {best_val_loss:.6f}")
    print(f"Pesos guardados en '{args.model_out}'")


if __name__ == "__main__":
    main()
