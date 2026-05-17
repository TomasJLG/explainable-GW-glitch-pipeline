"""
m1_autoencoder.py
=================
Módulo M1 — Autoencoder convolucional para detección de anomalías en
imágenes Q-transform del strain gravitacional.

Proyecto: Pipeline explicable para detección de Glitches en G.W.
Spec:     M1 Autoencoder Spec v1.0
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class GlitchAE(nn.Module):
    """
    Autoencoder convolucional simétrico para detección de glitches.

    Arquitectura:
        Entrada  : (N, 1, 128, 128)
        Encoder  : 4 bloques Conv2d con stride=2  → (N, 256, 8, 8)
        Cuello   : Flatten + 2 capas FC           → (N, latent_dim)
        Expansión: 2 capas FC + Unflatten         → (N, 256, 8, 8)
        Decoder  : 4 bloques ConvTranspose2d      → (N, 1, 128, 128)
        Salida   : Sigmoid  → valores en [0, 1]

    Parámetros
    ----------
    latent_dim : int
        Dimensión del espacio latente.
        - Empezar con 128.
        - Reducir a 64 si el modelo reconstruye demasiado bien (no detecta).
        - Aumentar a 256 si no converge.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # ------------------------------------------------------------------
        # ENCODER
        # ------------------------------------------------------------------
        self.encoder_conv = nn.Sequential(
            # Bloque 1: (N, 1, 128, 128) → (N, 32, 64, 64)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Bloque 2: (N, 32, 64, 64) → (N, 64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Bloque 3: (N, 64, 32, 32) → (N, 128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Bloque 4: (N, 128, 16, 16) → (N, 256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),                              # (N, 256*8*8) = (N, 16384)
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),                # (N, latent_dim)
        )

        # ------------------------------------------------------------------
        # DECODER
        # ------------------------------------------------------------------
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256 * 8 * 8),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 8, 8)),              # (N, 256, 8, 8)
        )

        self.decoder_conv = nn.Sequential(
            # Bloque 1: (N, 256, 8, 8) → (N, 128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Bloque 2: (N, 128, 16, 16) → (N, 64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Bloque 3: (N, 64, 32, 32) → (N, 32, 64, 64)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Bloque 4: (N, 32, 64, 64) → (N, 1, 128, 128)
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # salida en [0,1] para coincidir con X normalizado
        )

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Paso completo: encode → cuello → decode."""
        z = self.encoder_conv(x)
        z = self.encoder_fc(z)
        x_hat = self.decoder_fc(z)
        x_hat = self.decoder_conv(x_hat)
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Solo el encoder. Útil para extraer representaciones latentes."""
        z = self.encoder_conv(x)
        return self.encoder_fc(z)


# ---------------------------------------------------------------------------
# Score de anomalía
# ---------------------------------------------------------------------------

def compute_anomaly_scores(
    model: GlitchAE,
    X_tensor: torch.Tensor,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Calcula el error de reconstrucción (MSE por píxel) para cada ventana.

    Parámetros
    ----------
    model     : GlitchAE entrenado.
    X_tensor  : Tensor (N, 1, 128, 128) normalizado.
    device    : 'cuda' o 'cpu'.
    batch_size: Tamaño de lote para inferencia.

    Retorna
    -------
    scores : np.ndarray (N,)
        Error de reconstrucción medio por ventana.
        Valores altos → ventana anómala → posible glitch.
    """
    model.eval()
    scores = []
    dl = DataLoader(TensorDataset(X_tensor), batch_size=batch_size)

    with torch.no_grad():
        for (x,) in dl:
            x = x.to(device)
            x_hat = model(x)
            # MSE por imagen (media sobre canales, alto y ancho; NO sobre batch)
            mse = ((x - x_hat) ** 2).mean(dim=[1, 2, 3])
            scores.append(mse.cpu().numpy())

    return np.concatenate(scores)


def compute_combined_score(
    ae_scores: np.ndarray,
    log_energy: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Combina el score del AE con el log_energy del dataset.

    Parámetros
    ----------
    ae_scores  : (N,) scores de reconstrucción del autoencoder.
    log_energy : (N,) log1p(P90 del Q-transform) del dataset.
    alpha      : Peso del AE en el score combinado.
                 alpha=1 → solo AE,  alpha=0 → solo energía.
                 Empezar con 0.7 y ajustar según curvas ROC.

    Retorna
    -------
    combined : np.ndarray (N,) en [0, 1].
    """
    ae_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-8)
    en_norm = (log_energy - log_energy.min()) / (log_energy.max() - log_energy.min() + 1e-8)
    return alpha * ae_norm + (1 - alpha) * en_norm
