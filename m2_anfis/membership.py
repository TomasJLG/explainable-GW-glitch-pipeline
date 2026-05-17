"""
membership.py
Generalized Bell MF (GBellMF) as an nn.Module.

μ(x; a, b, c) = 1 / (1 + |(x - c) / a|^(2b))

Parameters shape: (n_rules, n_features) for a_raw, b_raw, c.
a and b are constrained positive via softplus + epsilon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GBellMF(nn.Module):
    """
    Computes per-rule firing strengths for a batch of feature vectors.

    Args:
        n_rules    : number of fuzzy rules
        n_features : number of input features

    Forward:
        x : (B, F) float tensor, features already in [0, 1]
    Returns:
        firing : (B, R) product T-norm firing strengths in (0, 1]
    """

    def __init__(self, n_rules: int, n_features: int):
        super().__init__()
        self.n_rules    = n_rules
        self.n_features = n_features

        # Raw (unconstrained) parameters; constrained in forward
        self.a_raw = nn.Parameter(torch.ones(n_rules, n_features))
        self.b_raw = nn.Parameter(torch.ones(n_rules, n_features))
        self.c     = nn.Parameter(torch.zeros(n_rules, n_features))

    def get_params(self):
        """Return constrained a (>0) and b (>0)."""
        a = F.softplus(self.a_raw) + 1e-6
        b = F.softplus(self.b_raw) + 1e-6
        return a, b, self.c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, F)
        Returns firing strengths: (B, R)
        """
        a, b, c = self.get_params()
        # x: (B, F) → (B, 1, F) broadcast with (R, F)
        x_exp = x.unsqueeze(1)            # (B, 1, F)
        ratio  = (x_exp - c) / a          # (B, R, F)
        mu     = 1.0 / (1.0 + torch.abs(ratio) ** (2.0 * b))  # (B, R, F)
        firing = mu.prod(dim=-1)          # (B, R) — product T-norm
        return firing

    def init_from_centers(self, centers: torch.Tensor, spread: float = 0.3):
        """
        Initialise c from cluster/grid centers, a to `spread`, b to 2.
        centers : (R, F)
        """
        with torch.no_grad():
            self.c.copy_(centers)
            # softplus^{-1}(spread - eps) ≈ log(exp(spread) - 1)
            val_a = torch.log(torch.exp(torch.tensor(spread - 1e-6)) - 1.0)
            self.a_raw.fill_(float(val_a))
            val_b = torch.log(torch.exp(torch.tensor(2.0 - 1e-6)) - 1.0)
            self.b_raw.fill_(float(val_b))
