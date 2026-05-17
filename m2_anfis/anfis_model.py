"""
anfis_model.py
Takagi-Sugeno ANFIS with 5 layers and a linear vector consequent.

Layer 1: GBellMF per-feature membership degrees  → (B, R, F)
Layer 2: Product T-norm firing strength           → (B, R)
Layer 3: Normalised firing strengths              → (B, R)
Layer 4: Consequent linear output per rule        → (B, R, C)
Layer 5: Weighted sum aggregation                 → (B, C)

C = n_classes for multi-class; C = 1 for binary (then squeeze + sigmoid).
"""

import torch
import torch.nn as nn

from m2_anfis.membership import GBellMF


class ANFIS(nn.Module):
    """
    Args:
        n_features : number of input features (F)
        n_rules    : number of fuzzy rules (R)
        n_classes  : output dimension (C); use 1 for binary, 5 for 5-class
    """

    def __init__(self, n_features: int, n_rules: int, n_classes: int):
        super().__init__()
        self.n_features = n_features
        self.n_rules    = n_rules
        self.n_classes  = n_classes

        # Layer 1+2: fuzzification + product T-norm
        self.mf = GBellMF(n_rules, n_features)

        # Layer 4: linear consequent — (F+1) terms per rule per output
        # consequent[r] : (F+1, C), last term is bias
        self.consequent = nn.Parameter(
            torch.zeros(n_rules, n_features + 1, n_classes)
        )
        nn.init.xavier_uniform_(self.consequent.view(n_rules, -1).unsqueeze(0)
                                .squeeze(0))

    # ------------------------------------------------------------------
    def get_firing_strengths(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw (unnormalised) firing strengths. Shape: (B, R)."""
        return self.mf(x)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, F) — already normalised to [0, 1]
        Returns logits : (B, C)
        """
        B = x.size(0)

        # Layer 2: (B, R)
        w = self.mf(x)

        # Layer 3: normalise
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-12)
        w_bar = w / w_sum                           # (B, R)

        # Layer 4: linear consequent per rule
        # x_aug: (B, F+1) — append bias column of ones
        x_aug = torch.cat([x, torch.ones(B, 1, device=x.device)], dim=1)  # (B, F+1)

        # consequent: (R, F+1, C)
        # rule_out[b, r, c] = sum_f x_aug[b,f] * consequent[r,f,c]
        rule_out = torch.einsum("bf,rfc->brc", x_aug, self.consequent)    # (B, R, C)

        # Layer 5: weighted sum
        output = (w_bar.unsqueeze(-1) * rule_out).sum(dim=1)              # (B, C)
        return output

    # ------------------------------------------------------------------
    def init_from_centers(self, centers: torch.Tensor, spread: float = 0.3):
        """Delegate center initialisation to the MF module."""
        self.mf.init_from_centers(centers, spread=spread)
