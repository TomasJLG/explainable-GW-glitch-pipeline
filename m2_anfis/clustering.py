"""
clustering.py
Subtractive clustering (Chiu 1994) for ANFIS rule initialisation.
Also provides grid_centers() for the lightweight binary POC.
"""

import itertools
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Grid centers (binary POC — 3 features, 3 levels each → 27 rules)
# ---------------------------------------------------------------------------

def grid_centers(n_features: int, n_levels: int = 3) -> torch.Tensor:
    """
    Return (n_levels^n_features, n_features) grid of equally-spaced centers
    in [0, 1]^n_features.
    """
    levels = np.linspace(0.0, 1.0, n_levels)
    combos = list(itertools.product(levels, repeat=n_features))
    return torch.tensor(combos, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Subtractive clustering (Chiu 1994)
# ---------------------------------------------------------------------------

def subtractive_clustering(
    data: np.ndarray,
    ra: float = 0.5,
    rb_factor: float = 1.5,
    accept_ratio: float = 0.5,
    reject_ratio: float = 0.15,
) -> np.ndarray:
    """
    Chiu's subtractive clustering algorithm.

    Args:
        data         : (N, F) float array, should be normalised to [0, 1]
        ra           : neighbourhood radius for potential calculation
        rb_factor    : rb = ra * rb_factor (mountain reduction radius)
        accept_ratio : candidate accepted if potential >= accept_ratio * p_max
        reject_ratio : candidate rejected if potential < reject_ratio * p_max

    Returns:
        centers : (K, F) array of cluster centres, K determined automatically
    """
    data = np.asarray(data, dtype=np.float64)
    N, F = data.shape
    rb   = ra * rb_factor

    # Initial potential for each data point
    sq_ra = (ra / 2.0) ** 2  # variance proxy
    sq_rb = (rb / 2.0) ** 2

    def compute_potentials(pts: np.ndarray, centres: list, p_arr: np.ndarray,
                           prev_center: np.ndarray, prev_potential: float,
                           sq_r: float) -> np.ndarray:
        """Reduce potentials based on a newly accepted center."""
        dist_sq = np.sum((pts - prev_center) ** 2, axis=1)
        p_arr = p_arr - prev_potential * np.exp(-dist_sq / sq_r)
        return p_arr

    # Compute initial potentials
    potentials = np.zeros(N)
    for i in range(N):
        dist_sq = np.sum((data - data[i]) ** 2, axis=1)
        potentials[i] = np.sum(np.exp(-dist_sq / sq_ra))

    p_max_init = potentials.max()
    p_max      = p_max_init

    centers = []
    iteration = 0
    max_iter = N  # safety ceiling

    while iteration < max_iter:
        best_idx = int(np.argmax(potentials))
        best_p   = potentials[best_idx]

        # Accept / reject / stop
        if best_p >= accept_ratio * p_max_init:
            centers.append(data[best_idx].copy())
        elif best_p < reject_ratio * p_max_init:
            break
        else:
            # Squash test: keep if it + nearest center exceed threshold
            center_arr = np.array(centers)
            d_min = np.min(np.linalg.norm(center_arr - data[best_idx], axis=1))
            if (d_min / ra) + (best_p / p_max_init) >= 1.0:
                centers.append(data[best_idx].copy())
            else:
                potentials[best_idx] = 0.0
                continue

        # Reduce potentials around the new center
        potentials = compute_potentials(
            data, centers, potentials, centers[-1], best_p, sq_rb
        )
        iteration += 1

        if len(centers) == 1:
            p_max = p_max_init

    if not centers:
        # Fallback: single center at data mean
        centers = [data.mean(axis=0)]

    return np.array(centers, dtype=np.float32)
