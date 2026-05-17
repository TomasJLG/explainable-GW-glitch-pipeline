#!/usr/bin/env python3
"""
diagnose_gps_overlap.py
=======================
Determines whether run02 NPZ windows temporally overlap with
Gravity Spy O3 trigger peak_times (Zenodo 5649212).

Hypothesis: random sampling in run02 produces statistically rare
coincidences with known glitch triggers.

Usage:
    python diagnose_gps_overlap.py

Requires: numpy (stdlib csv, json, math, pathlib used otherwise).
No plots. Writes only to stdout. Target runtime < 30 s.
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

# Force UTF-8 output so Unicode in f-strings works regardless of the
# Windows console codepage (cp1252 would raise UnicodeEncodeError otherwise).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Configuration ──────────────────────────────────────────────────────────────

NPZ_BASE = Path(
    "C:/Users/tlegu/Desktop/M1- Detector_de_anomalias"
    "/Dataset(v2)/qdataset_npz_v2/run02"
)
CSV_DIR = Path(
    "C:/Users/tlegu/Desktop/M1- Detector_de_anomalias"
    "/labeling/gravityspy_o3"
)

IFOS   = ["H1", "L1"]
EPOCHS = ["O3a", "O3b"]
SCALES = ["0p25", "1p0", "4p0"]


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _gps_from_row(row: dict) -> float | None:
    """
    Extract full-precision GPS time from a Gravity Spy CSV row.

    Priority:
      1. peak_time (int) + peak_time_ns (int) / 1e9  — most precise
      2. event_time (float)                           — already fractional
      3. peak_time (float alone)                      — integer-second fallback
    """
    pt  = row.get("peak_time",    "").strip()
    ns  = row.get("peak_time_ns", "").strip()
    et  = row.get("event_time",   "").strip()

    if pt and ns:
        try:
            return float(pt) + float(ns) * 1e-9
        except ValueError:
            pass
    if et:
        try:
            return float(et)
        except ValueError:
            pass
    if pt:
        try:
            return float(pt)
        except ValueError:
            pass
    return None


def load_peak_times(csv_path: Path, ifo: str):
    """
    Load trigger GPS times from a Gravity Spy CSV, filtered to `ifo`.

    Returns
    -------
    pts        : np.ndarray float64  — GPS times for the requested IFO
    total_rows : int                 — total data rows in the file
    kept       : int                 — rows kept after ifo filter
    """
    pts        = []
    total_rows = 0
    kept       = 0

    with open(str(csv_path), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Detect ifo column (try several spellings)
        ifo_col = next(
            (c for c in ["ifo", "IFO", "detector", "channel"] if c in fieldnames),
            None,
        )

        for row in reader:
            total_rows += 1

            # ifo filter — skip rows from a different detector
            if ifo_col is not None:
                row_ifo = row[ifo_col].strip()
                if row_ifo and row_ifo != ifo:
                    continue

            gps = _gps_from_row(row)
            if gps is not None:
                pts.append(gps)
                kept += 1

    return np.array(pts, dtype=np.float64), total_rows, kept


# ── NPZ helper ─────────────────────────────────────────────────────────────────

def load_npz_t0_dur(path: Path):
    """
    Load t0 array and window duration from a v2 NPZ dataset.

    Returns
    -------
    t0  : np.ndarray float64  — GPS start times of each window
    dur : float               — window duration in seconds (from meta_json)
    meta: dict
    """
    data = np.load(str(path), allow_pickle=False)
    t0   = data["t0"].astype(np.float64)
    meta = json.loads(str(data["meta_json"]))

    # 'dur' is the field used in v2 meta; fall back to other spellings
    dur = float(
        meta.get("dur",
        meta.get("window_dur",
        meta.get("duration", 1.0)))
    )
    return t0, dur, meta


# ── Per-window match counting ──────────────────────────────────────────────────

def count_matches_vectorised(
    t0: np.ndarray,
    dur: float,
    pts_sorted: np.ndarray,
) -> np.ndarray:
    """
    For each window [t0[i], t0[i]+dur), count triggers in pts_sorted.
    Uses np.searchsorted → O(N·log M) — fast even for N=100k, M=100k.
    """
    lo = np.searchsorted(pts_sorted, t0,       side="left")
    hi = np.searchsorted(pts_sorted, t0 + dur, side="left")
    return (hi - lo).astype(np.int32)


# ── Main per-combination routine ────────────────────────────────────────────────

def analyze(ifo: str, epoch: str, scale: str) -> dict:
    """
    Run the full overlap analysis for one (ifo, epoch, scale) combination.
    Prints a verbose section and returns a dict row for the summary table.
    """
    tag = f"{ifo}_{epoch}_scale_{scale}"
    bar = "=" * 74
    print(f"\n{bar}")
    print(f"  {tag}")
    print(bar)

    npz_path = (
        NPZ_BASE / ifo / epoch / f"scale_{scale}"
        / f"dataset_{ifo}_{epoch}_scale_{scale}_run02.npz"
    )
    csv_path = CSV_DIR / f"{ifo}_{epoch}.csv"

    base = dict(ifo=ifo, epoch=epoch, scale=scale)

    # ── Load NPZ ─────────────────────────────────────────────────────────────
    if not npz_path.exists():
        print(f"  [SKIP] NPZ not found:\n         {npz_path}")
        return {**base,
                "n_windows": "--", "n_triggers_in_span": "--",
                "pct_with_match": "--", "mean_matches_per_window": "--",
                "expected_pct": "--", "ratio_obs_exp": "--",
                "note": "NPZ missing"}

    t0, dur, _meta = load_npz_t0_dur(npz_path)
    n_win  = len(t0)
    t0_min = float(t0.min())
    t0_max = float(t0.max())
    # Full temporal extent covered by the NPZ (last window ends at t0_max+dur)
    span   = (t0_max - t0_min) + dur

    print(f"  NPZ : {n_win:,} windows")
    print(f"        t0 in [{t0_min:.3f}, {t0_max:.3f}] GPS")
    print(f"        span_total = {span:,.1f} s  |  dur/window = {dur} s")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    if not csv_path.exists():
        print(f"  [WARN] CSV not found — skipping overlap analysis:\n"
              f"         {csv_path}")
        return {**base,
                "n_windows": n_win, "n_triggers_in_span": "--",
                "pct_with_match": "--", "mean_matches_per_window": "--",
                "expected_pct": "--", "ratio_obs_exp": "--",
                "note": "CSV missing"}

    pts, total_rows, kept = load_peak_times(csv_path, ifo)

    if len(pts) == 0:
        print(f"  [WARN] 0 usable peak_times after ifo={ifo} filter.")
        return {**base,
                "n_windows": n_win, "n_triggers_in_span": 0,
                "pct_with_match": 0.0, "mean_matches_per_window": 0.0,
                "expected_pct": 0.0, "ratio_obs_exp": "N/A",
                "note": "0 triggers"}

    # Triggers that fall within the full NPZ span [t0_min, t0_max+dur)
    mask_in_span    = (pts >= t0_min) & (pts < t0_max + dur)
    pts_in_span     = pts[mask_in_span]
    n_trig_span     = len(pts_in_span)

    print(f"  CSV : {total_rows:,} rows total → {kept:,} triggers for {ifo}")
    print(f"        peak_time in [{pts.min():.3f}, {pts.max():.3f}] GPS")
    print(f"        {n_trig_span:,} triggers lie within NPZ span")

    # ── Per-window match counts ───────────────────────────────────────────────
    pts_sorted = np.sort(pts_in_span)
    counts     = count_matches_vectorised(t0, dur, pts_sorted)

    n0   = int(np.sum(counts == 0))
    n1   = int(np.sum(counts == 1))
    n2   = int(np.sum(counts == 2))
    n3p  = int(np.sum(counts >= 3))
    n_ge1     = n_win - n0
    pct_match = 100.0 * n_ge1 / n_win
    mean_m    = float(counts.mean())

    print(f"\n  Match distribution (one trigger peak_time inside [t0, t0+dur)):")
    print(f"    0 matches : {n0:>9,}  ({100.0*n0/n_win:6.2f}%)")
    print(f"    1 match   : {n1:>9,}  ({100.0*n1/n_win:6.2f}%)")
    print(f"    2 matches : {n2:>9,}  ({100.0*n2/n_win:6.2f}%)")
    print(f"    3+ matches: {n3p:>9,}  ({100.0*n3p/n_win:6.2f}%)")
    print(f"\n  Windows with >=1 match : {n_ge1:,} / {n_win:,}  ({pct_match:.4f}%)")
    print(f"  Mean matches / window  : {mean_m:.6f}")

    # ── Expected vs observed (Poisson model) ─────────────────────────────────
    # Under a uniform-random trigger distribution with rate ρ = n_trig_span/span,
    # the expected number of triggers per window is λ = ρ·dur.
    # P(≥1 match) = 1 − e^(−λ).
    lam     = (n_trig_span * dur) / span if span > 0 else 0.0
    exp_pct = 100.0 * (1.0 - math.exp(-lam)) if lam > 0 else 0.0
    ratio   = pct_match / exp_pct if exp_pct > 0 else float("nan")

    print(f"\n  Poisson lam (expected matches/window under H0) : {lam:.6f}")
    print(f"  Expected % windows with >=1 match  (Poisson)  : {exp_pct:.4f} %")
    print(f"  Observed % windows with >=1 match              : {pct_match:.4f} %")
    if not math.isnan(ratio):
        print(f"  Ratio  observed / expected                     : {ratio:.4f}x")
    else:
        print(f"  Ratio  observed / expected                     : N/A  (lam ~ 0)")

    return {
        **base,
        "n_windows":               n_win,
        "n_triggers_in_span":      n_trig_span,
        "pct_with_match":          round(pct_match, 4),
        "mean_matches_per_window": round(mean_m, 6),
        "expected_pct":            round(exp_pct, 4),
        "ratio_obs_exp":           round(ratio, 4) if not math.isnan(ratio) else "N/A",
        "note":                    "",
    }


# ── Summary table ──────────────────────────────────────────────────────────────

_COLS = [
    ("ifo",                    4),
    ("epoch",                  5),
    ("scale",                  6),
    ("n_windows",             10),
    ("n_triggers_in_span",    18),
    ("pct_with_match",        14),
    ("mean_matches_per_window", 24),
    ("expected_pct",          13),
    ("ratio_obs_exp",         13),
]


def _fmt(val, width: int) -> str:
    return f"{str(val):<{width}}"


def print_summary(rows: list[dict]) -> None:
    header = "  ".join(_fmt(h, w) for h, w in _COLS)
    div    = "-" * len(header)

    print(f"\n\n{'=' * len(header)}")
    print("SUMMARY TABLE  (12 combinations)")
    print(f"{'=' * len(header)}")
    print(header)
    print(div)

    for r in rows:
        line = "  ".join(_fmt(r.get(h, "--"), w) for h, w in _COLS)
        if r.get("note"):
            line += f"  [{r['note']}]"
        print(line)

    print(f"{'=' * len(header)}")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    results = []
    for ifo in IFOS:
        for epoch in EPOCHS:
            for scale in SCALES:
                results.append(analyze(ifo, epoch, scale))
    print_summary(results)


if __name__ == "__main__":
    main()
