"""
run03_minority_generator.py
Generates run03_minority dataset: Q-transform windows centred on Gravity Spy O3
peak times, filtered to minority classes only (excludes Low_Frequency_Burst and
Scattered_Light which are already over-represented in run03).

Output: run03_minority/{IFO}/{EPOCH}/scale_1p0s/dataset_{IFO}_{EPOCH}_scale_1p0s_run03_minority.npz

Combinations: H1/O3a, H1/O3b, L1/O3a, L1/O3b  (300 windows each, ~1200 total)
Skips any combination whose NPZ already exists.

Usage:
    python run03_minority_generator.py

Dependencies:
    pip install gwpy gwosc scipy numpy
"""

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_DIR      = PROJECT_ROOT / "gravityspy_o3"
HDF5_CACHE   = PROJECT_ROOT / "hdf5_cache"
OUT_ROOT     = PROJECT_ROOT / "run03_minority"

MAX_SAMPLES  = 300
MIN_SNR      = 7.5
SEED         = 42
SCALE        = 1.0
Q_HALF       = 8.5
Q_OUT_HALF   = 0.5
TARGET_SHAPE = (128, 128)
FRANGE       = (20, 1700)
QRANGE       = (4, 64)
CHECKPOINT_EVERY = 50
DOWNLOAD_TIMEOUT = 120
LOG_EVERY    = 5

GWOSC_DATASETS = {"O3a": "O3a", "O3b": "O3b"}

RUNS = [
    ("H1", "O3a"),
    ("H1", "O3b"),
    ("L1", "O3a"),
    ("L1", "O3b"),
]

CSV_FILES = {
    ("H1", "O3a"): "H1_O3a.csv",
    ("H1", "O3b"): "H1_O3b.csv",
    ("L1", "O3a"): "L1_O3a.csv",
    ("L1", "O3b"): "L1_O3b.csv",
}

# Classes to exclude (already over-represented in run03)
EXCLUDE_LABELS = {"Low_Frequency_Burst", "Scattered_Light"}

# ---------------------------------------------------------------------------
# Helpers  (identical to run03_bulk_generator.py)
# ---------------------------------------------------------------------------

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_checkpoint(path: Path) -> set:
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return set(data.get("done", []))
    return set()


def save_checkpoint(path: Path, done: set):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"done": sorted(done)}, f)


def read_csv_triggers(csv_path: Path, ifo: str):
    """Return list of dicts with peak_time, snr, label — minority classes only."""
    triggers = []
    excluded = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                peak_time = float(row["peak_time"])
                if "peak_time_ns" in row and row["peak_time_ns"].strip():
                    peak_time += float(row["peak_time_ns"]) / 1e9
                snr   = float(row["snr"])
                label = row["ml_label"].strip()
            except (KeyError, ValueError):
                continue
            if snr < MIN_SNR:
                continue
            if label in EXCLUDE_LABELS:
                excluded += 1
                continue
            triggers.append({"peak_time": peak_time, "snr": snr, "label": label})
    print(f"  {excluded} triggers excluded (Low_Frequency_Burst / Scattered_Light)")
    return triggers


def group_by_block(triggers):
    blocks = {}
    for t in triggers:
        block_gps = int(t["peak_time"] // 4096) * 4096
        blocks.setdefault(block_gps, []).append(t)
    return blocks


def get_hdf5_url(ifo: str, block_gps: int, dataset: str):
    try:
        from gwosc.locate import get_urls
        urls = get_urls(ifo, block_gps, block_gps + 4096, dataset=dataset)
        return urls[0] if urls else None
    except Exception as e:
        print(f"    [WARN] gwosc.locate failed for {ifo} {block_gps}: {e}")
        return None


def download_hdf5(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        print(f"    Downloading {url} ...")
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as resp:  # noqa: S310
            with open(dest, "wb") as fout:
                fout.write(resp.read())
        return True
    except Exception as e:
        print(f"    [WARN] Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def read_strain(hdf5_path: Path):
    from gwpy.timeseries import TimeSeries
    return TimeSeries.read(str(hdf5_path), format="hdf5.gwosc")


def process_trigger(strain, peak_time: float):
    """
    Crop -> Q-transform (whiten=True) -> validate -> resize.
    Returns (float32 (128,128) raw spectrogram, log_energy float) or (None, None).
    No per-window normalisation applied.
    """
    try:
        t_start = strain.t0.value
        t_end   = t_start + strain.duration.value

        if peak_time - Q_HALF < t_start or peak_time + Q_HALF > t_end:
            return None, None

        seg = strain.crop(peak_time - Q_HALF, peak_time + Q_HALF)
        if np.any(np.isnan(seg.value)):
            return None, None

        outseg = (peak_time - Q_OUT_HALF, peak_time + Q_OUT_HALF)
        qtrans = seg.q_transform(
            qrange=QRANGE,
            frange=FRANGE,
            outseg=outseg,
            whiten=True,
        )

        spec = qtrans.value.astype(np.float64)
        if np.any(np.isnan(spec)) or np.any(np.isinf(spec)):
            return None, None

        log_e = float(np.log1p(np.percentile(spec, 90)))

        if spec.shape != TARGET_SHAPE:
            zoom_factors = (TARGET_SHAPE[0] / spec.shape[0],
                            TARGET_SHAPE[1] / spec.shape[1])
            spec = zoom(spec, zoom_factors, order=1)

        return spec.astype(np.float32), log_e

    except Exception as e:
        print(f"      [WARN] process_trigger failed: {e}")
        return None, None


def save_npz(out_path: Path, X, t0_arr, log_e_arr, labels, peak_times, snrs, meta):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        X=np.array(X, dtype=np.float32),
        t0=np.array(t0_arr, dtype=np.float64),
        log_energy=np.array(log_e_arr, dtype=np.float32),
        labels=np.array(labels, dtype=object),
        peak_time=np.array(peak_times, dtype=np.float64),
        snr=np.array(snrs, dtype=np.float32),
        meta_json=json.dumps(meta),
    )


def eta_str(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "?"
    secs = elapsed / done * (total - done)
    return f"{secs/60:.1f}min"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_ifo_epoch(ifo: str, epoch: str):
    csv_file = CSV_DIR / CSV_FILES[(ifo, epoch)]
    if not csv_file.exists():
        print(f"\n[ERROR] CSV not found: {csv_file}")
        return

    dataset_name = GWOSC_DATASETS[epoch]
    scale_tag    = f"scale_{str(SCALE).replace('.', 'p')}s"
    out_dir      = OUT_ROOT / ifo / epoch / scale_tag
    out_path     = out_dir / f"dataset_{ifo}_{epoch}_{scale_tag}_run03_minority.npz"
    ckpt_path    = OUT_ROOT / f"checkpoint_{ifo}_{epoch}.json"

    ensure_dirs(HDF5_CACHE, out_dir)

    if out_path.exists():
        print(f"\n[SKIP] {ifo}/{epoch}: NPZ already exists at {out_path}")
        return

    print(f"\n{'='*60}")
    print(f"  IFO={ifo}  EPOCH={epoch}  max={MAX_SAMPLES}  snr>={MIN_SNR}")
    print(f"  Excluding: {sorted(EXCLUDE_LABELS)}")
    print(f"{'='*60}")

    # Load and filter triggers
    print(f"Loading triggers from {csv_file.name} ...")
    triggers = read_csv_triggers(csv_file, ifo)
    print(f"  {len(triggers)} minority triggers after SNR + label filter")

    if not triggers:
        print("  [WARN] No triggers remain after filtering — skipping.")
        return

    # Class summary before sampling
    from collections import Counter
    label_counts = Counter(t["label"] for t in triggers)
    print("  Available classes:")
    for cls, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:30s}: {cnt}")

    # Shuffle reproducibly
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(triggers))
    triggers = [triggers[i] for i in idx]

    # Resume from checkpoint
    done_set = load_checkpoint(ckpt_path)
    triggers = [t for t in triggers if t["peak_time"] not in done_set]
    print(f"  {len(triggers)} triggers remaining after checkpoint resume")

    blocks = group_by_block(triggers)
    block_keys = sorted(blocks.keys())
    total_blocks = len(block_keys)
    print(f"  {total_blocks} HDF5 blocks to process")

    X_all, t0_all, loge_all, labels_all, pt_all, snr_all = [], [], [], [], [], []

    t_start_run   = time.time()
    blocks_done   = 0
    blocks_failed = 0

    for bi, block_gps in enumerate(block_keys):
        if len(X_all) >= MAX_SAMPLES:
            break

        block_triggers = blocks[block_gps]
        hdf5_name = f"{ifo}-{block_gps}-4096.hdf5"
        hdf5_path = HDF5_CACHE / hdf5_name

        if bi % LOG_EVERY == 0:
            elapsed = time.time() - t_start_run
            print(f"  Block {bi+1}/{total_blocks} | "
                  f"windows={len(X_all)}/{MAX_SAMPLES} | "
                  f"ok={blocks_done} fail={blocks_failed} | "
                  f"elapsed={elapsed/60:.1f}min "
                  f"ETA={eta_str(elapsed, bi+1, total_blocks)}")

        url = get_hdf5_url(ifo, block_gps, dataset_name)
        if url is None:
            blocks_failed += 1
            continue

        if not download_hdf5(url, hdf5_path):
            blocks_failed += 1
            continue

        try:
            strain = read_strain(hdf5_path)
        except Exception as e:
            print(f"    [WARN] Cannot read strain {hdf5_path.name}: {e}")
            blocks_failed += 1
            hdf5_path.unlink(missing_ok=True)
            continue

        for t in block_triggers:
            if len(X_all) >= MAX_SAMPLES:
                break
            pt = t["peak_time"]
            spec, log_e = process_trigger(strain, pt)
            if spec is None:
                continue
            X_all.append(spec)
            t0_all.append(pt)
            loge_all.append(log_e)
            labels_all.append(t["label"])
            pt_all.append(pt)
            snr_all.append(t["snr"])
            done_set.add(pt)

        if len(X_all) % CHECKPOINT_EVERY < len(block_triggers):
            save_checkpoint(ckpt_path, done_set)

        hdf5_path.unlink(missing_ok=True)
        blocks_done += 1

    # Final save
    if not X_all:
        print("\n[WARN] No windows collected — check CSV paths and GWOSC connectivity.")
        return

    meta = {
        "run":          "run03_minority",
        "ifo":          ifo,
        "epoch":        epoch,
        "scale":        SCALE,
        "max_samples":  MAX_SAMPLES,
        "min_snr":      MIN_SNR,
        "seed":         SEED,
        "q_half":       Q_HALF,
        "q_out_half":   Q_OUT_HALF,
        "frange":       FRANGE,
        "qrange":       QRANGE,
        "target_shape": TARGET_SHAPE,
        "normalization": "none",
        "exclude_labels": sorted(EXCLUDE_LABELS),
        "n_windows":    len(X_all),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    save_npz(out_path, X_all, t0_all, loge_all, labels_all, pt_all, snr_all, meta)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved {len(X_all)} windows -> {out_path}  ({size_mb:.1f} MB)")

    dist = Counter(labels_all)
    print("Class distribution:")
    for cls, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {cls:30s}: {cnt}")

    save_checkpoint(ckpt_path, done_set)
    elapsed = time.time() - t_start_run
    print(f"\nDone in {elapsed/60:.1f} min  (blocks ok={blocks_done} fail={blocks_failed})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def check_dependencies():
    missing = []
    for pkg in ("gwpy", "gwosc", "scipy", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"  Run: pip install {' '.join(missing)}")
        sys.exit(1)


if __name__ == "__main__":
    check_dependencies()

    if not CSV_DIR.exists() or not any(CSV_DIR.glob("*.csv")):
        print(f"[ERROR] No CSVs found in {CSV_DIR}/")
        sys.exit(1)

    for ifo, epoch in RUNS:
        run_ifo_epoch(ifo, epoch)
