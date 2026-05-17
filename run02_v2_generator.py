"""
run02_v2_generator.py
Generates run02v2 dataset: Q-transform windows at uniformly random GPS times
within valid GWOSC science segments. No Gravity Spy CSV required.

The output spectrograms are stored raw (no per-window normalisation), matching
the convention of run03 v2 so that both can be normalised with the same global
p1/p99 at training/evaluation time.

Usage:
    python run02_v2_generator.py

Dependencies:
    pip install gwpy gwosc scipy numpy
"""

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
HDF5_CACHE   = PROJECT_ROOT / "hdf5_cache"
OUT_ROOT     = PROJECT_ROOT / "run02v2"

MAX_SAMPLES          = 2000
SCALE                = 1.0       # output window duration (seconds)
MARGIN               = 8.0       # context padding each side for Q-transform
Q_OUT_HALF           = SCALE / 2  # = 0.5s
IMG_SIZE             = 128
TARGET_SHAPE         = (IMG_SIZE, IMG_SIZE)
FS                   = 4096
FLOW                 = 20.0
FHIGH                = 1700.0
QRANGE               = (4, 64)
FRANGE               = (FLOW, FHIGH)
SEED                 = 42
N_CANDIDATES_FACTOR  = 3  # generate MAX_SAMPLES × this many initial candidates
CHECKPOINT_EVERY     = 50
DOWNLOAD_TIMEOUT     = 120
LOG_EVERY            = 5

# O3 GPS boundaries (GPS seconds)
GPS_RANGES = {
    "O3a": (1238166018, 1253977218),
    "O3b": (1256655618, 1269363618),
}

GWOSC_DATASETS = {"O3a": "O3a", "O3b": "O3b"}

IFOS   = ["H1", "L1"]
EPOCHS = ["O3a", "O3b"]
RUNS   = [(ifo, epoch) for ifo in IFOS for epoch in EPOCHS]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_science_segments(ifo: str, epoch: str):
    """Return list of (start, end) GPS tuples from the GWOSC timeline."""
    from gwosc.timeline import get_segments
    flag = f"{ifo}_DATA"
    gps_start, gps_end = GPS_RANGES[epoch]
    print(f"  Querying science segments: {flag}  {gps_start}–{gps_end} ...")
    segs = get_segments(flag, gps_start, gps_end)
    min_dur = 2 * MARGIN + 1.0 / FS
    valid = [(s, e) for s, e in segs if (e - s) >= min_dur]
    total_dur = sum(e - s for s, e in valid)
    print(f"  {len(valid)} valid segments  ({total_dur/3600:.1f} h of science data)")
    return valid


def sample_uniform_gps(segments, rng, n: int):
    """
    Sample n GPS times uniformly within segments.
    Each segment is weighted by its effective duration (length − 2×MARGIN).
    Returned list is sorted.
    """
    eff_durs = np.array([e - s - 2 * MARGIN for s, e in segments])
    probs    = eff_durs / eff_durs.sum()
    seg_idx  = rng.choice(len(segments), size=n, p=probs, replace=True)
    gps_list = []
    for i in seg_idx:
        s, e = segments[i]
        gps_list.append(float(rng.uniform(s + MARGIN, e - MARGIN)))
    return sorted(gps_list)


def group_by_block(gps_list):
    """Group GPS times into 4096-s HDF5 blocks."""
    blocks = {}
    for gps in gps_list:
        block_gps = int(gps // 4096) * 4096
        blocks.setdefault(block_gps, []).append(gps)
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


def process_window(strain, gps: float):
    """
    Crop ± MARGIN → Q-transform (whiten=True) → validate → resize.
    Returns (float32 (128,128) raw spectrogram, log_energy float) or (None, None).
    No per-window normalisation applied.
    """
    try:
        t_start = strain.t0.value
        t_end   = t_start + strain.duration.value

        if gps - MARGIN < t_start or gps + MARGIN > t_end:
            return None, None

        seg = strain.crop(gps - MARGIN, gps + MARGIN)
        if np.any(np.isnan(seg.value)):
            return None, None

        outseg = (gps - Q_OUT_HALF, gps + Q_OUT_HALF)
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
        print(f"      [WARN] process_window failed: {e}")
        return None, None


def save_npz(out_path: Path, X, t0_arr, loge_arr, meta):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        X=np.array(X, dtype=np.float32),
        t0=np.array(t0_arr, dtype=np.float64),
        log_energy=np.array(loge_arr, dtype=np.float32),
        meta_json=json.dumps(meta),
    )


def load_checkpoint(ckpt_path: Path):
    """Returns (done_blocks: set of int, X_all, t0_all, loge_all)."""
    if not ckpt_path.exists():
        return set(), [], [], []
    with open(ckpt_path) as f:
        data = json.load(f)
    done_blocks = set(data.get("done_blocks", []))
    n = data.get("n_windows", 0)

    # Restore accumulated data from partial NPZ if present
    partial_path = ckpt_path.with_suffix(".partial.npz")
    if partial_path.exists() and n > 0:
        try:
            npz = np.load(str(partial_path))
            X_all    = list(npz["X"])
            t0_all   = list(npz["t0"])
            loge_all = list(npz["log_energy"])
            print(f"  Resumed: {len(X_all)} windows from partial save.")
            return done_blocks, X_all, t0_all, loge_all
        except Exception as e:
            print(f"  [WARN] Could not load partial NPZ: {e}")
    return done_blocks, [], [], []


def save_checkpoint(ckpt_path: Path, done_blocks: set, X_all, t0_all, loge_all):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, "w") as f:
        json.dump({"done_blocks": sorted(done_blocks), "n_windows": len(X_all)}, f)
    # Persist accumulated data so a crash doesn't lose progress
    partial_path = ckpt_path.with_suffix(".partial.npz")
    if X_all:
        np.savez_compressed(
            str(partial_path),
            X=np.array(X_all, dtype=np.float32),
            t0=np.array(t0_all, dtype=np.float64),
            log_energy=np.array(loge_all, dtype=np.float32),
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
    dataset_name = GWOSC_DATASETS[epoch]
    scale_tag    = f"scale_{str(SCALE).replace('.', 'p')}s"
    out_dir      = OUT_ROOT / ifo / epoch / scale_tag
    out_path     = out_dir / f"dataset_{ifo}_{epoch}_{scale_tag}_run02v2.npz"
    ckpt_path    = OUT_ROOT / f"checkpoint_{ifo}_{epoch}.json"

    ensure_dirs(HDF5_CACHE, out_dir)

    if out_path.exists():
        print(f"\n[SKIP] {ifo}/{epoch}: NPZ already exists at {out_path}")
        return

    print(f"\n{'='*60}")
    print(f"  IFO={ifo}  EPOCH={epoch}  target={MAX_SAMPLES}  scale={SCALE}s")
    print(f"{'='*60}")

    # --- Science segments ---
    try:
        segments = get_science_segments(ifo, epoch)
    except Exception as e:
        print(f"  [ERROR] Cannot fetch science segments: {e}")
        return
    if not segments:
        print("  [ERROR] No valid science segments found.")
        return

    # --- Deterministic candidate GPS times ---
    rng = np.random.default_rng(SEED)
    n_candidates = MAX_SAMPLES * N_CANDIDATES_FACTOR
    candidates = sample_uniform_gps(segments, rng, n_candidates)
    print(f"  Sampled {n_candidates} candidate GPS times  (seed={SEED})")

    # --- Resume from checkpoint ---
    done_blocks, X_all, t0_all, loge_all = load_checkpoint(ckpt_path)
    print(f"  Checkpoint: {len(done_blocks)} blocks done, {len(X_all)} windows collected")

    # Group remaining candidates by block
    blocks = group_by_block(candidates)
    block_keys = sorted(
        (b for b in blocks if b not in done_blocks),
        key=lambda b: -len(blocks[b]),
    )
    total_blocks = len(block_keys)
    print(f"  {total_blocks} blocks remaining to process  ({len(blocks) - total_blocks} skipped by checkpoint)")

    t_start_run  = time.time()
    blocks_done  = 0
    blocks_failed = 0

    for bi, block_gps in enumerate(block_keys):
        if len(X_all) >= MAX_SAMPLES:
            break

        block_gps_list = blocks[block_gps]
        hdf5_name = f"{ifo}-{block_gps}-4096.hdf5"
        hdf5_path = HDF5_CACHE / hdf5_name

        # Progress log
        if bi % LOG_EVERY == 0:
            elapsed = time.time() - t_start_run
            print(f"  Block {bi+1}/{total_blocks} | "
                  f"windows={len(X_all)}/{MAX_SAMPLES} | "
                  f"ok={blocks_done} fail={blocks_failed} | "
                  f"elapsed={elapsed/60:.1f}min "
                  f"ETA={eta_str(elapsed, bi+1, total_blocks)}")

        # Get URL
        url = get_hdf5_url(ifo, block_gps, dataset_name)
        if url is None:
            blocks_failed += 1
            done_blocks.add(block_gps)
            continue

        # Download
        if not download_hdf5(url, hdf5_path):
            blocks_failed += 1
            done_blocks.add(block_gps)
            continue

        # Read strain
        try:
            strain = read_strain(hdf5_path)
        except Exception as e:
            print(f"    [WARN] Cannot read strain {hdf5_path.name}: {e}")
            blocks_failed += 1
            done_blocks.add(block_gps)
            hdf5_path.unlink(missing_ok=True)
            continue

        # Process all GPS candidates in this block
        prev_count = len(X_all)
        for gps in block_gps_list:
            if len(X_all) >= MAX_SAMPLES:
                break
            spec, log_e = process_window(strain, gps)
            if spec is None:
                continue
            X_all.append(spec)
            t0_all.append(gps)
            loge_all.append(log_e)

        # Checkpoint every N new windows
        new_count = len(X_all)
        if new_count // CHECKPOINT_EVERY > prev_count // CHECKPOINT_EVERY:
            save_checkpoint(ckpt_path, done_blocks, X_all, t0_all, loge_all)

        # Delete cached HDF5 to free disk
        hdf5_path.unlink(missing_ok=True)
        done_blocks.add(block_gps)
        blocks_done += 1

    # --- Final save ---
    if not X_all:
        print("\n[WARN] No windows collected — check GWOSC connectivity.")
        return

    meta = {
        "run":         "run02v2",
        "ifo":         ifo,
        "epoch":       epoch,
        "dur":         SCALE,
        "fs":          FS,
        "flow":        FLOW,
        "fhigh":       FHIGH,
        "qrange":      list(QRANGE),
        "img_size":    IMG_SIZE,
        "normalization": "none",
        "sampling":    "uniform_random",
        "seed":        SEED,
        "margin":      MARGIN,
        "n_generated": len(X_all),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    save_npz(out_path, X_all, t0_all, loge_all, meta)

    size_mb = out_path.stat().st_size / 1e6
    elapsed = time.time() - t_start_run
    print(f"\nSaved {len(X_all)} windows → {out_path}  ({size_mb:.1f} MB)")
    print(f"Done in {elapsed/60:.1f} min  (blocks ok={blocks_done} fail={blocks_failed})")

    # Final checkpoint + remove partial
    save_checkpoint(ckpt_path, done_blocks, X_all, t0_all, loge_all)
    partial_path = ckpt_path.with_suffix(".partial.npz")
    partial_path.unlink(missing_ok=True)


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
    for ifo, epoch in RUNS:
        run_ifo_epoch(ifo, epoch)
