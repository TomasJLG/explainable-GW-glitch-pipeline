#!/usr/bin/env python3
"""
generate_run03.py
=================
Dataset run03 — ventanas Q-transform centradas en peak_times de Gravity Spy O3,
con etiquetas reales por clase de glitch.

Pipeline de procesado IDENTICO a run02 (bandpass -> whiten -> notch -> Q-transform)
para que las distribuciones de pixeles sean directamente comparables.

Diferencias respecto a run02:
  - Sampling targeteado a peak_times de Gravity Spy (no aleatorio)
  - Cada ventana lleva metadata de etiqueta (ml_label, snr, etc.)
  - CLI recibe una sola escala; itera las 4 combinaciones IFO x epoch

Uso:
    python generate_run03.py --scale 1.0

# Descomentar en Kaggle si gwpy no esta instalado:
# !pip install gwpy
"""

# ---------- (0) Evitar oversubscription BLAS/OMP ----------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import collections
import csv
import json
import time
import random
import warnings
import multiprocessing as mp
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from gwpy.segments import DataQualityFlag, Segment, SegmentList
from gwpy.timeseries import TimeSeries
from scipy.signal import iirnotch, filtfilt
from scipy.ndimage import zoom
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor,
    wait, FIRST_COMPLETED,
)

# ============================================================
# CONFIG
# ============================================================

RUN_TAG = "run03"

# GPS ranges O3
O3A = (1238166018, 1253977218)
O3B = (1256655618, 1269363618)
EPOCHS: Dict[str, Tuple[int, int]] = {"O3a": O3A, "O3b": O3B}
IFOS = ["H1", "L1"]

# Señal — identico a run02
FS = 4096
FLOW, FHIGH = 20, 1291
QRANGE = (8, 64)
MISMATCH = 0.2
TARGET_SHAPE = (128, 128)
NOTCH_FREQS = [60, 120, 180, 240, 300, 360]
WHITEN_FFTLEN  = 4.0
WHITEN_OVERLAP = 2.0
FETCH_PAD      = 4.0
EDGE_CROP_MAX  = 0.5
ENERGY_PERCENTILE = 90

# Filtros run03
ML_CONFIDENCE_MIN = 0.9
EXCLUDED_LABELS   = ["None_of_the_Above"]
CAP_PER_CLASS     = 200
JITTER_FRACTION   = 0.25   # jitter = U(-dur*F, +dur*F)
SEED              = 1234

# Rutas
CSV_ROOT = "/kaggle/input/datasets/tomsjacoboleal/gravityspy-o3-triggers"
OUT_ROOT = "/kaggle/working/run03_glitches"

# Paralelismo — identico a run02
MAX_WORKERS    = min(4, os.cpu_count() or 4)
IN_FLIGHT      = max(8, MAX_WORKERS * 8)
HEARTBEAT_SECS = 20
CKPT_EVERY     = 20
CLEANUP_AFTER_PACK = False

# ============================================================
# Multiprocessing context
# ============================================================
try:
    CTX = mp.get_context("fork")
    USE_THREADS_FALLBACK = False
except ValueError:
    CTX = None
    USE_THREADS_FALLBACK = True


# ============================================================
# Utils IO / checkpoint  [identico a run02]
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_slug_scale(scale: float) -> str:
    return str(scale).replace(".", "p")


def effective_edge_crop(dur: float) -> float:
    edge = min(EDGE_CROP_MAX, 0.2 * dur)
    edge = min(edge, (dur / 2.0) - 1e-3)
    return max(0.0, edge)


def final_npz_path(scale_dir: str, ifo: str, epoch: str, dur: float) -> str:
    slug = safe_slug_scale(dur)
    return os.path.join(scale_dir, f"dataset_{ifo}_{epoch}_scale_{slug}_{RUN_TAG}.npz")


def open_or_create_memmap(
    scale_dir: str, n: int, shape_hw: Tuple[int, int]
) -> np.memmap:
    mm_path = os.path.join(scale_dir, "X.dat")
    H, W = shape_hw
    if not os.path.exists(mm_path):
        mm = np.memmap(mm_path, dtype="float32", mode="w+", shape=(n, H, W))
        mm[:] = np.nan
        mm.flush()
        return mm
    return np.memmap(mm_path, dtype="float32", mode="r+", shape=(n, H, W))


def open_or_create_energy_memmap(scale_dir: str, n: int) -> np.memmap:
    mm_path = os.path.join(scale_dir, "log_energy.dat")
    if not os.path.exists(mm_path):
        mm = np.memmap(mm_path, dtype="float32", mode="w+", shape=(n,))
        mm[:] = np.nan
        mm.flush()
        return mm
    return np.memmap(mm_path, dtype="float32", mode="r+", shape=(n,))


def load_or_create_done_mask(scale_dir: str, n: int) -> np.ndarray:
    path = os.path.join(scale_dir, "done.npy")
    if os.path.exists(path):
        done = np.load(path, allow_pickle=False)
        if done.shape == (n,) and done.dtype == np.bool_:
            return done
        print("[WARN] done.npy invalido. Regenerando.")
    done = np.zeros((n,), dtype=bool)
    np.save(path, done)
    return done


def save_done_mask(scale_dir: str, done: np.ndarray) -> None:
    path = os.path.join(scale_dir, "done.npy")
    tmp  = path + ".tmp"
    ghost = tmp + ".npy"
    if os.path.exists(ghost):
        try:
            os.remove(ghost)
        except Exception:
            pass
    with open(tmp, "wb") as f:
        np.save(f, done)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def cleanup_scale_temporals(scale_dir: str) -> None:
    for fn in ["X.dat", "log_energy.dat", "done.npy",
               "manifest.json", "checkpoint.json", "errors.log"]:
        p = os.path.join(scale_dir, fn)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


# ============================================================
# Segmentos de ciencia  [identico a run02]
# ============================================================

def query_science_segments(
    ifo: str, t_start: int, t_end: int, retries: int = 3
):
    """Descarga segmentos activos de GWOSC con reintentos."""
    flag = f"{ifo}_DATA"
    last = None
    for k in range(retries):
        try:
            dq = DataQualityFlag.fetch_open_data(flag, t_start, t_end)
            return dq.active
        except Exception as e:
            last = e
            time.sleep(1.5 * (k + 1))
    print(f"[WARN] fetch_open_data({flag}) fallo: {last}. Fallback: segmento completo.")
    return SegmentList([Segment(t_start, t_end)])


def segments_eligible_for_duration(
    segs, dur: float, pad: float
) -> List[Tuple[float, float]]:
    """Filtra segmentos donde cabe la ventana de fetch completa (dur + 2*pad)."""
    need = dur + 2.0 * pad
    out  = []
    for s in segs:
        a, b = float(s[0]), float(s[1])
        if (b - a) >= need:
            out.append((a, b))
    return out


def window_is_feasible(
    t0: float,
    dur: float,
    segs_eligible_sorted: List[Tuple[float, float]],
) -> bool:
    """
    True si [t0-FETCH_PAD, t0+dur+FETCH_PAD] cabe entero en algun segmento.
    segs_eligible_sorted debe estar ordenado por inicio (ascendente).
    """
    t_lo = t0 - FETCH_PAD
    t_hi = t0 + dur + FETCH_PAD
    for (a, b) in segs_eligible_sorted:
        if a > t_lo:
            break          # sorted: ningun segmento posterior empezara antes de t_lo
        if t_hi <= b:
            return True    # [t_lo, t_hi] cabe en [a, b]
    return False


# ============================================================
# Carga y filtrado de triggers  [nuevo para run03]
# ============================================================

def load_and_filter_triggers(
    csv_path: str,
    ifo: str,
    t_gps_start: int,
    t_gps_end: int,
) -> List[Dict[str, Any]]:
    """
    Lee el CSV de Gravity Spy y aplica los filtros de run03 en orden:
      1. ifo == IFO
      2. ml_confidence >= ML_CONFIDENCE_MIN
      3. ml_label not in EXCLUDED_LABELS (No_Glitch SI se mantiene)
      4. peak_time dentro de [t_gps_start, t_gps_end]
      5. Cap por clase: max CAP_PER_CLASS triggers por ml_label (seed=SEED)

    Retorna lista de dicts con t_peak (precision ns) y metadata de etiqueta.
    """
    if not os.path.exists(csv_path):
        print(f"  [WARN] CSV no encontrado: {csv_path}")
        return []

    raw: List[Dict[str, Any]] = []
    n_total = n_ifo_skip = n_conf_skip = n_label_skip = n_gps_skip = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1

            # Filtro 1: ifo
            if row.get("ifo", "").strip() != ifo:
                n_ifo_skip += 1
                continue

            # Filtro 2: ml_confidence
            try:
                conf = float(row["ml_confidence"])
            except (ValueError, KeyError):
                n_conf_skip += 1
                continue
            if conf < ML_CONFIDENCE_MIN:
                n_conf_skip += 1
                continue

            # Filtro 3: excluir etiquetas prohibidas
            label = row.get("ml_label", "").strip()
            if not label or label in EXCLUDED_LABELS:
                n_label_skip += 1
                continue

            # Filtro 4: rango GPS de la epoch
            try:
                t_peak = int(row["peak_time"]) + int(row["peak_time_ns"]) * 1e-9
            except (ValueError, KeyError):
                n_gps_skip += 1
                continue
            if not (t_gps_start <= t_peak <= t_gps_end):
                n_gps_skip += 1
                continue

            # Campos auxiliares con fallback a 0.0 si faltan o son vacios
            def _float(key: str) -> float:
                v = row.get(key, "").strip()
                try:
                    return float(v) if v else 0.0
                except ValueError:
                    return 0.0

            raw.append({
                "t_peak":         t_peak,
                "ml_label":       label,
                "ml_confidence":  conf,
                "snr":            _float("snr"),
                "peak_frequency": _float("peak_frequency"),
                "q_value":        _float("q_value"),
                "gravityspy_id":  row.get("gravityspy_id", "").strip(),
            })

    print(
        f"  [FILTER] total={n_total} | ifo_skip={n_ifo_skip} "
        f"conf_skip={n_conf_skip} label_skip={n_label_skip} "
        f"gps_skip={n_gps_skip} -> passing={len(raw)}"
    )

    # Filtro 5: cap por clase
    by_class: Dict[str, List] = {}
    for r in raw:
        by_class.setdefault(r["ml_label"], []).append(r)

    rng_cap = random.Random(SEED)
    result: List[Dict] = []
    for lbl in sorted(by_class):
        items = by_class[lbl]
        if len(items) > CAP_PER_CLASS:
            items = rng_cap.sample(items, CAP_PER_CLASS)
        result.extend(items)
        print(f"    [{lbl:<30s}]: {len(items):>4d} triggers")

    print(f"  [FILTER] Total tras cap: {len(result)}")
    return result


# ============================================================
# Manifest run03  [nuevo — almacena metadata de trigger + t0]
# ============================================================

def build_manifest_run03(
    triggers: List[Dict[str, Any]],
    dur: float,
    segs_eligible: List[Tuple[float, float]],
    seed: int,
) -> Dict[str, Any]:
    """
    Para cada trigger calcula t0 con jitter y valida contra segmentos.

    Jitter: t0 = t_peak + U(-dur*JITTER_FRACTION, +dur*JITTER_FRACTION) - dur/2
    Garantia: t_peak siempre cae en [t0 + dur/4, t0 + 3*dur/4].

    Triggers cuyo fetch window [t0-PAD, t0+dur+PAD] no cabe en ningun
    segmento elegible se descartan y se contabiliza el descarte.
    """
    rng            = random.Random(seed)
    segs_sorted    = sorted(segs_eligible, key=lambda s: s[0])
    entries        = []
    n_discarded    = 0

    for trig in triggers:
        t_peak = trig["t_peak"]
        jitter = rng.uniform(-dur * JITTER_FRACTION, dur * JITTER_FRACTION)
        t0     = t_peak + jitter - dur / 2.0

        if not window_is_feasible(t0, dur, segs_sorted):
            n_discarded += 1
            continue

        entries.append({
            "idx":            len(entries),   # indice en memmaps
            "t_peak":         t_peak,
            "t0":             t0,
            "ml_label":       trig["ml_label"],
            "ml_confidence":  trig["ml_confidence"],
            "snr":            trig["snr"],
            "peak_frequency": trig["peak_frequency"],
            "q_value":        trig["q_value"],
            "gravityspy_id":  trig["gravityspy_id"],
        })

    print(
        f"  [MANIFEST] {len(entries)} ventanas validas "
        f"({n_discarded} descartadas: t0 fuera de segmentos de ciencia)"
    )
    return {
        "n_target":     len(entries),
        "dur":          dur,
        "seed":         seed,
        "created_unix": time.time(),
        "filters": {
            "ml_confidence_min": ML_CONFIDENCE_MIN,
            "excluded_labels":   EXCLUDED_LABELS,
            "cap_per_class":     CAP_PER_CLASS,
            "jitter_fraction":   JITTER_FRACTION,
        },
        "entries": entries,
    }


def get_or_create_manifest_run03(
    scale_dir: str,
    triggers: List[Dict[str, Any]],
    dur: float,
    segs_eligible: List[Tuple[float, float]],
    seed: int,
) -> Dict[str, Any]:
    """
    Carga el manifest existente o construye uno nuevo.
    En sesiones de resume, el manifest persistido es la fuente de verdad:
    mismo t0 por trigger, mismo orden de indices en los memmaps.
    """
    manifest_path = os.path.join(scale_dir, "manifest.json")
    if os.path.exists(manifest_path):
        m = load_json(manifest_path)
        if isinstance(m.get("entries"), list) and len(m["entries"]) > 0:
            print(f"  [MANIFEST] Cargado: {len(m['entries'])} entradas.")
            return m
        print("[WARN] Manifest existente invalido o vacio. Regenerando.")

    m = build_manifest_run03(triggers, dur, segs_eligible, seed)
    if m["entries"]:
        save_json_atomic(manifest_path, m)
    return m


# ============================================================
# Preprocesado + Q-transform  [IDENTICO a run02 — no modificar]
# ============================================================

def apply_notches(
    x: np.ndarray, fs: int, freqs: List[float], Q: float = 30.0
) -> np.ndarray:
    y = x
    for f0 in freqs:
        if f0 <= 0 or f0 >= (fs / 2.0):
            continue
        b, a = iirnotch(w0=f0, Q=Q, fs=fs)
        y = filtfilt(b, a, y)
    return y


def preprocess_ts(
    ts: TimeSeries, fs: int, flow: float, fhigh: float
) -> TimeSeries:
    """
    Orden: bandpass -> whiten -> notch  (identico a run02).
    CRITICO: no alterar — las distribuciones run02/run03 deben ser comparables.
    """
    sr = float(ts.sample_rate.value)
    if abs(sr - fs) > 1e-6:
        ts = ts.resample(fs)
    ts = ts.bandpass(flow, fhigh, filtfilt=True)
    ts = ts.whiten(fftlength=WHITEN_FFTLEN, overlap=WHITEN_OVERLAP)
    x  = ts.value.astype(np.float64, copy=False)
    x  = apply_notches(x, fs, NOTCH_FREQS, Q=30.0)
    ts = TimeSeries(x, t0=ts.t0, dt=ts.dt, unit=ts.unit)
    return ts


def qtransform_to_image(
    ts: TimeSeries,
    outseg: Tuple[float, float],
    flow: float,
    fhigh: float,
    qrange: Tuple[float, float],
    mismatch: float,
    target_shape: Tuple[int, int],
) -> Tuple[np.ndarray, float]:
    """
    log1p(Q-transform) + log_energy + normalizacion por p99 de ventana.
    Identico a run02 — no modificar.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qspec = ts.q_transform(
            outseg=outseg,
            frange=(flow, fhigh),
            qrange=qrange,
            mismatch=mismatch,
        )

    Z = np.array(qspec.value, dtype=np.float64)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    Z = np.clip(Z, 0.0, None)
    Z = np.log1p(Z)

    log_energy = float(np.percentile(Z, ENERGY_PERCENTILE))

    p99 = float(np.percentile(Z, 99)) + 1e-6
    Z   = Z / p99
    Z   = np.clip(Z, 0.0, 1.0).astype(np.float32)

    if Z.shape != target_shape:
        zy = target_shape[0] / Z.shape[0]
        zx = target_shape[1] / Z.shape[1]
        Z  = zoom(Z, zoom=(zy, zx), order=1)

    return Z.astype(np.float32), log_energy


def fetch_open_data_retry(
    ifo: str, t_start: float, t_end: float, fs: int, retries: int = 3
) -> TimeSeries:
    last = None
    for k in range(retries):
        try:
            return TimeSeries.fetch_open_data(ifo, t_start, t_end, sample_rate=fs)
        except Exception as e:
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"fetch_open_data fallo tras {retries} intentos: {last}")


# ============================================================
# Worker  [misma firma que run02]
# ============================================================

def worker_compute(
    job: Tuple[int, str, str, float, float, float, float, int],
) -> Tuple[int, bool, Optional[np.ndarray], float, Optional[str]]:
    """
    Calcula la imagen Q-transform de una ventana.
    La metadata de etiqueta vive en el manifest (no en el worker),
    evitando pasar objetos grandes entre procesos.

    Retorna: (idx, ok, X, log_energy, error_msg)
    """
    idx, ifo, epoch, t0, dur, flow, fhigh, fs = job
    try:
        time.sleep((idx % 10) * 0.02)

        edge    = effective_edge_crop(dur)
        t_fetch0 = t0 - FETCH_PAD
        t_fetch1 = t0 + dur + FETCH_PAD
        out0    = t0 + edge
        out1    = t0 + dur - edge

        if out1 <= out0:
            raise ValueError(f"Ventana efectiva vacia: dur={dur}, edge={edge}")

        ts = fetch_open_data_retry(ifo, t_fetch0, t_fetch1, fs, retries=3)
        ts = preprocess_ts(ts, fs, flow, fhigh)

        X, log_energy = qtransform_to_image(
            ts=ts, outseg=(out0, out1),
            flow=flow, fhigh=fhigh,
            qrange=QRANGE, mismatch=MISMATCH,
            target_shape=TARGET_SHAPE,
        )
        return idx, True, X, log_energy, None

    except Exception as e:
        return idx, False, None, float("nan"), str(e)


def ping() -> int:
    return os.getpid()


# ============================================================
# Sanity checks  [checks de run02 + verificacion de labels]
# ============================================================

def sanity_check_dataset(
    Xmm: np.memmap,
    energies: np.memmap,
    t0s: List[float],
    entries: List[Dict[str, Any]],
    ifo: str,
    epoch: str,
    dur: float,
) -> Dict[str, Any]:
    """
    Verifica integridad antes de empaquetar.
    Lanza ValueError en fallos criticos.
    Checks adicionales run03: completitud de labels y distribucion de clases.
    """
    report: Dict[str, Any] = {}
    n = Xmm.shape[0]

    # 1. NaNs/Infs en imagenes
    nan_count = int(np.sum(np.isnan(Xmm)))
    inf_count = int(np.sum(np.isinf(Xmm)))
    report["nan_in_X"] = nan_count
    report["inf_in_X"] = inf_count
    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"[SANITY FAIL] X contiene {nan_count} NaNs y {inf_count} Infs."
        )

    # 2. Rango dinamico
    x_min  = float(np.min(Xmm))
    x_max  = float(np.max(Xmm))
    x_mean = float(np.mean(Xmm))
    report.update({"X_min": x_min, "X_max": x_max, "X_mean": round(x_mean, 4)})
    if x_min < -0.1 or x_max > 1.1:
        print(f"[SANITY WARN] Rango X fuera de [0,1]: min={x_min:.4f} max={x_max:.4f}")

    # 3. NaNs en energias
    nan_energy = int(np.sum(np.isnan(energies)))
    report["nan_in_log_energy"] = nan_energy
    if nan_energy > 0:
        raise ValueError(
            f"[SANITY FAIL] log_energy contiene {nan_energy} NaNs."
        )

    # 4. Distribucion de energias
    e_p10 = float(np.percentile(energies, 10))
    e_p50 = float(np.percentile(energies, 50))
    e_p90 = float(np.percentile(energies, 90))
    e_p99 = float(np.percentile(energies, 99))
    report.update({
        "log_energy_p10": round(e_p10, 4),
        "log_energy_p50": round(e_p50, 4),
        "log_energy_p90": round(e_p90, 4),
        "log_energy_p99": round(e_p99, 4),
    })

    # 5. t0s en rango GPS razonable
    t0_arr = np.array(t0s)
    t0_out = int(np.sum((t0_arr < 1e9) | (t0_arr > 2e9)))
    report["t0_out_of_range"] = t0_out
    if t0_out > 0:
        print(f"[SANITY WARN] {t0_out} t0s fuera de rango GPS razonable.")

    # 6. Imagenes planas (varianza casi cero)
    var_per_image = np.var(Xmm.reshape(n, -1), axis=1)
    flat_images   = int(np.sum(var_per_image < 1e-6))
    report["flat_images"] = flat_images
    if flat_images > n * 0.05:
        print(
            f"[SANITY WARN] {flat_images}/{n} imagenes planas "
            "(posiblemente corruptas)."
        )

    # 7. [run03] Completitud de labels
    if len(entries) != n:
        raise ValueError(
            f"[SANITY FAIL] len(entries)={len(entries)} != n={n}. "
            "Manifest desincronizado con memmaps."
        )
    bad_labels = sum(1 for e in entries if not e.get("ml_label", "").strip())
    report["entries_with_empty_label"] = bad_labels
    if bad_labels > 0:
        raise ValueError(
            f"[SANITY FAIL] {bad_labels} entries con ml_label vacio."
        )

    # 8. [run03] Distribucion de clases
    class_dist = dict(
        collections.Counter(e["ml_label"] for e in entries)
    )
    report["class_distribution"] = class_dist

    print(
        f"[SANITY OK] {ifo} {epoch} dur={dur}s | n={n} | "
        f"X in [{x_min:.3f},{x_max:.3f}] mean={x_mean:.3f} | "
        f"log_energy p50={e_p50:.3f} p99={e_p99:.3f} | "
        f"flat={flat_images} | classes={len(class_dist)}"
    )
    return report


# ============================================================
# Scheduler  [identico a run02]
# ============================================================

def iter_pending_jobs(
    done: np.ndarray,
    t0s: List[float],
    ifo: str,
    epoch: str,
    dur: float,
) -> Iterable:
    for i in range(len(done)):
        if not done[i]:
            yield (
                i, ifo, epoch,
                float(t0s[i]), float(dur),
                float(FLOW), float(FHIGH), int(FS),
            )


def run_jobs_streaming(
    EXEC,
    jobs_iter: Iterable,
    Xmm: np.memmap,
    energy_mm: np.memmap,
    done: np.ndarray,
    scale_dir: str,
    ckpt_path: str,
    ckpt: Dict[str, Any],
    t0s: List[float],
) -> None:
    errlog   = os.path.join(scale_dir, "errors.log")
    pending: Dict[Any, int] = {}
    completed = ok_added = fail_added = 0
    jobs_iter = iter(jobs_iter)

    def submit_one() -> bool:
        try:
            jb = next(jobs_iter)
        except StopIteration:
            return False
        fut = EXEC.submit(worker_compute, jb)
        pending[fut] = jb[0]
        return True

    for _ in range(IN_FLIGHT):
        if not submit_one():
            break

    while pending:
        done_set, _ = wait(
            set(pending.keys()), timeout=HEARTBEAT_SECS,
            return_when=FIRST_COMPLETED,
        )

        if not done_set:
            print(
                f"[HB] sin completados en {HEARTBEAT_SECS}s | "
                f"done={int(done.sum())}/{len(done)} | in_flight={len(pending)}"
            )
            continue

        for fut in done_set:
            _idx = pending.pop(fut)
            i, ok, X, log_energy, err = fut.result()

            if ok and X is not None:
                Xmm[i]       = X
                energy_mm[i] = log_energy
                done[i]      = True
                ok_added    += 1
            else:
                fail_added += 1
                with open(errlog, "a", encoding="utf-8") as f:
                    f.write(
                        f"{time.time():.0f}\tidx={i}\t"
                        f"t0={t0s[i]:.6f}\terr={err}\n"
                    )

            completed += 1
            submit_one()

            if (completed % CKPT_EVERY) == 0:
                Xmm.flush()
                energy_mm.flush()
                save_done_mask(scale_dir, done)
                done_now = int(done.sum())
                ckpt.update({
                    "status":            "running",
                    "done":              done_now,
                    "target":            len(done),
                    "ok_added_last":     ok_added,
                    "fail_added_last":   fail_added,
                    "completed_in_run":  completed,
                    "in_flight":         len(pending),
                    "updated_unix":      time.time(),
                })
                save_json_atomic(ckpt_path, ckpt)
                print(
                    f"[CKPT] +{CKPT_EVERY} | done={done_now}/{len(done)} | "
                    f"ok+{ok_added} fail+{fail_added} | in_flight={len(pending)}"
                )

    Xmm.flush()
    energy_mm.flush()
    save_done_mask(scale_dir, done)
    done_now = int(done.sum())
    ckpt.update({
        "status":           "ready_to_pack" if done_now == len(done) else "partial",
        "done":             done_now,
        "target":           len(done),
        "ok_added_last":    ok_added,
        "fail_added_last":  fail_added,
        "completed_in_run": completed,
        "in_flight":        0,
        "updated_unix":     time.time(),
    })
    save_json_atomic(ckpt_path, ckpt)


# ============================================================
# Runner por IFO + epoch
# ============================================================

def run_for_ifo_epoch(ifo: str, epoch_name: str, dur: float, EXEC) -> None:
    """
    Ejecuta el pipeline completo para una combinacion (ifo, epoch, scale).
    Resumible: si el NPZ final existe, salta. Si hay trabajo parcial,
    retoma desde el checkpoint y el manifest.
    """
    t_start, t_end = EPOCHS[epoch_name]
    slug = safe_slug_scale(dur)

    print("\n" + "=" * 64)
    print(f"  {ifo} - {epoch_name} | scale={dur}s | RUN={RUN_TAG}")
    print("=" * 64)

    scale_dir = os.path.join(OUT_ROOT, ifo, epoch_name, f"scale_{slug}")
    ensure_dir(scale_dir)

    ckpt_path = os.path.join(scale_dir, "checkpoint.json")
    ckpt = load_json(ckpt_path)
    for k, v in [
        ("ifo", ifo), ("epoch", epoch_name), ("dur", dur),
        ("fs", FS), ("flow", FLOW), ("fhigh", FHIGH),
        ("target_shape", list(TARGET_SHAPE)), ("run_tag", RUN_TAG),
    ]:
        ckpt.setdefault(k, v)

    out_npz = final_npz_path(scale_dir, ifo, epoch_name, dur)

    edge         = effective_edge_crop(dur)
    effective_dur = dur - 2 * edge
    print(f"  edge_crop={edge:.3f}s | ventana_efectiva={effective_dur:.3f}s")
    if effective_dur < 0.2:
        print(
            f"[WARN] Ventana efectiva={effective_dur:.3f}s muy corta "
            f"(escala {dur}s)."
        )

    if os.path.exists(out_npz):
        print(f"[INFO] NPZ ya existe: {os.path.basename(out_npz)} -> saltando.")
        ckpt.update({"status": "done", "final_npz": out_npz,
                     "updated_unix": time.time()})
        save_json_atomic(ckpt_path, ckpt)
        return

    # ── Paso 1: Segmentos de ciencia ─────────────────────────────────────────
    segs = query_science_segments(ifo, t_start, t_end)
    print(f"[INFO] Segmentos activos: {len(segs)}")
    segs_eligible = segments_eligible_for_duration(segs, dur, pad=FETCH_PAD)
    if not segs_eligible:
        print(f"[WARN] Sin segmentos elegibles para dur={dur}. Saltando.")
        ckpt.update({"status": "no_segments", "updated_unix": time.time()})
        save_json_atomic(ckpt_path, ckpt)
        return

    # ── Paso 2: Cargar y filtrar triggers ────────────────────────────────────
    csv_path = os.path.join(CSV_ROOT, f"{ifo}_{epoch_name}.csv")
    print(f"[INFO] Cargando triggers: {csv_path}")
    triggers = load_and_filter_triggers(csv_path, ifo, t_start, t_end)
    if not triggers:
        print(f"[WARN] 0 triggers tras filtros para {ifo} {epoch_name}. Saltando.")
        ckpt.update({"status": "no_triggers", "updated_unix": time.time()})
        save_json_atomic(ckpt_path, ckpt)
        return

    # ── Paso 3: Manifest (t0 con jitter + validacion de segmentos) ───────────
    seed = SEED + (abs(hash((ifo, epoch_name, dur, RUN_TAG))) % 10_000_000)
    manifest = get_or_create_manifest_run03(
        scale_dir, triggers, dur, segs_eligible, seed
    )

    entries  = manifest["entries"]
    n_target = len(entries)
    if n_target == 0:
        print("[WARN] 0 entradas validas en manifest. Saltando.")
        ckpt.update({"status": "no_valid_entries", "updated_unix": time.time()})
        save_json_atomic(ckpt_path, ckpt)
        return

    t0s     = [float(e["t0"])     for e in entries]
    t_peaks = [float(e["t_peak"]) for e in entries]
    print(f"[INFO] n_target={n_target} ventanas.")

    # ── Paso 4: Memmaps y mascara done ───────────────────────────────────────
    Xmm       = open_or_create_memmap(scale_dir, n_target, TARGET_SHAPE)
    energy_mm = open_or_create_energy_memmap(scale_dir, n_target)
    done      = load_or_create_done_mask(scale_dir, n_target)

    done_count = int(done.sum())
    remaining  = n_target - done_count
    print(f"[INFO] Hechos: {done_count}/{n_target} | Restantes: {remaining}")

    # ── Paso 5: Cómputo en paralelo ──────────────────────────────────────────
    if remaining > 0:
        ckpt.update({
            "status": "running", "done": done_count,
            "target": n_target, "updated_unix": time.time(),
        })
        save_json_atomic(ckpt_path, ckpt)

        jobs_iter = iter_pending_jobs(done, t0s, ifo, epoch_name, dur)
        t0_run    = time.time()
        run_jobs_streaming(
            EXEC, jobs_iter, Xmm, energy_mm, done,
            scale_dir, ckpt_path, ckpt, t0s,
        )
        t1_run = time.time()

        done_count = int(done.sum())
        ckpt.update({
            "last_run_seconds": round(t1_run - t0_run, 1),
            "updated_unix": time.time(),
        })
        save_json_atomic(ckpt_path, ckpt)
        print(
            f"[INFO] Fin computo. done={done_count}/{n_target}. "
            f"Tiempo={t1_run - t0_run:.1f}s"
        )

    # Releer done tras la sesion de computo
    done = load_or_create_done_mask(scale_dir, n_target)
    if int(done.sum()) < n_target:
        print(
            f"[WARN] Incompleto ({int(done.sum())}/{n_target}). "
            "Re-ejecuta para rellenar huecos."
        )
        return

    # ── Paso 6: Sanity checks ────────────────────────────────────────────────
    print("[SANITY] Verificando integridad antes de empaquetar...")
    Xmm       = open_or_create_memmap(scale_dir, n_target, TARGET_SHAPE)
    energy_mm = open_or_create_energy_memmap(scale_dir, n_target)
    try:
        sanity_report = sanity_check_dataset(
            Xmm, energy_mm, t0s, entries, ifo, epoch_name, dur
        )
    except ValueError as e:
        print(f"[SANITY FAIL] {e}. No se empaqueta. Re-ejecuta.")
        ckpt.update({
            "status": "sanity_failed",
            "error":  str(e),
            "updated_unix": time.time(),
        })
        save_json_atomic(ckpt_path, ckpt)
        return

    # ── Paso 7: Empaquetar NPZ ───────────────────────────────────────────────
    print(f"[PACK] Generando NPZ: {os.path.basename(out_npz)}")

    labels      = [e["ml_label"]       for e in entries]
    confidences = [e["ml_confidence"]  for e in entries]
    snrs        = [e["snr"]            for e in entries]
    peak_freqs  = [e["peak_frequency"] for e in entries]
    q_values    = [e["q_value"]        for e in entries]
    gs_ids      = [e["gravityspy_id"]  for e in entries]

    meta = {
        # Identificadores
        "ifo":          ifo,
        "epoch":        epoch_name,
        "dur":          dur,
        "run_tag":      RUN_TAG,
        # Parametros de senial (identicos a run02)
        "fs":           FS,
        "flow":         FLOW,
        "fhigh":        FHIGH,
        "qrange":       list(QRANGE),
        "mismatch":     MISMATCH,
        "fetch_pad":    FETCH_PAD,
        "edge_crop_max":     EDGE_CROP_MAX,
        "effective_window_dur": effective_dur,
        "target_shape":      list(TARGET_SHAPE),
        # Preprocesado (identico a run02)
        "preprocess_order":  "bandpass -> whiten -> notch",
        "whiten_fftlength":  WHITEN_FFTLEN,
        "whiten_overlap":    WHITEN_OVERLAP,
        "notch_freqs_hz":    NOTCH_FREQS,
        "log_energy_description": (
            "log1p(percentile(Q-transform, 90)) antes de normalizar. "
            "Feature de anomalia: valores altos indican ventana energetica."
        ),
        "log_energy_percentile": ENERGY_PERCENTILE,
        "normalization": "per-window p99 divisor (identico a run02)",
        # Sampling run03
        "sampling":          "targeted_gravity_spy",
        "ml_confidence_min": ML_CONFIDENCE_MIN,
        "excluded_labels":   EXCLUDED_LABELS,
        "cap_per_class":     CAP_PER_CLASS,
        "jitter_fraction":   JITTER_FRACTION,
        "jitter_description": (
            "t0 = t_peak + U(-dur*jitter_fraction, +dur*jitter_fraction) - dur/2. "
            "t_peak siempre cae en [t0 + dur/4, t0 + 3*dur/4]."
        ),
        "seed":              seed,
        # Estadisticas
        "n_windows":         n_target,
        "class_distribution": sanity_report.get("class_distribution", {}),
        "sanity_check":      sanity_report,
        "created_unix":      time.time(),
    }

    np.savez_compressed(
        out_npz,
        X              = np.array(Xmm,         dtype=np.float32),
        t0             = np.array(t0s,          dtype=np.float64),
        t_peak         = np.array(t_peaks,      dtype=np.float64),
        log_energy     = np.array(energy_mm,    dtype=np.float32),
        label          = np.array(labels,       dtype="<U30"),
        ml_confidence  = np.array(confidences,  dtype=np.float32),
        snr            = np.array(snrs,         dtype=np.float32),
        peak_frequency = np.array(peak_freqs,   dtype=np.float32),
        q_value        = np.array(q_values,     dtype=np.float32),
        gravityspy_id  = np.array(gs_ids,       dtype="<U40"),
        meta_json      = np.array(
            json.dumps(meta, ensure_ascii=False)
        ),
    )

    ckpt.update({
        "status": "done", "final_npz": out_npz,
        "updated_unix": time.time(),
    })
    save_json_atomic(ckpt_path, ckpt)
    print(f"[DONE] {ifo} {epoch_name} scale={dur}s -> {out_npz}")

    if CLEANUP_AFTER_PACK:
        cleanup_scale_temporals(scale_dir)


# ============================================================
# CLI + main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Genera dataset run03 — ventanas Q-transform centradas en "
            "triggers de Gravity Spy O3 con etiquetas reales."
        )
    )
    p.add_argument(
        "--scale",
        type=float,
        required=True,
        metavar="{0.25,1.0,4.0}",
        help="Duracion de ventana en segundos.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dur  = args.scale

    valid_scales = {0.25, 1.0, 4.0}
    if dur not in valid_scales:
        raise SystemExit(
            f"[ERROR] --scale debe ser uno de {valid_scales}. Recibido: {dur}"
        )

    ensure_dir(OUT_ROOT)

    if USE_THREADS_FALLBACK:
        Executor  = ThreadPoolExecutor
        ex_kwargs = {"max_workers": MAX_WORKERS}
        print("[WARN] Fork no disponible. Usando ThreadPoolExecutor.")
    else:
        Executor  = ProcessPoolExecutor
        ex_kwargs = {"max_workers": MAX_WORKERS, "mp_context": CTX}
        print(
            f"[INFO] ProcessPoolExecutor(fork) | "
            f"max_workers={MAX_WORKERS} | in_flight={IN_FLIGHT}"
        )

    print(
        f"[INFO] scale={dur}s | iterando 4 combinaciones "
        f"(H1/L1 x O3a/O3b) | OUT_ROOT={OUT_ROOT}"
    )

    with Executor(**ex_kwargs) as EXEC:
        # Warmup: verificar que los workers arrancan
        try:
            warm = [EXEC.submit(ping) for _ in range(min(MAX_WORKERS, 4))]
            [f.result(timeout=30) for f in warm]
            print("[INFO] Warmup workers OK.")
        except Exception as e:
            print(f"[WARN] Warmup fallo: {e}. Continuando...")

        for epoch_name in ["O3a", "O3b"]:
            for ifo in ["H1", "L1"]:
                run_for_ifo_epoch(ifo, epoch_name, dur, EXEC)

    print("\n" + "=" * 64)
    print(f"FIN. Output root: {OUT_ROOT}")
    print(
        "NPZ contiene: X, t0, t_peak, log_energy, label, ml_confidence,\n"
        "              snr, peak_frequency, q_value, gravityspy_id, meta_json"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
