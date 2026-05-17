"""
m1_dataloader.py
================
Ingesta del dataset v2 para M1 (detección de anomalías).

El dataset NPZ contiene:
    X           : (N, 128, 128) float32  — imágenes Q-transform normalizadas por p99
    t0          : (N,)          float64  — tiempos GPS de inicio de ventana
    log_energy  : (N,)          float32  — log1p(p90 del Q-transform pre-normalización)
    meta_json   : str                    — configuración del generador

Estrategia de ingesta para M1:
    M1 es un autoencoder entrenado SOLO sobre ventanas nominales (sin glitches).
    Su score de anomalía es el error de reconstrucción: ventanas con glitches
    se reconstruyen peor y tienen error alto.

    El log_energy se usa como:
    (a) Feature adicional de entrada al AE (opcional, modo "energy-aware")
    (b) Pre-filtro para crear el conjunto de entrenamiento nominal:
        ventanas con log_energy por encima del percentil P95 del dataset
        son candidatas a glitch y se excluyen del entrenamiento de M1.
    (c) Feature independiente para un detector de umbral simple como baseline.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# Carga de un NPZ individual
# ============================================================

def load_npz(path: str) -> Dict[str, np.ndarray]:
    """
    Carga un NPZ del dataset v2 y devuelve un dict con:
        X           : (N, 128, 128) float32
        t0          : (N,)          float64
        log_energy  : (N,)          float32
        meta        : dict          (parsed de meta_json)
    """
    data = np.load(path, allow_pickle=False)

    required = {"X", "t0", "log_energy", "meta_json"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"NPZ {path} no contiene: {missing}. "
                         f"¿Es un dataset v1? Regenerar con v2.")

    meta = json.loads(str(data["meta_json"]))

    return {
        "X": data["X"],
        "t0": data["t0"],
        "log_energy": data["log_energy"],
        "meta": meta,
    }


# ============================================================
# Carga multi-escala / multi-detector
# ============================================================

def load_dataset_collection(
    npz_paths: List[str],
    scales: Optional[List[float]] = None,
    ifos: Optional[List[str]] = None,
    epochs: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Carga varios NPZ y los indexa por (ifo, epoch, scale).
    Filtros opcionales por escala, detector y época.

    Retorna:
        {
          "H1_O3a_0.25": {"X": ..., "t0": ..., "log_energy": ..., "meta": ...},
          "H1_O3a_1.0":  {...},
          ...
        }
    """
    collection = {}
    for path in npz_paths:
        d = load_npz(path)
        meta = d["meta"]

        ifo = meta["ifo"]
        epoch = meta["epoch"]
        dur = float(meta["dur"])

        if ifos and ifo not in ifos:
            continue
        if epochs and epoch not in epochs:
            continue
        if scales and dur not in scales:
            continue

        key = f"{ifo}_{epoch}_{dur}"
        collection[key] = d
        print(f"[LOAD] {key}: X={d['X'].shape}, "
              f"log_energy p50={np.percentile(d['log_energy'], 50):.3f} "
              f"p99={np.percentile(d['log_energy'], 99):.3f}")

    return collection


# ============================================================
# Construcción del split nominal para M1
# ============================================================

def build_nominal_split(
    d: Dict,
    energy_threshold_percentile: float = 95.0,
    train_ratio: float = 0.85,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Construye train/val solo con ventanas nominales para M1.

    Estrategia:
    - Ventanas con log_energy > percentil(P) del dataset se consideran
      candidatas a anomalía y se excluyen del entrenamiento.
    - El resto se divide en train/val aleatoriamente (sin leakage temporal
      por ahora; el leakage temporal se trata en build_temporal_split).

    Parámetros:
        energy_threshold_percentile: percentil sobre el que se considera
            una ventana potencialmente anómala. 95 es conservador
            (excluye el 5% más energético). Bajar a 90 para más limpieza.
        train_ratio: fracción de nominales para entrenamiento.

    Retorna:
        {
            "X_train": (N_train, 128, 128),
            "X_val":   (N_val, 128, 128),
            "t0_train": ..., "t0_val": ...,
            "log_energy_train": ..., "log_energy_val": ...,
            "idx_nominal": ...,   # índices en el dataset original
            "idx_anomaly_candidate": ...,
            "energy_threshold": float,
        }
    """
    X = d["X"]
    t0 = d["t0"]
    log_energy = d["log_energy"]
    n = len(X)

    energy_thresh = float(np.percentile(log_energy, energy_threshold_percentile))
    is_nominal = log_energy <= energy_thresh

    idx_nominal = np.where(is_nominal)[0]
    idx_candidate = np.where(~is_nominal)[0]

    print(f"[SPLIT] Nominales: {len(idx_nominal)}/{n} ({100*len(idx_nominal)/n:.1f}%) | "
          f"Candidatos anomalía: {len(idx_candidate)} | "
          f"Umbral log_energy={energy_thresh:.4f} (P{energy_threshold_percentile})")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(idx_nominal)
    n_train = int(len(shuffled) * train_ratio)
    idx_train = shuffled[:n_train]
    idx_val = shuffled[n_train:]

    return {
        "X_train": X[idx_train],
        "X_val": X[idx_val],
        "t0_train": t0[idx_train],
        "t0_val": t0[idx_val],
        "log_energy_train": log_energy[idx_train],
        "log_energy_val": log_energy[idx_val],
        "idx_nominal": idx_nominal,
        "idx_anomaly_candidate": idx_candidate,
        "energy_threshold": energy_thresh,
        "n_total": n,
        "n_nominal": len(idx_nominal),
        "n_candidate": len(idx_candidate),
    }


# ============================================================
# Split temporal (control de leakage)
# ============================================================

def build_temporal_split(
    d: Dict,
    train_end_percentile: float = 70.0,
    val_end_percentile: float = 85.0,
    energy_threshold_percentile: float = 95.0,
) -> Dict[str, np.ndarray]:
    """
    Split por tiempo GPS para evitar leakage temporal.

    En lugar de shuffle aleatorio, dividimos por bloques temporales:
        train: t0 <= percentil 70 del tiempo
        val:   percentil 70 < t0 <= percentil 85
        test:  t0 > percentil 85

    Solo las ventanas nominales (por log_energy) van a train/val.
    Las candidatas a anomalía van siempre al conjunto de evaluación
    independientemente del tiempo.

    Este split es más conservador que el aleatorio y es el recomendado
    para la evaluación cruzada O3a -> O3b y H1 -> L1.
    """
    t0 = d["t0"]
    log_energy = d["log_energy"]
    X = d["X"]

    t_train_end = np.percentile(t0, train_end_percentile)
    t_val_end = np.percentile(t0, val_end_percentile)

    energy_thresh = float(np.percentile(log_energy, energy_threshold_percentile))
    is_nominal = log_energy <= energy_thresh

    mask_train = (t0 <= t_train_end) & is_nominal
    mask_val = (t0 > t_train_end) & (t0 <= t_val_end) & is_nominal
    mask_test_nominal = (t0 > t_val_end) & is_nominal
    mask_test_anomaly = ~is_nominal  # todos los candidatos van a test

    print(f"[TEMPORAL SPLIT] train={mask_train.sum()} val={mask_val.sum()} "
          f"test_nominal={mask_test_nominal.sum()} test_anomaly={mask_test_anomaly.sum()}")

    return {
        "X_train": X[mask_train],
        "X_val": X[mask_val],
        "X_test_nominal": X[mask_test_nominal],
        "X_test_anomaly": X[mask_test_anomaly],
        "t0_train": t0[mask_train],
        "t0_val": t0[mask_val],
        "t0_test_nominal": t0[mask_test_nominal],
        "t0_test_anomaly": t0[mask_test_anomaly],
        "log_energy_train": log_energy[mask_train],
        "log_energy_val": log_energy[mask_val],
        "log_energy_test_nominal": log_energy[mask_test_nominal],
        "log_energy_test_anomaly": log_energy[mask_test_anomaly],
        "energy_threshold": energy_thresh,
        "t_train_end": t_train_end,
        "t_val_end": t_val_end,
    }


# ============================================================
# Normalización global post-carga (paso de post-procesado)
# ============================================================

def compute_global_normalization(X_train: np.ndarray) -> Tuple[float, float]:
    """
    Calcula los percentiles globales P1 y P99 sobre el conjunto de entrenamiento.
    Estos se usan para normalizar TODO el dataset (train + val + test) de forma
    consistente, preservando diferencias de energía entre ventanas.

    No se usa z-score porque destruiría la información de magnitud relativa
    que es crítica para M1.
    """
    p1 = float(np.percentile(X_train, 1))
    p99 = float(np.percentile(X_train, 99))
    return p1, p99

def apply_global_normalization(X: np.ndarray, p1: float, p99: float) -> np.ndarray:
    """
    Normaliza X al rango [0, 1] usando percentiles globales del train set.
    Aplica clip para evitar outliers fuera de rango.
    """
    X_norm = (X - p1) / (p99 - p1 + 1e-8)
    return np.clip(X_norm, 0.0, 1.0).astype(np.float32)


# ============================================================
# Ejemplo de uso completo
# ============================================================

if __name__ == "__main__":
    # Ejemplo: cargar escala 1.0s de H1 O3a para M1
    NPZ_PATH = "/kaggle/working/qdataset_npz_v2/run02/H1/O3a/scale_1p0/dataset_H1_O3a_scale_1p0_run02.npz"

    # 1. Cargar
    d = load_npz(NPZ_PATH)
    print(f"Dataset cargado: X={d['X'].shape}, log_energy={d['log_energy'].shape}")
    print(f"Meta: ifo={d['meta']['ifo']}, epoch={d['meta']['epoch']}, "
          f"preprocess={d['meta']['preprocess_order']}")

    # 2. Split temporal (recomendado para evitar leakage)
    split = build_temporal_split(d, energy_threshold_percentile=95.0)

    X_train = split["X_train"]
    X_val = split["X_val"]

    # 3. Normalización global (usando solo train para calcular percentiles)
    p1, p99 = compute_global_normalization(X_train)
    print(f"Normalización global: p1={p1:.4f}, p99={p99:.4f}")

    X_train_norm = apply_global_normalization(X_train, p1, p99)
    X_val_norm = apply_global_normalization(X_val, p1, p99)

    # 4. Añadir dimensión de canal para PyTorch/Keras: (N, 1, H, W) o (N, H, W, 1)
    X_train_final = X_train_norm[:, np.newaxis, :, :]  # PyTorch: (N, C, H, W)
    X_val_final = X_val_norm[:, np.newaxis, :, :]

    print(f"\nListo para M1:")
    print(f"  X_train: {X_train_final.shape}")
    print(f"  X_val:   {X_val_final.shape}")
    print(f"  log_energy disponible como feature adicional o para umbral baseline")
