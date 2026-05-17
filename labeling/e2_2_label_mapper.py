"""
e2_2_label_mapper.py
====================
Módulo E2.2 — Mapeo de eventos Gravity Spy / Omicron a ventanas del dataset v2.

Responsabilidad:
    Lee un NPZ del dataset v2 (X, t0, log_energy, meta_json) y el CSV de
    metadatos de glitches (trainingset_v1d1_metadata.csv) y produce un array
    de etiquetas por ventana siguiendo la regla de matching temporal por IFO.

Regla de etiquetado:
    Una ventana cubre [t0_i, t0_i + dur).
    - 0 matches → "noise"    (label_id = -1)
    - 1 match   → label del evento  (label_id = índice de clase, 0..K-1)
    - 2+ matches → "ambiguous"  (label_id = -2)

    Las ventanas "ambiguous" se conservan en el output con su flag pero se
    excluyen del entrenamiento supervisado de M2.  Son válidas para M1 como
    anomalías sin clase confirmada.

Compatibilidad con m1_dataloader.load_npz():
    El NPZ de salida conserva las claves originales (X, t0, log_energy,
    meta_json) y añade (labels, label_ids, match_count). load_npz() lo
    cargará sin modificaciones.

Uso como módulo:
    from labeling.e2_2_label_mapper import build_label_array, build_temporal_split

Uso como script:
    python -m labeling.e2_2_label_mapper \\
        --npz_dir /kaggle/working/qdataset_npz_v2/run02 \\
        --metadata trainingset_v1d1_metadata.csv \\
        --output_dir /kaggle/working/labeled_run02 \\
        --scales 0.25 1.0 4.0 \\
        --ifos H1 L1 \\
        --epochs O3a O3b
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes internas
# ---------------------------------------------------------------------------

_NOISE_LABEL: str = "noise"
_AMBIGUOUS_LABEL: str = "ambiguous"
_NOISE_ID: int = -1
_AMBIGUOUS_ID: int = -2

# Nombre canónico de la columna de tiempo GPS en el CSV (Gravity Spy / Omicron)
_PEAK_TIME_COL: str = "peak_time"
_LABEL_COL: str = "label"
_IFO_COL: str = "ifo"

# Máxima longitud de string para el array numpy de etiquetas
_LABEL_DTYPE: str = "U64"


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _load_and_prepare_metadata(
    metadata_csv: str,
    ifo: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Carga el CSV de metadatos, filtra por IFO y construye el class_map global.

    Parámetros
    ----------
    metadata_csv : str
        Ruta al CSV (trainingset_v1d1_metadata.csv o similar).
    ifo : str
        Identificador del detector, p.ej. "H1" o "L1".

    Retorna
    -------
    sorted_peaks : np.ndarray (M,) float64
        Peak times de los eventos del IFO, ordenados ascendentemente.
    sorted_labels : np.ndarray (M,) str
        Etiqueta de clase para cada evento (mismo orden que sorted_peaks).
    class_map : dict {label_str: int_id}
        Mapa global construido sobre TODAS las clases únicas del CSV
        (excluyendo "noise" y "ambiguous"), con IDs 0..K-1.
        "noise" → -1 y "ambiguous" → -2 se añaden al final.
    """
    df = pd.read_csv(metadata_csv)

    # Validar columnas mínimas requeridas
    for col in (_PEAK_TIME_COL, _LABEL_COL, _IFO_COL):
        if col not in df.columns:
            raise ValueError(
                f"El CSV '{metadata_csv}' no contiene la columna '{col}'. "
                f"Columnas disponibles: {list(df.columns)}"
            )

    # Construir class_map global (todas las clases del CSV, orden alfabético)
    all_labels = sorted(df[_LABEL_COL].dropna().unique().tolist())
    # Excluir etiquetas especiales si estuvieran en el CSV
    all_labels = [l for l in all_labels if l not in (_NOISE_LABEL, _AMBIGUOUS_LABEL)]
    class_map: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(all_labels)}
    class_map[_NOISE_LABEL] = _NOISE_ID
    class_map[_AMBIGUOUS_LABEL] = _AMBIGUOUS_ID

    logger.info(
        "CSV cargado: %d eventos totales, %d clases únicas.",
        len(df), len(all_labels),
    )

    # Filtrar por IFO
    df_ifo = df[df[_IFO_COL] == ifo].copy()
    df_ifo = df_ifo.dropna(subset=[_PEAK_TIME_COL, _LABEL_COL])

    logger.info("IFO=%s: %d eventos tras filtrado.", ifo, len(df_ifo))

    if df_ifo.empty:
        return np.array([], dtype=np.float64), np.array([], dtype=_LABEL_DTYPE), class_map

    # Ordenar por peak_time para búsqueda binaria posterior
    df_ifo = df_ifo.sort_values(_PEAK_TIME_COL)
    sorted_peaks = df_ifo[_PEAK_TIME_COL].to_numpy(dtype=np.float64)
    sorted_labels = df_ifo[_LABEL_COL].to_numpy(dtype=_LABEL_DTYPE)

    return sorted_peaks, sorted_labels, class_map


def _match_windows(
    t0: np.ndarray,
    dur: float,
    sorted_peaks: np.ndarray,
    sorted_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matching vectorizado O(N log M) entre ventanas y eventos.

    Para cada ventana [t0_i, t0_i + dur) utiliza np.searchsorted para
    encontrar eficientemente todos los eventos cuyo peak_time cae dentro
    del intervalo, sin bucle Python por ventana.

    Parámetros
    ----------
    t0 : np.ndarray (N,) float64
        Tiempos GPS de inicio de cada ventana.
    dur : float
        Duración de la ventana en segundos (idéntica para todas).
    sorted_peaks : np.ndarray (M,) float64
        Peak times de los eventos, ordenados ascendentemente.
    sorted_labels : np.ndarray (M,) str
        Etiquetas correspondientes a sorted_peaks.

    Retorna
    -------
    match_count : np.ndarray (N,) int32
        Número de eventos que caen en cada ventana.
    first_label : np.ndarray (N,) str
        Etiqueta del primer evento encontrado para cada ventana
        (relevante solo donde match_count == 1; "" en los demás casos).
    """
    t1 = t0 + dur  # extremo derecho exclusivo

    # Índice del primer evento con peak_time >= t0_i
    left_idx = np.searchsorted(sorted_peaks, t0, side="left")
    # Índice del primer evento con peak_time >= t1_i (extremo excluido)
    right_idx = np.searchsorted(sorted_peaks, t1, side="left")

    match_count = (right_idx - left_idx).astype(np.int32)

    # Etiqueta del primer match (solo válida donde match_count >= 1)
    N = len(t0)
    M = len(sorted_peaks)
    first_label = np.full(N, "", dtype=_LABEL_DTYPE)
    has_match = match_count >= 1
    safe_left = np.clip(left_idx, 0, M - 1)
    first_label[has_match] = sorted_labels[safe_left[has_match]]

    return match_count, first_label


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def build_label_array(
    npz_path: str,
    metadata_csv: str,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Mapea eventos del CSV de metadatos a ventanas del NPZ del dataset v2.

    Algoritmo de matching:
        Los eventos se ordenan por peak_time y se usa np.searchsorted para
        encontrar, por ventana, cuántos eventos caen en [t0_i, t0_i + dur).
        Complejidad: O(N log M), donde N = ventanas, M = eventos del IFO.

    Parámetros
    ----------
    npz_path : str
        Ruta al NPZ del dataset v2 (debe contener X, t0, log_energy, meta_json).
    metadata_csv : str
        Ruta al CSV de metadatos Gravity Spy / Omicron.
    output_path : str, opcional
        Si se especifica, guarda el resultado como NPZ en esa ruta.
        El NPZ output conserva X, t0, log_energy, meta_json del original
        y añade labels, label_ids, match_count.

    Retorna
    -------
    dict con:
        labels        : np.ndarray (N,) str   — "noise", clase, o "ambiguous"
        label_ids     : np.ndarray (N,) int32 — -1, 0..K-1, o -2
        match_count   : np.ndarray (N,) int32 — nº de eventos por ventana
        class_map     : dict {label_str: int_id}
        stats         : dict con conteos: noise, ambiguous, total, y por clase
        t0            : np.ndarray (N,) float64 — copiado del NPZ
        log_energy    : np.ndarray (N,) float32 — copiado del NPZ
        meta          : dict — meta_json parseado + parámetros de etiquetado
    """
    # 1. Cargar NPZ (sin importar load_npz para evitar importación circular
    #    si este módulo se usa de forma independiente)
    raw = np.load(npz_path, allow_pickle=False)
    required = {"X", "t0", "log_energy", "meta_json"}
    missing = required - set(raw.files)
    if missing:
        raise ValueError(
            f"NPZ '{npz_path}' no contiene: {missing}. "
            "¿Es un dataset v1? Regenerar con v2."
        )

    t0: np.ndarray = raw["t0"].astype(np.float64)
    log_energy: np.ndarray = raw["log_energy"].astype(np.float32)
    X: np.ndarray = raw["X"]
    meta: dict = json.loads(str(raw["meta_json"]))

    ifo: str = meta["ifo"]
    dur: float = float(meta["dur"])
    N: int = len(t0)

    logger.info(
        "NPZ cargado: ifo=%s, dur=%.2fs, N=%d ventanas. Archivo: %s",
        ifo, dur, N, npz_path,
    )

    # 2. Cargar y preparar metadatos del CSV
    sorted_peaks, sorted_labels, class_map = _load_and_prepare_metadata(
        metadata_csv, ifo
    )

    # 3. Matching vectorizado
    match_count, first_label = _match_windows(t0, dur, sorted_peaks, sorted_labels)

    # 4. Construir arrays de labels y label_ids
    labels = np.full(N, _NOISE_LABEL, dtype=_LABEL_DTYPE)
    label_ids = np.full(N, _NOISE_ID, dtype=np.int32)

    # Ventanas con exactamente 1 match → asignar clase del evento
    single_mask = match_count == 1
    labels[single_mask] = first_label[single_mask]
    for i in np.where(single_mask)[0]:
        lbl = labels[i]
        label_ids[i] = class_map.get(lbl, _NOISE_ID)

    # Ventanas con 2+ matches → ambiguous
    ambiguous_mask = match_count >= 2
    labels[ambiguous_mask] = _AMBIGUOUS_LABEL
    label_ids[ambiguous_mask] = _AMBIGUOUS_ID

    # 5. Estadísticas
    n_noise = int(np.sum(match_count == 0))
    n_ambiguous = int(np.sum(ambiguous_mask))
    stats: Dict = {
        "total": N,
        "noise": n_noise,
        "ambiguous": n_ambiguous,
        "labeled": N - n_noise - n_ambiguous,
        "by_class": {},
    }
    for lbl, lid in class_map.items():
        if lid >= 0:
            count = int(np.sum(label_ids == lid))
            if count > 0:
                stats["by_class"][lbl] = count

    logger.info(
        "Etiquetado completado: noise=%d (%.1f%%), labeled=%d (%.1f%%), "
        "ambiguous=%d (%.1f%%)",
        n_noise,      100 * n_noise      / N,
        stats["labeled"], 100 * stats["labeled"] / N,
        n_ambiguous,  100 * n_ambiguous  / N,
    )
    for lbl, count in sorted(stats["by_class"].items(), key=lambda x: -x[1]):
        logger.info("  clase %-30s : %d ventanas", lbl, count)

    # 6. Actualizar meta con parámetros de etiquetado
    meta["labeling"] = {
        "metadata_csv": str(metadata_csv),
        "n_events_ifo": int(len(sorted_peaks)),
        "class_map": class_map,
        "stats": stats,
    }

    # 7. Guardar NPZ de salida si se solicita
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            X=X,
            t0=t0,
            log_energy=log_energy,
            meta_json=np.array(json.dumps(meta)),
            labels=labels,
            label_ids=label_ids,
            match_count=match_count,
        )
        logger.info("NPZ etiquetado guardado en: %s", output_path)

    return {
        "labels": labels,
        "label_ids": label_ids,
        "match_count": match_count,
        "class_map": class_map,
        "stats": stats,
        "t0": t0,
        "log_energy": log_energy,
        "meta": meta,
    }


def build_temporal_split(
    labels_dict: Dict,
    t0: np.ndarray,
    train_end_pct: float = 0.70,
    val_end_pct: float = 0.85,
    exclude: List[str] = None,
) -> Dict:
    """
    Divide ventanas en bloques temporales GPS para evitar leakage.

    La división se realiza por orden temporal (NO aleatoriamente):
        train : t0 <= percentil ``train_end_pct`` del tiempo
        val   : percentil ``train_end_pct`` < t0 <= percentil ``val_end_pct``
        test  : t0 > percentil ``val_end_pct``

    Las ventanas cuya etiqueta esté en ``exclude`` (por defecto "ambiguous")
    nunca se asignan a ningún split y se devuelven en ``idx_ambiguous``.

    Parámetros
    ----------
    labels_dict : dict
        Resultado de ``build_label_array()`` o dict con clave "labels".
    t0 : np.ndarray (N,) float64
        Tiempos GPS de inicio de ventana.
    train_end_pct : float
        Percentil GPS que delimita el fin del conjunto de entrenamiento.
    val_end_pct : float
        Percentil GPS que delimita el fin del conjunto de validación.
    exclude : list[str], opcional
        Etiquetas que no se asignan a ningún split (default: ["ambiguous"]).

    Retorna
    -------
    dict con:
        idx_train     : np.ndarray (int64) — índices del bloque de entrenamiento
        idx_val       : np.ndarray (int64) — índices del bloque de validación
        idx_test      : np.ndarray (int64) — índices del bloque de test
        idx_excluded  : np.ndarray (int64) — índices de ventanas excluidas
        t_train_end   : float — umbral GPS entre train y val
        t_val_end     : float — umbral GPS entre val y test
        split_stats   : dict  — distribución de clases por split
    """
    if exclude is None:
        exclude = [_AMBIGUOUS_LABEL]

    labels: np.ndarray = labels_dict["labels"]
    N = len(t0)

    # Máscara de exclusión
    exclude_mask = np.zeros(N, dtype=bool)
    for ex_lbl in exclude:
        exclude_mask |= (labels == ex_lbl)

    # Umbrales GPS calculados sobre TODAS las ventanas (incluidas las excluidas)
    t_train_end = float(np.percentile(t0, train_end_pct * 100))
    t_val_end   = float(np.percentile(t0, val_end_pct   * 100))

    # Máscaras temporales sobre ventanas no excluidas
    eligible = ~exclude_mask
    mask_train = eligible & (t0 <= t_train_end)
    mask_val   = eligible & (t0 > t_train_end) & (t0 <= t_val_end)
    mask_test  = eligible & (t0 > t_val_end)

    idx_train    = np.where(mask_train)[0]
    idx_val      = np.where(mask_val)[0]
    idx_test     = np.where(mask_test)[0]
    idx_excluded = np.where(exclude_mask)[0]

    # Estadísticas de distribución de clases por split
    def _class_dist(indices: np.ndarray) -> Dict[str, int]:
        if len(indices) == 0:
            return {}
        unique, counts = np.unique(labels[indices], return_counts=True)
        return {str(u): int(c) for u, c in zip(unique, counts)}

    split_stats = {
        "train":    {"n": len(idx_train),    "classes": _class_dist(idx_train)},
        "val":      {"n": len(idx_val),      "classes": _class_dist(idx_val)},
        "test":     {"n": len(idx_test),     "classes": _class_dist(idx_test)},
        "excluded": {"n": len(idx_excluded), "classes": _class_dist(idx_excluded)},
        "t_train_end": t_train_end,
        "t_val_end":   t_val_end,
    }

    logger.info(
        "Split temporal: train=%d | val=%d | test=%d | excluidas=%d",
        len(idx_train), len(idx_val), len(idx_test), len(idx_excluded),
    )
    logger.info(
        "Umbrales GPS: t_train_end=%.2f  t_val_end=%.2f",
        t_train_end, t_val_end,
    )

    return {
        "idx_train":    idx_train,
        "idx_val":      idx_val,
        "idx_test":     idx_test,
        "idx_excluded": idx_excluded,
        "t_train_end":  t_train_end,
        "t_val_end":    t_val_end,
        "split_stats":  split_stats,
    }


def validate_no_leakage(
    t0: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
) -> bool:
    """
    Verifica que los bloques temporales de train y test no se solapan.

    En un split temporal correcto, todo el conjunto de entrenamiento debe
    ocurrir antes que cualquier ventana del conjunto de test.

    Parámetros
    ----------
    t0 : np.ndarray (N,) float64
        Tiempos GPS de inicio de ventana.
    idx_train : np.ndarray
        Índices de las ventanas de entrenamiento.
    idx_test : np.ndarray
        Índices de las ventanas de test.

    Retorna
    -------
    bool
        True si no hay solapamiento.

    Lanza
    -----
    AssertionError
        Si max(t0[idx_train]) >= min(t0[idx_test]), es decir, si alguna
        ventana de entrenamiento ocurre al mismo tiempo o después que alguna
        ventana de test.
    ValueError
        Si alguno de los conjuntos está vacío.
    """
    if len(idx_train) == 0:
        raise ValueError("idx_train está vacío; no se puede validar leakage.")
    if len(idx_test) == 0:
        raise ValueError("idx_test está vacío; no se puede validar leakage.")

    max_train = float(t0[idx_train].max())
    min_test  = float(t0[idx_test].min())

    assert max_train < min_test, (
        f"Leakage temporal detectado: "
        f"max(t0[train])={max_train:.4f} >= min(t0[test])={min_test:.4f}. "
        "El split no es temporalmente puro. Revisar la lógica de split."
    )

    logger.info(
        "Validación leakage OK: max(t0_train)=%.4f < min(t0_test)=%.4f  (gap=%.4f s)",
        max_train, min_test, min_test - max_train,
    )
    return True


# ---------------------------------------------------------------------------
# Helpers del script __main__
# ---------------------------------------------------------------------------

def _scale_to_str(scale: float) -> str:
    """Convierte una escala numérica al formato de nombre de archivo: 1.0 → '1p0'."""
    return str(scale).replace(".", "p")


def _find_npz(npz_dir: Path, ifo: str, epoch: str, scale: float) -> Optional[Path]:
    """
    Busca el NPZ de run02 para una combinación (ifo, epoch, scale) dentro de
    npz_dir usando glob recursivo, tolerando distintas estructuras de carpetas.
    """
    scale_str = _scale_to_str(scale)
    pattern = f"*{ifo}*{epoch}*scale_{scale_str}*run02*.npz"
    matches = list(npz_dir.rglob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "Múltiples NPZ encontrados para %s/%s/%.2fs: %s. Usando el primero.",
            ifo, epoch, scale, matches,
        )
    return matches[0]


def _print_consolidated_report(all_results: List[Dict]) -> None:
    """Imprime un resumen consolidado de todos los NPZ procesados."""
    logger.info("=" * 70)
    logger.info("REPORTE CONSOLIDADO")
    logger.info("=" * 70)

    total_windows = 0
    total_noise = 0
    total_labeled = 0
    total_ambiguous = 0

    for r in all_results:
        tag = r["tag"]
        s = r["stats"]
        split = r.get("split")

        logger.info(
            "[%s]  total=%d | noise=%d (%.1f%%) | labeled=%d (%.1f%%) | "
            "ambiguous=%d (%.1f%%)",
            tag, s["total"],
            s["noise"],     100 * s["noise"]     / s["total"],
            s["labeled"],   100 * s["labeled"]   / s["total"],
            s["ambiguous"], 100 * s["ambiguous"] / s["total"],
        )
        if split:
            logger.info(
                "  → split: train=%d | val=%d | test=%d | excluidas=%d",
                split["split_stats"]["train"]["n"],
                split["split_stats"]["val"]["n"],
                split["split_stats"]["test"]["n"],
                split["split_stats"]["excluded"]["n"],
            )

        total_windows   += s["total"]
        total_noise     += s["noise"]
        total_labeled   += s["labeled"]
        total_ambiguous += s["ambiguous"]

    logger.info("-" * 70)
    logger.info(
        "TOTAL  %d ventanas | noise=%d | labeled=%d | ambiguous=%d",
        total_windows, total_noise, total_labeled, total_ambiguous,
    )
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Punto de entrada como script
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Procesa todos los NPZ run02 de una carpeta y genera NPZ etiquetados.

    Para cada combinación (ifo × epoch × scale) busca el NPZ correspondiente,
    aplica el etiquetado, valida el split temporal y guarda el resultado.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="E2.2 — Etiquetado de ventanas del dataset v2 con Gravity Spy."
    )
    parser.add_argument(
        "--npz_dir", required=True,
        help="Directorio raíz con los NPZ de run02 (búsqueda recursiva).",
    )
    parser.add_argument(
        "--metadata", required=True,
        help="CSV con metadatos de glitches (trainingset_v1d1_metadata.csv).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directorio donde guardar los NPZ etiquetados.",
    )
    parser.add_argument(
        "--scales", nargs="+", type=float, default=[0.25, 1.0, 4.0],
        help="Escalas de ventana a procesar (default: 0.25 1.0 4.0).",
    )
    parser.add_argument(
        "--ifos", nargs="+", default=["H1", "L1"],
        help="Detectores a procesar (default: H1 L1).",
    )
    parser.add_argument(
        "--epochs", nargs="+", default=["O3a", "O3b"],
        help="Épocas a procesar (default: O3a O3b).",
    )
    parser.add_argument(
        "--train_end_pct", type=float, default=0.70,
        help="Percentil GPS para fin de train (default: 0.70).",
    )
    parser.add_argument(
        "--val_end_pct", type=float, default=0.85,
        help="Percentil GPS para fin de val (default: 0.85).",
    )
    args = parser.parse_args()

    npz_dir    = Path(args.npz_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []

    for ifo in args.ifos:
        for epoch in args.epochs:
            for scale in args.scales:
                tag = f"{ifo}_{epoch}_scale_{_scale_to_str(scale)}"
                logger.info("Procesando: %s …", tag)

                # Localizar NPZ de entrada
                npz_path = _find_npz(npz_dir, ifo, epoch, scale)
                if npz_path is None:
                    logger.warning("[%s] NPZ no encontrado en '%s'. Saltando.", tag, npz_dir)
                    continue

                # Ruta de salida
                out_name = f"labels_{ifo}_{epoch}_scale_{_scale_to_str(scale)}_run02.npz"
                out_path = output_dir / out_name

                # Etiquetado
                result = build_label_array(
                    npz_path=str(npz_path),
                    metadata_csv=args.metadata,
                    output_path=str(out_path),
                )

                # Split temporal
                split = build_temporal_split(
                    labels_dict=result,
                    t0=result["t0"],
                    train_end_pct=args.train_end_pct,
                    val_end_pct=args.val_end_pct,
                )

                # Validación de no leakage
                try:
                    validate_no_leakage(
                        result["t0"],
                        split["idx_train"],
                        split["idx_test"],
                    )
                except (AssertionError, ValueError) as exc:
                    logger.error("[%s] %s", tag, exc)

                all_results.append({
                    "tag":   tag,
                    "stats": result["stats"],
                    "split": split,
                })

    _print_consolidated_report(all_results)


if __name__ == "__main__":
    main()
