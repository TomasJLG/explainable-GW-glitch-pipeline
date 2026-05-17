# Explainable Pipeline for Glitch Detection and Classification in LIGO O3 Data
# Pipeline Explicable para la Detección y Clasificación de Glitches en Datos LIGO O3

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![LIGO](https://img.shields.io/badge/Data-LIGO%20O3-purple)
![AUROC](https://img.shields.io/badge/AUROC-0.85-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-69.4%25-yellow)

---

## Description / Descripción

**[EN]**
Three-module pipeline for detecting and classifying transient instrumental noise artefacts
(*glitches*) in LIGO H1/L1 interferometer data from the O3 observing run (2019–2020).
The system combines a data generation stage (M0), a convolutional autoencoder for
unsupervised anomaly detection (M1), and an interpretable ANFIS fuzzy classifier (M2).

**[ES]**
Pipeline de tres módulos para detectar y clasificar ruidos instrumentales transitorios
(*glitches*) en datos de los interferómetros LIGO H1 y L1 durante la campaña O3 (2019–2020).
El sistema combina una etapa de generación de datos (M0), un autoencoder convolucional
para detección no supervisada (M1) y un clasificador difuso ANFIS interpretable (M2).

---

## Results / Resultados

| Module | Metric | Value |
|--------|--------|-------|
| M1 — GlitchAE | AUROC (500 labelled samples) | **0.85** |
| M1 — GlitchAE | TPR @ FPR=1% | **42%** |
| M2 — ANFIS v3 | Accuracy (4-class) | **69.4%** |
| M2 — ANFIS v3 | Macro-F1 | **0.6815** |
| M2 — ANFIS Stage 1 | Loud vs Rest F1 | **0.91** |

---

## Pipeline Architecture / Arquitectura del Pipeline

```
GWOSC strain (HDF5)
        │
        ▼
  ┌─────────────┐
  │     M0      │  Whitening · Bandpass 20–1700 Hz · Q-transform
  │  Generator  │  → 128×128 time-frequency windows (NPZ)
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │     M1      │  Convolutional Autoencoder (17.9M params)
  │  GlitchAE   │  MSE reconstruction error → anomaly score
  └─────────────┘
        │  latent vector (32-D) + metadata features
        ▼
  ┌─────────────┐
  │     M2      │  Hierarchical ANFIS (Takagi-Sugeno)
  │   ANFIS v3  │  Stage 1: Loud vs Rest  →  Stage 2: Burst / Scatter / Other
  └─────────────┘
```

---

## Modules / Módulos

### M0 — NPZ Dataset Generator / Generador de Datasets NPZ

**[EN]**
Downloads strain HDF5 blocks from GWOSC, applies preprocessing (whitening, bandpass
20–1700 Hz, 60/120/180 Hz notch) and computes the Q-transform to produce 128×128
time-frequency windows packed into NPZ files. Resumable via JSON checkpoints.
Supports 4 IFO×epoch combinations (H1/O3a, H1/O3b, L1/O3a, L1/O3b).

**[ES]**
Descarga bloques de strain HDF5 desde GWOSC, aplica preprocesado (blanqueado, filtro
pasa-banda 20–1700 Hz, notch en 60/120/180 Hz) y calcula la Q-transform para producir
ventanas tiempo-frecuencia de 128×128 píxeles empaquetadas en archivos NPZ.
Reanudable mediante checkpoints JSON.

```bash
# Nominal training dataset / Dataset nominal de entrenamiento
python run02_v2_generator.py

# Labelled Gravity Spy dataset (M1 evaluation) / Dataset etiquetado Gravity Spy
python run03_bulk_generator.py

# Minority-class dataset / Dataset clases minoritarias
python run03_minority_generator.py
```

### M1 — GlitchAE Autoencoder / Autoencoder GlitchAE

**[EN]**
Convolutional autoencoder (4 Conv2d blocks, latent_dim=32, 17.9M parameters) trained
exclusively on nominal windows. The MSE reconstruction error acts as the anomaly score.
Global P1/P99 normalisation computed on the training set only; temporal split at P80 of
GPS t0 to prevent data leakage.

**[ES]**
Autoencoder convolucional (4 bloques Conv2d, latent_dim=32, 17,9M parámetros) entrenado
exclusivamente sobre ventanas nominales. El error de reconstrucción MSE actúa como
puntuación de anomalía. Normalización P1/P99 global calculada solo sobre el conjunto de
entrenamiento; split temporal en P80 del GPS t0 para evitar data leakage.

```bash
# Train v3 on run02v2 / Entrenamiento v3 sobre run02v2
python m1_train_v3.py

# Evaluate on labelled run03 / Evaluación sobre run03 etiquetado
python m1_eval_run03.py
```

### M2 — Hierarchical ANFIS Classifier / Clasificador ANFIS Jerárquico

**[EN]**
Two-stage Takagi-Sugeno fuzzy inference system. Stage 1 separates Loud from the rest
(ra=0.15, 7 rules); Stage 2 classifies Burst/Scatter/Other (ra=0.10, 8 rules).
Independent P2/P98 normalisation per IFO×epoch group. Hybrid LSE+Adam learning.
Rule centres initialised via subtractive clustering (Chiu 1994).

**[ES]**
Sistema de inferencia difusa Takagi-Sugeno de dos etapas. Stage 1 separa Loud del
resto (ra=0,15, 7 reglas); Stage 2 clasifica Burst/Scatter/Other (ra=0,10, 8 reglas).
Normalización P2/P98 independiente por grupo IFO×época. Aprendizaje híbrido LSE+Adam.
Centros de reglas inicializados mediante clustering sustractivo (Chiu 1994).

```bash
# Feature extraction M1→M2 / Extracción de features M1→M2
python m2_feature_extractor.py

# Hierarchical ANFIS v3 training / Entrenamiento ANFIS jerárquico v3
python m2_anfis_v3.py

# End-to-end pipeline evaluation / Evaluación pipeline E2E
python m2_pipeline_eval.py
```

---

## Installation / Instalación

```bash
git clone https://github.com/TomasJLG/explainable-GW-glitch-pipeline.git
cd explainable-GW-glitch-pipeline
pip install -r requirements.txt
```

**Main dependencies / Dependencias principales:** `torch>=2.0`, `gwpy>=3.0`, `gwosc>=0.7`,
`numpy`, `scipy`, `scikit-learn`, `matplotlib`.

---

## Repository Structure / Estructura del Repositorio

```
.
├── run02_v2_generator.py        # M0: nominal dataset (random GPS sampling)
├── run03_bulk_generator.py      # M0: labelled Gravity Spy dataset
├── run03_minority_generator.py  # M0: minority-class dataset
├── dataset/                     # Canonical generator module
│
├── m1_anomaly/                  # M1: architecture + dataloader + train + eval
│   ├── m1_autoencoder.py        #     GlitchAE (Conv2d encoder-decoder)
│   ├── m1_dataloader.py         #     NPZ loading, temporal split, normalisation
│   ├── m1_train.py              #     Training (CLI)
│   └── m1_eval.py               #     AUROC, AUPRC, TPR@FPR1%, plots
├── m1_train_v3.py               # M1: v3 training on run02v2
├── m1_eval_run03.py             # M1: evaluation on labelled run03
├── m1_v3_outputs/               # M1: training curves, ROC, score distributions
│
├── m2_anfis_v3.py               # M2: hierarchical ANFIS v3 (Stage1 + Stage2)
├── m2_anfis/                    # M2: modular ANFIS package
├── m2_feature_extractor.py      # M2: encoder features + PCA + CSV metadata → 6-D vector
├── m2_pipeline_eval.py          # M2: end-to-end pipeline evaluation
├── m1m2_retrain_full.py         # Full M1v4 + M2v2 retraining script
├── m2_outputs_v3/               # M2: confusion matrix, membership functions, training curves
│
├── figures/                     # Architecture diagrams
├── configs/dataset.yaml         # Canonical pipeline parameters
├── requirements.txt
└── LICENSE
```

---

## Data / Datos

**[EN]**
LIGO strain data are publicly available and downloaded automatically from
[GWOSC](https://gwosc.org) via `gwosc` and `gwpy`. Gravity Spy labels are available
on [Zenodo](https://zenodo.org). NPZ files, model weights (`.pt`) and original datasets
are **not included** in this repository due to their size (>50 MB per file).

**[ES]**
Los datos de strain de LIGO son públicos y se descargan automáticamente desde
[GWOSC](https://gwosc.org) mediante `gwosc` y `gwpy`. Las etiquetas de Gravity Spy
están disponibles en [Zenodo](https://zenodo.org). Los archivos NPZ, pesos de modelo
(`.pt`) y datasets originales **no se incluyen** en este repositorio por su tamaño
(>50 MB por archivo).

---

## Authors / Autores

- **Tomás Legal** — main development / desarrollo principal
- **Daniel Tanco** — project collaborator / compañero de proyecto
- **Miguel Peralias** — project collaborator / compañero de proyecto

*Proyecto Intermodular — Ciclo Formativo de Grado Superior en Inteligencia Artificial y Big Data, 2025–2026*

---

## License / Licencia

[MIT License](LICENSE)
