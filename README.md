# Pipeline explicable para la detección y clasificación de glitches en datos LIGO O3
# Explainable Pipeline for Glitch Detection and Classification in LIGO O3 Data

---

## Descripción / Description

**[ES]**
Pipeline de tres módulos para detectar y clasificar ruidos instrumentales transitorios
(*glitches*) en datos de los interferómetros LIGO H1 y L1 durante la campaña O3 (2019–2020).
El sistema combina una etapa de generación de datos (M0), un autoencoder convolucional
para detección no supervisada (M1) y un clasificador difuso ANFIS interpretable (M2).

**[EN]**
Three-module pipeline for detecting and classifying transient instrumental noise artefacts
(*glitches*) in LIGO H1/L1 interferometer data from the O3 observing run (2019–2020).
The system combines a data generation stage (M0), a convolutional autoencoder for
unsupervised anomaly detection (M1), and an interpretable ANFIS fuzzy classifier (M2).

---

## Módulos / Modules

### M0 — Generador de datasets NPZ / NPZ Dataset Generator

**[ES]**
Descarga bloques de strain HDF5 desde GWOSC, aplica preprocesado (blanqueado, filtro
pasa-banda 20–1700 Hz, notch en 60/120/180 Hz) y calcula la Q-transform para producir
ventanas tiempo-frecuencia de 128×128 píxeles empaquetadas en archivos NPZ.
Reanudable mediante checkpoints JSON. Compatible con 4 combinaciones IFO×época
(H1/O3a, H1/O3b, L1/O3a, L1/O3b).

**[EN]**
Downloads strain HDF5 blocks from GWOSC, applies preprocessing (whitening, bandpass
20–1700 Hz, 60/120/180 Hz notch) and computes the Q-transform to produce 128×128
time-frequency windows packed into NPZ files. Resumable via JSON checkpoints.
Supports 4 IFO×epoch combinations (H1/O3a, H1/O3b, L1/O3a, L1/O3b).

```bash
# Dataset nominal de entrenamiento / Nominal training dataset
python run02_v2_generator.py

# Dataset etiquetado Gravity Spy (evaluación M1) / Labelled Gravity Spy dataset (M1 eval)
python run03_bulk_generator.py

# Dataset clases minoritarias / Minority-class dataset
python run03_minority_generator.py
```

### M1 — Autoencoder GlitchAE

**[ES]**
Autoencoder convolucional (4 bloques Conv2d, latent_dim=32, 17,9M parámetros) entrenado
exclusivamente sobre ventanas nominales. El error de reconstrucción MSE actúa como
puntuación de anomalía. Resultado: AUROC = 0,85 sobre 500 muestras etiquetadas de run03.

**[EN]**
Convolutional autoencoder (4 Conv2d blocks, latent_dim=32, 17.9M parameters) trained
exclusively on nominal windows. The MSE reconstruction error acts as the anomaly score.
Result: AUROC = 0.85 on 500 labelled run03 samples.

```bash
# Entrenamiento v3 / v3 Training
python m1_train_v3.py

# Evaluación sobre run03 / Evaluation on run03
python m1_eval_run03.py
```

### M2 — Clasificador ANFIS jerárquico / Hierarchical ANFIS Classifier

**[ES]**
Sistema de inferencia difusa Takagi-Sugeno de dos etapas: Stage 1 separa Loud del
resto (ra=0,15, 7 reglas); Stage 2 clasifica Burst/Scatter/Other (ra=0,10, 8 reglas).
Normalización P2/P98 independiente por IFO×época. Aprendizaje híbrido LSE+Adam.
Resultado: accuracy=0,694, macro-F1=0,6815.

**[EN]**
Two-stage Takagi-Sugeno fuzzy inference system: Stage 1 separates Loud from the rest
(ra=0.15, 7 rules); Stage 2 classifies Burst/Scatter/Other (ra=0.10, 8 rules).
Independent P2/P98 normalisation per IFO×epoch. Hybrid LSE+Adam learning.
Result: accuracy=0.694, macro-F1=0.6815.

```bash
# Extracción de features M1→M2 / Feature extraction M1→M2
python m2_feature_extractor.py

# Entrenamiento ANFIS v3 / ANFIS v3 training
python m2_anfis_v3.py

# Evaluación pipeline E2E / E2E pipeline evaluation
python m2_pipeline_eval.py
```

---

## Estructura del repositorio / Repository Structure

```
.
├── run02_v2_generator.py        # M0: dataset nominal (GPS aleatorio / random GPS)
├── run03_bulk_generator.py      # M0: dataset etiquetado Gravity Spy
├── run03_minority_generator.py  # M0: clases minoritarias / minority classes
├── dataset/                     # Módulo generador canónico / Canonical generator module
│
├── m1_anomaly/                  # M1: arquitectura + dataloader + train + eval
│   ├── m1_autoencoder.py        #     GlitchAE (Conv2d encoder-decoder)
│   ├── m1_dataloader.py         #     Carga NPZ, split temporal, normalización
│   ├── m1_train.py              #     Entrenamiento v1 (CLI)
│   └── m1_eval.py               #     AUROC, AUPRC, TPR@FPR1%, plots
├── m1_train_v3.py               # M1: entrenamiento v3 sobre run02v2
├── m1_eval_run03.py             # M1: evaluación sobre run03 etiquetado
├── m1_v3_outputs/               # M1: pesos JSON, curvas, ROC (sin .pt — >50 MB)
│
├── m2_anfis.py                  # M2: ANFIS plano v2 (4 clases)
├── m2_anfis_v3.py               # M2: ANFIS jerárquico v3 (Stage1 + Stage2)
├── m2_anfis/                    # M2: paquete modular ANFIS
├── m2_feature_extractor.py      # M2: extracción de features (encoder + CSV)
├── m2_pipeline_eval.py          # M2: evaluación E2E del pipeline
├── m1m2_retrain_full.py         # Reentrenamiento integral M1v4 + M2v2
├── m2_outputs/                  # M2: plots v1 + pipeline eval
├── m2_outputs_v2/               # M2: plots v2
├── m2_outputs_v3/               # M2: plots v3 + group_norms.json
│
├── figures/                     # Diagramas de arquitectura / Architecture diagrams
├── docs/                        # Informes DOCX / DOCX reports
├── Memoria_PI_LIGO.docx         # Memoria del Proyecto Intermodular
├── configs/dataset.yaml         # Parámetros canónicos / Canonical parameters
├── requirements.txt
└── LICENSE
```

---

## Requisitos / Requirements

```bash
pip install -r requirements.txt
```

Dependencias principales / Main dependencies: `torch>=2.0`, `gwpy>=3.0`, `gwosc>=0.7`,
`numpy`, `scipy`, `scikit-learn`, `matplotlib`, `python-docx`.

---

## Datos / Data

**[ES]**
Los datos de strain de LIGO son públicos y se descargan automáticamente desde
[GWOSC](https://gwosc.org) mediante `gwosc` y `gwpy`. Las etiquetas de Gravity Spy
están disponibles en [Zenodo](https://zenodo.org). Los archivos NPZ, pesos de modelo
(.pt) y datasets originales **no se incluyen** en este repositorio por su tamaño
(>50 MB por archivo).

**[EN]**
LIGO strain data are publicly available and downloaded automatically from
[GWOSC](https://gwosc.org) via `gwosc` and `gwpy`. Gravity Spy labels are available
on [Zenodo](https://zenodo.org). NPZ files, model weights (.pt) and original datasets
are **not included** in this repository due to their size (>50 MB per file).

---

## Autores / Authors

- **Tomás Legal** — desarrollo principal / main development
- **Daniel Tanco** — compañero de proyecto / project collaborator
- **Miguel Peralias** — compañero de proyecto / project collaborator

Proyecto Intermodular — Ciclo Formativo de Grado Superior en IA y Big Data, 2025–2026

---

## Licencia / License

MIT License — ver [LICENSE](LICENSE) / see [LICENSE](LICENSE)
