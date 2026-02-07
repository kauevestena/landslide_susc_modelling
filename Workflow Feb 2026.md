# Workflow — February 2026

> A comprehensive, code-derived description of how the landslide susceptibility
> modelling pipeline currently operates, stage by stage.  
> Written from direct inspection of the source code and produced artifacts.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Entry Point & Orchestration](#2-entry-point--orchestration)
3. [Stage 1 — Input Loading & Validation](#3-stage-1--input-loading--validation)
4. [Stage 2 — Preprocessing (`process_area`)](#4-stage-2--preprocessing-process_area)
   - 4.1 [DTM Conditioning](#41-dtm-conditioning)
   - 4.2 [Terrain Feature Derivation](#42-terrain-feature-derivation)
   - 4.3 [Orthophoto Reprojection & Normalisation](#43-orthophoto-reprojection--normalisation)
   - 4.4 [Land-Use / Land-Cover (LULC)](#44-land-use--land-cover-lulc)
   - 4.5 [Feature Stack Assembly & Normalisation](#45-feature-stack-assembly--normalisation)
   - 4.6 [Ground-Truth Remapping](#46-ground-truth-remapping)
5. [Stage 3 — Dataset Preparation (Tiling & Splitting)](#5-stage-3--dataset-preparation-tiling--splitting)
   - 5.1 [Mixed-Domain Merge](#51-mixed-domain-merge)
   - 5.2 [Spatial Block Splitting](#52-spatial-block-splitting)
   - 5.3 [Tile Generation & Class Balancing](#53-tile-generation--class-balancing)
   - 5.4 [Soft Label Smoothing](#54-soft-label-smoothing)
   - 5.5 [SMOTE Synthetic Oversampling](#55-smote-synthetic-oversampling)
6. [Stage 4 — Model Training](#6-stage-4--model-training)
   - 6.1 [Architecture](#61-architecture)
   - 6.2 [Loss Functions](#62-loss-functions)
   - 6.3 [Training Loop & Calibration](#63-training-loop--calibration)
   - 6.4 [Threshold Selection & Visualisations](#64-threshold-selection--visualisations)
7. [Stage 5 — Inference](#7-stage-5--inference)
   - 7.1 [Sliding-Window Prediction](#71-sliding-window-prediction)
   - 7.2 [Blending & CRF Post-Processing](#72-blending--crf-post-processing)
   - 7.3 [MC Dropout Uncertainty](#73-mc-dropout-uncertainty)
   - 7.4 [Susceptibility Score Computation](#74-susceptibility-score-computation)
   - 7.5 [Output Deliverables](#75-output-deliverables)
8. [Stage 6 — Standalone Evaluation](#8-stage-6--standalone-evaluation)
9. [Feature Channel Inventory](#9-feature-channel-inventory)
10. [Current Model Performance](#10-current-model-performance)
11. [Key Configuration Levers](#11-key-configuration-levers)
12. [Resumability & Artifact Lifecycle](#12-resumability--artifact-lifecycle)
13. [File Map](#13-file-map)

---

## 1. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  config.yaml  ←  Single source of truth for all parameters      │
│  inputs.py    ←  Absolute paths to raw GeoTIFFs per area        │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │   src/main_pipeline.py (orchestrator) │
         └───────┬───────┬───────┬───────┬───────┘
                 │       │       │       │
        ┌────────┘  ┌────┘  ┌────┘  ┌────┘
        ▼           ▼       ▼       ▼
  Preprocess     Tile/   Train    Infer
  (per area)     Split   Model    & Export
        │           │       │       │
        ▼           ▼       ▼       ▼
  artifacts/     artifacts/ artifacts/ outputs/
  derived/       tiles/    experiments/ *.tif
  metadata/      labels/               model_card.md
                 splits/
```

The pipeline is invoked with:

```bash
.venv/bin/python -m src.main_pipeline              # resume from last checkpoint
.venv/bin/python -m src.main_pipeline --force_recreate  # regenerate everything
```

Every stage checks for pre-existing artifacts on disk and **skips** if they
are already present (unless `--force_recreate` is passed).

---

## 2. Entry Point & Orchestration

`src/main_pipeline.py → main()` performs the following in order:

1. **Loads** `config.yaml` via PyYAML.
2. **Seeds** random generators (Python `random`, NumPy, and later PyTorch)
   using `reproducibility.seed` (currently `42`).
3. **Preprocessing** — calls `preprocess_data(config)` which iterates
   over `train` and `test` splits, producing feature stacks, valid masks,
   and aligned ground-truth rasters under `artifacts/derived/{train,test}/`.
4. **Dataset preparation** — depending on the boolean
   `dataset.use_mixed_domain` (currently **true**):
   - **True**: calls `prepare_mixed_domain_dataset()` from
     `src/prepare_mixed_domain_dataset.py`, which merges both areas into a
     single virtual raster, then tiles and splits with spatial blocking.
   - **False**: calls `prepare_dataset()` (internal), which tiles only the
     training area.
5. **Training** — calls `train_model()` from `src/train.py`. Produces
   `best_model.pth`, calibrators, metrics, and plots under
   `artifacts/experiments/`.
6. **Inference** — calls `run_inference()` from `src/inference.py`.
   Produces calibrated GeoTIFF outputs under `outputs/`.

---

## 3. Stage 1 — Input Loading & Validation

`load_input_paths()` dynamically imports the module specified in
`config.yaml → inputs.module` (currently the file `inputs.py` at project
root). For each declared split it resolves three attribute names:

| Split   | DTM attribute     | Orthophoto attribute  | Ground-truth attribute  |
|---------|-------------------|-----------------------|-------------------------|
| `train` | `DTM_train`       | `ortophoto_train`     | `ground_truth_train`    |
| `test`  | `DTM_test`        | `ortophoto_test`      | `ground_truth_test`     |

All paths point to local GeoTIFFs stored under `/home/kaue/data/landslide/`.
Existence is validated at startup; a `FileNotFoundError` is raised immediately
for any missing file.

The **DTM acts as the spatial reference grid** — all other rasters
(orthophoto, ground truth, external LULC) are reprojected and resampled
onto its CRS, transform, and pixel dimensions.

---

## 4. Stage 2 — Preprocessing (`process_area`)

Called once per area (first `train`, then `test`). Normalisation statistics
are computed on the **training** area and reused for the test area to prevent
data leakage.

### 4.1 DTM Conditioning

1. **NoData fill** — nearest-neighbour propagation via `scipy.ndimage.distance_transform_edt`.
2. **Sink fill** — approximate morphological closing
   (`maximum_filter` → `minimum_filter`) with a 5 × 5 kernel.
3. **Gaussian smoothing** — σ = 1.0 pixel applied to the conditioned DEM.

### 4.2 Terrain Feature Derivation

`compute_dem_features()` produces **14 terrain-derived channels** from the
conditioned DEM via pure NumPy/SciPy operations:

| # | Feature                      | Description                                   |
|---|------------------------------|-----------------------------------------------|
| 1 | `dtm_elevation`              | Conditioned elevation values                  |
| 2 | `slope_deg`                  | Slope angle in degrees                        |
| 3 | `aspect_sin`                 | Sine of aspect angle                          |
| 4 | `aspect_cos`                 | Cosine of aspect angle                        |
| 5 | `general_curvature`          | Laplacian (r + t)                             |
| 6 | `plan_curvature`             | Contour curvature                             |
| 7 | `profile_curvature`          | Down-slope curvature                          |
| 8 | `tpi`                        | Topographic Position Index (9×9 window)       |
| 9 | `tri`                        | Terrain Ruggedness Index (9×9 window)         |
|10 | `log_flow_accumulation`      | log₁₊(D8 flow accumulation × cell area)       |
|11 | `twi`                        | Topographic Wetness Index                     |
|12 | `spi`                        | Stream Power Index                            |
|13 | `sti`                        | Sediment Transport Index                      |
|14 | `log_distance_to_drainage`   | log₁₊(Euclidean distance to drainage network) |

D8 flow accumulation is computed in-house via a simple loop over pixels
sorted by descending elevation, tracing steepest-descent paths.
The drainage network is defined as pixels with contributing area ≥ 5 000 m².

### 4.3 Orthophoto Reprojection & Normalisation

The orthophoto raster is reprojected onto the DTM grid using **bilinear**
resampling, producing **3 RGB bands**. Per-band z-score normalisation is
applied constrained to valid pixels.

### 4.4 Land-Use / Land-Cover (LULC)

The current configuration (`external_lulc.enabled: true`,
`source: worldcover`) downloads **ESA WorldCover 2021** data via direct
S3 tile fetch. The workflow:

1. Extract the bounding box of the DTM in EPSG:4326.
2. Identify intersecting 3° × 3° WorldCover tiles on the ESA S3 bucket.
3. Download, mosaic, and cache the raw 10 m classification raster.
4. Reproject to the DTM grid using nearest-neighbour.
5. One-hot encode the WorldCover classes → produces **11 binary channels**
   (one per WorldCover class found in the study area).

A fallback path exists: if `external_lulc.enabled` is `false`, the pipeline
clusters the normalised orthophoto into k pseudo land-cover classes via
K-Means (configurable cluster count, historically 6).

### 4.5 Feature Stack Assembly & Normalisation

`build_feature_stack()` concatenates the channels in this order:

1. Terrain features (14 channels, gated by individual boolean toggles)
2. Normalised orthophoto bands (3 channels)
3. LULC one-hot channels (11 channels from WorldCover)

**Total: 28 channels** (confirmed by the stored normalization stats and
model checkpoint `in_channels: 28`).

`apply_normalization()` then z-score normalises every channel over valid
pixels. For the **training** area the statistics (mean, std per channel) are
**computed and saved** to `artifacts/metadata/normalization_stats.json`.
For the **test** area these same statistics are **loaded and reused**.

### 4.6 Ground-Truth Remapping

The original ground-truth raster is encoded as:
- 0 = invalid / background
- 1 = Low risk
- 2 = Moderate risk
- 3 = High risk

A critical remapping translates these to the model's expected classes:

| Ground-truth value | Model class | Meaning      |
|--------------------|-------------|--------------|
| 0                  | 255 (ignore)| No data      |
| 1                  | 0           | Low risk     |
| 2                  | 1           | Moderate risk|
| 3                  | 2           | High risk    |

Boundary pixels within 2 pixels of a class transition are also set to 255
(ignore) to avoid ambiguous labels at borders.

---

## 5. Stage 3 — Dataset Preparation (Tiling & Splitting)

### 5.1 Mixed-Domain Merge

Because `dataset.use_mixed_domain` is **true**, the pipeline calls
`prepare_mixed_domain_dataset()`, which:

1. Loads both the training and test feature stacks from
   `artifacts/derived/{train,test}/`.
2. **Vertically concatenates** them (train on top, test on bottom),
   zero-padding the narrower raster to the maximum width.
3. Saves the merged mosaic to `artifacts/derived/merged/`:
   - `merged_features.tif` — (28 × 20 722 × 5 964) float32
   - `merged_labels.tif` — (1 × 20 722 × 5 964) uint8
   - `merged_mask.tif` — (1 × 20 722 × 5 964) uint8
   - `merged_metadata.json` — records the row ranges for each source area.

The merged raster is:
- Train area: rows 0–1 573 (height 1 574, width 5 964)
- Test area: rows 1 574–20 721 (height 19 148, width 5 964)

### 5.2 Spatial Block Splitting

To avoid **spatial data leakage** (nearby pixels ending up in both
train and test), the mixed-domain tiling applies a **spatial block**
strategy:

1. Generate all valid candidate tile positions (256 × 256 tiles with
   128 px stride, requiring ≥ 40 % valid pixels).
2. Label each tile by whether its centre falls in the train-area or
   test-area row range.
3. For each source area independently:
   - Sort tiles by (y, x).
   - Group adjacent tiles into spatial **blocks** (minimum separation =
     5 tiles = 640 pixels).
   - Shuffle blocks, then allocate whole blocks to train / val / test
     at the configured ratios (65 % / 15 % / 20 %).
4. Combine the split lists from both areas, shuffle, and save.

Current tile counts (from `dataset_summary.json`):

| Split | Tiles | From train area | From test area |
|-------|-------|-----------------|----------------|
| train | 626   | 59              | 567            |
| val   | 111   | 10              | 101            |
| test  | 134   | 21              | 113            |

Each tile is saved as:
- `.npy` under `artifacts/tiles/{split}/` (fast loading for training)
- `.tif` under `artifacts/tiles/geotiff/{split}/` (GIS inspection)

### 5.3 Tile Generation & Class Balancing

After the initial split, two oversampling mechanisms address the severe
class imbalance (Low ≈ 66 %, Medium ≈ 34 %, High ≈ 0.7 % of pixels in the
mixed dataset):

1. **Tile duplication** — Tiles rich in the configured `oversample_class`
   (class 1 = Medium) that have > 5 % coverage are duplicated to push the
   class fraction towards `oversample_target_fraction` (7.5 %).

2. **SMOTE** — If duplication is insufficient, the pipeline applies SMOTE
   (from `imbalanced-learn`) at the pixel level. It samples up to 50 class-1-rich
   tiles, flattens their pixels, generates synthetic pixel feature vectors,
   reshapes them into new tiles, and saves them with a `train_smote_*` prefix.

### 5.4 Soft Label Smoothing

When `preprocessing.label_smoothing.enabled` is **true** (current setting),
`src/soft_labels.py` converts the hard per-pixel class label into a
probability distribution over the 3 classes.

The current method is **ordinal smoothing** with `alpha = 0.2`:

| True class    | P(Low) | P(Medium) | P(High) |
|---------------|--------|-----------|---------|
| 0 (Low)       | 0.80   | 0.20      | 0.00    |
| 1 (Medium)    | 0.10   | 0.80      | 0.10    |
| 2 (High)      | 0.00   | 0.20      | 0.80    |

Soft labels are stored as `float32` arrays of shape
`(3, tile_H, tile_W)` instead of `(tile_H, tile_W)`.

> **Important**: This is only active in the **single-area** prepare_dataset
> path. The mixed-domain path (`prepare_mixed_domain_dataset`) currently
> saves **hard labels** (`uint8`). The training code auto-detects whether
> labels are 2D (hard) or 3D (soft) and selects the appropriate loss function.

### 5.5 SMOTE Synthetic Oversampling

If `dataset.use_smote` is true (it is) and the minority class fraction
still falls below `oversample_target_fraction` after tile duplication,
pixel-level SMOTE is applied:

1. Up to 50 tiles containing class 1 are sampled.
2. Valid pixels are flattened into (n_pixels, 28) feature vectors.
3. SMOTE generates synthetic minority-class pixel vectors.
4. Synthetic pixels are reshaped into (28, 256, 256) tiles and saved.

---

## 6. Stage 4 — Model Training

### 6.1 Architecture

- **Backbone**: U-Net from `segmentation-models-pytorch`.
- **Encoder**: `efficientnet-b4` with ImageNet-pretrained weights (first
  convolutional layer is automatically adapted from 3 → 28 input channels
  by SMP).
- **Spatial Attention**: A `SpatialAttentionModule` (channel-pooled max + avg
  → 7×7 conv → sigmoid) wraps the deepest encoder features before decoding.
  This is a custom `UnetWithAttention` wrapper.
- **Dropout**: A `Dropout2d(p=0.4)` layer is prepended to the segmentation
  head (also enables MC-Dropout uncertainty at inference time).
- **Output**: 3-class softmax segmentation logits (Low / Medium / High).

### 6.2 Loss Functions

The loss function is selected by a hierarchy in the config:

| `use_focal_loss` | `use_ordinal_loss` | Resulting Loss                                  |
|-------------------|--------------------|------------------------------------------------|
| true              | true               | **CombinedOrdinalLoss** (current)               |
| true              | false              | FocalDiceLoss                                   |
| —                 | —                  | SoftDiceCrossEntropyLoss (if soft labels) or DiceCrossEntropyLoss |

**CombinedOrdinalLoss** (the currently active loss) is:

$$\mathcal{L} = \underbrace{0.7 \cdot \mathcal{L}_{\text{Focal}} + 0.3 \cdot \mathcal{L}_{\text{Dice}}}_{\text{FocalDice}} \;+\; 0.3 \cdot \mathcal{L}_{\text{CORAL}}$$

- **FocalDice**: A blended loss where the Focal component down-weights easy
  examples with γ = 2.5, and per-class weights `[0.4, 2.5, 2.0]` steer
  attention towards the minority Medium and High classes. It handles both
  hard and soft targets. When soft labels are present, it uses a focal-weighted
  KL divergence instead of cross-entropy.
- **CORAL** (Consistent Rank Logits): Adds an explicit ordinal constraint by
  modelling cumulative probabilities P(Y > 0) and P(Y > 1) via binary
  cross-entropy. This prevents the model from confusing Low with High while
  allowing softer errors between adjacent classes.

### 6.3 Training Loop & Calibration

| Parameter                 | Value      |
|---------------------------|------------|
| Optimizer                 | AdamW      |
| Learning rate             | 3 × 10⁻⁴  |
| Weight decay              | 1 × 10⁻⁴  |
| LR scheduler              | Cosine annealing |
| Epochs (max)              | 60         |
| Batch size                | 4          |
| Early stopping patience   | 15 epochs  |
| Gradient clipping norm    | 1.0        |
| Mixed precision           | true (GPU only) |
| Best-model criterion      | AUPRC (falls back to macro IoU) |

**Evaluation**: At each epoch the model is evaluated on the validation
split. A confusion matrix, macro IoU, macro F1, AUROC, and AUPRC are
computed. The checkpoint with the best AUPRC is persisted.

**Post-training calibration**:

1. **Isotonic Regression** — fitted on validation high-class probabilities.
   Saved as `isotonic_calibrator.joblib`.
2. **Temperature Scaling** — a scalar temperature parameter T is optimised
   via L-BFGS to minimise NLL on validation logits. Saved as
   `temperature_scaling.json` (T = 1.37 for the current model).
3. **Optimal thresholds** — both Youden's J and F1-optimal thresholds are
   computed on validation and cross-evaluated on test. The recommended
   threshold is **F1-optimal on validation** (currently 0.344).

**Visualisation artefacts** (under `artifacts/experiments/figures/`):
ROC curve, PR curve, calibration curve, training-history plot.

### 6.4 Current Training Results

The model trained for the full 60 epochs (no early-stop trigger):

| Metric             | Validation | Test   |
|--------------------|-----------|--------|
| Overall accuracy   | 0.8864    | 0.9405 |
| Macro IoU          | 0.8017    | 0.8358 |
| Macro F1           | 0.8886    | 0.9088 |
| AUROC              | 0.9996    | 0.9994 |
| AUPRC              | 0.9762    | 0.9700 |

Best epoch: **56**.

---

## 7. Stage 5 — Inference

`run_inference()` is invoked on the **test** area
(`artifacts/derived/test/feature_stack.tif`). If only a training area
exists, it infers on that instead.

### 7.1 Sliding-Window Prediction

The feature stack is read tile-by-tile through rasterio `Window` reads:

- **Window size**: 1 024 × 1 024 pixels
- **Overlap**: 256 pixels → stride 768
- **Skip threshold**: tiles with < 40 % valid pixels are skipped.

Each tile is pushed through the model. A temperature scaling factor
(T = 1.0 currently, because `temperature_override: 1.0` disables the
learned T = 1.37) is applied to logits before softmax.

### 7.2 Blending & CRF Post-Processing

**Gaussian blending**: each tile's predictions are multiplied by a 2-D
Gaussian weight mask (σ = overlap × 0.3 = 76.8 px), so overlapping
contributions are smoothly averaged with higher weight at tile centres.

**CRF post-processing** (enabled, `crf.enabled: true`):
After weighted blending, a Dense CRF (`pydensecrf`) is applied in tiles
(4 096 × 4 096 with 2 048 overlap) to encourage spatial coherence:

| CRF Parameter      | Value |
|---------------------|-------|
| Iterations          | 8     |
| Spatial weight (σ)  | 5.0   |
| Color weight (σ)    | 3.0   |
| Compat spatial      | 5.0   |
| Compat bilateral    | 15.0  |

The CRF uses the first 3 channels of the feature stack (normalised to
0–255) as the bilateral image.

### 7.3 MC Dropout Uncertainty

Currently **disabled** (`mc_dropout_iterations: 0`). When enabled, the
dropout layers are switched back to train mode and multiple stochastic
forward passes are performed; the pixel-level standard deviation of the
high-class probability across passes serves as the uncertainty estimate.

Because MC Dropout is off, the **fallback entropy** is used:

$$U(x) = -\sum_c P(c|x) \log P(c|x)$$

### 7.4 Susceptibility Score Computation

For the 3-class ordinal problem, two susceptibility scores are produced:

1. **Ordinal susceptibility** (primary):
   $$S = 0 \cdot P(\text{Low}) + 0.5 \cdot P(\text{Medium}) + 1.0 \cdot P(\text{High})$$
   Yields a continuous [0, 1] value.

2. **High-class probability** — raw P(High) after isotonic calibration.

**Class map generation** uses the ordinal susceptibility with configured
class breaks `[0.33, 0.67]`:
- S < 0.33 → Class 0 (Low)
- 0.33 ≤ S < 0.67 → Class 1 (Medium)
- S ≥ 0.67 → Class 2 (High)

### 7.5 Output Deliverables

All outputs are GeoTIFF rasters written to `outputs/`:

| File                              | Dtype   | Description                                    |
|-----------------------------------|---------|------------------------------------------------|
| `test_susceptibility.tif`         | float32 | Ordinal susceptibility score [0, 1]            |
| `test_susceptibility_high.tif`    | float32 | Calibrated P(High) [0, 1]                      |
| `test_class_probabilities.tif`    | float32 | All 3 class probabilities (multi-band)         |
| `test_uncertainty.tif`            | float32 | Entropy-based uncertainty                       |
| `test_class_map.tif`              | uint8   | Discrete class map (0/1/2, 255=nodata)         |
| `test_valid_mask.tif`             | uint8   | Binary valid-pixel mask                         |
| `test_feature_metadata.json`      | JSON    | Copy of the area's feature metadata             |
| `model_card.md`                   | Markdown| Summary of model, metrics, thresholds, outputs |

---

## 8. Stage 6 — Standalone Evaluation

`src/evaluate.py` is an independent CLI script that can be run **after**
the main pipeline to assess output quality against ground truth:

```bash
.venv/bin/python -m src.evaluate \
    --susceptibility outputs/test_susceptibility.tif \
    --ground_truth /path/to/ground_truth.tif \
    --output_dir outputs/evaluation
```

It implements multiple evaluation strategies appropriate for ordinal
susceptibility:

| Strategy | Positive class | Description                      |
|----------|----------------|----------------------------------|
| 1        | GT = 3         | High vs (Low + Medium)           |
| 2        | GT ≥ 2         | (Medium + High) vs Low           |
| 3        | —              | Spearman rank correlation        |
| 4        | GT = 2         | Medium vs Low (when High is rare)|

For each binary strategy it computes: AUROC, AUPRC, Cohen's Kappa, F1,
precision, recall, specificity, IoU. It also finds Youden-optimal and
F1-optimal thresholds independently.

An **analysis-only** mode (`--analysis_only`) computes spatial statistics
(susceptibility distribution, uncertainty correlation, coverage) without
needing ground truth.

Existing evaluation results live in
`outputs/evaluation_v2.5_mixed_domain_corrected/`.

---

## 9. Feature Channel Inventory

The model ingests **28 channels**. The full ordered list (from
`normalization_stats.json`) is:

| Index | Channel name               | Source            |
|-------|----------------------------|-------------------|
| 0     | `dtm_elevation`            | DEM               |
| 1     | `slope_deg`                | DEM derivative    |
| 2     | `aspect_sin`               | DEM derivative    |
| 3     | `aspect_cos`               | DEM derivative    |
| 4     | `general_curvature`        | DEM derivative    |
| 5     | `plan_curvature`           | DEM derivative    |
| 6     | `profile_curvature`        | DEM derivative    |
| 7     | `tpi`                      | DEM derivative    |
| 8     | `tri`                      | DEM derivative    |
| 9     | `log_flow_accumulation`    | DEM hydrology     |
| 10    | `twi`                      | DEM hydrology     |
| 11    | `spi`                      | DEM hydrology     |
| 12    | `sti`                      | DEM hydrology     |
| 13    | `log_distance_to_drainage` | DEM hydrology     |
| 14    | `ortho_norm_band_1`        | Orthophoto (R)    |
| 15    | `ortho_norm_band_2`        | Orthophoto (G)    |
| 16    | `ortho_norm_band_3`        | Orthophoto (B)    |
| 17–27 | `lulc_class_0` … `lulc_class_10` | ESA WorldCover one-hot |

---

## 10. Current Model Performance

### Training Metrics (best epoch = 56 / 60)

| Metric           | Validation | Test   |
|------------------|-----------|--------|
| Overall accuracy | 88.64 %   | 94.05 %|
| Macro IoU        | 0.8017    | 0.8358 |
| Macro F1         | 0.8886    | 0.9088 |
| AUROC            | 0.9996    | 0.9994 |
| AUPRC            | 0.9762    | 0.9700 |

### Recommended Threshold

F1-optimal on validation: **0.344** (method: `f1_validation`).

At this threshold on the test set:
- Precision: 93.9 %, Recall: 88.2 %, F1: 90.9 %, Specificity: 99.9 %

### Temperature Calibration

Learned temperature T = 1.37 (NLL ↓ 0.376 → 0.302), but currently
**overridden to T = 1.0** in the inference config.

---

## 11. Key Configuration Levers

| Parameter path (config.yaml)               | Current value  | Effect                                               |
|--------------------------------------------|----------------|------------------------------------------------------|
| `dataset.use_mixed_domain`                 | true           | Merge train + test areas before splitting             |
| `dataset.tile_size`                        | 256            | Tile edge length in pixels                            |
| `dataset.tile_overlap`                     | 128            | Overlap between adjacent tiles                        |
| `dataset.spatial_block_size`               | 5              | Minimum tile-distance separation between splits       |
| `dataset.positive_fraction`                | 0.3            | Target fraction of positive tiles per split           |
| `dataset.oversample_target_fraction`       | 0.075          | Target Class-1 pixel fraction via oversampling        |
| `dataset.use_smote`                        | true           | Enable SMOTE synthetic generation                     |
| `preprocessing.label_smoothing.enabled`    | true           | Ordinal soft labels                                   |
| `preprocessing.label_smoothing.alpha`      | 0.2            | Smoothing strength                                    |
| `preprocessing.external_lulc.enabled`      | true           | Use ESA WorldCover instead of K-Means                 |
| `model.encoder`                            | efficientnet-b4| U-Net encoder backbone                                |
| `model.attention`                          | true           | Spatial attention module on bottleneck                 |
| `model.dropout_prob`                       | 0.4            | Dropout rate (also enables MC uncertainty)             |
| `training.use_focal_loss`                  | true           | Focal loss for class imbalance                        |
| `training.use_ordinal_loss`                | true           | CORAL loss for ordinal consistency                    |
| `training.focal_gamma`                     | 2.5            | Focal focusing parameter                              |
| `training.coral_weight`                    | 0.3            | CORAL loss weight in combined loss                    |
| `training.class_weights`                   | [0.4, 2.5, 2.0]| Per-class weights [Low, Med, High]                   |
| `training.epochs`                          | 60             | Maximum training epochs                               |
| `inference.window_size`                    | 1024           | Sliding window size                                   |
| `inference.blending.method`                | gaussian       | Tile blending strategy                                |
| `inference.crf.enabled`                    | true           | Dense CRF post-processing                             |
| `inference.temperature_override`           | 1.0            | Override learned temperature (1.0 = disabled)         |
| `inference.class_breaks`                   | [0.33, 0.67]   | Ordinal score → class thresholds                      |

---

## 12. Resumability & Artifact Lifecycle

Every major function first checks whether its expected output files already
exist on disk. If they do (and `--force_recreate` is not set), the stage is
**skipped entirely**. This makes the pipeline safely re-entrant:

| Stage                | Checkpoint files                                          |
|----------------------|-----------------------------------------------------------|
| Preprocessing        | `artifacts/derived/{area}/feature_stack.tif`, `valid_mask.tif`, `feature_metadata.json`, `ground_truth_aligned.tif` |
| Mixed-domain merge   | `artifacts/derived/merged/merged_features.tif`, etc.       |
| Tiling & splitting   | `artifacts/splits/splits.json`, `dataset_summary.json`     |
| Training             | `artifacts/experiments/best_model.pth`                     |
| Inference            | `outputs/{area}_susceptibility.tif`, `_uncertainty.tif`, `_valid_mask.tif`, `model_card.md` |

To force a complete regeneration:
```bash
.venv/bin/python -m src.main_pipeline --force_recreate
```

---

## 13. File Map

```
.
├── config.yaml                          # All pipeline parameters
├── inputs.py                            # Absolute paths to source GeoTIFFs
├── requirements.txt                     # Python dependencies
│
├── src/
│   ├── main_pipeline.py                 # Orchestrator: preprocess → tile → train → infer
│   ├── preprocess_pipeline.py           # (Older/alternate preprocessing; not called by main)
│   ├── prepare_mixed_domain_dataset.py  # Merge + tile both areas with spatial blocking
│   ├── soft_labels.py                   # Ordinal & Gaussian label smoothing
│   ├── train.py                         # Dataset, losses (Focal, CORAL, Soft), training loop
│   ├── inference.py                     # Sliding window, blending, CRF, uncertainty, export
│   ├── evaluate.py                      # Standalone evaluation CLI
│   ├── metrics.py                       # Threshold selection (Youden, F1)
│   ├── visualize.py                     # ROC, PR, calibration, training-history plots
│   └── external_data_fetch/
│       ├── __init__.py
│       ├── fetch_esa_worldcover.py      # ESA WorldCover download (S3 + WMS) & one-hot
│       └── fetch_dynamic_world.py       # Google Dynamic World (requires EE auth)
│
├── artifacts/
│   ├── derived/{train,test,merged}/     # Feature stacks, masks, ground truth
│   ├── metadata/normalization_stats.json
│   ├── tiles/{train,val,test}/*.npy     # Feature tiles
│   ├── labels/{train,val,test}/*.npy    # Label tiles
│   ├── splits/splits.json               # Tile-to-split assignment
│   ├── splits/dataset_summary.json      # Tile counts, class distribution
│   └── experiments/
│       ├── best_model.pth               # Best model checkpoint
│       ├── isotonic_calibrator.joblib    # Post-hoc probability calibration
│       ├── temperature_scaling.json      # Learned temperature T
│       ├── training_metrics.json         # Full training report
│       └── figures/                      # ROC, PR, calibration, history plots
│
└── outputs/
    ├── test_susceptibility.tif          # Primary deliverable
    ├── test_susceptibility_high.tif     # Calibrated P(High)
    ├── test_class_probabilities.tif     # All 3 class probs
    ├── test_uncertainty.tif             # Entropy map
    ├── test_class_map.tif              # Discrete classification
    ├── test_valid_mask.tif             # Valid pixel mask
    ├── test_feature_metadata.json      # Channel metadata
    └── model_card.md                   # Human-readable summary
```

---

*Document generated from direct source-code analysis on 6 February 2026.*
