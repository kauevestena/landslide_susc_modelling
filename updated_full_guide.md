# Updated Full Guide - Landslide Susceptibility Modelling

Audit date: 2026-06-20.

This guide describes what the project currently does based on the implementation and the present generated artifact summaries. It replaces the removed historical Markdown files, which mixed old plans, completed fixes, experiments, and stale run instructions.

## 1. What This Project Does

The repository trains and applies a 3-class ordinal landslide susceptibility segmentation model from drone-derived data.

Inputs:

- DTM rasters.
- Orthophoto rasters.
- Ground-truth susceptibility classes.
- Optional external LULC data, currently ESA WorldCover by config.

Outputs:

- Continuous susceptibility raster.
- High-risk probability raster.
- Per-class probability raster.
- Class map raster.
- Uncertainty raster.
- Valid mask raster.
- Model card and training metrics.

The active command is:

```bash
.venv/bin/python manage.py pipeline
```

Use this to force a full rebuild:

```bash
.venv/bin/python manage.py pipeline --force_recreate
```

Always use `.venv/bin/python` and `.venv/bin/pip`; do not use system Python.

## 2. Repository Map

Maintained source files:

- `config.yaml`: central runtime configuration.
- `inputs.py`: absolute local file paths for training and test rasters.
- `manage.py`: canonical operations CLI for validation, CRF checks, preprocessing, spatial split checks, and pipeline launch.
- `requirements.txt`: Python dependencies.
- `src/main_pipeline.py`: end-to-end preprocessing, dataset preparation dispatch, training, and inference orchestration.
- `src/prepare_mixed_domain_dataset.py`: current active mixed-domain tiling path.
- `src/train.py`: PyTorch datasets, model wrapper, losses, metrics, training loop, calibration, plots.
- `src/inference.py`: sliding-window prediction, blending, TTA, CRF, calibration, uncertainty, output export, model card.
- `src/evaluate.py`: standalone analysis/evaluation of generated outputs.
- `src/metrics.py`: threshold selection helpers.
- `src/visualize.py`: ROC, PR, calibration, and training-history plots.
- `src/soft_labels.py`: ordinal and gaussian soft-label generation.
- `src/external_data_fetch/`: ESA WorldCover and Google Dynamic World fetchers.
- `*.sh`: compatibility wrappers that call `manage.py`.

Maintained Markdown docs:

- `README.md`: short entrypoint and orientation.
- `AGENTS.md`: operational instructions for agents and collaborators.
- `updated_full_guide.md`: this full guide.

Generated artifacts:

- `artifacts/`: derived rasters, merged stacks, tiles, model checkpoints, calibrators, metrics, figures.
- `outputs/`: final GeoTIFFs, model card, generated evaluation reports.

Generated artifacts are useful evidence but not source documentation.

## 3. Current Configuration Summary

The current `config.yaml` sets:

- `dataset.use_mixed_domain: true`.
- `preprocessing.external_lulc.enabled: true`.
- `preprocessing.external_lulc.source: worldcover`.
- `preprocessing.label_smoothing.enabled: true`.
- `preprocessing.label_smoothing.type: ordinal`.
- `preprocessing.label_smoothing.alpha: 0.2`.
- `dataset.tile_size: 256`.
- `dataset.tile_overlap: 128`.
- `dataset.positive_class: 2`.
- `model.encoder: efficientnet-b4`.
- `model.out_classes: 3`.
- `model.dropout_prob: 0.4`.
- `model.attention: true`.
- `training.use_focal_loss: true`.
- `training.use_ordinal_loss: true`.
- `training.coral_weight: 0.3`.
- `training.class_weights: [0.4, 2.5, 2.0]`.
- `inference.window_size: 1024`.
- `inference.overlap: 256`.
- `inference.tta: false`.
- `inference.mc_dropout_iterations: 0`.
- `inference.class_breaks: null`, meaning multi-class class maps use argmax in current source.
- `inference.temperature_override: null`, meaning learned temperature metadata is used when present.
- `inference.crf.enabled: true`; `pydensecrf` is required and setup is asserted before inference.

The current `inputs.py` points to local absolute paths under `/home/kaue/data/landslide/`.

## 4. End-to-End Pipeline

### 4.1 Preprocessing

Implemented in `src/main_pipeline.py`.

For each configured split, the pipeline:

1. Loads DTM, orthophoto, and optional ground truth from `inputs.py`.
2. Uses the DTM as the reference grid.
3. Builds a DTM valid mask from nodata and NaN cells.
4. Fills nodata by nearest-neighbor propagation.
5. Applies a simple sink-fill fallback and gaussian smoothing.
6. Derives terrain and hydrology features:
   - DTM elevation.
   - Slope.
   - Aspect sine and cosine.
   - General, plan, and profile curvature.
   - TPI and TRI.
   - Flow accumulation and flow area.
   - TWI, SPI, STI.
   - Distance to drainage.
7. Reprojects the orthophoto to the DTM grid.
8. Normalizes orthophoto bands by valid-pixel z-score.
9. Fetches external LULC when enabled, currently ESA WorldCover through S3-first processing.
10. One-hot encodes LULC classes and appends them to the feature stack.
11. Normalizes the feature stack.
12. Aligns and remaps ground truth:
    - source `1 -> 0` low.
    - source `2 -> 1` medium.
    - source `3 -> 2` high.
    - source `0` and unexpected values -> `255` ignore.
13. Applies an optional class-boundary ignore mask.
14. Writes derived area rasters and metadata under `artifacts/derived/{train,test}/`.

Current feature metadata reports 28 channels:

- 14 terrain/hydrology channels.
- 3 normalized orthophoto channels in the current metadata.
- 11 ESA WorldCover one-hot channels.

Note: historical logs show some runs with 4 orthophoto bands and 29 prepared channels, while current artifact metadata reports 28 channels. The current model card and feature metadata report 28 channels.

### 4.2 Mixed-Domain Dataset Preparation

Because `dataset.use_mixed_domain: true`, `src/main_pipeline.py` calls `prepare_mixed_domain_dataset()` rather than the older single-area `prepare_dataset()` path.

The mixed-domain path:

1. Loads train and test feature stacks, aligned labels, and masks.
2. Checks CRS and resolution match.
3. Vertically concatenates train and test arrays into a synthetic merged raster.
4. Pads width with zeros if needed.
5. Generates candidate tile positions using `tile_size=256` and `stride=128`.
6. Splits candidate tiles spatially by blocks within each source area.
7. Combines train-area and test-area tiles into train/val/test splits.
8. Recomputes normalization statistics from training tiles only in the current source.
9. Writes `.npy` feature and label tiles for training.
10. Writes GeoTIFF inspection tiles when configured.
11. Writes `artifacts/splits/splits.json` and `artifacts/splits/dataset_summary.json`.

Current generated `dataset_summary.json` reports:

- Train tiles: 626.
- Validation tiles: 111.
- Test tiles: 134.
- Mixed-domain: true.
- Train split contribution: 59 tiles from train area, 567 from test area.
- Validation contribution: 10 from train area, 101 from test area.
- Test contribution: 21 from train area, 113 from test area.
- Class pixel counts: class 0 = 31,779,863; class 1 = 16,336,765; class 2 = 355,096.

Important caveat: existing generated artifacts may predate the current schema fields. The pipeline now treats missing or stale metadata schemas as a reason to regenerate instead of silently resuming.

### 4.3 Training

Implemented in `src/train.py`.

The current training path:

1. Loads tiles from `artifacts/tiles/{train,val,test}` and labels from `artifacts/labels/{train,val,test}`.
2. Detects hard labels or soft labels from label array dimensionality.
3. Applies training augmentations:
   - horizontal and vertical flips.
   - 90-degree rotations.
   - brightness shift.
   - contrast scaling.
   - gaussian blur.
   - gaussian noise.
4. Builds a `segmentation_models_pytorch.Unet`.
5. Uses the configured encoder, currently `efficientnet-b4`.
6. Wraps the U-Net with a spatial attention module when `model.attention: true`.
7. Adds dropout to the segmentation head when configured.
8. Uses CPU unless `training.use_cuda: true` and a runtime CUDA convolution test succeeds.
9. Uses configured class weights when present.
10. Selects the loss:
    - current config uses `CombinedOrdinalLoss`.
    - this combines focal/dice loss with CORAL ordinal loss.
11. Trains with AdamW and cosine learning-rate schedule.
12. Tracks validation metrics and saves the best model by AUPRC, falling back to macro IoU if AUPRC is invalid.
13. Fits:
    - an isotonic calibrator for high-class probability.
    - an ordinal calibrator for weighted ordinal susceptibility in the current source when data is available.
    - temperature scaling metadata.
14. Evaluates on the test tile split if available.
15. Writes metrics and plots under `artifacts/experiments/`.

Current generated training summary reports:

- Best epoch: 56.
- Best validation metrics:
  - overall accuracy: 0.8864.
  - macro IoU: 0.8017.
  - macro F1: 0.8886.
  - AUROC: 0.9996.
  - AUPRC: 0.9762.
- Test metrics:
  - overall accuracy: 0.9405.
  - macro IoU: 0.8358.
  - macro F1: 0.9088.
  - AUROC: 0.9994.
  - AUPRC: 0.9700.
- Recommended high-risk threshold: 0.3441967070 from validation F1.
- Temperature: 1.3700920343.

These metrics are generated artifacts. They describe the last completed run, not a guarantee for future runs.

### 4.4 Inference

Implemented in `src/inference.py`.

The inference path:

1. Chooses the test area when test artifacts are available, otherwise train.
2. Loads feature metadata and the trained checkpoint.
3. Rebuilds the training architecture, including attention and dropout wrapper settings.
4. Loads calibrators and temperature scaling metadata when available.
5. Runs sliding-window inference over the feature stack.
6. Uses overlap blending when configured.
7. Applies probability smoothing when `inference.smoothing_alpha > 0`.
8. Applies CRF when enabled. If `pydensecrf` is not importable while CRF is enabled, inference fails fast with setup guidance.
9. Optionally runs MC dropout uncertainty when configured.
10. Computes:
    - `susceptibility_high`: calibrated probability for class 2.
    - `susceptibility`: ordinal weighted score `0*P(low) + 0.5*P(medium) + 1*P(high)`, then ordinal calibration when available.
    - `class_map`: argmax of class probabilities for multi-class models.
    - `uncertainty`: MC-dropout standard deviation if available, otherwise entropy.
11. Writes output GeoTIFFs and copies feature metadata.
12. Writes `outputs/model_card.md`.

Current output names:

- `outputs/test_susceptibility.tif`.
- `outputs/test_susceptibility_high.tif`.
- `outputs/test_class_probabilities.tif`.
- `outputs/test_class_map.tif`.
- `outputs/test_uncertainty.tif`.
- `outputs/test_valid_mask.tif`.
- `outputs/test_feature_metadata.json`.
- `outputs/model_card.md`.

### 4.5 Evaluation

Implemented in `src/evaluate.py`.

Evaluation can run in two modes:

```bash
.venv/bin/python -m src.evaluate --analysis_only
```

or with ground truth:

```bash
.venv/bin/python -m src.evaluate \
  --susceptibility outputs/test_susceptibility.tif \
  --ground_truth /home/kaue/data/landslide/test/Ground_truth_test.tif \
  --valid_mask outputs/test_valid_mask.tif \
  --class_map outputs/test_class_map.tif \
  --output_dir outputs/evaluation
```

With ground truth, it:

- resamples ground truth to output shape if needed.
- detects `{1,2,3}` or `{0,1,2}` encodings.
- normalizes labels to `{0,1,2}`.
- computes multi-class and binary strategy metrics.
- writes generated Markdown and JSON reports under the selected output directory.

Generated evaluation Markdown under `outputs/` is intentionally not maintained source documentation.

## 5. External LULC

The active config uses ESA WorldCover.

`src/external_data_fetch/fetch_esa_worldcover.py`:

- extracts the reference raster bbox in WGS84.
- downloads 3-degree WorldCover tiles from S3 first.
- falls back to WMS logic only when requested or needed.
- mosaics, reprojects to the DTM grid, and one-hot encodes all 11 WorldCover classes.

`src/external_data_fetch/fetch_dynamic_world.py`:

- requires Google Earth Engine.
- can build hard-class or probability-band composites over a date range.
- reprojects and one-hot encodes to the reference raster.

Dynamic World options now live under `preprocessing.external_lulc.dynamic_world`, and `force_download` lives under `preprocessing.external_lulc`, which is where `fetch_external_lulc()` expects them.

## 6. Current Artifact Snapshot

Current generated artifacts occupy approximately:

- `artifacts/`: 28G.
- `outputs/`: 2.4G.
- `.venv/`: 7.4G.

Current model card reports:

- Inference area: test.
- Feature stack: `artifacts/derived/test/feature_stack.tif`.
- Channels: 28.
- Encoder: `efficientnet-b4`.
- Output classes: 3.
- Positive class index: 2.
- Best epoch: 56.
- Calibration: `artifacts/experiments/isotonic_calibrator.joblib`.
- Temperature scaling: `artifacts/experiments/temperature_scaling.json`.

Current tracked source tree had pre-existing dirty non-doc changes at audit time:

- `config.yaml`.
- `requirements.txt`.
- `src/evaluate.py`.
- `src/inference.py`.
- `src/main_pipeline.py`.
- `src/prepare_mixed_domain_dataset.py`.
- `src/train.py`.

Those changes are treated as current implementation truth and should not be reverted without explicit user instruction.

## 7. Resolved Sharp Edges and Remaining Cautions

Resolved in current source:

- Stale preprocessing, merge, and tile artifacts now have schema/config checks before they are reused.
- The single-area `prepare_dataset()` path no longer shifts labels a second time after preprocessing.
- Inference skip checks use the same current output names that inference writes.
- Redundant standalone helper Python scripts were removed in favor of `manage.py`.
- Shell scripts are thin wrappers over `manage.py` and no longer contain old prompts, metric claims, or install side effects.
- Historical tracked training logs were removed from source.
- `pydensecrf` setup is asserted when CRF is enabled; missing CRF support is a hard setup error.
- Dynamic World and `force_download` config keys now sit under `preprocessing.external_lulc`.

Remaining cautions:

- Existing generated artifacts may be older than the current schema and will be regenerated when the pipeline reaches those stages.
- CUDA is disabled in current config. The code has a runtime CUDA test, but CPU is the expected path while `use_cuda: false`.
- Full retraining/inference is expensive and was not part of the sharp-edge cleanup.

## 8. Safe Operating Procedures

Before running:

```bash
.venv/bin/python manage.py check-crf
.venv/bin/python manage.py validate --config config.yaml
.venv/bin/python -m compileall -q inputs.py manage.py src
```

For a normal resumable run:

```bash
.venv/bin/python manage.py pipeline
```

For a clean rebuild:

```bash
.venv/bin/python manage.py pipeline --force_recreate
```

Use `--force_recreate` after changing:

- input rasters.
- preprocessing feature toggles.
- LULC source or LULC schema.
- label smoothing settings.
- tile size, overlap, split strategy, or mixed-domain mode.
- model encoder, attention, output classes, or channel count.
- loss strategy or class weights when old training artifacts should not be reused.
- inference architecture settings that must match training.

After training/inference:

```bash
.venv/bin/python -m src.evaluate --analysis_only
```

To inspect current mixed-domain split separation:

```bash
.venv/bin/python manage.py validate-spatial --metadata artifacts/derived/merged/merged_metadata.json
```

With ground truth, pass explicit paths as shown in section 4.5.

## 9. Documentation Maintenance Rule

Keep this repository to three maintained Markdown source docs:

- `README.md`.
- `AGENTS.md`.
- `updated_full_guide.md`.

When implementation changes, update these docs from the code and config. Do not add experiment-status Markdown files at the repo root. Put generated analyses under `outputs/` and treat them as artifacts.
