# AGENTS Guide

This document orients autonomous and human collaborators to the landslide susceptibility modelling repository.

## 1. Mission Overview
- **Objective:** Produce calibrated landslide susceptibility rasters (probabilities + classes + uncertainty) from drone-derived orthophotos and DTMs.
- **Primary entry point:** `src/main_pipeline.py` orchestrates preprocessing, tiling, training, and inference in a single run.
- **Execution config:** `config.yaml` controls directories, preprocessing toggles, sampling strategy, model hyperparameters, and inference behaviour.
- **Input registry:** Absolute file paths for each study area live in `inputs.py`. Adjust these when new surveys are introduced.

## 2. Repository Map
- `config.yaml` – Declarative project settings; keep it source-of-truth and prefer tweaking values here instead of hard-coding.
- `inputs.py` – Central definition of training/test rasters; avoid committing secrets or network paths that will break on other machines.
- `src/main_pipeline.py` – Preprocessing + dataset preparation utilities. Generates artefacts under `artifacts/` and drives the full run.
- `src/train.py` – Model training helpers (dataset loaders, losses, metrics, calibration).
- `src/inference.py` – Sliding-window inference with weighted blending, TTA, CRF post-processing, uncertainty estimation, deliverable exports, model card generation.
- `src/external_data_fetch/` – External LULC data fetchers (ESA WorldCover, Google Dynamic World) with standalone CLI interfaces.
- `descriptive_script.md` – Domain narrative for landslide workflow; use it for alignment when proposing changes.
- `INFERENCE_ENHANCEMENTS.md` – Detailed documentation of advanced inference techniques (blending, TTA, CRF).
- `EXTERNAL_LULC_IMPLEMENTATION.md` – Guide to external LULC integration, replacing K-means clustering with validated land cover products.

## 3. Standard Operating Procedure
1. **Activate virtual environment** – **CRITICAL**: Always activate `.venv` before running any commands:
   ```bash
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```
   **For agents/automated tools**: Always use `.venv/bin/python` or `.venv/bin/pip` instead of system `python`/`pip`.

2. **Install dependencies** – `pip install -r requirements.txt` (PyTorch build must match local CUDA/CPU stack).
3. **Validate inputs** – Ensure all paths in `inputs.py` exist and share projection, resolution, and extent (DTM is the reference grid).
4. **Tune config** – Key levers: preprocessing toggles, tile sampling parameters, model encoder, training schedule, inference window.
5. **Run pipeline** – `python -m src.main_pipeline`. The pipeline is **resumable**: it automatically detects existing artifacts and skips completed stages. To force recreation of all artifacts from scratch, use `python -m src.main_pipeline --force_recreate`. Outputs land in `artifacts/` (intermediates) and `outputs/` (GeoTIFFs + model card).
6. **Review results** – Inspect `outputs/` rasters in GIS, read `outputs/model_card.md`, and cross-check training metrics under `artifacts/experiments/`.
7. **Calibrate or iterate** – Adjust config, rerun pipeline (it will resume from the first incomplete stage). Use `--force_recreate` if you need to regenerate all artifacts from scratch.

## 4. Contribution Playbook
- **Plan first:** Summarise intent, affected modules, and validation strategy before editing.
- **Respect artefacts:** Never commit generated rasters/tiles; `.gitignore` already excludes `artifacts/` and `outputs/`.
- **Document decisions:** Maintain docstrings and in-code comments for complex geospatial logic (e.g., flow accumulation, sampling heuristics).
- **Tests & smoke checks:** When feasible, add synthetic or subset-based checks (e.g., run on cropped rasters) to guard regressions.
- **Calibration assets:** If you regenerate calibrators or metrics, keep the artifact structure (`best_model.pth`, `isotonic_calibrator.joblib`, `training_metrics.json`).
- **Performance awareness:** Large rasters demand streaming/tiling; avoid loading entire scenes unless necessary.

## 5. Interacting With Agents
- **Virtual environment**: **ALWAYS** use `.venv/bin/python` or `.venv/bin/pip` for all commands. Never use system Python or `python3`. The project has a dedicated virtual environment that must be used.
- **Small tasks:** Prefer targeted commands (e.g., editing a single function) over wholesale rewrites.
- **State reporting:** Log notable artefact changes (new files, schema adjustments) in pull requests or commit messages.
- **Validation requests:** After major pipeline edits, schedule a dry run or request user confirmation if data access is unavailable.
- **Error handling:** On preprocessing failure, capture stack trace plus offending raster metadata (`src`, `transform`, `shape`) to accelerate debugging.

## 6. Troubleshooting Cheatsheet
- *Import errors / Module not found* → **Activate `.venv`**: `source .venv/bin/activate` (or use `.venv/bin/python` directly).
- *Mismatched rasters* → Ensure the DTM is reference; resampling is handled in `process_area` but extreme nodata gaps require manual QA.
- *Missing tiles* → `prepare_dataset` drops patches with insufficient valid pixels; check `dataset_summary.json` for counts.
- *Training instability* → Verify class balance in summary, adjust `positive_fraction`, and consider freezing encoder if data is scarce.
- *Inference slowdowns* → Tune `window_size` / `overlap`, disable TTA, or reduce `mc_dropout_iterations`.
- *Calibration empty* → Happens when validation positives are absent; log this in metrics and consider alternative splits.
- *Pipeline stuck on old artifacts* → Use `--force_recreate` flag to regenerate all artifacts from scratch: `.venv/bin/python -m src.main_pipeline --force_recreate`.
- *External LULC download fails* → Check internet connection; for WorldCover verify WMS access, for Dynamic World authenticate via `.venv/bin/earthengine authenticate`.
- *Channel count mismatch after LULC change* → Switching between K-means and external LULC mid-project requires `--force_recreate` to regenerate all artifacts consistently.
- *Soft label warnings* → If switching label smoothing settings, use `--force_recreate` to regenerate tiles with consistent label format.
- *Test split missing classes* → Increase `dataset.max_split_attempts` in config.yaml (default: 20) or adjust `test_size` to include more spatial blocks. The pipeline automatically retries with different random seeds.
- *Class imbalance concerns* → **This is inherent to landslide susceptibility**: high-risk areas are naturally rarer than medium-risk, which are rarer than low-risk. The pipeline handles this via focal loss, positive_fraction sampling, and soft label smoothing. Use Cohen's Kappa instead of accuracy to evaluate properly.

## 7. Soft Label Smoothing (Ordinal Classification)

The pipeline supports **soft label smoothing** for ordinal 3-class landslide susceptibility (low/medium/high risk). Instead of training on hard discrete labels, the model can learn from probability distributions that express uncertainty and ordinal relationships.

### Why Use Soft Labels?

1. **Ordinal awareness**: Encodes that "medium" is between "low" and "high" risk
2. **Uncertainty expression**: Reflects ambiguity at class boundaries
3. **Better calibration**: Reduces model overconfidence on uncertain pixels
4. **Aligned with inference**: Training distribution matches continuous susceptibility output [0,1]

### Configuration

In `config.yaml` under `preprocessing.label_smoothing`:

```yaml
preprocessing:
  label_smoothing:
    enabled: true      # Enable soft label generation
    type: ordinal      # Options: ordinal (recommended), gaussian, none
    alpha: 0.1         # For ordinal: smoothing strength (0.0=hard, 0.3=very soft)
    sigma: 1.0         # For gaussian: spatial smoothing kernel size
```

### Smoothing Methods

**Ordinal smoothing** (default, recommended):
- Distributes probability mass to adjacent classes
- Example with `alpha=0.1`:
  - Class 0 (low): `[0.95, 0.05, 0.00]`
  - Class 1 (medium): `[0.05, 0.90, 0.05]`
  - Class 2 (high): `[0.00, 0.05, 0.95]`

**Gaussian smoothing**:
- Spatially smooths class boundaries using Gaussian filter
- Creates soft transitions where classes meet
- Pixels far from boundaries stay near one-hot

**None**:
- Standard hard labels (one-hot encoding)
- Traditional discrete classification

### Implementation Details

- Soft labels are generated in `src/main_pipeline.py:prepare_dataset()`
- Stored as float32 arrays: shape `(num_classes, H, W)` instead of `(H, W)`
- Training uses `SoftDiceCrossEntropyLoss` (KL divergence + soft Dice)
- Validation metrics computed via argmax of soft labels for consistency
- Dataset summary includes label smoothing metadata

### Switching Between Hard and Soft Labels

**CRITICAL**: Changing `label_smoothing.enabled` or `label_smoothing.type` requires regenerating tiles:

```bash
.venv/bin/python -m src.main_pipeline --force_recreate
```

Otherwise, the dataset will have mismatched label formats causing training errors.

### Troubleshooting Soft Labels

- *"Shape mismatch in loss"* → Tiles were created with different smoothing setting; use `--force_recreate`
- *"Probabilities don't sum to 1.0"* → Check `label_smoothing.alpha` is in valid range [0, 0.5]
- *Poor convergence* → Try lower `alpha` (0.05) or disable smoothing; some datasets benefit from hard labels
- *Memory issues* → Soft labels use 3× memory per pixel; reduce `tile_size` or `batch_size` if needed

## 8. Communication Templates
- **Status update:** Outline stage (preprocess/train/infer), config hash or Git commit, and notable metrics (macro IoU, AUROC, AUPRC).
- **Change proposal:** State motivation, impacted modules, validation evidence, and rollback plan.
- **Issue escalation:** Provide reproduction steps, command used, environment summary (OS, Python, PyTorch), and sample log snippets.

Stay mindful of reproducibility: pin random seeds, record config deltas, and prefer deterministic algorithms when feasible.
