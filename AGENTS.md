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
- `src/inference.py` – Sliding-window inference, uncertainty estimation, deliverable exports, model card generation.
- `descriptive_script.md` – Domain narrative for landslide workflow; use it for alignment when proposing changes.

## 3. Standard Operating Procedure
1. **Install dependencies** – `pip install -r requirements.txt` (PyTorch build must match local CUDA/CPU stack).
2. **Validate inputs** – Ensure all paths in `inputs.py` exist and share projection, resolution, and extent (DTM is the reference grid).
3. **Tune config** – Key levers: preprocessing toggles, tile sampling parameters, model encoder, training schedule, inference window.
4. **Run pipeline** – `python -m src.main_pipeline`. The pipeline is **resumable**: it automatically detects existing artifacts and skips completed stages. To force recreation of all artifacts from scratch, use `python -m src.main_pipeline --force_recreate`. Outputs land in `artifacts/` (intermediates) and `outputs/` (GeoTIFFs + model card).
5. **Review results** – Inspect `outputs/` rasters in GIS, read `outputs/model_card.md`, and cross-check training metrics under `artifacts/experiments/`.
6. **Calibrate or iterate** – Adjust config, rerun pipeline (it will resume from the first incomplete stage). Use `--force_recreate` if you need to regenerate all artifacts from scratch.

## 4. Contribution Playbook
- **Plan first:** Summarise intent, affected modules, and validation strategy before editing.
- **Respect artefacts:** Never commit generated rasters/tiles; `.gitignore` already excludes `artifacts/` and `outputs/`.
- **Document decisions:** Maintain docstrings and in-code comments for complex geospatial logic (e.g., flow accumulation, sampling heuristics).
- **Tests & smoke checks:** When feasible, add synthetic or subset-based checks (e.g., run on cropped rasters) to guard regressions.
- **Calibration assets:** If you regenerate calibrators or metrics, keep the artifact structure (`best_model.pth`, `isotonic_calibrator.joblib`, `training_metrics.json`).
- **Performance awareness:** Large rasters demand streaming/tiling; avoid loading entire scenes unless necessary.

## 5. Interacting With Agents
- **Small tasks:** Prefer targeted commands (e.g., editing a single function) over wholesale rewrites.
- **State reporting:** Log notable artefact changes (new files, schema adjustments) in pull requests or commit messages.
- **Validation requests:** After major pipeline edits, schedule a dry run or request user confirmation if data access is unavailable.
- **Error handling:** On preprocessing failure, capture stack trace plus offending raster metadata (`src`, `transform`, `shape`) to accelerate debugging.

## 6. Troubleshooting Cheatsheet
- *Mismatched rasters* → Ensure the DTM is reference; resampling is handled in `process_area` but extreme nodata gaps require manual QA.
- *Missing tiles* → `prepare_dataset` drops patches with insufficient valid pixels; check `dataset_summary.json` for counts.
- *Training instability* → Verify class balance in summary, adjust `positive_fraction`, and consider freezing encoder if data is scarce.
- *Inference slowdowns* → Tune `window_size` / `overlap`, disable TTA, or reduce `mc_dropout_iterations`.
- *Calibration empty* → Happens when validation positives are absent; log this in metrics and consider alternative splits.
- *Pipeline stuck on old artifacts* → Use `--force_recreate` flag to regenerate all artifacts from scratch: `python -m src.main_pipeline --force_recreate`.

## 7. Communication Templates
- **Status update:** Outline stage (preprocess/train/infer), config hash or Git commit, and notable metrics (macro IoU, AUROC, AUPRC).
- **Change proposal:** State motivation, impacted modules, validation evidence, and rollback plan.
- **Issue escalation:** Provide reproduction steps, command used, environment summary (OS, Python, PyTorch), and sample log snippets.

Stay mindful of reproducibility: pin random seeds, record config deltas, and prefer deterministic algorithms when feasible.
