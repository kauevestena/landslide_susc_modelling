# AGENTS Guide

This file is the operational guide for autonomous and human collaborators working in this repository.

## Source of Truth

- Treat `config.yaml`, `inputs.py`, and the Python implementation under `src/` as authoritative.
- Treat `README.md`, this file, and `updated_full_guide.md` as the only maintained Markdown source docs.
- Do not infer behavior from generated reports under `outputs/` or old logs unless you explicitly label them as artifact history.
- The current end-to-end entrypoint is `src/main_pipeline.py`.

## Environment Rules

- Always use `.venv/bin/python` and `.venv/bin/pip`.
- Do not use system `python`, `python3`, or `pip`.
- Standard commands:

```bash
.venv/bin/python manage.py validate --config config.yaml
.venv/bin/python manage.py check-crf
.venv/bin/python manage.py pipeline
.venv/bin/python manage.py pipeline --force_recreate
.venv/bin/python manage.py three-methods --force
.venv/bin/python -m src.evaluate --analysis_only
```

- `manage.py` is the only maintained helper/operations CLI. Do not revive deleted standalone helper scripts.
- `pydensecrf` is required when `inference.crf.enabled: true`; missing CRF support is a hard setup failure, not a silent runtime downgrade.

## Active Workflow

The current pipeline stages are:

1. `preprocess_data()` loads paths from `inputs.py`, aligns rasters to each DTM grid, computes terrain derivatives, fetches or derives LULC, normalizes features, remaps labels, and writes area artifacts.
2. `prepare_mixed_domain_dataset()` is active because `dataset.use_mixed_domain: true`; it vertically combines train/test area tensors, creates spatially blocked train/val/test tiles, and writes `.npy` tiles plus optional GeoTIFF inspection tiles.
3. `train_model()` trains the segmentation model, writes `best_model.pth`, metrics, plots, threshold selection, isotonic calibrator, optional ordinal calibrator, and temperature scaling metadata.
4. `run_inference()` loads the test area, predicts class probabilities by sliding window, applies smoothing/calibration/CRF when enabled, and writes final GeoTIFF outputs plus `outputs/model_card.md`.

`manage.py three-methods` is a separate comparison workflow. It aligns the existing DL outputs plus IBGE-adapted and SGB-style deterministic products to the feb26 16 cm drone DTM footprint, writing generated artifacts under `DL_method/`, `IBGE_method/`, and `SGB_method/`.

## Current Model and Data Contract

- Input registry: `inputs.py`.
- Output classes: `0=low`, `1=medium`, `2=high`, `255=ignore`.
- Current model config: EfficientNet-B4 U-Net with spatial attention and dropout.
- Current feature stack: terrain derivatives, normalized orthophoto bands, and ESA WorldCover one-hot channels.
- Current training objective: focal/dice plus CORAL ordinal loss when enabled in config.
- Current inference products use the `test_` prefix when test artifacts are present.

## Working Practices

- Keep edits targeted. Do not rewrite pipeline modules unless the task requires it.
- Do not delete or regenerate `artifacts/`, `outputs/`, or `.venv/` unless explicitly requested.
- Do not treat generated files under `DL_method/outputs/`, `IBGE_method/outputs/`, or `SGB_method/outputs/` as source files.
- Use `--force_recreate` when changing preprocessing, LULC, tiling, label smoothing, channel count, or model architecture.
- Before changing behavior, read the relevant implementation first. Generated artifacts may reflect older runs.
- Large rasters are expensive; prefer metadata checks and targeted validation before launching full retraining.

## Validation

For standard validation, use:

```bash
.venv/bin/python manage.py check-crf
.venv/bin/python manage.py validate --config config.yaml
.venv/bin/python manage.py validate-spatial --metadata artifacts/derived/merged/merged_metadata.json
.venv/bin/python -m compileall -q inputs.py manage.py src
bash -n START_V2.5_MIXED_DOMAIN.sh
bash -n START_V2.5_TRAINING.sh
bash -n run_fixed_pipeline.sh
bash -n validate_v2.5_config.sh
```

For pipeline behavior changes, add a focused smoke check when possible and state whether full preprocessing/training/inference was run.

## Known Cautions

- The worktree may already contain dirty code/config changes; do not revert them unless the user asks.
- Existing generated artifact metadata may predate current schema checks; the pipeline will regenerate stale artifacts when needed.
- `updated_full_guide.md` lists resolved sharp edges and remaining cautions; update it whenever implementation behavior changes.
