# landslide_susc_modelling

Drone-derived landslide susceptibility modelling pipeline. The project builds terrain, orthophoto, and land-cover feature stacks; trains a 3-class ordinal segmentation model; and exports georeferenced susceptibility, class, and uncertainty rasters.

The current source of truth is the implementation plus `config.yaml`. Older historical Markdown files have been removed because they mixed past experiments, proposed fixes, and stale operating instructions.

## Quick Start

Always use the repository virtual environment directly:

```bash
.venv/bin/python manage.py pipeline
```

To rebuild all preprocessing, tiles, model artifacts, and outputs from scratch:

```bash
.venv/bin/python manage.py pipeline --force_recreate
```

Install or refresh dependencies with:

```bash
.venv/bin/pip install -r requirements.txt
```

Do not use system `python`, `python3`, or `pip` for this repository.

Before running inference with the current config, CRF setup must pass:

```bash
.venv/bin/python manage.py check-crf
```

## Main Files

- `config.yaml` controls directories, preprocessing, tiling, model, training, and inference settings.
- `inputs.py` defines the absolute local raster paths for the train and test areas.
- `manage.py` is the canonical operations CLI for validation, preprocessing, CRF checks, and pipeline launch.
- `src/main_pipeline.py` is the active end-to-end entrypoint.
- `src/train.py` contains dataset loading, losses, training, metrics, and calibration.
- `src/inference.py` writes final GeoTIFF deliverables.
- `src/evaluate.py` can evaluate or summarize generated outputs after inference.
- `updated_full_guide.md` is the detailed current guide.

## Current Pipeline

With the current config, the pipeline:

1. Preprocesses train and test DTMs/orthophotos to the DTM grid.
2. Builds a 28-channel feature stack from terrain derivatives, normalized orthophoto bands, and ESA WorldCover one-hot LULC channels.
3. Remaps ground truth to model classes: `0=low`, `1=medium`, `2=high`, `255=ignore`.
4. Uses mixed-domain tiling from both train and test areas.
5. Trains an EfficientNet-B4 U-Net with spatial attention, soft labels, focal/dice loss, and CORAL ordinal loss.
6. Exports calibrated test-area outputs under `outputs/`.

## Current Outputs

The active inference output names are:

- `outputs/test_susceptibility.tif`
- `outputs/test_susceptibility_high.tif`
- `outputs/test_class_probabilities.tif`
- `outputs/test_class_map.tif`
- `outputs/test_uncertainty.tif`
- `outputs/test_valid_mask.tif`
- `outputs/model_card.md`

Generated rasters, tiles, model checkpoints, reports, and logs are runtime artifacts. They are not source documentation and should not be committed.

## Validation

```bash
.venv/bin/python manage.py validate --config config.yaml
.venv/bin/python manage.py validate-spatial --metadata artifacts/derived/merged/merged_metadata.json
.venv/bin/python -m compileall -q inputs.py manage.py src
```

## More Detail

Read `updated_full_guide.md` for the full project orientation, current artifact snapshot, resolved sharp edges, remaining cautions, and validation commands.
