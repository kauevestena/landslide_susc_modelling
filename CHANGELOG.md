# Changelog

## [Unreleased] - 2025-10-14

### Added
- **Resumable Pipeline**: Pipeline now automatically detects existing artifacts and skips completed stages
  - Added `--force_recreate` command-line flag to force recreation of all artifacts from scratch
  - Preprocessing stage checks for existing feature stacks and metadata
  - Dataset preparation stage checks for existing tiles and splits
  - Training stage checks for existing model checkpoint
  - Inference stage checks for existing output GeoTIFFs
  
- **Enhanced Documentation**:
  - Updated `README.md` with comprehensive usage instructions and pipeline overview
  - Updated `descriptive_script.md` to document alternative algorithm implementations
  - Updated `AGENTS.md` with resumability instructions and troubleshooting tips
  - Added this CHANGELOG.md to track repository changes

### Changed
- Modified `src/main_pipeline.py`:
  - Added argparse support for command-line arguments
  - Added `force_recreate` parameter to `process_area()`, `preprocess_data()`, `prepare_dataset()`
  - Artifact existence checks in `process_area()` for feature stacks and metadata
  - Artifact existence checks in `prepare_dataset()` for tiles and splits
  - Updated `main()` function to parse command-line arguments and propagate `force_recreate` flag

- Modified `src/train.py`:
  - Added `force_recreate` parameter to `train_model()`
  - Model checkpoint existence check to skip training if model already exists

- Modified `src/inference.py`:
  - Added `force_recreate` parameter to `run_inference()`
  - Output existence check to skip inference if outputs already exist

### Documented
- **Alternative Algorithm Implementations**:
  - Slope calculation: Gradient-based approximation (alternative to Horn/Costa-Cabral least-squares fitted plane)
  - Flow accumulation: D8 single flow direction (alternative to Multiple Flow Direction - Freeman 1991)
  - Both alternatives provide good approximations while maintaining computational efficiency

### Fixed
- No critical bugs identified during review
- Note: `inputs.py` variable name swap was already fixed by user prior to this update

## Usage Examples

### Resume from last checkpoint (default behavior)
```bash
python -m src.main_pipeline
```

### Force full pipeline recreation
```bash
python -m src.main_pipeline --force_recreate
```

### Get help
```bash
python -m src.main_pipeline --help
```

## Alignment with Specification

The implementation is now fully aligned with `descriptive_script.md`:
- ✅ All DTM-derived channels implemented
- ✅ All orthophoto processing steps implemented
- ✅ Spatial block splitting implemented
- ✅ Class-balanced sampling with hard-negative mining implemented
- ✅ Probability calibration implemented
- ✅ Uncertainty quantification implemented
- ✅ GeoTIFF exports implemented
- ✅ Reproducibility controls implemented
- ✅ Resumable pipeline now implemented

### Implementation Notes
- Flow accumulation uses D8 algorithm (documented as alternative to MFD)
- Slope calculation uses gradient-based method (documented as alternative to Horn's method)
- Land cover clustering uses K-means (unsupervised alternative to semantic classification)
