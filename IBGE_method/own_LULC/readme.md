

# IBGE Method: Custom LULC Generation Pipeline

## Overview

This subfolder generates a **high-resolution Land-Use Land-Cover (LULC) raster** specifically tailored for the IBGE method comparison workflow. The custom LULC replaces generic remote-sensing data with a model trained directly on the study area's training polygons.

## Data Specifications

### Input Data (defined in `lulc_inputs.py`)

1. **RGB Orthophoto**
   - High-resolution drone/aerial imagery
   - Covers the study area footprint

2. **Study Area Polygons**
   - Vector layer with training annotations
   - Each polygon labeled with:
     - Text class label
     - Integer class value (numeric encoding)
   - Represents ground truth for model training

### Configurable Parameters (in `lulc_inputs.py`)

- Output resolution of the LULC raster (e.g., pixel size in meters)
- Class definitions and mappings
- Train/validation split ratio
- Data augmentation settings (if applied)

## Implementation Plan

### Phase 1: Data Preparation & Preprocessing
- Load and validate orthophoto and polygon layers
- Rasterize polygons to create training labels at target resolution
- Normalize orthophoto bands (optional: histogram equalization, contrast stretching)
- Handle edge cases: nodata values, coordinate system misalignment
- **Decision needed**: Spatial blocking strategy for train/val split (avoid data leakage)

### Phase 2: Analysis & Design Decisions
- **Class Imbalance Investigation**: Compute class distribution; assess if resampling/weighting required
- **Data Augmentation**: Evaluate whether rotation, flipping, elastic deformation, or color jittering improves generalization
- **Model Architecture Selection**:
  - Consider lightweight models (e.g., FCN, UNet, SegFormer) vs. heavy models
  - Evaluate: parameter count, inference speed, memory footprint
  - Propose candidate architectures (e.g., EfficientNet-based UNet)
- **Hyperparameter Tuning**: Learning rate, batch size, optimizer, scheduler, loss function weights

### Phase 3: Model Training
- Implement training loop with validation monitoring
- Track metrics: pixel-level accuracy, per-class F1, mIoU
- Save checkpoints and select best model
- Log training curves and class-specific performance

### Phase 4: Inference & Post-Processing
- Run trained model on full study area (sliding window if needed)
- Generate probability maps for all classes
- Optional smoothing/filtering to reduce noise
- Output high-resolution LULC raster (GeoTIFF)

### Phase 5: Integration & Validation
- Verify LULC aligns spatially with original orthophoto and DTM
- Compare against original labels (holdout test set) to assess generalization
- Document model performance metrics
- Store model weights and metadata for reproducibility

## Implementation Status

- [ ] Finalize data loading and preprocessing pipeline
- [ ] Decide on class imbalance handling strategy
- [ ] Implement data augmentation (if needed)
- [ ] Select and configure model architecture
- [ ] Implement training and validation loop
- [ ] Run inference and generate output raster
- [ ] Validate spatial alignment and performance

## Related Files

- `lulc_inputs.py`: Data paths and configuration parameters
- `implementation/`: Core Python modules for preprocessing, training, and inference (to be created/organized)
- `runner.sh`: Execution script for the full pipeline
- `outputs/`: Generated LULC rasters and model artifacts
