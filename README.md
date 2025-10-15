# landslide_susc_modelling
Transforming Drone Direct Products into Landslide Susceptibility Index Raster

## Overview
An end-to-end deep learning pipeline for landslide susceptibility mapping from drone-derived orthophotos and Digital Terrain Models (DTMs). The pipeline includes preprocessing, feature engineering, model training, and inference with uncertainty quantification.

## Key Features
- **Multi-source feature engineering**: DTM-derived terrain features + orthophoto-derived land cover
- **Resumable pipeline**: Automatically detects existing artifacts and resumes from the last checkpoint
- **Spatial block cross-validation**: Prevents spatial leakage in train/val/test splits
- **Probability calibration**: Isotonic regression for reliable probability estimates
- **Uncertainty quantification**: Monte Carlo dropout for epistemic uncertainty
- **Production-ready outputs**: Georeferenced GeoTIFF exports with model card

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Usage
```bash
# Run the full pipeline (resumes from last checkpoint)
python -m src.main_pipeline

# Force recreation of all artifacts
python -m src.main_pipeline --force_recreate
```

### Configuration
Edit `config.yaml` to customize:
- Input/output directories
- Preprocessing parameters (DTM hygiene, feature toggles)
- Dataset tiling and sampling strategy
- Model architecture and training hyperparameters
- Inference settings (window size, TTA, MC dropout)

Edit `inputs.py` to specify paths to your DTM, orthophoto, and ground truth rasters.

## Pipeline Stages

1. **Preprocessing** (`process_area`)
   - DTM hygiene (sink filling, smoothing)
   - Terrain feature derivation (slope, aspect, curvatures, flow accumulation, TWI, SPI, STI)
   - Orthophoto normalization and land cover clustering
   - Feature stack assembly and normalization

2. **Dataset Preparation** (`prepare_dataset`)
   - Spatial block splitting (train/val/test)
   - Tile extraction with configurable overlap
   - Class-balanced sampling with hard-negative mining

3. **Model Training** (`train_model`)
   - U-Net with ResNet encoder
   - Dice + Cross-Entropy loss with class weights
   - Isotonic calibration on validation set

4. **Inference** (`run_inference`)
   - Sliding-window prediction with Gaussian blending
   - Optional test-time augmentation (TTA)
   - Monte Carlo dropout for uncertainty estimation
   - GeoTIFF export: susceptibility, uncertainty, valid mask

## Output Files
- `outputs/<area>_landslide_susceptibility.tif` - Calibrated probability map [0-1]
- `outputs/<area>_uncertainty.tif` - Epistemic uncertainty estimate
- `outputs/<area>_valid_mask.tif` - Valid pixel mask
- `outputs/model_card.md` - Model documentation and performance metrics
- `artifacts/experiments/best_model.pth` - Trained model checkpoint
- `artifacts/experiments/training_metrics.json` - Training history

## Documentation
- `descriptive_script.md` - Detailed methodology and design decisions
- `AGENTS.md` - Guide for autonomous and human collaborators

## Implementation Notes

### Alternative Algorithms
The implementation uses practical alternatives for computational efficiency:
- **Slope calculation**: Gradient-based approximation (alternative to least-squares fitted plane)
- **Flow accumulation**: D8 single flow direction (alternative to Multiple Flow Direction - Freeman 1991)

These alternatives provide good approximations while maintaining reasonable computational performance for large rasters.

## License
See LICENSE file for details.
