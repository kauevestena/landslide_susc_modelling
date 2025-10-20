# Inference Enhancement Summary

This document describes the advanced inference techniques implemented to eliminate tile borders and improve prediction quality.

## Overview

The inference pipeline now includes three sophisticated techniques:
1. **Weighted Blending** - Smooth transitions at tile boundaries
2. **Enhanced Test Time Augmentation (TTA)** - Robust predictions through multiple augmentations
3. **Conditional Random Fields (CRF)** - Spatial coherence post-processing

## 1. Weighted Blending

### Problem
The original implementation used overlapping tiles but simply **overwrote** predictions in overlap regions. The last tile to write to a pixel "won", creating hard borders at tile boundaries.

### Solution
Implement weighted averaging where each prediction is multiplied by a spatial weight before accumulation. Overlapping regions receive contributions from multiple tiles, smoothly blended.

### Methods Available

#### Gaussian Blending (Default)
- Creates 2D Gaussian weights centered at tile center
- Pixels near tile edges have lower weight
- Controlled by `sigma_factor` parameter
- Formula: `weight = exp(-(distance_from_center)² / (2σ²))` where `σ = overlap × sigma_factor`

#### Linear Blending
- Linear taper from tile edges
- Weight increases linearly from 0 at edges to 1 at distance > overlap
- Simpler but less smooth than Gaussian

#### No Blending
- Set `method: none` to disable (uniform weights)

### Configuration
```yaml
inference:
  overlap: 256  # Must be > 0 for blending
  blending:
    method: gaussian  # Options: gaussian, linear, none
    sigma_factor: 0.3  # For gaussian: controls taper steepness
```

## 2. Enhanced Test Time Augmentation

### Problem
Original TTA used only 4 basic augmentations (horizontal flip, vertical flip, 90°, 270° rotations). This limited the diversity of predictions.

### Solution
Expanded to 8 augmentations covering:
- Basic 4: horizontal flip, vertical flip, 90° rotation, 270° rotation
- Advanced 4: 180° rotation, combined flips, diagonal flips

### Benefits
- **Robustness**: Averages over more viewpoints → reduced sensitivity to orientation
- **Accuracy**: Ensemble effect improves prediction quality
- **Uncertainty**: Greater variation in augmentation responses indicates uncertain regions

### Configuration
```yaml
inference:
  tta: true
  tta_augmentations: 8  # 0=disabled, 4=basic, 8=full
```

### Performance Trade-off
- `tta_augmentations: 0` - Fastest (1× forward pass per tile)
- `tta_augmentations: 4` - Moderate (5× forward passes)
- `tta_augmentations: 8` - Slowest but most robust (9× forward passes)

## 3. Conditional Random Fields (CRF)

### Problem
Neural networks make pixel-wise predictions independently. This can produce:
- Salt-and-pepper noise
- Isolated misclassifications
- Predictions inconsistent with local context
- Edges that don't align with actual features

### Solution
CRF post-processing enforces spatial coherence using:
- **Unary potentials**: Original model predictions
- **Pairwise potentials**: Smoothness constraints

### Pairwise Potentials

#### Gaussian (Spatial) Kernel
- Encourages nearby pixels to have similar labels
- Controlled by `spatial_weight` (spatial standard deviation)
- Higher weight = more local smoothing

#### Bilateral Kernel
- Edge-aware smoothing using input features
- Preserves boundaries where features change rapidly (e.g., landslide edges)
- Controlled by:
  - `spatial_weight`: Spatial proximity
  - `color_weight`: Feature similarity threshold
  - Uses RGB channels from orthophoto when available

### Configuration
```yaml
inference:
  crf:
    enabled: true
    iterations: 5  # Mean-field inference iterations (more = smoother)
    spatial_weight: 3.0  # Spatial smoothing strength
    color_weight: 3.0  # Feature similarity threshold
    compat_spatial: 3.0  # Spatial compatibility weight
    compat_bilateral: 10.0  # Bilateral compatibility weight (edge preservation)
```

### Parameter Tuning Guide

**spatial_weight** (1.0 - 10.0)
- Lower: Less smoothing, preserves small features
- Higher: More smoothing, removes noise but may blur boundaries

**color_weight** (1.0 - 10.0)
- Lower: Smooth even across feature differences (may blur edges)
- Higher: Only smooth within similar features (preserves edges)

**compat_spatial** (1.0 - 10.0)
- Weight of spatial smoothness penalty
- Higher = stronger spatial coherence

**compat_bilateral** (1.0 - 20.0)
- Weight of edge-preserving penalty
- Higher = stronger edge preservation

**iterations** (1 - 10)
- More iterations = more refined solution
- Diminishing returns after ~5 iterations

## Implementation Details

### Processing Pipeline Order
1. **Sliding window inference** with weighted blending
2. **Normalize** by accumulated weights
3. **Apply CRF** for spatial coherence (uses raw features)
4. **MC Dropout** iterations (if enabled)
5. **Calibration** (if calibrator exists)
6. **Threshold** to generate class map

### Memory Considerations
- Blending adds minimal memory overhead (one weight array per class)
- CRF requires loading full feature stack into memory
- For very large rasters, consider processing in chunks or reducing `crf.iterations`

### Performance Impact

| Technique | Slowdown | Quality Gain |
|-----------|----------|--------------|
| Blending (Gaussian) | ~1% | High - eliminates tile borders |
| TTA (4 augmentations) | 5× | Moderate - improved robustness |
| TTA (8 augmentations) | 9× | High - maximum robustness |
| CRF | 10-30% | Moderate-High - spatial coherence |

**Recommended for production**: All enabled with `tta_augmentations: 4`

**Recommended for research/final maps**: All enabled with `tta_augmentations: 8`

**Recommended for fast iteration**: Only blending enabled, TTA off, CRF off

## Validation

To verify the improvements:

1. **Visual inspection**: Check tile boundaries in GIS
   - Before: Hard edges visible at tile boundaries
   - After: Smooth continuous predictions

2. **Compare outputs**:
   ```bash
   # Generate without enhancements
   # Edit config.yaml:
   #   blending.method: none
   #   tta: false
   #   crf.enabled: false
   python -m src.main_pipeline --force_recreate
   
   # Generate with enhancements
   # Edit config.yaml: enable all features
   python -m src.main_pipeline --force_recreate
   
   # Compare in GIS software (QGIS, ArcGIS)
   ```

3. **Quantitative metrics**:
   - Edge coherence: Compute gradient magnitude at tile boundaries
   - Spatial autocorrelation: Moran's I statistic
   - Classification smoothness: Local variance in predictions

## Dependencies

New requirement: `pydensecrf` for CRF post-processing

Install via:
```bash
pip install pydensecrf
```

If installation fails (C++ compilation issues), CRF will be automatically disabled with a warning.

## Troubleshooting

### "Hard borders still visible"
- Check that `overlap > 0` (recommend `overlap = window_size / 2`)
- Verify `blending.method != "none"`
- Increase `sigma_factor` for wider blending taper

### "Predictions too smooth / lost detail"
- Reduce CRF iterations
- Increase `color_weight` to preserve edges better
- Lower `compat_spatial` weight

### "Inference very slow"
- Reduce `tta_augmentations` from 8 to 4 or 0
- Disable CRF (`enabled: false`)
- Increase `window_size` (fewer tiles)
- Reduce `overlap` (less computation per pixel)

### "Out of memory during CRF"
- CRF loads entire feature stack - use smaller rasters
- Reduce `window_size` in main inference (affects tile count, not CRF)
- Process areas separately

## References

- **Dense CRF**: Krähenbühl & Koltun (2011) "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials"
- **Test Time Augmentation**: Krizhevsky et al. (2012) "ImageNet Classification with Deep CNNs"
- **Weighted Blending**: Standard technique in image stitching and panorama generation

## Future Enhancements

Potential additions:
- Multi-scale inference (pyramid approach)
- Learned blending weights (attention-based)
- 3D CRF using temporal information (if time series available)
- Uncertainty-aware blending (weight by prediction confidence)
