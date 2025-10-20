# Inference Configuration Quick Reference

## Essential Parameters

### Basic Inference
```yaml
inference:
  use_cuda: false           # Enable GPU if available
  window_size: 512          # Tile size for sliding window
  overlap: 256              # Overlap between tiles (≥ window_size/2 recommended)
  batch_size: 2             # Batch size (mainly for future batching support)
```

### Test Time Augmentation (TTA)
```yaml
inference:
  tta: true                 # Enable/disable TTA
  tta_augmentations: 8      # Number of augmentations
                            # 0 = disabled (fastest)
                            # 4 = basic (5× slower, good quality)
                            # 8 = full (9× slower, best quality)
```

### Weighted Blending
```yaml
inference:
  blending:
    method: gaussian        # gaussian | linear | none
    sigma_factor: 0.3       # For gaussian: controls taper width (0.2-0.5)
                            # Lower = sharper taper, higher = wider blend
```

### Conditional Random Fields (CRF)
```yaml
inference:
  crf:
    enabled: true           # Enable/disable CRF post-processing
    iterations: 5           # Mean-field iterations (3-10)
    spatial_weight: 3.0     # Spatial smoothing (1-10)
    color_weight: 3.0       # Feature similarity threshold (1-10)
    compat_spatial: 3.0     # Spatial compatibility (1-10)
    compat_bilateral: 10.0  # Edge preservation strength (1-20)
```

### MC Dropout Uncertainty
```yaml
inference:
  mc_dropout_iterations: 5  # Number of stochastic forward passes (0=disabled)
                            # More iterations = better uncertainty estimates
                            # Each iteration runs full sliding window inference
```

## Preset Configurations

### Fast Iteration (Testing)
```yaml
inference:
  window_size: 512
  overlap: 128              # Minimal overlap
  tta: false                # No TTA
  tta_augmentations: 0
  mc_dropout_iterations: 0  # No MC dropout
  blending:
    method: linear          # Simplest blending
  crf:
    enabled: false          # Skip CRF
```
**Speed**: Fastest  
**Quality**: Basic  
**Use case**: Quick tests, debugging

### Balanced (Production)
```yaml
inference:
  window_size: 512
  overlap: 256              # 50% overlap
  tta: true
  tta_augmentations: 4      # Basic TTA
  mc_dropout_iterations: 5
  blending:
    method: gaussian
    sigma_factor: 0.3
  crf:
    enabled: true
    iterations: 5
    spatial_weight: 3.0
    color_weight: 3.0
    compat_spatial: 3.0
    compat_bilateral: 10.0
```
**Speed**: Moderate (5-10× slower than fast)  
**Quality**: High  
**Use case**: Production deliverables

### Maximum Quality (Research)
```yaml
inference:
  window_size: 512
  overlap: 320              # ~62% overlap
  tta: true
  tta_augmentations: 8      # Full TTA
  mc_dropout_iterations: 10
  blending:
    method: gaussian
    sigma_factor: 0.4       # Wider blend
  crf:
    enabled: true
    iterations: 10          # More CRF iterations
    spatial_weight: 5.0     # Stronger smoothing
    color_weight: 2.0       # Better edge preservation
    compat_spatial: 5.0
    compat_bilateral: 15.0
```
**Speed**: Slowest (10-20× slower than fast)  
**Quality**: Maximum  
**Use case**: Final research outputs, publications

## Tuning Guide

### Problem: Tile borders visible
**Solutions**:
1. Increase `overlap` (minimum 128, recommended 256+)
2. Check `blending.method` is not `none`
3. Increase `sigma_factor` (0.3 → 0.5)

### Problem: Predictions too noisy
**Solutions**:
1. Enable CRF: `crf.enabled: true`
2. Increase `crf.compat_spatial` (3.0 → 5.0)
3. Increase `crf.iterations` (5 → 10)
4. Enable TTA: `tta: true`

### Problem: Lost small features / over-smoothed
**Solutions**:
1. Reduce `crf.spatial_weight` (3.0 → 2.0)
2. Increase `crf.color_weight` (3.0 → 5.0)
3. Reduce `crf.iterations` (5 → 3)
4. Check `crf.compat_bilateral` is higher than `compat_spatial`

### Problem: Inference too slow
**Solutions**:
1. Reduce TTA: `tta_augmentations: 4` or `0`
2. Disable MC dropout: `mc_dropout_iterations: 0`
3. Disable CRF: `crf.enabled: false`
4. Reduce overlap: `overlap: 128`
5. Increase window size: `window_size: 768` (fewer tiles)

### Problem: Out of memory
**Solutions**:
1. Disable CUDA: `use_cuda: false`
2. Reduce `window_size` (512 → 256)
3. Disable CRF (loads full raster)
4. Process smaller areas separately

## Performance Benchmarks

Approximate relative inference times (baseline = fast preset):

| Configuration | Time | Quality | RAM | GPU |
|--------------|------|---------|-----|-----|
| Fast | 1× | ⭐⭐ | 2GB | Optional |
| Balanced | 6× | ⭐⭐⭐⭐ | 4GB | Optional |
| Maximum | 15× | ⭐⭐⭐⭐⭐ | 8GB | Recommended |

*Actual times depend on raster size, hardware, and model complexity*

## Validation Checklist

After running inference with new settings:

- [ ] Open susceptibility map in GIS
- [ ] Check for visible tile borders (zoom to 1:500 scale)
- [ ] Verify edges align with features (landslide boundaries)
- [ ] Inspect small isolated predictions (should be reduced if CRF enabled)
- [ ] Compare uncertainty map (higher at tile borders = blending issue)
- [ ] Check log for processing time (compare with expectations)
- [ ] Verify valid mask matches expected coverage

## Example Commands

**Full pipeline with defaults**:
```bash
python -m src.main_pipeline
```

**Force regenerate all outputs** (e.g., after config changes):
```bash
python -m src.main_pipeline --force_recreate
```

**Test on small area first**:
1. Crop input rasters to small region
2. Update `inputs.py` paths
3. Run with fast preset
4. Verify results before full inference

## Dependencies

Ensure these are installed:
```bash
pip install pydensecrf  # For CRF (optional but recommended)
pip install torch torchvision  # For model inference
pip install rasterio numpy scipy  # For raster operations
```

If `pydensecrf` installation fails, CRF will be automatically disabled.
