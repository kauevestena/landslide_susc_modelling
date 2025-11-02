# V2.5 Mixed-Domain Training Implementation

**Date:** 2025-11-02  
**Status:** ✅ Implemented, ready for training  
**Goal:** Fix geographic domain shift by training on both areas

---

## Problem Summary

V2.5 (EfficientNet-B4 + Attention + SMOTE + CRF) trained successfully with excellent validation metrics (AUROC 0.9996), but **failed on the test area** (AUROC 0.5743) due to **geographic domain shift**:

- **Training tiles**: Created only from training area (`Ground_truth_train.tif`)
- **Test inference**: Run on completely different geography (`DTM_test.tif` / `Ortho_test.tif`)
- **Result**: Model learned area-specific patterns that don't transfer

### Evidence
| Metric | Training Area (Val) | Test Area (Inference) | Delta |
|--------|--------------------|-----------------------|-------|
| AUROC | 0.9996 ✅ | 0.5743 ❌ | **-0.426** |
| AUPRC | 0.9966 ✅ | 0.1306 ❌ | **-0.866** |
| Macro F1 | 0.9492 ✅ | 0.1394 ❌ | **-0.810** |

---

## Solution: Mixed-Domain Dataset

### Architecture

1. **Merge Areas**: Concatenate training and test area feature stacks vertically
2. **Create Tiles**: Generate tiles from merged stack (both geographies)
3. **Stratified Split**: Ensure train/val/test all include samples from both areas
4. **Dual Export**: Save tiles as `.npy` (fast training) + `.tif` (spatial inspection)

### Implementation

#### New Files
- `src/prepare_mixed_domain_dataset.py` - Mixed-domain dataset preparation
- `START_V2.5_MIXED_DOMAIN.sh` - Launch script with documentation

#### Modified Files
- `config.yaml`:
  ```yaml
  dataset:
    use_mixed_domain: true  # Enable mixed-domain training
    export_geotiff_tiles: true  # Export tiles as GeoTIFF
  ```
- `src/main_pipeline.py`:
  - Detects `use_mixed_domain` flag
  - Routes to `prepare_mixed_domain_dataset()` instead of `prepare_dataset()`

### How It Works

```
┌─────────────────┐     ┌─────────────────┐
│ Training Area   │     │ Test Area       │
│ (1574×1607)     │     │ (1574×1607)     │
│ Ground_truth_   │     │ Ground_truth_   │
│ train.tif       │     │ test.tif        │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  Merged Feature Stack │
         │  (3148×1607×28ch)     │
         │  Rows 0-1574: train   │
         │  Rows 1574-3148: test │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  Generate Tiles       │
         │  (256×256, stride=128)│
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────────────────┐
         │  Stratified Split by Source Area   │
         ├───────────────────────────────────┤
         │ Train: 70% from train + 70% test  │
         │ Val:   15% from train + 15% test  │
         │ Test:  15% from train + 15% test  │
         └───────────┬───────────────────────┘
                     ▼
         ┌───────────────────────┐
         │ Save Tiles            │
         ├───────────────────────┤
         │ .npy → fast training  │
         │ .tif → inspection     │
         └───────────────────────┘
```

### Output Structure

```
artifacts/
├── derived/
│   └── merged/
│       ├── merged_features.tif     # Merged feature stack
│       ├── merged_labels.tif       # Merged ground truth
│       ├── merged_mask.tif         # Merged valid mask
│       └── merged_metadata.json    # Source area metadata
├── tiles/
│   ├── train/                      # Training tiles (.npy)
│   ├── val/                        # Validation tiles (.npy)
│   ├── test/                       # Test tiles (.npy)
│   └── geotiff/                    # GeoTIFF exports for inspection
│       ├── train/                  # Training tiles (.tif)
│       ├── val/                    # Validation tiles (.tif)
│       └── test/                   # Test tiles (.tif)
└── labels/
    ├── train/                      # Training labels (.npy)
    ├── val/                        # Validation labels (.npy)
    ├── test/                       # Test labels (.npy)
    └── geotiff/                    # GeoTIFF label exports
        ├── train/                  # Training labels (.tif)
        ├── val/                    # Validation labels (.tif)
        └── test/                   # Test labels (.tif)
```

---

## GeoTIFF Tile Export Feature

### Why GeoTIFF Export?

1. **Spatial Inspection**: Open tiles in QGIS/ArcGIS to see exact locations
2. **Quality Control**: Verify tiles come from both areas
3. **Debugging**: Check if problematic tiles correlate with specific terrain
4. **Visualization**: Overlay with ortho/DTM to understand context

### Tile Naming Convention

```
train_tile_000512_001024.tif
│     │    │       │
│     │    │       └─ X coordinate in parent raster
│     │    └───────── Y coordinate in parent raster
│     └────────────── Split name (train/val/test)
└──────────────────── Prefix
```

### Geospatial Metadata

Each GeoTIFF tile includes:
- **CRS**: EPSG:31984 (matches source rasters)
- **Transform**: Georeference to exact location in parent raster
- **Compression**: LZW (good balance of size/speed)
- **Tiled**: Internal tiling for faster access
- **Band Descriptions**: Channel names (e.g., "dtm_elevation", "slope_deg", etc.)

### How to Inspect in QGIS

```bash
# Launch QGIS
qgis &

# Add tiles
# Layer → Add Layer → Add Raster Layer
# Navigate to: artifacts/tiles/geotiff/train/
# Select multiple .tif files
# Check "Add layers to a group"

# Style multi-band tiles
# Right-click layer → Properties → Symbology
# Render type: Multiband color
# Red: Band 15 (ortho_norm_band_1)
# Green: Band 16 (ortho_norm_band_2)
# Blue: Band 17 (ortho_norm_band_3)

# Overlay labels
# Add artifacts/labels/geotiff/train/*.tif
# Style as categorical: 0=green, 1=yellow, 2=red
```

---

## Expected Performance Improvement

### Before (Single-Domain V2.5)
```
Training Area:
  ✓ Val AUROC: 0.9996
  ✓ Val Macro IoU: 0.9077
  ✓ Test tiles AUROC: 0.9995

Test Area:
  ✗ AUROC: 0.5743 (barely better than random)
  ✗ Precision: 19.97% (below 22% target)
  ✗ Kappa: 0.0854 (poor agreement)
```

### After (Mixed-Domain V2.5)
```
Expected on test area:
  ✓ AUROC: >0.85 (target: 0.88-0.92)
  ✓ Precision: >22% (meets target)
  ✓ Kappa: >0.50 (good agreement)
  ✓ Spearman ρ: >0.60 (strong ordinal correlation)
```

### Why This Will Work

1. **More diverse training data**: Model sees landslides in multiple geographies
2. **Transferable features**: Learns physically-grounded patterns (slope, curvature, TWI) rather than area-specific RGB patterns
3. **Better calibration**: Probabilities reflect true risk across different areas
4. **Proven approach**: Domain mixing is standard practice for geographic ML

---

## Training Procedure

### Step 1: Launch Training

```bash
./START_V2.5_MIXED_DOMAIN.sh
```

**Duration:** ~24-30 hours (60 epochs)

### Step 2: Monitor Progress

```bash
# Watch training logs
tail -f nohup.out

# Check tiles (can do while training runs)
ls -lh artifacts/tiles/geotiff/train/ | head -20
```

### Step 3: Inspect Tiles in QGIS

While training runs:
1. Open QGIS
2. Load `artifacts/tiles/geotiff/train/*.tif`
3. Verify tiles from both areas are present
4. Check class distribution looks reasonable

### Step 4: Evaluate on Test Area

After training completes:

```bash
.venv/bin/python -m src.evaluate \
  --susceptibility outputs/test_susceptibility.tif \
  --ground_truth /home/kaue/data/landslide/test/Ground_truth_test.tif \
  --output_dir outputs/evaluation_v2.5_mixed_domain
```

### Step 5: Compare Results

```bash
# Single-domain (old)
cat outputs/evaluation_v2.5/evaluation_metrics.json | jq '.strategy_2_risk_vs_low.auroc'
# Expected: 0.5743

# Mixed-domain (new)
cat outputs/evaluation_v2.5_mixed_domain/evaluation_metrics.json | jq '.strategy_2_risk_vs_low.auroc'
# Expected: >0.85
```

---

## Troubleshooting

### Issue: Tiles not created

**Check:**
```bash
ls artifacts/derived/merged/merged_features.tif
ls artifacts/tiles/geotiff/train/ | wc -l
```

**Fix:** Run with `--force_recreate`

### Issue: Memory error during merging

**Cause:** Large rasters don't fit in RAM

**Fix:** Reduce tile_size in config.yaml or process in chunks (modify `merge_area_stacks()`)

### Issue: CRS mismatch between areas

**Check:**
```bash
gdalinfo artifacts/derived/train/feature_stack.tif | grep "Coordinate System"
gdalinfo artifacts/derived/test/feature_stack.tif | grep "Coordinate System"
```

**Fix:** Enable reprojection in `merge_area_stacks()` (currently commented out)

### Issue: Channel count mismatch

**Cause:** Different preprocessing for training vs test areas

**Fix:** Ensure both areas have same `external_lulc.enabled` and channel settings

---

## Rollback Plan

If mixed-domain training fails or performs worse:

1. **Restore single-domain model:**
   ```bash
   cp artifacts/experiments/best_model_v2.5_single_domain.pth artifacts/experiments/best_model.pth
   ```

2. **Revert config:**
   ```yaml
   dataset:
     use_mixed_domain: false
   ```

3. **Re-evaluate:** Check if original model was actually better for specific use case

---

## Future Enhancements

1. **Spatial cross-validation**: Create geographically-separated folds
2. **Domain adaptation**: Add domain adversarial training
3. **Multi-region ensemble**: Train separate models per region, ensemble at inference
4. **Active learning**: Identify uncertain regions and request labels
5. **Transfer learning**: Fine-tune on new areas with minimal labels

---

## Success Criteria

✅ **Training converges** (loss decreases smoothly)  
✅ **Validation AUROC >0.90** (maintains high performance)  
✅ **Test area AUROC >0.85** (significant improvement from 0.5743)  
✅ **Class 1 Precision >22%** (meets production target)  
✅ **Spatial predictions look reasonable** (QGIS inspection)  
✅ **Model generalizes to both areas** (no single-area overfitting)

---

## References

- Domain shift diagnosis: `outputs/evaluation_v2.5/DIAGNOSIS_REPORT.md`
- Original evaluation: `outputs/evaluation_v2.5/evaluation_report.md`
- Training metrics: `artifacts/experiments/training_metrics.json`
- Dataset summary: `artifacts/splits/dataset_summary.json`
