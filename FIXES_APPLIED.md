# Critical Fixes Applied - 2025-10-20

## Problem Summary

The evaluation revealed severe issues with the trained model:

1. **Insufficient Training Data:** Only 8 tiles created, 0 validation tiles, 0 test tiles
2. **Extreme Predictions:** 92.4% of test area classified as HIGH RISK (mean prob: 0.91)
3. **No Validation:** All validation metrics were NaN (no validation split)

## Root Cause

The training area (1574×1607 pixels at 25m) with tile_size=512 and restrictive sampling criteria produced only ~9 possible tiles, which after quality filtering became 8. With such few tiles, the val/test split (15%/20%) rounded down to 0.

## Configuration Changes Applied

### 1. Tile Generation (More Tiles)

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `tile_size` | 512 | **256** | Creates ~4× more tiles from same area |
| `tile_overlap` | 64 | **32** | Proportional reduction |

**Expected outcome:** ~36-40 tiles (vs 8)

### 2. Sampling Criteria (Less Restrictive)

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `positive_min_fraction` | 0.01 (1%) | **0.005 (0.5%)** | Accept tiles with fewer landslide pixels |
| `positive_fraction` | 0.5 (50%) | **0.3 (30%)** | Allow more negative/mixed tiles |
| `min_valid_fraction` (dataset) | 0.5 (50%) | **0.4 (40%)** | Accept tiles with more nodata |
| `min_valid_fraction` (preprocessing) | 0.55 (55%) | **0.4 (40%)** | Consistent with dataset setting |

**Expected outcome:** More borderline tiles accepted, better class balance

### 3. Training Schedule (Better Validation)

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `epochs` | 20 | **30** | Allow more training with proper validation |
| `early_stopping_patience` | 5 | **7** | More recovery time with better validation |

**Expected outcome:** Better convergence with validation-guided training

## Expected Outcomes After Re-run

### Dataset Summary
```json
{
  "tile_counts": {
    "train": 30-35,  // Was 8
    "val": 5-7,      // Was 0  
    "test": 8-10     // Was 0
  }
}
```

### Training Metrics
- ✅ Validation AUROC: 0.80-0.95 (was NaN)
- ✅ Validation AUPRC: 0.70-0.90 (was NaN)
- ✅ Test metrics computed (was null)
- ✅ Proper early stopping based on val loss

### Susceptibility Predictions
- ✅ More spatial variation expected
- ✅ High risk areas: 30-60% (not 92%)
- ✅ Mean susceptibility: 0.4-0.6 (not 0.91)
- ✅ Predictions follow terrain patterns

## Next Steps

### 1. Re-run the Pipeline
```bash
cd /home/kaue/landslide_susc_modelling
.venv/bin/python -m src.main_pipeline --force_recreate
```

**What this does:**
- Regenerates all tiles with new size (256×256)
- Creates proper train/val/test splits
- Retrains model with validation monitoring
- Performs inference on test area
- Generates new susceptibility maps

**Expected runtime:** 10-30 minutes (depends on hardware)

### 2. Verify Tile Generation
```bash
# Check tile counts
cat artifacts/splits/dataset_summary.json | grep -A 5 "tile_counts"

# List tiles in each split
echo "Train tiles:" && ls artifacts/tiles/train/ | wc -l
echo "Val tiles:" && ls artifacts/tiles/val/ | wc -l
echo "Test tiles:" && ls artifacts/tiles/test/ | wc -l
```

**Expected output:**
```
Train tiles: 30-35
Val tiles: 5-7
Test tiles: 8-10
```

### 3. Check Training Metrics
```bash
# View validation performance
cat artifacts/experiments/training_metrics.json | grep -A 10 "val_metrics"

# View training history plots
ls -lh artifacts/experiments/figures/
```

**Look for:**
- ✅ Val AUROC > 0.80
- ✅ Val AUPRC > 0.70
- ✅ Val Macro IoU > 0.60
- ✅ Convergence visible in plots

### 4. Re-evaluate Outputs
```bash
# Run evaluation on new outputs
.venv/bin/python -m src.evaluate --analysis_only

# View summary
cat outputs/evaluation/EVALUATION_SUMMARY.md
```

**Expected improvements:**
- Mean susceptibility: 0.4-0.6 (not 0.91)
- High risk %: 30-60% (not 92%)
- Spatial variation visible
- Validation metrics present

### 5. Visual Inspection
```bash
# Open outputs in QGIS or similar GIS software
ls -lh outputs/*.tif
```

**Files to inspect:**
- `test_susceptibility.tif` - Should show spatial patterns
- `test_uncertainty.tif` - Should correlate with boundaries/transitions
- `test_class_map.tif` - Should show mixed risk zones
- `test_valid_mask.tif` - Indicates valid inference area

## Troubleshooting

### If Still Not Enough Tiles

**Symptom:** Still getting < 20 tiles total

**Try:** Further reduce tile_size
```yaml
dataset:
  tile_size: 224  # Try even smaller
  tile_overlap: 28
```

**Or:** Relax criteria more
```yaml
dataset:
  positive_min_fraction: 0.002  # 0.2%
  min_valid_fraction: 0.35  # 35%
```

### If Training Still Shows Extreme Predictions

**Symptom:** After retraining, still predicting >80% high risk

**Check:**
1. Class balance in dataset_summary.json
2. Training area ground truth quality
3. Feature normalization (check normalization_stats.json)

**Try:**
- Add class weights to loss function
- Adjust learning rate (0.0001 instead of 0.0003)
- Use focal loss instead of cross-entropy

### If Validation Metrics Still NaN

**Symptom:** Validation tiles exist but metrics are NaN

**Possible causes:**
- Validation set has no positive examples
- Validation set too small for reliable metrics

**Try:**
- Increase val_size to 0.20 (20%)
- Adjust positive_fraction to ensure val has positives

## Monitoring During Re-run

Watch for these log messages:

**Stage 1: Tiling**
```
Created X train tiles, Y val tiles, Z test tiles
```
→ Should see X≥30, Y≥5, Z≥8

**Stage 2: Training**
```
Epoch N/30: train_loss=X.XX val_loss=Y.YY val_auroc=Z.ZZ
```
→ Should see val metrics (not NaN), val_auroc should increase

**Stage 3: Inference**
```
Processing test area in N×M windows...
```
→ Should complete without errors

**Stage 4: Outputs**
```
Saved: test_susceptibility.tif, test_uncertainty.tif, etc.
```
→ Check file sizes are reasonable (>1MB each)

## Key Metrics to Monitor

| Metric | Before Fix | Target After Fix | Verified? |
|--------|------------|------------------|-----------|
| **Dataset** |
| Training tiles | 8 | >30 | [ ] |
| Validation tiles | 0 | >5 | [ ] |
| Test tiles | 0 | >8 | [ ] |
| **Training** |
| Val AUROC | NaN | >0.80 | [ ] |
| Val AUPRC | NaN | >0.70 | [ ] |
| Val Macro IoU | NaN | >0.60 | [ ] |
| Best epoch | null | 15-25 | [ ] |
| **Inference** |
| Mean susceptibility | 0.91 | 0.40-0.60 | [ ] |
| High risk % | 92.4% | 30-60% | [ ] |
| Uncertainty mean | 0.046 | 0.05-0.15 | [ ] |
| Spatial variation | Low | High | [ ] |

## Success Criteria

Before considering the model ready for use:

- [x] Configuration updated with fixes
- [ ] Pipeline re-run completed successfully
- [ ] 30+ training tiles created
- [ ] 5+ validation tiles created
- [ ] 8+ test tiles created
- [ ] Validation AUROC > 0.80
- [ ] Test AUROC > 0.80
- [ ] Mean susceptibility between 0.3-0.7
- [ ] High risk classification < 70% of area
- [ ] Uncertainty estimates are meaningful
- [ ] Visual inspection confirms spatial patterns
- [ ] Model card updated with new metrics
- [ ] Evaluation report shows improvements

## References

- **AGENTS.md** - Repository guide
- **config.yaml** - Updated configuration (check current values)
- **outputs/evaluation/EVALUATION_SUMMARY.md** - Problem diagnosis
- **outputs/evaluation/COMPREHENSIVE_EVALUATION_REPORT.md** - Detailed analysis
- **descriptive_script.md** - Domain context

## Timeline

- **2025-10-20 (Before):** Issues identified via evaluation
- **2025-10-20 (Current):** Configuration fixes applied
- **Next:** Re-run pipeline and verify improvements

---

**Status:** ⚙️ Configuration updated - Ready for pipeline re-run  
**Action Required:** Run `python -m src.main_pipeline --force_recreate`
