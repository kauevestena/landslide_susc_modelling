# Pipeline Fixes Status

**Date:** 2025-10-20  
**Issue:** Insufficient tiles generated (8 train, 0 val, 0 test)  
**Root Cause:** Tile size too large (512px) for small training area (1574×1607px)

---

## ✅ Configuration Fixed

The following changes have been applied to `config.yaml`:

### 1. Tile Size Reduction
- **Before:** `tile_size: 512`, `tile_overlap: 64`
- **After:** `tile_size: 256`, `tile_overlap: 32`
- **Expected Impact:** ~4× more tiles (from 8 to ~32)

### 2. Relaxed Quality Thresholds
- `min_valid_fraction: 0.55` → `0.4` (preprocessing)
- `min_valid_fraction: 0.5` → `0.4` (dataset)
- `positive_min_fraction: 0.01` → `0.005` (0.5% landslide pixels)
- `positive_fraction: 0.5` → `0.3` (30% positive tiles)
- **Expected Impact:** More tiles pass quality filters

### 3. Extended Training Schedule
- `epochs: 20` → `30`
- `early_stopping_patience: 5` → `7`
- **Expected Impact:** Better convergence with proper validation

---

## ❌ Artifacts NOT Yet Regenerated

The current artifacts still reflect the **old configuration**:

```
artifacts/splits/dataset_summary.json:
  - tile_size: 512 (OLD)
  - train: 8, val: 0, test: 0 (OLD)

artifacts/tiles/:
  - train/: 8 tiles ✓
  - val/: EMPTY ❌
  - test/: EMPTY ❌

artifacts/labels/:
  - train/: 8 tiles ✓
  - val/: EMPTY ❌
  - test/: EMPTY ❌
```

---

## 🚀 Next Step: Force Regeneration

Run the pipeline with `--force_recreate` to apply the new configuration:

```bash
.venv/bin/python -m src.main_pipeline --force_recreate
```

### What This Will Do:

1. **Delete old artifacts** (splits, tiles, labels, metadata, experiments)
2. **Regenerate with NEW config:**
   - Smaller tiles (256×256)
   - Relaxed thresholds
   - Expected: ~32 tiles total
   - Expected distribution:
     - Train: ~21 tiles (65%)
     - Val: ~5 tiles (15%)
     - Test: ~6 tiles (20%)

3. **Retrain model** with proper validation
4. **Run inference** on test area
5. **Generate outputs** with better calibration

### Expected Duration:
- Preprocessing: ~5-10 min
- Training: ~15-30 min (30 epochs, CPU)
- Inference: ~10-15 min
- **Total: ~30-55 min**

---

## 📊 Expected Results After Regeneration

### Dataset Summary (artifacts/splits/dataset_summary.json)
```json
{
  "tile_size": 256,  // ← Should be 256 now
  "tile_counts": {
    "train": 21,     // ← Should be 20-25
    "val": 5,        // ← Should be 4-6
    "test": 6        // ← Should be 5-8
  }
}
```

### Directory Contents
```
artifacts/tiles/train/  → 21 .npy files
artifacts/tiles/val/    → 5 .npy files ✓
artifacts/tiles/test/   → 6 .npy files ✓

artifacts/labels/train/ → 21 .npy files
artifacts/labels/val/   → 5 .npy files ✓
artifacts/labels/test/  → 6 .npy files ✓
```

### Training Metrics (artifacts/experiments/training_metrics.json)
```json
{
  "val_loss": [...]         // ← Should have values (not NaN)
  "val_macro_iou": [...]    // ← Should have values
  "best_epoch": 15          // ← Should be < 30
}
```

### Model Performance
- **Train AUROC:** 0.80-0.90 (expected)
- **Val AUROC:** 0.75-0.85 (expected)
- **Test High Risk %:** 10-30% (expected, was 92%!)
- **Mean Susceptibility:** 0.3-0.5 (expected, was 0.91!)

---

## 🔍 Validation Checklist

After running `--force_recreate`, verify:

### 1. Dataset Generation
```bash
cat artifacts/splits/dataset_summary.json | jq '.tile_counts'
```
Expected: `{"train": 21, "val": 5, "test": 6}` (approximately)

### 2. Files Created
```bash
ls artifacts/tiles/train/ | wc -l   # Should be ~21
ls artifacts/tiles/val/ | wc -l     # Should be ~5
ls artifacts/tiles/test/ | wc -l    # Should be ~6
```

### 3. Training Metrics
```bash
cat artifacts/experiments/training_metrics.json | jq '.val_macro_iou[-1]'
```
Expected: 0.30-0.50 (not NaN or null)

### 4. Susceptibility Statistics
```bash
.venv/bin/python -m src.evaluate --analysis_only
cat outputs/evaluation/output_statistics.json | jq '.susceptibility_stats.mean'
```
Expected: 0.30-0.50 (not 0.91!)

### 5. Risk Distribution
```bash
cat outputs/evaluation/output_statistics.json | jq '.risk_distribution'
```
Expected high_risk: 10-30% (not 92%!)

---

## 🐛 If Issues Persist

### Issue 1: Still 0 Val/Test Tiles
**Diagnosis:** Training area still too small for split
**Solution:** Further reduce tile_size to 128 or use stratified sampling

### Issue 2: Still Extreme Predictions
**Diagnosis:** Model overfitting or class imbalance
**Solution:** Check `positive_fraction` in config, verify label encoding

### Issue 3: Training Crashes
**Diagnosis:** Out of memory or data loading issue
**Solution:** Reduce `batch_size`, check tile integrity

---

## 📝 Summary

| Status | Component |
|--------|-----------|
| ✅ | Config updated with fixes |
| ✅ | Python environment ready |
| ❌ | Artifacts NOT regenerated yet |
| ⏳ | Awaiting `--force_recreate` run |

**Current State:** Ready to regenerate  
**Action Required:** Run force_recreate command  
**ETA to Fixed Model:** ~30-55 minutes

---

## Command to Run Now

```bash
.venv/bin/python -m src.main_pipeline --force_recreate
```

Then verify with:

```bash
# Check dataset
cat artifacts/splits/dataset_summary.json | jq '.tile_counts'

# Re-evaluate
.venv/bin/python -m src.evaluate --analysis_only

# Check results
cat outputs/evaluation/EVALUATION_SUMMARY.md
```

---

**Status:** 🟡 Configuration fixed, awaiting artifact regeneration  
**Next Action:** Run `--force_recreate` command above
