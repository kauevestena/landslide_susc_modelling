# Test Split Class Distribution Fix

**Date:** 2025-10-27  
**Issue:** Test tiles missing Class 3 (High risk), preventing proper 3-class evaluation  
**Root Cause:** Random spatial block splitting can exclude rare classes by chance  
**Solution:** Intelligent retry mechanism with class distribution validation

---

## Problem Statement

The evaluation analysis revealed:
- **Ground truth contains all 3 classes**: Low (39.54%), Medium (7.52%), High (8.50%)
- **Test tiles contained only 2 classes**: Low (91.55%), Medium (8.45%), High (0%)
- **Impact**: Cannot evaluate Strategy 1 (High vs Rest), incomplete model assessment

This was NOT due to:
- ❌ Incorrect ground truth file paths
- ❌ Data corruption or preprocessing errors
- ❌ Inference bugs

This WAS due to:
- ✅ Random spatial splitting that happened to exclude high-risk areas from test blocks
- ✅ Class 3 being naturally rare (8.5% of pixels) and spatially localized

---

## Solution Architecture

### Two-Stage Validation:

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: BLOCK-LEVEL PRE-CHECK (before tile generation)   │
├─────────────────────────────────────────────────────────────┤
│ For attempt = 1 to max_split_attempts:                     │
│   1. Generate random train/test block split                │
│   2. Scan test blocks for potential tile class coverage    │
│   3. If all classes present → accept split, break          │
│   4. Else track best split and continue                    │
│                                                             │
│ If max attempts exhausted:                                  │
│   → Use best available split                               │
│   → Issue warning about missing classes                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: TILE-LEVEL POST-CHECK (after tile generation)     │
├─────────────────────────────────────────────────────────────┤
│ For each generated test tile:                              │
│   1. Load tile labels (hard or soft)                       │
│   2. Detect which classes are present                      │
│   3. Accumulate test_classes_found set                     │
│                                                             │
│ After all tiles processed:                                 │
│   → Report found classes in console                        │
│   → Save to dataset_summary.json                           │
│   → Warn if any classes missing                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### 1. Block-Level Validation Function

```python
def check_test_has_all_classes(test_block_list, num_classes):
    """
    Scan test blocks to check if they contain tiles with all classes.
    
    Returns:
        tuple: (has_all_classes: bool, classes_found: set)
    """
    classes_found = set()
    
    for each spatial block in test_block_list:
        for each potential tile position:
            if tile is valid:
                extract tile labels
                detect unique classes
                update classes_found set
                
                if len(classes_found) >= num_classes:
                    return True, classes_found
    
    return len(classes_found) >= num_classes, classes_found
```

### 2. Retry Loop with Random Seeds

```python
max_attempts = config["dataset"]["max_split_attempts"]  # Default: 20
best_split = None
best_classes_found = set()

for attempt in range(max_attempts):
    current_seed = random_state + attempt  # 42, 43, 44, ..., 61
    
    train_blocks, test_blocks = train_test_split(
        blocks, test_size=test_size, random_state=current_seed
    )
    
    has_all_classes, classes_found = check_test_has_all_classes(
        test_blocks, num_classes
    )
    
    if has_all_classes:
        # Success! Use this split
        print(f"Found valid split on attempt {attempt + 1}")
        break
    
    # Track best attempt
    if len(classes_found) > len(best_classes_found):
        best_classes_found = classes_found
        best_split = (train_blocks, test_blocks)
else:
    # Max attempts exhausted
    print(f"WARNING: Could not find complete split after {max_attempts} attempts")
    if best_split:
        train_blocks, test_blocks = best_split
```

### 3. Tile-Level Validation (Post-Generation)

```python
test_classes_found = set()

for tile_name in test_tiles:
    tile_labels = np.load(tile_path)
    
    if soft_labels:
        # Check which classes have significant probability
        for cls_idx in range(num_classes):
            if np.any(tile_labels[cls_idx] > 0.1):
                test_classes_found.add(cls_idx)
    else:
        # Check discrete class assignments
        unique_classes = np.unique(tile_labels[tile_labels != 255])
        test_classes_found.update(unique_classes)

# Report results
if len(test_classes_found) == num_classes:
    print(f"✓ Test split contains all {num_classes} classes")
else:
    missing = set(range(num_classes)) - test_classes_found
    print(f"WARNING: Test split missing classes: {missing}")
```

---

## Configuration Parameters

### In `config.yaml`:

```yaml
dataset:
  # Existing parameters
  test_size: 0.2            # Fraction of blocks for testing
  val_size: 0.15            # Fraction of blocks for validation
  block_grid: [1, 1]        # Spatial block subdivision (rows, cols)
  random_state: 42          # Initial random seed
  
  # NEW PARAMETER
  max_split_attempts: 20    # Max tries to find split with all classes
```

### Tuning Guidance:

| Parameter | Current | Increase If | Decrease If |
|-----------|---------|-------------|-------------|
| `max_split_attempts` | 20 | Classes consistently missing | Slow startup, acceptable to miss rare classes |
| `test_size` | 0.2 | Rare classes excluded | Need more training data |
| `block_grid` | [1, 1] | Need more split diversity | Blocks too small, overfitting risk |

---

## Output Changes

### Console Output (Success):

```
[main_pipeline] prepare_dataset: found valid test split on attempt 3 with all 3 classes: [0, 1, 2]
[main_pipeline] prepare_dataset: ✓ Test split contains all 3 classes: [0, 1, 2]
```

### Console Output (Partial Success):

```
[main_pipeline] prepare_dataset: WARNING - could not find test split with all 3 classes after 20 attempts.
Best split found classes: [0, 1]. Consider increasing test_size or max_split_attempts.
[main_pipeline] prepare_dataset: WARNING - Test split missing classes: [2]. Found classes: [0, 1].
This may affect evaluation. Consider increasing test_size or max_split_attempts.
```

### `dataset_summary.json` Enhancement:

```json
{
  "tile_size": 256,
  "tile_overlap": 128,
  "tile_counts": {
    "train": 51,
    "val": 12,
    "test": 16
  },
  "test_classes_found": [0, 1, 2],  ← NEW FIELD
  "class_pixel_counts": {
    "0": 2093964.49,
    "1": 697974.48,
    "2": 282487.41
  },
  ...
}
```

---

## Class Imbalance: Expected Behavior

### ⚠️ Important: Class Imbalance is NOT a Bug

**Ground Truth Distribution (typical):**
- Class 0 (Low risk): ~40-90% of pixels
- Class 1 (Medium risk): ~7-15% of pixels
- Class 2 (High risk): ~5-10% of pixels

**Why This is Correct:**
1. **Physical Reality:** High-risk landslide areas are spatially rare
   - Require specific conditions: steep slopes + unstable geology + triggering factors
   - Most terrain is stable (low risk)

2. **Ordinal Nature:** Risk classes form a natural hierarchy
   - Low → Medium → High represents increasing severity
   - Not all areas can be high risk (otherwise "high" loses meaning)

3. **Spatial Clustering:** Landslide susceptibility is spatially autocorrelated
   - High-risk zones cluster together (geologic formations)
   - Low-risk zones dominate flat/stable areas

**Pipeline Design Handles Imbalance:**
- ✅ **Focal loss:** Emphasizes learning from rare/hard examples
- ✅ **Positive fraction sampling:** Oversamples at-risk tiles during training
- ✅ **Ordinal soft labels:** Encodes class relationships and reduces boundary ambiguity
- ✅ **Cohen's Kappa:** Evaluation metric that accounts for imbalance

**DO NOT Artificially Balance Classes:**
- ❌ Do not oversample/duplicate high-risk tiles excessively
- ❌ Do not undersample low-risk areas aggressively
- ❌ Do not use class weights that drastically skew predictions

**Result:** The model learns realistic spatial distributions that generalize to new areas.

---

## Validation Checklist

After running pipeline with `--force_recreate`:

- [ ] Console shows "found valid test split on attempt N"
- [ ] Console shows "Test split contains all 3 classes: [0, 1, 2]"
- [ ] `artifacts/splits/dataset_summary.json` has `"test_classes_found": [0, 1, 2]`
- [ ] Test tile count > 0 in `dataset_summary.json`
- [ ] No warnings about missing classes

If validation fails:
1. Check `max_split_attempts` in config.yaml (try 30-50)
2. Increase `test_size` (try 0.25 or 0.3)
3. Consider `block_grid: [2, 2]` for more split options
4. Verify ground truth contains all classes:
   ```bash
   .venv/bin/python -c "import rasterio; import numpy as np; \
   src = rasterio.open('path/to/ground_truth.tif'); \
   data = src.read(1); print(np.unique(data, return_counts=True))"
   ```

---

## Expected Impact on Evaluation

### Before Fix:
```
Strategy 1: High Risk (Class 3) vs Rest (Classes 1-2)
  Error: Only one class present  ← BLOCKED

Strategy 2: At-Risk (Classes 2-3) vs Low (Class 1)
  AUROC: 0.6178
  Cohen's Kappa: 0.0292  ← Very poor (missing high-risk evaluation)
```

### After Fix:
```
Strategy 1: High Risk (Class 3) vs Rest (Classes 1-2)
  AUROC: [expected improvement]
  Cohen's Kappa: [computable with all classes]

Strategy 2: At-Risk (Classes 2-3) vs Low (Class 1)
  AUROC: [expected similar or better]
  Cohen's Kappa: [improved with complete class coverage]
```

### Metrics Expected to Improve:
- ✅ **Strategy 1** now computable (was blocked)
- ✅ **Cohen's Kappa** more representative (all classes evaluated)
- ✅ **Confusion matrices** complete for all strategies
- ✅ **ROC/PR curves** for high-risk class available

### Metrics May Appear Worse (But More Honest):
- ⚠️ **Precision** may decrease (high-risk is hard to predict)
- ⚠️ **AUROC** may be lower for Strategy 1 (Class 3 vs rest is hardest task)
- ⚠️ **F1 score** reflects true difficulty of rare class prediction

**Interpretation:** Lower metrics with complete evaluation are more valuable than inflated metrics from incomplete evaluation!

---

## Troubleshooting

### Issue: Still missing classes after 20 attempts

**Solution 1:** Increase `max_split_attempts`
```yaml
dataset:
  max_split_attempts: 50  # Try more random seeds
```

**Solution 2:** Increase `test_size`
```yaml
dataset:
  test_size: 0.3  # More blocks = higher chance of coverage
```

**Solution 3:** Add spatial block subdivision
```yaml
dataset:
  block_grid: [2, 2]  # 4 blocks instead of 1 = more split options
```

### Issue: Only Class 2 missing, not Class 3

Check if rare class is spatially isolated:
```python
# Visualize class spatial distribution
import rasterio
import matplotlib.pyplot as plt

with rasterio.open('ground_truth.tif') as src:
    data = src.read(1)
    plt.imshow(data, cmap='viridis')
    plt.colorbar()
    plt.title('Ground Truth Class Distribution')
    plt.savefig('class_distribution.png')
```

If rare class is in one corner → increase `test_size` or adjust `block_grid`

### Issue: Performance degraded after fix

This is expected! Previous evaluation was incomplete:
- Before: Evaluated only easy classes (Low vs Medium)
- After: Evaluates all classes including hardest (High risk)

**Lower metrics with complete evaluation > Higher metrics with missing data**

Use Cohen's Kappa to compare fairly across runs.

---

## Files Modified

1. **`src/main_pipeline.py`**
   - Added `check_test_has_all_classes()` function (lines ~1145-1180)
   - Implemented retry loop with random seeds (lines ~1185-1225)
   - Added tile-level validation (lines ~1465-1500)
   - Enhanced `dataset_summary.json` with `test_classes_found` field

2. **`config.yaml`**
   - Added `dataset.max_split_attempts: 20` parameter

3. **`AGENTS.md`**
   - Added troubleshooting entry for "Test split missing classes"
   - Added note about class imbalance being inherent to landslide susceptibility
   - Clarified evaluation best practices (use Cohen's Kappa)

4. **`TEST_SPLIT_FIX.md`** (this document)
   - Comprehensive documentation of the fix

---

## Next Steps

1. **Regenerate tiles with fix:**
   ```bash
   .venv/bin/python -m src.main_pipeline --force_recreate
   ```

2. **Verify test classes:**
   ```bash
   cat artifacts/splits/dataset_summary.json | .venv/bin/python -m json.tool | grep -A 3 test_classes
   ```

3. **Re-run evaluation:**
   ```bash
   .venv/bin/python -m src.evaluate \
     --susceptibility outputs/test_susceptibility.tif \
     --ground_truth path/to/ground_truth.tif \
     --uncertainty outputs/test_uncertainty.tif \
     --output_dir outputs/evaluation_fixed
   ```

4. **Compare results:**
   - Check if Strategy 1 now works
   - Review Cohen's Kappa improvement
   - Analyze per-class metrics

---

**Status:** Ready for testing. Run pipeline with `--force_recreate` to apply fix.
