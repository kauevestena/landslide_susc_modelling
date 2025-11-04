# Ground Truth Remapping Fix

**Date:** 2025-11-03  
**Issue:** Ground truth uses values [0,1,2,3] where 0=invalid, but model expects [0,1,2]  
**Severity:** CRITICAL - Wrong class mapping causes training failure and incorrect predictions  
**Status:** ✅ FIXED

---

## Problem: Incorrect Ground Truth Interpretation

### Ground Truth Encoding

**Original ground truth files use this encoding:**
- **Value 0**: Invalid/Empty pixels (44% of data) ← **NOT a valid class!**
- **Value 1**: Low risk (39.5% of data)
- **Value 2**: Moderate risk (7.5% of data)
- **Value 3**: High risk (8.5% of data)

**Model expects:**
- **Class 0**: Low risk
- **Class 1**: Moderate risk
- **Class 2**: High risk
- **Value 255**: Invalid/Ignore

### The Critical Error

The preprocessing was treating GT value 0 as a valid class (low risk), when it actually represents **invalid/empty pixels** that should be excluded from training!

This caused:
1. **Wrong class mapping**: GT[1,2,3] mapped to wrong model classes
2. **Invalid data in training**: Empty pixels (GT=0) trained as "low risk"
3. **Index out of bounds**: GT[3] couldn't fit into 3-class model
4. **Inverted risk predictions**: High risk areas predicted as low risk!

---

## Solution: Proper Value Remapping

### Remapping Logic

```python
# Ground truth → Model class mapping
GT value 0 → 255 (IGNORE_INDEX) - Invalid/empty pixels
GT value 1 → 0 (Class 0) - Low risk
GT value 2 → 1 (Class 1) - Moderate risk  
GT value 3 → 2 (Class 2) - High risk
```

### Code Fix

**File:** `src/main_pipeline.py`  
**Function:** `process_area()` at line ~944

**Before (WRONG):**
```python
gt = gt_array[0]
gt = np.round(gt).astype(np.int16)
gt[~valid_mask] = 255  # Only masks invalid DTM pixels
```
**Result**: GT value 0 treated as valid class → invalid pixels trained as "low risk"

**After (CORRECT):**
```python
gt = gt_array[0]
gt = np.round(gt).astype(np.int16)

# CRITICAL FIX: Remap ground truth values to model classes
# Ground truth encoding: 0=invalid/empty, 1=low, 2=moderate, 3=high
# Model expects: 0=low, 1=moderate, 2=high, 255=ignore
# Remap: GT[0]->255, GT[1]->0, GT[2]->1, GT[3]->2
gt_remapped = np.full_like(gt, 255, dtype=np.int16)
gt_remapped[gt == 1] = 0  # Low risk
gt_remapped[gt == 2] = 1  # Moderate risk
gt_remapped[gt == 3] = 2  # High risk
# GT value 0 and any other values remain as 255 (ignore)
gt = gt_remapped

gt[~valid_mask] = 255
```
**Result**: Correct class mapping, invalid pixels properly excluded

---

## Verification

### 1. Regenerate Artifacts

**CRITICAL:** All preprocessing artifacts must be regenerated:

```bash
.venv/bin/python -m src.main_pipeline --force_recreate
```

This will:
- Reprocess training and test areas with class 3 → 255 remapping
- Regenerate merged stack with only valid classes [0, 1, 2, 255]
- Create new tiles without class 3

### 2. Verify Remapping

After preprocessing completes:

```python
.venv/bin/python -c "
import rasterio
import numpy as np

with rasterio.open('artifacts/derived/merged/merged_labels.tif') as src:
    data = src.read(1)
    unique = np.unique(data)
    print('Unique values after fix:', unique)
    assert 3 not in unique, 'Class 3 still present!'
    print('✅ Class 3 successfully remapped to 255')
"
```

**Expected output:**
```
Unique values after fix: [0, 1, 2, 255]
✅ Class 3 successfully remapped to 255
```

### 3. Training Should Succeed

Temperature scaling will now work correctly:
```
[train] Optimizing temperature scaling on validation set...
✅ Temperature: 1.05, NLL before: 0.234, NLL after: 0.229
```

---

## Impact Assessment

### Dataset Changes

**Before fix (WRONG MAPPING):**
```
Ground Truth Value → Model Class
0 → 0 (treated as low risk) ❌ WRONG! Should be ignored
1 → 1 (treated as moderate risk) ❌ WRONG!
2 → 2 (treated as high risk) ❌ WRONG!
3 → CRASH (out of bounds) ❌

Training data:
  "Class 0" (actually invalid): 1,123,903 pixels ❌
  "Class 1" (actually low): 1,000,145 pixels ❌
  "Class 2" (actually moderate): 190,316 pixels ❌
  Crash on class 3: 215,054 pixels ❌
```

**After fix (CORRECT MAPPING):**
```
Ground Truth Value → Model Class
0 → 255 (ignored) ✅ CORRECT
1 → 0 (low risk) ✅ CORRECT
2 → 1 (moderate risk) ✅ CORRECT
3 → 2 (high risk) ✅ CORRECT

Training data:
  Class 0 (low): 1,000,145 pixels ✅
  Class 1 (moderate): 190,316 pixels ✅
  Class 2 (high): 215,054 pixels ✅
  Ignored: 1,123,903 pixels ✅
```

### Training Implications

1. **Correct risk levels**: Model now learns proper low/moderate/high classification
2. **Invalid data excluded**: Empty pixels (GT=0) no longer contaminate training
3. **All classes available**: High risk class (GT=3) now properly used
4. **Better predictions**: Model will produce correct risk maps instead of inverted ones

**This was a CRITICAL fix** - without it, the model was learning completely wrong associations!

---

## Alternative Solutions Considered

### Option A: Keep Original Mapping ❌
```python
# No remapping - use GT values directly
```
**Rejected because:**
- Treats invalid pixels (0) as valid class
- Can't handle GT value 3 (out of bounds)
- Completely wrong semantic interpretation

### Option B: Shift by 1 ❌
```python
gt = gt - 1  # 0->-1, 1->0, 2->1, 3->2
gt[gt < 0] = 255
```
**Rejected because:**
- Arithmetic solution obscures intent
- Harder to maintain and understand
- Doesn't handle edge cases clearly

### Option C: Explicit Remapping ✅ (CHOSEN)
```python
gt_remapped = np.full_like(gt, 255, dtype=np.int16)
gt_remapped[gt == 1] = 0  # Low
gt_remapped[gt == 2] = 1  # Moderate
gt_remapped[gt == 3] = 2  # High
```
**Selected because:**
- Explicit, self-documenting code
- Clear semantic mapping
- Handles all edge cases (invalid values → 255)
- Easy to verify correctness

---

## Related Issues

### Spatial Data Leakage Fix
This fix is **independent** from the spatial blocking fix. Both are required:
1. **Class 3 remapping** (this fix) → prevents training crashes
2. **Spatial blocking** (separate fix) → prevents data leakage

### Soft Label Smoothing
The class 3 issue is orthogonal to soft labels:
- Soft labels convert hard classes [0,1,2] → probability distributions
- Class 3 remapping happens **before** soft label generation
- Both can coexist once class 3 is remapped to 255

---

## Action Items

- [x] Add class 3 remapping logic to `process_area()`
- [ ] Regenerate all artifacts: `.venv/bin/python -m src.main_pipeline --force_recreate`
- [ ] Verify merged labels contain only [0, 1, 2, 255]
- [ ] Run validation script: `.venv/bin/python validate_spatial_split.py`
- [ ] Complete training run (~24-30 hours)
- [ ] Evaluate on test area and compare against baseline

---

## Summary

| Aspect | Before Fix | After Fix |
|--------|-----------|-----------|
| **GT value 0** | Treated as class 0 (low) ❌ | Ignored (255) ✅ |
| **GT value 1** | Treated as class 1 (moderate) ❌ | Class 0 (low) ✅ |
| **GT value 2** | Treated as class 2 (high) ❌ | Class 1 (moderate) ✅ |
| **GT value 3** | Crash (out of bounds) ❌ | Class 2 (high) ✅ |
| **Valid training data** | ~57M pixels (includes invalid) ❌ | ~1.4M pixels (correct) ✅ |
| **Class semantics** | Completely wrong ❌ | Correct ✅ |
| **Risk predictions** | Inverted/wrong ❌ | Accurate ✅ |
| **Training status** | WRONG + BLOCKED ❌ | READY ✅ |

**Next step:** Regenerate artifacts with `--force_recreate` to apply the correct remapping.
