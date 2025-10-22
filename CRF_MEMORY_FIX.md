# CRF Memory Layout Fix

## Issue
The CRF post-processing was failing with a Cython memory view error:
```
File "pydensecrf/densecrf.pyx", line 126, in pydensecrf.densecrf.DenseCRF2D.addPairwiseBilateral
File "stringsource", line 660, in View.MemoryView.memoryview_cwrapper
File "stringsource", line 350, in View.MemoryView.memoryview.__cinit__
```

## Root Cause
The `pydensecrf` library requires all numpy arrays to be:
1. **C-contiguous** in memory (not Fortran-contiguous or fragmented)
2. Proper **data types** (float32 for probabilities, uint8 for RGB images)
3. Properly **formatted** (HWC layout for RGB images)

The error occurred because arrays were not guaranteed to be C-contiguous after various numpy operations (transpose, slicing, arithmetic).

## Solution Applied

### 1. Input Array Sanitization
Added explicit C-contiguous conversion at the start of `apply_crf`:
```python
# Ensure input arrays are C-contiguous and proper dtype
probabilities = np.ascontiguousarray(probabilities, dtype=np.float32)
features = np.ascontiguousarray(features, dtype=np.float32)
mask = np.ascontiguousarray(mask, dtype=bool)
```

### 2. Unary Energy Preparation
Ensured unary potentials are C-contiguous:
```python
probs_clipped = np.clip(probabilities, 1e-10, 1.0)
probs_clipped = np.ascontiguousarray(probs_clipped, dtype=np.float32)
unary = unary_from_softmax(probs_clipped)
unary = np.ascontiguousarray(unary, dtype=np.float32)
d.setUnaryEnergy(unary)
```

### 3. Bilateral Kernel RGB Image
Properly prepared the RGB image with guaranteed memory layout:
```python
# Prepare RGB image in HWC format with proper data type and memory layout
rgbim = feat_for_crf.transpose(1, 2, 0)
rgbim = np.ascontiguousarray(rgbim, dtype=np.uint8)
d.addPairwiseBilateral(
    sxy=spatial_weight,
    srgb=color_weight,
    rgbim=rgbim,
    ...
)
```

## Technical Background

### Memory Contiguity
- **C-contiguous**: Row-major order, elements stored row-by-row (C/Python default)
- **Fortran-contiguous**: Column-major order, elements stored column-by-column
- **Non-contiguous**: Array is a view with strides, not a continuous memory block

Operations that can break contiguity:
- `transpose()` - creates a view with different strides
- Slicing with steps - may create non-contiguous views
- Arithmetic operations - may return non-contiguous results

### Why pydensecrf Needs This
The library uses Cython for performance, directly accessing numpy array memory buffers. Cython's typed memory views require:
- Contiguous memory for efficient iteration
- Known data types for type safety
- Proper layout for multi-dimensional access

## Testing
Re-run the pipeline to verify CRF post-processing completes successfully:
```bash
python -m src.main_pipeline
```

The CRF stage should now complete without memory view errors.

## Future Considerations
- Consider adding array validation utilities if memory layout issues arise elsewhere
- Monitor CRF performance on large rasters (the `ascontiguousarray` calls may copy data)
- For extremely large scenes, CRF could be applied on tiles rather than the full scene
