# Summary: Fixed 3-Class Ordinal Landslide Susceptibility Pipeline

## 🔍 Root Cause Analysis

### The Problem
Your evaluation showed **terrible metrics** (AUROC ~0.50 = random chance) because:

```
Ground Truth:  1, 2, 3  (Low, Medium, High landslide probability)
                ↓  ↓  ↓
Old Evaluation: ❌  ✅  ❌  (Only class 2 = positive, rest = negative)
```

This is **completely wrong** for an ordinal 3-class problem where:
- Class 1 < Class 2 < Class 3 (ordered categories)
- All classes should be evaluated, not just one!

### What Actually Happens in Training
```python
# src/main_pipeline.py line 1069-1070
labels = np.where(
    ignore_mask, 255, np.clip(labels, 1, config["model"]["out_classes"]) - 1
)
# Remaps: 1→0, 2→1, 3→2  ✅ Correct!
```

The model **trains correctly** on 3 classes (0, 1, 2), but:
1. Inference only saved P(class=2) as "susceptibility"
2. Evaluation compared this to (ground_truth == 2), treating medium as positive!

## ✅ Solution: Multi-Strategy Ordinal Evaluation

### New Inference Outputs

| File | Description | Range |
|------|-------------|-------|
| `test_susceptibility.tif` | **Ordinal weighted score**: 0×P(low) + 0.5×P(med) + 1×P(high) | 0.0 - 1.0 |
| `test_susceptibility_high.tif` | Calibrated P(high risk) | 0.0 - 1.0 |
| `test_class_probabilities.tif` | 3-band: [P(low), P(med), P(high)] | 0.0 - 1.0 each |
| `test_uncertainty.tif` | Uncertainty estimate | varies |
| `test_class_map.tif` | Argmax class | 0, 1, 2 |
| `test_valid_mask.tif` | Valid pixels | 0, 1 |

### New Evaluation Strategies

#### Strategy 1: High vs Rest
```
Positive: Class 3 (High risk)
Negative: Classes 1-2 (Low/Medium)

Use case: Identify critical intervention zones
```

#### Strategy 2: At-Risk vs Low
```
Positive: Classes 2-3 (Medium/High risk)
Negative: Class 1 (Low risk)

Use case: Broader hazard mapping
```

#### Strategy 3: Ordinal Correlation
```
Spearman's ρ between predicted probabilities and ground truth classes

Measures: How well the model preserves ordinal relationships
Perfect: ρ = 1.0 (higher predictions → higher classes)
Random: ρ ≈ 0.0
```

## 📊 Metrics You'll Get

```json
{
  "ground_truth_encoding": {
    "class_1": "Low landslide probability",
    "class_2": "Medium landslide probability",
    "class_3": "High landslide probability"
  },
  "ordinal_correlation": {
    "spearman_rho": 0.85,  // ← Key metric!
    "p_value": 1.2e-10
  },
  "strategy_1_high_vs_rest": {
    "auroc": 0.92,
    "auprc": 0.87,
    "optimal_thresholds": {...},
    "confusion_matrix": {...}
  },
  "strategy_2_risk_vs_low": {
    "auroc": 0.89,
    "auprc": 0.84,
    ...
  }
}
```

## 🎯 Expected Performance

| Metric | Good | Excellent |
|--------|------|-----------|
| Spearman's ρ | > 0.70 | > 0.85 |
| AUROC (Strategy 1) | > 0.80 | > 0.90 |
| AUROC (Strategy 2) | > 0.75 | > 0.85 |
| AUPRC | >> random | > 0.80 |

## 🔧 Files Modified

1. **`src/inference.py`** (+80 lines)
   - Added `write_multiband_geotiff()` for 3-band output
   - Compute ordinal susceptibility: weighted class probabilities
   - Save all 3 probabilities + high-risk prob separately

2. **`src/evaluate.py`** (+150 lines, refactored)
   - New `evaluate_binary_strategy()` helper function
   - Multi-strategy evaluation (2 binary + 1 ordinal)
   - Spearman correlation computation
   - Updated report generation for all strategies

3. **New docs**:
   - `ORDINAL_3CLASS_FIX.md` - Technical details
   - `QUICKSTART_FIXED_EVAL.md` - Quick start guide

## 🚀 Next Steps

### 0. Activate Virtual Environment (CRITICAL!)
```bash
source .venv/bin/activate  # Linux/macOS
# OR: .venv\Scripts\activate  # Windows
```

### 1. Re-run Inference
```bash
# With venv activated:
python -m src.main_pipeline
```
This regenerates outputs with ordinal scoring.

### 2. Run Fixed Evaluation
```bash
# With venv activated:
python -m src.evaluate \
  --susceptibility outputs/test_susceptibility.tif \
  --ground_truth /path/to/Ground_truth_train.tif
```

### 3. Analyze Results
- Check Spearman's ρ (should be > 0.70)
- Compare AUROC across both strategies
- Review confusion matrices (errors should be mostly between adjacent classes)

### 4. If Performance is Still Low
- **Class imbalance**: Check `ground_truth_distribution` in metrics
- **Wrong loss function**: Consider ordinal regression loss instead of cross-entropy
- **Training issues**: Review `artifacts/experiments/training_metrics.json`
- **Feature quality**: Ensure DTM and orthophoto are properly aligned

## 📖 Theory: Why Ordinal Matters

### Cross-Entropy (Current)
```
Loss(1→3) = Loss(1→2) = Loss(2→3)
```
Treats all misclassifications equally ❌

### Ordinal Regression (Future)
```
Loss(1→3) > Loss(1→2)
```
Penalizes distant errors more ✅

Consider implementing ordinal regression for better performance on this ordered problem.

## 🎓 Key Takeaways

1. **Ordinal data ≠ Multi-class**: Need special evaluation strategies
2. **Multiple metrics**: Different use cases need different binarizations
3. **Spearman's ρ**: Critical for measuring ordinal relationships
4. **Weighted scoring**: Ordinal susceptibility better reflects risk continuum
5. **Always check encoding**: Make sure evaluation matches training!

---

**Status**: ✅ Complete - Ready to run
**Impact**: Evaluation metrics will now be meaningful
**Breaking**: Old evaluation results invalid, re-run needed

For questions, see `ORDINAL_3CLASS_FIX.md` or `QUICKSTART_FIXED_EVAL.md`.
