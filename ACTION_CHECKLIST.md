# ✅ Action Checklist: Fixed 3-Class Ordinal Pipeline

## Immediate Actions

### ☐ 1. Review the Fix
- [ ] Read `ORDINAL_FIX_SUMMARY.md` - High-level overview
- [ ] Read `ORDINAL_3CLASS_FIX.md` - Technical details
- [ ] Understand why old evaluation was wrong

### ☐ 2. Verify Environment
```bash
# CRITICAL: Activate virtual environment first!
source .venv/bin/activate

# Check if dependencies are installed
python -c "import rasterio, scipy, sklearn; print('✅ Dependencies OK')"

# If not, install (with venv activated)
pip install -r requirements.txt
```

**Note**: Always use `.venv/bin/python` or activate the venv before running commands!

### ☐ 3. Re-run Inference
```bash
# This regenerates outputs with ordinal susceptibility
# (with venv activated)
python -m src.main_pipeline

# Expected new outputs:
# - outputs/test_susceptibility.tif (ordinal weighted)
# - outputs/test_susceptibility_high.tif (P(high))
# - outputs/test_class_probabilities.tif (3-band)
```

### ☐ 4. Run Fixed Evaluation
```bash
# (with venv activated)
python -m src.evaluate \
  --susceptibility outputs/test_susceptibility.tif \
  --ground_truth /home/kaue/data/landslide/training/Ground_truth_train.tif \
  --output_dir outputs/evaluation
```

### ☐ 5. Check Results
```bash
# View metrics JSON
cat outputs/evaluation/evaluation_metrics.json | grep -A2 "spearman_rho"
cat outputs/evaluation/evaluation_metrics.json | grep "auroc"

# View full report
cat outputs/evaluation/evaluation_report.md

# View plots
ls -lh outputs/evaluation/figures/
```

## Validation Checklist

### ✅ Outputs Generated
- [ ] `outputs/test_susceptibility.tif` exists
- [ ] `outputs/test_class_probabilities.tif` exists (3 bands)
- [ ] `outputs/evaluation/evaluation_metrics.json` exists

### ✅ Metrics Make Sense
- [ ] Spearman's ρ > 0.70 (ordinal correlation is strong)
- [ ] AUROC (Strategy 1) > 0.80 (can identify high risk)
- [ ] AUROC (Strategy 2) > 0.75 (can distinguish risk levels)
- [ ] Confusion matrices show errors mostly between adjacent classes

### ✅ Sanity Checks
- [ ] Ordinal susceptibility range: 0.0 to 1.0 ✓
- [ ] Class probabilities sum to ~1.0 per pixel ✓
- [ ] Ground truth distribution matches expectations
- [ ] No NaN or Inf values in metrics

## If Metrics Are Still Low

### 🔍 Diagnostics

1. **Check Ground Truth Distribution**
   ```bash
   # (with venv activated)
   python -c "
   import rasterio, numpy as np
   with rasterio.open('/home/kaue/data/landslide/training/Ground_truth_train.tif') as src:
       gt = src.read(1)
       unique, counts = np.unique(gt[gt > 0], return_counts=True)
       for val, count in zip(unique, counts):
           print(f'Class {int(val)}: {count:,} pixels')
   "
   ```
   
   **Issue**: Severe class imbalance (e.g., <1% high risk)
   **Fix**: Adjust sampling strategy or class weights

2. **Check Training Performance**
   ```bash
   cat artifacts/experiments/training_metrics.json | grep -A5 "best_metrics"
   ```
   
   **Issue**: Training metrics are also low
   **Fix**: Model didn't learn properly, need to retrain

3. **Check Feature Alignment**
   ```bash
   gdalinfo outputs/test_susceptibility.tif | grep -A2 "Size is"
   gdalinfo /home/kaue/data/landslide/training/Ground_truth_train.tif | grep -A2 "Size is"
   ```
   
   **Issue**: Size mismatch
   **Fix**: Verify preprocessing aligned everything to DTM grid

### 🔧 Potential Improvements

If evaluation shows model isn't learning well:

1. **Use Ordinal Regression Loss**
   - Replace cross-entropy with ordinal-aware loss
   - Penalizes distant misclassifications (1→3) more than adjacent (1→2)

2. **Adjust Class Weights**
   - If classes are imbalanced, weight them in loss function
   - See `config.yaml` training section

3. **More Data/Features**
   - Ensure all relevant features are included
   - Check if external LULC data helps

4. **Architecture Changes**
   - Try different encoders (ResNet50, EfficientNet)
   - Adjust dropout probability

## Success Criteria

### ✅ Minimum Viable
- Spearman's ρ > 0.60
- AUROC (either strategy) > 0.70
- Model shows some discrimination ability

### ✅ Good Performance
- Spearman's ρ > 0.75
- AUROC (both strategies) > 0.80
- Confusion mostly between adjacent classes

### ✅ Excellent Performance
- Spearman's ρ > 0.85
- AUROC > 0.90
- Strong ordinal relationship preserved
- High practical utility

## Documentation Updates

### ☐ After Successful Evaluation
- [ ] Update `README.md` with new output descriptions
- [ ] Add evaluation results to project docs
- [ ] Consider updating `AGENTS.md` with new evaluation flow
- [ ] Archive old evaluation results (mark as invalid)

## Questions to Ask

1. **What's the Spearman correlation?**
   - This is THE key metric for ordinal problems
   - Tells you if model preserves risk ordering

2. **Which strategy performs better?**
   - Strategy 1 (high vs rest): Stricter, for critical zones
   - Strategy 2 (risk vs low): Broader, for general mapping

3. **Where are the errors?**
   - Look at confusion matrices
   - Are errors between adjacent classes (OK) or distant (bad)?

4. **Do optimal thresholds make sense?**
   - Youden vs F1 thresholds
   - Choose based on your risk tolerance

## Git Commit

After validation:
```bash
git add src/inference.py src/evaluate.py
git add ORDINAL_3CLASS_FIX.md ORDINAL_FIX_SUMMARY.md QUICKSTART_FIXED_EVAL.md ACTION_CHECKLIST.md
git commit -m "Fix: Ordinal 3-class evaluation pipeline

- Compute ordinal susceptibility score (weighted by risk)
- Save all 3 class probabilities as multi-band GeoTIFF
- Implement multi-strategy evaluation (high vs rest, risk vs low, Spearman correlation)
- Generate comprehensive metrics and plots for each strategy
- Add detailed documentation

Fixes evaluation metrics that were showing random performance due to incorrect binarization of ordinal ground truth."
```

---

**Next Step**: Complete checkboxes above, starting with #1! 🚀

**Need Help?**: Share your evaluation results and we can diagnose together.
