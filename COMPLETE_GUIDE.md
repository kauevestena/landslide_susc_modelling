# 🚀 Complete Setup & Evaluation Guide

## 📋 Pre-Flight Checklist

### ✅ Step 0: Virtual Environment (MANDATORY!)

**Before anything else**, set up your virtual environment:

```bash
# 1. Create virtual environment (first time only)
cd /home/kaue/landslide_susc_modelling
python3 -m venv .venv

# 2. Activate it (EVERY session)
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import rasterio, scipy, sklearn, torch; print('✅ Ready to go!')"
```

**See [VENV_SETUP.md](VENV_SETUP.md) for detailed instructions.**

---

## 🔧 The Fix: Ordinal 3-Class Evaluation

### What Was Broken
Your evaluation showed random performance (AUROC ~0.50) because:
- Ground truth: **1=Low, 2=Medium, 3=High** (ordinal classes)
- Old evaluation: Treated only class 2 as "positive" ❌
- This is wrong for ordinal problems!

### What Was Fixed
1. **Inference** - Now outputs:
   - Ordinal susceptibility (weighted score)
   - All 3 class probabilities
   - High-risk probability separately

2. **Evaluation** - Multi-strategy approach:
   - Strategy 1: High vs Rest
   - Strategy 2: At-Risk vs Low
   - **Spearman correlation** (key ordinal metric!)

**Full details**: [ORDINAL_FIX_SUMMARY.md](ORDINAL_FIX_SUMMARY.md)

---

## 🎯 Run the Fixed Pipeline

### 1. Re-run Inference
```bash
# With venv activated!
python -m src.main_pipeline
```

**Outputs**:
- `outputs/test_susceptibility.tif` - **Ordinal weighted score** (primary)
- `outputs/test_susceptibility_high.tif` - P(high risk)
- `outputs/test_class_probabilities.tif` - All 3 class probabilities
- `outputs/test_uncertainty.tif` - Uncertainty
- `outputs/test_class_map.tif` - Argmax classes
- `outputs/test_valid_mask.tif` - Valid pixels

### 2. Run Evaluation
```bash
# With venv activated!
python -m src.evaluate \
  --susceptibility outputs/test_susceptibility.tif \
  --ground_truth /home/kaue/data/landslide/training/Ground_truth_train.tif \
  --output_dir outputs/evaluation
```

**Outputs**:
- `outputs/evaluation/evaluation_metrics.json` - All metrics
- `outputs/evaluation/evaluation_report.md` - Comprehensive report
- `outputs/evaluation/figures/` - ROC/PR curves for each strategy

### 3. Check Results
```bash
# View key metrics
cat outputs/evaluation/evaluation_metrics.json | grep -A2 "spearman_rho"
cat outputs/evaluation/evaluation_metrics.json | grep "auroc"

# Read full report
less outputs/evaluation/evaluation_report.md

# View visualizations
ls -lh outputs/evaluation/figures/
```

---

## 📊 Interpreting Results

### Good Performance Indicators

| Metric | Good | Excellent |
|--------|------|-----------|
| **Spearman's ρ** | > 0.70 | > 0.85 |
| **AUROC (Strategy 1)** | > 0.80 | > 0.90 |
| **AUROC (Strategy 2)** | > 0.75 | > 0.85 |

### What Each Metric Means

**Spearman's ρ (Ordinal Correlation)**:
- Measures if higher predictions → higher ground truth classes
- **KEY METRIC** for ordinal problems!
- > 0.70 = Strong ordinal relationship ✅

**Strategy 1 (High vs Rest)**:
- Can the model identify highest-risk areas?
- Use for prioritizing critical interventions

**Strategy 2 (At-Risk vs Low)**:
- Can the model distinguish risky from safe areas?
- Use for broader hazard mapping

**Confusion Matrices**:
- Check if errors are between adjacent classes (OK)
- Distant errors (1↔3) indicate poor ordinal learning

---

## 🆘 Troubleshooting

### "Module not found" errors
```bash
# SOLUTION: Activate venv!
source .venv/bin/activate
pip install -r requirements.txt
```

### Metrics still low after re-running
1. **Check ground truth distribution**:
   ```bash
   python -c "import rasterio, numpy as np; \
   gt = rasterio.open('/home/kaue/data/landslide/training/Ground_truth_train.tif').read(1); \
   unique, counts = np.unique(gt[gt>0], return_counts=True); \
   print(dict(zip(unique, counts)))"
   ```

2. **Check training metrics**:
   ```bash
   cat artifacts/experiments/training_metrics.json | grep -A5 "best_metrics"
   ```

3. **Severe class imbalance?**
   - Consider adjusting class weights in training
   - Use stratified sampling

4. **Model didn't learn well?**
   - Consider ordinal regression loss (not cross-entropy)
   - Try different encoder architectures
   - Increase training epochs

---

## 📚 Documentation Index

| Document | Purpose |
|----------|---------|
| **[VENV_SETUP.md](VENV_SETUP.md)** | Virtual environment setup (START HERE) |
| **[ORDINAL_FIX_SUMMARY.md](ORDINAL_FIX_SUMMARY.md)** | Overview of the ordinal evaluation fix |
| **[ORDINAL_3CLASS_FIX.md](ORDINAL_3CLASS_FIX.md)** | Technical details of the fix |
| **[QUICKSTART_FIXED_EVAL.md](QUICKSTART_FIXED_EVAL.md)** | Quick reference guide |
| **[ACTION_CHECKLIST.md](ACTION_CHECKLIST.md)** | Step-by-step checklist |
| **[AGENTS.md](AGENTS.md)** | Guide for AI agents/collaborators |
| **[README.md](README.md)** | Project overview |

---

## ⚡ Quick Commands Reference

```bash
# Activate venv (ALWAYS FIRST!)
source .venv/bin/activate

# Run pipeline
python -m src.main_pipeline

# Force recreate all artifacts
python -m src.main_pipeline --force_recreate

# Evaluate
python -m src.evaluate \
  --susceptibility outputs/test_susceptibility.tif \
  --ground_truth /path/to/Ground_truth_train.tif

# Check training history
cat artifacts/experiments/training_metrics.json | jq '.best_metrics'

# View dataset summary
cat artifacts/splits/dataset_summary.json

# Deactivate venv when done
deactivate
```

---

## ✨ Success Criteria

Your pipeline is working well if:
- ✅ Spearman's ρ > 0.70 (strong ordinal relationship)
- ✅ AUROC > 0.80 (good discrimination)
- ✅ Confusion errors mostly between adjacent classes
- ✅ Susceptibility maps visually match terrain/features
- ✅ No NaN/Inf values in outputs

---

## 🎓 Next Steps After Validation

1. **Analyze results** - Which strategy performs best for your use case?
2. **Visualize in GIS** - Load outputs in QGIS/ArcGIS
3. **Iterate if needed** - Tune hyperparameters, try different features
4. **Consider improvements**:
   - Ordinal regression loss
   - Ensemble models
   - More training data
   - External validation dataset

---

**Need help?** Check the documentation index above or refer to specific guides! 🚀

**Remember**: Always activate `.venv` before running any commands!
