# Quick Start: Fixed 3-Class Ordinal Pipeline

## ✅ What Was Fixed

Your evaluation metrics were broken because:
1. Ground truth has **3 ordinal classes** (1=low, 2=medium, 3=high landslide probability)
2. Old evaluation treated class 2 as "positive" and classes 1+3 as "negative" ❌
3. This is completely wrong for an ordinal problem!

## 🔧 Changes Made

### 1. Inference Now Outputs:
- **`test_susceptibility.tif`** - Ordinal score (0→1, weighted by risk)
- **`test_susceptibility_high.tif`** - Probability of high risk class
- **`test_class_probabilities.tif`** - All 3 class probabilities (3 bands)
- Plus uncertainty, class map, valid mask

### 2. Evaluation Now Provides:
- **Strategy 1**: High (class 3) vs Rest (classes 1-2)
- **Strategy 2**: At-Risk (classes 2-3) vs Low (class 1)
- **Strategy 3**: Ordinal correlation (Spearman's ρ)
- Separate metrics, plots, and optimal thresholds for each

## 🚀 How to Run

### Step 1: Activate Virtual Environment
```bash
# CRITICAL: Always activate .venv first!
source .venv/bin/activate  # Linux/macOS
# OR on Windows: .venv\Scripts\activate
```

### Step 2: Install Dependencies (if needed)
```bash
# With venv activated:
pip install -r requirements.txt
```

### Step 3: Re-run Inference
```bash
# With venv activated:
python -m src.main_pipeline
```

This will generate new outputs with the fixed inference code.

### Step 4: Evaluate with Fixed Script
```bash
# With venv activated:
python -m src.evaluate \
  --susceptibility outputs/test_susceptibility.tif \
  --ground_truth /home/kaue/data/landslide/training/Ground_truth_train.tif \
  --output_dir outputs/evaluation
```

### Step 4: Check Results
```bash
# View metrics
cat outputs/evaluation/evaluation_metrics.json

# View report
cat outputs/evaluation/evaluation_report.md

# View ROC curves
ls outputs/evaluation/figures/
```

## 📊 What to Expect

### Good Performance Indicators:
- **Spearman's ρ > 0.70**: Strong ordinal relationship ✅
- **AUROC > 0.80** (both strategies): Good discrimination ✅
- **AUPRC >> random baseline**: Model is useful ✅

### If Performance is Still Low:
1. Check class balance in `ground_truth_distribution`
2. Consider using ordinal regression loss (not cross-entropy)
3. Add class-distance weighting to penalize 1↔3 errors more than 1↔2
4. Verify training converged properly

## 📁 Modified Files

- `src/inference.py` - Compute ordinal score, save all probabilities
- `src/evaluate.py` - Multi-strategy evaluation
- `ORDINAL_3CLASS_FIX.md` - Detailed documentation

## ⚠️ Important Notes

1. **Virtual environment required** - Always activate `.venv` before running any commands!
2. **Old outputs are invalid** - Re-run inference to get corrected outputs
3. **Ground truth encoding**: 1=Low, 2=Medium, 3=High (remapped to 0,1,2 internally)
4. **Primary output**: `test_susceptibility.tif` is now the ordinal weighted score
5. **Use multiple strategies**: Different use cases need different metrics

## 🆘 Troubleshooting

**Import errors / Module not found?**
```bash
# First, activate virtual environment!
source .venv/bin/activate

# Then install dependencies
pip install -r requirements.txt
```

**Still low metrics after re-running?**
- Check training metrics in `artifacts/experiments/training_metrics.json`
- Model might need retraining with ordinal-aware loss
- Verify ground truth alignment with features

---

**Next**: Run the 3 steps above and share the new evaluation results!
