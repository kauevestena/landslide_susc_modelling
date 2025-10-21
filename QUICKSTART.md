# Quick Start Guide - Enhanced Evaluation Pipeline

## Prerequisites
Ensure you have a working Python environment with all dependencies installed:
```bash
pip install -r requirements.txt
```

## Running the Enhanced Pipeline

### Full Pipeline (Recommended for First Run)
```bash
# Force recreation of all artifacts to get new evaluation features
python -m src.main_pipeline --force_recreate
```

### Resume from Checkpoint
```bash
# Resume from last checkpoint (skips already-completed stages)
python -m src.main_pipeline
```

## What's New - Expected Outputs

### 1. Test Set Evaluation
**Location**: `artifacts/experiments/training_metrics.json`

```bash
# View test metrics
cat artifacts/experiments/training_metrics.json | jq '.test_metrics'
```

Expected output:
```json
{
  "overall_accuracy": 0.8234,
  "macro_iou": 0.6543,
  "macro_f1": 0.7123,
  "auroc": 0.8876,
  "auprc": 0.7234
}
```

### 2. Optimal Thresholds
**Location**: `artifacts/experiments/training_metrics.json`

```bash
# View recommended threshold
cat artifacts/experiments/training_metrics.json | jq '.thresholds.recommended_threshold'
```

Expected output:
```
0.3456
```

### 3. Performance Visualizations
**Location**: `artifacts/experiments/figures/`

```bash
# List generated plots
ls -lh artifacts/experiments/figures/
```

Expected files:
- `roc_curve.png` - ROC curves for validation and test sets
- `pr_curve.png` - Precision-Recall curves
- `calibration_curve.png` - Calibration analysis
- `training_history.png` - Training metrics over epochs

**View plots**:
```bash
# On Linux with GUI
xdg-open artifacts/experiments/figures/roc_curve.png

# On macOS
open artifacts/experiments/figures/roc_curve.png

# Copy to local machine if on remote server
scp user@server:~/landslide_susc_modelling/artifacts/experiments/figures/*.png ./
```

### 4. Enhanced Model Card
**Location**: `outputs/model_card.md`

```bash
# View model card
cat outputs/model_card.md
```

New sections included:
- **Test Metrics** - Performance on held-out test set
- **Classification Thresholds** - Optimal operating points

## Verifying Implementation

### Check Console Output
During training, you should see:
```
[train] Evaluating on test set...
[train] Test set metrics: {'overall_accuracy': 0.82, ...}
[train] Selecting optimal classification thresholds...
[train] Recommended threshold: 0.3456 (method: f1_validation)
[train] Generating performance visualizations...
[visualize] Saved ROC curve to artifacts/experiments/figures/roc_curve.png
[visualize] Saved PR curve to artifacts/experiments/figures/pr_curve.png
[visualize] Saved calibration curve to artifacts/experiments/figures/calibration_curve.png
[visualize] Saved training history to artifacts/experiments/figures/training_history.png
```

During inference:
```
[inference] Using optimal threshold 0.3456 (method: f1_validation)
```

### Validate All Outputs Exist
```bash
#!/bin/bash
# Comprehensive validation script

echo "=== Checking Training Artifacts ==="
test -f artifacts/experiments/best_model.pth && echo "✓ Model checkpoint exists" || echo "✗ Model missing"
test -f artifacts/experiments/training_metrics.json && echo "✓ Metrics JSON exists" || echo "✗ Metrics missing"
test -f artifacts/experiments/isotonic_calibrator.joblib && echo "✓ Calibrator exists" || echo "✗ Calibrator missing"

echo -e "\n=== Checking Visualizations ==="
test -f artifacts/experiments/figures/roc_curve.png && echo "✓ ROC curve exists" || echo "✗ ROC curve missing"
test -f artifacts/experiments/figures/pr_curve.png && echo "✓ PR curve exists" || echo "✗ PR curve missing"
test -f artifacts/experiments/figures/calibration_curve.png && echo "✓ Calibration exists" || echo "✗ Calibration missing"
test -f artifacts/experiments/figures/training_history.png && echo "✓ History exists" || echo "✗ History missing"

echo -e "\n=== Checking Test Metrics ==="
test -n "$(cat artifacts/experiments/training_metrics.json | jq -r '.test_metrics')" && \
  echo "✓ Test metrics present in JSON" || echo "✗ Test metrics missing"

echo -e "\n=== Checking Thresholds ==="
THRESHOLD=$(cat artifacts/experiments/training_metrics.json | jq -r '.thresholds.recommended_threshold')
if [ "$THRESHOLD" != "null" ]; then
  echo "✓ Recommended threshold: $THRESHOLD"
else
  echo "✗ Threshold not found"
fi

echo -e "\n=== Checking Inference Outputs ==="
test -f outputs/model_card.md && echo "✓ Model card exists" || echo "✗ Model card missing"
test -f outputs/train_susceptibility.tif && echo "✓ Susceptibility map exists" || echo "✗ Susceptibility missing"
test -f outputs/train_uncertainty.tif && echo "✓ Uncertainty map exists" || echo "✗ Uncertainty missing"
test -f outputs/train_class_map.tif && echo "✓ Class map exists" || echo "✗ Class map missing"

echo -e "\n=== Model Card Validation ==="
grep -q "## Test Metrics" outputs/model_card.md && \
  echo "✓ Test metrics section in model card" || echo "✗ Test metrics section missing"
grep -q "## Classification Thresholds" outputs/model_card.md && \
  echo "✓ Thresholds section in model card" || echo "✗ Thresholds section missing"

echo -e "\n=== Done ==="
```

Save as `validate_outputs.sh` and run:
```bash
chmod +x validate_outputs.sh
./validate_outputs.sh
```

## Interpreting Results

### ROC Curve (`roc_curve.png`)
- **AUC close to 1.0**: Excellent discrimination
- **AUC around 0.5**: Random classifier
- **Test < Validation**: Possible overfitting
- **Test ≈ Validation**: Good generalization

### PR Curve (`pr_curve.png`)
- **AP (Average Precision)**: Higher is better
- **Curve above baseline**: Model is useful
- **Curve at baseline**: Model no better than predicting majority class
- **Important for imbalanced data**: More informative than ROC when positive class is rare

### Calibration Curve (`calibration_curve.png`)
- **Close to diagonal**: Well-calibrated probabilities
- **Above diagonal**: Overconfident (predicted probs too high)
- **Below diagonal**: Underconfident (predicted probs too low)
- **Distribution plot**: Shows prediction confidence distribution

### Training History (`training_history.png`)
- **Loss decreasing**: Model learning
- **Val loss increasing after epoch X**: Overfitting after epoch X
- **Metrics plateauing**: Model converged
- **Erratic curves**: May need lower learning rate or more data

## Troubleshooting

### Issue: No test tiles found
```
ValueError: No tiles found for split "test" in artifacts/tiles/test
```

**Solution**: Test split may not have been created. Check:
```bash
ls artifacts/tiles/test/ | wc -l
```

If zero, check `config.yaml` for `test_size` parameter:
```yaml
dataset:
  test_size: 0.2  # Should be > 0
```

### Issue: Plots not generated
```
[visualize] Could not compute validation calibration: ...
```

**Solution**: Edge case handling. Check:
- At least 2 classes present in dataset
- Sufficient samples in validation/test sets
- No NaN values in probabilities

### Issue: Threshold is 0.5 (default)
```
[inference] Using optimal threshold 0.5000 (method: default)
```

**Solution**: Threshold selection failed. Check:
- `training_metrics.json` has `thresholds` key
- Validation or test set has both positive and negative samples
- No errors during threshold selection step

### Issue: Model card missing new sections
**Solution**: Model card uses cached metrics. Force regeneration:
```bash
rm outputs/model_card.md
python -m src.main_pipeline  # Will regenerate during inference
```

## Performance Tips

### Speed Up Training (Development)
Reduce dataset size in `config.yaml`:
```yaml
dataset:
  max_tiles_per_split: 100  # Limit tiles for faster iteration
  
training:
  epochs: 5  # Fewer epochs for testing
```

### High-Quality Production Run
```yaml
dataset:
  max_tiles_per_split: null  # Use all tiles
  
training:
  epochs: 50
  early_stopping_patience: 15
  
inference:
  mc_dropout_iterations: 10  # Uncertainty quantification
  tta: true  # Test-time augmentation
```

## Next Steps

After reviewing evaluation results:

1. **If test performance is poor**:
   - Check calibration curve for confidence issues
   - Review training history for overfitting
   - Consider data augmentation or regularization

2. **If threshold seems wrong**:
   - Examine F1 vs Youden trade-offs in `training_metrics.json`
   - Consider domain requirements (prefer precision vs recall?)
   - May need to manually set threshold based on cost matrix

3. **For production deployment**:
   - Document threshold selection rationale
   - Set up monitoring for prediction distribution drift
   - Consider A/B testing different thresholds

## Documentation

- **Full Implementation**: See `IMPLEMENTATION_SUMMARY.md`
- **Evaluation Roadmap**: See `EVALUATION_ROADMAP.md` for future features
- **Quick Reference**: See `EVALUATION_QUICK_REF.md` for code snippets
- **External LULC Integration**: See `EXTERNAL_LULC_IMPLEMENTATION.md` for using validated land cover datasets

## External LULC Quick Start

### Option 1: ESA WorldCover (Recommended - No Auth Required)

1. **Enable in config.yaml**:
```yaml
preprocessing:
  external_lulc:
    enabled: true
    source: worldcover
    worldcover:
      year: 2021
```

2. **Install dependencies**:
```bash
pip install requests
```

3. **Run pipeline**:
```bash
python -m src.main_pipeline --force_recreate
```

WorldCover will be automatically downloaded and cached in `artifacts/derived/*/lulc_cache/`.

### Option 2: Google Dynamic World (Requires Earth Engine)

1. **Setup Earth Engine**:
```bash
pip install earthengine-api
earthengine authenticate
```

2. **Enable in config.yaml**:
```yaml
preprocessing:
  external_lulc:
    enabled: true
    source: dynamic_world
    dynamic_world:
      start_date: "2023-01-01"
      end_date: "2023-06-30"
```

3. **Run pipeline**:
```bash
python -m src.main_pipeline --force_recreate
```

### Verify External LULC Integration

```bash
# Check LULC source in metadata
cat artifacts/derived/train/feature_metadata.json | grep -A 5 lulc_source

# Verify LULC classes
cat artifacts/derived/train/feature_metadata.json | jq '.lulc_class_info'

# View cached LULC data
ls -lh artifacts/derived/train/lulc_cache/
```

### Disable External LULC (Use K-means)

```yaml
preprocessing:
  external_lulc:
    enabled: false
```

This reverts to the original K-means clustering approach.

## Support

If you encounter issues:
1. Check `IMPLEMENTATION_SUMMARY.md` for troubleshooting
2. Validate environment: `pip list | grep -E 'numpy|scikit-learn|matplotlib'`
3. Review console output for specific error messages
4. Ensure all dependencies are installed: `pip install -r requirements.txt`
