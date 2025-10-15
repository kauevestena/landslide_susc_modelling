# Evaluation Pipeline Analysis

## Current State of Evaluation

### ✅ What EXISTS

#### 1. **Training-time Validation** (Fully Implemented)
**Location**: `src/train.py` - `evaluate()` function

The pipeline includes a comprehensive validation loop during training:

```python
def evaluate(model, loader, loss_fn, device, num_classes, positive_class, collect_probs=False)
```

**Metrics Computed**:
- ✅ Overall Accuracy
- ✅ Macro IoU (Intersection over Union)
- ✅ Macro F1-score
- ✅ AUROC (Area Under ROC Curve)
- ✅ AUPRC (Area Under Precision-Recall Curve)
- ✅ Per-class confusion matrix
- ✅ Loss tracking (Dice + Cross-Entropy blend)

**When it runs**: 
- Every epoch during training
- On validation split (if configured)
- Results saved to `artifacts/experiments/training_metrics.json`

**Outputs**:
```json
{
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.45,
      "val_loss": 0.52,
      "val_metrics": {
        "overall_accuracy": 0.87,
        "macro_iou": 0.65,
        "macro_f1": 0.72,
        "auroc": 0.89,
        "auprc": 0.78
      }
    },
    ...
  ],
  "best_epoch": 15,
  "best_metrics": {
    "overall_accuracy": 0.91,
    "macro_iou": 0.73,
    "macro_f1": 0.81,
    "auroc": 0.93,
    "auprc": 0.86
  }
}
```

#### 2. **Probability Calibration** (Implemented)
**Location**: `src/train.py` - end of `train_model()`

- Isotonic regression fitted on validation predictions
- Stored at `artifacts/experiments/isotonic_calibrator.joblib`
- Applied during inference to produce calibrated probabilities

#### 3. **Model Card Generation** (Implemented)
**Location**: `src/inference.py` - `generate_model_card()`

- Summarizes validation metrics
- Documents model architecture
- Lists output products
- Saved to `outputs/model_card.md`

### ⚠️ What is PARTIALLY Implemented

#### 1. **Test Set Tiles Created but Not Evaluated**
**Location**: `src/main_pipeline.py` - `prepare_dataset()`

**What exists**:
- Spatial block splitting creates train/val/**test** tiles
- Test tiles saved to `artifacts/tiles/test/`
- Test labels saved to `artifacts/labels/test/`
- Test split assignments in `artifacts/splits/splits.json`

**What's missing**:
- Test set is **never loaded or evaluated** in `train.py`
- No test metrics computed
- No final held-out performance reported
- Model card only shows **validation** metrics, not test metrics

**Current behavior**:
```python
# In train.py
train_dataset = LandslideDataset(tiles_dir, labels_dir, 'train', augment_cfg)
val_dataset = LandslideDataset(tiles_dir, labels_dir, 'val', None)
# ❌ test_dataset is NEVER created
```

### ❌ What is MISSING

#### 1. **Dedicated Test Set Evaluation**
**Status**: Not implemented

**Expected workflow** (per descriptive_script.md Section 13):
1. After training completes, load best model
2. Load test tiles (held-out spatial blocks)
3. Run evaluation on test set
4. Compute final metrics: AUROC, AUPRC, IoU, calibration curves
5. Generate confusion matrix
6. Save test metrics separately

**Where it should be**:
- Option A: Add test evaluation at end of `train_model()`
- Option B: Create separate `src/evaluate.py` script
- Option C: Add `--evaluate` flag to main pipeline

#### 2. **Spatial Cross-Validation**
**Status**: Not implemented

**Specification** (descriptive_script.md Section 13):
> "Spatial cross-validation"

**What's needed**:
- K-fold cross-validation with spatial block assignments
- Each fold should use different spatial blocks
- Average metrics across folds
- Report mean ± std for all metrics

#### 3. **Threshold Selection Analysis**
**Status**: Not implemented

**Specification** (descriptive_script.md Section 13):
> "Threshold selection: Youden's J, fixed precision, or quantiles"

**What's needed**:
- Analyze ROC curve to find optimal threshold
- Methods: Youden's J (sensitivity + specificity), fixed precision, quantile-based
- Apply threshold to generate discrete class maps
- Report metrics at selected threshold

#### 4. **Comprehensive Reporting**
**Status**: Partially implemented (model card exists but incomplete)

**Specification** (descriptive_script.md Section 13):
> "Deliverables: PDF summary with maps, ROC/PR curves, calibration, confusion matrices"

**Currently have**:
- ✅ Model card (text-based)
- ❌ ROC curves
- ❌ PR curves  
- ❌ Calibration plots
- ❌ Confusion matrix visualization
- ❌ Per-class performance breakdown
- ❌ Spatial distribution of errors

#### 5. **Ablation Studies**
**Status**: Not implemented

**Specification** (descriptive_script.md Section 13):
> "Ablation: RGB only vs Terrain only vs Fused"

**What's needed**:
- Train models with different feature subsets
- Compare performance across ablations
- Quantify contribution of each feature group

#### 6. **Explainability/Attribution**
**Status**: Not implemented

**Specification** (descriptive_script.md Section 14):
> "Attribution: Grad-CAM / Integrated Gradients overlays"

**What's needed**:
- Generate attribution maps for sample predictions
- Overlay on input features to show what model "looks at"
- Export as visualization

## Expected Evaluation Workflow

### As Specified in descriptive_script.md

```
Training Phase:
├── Validation metrics computed every epoch ✅ DONE
├── Best model selected by AUPRC ✅ DONE
├── Calibrator fitted on validation set ✅ DONE
└── Training metrics saved ✅ DONE

Evaluation Phase: ❌ MISSING
├── Load best model checkpoint
├── Load test tiles (held-out spatial blocks)
├── Run inference on test set
├── Compute test metrics (AUROC, AUPRC, IoU, F1)
├── Generate confusion matrix
├── Plot ROC and PR curves
├── Generate calibration plot
├── Threshold selection (Youden's J)
├── Save test results JSON
└── Generate comprehensive PDF report

Cross-Validation Phase: ❌ MISSING
├── Define K spatial folds
├── For each fold:
│   ├── Train model
│   ├── Evaluate on fold's test set
│   └── Store fold metrics
├── Aggregate metrics (mean ± std)
└── Report cross-validated performance

Ablation Phase: ❌ MISSING
├── Train with RGB-only features
├── Train with Terrain-only features
├── Train with Fused features
├── Compare metrics across ablations
└── Visualize feature importance
```

## Recommendations for Implementation

### Priority 1: Add Test Set Evaluation (Critical)

**Why**: Model card currently shows only validation metrics, which may be optimistically biased due to early stopping on validation AUPRC.

**Implementation**:
1. Modify `src/train.py` to add test evaluation after training:
   ```python
   # After training loop completes
   if test_dataset:
       test_loss, test_metrics, _ = evaluate(
           model, test_loader, loss_fn, device, num_classes, positive_class
       )
       training_report['test_metrics'] = test_metrics
   ```

2. Update model card to include test metrics

**Effort**: Low (1-2 hours)

### Priority 2: Generate Evaluation Visualizations (High)

**Why**: Visualizations are critical for understanding model behavior and communicating results.

**Implementation**:
1. Create `src/visualize.py` with functions:
   - `plot_roc_curve(y_true, y_scores, output_path)`
   - `plot_pr_curve(y_true, y_scores, output_path)`
   - `plot_calibration(y_true, y_scores, bins, output_path)`
   - `plot_confusion_matrix(confusion, class_names, output_path)`

2. Call visualization functions after test evaluation

3. Save plots to `outputs/figures/`

**Effort**: Medium (4-6 hours)

### Priority 3: Threshold Selection (Medium)

**Why**: Converting probabilities to discrete classes requires choosing a threshold.

**Implementation**:
1. Add `find_optimal_threshold()` function
2. Methods: Youden's J, F1-maximizing, fixed precision
3. Apply threshold and generate class map
4. Export as GeoTIFF

**Effort**: Low-Medium (2-3 hours)

### Priority 4: Create Standalone Evaluation Script (Medium)

**Why**: Allows re-evaluation without retraining.

**Implementation**:
```python
# src/evaluate.py
def evaluate_model(config, model_path, test_tiles_dir, test_labels_dir, output_dir):
    """Evaluate a trained model on test set and generate comprehensive report."""
    # Load model
    # Load test data
    # Compute metrics
    # Generate visualizations
    # Save report
```

Run with: `python -m src.evaluate`

**Effort**: Medium (3-4 hours)

### Priority 5: Spatial Cross-Validation (Lower Priority)

**Why**: Provides robust performance estimates, but computationally expensive.

**Implementation**:
1. Create `src/cross_validate.py`
2. Define K-fold spatial splits
3. Train K models
4. Aggregate metrics
5. Report mean ± std

**Effort**: High (8-12 hours)

### Priority 6: Ablation Studies (Lower Priority)

**Why**: Provides insights into feature importance but is research-oriented.

**Implementation**:
1. Modify config to specify feature subsets
2. Run pipeline multiple times with different subsets
3. Compare results
4. Create comparison table/plot

**Effort**: High (time-consuming due to multiple training runs)

## Quick Implementation Plan

### Phase 1: Complete Test Evaluation (Week 1)
- [ ] Add test set evaluation to `train.py`
- [ ] Include test metrics in model card
- [ ] Verify test tiles are being used correctly

### Phase 2: Add Visualizations (Week 2)
- [ ] Create `src/visualize.py`
- [ ] Generate ROC, PR, calibration curves
- [ ] Generate confusion matrix heatmap
- [ ] Update model card with figure references

### Phase 3: Threshold Selection (Week 2)
- [ ] Implement threshold optimization
- [ ] Generate discrete class maps at optimal threshold
- [ ] Add class map to outputs

### Phase 4: Standalone Evaluation (Week 3)
- [ ] Create `src/evaluate.py` script
- [ ] Add `--evaluate` mode to pipeline
- [ ] Document evaluation-only workflow

### Phase 5: Advanced Features (Future)
- [ ] Cross-validation framework
- [ ] Ablation study tools
- [ ] Explainability/attribution

## Summary

**Evaluation Status: 🟡 PARTIALLY IMPLEMENTED**

### Current Strengths ✅
- Validation metrics computed during training
- Comprehensive metrics (AUROC, AUPRC, IoU, F1)
- Probability calibration
- Basic model card

### Critical Gaps ❌
- Test set never evaluated (tiles exist but unused)
- No test metrics in final report
- No visualization of performance curves
- No threshold selection
- No spatial cross-validation
- No ablation studies

### Immediate Action Required
**Priority 1**: Add test set evaluation to prevent reporting overly optimistic validation metrics as final results. This is a critical gap that should be addressed before any production deployment.
