# Evaluation Pipeline Implementation Roadmap

## Implementation Status Update (October 14, 2025)

**🎉 Phase 1 COMPLETED!** All critical Week 1 features have been successfully implemented.

### Completed Milestones ✅
- ✅ **Milestone 1.1**: Test Set Evaluation (2-3 hours) - DONE
- ✅ **Milestone 1.2**: Threshold Selection & Class Maps (3-4 hours) - DONE
- ✅ **Milestone 2.1**: Performance Curve Visualizations (4-5 hours) - DONE

### Implementation Summary
- **7 tasks completed** across 3 milestones
- **New modules created**: `src/metrics.py`, `src/visualize.py`
- **Files modified**: `src/train.py`, `src/inference.py`
- **Total implementation time**: ~9-12 hours (completed in 1 session)
- **Documentation**: `IMPLEMENTATION_SUMMARY.md`, `QUICKSTART.md`

### Key Features Added
1. ✅ Test set evaluation with comprehensive metrics
2. ✅ Optimal threshold selection (Youden's J and F1-maximizing)
3. ✅ Four visualization types (ROC, PR, calibration, training history)
4. ✅ Enhanced model card with test metrics and threshold recommendations
5. ✅ Data-driven threshold application during inference
6. ✅ Automated plot generation and storage

### Next Steps
Continue with **Phase 2** (Milestone 2.2+) and **Phase 3** for:
- Confusion matrix visualizations
- Standalone evaluation script
- Feature importance analysis
- Cross-validation framework
- Ablation studies

---

## Overview
This roadmap outlines the implementation plan for completing the evaluation pipeline as specified in `descriptive_script.md` Section 13 (Evaluation & Reporting) and Section 14 (Uncertainty & Explainability).

**Current Status**: � Phase 1 Complete - Core evaluation features implemented  
**Target Status**: ✅ Fully Implemented (remaining: confusion matrices, standalone tools, cross-validation, ablations)

---

## Phase 1: Critical Foundation (Week 1) 🔴 HIGH PRIORITY

### Milestone 1.1: Test Set Evaluation ✅ COMPLETED
**Goal**: Ensure held-out test set is evaluated and reported  
**Effort**: 2-3 hours  
**Status**: ✅ **COMPLETED** (Oct 14, 2025)

#### Tasks:
- [x] **Task 1.1.1**: Modify `src/train.py` to create test dataset
  - ✅ Added test dataset loader after validation dataset
  - ✅ Created test DataLoader with same settings as validation
  - **Files**: `src/train.py` (lines ~295, ~318)

- [x] **Task 1.1.2**: Add test evaluation after training loop
  - ✅ Calls `evaluate()` on test set after training completes
  - ✅ Stores test metrics in training report
  - **Files**: `src/train.py` (lines ~502-520)

- [x] **Task 1.1.3**: Update model card with test metrics
  - ✅ Modified `write_model_card()` to include test metrics section
  - ✅ Distinguishes between validation and test metrics in output
  - **Files**: `src/inference.py` (function `write_model_card()`)

- [x] **Task 1.1.4**: Add test metrics to training report JSON
  - ✅ Updated `training_metrics.json` schema to include test results
  - **Files**: `src/train.py` (JSON export with `"test_metrics"` key)

#### Acceptance Criteria:
- ✅ Test dataset is loaded from `artifacts/tiles/test/`
- ✅ Test metrics (AUROC, AUPRC, IoU, F1, Accuracy) are computed
- ✅ Test metrics appear in `training_metrics.json`
- ✅ Model card shows both validation and test metrics
- ✅ Test evaluation runs automatically after training

#### Dependencies: None

**Implementation Notes**: Test evaluation successfully closes critical gap where test tiles were created but never evaluated. Console output shows test metrics during training, and model card now includes dedicated test metrics section.

---

### Milestone 1.2: Threshold Selection & Class Maps ✅ COMPLETED
**Goal**: Implement optimal threshold selection and generate discrete class maps  
**Effort**: 3-4 hours  
**Status**: ✅ **COMPLETED** (Oct 14, 2025)

#### Tasks:
- [x] **Task 1.2.1**: Implement threshold optimization functions
  - ✅ Created `src/metrics.py` with comprehensive threshold functions
  - ✅ Implemented Youden's J (sensitivity + specificity - 1)
  - ✅ Implemented F1-maximizing threshold
  - ✅ Added `compute_threshold_metrics()` for evaluation at specific thresholds
  - ✅ Added `select_optimal_thresholds()` orchestrator function
  - **Files**: New `src/metrics.py` (276 lines)

- [x] **Task 1.2.2**: Apply threshold selection on validation/test sets
  - ✅ Computes optimal thresholds on validation probabilities
  - ✅ Reports metrics at optimal threshold on both val and test
  - ✅ Stores threshold values in training artifacts
  - **Files**: `src/train.py` (lines ~524-528, threshold selection after test eval)

- [x] **Task 1.2.3**: Generate discrete class maps in inference
  - ✅ Applies optimal threshold to susceptibility probabilities
  - ✅ Generates binary/multi-class maps using data-driven thresholds
  - ✅ Exports as GeoTIFF with nodata=255
  - **Files**: `src/inference.py` (lines ~368-394, threshold loading and application)

- [x] **Task 1.2.4**: Update config for threshold strategy
  - ✅ Threshold strategy automatically determined (validation preferred, test fallback)
  - ✅ Manual threshold override possible via editing `training_metrics.json`
  - **Files**: Automatic selection in `src/metrics.py`

#### Acceptance Criteria:
- ✅ Optimal thresholds computed using multiple methods (Youden & F1)
- ✅ Threshold values saved in `training_metrics.json` under `"thresholds"` key
- ✅ Discrete class maps exported as `<area>_class_map.tif`
- ✅ Threshold selection method recorded (`"recommendation_method"`)
- ✅ Metrics reported at optimal threshold for both validation and test

#### Dependencies: Milestone 1.1 (test evaluation needed for threshold validation) ✅

**Implementation Notes**: Threshold selection uses F1-optimal by default (better for imbalanced data). Both Youden and F1 methods are computed and stored. Inference automatically loads and applies recommended threshold. Console output shows which threshold and method are being used.

---

## Phase 2: Visualization & Reporting (Week 2) 🟠 MEDIUM-HIGH PRIORITY

### Milestone 2.1: Performance Curve Visualizations ✅ COMPLETED
**Goal**: Generate ROC, PR, and calibration curves  
**Effort**: 4-5 hours  
**Status**: ✅ **COMPLETED** (Oct 14, 2025)

#### Tasks:
- [x] **Task 2.1.1**: Create visualization module
  - ✅ Created new file: `src/visualize.py` (425 lines)
  - ✅ Set up matplotlib with Agg backend (server-compatible)
  - ✅ Created output directory: `artifacts/experiments/figures/`
  - **Files**: New `src/visualize.py`

- [x] **Task 2.1.2**: Implement ROC curve plotting
  - ✅ Function: `plot_roc_curve(val_probs, val_labels, test_probs, test_labels, save_path, title)`
  - ✅ Plots for validation and test sets on same figure
  - ✅ Displays AUC values in legend
  - ✅ Saves as high-res PNG (150 DPI)
  - **Files**: `src/visualize.py` (lines 11-72)

- [x] **Task 2.1.3**: Implement PR curve plotting
  - ✅ Function: `plot_pr_curve(val_probs, val_labels, test_probs, test_labels, save_path, title)`
  - ✅ Plots for positive class (landslide) for both sets
  - ✅ Displays AUPRC/AP values on plot
  - ✅ Shows baseline (class prevalence) for context
  - **Files**: `src/visualize.py` (lines 75-151)

- [x] **Task 2.1.4**: Implement calibration plot
  - ✅ Function: `plot_calibration_curve(val_probs, val_labels, test_probs, test_labels, save_path, n_bins, title)`
  - ✅ Two-panel: reliability diagram + probability distribution
  - ✅ Shows perfect calibration line
  - ✅ Handles edge cases gracefully
  - **Files**: `src/visualize.py` (lines 154-256)

- [x] **Task 2.1.5**: Integrate visualization into training
  - ✅ Calls `generate_all_plots()` after test evaluation and threshold selection
  - ✅ Saves plots to `artifacts/experiments/figures/`
  - ✅ Stores plot paths in `training_metrics.json` under `"plots"` key
  - **Files**: `src/train.py` (lines ~528-531)

- [x] **BONUS**: Training history visualization
  - ✅ Function: `plot_training_history()` - 4-panel plot of loss and metrics over epochs
  - ✅ Helps diagnose overfitting and convergence
  - **Files**: `src/visualize.py` (lines 259-316)

#### Acceptance Criteria:
- ✅ ROC curves generated for validation and test sets
- ✅ PR curves generated for positive class (both sets)
- ✅ Calibration plots show predicted vs observed frequencies
- ✅ All plots saved as PNG (150 DPI - suitable for reports)
- ✅ Plot paths stored in training metrics JSON
- ✅ Plots are publication-quality with clear labels and legends
- ✅ Training history plot included as bonus feature

#### Dependencies: Milestone 1.1 (need test predictions) ✅

**Implementation Notes**: All four plot types automatically generated during training. Uses matplotlib Agg backend for server compatibility. Plots saved to `artifacts/experiments/figures/` with paths recorded in training_metrics.json for easy reference.

---

### Milestone 2.2: Confusion Matrix & Error Analysis
**Goal**: Generate confusion matrices and spatial error maps  
**Effort**: 3-4 hours  
**Status**: ❌ Not Started

#### Tasks:
- [ ] **Task 2.2.1**: Implement confusion matrix visualization
  - Function: `plot_confusion_matrix(confusion, class_names, output_path)`
  - Normalized and absolute counts
  - Heatmap with annotations
  - **Files**: `src/visualize.py`

- [ ] **Task 2.2.2**: Per-class performance breakdown
  - Function: `plot_per_class_metrics(metrics_dict, class_names, output_path)`
  - Bar charts for IoU, F1, Precision, Recall per class
  - **Files**: `src/visualize.py`

- [ ] **Task 2.2.3**: Generate spatial error maps
  - During inference, track prediction errors spatially
  - Export error map as GeoTIFF (correct/FP/FN/TN)
  - Visualize error distribution
  - **Files**: `src/inference.py`, `src/visualize.py`

- [ ] **Task 2.2.4**: Integrate into evaluation pipeline
  - Generate confusion matrices for val and test
  - Create per-class performance charts
  - Export error maps during inference
  - **Files**: `src/train.py`, `src/inference.py`

#### Acceptance Criteria:
- ✅ Confusion matrix heatmaps generated
- ✅ Per-class performance visualizations created
- ✅ Spatial error maps exported as GeoTIFF
- ✅ Error analysis included in model card

#### Dependencies: Milestone 1.1 (need test predictions)

---

### Milestone 2.3: Enhanced Model Card & Report
**Goal**: Create comprehensive evaluation report  
**Effort**: 2-3 hours  
**Status**: ❌ Not Started

#### Tasks:
- [ ] **Task 2.3.1**: Expand model card content
  - Add sections: Data Split Details, Threshold Selection, Limitations
  - Include figure references with captions
  - Add performance summary table
  - **Files**: `src/inference.py`

- [ ] **Task 2.3.2**: Generate PDF report (optional)
  - Use ReportLab or Markdown → PDF conversion
  - Include all figures and tables
  - Professional formatting
  - **Files**: New `src/report.py` (optional)

- [ ] **Task 2.3.3**: Add data provenance tracking
  - Record input file checksums
  - Track config hash
  - Version all artifacts
  - **Files**: `src/main_pipeline.py`, `src/inference.py`

#### Acceptance Criteria:
- ✅ Model card includes all specified sections
- ✅ Figures embedded or referenced with paths
- ✅ Performance summary table present
- ✅ Data provenance tracked
- ✅ (Optional) PDF report generated

#### Dependencies: Milestones 1.1, 2.1, 2.2

---

## Phase 3: Standalone Evaluation Tools (Week 3) 🟡 MEDIUM PRIORITY

### Milestone 3.1: Standalone Evaluation Script
**Goal**: Allow re-evaluation without retraining  
**Effort**: 4-5 hours  
**Status**: ❌ Not Started

#### Tasks:
- [ ] **Task 3.1.1**: Create evaluation script
  - New file: `src/evaluate.py`
  - Load model checkpoint
  - Load specified split (val/test)
  - Compute all metrics
  - Generate all visualizations
  - **Files**: New `src/evaluate.py`

- [ ] **Task 3.1.2**: Add command-line interface
  - Arguments: `--model_path`, `--split`, `--output_dir`
  - Support for custom test data
  - Verbose/quiet modes
  - **Files**: `src/evaluate.py`

- [ ] **Task 3.1.3**: Integrate with main pipeline
  - Add `--evaluate_only` flag to `main_pipeline.py`
  - Skip training if only evaluating
  - **Files**: `src/main_pipeline.py`

- [ ] **Task 3.1.4**: Document evaluation workflow
  - Update README with evaluation examples
  - Add to AGENTS.md
  - **Files**: `README.md`, `AGENTS.md`

#### Acceptance Criteria:
- ✅ Can run: `python -m src.evaluate --model_path <path> --split test`
- ✅ All metrics and visualizations generated
- ✅ Works with arbitrary checkpoints
- ✅ Documented in README

#### Dependencies: Milestones 1.1, 2.1, 2.2

---

### Milestone 3.2: Inference with Ground Truth Comparison
**Goal**: Evaluate model predictions against ground truth during inference  
**Effort**: 3-4 hours  
**Status**: ❌ Not Started

#### Tasks:
- [ ] **Task 3.2.1**: Modify inference to optionally load ground truth
  - Check if ground truth exists for inference area
  - Load ground truth alongside predictions
  - **Files**: `src/inference.py`

- [ ] **Task 3.2.2**: Compute inference-time metrics
  - Compare predictions to ground truth pixel-wise
  - Compute AUROC, AUPRC, IoU on full inference area
  - Generate confusion matrix
  - **Files**: `src/inference.py`

- [ ] **Task 3.2.3**: Export comparison visualizations
  - Side-by-side: prediction vs ground truth
  - Difference map (errors highlighted)
  - Save to `outputs/figures/`
  - **Files**: `src/inference.py`, `src/visualize.py`

- [ ] **Task 3.2.4**: Add to model card
  - Include inference-area metrics if available
  - Distinguish from train/val/test splits
  - **Files**: `src/inference.py`

#### Acceptance Criteria:
- ✅ Inference can optionally compare to ground truth
- ✅ Inference-area metrics computed if ground truth available
- ✅ Comparison visualizations exported
- ✅ Metrics included in model card

#### Dependencies: Milestones 2.1, 2.2

---

## Phase 4: Spatial Cross-Validation (Week 4-5) 🟢 LOWER PRIORITY

### Milestone 4.1: Cross-Validation Framework
**Goal**: Implement K-fold spatial cross-validation  
**Effort**: 8-10 hours  
**Status**: ❌ Not Started

#### Tasks:
- [ ] **Task 4.1.1**: Design cross-validation scheme
  - Define K spatial folds (e.g., K=5)
  - Ensure spatial blocking per fold
  - Store fold assignments
  - **Files**: New `src/cross_validate.py`

- [ ] **Task 4.1.2**: Implement cross-validation loop
  - For each fold: train model, evaluate on fold's test set
  - Track metrics per fold
  - Save fold checkpoints
  - **Files**: `src/cross_validate.py`

- [ ] **Task 4.1.3**: Aggregate fold metrics
  - Compute mean ± std for all metrics
  - Statistical significance tests (optional)
  - Best/worst fold analysis
  - **Files**: `src/cross_validate.py`

- [ ] **Task 4.1.4**: Generate cross-validation report
  - Summary table: mean ± std for AUROC, AUPRC, IoU, F1
  - Per-fold performance plots
  - Export to `outputs/cross_validation/`
  - **Files**: `src/cross_validate.py`, `src/visualize.py`

- [ ] **Task 4.1.5**: Add to main pipeline
  - Add `--cross_validate` flag
  - Integrate with existing pipeline
  - **Files**: `src/main_pipeline.py`

#### Acceptance Criteria:
- ✅ K-fold spatial cross-validation runs successfully
- ✅ Per-fold metrics saved
- ✅ Aggregate statistics computed (mean ± std)
- ✅ Cross-validation report generated
- ✅ Can run: `python -m src.main_pipeline --cross_validate`

#### Dependencies: Phase 1 complete (need robust evaluation)

---

## Phase 5: Ablation Studies (Week 6-7) 🟢 LOWER PRIORITY

### Milestone 5.1: Feature Ablation Framework
**Goal**: Quantify contribution of feature groups (RGB, Terrain, Land Cover)  
**Effort**: 6-8 hours  
**Status**: ❌ Not Started

#### Tasks:
- [ ] **Task 5.1.1**: Define feature groups
  - RGB channels
  - DTM-derived (slope, aspect, curvatures, flow, TWI, etc.)
  - Land cover (K-means clusters)
  - **Files**: `config.yaml`, `src/main_pipeline.py`

- [ ] **Task 5.1.2**: Modify config for ablation
  - Add feature group toggles to config
  - Ensure preprocessing respects toggles
  - **Files**: `config.yaml`, `src/main_pipeline.py`

- [ ] **Task 5.1.3**: Implement ablation script
  - New file: `src/ablation.py`
  - Run pipeline multiple times with different feature sets
  - Store results per ablation
  - **Files**: New `src/ablation.py`

- [ ] **Task 5.1.4**: Compare ablation results
  - Create comparison table: features vs metrics
  - Visualize feature importance
  - Statistical tests for significance
  - **Files**: `src/ablation.py`, `src/visualize.py`

- [ ] **Task 5.1.5**: Generate ablation report
  - Summary: which features matter most
  - Recommendation: minimal feature set for target performance
  - Export to `outputs/ablation/`
  - **Files**: `src/ablation.py`

#### Acceptance Criteria:
- ✅ Can train models with feature subsets
- ✅ Ablation results compared systematically
- ✅ Feature importance quantified
- ✅ Ablation report generated
- ✅ Can run: `python -m src.ablation`

#### Dependencies: Phase 1 complete

---

## Phase 6: Explainability & Attribution (Week 8-9) 🔵 FUTURE WORK

### Milestone 6.1: Grad-CAM Attribution
**Goal**: Generate attribution maps showing what model focuses on  
**Effort**: 8-10 hours  
**Status**: ❌ Not Started

#### Tasks:
- [ ] **Task 6.1.1**: Implement Grad-CAM
  - Adapt Grad-CAM for segmentation (Grad-CAM++)
  - Target layer: decoder bottleneck
  - **Files**: New `src/explainability.py`

- [ ] **Task 6.1.2**: Generate attribution maps
  - For sample predictions, compute attributions
  - Overlay on input features
  - Highlight most influential channels
  - **Files**: `src/explainability.py`

- [ ] **Task 6.1.3**: Visualize attributions
  - Heatmap overlays on RGB orthophoto
  - Channel-wise attribution scores
  - Export as GeoTIFF and PNG
  - **Files**: `src/explainability.py`, `src/visualize.py`

- [ ] **Task 6.1.4**: Integrate into inference
  - Optional attribution generation during inference
  - Save attribution maps to `outputs/attributions/`
  - **Files**: `src/inference.py`

#### Acceptance Criteria:
- ✅ Grad-CAM attribution maps generated
- ✅ Attributions overlaid on inputs
- ✅ Channel importance quantified
- ✅ Attribution maps exported

#### Dependencies: Phase 1 complete

---

### Milestone 6.2: Integrated Gradients (Optional)
**Goal**: Alternative attribution method for comparison  
**Effort**: 6-8 hours  
**Status**: ❌ Not Started

#### Tasks:
- [ ] **Task 6.2.1**: Implement Integrated Gradients
  - Path integration from baseline to input
  - Efficient computation
  - **Files**: `src/explainability.py`

- [ ] **Task 6.2.2**: Compare attribution methods
  - Grad-CAM vs Integrated Gradients
  - Qualitative comparison
  - **Files**: `src/explainability.py`, `src/visualize.py`

#### Acceptance Criteria:
- ✅ Integrated Gradients implemented
- ✅ Comparison with Grad-CAM provided
- ✅ Method selection documented

#### Dependencies: Milestone 6.1

---

## Phase 7: Production Hardening (Week 10) 🔵 FUTURE WORK

### Milestone 7.1: Automated Testing
**Goal**: Ensure evaluation pipeline reliability  
**Effort**: 6-8 hours  
**Status**: ❌ Not Started

#### Tasks:
- [ ] **Task 7.1.1**: Create test data
  - Synthetic small rasters for testing
  - Known ground truth
  - **Files**: New `tests/fixtures/`

- [ ] **Task 7.1.2**: Write unit tests
  - Test evaluation functions
  - Test visualization functions
  - Test threshold selection
  - **Files**: New `tests/test_evaluation.py`

- [ ] **Task 7.1.3**: Write integration tests
  - End-to-end evaluation workflow
  - Cross-validation workflow
  - **Files**: New `tests/test_integration.py`

- [ ] **Task 7.1.4**: Set up CI/CD
  - GitHub Actions or similar
  - Run tests on push
  - **Files**: New `.github/workflows/test.yml`

#### Acceptance Criteria:
- ✅ Unit tests pass (>80% coverage for evaluation code)
- ✅ Integration tests pass
- ✅ CI/CD pipeline configured
- ✅ Tests documented in README

#### Dependencies: Phases 1-3 complete

---

### Milestone 7.2: Documentation & Examples
**Goal**: Complete documentation for evaluation features  
**Effort**: 4-5 hours  
**Status**: ❌ Not Started

#### Tasks:
- [ ] **Task 7.2.1**: Update README
  - Document all evaluation modes
  - Add usage examples
  - **Files**: `README.md`

- [ ] **Task 7.2.2**: Update AGENTS.md
  - Add evaluation workflow guidance
  - Troubleshooting for evaluation
  - **Files**: `AGENTS.md`

- [ ] **Task 7.2.3**: Create evaluation tutorial
  - Jupyter notebook or markdown guide
  - Step-by-step evaluation walkthrough
  - **Files**: New `docs/evaluation_tutorial.md`

- [ ] **Task 7.2.4**: Document config options
  - All evaluation-related config parameters
  - Examples for common scenarios
  - **Files**: `docs/config_reference.md` or in README

#### Acceptance Criteria:
- ✅ All evaluation features documented
- ✅ Usage examples provided
- ✅ Tutorial created
- ✅ Config options documented

#### Dependencies: Phases 1-6 complete

---

## Timeline Summary

```
Week 1:  Phase 1 - Critical Foundation (Test eval, Threshold selection)
Week 2:  Phase 2 - Visualization & Reporting (Curves, confusion matrices)
Week 3:  Phase 3 - Standalone Tools (Evaluation script, inference comparison)
Week 4-5: Phase 4 - Cross-Validation (K-fold spatial CV)
Week 6-7: Phase 5 - Ablation Studies (Feature importance)
Week 8-9: Phase 6 - Explainability (Grad-CAM, Integrated Gradients)
Week 10: Phase 7 - Production Hardening (Tests, Docs)
```

**Total Effort**: ~55-70 hours (7-9 weeks for one developer)

---

## Quick Start Checklist

### Minimum Viable Evaluation (Week 1)
- [ ] Test set evaluation (Milestone 1.1)
- [ ] Threshold selection (Milestone 1.2)
- [ ] Basic visualizations (ROC, PR curves from Milestone 2.1)

### Production-Ready Evaluation (Weeks 1-3)
- [ ] Everything in Minimum Viable
- [ ] Confusion matrices (Milestone 2.2)
- [ ] Enhanced model card (Milestone 2.3)
- [ ] Standalone evaluation script (Milestone 3.1)

### Research-Grade Evaluation (Weeks 1-9)
- [ ] Everything in Production-Ready
- [ ] Cross-validation (Milestone 4.1)
- [ ] Ablation studies (Milestone 5.1)
- [ ] Explainability (Milestone 6.1)

---

## Success Metrics

### Phase 1 Success:
- ✅ Test metrics reported in model card
- ✅ Test AUROC within 0.05 of validation AUROC (no overfitting)
- ✅ Optimal thresholds determined and applied

### Phase 2 Success:
- ✅ All visualizations generated automatically
- ✅ Model card includes figure references
- ✅ Calibration error < 0.1

### Phase 3 Success:
- ✅ Can evaluate any checkpoint without retraining
- ✅ Inference can optionally compute metrics

### Phase 4 Success:
- ✅ Cross-validated performance reported with confidence intervals
- ✅ Performance variance across folds < 0.1 (AUROC)

### Phase 5 Success:
- ✅ Feature importance quantified
- ✅ Minimal feature set identified

### Phase 6 Success:
- ✅ Attribution maps interpretable by domain experts
- ✅ Model decisions explainable

### Phase 7 Success:
- ✅ All tests pass
- ✅ Complete documentation
- ✅ CI/CD pipeline operational

---

## Risk Assessment

### High Risk:
- **Cross-validation computational cost**: May require distributed computing for large datasets
- **Explainability complexity**: Grad-CAM for segmentation is more complex than classification

### Medium Risk:
- **Visualization library compatibility**: Matplotlib/seaborn versions may conflict
- **Memory constraints**: Generating all visualizations may require substantial RAM

### Low Risk:
- **Test evaluation**: Straightforward extension of existing code
- **Threshold selection**: Well-established algorithms

---

## Notes

- **Incremental delivery**: Each milestone can be merged independently
- **Backward compatibility**: Existing pipeline continues to work throughout
- **Config-driven**: New features controlled via `config.yaml`
- **Documentation-first**: Update docs as features are implemented
- **Testing**: Add tests incrementally, not as final phase

---

## Appendix: File Structure After Completion

```
landslide_susc_modelling/
├── src/
│   ├── main_pipeline.py (enhanced with flags)
│   ├── train.py (test evaluation added)
│   ├── inference.py (enhanced model card)
│   ├── evaluate.py (NEW - standalone evaluation)
│   ├── cross_validate.py (NEW - K-fold CV)
│   ├── ablation.py (NEW - feature ablations)
│   ├── visualize.py (NEW - all plotting functions)
│   ├── explainability.py (NEW - Grad-CAM, IG)
│   └── metrics.py (NEW - threshold selection, etc.)
├── outputs/
│   ├── figures/ (NEW - ROC, PR, calibration, confusion)
│   ├── cross_validation/ (NEW - CV results)
│   ├── ablation/ (NEW - ablation results)
│   └── attributions/ (NEW - attribution maps)
├── tests/ (NEW)
│   ├── fixtures/
│   ├── test_evaluation.py
│   └── test_integration.py
├── docs/ (NEW)
│   ├── evaluation_tutorial.md
│   └── config_reference.md
├── EVALUATION_ANALYSIS.md (DONE)
├── EVALUATION_ROADMAP.md (THIS FILE)
└── config.yaml (enhanced with evaluation options)
```

---

**Last Updated**: October 14, 2025  
**Status**: Planning Phase  
**Next Action**: Begin Phase 1, Milestone 1.1 (Test Set Evaluation)
