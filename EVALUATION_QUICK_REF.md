# Evaluation Implementation - Quick Reference

## 🎯 Priority Matrix

### 🔴 **Critical (Week 1)** - DO FIRST
| Task | File | Effort | Impact |
|------|------|--------|--------|
| Test set evaluation | `src/train.py` | 2h | HIGH - Fix misleading validation-only metrics |
| Threshold selection | `src/metrics.py` (new) | 3h | HIGH - Enable discrete class maps |
| Basic ROC/PR curves | `src/visualize.py` (new) | 2h | HIGH - Essential reporting |

### 🟠 **High Priority (Week 2)**
| Task | File | Effort | Impact |
|------|------|--------|--------|
| Confusion matrices | `src/visualize.py` | 2h | MEDIUM-HIGH - Error analysis |
| Calibration plots | `src/visualize.py` | 2h | MEDIUM-HIGH - Validate calibration |
| Enhanced model card | `src/inference.py` | 2h | MEDIUM - Better reporting |

### 🟡 **Medium Priority (Week 3)**
| Task | File | Effort | Impact |
|------|------|--------|--------|
| Standalone eval script | `src/evaluate.py` (new) | 4h | MEDIUM - Operational flexibility |
| Inference w/ GT comparison | `src/inference.py` | 3h | MEDIUM - Production validation |

### 🟢 **Lower Priority (Week 4+)**
| Task | File | Effort | Impact |
|------|------|--------|--------|
| Cross-validation | `src/cross_validate.py` (new) | 10h | LOW-MEDIUM - Research quality |
| Ablation studies | `src/ablation.py` (new) | 8h | LOW - Feature understanding |
| Explainability | `src/explainability.py` (new) | 10h | LOW - Interpretability |

---

## 📋 Week 1 Implementation Checklist

### Day 1: Test Set Evaluation (2-3 hours)

**Step 1**: Add test dataset loader to `src/train.py` (line ~250)
```python
# After val_dataset creation
try:
    test_dataset = LandslideDataset(tiles_dir, labels_dir, 'test', None)
except ValueError:
    test_dataset = None

test_loader = None
if test_dataset and len(test_dataset) > 0:
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
```

**Step 2**: Add test evaluation after training loop (line ~380)
```python
# After training loop and before returning
test_metrics = None
if test_loader:
    test_loss, test_metrics, _ = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        num_classes,
        dataset_cfg.get('positive_class', num_classes - 1),
        collect_probs=False,
    )
    print(f"[train] Test metrics: {test_metrics}")
```

**Step 3**: Add to training report (line ~400)
```python
training_report = {
    'history': history,
    'best_epoch': best_epoch + 1 if best_epoch >= 0 else None,
    'best_metrics': best_metrics,  # validation metrics
    'test_metrics': test_metrics,   # ADD THIS
    'model': {...},
}
```

**Step 4**: Update model card in `src/inference.py` (~line 145)
```python
if best_metrics:
    lines.append('## Validation Metrics')
    for key, value in best_metrics.items():
        # ... existing code ...

# ADD THIS
if report.get('test_metrics'):
    lines.append('')
    lines.append('## Test Metrics (Held-out)')
    for key, value in report['test_metrics'].items():
        if value is None or (isinstance(value, (int, float)) and np.isnan(value)):
            continue
        if isinstance(value, float):
            lines.append(f'- {key}: {value:.4f}')
        else:
            lines.append(f'- {key}: {value}')
```

**Test**: Run pipeline and verify test metrics appear in:
- Console output during training
- `artifacts/experiments/training_metrics.json`
- `outputs/model_card.md`

---

### Day 2: Threshold Selection (3-4 hours)

**Step 1**: Create `src/metrics.py`
```python
"""Additional metrics and threshold selection utilities."""

import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import roc_curve, precision_recall_curve

def find_optimal_threshold_youden(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx], {
        'threshold': float(thresholds[optimal_idx]),
        'sensitivity': float(tpr[optimal_idx]),
        'specificity': float(1 - fpr[optimal_idx]),
        'j_score': float(j_scores[optimal_idx]),
    }

def find_optimal_threshold_f1(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Find threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx], {
        'threshold': float(thresholds[optimal_idx]),
        'precision': float(precision[optimal_idx]),
        'recall': float(recall[optimal_idx]),
        'f1_score': float(f1_scores[optimal_idx]),
    }
```

**Step 2**: Apply in `src/train.py` after validation evaluation
```python
from src.metrics import find_optimal_threshold_youden, find_optimal_threshold_f1

# After collecting validation probabilities
if best_calibration and best_calibration[0] is not None:
    y_scores, y_true = best_calibration
    threshold_youden, youden_stats = find_optimal_threshold_youden(y_true, y_scores)
    threshold_f1, f1_stats = find_optimal_threshold_f1(y_true, y_scores)
    
    training_report['optimal_thresholds'] = {
        'youden': youden_stats,
        'f1_max': f1_stats,
    }
```

**Step 3**: Apply threshold in `src/inference.py` (after susceptibility calculation)
```python
# After calibrating probabilities
threshold_info = None
metrics_path = training_artifacts.get('metrics_path')
if metrics_path and os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        report = json.load(f)
    threshold_info = report.get('optimal_thresholds', {})

if threshold_info:
    # Use Youden's threshold by default (configurable)
    method = config['inference'].get('threshold_method', 'youden')
    threshold = threshold_info.get(method, {}).get('threshold', 0.5)
else:
    threshold = 0.5

# Generate class map
class_map = np.zeros_like(susceptibility, dtype=np.uint8)
class_map[valid_mask] = (susceptibility[valid_mask] >= threshold).astype(np.uint8)
# Map to 3-class: 0->1 (low), 1->3 (high) or use quantiles for medium
```

**Test**: Verify optimal thresholds computed and class maps generated

---

### Day 3: Basic Visualizations (3-4 hours)

**Step 1**: Create `src/visualize.py` skeleton
```python
"""Visualization utilities for model evaluation."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_roc_curve(y_true, y_scores, output_path, title="ROC Curve"):
    """Generate ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curve(y_true, y_scores, output_path, title="Precision-Recall Curve"):
    """Generate PR curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    baseline = np.sum(y_true) / len(y_true)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR (AP = {avg_precision:.3f})')
    plt.axhline(y=baseline, color='navy', lw=2, linestyle='--', label=f'Baseline ({baseline:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

**Step 2**: Integrate into `src/train.py`
```python
from src.visualize import plot_roc_curve, plot_pr_curve

# After collecting validation/test probabilities
figures_dir = os.path.join(experiments_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

if best_calibration and best_calibration[0] is not None:
    y_scores, y_true = best_calibration
    plot_roc_curve(y_true, y_scores, 
                   os.path.join(figures_dir, 'roc_validation.png'),
                   title='ROC Curve (Validation)')
    plot_pr_curve(y_true, y_scores,
                  os.path.join(figures_dir, 'pr_validation.png'),
                  title='Precision-Recall (Validation)')

if test_loader and test_probabilities is not None:
    # Similar for test set
    plot_roc_curve(test_y_true, test_y_scores,
                   os.path.join(figures_dir, 'roc_test.png'),
                   title='ROC Curve (Test)')
```

**Test**: Verify PNG files generated in `artifacts/experiments/figures/`

---

## 🚀 Quick Commands Reference

### Run full pipeline with evaluation
```bash
python -m src.main_pipeline
```

### Force recreation (after adding evaluation features)
```bash
python -m src.main_pipeline --force_recreate
```

### Check test metrics
```bash
cat artifacts/experiments/training_metrics.json | grep -A 10 "test_metrics"
```

### View model card
```bash
cat outputs/model_card.md
```

### Future: Standalone evaluation
```bash
python -m src.evaluate --model_path artifacts/experiments/best_model.pth --split test
```

### Future: Cross-validation
```bash
python -m src.main_pipeline --cross_validate --folds 5
```

---

## 📊 Expected Outputs After Week 1

```
artifacts/
└── experiments/
    ├── best_model.pth (existing)
    ├── isotonic_calibrator.joblib (existing)
    ├── training_metrics.json (ENHANCED with test_metrics, optimal_thresholds)
    └── figures/ (NEW)
        ├── roc_validation.png
        ├── roc_test.png
        ├── pr_validation.png
        └── pr_test.png

outputs/
├── model_card.md (ENHANCED with test metrics, figure references)
├── <area>_landslide_susceptibility.tif (existing)
├── <area>_uncertainty.tif (existing)
├── <area>_class_map.tif (NEW)
└── <area>_valid_mask.tif (existing)
```

---

## ⚠️ Common Issues & Solutions

### Issue: Test tiles not found
**Solution**: Verify `config.yaml` has `test_size: 0.2` and tiles exist in `artifacts/tiles/test/`

### Issue: Visualizations fail with backend error
**Solution**: Add `matplotlib.use('Agg')` before importing pyplot

### Issue: Test metrics show NaN
**Solution**: Check that test split has both positive and negative samples

### Issue: Memory error during visualization
**Solution**: Process probabilities in batches or reduce plot DPI

### Issue: Threshold selection unstable
**Solution**: Ensure sufficient validation samples (>1000) and class balance

---

## 📚 Documentation References

- **Full Roadmap**: `EVALUATION_ROADMAP.md` (this file's companion)
- **Current Analysis**: `EVALUATION_ANALYSIS.md`
- **Pipeline Guide**: `AGENTS.md`
- **Specification**: `descriptive_script.md` Section 13

---

## 🎓 Learning Resources

### Threshold Selection
- [Youden's J Statistic](https://en.wikipedia.org/wiki/Youden%27s_J_statistic)
- [ROC Analysis](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)

### Calibration
- [Calibration Curves](https://scikit-learn.org/stable/modules/calibration.html)
- [Expected Calibration Error](https://arxiv.org/abs/1706.04599)

### Cross-Validation
- [Spatial CV for Geospatial Data](https://geomoer.github.io/moer-mpg-data-analysis/unit12/unit12-01_cross_validation.html)

---

**Next Steps**: Start with Day 1 tasks above, then proceed to Week 2 visualization tasks in the full roadmap.
