# Aggressive Improvements - Version 2 (CORAL + Oversampling)

**Date:** October 28, 2025  
**Baseline:** ResNet50 + Weighted Focal Loss (v1)  
**Target:** Production-ready model (Kappa ≥0.40, Class 1 Prec ≥25%, Class 2 Rec ≥90%)

---

## 🎯 V1 Performance (Baseline for V2)

**Model:** ResNet50 + FocalDiceLoss  
**Configuration:**
- Encoder: ResNet50
- Class weights: [0.5, 3.0, 1.5]
- Focal gamma: 2.5
- Epochs: 50
- Dropout: 0.4

**Results:**
- **Cohen's Kappa:** 0.3108 (Target: ≥0.40) ❌
- **Class 1 Precision:** 14.61% (Target: ≥25%) ❌
- **Class 2 Recall:** 94.12% (Target: ≥90%) ✅
- **Overall Accuracy:** 74.33%
- **AUROC (High vs Rest):** 0.9689 ✅

**Key Issues Identified:**
1. Class 1 (Medium-Risk) still severely underrepresented (~2.65% of data)
2. Class weight 3.0 for Class 1 may be too aggressive → hurt Class 2 precision
3. Model doesn't explicitly enforce ordinal relationships (Low < Medium < High)
4. Confusion matrix shows 140,100 false positives (Low → Medium)

---

## 🚀 V2 Aggressive Improvements

### 1. **CORAL (Ordinal Regression) Loss** ✅ IMPLEMENTED

**What it does:**
- Explicitly models cumulative probabilities: P(Y > 0) and P(Y > 1)
- Enforces ordinal constraint: Low < Medium < High
- Reduces confusion between distant classes (e.g., Low ↔ High)

**Implementation:**
- New `CORALLoss` class in `src/train.py` (lines ~400-500)
- New `CombinedOrdinalLoss` class combines FocalDice + CORAL
- Configurable via `training.use_ordinal_loss: true` in config.yaml

**Expected Impact:**
- ✅ Better Class 1 boundary definition (reduce Low → Medium confusion)
- ✅ Improved ordinal correlation (currently ρ=0.4355)
- ✅ Maintain Class 2 recall while improving precision
- ⚠️ Slight increase in training time (~10%)

**Configuration:**
```yaml
training:
  use_ordinal_loss: true  # Enable CORAL
  coral_weight: 0.3  # Weight for CORAL component
```

**References:**
- Cao et al. "Rank Consistent Ordinal Regression for Neural Networks" (2020)
- Niu et al. "Ordinal Regression with Multiple Output CNN" (2016)

---

### 2. **Class 1 (Medium-Risk) Oversampling** ✅ IMPLEMENTED

**What it does:**
- Duplicates tiles rich in Class 1 pixels
- Increases Class 1 representation from ~2.65% → ~5.0%
- Applies only to training set (not validation/test)

**Implementation:**
- New oversampling logic in `src/main_pipeline.py prepare_dataset()` (lines ~1458-1550)
- Identifies tiles where Class 1 comprises >5% of pixels
- Creates duplicates until target fraction reached

**Strategy:**
- Sort tiles by Class 1 richness (descending)
- Duplicate top tiles until deficit filled
- Duplicates saved with `_dup{N}` suffix

**Expected Impact:**
- ✅ Better Class 1 feature learning (more training examples)
- ✅ Improved Class 1 recall AND precision
- ⚠️ Increased training time (~15% more tiles)
- ⚠️ Risk of overfitting on duplicates (mitigated by augmentation)

**Configuration:**
```yaml
dataset:
  oversample_class: 1  # Class to oversample
  oversample_target_fraction: 0.05  # Target 5% representation
```

**Stats Before/After:**
- **Before:** Class 1 = 2.65% of pixels (~27,742 pixels per area)
- **Target:** Class 1 = 5.00% of pixels (~50,000 pixels per area)
- **Method:** Duplicate ~47% more Class 1-rich tiles

---

### 3. **Balanced Class Weights** ✅ UPDATED

**Previous (V1):**
```yaml
class_weights: [0.5, 3.0, 1.5]  # [Low, Medium, High]
```

**New (V2):**
```yaml
class_weights: [0.4, 2.5, 2.0]  # [Low, Medium, High]
```

**Rationale:**
1. **Class 0 (Low):** 0.5 → 0.4
   - Slightly more down-weighting (91% of data)
   - Reduces Low → Medium false positives

2. **Class 1 (Medium):** 3.0 → 2.5
   - Less aggressive weight (was too high)
   - Prevents over-prediction of medium-risk
   - Combined with oversampling for balanced approach

3. **Class 2 (High):** 1.5 → 2.0
   - Increased weight to maintain precision
   - Prevents precision drop seen in v1 (35.7% → 31.2%)
   - Keeps recall high (≥90%)

**Expected Impact:**
- ✅ Better balance between Class 1 and Class 2
- ✅ Maintain Class 2 precision while improving Class 1
- ✅ Reduce over-prediction of medium-risk

---

### 4. **Extended Training Schedule** ✅ UPDATED

**Previous (V1):**
- Epochs: 50
- Early stopping patience: 12

**New (V2):**
- Epochs: 60 (+20%)
- Early stopping patience: 15 (+25%)

**Rationale:**
- CORAL loss needs more epochs to converge
- Ordinal constraints are harder to learn
- Oversampled data benefits from longer training

**Expected Impact:**
- ✅ Better ordinal loss convergence
- ✅ More stable validation metrics
- ⚠️ +20% training time (~16-24 hours total)

---

### 5. **Ensemble Preparation** (DEFERRED for V3)

**Planned for future iteration:**
```yaml
ensemble:
  enabled: true
  num_models: 3
  random_seeds: [42, 123, 456]
  aggregation: mean  # Options: mean, median, weighted
```

**Why deferred:**
- V2 improvements (CORAL + oversampling) may be sufficient
- Ensemble adds 3× training time (48-72 hours)
- Best to validate V2 first, then add ensemble if needed

**Future V3 Implementation:**
- Train 3 models with different seeds
- Average predictions during inference
- Expected +5-10% Kappa improvement

---

## 📊 Expected V2 Performance Targets

### Conservative Estimates (70% confidence)

| Metric | V1 (Baseline) | V2 Target | V2 Stretch |
|--------|---------------|-----------|------------|
| **Cohen's Kappa** | 0.3108 | **≥0.38** | ≥0.42 |
| **Overall Accuracy** | 74.33% | **≥76%** | ≥78% |
| **Class 0 Precision** | 99.63% | ≥99.5% | ≥99.6% |
| **Class 0 Recall** | 72.70% | ≥74% | ≥76% |
| **Class 1 Precision** | 14.61% | **≥22%** | **≥28%** |
| **Class 1 Recall** | 88.42% | ≥90% | ≥92% |
| **Class 1 F1** | 25.07% | **≥35%** | **≥42%** |
| **Class 2 Precision** | 31.19% | **≥32%** | ≥35% |
| **Class 2 Recall** | 94.12% | **≥92%** | ≥94% |
| **Class 2 F1** | 46.85% | **≥48%** | ≥52% |
| **AUROC (High)** | 0.9689 | ≥0.96 | ≥0.97 |
| **Ordinal ρ** | 0.4355 | **≥0.50** | ≥0.55 |

### Key Success Metrics (Must Achieve 3/4)

1. ✅ **Cohen's Kappa ≥ 0.38** → Approaching production target (0.40)
2. ✅ **Class 1 Precision ≥ 22%** → +50% improvement over v1
3. ✅ **Class 2 Recall ≥ 92%** → Maintain safety-critical performance
4. ✅ **Ordinal ρ ≥ 0.50** → Strong ordinal progression

---

## 🔬 Implementation Details

### Files Modified

1. **`src/train.py`** (MAJOR)
   - Added `CORALLoss` class (~80 lines)
   - Added `CombinedOrdinalLoss` class (~60 lines)
   - Updated loss selection logic to support ordinal loss
   - Total additions: ~150 lines

2. **`src/main_pipeline.py`** (MAJOR)
   - Added Class 1 oversampling logic in `prepare_dataset()` (~120 lines)
   - Detects Class 1-rich tiles (>5% Class 1 pixels)
   - Duplicates tiles until target fraction reached
   - Updates dataset summary with oversampling stats

3. **`config.yaml`** (MODIFIED)
   - Updated `dataset.oversample_class: 1`
   - Updated `dataset.oversample_target_fraction: 0.05`
   - Updated `training.use_ordinal_loss: true`
   - Updated `training.coral_weight: 0.3`
   - Updated `training.class_weights: [0.4, 2.5, 2.0]`
   - Updated `training.epochs: 60`
   - Updated `training.early_stopping_patience: 15`

### Backup Files Created

- `artifacts/experiments/best_model_v1_resnet50_focal.pth` (125 MB)
- `outputs/evaluation_production/production_metrics_v1.json` (1.7 KB)
- `outputs/evaluation_production/PRODUCTION_ANALYSIS_v1.md` (15 KB)

---

## 🎓 Technical Rationale

### Why CORAL Loss?

**Problem:** Standard cross-entropy treats classes as independent categories
- Low (0), Medium (1), High (2) are ordinal, not nominal
- Misclassifying Low as High is WORSE than Low as Medium
- But cross-entropy penalizes them equally

**Solution:** CORAL models cumulative probabilities
- Task 1: P(Y > 0) = Probability of Medium OR High
- Task 2: P(Y > 1) = Probability of High
- Enforces: If P(High) is high, then P(Medium OR High) must be even higher

**Benefits:**
- Reduces distant-class confusion (Low ↔ High)
- Better boundary definition for Class 1
- Improved ordinal correlation
- Maintains class imbalance handling from focal loss

### Why Oversampling Class 1?

**Problem:** Severe class imbalance makes Class 1 hard to learn
- Class 1 is only 2.65% of data (27,742 pixels)
- Model sees ~50× more Class 0 examples
- Even with class weight 3.0, still insufficient

**Solution:** Duplicate Class 1-rich tiles
- Increases exposure to Class 1 patterns
- More examples → better feature learning
- Combined with augmentation → variety maintained

**Why 5% target?**
- Doubles Class 1 representation (2.65% → 5%)
- Keeps balance reasonable (not 33% like Class 0)
- Prevents over-correction that hurt Class 2

**Alternative considered:** SMOTE (synthetic oversampling)
- More complex, requires sklearn-compatible data format
- CORAL + simple duplication is simpler first step
- Can add SMOTE in V3 if V2 insufficient

### Why Balanced Class Weights?

**V1 Analysis:**
- Class weight 3.0 for Class 1 → improved precision from 11.37% to 14.61% ✅
- BUT Class 2 precision dropped from 35.7% to 31.2% ❌
- Trade-off was acceptable but not optimal

**V2 Strategy:**
- Oversampling handles Class 1 data scarcity
- Lower weight 2.5 prevents over-emphasis
- Higher weight 2.0 for Class 2 maintains precision
- Combined approach is more balanced

---

## 📋 Execution Plan

### Step 1: Backup Complete ✅
```bash
# Backed up v1 artifacts
artifacts/experiments/best_model_v1_resnet50_focal.pth
outputs/evaluation_production/production_metrics_v1.json
outputs/evaluation_production/PRODUCTION_ANALYSIS_v1.md
```

### Step 2: Validate Configuration ✅
```bash
# Check config.yaml changes
grep -A 5 "use_ordinal_loss" config.yaml
grep -A 3 "oversample_class" config.yaml
grep "class_weights" config.yaml
```

### Step 3: Run Pipeline (NEXT)
```bash
source .venv/bin/activate
.venv/bin/python -m src.main_pipeline --force_recreate
```

**Expected Duration:** 18-24 hours on CPU
- Preprocessing: ~30 min (same as before)
- Dataset prep with oversampling: ~45 min (+15 min for duplicates)
- Training with CORAL loss: ~20-22 hours (+2-4 hours for ordinal loss)
- Inference: ~30 min (same as before)

### Step 4: Evaluate V2
```bash
.venv/bin/python -m src.evaluate
```

### Step 5: Compare V2 vs V1
- Load `production_metrics_v1.json` and `production_metrics.json`
- Generate comparison tables and charts
- Validate target achievement
- Decision: Deploy, iterate, or add ensemble

---

## ⚠️ Risks and Mitigations

### Risk 1: CORAL Loss Instability
**Risk:** Ordinal loss may destabilize training  
**Mitigation:**
- Moderate coral_weight: 0.3 (30% of total loss)
- FocalDice remains primary loss (70%)
- Extended patience: 15 epochs

### Risk 2: Oversampling Overfitting
**Risk:** Duplicates may cause overfitting  
**Mitigation:**
- Only duplicate Class 1-rich tiles (not all)
- Augmentation creates variety (brightness, flip, rotate)
- Validation split unchanged (no duplicates)

### Risk 3: Training Time
**Risk:** 20-24 hours is long on CPU  
**Mitigation:**
- Monitor training progress via `training_log_v2.txt`
- Early stopping at 15 patience (prevents wasted epochs)
- Resume capability if interrupted

### Risk 4: Class 2 Regression
**Risk:** Improvements to Class 1 hurt Class 2  
**Mitigation:**
- Increased Class 2 weight: 1.5 → 2.0
- Balanced approach (not over-emphasizing Class 1)
- Monitoring both precision and recall

---

## 📈 Success Criteria

### Minimum Viable Product (MVP)
✅ Pass 3 of 4 core metrics:
1. Cohen's Kappa ≥ 0.38
2. Class 1 Precision ≥ 22%
3. Class 2 Recall ≥ 92%
4. Ordinal ρ ≥ 0.50

### Production Ready
✅ Pass ALL of:
1. Cohen's Kappa ≥ 0.40
2. Class 1 Precision ≥ 25%
3. Class 2 Recall ≥ 90%
4. AUROC ≥ 0.90
5. Ordinal ρ ≥ 0.50

### Stretch Goal (Ideal)
✅ Exceed all targets:
1. Cohen's Kappa ≥ 0.42
2. Class 1 Precision ≥ 28%
3. Class 2 Recall ≥ 94%
4. Macro F1 ≥ 0.60

---

## 🔄 Next Steps After V2

### If V2 Meets MVP (Kappa ≥0.38):
1. **Deploy as Pilot**
   - Use for expert-reviewed surveys
   - Collect field validation data
   - Monitor real-world performance

2. **Iterate to V3** (Optional ensemble)
   - Train 3 models with seeds [42, 123, 456]
   - Average predictions
   - Expected +5-10% Kappa boost

### If V2 Falls Short (Kappa <0.38):
1. **Diagnose Issues**
   - Check training curves for convergence
   - Analyze confusion matrix patterns
   - Review Class 1 precision improvements

2. **Next Iteration Options:**
   - Increase coral_weight: 0.3 → 0.5
   - More aggressive oversampling: 5% → 7.5%
   - Try SMOTE instead of simple duplication
   - Consider EfficientNet-B3 encoder

### If V2 Exceeds Stretch Goal (Kappa ≥0.42):
1. **Production Deployment** 🎉
   - Full autonomous susceptibility mapping
   - Regulatory compliance applications
   - Resource allocation without expert review

2. **Optimization**
   - Dockerize inference pipeline
   - Set up monitoring and logging
   - Create user documentation
   - Deploy to web service

---

## 📚 References and Resources

### Papers
1. Cao et al. (2020) "Rank Consistent Ordinal Regression for Neural Networks"
2. Lin et al. (2017) "Focal Loss for Dense Object Detection"
3. Milletari et al. (2016) "V-Net: Fully Convolutional Neural Networks"

### Documentation
- `PRODUCTION_IMPROVEMENTS.md` - V1 improvements documentation
- `PRODUCTION_ANALYSIS_v1.md` - V1 evaluation analysis
- `AGENTS.md` - Project overview and troubleshooting
- `INFERENCE_ENHANCEMENTS.md` - Advanced inference techniques

### Code References
- `src/train.py` lines 400-600 - CORAL loss implementation
- `src/main_pipeline.py` lines 1458-1550 - Oversampling logic
- `config.yaml` - All configuration parameters

---

**Prepared by:** GitHub Copilot  
**Date:** October 28, 2025  
**Status:** Ready for retraining  
**Estimated Completion:** October 29-30, 2025  

**Commands to start:**
```bash
source .venv/bin/activate
.venv/bin/python -m src.main_pipeline --force_recreate > training_log_v2.txt 2>&1 &
tail -f training_log_v2.txt
```
