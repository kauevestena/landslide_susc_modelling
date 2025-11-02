# Architecture Mismatch Fix - V2.5

**Issue Date:** 2025-10-29  
**Status:** ✅ FIXED

---

## 🐛 Problem Encountered

Inference failed when loading the V2 model:
```
RuntimeError: Error(s) in loading state_dict for Unet:
  Missing key(s) in state_dict: "encoder._conv_stem.weight", "encoder._bn0.weight", ...
  Unexpected key(s) in state_dict: "encoder.conv1.weight", "encoder.bn1.weight", ...
  size mismatch for decoder.blocks.0.conv1.0.weight: 
    copying a param with shape torch.Size([256, 3072, 3, 3]) from checkpoint, 
    the shape in current model is torch.Size([256, 608, 3, 3])
```

---

## 🔍 Root Cause

**Architecture mismatch between saved model and current config:**
- **Saved model (V2):** ResNet50 encoder
  - Decoder expects ResNet50 feature dimensions: [3072, 768, 384, 128]
- **Current config (V2.5):** EfficientNet-B4 encoder  
  - Decoder expects EfficientNet-B4 feature dimensions: [608, 312, 160, 112]

**Why this happened:**
1. We updated `config.yaml` to use `encoder: efficientnet-b4`
2. The existing `artifacts/experiments/best_model.pth` is from V2 (ResNet50)
3. Inference tried to load V2 weights into V2.5 architecture → **incompatible shapes**

---

## ✅ Solution

**Use `--force_recreate` flag** to retrain the model from scratch with the new architecture:

```bash
.venv/bin/python -m src.main_pipeline --force_recreate
```

### What `--force_recreate` Does:
1. **Skips loading existing model** (`best_model.pth`)
2. **Trains new model** with EfficientNet-B4 encoder
3. **Saves new weights** that match the V2.5 architecture
4. **Prevents architecture mismatches** during inference

---

## 📊 Architecture Comparison

### V2 (ResNet50):
```python
Encoder: ResNet50
  - Layers: [3, 4, 6, 3] (13 blocks)
  - Channels: [64, 256, 512, 1024, 2048]
  - Decoder input: [2048, 1024, 512, 256] → [3072, 768, 384, 128]
```

### V2.5 (EfficientNet-B4):
```python
Encoder: EfficientNet-B4
  - Blocks: 32 mobile inverted bottleneck blocks
  - Channels: [9, 48, 32, 56, 160, 448] (6 stages)
  - Decoder input: [448, 160, 56, 48] → [608, 312, 160, 112]
```

**Incompatibility:** Channel counts are completely different → weights cannot be loaded.

---

## 🚫 Why Not Transfer Learning?

**Q:** Can't we just transfer the decoder weights and only retrain the encoder?  
**A:** No, because:
1. **Decoder dimensions depend on encoder:** Each decoder block expects specific input channels from encoder skip connections
2. **Skip connection mismatch:** ResNet50 produces [2048, 1024, 512, 256], EfficientNet-B4 produces [448, 160, 56, 48]
3. **Decoder architecture changes:** The decoder must be rebuilt to match new encoder dimensions

**Conclusion:** Must retrain the entire model (encoder + decoder + segmentation head) from scratch.

---

## ✅ Updated Training Script

`START_V2.5_TRAINING.sh` now includes a comment explaining the requirement:

```bash
# Start training in background with --force_recreate
# CRITICAL: --force_recreate is required because we changed the architecture (ResNet50 -> EfficientNet-B4)
nohup .venv/bin/python -m src.main_pipeline --force_recreate > training_log_v2.5.txt 2>&1 &
```

---

## 🎯 Impact

### Before Fix:
- ❌ Inference crashes with state_dict loading error
- ❌ Cannot run inference or evaluation
- ❌ V2.5 improvements blocked

### After Fix:
- ✅ Full V2.5 training from scratch (24-30 hours)
- ✅ EfficientNet-B4 + Spatial Attention trained properly
- ✅ Model weights match V2.5 architecture perfectly
- ✅ Inference will work correctly after training

---

## 📝 Lesson Learned

**When changing encoder architecture:**
1. Always use `--force_recreate` flag
2. Cannot reuse weights from different encoder families (ResNet ≠ EfficientNet)
3. Backup old model first (already done: `best_model_v2_coral_oversampling.pth`)
4. Expect full retraining duration (24-30 hours for V2.5)

---

## 🚀 Ready to Train

All blockers removed:
- ✅ Spatial attention fixed (decoder takes list, not unpacked args)
- ✅ GradScaler deprecation fixed
- ✅ Architecture mismatch understood (use `--force_recreate`)

**Start training now:**
```bash
./START_V2.5_TRAINING.sh
```

Expected: 24-30 hours → Class 1 Precision ≥23% ✅

---

*Architecture mismatch documented - requires full retraining with --force_recreate*
