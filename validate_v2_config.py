#!/usr/bin/env python3
"""Validate V2 configuration before retraining."""

import yaml
import sys

print("=" * 70)
print("VALIDATING V2 AGGRESSIVE IMPROVEMENTS CONFIGURATION")
print("=" * 70)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

all_valid = True

# Check dataset oversampling
print("\n1. Dataset Configuration:")
oversample_class = config["dataset"].get("oversample_class")
oversample_target = config["dataset"].get("oversample_target_fraction")
if oversample_class == 1 and oversample_target == 0.05:
    print(f"   ✅ Class 1 oversampling enabled: target {oversample_target*100}%")
else:
    print(f"   ❌ Oversampling config incorrect: class={oversample_class}, target={oversample_target}")
    all_valid = False

# Check training configuration
print("\n2. Training Configuration:")
use_ordinal = config["training"].get("use_ordinal_loss", False)
coral_weight = config["training"].get("coral_weight", 0.0)
class_weights = config["training"].get("class_weights", [])
epochs = config["training"].get("epochs", 0)
patience = config["training"].get("early_stopping_patience", 0)

if use_ordinal and coral_weight == 0.3:
    print(f"   ✅ CORAL loss enabled with weight {coral_weight}")
else:
    print(f"   ❌ CORAL config incorrect: enabled={use_ordinal}, weight={coral_weight}")
    all_valid = False

if class_weights == [0.4, 2.5, 2.0]:
    print(f"   ✅ Balanced class weights: {class_weights}")
else:
    print(f"   ❌ Class weights incorrect: {class_weights} (expected [0.4, 2.5, 2.0])")
    all_valid = False

if epochs >= 60:
    print(f"   ✅ Extended training: {epochs} epochs, patience {patience}")
else:
    print(f"   ❌ Insufficient epochs: {epochs} (expected ≥60)")
    all_valid = False

# Check backup files
print("\n3. Backup Files:")
import os
backup_files = [
    "artifacts/experiments/best_model_v1_resnet50_focal.pth",
    "outputs/evaluation_production/production_metrics_v1.json",
    "outputs/evaluation_production/PRODUCTION_ANALYSIS_v1.md"
]
for f in backup_files:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024*1024)  # MB
        print(f"   ✅ {f} ({size:.1f} MB)")
    else:
        print(f"   ❌ Missing: {f}")
        all_valid = False

# Check model configuration
print("\n4. Model Configuration:")
encoder = config["model"]["encoder"]
dropout = config["model"]["dropout_prob"]
if encoder == "resnet50" and dropout == 0.4:
    print(f"   ✅ Encoder: {encoder}, Dropout: {dropout}")
else:
    print(f"   ⚠️  Encoder: {encoder}, Dropout: {dropout}")

# Summary
print("\n" + "=" * 70)
if all_valid:
    print("✅ ALL VALIDATIONS PASSED - READY FOR V2 RETRAINING")
    print("=" * 70)
    print("\nTo start retraining:")
    print("  source .venv/bin/activate")
    print("  .venv/bin/python -m src.main_pipeline --force_recreate > training_log_v2.txt 2>&1 &")
    print("  tail -f training_log_v2.txt")
    print("\nExpected duration: 18-24 hours")
    print("Expected improvements:")
    print("  • Cohen's Kappa: 0.31 → ≥0.38 (+23%)")
    print("  • Class 1 Precision: 14.6% → ≥22% (+50%)")
    print("  • Class 2 Recall: maintain ≥92%")
    sys.exit(0)
else:
    print("❌ VALIDATION FAILED - FIX ISSUES BEFORE RETRAINING")
    print("=" * 70)
    sys.exit(1)
