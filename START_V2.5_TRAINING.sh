#!/bin/bash
# V2.5 Training Start Script
# This script starts the V2.5 model training with all hybrid improvements

echo "=============================================="
echo "V2.5 HYBRID IMPROVEMENTS - TRAINING START"
echo "=============================================="
echo ""
echo "Configuration Summary:"
echo "  - Encoder: EfficientNet-B4 (upgraded from ResNet50)"
echo "  - Spatial Attention: ENABLED (new feature)"
echo "  - SMOTE: ENABLED (synthetic minority oversampling)"
echo "  - Oversample Target: 7.5% (increased from 5%)"
echo "  - CRF: Enhanced (8 iterations, stronger smoothing)"
echo "  - CORAL Loss: Weight 0.3 (ordinal constraints)"
echo "  - Class Weights: [0.4, 2.5, 2.0]"
echo ""
echo "Expected Improvements over V2:"
echo "  - Class 1 Precision: 17.85% → ≥23% (+29% improvement)"
echo "  - Cohen's Kappa: 0.4534 → ≥0.48 (+7% improvement)"
echo "  - Class 2 Recall: 93.86% → ≥94.5% (maintained)"
echo "  - Ordinal ρ: 0.5666 → ≥0.585 (+3% improvement)"
echo ""
echo "Expected Duration: 24-30 hours"
echo ""
echo "=============================================="

# Confirm before starting
read -p "Start V2.5 training now? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "Starting training..."
echo "Logs will be saved to: training_log_v2.5.txt"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Start training in background with --force_recreate
# CRITICAL: --force_recreate is required because we changed the architecture (ResNet50 -> EfficientNet-B4)
nohup .venv/bin/python -m src.main_pipeline --force_recreate > training_log_v2.5.txt 2>&1 &

# Get process ID
TRAINING_PID=$!

echo "✓ Training started successfully!"
echo "  Process ID: $TRAINING_PID"
echo ""
echo "Monitor progress with:"
echo "  tail -f training_log_v2.5.txt"
echo ""
echo "Check if still running:"
echo "  ps aux | grep main_pipeline"
echo ""
echo "Stop training (if needed):"
echo "  kill $TRAINING_PID"
echo ""
echo "=============================================="
echo "Training launched! Good luck! 🚀"
echo "=============================================="
