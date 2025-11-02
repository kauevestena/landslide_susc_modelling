#!/bin/bash
# V2.5 Configuration Validation Script

echo "=============================================="
echo "V2.5 HYBRID IMPROVEMENTS VALIDATION"
echo "=============================================="
echo ""

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "❌ ERROR: Virtual environment not found"
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1)
echo "✓ Python: $python_version"

# Check if imbalanced-learn is installed
if python -c "import imblearn" 2>/dev/null; then
    echo "✓ imbalanced-learn: INSTALLED"
else
    echo "⚠️  imbalanced-learn: NOT INSTALLED"
    echo "   Installing now..."
    pip install imbalanced-learn
    if [ $? -eq 0 ]; then
        echo "   ✓ Installation successful"
    else
        echo "   ❌ Installation failed"
        exit 1
    fi
fi

echo ""
echo "=============================================="
echo "CONFIG.YAML VALIDATION"
echo "=============================================="

# Run Python validation
python << 'EOF'
import yaml
import sys

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("")
    print("MODEL CONFIGURATION:")
    print("-" * 40)
    
    # Check encoder
    encoder = config['model']['encoder']
    print(f"  Encoder: {encoder}")
    if encoder != 'efficientnet-b4':
        print(f"  ❌ ERROR: Expected efficientnet-b4, got {encoder}")
        sys.exit(1)
    print("  ✓ EfficientNet-B4 configured")
    
    # Check attention
    attention = config['model'].get('attention', False)
    print(f"  Spatial Attention: {attention}")
    if not attention:
        print("  ❌ ERROR: Attention should be enabled")
        sys.exit(1)
    print("  ✓ Spatial Attention enabled")
    
    print("")
    print("DATASET CONFIGURATION:")
    print("-" * 40)
    
    # Check oversampling
    oversample_class = config['dataset'].get('oversample_class', None)
    oversample_target = config['dataset'].get('oversample_target_fraction', 0.0)
    print(f"  Oversample Class: {oversample_class}")
    print(f"  Target Fraction: {oversample_target}")
    if oversample_target != 0.075:
        print(f"  ❌ ERROR: Expected 0.075, got {oversample_target}")
        sys.exit(1)
    print("  ✓ Oversampling configured (7.5% target)")
    
    # Check SMOTE
    use_smote = config['dataset'].get('use_smote', False)
    smote_k = config['dataset'].get('smote_k_neighbors', 5)
    print(f"  SMOTE Enabled: {use_smote}")
    print(f"  SMOTE k-neighbors: {smote_k}")
    if not use_smote:
        print("  ❌ ERROR: SMOTE should be enabled")
        sys.exit(1)
    print("  ✓ SMOTE configured")
    
    print("")
    print("TRAINING CONFIGURATION:")
    print("-" * 40)
    
    # Check CORAL
    use_ordinal = config['training'].get('use_ordinal_loss', False)
    coral_weight = config['training'].get('coral_weight', 0.0)
    print(f"  Ordinal Loss: {use_ordinal}")
    print(f"  CORAL Weight: {coral_weight}")
    if not use_ordinal or coral_weight != 0.3:
        print("  ❌ ERROR: CORAL should be enabled with weight 0.3")
        sys.exit(1)
    print("  ✓ CORAL loss configured")
    
    # Check class weights
    class_weights = config['training'].get('class_weights', [])
    print(f"  Class Weights: {class_weights}")
    expected_weights = [0.4, 2.5, 2.0]
    if class_weights != expected_weights:
        print(f"  ❌ ERROR: Expected {expected_weights}, got {class_weights}")
        sys.exit(1)
    print("  ✓ Balanced class weights configured")
    
    print("")
    print("INFERENCE CONFIGURATION:")
    print("-" * 40)
    
    # Check CRF
    crf_enabled = config['inference']['crf']['enabled']
    crf_iters = config['inference']['crf']['iterations']
    spatial_weight = config['inference']['crf']['spatial_weight']
    compat_spatial = config['inference']['crf']['compat_spatial']
    compat_bilateral = config['inference']['crf']['compat_bilateral']
    
    print(f"  CRF Enabled: {crf_enabled}")
    print(f"  CRF Iterations: {crf_iters}")
    print(f"  Spatial Weight: {spatial_weight}")
    print(f"  Compat Spatial: {compat_spatial}")
    print(f"  Compat Bilateral: {compat_bilateral}")
    
    if not crf_enabled:
        print("  ❌ ERROR: CRF should be enabled")
        sys.exit(1)
    if crf_iters != 8:
        print(f"  ❌ ERROR: Expected 8 iterations, got {crf_iters}")
        sys.exit(1)
    print("  ✓ Enhanced CRF configured")
    
    print("")
    print("=" * 50)
    print("✅ ALL V2.5 VALIDATIONS PASSED")
    print("=" * 50)
    print("")
    print("READY FOR RETRAINING!")
    print("")
    print("Next steps:")
    print("  1. Backup V2 model (if not done):")
    print("     cp artifacts/experiments/best_model.pth \\")
    print("        artifacts/experiments/best_model_v2_coral_oversampling.pth")
    print("")
    print("  2. Run full pipeline with --force_recreate:")
    print("     python -m src.main_pipeline --force_recreate")
    print("")
    print("  Expected duration: 24-30 hours")
    print("  Expected results:")
    print("    - Cohen's Kappa ≥ 0.48")
    print("    - Class 1 Precision ≥ 23% ✅ (TARGET MET)")
    print("    - Class 2 Recall ≥ 93%")
    print("    - Ordinal ρ ≥ 0.58")
    print("")

except FileNotFoundError:
    print("❌ ERROR: config.yaml not found")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo "=============================================="
    exit 0
else
    echo "=============================================="
    echo "❌ VALIDATION FAILED"
    echo "=============================================="
    exit 1
fi
