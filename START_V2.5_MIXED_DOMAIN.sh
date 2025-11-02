#!/bin/bash
# V2.5 Mixed-Domain Training Launch Script
# Fixes geographic domain shift by merging training and test areas into a single dataset
# Exports tiles as GeoTIFF for spatial inspection alongside .npy for fast training

set -e

echo "================================================================================"
echo "V2.5 MIXED-DOMAIN TRAINING - Geographic Generalization Fix"
echo "================================================================================"
echo ""
echo "BACKGROUND:"
echo "  Previous V2.5 run trained only on training area → excellent validation (AUROC 0.9996)"
echo "  But failed on test area (AUROC 0.5743) due to geographic domain shift."
echo ""
echo "SOLUTION:"
echo "  - Merge training and test areas into single feature stack"
echo "  - Create tiles from BOTH geographies"
echo "  - Split ensuring all sets have samples from both areas"
echo "  - Export tiles as GeoTIFF for spatial inspection"
echo ""
echo "EXPECTED OUTCOME:"
echo "  - Model learns transferable landslide features"
echo "  - Test area performance improves significantly (target: AUROC >0.85)"
echo "  - Can generalize to new geographic areas"
echo ""
echo "CONFIG CHANGES:"
echo "  ✓ dataset.use_mixed_domain: true"
echo "  ✓ dataset.export_geotiff_tiles: true"
echo ""
echo "================================================================================"
echo ""

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment .venv not found"
    echo "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "[1/5] Activating virtual environment..."
source .venv/bin/activate

# Verify config
echo "[2/5] Verifying configuration..."
if ! grep -q "use_mixed_domain: true" config.yaml; then
    echo "ERROR: use_mixed_domain not enabled in config.yaml"
    echo "Please set dataset.use_mixed_domain: true"
    exit 1
fi

echo "  ✓ Mixed-domain enabled"
echo "  ✓ GeoTIFF export enabled"

# Backup current model (if exists)
if [ -f "artifacts/experiments/best_model.pth" ]; then
    echo "[3/5] Backing up current V2.5 model..."
    cp artifacts/experiments/best_model.pth artifacts/experiments/best_model_v2.5_single_domain.pth
    cp artifacts/experiments/training_metrics.json artifacts/experiments/training_metrics_v2.5_single_domain.json
    echo "  ✓ Backed up to best_model_v2.5_single_domain.pth"
else
    echo "[3/5] No existing model to backup"
fi

# Launch training
echo "[4/5] Launching mixed-domain training pipeline..."
echo ""
echo "  This will:"
echo "    1. Preprocess both training and test areas"
echo "    2. Merge into single feature stack"
echo "    3. Create ~150-200 tiles from both geographies"
echo "    4. Export tiles as .npy (training) and .tif (inspection)"
echo "    5. Train for 60 epochs (~24-30 hours)"
echo ""
echo "  Tile locations:"
echo "    - Training tiles (.npy): artifacts/tiles/{train,val,test}/"
echo "    - GeoTIFF tiles (.tif): artifacts/tiles/geotiff/{train,val,test}/"
echo "    - Labels (.tif): artifacts/labels/geotiff/{train,val,test}/"
echo ""
echo "  You can inspect tiles in QGIS while training runs!"
echo ""
read -p "Press ENTER to start, or Ctrl+C to cancel..."
echo ""

.venv/bin/python -m src.main_pipeline --force_recreate

echo ""
echo "[5/5] Training complete!"
echo ""
echo "================================================================================"
echo "NEXT STEPS"
echo "================================================================================"
echo ""
echo "1. INSPECT TILES (while training or after):"
echo "   - Open QGIS and load: artifacts/tiles/geotiff/train/*.tif"
echo "   - Check that tiles come from both training and test areas"
echo "   - Verify class distribution looks balanced"
echo ""
echo "2. EVALUATE ON TEST AREA:"
echo "   .venv/bin/python -m src.evaluate \\"
echo "     --susceptibility outputs/test_susceptibility.tif \\"
echo "     --ground_truth /home/kaue/data/landslide/test/Ground_truth_test.tif \\"
echo "     --output_dir outputs/evaluation_v2.5_mixed_domain"
echo ""
echo "3. COMPARE RESULTS:"
echo "   - V2.5 single-domain: outputs/evaluation_v2.5/"
echo "   - V2.5 mixed-domain: outputs/evaluation_v2.5_mixed_domain/"
echo "   - Expected: AUROC improves from 0.5743 to >0.85"
echo ""
echo "4. IF SUCCESSFUL:"
echo "   - Deploy model for production use"
echo "   - Can now generalize to new geographic areas"
echo ""
echo "================================================================================"
