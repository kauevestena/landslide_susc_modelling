#!/bin/bash
# Script to re-run the pipeline with fixes applied
# Created: 2025-10-20

set -e  # Exit on error

echo "=========================================="
echo "Landslide Susceptibility Pipeline - Fixed"
echo "=========================================="
echo ""
echo "Configuration changes applied:"
echo "  ✓ tile_size: 512 → 256"
echo "  ✓ tile_overlap: 64 → 32"
echo "  ✓ positive_min_fraction: 0.01 → 0.005"
echo "  ✓ positive_fraction: 0.5 → 0.3"
echo "  ✓ min_valid_fraction: 0.5 → 0.4"
echo "  ✓ epochs: 20 → 30"
echo "  ✓ early_stopping_patience: 5 → 7"
echo ""
echo "Expected outcomes:"
echo "  • 30-40 training tiles (was 8)"
echo "  • 5-7 validation tiles (was 0)"
echo "  • 8-10 test tiles (was 0)"
echo "  • Validation metrics computed (was NaN)"
echo "  • More realistic susceptibility predictions"
echo ""
echo "=========================================="
echo ""

# Change to project directory
cd /home/kaue/landslide_susc_modelling

# Run the pipeline with force_recreate
echo "🚀 Starting pipeline with --force_recreate..."
echo ""
.venv/bin/python -m src.main_pipeline --force_recreate

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "📊 Next Steps:"
echo ""
echo "1. Check tile counts:"
echo "   cat artifacts/splits/dataset_summary.json | grep -A 5 'tile_counts'"
echo ""
echo "2. View training metrics:"
echo "   cat artifacts/experiments/training_metrics.json | grep -A 10 'val_metrics'"
echo ""
echo "3. Re-run evaluation:"
echo "   .venv/bin/python -m src.evaluate --analysis_only"
echo ""
echo "4. View results:"
echo "   cat outputs/evaluation/EVALUATION_SUMMARY.md"
echo ""
echo "5. Open outputs in GIS:"
echo "   ls -lh outputs/*.tif"
echo ""
echo "See FIXES_APPLIED.md for detailed guidance."
echo ""
