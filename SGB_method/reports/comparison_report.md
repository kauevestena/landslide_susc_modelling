# Three-Method Validation Summary

Reference layers are real data clipped/reprojected to the drone grid.

## References
- official_sgb: `SGB_method/outputs/reference_official_sgb_class_map.tif`
- ground_truth_25m: `SGB_method/outputs/reference_ground_truth_25m_class_map.tif`

## Overall accuracy against references
- dl: SGB=0.8894, GT25m=0.9000
- ibge: SGB=0.0247, GT25m=0.0247
- sgb: SGB=0.5045, GT25m=0.5015

See `validation.json` for confusion matrices and class areas.