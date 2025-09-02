# Drone-based Landslide Vulnerability Scoring Pipeline (Deep Learning)

## 0) Objective & Framing

- *Goal:* Produce a georeferenced raster where each pixel stores *vulnerability* (probability of landslide susceptibility given terrain/cover conditions).  
- *Inputs:*  
  1. *Orthophoto* (RGB; more bands if available).  
  2. *DEM* (from same survey).  
  3. *Ground truth* (inventory polygons or masks, for training).  
- *Outputs:*  
  - Primary: *Vulnerability score raster* (0–1 float).  
  - Secondary: *Uncertainty raster, **QA maps, and a **model card*.

---

## 1) Data Contracts

- *CRS & units:* Single projected CRS, meters.  
- *Resolution:* Fix target GSD (e.g., 0.1–0.2 m). Resample all inputs.  
- *Extent & alignment:* Same grid alignment, same nodata footprint.  
- *Metadata:* CRS, transform, nodata, acquisition date.  
- *Tiling:* Split into overlapping tiles for large rasters.  
- *Ground truth:*  
  - Rasterize polygons.  
  - Binary or categorical classes.  
  - Store *ignore mask* for uncertain edges.

---

## 2) Pre-processing & Feature Stack

### DEM Hygiene
- Sink filling, conditioning, edge padding.

### DEM-Derived Channels
- Slope (degrees).  
- Aspect: *sin* + *cos*.  
- Plan/profile curvature.  
- TPI (multi-scale).  
- TRI / roughness.  
- Flow accumulation (log).  
- TWI.  
- LS factor (optional).  
- Distance to drainage (log).

### Orthophoto Channels
- RGB (+NIR if available).  
- Radiometric normalization.  
- Shadow mask (optional).

### Final Feature Tensor
- Channels stacked in consistent order.  
- Z-score normalization.  
- Nodata masked.

---

## 3) Label Engineering

- Binary: 1 = landslide, 0 = non-landslide.  
- Ordered classes: keep as ordinal or map to continuous.  
- Boundary uncertainty: ignore band ±1–2 px.  
- Hard-negative sampling: steep slopes with no slides.

---

## 4) Dataset Splits

- *Spatial block splits* to prevent leakage.  
- No patch overlaps between Train/Val/Test.  
- If multi-date: hold out entire dates for test.

---

## 5) Modeling Choices

### Baseline
- U-Net (or DeepLabv3+) with ImageNet pretrained encoder.  
- Input: stacked patches.  
- Output: vulnerability ∈ [0,1].

### Alternatives
- Two-branch fusion (RGB vs terrain).  
- Ordinal regression head.  
- Multi-task head for vulnerability + boundary.

---

## 6) Tiling, Patching, Sampling

- Train patch: 512×512 (overlap 64).  
- Inference window: 1024×1024 (overlap 128).  
- Sampling:  
  - 50% landslide-positive patches.  
  - 50% negatives (biased to steep terrain).  
  - Cap duplicates with NMS.

---

## 7) Losses, Metrics, Class Imbalance

- *Loss:* BCEWithLogits + Soft Dice.  
- Add Focal if positives <5%.  
- Ignore mask for excluded pixels.  
- *Metrics:* AUROC, AUPRC, IoU/Dice, Brier score, calibration metrics.

---

## 8) Data Augmentation

- *Geometric:* Rotations, flips, scaling, translations.  
- *Photometric (RGB):* Brightness/contrast/hue jitter, blur, artifacts.  
- *Terrain:* Gaussian noise, slight DEM perturbations.  
- *Cutout:* Occlusions.  
- No unrealistic terrain warps.

---

## 9) Training Protocol

- Batch size per GPU capacity; cosine LR with warmup.  
- Early stopping on AUPRC.  
- Checkpoint top-k.  
- Mixed precision if stable.  
- Stage-2 hard-negative mining.

---

## 10) Probability Calibration

- Fit isotonic regression (or Platt scaling) on validation.  
- Apply at inference for calibrated scores.

---

## 11) Inference & Mosaic

- Sliding-window with overlap + Gaussian blending.  
- Test-time augmentation (rot/flip).  
- Calibrate scores.  
- Export GeoTIFF:  
  - Band 1: vulnerability.  
  - Band 2: uncertainty.  
  - Band 3: valid mask.

---

## 12) Post-processing (Optional)

- Morphology for presentation only.  
- CRF or boundary loss if edges ragged.  
- Sanity checks: exclude near-flat unless supported by imagery.

---

## 13) Evaluation & Reporting

- Spatial cross-validation.  
- Threshold selection: Youden’s J, fixed precision, or quantiles.  
- Ablation: RGB only vs Terrain only vs Fused.  
- Deliverables: PDF summary with maps, ROC/PR curves, calibration, confusion matrices.

---

## 14) Uncertainty & Explainability

- Epistemic: Monte Carlo dropout or ensembles.  
- Attribution: Grad-CAM / Integrated Gradients overlays.

---

## 15) Reproducibility & Ops

- Config-first (YAML).  
- Fixed random seeds.  
- File structure: raw, derivs, tiles, labels, splits, experiments.  
- Versioning: models, scalers, calibrators.  
- Model card required.

---

## 16) Known Limitations

- No triggers (rainfall, soil, seismic).  
- Domain shift risks.  
- Micro-failures below GSD may be undetectable.

---

## 17) Acceptance Checklist

- [ ] Inputs aligned (CRS, GSD, grid).  
- [ ] DEM derivatives computed.  
- [ ] Labels rasterized + ignore mask.  
- [ ] Spatial splits verified.  
- [ ] Baseline model agreed.  
- [ ] Inference pipeline defined.  
- [ ] Calibration chosen.  
- [ ] Evaluation thresholds set.  
- [ ] Reproducibility plan ready.

---

## 18) Suggested Toolchain

- *I/O & processing:* GDAL, rasterio, richdem/whitebox, numpy.  
- *Modeling:* PyTorch, segmentation_models_pytorch, Albumentations.  
- *Tracking:* W&B or MLflow, DVC.  
- *Visualization:* QGIS, matplotlib.

---

## Minimal Baseline

1. Channels: RGB + slope, sin/cos aspect, TPI (small/medium), roughness (small/medium), logFlowAcc, TWI.  
2. U-Net ResNet-34 encoder (N channels).  
3. BCE + Dice, block split, calibrated sigmoid.  
4. Sliding-window inference with blending.  
5. Export GeoTIFF vulnerability (0–1) + uncertainty.
