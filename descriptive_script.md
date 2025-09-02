# Drone-based Landslide Vulnerability Scoring Pipeline (Deep Learning)

## 0) Objective & Framing

- *Goal:* Produce a georeferenced raster where each pixel stores *landslide susceptibility* (probability of landslide susceptibility given terrain/cover conditions).  
- *Inputs:*  
  1. *Orthophoto* (RGB only from drone survey).  
  2. *DTM* (only ground points classified from raw DSM from the same drone survey).
  3. *Ground truth* (raster masks for training, discretized into three classes: 1 - low; 2 - medium; 3 - high).  
- *Outputs:*  
  - Primary: *landslide susceptibility score raster* (0–1 float to be later discretized into the same three classes - low, medium, high).  
  - Secondary: *Uncertainty raster, **QA maps, and a **model card*.

---

## 1) Data Contracts

- *CRS & units:* Single projected CRS defined by the DTM, meters.  
- *Resolution:* Fix target GSD defined by the DTM. Resample all inputs.  
- *Extent & alignment:* Same grid alignment defined by the DTM, same nodata footprint.  
- *Metadata:* CRS, transform, nodata, acquisition date.  
- *Tiling:* Split into overlapping tiles for large rasters.  
- *Ground truth:*  Categorical classes (three classes: 1 - low; 2 - medium; 3 - high)

---

## 2) Pre-processing & Feature Stack

### DTM Hygiene
- Sink filling, conditioning, edge padding.

### DTM-Derived Channels
- Slope (degrees): using least squares fitted plane - Horn\Costa-Cabralno
- Aspect (degrees): *sin* + *cos*.  
- General, Plan and Profile curvature.  
- Topographic Position Index (TPI). 
- Terrain Ruggedness Index (TRI) 
- Flow Accumulation (Top-Down) - (logarithm scaled) using Multiple Flow Direction (MDT) Freeman, G.T. (1991) 
- Topographic Wetness Index (TWI) 
- Stream Power Index (SPI)
- Sediment Transport Index (STI)
- Distance to drainage (log).

### Orthophoto Channels
- RGB.  
- Radiometric normalization.  
- Shadow mask (optional).

### Orthophoto-Derived Channels
- Land use/cover (categorized into classes - low vegetation, high vegetation, barefoot terrain, building, road, water)

### Final Feature Tensor
- Channels stacked in consistent order.  
- Z-score normalization.  
- Nodata masked.

---

## 3) Label Engineering

- Categorical classes (three classes: 1 - low; 2 - medium; 3 - high)
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
- Output: landslide susceptibility ∈ [0,1].

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

- *Photometric (RGB):* Brightness/contrast/hue jitter, blur, artifacts.  
- *Terrain:* Gaussian noise, slight DTM perturbations.  
- *Cutout:* Occlusions.  

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
- Export GeoTIFFS:  
  - File 1: landslide susceptibility.  
  - File 2: uncertainty.  
  - File 3: valid mask.

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


## Suggested Toolchain

- *I/O & processing:* GDAL, SAGA, rasterio, richdem/whitebox, numpy.  
- *Modeling:* PyTorch, segmentation_models_pytorch, Albumentations.  
- *Tracking:* W&B or MLflow, DVC.  
- *Visualization:* QGIS, matplotlib.

---

## Minimal Baseline

1. Channels: see Orthophoto Derived Channels and DTM Derived Channels
2. U-Net ResNet-34 encoder (N channels).  
3. BCE + Dice, block split, calibrated sigmoid.  
4. Sliding-window inference with blending.  
5. Export GeoTIFF vulnerability (0–1) + uncertainty.
