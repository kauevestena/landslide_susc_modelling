# Critique — Landslide Susceptibility Modelling Pipeline

> **Date:** February 2025  
> **Scope:** Critical analysis of the full pipeline — from preprocessing through training to inference and evaluation — identifying flaws, pitfalls, and design decisions that may undermine the reliability of the delivered susceptibility maps.

---

## Table of Contents

1. [Label Encoding & Ground-Truth Representation](#1-label-encoding--ground-truth-representation)
2. [Mixed-Domain Merge: Spatial & Geometric Pitfalls](#2-mixed-domain-merge-spatial--geometric-pitfalls)
3. [Normalization Leakage Across Domains](#3-normalization-leakage-across-domains)
4. [Spatial Splitting & Data Leakage](#4-spatial-splitting--data-leakage)
5. [Class Imbalance: Overlapping Remedies, Uncertain Net Effect](#5-class-imbalance-overlapping-remedies-uncertain-net-effect)
6. [Soft Labels: Silently Disabled in the Active Path](#6-soft-labels-silently-disabled-in-the-active-path)
7. [Training on a Single Split Without Cross-Validation](#7-training-on-a-single-split-without-cross-validation)
8. [Temperature Override Defeating Learned Calibration](#8-temperature-override-defeating-learned-calibration)
9. [Calibration–Susceptibility Misalignment](#9-calibration--susceptibility-misalignment)
10. [CRF Post-Processing: Questionable Bilateral Signal](#10-crf-post-processing-questionable-bilateral-signal)
11. [D8 Flow Accumulation: Critical Performance Bottleneck](#11-d8-flow-accumulation-critical-performance-bottleneck)
12. [Evaluation Framework: Encoding Mismatch and Reductionism](#12-evaluation-framework-encoding-mismatch-and-reductionism)
13. [Augmentation Strategy: Minimal Diversity](#13-augmentation-strategy-minimal-diversity)
14. [Ordinal Loss Design: Weak CORAL Signal](#14-ordinal-loss-design-weak-coral-signal)
15. [Inference Class-Map Generation: Ambiguous Logic](#15-inference-class-map-generation-ambiguous-logic)
16. [Generalisability and Transferability Gaps](#16-generalisability-and-transferability-gaps)
17. [Summary of Risk Levels](#17-summary-of-risk-levels)

---

## 1. Label Encoding & Ground-Truth Representation

### 1.1 Nodata Value Collision

In `prepare_mixed_domain_dataset.py`, the merged label raster is written with a GeoTIFF profile that sets `nodata=0`. However, after the ground truth remapping performed in `main_pipeline.py` (original GT 1→0, GT 2→1, GT 3→2), the value `0` is a **valid class** — it represents "Low risk." Any downstream tool that honours the GeoTIFF nodata tag will silently mask out every Low-risk pixel. While the `.npy` training path bypasses this because labels are loaded directly and nodata is marked as `255`, any GIS inspection of the exported GeoTIFF tiles will display misleading data. If someone were to reload from the GeoTIFF rather than the `.npy`, the entire Low-risk class would vanish.

**Impact:** Data misinterpretation during QA; risk of silent data loss if the workflow is modified to read from GeoTIFFs.

### 1.2 Dual Encoding Across Training and Evaluation

The training pipeline remaps the ground truth: original values `{0→nodata, 1→0, 2→1, 3→2}`. All internal model operations (loss functions, metrics, confusion matrices) use the `{0, 1, 2}` encoding. However, `evaluate.py` directly loads the original ground truth raster and expects values `{1, 2, 3}`. Its binary evaluation strategies hard-code comparisons like `y_true == 3` for "High risk" and `y_true >= 2` for "Medium+High". This means the training metrics and the standalone evaluation metrics are computed on **different label encodings**, making them non-comparable without explicit mental remapping. There is no automated check that the loaded ground truth matches the expected encoding. If a raster were already remapped, the evaluation would silently produce meaningless results.

**Impact:** Fragile, error-prone; direct comparison of training and evaluation metrics requires manual bookkeeping.

### 1.3 The `tile_labels != 0` Filter in Candidate Selection

In the mixed-domain spatial split (`prepare_mixed_domain_dataset.py`, candidate tile filtering), the code discards tiles where the label contains only zeros. After remapping, `0` is "Low risk." This means tiles that are entirely Low-risk **are silently dropped** from the dataset. This introduces a systematic bias: the model never sees pure Low-risk tiles during training, inflating the perceived prevalence of Medium and High risk. The model may learn to overpredict risk because it never encounters the "boring" regions.

**Impact:** Systematic positive bias in training; the model is deprived of negative examples that would anchor the Low-risk end of the susceptibility spectrum.

---

## 2. Mixed-Domain Merge: Spatial & Geometric Pitfalls

### 2.1 Unimplemented CRS Reprojection

`prepare_mixed_domain_dataset.py` contains a code comment at the CRS check: `"# (Implementation would go here - for now, assume they match)"`. If the train and test areas are ever supplied in different coordinate reference systems, the pipeline will **silently produce spatially misaligned data**. The vertical concatenation will proceed, overlaying features from different projections without warning. There is no runtime assertion; the code simply logs a message and continues.

**Impact:** Silent geometric corruption if inputs in different CRS are ever used.

### 2.2 Vertical Concatenation Destroys Spatial Context

The mixed-domain merge vertically stacks the train area (1,574 rows) on top of the test area (19,148 rows) to create a single 20,722-row raster. This concatenation is a purely computational convenience — the two areas are **geographically disjoint** sites. Yet the spatial blocking algorithm treats them as a single continuous space, meaning:

- **Blocks near the seam** (row ~1,574) may contain tiles from both areas, artificially bridging two different geographies.
- Any spatial feature that depends on absolute position (flow accumulation, aspect) will have meaningless values at the boundary.
- The merged geotransform is inherited from the train area profile without adjustment, so pixel coordinates in the test portion of the merged raster **do not correspond to real-world coordinates**.

**Impact:** Fundamentally flawed spatial representation; spatial features at the merge boundary are geophysically meaningless.

### 2.3 Geotransform Inheritance Without Correction

The merged raster's profile is cloned from the train area's profile, with only `height` updated to the combined total. The `transform` (affine mapping from pixel to geographic coordinates) is kept as-is. This means:

- For the first 1,574 rows, the geotransform is correct (pointing to the train area's real-world location).
- For rows 1,574–20,722, the geotransform maps pixels to coordinates that extend **south of the train area**, not to the test area's actual location.
- The exported GeoTIFF tiles carry these wrong coordinates. Any GIS overlay will place test-area tiles at the wrong location on the map.

**Impact:** All GeoTIFF tiles from the test area portion carry incorrect spatial references.

### 2.4 Zero-Padding for Width Mismatch

When the train and test areas have different widths, the narrower one is zero-padded. Zero-filled feature columns represent **physical impossibilities** (e.g., zero elevation, zero slope, zero NDVI). These fabricated pixels:

- Contaminate normalization if stats were computed after padding (currently they are not, but the padding is not masked).
- May end up in training tiles near the edge, teaching the model patterns that do not exist in nature.
- Are not covered by the valid mask unless the original area's mask also happened to exclude them.

**Impact:** Introduction of unrealistic feature values at raster margins.

---

## 3. Normalization Leakage Across Domains

Normalization statistics (per-channel mean and standard deviation) are computed in `main_pipeline.py:compute_normalization_stats()` using **only the train area's** preprocessed feature stack. When the mixed-domain path is active, these same statistics are applied to normalise both the train and test areas before merging.

This is problematic in two directions:

1. **Distribution shift:** If the test area has a different elevation range, vegetation density, or spectral distribution, the train-area statistics will miscenter and misscale the test features. Normalised values may fall outside the expected [-3, 3] range, pushing activations into saturated regimes.
2. **Information leakage (subtle):** The normalization stats are computed before the train/val/test split. In the non-mixed-domain path, all pixels in the train area contribute to the stats, including those that will end up in val and test tiles. This is a mild form of data leakage.

**Impact:** Potential domain adaptation gap when the test area has different feature distributions; mild normalization leakage from future test pixels.

---

## 4. Spatial Splitting & Data Leakage

### 4.1 The Blocking Algorithm Is Chain-Based, Not Cluster-Based

The spatial blocking in `prepare_mixed_domain_dataset.py:split_area_tiles_spatially()` sorts tiles by position and then groups consecutive tiles into blocks by checking whether each tile is within a stride distance of the **previous tile in the sorted list**. This produces a 1D chain decomposition that is sensitive to the sort order and to the shape of the tile distribution.

- **Failure mode:** A U-shaped or L-shaped cluster of tiles will be split into multiple blocks because the sort order traverses one arm before the other. Tiles that are spatially adjacent (same 2D neighbourhood) end up in different blocks, and those blocks can be assigned to different splits (train vs. test), re-introducing spatial leakage.
- A true spatial clustering (e.g., DBSCAN, grid-based partitioning) would be more robust.

### 4.2 Tile Overlap Creates Correlated Samples Across Splits

Tiles are generated with `tile_overlap=128` on a `tile_size=256` grid, meaning adjacent tiles share **75% of their pixels** (128×256 shared region per adjacent pair). Even though the block-based split tries to assign nearby tiles to the same split, the blocking granularity may not fully capture overlap zones. Two tiles with 75% pixel overlap landing in train and validation would make validation metrics **overly optimistic** — the model has already seen most of those pixels during training.

### 4.3 Train/Test Area Size Imbalance

The train area contributes 1,574 rows while the test area contributes 19,148 rows — a **1:12 ratio**. After the spatial split, the vast majority of tiles come from the test area. This means:

- The "train" split is dominated by test-area geography and class distributions.
- If the two areas have different geomorphological characteristics, the model will be biased towards the larger (test) area's patterns.
- The model is effectively being trained primarily on the test area's data, then evaluated on a subset of that same area, which defeats the purpose of having separate areas.

**Impact:** The train/test area distinction is largely cosmetic; the model is overwhelmingly trained on test-area data.

---

## 5. Class Imbalance: Overlapping Remedies, Uncertain Net Effect

The pipeline stacks multiple class-imbalance mitigations without evidence that their combined effect is well-calibrated:

| Mechanism | Where | Effect |
|-----------|-------|--------|
| `positive_fraction` tile sampling | `main_pipeline.py` | Oversamples tiles containing rare classes |
| Class weights `[0.4, 2.5, 2.0]` | `train.py` (Focal loss) | Per-pixel weighting in loss function |
| Focal loss `γ=2.5` | `train.py` | Down-weights easy examples |
| Tile duplication | `train.py` (`_balance_tiles`) | Repeats minority-class tiles up to 3× |
| SMOTE | `train.py` (`_apply_smote`) | Synthesizes new pixel vectors for minority classes |
| Soft label smoothing `α=0.2` | `soft_labels.py` | Redistributes probability mass to adjacent classes |
| Ordinal CORAL loss | `train.py` | Penalises rank violations |

Each mechanism individually is defensible, but their interaction is uncharacterised. For example:

- **Class weights + Focal loss** already dramatically up-weight the rare high-risk class. Adding **tile duplication** on top further inflates its influence. The effective weight on a High-risk pixel could be `2.0 (class weight) × (1-p)^2.5 (focal) × 3 (tile duplication)`, which can easily exceed 20× the weight of a Low-risk pixel.
- **SMOTE at the pixel level** creates synthetic feature vectors by interpolating between pixel values in the 28-dimensional feature space. This destroys spatial structure — the synthesized pixels do not form spatially coherent patches. They are inserted back into tiles, creating **chimeric tiles** where some pixels are real and some are artificial, with the artificial pixels having no spatial autocorrelation.
- **Soft label smoothing** redistributes 20% of the target probability to adjacent classes. This means the model is trained to partially predict "Medium" even for clearly "High" pixels. Combined with Focal loss (which focuses on hard examples), the training signal becomes self-contradictory: Focal loss pushes the model to commit strongly, while soft labels tell it to hedge.

**Impact:** The net effect on calibration is unpredictable. Risk of oscillating gradients, poor convergence, or overcorrected class predictions.

---

## 6. Soft Labels: Silently Disabled in the Active Path

The `config.yaml` sets `label_smoothing.enabled: true` with `type: ordinal` and `alpha: 0.2`. However, when `mixed_domain: true` is active (which it is in the current configuration), the pipeline follows `prepare_mixed_domain_dataset()` instead of `prepare_dataset()`. 

**`prepare_mixed_domain_dataset()` does not implement soft label generation.** It saves all labels as `uint8` hard class indices. The soft label smoothing code in `soft_labels.py` is only invoked by `prepare_dataset()` in the single-area path.

As a result:

- The config declares soft labels are enabled, but they are **never generated** for the actual tiles used in training.
- The `LandslideDataset` in `train.py` auto-detects the label format (if 2D → hard, if 3D → soft). Since the labels are 2D `uint8`, the dataset loads them as hard labels and the `SoftDiceCrossEntropyLoss` falls back to its hard-label codepath.
- The `CombinedOrdinalLoss` used in training explicitly handles both paths, but because labels are hard, the soft-label branch (KL divergence) is never exercised.

The config creates a false sense that ordinal soft labels are contributing to training. They are not.

**Impact:** Misleading configuration; the training runs with standard hard labels despite the config suggesting otherwise. Any hyperparameter tuning done under the assumption of soft labels was wasted.

---

## 7. Training on a Single Split Without Cross-Validation

The entire model selection — architecture, hyperparameters, class weights, loss function blend — is validated on **a single train/val/test split**. There is no k-fold cross-validation and no repeated random sub-sampling.

With only 111 validation tiles and 134 test tiles (per `dataset_summary.json`), the metric estimates have high variance. A different random seed for the spatial split could produce materially different results:

- The reported macro IoU of 0.8358 could fluctuate by ±0.05 or more depending on which tiles land in which split.
- The best epoch (56 out of 60) might change, leading to a different model checkpoint.
- The isotonic calibrator is fitted on one validation split's probabilities — a different split could yield a different calibration curve.

Without cross-validation, it is impossible to quantify this instability. The reported metrics are **point estimates without confidence intervals**.

**Impact:** No statistical confidence in reported performance; the model may be overfit to the specific split rather than to the underlying phenomenon.

---

## 8. Temperature Override Defeating Learned Calibration

The pipeline learns a temperature scaling parameter via NLL minimisation on the validation set. The `training_metrics.json` records `T=1.37`, meaning the model's logits need to be divided by 1.37 to produce well-calibrated softmax probabilities.

However, `config.yaml` sets `inference.temperature_override: 1.0`. The inference code checks for this override and **replaces the learned temperature** with 1.0. This means:

- The model's probabilities are systematically **overconfident** (since `T=1.0 < T_optimal=1.37`).
- The entire temperature calibration procedure (collecting validation logits, optimising NLL) is performed but its result is discarded.
- This overconfidence propagates into the susceptibility map, the class map, and the uncertainty estimate.

The override was presumably set for debugging or experimentation but was never reverted.

**Impact:** The delivered susceptibility maps are miscalibrated — probabilities are too extreme. The uncertainty map underestimates true uncertainty.

---

## 9. Calibration–Susceptibility Misalignment

The pipeline computes **two** susceptibility outputs:

1. **Ordinal susceptibility** = `0×P(Low) + 0.5×P(Med) + 1.0×P(High)` — used as the primary deliverable.
2. **P(High)** — the probability of the highest risk class — used for isotonic calibration.

The isotonic calibrator is fitted to `P(High)` on the validation set, mapping raw probabilities to calibrated probabilities. But the calibration is then applied only to `P(High)`, **not to the ordinal susceptibility score** that is actually exported as the main deliverable.

The ordinal score is a weighted sum of all three class probabilities. It is a different quantity than `P(High)` and has a different distribution. The calibration guarantees (if `P(High)` says 0.3, then roughly 30% of such pixels should indeed be high-risk) do not transfer to the ordinal score.

Additionally, the threshold selection (`select_optimal_thresholds` in `train.py`) is performed on the `P(High_class)` probability vector, yielding a threshold designed for binary classification of the highest class. This threshold is then applied to the **ordinal susceptibility score** in inference to generate a binary positive map — but the ordinal score and `P(High)` are on different scales, making the threshold semantically meaningless for the ordinal score.

**Impact:** The primary deliverable (ordinal susceptibility) is neither calibrated nor meaningfully thresholded. Calibration effort is wasted on a secondary output.

---

## 10. CRF Post-Processing: Questionable Bilateral Signal

The Dense CRF post-processing uses `pydensecrf` with a bilateral filter that takes "colour" channels as input to enforce that similar-looking pixels should have similar labels. In a typical image segmentation CRF, these would be RGB channels.

In this pipeline, the CRF loads the **entire feature stack** (28 channels of normalised geoscientific features) and uses the first 3 channels as the bilateral signal. Looking at the feature stack, the first 3 channels correspond to **terrain derivatives** (e.g., DTM, slope, aspect — depending on the stacking order), not colour information.

This means:

- The bilateral term enforces that pixels with similar **elevation and slope** should have the same class. While this has geomorphological merit, it is not what the CRF's bilateral kernel was designed for — it assumes approximately Euclidean distance in a perceptual colour space.
- The `color_weight` and `compat_bilateral` hyperparameters were presumably set with RGB semantics in mind. Their effect on terrain-derivative channels is unpredictable.
- Only 3 of 28 channels are used. The orthophoto RGB channels (channels 14–16 in the feature stack) would be more appropriate for a bilateral CRF, but they are not at indices 0–2.

**Impact:** The CRF bilateral filter operates on the wrong features, potentially introducing spatial artefacts rather than correcting them.

---

## 11. D8 Flow Accumulation: Critical Performance Bottleneck

The `d8_flow_accumulation()` function in `main_pipeline.py` implements the D8 flow routing algorithm as a **pure Python for-loop** over every pixel in the DTM, in descending elevation order:

```python
order = np.argsort(elevation, axis=None)[::-1]
for index in tqdm(order, desc="d8_flow_accumulation", leave=False):
    r = index // ncols
    c = index % ncols
    ...
```

For the merged raster of 20,722×5,964 = ~123 million pixels, this loop iterates 123 million times in CPython, with each iteration performing 8 neighbour lookups and conditional branches. Estimated wall-clock time: **hours** on a modern CPU.

Established GIS libraries (GRASS, WhiteboxTools, RichDEM) compute flow accumulation in seconds using optimized C/Fortran backends. The custom implementation also has algorithmic limitations:

- **No flat resolution**: Pixels on flat surfaces have no steepest downhill neighbour, causing `target = None` and zero flow propagation. Real D8 implementations include flat resolution algorithms (e.g., GRASS's r.watershed).
- **No depression filling**: The `fill_sinks()` function uses morphological closing, which is a crude approximation. True hydrological sink filling (e.g., Planchon-Darboux) is required for correct flow routing.
- **Edge effects**: Flow reaching the raster boundary simply stops accumulating, which is correct but means that catchments truncated by the raster extent produce underestimated accumulation values.

**Impact:** Prohibitive runtime for large rasters; algorithmically inferior to established tools; may produce incorrect flow accumulation values in flat or depressed areas.

---

## 12. Evaluation Framework: Encoding Mismatch and Reductionism

### 12.1 Label Encoding Mismatch

As noted in §1.2, `evaluate.py` expects ground truth values `{1, 2, 3}` (original encoding), while the model outputs class maps with values `{0, 1, 2}` (remapped encoding). The susceptibility map is a continuous `[0, 1]` score that is compared against discrete ordinal classes. The evaluation binary strategies use hard-coded comparisons:

```python
y_true_binary_high = (y_true == 3).astype(int)   # Expects original encoding
y_true_binary_risk = (y_true >= 2).astype(int)    # Medium + High in original
```

If someone passes the model's class map output (values 0, 1, 2) instead of the original ground truth, all thresholds and metrics will be wrong. There is no validation of the expected encoding.

### 12.2 Reductive Binary Strategies

The pipeline produces a 3-class ordinal segmentation, but all four evaluation strategies collapse it to binary:

1. High vs. rest
2. Risk (Medium+High) vs. Low
3. Spearman correlation (ordinal, but a single scalar)
4. Medium vs. Low (only when High is absent)

There is **no true multi-class evaluation**:

- No 3×3 confusion matrix
- No per-class precision/recall
- No ordinal-aware metrics (quadratic weighted kappa, ordinal c-statistic)
- No reliability diagrams or calibration curves for the ordinal score
- No spatial error analysis (where does the model fail?)

The binary reductions can paint an optimistic picture: a model that correctly separates High from Low will score well on Strategies 1 and 2, even if it completely confuses Medium with both other classes. Medium-risk discrimination — arguably the most challenging and most useful capability — is only assessed as a fallback when High is absent.

**Impact:** The evaluation does not adequately assess what it should: the model's ability to discriminate all three ordinal risk levels. Critical failure modes (Medium misclassification) may go undetected.

---

## 13. Augmentation Strategy: Minimal Diversity

The training augmentations consist of:

- Random horizontal flip (probability configurable via `flip_prob`)
- Random vertical flip (same probability)
- Random 90° rotations (k ∈ {0, 1, 2, 3})
- Optional Gaussian noise (`noise_std`, currently likely 0)

Missing augmentations that are standard in remote sensing segmentation:

| Augmentation | Relevance |
|-------------|-----------|
| Elastic deformation | Simulates terrain distortion from varying DTM sources |
| Brightness/contrast jitter | Simulates different illumination in orthophotos |
| Channel dropout | Builds robustness to missing or noisy bands |
| Random crop-and-resize | Introduces scale invariance |
| CutMix / MixUp | Regularisation that has proven effective for imbalanced segmentation |
| Colour space transforms | For the orthophoto RGB channels |

The geometric augmentations (flip, rotate90) only generate 8 unique orientations. For a dataset with 626 training tiles, this is a limited diversity multiplier.

**Impact:** The model may overfit to specific feature orientations and spectral characteristics present in the training data.

---

## 14. Ordinal Loss Design: Weak CORAL Signal

The `CombinedOrdinalLoss` is structured as:

```
Total = (0.7 × FocalLoss + 0.3 × DiceLoss) + 0.3 × CORALLoss
```

The CORAL component models cumulative probabilities P(Y > 0) and P(Y > 1) via binary cross-entropy. However:

- **The CORAL weight is only 0.3**, and it is added to a FocalDice term that already strongly shapes the loss landscape. In practice, CORAL contributes a minor perturbation.
- **CORAL with only 2 binary tasks** (for 3 classes) has very limited capacity. The gradient signal from two binary cross-entropy terms, weighted at 0.3, is easily dominated by the Focal loss gradient (which directly optimises multi-class discrimination at every pixel).
- **Ordinal consistency is not enforced**: the CORAL loss encourages ordinal consistency but does not guarantee it. The Focal+Dice loss can override the CORAL signal, producing predictions where `P(Y > 1) > P(Y > 0)` — an ordinal rank violation. There is no constraint or projection to enforce cumulative probability monotonicity.
- The CORAL loss operates on **raw logits passed through sigmoid**, while the FocalDice operates on logits passed through **softmax**. These two normalisations are incompatible — sigmoid and softmax produce different probability spaces. The model receives conflicting training signals about what the logits should represent.

**Impact:** The ordinal structure is weakly enforced; the model may produce rank-violated predictions where the cumulative probabilities are inconsistent.

---

## 15. Inference Class-Map Generation: Ambiguous Logic

The inference class map is generated with competing and overlapping logic:

1. First, `argmax(probabilities)` is computed — the standard maximum-probability class assignment.
2. Then, if `class_breaks` are configured (currently `[0.33, 0.67]`), the ordinal susceptibility score is **digitized** into classes based on fixed thresholds.
3. Otherwise, a binary positive mask (`susceptibility >= optimal_threshold`) overrides the argmax.

With the current config (`class_breaks: [0.33, 0.67]`), the class map is produced by:
```python
class_map = np.digitize(susceptibility, bins=[0.33, 0.67])
```

This means the class assignment depends on the **ordinal susceptibility score** (a weighted average of probabilities), not on the class probabilities themselves. A pixel with probabilities `[0.34, 0.33, 0.33]` (argmax = Low) gets an ordinal score of `0.33*0 + 0.33*0.5 + 0.33*1.0 ≈ 0.495`, which maps to **Medium risk** via the class breaks. The argmax says Low; the class breaks say Medium.

The class breaks are hard-coded to divide the `[0, 1]` interval into equal thirds, with no empirical justification. They do not correspond to any trained decision boundary. In a well-calibrated model, the natural class boundaries would emerge from the probability distribution, not from arbitrary fixed thresholds.

**Impact:** The delivered class map may disagree with the probability-based argmax classification. The class breaks are arbitrary and not validated against the model's actual output distribution.

---

## 16. Generalisability and Transferability Gaps

### 16.1 Single Geographic Context

The model is trained on two study areas from a single survey campaign (presumably same drone, same flight conditions, same season). There is no evidence that the model generalises to:

- Different geographic regions with different geology, vegetation, or climate
- Different seasons (snow cover, vegetation phenology)
- Different sensor platforms (satellite imagery, different drone cameras)
- Different DTM resolutions or sources (LiDAR vs. photogrammetric)

### 16.2 No Domain Adaptation

When the model is applied to a new area (the intended use case for susceptibility mapping), the feature distributions will differ from training. The pipeline has no domain adaptation mechanism:

- No feature standardisation relative to local statistics
- No test-time adaptation
- No domain-adversarial training
- No ensemble of models from different areas

### 16.3 Temporal Blindness

Landslide susceptibility is inherently dynamic — it changes with rainfall, seismicity, land use change, and progressive slope weakening. The model treats it as a static classification problem. There is no temporal dimension in the input features, no consideration of antecedent conditions, and no mechanism to update predictions as conditions change.

### 16.4 ESA WorldCover as a Fixed Snapshot

The 11-class WorldCover land cover (one-hot encoded) contributes 11 of the 28 input channels. WorldCover is a **2020/2021 annual product**. If the orthophoto and DTM were acquired in a different year, the land cover may have changed (deforestation, urbanisation). The model learns associations between land cover patterns and landslide susceptibility that may not hold if the land cover vintage does not match the terrain data vintage.

**Impact:** The model is a snapshot classifier with no transferability guarantees. Applying it to new areas or times without retraining is speculative.

---

## 17. Summary of Risk Levels

| Issue | Severity | Likelihood of Impact | Remediation Difficulty |
|-------|----------|---------------------|----------------------|
| Soft labels silently disabled (§6) | **High** | Certain | Low — fix the path or disable in config |
| Temperature override (§8) | **High** | Certain | Trivial — remove override or set to learned value |
| Label nodata=0 collision (§1.1) | **Medium** | Depends on workflow | Low — use 255 as nodata |
| Candidate tile `!= 0` filter (§1.3) | **High** | Certain | Low — check for valid pixels instead |
| Unimplemented CRS reprojection (§2.1) | **Medium** | If CRS differ | Medium — use rasterio.warp |
| Vertical concatenation (§2.2) | **High** | Certain | High — fundamental redesign needed |
| Geotransform inheritance (§2.3) | **Medium** | Certain for GeoTIFFs | Medium — compute correct transforms |
| Normalization leakage (§3) | **Medium** | Certain | Medium — compute on merged or per-area |
| Spatial blocking fragility (§4.1) | **Medium** | Depends on geometry | Medium — use DBSCAN or grid blocks |
| Tile overlap leakage (§4.2) | **Medium** | Likely | Medium — ensure overlap zones stay in same split |
| Area size imbalance (§4.3) | **High** | Certain | Medium — stratified or proportional sampling |
| Overlapping class balance (§5) | **Medium** | Likely | Medium — ablation study needed |
| SMOTE spatial destruction (§5) | **High** | Certain | Low — remove SMOTE or apply per-tile |
| No cross-validation (§7) | **Medium** | Certain | High — significant compute cost |
| Calibration misalignment (§9) | **High** | Certain | Medium — calibrate the ordinal score |
| CRF bilateral channels (§10) | **Medium** | Certain | Low — use orthophoto bands |
| D8 performance (§11) | **High** | Certain for large rasters | Low — use WhiteboxTools or RichDEM |
| Evaluation encoding mismatch (§12.1) | **Medium** | If misused | Low — add encoding validation |
| Evaluation reductionism (§12.2) | **Medium** | Certain | Medium — add multi-class metrics |
| Minimal augmentation (§13) | **Medium** | Likely | Low — add standard augmentations |
| Weak CORAL signal (§14) | **Low** | Uncertain | Medium — increase weight or enforce constraints |
| Ambiguous class map (§15) | **Medium** | Certain | Low — choose one strategy |
| No transferability (§16) | **High** | If applied to new areas | High — fundamental limitation |

---

*This critique is intended as a constructive assessment to guide improvements. The pipeline demonstrates substantial engineering effort and contains many sound design choices (sliding-window inference, Gaussian blending, spatial blocking intent, multi-faceted loss). The issues identified above represent areas where the implementation either diverges from its stated intent, introduces uncharacterised biases, or fails to meet the robustness standards needed for operational landslide susceptibility mapping.*
