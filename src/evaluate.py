"""
Standalone evaluation script for landslide susceptibility model outputs.

This script performs comprehensive evaluation of model predictions, including:
- Metrics computation (when ground truth is available)
- Visualization generation (ROC, PR, calibration curves)
- Spatial statistics and uncertainty analysis
- Enhanced reporting

Usage:
    # Evaluate inference outputs with ground truth
    python -m src.evaluate --outputs_dir outputs --ground_truth_path <path>

    # Analyze outputs without ground truth (statistics only)
    python -m src.evaluate --outputs_dir outputs --analysis_only

    # Evaluate test tiles if available
    python -m src.evaluate --mode tiles --split test
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Import from project modules
from src.metrics import (
    compute_threshold_metrics,
    find_optimal_threshold_f1,
    find_optimal_threshold_youden,
    select_optimal_thresholds,
)
from src.visualize import generate_all_plots, plot_roc_curve, plot_pr_curve


def evaluate_binary_strategy(
    y_true_binary: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
    output_dir: str,
    strategy_name: str,
) -> Dict:
    """
    Evaluate binary classification for a specific strategy.

    Args:
        y_true_binary: Binary ground truth (0/1)
        y_probs: Predicted probabilities
        threshold: Classification threshold
        output_dir: Output directory
        strategy_name: Name for this strategy (for file naming)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Check if we have both classes
    if len(np.unique(y_true_binary)) < 2:
        print(f"  WARNING: Only one class present!")
        return {"error": "Only one class present"}

    # Threshold-independent metrics
    try:
        auroc = roc_auc_score(y_true_binary, y_probs)
        metrics["auroc"] = float(auroc)
        print(f"  AUROC: {auroc:.4f}")
    except Exception as e:
        print(f"  AUROC: ERROR - {e}")
        metrics["auroc"] = None

    try:
        auprc = average_precision_score(y_true_binary, y_probs)
        metrics["auprc"] = float(auprc)
        print(f"  AUPRC: {auprc:.4f}")
    except Exception as e:
        print(f"  AUPRC: ERROR - {e}")
        metrics["auprc"] = None

    # Threshold-dependent metrics
    y_pred = (y_probs >= threshold).astype(int)
    acc = accuracy_score(y_true_binary, y_pred)
    f1 = f1_score(y_true_binary, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred, labels=[0, 1]).ravel()

    # Cohen's Kappa (agreement beyond chance)
    kappa = cohen_kappa_score(y_true_binary, y_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    iou_pos = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    iou_neg = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0
    macro_iou = (iou_pos + iou_neg) / 2.0

    metrics.update(
        {
            "threshold": float(threshold),
            "accuracy": float(acc),
            "f1": float(f1),
            "cohen_kappa": float(kappa),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "iou_positive": float(iou_pos),
            "iou_negative": float(iou_neg),
            "macro_iou": float(macro_iou),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }
    )

    print(f"  At threshold {threshold:.3f}:")
    print(f"    Accuracy: {acc:.4f}, F1: {f1:.4f}, Cohen's Kappa: {kappa:.4f}")
    print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"    IoU: {iou_pos:.4f}, Macro IoU: {macro_iou:.4f}")
    print(f"    Confusion: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")

    # Find optimal thresholds
    threshold_youden, youden_metrics = find_optimal_threshold_youden(
        y_true_binary, y_probs
    )
    threshold_f1, f1_metrics = find_optimal_threshold_f1(y_true_binary, y_probs)

    metrics["optimal_thresholds"] = {
        "youden": {
            "threshold": float(threshold_youden),
            "sensitivity": youden_metrics["sensitivity"],
            "specificity": youden_metrics["specificity"],
            "youden_j": youden_metrics["youden_j"],
        },
        "f1": {
            "threshold": float(threshold_f1),
            "precision": f1_metrics["precision"],
            "recall": f1_metrics["recall"],
            "f1": f1_metrics["f1"],
        },
    }

    print(f"  Optimal thresholds:")
    print(f"    Youden: {threshold_youden:.3f} (J={youden_metrics['youden_j']:.4f})")
    print(f"    F1: {threshold_f1:.3f} (F1={f1_metrics['f1']:.4f})")

    # Generate plots
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    roc_path = os.path.join(figures_dir, f"roc_curve_{strategy_name}.png")
    plot_roc_curve(
        None,
        None,
        y_probs,
        y_true_binary,
        roc_path,
        title=f"ROC Curve - {strategy_name.replace('_', ' ').title()}",
    )
    metrics["plot_roc"] = roc_path

    pr_path = os.path.join(figures_dir, f"pr_curve_{strategy_name}.png")
    plot_pr_curve(
        None,
        None,
        y_probs,
        y_true_binary,
        pr_path,
        title=f"PR Curve - {strategy_name.replace('_', ' ').title()}",
    )
    metrics["plot_pr"] = pr_path

    return metrics


def _quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> float:
    """
    Compute quadratic weighted kappa — an ordinal-aware agreement metric.

    Unlike standard Cohen's kappa, this penalises disagreements proportionally
    to the squared distance between classes. Predicting 'High' when truth is 'Low'
    is penalised more than predicting 'Medium' when truth is 'Low'.
    """
    # Build quadratic weight matrix
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = (i - j) ** 2 / (num_classes - 1) ** 2

    # Observed confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm = cm.astype(np.float64)

    # Expected confusion matrix under independence
    hist_true = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    hist_pred = np.bincount(y_pred, minlength=num_classes).astype(np.float64)
    expected = np.outer(hist_true, hist_pred)
    expected = expected / expected.sum() if expected.sum() > 0 else expected

    cm_norm = cm / cm.sum() if cm.sum() > 0 else cm

    numerator = (weights * cm_norm).sum()
    denominator = (weights * expected).sum()

    if denominator == 0:
        return 1.0  # Perfect agreement
    return 1.0 - numerator / denominator


def _evaluate_multiclass(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int,
    output_dir: str,
) -> Dict:
    """
    §12 fix: Proper multi-class evaluation with 3×3 confusion matrix,
    per-class precision/recall/F1, and ordinal-aware metrics.

    Args:
        y_true: Ground truth class labels {0, 1, 2}
        y_probs: Predicted susceptibility scores [0, 1]
        num_classes: Number of classes (3)
        output_dir: Directory for saving figures

    Returns:
        Dictionary of multi-class metrics
    """
    CLASS_NAMES = ["Low", "Medium", "High"]

    # Generate predicted classes from susceptibility scores
    # Use equal-percentile breaks based on class proportions in GT
    # as the most unbiased mapping from continuous scores → ordinal classes
    class_fractions = np.array([
        np.mean(y_true == c) for c in range(num_classes)
    ])
    cum_fractions = np.cumsum(class_fractions)
    breaks = [np.percentile(y_probs, 100 * cum_fractions[i])
              for i in range(num_classes - 1)]

    y_pred = np.digitize(y_probs, bins=breaks).astype(int)
    y_pred = np.clip(y_pred, 0, num_classes - 1)

    print(f"\n  Predicted class breaks (percentile-matched): {[f'{b:.4f}' for b in breaks]}")
    for c in range(num_classes):
        print(f"    Predicted {CLASS_NAMES[c]}: {np.sum(y_pred == c):,} pixels")

    # 3×3 Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    print(f"\n  3×3 Confusion Matrix:")
    print(f"  {'':>12} {'Pred Low':>10} {'Pred Med':>10} {'Pred High':>10}")
    for i, name in enumerate(CLASS_NAMES):
        row = "  " + f"True {name:>6}" + "".join(f"{cm[i, j]:>10,}" for j in range(num_classes))
        print(row)

    # Per-class precision, recall, F1
    per_class = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        per_class[CLASS_NAMES[c]] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "iou": float(iou),
            "support": int(cm[c, :].sum()),
        }
        print(f"\n  {CLASS_NAMES[c]:>6}: P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  IoU={iou:.4f}  Support={cm[c, :].sum():,}")

    # Macro-averaged metrics
    macro_prec = np.mean([v["precision"] for v in per_class.values()])
    macro_rec = np.mean([v["recall"] for v in per_class.values()])
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])
    macro_iou = np.mean([v["iou"] for v in per_class.values()])

    # Overall accuracy
    accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0

    # Cohen's kappa (standard)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Quadratic weighted kappa (ordinal-aware)
    qw_kappa = _quadratic_weighted_kappa(y_true, y_pred, num_classes)

    print(f"\n  Macro Precision: {macro_prec:.4f}")
    print(f"  Macro Recall:    {macro_rec:.4f}")
    print(f"  Macro F1:        {macro_f1:.4f}")
    print(f"  Macro IoU:       {macro_iou:.4f}")
    print(f"  Accuracy:        {accuracy:.4f}")
    print(f"  Cohen's Kappa:   {kappa:.4f}")
    print(f"  QW Kappa:        {qw_kappa:.4f} (ordinal-aware)")

    return {
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "predicted_class_breaks": [float(b) for b in breaks],
        "per_class": per_class,
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "macro_iou": float(macro_iou),
        "accuracy": float(accuracy),
        "cohen_kappa": float(kappa),
        "quadratic_weighted_kappa": float(qw_kappa),
    }


def load_raster(path: str) -> Tuple[np.ndarray, dict, Optional[float]]:
    """Load a GeoTIFF raster and return data + metadata."""
    with rasterio.open(path) as src:
        data = src.read(1)  # Read first band
        meta = src.meta.copy()
        nodata = src.nodata
    return data, meta, nodata


def compute_raster_statistics(
    susceptibility: np.ndarray,
    uncertainty: Optional[np.ndarray],
    valid_mask: np.ndarray,
) -> Dict:
    """Compute spatial statistics for susceptibility and uncertainty maps."""

    stats = {}

    # Susceptibility statistics
    valid_susc = susceptibility[valid_mask > 0]
    if len(valid_susc) > 0:
        stats["susceptibility"] = {
            "mean": float(np.mean(valid_susc)),
            "std": float(np.std(valid_susc)),
            "min": float(np.min(valid_susc)),
            "max": float(np.max(valid_susc)),
            "median": float(np.median(valid_susc)),
            "q25": float(np.percentile(valid_susc, 25)),
            "q75": float(np.percentile(valid_susc, 75)),
            "q95": float(np.percentile(valid_susc, 95)),
            "high_risk_fraction": float(np.mean(valid_susc > 0.7)),
            "moderate_risk_fraction": float(
                np.mean((valid_susc > 0.3) & (valid_susc <= 0.7))
            ),
            "low_risk_fraction": float(np.mean(valid_susc <= 0.3)),
        }

    # Uncertainty statistics
    if uncertainty is not None:
        valid_unc = uncertainty[valid_mask > 0]
        if len(valid_unc) > 0:
            stats["uncertainty"] = {
                "mean": float(np.mean(valid_unc)),
                "std": float(np.std(valid_unc)),
                "min": float(np.min(valid_unc)),
                "max": float(np.max(valid_unc)),
                "median": float(np.median(valid_unc)),
                "q25": float(np.percentile(valid_unc, 25)),
                "q75": float(np.percentile(valid_unc, 75)),
                "high_uncertainty_fraction": float(
                    np.mean(valid_unc > np.median(valid_unc))
                ),
            }

            # Correlation between susceptibility and uncertainty
            if len(valid_susc) == len(valid_unc):
                corr = np.corrcoef(valid_susc, valid_unc)[0, 1]
                stats["susceptibility_uncertainty_correlation"] = float(corr)

    # Coverage statistics
    total_pixels = susceptibility.size
    valid_pixels = np.sum(valid_mask > 0)
    stats["coverage"] = {
        "total_pixels": int(total_pixels),
        "valid_pixels": int(valid_pixels),
        "valid_fraction": float(valid_pixels / total_pixels),
        "invalid_pixels": int(total_pixels - valid_pixels),
    }

    return stats


def evaluate_with_ground_truth(
    susceptibility_path: str,
    ground_truth_path: str,
    valid_mask_path: Optional[str] = None,
    output_dir: str = "outputs/evaluation",
    threshold: float = 0.5,
) -> Dict:
    """
    Evaluate model predictions against ground truth.

    Args:
        susceptibility_path: Path to susceptibility GeoTIFF (probabilities)
        ground_truth_path: Path to ground truth GeoTIFF
        valid_mask_path: Path to valid mask GeoTIFF (optional)
        output_dir: Directory to save evaluation results
        threshold: Classification threshold

    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 80)
    print("EVALUATING MODEL PREDICTIONS AGAINST GROUND TRUTH")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Load rasters
    print(f"\n[evaluate] Loading susceptibility map: {susceptibility_path}")
    susc_data, susc_meta, susc_nodata = load_raster(susceptibility_path)

    print(f"[evaluate] Loading ground truth: {ground_truth_path}")
    gt_data, gt_meta, gt_nodata = load_raster(ground_truth_path)

    # Check if ground truth needs spatial resampling
    needs_resampling = susc_data.shape != gt_data.shape

    if needs_resampling:
        print(f"[evaluate] Shape mismatch detected - performing spatial resampling:")
        print(f"  Susceptibility: {susc_data.shape}")
        print(f"  Ground truth: {gt_data.shape}")

        # Check CRS compatibility
        if susc_meta.get("crs") != gt_meta.get("crs"):
            print(f"[evaluate] WARNING: CRS mismatch!")
            print(f"  Susceptibility CRS: {susc_meta.get('crs')}")
            print(f"  Ground truth CRS: {gt_meta.get('crs')}")
            print(
                f"[evaluate] Ground truth will be reprojected to match susceptibility."
            )

        # Resample ground truth to match susceptibility map
        # Using nearest neighbor to preserve class labels
        gt_resampled = np.empty_like(susc_data, dtype=gt_data.dtype)

        reproject(
            source=gt_data,
            destination=gt_resampled,
            src_transform=gt_meta["transform"],
            src_crs=gt_meta.get("crs"),
            dst_transform=susc_meta["transform"],
            dst_crs=susc_meta.get("crs"),
            resampling=Resampling.nearest,
        )

        print(f"[evaluate] Ground truth resampled to: {gt_resampled.shape}")
        gt_data = gt_resampled

    # Load or create valid mask
    if valid_mask_path and os.path.exists(valid_mask_path):
        print(f"[evaluate] Loading valid mask: {valid_mask_path}")
        mask_data, _, _ = load_raster(valid_mask_path)
    else:
        print("[evaluate] Creating valid mask from nodata values")
        mask_data = np.ones_like(susc_data, dtype=np.uint8)
        if susc_nodata is not None:
            mask_data[susc_data == susc_nodata] = 0
        if gt_nodata is not None:
            mask_data[gt_data == gt_nodata] = 0

    # Final shape check (should match after resampling)
    if susc_data.shape != gt_data.shape:
        print(f"[evaluate] ERROR: Shape mismatch persists after resampling!")
        print(f"  Susceptibility: {susc_data.shape}")
        print(f"  Ground truth: {gt_data.shape}")
        sys.exit(1)

    # Extract valid pixels
    valid_pixels = mask_data > 0
    y_probs = susc_data[valid_pixels].flatten()
    y_true = gt_data[valid_pixels].flatten()

    # §1.2 fix: Auto-detect ground truth encoding and normalise to {0, 1, 2}
    # Training uses {0, 1, 2} (Low, Medium, High) but original GT may use {1, 2, 3}.
    # Instead of hard-coding comparisons like `y_true == 3`, detect and remap.
    unique_gt_raw, counts_gt_raw = np.unique(y_true, return_counts=True)

    CLASS_NAMES = {0: "Low risk", 1: "Medium risk", 2: "High risk"}

    if set(unique_gt_raw).issubset({1, 2, 3}):
        # Original encoding {1, 2, 3} — remap to {0, 1, 2}
        print(f"\n[evaluate] Detected original GT encoding {{1, 2, 3}} — remapping to {{0, 1, 2}}")
        y_true = y_true - 1
        gt_encoding = "original_1_2_3"
    elif set(unique_gt_raw).issubset({0, 1, 2}):
        # Already in model encoding {0, 1, 2}
        print(f"\n[evaluate] Detected model GT encoding {{0, 1, 2}} — no remapping needed")
        gt_encoding = "model_0_1_2"
    elif set(unique_gt_raw).issubset({0, 1, 2, 3}):
        # Mixed — could be {0=nodata, 1, 2, 3}
        print(f"\n[evaluate] Detected encoding with 0 present alongside {{1, 2, 3}}")
        print(f"  Treating 0 as nodata and remapping {{1, 2, 3}} → {{0, 1, 2}}")
        valid_class_mask = y_true > 0
        y_true = y_true[valid_class_mask] - 1
        y_probs = y_probs[valid_class_mask]
        gt_encoding = "original_with_nodata"
    else:
        print(f"\n[evaluate] WARNING: Unexpected GT values: {unique_gt_raw}")
        print(f"  Proceeding without remapping — results may be incorrect")
        gt_encoding = "unknown"

    # Recompute distribution after remapping
    unique_gt, counts_gt = np.unique(y_true, return_counts=True)

    print(f"\n[evaluate] Ground truth value distribution (after normalisation):")
    for val, count in zip(unique_gt, counts_gt):
        label = CLASS_NAMES.get(int(val), f"Unknown({int(val)})")
        print(f"  Class {int(val)} ({label}): {count:,} pixels ({100*count/len(y_true):.2f}%)")

    print(f"\n[evaluate] Susceptibility statistics:")
    print(f"  Min: {y_probs.min():.4f}, Max: {y_probs.max():.4f}")
    print(f"  Mean: {y_probs.mean():.4f}, Median: {np.median(y_probs):.4f}")

    num_classes = 3

    # =========================================================================
    # §12 fix: TRUE MULTI-CLASS EVALUATION (3×3 confusion matrix + ordinal metrics)
    # =========================================================================
    print(f"\n[evaluate] {'='*60}")
    print(f"[evaluate] MULTI-CLASS EVALUATION (3-class ordinal)")
    print(f"[evaluate] {'='*60}")

    multiclass_metrics = _evaluate_multiclass(y_true, y_probs, num_classes, output_dir)

    # Strategy 1: Binary - High risk (class 2) vs rest (classes 0-1)
    print(f"\n[evaluate] STRATEGY 1: Binary evaluation (High vs Low+Medium)")
    y_true_binary_high = (y_true == 2).astype(int)

    # Strategy 2: Binary - Medium+High risk (classes 1-2) vs Low (class 0)
    print(f"[evaluate] STRATEGY 2: Binary evaluation (Medium+High vs Low)")
    y_true_binary_risk = (y_true >= 1).astype(int)

    # Strategy 3: Ordinal - Treat as ordered categories and compute correlation
    print(f"[evaluate] STRATEGY 3: Ordinal correlation")
    from scipy.stats import spearmanr

    corr, pval = spearmanr(y_true, y_probs)
    print(f"  Spearman's rho: {corr:.4f} (p={pval:.4e})")

    # Evaluate Strategy 1: High vs rest
    print(f"\n[evaluate] Computing metrics for Strategy 1 (High risk vs rest)...")
    print(
        f"  Positive (Class 2 - High): {np.sum(y_true_binary_high):,} ({100*np.mean(y_true_binary_high):.2f}%)"
    )
    metrics_high = evaluate_binary_strategy(
        y_true_binary_high, y_probs, threshold, output_dir, "high_vs_rest"
    )

    # Evaluate Strategy 2: Medium+High vs Low
    print(f"\n[evaluate] Computing metrics for Strategy 2 (Medium+High vs Low)...")
    print(
        f"  Positive (Classes 1-2): {np.sum(y_true_binary_risk):,} ({100*np.mean(y_true_binary_risk):.2f}%)"
    )
    metrics_risk = evaluate_binary_strategy(
        y_true_binary_risk, y_probs, threshold, output_dir, "risk_vs_low"
    )

    # Strategy 4: Medium vs Low (when High risk is absent or rare)
    metrics_medium = None
    if 2 not in unique_gt or np.sum(y_true == 2) < 100:
        print(f"\n[evaluate] STRATEGY 4: Binary evaluation (Medium vs Low)")
        print(f"  Note: Class 2 (High) absent or rare, evaluating Classes 0 vs 1 only")

        # Filter to only Class 0 and Class 1
        mask_01 = (y_true == 0) | (y_true == 1)
        y_true_01 = y_true[mask_01]
        y_probs_01 = y_probs[mask_01]

        y_true_binary_medium = (y_true_01 == 1).astype(int)

        print(
            f"  Positive (Class 1 - Medium): {np.sum(y_true_binary_medium):,} ({100*np.mean(y_true_binary_medium):.2f}%)"
        )
        print(
            f"  Negative (Class 0 - Low): {np.sum(y_true_binary_medium == 0):,} ({100*np.mean(y_true_binary_medium == 0):.2f}%)"
        )

        metrics_medium = evaluate_binary_strategy(
            y_true_binary_medium, y_probs_01, threshold, output_dir, "medium_vs_low"
        )

    # Combine all metrics
    metrics = {
        "ground_truth_encoding": {
            "detected": gt_encoding,
            "normalised_to": "{0, 1, 2}",
            "class_0": "Low landslide probability",
            "class_1": "Medium landslide probability",
            "class_2": "High landslide probability",
        },
        "ground_truth_distribution": {
            int(val): {"count": int(count), "fraction": float(count / len(y_true))}
            for val, count in zip(unique_gt, counts_gt)
        },
        "multiclass_evaluation": multiclass_metrics,
        "ordinal_correlation": {
            "spearman_rho": float(corr),
            "p_value": float(pval),
        },
        "strategy_1_high_vs_rest": metrics_high,
        "strategy_2_risk_vs_low": metrics_risk,
        "strategy_4_medium_vs_low": metrics_medium if metrics_medium else None,
    }

    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[evaluate] Saved metrics to: {metrics_path}")

    # Generate evaluation report
    report_path = os.path.join(output_dir, "evaluation_report.md")
    write_evaluation_report(
        metrics, report_path, susceptibility_path, ground_truth_path
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return metrics


def analyze_outputs_only(
    susceptibility_path: str,
    uncertainty_path: Optional[str] = None,
    valid_mask_path: Optional[str] = None,
    class_map_path: Optional[str] = None,
    output_dir: str = "outputs/evaluation",
) -> Dict:
    """
    Analyze model outputs without ground truth (spatial statistics only).

    Args:
        susceptibility_path: Path to susceptibility GeoTIFF
        uncertainty_path: Path to uncertainty GeoTIFF (optional)
        valid_mask_path: Path to valid mask GeoTIFF (optional)
        class_map_path: Path to class map GeoTIFF (optional)
        output_dir: Directory to save analysis results

    Returns:
        Dictionary of analysis statistics
    """
    print("\n" + "=" * 80)
    print("ANALYZING MODEL OUTPUTS (NO GROUND TRUTH)")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Load susceptibility
    print(f"\n[analyze] Loading susceptibility map: {susceptibility_path}")
    susc_data, susc_meta, susc_nodata = load_raster(susceptibility_path)

    # Load uncertainty if available
    uncertainty_data = None
    if uncertainty_path and os.path.exists(uncertainty_path):
        print(f"[analyze] Loading uncertainty map: {uncertainty_path}")
        uncertainty_data, _, _ = load_raster(uncertainty_path)

    # Load or create valid mask
    if valid_mask_path and os.path.exists(valid_mask_path):
        print(f"[analyze] Loading valid mask: {valid_mask_path}")
        mask_data, _, _ = load_raster(valid_mask_path)
    else:
        print("[analyze] Creating valid mask from nodata values")
        mask_data = np.ones_like(susc_data, dtype=np.uint8)
        if susc_nodata is not None:
            mask_data[susc_data == susc_nodata] = 0

    # Compute statistics
    print("\n[analyze] Computing spatial statistics...")
    stats = compute_raster_statistics(susc_data, uncertainty_data, mask_data)

    # Print statistics
    print("\n" + "-" * 80)
    print("SUSCEPTIBILITY STATISTICS")
    print("-" * 80)
    if "susceptibility" in stats:
        s = stats["susceptibility"]
        print(f"  Mean: {s['mean']:.4f}")
        print(f"  Std Dev: {s['std']:.4f}")
        print(f"  Min: {s['min']:.4f}")
        print(f"  Max: {s['max']:.4f}")
        print(f"  Median: {s['median']:.4f}")
        print(f"  Q25: {s['q25']:.4f}")
        print(f"  Q75: {s['q75']:.4f}")
        print(f"  Q95: {s['q95']:.4f}")
        print(f"\n  High Risk (>0.7): {s['high_risk_fraction']*100:.2f}%")
        print(f"  Moderate Risk (0.3-0.7): {s['moderate_risk_fraction']*100:.2f}%")
        print(f"  Low Risk (<0.3): {s['low_risk_fraction']*100:.2f}%")

    if "uncertainty" in stats:
        print("\n" + "-" * 80)
        print("UNCERTAINTY STATISTICS")
        print("-" * 80)
        u = stats["uncertainty"]
        print(f"  Mean: {u['mean']:.4f}")
        print(f"  Std Dev: {u['std']:.4f}")
        print(f"  Min: {u['min']:.4f}")
        print(f"  Max: {u['max']:.4f}")
        print(f"  Median: {u['median']:.4f}")
        print(f"  Q25: {u['q25']:.4f}")
        print(f"  Q75: {u['q75']:.4f}")
        print(
            f"  High Uncertainty (>median): {u['high_uncertainty_fraction']*100:.2f}%"
        )

        if "susceptibility_uncertainty_correlation" in stats:
            print(
                f"\n  Correlation with Susceptibility: {stats['susceptibility_uncertainty_correlation']:.4f}"
            )

    print("\n" + "-" * 80)
    print("COVERAGE STATISTICS")
    print("-" * 80)
    c = stats["coverage"]
    print(f"  Total Pixels: {c['total_pixels']:,}")
    print(f"  Valid Pixels: {c['valid_pixels']:,} ({c['valid_fraction']*100:.2f}%)")
    print(f"  Invalid Pixels: {c['invalid_pixels']:,}")

    # Load class map if available
    if class_map_path and os.path.exists(class_map_path):
        print(f"\n[analyze] Loading class map: {class_map_path}")
        class_data, _, class_nodata = load_raster(class_map_path)

        valid_classes = class_data[mask_data > 0]
        unique, counts = np.unique(valid_classes, return_counts=True)

        print("\n" + "-" * 80)
        print("CLASS DISTRIBUTION")
        print("-" * 80)
        stats["class_distribution"] = {}
        for cls, count in zip(unique, counts):
            if class_nodata is not None and cls == class_nodata:
                continue
            fraction = count / len(valid_classes)
            stats["class_distribution"][int(cls)] = {
                "count": int(count),
                "fraction": float(fraction),
            }
            print(f"  Class {cls}: {count:,} ({fraction*100:.2f}%)")

    # Save statistics to JSON
    stats_path = os.path.join(output_dir, "output_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[analyze] Saved statistics to: {stats_path}")

    # Generate analysis report
    report_path = os.path.join(output_dir, "analysis_report.md")
    write_analysis_report(stats, report_path, susceptibility_path)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return stats


def write_evaluation_report(
    metrics: Dict,
    report_path: str,
    susceptibility_path: str,
    ground_truth_path: str,
) -> None:
    """Write a comprehensive evaluation report in Markdown format."""

    with open(report_path, "w") as f:
        f.write("# Landslide Susceptibility Model - Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Ground Truth Encoding\n\n")
        if "ground_truth_encoding" in metrics:
            enc = metrics["ground_truth_encoding"]
            f.write(f"- **Detected encoding:** {enc.get('detected', 'unknown')}\n")
            f.write(f"- **Normalised to:** {enc.get('normalised_to', '{0, 1, 2}')}\n")
            f.write(f"- **Class 0:** {enc.get('class_0', 'Low')}\n")
            f.write(f"- **Class 1:** {enc.get('class_1', 'Medium')}\n")
            f.write(f"- **Class 2:** {enc.get('class_2', 'High')}\n\n")

        f.write("## Input Files\n\n")
        f.write(f"- **Susceptibility Map:** `{susceptibility_path}`\n")
        f.write(f"- **Ground Truth:** `{ground_truth_path}`\n\n")

        if "ground_truth_distribution" in metrics:
            f.write("## Ground Truth Distribution\n\n")
            for cls, info in metrics["ground_truth_distribution"].items():
                f.write(
                    f"- **Class {cls}:** {info['count']:,} pixels ({info['fraction']*100:.2f}%)\n"
                )
            f.write("\n")

        if "ordinal_correlation" in metrics:
            corr = metrics["ordinal_correlation"]
            f.write("## Ordinal Correlation\n\n")
            f.write(f"- **Spearman's ρ:** {corr['spearman_rho']:.4f}\n")
            f.write(f"- **P-value:** {corr['p_value']:.4e}\n\n")

        # Multi-class evaluation
        if "multiclass_evaluation" in metrics and metrics["multiclass_evaluation"]:
            mc = metrics["multiclass_evaluation"]
            f.write("## Multi-Class Evaluation (3×3)\n\n")
            f.write("### Confusion Matrix\n\n")
            names = mc.get("class_names", ["Low", "Medium", "High"])
            cm = mc.get("confusion_matrix", [])
            if cm:
                f.write("| | " + " | ".join(f"Pred {n}" for n in names) + " |\n")
                f.write("|" + "---|" * (len(names) + 1) + "\n")
                for i, name in enumerate(names):
                    row = f"| **True {name}** | " + " | ".join(f"{cm[i][j]:,}" for j in range(len(names))) + " |\n"
                    f.write(row)
                f.write("\n")

            f.write("### Per-Class Metrics\n\n")
            f.write("| Class | Precision | Recall | F1 | IoU | Support |\n")
            f.write("|---|---|---|---|---|---|\n")
            for name in names:
                pc = mc.get("per_class", {}).get(name, {})
                f.write(f"| {name} | {pc.get('precision', 0):.4f} | {pc.get('recall', 0):.4f} | "
                        f"{pc.get('f1', 0):.4f} | {pc.get('iou', 0):.4f} | {pc.get('support', 0):,} |\n")
            f.write("\n")

            f.write("### Aggregate Metrics\n\n")
            f.write(f"- **Macro Precision:** {mc.get('macro_precision', 0):.4f}\n")
            f.write(f"- **Macro Recall:** {mc.get('macro_recall', 0):.4f}\n")
            f.write(f"- **Macro F1:** {mc.get('macro_f1', 0):.4f}\n")
            f.write(f"- **Macro IoU:** {mc.get('macro_iou', 0):.4f}\n")
            f.write(f"- **Accuracy:** {mc.get('accuracy', 0):.4f}\n")
            f.write(f"- **Cohen's Kappa:** {mc.get('cohen_kappa', 0):.4f}\n")
            f.write(f"- **Quadratic Weighted Kappa:** {mc.get('quadratic_weighted_kappa', 0):.4f} *(ordinal-aware)*\n\n")

            if mc.get("predicted_class_breaks"):
                f.write(f"*Predicted class breaks (percentile-matched):* {mc['predicted_class_breaks']}\n\n")

        # Strategy 1: High vs Rest
        if "strategy_1_high_vs_rest" in metrics:
            f.write("## Strategy 1: High Risk (Class 2) vs Rest (Classes 0-1)\n\n")
            write_strategy_section(f, metrics["strategy_1_high_vs_rest"])

        # Strategy 2: Risk vs Low
        if "strategy_2_risk_vs_low" in metrics:
            f.write("## Strategy 2: At-Risk (Classes 1-2) vs Low (Class 0)\n\n")
            write_strategy_section(f, metrics["strategy_2_risk_vs_low"])

        # Strategy 4: Medium vs Low (when Class 3 absent)
        if metrics.get("strategy_4_medium_vs_low"):
            f.write("## Strategy 4: Medium Risk (Class 2) vs Low Risk (Class 1)\n\n")
            f.write(
                "*Note: This strategy evaluates only Classes 0 and 1, excluding Class 2 predictions.*\n\n"
            )
            write_strategy_section(f, metrics["strategy_4_medium_vs_low"])

        f.write("---\n")
        f.write(
            "*Generated by src/evaluate.py (fixed for 3-class ordinal evaluation)*\n"
        )

    print(f"[evaluate] Saved evaluation report to: {report_path}")


def write_strategy_section(f, strategy_metrics: Dict) -> None:
    """Write metrics section for a specific evaluation strategy."""
    if "error" in strategy_metrics:
        f.write(f"**Error:** {strategy_metrics['error']}\n\n")
        return

    if strategy_metrics.get("auroc"):
        f.write(f"- **AUROC:** {strategy_metrics['auroc']:.4f}\n")
    if strategy_metrics.get("auprc"):
        f.write(f"- **AUPRC:** {strategy_metrics['auprc']:.4f}\n")

    f.write(f"\n**At threshold {strategy_metrics.get('threshold', 0.5):.3f}:**\n\n")
    f.write(f"- Accuracy: {strategy_metrics.get('accuracy', 0):.4f}\n")
    f.write(f"- F1 Score: {strategy_metrics.get('f1', 0):.4f}\n")
    f.write(f"- Cohen's Kappa: {strategy_metrics.get('cohen_kappa', 0):.4f}\n")
    f.write(f"- Precision: {strategy_metrics.get('precision', 0):.4f}\n")
    f.write(f"- Recall: {strategy_metrics.get('recall', 0):.4f}\n")
    f.write(f"- Specificity: {strategy_metrics.get('specificity', 0):.4f}\n")
    f.write(f"- Macro IoU: {strategy_metrics.get('macro_iou', 0):.4f}\n\n")

    if "confusion_matrix" in strategy_metrics:
        cm = strategy_metrics["confusion_matrix"]
        f.write("**Confusion Matrix:**\n\n")
        f.write("```\n")
        f.write(f"TN: {cm['true_negative']:,}  FP: {cm['false_positive']:,}\n")
        f.write(f"FN: {cm['false_negative']:,}  TP: {cm['true_positive']:,}\n")
        f.write("```\n\n")

    if "optimal_thresholds" in strategy_metrics:
        opt = strategy_metrics["optimal_thresholds"]
        f.write("**Optimal Thresholds:**\n\n")

        if "youden" in opt:
            y = opt["youden"]
            f.write(f"- **Youden's J:** {y['threshold']:.3f} (J={y['youden_j']:.4f})\n")

        if "f1" in opt:
            f1_opt = opt["f1"]
            f.write(
                f"- **F1-Optimal:** {f1_opt['threshold']:.3f} (F1={f1_opt['f1']:.4f})\n"
            )

        f.write("\n")


def write_analysis_report(
    stats: Dict,
    report_path: str,
    susceptibility_path: str,
) -> None:
    """Write a spatial analysis report in Markdown format."""

    with open(report_path, "w") as f:
        f.write("# Landslide Susceptibility Model - Output Analysis Report\n\n")

        f.write("## Input Files\n\n")
        f.write(f"- **Susceptibility Map:** `{susceptibility_path}`\n\n")

        if "susceptibility" in stats:
            s = stats["susceptibility"]
            f.write("## Susceptibility Statistics\n\n")
            f.write(f"- **Mean:** {s['mean']:.4f}\n")
            f.write(f"- **Std Dev:** {s['std']:.4f}\n")
            f.write(f"- **Min:** {s['min']:.4f}\n")
            f.write(f"- **Max:** {s['max']:.4f}\n")
            f.write(f"- **Median:** {s['median']:.4f}\n")
            f.write(f"- **Q25:** {s['q25']:.4f}\n")
            f.write(f"- **Q75:** {s['q75']:.4f}\n")
            f.write(f"- **Q95:** {s['q95']:.4f}\n\n")

            f.write("### Risk Distribution\n\n")
            f.write(f"- **High Risk (>0.7):** {s['high_risk_fraction']*100:.2f}%\n")
            f.write(
                f"- **Moderate Risk (0.3-0.7):** {s['moderate_risk_fraction']*100:.2f}%\n"
            )
            f.write(f"- **Low Risk (<0.3):** {s['low_risk_fraction']*100:.2f}%\n\n")

        if "uncertainty" in stats:
            u = stats["uncertainty"]
            f.write("## Uncertainty Statistics\n\n")
            f.write(f"- **Mean:** {u['mean']:.4f}\n")
            f.write(f"- **Std Dev:** {u['std']:.4f}\n")
            f.write(f"- **Median:** {u['median']:.4f}\n")
            f.write(
                f"- **High Uncertainty (>median):** {u['high_uncertainty_fraction']*100:.2f}%\n\n"
            )

            if "susceptibility_uncertainty_correlation" in stats:
                f.write(
                    f"**Correlation with Susceptibility:** {stats['susceptibility_uncertainty_correlation']:.4f}\n\n"
                )

        if "coverage" in stats:
            c = stats["coverage"]
            f.write("## Coverage Statistics\n\n")
            f.write(f"- **Total Pixels:** {c['total_pixels']:,}\n")
            f.write(
                f"- **Valid Pixels:** {c['valid_pixels']:,} ({c['valid_fraction']*100:.2f}%)\n"
            )
            f.write(f"- **Invalid Pixels:** {c['invalid_pixels']:,}\n\n")

        if "class_distribution" in stats:
            f.write("## Class Distribution\n\n")
            for cls, info in stats["class_distribution"].items():
                f.write(
                    f"- **Class {cls}:** {info['count']:,} ({info['fraction']*100:.2f}%)\n"
                )
            f.write("\n")

        f.write("---\n")
        f.write("*Generated by src/evaluate.py*\n")

    print(f"[analyze] Saved analysis report to: {report_path}")


def main():
    """Main evaluation function with CLI argument parsing."""

    parser = argparse.ArgumentParser(
        description="Evaluate landslide susceptibility model outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate outputs with ground truth
  python -m src.evaluate --susceptibility outputs/test_susceptibility.tif \\
                          --ground_truth path/to/ground_truth.tif \\
                          --output_dir outputs/evaluation
  
  # Analyze outputs without ground truth
  python -m src.evaluate --susceptibility outputs/test_susceptibility.tif \\
                          --uncertainty outputs/test_uncertainty.tif \\
                          --analysis_only
  
  # Specify custom threshold
  python -m src.evaluate --susceptibility outputs/test_susceptibility.tif \\
                          --ground_truth path/to/ground_truth.tif \\
                          --threshold 0.6
        """,
    )

    parser.add_argument(
        "--susceptibility",
        type=str,
        default="outputs/test_susceptibility.tif",
        help="Path to susceptibility GeoTIFF (probabilities)",
    )

    parser.add_argument(
        "--ground_truth",
        type=str,
        default=None,
        help="Path to ground truth GeoTIFF (required for evaluation metrics)",
    )

    parser.add_argument(
        "--uncertainty",
        type=str,
        default="outputs/test_uncertainty.tif",
        help="Path to uncertainty GeoTIFF (optional)",
    )

    parser.add_argument(
        "--valid_mask",
        type=str,
        default="outputs/test_valid_mask.tif",
        help="Path to valid mask GeoTIFF (optional)",
    )

    parser.add_argument(
        "--class_map",
        type=str,
        default="outputs/test_class_map.tif",
        help="Path to class map GeoTIFF (optional)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,  # Changed from 0.5
        help="Classification threshold (default: auto-load from checkpoint or 0.5)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/experiments/run_*/best_model.pth",
        help="Path to model checkpoint (to load optimal threshold)",
    )

    parser.add_argument(
        "--analysis_only",
        action="store_true",
        help="Only perform spatial analysis (no ground truth needed)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.susceptibility):
        print(f"ERROR: Susceptibility file not found: {args.susceptibility}")
        sys.exit(1)

    if args.analysis_only or args.ground_truth is None:
        # Analysis mode (no ground truth)
        print("[evaluate] Running in ANALYSIS mode (no ground truth)")
        results = analyze_outputs_only(
            susceptibility_path=args.susceptibility,
            uncertainty_path=(
                args.uncertainty if os.path.exists(args.uncertainty) else None
            ),
            valid_mask_path=(
                args.valid_mask if os.path.exists(args.valid_mask) else None
            ),
            class_map_path=args.class_map if os.path.exists(args.class_map) else None,
            output_dir=args.output_dir,
        )
    else:
        # Evaluation mode (with ground truth)
        if not os.path.exists(args.ground_truth):
            print(f"ERROR: Ground truth file not found: {args.ground_truth}")
            sys.exit(1)

        print("[evaluate] Running in EVALUATION mode (with ground truth)")

        # Use default threshold if not specified
        threshold = args.threshold if args.threshold is not None else 0.5

        results = evaluate_with_ground_truth(
            susceptibility_path=args.susceptibility,
            ground_truth_path=args.ground_truth,
            valid_mask_path=(
                args.valid_mask if os.path.exists(args.valid_mask) else None
            ),
            output_dir=args.output_dir,
            threshold=threshold,
        )

    print(f"\n[evaluate] Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
