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
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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
    print(f"    Accuracy: {acc:.4f}, F1: {f1:.4f}")
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

    # CRITICAL FIX: Ground truth has values 1, 2, 3 (low, medium, high)
    # These were remapped to 0, 1, 2 during training
    # For ordinal evaluation, we have multiple strategies:

    print(f"\n[evaluate] Ground truth value distribution:")
    unique_gt, counts_gt = np.unique(y_true, return_counts=True)
    for val, count in zip(unique_gt, counts_gt):
        print(f"  Value {int(val)}: {count:,} pixels ({100*count/len(y_true):.2f}%)")

    print(f"\n[evaluate] Susceptibility statistics:")
    print(f"  Min: {y_probs.min():.4f}, Max: {y_probs.max():.4f}")
    print(f"  Mean: {y_probs.mean():.4f}, Median: {np.median(y_probs):.4f}")

    # Strategy 1: Binary - High risk (class 3) vs rest (classes 1-2)
    print(f"\n[evaluate] STRATEGY 1: Binary evaluation (High vs Low+Medium)")
    y_true_binary_high = (y_true == 3).astype(int)

    # Strategy 2: Binary - Medium+High risk (classes 2-3) vs Low (class 1)
    print(f"[evaluate] STRATEGY 2: Binary evaluation (Medium+High vs Low)")
    y_true_binary_risk = (y_true >= 2).astype(int)

    # Strategy 3: Ordinal - Treat as ordered categories and compute correlation
    print(f"[evaluate] STRATEGY 3: Ordinal correlation")
    from scipy.stats import spearmanr

    corr, pval = spearmanr(y_true, y_probs)
    print(f"  Spearman's rho: {corr:.4f} (p={pval:.4e})")

    # Evaluate Strategy 1: High vs rest
    print(f"\n[evaluate] Computing metrics for Strategy 1 (High risk vs rest)...")
    print(
        f"  Positive (Class 3): {np.sum(y_true_binary_high):,} ({100*np.mean(y_true_binary_high):.2f}%)"
    )
    metrics_high = evaluate_binary_strategy(
        y_true_binary_high, y_probs, threshold, output_dir, "high_vs_rest"
    )

    # Evaluate Strategy 2: Medium+High vs Low
    print(f"\n[evaluate] Computing metrics for Strategy 2 (Medium+High vs Low)...")
    print(
        f"  Positive (Classes 2-3): {np.sum(y_true_binary_risk):,} ({100*np.mean(y_true_binary_risk):.2f}%)"
    )
    metrics_risk = evaluate_binary_strategy(
        y_true_binary_risk, y_probs, threshold, output_dir, "risk_vs_low"
    )

    # Strategy 4: Medium vs Low (when Class 3 is absent or rare)
    metrics_medium = None
    if 3 not in unique_gt or np.sum(y_true == 3) < 100:
        print(f"\n[evaluate] STRATEGY 4: Binary evaluation (Medium vs Low)")
        print(f"  Note: Class 3 absent or rare, evaluating Classes 1 vs 2 only")

        # Filter to only Class 1 and Class 2
        mask_12 = (y_true == 1) | (y_true == 2)
        y_true_12 = y_true[mask_12]
        y_probs_12 = y_probs[mask_12]

        y_true_binary_medium = (y_true_12 == 2).astype(int)

        print(
            f"  Positive (Class 2): {np.sum(y_true_binary_medium):,} ({100*np.mean(y_true_binary_medium):.2f}%)"
        )
        print(
            f"  Negative (Class 1): {np.sum(y_true_binary_medium == 0):,} ({100*np.mean(y_true_binary_medium == 0):.2f}%)"
        )

        metrics_medium = evaluate_binary_strategy(
            y_true_binary_medium, y_probs_12, threshold, output_dir, "medium_vs_low"
        )

    # Combine all metrics
    metrics = {
        "ground_truth_encoding": {
            "class_1": "Low landslide probability",
            "class_2": "Medium landslide probability",
            "class_3": "High landslide probability",
        },
        "ground_truth_distribution": {
            int(val): {"count": int(count), "fraction": float(count / len(y_true))}
            for val, count in zip(unique_gt, counts_gt)
        },
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
            f.write(f"- **Class 1:** {enc['class_1']}\n")
            f.write(f"- **Class 2:** {enc['class_2']}\n")
            f.write(f"- **Class 3:** {enc['class_3']}\n\n")

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

        # Strategy 1: High vs Rest
        if "strategy_1_high_vs_rest" in metrics:
            f.write("## Strategy 1: High Risk (Class 3) vs Rest (Classes 1-2)\n\n")
            write_strategy_section(f, metrics["strategy_1_high_vs_rest"])

        # Strategy 2: Risk vs Low
        if "strategy_2_risk_vs_low" in metrics:
            f.write("## Strategy 2: At-Risk (Classes 2-3) vs Low (Class 1)\n\n")
            write_strategy_section(f, metrics["strategy_2_risk_vs_low"])

        # Strategy 4: Medium vs Low (when Class 3 absent)
        if metrics.get("strategy_4_medium_vs_low"):
            f.write("## Strategy 4: Medium Risk (Class 2) vs Low Risk (Class 1)\n\n")
            f.write(
                "*Note: This strategy evaluates only Classes 1 and 2, excluding Class 3 predictions.*\n\n"
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
