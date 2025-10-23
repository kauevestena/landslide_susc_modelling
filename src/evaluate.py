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


def load_raster(path: str) -> Tuple[np.ndarray, dict]:
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

    # Convert ground truth to binary (assuming class 2 is positive)
    # Adjust if your ground truth has different encoding
    y_true_binary = (y_true == 2).astype(int)

    print(f"\n[evaluate] Valid pixels: {len(y_true):,}")
    print(
        f"[evaluate] Positive samples: {np.sum(y_true_binary):,} ({100*np.mean(y_true_binary):.2f}%)"
    )
    print(
        f"[evaluate] Negative samples: {np.sum(1-y_true_binary):,} ({100*np.mean(1-y_true_binary):.2f}%)"
    )

    # Check if we have both classes
    if len(np.unique(y_true_binary)) < 2:
        print("[evaluate] WARNING: Only one class present in ground truth!")
        print("[evaluate] Cannot compute meaningful metrics.")
        return {"error": "Only one class present in ground truth"}

    # Compute metrics
    print("\n[evaluate] Computing performance metrics...")
    metrics = {}

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

    # Threshold-dependent metrics at specified threshold
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

    metrics["threshold"] = float(threshold)
    metrics["accuracy"] = float(acc)
    metrics["f1"] = float(f1)
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["specificity"] = float(specificity)
    metrics["iou_positive"] = float(iou_pos)
    metrics["iou_negative"] = float(iou_neg)
    metrics["macro_iou"] = float(macro_iou)

    print(f"\n  Threshold: {threshold:.3f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  IoU (Positive): {iou_pos:.4f}")
    print(f"  IoU (Negative): {iou_neg:.4f}")
    print(f"  Macro IoU: {macro_iou:.4f}")

    # Confusion matrix
    metrics["confusion_matrix"] = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    print(f"\n  Confusion Matrix:")
    print(f"    TN: {tn:,}  FP: {fp:,}")
    print(f"    FN: {fn:,}  TP: {tp:,}")

    # Find optimal thresholds
    print("\n[evaluate] Computing optimal thresholds...")

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

    print(f"  Youden's J Optimal Threshold: {threshold_youden:.3f}")
    print(f"    Sensitivity: {youden_metrics['sensitivity']:.4f}")
    print(f"    Specificity: {youden_metrics['specificity']:.4f}")
    print(f"  F1-Optimal Threshold: {threshold_f1:.3f}")
    print(f"    Precision: {f1_metrics['precision']:.4f}")
    print(f"    Recall: {f1_metrics['recall']:.4f}")
    print(f"    F1: {f1_metrics['f1']:.4f}")

    # Generate visualizations
    print("\n[evaluate] Generating visualization plots...")

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_paths = {}

    # ROC curve
    roc_path = os.path.join(figures_dir, "roc_curve.png")
    plot_roc_curve(
        None,
        None,
        y_probs,
        y_true_binary,
        roc_path,
        title="ROC Curve - Inference Evaluation",
    )
    plot_paths["roc_curve"] = roc_path

    # PR curve
    pr_path = os.path.join(figures_dir, "pr_curve.png")
    plot_pr_curve(
        None,
        None,
        y_probs,
        y_true_binary,
        pr_path,
        title="Precision-Recall Curve - Inference Evaluation",
    )
    plot_paths["pr_curve"] = pr_path

    metrics["plot_paths"] = plot_paths

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

        f.write("## Input Files\n\n")
        f.write(f"- **Susceptibility Map:** `{susceptibility_path}`\n")
        f.write(f"- **Ground Truth:** `{ground_truth_path}`\n\n")

        f.write("## Performance Summary\n\n")

        # Threshold-independent metrics
        if metrics.get("auroc") is not None:
            f.write(f"### Threshold-Independent Metrics\n\n")
            f.write(f"- **AUROC:** {metrics['auroc']:.4f}\n")
            if metrics.get("auprc") is not None:
                f.write(f"- **AUPRC:** {metrics['auprc']:.4f}\n")
            f.write("\n")

        # Threshold-dependent metrics
        f.write(f"### Metrics at Threshold = {metrics['threshold']:.3f}\n\n")
        f.write(f"- **Accuracy:** {metrics['accuracy']:.4f}\n")
        f.write(f"- **F1 Score:** {metrics['f1']:.4f}\n")
        f.write(f"- **Precision:** {metrics['precision']:.4f}\n")
        f.write(f"- **Recall (Sensitivity):** {metrics['recall']:.4f}\n")
        f.write(f"- **Specificity:** {metrics['specificity']:.4f}\n")
        f.write(f"- **Macro IoU:** {metrics['macro_iou']:.4f}\n\n")

        # Confusion matrix
        cm = metrics["confusion_matrix"]
        f.write("### Confusion Matrix\n\n")
        f.write("```\n")
        f.write("                 Predicted\n")
        f.write("               Neg      Pos\n")
        f.write(
            f"Actual  Neg   {cm['true_negative']:>8,}  {cm['false_positive']:>8,}\n"
        )
        f.write(
            f"        Pos   {cm['false_negative']:>8,}  {cm['true_positive']:>8,}\n"
        )
        f.write("```\n\n")

        # Optimal thresholds
        if "optimal_thresholds" in metrics:
            f.write("## Optimal Threshold Recommendations\n\n")

            youden = metrics["optimal_thresholds"]["youden"]
            f.write(f"### Youden's J Method\n\n")
            f.write(f"- **Optimal Threshold:** {youden['threshold']:.3f}\n")
            f.write(f"- **Sensitivity:** {youden['sensitivity']:.4f}\n")
            f.write(f"- **Specificity:** {youden['specificity']:.4f}\n")
            f.write(f"- **Youden's J:** {youden['youden_j']:.4f}\n\n")

            f1_opt = metrics["optimal_thresholds"]["f1"]
            f.write(f"### F1-Maximizing Method\n\n")
            f.write(f"- **Optimal Threshold:** {f1_opt['threshold']:.3f}\n")
            f.write(f"- **Precision:** {f1_opt['precision']:.4f}\n")
            f.write(f"- **Recall:** {f1_opt['recall']:.4f}\n")
            f.write(f"- **F1 Score:** {f1_opt['f1']:.4f}\n\n")

        # Visualizations
        if "plot_paths" in metrics:
            f.write("## Visualizations\n\n")
            for plot_name, plot_path in metrics["plot_paths"].items():
                f.write(f"- **{plot_name.replace('_', ' ').title()}:** `{plot_path}`\n")
            f.write("\n")

        f.write("---\n")
        f.write("*Generated by src/evaluate.py*\n")

    print(f"[evaluate] Saved evaluation report to: {report_path}")


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
