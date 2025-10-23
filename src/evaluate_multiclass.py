"""
Fixed evaluation script for 3-class landslide susceptibility model.

This script properly handles the 3-class ground truth:
- Class 1: Low landslide probability
- Class 2: Medium landslide probability
- Class 3: High landslide probability

It provides two evaluation modes:
1. Multi-class evaluation (all 3 classes)
2. Binary evaluation (treating class 3 as positive, classes 1-2 as negative)
3. Ordinal evaluation (weighted by severity)
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    cohen_kappa_score,
)

from src.metrics import (
    find_optimal_threshold_f1,
    find_optimal_threshold_youden,
)
from src.visualize import plot_roc_curve, plot_pr_curve


def load_raster(path: str) -> Tuple[np.ndarray, dict, Optional[float]]:
    """Load a GeoTIFF raster and return data + metadata."""
    with rasterio.open(path) as src:
        data = src.read(1)  # Read first band
        meta = src.meta.copy()
        nodata = src.nodata
    return data, meta, nodata


def evaluate_multiclass(
    susceptibility_path: str,
    ground_truth_path: str,
    valid_mask_path: Optional[str] = None,
    output_dir: str = "outputs/evaluation",
    mode: str = "ordinal",  # 'binary', 'multiclass', or 'ordinal'
) -> Dict:
    """
    Evaluate 3-class landslide model with proper handling.

    Args:
        susceptibility_path: Path to susceptibility GeoTIFF (probability of class 3)
        ground_truth_path: Path to ground truth with values 1, 2, 3
        valid_mask_path: Path to valid mask (optional)
        output_dir: Output directory
        mode: Evaluation mode - 'binary', 'multiclass', or 'ordinal'

    Returns:
        Dictionary of metrics
    """
    print("\n" + "=" * 80)
    print("EVALUATING 3-CLASS LANDSLIDE SUSCEPTIBILITY MODEL")
    print(f"Mode: {mode.upper()}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Load rasters
    print(f"\n[evaluate] Loading susceptibility map: {susceptibility_path}")
    susc_data, susc_meta, susc_nodata = load_raster(susceptibility_path)

    print(f"[evaluate] Loading ground truth: {ground_truth_path}")
    gt_data, gt_meta, gt_nodata = load_raster(ground_truth_path)

    # Resample if needed
    if susc_data.shape != gt_data.shape:
        print(
            f"[evaluate] Resampling ground truth from {gt_data.shape} to {susc_data.shape}"
        )
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

    # Extract valid pixels
    valid_pixels = mask_data > 0
    y_probs = susc_data[valid_pixels].flatten()
    y_true_raw = gt_data[valid_pixels].flatten()

    # Ground truth statistics
    print(f"\n[evaluate] Ground truth class distribution:")
    unique_classes, counts = np.unique(y_true_raw, return_counts=True)
    for cls, count in zip(unique_classes, counts):
        pct = 100 * count / len(y_true_raw)
        print(f"  Class {int(cls)}: {count:,} ({pct:.2f}%)")

    # Susceptibility statistics
    print(f"\n[evaluate] Susceptibility statistics:")
    print(f"  Min: {y_probs.min():.4f}")
    print(f"  Max: {y_probs.max():.4f}")
    print(f"  Mean: {y_probs.mean():.4f}")
    print(f"  Median: {np.median(y_probs):.4f}")

    metrics = {}

    if mode == "binary":
        # Binary: Class 3 (high) vs Classes 1-2 (low+medium)
        print("\n[evaluate] Binary evaluation: Class 3 (HIGH) as positive")
        y_true_binary = (y_true_raw == 3).astype(int)

        print(
            f"  Positive (Class 3): {np.sum(y_true_binary):,} ({100*np.mean(y_true_binary):.2f}%)"
        )
        print(
            f"  Negative (Classes 1-2): {np.sum(1-y_true_binary):,} ({100*np.mean(1-y_true_binary):.2f}%)"
        )

        if len(np.unique(y_true_binary)) < 2:
            print("[evaluate] ERROR: Only one class present!")
            return {"error": "Only one class in ground truth"}

        # Compute binary metrics
        metrics = compute_binary_metrics(y_true_binary, y_probs, output_dir)

    elif mode == "ordinal":
        # Ordinal: Weight classes by severity (1=low, 2=med, 3=high)
        print("\n[evaluate] Ordinal evaluation: Weighted by severity")

        # Option 1: Evaluate treating 2 or 3 as positive
        y_true_binary_23 = (y_true_raw >= 2).astype(int)
        print(f"  Treating Classes 2-3 as positive (medium+high risk)")
        print(
            f"    Positive: {np.sum(y_true_binary_23):,} ({100*np.mean(y_true_binary_23):.2f}%)"
        )
        print(
            f"    Negative: {np.sum(1-y_true_binary_23):,} ({100*np.mean(1-y_true_binary_23):.2f}%)"
        )

        metrics["binary_med_high_vs_low"] = compute_binary_metrics(
            y_true_binary_23, y_probs, output_dir, suffix="_med_high"
        )

        # Option 2: Evaluate treating only 3 as positive
        y_true_binary_3 = (y_true_raw == 3).astype(int)
        print(f"\n  Treating Class 3 as positive (high risk only)")
        print(
            f"    Positive: {np.sum(y_true_binary_3):,} ({100*np.mean(y_true_binary_3):.2f}%)"
        )
        print(
            f"    Negative: {np.sum(1-y_true_binary_3):,} ({100*np.mean(1-y_true_binary_3):.2f}%)"
        )

        metrics["binary_high_only"] = compute_binary_metrics(
            y_true_binary_3, y_probs, output_dir, suffix="_high_only"
        )

        # Ordinal correlation
        from scipy.stats import spearmanr

        corr, pval = spearmanr(y_true_raw, y_probs)
        metrics["ordinal_correlation"] = {
            "spearman_r": float(corr),
            "p_value": float(pval),
        }
        print(f"\n  Spearman correlation: {corr:.4f} (p={pval:.4e})")

    elif mode == "multiclass":
        # Multi-class: This doesn't make sense with single probability output
        print(
            "\n[evaluate] WARNING: Multi-class evaluation requested but model outputs single probability!"
        )
        print(
            "[evaluate] This susceptibility map is P(class=3), not full 3-class probabilities."
        )
        print("[evaluate] Falling back to binary evaluation (class 3 vs rest)")

        y_true_binary = (y_true_raw == 3).astype(int)
        metrics = compute_binary_metrics(y_true_binary, y_probs, output_dir)

    # Save metrics
    metrics_path = os.path.join(output_dir, "evaluation_metrics_fixed.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[evaluate] Saved metrics to: {metrics_path}")

    # Generate report
    report_path = os.path.join(output_dir, "evaluation_report_fixed.md")
    write_report(metrics, report_path, susceptibility_path, ground_truth_path, mode)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return metrics


def compute_binary_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_dir: str,
    threshold: float = 0.5,
    suffix: str = "",
) -> Dict:
    """Compute binary classification metrics."""

    metrics = {}

    # Threshold-independent
    try:
        auroc = roc_auc_score(y_true, y_probs)
        metrics["auroc"] = float(auroc)
        print(f"    AUROC: {auroc:.4f}")
    except Exception as e:
        print(f"    AUROC: ERROR - {e}")
        metrics["auroc"] = None

    try:
        auprc = average_precision_score(y_true, y_probs)
        metrics["auprc"] = float(auprc)
        print(f"    AUPRC: {auprc:.4f}")
    except Exception as e:
        print(f"    AUPRC: ERROR - {e}")
        metrics["auprc"] = None

    # Threshold-dependent
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    kappa = cohen_kappa_score(y_true, y_pred)

    metrics.update(
        {
            "threshold": float(threshold),
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "iou": float(iou),
            "kappa": float(kappa),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }
    )

    print(f"\n    Metrics at threshold {threshold:.3f}:")
    print(f"      Accuracy: {acc:.4f}")
    print(f"      F1: {f1:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall: {recall:.4f}")
    print(f"      Specificity: {specificity:.4f}")
    print(f"      IoU: {iou:.4f}")
    print(f"      Cohen's Kappa: {kappa:.4f}")

    # Find optimal thresholds
    threshold_youden, youden_metrics = find_optimal_threshold_youden(y_true, y_probs)
    threshold_f1, f1_metrics = find_optimal_threshold_f1(y_true, y_probs)

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

    print(f"\n    Optimal thresholds:")
    print(
        f"      Youden's J: {threshold_youden:.3f} (J={youden_metrics['youden_j']:.4f})"
    )
    print(f"      F1-optimal: {threshold_f1:.3f} (F1={f1_metrics['f1']:.4f})")

    # Generate plots
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    roc_path = os.path.join(figures_dir, f"roc_curve{suffix}.png")
    plot_roc_curve(None, None, y_probs, y_true, roc_path, title=f"ROC Curve{suffix}")
    metrics["plot_roc"] = roc_path

    pr_path = os.path.join(figures_dir, f"pr_curve{suffix}.png")
    plot_pr_curve(None, None, y_probs, y_true, pr_path, title=f"PR Curve{suffix}")
    metrics["plot_pr"] = pr_path

    return metrics


def write_report(
    metrics: Dict,
    report_path: str,
    susceptibility_path: str,
    ground_truth_path: str,
    mode: str,
) -> None:
    """Write evaluation report."""

    with open(report_path, "w") as f:
        f.write("# Landslide Susceptibility - Fixed Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Evaluation Mode:** {mode}\n\n")

        f.write("## Ground Truth Encoding\n\n")
        f.write("- **Class 1:** Low landslide probability\n")
        f.write("- **Class 2:** Medium landslide probability\n")
        f.write("- **Class 3:** High landslide probability\n\n")

        f.write("## Input Files\n\n")
        f.write(f"- **Susceptibility:** `{susceptibility_path}`\n")
        f.write(f"- **Ground Truth:** `{ground_truth_path}`\n\n")

        if mode == "ordinal":
            f.write("## Ordinal Evaluation Results\n\n")

            if "ordinal_correlation" in metrics:
                corr = metrics["ordinal_correlation"]
                f.write(f"### Spearman Rank Correlation\n\n")
                f.write(f"- **Correlation:** {corr['spearman_r']:.4f}\n")
                f.write(f"- **P-value:** {corr['p_value']:.4e}\n\n")

            if "binary_med_high_vs_low" in metrics:
                f.write("### Binary: Medium+High (2-3) vs Low (1)\n\n")
                write_binary_section(f, metrics["binary_med_high_vs_low"])

            if "binary_high_only" in metrics:
                f.write("### Binary: High (3) vs Medium+Low (1-2)\n\n")
                write_binary_section(f, metrics["binary_high_only"])

        else:
            write_binary_section(f, metrics)

        f.write("\n---\n")
        f.write("*Generated by src/evaluate_multiclass.py*\n")

    print(f"[evaluate] Saved report to: {report_path}")


def write_binary_section(f, metrics: Dict) -> None:
    """Write binary metrics section."""
    if metrics.get("auroc"):
        f.write(f"- **AUROC:** {metrics['auroc']:.4f}\n")
    if metrics.get("auprc"):
        f.write(f"- **AUPRC:** {metrics['auprc']:.4f}\n")

    f.write(f"\n**At threshold {metrics.get('threshold', 0.5):.3f}:**\n\n")
    f.write(f"- Accuracy: {metrics.get('accuracy', 0):.4f}\n")
    f.write(f"- F1 Score: {metrics.get('f1', 0):.4f}\n")
    f.write(f"- Precision: {metrics.get('precision', 0):.4f}\n")
    f.write(f"- Recall: {metrics.get('recall', 0):.4f}\n")
    f.write(f"- IoU: {metrics.get('iou', 0):.4f}\n")
    f.write(f"- Cohen's Kappa: {metrics.get('kappa', 0):.4f}\n\n")

    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        f.write("**Confusion Matrix:**\n\n")
        f.write("```\n")
        f.write(f"TN: {cm['tn']:,}  FP: {cm['fp']:,}\n")
        f.write(f"FN: {cm['fn']:,}  TP: {cm['tp']:,}\n")
        f.write("```\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fixed evaluation for 3-class landslide model"
    )

    parser.add_argument(
        "--susceptibility",
        type=str,
        default="outputs/test_susceptibility.tif",
        help="Susceptibility map (probability of high class)",
    )

    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Ground truth with values 1, 2, 3",
    )

    parser.add_argument(
        "--valid_mask",
        type=str,
        default="outputs/test_valid_mask.tif",
        help="Valid mask (optional)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="ordinal",
        choices=["binary", "multiclass", "ordinal"],
        help="Evaluation mode",
    )

    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        print(f"ERROR: Ground truth not found: {args.ground_truth}")
        sys.exit(1)

    evaluate_multiclass(
        susceptibility_path=args.susceptibility,
        ground_truth_path=args.ground_truth,
        valid_mask_path=args.valid_mask if os.path.exists(args.valid_mask) else None,
        output_dir=args.output_dir,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
