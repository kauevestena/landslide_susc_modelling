"""Visualization utilities for model performance analysis."""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve


def plot_roc_curve(
    val_probs: Optional[np.ndarray],
    val_labels: Optional[np.ndarray],
    test_probs: Optional[np.ndarray],
    test_labels: Optional[np.ndarray],
    save_path: str,
    title: str = "ROC Curve - Landslide Detection",
) -> None:
    """
    Plot ROC curve comparing validation and test set performance.

    Args:
        val_probs: Validation set predicted probabilities
        val_labels: Validation set binary labels
        test_probs: Test set predicted probabilities
        test_labels: Test set binary labels
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(8, 8))

    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

    # Plot validation ROC if available
    if val_probs is not None and val_labels is not None and len(val_labels) > 0:
        if len(np.unique(val_labels)) > 1:
            fpr_val, tpr_val, _ = roc_curve(val_labels, val_probs)
            roc_auc_val = auc(fpr_val, tpr_val)
            plt.plot(
                fpr_val,
                tpr_val,
                linewidth=2,
                label=f"Validation (AUC = {roc_auc_val:.4f})",
                color="blue",
            )

    # Plot test ROC if available
    if test_probs is not None and test_labels is not None and len(test_labels) > 0:
        if len(np.unique(test_labels)) > 1:
            fpr_test, tpr_test, _ = roc_curve(test_labels, test_probs)
            roc_auc_test = auc(fpr_test, tpr_test)
            plt.plot(
                fpr_test,
                tpr_test,
                linewidth=2,
                label=f"Test (AUC = {roc_auc_test:.4f})",
                color="red",
            )

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Saved ROC curve to {save_path}")


def plot_pr_curve(
    val_probs: Optional[np.ndarray],
    val_labels: Optional[np.ndarray],
    test_probs: Optional[np.ndarray],
    test_labels: Optional[np.ndarray],
    save_path: str,
    title: str = "Precision-Recall Curve - Landslide Detection",
) -> None:
    """
    Plot Precision-Recall curve comparing validation and test set performance.

    Args:
        val_probs: Validation set predicted probabilities
        val_labels: Validation set binary labels
        test_probs: Test set predicted probabilities
        test_labels: Test set binary labels
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(8, 8))

    # Plot validation PR curve if available
    if val_probs is not None and val_labels is not None and len(val_labels) > 0:
        if len(np.unique(val_labels)) > 1:
            precision_val, recall_val, _ = precision_recall_curve(val_labels, val_probs)
            ap_val = average_precision_score(val_labels, val_probs)
            plt.plot(
                recall_val,
                precision_val,
                linewidth=2,
                label=f"Validation (AP = {ap_val:.4f})",
                color="blue",
            )

            # Baseline: prevalence of positive class
            baseline_val = np.mean(val_labels)
            plt.axhline(
                y=baseline_val,
                color="blue",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label=f"Val Baseline (Prevalence = {baseline_val:.4f})",
            )

    # Plot test PR curve if available
    if test_probs is not None and test_labels is not None and len(test_labels) > 0:
        if len(np.unique(test_labels)) > 1:
            precision_test, recall_test, _ = precision_recall_curve(
                test_labels, test_probs
            )
            ap_test = average_precision_score(test_labels, test_probs)
            plt.plot(
                recall_test,
                precision_test,
                linewidth=2,
                label=f"Test (AP = {ap_test:.4f})",
                color="red",
            )

            # Baseline: prevalence of positive class
            baseline_test = np.mean(test_labels)
            plt.axhline(
                y=baseline_test,
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label=f"Test Baseline (Prevalence = {baseline_test:.4f})",
            )

    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Saved PR curve to {save_path}")


def plot_calibration_curve(
    val_probs: Optional[np.ndarray],
    val_labels: Optional[np.ndarray],
    test_probs: Optional[np.ndarray],
    test_labels: Optional[np.ndarray],
    save_path: str,
    n_bins: int = 10,
    title: str = "Calibration Curve - Landslide Detection",
) -> None:
    """
    Plot calibration curve showing how well predicted probabilities match actual frequencies.

    Args:
        val_probs: Validation set predicted probabilities
        val_labels: Validation set binary labels
        test_probs: Test set predicted probabilities
        test_labels: Test set binary labels
        save_path: Path to save figure
        n_bins: Number of bins for calibration curve
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly Calibrated")

    # Plot validation calibration if available
    if val_probs is not None and val_labels is not None and len(val_labels) > 0:
        if len(np.unique(val_labels)) > 1:
            try:
                frac_pos_val, mean_pred_val = calibration_curve(
                    val_labels, val_probs, n_bins=n_bins, strategy="uniform"
                )
                ax1.plot(
                    mean_pred_val,
                    frac_pos_val,
                    marker="o",
                    linewidth=2,
                    label="Validation",
                    color="blue",
                )
            except (ValueError, RuntimeError) as e:
                print(f"[visualize] Could not compute validation calibration: {e}")

    # Plot test calibration if available
    if test_probs is not None and test_labels is not None and len(test_labels) > 0:
        if len(np.unique(test_labels)) > 1:
            try:
                frac_pos_test, mean_pred_test = calibration_curve(
                    test_labels, test_probs, n_bins=n_bins, strategy="uniform"
                )
                ax1.plot(
                    mean_pred_test,
                    frac_pos_test,
                    marker="s",
                    linewidth=2,
                    label="Test",
                    color="red",
                )
            except (ValueError, RuntimeError) as e:
                print(f"[visualize] Could not compute test calibration: {e}")

    ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax1.set_ylabel("Fraction of Positives", fontsize=12)
    ax1.set_title("Calibration Curve", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])

    # Plot 2: Distribution of predicted probabilities
    bins = np.linspace(0, 1, 50)

    if val_probs is not None and len(val_probs) > 0:
        ax2.hist(
            val_probs,
            bins=bins,
            alpha=0.5,
            label="Validation",
            color="blue",
            edgecolor="black",
        )

    if test_probs is not None and len(test_probs) > 0:
        ax2.hist(
            test_probs,
            bins=bins,
            alpha=0.5,
            label="Test",
            color="red",
            edgecolor="black",
        )

    ax2.set_xlabel("Predicted Probability", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Distribution of Predictions", fontsize=14, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Saved calibration curve to {save_path}")


def plot_training_history(
    history: list,
    save_path: str,
    title: str = "Training History",
) -> None:
    """
    Plot training and validation loss/metrics over epochs.

    Args:
        history: List of dicts containing epoch metrics
        save_path: Path to save figure
        title: Plot title
    """
    if not history:
        print("[visualize] No training history available to plot")
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]

    # Extract validation metrics
    val_metrics_keys = list(history[0].get("val_metrics", {}).keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Loss
    axes[0, 0].plot(epochs, train_loss, "b-o", label="Train Loss", linewidth=2)
    axes[0, 0].plot(epochs, val_loss, "r-s", label="Val Loss", linewidth=2)
    axes[0, 0].set_xlabel("Epoch", fontsize=12)
    axes[0, 0].set_ylabel("Loss", fontsize=12)
    axes[0, 0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0, 0].legend(loc="best", fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot remaining validation metrics
    metric_names = ["macro_iou", "macro_f1", "auroc", "auprc"]
    for idx, metric_name in enumerate(
        metric_names[:3]
    ):  # Plot first 3 additional metrics
        if metric_name in val_metrics_keys:
            row = (idx + 1) // 2
            col = (idx + 1) % 2
            values = [
                entry["val_metrics"].get(metric_name, float("nan")) for entry in history
            ]
            axes[row, col].plot(epochs, values, "g-d", linewidth=2)
            axes[row, col].set_xlabel("Epoch", fontsize=12)
            axes[row, col].set_ylabel(
                metric_name.upper().replace("_", " "), fontsize=12
            )
            axes[row, col].set_title(
                f"Validation {metric_name.upper()}", fontsize=14, fontweight="bold"
            )
            axes[row, col].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Saved training history to {save_path}")


def generate_all_plots(
    val_probs: Optional[np.ndarray],
    val_labels: Optional[np.ndarray],
    test_probs: Optional[np.ndarray],
    test_labels: Optional[np.ndarray],
    history: list,
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate all visualization plots for model evaluation.

    Args:
        val_probs: Validation set predicted probabilities
        val_labels: Validation set binary labels
        test_probs: Test set predicted probabilities
        test_labels: Test set binary labels
        history: Training history list
        output_dir: Directory to save figures

    Returns:
        Dictionary mapping plot names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_paths = {}

    # ROC curve
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plot_roc_curve(val_probs, val_labels, test_probs, test_labels, roc_path)
    plot_paths["roc_curve"] = roc_path

    # PR curve
    pr_path = os.path.join(output_dir, "pr_curve.png")
    plot_pr_curve(val_probs, val_labels, test_probs, test_labels, pr_path)
    plot_paths["pr_curve"] = pr_path

    # Calibration curve
    calib_path = os.path.join(output_dir, "calibration_curve.png")
    plot_calibration_curve(val_probs, val_labels, test_probs, test_labels, calib_path)
    plot_paths["calibration_curve"] = calib_path

    # Training history
    history_path = os.path.join(output_dir, "training_history.png")
    plot_training_history(history, history_path)
    plot_paths["training_history"] = history_path

    return plot_paths
