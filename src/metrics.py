"""Metrics utilities for threshold selection and performance analysis."""

from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


def find_optimal_threshold_youden(
    y_true: np.ndarray, y_scores: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal classification threshold using Youden's J statistic.

    Youden's J = sensitivity + specificity - 1
    This maximizes the vertical distance from the ROC curve to the diagonal.

    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted probabilities for positive class

    Returns:
        Tuple of (optimal_threshold, metrics_dict)
        metrics_dict contains: threshold, sensitivity, specificity, youden_j
    """
    if len(np.unique(y_true)) < 2:
        # Only one class present, cannot compute meaningful threshold
        return 0.5, {
            "threshold": 0.5,
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "youden_j": float("nan"),
        }

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Youden's J statistic
    j_scores = (
        tpr - fpr
    )  # sensitivity - (1 - specificity) = sensitivity + specificity - 1

    # Find threshold that maximizes J
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[optimal_idx])
    optimal_sensitivity = float(tpr[optimal_idx])
    optimal_specificity = float(1 - fpr[optimal_idx])
    optimal_j = float(j_scores[optimal_idx])

    metrics = {
        "threshold": optimal_threshold,
        "sensitivity": optimal_sensitivity,
        "specificity": optimal_specificity,
        "youden_j": optimal_j,
    }

    return optimal_threshold, metrics


def find_optimal_threshold_f1(
    y_true: np.ndarray, y_scores: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal classification threshold that maximizes F1 score.

    F1 = 2 * (precision * recall) / (precision + recall)
    This balances precision and recall.

    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted probabilities for positive class

    Returns:
        Tuple of (optimal_threshold, metrics_dict)
        metrics_dict contains: threshold, precision, recall, f1
    """
    if len(np.unique(y_true)) < 2:
        # Only one class present, cannot compute meaningful threshold
        return 0.5, {
            "threshold": 0.5,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Compute F1 scores (handle division by zero)
    denom = precision + recall
    f1_scores = np.zeros_like(precision)
    mask = denom > 0
    f1_scores[mask] = 2 * (precision[mask] * recall[mask]) / denom[mask]

    # Find threshold that maximizes F1
    # Note: precision_recall_curve returns n+1 values for n thresholds
    # Last values correspond to threshold=0 (all positive predictions)
    optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last point

    # Handle edge case where thresholds array might be empty
    if len(thresholds) == 0:
        optimal_threshold = 0.5
    else:
        optimal_threshold = float(thresholds[optimal_idx])

    optimal_precision = float(precision[optimal_idx])
    optimal_recall = float(recall[optimal_idx])
    optimal_f1 = float(f1_scores[optimal_idx])

    metrics = {
        "threshold": optimal_threshold,
        "precision": optimal_precision,
        "recall": optimal_recall,
        "f1": optimal_f1,
    }

    return optimal_threshold, metrics


def compute_threshold_metrics(
    y_true: np.ndarray, y_scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    """
    Compute classification metrics at a specific threshold.

    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted probabilities for positive class
        threshold: Classification threshold

    Returns:
        Dictionary of metrics: accuracy, precision, recall, f1, specificity
    """
    if len(np.unique(y_true)) < 2:
        return {
            "threshold": threshold,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "specificity": float("nan"),
        }

    # Apply threshold
    y_pred = (y_scores >= threshold).astype(int)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Compute metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
    }


def select_optimal_thresholds(
    val_probs: Optional[np.ndarray],
    val_labels: Optional[np.ndarray],
    test_probs: Optional[np.ndarray],
    test_labels: Optional[np.ndarray],
) -> Dict[str, Dict]:
    """
    Select optimal classification thresholds using validation and test sets.

    Strategy:
    1. Use validation set to find candidate thresholds (Youden's J and F1-optimal)
    2. Report performance of these thresholds on test set
    3. If validation set unavailable, use test set to find thresholds

    Args:
        val_probs: Validation set predicted probabilities
        val_labels: Validation set binary labels
        test_probs: Test set predicted probabilities
        test_labels: Test set binary labels

    Returns:
        Dictionary with threshold selection results:
        {
            "youden": {"val": {...}, "test": {...}},
            "f1": {"val": {...}, "test": {...}},
            "recommended_threshold": float,
            "recommendation_method": str
        }
    """
    results = {
        "youden": {"val": None, "test": None},
        "f1": {"val": None, "test": None},
        "recommended_threshold": 0.5,
        "recommendation_method": "default",
    }

    # Determine which dataset to use for threshold selection
    use_val_for_selection = (
        val_probs is not None
        and val_labels is not None
        and len(val_labels) > 0
        and len(np.unique(val_labels)) > 1
    )

    if use_val_for_selection:
        # Select thresholds on validation set
        youden_threshold, youden_val_metrics = find_optimal_threshold_youden(
            val_labels, val_probs
        )
        f1_threshold, f1_val_metrics = find_optimal_threshold_f1(val_labels, val_probs)

        results["youden"]["val"] = youden_val_metrics
        results["f1"]["val"] = f1_val_metrics

        # Evaluate selected thresholds on test set
        if test_probs is not None and test_labels is not None and len(test_labels) > 0:
            youden_test_metrics = compute_threshold_metrics(
                test_labels, test_probs, youden_threshold
            )
            f1_test_metrics = compute_threshold_metrics(
                test_labels, test_probs, f1_threshold
            )

            results["youden"]["test"] = youden_test_metrics
            results["f1"]["test"] = f1_test_metrics

        # Recommend F1-optimal threshold (generally better for imbalanced data)
        results["recommended_threshold"] = f1_threshold
        results["recommendation_method"] = "f1_validation"

    elif (
        test_probs is not None
        and test_labels is not None
        and len(test_labels) > 0
        and len(np.unique(test_labels)) > 1
    ):
        # Fallback: select thresholds on test set (not ideal but better than default)
        youden_threshold, youden_test_metrics = find_optimal_threshold_youden(
            test_labels, test_probs
        )
        f1_threshold, f1_test_metrics = find_optimal_threshold_f1(
            test_labels, test_probs
        )

        results["youden"]["test"] = youden_test_metrics
        results["f1"]["test"] = f1_test_metrics

        # Recommend F1-optimal threshold
        results["recommended_threshold"] = f1_threshold
        results["recommendation_method"] = "f1_test"

    # If no valid data, keep default 0.5 threshold

    return results
