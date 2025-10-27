"""
Soft Label Generation for Ordinal Classification

Converts discrete class labels to probability distributions to express uncertainty
and ordinal relationships during training.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Literal


def ordinal_soft_labels(
    labels: np.ndarray,
    num_classes: int = 3,
    alpha: float = 0.1,
    ignore_value: int = 255,
) -> np.ndarray:
    """
    Generate ordinal-aware soft labels where adjacent classes receive probability mass.

    For ordinal classes (e.g., low < medium < high landslide risk), this smoothing
    reflects that misclassifying into an adjacent class is less severe than
    misclassifying across the full range.

    Args:
        labels: (H, W) array with discrete class labels (0, 1, 2, ..., num_classes-1)
                and optionally ignore_value for invalid pixels
        num_classes: Number of valid classes (default: 3)
        alpha: Smoothing strength.
               - 0.0 = hard labels (one-hot)
               - 0.1 = gentle smoothing (default)
               - 0.3 = strong smoothing
        ignore_value: Value in labels array to ignore (default: 255)

    Returns:
        (num_classes, H, W) array of class probabilities that sum to 1.0 per pixel.
        Ignored pixels have all zeros.

    Example:
        For alpha=0.1, num_classes=3:
        - Class 0 (low):    [0.95, 0.05, 0.00]
        - Class 1 (medium): [0.05, 0.90, 0.05]
        - Class 2 (high):   [0.00, 0.05, 0.95]
    """
    h, w = labels.shape
    soft = np.zeros((num_classes, h, w), dtype=np.float32)

    # Identify valid pixels
    valid_mask = labels != ignore_value

    for c in range(num_classes):
        class_mask = (labels == c) & valid_mask

        if not class_mask.any():
            continue

        # Main class gets (1 - alpha) probability
        soft[c, class_mask] = 1.0 - alpha

        # Distribute alpha to adjacent classes
        # Edge classes (0 and num_classes-1) have only one neighbor
        if c == 0:
            # Low risk: only distribute to medium
            if num_classes > 1:
                soft[c + 1, class_mask] = alpha
        elif c == num_classes - 1:
            # High risk: only distribute to medium
            soft[c - 1, class_mask] = alpha
        else:
            # Middle class(es): distribute equally to neighbors
            soft[c - 1, class_mask] = alpha / 2
            soft[c + 1, class_mask] = alpha / 2

    return soft


def gaussian_soft_labels(
    labels: np.ndarray,
    num_classes: int = 3,
    sigma: float = 1.0,
    ignore_value: int = 255,
) -> np.ndarray:
    """
    Generate spatially-smoothed soft labels using Gaussian filtering.

    This approach creates soft boundaries between classes based on spatial context,
    reflecting uncertainty at class transitions.

    Args:
        labels: (H, W) array with discrete class labels (0, 1, 2, ..., num_classes-1)
                and optionally ignore_value for invalid pixels
        num_classes: Number of valid classes (default: 3)
        sigma: Standard deviation for Gaussian kernel (default: 1.0)
               - Higher values create smoother transitions
               - Lower values keep labels sharper
        ignore_value: Value in labels array to ignore (default: 255)

    Returns:
        (num_classes, H, W) array of class probabilities that sum to 1.0 per pixel.
        Ignored pixels have all zeros.

    Example:
        For sigma=1.0:
        - Pixels far from boundaries remain ~one-hot [1, 0, 0]
        - Pixels at class boundaries become soft [0.6, 0.35, 0.05]
    """
    h, w = labels.shape
    soft = np.zeros((num_classes, h, w), dtype=np.float32)

    # Identify valid pixels
    valid_mask = (labels != ignore_value).astype(np.float32)

    # Create one-hot encoding for each class
    for c in range(num_classes):
        class_mask = (labels == c).astype(np.float32)
        # Mask invalid pixels before smoothing
        class_mask *= valid_mask
        # Apply Gaussian smoothing
        soft[c] = gaussian_filter(class_mask, sigma=sigma, mode="constant", cval=0.0)

    # Normalize to sum to 1.0 per pixel (only where we have valid data)
    soft_sum = soft.sum(axis=0, keepdims=True)
    soft_sum = np.where(soft_sum > 0, soft_sum, 1.0)  # Avoid division by zero
    soft = soft / soft_sum

    # Zero out ignored pixels
    ignore_mask = labels == ignore_value
    soft[:, ignore_mask] = 0.0

    return soft


def apply_label_smoothing(
    labels: np.ndarray,
    smoothing_type: Literal["ordinal", "gaussian", "none"] = "ordinal",
    num_classes: int = 3,
    alpha: float = 0.1,
    sigma: float = 1.0,
    ignore_value: int = 255,
) -> np.ndarray:
    """
    Apply label smoothing based on configuration.

    Args:
        labels: (H, W) array with discrete class labels
        smoothing_type: Type of smoothing to apply:
                        - "ordinal": Ordinal-aware smoothing (recommended)
                        - "gaussian": Spatial Gaussian smoothing
                        - "none": No smoothing (hard labels)
        num_classes: Number of valid classes
        alpha: Smoothing parameter for ordinal method
        sigma: Smoothing parameter for gaussian method
        ignore_value: Value to treat as invalid

    Returns:
        Either (num_classes, H, W) soft labels or (H, W) hard labels depending on type.
    """
    if smoothing_type == "none":
        return labels
    elif smoothing_type == "ordinal":
        return ordinal_soft_labels(labels, num_classes, alpha, ignore_value)
    elif smoothing_type == "gaussian":
        return gaussian_soft_labels(labels, num_classes, sigma, ignore_value)
    else:
        raise ValueError(
            f"Unknown smoothing_type: {smoothing_type}. "
            f"Must be 'ordinal', 'gaussian', or 'none'."
        )


def validate_soft_labels(soft_labels: np.ndarray, tolerance: float = 1e-5) -> bool:
    """
    Validate that soft labels are proper probability distributions.

    Args:
        soft_labels: (num_classes, H, W) array
        tolerance: Numerical tolerance for sum-to-one check

    Returns:
        True if valid, raises AssertionError otherwise
    """
    # Check shape
    assert soft_labels.ndim == 3, f"Expected 3D array, got {soft_labels.ndim}D"

    # Check range [0, 1]
    assert (
        soft_labels.min() >= -tolerance
    ), f"Found negative probabilities: {soft_labels.min()}"
    assert (
        soft_labels.max() <= 1.0 + tolerance
    ), f"Found probabilities > 1: {soft_labels.max()}"

    # Check sum to 1.0 per pixel (ignoring all-zero pixels which are masked)
    sums = soft_labels.sum(axis=0)
    non_zero_mask = sums > tolerance
    if non_zero_mask.any():
        sums_valid = sums[non_zero_mask]
        assert np.allclose(
            sums_valid, 1.0, atol=tolerance
        ), f"Probabilities don't sum to 1.0: min={sums_valid.min()}, max={sums_valid.max()}"

    return True
