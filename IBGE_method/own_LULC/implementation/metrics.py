"""Metrics and summaries for custom LULC training."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np


def class_distribution(labels: np.ndarray, ignore_index: int) -> Dict[str, Any]:
    valid = labels != ignore_index
    total = int(np.count_nonzero(valid))
    result: Dict[str, Any] = {"total_labeled_pixels": total, "classes": {}}
    for class_value in sorted(int(v) for v in np.unique(labels[valid])) if total else []:
        count = int(np.count_nonzero(labels == class_value))
        result["classes"][str(class_value)] = {
            "pixels": count,
            "fraction": float(count / total) if total else 0.0,
        }
    return result


def confusion_metrics(confusion: np.ndarray) -> Dict[str, Any]:
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    metrics: Dict[str, Any] = {
        "overall_accuracy": float(correct / total) if total else 0.0,
        "macro_iou": 0.0,
        "macro_f1": 0.0,
        "per_class": {},
    }
    ious = []
    f1s = []
    for idx in range(confusion.shape[0]):
        tp = float(confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - confusion[idx, idx])
        fn = float(confusion[idx, :].sum() - confusion[idx, idx])
        support = int(confusion[idx, :].sum())
        predicted = int(confusion[:, idx].sum())
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        ious.append(iou)
        f1s.append(f1)
        metrics["per_class"][str(idx)] = {"iou": float(iou), "f1": float(f1)}
        metrics["per_class"][str(idx)]["support_pixels"] = support
        metrics["per_class"][str(idx)]["predicted_pixels"] = predicted
    metrics["macro_iou"] = float(np.mean(ious)) if ious else 0.0
    metrics["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    metrics["confusion"] = confusion.astype(int).tolist()
    metrics["evaluated_pixels"] = total
    return metrics
