"""Ensemble voting for generated custom LULC experiment outputs."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import rasterio

from .metrics import confusion_metrics


def log(message: str) -> None:
    print(f"[lulc] {message}", flush=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False, default=str)
    log(f"wrote JSON: {path}")


def run_metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    return read_json(metadata_path(row))


def run_output_path(row: Mapping[str, Any], filename_key: str) -> Path:
    metadata = run_metadata(row)
    filename = str(metadata["params"][filename_key])
    return Path(str(row["output_dir"])) / filename


def probability_path(row: Mapping[str, Any]) -> Path:
    return run_output_path(row, "probabilities_filename")


def lulc_path(row: Mapping[str, Any]) -> Path:
    return run_output_path(row, "lulc_filename")


def metadata_path(row: Mapping[str, Any]) -> Path:
    return Path(str(row["output_dir"])) / "lulc_model_metadata.json"


def eligible_runs(config: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ensemble_cfg = config["ensemble"]
    min_val = float(ensemble_cfg["min_val_macro_iou"])
    exclude_smoke = bool(ensemble_cfg["exclude_smoke_runs"])
    target_resolution = float(config["params"]["output_resolution"])
    eligible: List[Dict[str, Any]] = []
    for row in runs:
        run_id = str(row.get("run_id", ""))
        if exclude_smoke and run_id.startswith("smoke_"):
            continue
        if not metadata_path(row).exists():
            continue
        row_resolution = float(row.get("resolution", run_metadata(row)["params"]["output_resolution"]))
        if abs(row_resolution - target_resolution) > 1e-9:
            continue
        if float(row.get("val_macro_iou", 0.0)) < min_val:
            continue
        if not probability_path(row).exists() or not lulc_path(row).exists():
            continue
        eligible.append(dict(row))
    eligible.sort(key=lambda item: float(item.get("val_macro_iou", 0.0)), reverse=True)
    log(f"ensemble eligible runs: {len(eligible)}")
    return eligible


def signature(row: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row["architecture"]),
        str(row["encoder"]),
        str(row["loss"]),
        str(row["feature_set"]),
    )


def select_diverse_voters(config: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    max_models = int(config["ensemble"]["max_models"])
    eligible = eligible_runs(config, runs)
    preferred = [
        ("unetplusplus", "resnet18", "focal_lovasz", "rgb_indices"),
        ("deeplabv3plus", "resnet18", "focal_dice", "rgb"),
        ("unet", "resnet34", "weighted_ce_lovasz", "rgb"),
        ("unet", "mit_b0", "focal_lovasz", "rgb"),
        ("fpn", "resnet34", "weighted_ce_lovasz", "rgb"),
    ]
    selected: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str]] = set()
    for preferred_key in preferred:
        matches = [row for row in eligible if signature(row) == preferred_key]
        if not matches:
            continue
        row = max(matches, key=lambda item: float(item.get("val_macro_iou", 0.0)))
        selected.append(dict(row))
        seen.add(preferred_key)
        if len(selected) >= max_models:
            break
    for row in eligible:
        if len(selected) >= max_models:
            break
        key = signature(row)
        if key in seen:
            continue
        selected.append(dict(row))
        seen.add(key)
    if len(selected) < 2:
        raise ValueError("Need at least two eligible diverse LULC runs for ensemble voting.")
    log(f"ensemble selected voters: {[row['run_id'] for row in selected]}")
    return selected


def assert_same_grid(reference: rasterio.io.DatasetReader, candidate: rasterio.io.DatasetReader, path: Path) -> None:
    mismatches = []
    if reference.crs != candidate.crs:
        mismatches.append(f"crs {candidate.crs} != {reference.crs}")
    if reference.transform != candidate.transform:
        mismatches.append("transform differs")
    if reference.width != candidate.width or reference.height != candidate.height:
        mismatches.append(
            f"shape {candidate.width}x{candidate.height} != {reference.width}x{reference.height}"
        )
    if reference.count != candidate.count:
        mismatches.append(f"band count {candidate.count} != {reference.count}")
    if mismatches:
        raise ValueError(f"Ensemble raster grid mismatch for {path}: {', '.join(mismatches)}")


def average_probabilities(voters: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    first_path = probability_path(voters[0])
    with rasterio.open(first_path) as reference:
        profile = reference.profile.copy()
        probabilities = reference.read().astype(np.float64)
        if reference.count != 5:
            raise ValueError(f"Expected 5 probability bands in {first_path}, found {reference.count}")
        for row in voters[1:]:
            path = probability_path(row)
            with rasterio.open(path) as src:
                assert_same_grid(reference, src, path)
                probabilities += src.read().astype(np.float64)
    probabilities = probabilities / float(len(voters))
    sums = probabilities.sum(axis=0, keepdims=True)
    valid = sums > 1e-6
    probabilities = np.where(valid, probabilities / np.maximum(sums, 1e-6), 0.0)
    return probabilities.astype(np.float32), profile


def hard_vote_agreement(voters: Sequence[Mapping[str, Any]], ensemble_lulc: np.ndarray) -> np.ndarray:
    votes = np.zeros((len(voters), *ensemble_lulc.shape), dtype=np.uint8)
    for idx, row in enumerate(voters):
        with rasterio.open(lulc_path(row)) as src:
            votes[idx] = src.read(1).astype(np.uint8)
    valid = ensemble_lulc != 0
    agreement = np.zeros(ensemble_lulc.shape, dtype=np.float32)
    agreement[valid] = np.mean(votes[:, valid] == ensemble_lulc[valid], axis=0).astype(np.float32)
    return agreement


def probability_diagnostics(probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_probs = np.sort(probabilities, axis=0)
    confidence = sorted_probs[-1].astype(np.float32)
    margin = (sorted_probs[-1] - sorted_probs[-2]).astype(np.float32)
    safe = np.clip(probabilities, 1e-8, 1.0)
    entropy = (-np.sum(safe * np.log(safe), axis=0) / math.log(probabilities.shape[0])).astype(np.float32)
    invalid = probabilities.sum(axis=0) <= 1e-6
    confidence[invalid] = 0.0
    margin[invalid] = 0.0
    entropy[invalid] = 0.0
    return confidence, margin, entropy


def write_single_band(path: Path, profile: Mapping[str, Any], data: np.ndarray, description: str) -> None:
    output_profile = dict(profile)
    output_profile.update(
        count=1,
        dtype=str(data.dtype),
        nodata=0.0,
        compress="deflate",
        predictor=2 if np.issubdtype(data.dtype, np.floating) else None,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    output_profile = {key: value for key, value in output_profile.items() if value is not None}
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, description)
    log(f"wrote ensemble raster: {path}")


def write_probabilities(path: Path, profile: Mapping[str, Any], probabilities: np.ndarray) -> None:
    output_profile = dict(profile)
    output_profile.update(
        count=probabilities.shape[0],
        dtype="float32",
        nodata=0.0,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(probabilities.astype(np.float32))
        for band_idx in range(probabilities.shape[0]):
            dst.set_band_description(band_idx + 1, f"Ensemble P(class={band_idx + 1})")
    log(f"wrote ensemble probabilities: {path}")


def window_confusion(
    labels: np.ndarray,
    predictions: np.ndarray,
    positions: Iterable[Mapping[str, Any]],
    tile_size: int,
    ignore_index: int,
) -> np.ndarray:
    confusion = np.zeros((5, 5), dtype=np.int64)
    height, width = labels.shape
    for pos in positions:
        y = int(pos["y"])
        x = int(pos["x"])
        y2 = min(y + tile_size, height)
        x2 = min(x + tile_size, width)
        label_window = labels[y:y2, x:x2]
        pred_window = predictions[y:y2, x:x2]
        valid = (label_window != ignore_index) & (pred_window > 0)
        truth = label_window[valid].astype(np.int64) - 1
        pred = pred_window[valid].astype(np.int64) - 1
        valid_idx = (truth >= 0) & (truth < 5) & (pred >= 0) & (pred < 5)
        for t, p in zip(truth[valid_idx], pred[valid_idx]):
            confusion[int(t), int(p)] += 1
    return confusion


def evaluate_ensemble(config: Mapping[str, Any], voters: Sequence[Mapping[str, Any]], lulc: np.ndarray) -> Dict[str, Any]:
    metadata = read_json(metadata_path(voters[0]))
    labels_path = Path(str(voters[0]["output_dir"])) / str(metadata["params"]["training_labels_filename"])
    if not labels_path.exists():
        labels_path = Path(str(metadata["canonical_output_dir"])) / str(metadata["params"]["training_labels_filename"])
    with rasterio.open(labels_path) as src:
        labels = src.read(1)
    tile_size = int(metadata["params"]["tile_size"])
    ignore_index = int(metadata["params"]["ignore_index"])
    metrics = {}
    for split_name in ("val", "test"):
        positions = metadata["splits"][split_name]["positions"]
        metrics[split_name] = confusion_metrics(
            window_confusion(labels, lulc, positions, tile_size, ignore_index)
        )
    return metrics


def agreement_summary(
    lulc: np.ndarray,
    agreement: np.ndarray,
    confidence: np.ndarray,
    margin: np.ndarray,
    entropy: np.ndarray,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    valid = lulc != 0
    ensemble_cfg = config["ensemble"]
    by_class = {}
    for class_value in range(1, 6):
        mask = lulc == class_value
        by_class[str(class_value)] = {
            "pixels": int(np.count_nonzero(mask)),
            "mean_agreement": float(np.mean(agreement[mask])) if np.any(mask) else 0.0,
            "mean_confidence": float(np.mean(confidence[mask])) if np.any(mask) else 0.0,
            "mean_margin": float(np.mean(margin[mask])) if np.any(mask) else 0.0,
            "mean_entropy": float(np.mean(entropy[mask])) if np.any(mask) else 0.0,
        }
    return {
        "valid_pixels": int(np.count_nonzero(valid)),
        "mean_agreement": float(np.mean(agreement[valid])) if np.any(valid) else 0.0,
        "mean_confidence": float(np.mean(confidence[valid])) if np.any(valid) else 0.0,
        "mean_margin": float(np.mean(margin[valid])) if np.any(valid) else 0.0,
        "mean_entropy": float(np.mean(entropy[valid])) if np.any(valid) else 0.0,
        "low_agreement_pixels": int(
            np.count_nonzero(valid & (agreement < float(ensemble_cfg["low_agreement_threshold"])))
        ),
        "low_margin_pixels": int(
            np.count_nonzero(valid & (margin < float(ensemble_cfg["low_margin_threshold"])))
        ),
        "high_entropy_pixels": int(
            np.count_nonzero(valid & (entropy > float(ensemble_cfg["high_entropy_threshold"])))
        ),
        "by_predicted_class": by_class,
    }


def write_results_csv(path: Path, result: Mapping[str, Any]) -> None:
    rows = []
    for voter in result["voters"]:
        rows.append(
            {
                "row_type": "voter",
                "run_id": voter["run_id"],
                "architecture": voter["architecture"],
                "encoder": voter["encoder"],
                "feature_set": voter["feature_set"],
                "loss": voter["loss"],
                "seed": voter["seed"],
                "val_macro_iou": voter["val_macro_iou"],
                "test_macro_iou": voter["test_macro_iou"],
            }
        )
    rows.append(
        {
            "row_type": "ensemble",
            "run_id": result["run_id"],
            "architecture": "ensemble",
            "encoder": "",
            "feature_set": "mixed",
            "loss": "probability_average",
            "seed": "",
            "val_macro_iou": result["metrics"]["val"]["macro_iou"],
            "test_macro_iou": result["metrics"]["test"]["macro_iou"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log(f"wrote CSV: {path}")


def copy_to_canonical(config: Mapping[str, Any], result: Dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    params = config["params"]
    ensemble_cfg = config["ensemble"]
    copies = {
        ensemble_cfg["lulc_filename"]: params["lulc_filename"],
        ensemble_cfg["probabilities_filename"]: params["probabilities_filename"],
    }
    for src_name, dst_name in copies.items():
        src = output_dir / src_name
        dst = output_dir / dst_name
        shutil.copy2(src, dst)
        log(f"promoted ensemble {src} -> {dst}")
    result["promoted_lulc"] = str(output_dir / params["lulc_filename"])
    result["promoted_probabilities"] = str(output_dir / params["probabilities_filename"])


def build_ensemble(config: Mapping[str, Any], runs: Sequence[Mapping[str, Any]], promote: bool = True) -> Dict[str, Any]:
    ensemble_cfg = config["ensemble"]
    if not bool(ensemble_cfg["enabled"]):
        raise ValueError("Ensemble is disabled in lulc_inputs.py")
    if ensemble_cfg["strategy"] != "probability_average":
        raise ValueError(f"Unsupported ensemble strategy: {ensemble_cfg['strategy']}")
    if ensemble_cfg["voter_policy"] != "diverse_top_models":
        raise ValueError(f"Unsupported ensemble voter policy: {ensemble_cfg['voter_policy']}")

    output_dir = Path(config["output_dir"])
    voters = select_diverse_voters(config, runs)
    probabilities, profile = average_probabilities(voters)
    valid = probabilities.sum(axis=0) > 1e-6
    lulc = (np.argmax(probabilities, axis=0).astype(np.uint8) + 1)
    lulc[~valid] = int(config["params"]["output_nodata"])
    confidence, margin, entropy = probability_diagnostics(probabilities)
    agreement = hard_vote_agreement(voters, lulc)

    write_single_band(
        output_dir / ensemble_cfg["lulc_filename"],
        {**profile, "dtype": "uint8", "nodata": int(config["params"]["output_nodata"])},
        lulc.astype(np.uint8),
        "Ensemble custom polygon-trained LULC class",
    )
    write_probabilities(output_dir / ensemble_cfg["probabilities_filename"], profile, probabilities)
    write_single_band(output_dir / ensemble_cfg["agreement_filename"], profile, agreement, "Hard-vote agreement")
    write_single_band(output_dir / ensemble_cfg["confidence_filename"], profile, confidence, "Ensemble confidence")
    write_single_band(output_dir / ensemble_cfg["margin_filename"], profile, margin, "Ensemble top-two margin")
    write_single_band(output_dir / ensemble_cfg["entropy_filename"], profile, entropy, "Ensemble normalized entropy")

    metrics = evaluate_ensemble(config, voters, lulc)
    result: Dict[str, Any] = {
        "selection_type": "ensemble",
        "run_id": "ensemble_probability_average_diverse_top_models",
        "resolution": float(config["params"]["output_resolution"]),
        "strategy": ensemble_cfg["strategy"],
        "voter_policy": ensemble_cfg["voter_policy"],
        "voter_run_ids": [str(row["run_id"]) for row in voters],
        "voters": voters,
        "metrics": metrics,
        "agreement_summary": agreement_summary(lulc, agreement, confidence, margin, entropy, config),
        "outputs": {
            "ensemble_lulc": str(output_dir / ensemble_cfg["lulc_filename"]),
            "ensemble_probabilities": str(output_dir / ensemble_cfg["probabilities_filename"]),
            "agreement": str(output_dir / ensemble_cfg["agreement_filename"]),
            "confidence": str(output_dir / ensemble_cfg["confidence_filename"]),
            "margin": str(output_dir / ensemble_cfg["margin_filename"]),
            "entropy": str(output_dir / ensemble_cfg["entropy_filename"]),
        },
    }
    if promote:
        copy_to_canonical(config, result)
    write_json(output_dir / ensemble_cfg["results_json_filename"], result)
    write_results_csv(output_dir / ensemble_cfg["results_csv_filename"], result)
    return result
