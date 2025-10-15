"""Inference helpers for generating calibrated landslide susceptibility and uncertainty maps."""

import os
import json
import shutil
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING

import joblib
import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.windows import Window

import segmentation_models_pytorch as smp

if TYPE_CHECKING:
    from src.main_pipeline import AreaArtifacts


def infer_with_tta(model: nn.Module, tensor: torch.Tensor, tta: bool) -> np.ndarray:
    """Forward a tile through the model with optional flip/rotation TTA."""
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
    base = probs.squeeze(0).cpu().numpy()
    if not tta:
        return base

    transformations = [
        (lambda x: torch.flip(x, dims=[-1]), lambda y: np.flip(y, axis=2)),
        (lambda x: torch.flip(x, dims=[-2]), lambda y: np.flip(y, axis=1)),
        (
            lambda x: torch.rot90(x, k=1, dims=[-2, -1]),
            lambda y: np.rot90(y, k=3, axes=(1, 2)),
        ),
        (
            lambda x: torch.rot90(x, k=3, dims=[-2, -1]),
            lambda y: np.rot90(y, k=1, axes=(1, 2)),
        ),
    ]

    acc = base.copy()
    for forward, inverse in transformations:
        with torch.no_grad():
            aug_logits = model(forward(tensor))
            aug_probs = torch.softmax(aug_logits, dim=1)
        acc += inverse(aug_probs.squeeze(0).cpu().numpy())
    acc /= len(transformations) + 1
    return acc


def sliding_window_predict(
    model: nn.Module,
    feature_stack_path: str,
    mask: np.ndarray,
    num_classes: int,
    window_size: int,
    overlap: int,
    device: torch.device,
    tta: bool,
    min_valid_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterate over the raster with overlap and accumulate class probabilities."""
    with rasterio.open(feature_stack_path) as src:
        height, width = src.height, src.width
        stride = max(1, window_size - overlap)
        prob_sum = np.zeros((num_classes, height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)

        for y in range(0, height, stride):
            for x in range(0, width, stride):
                h = min(window_size, height - y)
                w = min(window_size, width - x)
                window = Window(x, y, w, h)
                tile = src.read(window=window).astype(np.float32)
                tile_mask = mask[y : y + h, x : x + w]
                if tile_mask.mean() < min_valid_fraction:
                    continue

                tensor = (
                    torch.from_numpy(np.ascontiguousarray(tile)).unsqueeze(0).to(device)
                )
                probs = infer_with_tta(model, tensor, tta)[:, :h, :w]
                prob_sum[:, y : y + h, x : x + w] += probs
                counts[y : y + h, x : x + w] += 1

    return prob_sum, counts


def enable_dropout_layers(module: nn.Module) -> None:
    """Switch Dropout modules to train mode for MC Dropout sampling."""
    if isinstance(module, (nn.Dropout, nn.Dropout2d)):
        module.train()


def compute_uncertainty(
    mc_stack: Optional[np.ndarray],
    probabilities: np.ndarray,
    positive_class: int,
    mask: np.ndarray,
) -> np.ndarray:
    """Estimate pixel-level uncertainty from MC samples or entropy fallback."""
    if mc_stack is not None and mc_stack.size > 0:
        uncertainty = np.std(mc_stack[:, positive_class, :, :], axis=0).astype(
            np.float32
        )
    else:
        probs_clipped = np.clip(probabilities, 1e-6, 1.0)
        entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=0)
        uncertainty = entropy.astype(np.float32)
    uncertainty[~mask] = 0.0
    return uncertainty


def write_geotiff(
    path: str,
    array: np.ndarray,
    reference_path: str,
    dtype: str,
    nodata: Optional[float] = None,
) -> None:
    """Write an array using metadata from an existing raster on disk."""
    with rasterio.open(reference_path) as src:
        meta = src.meta.copy()
    data = array
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    meta.update(count=data.shape[0], dtype=dtype)
    if nodata is not None:
        meta["nodata"] = nodata
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data.astype(dtype))


def write_model_card(
    config: Dict,
    training_artifacts: Dict[str, Optional[str]],
    area: "AreaArtifacts",
    outputs_dir: str,
    products: Dict[str, str],
    channel_count: int,
) -> None:
    """Generate a lightweight markdown model card summarizing data, model, and outputs."""
    metrics_path = training_artifacts.get("metrics_path")
    best_metrics = {}
    test_metrics = {}
    best_epoch = None
    thresholds = {}
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            report = json.load(f)
        best_metrics = report.get("best_metrics", {}) or {}
        test_metrics = report.get("test_metrics", {}) or {}
        best_epoch = report.get("best_epoch")
        thresholds = report.get("thresholds", {})

    lines: List[str] = [
        "# Landslide Susceptibility Model Card",
        "## Data & Features",
        f"- Inference area: {area.name}",
        f"- Feature stack: {area.feature_stack_path}",
        f"- Channels: {channel_count}",
        f"- Valid mask: {area.mask_path}",
        "## Model",
        f'- Encoder: {config["model"]["encoder"]}',
        f'- Output classes: {config["model"]["out_classes"]}',
        f'- Positive class index: {config["dataset"].get("positive_class", config["model"]["out_classes"] - 1)}',
    ]
    if best_epoch is not None:
        lines.append(f"- Best epoch: {best_epoch}")
    
    # Validation metrics section
    if best_metrics:
        lines.append("\n## Validation Metrics")
        for key, value in best_metrics.items():
            if value is None:
                continue
            if isinstance(value, (int, float)) and np.isnan(value):
                continue
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
    
    # Test metrics section
    if test_metrics:
        lines.append("\n## Test Metrics")
        for key, value in test_metrics.items():
            if value is None:
                continue
            if isinstance(value, (int, float)) and np.isnan(value):
                continue
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
    
    # Threshold selection section
    if thresholds and "recommended_threshold" in thresholds:
        lines.append("\n## Classification Thresholds")
        lines.append(f"- Recommended threshold: {thresholds['recommended_threshold']:.4f}")
        lines.append(f"- Selection method: {thresholds.get('recommendation_method', 'unknown')}")
        
        # Add F1-optimal threshold details if available
        if "f1" in thresholds and thresholds["f1"].get("val"):
            f1_metrics = thresholds["f1"]["val"]
            lines.append(f"- F1-optimal (validation): threshold={f1_metrics['threshold']:.4f}, "
                        f"F1={f1_metrics['f1']:.4f}, precision={f1_metrics['precision']:.4f}, "
                        f"recall={f1_metrics['recall']:.4f}")
        
        # Add Youden threshold details if available
        if "youden" in thresholds and thresholds["youden"].get("val"):
            youden_metrics = thresholds["youden"]["val"]
            lines.append(f"- Youden-optimal (validation): threshold={youden_metrics['threshold']:.4f}, "
                        f"J={youden_metrics['youden_j']:.4f}, sensitivity={youden_metrics['sensitivity']:.4f}, "
                        f"specificity={youden_metrics['specificity']:.4f}")
    
    lines.extend(
        [
            "\n## Outputs",
            f'- Susceptibility map: {products["susceptibility"]}',
            f'- Uncertainty map: {products["uncertainty"]}',
            f'- Class map: {products["class_map"]}',
            f'- Valid mask: {products["valid_mask"]}',
        ]
    )
    if training_artifacts.get("calibrator_path"):
        lines.append(f'- Calibration: {training_artifacts["calibrator_path"]}')

    card_path = os.path.join(outputs_dir, "model_card.md")
    with open(card_path, "w") as f:
        f.write("\n".join(lines))


def run_inference(
    config: Dict,
    artifacts: Dict[str, "AreaArtifacts"],
    training_artifacts: Dict[str, Optional[str]],
    force_recreate: bool = False,
) -> None:
    """Perform sliding-window inference, optional calibration, and export deliverables."""
    structure_cfg = config["project_structure"]
    outputs_dir = structure_cfg.get("outputs_dir", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    area = artifacts.get("test") or artifacts["train"]
    area_name = area.name

    # Check if inference outputs already exist
    susceptibility_path = os.path.join(
        outputs_dir, f"{area_name}_landslide_susceptibility.tif"
    )
    uncertainty_path = os.path.join(outputs_dir, f"{area_name}_uncertainty.tif")
    valid_mask_path = os.path.join(outputs_dir, f"{area_name}_valid_mask.tif")
    model_card_path = os.path.join(outputs_dir, "model_card.md")

    outputs_exist = (
        os.path.exists(susceptibility_path)
        and os.path.exists(uncertainty_path)
        and os.path.exists(valid_mask_path)
        and os.path.exists(model_card_path)
    )

    if outputs_exist and not force_recreate:
        print(
            f"[inference] Inference outputs already exist in {outputs_dir}, skipping inference"
        )
        return

    print(f"[inference] Running inference (force_recreate={force_recreate})")

    device = torch.device("cuda")

    with open(area.metadata_path, "r") as f:
        metadata = json.load(f)
    channel_names = metadata.get("channel_names", [])
    in_channels = len(channel_names)

    checkpoint = torch.load(training_artifacts["model_path"], map_location=device)
    num_classes = checkpoint.get("num_classes", config["model"]["out_classes"])
    if in_channels == 0:
        in_channels = checkpoint.get("in_channels")
    if in_channels is None:
        raise ValueError("Unable to determine number of input channels for inference.")

    model = smp.Unet(
        encoder_name=config["model"]["encoder"],
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
    )
    dropout_prob = config["model"].get("dropout_prob", 0.0)
    if dropout_prob and dropout_prob > 0:
        model.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=dropout_prob), model.segmentation_head
        )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    calibrator_path = training_artifacts.get("calibrator_path")
    calibrator = None
    positive_class = config["dataset"].get("positive_class", num_classes - 1)
    if calibrator_path and os.path.exists(calibrator_path):
        payload = joblib.load(calibrator_path)
        calibrator = payload.get("calibrator")
        positive_class = payload.get("positive_class", positive_class)

    with rasterio.open(area.mask_path) as src:
        valid_mask = src.read(1).astype(np.uint8).astype(bool)

    window_size = config["inference"]["window_size"]
    overlap = config["inference"]["overlap"]
    tta = config["inference"].get("tta", False)
    min_valid_fraction = config["preprocessing"].get("min_valid_fraction", 0.85)

    prob_sum, counts = sliding_window_predict(
        model,
        area.feature_stack_path,
        valid_mask,
        num_classes,
        window_size,
        overlap,
        device,
        tta,
        min_valid_fraction,
    )

    counts_mask = counts > 0
    counts_safe = counts.copy()
    counts_safe[counts_safe == 0] = 1
    probabilities = prob_sum / counts_safe[np.newaxis, ...]
    probabilities[:, ~counts_mask] = 0.0
    probabilities[:, ~valid_mask] = 0.0

    mc_iterations = config["inference"].get("mc_dropout_iterations", 0)
    mc_stack = None
    if mc_iterations and mc_iterations > 0:
        model.apply(enable_dropout_layers)
        mc_maps: List[np.ndarray] = []
        for _ in range(mc_iterations):
            mc_sum, mc_counts = sliding_window_predict(
                model,
                area.feature_stack_path,
                valid_mask,
                num_classes,
                window_size,
                overlap,
                device,
                tta,
                min_valid_fraction,
            )
            mc_counts_safe = mc_counts.copy()
            mc_counts_safe[mc_counts_safe == 0] = 1
            mc_prob = mc_sum / mc_counts_safe[np.newaxis, ...]
            mc_prob[:, mc_counts == 0] = 0.0
            mc_prob[:, ~valid_mask] = 0.0
            mc_maps.append(mc_prob)
        mc_stack = np.stack(mc_maps, axis=0)
        model.eval()

    susceptibility = probabilities[positive_class, :, :].copy()
    valid_pixels = counts_mask & valid_mask
    if calibrator is not None and np.any(valid_pixels):
        calibrated = calibrator.predict(susceptibility[valid_pixels])
        susceptibility[valid_pixels] = calibrated
    susceptibility = np.clip(susceptibility, 0.0, 1.0).astype(np.float32)
    susceptibility[~valid_pixels] = 0.0

    # Load optimal threshold if available
    optimal_threshold = 0.5  # Default
    metrics_path = training_artifacts.get("metrics_path")
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_report = json.load(f)
        threshold_info = metrics_report.get("thresholds", {})
        if "recommended_threshold" in threshold_info:
            optimal_threshold = threshold_info["recommended_threshold"]
            print(
                f"[inference] Using optimal threshold {optimal_threshold:.4f} "
                f"(method: {threshold_info.get('recommendation_method', 'unknown')})"
            )
    
    # Generate class map using optimal threshold for positive class
    # Create binary mask for positive class first
    positive_binary = (susceptibility >= optimal_threshold).astype(np.uint8)
    
    # Use argmax for multi-class, but override with threshold-based positive class
    class_map = np.argmax(probabilities, axis=0).astype(np.uint8)
    
    # Override positive class pixels based on threshold
    # Only mark as positive if threshold is exceeded, otherwise use argmax result
    if num_classes == 2:
        # Binary case: use threshold directly
        class_map = positive_binary
    else:
        # Multi-class case: if positive class probability exceeds threshold, assign positive
        # Otherwise keep argmax result
        class_map = np.where(positive_binary, positive_class, class_map)
    
    class_map[~valid_pixels] = 255

    uncertainty_map = compute_uncertainty(
        mc_stack, probabilities, positive_class, valid_pixels
    )

    susceptibility_path = os.path.join(outputs_dir, f"{area.name}_susceptibility.tif")
    uncertainty_path = os.path.join(outputs_dir, f"{area.name}_uncertainty.tif")
    class_map_path = os.path.join(outputs_dir, f"{area.name}_class_map.tif")
    valid_mask_path = os.path.join(outputs_dir, f"{area.name}_valid_mask.tif")

    write_geotiff(
        susceptibility_path,
        susceptibility,
        area.feature_stack_path,
        dtype="float32",
        nodata=0.0,
    )
    write_geotiff(
        uncertainty_path,
        uncertainty_map,
        area.feature_stack_path,
        dtype="float32",
        nodata=0.0,
    )
    write_geotiff(
        class_map_path, class_map, area.feature_stack_path, dtype="uint8", nodata=255
    )
    write_geotiff(
        valid_mask_path,
        valid_pixels.astype(np.uint8),
        area.feature_stack_path,
        dtype="uint8",
        nodata=0,
    )

    metadata_copy_path = os.path.join(outputs_dir, f"{area.name}_feature_metadata.json")
    shutil.copyfile(area.metadata_path, metadata_copy_path)

    write_model_card(
        config,
        training_artifacts,
        area,
        outputs_dir,
        products={
            "susceptibility": susceptibility_path,
            "uncertainty": uncertainty_path,
            "class_map": class_map_path,
            "valid_mask": valid_mask_path,
        },
        channel_count=len(channel_names),
    )


if __name__ == "__main__":
    raise RuntimeError("Run main_pipeline.py to execute inference end-to-end.")
