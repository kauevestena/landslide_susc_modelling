"""Inference helpers for generating calibrated landslide susceptibility and uncertainty maps."""

import os
import json
import shutil
import logging
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter, distance_transform_edt

import segmentation_models_pytorch as smp

from tqdm import tqdm

# Lazy import for CRF (optional dependency)
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    logging.warning("pydensecrf not available - CRF post-processing will be disabled")

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.main_pipeline import AreaArtifacts


def create_blend_weights(
    window_size: int, overlap: int, method: str = "gaussian", sigma_factor: float = 0.3
) -> np.ndarray:
    """
    Create weight masks for smooth blending of overlapping tiles.

    Args:
        window_size: Size of the inference window (assumed square)
        overlap: Overlap between adjacent windows in pixels
        method: Blending method - 'gaussian', 'linear', or 'none'
        sigma_factor: For gaussian, sigma = overlap * sigma_factor

    Returns:
        2D weight array with same shape as window
    """
    weights = np.ones((window_size, window_size), dtype=np.float32)

    if overlap <= 0 or method == "none":
        return weights

    if method == "gaussian":
        # Create 2D Gaussian that peaks at center and tapers at edges
        center = window_size / 2.0
        sigma = overlap * sigma_factor
        y, x = np.ogrid[:window_size, :window_size]

        # Distance from center
        dist_from_center = np.sqrt((x - center + 0.5) ** 2 + (y - center + 0.5) ** 2)

        # Gaussian weight (higher in center, lower at edges)
        weights = np.exp(-(dist_from_center**2) / (2 * sigma**2))

    elif method == "linear":
        # Linear taper from edges
        # Distance from nearest edge
        y_indices = np.arange(window_size)
        x_indices = np.arange(window_size)

        # Distance to nearest edge
        y_dist = np.minimum(y_indices, window_size - 1 - y_indices)
        x_dist = np.minimum(x_indices, window_size - 1 - x_indices)

        y_weights = np.minimum(y_dist / overlap, 1.0)
        x_weights = np.minimum(x_dist / overlap, 1.0)

        weights = np.outer(y_weights, x_weights)

    return weights


def infer_with_tta(
    model: nn.Module, tensor: torch.Tensor, tta: bool, num_augmentations: int = 8
) -> np.ndarray:
    """
    Forward a tile through the model with enhanced Test Time Augmentation.

    Args:
        model: Neural network model
        tensor: Input tensor [1, C, H, W]
        tta: Whether to enable TTA
        num_augmentations: Number of augmentations (0=off, 4=basic, 8=full)

    Returns:
        Averaged probability predictions [num_classes, H, W]
    """
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
    base = probs.squeeze(0).cpu().numpy()

    if not tta or num_augmentations == 0:
        return base

    # Define augmentation pairs (forward transform, inverse transform)
    transformations = []

    # Basic augmentations (4 total)
    if num_augmentations >= 4:
        # Horizontal flip
        transformations.append(
            (lambda x: torch.flip(x, dims=[-1]), lambda y: np.flip(y, axis=2))
        )
        # Vertical flip
        transformations.append(
            (lambda x: torch.flip(x, dims=[-2]), lambda y: np.flip(y, axis=1))
        )
        # 90° rotation
        transformations.append(
            (
                lambda x: torch.rot90(x, k=1, dims=[-2, -1]),
                lambda y: np.rot90(y, k=3, axes=(1, 2)),
            )
        )
        # 270° rotation
        transformations.append(
            (
                lambda x: torch.rot90(x, k=3, dims=[-2, -1]),
                lambda y: np.rot90(y, k=1, axes=(1, 2)),
            )
        )

    # Full augmentations (8 total = 4 basic + 4 combined)
    if num_augmentations >= 8:
        # 180° rotation
        transformations.append(
            (
                lambda x: torch.rot90(x, k=2, dims=[-2, -1]),
                lambda y: np.rot90(y, k=2, axes=(1, 2)),
            )
        )
        # Horizontal flip + 90° rotation
        transformations.append(
            (
                lambda x: torch.rot90(torch.flip(x, dims=[-1]), k=1, dims=[-2, -1]),
                lambda y: np.flip(np.rot90(y, k=3, axes=(1, 2)), axis=2),
            )
        )
        # Vertical flip + 90° rotation
        transformations.append(
            (
                lambda x: torch.rot90(torch.flip(x, dims=[-2]), k=1, dims=[-2, -1]),
                lambda y: np.flip(np.rot90(y, k=3, axes=(1, 2)), axis=1),
            )
        )
        # Both flips (equivalent to 180° rotation, but different for non-square features)
        transformations.append(
            (
                lambda x: torch.flip(torch.flip(x, dims=[-1]), dims=[-2]),
                lambda y: np.flip(np.flip(y, axis=2), axis=1),
            )
        )

    # Accumulate predictions from all augmentations
    acc = base.copy()
    for forward, inverse in transformations[:num_augmentations]:
        with torch.no_grad():
            aug_logits = model(forward(tensor))
            aug_probs = torch.softmax(aug_logits, dim=1)
        acc += inverse(aug_probs.squeeze(0).cpu().numpy())

    # Average across all augmentations
    acc /= len(transformations[:num_augmentations]) + 1
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
    num_tta_augmentations: int,
    min_valid_fraction: float,
    blend_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterate over the raster with overlap and accumulate weighted class probabilities.

    Args:
        model: Neural network model
        feature_stack_path: Path to input raster
        mask: Valid pixel mask
        num_classes: Number of output classes
        window_size: Size of sliding window
        overlap: Overlap between windows
        device: Torch device
        tta: Enable test time augmentation
        num_tta_augmentations: Number of TTA augmentations
        min_valid_fraction: Minimum valid pixel fraction to process tile
        blend_weights: Optional pre-computed blending weights

    Returns:
        (weighted_prob_sum, weight_sum) both [num_classes, H, W] and [H, W]
    """
    logger.info(f"[sliding_window_predict] Starting inference on {feature_stack_path}")
    logger.info(
        f"[sliding_window_predict] Window size: {window_size}, overlap: {overlap}"
    )

    with rasterio.open(feature_stack_path) as src:
        height, width = src.height, src.width
        logger.info(f"[sliding_window_predict] Raster dimensions: {height}x{width}")

        stride = max(1, window_size - overlap)
        logger.info(f"[sliding_window_predict] Stride: {stride}")

        prob_sum = np.zeros((num_classes, height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        # Calculate total windows for progress bar
        y_windows = list(range(0, height, stride))
        x_windows = list(range(0, width, stride))
        total_windows = len(y_windows) * len(x_windows)
        logger.info(
            f"[sliding_window_predict] Total windows to process: {total_windows}"
        )

        processed = 0
        skipped = 0

        # Single progress bar for all windows
        with tqdm(
            total=total_windows, desc="Sliding window inference", unit="window"
        ) as pbar:
            for y in y_windows:
                for x in x_windows:
                    h = min(window_size, height - y)
                    w = min(window_size, width - x)
                    window = Window(x, y, w, h)
                    tile = src.read(window=window).astype(np.float32)
                    tile_mask = mask[y : y + h, x : x + w]

                    if tile_mask.mean() < min_valid_fraction:
                        skipped += 1
                        pbar.update(1)
                        continue

                    tensor = (
                        torch.from_numpy(np.ascontiguousarray(tile))
                        .unsqueeze(0)
                        .to(device)
                    )
                    probs = infer_with_tta(model, tensor, tta, num_tta_augmentations)[
                        :, :h, :w
                    ]

                    # Apply blending weights
                    if blend_weights is not None:
                        tile_weights = blend_weights[:h, :w]
                        weighted_probs = probs * tile_weights[np.newaxis, :, :]
                    else:
                        tile_weights = np.ones((h, w), dtype=np.float32)
                        weighted_probs = probs

                    # Accumulate weighted predictions
                    prob_sum[:, y : y + h, x : x + w] += weighted_probs
                    weight_sum[y : y + h, x : x + w] += tile_weights

                    processed += 1
                    pbar.update(1)

    logger.info(
        f"[sliding_window_predict] Processed: {processed}, Skipped: {skipped} (low valid fraction)"
    )
    return prob_sum, weight_sum


def enable_dropout_layers(module: nn.Module) -> None:
    """Switch Dropout modules to train mode for MC Dropout sampling."""
    if isinstance(module, (nn.Dropout, nn.Dropout2d)):
        module.train()


def apply_crf(
    probabilities: np.ndarray,
    features: np.ndarray,
    mask: np.ndarray,
    iterations: int = 5,
    spatial_weight: float = 3.0,
    color_weight: float = 3.0,
    compat_spatial: float = 3.0,
    compat_bilateral: float = 10.0,
) -> np.ndarray:
    """
    Apply Conditional Random Field (CRF) post-processing for spatial coherence.

    Uses dense CRF with Gaussian pairwise potentials based on spatial proximity
    and feature similarity. This helps smooth predictions while respecting edges.

    Args:
        probabilities: Class probabilities [num_classes, H, W]
        features: Input features for bilateral filtering [C, H, W]
        mask: Valid pixel mask [H, W]
        iterations: Number of mean-field inference iterations
        spatial_weight: Spatial standard deviation for Gaussian kernel
        color_weight: Feature/color standard deviation for bilateral kernel
        compat_spatial: Compatibility weight for spatial kernel
        compat_bilateral: Compatibility weight for bilateral kernel

    Returns:
        Refined probabilities [num_classes, H, W]
    """
    if not CRF_AVAILABLE:
        logger.warning(
            "[apply_crf] pydensecrf not available, returning original probabilities"
        )
        return probabilities

    # Ensure input arrays are C-contiguous and proper dtype
    probabilities = np.ascontiguousarray(probabilities, dtype=np.float32)
    features = np.ascontiguousarray(features, dtype=np.float32)
    mask = np.ascontiguousarray(mask, dtype=bool)
    
    num_classes, height, width = probabilities.shape

    # Normalize features to 0-255 range for CRF (use first 3 channels if available)
    if features.shape[0] >= 3:
        # Use RGB-like channels (orthophoto if available)
        feat_for_crf = features[:3].copy()
    else:
        # Repeat single channel or use available channels
        feat_for_crf = np.repeat(features[:1], 3, axis=0)

    # Normalize to 0-255
    for i in range(feat_for_crf.shape[0]):
        channel = feat_for_crf[i]
        channel_min = channel[mask].min() if mask.any() else channel.min()
        channel_max = channel[mask].max() if mask.any() else channel.max()
        if channel_max > channel_min:
            feat_for_crf[i] = (
                (channel - channel_min) / (channel_max - channel_min) * 255
            ).astype(np.uint8)
        else:
            feat_for_crf[i] = 128  # Neutral gray if constant

    # Create DenseCRF object
    d = dcrf.DenseCRF2D(width, height, num_classes)

    # Set unary potentials (negative log probabilities)
    # Ensure C-contiguous arrays for pydensecrf
    probs_clipped = np.clip(probabilities, 1e-10, 1.0)
    probs_clipped = np.ascontiguousarray(probs_clipped, dtype=np.float32)
    unary = unary_from_softmax(probs_clipped)
    unary = np.ascontiguousarray(unary, dtype=np.float32)
    d.setUnaryEnergy(unary)

    # Add pairwise Gaussian potential (spatial smoothness)
    d.addPairwiseGaussian(
        sxy=spatial_weight,
        compat=compat_spatial,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    # Add pairwise bilateral potential (edge-aware smoothing)
    # Prepare RGB image in HWC format with proper data type and memory layout
    rgbim = feat_for_crf.transpose(1, 2, 0)
    rgbim = np.ascontiguousarray(rgbim, dtype=np.uint8)
    d.addPairwiseBilateral(
        sxy=spatial_weight,
        srgb=color_weight,
        rgbim=rgbim,
        compat=compat_bilateral,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    # Perform mean-field inference
    Q = d.inference(iterations)
    refined_probs = np.array(Q).reshape((num_classes, height, width))

    # Apply mask
    refined_probs[:, ~mask] = 0.0

    return refined_probs


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
        lines.append(
            f"- Recommended threshold: {thresholds['recommended_threshold']:.4f}"
        )
        lines.append(
            f"- Selection method: {thresholds.get('recommendation_method', 'unknown')}"
        )

        # Add F1-optimal threshold details if available
        if "f1" in thresholds and thresholds["f1"].get("val"):
            f1_metrics = thresholds["f1"]["val"]
            lines.append(
                f"- F1-optimal (validation): threshold={f1_metrics['threshold']:.4f}, "
                f"F1={f1_metrics['f1']:.4f}, precision={f1_metrics['precision']:.4f}, "
                f"recall={f1_metrics['recall']:.4f}"
            )

        # Add Youden threshold details if available
        if "youden" in thresholds and thresholds["youden"].get("val"):
            youden_metrics = thresholds["youden"]["val"]
            lines.append(
                f"- Youden-optimal (validation): threshold={youden_metrics['threshold']:.4f}, "
                f"J={youden_metrics['youden_j']:.4f}, sensitivity={youden_metrics['sensitivity']:.4f}, "
                f"specificity={youden_metrics['specificity']:.4f}"
            )

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
    logger.info("=" * 80)
    logger.info("[run_inference] Starting inference pipeline")
    logger.info("=" * 80)

    structure_cfg = config["project_structure"]
    outputs_dir = structure_cfg.get("outputs_dir", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    logger.info(f"[run_inference] Output directory: {outputs_dir}")

    area = artifacts.get("test") or artifacts["train"]
    area_name = area.name
    logger.info(f"[run_inference] Inference area: {area_name}")

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
        logger.info(
            f"[run_inference] Inference outputs already exist in {outputs_dir}, skipping inference"
        )
        return

    logger.info(f"[run_inference] Running inference (force_recreate={force_recreate})")

    # Device selection
    logger.info("[run_inference] Configuring compute device...")
    use_cuda = config.get("inference", {}).get("use_cuda", False)

    if use_cuda and torch.cuda.is_available():
        try:
            # Test if GPU is actually usable with a convolution operation
            test_input = torch.randn(1, 3, 32, 32).cuda()
            test_conv = torch.nn.Conv2d(3, 3, 3).cuda()
            _ = test_conv(test_input)
            device = torch.device("cuda")
            logger.info("[run_inference] Using CUDA device for inference")
        except (RuntimeError, AssertionError) as e:
            logger.warning(f"[run_inference] CUDA test failed: {str(e)[:100]}...")
            logger.info("[run_inference] Falling back to CPU")
            device = torch.device("cpu")
    else:
        if not use_cuda:
            logger.info("[run_inference] use_cuda=False in config, using CPU")
        else:
            logger.info("[run_inference] CUDA not available, using CPU")
        device = torch.device("cpu")

    # Load metadata
    logger.info(f"[run_inference] Loading metadata from {area.metadata_path}")
    with open(area.metadata_path, "r") as f:
        metadata = json.load(f)
    channel_names = metadata.get("channel_names", [])
    in_channels = len(channel_names)
    logger.info(f"[run_inference] Input channels: {in_channels}")

    # Load model
    logger.info(
        f"[run_inference] Loading model from {training_artifacts['model_path']}"
    )
    checkpoint = torch.load(training_artifacts["model_path"], map_location=device)
    num_classes = checkpoint.get("num_classes", config["model"]["out_classes"])
    if in_channels == 0:
        in_channels = checkpoint.get("in_channels")
    if in_channels is None:
        raise ValueError("Unable to determine number of input channels for inference.")
    logger.info(f"[run_inference] Model classes: {num_classes}")

    logger.info(
        f"[run_inference] Building model architecture: {config['model']['encoder']}"
    )
    model = smp.Unet(
        encoder_name=config["model"]["encoder"],
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
    )

    dropout_prob = config["model"].get("dropout_prob", 0.0)
    if dropout_prob and dropout_prob > 0:
        logger.info(f"[run_inference] Adding dropout layer (p={dropout_prob})")
        model.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=dropout_prob), model.segmentation_head
        )

    logger.info("[run_inference] Loading model weights...")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    logger.info("[run_inference] Model loaded and set to evaluation mode")

    # Load calibrator
    calibrator_path = training_artifacts.get("calibrator_path")
    calibrator = None
    positive_class = config["dataset"].get("positive_class", num_classes - 1)
    if calibrator_path and os.path.exists(calibrator_path):
        logger.info(f"[run_inference] Loading calibrator from {calibrator_path}")
        payload = joblib.load(calibrator_path)
        calibrator = payload.get("calibrator")
        positive_class = payload.get("positive_class", positive_class)
        logger.info(
            f"[run_inference] Calibrator loaded for positive class: {positive_class}"
        )
    else:
        logger.info("[run_inference] No calibrator found, using raw probabilities")

    # Load valid mask
    logger.info(f"[run_inference] Loading valid mask from {area.mask_path}")
    with rasterio.open(area.mask_path) as src:
        valid_mask = src.read(1).astype(np.uint8).astype(bool)
    logger.info(
        f"[run_inference] Valid mask shape: {valid_mask.shape}, valid pixels: {valid_mask.sum()}"
    )

    window_size = config["inference"]["window_size"]
    overlap = config["inference"]["overlap"]
    tta = config["inference"].get("tta", False)
    num_tta_augmentations = config["inference"].get("tta_augmentations", 4)
    min_valid_fraction = config["preprocessing"].get("min_valid_fraction", 0.85)

    # Blending configuration
    blend_config = config["inference"].get("blending", {})
    blend_method = blend_config.get("method", "gaussian")
    sigma_factor = blend_config.get("sigma_factor", 0.3)

    # CRF configuration
    crf_config = config["inference"].get("crf", {})
    crf_enabled = crf_config.get("enabled", False) and CRF_AVAILABLE

    logger.info(
        f"[run_inference] Inference config: window={window_size}, overlap={overlap}, "
        f"TTA={tta} (augmentations={num_tta_augmentations}), blending={blend_method}"
    )
    if crf_enabled:
        logger.info(
            f"[run_inference] CRF enabled: iterations={crf_config.get('iterations', 5)}, "
            f"spatial_weight={crf_config.get('spatial_weight', 3.0)}, "
            f"color_weight={crf_config.get('color_weight', 3.0)}"
        )
    else:
        logger.info("[run_inference] CRF post-processing disabled")

    # Create blending weights
    if overlap > 0 and blend_method != "none":
        logger.info(
            f"[run_inference] Creating blend weights (method={blend_method}, sigma_factor={sigma_factor})"
        )
        blend_weights = create_blend_weights(
            window_size, overlap, blend_method, sigma_factor
        )
    else:
        logger.info("[run_inference] No blending (overlap=0 or method=none)")
        blend_weights = None

    # Main sliding window prediction
    logger.info("[run_inference] Starting main sliding window prediction...")
    prob_sum, weight_sum = sliding_window_predict(
        model,
        area.feature_stack_path,
        valid_mask,
        num_classes,
        window_size,
        overlap,
        device,
        tta,
        num_tta_augmentations,
        min_valid_fraction,
        blend_weights,
    )
    logger.info("[run_inference] Main prediction complete")

    # Average probabilities with weighted blending
    logger.info("[run_inference] Computing weighted averaged probabilities...")
    weight_sum_safe = weight_sum.copy()
    weight_sum_safe[weight_sum_safe == 0] = 1.0  # Avoid division by zero
    probabilities = prob_sum / weight_sum_safe[np.newaxis, ...]
    probabilities[:, weight_sum == 0] = 0.0
    probabilities[:, ~valid_mask] = 0.0
    logger.info(f"[run_inference] Probabilities computed, shape: {probabilities.shape}")

    # Apply CRF post-processing if enabled
    if crf_enabled:
        logger.info(
            "[run_inference] Applying CRF post-processing for spatial coherence..."
        )
        # Load features for bilateral filtering
        with rasterio.open(area.feature_stack_path) as src:
            features = src.read().astype(np.float32)

        probabilities = apply_crf(
            probabilities,
            features,
            valid_mask,
            iterations=crf_config.get("iterations", 5),
            spatial_weight=crf_config.get("spatial_weight", 3.0),
            color_weight=crf_config.get("color_weight", 3.0),
            compat_spatial=crf_config.get("compat_spatial", 3.0),
            compat_bilateral=crf_config.get("compat_bilateral", 10.0),
        )
        logger.info("[run_inference] CRF post-processing complete")

    # MC Dropout uncertainty
    mc_iterations = config["inference"].get("mc_dropout_iterations", 0)
    mc_stack = None
    if mc_iterations and mc_iterations > 0:
        logger.info(
            f"[run_inference] Starting MC Dropout with {mc_iterations} iterations..."
        )
        model.apply(enable_dropout_layers)
        mc_maps: List[np.ndarray] = []

        for i in tqdm(range(mc_iterations), desc="MC Dropout iterations", unit="iter"):
            logger.info(f"[run_inference] MC Dropout iteration {i+1}/{mc_iterations}")
            mc_sum, mc_weight_sum = sliding_window_predict(
                model,
                area.feature_stack_path,
                valid_mask,
                num_classes,
                window_size,
                overlap,
                device,
                tta,
                num_tta_augmentations,
                min_valid_fraction,
                blend_weights,
            )
            mc_weight_sum_safe = mc_weight_sum.copy()
            mc_weight_sum_safe[mc_weight_sum_safe == 0] = 1.0
            mc_prob = mc_sum / mc_weight_sum_safe[np.newaxis, ...]
            mc_prob[:, mc_weight_sum == 0] = 0.0
            mc_prob[:, ~valid_mask] = 0.0
            mc_maps.append(mc_prob)

        mc_stack = np.stack(mc_maps, axis=0)
        model.eval()
        logger.info(
            f"[run_inference] MC Dropout complete, stack shape: {mc_stack.shape}"
        )
    else:
        logger.info("[run_inference] MC Dropout disabled (iterations=0)")

    # Extract and calibrate susceptibility
    logger.info("[run_inference] Extracting susceptibility map for positive class...")
    susceptibility = probabilities[positive_class, :, :].copy()
    valid_pixels = (weight_sum > 0) & valid_mask

    if calibrator is not None and np.any(valid_pixels):
        logger.info("[run_inference] Applying calibration to susceptibility values...")
        calibrated = calibrator.predict(susceptibility[valid_pixels])
        susceptibility[valid_pixels] = calibrated
        logger.info("[run_inference] Calibration applied")
    else:
        logger.info(
            "[run_inference] Skipping calibration (no calibrator or no valid pixels)"
        )

    susceptibility = np.clip(susceptibility, 0.0, 1.0).astype(np.float32)
    susceptibility[~valid_pixels] = 0.0
    logger.info(
        f"[run_inference] Susceptibility range: [{susceptibility.min():.4f}, {susceptibility.max():.4f}]"
    )

    # Load optimal threshold
    logger.info("[run_inference] Loading optimal threshold...")
    optimal_threshold = 0.5  # Default
    metrics_path = training_artifacts.get("metrics_path")
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_report = json.load(f)
        threshold_info = metrics_report.get("thresholds", {})
        if "recommended_threshold" in threshold_info:
            optimal_threshold = threshold_info["recommended_threshold"]
            logger.info(
                f"[run_inference] Using optimal threshold {optimal_threshold:.4f} "
                f"(method: {threshold_info.get('recommendation_method', 'unknown')})"
            )
    else:
        logger.info(f"[run_inference] Using default threshold: {optimal_threshold}")

    # Generate class map
    logger.info("[run_inference] Generating class map...")
    positive_binary = (susceptibility >= optimal_threshold).astype(np.uint8)

    class_map = np.argmax(probabilities, axis=0).astype(np.uint8)

    if num_classes == 2:
        class_map = positive_binary
        logger.info("[run_inference] Binary classification: using threshold-based map")
    else:
        class_map = np.where(positive_binary, positive_class, class_map)
        logger.info(
            "[run_inference] Multi-class classification: applied threshold for positive class"
        )

    class_map[~valid_pixels] = 255
    logger.info(
        f"[run_inference] Class map generated, unique values: {np.unique(class_map)}"
    )

    # Compute uncertainty
    logger.info("[run_inference] Computing uncertainty map...")
    uncertainty_map = compute_uncertainty(
        mc_stack, probabilities, positive_class, valid_pixels
    )
    logger.info(
        f"[run_inference] Uncertainty range: [{uncertainty_map.min():.4f}, {uncertainty_map.max():.4f}]"
    )

    # Write outputs
    logger.info("[run_inference] Writing output GeoTIFFs...")
    susceptibility_path = os.path.join(outputs_dir, f"{area.name}_susceptibility.tif")
    uncertainty_path = os.path.join(outputs_dir, f"{area.name}_uncertainty.tif")
    class_map_path = os.path.join(outputs_dir, f"{area.name}_class_map.tif")
    valid_mask_path = os.path.join(outputs_dir, f"{area.name}_valid_mask.tif")

    logger.info(f"[run_inference] Writing: {susceptibility_path}")
    write_geotiff(
        susceptibility_path,
        susceptibility,
        area.feature_stack_path,
        dtype="float32",
        nodata=0.0,
    )

    logger.info(f"[run_inference] Writing: {uncertainty_path}")
    write_geotiff(
        uncertainty_path,
        uncertainty_map,
        area.feature_stack_path,
        dtype="float32",
        nodata=0.0,
    )

    logger.info(f"[run_inference] Writing: {class_map_path}")
    write_geotiff(
        class_map_path, class_map, area.feature_stack_path, dtype="uint8", nodata=255
    )

    logger.info(f"[run_inference] Writing: {valid_mask_path}")
    write_geotiff(
        valid_mask_path,
        valid_pixels.astype(np.uint8),
        area.feature_stack_path,
        dtype="uint8",
        nodata=0,
    )

    logger.info("[run_inference] Copying feature metadata...")
    metadata_copy_path = os.path.join(outputs_dir, f"{area.name}_feature_metadata.json")
    shutil.copyfile(area.metadata_path, metadata_copy_path)

    logger.info("[run_inference] Writing model card...")
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

    logger.info("=" * 80)
    logger.info("[run_inference] Inference pipeline complete!")
    logger.info(f"[run_inference] All outputs written to: {outputs_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    raise RuntimeError("Run main_pipeline.py to execute inference end-to-end.")
