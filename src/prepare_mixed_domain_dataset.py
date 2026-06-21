"""
Enhanced dataset preparation that merges training and test areas for mixed-domain learning.
Exports tiles as both .npy (for fast training) and GeoTIFF (for spatial inspection).
"""

import os
import json
import glob
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from sklearn.model_selection import train_test_split


MIXED_MERGE_SCHEMA_VERSION = 2
MIXED_DATASET_SCHEMA_VERSION = 2


def debug_print(msg: str):
    """Print debug message."""
    print(f"[prepare_mixed_domain] {msg}")


def merge_artifacts_current(
    merged_features_path: str,
    merged_labels_path: str,
    merged_mask_path: str,
    metadata_path: str,
) -> Tuple[bool, str]:
    """Validate merged raster artifacts before reusing them."""
    required_paths = [
        merged_features_path,
        merged_labels_path,
        merged_mask_path,
        metadata_path,
    ]
    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        return False, f"missing merged artifacts: {missing}"

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"merged metadata unreadable: {exc}"

    if metadata.get("schema_version") != MIXED_MERGE_SCHEMA_VERSION:
        return False, "merged metadata schema is missing or stale"

    for area_name in ("train_area", "test_area"):
        area = metadata.get(area_name, {})
        for key in ("height", "width", "row_range", "source", "transform"):
            if key not in area:
                return False, f"merged metadata missing {area_name}.{key}"

    try:
        with rasterio.open(merged_features_path) as features_src:
            merged = metadata.get("merged", {})
            if features_src.count != merged.get("channels"):
                return False, "merged feature channel count differs from metadata"
            if features_src.height != merged.get("height"):
                return False, "merged feature height differs from metadata"
            if features_src.width != merged.get("width"):
                return False, "merged feature width differs from metadata"
            feature_shape = (features_src.height, features_src.width)
        with rasterio.open(merged_labels_path) as labels_src:
            if (labels_src.height, labels_src.width) != feature_shape:
                return False, "merged label shape differs from features"
        with rasterio.open(merged_mask_path) as mask_src:
            if (mask_src.height, mask_src.width) != feature_shape:
                return False, "merged mask shape differs from features"
    except rasterio.errors.RasterioIOError as exc:
        return False, f"merged raster unreadable: {exc}"

    return True, "current"


def mixed_dataset_artifacts_current(
    splits_path: str,
    summary_path: str,
    structure_cfg: Dict,
    config: Dict,
) -> Tuple[bool, str]:
    """Validate mixed-domain tile artifacts before resuming."""
    if not os.path.exists(splits_path) or not os.path.exists(summary_path):
        return False, "splits or dataset summary missing"

    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
        with open(splits_path, "r") as f:
            splits = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"dataset metadata unreadable: {exc}"

    dataset_cfg = config["dataset"]
    smoothing_cfg = config["preprocessing"].get("label_smoothing", {})
    use_soft_labels = bool(smoothing_cfg.get("enabled", False))
    expected_smoothing = {
        "enabled": use_soft_labels,
        "type": smoothing_cfg.get("type", "none") if use_soft_labels else "none",
        "alpha": smoothing_cfg.get("alpha", 0.0) if use_soft_labels else 0.0,
        "sigma": smoothing_cfg.get("sigma", 0.0) if use_soft_labels else 0.0,
    }

    if summary.get("schema_version") != MIXED_DATASET_SCHEMA_VERSION:
        return False, "dataset summary schema is missing or stale"
    if summary.get("mixed_domain") is not True:
        return False, "dataset summary is not mixed-domain"
    if summary.get("tile_size") != dataset_cfg["tile_size"]:
        return False, "dataset tile_size changed"
    if summary.get("tile_overlap") != dataset_cfg["tile_overlap"]:
        return False, "dataset tile_overlap changed"
    if summary.get("label_smoothing") != expected_smoothing:
        return False, "label smoothing config changed"

    for split_name, tile_names in splits.items():
        if not tile_names:
            continue
        tile_dir = os.path.join(structure_cfg["tiles_dir"], split_name)
        label_dir = os.path.join(structure_cfg["labels_dir"], split_name)
        for tile_name in [tile_names[0], tile_names[-1]]:
            if not os.path.exists(os.path.join(tile_dir, tile_name)):
                return False, f"missing tile file for {split_name}: {tile_name}"
            if not os.path.exists(os.path.join(label_dir, tile_name)):
                return False, f"missing label file for {split_name}: {tile_name}"

    return True, "current"


def merge_area_stacks(
    train_artifacts, test_artifacts, output_dir: str, force_recreate: bool = False
) -> Tuple[str, str, str, Dict]:
    """
    Merge training and test area feature stacks into a single mosaic.

    Returns:
        (merged_features_path, merged_labels_path, merged_mask_path, metadata)
    """
    merged_features_path = os.path.join(output_dir, "merged_features.tif")
    merged_labels_path = os.path.join(output_dir, "merged_labels.tif")
    merged_mask_path = os.path.join(output_dir, "merged_mask.tif")
    metadata_path = os.path.join(output_dir, "merged_metadata.json")

    merge_current, merge_reason = merge_artifacts_current(
        merged_features_path,
        merged_labels_path,
        merged_mask_path,
        metadata_path,
    )
    if merge_current and not force_recreate:
        debug_print("Merged stacks already exist, loading metadata")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return merged_features_path, merged_labels_path, merged_mask_path, metadata
    elif not force_recreate:
        debug_print(f"Merged stacks are stale ({merge_reason}); regenerating")

    debug_print("Merging training and test area stacks...")

    # Load training area
    with rasterio.open(train_artifacts.feature_stack_path) as src:
        train_features = src.read()
        train_profile = src.profile.copy()
        train_transform = src.transform
        train_crs = src.crs
        debug_print(f"Training area: {train_features.shape}, bounds={src.bounds}")

    with rasterio.open(train_artifacts.ground_truth_path) as src:
        train_labels = src.read(1)

    with rasterio.open(train_artifacts.mask_path) as src:
        train_mask = src.read(1)

    # Load test area
    with rasterio.open(test_artifacts.feature_stack_path) as src:
        test_features = src.read()
        test_profile = src.profile.copy()
        test_transform = src.transform
        test_crs = src.crs
        debug_print(f"Test area: {test_features.shape}, bounds={src.bounds}")

    with rasterio.open(test_artifacts.ground_truth_path) as src:
        test_labels = src.read(1)

    with rasterio.open(test_artifacts.mask_path) as src:
        test_mask = src.read(1)

    # §2.1 fix: Validate CRS match — raise explicit error instead of silently continuing
    if train_crs != test_crs:
        raise ValueError(
            f"CRS mismatch between train ({train_crs}) and test ({test_crs}) areas. "
            f"Reproject one of the inputs so both share the same CRS before running "
            f"the mixed-domain pipeline. Vertical concatenation requires identical CRS."
        )

    # Also validate resolution match
    train_res = (abs(train_transform.a), abs(train_transform.e))
    test_res = (abs(test_transform.a), abs(test_transform.e))
    res_tol = 1e-6
    if abs(train_res[0] - test_res[0]) > res_tol or abs(train_res[1] - test_res[1]) > res_tol:
        raise ValueError(
            f"Resolution mismatch between train ({train_res}) and test ({test_res}) areas. "
            f"Resample one of the inputs to match before running the mixed-domain pipeline."
        )

    # Check channel count
    if train_features.shape[0] != test_features.shape[0]:
        raise ValueError(
            f"Channel mismatch: train={train_features.shape[0]}, test={test_features.shape[0]}"
        )

    # Strategy: Stack vertically (concatenate along height dimension)
    debug_print("Concatenating areas vertically...")

    num_channels = train_features.shape[0]
    merged_height = train_features.shape[1] + test_features.shape[1]
    merged_width = max(train_features.shape[2], test_features.shape[2])

    # Pad narrower array with zeros
    if train_features.shape[2] < merged_width:
        pad_width = [(0, 0), (0, 0), (0, merged_width - train_features.shape[2])]
        train_features = np.pad(
            train_features, pad_width, mode="constant", constant_values=0
        )
        train_labels = np.pad(
            train_labels,
            [(0, 0), (0, merged_width - train_labels.shape[1])],
            mode="constant",
            constant_values=0,
        )
        train_mask = np.pad(
            train_mask,
            [(0, 0), (0, merged_width - train_mask.shape[1])],
            mode="constant",
            constant_values=0,
        )

    if test_features.shape[2] < merged_width:
        pad_width = [(0, 0), (0, 0), (0, merged_width - test_features.shape[2])]
        test_features = np.pad(
            test_features, pad_width, mode="constant", constant_values=0
        )
        test_labels = np.pad(
            test_labels,
            [(0, 0), (0, merged_width - test_labels.shape[1])],
            mode="constant",
            constant_values=0,
        )
        test_mask = np.pad(
            test_mask,
            [(0, 0), (0, merged_width - test_mask.shape[1])],
            mode="constant",
            constant_values=0,
        )

    # Concatenate
    merged_features = np.concatenate([train_features, test_features], axis=1)
    merged_labels = np.concatenate([train_labels, test_labels], axis=0)
    merged_mask = np.concatenate([train_mask, test_mask], axis=0)

    debug_print(
        f"Merged shape: features={merged_features.shape}, labels={merged_labels.shape}"
    )

    # Update transform for merged raster
    # Since we stacked vertically, adjust the transform
    merged_profile = train_profile.copy()
    merged_profile.update(
        {"height": merged_height, "width": merged_width, "count": num_channels}
    )

    # Save merged stacks
    debug_print(f"Saving merged features to {merged_features_path}")
    with rasterio.open(merged_features_path, "w", **merged_profile) as dst:
        dst.write(merged_features.astype(np.float32))

    label_profile = merged_profile.copy()
    label_profile.update({"count": 1, "dtype": "uint8", "nodata": 255})

    debug_print(f"Saving merged labels to {merged_labels_path}")
    with rasterio.open(merged_labels_path, "w", **label_profile) as dst:
        dst.write(merged_labels, 1)

    debug_print(f"Saving merged mask to {merged_mask_path}")
    with rasterio.open(merged_mask_path, "w", **label_profile) as dst:
        dst.write(merged_mask, 1)

    # Save metadata (§2.3 fix: include per-area transforms for correct GeoTIFF export)
    metadata = {
        "schema_version": MIXED_MERGE_SCHEMA_VERSION,
        "train_area": {
            "height": train_features.shape[1],
            "width": train_features.shape[2],
            "row_range": [0, train_features.shape[1]],
            "source": train_artifacts.feature_stack_path,
            "transform": list(train_transform)[:6],
        },
        "test_area": {
            "height": test_features.shape[1],
            "width": test_features.shape[2],
            "row_range": [train_features.shape[1], merged_height],
            "source": test_artifacts.feature_stack_path,
            "transform": list(test_transform)[:6],
        },
        "merged": {
            "height": merged_height,
            "width": merged_width,
            "channels": num_channels,
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    debug_print("✓ Merged stacks created successfully")

    return merged_features_path, merged_labels_path, merged_mask_path, metadata


def save_tile_as_geotiff(
    tile_array: np.ndarray,
    output_path: str,
    parent_transform: Affine,
    parent_crs,
    tile_y: int,
    tile_x: int,
    channel_names: Optional[List[str]] = None,
):
    """
    Save a tile as GeoTIFF with proper geospatial metadata.

    Args:
        tile_array: (channels, height, width) or (height, width)
        output_path: Path to save GeoTIFF
        parent_transform: Transform of parent raster
        parent_crs: CRS of parent raster
        tile_y, tile_x: Top-left coordinates of tile in parent raster
        channel_names: Optional list of channel names for descriptions
    """
    if tile_array.ndim == 2:
        tile_array = tile_array[np.newaxis, ...]

    channels, height, width = tile_array.shape

    # Calculate tile transform (offset from parent)
    tile_transform = parent_transform * Affine.translation(tile_x, tile_y)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": channels,
        "dtype": tile_array.dtype.name,
        "crs": parent_crs,
        "transform": tile_transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": min(256, width),
        "blockysize": min(256, height),
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(tile_array)
        if channel_names:
            for i, name in enumerate(channel_names[:channels], 1):
                dst.set_band_description(i, name)


def prepare_mixed_domain_dataset(
    config: Dict, train_artifacts, test_artifacts, force_recreate: bool = False
) -> None:
    """
    Prepare dataset from merged training and test areas.
    Exports tiles as both .npy (fast training) and .tif (spatial inspection).
    Supports soft label smoothing when configured (§6 fix).
    """
    debug_print("=" * 80)
    debug_print("PREPARING MIXED-DOMAIN DATASET")
    debug_print("=" * 80)

    dataset_cfg = config["dataset"]
    structure_cfg = config["project_structure"]

    # Check if dataset already exists
    splits_path = os.path.join(structure_cfg["splits_dir"], "splits.json")
    summary_path = os.path.join(structure_cfg["splits_dir"], "dataset_summary.json")

    if (
        os.path.exists(splits_path)
        and os.path.exists(summary_path)
        and not force_recreate
    ):
        dataset_current, dataset_reason = mixed_dataset_artifacts_current(
            splits_path, summary_path, structure_cfg, config
        )
        if dataset_current:
            debug_print("Dataset artifacts already exist, skipping")
            return
        debug_print(f"Dataset artifacts are stale ({dataset_reason}); regenerating")
    elif os.path.exists(splits_path) or os.path.exists(summary_path):
        debug_print("Partial dataset artifacts found, regenerating")

    debug_print(f"Preparing mixed-domain dataset (force_recreate={force_recreate})")

    # Step 1: Merge training and test areas
    merged_dir = os.path.join(structure_cfg["derived_data_dir"], "merged")
    os.makedirs(merged_dir, exist_ok=True)

    merged_features_path, merged_labels_path, merged_mask_path, merge_metadata = (
        merge_area_stacks(train_artifacts, test_artifacts, merged_dir, force_recreate)
    )

    # Step 2: Load merged stacks
    with rasterio.open(merged_features_path) as src:
        features = src.read().astype(np.float32)
        height, width = src.height, src.width
        profile = src.profile
        transform = src.transform
        crs = src.crs
        debug_print(f"Loaded merged features: {features.shape}")

    with rasterio.open(merged_labels_path) as src:
        labels = src.read(1).astype(np.uint8)

    with rasterio.open(merged_mask_path) as src:
        valid_mask = src.read(1).astype(bool)

    # Load channel names if available
    channel_names = None
    train_metadata_path = train_artifacts.feature_stack_path.replace(
        ".tif", "_metadata.json"
    )
    if os.path.exists(train_metadata_path):
        with open(train_metadata_path, "r") as f:
            metadata = json.load(f)
            channel_names = metadata.get("channel_names", None)

    # Apply label smoothing if enabled (§6 fix: was missing from mixed-domain path)
    label_smoothing_cfg = config["preprocessing"].get("label_smoothing", {})
    use_soft_labels = label_smoothing_cfg.get("enabled", False)
    soft_labels = None

    if use_soft_labels:
        from src.soft_labels import apply_label_smoothing, validate_soft_labels

        smoothing_type = label_smoothing_cfg.get("type", "ordinal")
        alpha = label_smoothing_cfg.get("alpha", 0.1)
        sigma = label_smoothing_cfg.get("sigma", 1.0)
        num_classes = config["model"]["out_classes"]

        debug_print(
            f"Applying {smoothing_type} label smoothing "
            f"(alpha={alpha}, sigma={sigma}, num_classes={num_classes})"
        )

        soft_labels = apply_label_smoothing(
            labels,
            smoothing_type=smoothing_type,
            num_classes=num_classes,
            alpha=alpha,
            sigma=sigma,
            ignore_value=255,
        )

        try:
            validate_soft_labels(soft_labels)
            debug_print("Soft labels validated successfully")
        except AssertionError as e:
            debug_print(f"WARNING - soft label validation failed: {e}")
    else:
        debug_print("Using hard labels (label smoothing disabled)")

    # Step 3: Create tile directories
    tile_size = dataset_cfg["tile_size"]
    tile_overlap = dataset_cfg["tile_overlap"]
    stride = max(1, tile_size - tile_overlap)
    min_valid_fraction = dataset_cfg.get("min_valid_fraction", 0.8)

    os.makedirs(structure_cfg["tiles_dir"], exist_ok=True)
    os.makedirs(structure_cfg["labels_dir"], exist_ok=True)
    os.makedirs(structure_cfg["splits_dir"], exist_ok=True)

    # Create subdirectories for .tif exports
    tif_tiles_dir = os.path.join(structure_cfg["tiles_dir"], "geotiff")
    tif_labels_dir = os.path.join(structure_cfg["labels_dir"], "geotiff")
    os.makedirs(tif_tiles_dir, exist_ok=True)
    os.makedirs(tif_labels_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(structure_cfg["tiles_dir"], split), exist_ok=True)
        os.makedirs(os.path.join(structure_cfg["labels_dir"], split), exist_ok=True)
        os.makedirs(os.path.join(tif_tiles_dir, split), exist_ok=True)
        os.makedirs(os.path.join(tif_labels_dir, split), exist_ok=True)

        # Clean existing files
        for path in glob.glob(os.path.join(structure_cfg["tiles_dir"], split, "*.npy")):
            os.remove(path)
        for path in glob.glob(
            os.path.join(structure_cfg["labels_dir"], split, "*.npy")
        ):
            os.remove(path)
        for path in glob.glob(os.path.join(tif_tiles_dir, split, "*.tif")):
            os.remove(path)
        for path in glob.glob(os.path.join(tif_labels_dir, split, "*.tif")):
            os.remove(path)

    # Step 4: Generate candidate tile positions
    debug_print(f"Generating candidate tiles (size={tile_size}, stride={stride})...")

    candidate_positions = []
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile_mask = valid_mask[y : y + tile_size, x : x + tile_size]
            if tile_mask.mean() >= min_valid_fraction:
                tile_labels = labels[y : y + tile_size, x : x + tile_size]
                # Check for valid labelled pixels (not nodata=255).
                # Previously used `tile_labels != 0` which dropped pure Low-risk
                # tiles (class 0) — introducing systematic positive bias (§1.3).
                valid_pixels = tile_labels != 255
                if np.any(valid_pixels):
                    candidate_positions.append((y, x))

    debug_print(f"Found {len(candidate_positions)} valid candidate tiles")

    # Step 5: Label each tile with source area (train or test)
    train_row_end = merge_metadata["train_area"]["row_range"][1]

    train_area_tiles = []
    test_area_tiles = []

    for y, x in candidate_positions:
        tile_center_y = y + tile_size // 2
        if tile_center_y < train_row_end:
            train_area_tiles.append((y, x))
        else:
            test_area_tiles.append((y, x))

    debug_print(
        f"Tiles by area: train={len(train_area_tiles)}, test={len(test_area_tiles)}"
    )

    # Step 6: Spatial block split to prevent data leakage
    # CRITICAL: Adjacent tiles MUST be in same split to avoid spatial autocorrelation
    test_size = dataset_cfg.get("test_size", 0.2)
    val_size = dataset_cfg.get("val_size", 0.15)
    random_state = dataset_cfg.get("random_state", 42)
    rng = random.Random(config["reproducibility"]["seed"])

    # Define spatial block size (minimum distance between train/test tiles)
    block_size = dataset_cfg.get(
        "spatial_block_size", 5
    )  # 5 tiles = 5*128 = 640 pixels minimum separation

    debug_print(
        f"Using spatial block splitting (block_size={block_size} tiles, min separation={block_size * stride} pixels)"
    )

    # Split each area separately, then combine
    def split_area_tiles_spatially(area_tiles, test_fraction, val_fraction, seed):
        """
        Split tiles using spatial blocking to prevent data leakage.
        Adjacent tiles are grouped into blocks and blocks are split, not individual tiles.
        """
        if len(area_tiles) < 3:
            # Too few tiles, put all in train
            return area_tiles, [], []

        # Group tiles into spatial blocks
        # Sort by Y coordinate, then X coordinate
        sorted_tiles = sorted(area_tiles, key=lambda t: (t[0], t[1]))

        # Create spatial blocks (groups of nearby tiles)
        blocks = []
        current_block = []

        for i, (y, x) in enumerate(sorted_tiles):
            if not current_block:
                current_block.append((y, x))
            else:
                # Check if this tile is close to the block (within block_size tiles)
                last_y, last_x = current_block[-1]
                distance_tiles = max(
                    abs(y - last_y) // stride, abs(x - last_x) // stride
                )

                if distance_tiles < block_size:
                    # Close enough, add to current block
                    current_block.append((y, x))
                else:
                    # Too far, start new block
                    blocks.append(current_block)
                    current_block = [(y, x)]

        # Add last block
        if current_block:
            blocks.append(current_block)

        debug_print(
            f"  Created {len(blocks)} spatial blocks from {len(area_tiles)} tiles"
        )

        # Now split blocks (not individual tiles)
        if len(blocks) < 3:
            # Too few blocks, put all tiles in train
            all_tiles = [tile for block in blocks for tile in block]
            return all_tiles, [], []

        # Shuffle blocks randomly
        random.Random(seed).shuffle(blocks)

        # Split blocks into train/test
        n_test_blocks = max(1, int(len(blocks) * test_fraction))
        test_blocks = blocks[:n_test_blocks]
        remaining_blocks = blocks[n_test_blocks:]

        if len(remaining_blocks) < 2 or val_fraction <= 0:
            train_blocks = remaining_blocks
            val_blocks = []
        else:
            n_val_blocks = max(
                1, int(len(remaining_blocks) * val_fraction / (1.0 - test_fraction))
            )
            val_blocks = remaining_blocks[:n_val_blocks]
            train_blocks = remaining_blocks[n_val_blocks:]

        # Flatten blocks back to tile lists
        area_train = [tile for block in train_blocks for tile in block]
        area_val = [tile for block in val_blocks for tile in block]
        area_test = [tile for block in test_blocks for tile in block]

        debug_print(
            f"  Split into {len(train_blocks)} train, {len(val_blocks)} val, {len(test_blocks)} test blocks"
        )

        return area_train, area_val, area_test

    (
        train_tiles_from_train_area,
        val_tiles_from_train_area,
        test_tiles_from_train_area,
    ) = split_area_tiles_spatially(train_area_tiles, test_size, val_size, random_state)

    train_tiles_from_test_area, val_tiles_from_test_area, test_tiles_from_test_area = (
        split_area_tiles_spatially(
            test_area_tiles, test_size, val_size, random_state + 100
        )
    )

    # Combine splits
    train_tiles = train_tiles_from_train_area + train_tiles_from_test_area
    val_tiles = val_tiles_from_train_area + val_tiles_from_test_area
    test_tiles = test_tiles_from_train_area + test_tiles_from_test_area

    rng.shuffle(train_tiles)
    rng.shuffle(val_tiles)
    rng.shuffle(test_tiles)

    debug_print(
        f"Split: train={len(train_tiles)}, val={len(val_tiles)}, test={len(test_tiles)}"
    )
    debug_print(
        f"  Train from train_area={len(train_tiles_from_train_area)}, test_area={len(train_tiles_from_test_area)}"
    )
    debug_print(
        f"  Val from train_area={len(val_tiles_from_train_area)}, test_area={len(val_tiles_from_test_area)}"
    )
    debug_print(
        f"  Test from train_area={len(test_tiles_from_train_area)}, test_area={len(test_tiles_from_test_area)}"
    )

    # Step 7: §3 fix — Recompute normalization stats from TRAINING tiles only
    # The features array was normalized using full train-area stats (computed before
    # the spatial split). This leaks val/test tile information into the normalisation.
    # Fix: recompute stats from training tiles only, apply linear correction.
    normalization_stats_path = os.path.join(
        config["project_structure"]["metadata_dir"], "normalization_stats.json"
    )
    if os.path.exists(normalization_stats_path) and len(train_tiles) > 0:
        debug_print("§3 fix: Recomputing normalization stats from training tiles only...")
        with open(normalization_stats_path, "r") as f:
            old_stats = json.load(f)

        old_mean = np.array(old_stats["mean"], dtype=np.float32)
        old_std = np.array(old_stats["std"], dtype=np.float32)
        epsilon = old_stats.get("epsilon", 1e-7)
        num_ch = features.shape[0]

        # Accumulate per-channel stats from training tiles only
        ch_sum = np.zeros(num_ch, dtype=np.float64)
        ch_sq_sum = np.zeros(num_ch, dtype=np.float64)
        ch_count = np.zeros(num_ch, dtype=np.float64)

        for y, x in train_tiles:
            tile_f = features[:, y : y + tile_size, x : x + tile_size]
            tile_m = valid_mask[y : y + tile_size, x : x + tile_size]
            if not np.any(tile_m):
                continue
            # Denormalize back to raw values: raw = normalized * std + mean
            for c in range(num_ch):
                vals = tile_f[c][tile_m].astype(np.float64)
                raw_vals = vals * old_std[c] + old_mean[c]
                ch_sum[c] += raw_vals.sum()
                ch_sq_sum[c] += (raw_vals ** 2).sum()
                ch_count[c] += raw_vals.size

        # Compute clean stats
        new_mean = np.where(ch_count > 0, ch_sum / ch_count, old_mean.astype(np.float64))
        new_var = np.where(
            ch_count > 1,
            ch_sq_sum / ch_count - new_mean ** 2,
            old_std.astype(np.float64) ** 2,
        )
        new_std = np.sqrt(np.maximum(new_var, 0.0))
        new_std = np.where(new_std < epsilon, epsilon, new_std)

        # Log the delta
        mean_delta = np.abs(new_mean - old_mean.astype(np.float64))
        std_delta = np.abs(new_std - old_std.astype(np.float64))
        debug_print(f"  Max mean delta: {mean_delta.max():.6f} (channel {np.argmax(mean_delta)})")
        debug_print(f"  Max std delta:  {std_delta.max():.6f} (channel {np.argmax(std_delta)})")

        # Apply linear correction to the entire features array in-place:
        # new_normalized = (raw - new_mean) / new_std
        #                = ((old_normalized * old_std + old_mean) - new_mean) / new_std
        #                = old_normalized * (old_std / new_std) + (old_mean - new_mean) / new_std
        new_mean_f32 = new_mean.astype(np.float32)
        new_std_f32 = new_std.astype(np.float32)
        for c in range(num_ch):
            scale = old_std[c] / new_std_f32[c]
            shift = (old_mean[c] - new_mean_f32[c]) / new_std_f32[c]
            features[c] = features[c] * scale + shift

        # Save clean stats (overwrite old file)
        clean_stats = {
            "strategy": old_stats.get("strategy", "zscore"),
            "mean": new_mean_f32.tolist(),
            "std": new_std_f32.tolist(),
            "epsilon": epsilon,
            "channel_names": old_stats.get("channel_names", []),
            "note": "Recomputed from training tiles only (§3 fix)",
        }
        with open(normalization_stats_path, "w") as f:
            json.dump(clean_stats, f, indent=2)
        debug_print(f"  Saved clean normalization stats to {normalization_stats_path}")
    else:
        debug_print("Skipping normalization stat recomputation (no stats file or no training tiles)")

    # Step 8: Save tiles
    # §2.3 fix: Use correct per-area transforms for GeoTIFF exports
    train_area_height = merge_metadata["train_area"]["height"]
    train_area_transform = transform  # The merged transform == train area's transform
    test_area_transform_raw = merge_metadata.get("test_area", {}).get("transform")
    if test_area_transform_raw:
        test_area_transform = Affine(*test_area_transform_raw)
        debug_print("§2.3 fix: Using per-area transforms for GeoTIFF tile export")
    else:
        test_area_transform = None
        debug_print("§2.3 fix: No test area transform in metadata, using merged transform")

    split_positions = {"train": train_tiles, "val": val_tiles, "test": test_tiles}

    tile_records = {"train": [], "val": [], "test": []}
    class_pixel_counts = {}

    for split_name, positions in split_positions.items():
        debug_print(f"Saving {split_name} tiles ({len(positions)} tiles)...")

        npy_tile_dir = os.path.join(structure_cfg["tiles_dir"], split_name)
        npy_label_dir = os.path.join(structure_cfg["labels_dir"], split_name)
        tif_tile_dir = os.path.join(tif_tiles_dir, split_name)
        tif_label_dir = os.path.join(tif_labels_dir, split_name)

        for idx, (y, x) in enumerate(positions):
            tile_features = features[:, y : y + tile_size, x : x + tile_size]
            tile_labels = labels[y : y + tile_size, x : x + tile_size].copy()
            tile_mask = valid_mask[y : y + tile_size, x : x + tile_size]

            tile_labels[~tile_mask] = 255

            if not np.any(tile_labels != 255):
                continue

            tile_name = f"{split_name}_tile_{y:06d}_{x:06d}.npy"
            tif_name = f"{split_name}_tile_{y:06d}_{x:06d}.tif"

            # Save as .npy for training
            np.save(
                os.path.join(npy_tile_dir, tile_name), tile_features.astype(np.float32)
            )

            # Save either soft or hard labels (§6 fix)
            if use_soft_labels and soft_labels is not None:
                # Extract soft label tile: shape (num_classes, tile_h, tile_w)
                tile_soft_labels = soft_labels[
                    :, y : y + tile_size, x : x + tile_size
                ].copy()
                # Zero out invalid pixels in all class channels
                tile_soft_labels[:, ~tile_mask] = 0.0
                np.save(
                    os.path.join(npy_label_dir, tile_name),
                    tile_soft_labels.astype(np.float32),
                )
            else:
                np.save(
                    os.path.join(npy_label_dir, tile_name),
                    tile_labels.astype(np.uint8),
                )

            # Save as .tif for inspection
            # §2.3 fix: Use correct area transform for GeoTIFF metadata
            if y >= train_area_height and test_area_transform is not None:
                # Tile is from test area — use test area's transform with adjusted row
                tile_geo_transform = test_area_transform
                tile_geo_y = y - train_area_height
            else:
                # Tile is from train area — use train area's (merged) transform
                tile_geo_transform = train_area_transform
                tile_geo_y = y

            save_tile_as_geotiff(
                tile_features,
                os.path.join(tif_tile_dir, tif_name),
                tile_geo_transform,
                crs,
                tile_geo_y,
                x,
                channel_names,
            )

            save_tile_as_geotiff(
                tile_labels,
                os.path.join(tif_label_dir, tif_name),
                tile_geo_transform,
                crs,
                tile_geo_y,
                x,
                ["labels"],
            )

            tile_records[split_name].append(tile_name)

            # Count pixels per class (works for both hard and soft labels)
            valid_pixels = tile_labels != 255
            if np.any(valid_pixels):
                if use_soft_labels and soft_labels is not None:
                    # For soft labels, count fractional class membership
                    tile_soft_valid = soft_labels[
                        :, y : y + tile_size, x : x + tile_size
                    ][:, valid_pixels]
                    num_classes = config["model"]["out_classes"]
                    for cls_idx in range(num_classes):
                        count = float(np.sum(tile_soft_valid[cls_idx]))
                        class_pixel_counts[str(cls_idx)] = (
                            class_pixel_counts.get(str(cls_idx), 0) + count
                        )
                else:
                    # For hard labels, count discrete assignments
                    unique, counts = np.unique(
                        tile_labels[valid_pixels], return_counts=True
                    )
                    for cls, count in zip(unique, counts):
                        class_pixel_counts[str(int(cls))] = class_pixel_counts.get(
                            str(int(cls)), 0
                        ) + int(count)

    # Step 9: Save splits and summary
    splits_json = os.path.join(structure_cfg["splits_dir"], "splits.json")
    with open(splits_json, "w") as f:
        json.dump(tile_records, f, indent=2)

    dataset_summary = {
        "schema_version": MIXED_DATASET_SCHEMA_VERSION,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "tile_counts": {split: len(names) for split, names in tile_records.items()},
        "class_pixel_counts": class_pixel_counts,
        "mixed_domain": True,
        "label_smoothing": {
            "enabled": use_soft_labels,
            "type": label_smoothing_cfg.get("type", "none") if use_soft_labels else "none",
            "alpha": label_smoothing_cfg.get("alpha", 0.0) if use_soft_labels else 0.0,
            "sigma": label_smoothing_cfg.get("sigma", 0.0) if use_soft_labels else 0.0,
        },
        "train_area_contribution": {
            "train": len(train_tiles_from_train_area),
            "val": len(val_tiles_from_train_area),
            "test": len(test_tiles_from_train_area),
        },
        "test_area_contribution": {
            "train": len(train_tiles_from_test_area),
            "val": len(val_tiles_from_test_area),
            "test": len(test_tiles_from_test_area),
        },
        "geotiff_exports": {
            "features": os.path.join(tif_tiles_dir, "{split}/"),
            "labels": os.path.join(tif_labels_dir, "{split}/"),
        },
    }

    summary_json = os.path.join(structure_cfg["splits_dir"], "dataset_summary.json")
    with open(summary_json, "w") as f:
        json.dump(dataset_summary, f, indent=2)

    debug_print("=" * 80)
    debug_print("MIXED-DOMAIN DATASET PREPARATION COMPLETE")
    debug_print(f"  Total tiles: {sum(len(v) for v in tile_records.values())}")
    debug_print(
        f"  Train: {len(tile_records['train'])} (train_area={len(train_tiles_from_train_area)}, test_area={len(train_tiles_from_test_area)})"
    )
    debug_print(
        f"  Val: {len(tile_records['val'])} (train_area={len(val_tiles_from_train_area)}, test_area={len(val_tiles_from_test_area)})"
    )
    debug_print(
        f"  Test: {len(tile_records['test'])} (train_area={len(test_tiles_from_train_area)}, test_area={len(test_tiles_from_test_area)})"
    )
    debug_print(f"  GeoTIFF tiles saved to: {tif_tiles_dir}")
    debug_print("=" * 80)
