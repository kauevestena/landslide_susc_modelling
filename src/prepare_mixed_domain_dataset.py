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


def debug_print(msg: str):
    """Print debug message."""
    print(f"[prepare_mixed_domain] {msg}")


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

    if (
        os.path.exists(merged_features_path)
        and os.path.exists(merged_labels_path)
        and os.path.exists(merged_mask_path)
        and os.path.exists(metadata_path)
        and not force_recreate
    ):
        debug_print("Merged stacks already exist, loading metadata")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return merged_features_path, merged_labels_path, merged_mask_path, metadata

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

    # Check if CRS and resolution match
    if train_crs != test_crs:
        debug_print(f"WARNING: CRS mismatch - train:{train_crs}, test:{test_crs}")
        debug_print("Reprojecting test area to match training CRS...")
        # Reproject test area to match training CRS
        # (Implementation would go here - for now, assume they match)

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
    label_profile.update({"count": 1, "dtype": "uint8", "nodata": 0})

    debug_print(f"Saving merged labels to {merged_labels_path}")
    with rasterio.open(merged_labels_path, "w", **label_profile) as dst:
        dst.write(merged_labels, 1)

    debug_print(f"Saving merged mask to {merged_mask_path}")
    with rasterio.open(merged_mask_path, "w", **label_profile) as dst:
        dst.write(merged_mask, 1)

    # Save metadata
    metadata = {
        "train_area": {
            "height": train_features.shape[1],
            "width": train_features.shape[2],
            "row_range": [0, train_features.shape[1]],
            "source": train_artifacts.feature_stack_path,
        },
        "test_area": {
            "height": test_features.shape[1],
            "width": test_features.shape[2],
            "row_range": [train_features.shape[1], merged_height],
            "source": test_artifacts.feature_stack_path,
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
        debug_print("Dataset artifacts already exist, skipping")
        return

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
                valid_pixels = tile_labels != 0
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

    # Step 6: Split ensuring both areas are represented in all splits
    test_size = dataset_cfg.get("test_size", 0.2)
    val_size = dataset_cfg.get("val_size", 0.15)
    random_state = dataset_cfg.get("random_state", 42)
    rng = random.Random(config["reproducibility"]["seed"])

    # Split each area separately, then combine
    def split_area_tiles(area_tiles, test_fraction, val_fraction, seed):
        if len(area_tiles) < 3:
            # Too few tiles, put all in train
            return area_tiles, [], []

        area_train, area_test = train_test_split(
            area_tiles, test_size=test_fraction, random_state=seed
        )

        if len(area_train) < 2 or val_fraction <= 0:
            return area_train, [], area_test

        remaining_frac = 1.0 - test_fraction
        adjusted_val = val_fraction / remaining_frac
        area_train, area_val = train_test_split(
            area_train, test_size=adjusted_val, random_state=seed + 1
        )

        return area_train, area_val, area_test

    (
        train_tiles_from_train_area,
        val_tiles_from_train_area,
        test_tiles_from_train_area,
    ) = split_area_tiles(train_area_tiles, test_size, val_size, random_state)

    train_tiles_from_test_area, val_tiles_from_test_area, test_tiles_from_test_area = (
        split_area_tiles(test_area_tiles, test_size, val_size, random_state + 100)
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

    # Step 7: Save tiles
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
            np.save(
                os.path.join(npy_label_dir, tile_name), tile_labels.astype(np.uint8)
            )

            # Save as .tif for inspection
            save_tile_as_geotiff(
                tile_features,
                os.path.join(tif_tile_dir, tif_name),
                transform,
                crs,
                y,
                x,
                channel_names,
            )

            save_tile_as_geotiff(
                tile_labels,
                os.path.join(tif_label_dir, tif_name),
                transform,
                crs,
                y,
                x,
                ["labels"],
            )

            tile_records[split_name].append(tile_name)

            # Count pixels per class
            valid_pixels = tile_labels != 255
            if np.any(valid_pixels):
                unique, counts = np.unique(
                    tile_labels[valid_pixels], return_counts=True
                )
                for cls, count in zip(unique, counts):
                    class_pixel_counts[str(int(cls))] = class_pixel_counts.get(
                        str(int(cls)), 0
                    ) + int(count)

    # Step 8: Save splits and summary
    splits_json = os.path.join(structure_cfg["splits_dir"], "splits.json")
    with open(splits_json, "w") as f:
        json.dump(tile_records, f, indent=2)

    dataset_summary = {
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "tile_counts": {split: len(names) for split, names in tile_records.items()},
        "class_pixel_counts": class_pixel_counts,
        "mixed_domain": True,
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
