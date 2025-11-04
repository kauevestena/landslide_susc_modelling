#!/usr/bin/env python3
"""
Validate that train/val/test splits have proper spatial separation.
Checks for data leakage by measuring minimum distances between splits.
"""

import json
import numpy as np
from pathlib import Path


def parse_tile_coords(tile_name):
    """Extract Y, X coordinates from tile filename."""
    # Format: train_tile_000512_001024.npy
    parts = tile_name.replace(".npy", "").split("_")
    y = int(parts[2])
    x = int(parts[3])
    return y, x


def compute_min_distance_between_splits(split1_tiles, split2_tiles, stride=128):
    """
    Compute minimum distance (in pixels) between two splits.
    Returns distance in pixels and in number of tiles.
    """
    min_distance_pixels = float("inf")
    closest_pair = (None, None)

    for tile1 in split1_tiles:
        y1, x1 = parse_tile_coords(tile1)

        for tile2 in split2_tiles:
            y2, x2 = parse_tile_coords(tile2)

            # Euclidean distance between tile centers
            distance = np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)

            if distance < min_distance_pixels:
                min_distance_pixels = distance
                closest_pair = (tile1, tile2)

    # Convert to tile units
    distance_tiles = min_distance_pixels / stride

    return min_distance_pixels, distance_tiles, closest_pair


def validate_spatial_split(splits_path="artifacts/splits/splits.json", stride=128):
    """Validate spatial separation between train/val/test splits."""

    print("=" * 80)
    print("SPATIAL SPLIT VALIDATION")
    print("=" * 80)
    print()

    # Load splits
    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_tiles = splits["train"]
    val_tiles = splits.get("val", [])
    test_tiles = splits["test"]

    print(f"Split sizes:")
    print(f"  Train: {len(train_tiles)} tiles")
    print(f"  Val:   {len(val_tiles)} tiles")
    print(f"  Test:  {len(test_tiles)} tiles")
    print()

    # Check train-test separation
    print("Checking Train-Test separation...")
    dist_px, dist_tiles, pair = compute_min_distance_between_splits(
        train_tiles, test_tiles, stride
    )
    print(f"  Minimum distance: {dist_px:.1f} pixels ({dist_tiles:.2f} tiles)")
    print(f"  Closest pair:")
    print(f"    Train: {pair[0]}")
    print(f"    Test:  {pair[1]}")

    if dist_tiles < 3:
        print(f"  ⚠️  WARNING: Very close tiles! Risk of data leakage.")
    elif dist_tiles < 5:
        print(f"  ⚠️  CAUTION: Moderately close. Some spatial correlation may exist.")
    else:
        print(f"  ✅ GOOD: Sufficient spatial separation.")
    print()

    # Check train-val separation
    if val_tiles:
        print("Checking Train-Val separation...")
        dist_px, dist_tiles, pair = compute_min_distance_between_splits(
            train_tiles, val_tiles, stride
        )
        print(f"  Minimum distance: {dist_px:.1f} pixels ({dist_tiles:.2f} tiles)")
        print(f"  Closest pair:")
        print(f"    Train: {pair[0]}")
        print(f"    Val:   {pair[1]}")

        if dist_tiles < 3:
            print(f"  ⚠️  WARNING: Very close tiles! Risk of data leakage.")
        elif dist_tiles < 5:
            print(
                f"  ⚠️  CAUTION: Moderately close. Some spatial correlation may exist."
            )
        else:
            print(f"  ✅ GOOD: Sufficient spatial separation.")
        print()

    # Check val-test separation
    if val_tiles:
        print("Checking Val-Test separation...")
        dist_px, dist_tiles, pair = compute_min_distance_between_splits(
            val_tiles, test_tiles, stride
        )
        print(f"  Minimum distance: {dist_px:.1f} pixels ({dist_tiles:.2f} tiles)")
        print(f"  Closest pair:")
        print(f"    Val:  {pair[0]}")
        print(f"    Test: {pair[1]}")

        if dist_tiles < 3:
            print(f"  ⚠️  WARNING: Very close tiles! Risk of data leakage.")
        elif dist_tiles < 5:
            print(
                f"  ⚠️  CAUTION: Moderately close. Some spatial correlation may exist."
            )
        else:
            print(f"  ✅ GOOD: Sufficient spatial separation.")
        print()

    # Visualize split distribution
    print("Split distribution in merged raster:")
    all_tiles = train_tiles + val_tiles + test_tiles
    coords = [parse_tile_coords(t) for t in all_tiles]
    ys, xs = zip(*coords)

    min_y, max_y = min(ys), max(ys)
    min_x, max_x = min(xs), max(xs)

    print(f"  Y range: {min_y} to {max_y} (span: {max_y - min_y} pixels)")
    print(f"  X range: {min_x} to {max_x} (span: {max_x - min_x} pixels)")
    print()

    # Check for train/test area mixing
    train_area_split = 1574  # From merged metadata (row where test area starts)
    train_from_train_area = sum(
        1 for t in train_tiles if parse_tile_coords(t)[0] < train_area_split
    )
    train_from_test_area = len(train_tiles) - train_from_train_area
    test_from_train_area = sum(
        1 for t in test_tiles if parse_tile_coords(t)[0] < train_area_split
    )
    test_from_test_area = len(test_tiles) - test_from_train_area

    print("Mixed-domain distribution:")
    print(f"  Train split:")
    print(f"    From training area: {train_from_train_area} tiles")
    print(f"    From test area:     {train_from_test_area} tiles")
    print(f"  Test split:")
    print(f"    From training area: {test_from_train_area} tiles")
    print(f"    From test area:     {test_from_test_area} tiles")
    print()

    if train_from_test_area == 0 or test_from_test_area == 0:
        print("  ⚠️  WARNING: One area not represented in training or test!")
    else:
        print("  ✅ GOOD: Both areas represented in train and test splits.")

    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    splits_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/splits/splits.json"

    if not Path(splits_path).exists():
        print(f"ERROR: Splits file not found: {splits_path}")
        print("Run the pipeline first to generate splits.")
        sys.exit(1)

    validate_spatial_split(splits_path)
