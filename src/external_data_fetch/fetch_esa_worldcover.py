"""
Fetch and process ESA WorldCover (10m) land cover data for a given area of interest.

ESA WorldCover provides global land cover maps at 10m resolution with 11 classes:
- 10: Tree cover
- 20: Shrubland
- 30: Grassland
- 40: Cropland
- 50: Built-up
- 60: Bare / sparse vegetation
- 70: Snow and ice
- 80: Permanent water bodies
- 90: Herbaceous wetland
- 95: Mangroves
- 100: Moss and lichen

Documentation: https://esa-worldcover.org/en
API: https://services.terrascope.be/wms/v2
"""

import os
from typing import Tuple, Optional
import numpy as np
import rasterio
from rasterio.warp import reproject, calculate_default_transform, Resampling
from rasterio.transform import Affine
from rasterio.crs import CRS
import requests
from pathlib import Path


def get_worldcover_bbox(reference_raster: str) -> Tuple[float, float, float, float]:
    """
    Extract bounding box from reference raster in WGS84 (EPSG:4326).

    Args:
        reference_raster: Path to DTM or other georeferenced raster

    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat)
    """
    with rasterio.open(reference_raster) as src:
        bounds = src.bounds
        src_crs = src.crs

        # Transform to WGS84 if needed
        if src_crs != CRS.from_epsg(4326):
            from rasterio.warp import transform_bounds

            bounds = transform_bounds(src_crs, CRS.from_epsg(4326), *bounds)

    return bounds.left, bounds.bottom, bounds.right, bounds.top


def download_worldcover_tile(
    bbox: Tuple[float, float, float, float],
    output_path: str,
    year: int = 2021,
    width: int = 2048,
    height: int = 2048,
) -> bool:
    """
    Download ESA WorldCover tile via WMS for the specified bounding box.

    Args:
        bbox: Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84
        output_path: Path to save the downloaded GeoTIFF
        year: Year of WorldCover product (2020 or 2021)
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        True if download successful, False otherwise
    """
    base_url = "https://services.terrascope.be/wms/v2"

    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": f"WORLDCOVER_{year}_MAP",
        "bbox": f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",  # WMS 1.3.0 uses lat,lon order
        "crs": "EPSG:4326",
        "width": width,
        "height": height,
        "format": "image/geotiff",
    }

    try:
        print(f"[fetch_esa_worldcover] Downloading WorldCover {year} for bbox {bbox}")
        response = requests.get(base_url, params=params, timeout=120)
        response.raise_for_status()

        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"[fetch_esa_worldcover] Downloaded to {output_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"[fetch_esa_worldcover] Download failed: {e}")
        return False


def reproject_worldcover_to_reference(
    worldcover_path: str,
    reference_raster: str,
    output_path: str,
    resampling_method: Resampling = Resampling.nearest,
) -> bool:
    """
    Reproject and align WorldCover data to match reference raster grid.

    Args:
        worldcover_path: Path to downloaded WorldCover GeoTIFF
        reference_raster: Path to reference raster (typically DTM)
        output_path: Path to save reprojected GeoTIFF
        resampling_method: Resampling method (nearest for categorical data)

    Returns:
        True if reprojection successful, False otherwise
    """
    try:
        print(
            f"[fetch_esa_worldcover] Reprojecting WorldCover to match {reference_raster}"
        )

        with rasterio.open(reference_raster) as ref_src:
            ref_profile = ref_src.profile.copy()
            ref_transform = ref_src.transform
            ref_crs = ref_src.crs
            ref_shape = (ref_src.height, ref_src.width)

        with rasterio.open(worldcover_path) as wc_src:
            # Prepare output array
            lulc_data = np.zeros(ref_shape, dtype=np.uint8)

            # Reproject
            reproject(
                source=rasterio.band(wc_src, 1),
                destination=lulc_data,
                src_transform=wc_src.transform,
                src_crs=wc_src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling_method,
                src_nodata=0,
                dst_nodata=0,
            )

        # Write reprojected data
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ref_profile.update(dtype=rasterio.uint8, count=1, nodata=0)

        with rasterio.open(output_path, "w", **ref_profile) as dst:
            dst.write(lulc_data, 1)

        print(f"[fetch_esa_worldcover] Reprojected to {output_path}")
        return True

    except Exception as e:
        print(f"[fetch_esa_worldcover] Reprojection failed: {e}")
        return False


def one_hot_encode_worldcover(
    lulc_raster: str,
    output_path: str,
    class_mapping: Optional[dict] = None,
) -> Tuple[np.ndarray, dict]:
    """
    One-hot encode WorldCover classes into binary channels.

    Args:
        lulc_raster: Path to reprojected WorldCover GeoTIFF
        output_path: Path to save one-hot encoded multi-band GeoTIFF
        class_mapping: Optional custom class mapping for aggregation

    Returns:
        Tuple of (one_hot_array, class_info_dict)
    """
    # Default WorldCover classes
    default_classes = {
        10: "tree_cover",
        20: "shrubland",
        30: "grassland",
        40: "cropland",
        50: "built_up",
        60: "bare_sparse_veg",
        70: "snow_ice",
        80: "water",
        90: "herbaceous_wetland",
        95: "mangroves",
        100: "moss_lichen",
    }

    if class_mapping is None:
        class_mapping = default_classes

    with rasterio.open(lulc_raster) as src:
        lulc_data = src.read(1)
        profile = src.profile.copy()

        # Get unique classes present in the data
        unique_classes = np.unique(lulc_data)
        unique_classes = unique_classes[unique_classes != 0]  # Exclude nodata

        # Create one-hot encoding
        num_classes = len(unique_classes)
        one_hot = np.zeros(
            (num_classes, lulc_data.shape[0], lulc_data.shape[1]), dtype=np.float32
        )

        class_info = {}
        for idx, class_val in enumerate(sorted(unique_classes)):
            one_hot[idx] = (lulc_data == class_val).astype(np.float32)
            class_name = class_mapping.get(int(class_val), f"class_{class_val}")
            class_info[f"lulc_class_{idx}"] = {
                "worldcover_code": int(class_val),
                "name": class_name,
                "channel_index": idx,
            }

        # Write one-hot encoded raster
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        profile.update(dtype=rasterio.float32, count=num_classes)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(one_hot)

        print(
            f"[fetch_esa_worldcover] One-hot encoded {num_classes} classes to {output_path}"
        )
        return one_hot, class_info


def fetch_and_process_worldcover(
    reference_raster: str,
    output_dir: str,
    year: int = 2021,
    force_download: bool = False,
) -> Tuple[str, str, dict]:
    """
    Complete workflow: download, reproject, and one-hot encode ESA WorldCover.

    Args:
        reference_raster: Path to reference raster (typically DTM)
        output_dir: Directory to save intermediate and final outputs
        year: Year of WorldCover product (2020 or 2021)
        force_download: Force re-download even if files exist

    Returns:
        Tuple of (one_hot_raster_path, raw_lulc_path, class_info_dict)
    """
    os.makedirs(output_dir, exist_ok=True)

    raw_download_path = os.path.join(output_dir, f"worldcover_{year}_raw.tif")
    reprojected_path = os.path.join(output_dir, f"worldcover_{year}_reprojected.tif")
    one_hot_path = os.path.join(output_dir, f"worldcover_{year}_onehot.tif")

    # Step 1: Download if needed
    if force_download or not os.path.exists(raw_download_path):
        bbox = get_worldcover_bbox(reference_raster)
        success = download_worldcover_tile(bbox, raw_download_path, year=year)
        if not success:
            raise RuntimeError("Failed to download ESA WorldCover data")
    else:
        print(f"[fetch_esa_worldcover] Using existing download: {raw_download_path}")

    # Step 2: Reproject to reference grid
    if force_download or not os.path.exists(reprojected_path):
        success = reproject_worldcover_to_reference(
            raw_download_path, reference_raster, reprojected_path
        )
        if not success:
            raise RuntimeError("Failed to reproject WorldCover data")
    else:
        print(f"[fetch_esa_worldcover] Using existing reprojection: {reprojected_path}")

    # Step 3: One-hot encode
    if force_download or not os.path.exists(one_hot_path):
        _, class_info = one_hot_encode_worldcover(reprojected_path, one_hot_path)
    else:
        print(f"[fetch_esa_worldcover] Using existing one-hot encoding: {one_hot_path}")
        # Load class info from existing file
        with rasterio.open(one_hot_path) as src:
            num_classes = src.count
            class_info = {
                f"lulc_class_{i}": {"channel_index": i} for i in range(num_classes)
            }

    return one_hot_path, reprojected_path, class_info


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python fetch_esa_worldcover.py <reference_raster> <output_dir> [year]"
        )
        sys.exit(1)

    reference = sys.argv[1]
    output = sys.argv[2]
    year = int(sys.argv[3]) if len(sys.argv) > 3 else 2021

    one_hot_path, raw_path, class_info = fetch_and_process_worldcover(
        reference, output, year=year, force_download=True
    )

    print(f"\n=== ESA WorldCover Processing Complete ===")
    print(f"One-hot encoded raster: {one_hot_path}")
    print(f"Raw LULC raster: {raw_path}")
    print(f"Classes found: {class_info}")
