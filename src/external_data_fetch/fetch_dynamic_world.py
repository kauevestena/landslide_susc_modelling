"""
Fetch and process Google Dynamic World (10m) land cover data for a given area of interest.

Dynamic World provides near real-time global land cover at 10m resolution with 9 classes:
- 0: Water
- 1: Trees
- 2: Grass
- 3: Flooded vegetation
- 4: Crops
- 5: Shrub and scrub
- 6: Built
- 7: Bare
- 8: Snow and ice

The product provides probability layers for each class plus a discrete classification.

Documentation: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
Access: Requires Google Earth Engine API
"""

import os
from typing import Tuple, Optional, List
from datetime import datetime, timedelta
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from pathlib import Path

# Google Earth Engine imports (optional, gracefully handle if not installed)
try:
    import ee

    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    print("[fetch_dynamic_world] WARNING: Google Earth Engine not installed.")
    print("Install with: pip install earthengine-api")


def initialize_earth_engine(project: Optional[str] = None) -> bool:
    """
    Initialize Google Earth Engine API.

    Args:
        project: Optional GCP project ID for Earth Engine

    Returns:
        True if initialization successful, False otherwise
    """
    if not EE_AVAILABLE:
        return False

    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        print("[fetch_dynamic_world] Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"[fetch_dynamic_world] Earth Engine initialization failed: {e}")
        print("Run 'earthengine authenticate' to authenticate")
        return False


def get_bbox_from_reference(reference_raster: str) -> Tuple[float, float, float, float]:
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

        if src_crs != CRS.from_epsg(4326):
            from rasterio.warp import transform_bounds

            bounds = transform_bounds(src_crs, CRS.from_epsg(4326), *bounds)

    return bounds.left, bounds.bottom, bounds.right, bounds.top


def download_dynamic_world_composite(
    bbox: Tuple[float, float, float, float],
    output_path: str,
    start_date: str,
    end_date: str,
    scale: int = 10,
    max_pixels: int = 1e8,
) -> bool:
    """
    Download Dynamic World composite via Google Earth Engine API.

    Args:
        bbox: Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84
        output_path: Path to save the downloaded GeoTIFF
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        scale: Spatial resolution in meters (default 10m)
        max_pixels: Maximum number of pixels for export

    Returns:
        True if download successful, False otherwise
    """
    if not EE_AVAILABLE or not initialize_earth_engine():
        return False

    try:
        print(
            f"[fetch_dynamic_world] Fetching Dynamic World from {start_date} to {end_date}"
        )

        # Define region
        region = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])

        # Load Dynamic World collection
        dw = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterDate(start_date, end_date)
            .filterBounds(region)
        )

        # Get the most common classification (mode) over the time period
        classification = dw.select("label").mode()

        # Also get mean probabilities for each class
        prob_bands = [
            "water",
            "trees",
            "grass",
            "flooded_vegetation",
            "crops",
            "shrub_and_scrub",
            "built",
            "bare",
            "snow_and_ice",
        ]
        probabilities = dw.select(prob_bands).mean()

        # Combine classification and probabilities
        export_image = classification.addBands(probabilities)

        # Export to local file via download URL
        url = export_image.getDownloadURL(
            {
                "region": region,
                "scale": scale,
                "format": "GEO_TIFF",
                "maxPixels": max_pixels,
            }
        )

        print(f"[fetch_dynamic_world] Downloading from Earth Engine...")
        import requests

        response = requests.get(url, timeout=300)
        response.raise_for_status()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"[fetch_dynamic_world] Downloaded to {output_path}")
        return True

    except Exception as e:
        print(f"[fetch_dynamic_world] Download failed: {e}")
        return False


def reproject_dynamic_world_to_reference(
    dynamic_world_path: str,
    reference_raster: str,
    output_path: str,
    resampling_method: Resampling = Resampling.nearest,
) -> bool:
    """
    Reproject and align Dynamic World data to match reference raster grid.

    Args:
        dynamic_world_path: Path to downloaded Dynamic World GeoTIFF
        reference_raster: Path to reference raster (typically DTM)
        output_path: Path to save reprojected GeoTIFF
        resampling_method: Resampling method (nearest for classification, bilinear for probabilities)

    Returns:
        True if reprojection successful, False otherwise
    """
    try:
        print(
            f"[fetch_dynamic_world] Reprojecting Dynamic World to match {reference_raster}"
        )

        with rasterio.open(reference_raster) as ref_src:
            ref_profile = ref_src.profile.copy()
            ref_transform = ref_src.transform
            ref_crs = ref_src.crs
            ref_shape = (ref_src.height, ref_src.width)

        with rasterio.open(dynamic_world_path) as dw_src:
            num_bands = dw_src.count

            # Prepare output array
            reprojected_data = np.zeros((num_bands, *ref_shape), dtype=np.float32)

            # Reproject each band
            for band_idx in range(1, num_bands + 1):
                # Use nearest for classification (band 1), bilinear for probabilities
                method = Resampling.nearest if band_idx == 1 else Resampling.bilinear

                reproject(
                    source=rasterio.band(dw_src, band_idx),
                    destination=reprojected_data[band_idx - 1],
                    src_transform=dw_src.transform,
                    src_crs=dw_src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=method,
                    src_nodata=dw_src.nodata,
                    dst_nodata=-9999,
                )

        # Write reprojected data
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ref_profile.update(dtype=rasterio.float32, count=num_bands, nodata=-9999)

        with rasterio.open(output_path, "w", **ref_profile) as dst:
            dst.write(reprojected_data)

        print(f"[fetch_dynamic_world] Reprojected to {output_path}")
        return True

    except Exception as e:
        print(f"[fetch_dynamic_world] Reprojection failed: {e}")
        return False


def one_hot_encode_dynamic_world(
    lulc_raster: str,
    output_path: str,
    use_probabilities: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    One-hot encode Dynamic World classes into binary channels.

    Args:
        lulc_raster: Path to reprojected Dynamic World GeoTIFF
        output_path: Path to save one-hot encoded multi-band GeoTIFF
        use_probabilities: If True, use probability bands instead of hard classification

    Returns:
        Tuple of (one_hot_array, class_info_dict)
    """
    class_names = [
        "water",
        "trees",
        "grass",
        "flooded_vegetation",
        "crops",
        "shrub_and_scrub",
        "built",
        "bare",
        "snow_and_ice",
    ]

    with rasterio.open(lulc_raster) as src:
        profile = src.profile.copy()

        if use_probabilities and src.count > 1:
            # Use probability bands (bands 2-10)
            print("[fetch_dynamic_world] Using probability bands")
            probabilities = src.read(list(range(2, min(11, src.count + 1))))
            one_hot = probabilities.astype(np.float32)
            num_classes = one_hot.shape[0]
        else:
            # Use hard classification (band 1)
            print("[fetch_dynamic_world] Using hard classification")
            classification = src.read(1)

            # Get unique classes
            unique_classes = np.unique(classification)
            unique_classes = unique_classes[unique_classes >= 0]  # Exclude nodata

            num_classes = len(unique_classes)
            one_hot = np.zeros(
                (num_classes, classification.shape[0], classification.shape[1]),
                dtype=np.float32,
            )

            for idx, class_val in enumerate(sorted(unique_classes)):
                one_hot[idx] = (classification == class_val).astype(np.float32)

        # Build class info
        class_info = {}
        for idx in range(num_classes):
            class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            class_info[f"lulc_class_{idx}"] = {
                "dynamic_world_code": idx,
                "name": class_name,
                "channel_index": idx,
            }

        # Write one-hot encoded raster
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        profile.update(dtype=rasterio.float32, count=num_classes, nodata=-9999)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(one_hot)

        print(
            f"[fetch_dynamic_world] One-hot encoded {num_classes} classes to {output_path}"
        )
        return one_hot, class_info


def fetch_and_process_dynamic_world(
    reference_raster: str,
    output_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_probabilities: bool = False,
    force_download: bool = False,
    ee_project: Optional[str] = None,
) -> Tuple[str, str, dict]:
    """
    Complete workflow: download, reproject, and one-hot encode Dynamic World.

    Args:
        reference_raster: Path to reference raster (typically DTM)
        output_dir: Directory to save intermediate and final outputs
        start_date: Start date in 'YYYY-MM-DD' format (defaults to 6 months ago)
        end_date: End date in 'YYYY-MM-DD' format (defaults to today)
        use_probabilities: Use probability bands instead of hard classification
        force_download: Force re-download even if files exist
        ee_project: Optional GCP project ID for Earth Engine

    Returns:
        Tuple of (one_hot_raster_path, raw_lulc_path, class_info_dict)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Default date range: last 6 months
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    raw_download_path = os.path.join(output_dir, "dynamic_world_raw.tif")
    reprojected_path = os.path.join(output_dir, "dynamic_world_reprojected.tif")
    one_hot_path = os.path.join(output_dir, "dynamic_world_onehot.tif")

    # Step 1: Download if needed
    if force_download or not os.path.exists(raw_download_path):
        if ee_project:
            initialize_earth_engine(ee_project)
        bbox = get_bbox_from_reference(reference_raster)
        success = download_dynamic_world_composite(
            bbox, raw_download_path, start_date, end_date
        )
        if not success:
            raise RuntimeError("Failed to download Dynamic World data")
    else:
        print(f"[fetch_dynamic_world] Using existing download: {raw_download_path}")

    # Step 2: Reproject to reference grid
    if force_download or not os.path.exists(reprojected_path):
        success = reproject_dynamic_world_to_reference(
            raw_download_path, reference_raster, reprojected_path
        )
        if not success:
            raise RuntimeError("Failed to reproject Dynamic World data")
    else:
        print(f"[fetch_dynamic_world] Using existing reprojection: {reprojected_path}")

    # Step 3: One-hot encode
    if force_download or not os.path.exists(one_hot_path):
        _, class_info = one_hot_encode_dynamic_world(
            reprojected_path, one_hot_path, use_probabilities=use_probabilities
        )
    else:
        print(f"[fetch_dynamic_world] Using existing one-hot encoding: {one_hot_path}")
        with rasterio.open(one_hot_path) as src:
            num_classes = src.count
            class_names = [
                "water",
                "trees",
                "grass",
                "flooded_vegetation",
                "crops",
                "shrub_and_scrub",
                "built",
                "bare",
                "snow_and_ice",
            ]
            class_info = {
                f"lulc_class_{i}": {
                    "dynamic_world_code": i,
                    "name": class_names[i] if i < len(class_names) else f"class_{i}",
                    "channel_index": i,
                }
                for i in range(num_classes)
            }

    return one_hot_path, reprojected_path, class_info


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python fetch_dynamic_world.py <reference_raster> <output_dir> [start_date] [end_date] [ee_project]"
        )
        print("Dates in YYYY-MM-DD format")
        sys.exit(1)

    reference = sys.argv[1]
    output = sys.argv[2]
    start = sys.argv[3] if len(sys.argv) > 3 else None
    end = sys.argv[4] if len(sys.argv) > 4 else None
    project = sys.argv[5] if len(sys.argv) > 5 else None

    one_hot_path, raw_path, class_info = fetch_and_process_dynamic_world(
        reference,
        output,
        start_date=start,
        end_date=end,
        force_download=True,
        ee_project=project,
    )

    print(f"\n=== Dynamic World Processing Complete ===")
    print(f"One-hot encoded raster: {one_hot_path}")
    print(f"Raw LULC raster: {raw_path}")
    print(f"Classes found: {class_info}")
