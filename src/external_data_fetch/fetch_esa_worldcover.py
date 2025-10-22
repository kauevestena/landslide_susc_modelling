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
Data Access: AWS S3 bucket s3://esa-worldcover (no authentication required)
Tile Grid: 3×3 degree tiles in EPSG:4326

DOWNLOAD METHODS:
1. S3 Direct (preferred): Downloads raw classification GeoTIFFs from AWS S3
   - No authentication required
   - Direct class values (no color mapping)
   - Cloud-Optimized GeoTIFFs for efficient access

2. WMS Fallback: Downloads rendered visualization tiles via WMS
   - Maps RGB colors back to class codes
   - Less accurate but works when S3 is unavailable
"""

import os
from typing import Tuple, Optional, List
import numpy as np
import rasterio
from rasterio.warp import reproject, calculate_default_transform, Resampling
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.io import MemoryFile
import requests
from pathlib import Path
import subprocess


def get_worldcover_bbox(reference_raster: str) -> Tuple[float, float, float, float]:
    """
    Extract bounding box from reference raster in WGS84 (EPSG:4326).

    Args:
        reference_raster: Path to DTM or other georeferenced raster

    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84
    """
    with rasterio.open(reference_raster) as src:
        bounds = src.bounds
        src_crs = src.crs

        # Transform to WGS84 if needed
        if src_crs != CRS.from_epsg(4326):
            from rasterio.warp import transform_bounds

            bounds = transform_bounds(src_crs, CRS.from_epsg(4326), *bounds)
            # transform_bounds returns a tuple (left, bottom, right, top)
            return bounds

    # bounds is a BoundingBox object
    return bounds.left, bounds.bottom, bounds.right, bounds.top


def get_worldcover_tile_names(bbox: Tuple[float, float, float, float]) -> List[str]:
    """
    Get the names of ESA WorldCover 3×3 degree tiles that intersect the bbox.

    Tile naming convention: ESA_WorldCover_10m_2021_v200_<LON><LAT>_Map.tif
    where LON is W###E### (west/east) and LAT is N##S## (north/south)
    Example: ESA_WorldCover_10m_2021_v200_N42W003_Map.tif

    Args:
        bbox: Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84

    Returns:
        List of tile names that cover the bbox
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # WorldCover tiles are 3×3 degrees, aligned to -180, -90
    # Calculate tile indices
    tile_lon_start = int(np.floor((min_lon + 180) / 3)) * 3 - 180
    tile_lon_end = int(np.floor((max_lon + 180) / 3)) * 3 - 180
    tile_lat_start = int(np.floor((min_lat + 90) / 3)) * 3 - 90
    tile_lat_end = int(np.floor((max_lat + 90) / 3)) * 3 - 90

    tile_names = []
    for lat in range(tile_lat_start, tile_lat_end + 3, 3):
        for lon in range(tile_lon_start, tile_lon_end + 3, 3):
            # Format tile name
            lat_str = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
            lon_str = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
            tile_name = f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map.tif"
            tile_names.append(tile_name)

    return tile_names


def download_s3_tile(
    tile_name: str,
    output_path: str,
    year: int = 2021,
    use_aws_cli: bool = True,
) -> bool:
    """
    Download a single WorldCover tile from AWS S3.

    Args:
        tile_name: Name of the tile (e.g., ESA_WorldCover_10m_2021_v200_N42W003_Map.tif)
        output_path: Local path to save the tile
        year: Year (2020 or 2021)
        use_aws_cli: Use AWS CLI if available (faster), else use HTTP

    Returns:
        True if download successful, False otherwise
    """
    # Determine S3 path based on year
    if year == 2021:
        s3_path = f"s3://esa-worldcover/v200/2021/map/{tile_name}"
        http_url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{tile_name}"
    elif year == 2020:
        s3_path = f"s3://esa-worldcover/v100/2020/map/{tile_name}"
        http_url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/{tile_name}"
    else:
        raise ValueError(f"Unsupported year: {year}. Must be 2020 or 2021.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Try AWS CLI first (faster and more reliable)
    if use_aws_cli:
        try:
            print(f"[fetch_esa_worldcover] Downloading {tile_name} via AWS CLI...")
            result = subprocess.run(
                ["aws", "s3", "cp", s3_path, output_path, "--no-sign-request"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                print(f"[fetch_esa_worldcover] Downloaded to {output_path}")
                return True
            else:
                print(f"[fetch_esa_worldcover] AWS CLI failed: {result.stderr}")
                print(f"[fetch_esa_worldcover] Falling back to HTTP download...")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"[fetch_esa_worldcover] AWS CLI unavailable ({e}), using HTTP...")

    # Fallback to HTTP download
    try:
        print(f"[fetch_esa_worldcover] Downloading {tile_name} via HTTP...")
        response = requests.get(http_url, timeout=300, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"[fetch_esa_worldcover] Downloaded to {output_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"[fetch_esa_worldcover] HTTP download failed: {e}")
        return False


def download_and_mosaic_s3_tiles(
    bbox: Tuple[float, float, float, float],
    output_path: str,
    year: int = 2021,
    cache_dir: Optional[str] = None,
) -> bool:
    """
    Download and mosaic WorldCover tiles from S3 for the given bbox.

    Args:
        bbox: Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84
        output_path: Path to save the mosaicked raster
        year: Year (2020 or 2021)
        cache_dir: Directory to cache individual tiles (optional)

    Returns:
        True if successful, False otherwise
    """
    tile_names = get_worldcover_tile_names(bbox)
    print(f"[fetch_esa_worldcover] Need {len(tile_names)} tile(s) for bbox {bbox}")

    if cache_dir is None:
        cache_dir = Path(output_path).parent / "s3_tiles_cache"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Download tiles
    tile_paths = []
    for tile_name in tile_names:
        tile_path = os.path.join(cache_dir, tile_name)

        # Skip if already cached
        if os.path.exists(tile_path):
            print(f"[fetch_esa_worldcover] Using cached tile: {tile_name}")
            tile_paths.append(tile_path)
            continue

        # Download
        success = download_s3_tile(tile_name, tile_path, year=year)
        if success:
            tile_paths.append(tile_path)
        else:
            print(
                f"[fetch_esa_worldcover] Warning: Failed to download {tile_name}, continuing..."
            )

    if not tile_paths:
        print("[fetch_esa_worldcover] Error: No tiles were downloaded successfully")
        return False

    # Mosaic tiles if multiple, or just copy if single
    if len(tile_paths) == 1:
        print(f"[fetch_esa_worldcover] Single tile, copying to output...")
        import shutil

        shutil.copy(tile_paths[0], output_path)
        return True

    # Mosaic multiple tiles
    print(f"[fetch_esa_worldcover] Mosaicking {len(tile_paths)} tiles...")
    try:
        src_files = [rasterio.open(p) for p in tile_paths]
        mosaic_data, mosaic_transform = merge(src_files, nodata=0)

        # Get profile from first tile
        profile = src_files[0].profile.copy()
        profile.update(
            {
                "height": mosaic_data.shape[1],
                "width": mosaic_data.shape[2],
                "transform": mosaic_transform,
            }
        )

        # Write mosaic
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic_data)

        # Close source files
        for src in src_files:
            src.close()

        print(f"[fetch_esa_worldcover] Mosaic saved to {output_path}")
        return True

    except Exception as e:
        print(f"[fetch_esa_worldcover] Mosaic failed: {e}")
        return False


def bbox_to_web_mercator(
    bbox_4326: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    Convert a WGS84 bbox to Web Mercator (EPSG:3857).

    Args:
        bbox_4326: Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84

    Returns:
        Tuple of (minx, miny, maxx, maxy) in EPSG:3857
    """
    from rasterio.warp import transform_bounds

    bbox_3857 = transform_bounds(CRS.from_epsg(4326), CRS.from_epsg(3857), *bbox_4326)
    return bbox_3857


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

    # Convert bbox to Web Mercator (EPSG:3857) which is the only CRS supported
    bbox_3857 = bbox_to_web_mercator(bbox)

    # Use WMS 1.1.1 which uses lon,lat order consistently
    params = {
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetMap",
        "layers": f"WORLDCOVER_{year}_MAP",
        "bbox": f"{bbox_3857[0]},{bbox_3857[1]},{bbox_3857[2]},{bbox_3857[3]}",
        "srs": "EPSG:3857",  # WorldCover WMS only supports EPSG:3857
        "width": width,
        "height": height,
        "format": "image/tiff",  # Use image/tiff instead of image/geotiff
    }

    try:
        print(f"[fetch_esa_worldcover] Downloading WorldCover {year} for bbox {bbox}")
        response = requests.get(base_url, params=params, timeout=120)

        # Check if we got an error response
        if response.status_code != 200:
            print(
                f"[fetch_esa_worldcover] HTTP {response.status_code}: {response.text[:500]}"
            )

        response.raise_for_status()

        # Save to temporary file first
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path + ".tmp"
        with open(temp_path, "wb") as f:
            f.write(response.content)

        # Add georeference information since WMS doesn't always include it
        import numpy as np

        with rasterio.open(temp_path) as src:
            data = src.read()
            src_count = src.count

        # If the WMS returned an RGB(A) rendered image, map palette/colors to class codes
        if src_count in (3, 4):
            print(
                f"[fetch_esa_worldcover] Detected rendered RGB(A) image with {src_count} bands. Mapping colors to WorldCover class codes."
            )
            # Build palette mapping for WorldCover (visualization palette used by Terrascope)
            wc_palette = {
                (0, 0, 0): 0,  # nodata / background
                (0, 104, 55): 10,  # tree
                (85, 176, 115): 20,  # shrubland
                (69, 139, 116): 30,  # grassland (approx)
                (227, 26, 28): 40,  # cropland (approx)
                (240, 59, 32): 50,  # built-up (approx)
                (255, 255, 204): 60,  # bare/sparse veg (approx)
                (255, 255, 255): 70,  # snow/ice
                (0, 0, 255): 80,  # water
                (166, 255, 255): 90,  # wetland (approx)
                (166, 215, 83): 95,  # mangroves (approx)
                (201, 165, 93): 100,  # moss/lichen (approx)
            }

            # Read RGB(A) into H,W,3 array
            rgb = np.transpose(data[:3], (1, 2, 0))  # shape H,W,3

            # Convert to single-band class raster using palette mapping
            mapped = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

            # Create a viewable tuple array for faster lookup
            flat_rgb = rgb.reshape(-1, 3)
            mapped_flat = np.zeros((flat_rgb.shape[0],), dtype=np.uint8)

            # Build a color->class dict with fallbacks using nearest color if exact match not found
            color_keys = np.array(list(wc_palette.keys()), dtype=np.uint8)
            color_vals = np.array(list(wc_palette.values()), dtype=np.uint8)

            # Exact matches first
            for i, px in enumerate(flat_rgb):
                key = tuple(int(x) for x in px)
                mapped_flat[i] = wc_palette.get(key, 0)

            # For pixels still unmapped (mapped_flat == 0) but not true background (0,0,0),
            # assign the nearest palette color by Euclidean distance in RGB space.
            unmapped_idx = np.where(
                (mapped_flat == 0) & ~(np.all(flat_rgb == 0, axis=1))
            )[0]
            if unmapped_idx.size > 0:
                # Build palette arrays
                palette_colors = np.array(
                    list(wc_palette.keys()), dtype=np.int16
                )  # Kx3
                palette_values = np.array(
                    list(wc_palette.values()), dtype=np.uint8
                )  # K

                # Vectorized nearest neighbor: compute distances in chunks to save memory
                chunk_size = 100000
                for start in range(0, unmapped_idx.size, chunk_size):
                    end = start + chunk_size
                    idx_chunk = unmapped_idx[start:end]
                    pixels = flat_rgb[idx_chunk].astype(np.int16)[
                        :, None, :
                    ]  # N x 1 x 3
                    diffs = pixels - palette_colors[None, :, :]  # N x K x 3
                    dists = np.sum(diffs * diffs, axis=2)  # N x K
                    nearest = np.argmin(dists, axis=1)
                    mapped_flat[idx_chunk] = palette_values[nearest]

            mapped = mapped_flat.reshape(rgb.shape[0], rgb.shape[1])

            data = mapped[np.newaxis, ...]  # shape 1,H,W

        # Calculate transform from bbox_3857
        transform = Affine(
            (bbox_3857[2] - bbox_3857[0]) / width,  # pixel width
            0,
            bbox_3857[0],  # left
            0,
            -(bbox_3857[3] - bbox_3857[1])
            / height,  # pixel height (negative for north-up)
            bbox_3857[3],  # top
        )

        # Write with proper georeference
        profile = {
            "driver": "GTiff",
            "height": data.shape[1],
            "width": data.shape[2],
            "count": data.shape[0],
            "dtype": data.dtype,
            "crs": CRS.from_epsg(3857),
            "transform": transform,
            "compress": "lzw",
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

        # Remove temp file
        os.remove(temp_path)

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

        # IMPORTANT: Always encode ALL WorldCover classes (not just those present)
        # This ensures consistent channel counts across different study areas
        all_class_codes = sorted(default_classes.keys())  # [10, 20, 30, ..., 100]
        num_classes = len(all_class_codes)

        # Create one-hot encoding for all classes
        one_hot = np.zeros(
            (num_classes, lulc_data.shape[0], lulc_data.shape[1]), dtype=np.float32
        )

        class_info = {}
        for idx, class_val in enumerate(all_class_codes):
            # Set 1.0 where this class is present, 0.0 elsewhere
            one_hot[idx] = (lulc_data == class_val).astype(np.float32)
            class_name = class_mapping.get(int(class_val), f"class_{class_val}")
            class_info[f"lulc_class_{idx}"] = {
                "worldcover_code": int(class_val),
                "name": class_name,
                "channel_index": idx,
            }

        # Log which classes are actually present vs absent
        unique_classes = np.unique(lulc_data)
        unique_classes = unique_classes[unique_classes != 0]  # Exclude nodata
        present_classes = [c for c in all_class_codes if c in unique_classes]
        absent_classes = [c for c in all_class_codes if c not in unique_classes]

        print(f"[fetch_esa_worldcover] Classes present in data: {present_classes}")
        if absent_classes:
            print(
                f"[fetch_esa_worldcover] Classes absent (encoded as zeros): {absent_classes}"
            )

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
    use_s3: bool = True,
) -> Tuple[str, str, dict]:
    """
    Complete workflow: download, reproject, and one-hot encode ESA WorldCover.

    Args:
        reference_raster: Path to reference raster (typically DTM)
        output_dir: Directory to save intermediate and final outputs
        year: Year of WorldCover product (2020 or 2021)
        force_download: Force re-download even if files exist
        use_s3: Use S3 direct download (preferred) vs WMS (fallback)

    Returns:
        Tuple of (one_hot_raster_path, raw_lulc_path, class_info_dict)
    """
    os.makedirs(output_dir, exist_ok=True)

    method = "s3" if use_s3 else "wms"
    raw_download_path = os.path.join(output_dir, f"worldcover_{year}_raw_{method}.tif")
    reprojected_path = os.path.join(output_dir, f"worldcover_{year}_reprojected.tif")
    one_hot_path = os.path.join(output_dir, f"worldcover_{year}_onehot.tif")

    # Step 1: Download if needed
    if force_download or not os.path.exists(raw_download_path):
        bbox = get_worldcover_bbox(reference_raster)

        if use_s3:
            print(
                f"[fetch_esa_worldcover] Using S3 direct download (raw classification data)"
            )
            success = download_and_mosaic_s3_tiles(
                bbox,
                raw_download_path,
                year=year,
                cache_dir=os.path.join(output_dir, "s3_tiles_cache"),
            )
        else:
            print(f"[fetch_esa_worldcover] Using WMS download (rendered visualization)")
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
            "Usage: python fetch_esa_worldcover.py <reference_raster> <output_dir> [year] [method]"
        )
        print("  year: 2020 or 2021 (default: 2021)")
        print("  method: s3 or wms (default: s3)")
        sys.exit(1)

    reference = sys.argv[1]
    output = sys.argv[2]
    year = int(sys.argv[3]) if len(sys.argv) > 3 else 2021
    method = sys.argv[4] if len(sys.argv) > 4 else "s3"
    use_s3 = method.lower() == "s3"

    one_hot_path, raw_path, class_info = fetch_and_process_worldcover(
        reference, output, year=year, force_download=True, use_s3=use_s3
    )

    print(f"\n=== ESA WorldCover Processing Complete ===")
    print(
        f"Method: {'S3 Direct (raw classification)' if use_s3 else 'WMS (color-mapped visualization)'}"
    )
    print(f"One-hot encoded raster: {one_hot_path}")
    print(f"Raw LULC raster: {raw_path}")
    print(f"Classes found: {class_info}")
