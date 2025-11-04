"""End-to-end landslide susceptibility pipeline covering preprocessing, tiling, training, and inference."""

import argparse
import os
import json
import math
import random
import importlib
from dataclasses import dataclass
import glob
from typing import Dict, Optional, Tuple, List

try:
    from tqdm import tqdm
except ImportError:

    class _TqdmDummy:
        def __init__(self, iterable=None):
            self.iterable = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self.iterable)

        def update(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def set_description(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def tqdm(iterable=None, **kwargs):
        return _TqdmDummy(iterable)


import yaml
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.ndimage import (
    gaussian_filter,
    uniform_filter,
    distance_transform_edt,
    maximum_filter,
    minimum_filter,
    binary_erosion,
)

# V2.5: Import SMOTE for synthetic minority oversampling
try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("WARNING: imbalanced-learn not installed. SMOTE will be unavailable.")
    print("Install with: pip install imbalanced-learn")

from src.train import train_model
from src.inference import run_inference

# Import external LULC fetchers
try:
    from src.external_data_fetch import (
        fetch_and_process_worldcover,
        fetch_and_process_dynamic_world,
        DYNAMIC_WORLD_AVAILABLE,
    )

    EXTERNAL_LULC_AVAILABLE = True
except ImportError:
    EXTERNAL_LULC_AVAILABLE = False
    DYNAMIC_WORLD_AVAILABLE = False
    print("[main_pipeline] WARNING: External LULC fetchers not available")


@dataclass
class AreaInputs:
    """Absolute paths for DTM/orthophoto/ground-truth inputs for a spatial area."""

    name: str
    dtm: str
    ortho: str
    ground_truth: Optional[str] = None


@dataclass
class AreaArtifacts:
    """Paths to derived artifacts produced during preprocessing for an area."""

    name: str
    feature_stack_path: str
    mask_path: str
    metadata_path: str
    ground_truth_path: Optional[str] = None
    normalization_stats_path: Optional[str] = None


def debug_print(message: str) -> None:
    print(f"[main_pipeline] {message}", flush=True)


def prepare_directories(structure_cfg: Dict[str, str]) -> None:
    """Create required project directories declared in configuration."""
    debug_print("Preparing project directories from configuration")
    for key, path in structure_cfg.items():
        if key.endswith("_dir"):
            debug_print(f"Ensuring directory '{key}' at '{path}'")
            os.makedirs(path, exist_ok=True)


def load_input_paths(config: Dict) -> Dict[str, AreaInputs]:
    """Resolve input raster paths from the configured module and validate their existence."""
    module_name = config["inputs"]["module"]
    debug_print(f"Importing input registry module '{module_name}'")
    data_module = importlib.import_module(module_name)
    debug_print("Resolving configured input splits")
    inputs_map: Dict[str, AreaInputs] = {}

    for split_name, split_cfg in config["inputs"].items():
        debug_print(f"Processing input split '{split_name}'")
        if split_name == "module":
            continue
        dtm_attr = split_cfg.get("dtm_attr")
        ortho_attr = split_cfg.get("ortho_attr")
        if not dtm_attr or not ortho_attr:
            continue
        ground_attr = split_cfg.get("ground_truth_attr")
        dtm_path = getattr(data_module, dtm_attr)
        ortho_path = getattr(data_module, ortho_attr)
        ground_path = getattr(data_module, ground_attr) if ground_attr else None

        for candidate in [dtm_path, ortho_path, ground_path]:
            if candidate and not os.path.exists(candidate):
                raise FileNotFoundError(f"Input path does not exist: {candidate}")

        inputs_map[split_name] = AreaInputs(
            name=split_name,
            dtm=str(dtm_path),
            ortho=str(ortho_path),
            ground_truth=str(ground_path) if ground_path else None,
        )
        debug_print(
            f"Registered inputs for split '{split_name}': DTM={dtm_path}, Ortho={ortho_path}, GroundTruth={ground_path}"
        )

    debug_print(f"Resolved input splits: {list(inputs_map.keys())}")
    if "train" not in inputs_map:
        raise ValueError("Training inputs not defined in configuration.")
    return inputs_map


def save_geotiff(
    path: str,
    array: np.ndarray,
    reference_profile: Dict,
    dtype: Optional[str] = None,
    nodata: Optional[float] = None,
) -> None:
    """Persist a numpy array to GeoTIFF using a reference profile for spatial metadata."""
    debug_print(f"Saving GeoTIFF to {path}")
    data = array
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    profile = reference_profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "count": data.shape[0],
            "dtype": dtype or data.dtype.name,
        }
    )
    if nodata is not None:
        profile["nodata"] = nodata
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(profile["dtype"]))


def fill_nodata(array: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    """Fill invalid cells by nearest-neighbour propagation."""
    debug_print("fill_nodata: evaluating invalid mask")
    if not np.any(invalid_mask):
        debug_print("fill_nodata: no invalid cells detected, returning original array")
        return array
    filled = array.copy()
    debug_print("fill_nodata: propagating nearest valid values")
    _, indices = distance_transform_edt(invalid_mask, return_indices=True)
    filled[invalid_mask] = array[tuple(indices[:, invalid_mask])]
    return filled


def simple_sink_fill(array: np.ndarray, kernel_size: int) -> np.ndarray:
    """Approximate sink filling via morphological closing."""
    if not kernel_size or kernel_size < 3:
        return array
    size = (kernel_size, kernel_size)
    dilated = maximum_filter(array, size=size)
    eroded = minimum_filter(dilated, size=size)
    return eroded


def d8_flow_accumulation(
    elevation: np.ndarray, valid_mask: np.ndarray, cellsize: float
) -> np.ndarray:
    """Compute a simple D8-style flow accumulation grid respecting nodata."""
    debug_print("d8_flow_accumulation: computing flow accumulation grid")
    nrows, ncols = elevation.shape
    debug_print(f"d8_flow_accumulation: grid size {nrows}x{ncols}")
    flow = np.ones((nrows, ncols), dtype=np.float32)
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    distances = [
        math.sqrt(2) * cellsize,
        cellsize,
        math.sqrt(2) * cellsize,
        cellsize,
        cellsize,
        math.sqrt(2) * cellsize,
        cellsize,
        math.sqrt(2) * cellsize,
    ]
    order = np.argsort(elevation, axis=None)[::-1]
    for index in tqdm(order, desc="d8_flow_accumulation", leave=False):
        r = index // ncols
        c = index % ncols
        if not valid_mask[r, c]:
            continue
        z = elevation[r, c]
        best_drop = 0.0
        target = None
        for (dr, dc), dist in zip(offsets, distances):
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                continue
            if not valid_mask[rr, cc]:
                continue
            drop = (z - elevation[rr, cc]) / dist
            if drop > best_drop:
                best_drop = drop
                target = (rr, cc)
        if target is None:
            continue
        rr, cc = target
        flow[rr, cc] += flow[r, c]
    flow[~valid_mask] = 0.0
    return flow


def compute_dem_features(
    elevation: np.ndarray,
    transform: Affine,
    valid_mask: np.ndarray,
    preprocessing_cfg: Dict,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Derive terrain attributes and hydrologic metrics from the conditioned DEM."""
    debug_print("compute_dem_features: starting terrain feature derivation")
    cellsize_x = abs(transform.a)
    cellsize_y = abs(transform.e)
    dz_dy, dz_dx = np.gradient(elevation, cellsize_y, cellsize_x)
    debug_print("compute_dem_features: computed slope and aspect gradients")
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
    aspect_sin = np.sin(np.deg2rad(aspect_deg))
    aspect_cos = np.cos(np.deg2rad(aspect_deg))

    d_p_dy, d_p_dx = np.gradient(dz_dx, cellsize_y, cellsize_x)
    d_q_dy, d_q_dx = np.gradient(dz_dy, cellsize_y, cellsize_x)
    r = d_p_dx
    t = d_q_dy
    s = 0.5 * (d_p_dy + d_q_dx)

    denom = dz_dx**2 + dz_dy**2
    denom_safe = np.where(denom == 0.0, 1e-6, denom)
    plan_curvature = (dz_dx**2 * t - 2.0 * dz_dx * dz_dy * s + dz_dy**2 * r) / (
        denom_safe * np.sqrt(1.0 + denom_safe)
    )
    profile_curvature = (dz_dx**2 * r + 2.0 * dz_dx * dz_dy * s + dz_dy**2 * t) / (
        denom_safe * (1.0 + denom_safe) ** 1.5
    )
    general_curvature = r + t
    debug_print("compute_dem_features: derived curvature metrics")

    window = 9
    mean_local = uniform_filter(elevation, size=window, mode="nearest")
    tpi = elevation - mean_local
    tri = np.sqrt(
        uniform_filter((elevation - mean_local) ** 2, size=window, mode="nearest")
    )
    debug_print("compute_dem_features: calculated TPI and TRI")

    cell_area = cellsize_x * cellsize_y
    flow = d8_flow_accumulation(
        elevation, valid_mask, cellsize=(cellsize_x + cellsize_y) * 0.5
    )
    flow_area = flow * cell_area
    debug_print("compute_dem_features: computed flow accumulation and area")
    log_flow = np.log1p(flow_area)

    slope_safe = np.where(slope_rad < math.radians(0.1), math.radians(0.1), slope_rad)
    tan_slope = np.tan(slope_safe)
    tan_slope = np.where(tan_slope < 1e-6, 1e-6, tan_slope)
    twi = np.log((flow_area + 1.0) / tan_slope)
    spi = np.log1p(flow_area * tan_slope)
    sti = np.log1p(
        ((flow_area / 22.13) ** 0.6) * ((np.sin(slope_safe) / 0.0896) ** 1.3)
    )

    drainage_threshold = preprocessing_cfg["hydrology"].get(
        "drainage_threshold", 5000.0
    )
    drainage_mask = flow_area >= drainage_threshold
    if np.any(drainage_mask):
        distance = distance_transform_edt(
            ~drainage_mask, sampling=(cellsize_y, cellsize_x)
        )
    else:
        distance = np.zeros_like(elevation, dtype=np.float32)
    distance_log = np.log1p(distance)
    debug_print("compute_dem_features: computed distance to drainage metrics")

    debug_print("compute_dem_features: assembling feature dictionary")
    features = {
        "dtm_elevation": elevation.astype(np.float32),
        "slope_deg": slope_deg.astype(np.float32),
        "aspect_sin": aspect_sin.astype(np.float32),
        "aspect_cos": aspect_cos.astype(np.float32),
        "general_curvature": general_curvature.astype(np.float32),
        "plan_curvature": plan_curvature.astype(np.float32),
        "profile_curvature": profile_curvature.astype(np.float32),
        "tpi": tpi.astype(np.float32),
        "tri": tri.astype(np.float32),
        "log_flow_accumulation": log_flow.astype(np.float32),
        "twi": twi.astype(np.float32),
        "spi": spi.astype(np.float32),
        "sti": sti.astype(np.float32),
        "log_distance_to_drainage": distance_log.astype(np.float32),
    }
    for key, value in features.items():
        value[~valid_mask] = 0.0
        features[key] = value

    debug_print("compute_dem_features: compiling metadata layers")
    meta = {
        "slope_deg": slope_deg.astype(np.float32),
        "aspect_deg": aspect_deg.astype(np.float32),
        "flow_area": flow_area.astype(np.float32),
        "drainage_mask": drainage_mask.astype(np.uint8),
    }
    debug_print("compute_dem_features: finished terrain feature derivation")
    return features, meta


def normalize_radiometry(
    ortho: np.ndarray, valid_mask: np.ndarray, epsilon: float
) -> np.ndarray:
    """Apply per-band z-score normalization constrained to valid pixels."""
    debug_print(f"normalize_radiometry: normalizing {ortho.shape[0]} bands")
    normalized = ortho.astype(np.float32).copy()
    for band_idx in tqdm(
        range(normalized.shape[0]), desc="normalize_radiometry", leave=False
    ):
        debug_print(f"normalize_radiometry: processing band {band_idx + 1}")
        band = normalized[band_idx]
        values = band[valid_mask]
        if values.size == 0:
            mean = 0.0
            std = 1.0
        else:
            mean = float(values.mean())
            std = float(values.std())
            if std < epsilon:
                std = 1.0
        band = (band - mean) / (std + epsilon)
        band[~valid_mask] = 0.0
        normalized[band_idx] = band
    return normalized


def fetch_external_lulc(
    reference_raster: str,
    area_name: str,
    area_dir: str,
    lulc_config: Dict,
    valid_mask: np.ndarray,
    reference_profile: Dict,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Fetch and process external LULC data from WorldCover or Dynamic World.

    Args:
        reference_raster: Path to reference DTM raster
        area_name: Name of the area (train/test)
        area_dir: Directory to cache downloaded LULC data
        lulc_config: Configuration for external LULC fetching
        valid_mask: Valid pixel mask from DTM
        reference_profile: Rasterio profile of the reference grid

    Returns:
        Tuple of (one_hot_array, raw_labels, class_info_dict)
    """
    if not EXTERNAL_LULC_AVAILABLE:
        raise RuntimeError(
            "External LULC fetchers not available. Install required dependencies."
        )

    source = lulc_config.get("source", "worldcover").lower()
    force_download = lulc_config.get("force_download", False)
    lulc_cache_dir = os.path.join(area_dir, "lulc_cache")

    debug_print(f"fetch_external_lulc: fetching {source} data for {area_name}")

    if source == "worldcover":
        year = lulc_config.get("worldcover", {}).get("year", 2021)
        one_hot_path, raw_path, class_info = fetch_and_process_worldcover(
            reference_raster,
            lulc_cache_dir,
            year=year,
            force_download=force_download,
            use_s3=True,  # Use S3 direct download (raw classification) instead of WMS
        )
    elif source == "dynamic_world":
        if not DYNAMIC_WORLD_AVAILABLE:
            raise RuntimeError(
                "Dynamic World requires Google Earth Engine. Install: pip install earthengine-api"
            )
        dw_config = lulc_config.get("dynamic_world", {})
        one_hot_path, raw_path, class_info = fetch_and_process_dynamic_world(
            reference_raster,
            lulc_cache_dir,
            start_date=dw_config.get("start_date"),
            end_date=dw_config.get("end_date"),
            use_probabilities=dw_config.get("use_probabilities", False),
            force_download=force_download,
            ee_project=dw_config.get("ee_project"),
        )
    else:
        raise ValueError(f"Unknown LULC source: {source}")

    # Load the one-hot encoded raster
    with rasterio.open(one_hot_path) as src:
        one_hot_data = src.read()  # Shape: (num_classes, height, width)

    # Load the raw labels
    with rasterio.open(raw_path) as src:
        raw_labels = src.read(1).astype(np.int16)

    # Mask invalid pixels
    one_hot_data[:, ~valid_mask] = 0.0
    raw_labels[~valid_mask] = -1

    debug_print(
        f"fetch_external_lulc: loaded {one_hot_data.shape[0]} classes from {source}"
    )

    return one_hot_data.astype(np.float32), raw_labels, class_info


def derive_land_cover(
    ortho: np.ndarray, valid_mask: np.ndarray, clusters: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster normalized orthophoto pixels into pseudo land-cover classes."""
    debug_print(f"derive_land_cover: clustering into {clusters} classes")
    height, width = valid_mask.shape
    flattened = ortho.reshape(ortho.shape[0], -1).T
    valid_flat = flattened[valid_mask.ravel()]
    if valid_flat.size == 0:
        debug_print(
            "derive_land_cover: no valid pixels, returning empty land cover stack"
        )
        one_hot = np.zeros((clusters, height, width), dtype=np.float32)
        labels = np.full((height, width), fill_value=-1, dtype=np.int16)
        debug_print("derive_land_cover: returning empty land cover stack")
        return one_hot, labels
    max_samples = 200000
    rng = np.random.default_rng(seed)
    sample = valid_flat
    if sample.shape[0] > max_samples:
        debug_print(
            f"derive_land_cover: subsampling from {sample.shape[0]} to {max_samples} pixels"
        )
        idx = rng.choice(sample.shape[0], size=max_samples, replace=False)
        sample = sample[idx]
    kmeans = KMeans(n_clusters=clusters, random_state=seed, n_init="auto")
    kmeans.fit(sample)
    debug_print("derive_land_cover: fitted k-means model")
    labels_flat = np.full(flattened.shape[0], fill_value=-1, dtype=np.int32)
    labels_flat[valid_mask.ravel()] = kmeans.predict(valid_flat)
    labels = labels_flat.reshape(height, width)
    one_hot = np.zeros((clusters, height, width), dtype=np.float32)
    for idx in range(clusters):
        one_hot[idx] = np.where(labels == idx, 1.0, 0.0)
    one_hot[:, ~valid_mask] = 0.0
    debug_print("derive_land_cover: finished clustering, returning tensors")
    return one_hot, labels.astype(np.int16)


def build_feature_stack(
    dem_feature_map: Dict[str, np.ndarray],
    ortho_channels: np.ndarray,
    land_cover: np.ndarray,
    preprocessing_cfg: Dict,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Assemble the multi-channel tensor following configuration toggles."""
    debug_print("build_feature_stack: assembling configured features")
    include_cfg = preprocessing_cfg["derived_channels"]
    feature_arrays: List[np.ndarray] = []
    channel_names: List[str] = []

    if include_cfg.get("include_dtm", True):
        debug_print("build_feature_stack: including dtm_elevation")
        feature_arrays.append(dem_feature_map["dtm_elevation"])
        channel_names.append("dtm_elevation")
    if include_cfg.get("include_slope", True):
        debug_print("build_feature_stack: including slope_deg")
        feature_arrays.append(dem_feature_map["slope_deg"])
        channel_names.append("slope_deg")
    if include_cfg.get("include_aspect_sin_cos", True):
        debug_print("build_feature_stack: including aspect_sin")
        feature_arrays.append(dem_feature_map["aspect_sin"])
        channel_names.append("aspect_sin")
        debug_print("build_feature_stack: including aspect_cos")
        feature_arrays.append(dem_feature_map["aspect_cos"])
        channel_names.append("aspect_cos")
    if include_cfg.get("include_curvatures", True):
        debug_print("build_feature_stack: including general_curvature")
        feature_arrays.append(dem_feature_map["general_curvature"])
        channel_names.append("general_curvature")
        debug_print("build_feature_stack: including plan_curvature")
        feature_arrays.append(dem_feature_map["plan_curvature"])
        channel_names.append("plan_curvature")
        debug_print("build_feature_stack: including profile_curvature")
        feature_arrays.append(dem_feature_map["profile_curvature"])
        channel_names.append("profile_curvature")
    if include_cfg.get("include_tpi", True):
        debug_print("build_feature_stack: including tpi")
        feature_arrays.append(dem_feature_map["tpi"])
        channel_names.append("tpi")
    if include_cfg.get("include_tri", True):
        debug_print("build_feature_stack: including tri")
        feature_arrays.append(dem_feature_map["tri"])
        channel_names.append("tri")
    if include_cfg.get("include_flow_accumulation", True):
        debug_print("build_feature_stack: including log_flow_accumulation")
        feature_arrays.append(dem_feature_map["log_flow_accumulation"])
        channel_names.append("log_flow_accumulation")
    if include_cfg.get("include_twi", True):
        debug_print("build_feature_stack: including twi")
        feature_arrays.append(dem_feature_map["twi"])
        channel_names.append("twi")
    if include_cfg.get("include_spi", True):
        debug_print("build_feature_stack: including spi")
        feature_arrays.append(dem_feature_map["spi"])
        channel_names.append("spi")
    if include_cfg.get("include_sti", True):
        debug_print("build_feature_stack: including sti")
        feature_arrays.append(dem_feature_map["sti"])
        channel_names.append("sti")
    if include_cfg.get("include_distance_to_drainage", True):
        debug_print("build_feature_stack: including log_distance_to_drainage")
        feature_arrays.append(dem_feature_map["log_distance_to_drainage"])
        channel_names.append("log_distance_to_drainage")

    for band_idx in tqdm(
        range(ortho_channels.shape[0]),
        desc="build_feature_stack: ortho bands",
        leave=False,
    ):
        debug_print(
            f"build_feature_stack: including normalized ortho band {band_idx + 1}"
        )
        feature_arrays.append(ortho_channels[band_idx])
        channel_names.append(f"ortho_norm_band_{band_idx + 1}")

    for cls_idx in tqdm(
        range(land_cover.shape[0]), desc="build_feature_stack: land cover", leave=False
    ):
        debug_print(f"build_feature_stack: including land cover class {cls_idx}")
        feature_arrays.append(land_cover[cls_idx])
        channel_names.append(f"lulc_class_{cls_idx}")

    stack = np.stack(feature_arrays, axis=0).astype(np.float32)
    stack[:, ~valid_mask] = 0.0
    debug_print(f"build_feature_stack: prepared {len(channel_names)} channels")
    return stack, channel_names


def apply_normalization(
    stack: np.ndarray,
    valid_mask: np.ndarray,
    normalization_cfg: Dict,
    existing_stats: Optional[Dict],
) -> Tuple[np.ndarray, Dict]:
    """Normalize the feature tensor and optionally reuse stored statistics."""
    strategy = normalization_cfg.get("strategy", "zscore")
    debug_print(
        f"apply_normalization: strategy={strategy}, reuse_stats={existing_stats is not None}"
    )
    epsilon = float(normalization_cfg.get("epsilon", 1e-6))
    stats = existing_stats
    normalized = stack.copy()

    debug_print("apply_normalization: beginning normalization loop")
    if strategy == "zscore":
        if stats is None:
            debug_print("apply_normalization: computing new normalization statistics")
            means = []
            stds = []
            flat = normalized.reshape(normalized.shape[0], -1)
            valid_flat = valid_mask.ravel()
            for channel_idx in tqdm(
                range(normalized.shape[0]), desc="apply_normalization", leave=False
            ):
                debug_print(f"apply_normalization: processing channel {channel_idx}")
                values = flat[channel_idx, valid_flat]
                if values.size == 0:
                    mean = 0.0
                    std = 1.0
                else:
                    mean = float(values.mean())
                    std = float(values.std())
                    if std < epsilon:
                        std = 1.0
                means.append(mean)
                stds.append(std)
                flat[channel_idx, :] = (flat[channel_idx, :] - mean) / (std + epsilon)
            normalized = flat.reshape(normalized.shape)
            stats = {
                "strategy": "zscore",
                "mean": means,
                "std": stds,
                "epsilon": epsilon,
            }
        else:
            debug_print("apply_normalization: applying existing statistics")
            for channel_idx in tqdm(
                range(normalized.shape[0]),
                desc="apply_normalization: reuse",
                leave=False,
            ):
                mean = stats["mean"][channel_idx]
                std = stats["std"][channel_idx]
                normalized[channel_idx] = (normalized[channel_idx] - mean) / (
                    std + epsilon
                )
    debug_print("apply_normalization: masking invalid pixels to zero")
    normalized[:, ~valid_mask] = 0.0
    debug_print("apply_normalization: completed normalization step")
    return normalized.astype(np.float32), stats


def compute_ignore_mask(labels: np.ndarray, pixels: int) -> np.ndarray:
    """Construct a boundary ignore mask by eroding class extents."""
    debug_print(f"compute_ignore_mask: eroding classes with {pixels} iterations")
    if labels.size == 0 or pixels is None or pixels <= 0:
        return np.zeros_like(labels, dtype=bool)
    ignore_mask = np.zeros_like(labels, dtype=bool)
    classes = np.unique(labels[(labels != 0) & (labels != 255)])
    for cls in classes:
        class_mask = labels == cls
        if not class_mask.any():
            continue
        eroded = binary_erosion(class_mask, iterations=pixels, border_value=0)
        boundary = class_mask & ~eroded
        ignore_mask |= boundary
    return ignore_mask


def reproject_raster(
    src_path: str,
    reference_profile: Dict,
    resampling: Resampling,
    dst_dtype: Optional[np.dtype] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    """Warp a raster onto the reference grid using the requested resampling."""
    debug_print(f"reproject_raster: reprojecting {src_path}")
    dtype = dst_dtype or np.float32
    with rasterio.open(src_path) as src:
        count = src.count
        dst = np.zeros(
            (count, reference_profile["height"], reference_profile["width"]),
            dtype=dtype,
        )
        for band_idx in tqdm(range(1, count + 1), desc="reproject_raster", leave=False):
            reproject(
                source=rasterio.band(src, band_idx),
                destination=dst[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=reference_profile["transform"],
                dst_crs=reference_profile["crs"],
                resampling=resampling,
            )
        nodata = src.nodata
    return dst.astype(dtype), nodata


def process_area(
    area_inputs: AreaInputs,
    area_dir: str,
    preprocessing_cfg: Dict,
    metadata_dir: str,
    stats: Optional[Dict],
    seed: int,
    force_recreate: bool = False,
) -> Tuple[AreaArtifacts, Optional[Dict]]:
    """Run preprocessing for a single area and persist derived artifacts."""
    debug_print(f"process_area: begin processing area '{area_inputs.name}'")

    # Define expected artifact paths
    feature_stack_path = os.path.join(area_dir, "feature_stack.tif")
    mask_path = os.path.join(area_dir, "valid_mask.tif")
    metadata_path = os.path.join(area_dir, "feature_metadata.json")
    normalization_stats_path = os.path.join(metadata_dir, "normalization_stats.json")
    ground_truth_path = None
    if area_inputs.ground_truth:
        ground_truth_path = os.path.join(area_dir, "ground_truth_aligned.tif")

    # Check if artifacts already exist
    artifacts_exist = (
        os.path.exists(feature_stack_path)
        and os.path.exists(mask_path)
        and os.path.exists(metadata_path)
    )

    if area_inputs.ground_truth:
        artifacts_exist = artifacts_exist and os.path.exists(ground_truth_path)

    if artifacts_exist and not force_recreate:
        debug_print(
            f"process_area: artifacts for '{area_inputs.name}' already exist, skipping preprocessing"
        )

        # Load existing normalization stats if this is the training area
        existing_stats = stats
        if area_inputs.name == "train" and os.path.exists(normalization_stats_path):
            with open(normalization_stats_path, "r") as f:
                existing_stats = json.load(f)
            debug_print(
                "process_area: loaded existing normalization stats from training area"
            )

        # Return existing artifacts
        artifacts = AreaArtifacts(
            name=area_inputs.name,
            feature_stack_path=feature_stack_path,
            mask_path=mask_path,
            metadata_path=metadata_path,
            ground_truth_path=ground_truth_path,
            normalization_stats_path=(
                normalization_stats_path if area_inputs.name == "train" else None
            ),
        )
        return artifacts, existing_stats

    debug_print(
        f"process_area: processing area '{area_inputs.name}' (force_recreate={force_recreate})"
    )

    with rasterio.open(area_inputs.dtm) as src:
        dtm = src.read(1).astype(np.float32)
        profile = src.profile
    debug_print(f"process_area: loaded DTM with shape {dtm.shape}")

    valid_mask = np.ones_like(dtm, dtype=bool)
    nodata = profile.get("nodata")
    if nodata is not None:
        valid_mask &= dtm != nodata
    valid_mask &= ~np.isnan(dtm)

    dtm_filled = fill_nodata(dtm, ~valid_mask)
    debug_print("process_area: filled nodata regions in DTM")
    dtm_filled = simple_sink_fill(
        dtm_filled, preprocessing_cfg["dtm_hygiene"].get("sink_fill_kernel", 0)
    )
    debug_print("process_area: applied sink fill to DTM")
    gaussian_sigma = preprocessing_cfg["dtm_hygiene"].get("gaussian_sigma", 0.0)
    if gaussian_sigma and gaussian_sigma > 0.0:
        dtm_filled = gaussian_filter(dtm_filled, sigma=gaussian_sigma)
        debug_print(
            f"process_area: applied Gaussian smoothing with sigma={gaussian_sigma}"
        )

    dem_features, dem_meta = compute_dem_features(
        dtm_filled, profile["transform"], valid_mask, preprocessing_cfg
    )
    debug_print("process_area: computed DEM derivative features")

    resampling_map = {
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
        "cubic": Resampling.cubic,
    }
    ortho_resampling = resampling_map.get(
        preprocessing_cfg["resampling"].get("ortho", "bilinear"), Resampling.bilinear
    )
    ortho_array, _ = reproject_raster(
        area_inputs.ortho, profile, ortho_resampling, dst_dtype=np.float32
    )
    debug_print(f"process_area: reprojected orthophoto {area_inputs.ortho}")
    ortho_norm = normalize_radiometry(
        ortho_array, valid_mask, preprocessing_cfg["normalization"]["epsilon"]
    )
    debug_print("process_area: normalized orthophoto radiometry")

    # Fetch LULC data: external or K-means clustering
    lulc_class_info = {}
    if preprocessing_cfg.get("external_lulc", {}).get("enabled", False):
        debug_print("process_area: using external LULC data")
        land_cover_one_hot, land_cover_labels, lulc_class_info = fetch_external_lulc(
            area_inputs.dtm,
            area_inputs.name,
            area_dir,
            preprocessing_cfg["external_lulc"],
            valid_mask,
            profile,
        )
    else:
        debug_print("process_area: using K-means clustering for LULC")
        land_cover_one_hot, land_cover_labels = derive_land_cover(
            ortho_norm,
            valid_mask,
            preprocessing_cfg["orthophoto_channels"].get("land_cover_clusters", 6),
            seed,
        )

    feature_stack, channel_names = build_feature_stack(
        dem_features,
        ortho_norm,
        land_cover_one_hot,
        preprocessing_cfg,
        valid_mask,
    )

    if stats is not None and "channel_names" in stats:
        reference_names = list(stats["channel_names"])
        if channel_names != reference_names:
            name_to_idx = {name: idx for idx, name in enumerate(channel_names)}
            missing = [name for name in reference_names if name not in name_to_idx]
            if missing:
                raise ValueError(f"Channels missing in {area_inputs.name}: {missing}")
            selected_indices = [name_to_idx[name] for name in reference_names]
            feature_stack = feature_stack[selected_indices, :, :]
            channel_names = reference_names

    normalization_stats_path = os.path.join(metadata_dir, "normalization_stats.json")
    if stats is None:
        normalized_stack, stats = apply_normalization(
            feature_stack,
            valid_mask,
            preprocessing_cfg["normalization"],
            existing_stats=None,
        )
        stats["channel_names"] = list(channel_names)
    else:
        normalized_stack, _ = apply_normalization(
            feature_stack,
            valid_mask,
            preprocessing_cfg["normalization"],
            existing_stats=stats,
        )

    feature_stack_path = os.path.join(area_dir, "feature_stack.tif")
    save_geotiff(feature_stack_path, normalized_stack, profile, dtype="float32")

    mask_path = os.path.join(area_dir, "valid_mask.tif")
    save_geotiff(
        mask_path, valid_mask.astype(np.uint8), profile, dtype="uint8", nodata=0
    )

    slope_qc_path = os.path.join(area_dir, "slope_deg.tif")
    save_geotiff(slope_qc_path, dem_features["slope_deg"], profile, dtype="float32")
    lulc_qc_path = os.path.join(area_dir, "land_cover.tif")
    save_geotiff(
        lulc_qc_path,
        land_cover_labels.astype(np.int16),
        profile,
        dtype="int16",
        nodata=-1,
    )

    metadata = {
        "channel_names": channel_names,
        "channel_map": {name: idx for idx, name in enumerate(channel_names)},
        "land_cover_clusters": (
            len(lulc_class_info)
            if lulc_class_info
            else preprocessing_cfg["orthophoto_channels"].get("land_cover_clusters", 6)
        ),
        "lulc_source": (
            preprocessing_cfg.get("external_lulc", {}).get("source", "kmeans")
            if preprocessing_cfg.get("external_lulc", {}).get("enabled", False)
            else "kmeans"
        ),
        "lulc_class_info": lulc_class_info if lulc_class_info else {},
        "source_files": {
            "dtm": area_inputs.dtm,
            "ortho": area_inputs.ortho,
            "ground_truth": area_inputs.ground_truth,
        },
        "auxiliary_layers": {
            "slope_deg": slope_qc_path,
            "land_cover": lulc_qc_path,
        },
    }
    metadata_path = os.path.join(area_dir, "feature_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if area_inputs.name == "train":
        with open(normalization_stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    ground_truth_path = None
    if area_inputs.ground_truth:
        gt_resampling = resampling_map.get(
            preprocessing_cfg["resampling"].get("ground_truth", "nearest"),
            Resampling.nearest,
        )
        gt_array, _ = reproject_raster(
            area_inputs.ground_truth, profile, gt_resampling, dst_dtype=np.float32
        )
        gt = gt_array[0]
        gt = np.round(gt).astype(np.int16)

        # CRITICAL FIX: Remap ground truth values to model classes
        # Ground truth encoding: 0=invalid/empty, 1=low, 2=moderate, 3=high
        # Model expects: 0=low, 1=moderate, 2=high, 255=ignore
        # Remap: GT[0]->255, GT[1]->0, GT[2]->1, GT[3]->2
        gt_remapped = np.full_like(gt, 255, dtype=np.int16)
        gt_remapped[gt == 1] = 0  # Low risk
        gt_remapped[gt == 2] = 1  # Moderate risk
        gt_remapped[gt == 3] = 2  # High risk
        # GT value 0 and any other values remain as 255 (ignore)
        gt = gt_remapped

        gt[~valid_mask] = 255
        ignore_mask = compute_ignore_mask(
            gt, preprocessing_cfg.get("boundary_ignore_pixels", 0)
        )
        gt[ignore_mask] = 255
        ground_truth_path = os.path.join(area_dir, "ground_truth_aligned.tif")
        save_geotiff(
            ground_truth_path, gt.astype(np.uint8), profile, dtype="uint8", nodata=255
        )

    artifacts = AreaArtifacts(
        name=area_inputs.name,
        feature_stack_path=feature_stack_path,
        mask_path=mask_path,
        metadata_path=metadata_path,
        ground_truth_path=ground_truth_path,
        normalization_stats_path=normalization_stats_path,
    )
    return artifacts, stats


def preprocess_data(
    config: Dict, force_recreate: bool = False
) -> Dict[str, AreaArtifacts]:
    """Generate feature stacks and metadata for configured train/test splits."""
    debug_print("preprocess_data: starting preprocessing stage")
    structure_cfg = config["project_structure"]
    prepare_directories(structure_cfg)
    inputs_map = load_input_paths(config)
    debug_print(f"preprocess_data: available splits {list(inputs_map.keys())}")
    preprocessing_cfg = config["preprocessing"]
    metadata_dir = structure_cfg["metadata_dir"]
    os.makedirs(metadata_dir, exist_ok=True)

    artifacts: Dict[str, AreaArtifacts] = {}
    stats: Optional[Dict] = None
    seed = config["reproducibility"]["seed"]

    for split_name in ["train", "test"]:
        debug_print(f"preprocess_data: handling split '{split_name}'")
        if split_name not in inputs_map:
            continue
        area_dir = os.path.join(structure_cfg["derived_data_dir"], split_name)
        os.makedirs(area_dir, exist_ok=True)
        artifacts[split_name], stats = process_area(
            inputs_map[split_name],
            area_dir,
            preprocessing_cfg,
            metadata_dir,
            stats,
            seed,
            force_recreate,
        )

    return artifacts


def prepare_dataset(
    config: Dict, train_artifacts: AreaArtifacts, force_recreate: bool = False
) -> None:
    """Tile the training area into spatially blocked splits with sampling heuristics."""
    dataset_cfg = config["dataset"]
    structure_cfg = config["project_structure"]

    # Check if dataset artifacts already exist
    splits_path = os.path.join(structure_cfg["splits_dir"], "splits.json")
    summary_path = os.path.join(structure_cfg["splits_dir"], "dataset_summary.json")

    if (
        os.path.exists(splits_path)
        and os.path.exists(summary_path)
        and not force_recreate
    ):
        debug_print("prepare_dataset: dataset artifacts already exist, skipping tiling")
        # Verify that tiles exist
        with open(splits_path, "r") as f:
            splits_data = json.load(f)

        tiles_exist = True
        for split_name, tile_list in splits_data.items():
            if not tile_list:
                continue
            # Check if at least first and last tile exist
            first_tile = os.path.join(
                structure_cfg["tiles_dir"], split_name, tile_list[0]
            )
            if not os.path.exists(first_tile):
                tiles_exist = False
                break

        if tiles_exist:
            debug_print(
                "prepare_dataset: verified tile files exist, skipping dataset preparation"
            )
            return
        else:
            debug_print(
                "prepare_dataset: splits.json exists but tiles missing, regenerating"
            )

    debug_print(f"prepare_dataset: preparing dataset (force_recreate={force_recreate})")

    tile_size = dataset_cfg["tile_size"]
    tile_overlap = dataset_cfg["tile_overlap"]
    stride = max(1, tile_size - tile_overlap)
    min_valid_fraction = dataset_cfg.get("min_valid_fraction", 0.8)
    rng = random.Random(config["reproducibility"]["seed"])

    with rasterio.open(train_artifacts.feature_stack_path) as src:
        features = src.read().astype(np.float32)
        height = src.height
        width = src.width
        profile = src.profile

    with rasterio.open(train_artifacts.mask_path) as src:
        valid_mask = src.read(1).astype(np.uint8).astype(bool)

    if not train_artifacts.ground_truth_path:
        raise ValueError("Ground truth required for dataset preparation.")

    with rasterio.open(train_artifacts.ground_truth_path) as src:
        ground_truth = src.read(1).astype(np.uint8)

    with open(train_artifacts.metadata_path, "r") as f:
        metadata = json.load(f)

    slope_path = metadata.get("auxiliary_layers", {}).get("slope_deg")
    if slope_path and os.path.exists(slope_path):
        with rasterio.open(slope_path) as src:
            slope_deg = src.read(1).astype(np.float32)
    else:
        slope_deg = features[metadata["channel_map"].get("slope_deg", 0)]

    labels = ground_truth.copy()
    ignore_mask = labels == 255
    labels = np.where(
        ignore_mask, 255, np.clip(labels, 1, config["model"]["out_classes"]) - 1
    )
    labels[~valid_mask] = 255

    # Apply label smoothing if enabled
    label_smoothing_cfg = config["preprocessing"].get("label_smoothing", {})
    use_soft_labels = label_smoothing_cfg.get("enabled", False)

    if use_soft_labels:
        from src.soft_labels import apply_label_smoothing

        smoothing_type = label_smoothing_cfg.get("type", "ordinal")
        alpha = label_smoothing_cfg.get("alpha", 0.1)
        sigma = label_smoothing_cfg.get("sigma", 1.0)
        num_classes = config["model"]["out_classes"]

        debug_print(
            f"prepare_dataset: applying {smoothing_type} label smoothing "
            f"(alpha={alpha}, sigma={sigma})"
        )

        soft_labels = apply_label_smoothing(
            labels,
            smoothing_type=smoothing_type,
            num_classes=num_classes,
            alpha=alpha,
            sigma=sigma,
            ignore_value=255,
        )

        # Validate soft labels
        from src.soft_labels import validate_soft_labels

        try:
            validate_soft_labels(soft_labels)
            debug_print("prepare_dataset: soft labels validated successfully")
        except AssertionError as e:
            debug_print(f"prepare_dataset: WARNING - soft label validation failed: {e}")
    else:
        soft_labels = None
        debug_print("prepare_dataset: using hard labels (label smoothing disabled)")

    os.makedirs(structure_cfg["tiles_dir"], exist_ok=True)
    os.makedirs(structure_cfg["labels_dir"], exist_ok=True)
    os.makedirs(structure_cfg["splits_dir"], exist_ok=True)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(structure_cfg["tiles_dir"], split), exist_ok=True)
        os.makedirs(os.path.join(structure_cfg["labels_dir"], split), exist_ok=True)
        for path in glob.glob(os.path.join(structure_cfg["tiles_dir"], split, "*.npy")):
            os.remove(path)
        for path in glob.glob(
            os.path.join(structure_cfg["labels_dir"], split, "*.npy")
        ):
            os.remove(path)

    block_rows, block_cols = dataset_cfg.get("block_grid", [4, 4])
    y_edges = np.linspace(0, height, block_rows + 1, dtype=int)
    x_edges = np.linspace(0, width, block_cols + 1, dtype=int)
    blocks: List[Tuple[int, int, int, int]] = []
    for i in range(block_rows):
        for j in range(block_cols):
            y0, y1 = y_edges[i], y_edges[i + 1]
            x0, x1 = x_edges[j], x_edges[j + 1]
            blocks.append((y0, y1, x0, x1))

    test_size = dataset_cfg.get("test_size", 0.2)
    val_size = dataset_cfg.get("val_size", 0.15)
    random_state = dataset_cfg.get("random_state", 42)

    # Helper function to check if test split contains all classes
    def check_test_has_all_classes(test_block_list, num_classes):
        """Check if test blocks contain tiles with all classes."""
        classes_found = set()
        positive_class = dataset_cfg.get("positive_class", num_classes - 1)
        positive_min_fraction = dataset_cfg.get("positive_min_fraction", 0.02)

        for y0, y1, x0, x1 in test_block_list:
            for y in range(y0, max(y1 - tile_size + 1, y0), stride):
                for x in range(x0, max(x1 - tile_size + 1, x0), stride):
                    y_end = y + tile_size
                    x_end = x + tile_size
                    if y_end > height or x_end > width:
                        continue
                    tile_mask = valid_mask[y:y_end, x:x_end]
                    if tile_mask.mean() < min_valid_fraction:
                        continue
                    tile_labels = labels[y:y_end, x:x_end]
                    valid_pixels = tile_labels != 255
                    if not np.any(valid_pixels):
                        continue
                    # Check which classes are present in this tile
                    unique_classes = np.unique(tile_labels[valid_pixels])
                    classes_found.update(int(c) for c in unique_classes if c != 255)

                    # Early exit if all classes found
                    if len(classes_found) >= num_classes:
                        return True, classes_found

        return len(classes_found) >= num_classes, classes_found

    if len(blocks) < 2 or test_size <= 0:
        train_blocks = blocks
        test_blocks = []
        effective_test_size = 0.0
    else:
        min_fraction = 1.0 / len(blocks)
        effective_test_size = max(test_size, min_fraction)
        if effective_test_size >= 1.0:
            effective_test_size = max(min_fraction, min(0.5, 1.0 - min_fraction))

        # Try multiple random splits to ensure test has all classes
        max_attempts = dataset_cfg.get("max_split_attempts", 10)
        num_classes = config["model"]["out_classes"]
        best_split = None
        best_classes_found = set()

        for attempt in range(max_attempts):
            current_seed = random_state + attempt
            train_blocks_attempt, test_blocks_attempt = train_test_split(
                blocks, test_size=effective_test_size, random_state=current_seed
            )

            has_all_classes, classes_found = check_test_has_all_classes(
                test_blocks_attempt, num_classes
            )

            if has_all_classes:
                debug_print(
                    f"prepare_dataset: found valid test split on attempt {attempt + 1} "
                    f"with all {num_classes} classes: {sorted(classes_found)}"
                )
                train_blocks = train_blocks_attempt
                test_blocks = test_blocks_attempt
                break

            # Track best attempt even if not perfect
            if len(classes_found) > len(best_classes_found):
                best_classes_found = classes_found
                best_split = (train_blocks_attempt, test_blocks_attempt)
        else:
            # Max attempts reached without finding all classes
            debug_print(
                f"prepare_dataset: WARNING - could not find test split with all {num_classes} classes "
                f"after {max_attempts} attempts. Best split found classes: {sorted(best_classes_found)}"
            )
            if best_split:
                train_blocks, test_blocks = best_split
            else:
                train_blocks, test_blocks = train_test_split(
                    blocks, test_size=effective_test_size, random_state=random_state
                )

    if not train_blocks:
        train_blocks = blocks
        test_blocks = []
        effective_test_size = 0.0

    if val_size > 0 and len(train_blocks) > 1:
        remaining_fraction = 1.0 - effective_test_size if test_blocks else 1.0
        adjusted_val = val_size / max(1e-6, remaining_fraction)
        adjusted_val = max(adjusted_val, 1.0 / len(train_blocks))
        adjusted_val = min(adjusted_val, 0.9)
        if adjusted_val >= 1.0:
            adjusted_val = 0.5
        train_blocks, val_blocks = train_test_split(
            train_blocks, test_size=adjusted_val, random_state=random_state
        )
    else:
        val_blocks = []
    split_blocks = {
        "train": train_blocks,
        "val": val_blocks,
        "test": test_blocks,
    }

    split_positions: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
        "train": {"pos": [], "neg": []},
        "val": {"pos": [], "neg": []},
        "test": {"pos": [], "neg": []},
    }

    positive_class = dataset_cfg.get(
        "positive_class", config["model"]["out_classes"] - 1
    )
    positive_min_fraction = dataset_cfg.get("positive_min_fraction", 0.02)
    slope_threshold = dataset_cfg.get("negative_slope_threshold_deg", 30.0)

    for split_name, block_list in split_blocks.items():
        for y0, y1, x0, x1 in block_list:
            for y in range(y0, max(y1 - tile_size + 1, y0), stride):
                for x in range(x0, max(x1 - tile_size + 1, x0), stride):
                    y_end = y + tile_size
                    x_end = x + tile_size
                    if y_end > height or x_end > width:
                        continue
                    tile_mask = valid_mask[y:y_end, x:x_end]
                    if tile_mask.mean() < min_valid_fraction:
                        continue
                    tile_labels = labels[y:y_end, x:x_end]
                    valid_pixels = tile_labels != 255
                    if not np.any(valid_pixels):
                        continue
                    positive_fraction = np.mean(
                        tile_labels[valid_pixels] == positive_class
                    )
                    slope_tile = slope_deg[y:y_end, x:x_end]
                    slope_mean = float(np.mean(slope_tile[tile_mask]))
                    entry = (y, x)
                    if positive_fraction >= positive_min_fraction:
                        split_positions[split_name]["pos"].append(entry)
                    elif slope_mean >= slope_threshold:
                        split_positions[split_name]["neg"].append(entry)

    def _count_positions(split_name: str) -> int:
        groups = split_positions[split_name]
        return len(groups["pos"]) + len(groups["neg"])

    def _pop_random(
        entries: List[Tuple[int, int]], count: int
    ) -> List[Tuple[int, int]]:
        if count <= 0 or not entries:
            return []
        count = min(count, len(entries))
        indices = rng.sample(range(len(entries)), count)
        indices.sort(reverse=True)
        return [entries.pop(idx) for idx in indices]

    def _reassign_from_train(target_split: str, required: int) -> None:
        if required <= 0:
            return
        train_total = _count_positions("train")
        if train_total <= 1:
            return
        move_total = min(required, train_total - 1)
        if move_total <= 0:
            return
        train_pos = split_positions["train"]["pos"]
        train_neg = split_positions["train"]["neg"]
        train_total = len(train_pos) + len(train_neg)
        if train_total == 0:
            return
        pos_ratio = len(train_pos) / train_total if train_total else 0.0
        pos_to_move = min(len(train_pos), int(round(move_total * pos_ratio)))
        neg_to_move = move_total - pos_to_move
        if neg_to_move > len(train_neg):
            neg_to_move = len(train_neg)
        assigned = pos_to_move + neg_to_move
        if assigned < move_total:
            remaining = move_total - assigned
            extra_pos = min(remaining, len(train_pos) - pos_to_move)
            pos_to_move += extra_pos
            remaining -= extra_pos
            if remaining > 0:
                extra_neg = min(remaining, len(train_neg) - neg_to_move)
                neg_to_move += extra_neg
        if pos_to_move == 0 and len(train_pos) > 0 and move_total > 0:
            pos_to_move = min(len(train_pos), move_total)
            neg_to_move = move_total - pos_to_move
            if neg_to_move > len(train_neg):
                neg_to_move = len(train_neg)
        neg_to_move = max(0, neg_to_move)
        moved_pos = _pop_random(train_pos, pos_to_move)
        moved_neg = _pop_random(train_neg, neg_to_move)
        split_positions[target_split]["pos"].extend(moved_pos)
        split_positions[target_split]["neg"].extend(moved_neg)

    total_candidates = sum(_count_positions(split) for split in split_positions)

    if (
        (test_size > 0 and _count_positions("test") == 0)
        or (val_size > 0 and _count_positions("val") == 0)
    ) and total_candidates > 0:
        debug_print(
            "prepare_dataset: insufficient hold-out tiles from block grid; reallocating from training set"
        )

    if test_size > 0 and total_candidates > 0:
        desired_test = int(round(total_candidates * test_size))
        if desired_test == 0:
            desired_test = 1
        need_test = desired_test - _count_positions("test")
        if need_test > 0:
            _reassign_from_train("test", need_test)

    if val_size > 0 and total_candidates > 0:
        desired_val = int(round(total_candidates * val_size))
        if desired_val == 0:
            desired_val = 1
        need_val = desired_val - _count_positions("val")
        if need_val > 0:
            _reassign_from_train("val", need_val)

    tile_records: Dict[str, List[str]] = {split: [] for split in split_blocks}
    class_pixel_counts: Dict[str, int] = {
        str(idx): 0 for idx in range(config["model"]["out_classes"])
    }
    ignore_pixel_count = 0

    for split_name, groups in split_positions.items():
        pos_positions = groups["pos"]
        neg_positions = groups["neg"]
        desired_pos_fraction = dataset_cfg.get("positive_fraction", 0.5)
        if not pos_positions:
            selected_positions = neg_positions
        else:
            neg_target = int(
                round(
                    len(pos_positions)
                    * (1.0 - desired_pos_fraction)
                    / max(desired_pos_fraction, 1e-6)
                )
            )
            neg_target = min(len(neg_positions), neg_target)
            neg_selected = (
                rng.sample(neg_positions, neg_target) if neg_target > 0 else []
            )
            selected_positions = pos_positions + neg_selected
        if dataset_cfg.get("max_tiles_per_split"):
            max_tiles = dataset_cfg["max_tiles_per_split"]
            if max_tiles and len(selected_positions) > max_tiles:
                selected_positions = rng.sample(selected_positions, max_tiles)
        rng.shuffle(selected_positions)

        split_tile_dir = os.path.join(structure_cfg["tiles_dir"], split_name)
        split_label_dir = os.path.join(structure_cfg["labels_dir"], split_name)

        for y, x in selected_positions:
            tile_features = features[:, y : y + tile_size, x : x + tile_size]
            tile_labels = labels[y : y + tile_size, x : x + tile_size].copy()
            tile_mask = valid_mask[y : y + tile_size, x : x + tile_size]
            tile_labels[~tile_mask] = 255
            if not np.any(tile_labels != 255):
                continue
            tile_name = f"{split_name}_tile_{y}_{x}.npy"
            np.save(
                os.path.join(split_tile_dir, tile_name),
                tile_features.astype(np.float32),
            )

            # Save either soft or hard labels
            if use_soft_labels:
                # Extract soft label tile: shape (num_classes, tile_size, tile_size)
                tile_soft_labels = soft_labels[
                    :, y : y + tile_size, x : x + tile_size
                ].copy()
                # Zero out invalid pixels
                tile_soft_labels[:, ~tile_mask] = 0.0
                np.save(
                    os.path.join(split_label_dir, tile_name),
                    tile_soft_labels.astype(np.float32),
                )
            else:
                # Save hard labels as uint8
                np.save(
                    os.path.join(split_label_dir, tile_name),
                    tile_labels.astype(np.uint8),
                )

            tile_records[split_name].append(tile_name)

            # Count pixels per class (works for both hard and soft labels)
            valid_pixels = tile_labels != 255
            if np.any(valid_pixels):
                if use_soft_labels:
                    # For soft labels, count fractional class membership
                    tile_soft_valid = tile_soft_labels[:, valid_pixels]
                    for cls_idx in range(config["model"]["out_classes"]):
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
            ignore_pixel_count += int(np.sum(tile_labels == 255))

    # Class 1 (Medium-Risk) Oversampling for Training Set
    # Addresses severe class imbalance: Class 1 is only ~2.65% of data
    # Strategy: Duplicate Class 1-rich tiles to increase representation to ~5%
    oversample_class = dataset_cfg.get("oversample_class", None)
    oversample_target_fraction = dataset_cfg.get("oversample_target_fraction", 0.05)

    if oversample_class is not None and "train" in split_positions:
        debug_print(f"prepare_dataset: Applying Class {oversample_class} oversampling")

        train_tile_dir = os.path.join(structure_cfg["tiles_dir"], "train")
        train_label_dir = os.path.join(structure_cfg["labels_dir"], "train")

        # Identify tiles rich in target class
        class_rich_tiles = []
        for tile_name in tile_records["train"]:
            tile_label_path = os.path.join(train_label_dir, tile_name)
            tile_labels_arr = np.load(tile_label_path)

            # Calculate fraction of target class
            if use_soft_labels:
                # Soft labels: shape (num_classes, H, W)
                class_fraction = np.mean(tile_labels_arr[oversample_class])
            else:
                # Hard labels: shape (H, W)
                valid_pixels = tile_labels_arr != 255
                if np.any(valid_pixels):
                    class_fraction = np.mean(
                        tile_labels_arr[valid_pixels] == oversample_class
                    )
                else:
                    class_fraction = 0.0

            # Consider "rich" if class comprises >5% of tile
            if class_fraction > 0.05:
                class_rich_tiles.append((tile_name, class_fraction))

        if class_rich_tiles:
            # Sort by class fraction (most rich first)
            class_rich_tiles.sort(key=lambda x: x[1], reverse=True)

            # Calculate current and target counts
            current_train_count = len(tile_records["train"])
            current_class_pixels = class_pixel_counts.get(str(oversample_class), 0)
            total_pixels = sum(class_pixel_counts.values())
            current_fraction = current_class_pixels / max(total_pixels, 1)

            # Determine how many duplicates needed
            # Target: oversample_target_fraction of dataset
            target_class_pixels = oversample_target_fraction * total_pixels
            deficit_pixels = max(0, target_class_pixels - current_class_pixels)

            debug_print(
                f"  Current Class {oversample_class} fraction: {current_fraction:.4f} "
                f"({current_class_pixels:.0f} / {total_pixels:.0f} pixels)"
            )
            debug_print(
                f"  Target Class {oversample_class} fraction: {oversample_target_fraction:.4f} "
                f"({target_class_pixels:.0f} pixels)"
            )
            debug_print(f"  Deficit: {deficit_pixels:.0f} pixels")
            debug_print(
                f"  Found {len(class_rich_tiles)} Class {oversample_class}-rich tiles"
            )

            # Duplicate tiles until deficit is filled
            duplicates_added = 0
            pixels_added = 0

            for tile_name, class_frac in class_rich_tiles:
                if pixels_added >= deficit_pixels:
                    break

                # Load original tile
                tile_feature_path = os.path.join(train_tile_dir, tile_name)
                tile_label_path = os.path.join(train_label_dir, tile_name)
                tile_features = np.load(tile_feature_path)
                tile_labels_arr = np.load(tile_label_path)

                # Create duplicate with different name
                base_name = tile_name.replace(".npy", "")
                dup_name = f"{base_name}_dup{duplicates_added}.npy"

                # Save duplicate
                np.save(os.path.join(train_tile_dir, dup_name), tile_features)
                np.save(os.path.join(train_label_dir, dup_name), tile_labels_arr)
                tile_records["train"].append(dup_name)

                # Update pixel counts
                if use_soft_labels:
                    valid_mask_dup = tile_labels_arr.sum(axis=0) > 1e-6
                    tile_soft_valid = tile_labels_arr[:, valid_mask_dup]
                    for cls_idx in range(config["model"]["out_classes"]):
                        count = float(np.sum(tile_soft_valid[cls_idx]))
                        class_pixel_counts[str(cls_idx)] += count
                        if cls_idx == oversample_class:
                            pixels_added += count
                else:
                    valid_pixels = tile_labels_arr != 255
                    if np.any(valid_pixels):
                        unique, counts = np.unique(
                            tile_labels_arr[valid_pixels], return_counts=True
                        )
                        for cls, count in zip(unique, counts):
                            class_pixel_counts[str(int(cls))] += int(count)
                            if int(cls) == oversample_class:
                                pixels_added += int(count)

                duplicates_added += 1

            # Final stats
            total_pixels_after = sum(class_pixel_counts.values())
            final_class_pixels = class_pixel_counts.get(str(oversample_class), 0)
            final_fraction = final_class_pixels / max(total_pixels_after, 1)

            debug_print(
                f"  ✓ Added {duplicates_added} duplicate tiles (+{pixels_added:.0f} Class {oversample_class} pixels)"
            )
            debug_print(
                f"  Final Class {oversample_class} fraction: {final_fraction:.4f} "
                f"({final_class_pixels:.0f} / {total_pixels_after:.0f} pixels)"
            )
        else:
            debug_print(
                f"  WARNING: No Class {oversample_class}-rich tiles found for oversampling"
            )

    # V2.5: SMOTE Synthetic Data Generation
    # Generates synthetic Class 1 examples to reach target fraction
    use_smote = dataset_cfg.get("use_smote", False)
    smote_k_neighbors = dataset_cfg.get("smote_k_neighbors", 5)

    if (
        use_smote
        and oversample_class is not None
        and "train" in split_positions
        and SMOTE_AVAILABLE
    ):
        debug_print(f"prepare_dataset: Applying SMOTE for Class {oversample_class}")

        train_tile_dir = os.path.join(structure_cfg["tiles_dir"], "train")
        train_label_dir = os.path.join(structure_cfg["labels_dir"], "train")

        # Check if we need more samples
        current_class_pixels = class_pixel_counts.get(str(oversample_class), 0)
        total_pixels = sum(class_pixel_counts.values())
        current_fraction = current_class_pixels / max(total_pixels, 1)

        if current_fraction < oversample_target_fraction:
            debug_print(
                f"  Current fraction: {current_fraction:.4f}, Target: {oversample_target_fraction:.4f}"
            )
            debug_print(f"  Generating synthetic tiles with SMOTE...")

            # Collect tiles with target class
            class_tiles = []
            for tile_name in tile_records["train"]:
                tile_label_path = os.path.join(train_label_dir, tile_name)
                tile_labels_arr = np.load(tile_label_path)

                # Check if tile has target class
                has_class = False
                if use_soft_labels:
                    has_class = np.mean(tile_labels_arr[oversample_class]) > 0.01
                else:
                    valid_pixels = tile_labels_arr != 255
                    if np.any(valid_pixels):
                        has_class = np.any(
                            tile_labels_arr[valid_pixels] == oversample_class
                        )

                if has_class:
                    class_tiles.append(tile_name)

            debug_print(
                f"  Found {len(class_tiles)} tiles with Class {oversample_class}"
            )

            if len(class_tiles) >= smote_k_neighbors + 1:
                # Load tiles into array for SMOTE
                # Sample max 50 tiles to keep memory reasonable
                sample_size = min(50, len(class_tiles))
                sampled_tiles = random.sample(class_tiles, sample_size)

                # Flatten tiles: each tile becomes one "sample" with flattened features
                X_samples = []
                y_samples = []

                for tile_name in sampled_tiles:
                    tile_feature_path = os.path.join(train_tile_dir, tile_name)
                    tile_label_path = os.path.join(train_label_dir, tile_name)

                    tile_features = np.load(tile_feature_path)  # (C, H, W)
                    tile_labels = np.load(tile_label_path)

                    # Flatten: (C, H, W) -> (C, H*W) -> (H*W, C)
                    C, H, W = tile_features.shape
                    flat_features = tile_features.reshape(C, H * W).T  # (H*W, C)

                    if use_soft_labels:
                        # Use argmax for SMOTE
                        flat_labels = np.argmax(
                            tile_labels.reshape(config["model"]["out_classes"], H * W),
                            axis=0,
                        )
                    else:
                        flat_labels = tile_labels.flatten()

                    # Filter out ignore pixels
                    valid_mask = flat_labels != 255
                    X_samples.append(flat_features[valid_mask])
                    y_samples.append(flat_labels[valid_mask])

                # Concatenate all samples
                X = np.vstack(X_samples)
                y = np.concatenate(y_samples)

                # Calculate sampling strategy
                unique_classes, class_counts = np.unique(y, return_counts=True)
                class_count_dict = dict(zip(unique_classes, class_counts))

                # Target count for minority class
                target_count = int(oversample_target_fraction * len(y))
                current_count = class_count_dict.get(oversample_class, 0)

                if current_count > 0 and target_count > current_count:
                    sampling_strategy = {oversample_class: target_count}

                    debug_print(
                        f"  Running SMOTE: {current_count} -> {target_count} samples"
                    )

                    try:
                        smote = SMOTE(
                            sampling_strategy=sampling_strategy,
                            k_neighbors=smote_k_neighbors,
                            random_state=dataset_cfg.get("random_state", 42),
                        )
                        X_resampled, y_resampled = smote.fit_resample(X, y)

                        # Create synthetic tiles from resampled data
                        synthetic_pixels = len(y_resampled) - len(y)
                        synthetic_tiles_needed = int(
                            np.ceil(synthetic_pixels / (H * W * 0.8))
                        )  # 80% valid pixels

                        debug_print(f"  Generated {synthetic_pixels} synthetic pixels")
                        debug_print(
                            f"  Creating {synthetic_tiles_needed} synthetic tiles..."
                        )

                        # Create synthetic tiles
                        synth_tile_count = 0
                        idx_offset = len(y)  # Start from synthetic samples

                        for i in range(synthetic_tiles_needed):
                            if idx_offset >= len(y_resampled):
                                break

                            # Sample pixels for this tile
                            tile_pixels = min(H * W, len(y_resampled) - idx_offset)
                            tile_X = X_resampled[idx_offset : idx_offset + tile_pixels]
                            tile_y = y_resampled[idx_offset : idx_offset + tile_pixels]

                            # Reshape to tile format
                            # Pad if needed
                            if tile_pixels < H * W:
                                pad_size = H * W - tile_pixels
                                tile_X = np.vstack([tile_X, np.zeros((pad_size, C))])
                                tile_y = np.concatenate(
                                    [tile_y, np.full(pad_size, 255)]
                                )

                            tile_features = tile_X.T.reshape(C, H, W)
                            tile_labels = tile_y.reshape(H, W)

                            # Save synthetic tile
                            synth_name = f"train_smote_{synth_tile_count}.npy"
                            np.save(
                                os.path.join(train_tile_dir, synth_name),
                                tile_features.astype(np.float32),
                            )

                            if use_soft_labels:
                                # Convert to soft labels
                                tile_labels_soft = np.zeros(
                                    (config["model"]["out_classes"], H, W),
                                    dtype=np.float32,
                                )
                                for cls in range(config["model"]["out_classes"]):
                                    tile_labels_soft[cls] = (tile_labels == cls).astype(
                                        np.float32
                                    )
                                np.save(
                                    os.path.join(train_label_dir, synth_name),
                                    tile_labels_soft,
                                )
                            else:
                                np.save(
                                    os.path.join(train_label_dir, synth_name),
                                    tile_labels.astype(np.uint8),
                                )

                            tile_records["train"].append(synth_name)

                            # Update pixel counts
                            valid_pixels = tile_labels != 255
                            if np.any(valid_pixels):
                                unique, counts = np.unique(
                                    tile_labels[valid_pixels], return_counts=True
                                )
                                for cls, count in zip(unique, counts):
                                    class_pixel_counts[str(int(cls))] += int(count)

                            synth_tile_count += 1
                            idx_offset += tile_pixels

                        # Final stats
                        total_pixels_after_smote = sum(class_pixel_counts.values())
                        final_class_pixels_smote = class_pixel_counts.get(
                            str(oversample_class), 0
                        )
                        final_fraction_smote = final_class_pixels_smote / max(
                            total_pixels_after_smote, 1
                        )

                        debug_print(f"  ✓ Added {synth_tile_count} synthetic tiles")
                        debug_print(
                            f"  Final Class {oversample_class} fraction after SMOTE: {final_fraction_smote:.4f} "
                            f"({final_class_pixels_smote:.0f} / {total_pixels_after_smote:.0f} pixels)"
                        )

                    except Exception as e:
                        debug_print(f"  WARNING: SMOTE failed: {e}")
                else:
                    debug_print(
                        f"  Target already met or insufficient samples, skipping SMOTE"
                    )
            else:
                debug_print(
                    f"  WARNING: Not enough tiles ({len(class_tiles)}) for SMOTE (need >{smote_k_neighbors})"
                )
        else:
            debug_print(
                f"  Target fraction already met ({current_fraction:.4f} >= {oversample_target_fraction:.4f})"
            )

    elif use_smote and not SMOTE_AVAILABLE:
        debug_print(
            "  WARNING: SMOTE enabled but imbalanced-learn not installed. Skipping SMOTE."
        )

    splits_path = os.path.join(structure_cfg["splits_dir"], "splits.json")
    with open(splits_path, "w") as f:
        json.dump(tile_records, f, indent=2)

    # Validate test split class distribution
    num_classes = config["model"]["out_classes"]
    test_classes_found = set()

    if tile_records["test"]:
        debug_print("prepare_dataset: validating test split class distribution")
        test_label_dir = os.path.join(structure_cfg["labels_dir"], "test")

        for tile_name in tile_records["test"]:
            tile_label_path = os.path.join(test_label_dir, tile_name)
            tile_labels = np.load(tile_label_path)

            # Handle both soft and hard labels
            if use_soft_labels:
                # For soft labels: shape (num_classes, H, W)
                # Check which classes have significant probability mass
                for cls_idx in range(num_classes):
                    if np.any(tile_labels[cls_idx] > 0.1):  # Threshold for soft labels
                        test_classes_found.add(cls_idx)
            else:
                # For hard labels: shape (H, W)
                valid_pixels = tile_labels != 255
                if np.any(valid_pixels):
                    unique_classes = np.unique(tile_labels[valid_pixels])
                    test_classes_found.update(
                        int(c) for c in unique_classes if c != 255
                    )

        missing_classes = set(range(num_classes)) - test_classes_found
        if missing_classes:
            debug_print(
                f"prepare_dataset: WARNING - Test split missing classes: {sorted(missing_classes)}. "
                f"Found classes: {sorted(test_classes_found)}. "
                "This may affect evaluation. Consider increasing test_size or max_split_attempts."
            )
        else:
            debug_print(
                f"prepare_dataset: ✓ Test split contains all {num_classes} classes: {sorted(test_classes_found)}"
            )

    summary_path = os.path.join(structure_cfg["splits_dir"], "dataset_summary.json")
    dataset_summary = {
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "tile_counts": {split: len(names) for split, names in tile_records.items()},
        "class_pixel_counts": class_pixel_counts,
        "ignore_pixel_count": ignore_pixel_count,
        "test_classes_found": sorted(test_classes_found) if test_classes_found else [],
        "feature_stack_profile": {
            "height": height,
            "width": width,
            "transform": (
                list(profile["transform"])
                if isinstance(profile["transform"], Affine)
                else profile["transform"]
            ),
        },
        "label_smoothing": {
            "enabled": use_soft_labels,
            "type": (
                label_smoothing_cfg.get("type", "none") if use_soft_labels else "none"
            ),
            "alpha": label_smoothing_cfg.get("alpha", 0.0) if use_soft_labels else 0.0,
            "sigma": label_smoothing_cfg.get("sigma", 0.0) if use_soft_labels else 0.0,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(dataset_summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end landslide susceptibility pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main_pipeline                    # Resume from last checkpoint
  python -m src.main_pipeline --force_recreate   # Force full pipeline recreation
        """,
    )
    parser.add_argument(
        "--force_recreate",
        action="store_true",
        help="Force recreation of all artifacts, ignoring existing checkpoints",
    )
    args = parser.parse_args()

    debug_print("main: loading configuration from config.yaml")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    debug_print("main: initializing random seeds")
    random.seed(config["reproducibility"]["seed"])
    np.random.seed(config["reproducibility"]["seed"])

    debug_print(f"main: starting pipeline (force_recreate={args.force_recreate})")

    debug_print("main: starting preprocessing stage")
    artifacts = preprocess_data(config, force_recreate=args.force_recreate)
    debug_print("main: preprocessing complete")

    # V2.5 Enhancement: Mixed-domain dataset preparation
    # Merges training and test areas to improve geographic generalization
    use_mixed_domain = config.get("dataset", {}).get("use_mixed_domain", False)

    if use_mixed_domain and "train" in artifacts and "test" in artifacts:
        debug_print("main: preparing MIXED-DOMAIN dataset (train + test areas)")
        from src.prepare_mixed_domain_dataset import prepare_mixed_domain_dataset

        prepare_mixed_domain_dataset(
            config,
            artifacts["train"],
            artifacts["test"],
            force_recreate=args.force_recreate,
        )
        debug_print("main: mixed-domain dataset preparation complete")
    else:
        debug_print("main: preparing training dataset tiles (single area)")
        prepare_dataset(config, artifacts["train"], force_recreate=args.force_recreate)
        debug_print("main: dataset preparation complete")

    debug_print("main: launching model training")
    training_artifacts = train_model(
        config, artifacts["train"], force_recreate=args.force_recreate
    )
    debug_print("main: training complete")

    debug_print("main: running inference and exports")
    run_inference(
        config, artifacts, training_artifacts, force_recreate=args.force_recreate
    )
    debug_print("main: pipeline finished successfully")


if __name__ == "__main__":
    main()
