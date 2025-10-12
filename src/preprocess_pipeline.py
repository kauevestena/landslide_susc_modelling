"""
Standalone preprocessing runner for landslide susceptibility modelling.

- Reads input paths from config.yaml/inputs.py (or CLI overrides).
- Generates cleaned DEM, terrain/hydrology derivatives, normalized RGB, optional
  shadow mask, land-cover clusters, and writes an aligned feature stack with
  z-score normalization plus metadata.

Usage:
    python -m src.preprocess_pipeline --config config.yaml --area train
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    maximum_filter,
    minimum_filter,
    uniform_filter,
)
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AreaInputs:
    name: str
    dtm: str
    ortho: str
    ground_truth: Optional[str] = None


@dataclass
class AreaArtifacts:
    name: str
    feature_stack_path: str
    mask_path: str
    metadata_path: str
    ground_truth_path: Optional[str] = None
    normalization_stats_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Utility helpers (ported from main pipeline)
# ---------------------------------------------------------------------------

def fill_nodata(array: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    if not np.any(invalid_mask):
        return array
    filled = array.copy()
    _, indices = distance_transform_edt(invalid_mask, return_indices=True)
    filled[invalid_mask] = array[tuple(indices[:, invalid_mask])]
    return filled


def simple_sink_fill(array: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size < 3:
        return array
    size = (kernel_size, kernel_size)
    return minimum_filter(maximum_filter(array, size=size), size=size)


def d8_flow_accumulation(elevation: np.ndarray, valid_mask: np.ndarray, cellsize: float) -> np.ndarray:
    nrows, ncols = elevation.shape
    flow = np.ones((nrows, ncols), dtype=np.float32)
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),          (0, 1),
               (1, -1), (1, 0),  (1, 1)]
    distances = [math.sqrt(2) * cellsize,
                 cellsize,
                 math.sqrt(2) * cellsize,
                 cellsize,
                 cellsize,
                 math.sqrt(2) * cellsize,
                 cellsize,
                 math.sqrt(2) * cellsize]
    order = np.argsort(elevation, axis=None)[::-1]
    for index in order:
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


def compute_dem_features(elevation: np.ndarray, transform: Affine, valid_mask: np.ndarray,
                         drainage_threshold: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    cellsize_x = abs(transform.a)
    cellsize_y = abs(transform.e)

    dz_dy, dz_dx = np.gradient(elevation, cellsize_y, cellsize_x)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
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

    denom = dz_dx ** 2 + dz_dy ** 2
    denom_safe = np.where(denom == 0.0, 1e-6, denom)
    plan_curvature = (dz_dx ** 2 * t - 2.0 * dz_dx * dz_dy * s + dz_dy ** 2 * r) / (denom_safe * np.sqrt(1.0 + denom_safe))
    profile_curvature = (dz_dx ** 2 * r + 2.0 * dz_dx * dz_dy * s + dz_dy ** 2 * t) / (denom_safe * (1.0 + denom_safe) ** 1.5)
    general_curvature = r + t

    window = 9
    mean_local = uniform_filter(elevation, size=window, mode='nearest')
    tpi = elevation - mean_local
    tri = np.sqrt(uniform_filter((elevation - mean_local) ** 2, size=window, mode='nearest'))

    cellsize = (cellsize_x + cellsize_y) * 0.5
    flow = d8_flow_accumulation(elevation, valid_mask, cellsize=cellsize)
    flow_area = flow * cellsize_x * cellsize_y
    log_flow = np.log1p(flow_area)

    slope_safe = np.clip(slope_rad, a_min=math.radians(0.1), a_max=None)
    tan_slope = np.tan(slope_safe)
    tan_slope = np.where(tan_slope < 1e-6, 1e-6, tan_slope)
    twi = np.log((flow_area + 1.0) / tan_slope)
    spi = np.log1p(flow_area * tan_slope)
    sti = np.log1p(((flow_area / 22.13) ** 0.6) * ((np.sin(slope_safe) / 0.0896) ** 1.3))

    drainage_mask = flow_area >= drainage_threshold
    if np.any(drainage_mask):
        distance = distance_transform_edt(~drainage_mask, sampling=(cellsize_y, cellsize_x))
    else:
        distance = np.zeros_like(elevation, dtype=np.float32)
    distance_log = np.log1p(distance)

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

    meta = {"slope_deg": slope_deg.astype(np.float32),
            "aspect_deg": aspect_deg.astype(np.float32),
            "flow_area": flow_area.astype(np.float32),
            "drainage_mask": drainage_mask.astype(np.uint8)}
    return features, meta


def reproject_raster(src_path: str, reference_profile: Dict, resampling: Resampling,
                     dst_dtype: Optional[np.dtype] = None) -> Tuple[np.ndarray, Optional[float]]:
    dtype = dst_dtype or np.float32
    with rasterio.open(src_path) as src:
        count = src.count
        dst = np.zeros((count, reference_profile["height"], reference_profile["width"]), dtype=dtype)
        for band_idx in range(1, count + 1):
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


def normalize_radiometry(ortho: np.ndarray, valid_mask: np.ndarray, epsilon: float) -> np.ndarray:
    normalized = ortho.astype(np.float32).copy()
    for band_idx in range(normalized.shape[0]):
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


def derive_shadow_mask(ortho: np.ndarray, valid_mask: np.ndarray, percentile: float = 15.0) -> np.ndarray:
    intensity = ortho.mean(axis=0)
    valid_intensity = intensity[valid_mask]
    if valid_intensity.size == 0:
        shadow_mask = np.zeros_like(intensity, dtype=np.uint8)
    else:
        threshold = np.percentile(valid_intensity, percentile)
        shadow_mask = (intensity < threshold).astype(np.uint8)
    shadow_mask[~valid_mask] = 0
    return shadow_mask


def derive_land_cover(ortho: np.ndarray, valid_mask: np.ndarray, clusters: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    height, width = valid_mask.shape
    flattened = ortho.reshape(ortho.shape[0], -1).T
    valid_flat = flattened[valid_mask.ravel()]
    if valid_flat.size == 0:
        one_hot = np.zeros((clusters, height, width), dtype=np.float32)
        labels = np.full((height, width), fill_value=-1, dtype=np.int16)
        return one_hot, labels
    max_samples = 200000
    rng = np.random.default_rng(seed)
    sample = valid_flat
    if sample.shape[0] > max_samples:
        idx = rng.choice(sample.shape[0], size=max_samples, replace=False)
        sample = sample[idx]
    kmeans = KMeans(n_clusters=clusters, random_state=seed, n_init="auto")
    kmeans.fit(sample)
    labels_flat = np.full(flattened.shape[0], fill_value=-1, dtype=np.int32)
    labels_flat[valid_mask.ravel()] = kmeans.predict(valid_flat)
    labels = labels_flat.reshape(height, width)
    one_hot = np.zeros((clusters, height, width), dtype=np.float32)
    for idx_cluster in range(clusters):
        one_hot[idx_cluster] = np.where(labels == idx_cluster, 1.0, 0.0)
    one_hot[:, ~valid_mask] = 0.0
    return one_hot, labels.astype(np.int16)


def build_feature_stack(feature_map: Dict[str, np.ndarray],
                        ortho_channels: np.ndarray,
                        land_cover: np.ndarray,
                        shadow_mask: Optional[np.ndarray],
                        derived_cfg: Dict,
                        valid_mask: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    include_cfg = derived_cfg["derived_channels"]
    feature_arrays: List[np.ndarray] = []
    channel_names: List[str] = []

    if include_cfg.get("include_dtm", True):
        feature_arrays.append(feature_map["dtm_elevation"])
        channel_names.append("dtm_elevation")
    if include_cfg.get("include_slope", True):
        feature_arrays.append(feature_map["slope_deg"])
        channel_names.append("slope_deg")
    if include_cfg.get("include_aspect_sin_cos", True):
        feature_arrays.extend([feature_map["aspect_sin"], feature_map["aspect_cos"]])
        channel_names.extend(["aspect_sin", "aspect_cos"])
    if include_cfg.get("include_curvatures", True):
        feature_arrays.extend([
            feature_map["general_curvature"],
            feature_map["plan_curvature"],
            feature_map["profile_curvature"],
        ])
        channel_names.extend(["general_curvature", "plan_curvature", "profile_curvature"])
    if include_cfg.get("include_tpi", True):
        feature_arrays.append(feature_map["tpi"])
        channel_names.append("tpi")
    if include_cfg.get("include_tri", True):
        feature_arrays.append(feature_map["tri"])
        channel_names.append("tri")
    if include_cfg.get("include_flow_accumulation", True):
        feature_arrays.append(feature_map["log_flow_accumulation"])
        channel_names.append("log_flow_accumulation")
    if include_cfg.get("include_twi", True):
        feature_arrays.append(feature_map["twi"])
        channel_names.append("twi")
    if include_cfg.get("include_spi", True):
        feature_arrays.append(feature_map["spi"])
        channel_names.append("spi")
    if include_cfg.get("include_sti", True):
        feature_arrays.append(feature_map["sti"])
        channel_names.append("sti")
    if include_cfg.get("include_distance_to_drainage", True):
        feature_arrays.append(feature_map["log_distance_to_drainage"])
        channel_names.append("log_distance_to_drainage")

    for band_idx in range(ortho_channels.shape[0]):
        feature_arrays.append(ortho_channels[band_idx])
        channel_names.append(f"ortho_norm_band_{band_idx + 1}")

    if shadow_mask is not None:
        feature_arrays.append(shadow_mask.astype(np.float32))
        channel_names.append("shadow_mask")

    for cls_idx in range(land_cover.shape[0]):
        feature_arrays.append(land_cover[cls_idx])
        channel_names.append(f"land_cover_class_{cls_idx}")

    stack = np.stack(feature_arrays, axis=0).astype(np.float32)
    stack[:, ~valid_mask] = 0.0
    return stack, channel_names


def apply_normalization(stack: np.ndarray, valid_mask: np.ndarray, normalization_cfg: Dict,
                        existing_stats: Optional[Dict]) -> Tuple[np.ndarray, Dict]:
    strategy = normalization_cfg.get("strategy", "zscore")
    epsilon = float(normalization_cfg.get("epsilon", 1e-6))
    stats = existing_stats
    normalized = stack.copy()

    if strategy == "zscore":
        if stats is None:
            means = []
            stds = []
            flat = normalized.reshape(normalized.shape[0], -1)
            valid_flat = valid_mask.ravel()
            for channel_idx in range(normalized.shape[0]):
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
            stats = {"strategy": "zscore", "mean": means, "std": stds, "epsilon": epsilon}
        else:
            for channel_idx in range(normalized.shape[0]):
                mean = stats["mean"][channel_idx]
                std = stats["std"][channel_idx]
                normalized[channel_idx] = (normalized[channel_idx] - mean) / (std + epsilon)

    normalized[:, ~valid_mask] = 0.0
    return normalized.astype(np.float32), stats


def save_geotiff(path: str, array: np.ndarray, reference_profile: Dict,
                 dtype: Optional[str] = None, nodata: Optional[float] = None) -> None:
    data = array
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    profile = reference_profile.copy()
    profile.update({
        "driver": "GTiff",
        "count": data.shape[0],
        "dtype": dtype or data.dtype.name,
    })
    if nodata is not None:
        profile["nodata"] = nodata
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(profile["dtype"]))


# ---------------------------------------------------------------------------
# Core preprocessing workflow
# ---------------------------------------------------------------------------

def preprocess_area(area_inputs: AreaInputs, area_dir: str, config: Dict,
                    stats: Optional[Dict], seed: int) -> Tuple[AreaArtifacts, Optional[Dict]]:
    preprocessing_cfg = config["preprocessing"]
    metadata_dir = config["project_structure"]["metadata_dir"]
    os.makedirs(metadata_dir, exist_ok=True)

    with rasterio.open(area_inputs.dtm) as src:
        dtm = src.read(1).astype(np.float32)
        profile = src.profile

    valid_mask = np.ones_like(dtm, dtype=bool)
    nodata = profile.get("nodata")
    if nodata is not None:
        valid_mask &= dtm != nodata
    valid_mask &= ~np.isnan(dtm)

    dtm_filled = fill_nodata(dtm, ~valid_mask)
    dtm_filled = simple_sink_fill(dtm_filled, preprocessing_cfg["dtm_hygiene"].get("sink_fill_kernel", 5))
    gaussian_sigma = preprocessing_cfg["dtm_hygiene"].get("gaussian_sigma", 0.0)
    if gaussian_sigma and gaussian_sigma > 0.0:
        dtm_filled = gaussian_filter(dtm_filled, sigma=gaussian_sigma)

    dem_features, dem_meta = compute_dem_features(
        dtm_filled,
        profile["transform"],
        valid_mask,
        preprocessing_cfg["hydrology"].get("drainage_threshold", 5000.0),
    )

    resampling_map = {
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
        "cubic": Resampling.cubic,
    }
    ortho_resampling = resampling_map.get(preprocessing_cfg["resampling"].get("ortho", "bilinear"), Resampling.bilinear)
    ortho_array, _ = reproject_raster(area_inputs.ortho, profile, ortho_resampling, dst_dtype=np.float32)
    ortho_norm = normalize_radiometry(ortho_array, valid_mask, preprocessing_cfg["normalization"]["epsilon"])

    shadow_mask = None
    if preprocessing_cfg["orthophoto_channels"].get("shadow_mask", True):
        shadow_mask = derive_shadow_mask(ortho_array, valid_mask)

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
        shadow_mask,
        preprocessing_cfg,
        valid_mask,
    )

    area_dir = os.path.join(config["project_structure"]["derived_data_dir"], area_inputs.name)
    os.makedirs(area_dir, exist_ok=True)

    if stats is not None and "channel_names" in stats:
        reference_names = list(stats["channel_names"])
        if channel_names != reference_names:
            name_to_idx = {name: idx for idx, name in enumerate(channel_names)}
            missing = [name for name in reference_names if name not in name_to_idx]
            if missing:
                raise ValueError(f"Channels missing in {area_inputs.name}: {missing}")
            feature_stack = feature_stack[[name_to_idx[name] for name in reference_names], :, :]
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
        with open(normalization_stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
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
    save_geotiff(mask_path, valid_mask.astype(np.uint8), profile, dtype="uint8", nodata=0)

    slope_qc_path = os.path.join(area_dir, "slope_deg.tif")
    save_geotiff(slope_qc_path, dem_features["slope_deg"], profile, dtype="float32")
    lulc_qc_path = os.path.join(area_dir, "land_cover.tif")
    save_geotiff(lulc_qc_path, land_cover_labels.astype(np.int16), profile, dtype="int16", nodata=-1)

    metadata = {
        "channel_names": channel_names,
        "channel_map": {name: idx for idx, name in enumerate(channel_names)},
        "land_cover_clusters": preprocessing_cfg["orthophoto_channels"].get("land_cover_clusters", 6),
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
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    artifacts = AreaArtifacts(
        name=area_inputs.name,
        feature_stack_path=feature_stack_path,
        mask_path=mask_path,
        metadata_path=metadata_path,
        ground_truth_path=None,
        normalization_stats_path=normalization_stats_path if stats is None else None,
    )
    return artifacts, stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    import yaml  # local import to avoid mandatory dependency for other modules
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_inputs(config: Dict, split: str) -> AreaInputs:
    import importlib

    module_name = config["inputs"]["module"]
    data_module = importlib.import_module(module_name)
    split_cfg = config["inputs"][split]
    dtm = getattr(data_module, split_cfg["dtm_attr"])
    ortho = getattr(data_module, split_cfg["ortho_attr"])
    ground = split_cfg.get("ground_truth_attr")
    ground_path = getattr(data_module, ground) if ground else None
    for path in [dtm, ortho, ground_path]:
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"{split} input missing: {path}")
    return AreaInputs(name=split, dtm=dtm, ortho=ortho, ground_truth=ground_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone preprocessing.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration.")
    parser.add_argument("--area", default="train", help="Area key in config.inputs (e.g., train/test).")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config.")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(config["project_structure"]["derived_data_dir"], exist_ok=True)
    os.makedirs(config["project_structure"]["metadata_dir"], exist_ok=True)

    seed = args.seed or config["reproducibility"]["seed"]
    area_inputs = resolve_inputs(config, args.area)

    stats = None
    stats_path = os.path.join(config["project_structure"]["metadata_dir"], "normalization_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

    artifacts, stats = preprocess_area(area_inputs, config["project_structure"]["derived_data_dir"],
                                       config, stats, seed)

    print("--- Preprocessing complete ---")
    print(f"Feature stack: {artifacts.feature_stack_path}")
    print(f"Valid mask:    {artifacts.mask_path}")
    print(f"Metadata:      {artifacts.metadata_path}")
    if artifacts.normalization_stats_path:
        print(f"Normalization stats: {artifacts.normalization_stats_path}")


if __name__ == "__main__":
    main()
