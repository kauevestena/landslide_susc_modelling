"""Raster preparation for the custom LULC workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from .config import output_path


def target_profile_from_ortho(src: rasterio.io.DatasetReader, resolution: float) -> Dict[str, Any]:
    width = int(np.ceil((src.bounds.right - src.bounds.left) / resolution))
    height = int(np.ceil((src.bounds.top - src.bounds.bottom) / resolution))
    transform = from_origin(src.bounds.left, src.bounds.top, resolution, resolution)
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": src.crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }


def resample_rgb(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    params = config["params"]
    resolution = float(params["output_resolution"])
    with rasterio.open(config["input_ortho"]) as src:
        base_profile = target_profile_from_ortho(src, resolution)
        rgb = np.zeros((3, base_profile["height"], base_profile["width"]), dtype=np.uint8)
        for band_idx in range(1, 4):
            reproject(
                source=rasterio.band(src, band_idx),
                destination=rgb[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=base_profile["transform"],
                dst_crs=base_profile["crs"],
                dst_nodata=0,
                resampling=Resampling.average,
            )

        valid = np.ones((base_profile["height"], base_profile["width"]), dtype=bool)
        if bool(params["use_alpha_valid_mask"]) and src.count >= 4:
            alpha = np.zeros(valid.shape, dtype=np.uint8)
            reproject(
                source=rasterio.band(src, 4),
                destination=alpha,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=base_profile["transform"],
                dst_crs=base_profile["crs"],
                dst_nodata=0,
                resampling=Resampling.average,
            )
            valid &= alpha > 0

    profile = base_profile.copy()
    profile.update(count=3, dtype="uint8", nodata=0)
    with rasterio.open(output_path(config, "resampled_rgb_filename"), "w", **profile) as dst:
        dst.write(rgb)
        for idx, description in enumerate(("red", "green", "blue"), start=1):
            dst.set_band_description(idx, description)

    return rgb, valid, base_profile


def rasterize_labels(
    config: Dict[str, Any], polygons: gpd.GeoDataFrame, reference_profile: Dict[str, Any]
) -> np.ndarray:
    params = config["params"]
    ignore_index = int(params["ignore_index"])
    value_field = params["class_value_field"]
    labels = np.full(
        (reference_profile["height"], reference_profile["width"]),
        ignore_index,
        dtype=np.uint8,
    )
    shapes = (
        (geometry, int(class_value))
        for geometry, class_value in zip(polygons.geometry, polygons[value_field])
        if geometry is not None and not geometry.is_empty
    )
    burned = features.rasterize(
        shapes=shapes,
        out_shape=labels.shape,
        fill=ignore_index,
        transform=reference_profile["transform"],
        dtype="uint8",
        all_touched=bool(params["rasterize_all_touched"]),
    )
    labels[:, :] = burned

    profile = reference_profile.copy()
    profile.update(count=1, dtype="uint8", nodata=ignore_index)
    with rasterio.open(output_path(config, "training_labels_filename"), "w", **profile) as dst:
        dst.write(labels, 1)
        dst.set_band_description(1, "LULC training labels from polygons")
    return labels


def prepare_rasters(
    config: Dict[str, Any], polygons: gpd.GeoDataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    rgb, valid_mask, profile = resample_rgb(config)
    labels = rasterize_labels(config, polygons, profile)
    return rgb, labels, valid_mask, profile
