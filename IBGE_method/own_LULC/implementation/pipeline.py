"""Entrypoint for generating the custom IBGE LULC raster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import rasterio

from .config import load_config, output_path
from .metrics import class_distribution
from .model import infer_full_raster, train_model
from .preprocessing import prepare_rasters
from .tiles import build_tiles, limit_split_for_smoke, split_summary, split_tiles
from .validation import validate_config


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str, allow_nan=False)


def write_lulc_outputs(
    config: Dict[str, Any],
    profile: Dict[str, Any],
    lulc: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    params = config["params"]
    lulc_profile = profile.copy()
    lulc_profile.update(
        count=1,
        dtype="uint8",
        nodata=int(params["output_nodata"]),
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(output_path(config, "lulc_filename"), "w", **lulc_profile) as dst:
        dst.write(lulc.astype(np.uint8), 1)
        dst.set_band_description(1, "Custom polygon-trained LULC class")

    prob_profile = profile.copy()
    prob_profile.update(
        count=probabilities.shape[0],
        dtype="float32",
        nodata=0.0,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(output_path(config, "probabilities_filename"), "w", **prob_profile) as dst:
        dst.write(probabilities.astype(np.float32))
        class_values = sorted(int(v) for v in config["class_definitions"].keys())
        for band_idx, class_value in enumerate(class_values, start=1):
            class_name = config["class_definitions"][class_value]["name"]
            dst.set_band_description(band_idx, f"P(class={class_value}:{class_name})")


def run_pipeline(smoke: bool = False, preprocess_only: bool = False) -> Dict[str, Any]:
    config = load_config()
    if smoke:
        smoke_cfg = config["params"]["smoke_test"]
        config["params"]["training"]["epochs"] = int(smoke_cfg["epochs"])

    source_meta, loaded = validate_config(config)
    rgb, labels, valid_mask, profile = prepare_rasters(config, loaded["polygons"])
    distribution = class_distribution(labels, int(config["params"]["ignore_index"]))
    write_json(output_path(config, "class_distribution_filename"), distribution)

    records = build_tiles(config, rgb, labels, valid_mask)
    splits = split_tiles(config, records)
    if smoke:
        splits = limit_split_for_smoke(config, splits)

    metadata: Dict[str, Any] = {
        "source": source_meta,
        "class_definitions": config["class_definitions"],
        "params": config["params"],
        "target_grid": {
            "crs": str(profile["crs"]),
            "width": int(profile["width"]),
            "height": int(profile["height"]),
            "resolution": [
                float(profile["transform"].a),
                float(abs(profile["transform"].e)),
            ],
        },
        "splits": split_summary(splits),
        "smoke": bool(smoke),
    }
    if preprocess_only:
        write_json(output_path(config, "metadata_filename"), metadata)
        return metadata

    model, training_summary = train_model(config, splits)
    lulc, probabilities = infer_full_raster(config, model, rgb, valid_mask)
    write_lulc_outputs(config, profile, lulc, probabilities)

    metadata["training"] = training_summary
    metadata["outputs"] = {
        "lulc": str(output_path(config, "lulc_filename")),
        "probabilities": str(output_path(config, "probabilities_filename")),
        "model": str(output_path(config, "model_filename")),
        "metrics": str(output_path(config, "metrics_filename")),
        "class_distribution": str(output_path(config, "class_distribution_filename")),
    }
    write_json(output_path(config, "metadata_filename"), metadata)
    write_json(output_path(config, "metrics_filename"), training_summary)
    return metadata


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run the configured short smoke mode")
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Validate inputs and write resampled RGB/label rasters without training",
    )
    args = parser.parse_args(argv)
    metadata = run_pipeline(smoke=args.smoke, preprocess_only=args.preprocess_only)
    print(json.dumps({"outputs": metadata.get("outputs", {}), "smoke": args.smoke}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
