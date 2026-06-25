"""Input and configuration validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import geopandas as gpd
import rasterio


def validate_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    params = config["params"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    polygon_path = Path(config["input_polygons"])
    ortho_path = Path(config["input_ortho"])
    if not polygon_path.exists():
        raise FileNotFoundError(polygon_path)
    if not ortho_path.exists():
        raise FileNotFoundError(ortho_path)

    split_total = (
        float(params["train_percentage"])
        + float(params["validation_percentage"])
        + float(params["test_percentage"])
    )
    if abs(split_total - 1.0) > 1e-6:
        raise ValueError(f"LULC split percentages must sum to 1.0; got {split_total}")

    expected_classes = set(int(value) for value in config["class_definitions"].keys())
    if len(expected_classes) != int(params["model"]["output_classes"]):
        raise ValueError("class_definitions count must match model.output_classes")

    with rasterio.open(ortho_path) as src:
        if src.count < 3:
            raise ValueError(
                f"Orthophoto has {src.count} bands, but the LULC workflow requires RGB."
            )
        feature_set = params["feature_set"]
        if feature_set not in config["feature_sets"]:
            raise ValueError(f"Unknown LULC feature_set: {feature_set}")
        expected_channels = len(config["feature_sets"][feature_set]["channels"])
        if expected_channels != int(params["model"]["input_channels"]):
            raise ValueError(
                f"feature_set {feature_set} has {expected_channels} channels, "
                f"but model.input_channels={params['model']['input_channels']}"
            )
        ortho_meta = {
            "path": str(ortho_path),
            "crs": str(src.crs),
            "width": int(src.width),
            "height": int(src.height),
            "count": int(src.count),
            "dtypes": list(src.dtypes),
            "resolution": [float(src.res[0]), float(src.res[1])],
            "bounds": [float(v) for v in src.bounds],
        }

    polygons = gpd.read_file(polygon_path)
    text_field = params["class_text_field"]
    value_field = params["class_value_field"]
    missing = [field for field in (text_field, value_field) if field not in polygons.columns]
    if missing:
        raise ValueError(f"Polygon layer is missing required fields: {missing}")
    if polygons.empty:
        raise ValueError("Polygon layer has no rows.")
    if polygons.crs is None:
        raise ValueError("Polygon layer has no CRS.")
    if str(polygons.crs) != ortho_meta["crs"]:
        polygons = polygons.to_crs(ortho_meta["crs"])

    observed_classes = set(int(v) for v in polygons[value_field].dropna().unique())
    if observed_classes - expected_classes:
        raise ValueError(
            "Polygon layer contains classes absent from class_definitions: "
            f"{sorted(observed_classes - expected_classes)}"
        )

    vector_meta = {
        "path": str(polygon_path),
        "crs": str(polygons.crs),
        "rows": int(len(polygons)),
        "class_text_field": text_field,
        "class_value_field": value_field,
        "observed_classes": sorted(observed_classes),
        "bounds": [float(v) for v in polygons.total_bounds],
    }
    return {"ortho": ortho_meta, "polygons": vector_meta}, {"polygons": polygons}
