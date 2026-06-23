"""Configuration loading for the custom LULC workflow."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from IBGE_method.own_LULC import lulc_inputs


ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def load_config() -> Dict[str, Any]:
    params = deepcopy(lulc_inputs.lulc_params)
    config: Dict[str, Any] = {
        "input_polygons": resolve_path(lulc_inputs.input_polygons),
        "input_ortho": resolve_path(lulc_inputs.input_ortho),
        "class_definitions": deepcopy(lulc_inputs.class_definitions),
        "params": params,
    }
    config["output_dir"] = resolve_path(params["output_folderpath"])
    return config


def output_path(config: Dict[str, Any], filename_key: str) -> Path:
    return Path(config["output_dir"]) / str(config["params"][filename_key])
