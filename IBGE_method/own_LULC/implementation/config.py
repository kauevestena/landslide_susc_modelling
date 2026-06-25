"""Configuration loading for the custom LULC workflow."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Dict, Optional

from IBGE_method.own_LULC import lulc_inputs


ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def resolution_label(resolution: float) -> str:
    if float(resolution).is_integer():
        return f"{int(resolution)}m"
    if 0 < float(resolution) < 1:
        centimeters = float(resolution) * 100.0
        if float(centimeters).is_integer():
            return f"{int(centimeters)}cm"
    return f"{str(resolution).replace('.', 'p')}m"


def apply_output_filenames(config: Dict[str, Any], resolution: float) -> None:
    params = config["params"]
    label = resolution_label(resolution)
    params["lulc_filename"] = f"lulc_custom_{label}.tif"
    params["probabilities_filename"] = f"lulc_custom_probabilities_{label}.tif"
    params["resampled_rgb_filename"] = f"resampled_rgb_{label}.tif"
    params["training_labels_filename"] = f"training_labels_{label}.tif"
    ensemble = config.get("ensemble")
    if ensemble:
        ensemble["lulc_filename"] = f"lulc_custom_ensemble_{label}.tif"
        ensemble["probabilities_filename"] = f"lulc_custom_ensemble_probabilities_{label}.tif"
        ensemble["agreement_filename"] = f"lulc_custom_ensemble_agreement_{label}.tif"
        ensemble["confidence_filename"] = f"lulc_custom_ensemble_confidence_{label}.tif"
        ensemble["margin_filename"] = f"lulc_custom_ensemble_margin_{label}.tif"
        ensemble["entropy_filename"] = f"lulc_custom_ensemble_entropy_{label}.tif"


def apply_experiment(config: Dict[str, Any], experiment_name: Optional[str] = None) -> Dict[str, Any]:
    params = config["params"]
    name = experiment_name or config["active_experiment"]
    if not name:
        return config
    grid = config["experiment_grid"]
    if name not in grid:
        raise ValueError(f"Unknown LULC experiment: {name}")
    experiment = deepcopy(grid[name])
    resolution = float(experiment.get("resolution", params["output_resolution"]))
    params["output_resolution"] = resolution
    params["tile_size"] = int(experiment.get("tile_size", params["tile_size"]))
    params["stride"] = int(experiment.get("stride", params["stride"]))
    params["feature_set"] = experiment.get("feature_set", params["feature_set"])
    params["random_seed"] = int(experiment.get("seed", params["random_seed"]))
    apply_output_filenames(config, resolution)
    if "inference_window_size" in experiment:
        params["inference"]["window_size"] = int(experiment["inference_window_size"])
    if "inference_overlap" in experiment:
        params["inference"]["overlap"] = int(experiment["inference_overlap"])
    if "model" in experiment:
        params["model"].update(deepcopy(experiment["model"]))
    if "loss" in experiment:
        params["loss"].update(deepcopy(experiment["loss"]))
    feature_set = params["feature_set"]
    params["model"]["input_channels"] = len(config["feature_sets"][feature_set]["channels"])
    config["active_experiment"] = name
    config["active_experiment_config"] = experiment
    return config


def load_config(experiment_name: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    params = deepcopy(lulc_inputs.lulc_params)
    output_override = os.environ.get("LULC_OUTPUT_DIR")
    if output_override:
        params["output_folderpath"] = output_override
    config: Dict[str, Any] = {
        "input_polygons": resolve_path(os.environ.get("LULC_INPUT_POLYGONS", lulc_inputs.input_polygons)),
        "input_ortho": resolve_path(os.environ.get("LULC_INPUT_ORTHO", lulc_inputs.input_ortho)),
        "class_definitions": deepcopy(lulc_inputs.class_definitions),
        "feature_sets": deepcopy(lulc_inputs.feature_sets),
        "split_constraints": deepcopy(lulc_inputs.split_constraints),
        "sampler": deepcopy(lulc_inputs.sampler),
        "ensemble": deepcopy(lulc_inputs.ensemble),
        "selection_metric": lulc_inputs.selection_metric,
        "active_experiment": lulc_inputs.active_experiment,
        "experiment_grid": deepcopy(lulc_inputs.experiment_grid),
        "params": params,
    }
    config["output_dir"] = resolve_path(params["output_folderpath"])
    apply_output_filenames(config, float(params["output_resolution"]))
    apply_experiment(config, experiment_name)
    if seed is not None:
        config["params"]["random_seed"] = int(seed)
        config["active_experiment_config"]["seed"] = int(seed)
    return config


def load_config_with_experiment(
    experiment_name: str,
    experiment_config: Dict[str, Any],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    config = load_config(experiment_name=None, seed=None)
    config["experiment_grid"][experiment_name] = deepcopy(experiment_config)
    apply_experiment(config, experiment_name)
    if seed is not None:
        config["params"]["random_seed"] = int(seed)
        config["active_experiment_config"]["seed"] = int(seed)
    return config


def output_path(config: Dict[str, Any], filename_key: str) -> Path:
    return Path(config["output_dir"]) / str(config["params"][filename_key])
