"""Entrypoint for generating the custom IBGE LULC raster."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import rasterio

from .config import apply_output_filenames, load_config, load_config_with_experiment, output_path
from .ensemble import build_ensemble
from .metrics import class_distribution
from .model import infer_full_raster, train_model
from .preprocessing import prepare_rasters
from .tiles import build_tiles, limit_split_for_smoke, split_summary, split_tiles
from .validation import validate_config


def log(message: str) -> None:
    print(f"[lulc] {message}", flush=True)


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str, allow_nan=False)
    log(f"wrote JSON: {path}")


def write_lulc_outputs(
    config: Dict[str, Any],
    profile: Dict[str, Any],
    lulc: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    params = config["params"]
    log(
        "writing LULC rasters: "
        f"{output_path(config, 'lulc_filename')} and "
        f"{output_path(config, 'probabilities_filename')}"
    )
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


OUTPUT_FILE_KEYS = (
    "lulc_filename",
    "probabilities_filename",
    "metrics_filename",
    "class_distribution_filename",
    "metadata_filename",
    "model_filename",
    "resampled_rgb_filename",
    "training_labels_filename",
)


def run_id(config: Dict[str, Any], smoke: bool) -> str:
    prefix = "smoke_" if smoke else ""
    return (
        f"{prefix}{config['active_experiment']}_"
        f"seed{config['params']['random_seed']}"
    )


def run_pipeline(
    smoke: bool = False,
    preprocess_only: bool = False,
    experiment: Optional[str] = None,
    seed: Optional[int] = None,
    run_dir: Optional[Path] = None,
    experiment_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    started = time.time()
    if experiment_override is not None:
        if experiment is None:
            raise ValueError("experiment must be set when experiment_override is provided")
        config = load_config_with_experiment(experiment, experiment_override, seed=seed)
    else:
        config = load_config(experiment_name=experiment, seed=seed)
    canonical_output_dir = Path(config["output_dir"])
    if run_dir is not None:
        config["output_dir"] = Path(run_dir)
        config["output_dir"].mkdir(parents=True, exist_ok=True)
    if smoke:
        smoke_cfg = config["params"]["smoke_test"]
        config["params"]["training"]["epochs"] = int(smoke_cfg["epochs"])

    log(
        "starting run "
        f"run_id={run_id(config, smoke)} experiment={config['active_experiment']} "
        f"seed={config['params']['random_seed']} smoke={smoke} "
        f"preprocess_only={preprocess_only}"
    )
    log(
        "settings: "
        f"resolution={config['params']['output_resolution']}m "
        f"feature_set={config['params']['feature_set']} "
        f"tile={config['params']['tile_size']} stride={config['params']['stride']} "
        f"model={config['params']['model']['architecture']}/"
        f"{config['params']['model']['encoder']} loss={config['params']['loss']['name']} "
        f"epochs={config['params']['training']['epochs']}"
    )
    log(f"output directory: {config['output_dir']}")

    log("validating inputs and config")
    source_meta, loaded = validate_config(config)
    log(
        "input summary: "
        f"ortho={source_meta['ortho']['width']}x{source_meta['ortho']['height']} "
        f"bands={source_meta['ortho']['count']} polygons={source_meta['polygons']['rows']}"
    )
    log("preprocessing rasters")
    rgb, labels, valid_mask, profile = prepare_rasters(config, loaded["polygons"])
    log(
        "target grid: "
        f"{profile['width']}x{profile['height']} crs={profile['crs']} "
        f"res={abs(profile['transform'].a):g}m"
    )
    distribution = class_distribution(labels, int(config["params"]["ignore_index"]))
    write_json(output_path(config, "class_distribution_filename"), distribution)
    log(f"class distribution: {distribution['classes']}")

    log("building tiles")
    records = build_tiles(config, rgb, labels, valid_mask)
    log(f"candidate labeled tiles: {len(records)}")
    log("creating stratified spatial split")
    splits = split_tiles(config, records)
    if smoke:
        log("limiting split for smoke mode")
        splits = limit_split_for_smoke(config, splits)
    split_meta = split_summary(splits, config)
    for split_name in ("train", "val", "test"):
        split_info = split_meta[split_name]
        log(
            f"split {split_name}: tiles={split_info['tiles']} "
            f"labeled_pixels={split_info['labeled_pixels']} "
            f"class_counts={split_info['class_counts']}"
        )

    metadata: Dict[str, Any] = {
        "source": source_meta,
        "class_definitions": config["class_definitions"],
        "params": config["params"],
        "active_experiment": config["active_experiment"],
        "active_experiment_config": config.get("active_experiment_config", {}),
        "run_id": run_id(config, smoke),
        "canonical_output_dir": str(canonical_output_dir),
        "target_grid": {
            "crs": str(profile["crs"]),
            "width": int(profile["width"]),
            "height": int(profile["height"]),
            "resolution": [
                float(profile["transform"].a),
                float(abs(profile["transform"].e)),
            ],
        },
        "splits": split_meta,
        "smoke": bool(smoke),
    }
    if preprocess_only:
        write_json(output_path(config, "metadata_filename"), metadata)
        log(f"preprocess-only run complete in {format_seconds(time.time() - started)}")
        return metadata

    log("training model")
    model, training_summary = train_model(config, splits)
    log(
        "training complete: "
        f"best_epoch={training_summary['best_epoch']} "
        f"best_score={training_summary['best_score']:.6f} "
        f"device={training_summary['device']}"
    )
    log("running full-raster inference")
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
    log(f"run complete in {format_seconds(time.time() - started)}")
    return metadata


def summarize_run(metadata: Dict[str, Any]) -> Dict[str, Any]:
    training = metadata.get("training", {})
    history = training.get("history", [])
    best_epoch = training.get("best_epoch", 0)
    best_record = None
    for record in history:
        if int(record.get("epoch", -1)) == int(best_epoch):
            best_record = record
            break
    if best_record is None and history:
        best_record = max(history, key=lambda item: item.get("score", 0.0))
    best_val_metrics = (best_record or {}).get("val_metrics", {}) or {}
    test_metrics = training.get("test", {}).get("metrics", {}) or {}
    return {
        "run_id": metadata["run_id"],
        "experiment": metadata["active_experiment"],
        "seed": int(metadata["params"]["random_seed"]),
        "resolution": float(metadata["params"]["output_resolution"]),
        "feature_set": metadata["params"]["feature_set"],
        "architecture": metadata["params"]["model"]["architecture"],
        "encoder": metadata["params"]["model"]["encoder"],
        "loss": metadata["params"]["loss"]["name"],
        "tile_size": int(metadata["params"]["tile_size"]),
        "stride": int(metadata["params"]["stride"]),
        "epochs": len(history),
        "best_epoch": int(best_epoch),
        "best_score": float(training.get("best_score", 0.0)),
        "val_macro_iou": float(best_val_metrics.get("macro_iou", 0.0)),
        "val_macro_f1": float(best_val_metrics.get("macro_f1", 0.0)),
        "val_overall_accuracy": float(best_val_metrics.get("overall_accuracy", 0.0)),
        "test_macro_iou": float(test_metrics.get("macro_iou", 0.0)),
        "test_macro_f1": float(test_metrics.get("macro_f1", 0.0)),
        "test_overall_accuracy": float(test_metrics.get("overall_accuracy", 0.0)),
        "output_dir": str(Path(metadata["outputs"]["metrics"]).parent),
    }


def write_sweep_results(output_dir: Path, results: Sequence[Dict[str, Any]]) -> None:
    write_json(output_dir / "sweep_results.json", {"runs": list(results)})
    if not results:
        return
    fieldnames = list(results[0].keys())
    with (output_dir / "sweep_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    log(f"wrote CSV: {output_dir / 'sweep_results.csv'}")


def apply_target_resolution(config: Dict[str, Any], target_resolution: Optional[float]) -> Dict[str, Any]:
    if target_resolution is None:
        return config
    config["params"]["output_resolution"] = float(target_resolution)
    apply_output_filenames(config, float(target_resolution))
    return config


def completed_run_valid(config: Dict[str, Any], run_path: Path) -> bool:
    params = config["params"]
    required = [
        params["metadata_filename"],
        params["metrics_filename"],
        params["model_filename"],
        params["lulc_filename"],
        params["probabilities_filename"],
    ]
    if any(not (run_path / name).exists() for name in required):
        return False
    try:
        with (run_path / params["metadata_filename"]).open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        with (run_path / params["metrics_filename"]).open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        if not metadata.get("training") or not metrics.get("history"):
            return False
        with rasterio.open(run_path / params["lulc_filename"]) as src:
            if src.count != 1 or src.width <= 0 or src.height <= 0:
                return False
        with rasterio.open(run_path / params["probabilities_filename"]) as src:
            if src.count != int(params["model"]["output_classes"]) or src.width <= 0 or src.height <= 0:
                return False
    except Exception as exc:
        log(f"completed run validation failed for {run_path}: {exc}")
        return False
    return True


def run_ensemble_from_results(
    results: Sequence[Dict[str, Any]],
    promote: bool = True,
    target_resolution: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    config = apply_target_resolution(load_config(), target_resolution)
    if not bool(config.get("ensemble", {}).get("enabled", False)):
        log("ensemble disabled; keeping single-model promotion")
        return None
    try:
        result = build_ensemble(config, results, promote=promote)
    except ValueError as exc:
        log(f"ensemble unavailable: {exc}")
        return None
    if promote:
        write_json(Path(config["output_dir"]) / "selected_experiment.json", result)
        log(
            "ensemble promoted: "
            f"val_macro_iou={result['metrics']['val']['macro_iou']:.6f} "
            f"test_macro_iou={result['metrics']['test']['macro_iou']:.6f}"
        )
    return result


def run_ensemble_only(promote: bool = True, target_resolution: Optional[float] = None) -> Dict[str, Any]:
    config = apply_target_resolution(load_config(), target_resolution)
    output_dir = Path(config["output_dir"])
    sweep_path = output_dir / "sweep_results.json"
    if not sweep_path.exists():
        raise FileNotFoundError(f"Missing sweep results for ensemble: {sweep_path}")
    with sweep_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    runs = list(payload.get("runs", []))
    if not runs:
        raise ValueError(f"No runs found in {sweep_path}")
    result = run_ensemble_from_results(runs, promote=promote, target_resolution=target_resolution)
    if result is None:
        raise ValueError("Ensemble is disabled in lulc_inputs.py")
    selection_path = output_dir / "sweep_selection.json"
    if selection_path.exists():
        with selection_path.open("r", encoding="utf-8") as f:
            selection = json.load(f)
    else:
        selection = {"runs": runs}
    selection["selected"] = result
    selection["ensemble"] = result
    write_json(selection_path, selection)
    return result


def promote_run(summary: Dict[str, Any]) -> None:
    log(f"promoting selected run: {summary['run_id']}")
    if "experiment_config" in summary:
        config = load_config_with_experiment(
            summary["experiment"],
            summary["experiment_config"],
            seed=int(summary["seed"]),
        )
    else:
        config = load_config(experiment_name=summary["experiment"], seed=int(summary["seed"]))
    canonical_dir = Path(config["output_dir"])
    source_dir = Path(summary["output_dir"])
    canonical_dir.mkdir(parents=True, exist_ok=True)
    for filename_key in OUTPUT_FILE_KEYS:
        name = config["params"][filename_key]
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, canonical_dir / name)
            log(f"promoted {src} -> {canonical_dir / name}")
    summary["promoted_lulc"] = str(canonical_dir / config["params"]["lulc_filename"])
    summary["promoted_probabilities"] = str(canonical_dir / config["params"]["probabilities_filename"])
    write_json(canonical_dir / "selected_experiment.json", summary)


def run_sweep(
    experiment_names: Sequence[str],
    smoke: bool = False,
    promote: bool = True,
    experiment_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    force: bool = False,
    target_resolution: Optional[float] = None,
) -> Dict[str, Any]:
    log(
        "starting sweep: "
        f"experiments={list(experiment_names)} smoke={smoke} promote={promote}"
    )
    base_config = load_config()
    root_output = Path(base_config["output_dir"])
    experiments_dir = root_output / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for experiment_name in experiment_names:
        experiment = (
            experiment_overrides.get(experiment_name)
            if experiment_overrides and experiment_name in experiment_overrides
            else base_config["experiment_grid"][experiment_name]
        )
        for seed in experiment.get("seeds", [base_config["params"]["random_seed"]]):
            log(f"sweep run start: experiment={experiment_name} seed={seed}")
            if experiment_overrides and experiment_name in experiment_overrides:
                run_config = load_config_with_experiment(
                    experiment_name, experiment_overrides[experiment_name], seed=int(seed)
                )
            else:
                run_config = load_config(experiment_name=experiment_name, seed=int(seed))
            run_path = experiments_dir / run_id(run_config, smoke)
            if not force and completed_run_valid(run_config, run_path):
                log(f"resume: using completed run {run_path}")
                with (run_path / run_config["params"]["metadata_filename"]).open(
                    "r", encoding="utf-8"
                ) as f:
                    metadata = json.load(f)
            else:
                metadata = run_pipeline(
                    smoke=smoke,
                    preprocess_only=False,
                    experiment=experiment_name,
                    seed=int(seed),
                    run_dir=run_path,
                    experiment_override=(
                        experiment_overrides.get(experiment_name)
                        if experiment_overrides and experiment_name in experiment_overrides
                        else None
                    ),
                )
            summary = summarize_run(metadata)
            summary["experiment_config"] = experiment
            results.append(summary)
            log(
                "sweep run done: "
                f"{summary['run_id']} val_macro_iou={summary['val_macro_iou']:.6f} "
                f"test_macro_iou={summary['test_macro_iou']:.6f}"
            )

    write_sweep_results(root_output, results)
    if not results:
        return {"runs": [], "selected": None}
    selected = max(results, key=lambda row: row["val_macro_iou"])
    log(f"sweep selected run: {selected['run_id']} val_macro_iou={selected['val_macro_iou']:.6f}")
    if promote:
        ensemble_result = run_ensemble_from_results(
            results, promote=True, target_resolution=target_resolution
        )
        if ensemble_result is None:
            promote_run(selected)
            selected_payload = selected
        else:
            selected_payload = ensemble_result
    else:
        selected_payload = selected
    write_json(
        root_output / "sweep_selection.json",
        {
            "selected": selected_payload,
            "single_model_selected": selected,
            "ensemble": selected_payload if selected_payload is not selected else None,
            "runs": results,
        },
    )
    return {"runs": results, "selected": selected_payload}


def grouped_selection(results: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not results:
        return None
    groups: Dict[Tuple[Any, ...], Dict[int, Dict[str, Any]]] = {}
    for result in results:
        experiment_config = result.get("experiment_config", {})
        key = (
            result["resolution"],
            result["feature_set"],
            result["architecture"],
            result["encoder"],
            result["loss"],
            result.get("tile_size", experiment_config.get("tile_size")),
            result.get("stride", experiment_config.get("stride")),
        )
        seed = int(result["seed"])
        current = groups.setdefault(key, {}).get(seed)
        if current is None or result["val_macro_iou"] >= current["val_macro_iou"]:
            groups[key][seed] = result
    aggregate = []
    for rows_by_seed in groups.values():
        rows = list(rows_by_seed.values())
        mean_val = float(np.mean([row["val_macro_iou"] for row in rows]))
        best_row = max(rows, key=lambda row: row["val_macro_iou"])
        best_row = dict(best_row)
        best_row["selection_mean_val_macro_iou"] = mean_val
        best_row["selection_seed_count"] = len(rows)
        aggregate.append((mean_val, best_row))
    return max(aggregate, key=lambda item: item[0])[1]


def clone_experiment_for_feature_set(
    name: str, source: Dict[str, Any], feature_set: str, seeds: Sequence[int]
) -> Tuple[str, Dict[str, Any]]:
    payload = dict(source["experiment_config"])
    payload["feature_set"] = feature_set
    payload["model"] = {
        "architecture": source["architecture"],
        "encoder": source["encoder"],
    }
    payload["loss"] = {"name": source["loss"]}
    payload["seeds"] = list(seeds)
    return name, payload


def run_planned_sweep(smoke: bool = False, promote: bool = True, force: bool = False) -> Dict[str, Any]:
    log(f"starting planned staged sweep smoke={smoke} promote={promote}")
    base_config = load_config()
    stage_names = default_sweep_names(base_config)
    stage_result = run_sweep(stage_names, smoke=smoke, promote=False, force=force)
    results = list(stage_result["runs"])

    stage3 = [row for row in results if row["experiment"].startswith("stage3_")]
    top_stage3 = sorted(stage3, key=lambda row: row["val_macro_iou"], reverse=True)[:2]
    log(f"top stage3 runs for rgb_indices: {[row['run_id'] for row in top_stage3]}")
    overrides: Dict[str, Dict[str, Any]] = {}
    rgb_indices_names = []
    for row in top_stage3:
        name, payload = clone_experiment_for_feature_set(
            f"stage4_rgb_indices_{row['experiment']}",
            row,
            "rgb_indices",
            [int(row["seed"])],
        )
        overrides[name] = payload
        rgb_indices_names.append(name)
    if rgb_indices_names:
        rgb_result = run_sweep(
            rgb_indices_names,
            smoke=smoke,
            promote=False,
            experiment_overrides=overrides,
            force=force,
        )
        results.extend(rgb_result["runs"])

    best = grouped_selection(results)
    if best is not None:
        log(
            "best after rgb_indices stage: "
            f"{best['run_id']} mean_val_macro_iou="
            f"{best.get('selection_mean_val_macro_iou', best['val_macro_iou']):.6f}"
        )
    robustness_overrides = {}
    robustness_names = []
    if best is not None:
        name, payload = clone_experiment_for_feature_set(
            f"stage4_seed_robustness_{best['experiment']}",
            best,
            best["feature_set"],
            [42, 1337, 2026],
        )
        robustness_overrides[name] = payload
        robustness_names.append(name)
        robust_result = run_sweep(
            robustness_names,
            smoke=smoke,
            promote=False,
            experiment_overrides=robustness_overrides,
            force=force,
        )
        results.extend(robust_result["runs"])

    selected = grouped_selection(results)
    output_dir = Path(base_config["output_dir"])
    write_sweep_results(output_dir, results)
    ensemble_result = None
    if selected is not None and promote:
        ensemble_result = run_ensemble_from_results(results, promote=True)
        if ensemble_result is None:
            promote_run(selected)
    if selected is not None:
        log(
            "planned sweep selected: "
            f"{selected['run_id']} mean_val_macro_iou="
            f"{selected.get('selection_mean_val_macro_iou', selected['val_macro_iou']):.6f}"
        )
    write_json(
        output_dir / "sweep_selection.json",
        {
            "selected": ensemble_result or selected,
            "single_model_selected": selected,
            "ensemble": ensemble_result,
            "runs": results,
        },
    )
    return {"runs": results, "selected": ensemble_result or selected}


def default_sweep_names(config: Dict[str, Any]) -> Sequence[str]:
    return [
        name
        for name in config["experiment_grid"]
        if name.startswith("stage1_") or name.startswith("stage2_") or name.startswith("stage3_")
    ]


def fullres_sweep_names(config: Dict[str, Any]) -> Sequence[str]:
    return [name for name in config["experiment_grid"] if name.startswith("fullres_")]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run the configured short smoke mode")
    parser.add_argument("--experiment", default=None, help="Experiment name from lulc_inputs.experiment_grid")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed for this run")
    parser.add_argument(
        "--sweep",
        nargs="*",
        help="Run a sweep. Omit names after --sweep to run stage1/stage2/stage3 experiments.",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Do not copy the selected sweep run to canonical outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun experiments even when complete outputs already exist.",
    )
    parser.add_argument(
        "--target-resolution",
        type=float,
        default=None,
        help="Resolution to use when rebuilding an ensemble from existing sweep results.",
    )
    parser.add_argument(
        "--fullres-sweep",
        action="store_true",
        help="Run the configured full-resolution external CUDA voter sweep.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Validate inputs and write resampled RGB/label rasters without training",
    )
    parser.add_argument(
        "--ensemble-only",
        action="store_true",
        help="Build and optionally promote the ensemble from existing sweep_results.json",
    )
    args = parser.parse_args(argv)
    if args.ensemble_only:
        result = run_ensemble_only(
            promote=not args.no_promote,
            target_resolution=args.target_resolution,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.fullres_sweep:
        base_config = load_config()
        names = fullres_sweep_names(base_config)
        result = run_sweep(
            names,
            smoke=args.smoke,
            promote=not args.no_promote,
            force=args.force,
            target_resolution=0.16,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.sweep is not None:
        base_config = load_config()
        names = args.sweep if args.sweep else default_sweep_names(base_config)
        if args.sweep:
            result = run_sweep(
                names,
                smoke=args.smoke,
                promote=not args.no_promote,
                force=args.force,
                target_resolution=args.target_resolution,
            )
        else:
            result = run_planned_sweep(
                smoke=args.smoke,
                promote=not args.no_promote,
                force=args.force,
            )
        print(json.dumps(result, indent=2))
        return 0
    metadata = run_pipeline(
        smoke=args.smoke,
        preprocess_only=args.preprocess_only,
        experiment=args.experiment,
        seed=args.seed,
    )
    print(json.dumps({"outputs": metadata.get("outputs", {}), "smoke": args.smoke}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
