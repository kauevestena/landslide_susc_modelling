#!/usr/bin/env python3
"""Canonical operations CLI for the landslide susceptibility project."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"
DEFAULT_SPLITS = ROOT / "artifacts" / "splits" / "splits.json"
DEFAULT_MERGED_METADATA = (
    ROOT / "artifacts" / "derived" / "merged" / "merged_metadata.json"
)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def print_items(title: str, items: Sequence[str]) -> None:
    if not items:
        return
    print(title)
    for item in items:
        print(f"  - {item}")


def running_in_project_venv() -> bool:
    executable = Path(sys.executable).resolve()
    venv_python = (ROOT / ".venv" / "bin" / "python").resolve()
    return executable == venv_python


def import_pydensecrf() -> None:
    try:
        import pydensecrf.densecrf  # noqa: F401
        from pydensecrf.utils import unary_from_softmax  # noqa: F401
    except Exception as exc:  # pragma: no cover - exact failure depends on platform
        raise RuntimeError(
            "pydensecrf is required when inference.crf.enabled is true. "
            "Install dependencies with '.venv/bin/pip install -r requirements.txt'. "
            f"Import failed with {type(exc).__name__}: {exc}"
        ) from exc


def check_crf(args: argparse.Namespace) -> int:
    try:
        import_pydensecrf()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("pydensecrf import check passed")
    return 0


def configured_input_paths(config: Dict[str, Any]) -> List[Tuple[str, str, Path]]:
    module_name = config["inputs"]["module"]
    data_module = importlib.import_module(module_name)
    paths: List[Tuple[str, str, Path]] = []

    for split_name, split_cfg in config["inputs"].items():
        if split_name == "module":
            continue
        for key in ("dtm_attr", "ortho_attr", "ground_truth_attr"):
            attr = split_cfg.get(key)
            if not attr:
                continue
            paths.append((split_name, attr, Path(getattr(data_module, attr))))
    return paths


def validate_config(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    errors: List[str] = []
    warnings: List[str] = []

    if not running_in_project_venv():
        errors.append(
            f"Use the project virtualenv: expected {rel(ROOT / '.venv/bin/python')}, "
            f"got {sys.executable}"
        )

    required_sections = [
        "project_structure",
        "inputs",
        "preprocessing",
        "dataset",
        "model",
        "training",
        "inference",
        "reproducibility",
    ]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing config section: {section}")

    preprocessing = config.get("preprocessing", {})
    external_lulc = preprocessing.get("external_lulc", {})
    label_smoothing = preprocessing.get("label_smoothing", {})

    if "dynamic_world" in label_smoothing:
        errors.append(
            "preprocessing.label_smoothing.dynamic_world is misplaced; move it to "
            "preprocessing.external_lulc.dynamic_world"
        )
    if "force_download" in label_smoothing:
        errors.append(
            "preprocessing.label_smoothing.force_download is misplaced; move it to "
            "preprocessing.external_lulc.force_download"
        )
    if external_lulc.get("enabled") and external_lulc.get("source") not in {
        "worldcover",
        "dynamic_world",
        "none",
    }:
        errors.append(
            "preprocessing.external_lulc.source must be worldcover, dynamic_world, or none"
        )

    crf_enabled = (
        config.get("inference", {}).get("crf", {}).get("enabled", False)
    )
    if crf_enabled:
        try:
            import_pydensecrf()
        except RuntimeError as exc:
            errors.append(str(exc))
    else:
        warnings.append("CRF is disabled in config; pydensecrf is not required for this run")

    requirements_path = ROOT / "requirements.txt"
    if requirements_path.exists():
        requirements_text = requirements_path.read_text(encoding="utf-8")
        if "pydensecrf" not in requirements_text:
            errors.append("requirements.txt must include pydensecrf")
    else:
        errors.append("requirements.txt not found")

    model = config.get("model", {})
    if model.get("out_classes") != 3:
        warnings.append(f"Expected 3 output classes for current docs, got {model.get('out_classes')}")
    if not model.get("attention", False):
        warnings.append("model.attention is disabled")

    dataset = config.get("dataset", {})
    if not dataset.get("use_mixed_domain", False):
        warnings.append("dataset.use_mixed_domain is disabled; single-area tiling path will be used")
    if dataset.get("tile_size", 0) <= dataset.get("tile_overlap", 0):
        errors.append("dataset.tile_size must be greater than dataset.tile_overlap")

    try:
        for split_name, attr, path in configured_input_paths(config):
            if not path.exists():
                errors.append(f"Configured input does not exist ({split_name}.{attr}): {path}")
    except Exception as exc:
        errors.append(f"Unable to resolve configured inputs: {exc}")

    if args.check_artifacts:
        errors.extend(validate_artifact_contracts(config))

    print(f"Config: {rel(config_path)}")
    print_items("Warnings:", warnings)
    print_items("Errors:", errors)
    if errors:
        print("Validation failed")
        return 1
    print("Validation passed")
    return 0


def validate_artifact_contracts(config: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    structure = config["project_structure"]
    derived = ROOT / structure["derived_data_dir"]
    metadata_dir = ROOT / structure["metadata_dir"]
    splits_dir = ROOT / structure["splits_dir"]

    for area in ("train", "test"):
        area_dir = derived / area
        feature_metadata = area_dir / "feature_metadata.json"
        if not feature_metadata.exists():
            errors.append(f"Missing feature metadata: {rel(feature_metadata)}")
            continue
        with feature_metadata.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata.get("schema_version") != 2:
            errors.append(f"Stale feature metadata schema: {rel(feature_metadata)}")
        if not metadata.get("channel_names"):
            errors.append(f"Missing channel_names in {rel(feature_metadata)}")

    stats_path = metadata_dir / "normalization_stats.json"
    if not stats_path.exists():
        errors.append(f"Missing normalization stats: {rel(stats_path)}")

    summary_path = splits_dir / "dataset_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        if summary.get("schema_version") != 2:
            errors.append(f"Stale dataset summary schema: {rel(summary_path)}")
    else:
        errors.append(f"Missing dataset summary: {rel(summary_path)}")

    return errors


def parse_tile_coords(tile_name: str) -> Tuple[int, int]:
    parts = tile_name.replace(".npy", "").split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected tile name format: {tile_name}")
    return int(parts[-2]), int(parts[-1])


def min_distance_between_splits(
    split_a: Sequence[str], split_b: Sequence[str], stride: int
) -> Tuple[float, float, Tuple[str | None, str | None]]:
    if not split_a or not split_b:
        return float("inf"), float("inf"), (None, None)

    min_pixels = float("inf")
    closest = (None, None)
    coords_a = [(name, parse_tile_coords(name)) for name in split_a]
    coords_b = [(name, parse_tile_coords(name)) for name in split_b]

    for name_a, (y_a, x_a) in coords_a:
        for name_b, (y_b, x_b) in coords_b:
            distance = float(np.hypot(y_a - y_b, x_a - x_b))
            if distance < min_pixels:
                min_pixels = distance
                closest = (name_a, name_b)

    return min_pixels, min_pixels / stride, closest


def train_row_end_from_metadata(metadata_path: Path) -> int:
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    train_area = metadata.get("train_area", {})
    row_range = train_area.get("row_range")
    if row_range and len(row_range) == 2:
        return int(row_range[1])
    if "height" in train_area:
        return int(train_area["height"])
    raise ValueError(f"Unable to derive train/test split row from {metadata_path}")


def validate_spatial(args: argparse.Namespace) -> int:
    splits_path = Path(args.splits).resolve()
    metadata_path = Path(args.metadata).resolve()
    if not splits_path.exists():
        print(f"ERROR: splits file not found: {splits_path}", file=sys.stderr)
        return 1
    if not metadata_path.exists():
        print(f"ERROR: merged metadata not found: {metadata_path}", file=sys.stderr)
        return 1

    with splits_path.open("r", encoding="utf-8") as f:
        splits = json.load(f)

    train_tiles = splits.get("train", [])
    val_tiles = splits.get("val", [])
    test_tiles = splits.get("test", [])
    stride = int(args.stride)
    train_row_end = train_row_end_from_metadata(metadata_path)

    print("Spatial split validation")
    print(f"Splits: train={len(train_tiles)}, val={len(val_tiles)}, test={len(test_tiles)}")
    print(f"Train/test area row boundary: {train_row_end}")

    checks = [
        ("train-test", train_tiles, test_tiles),
        ("train-val", train_tiles, val_tiles),
        ("val-test", val_tiles, test_tiles),
    ]
    errors: List[str] = []
    warnings: List[str] = []
    for label, left, right in checks:
        if not left or not right:
            continue
        dist_px, dist_tiles, pair = min_distance_between_splits(left, right, stride)
        print(
            f"{label}: min distance {dist_px:.1f}px ({dist_tiles:.2f} tiles), "
            f"closest={pair[0]} <-> {pair[1]}"
        )
        if dist_tiles < args.min_tiles:
            message = (
                f"{label} minimum distance {dist_tiles:.2f} tiles is below {args.min_tiles}"
            )
            if args.strict_distance:
                errors.append(message)
            else:
                warnings.append(message)

    train_from_test_area = sum(
        1 for tile in train_tiles if parse_tile_coords(tile)[0] >= train_row_end
    )
    test_from_test_area = sum(
        1 for tile in test_tiles if parse_tile_coords(tile)[0] >= train_row_end
    )
    train_from_train_area = len(train_tiles) - train_from_test_area
    test_from_train_area = len(test_tiles) - test_from_test_area

    print(
        "Mixed-domain contribution: "
        f"train split train_area={train_from_train_area}, test_area={train_from_test_area}; "
        f"test split train_area={test_from_train_area}, test_area={test_from_test_area}"
    )
    if train_from_test_area == 0 or test_from_test_area == 0:
        errors.append("Test area is not represented in both train and test splits")

    print_items("Warnings:", warnings)
    print_items("Errors:", errors)
    if errors:
        print("Spatial validation failed")
        return 1
    print("Spatial validation passed")
    return 0


def preprocess(args: argparse.Namespace) -> int:
    from src.main_pipeline import (
        load_input_paths,
        prepare_directories,
        preprocess_data,
        process_area,
    )

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.area == "all":
        artifacts = preprocess_data(config, force_recreate=args.force_recreate)
        for area_name, artifact in artifacts.items():
            print(f"{area_name}: {artifact.feature_stack_path}")
        return 0

    structure = config["project_structure"]
    prepare_directories(structure)
    inputs = load_input_paths(config)
    if args.area not in inputs:
        print(f"ERROR: area '{args.area}' is not configured in inputs", file=sys.stderr)
        return 1

    stats = None
    stats_path = Path(structure["metadata_dir"]) / "normalization_stats.json"
    if args.area != "train":
        if not stats_path.exists():
            print(
                "ERROR: normalization stats are required before preprocessing non-train areas. "
                "Run 'manage.py preprocess --area train' or '--area all' first.",
                file=sys.stderr,
            )
            return 1
        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)

    area_dir = Path(structure["derived_data_dir"]) / args.area
    area_dir.mkdir(parents=True, exist_ok=True)
    Path(structure["metadata_dir"]).mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else config["reproducibility"]["seed"]
    artifact, _ = process_area(
        inputs[args.area],
        str(area_dir),
        config["preprocessing"],
        structure["metadata_dir"],
        stats,
        seed,
        args.force_recreate,
    )
    print(f"Feature stack: {artifact.feature_stack_path}")
    print(f"Valid mask: {artifact.mask_path}")
    print(f"Metadata: {artifact.metadata_path}")
    return 0


def run_pipeline(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "-m", "src.main_pipeline"]
    if args.force_recreate:
        cmd.append("--force_recreate")
    return subprocess.call(cmd, cwd=ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    crf_parser = subparsers.add_parser("check-crf", help="Assert pydensecrf imports")
    crf_parser.set_defaults(func=check_crf)

    validate_parser = subparsers.add_parser("validate", help="Validate current config")
    validate_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    validate_parser.add_argument(
        "--check-artifacts",
        action="store_true",
        help="Also validate generated artifact metadata contracts",
    )
    validate_parser.set_defaults(func=validate_config)

    spatial_parser = subparsers.add_parser(
        "validate-spatial", help="Validate spatial train/val/test separation"
    )
    spatial_parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    spatial_parser.add_argument("--metadata", default=str(DEFAULT_MERGED_METADATA))
    spatial_parser.add_argument("--stride", type=int, default=128)
    spatial_parser.add_argument("--min-tiles", type=float, default=3.0)
    spatial_parser.add_argument(
        "--strict-distance",
        action="store_true",
        help="Fail when split distances are below --min-tiles instead of warning",
    )
    spatial_parser.set_defaults(func=validate_spatial)

    preprocess_parser = subparsers.add_parser("preprocess", help="Run preprocessing")
    preprocess_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    preprocess_parser.add_argument("--area", default="all")
    preprocess_parser.add_argument("--seed", type=int, default=None)
    preprocess_parser.add_argument(
        "--force_recreate",
        "--force-recreate",
        action="store_true",
        dest="force_recreate",
    )
    preprocess_parser.set_defaults(func=preprocess)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run the main pipeline")
    pipeline_parser.add_argument(
        "--force_recreate",
        "--force-recreate",
        action="store_true",
        dest="force_recreate",
    )
    pipeline_parser.set_defaults(func=run_pipeline)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
