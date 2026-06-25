"""Preflight checks for external full-resolution custom LULC runs."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import rasterio
import torch

from IBGE_method.own_LULC.implementation.config import load_config
from IBGE_method.own_LULC.implementation.model import create_model


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def fail(message: str) -> None:
    raise SystemExit(f"[lulc-preflight] ERROR: {message}")


def set_overrides(args: argparse.Namespace) -> None:
    os.environ["LULC_INPUT_ORTHO"] = str(Path(args.ortho).expanduser().resolve())
    os.environ["LULC_INPUT_POLYGONS"] = str(Path(args.polygons).expanduser().resolve())
    os.environ["LULC_OUTPUT_DIR"] = str(Path(args.output_dir).expanduser().resolve())


def check_paths(args: argparse.Namespace) -> None:
    ortho = Path(args.ortho).expanduser()
    polygons = Path(args.polygons).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not ortho.exists():
        fail(f"orthophoto not found: {ortho}")
    if not polygons.exists():
        fail(f"polygon file not found: {polygons}")
    output_dir.mkdir(parents=True, exist_ok=True)


def check_rasters(args: argparse.Namespace) -> None:
    with rasterio.open(args.ortho) as src:
        if src.count < 3:
            fail(f"orthophoto must have at least 3 bands, found {src.count}")
        if src.crs is None:
            fail("orthophoto has no CRS")
        if src.transform is None:
            fail("orthophoto has no affine transform")
        if src.width <= 0 or src.height <= 0:
            fail(f"invalid orthophoto shape: {src.width}x{src.height}")
        print(
            "[lulc-preflight] orthophoto: "
            f"{src.width}x{src.height} bands={src.count} crs={src.crs} res={src.res}",
            flush=True,
        )
        if args.target_resolution:
            expected = float(args.target_resolution)
            x_res, y_res = src.res
            if abs(float(x_res) - expected) > 1e-6 or abs(float(y_res) - expected) > 1e-6:
                print(
                    "[lulc-preflight] warning: input resolution differs from target "
                    f"{expected:g}m; pipeline will resample to target.",
                    flush=True,
                )


def check_polygons(args: argparse.Namespace) -> None:
    polygons = gpd.read_file(args.polygons, engine="pyogrio")
    missing = [field for field in ("Classe", "Class_num") if field not in polygons.columns]
    if missing:
        fail(f"polygon file missing required fields: {missing}")
    if polygons.empty:
        fail("polygon file has no rows")
    class_values = sorted(int(v) for v in polygons["Class_num"].dropna().unique())
    if class_values != [1, 2, 3, 4, 5]:
        fail(f"expected polygon Class_num values [1,2,3,4,5], found {class_values}")
    print(f"[lulc-preflight] polygons: rows={len(polygons)} classes={class_values}", flush=True)


def check_disk(args: argparse.Namespace) -> None:
    usage = shutil.disk_usage(args.output_dir)
    print(
        "[lulc-preflight] output disk: "
        f"free={human_bytes(usage.free)} total={human_bytes(usage.total)} path={args.output_dir}",
        flush=True,
    )
    if usage.free < int(args.min_free_gb * 1024**3):
        fail(f"output disk has less than {args.min_free_gb:g} GiB free")


def check_cuda(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        if args.allow_cpu:
            print("[lulc-preflight] CUDA unavailable; continuing because --allow-cpu was set", flush=True)
            return
        fail("CUDA is unavailable in PyTorch")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(
        "[lulc-preflight] CUDA device: "
        f"{torch.cuda.get_device_name(0)} memory={props.total_memory / 1024**3:.2f} GiB",
        flush=True,
    )
    if props.total_memory < int(args.min_gpu_memory_gb * 1024**3):
        fail(f"GPU has less than {args.min_gpu_memory_gb:g} GiB memory")

    try:
        config = load_config(experiment_name="fullres_unetplusplus_resnet18_focal_lovasz_rgb_indices")
        model = create_model(config).to(device)
        batch_size = int(config["params"]["training"]["batch_size"])
        channels = int(config["params"]["model"]["input_channels"])
        tile_size = int(config["params"]["tile_size"])
        with torch.no_grad():
            dummy = torch.zeros((batch_size, channels, tile_size, tile_size), device=device)
            _ = model(dummy)
        del model
        torch.cuda.empty_cache()
    except (RuntimeError, AssertionError) as exc:
        if args.allow_cpu:
            print(
                "[lulc-preflight] CUDA model check failed; continuing because "
                f"--allow-cpu was set: {exc}",
                flush=True,
            )
            return
        fail(f"CUDA model check failed: {exc}")
    print(
        "[lulc-preflight] CUDA model check passed: "
        f"batch={batch_size} channels={channels} tile={tile_size}",
        flush=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ortho", required=True, help="Input RGB/RGBA orthophoto path")
    parser.add_argument(
        "--polygons",
        default="IBGE_method/own_LULC/extra_data/Classes_Uso_Solo.gpkg",
        help="Training polygon GeoPackage path",
    )
    parser.add_argument(
        "--output-dir",
        default="IBGE_method/own_LULC/outputs_fullres",
        help="External full-resolution output directory",
    )
    parser.add_argument("--target-resolution", type=float, default=0.16)
    parser.add_argument("--min-free-gb", type=float, default=80.0)
    parser.add_argument("--min-gpu-memory-gb", type=float, default=6.0)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CUDA checks to fail; intended for local dry-runs only.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    check_paths(args)
    set_overrides(args)
    check_rasters(args)
    check_polygons(args)
    check_disk(args)
    check_cuda(args)
    print("[lulc-preflight] all checks passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
