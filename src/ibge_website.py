"""Build the Portuguese static report for the IBGE high-resolution product."""

from __future__ import annotations

import html
import csv
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import transform

from IBGE_method.own_LULC import lulc_inputs
from src.three_method_comparison import (
    IBGE_WEIGHTS,
    ROOT,
    SIG_PATHS,
    custom_lulc_land_use_mapping,
)


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


OUTPUTS = env_path("IBGE_WEBSITE_OUTPUTS_DIR", ROOT / "IBGE_method" / "outputs")
REPORTS = env_path("IBGE_WEBSITE_REPORTS_DIR", ROOT / "IBGE_method" / "reports")
CONFIGS = env_path("IBGE_WEBSITE_CONFIGS_DIR", ROOT / "IBGE_method" / "configs")
WEBSITE = env_path("IBGE_WEBSITE_DIR", ROOT / "IBGE_method" / "website")
ASSETS = WEBSITE / "assets"
WEBSITE_VARIANT = os.environ.get("IBGE_WEBSITE_VARIANT", "standard").strip().lower()
LULC_REPORT_MODE = os.environ.get("LULC_REPORT_MODE", "ensemble").strip().lower()
LULC_SINGLE_SELECTION_PATH = env_path(
    "IBGE_LULC_SINGLE_SELECTION_PATH",
    REPORTS / "lulc_single_model_selection.json",
)
COMPLETE_IBGE_OUTPUTS = ROOT / "IBGE_method" / "outputs"
FINAL_DTM = Path(os.environ.get("IBGE_DTM_PATH", "/home/kaue/data/landslide/dtm_final.tif")).expanduser().resolve()
DTM_SENSITIVITY_ROOT = ROOT / "IBGE_method" / "dtm_sensitivity" / "DTM_OTJC_3_1_16cm"
DTM_SENSITIVITY_COMPARISON = DTM_SENSITIVITY_ROOT / "comparison"
DTM_SENSITIVITY_OUTPUTS = DTM_SENSITIVITY_ROOT / "outputs"
ALT_DTM = Path("/home/kaue/data/landslide/DTM_OTJC_3_1_16cm.tif")
LULC_ROOT = ROOT / "IBGE_method" / "own_LULC"
LULC_FULLRES_OUTPUTS = LULC_ROOT / "outputs_fullres"
LULC_STANDARD_OUTPUTS = LULC_ROOT / "outputs"

THEME_RASTERS = {
    "dtm": FINAL_DTM,
    "lulc": OUTPUTS / "ibge_land_use_custom_lulc.tif",
    "slope": OUTPUTS / "ibge_note_slope.tif",
    "geology": OUTPUTS / "ibge_note_geology.tif",
    "pedology": OUTPUTS / "ibge_note_pedology.tif",
    "geomorphology": OUTPUTS / "ibge_note_geomorphology.tif",
    "pluviosity": OUTPUTS / "ibge_note_pluviosity.tif",
    "pluviosity_mm": OUTPUTS / "ibge_pluviosity_pma_mm.tif",
    "score": OUTPUTS / "ibge_susceptibility_score_1to10.tif",
    "class5": OUTPUTS / "ibge_class_map_5class.tif",
    "valid": OUTPUTS / "ibge_valid_mask.tif",
}

DISCRETE_COLORS = {
    "lulc": {
        0: (245, 247, 250),
        1: (188, 32, 111),
        2: (210, 180, 91),
        3: (64, 135, 194),
        4: (133, 180, 84),
        5: (41, 118, 80),
    },
    "class5": {
        1: (47, 111, 166),
        2: (77, 163, 121),
        3: (232, 196, 92),
        4: (218, 130, 72),
        5: (174, 64, 67),
        255: (232, 235, 238),
    },
    "valid": {
        0: (232, 235, 238),
        1: (53, 128, 88),
    },
}

LULC_NAMES = {
    1: "Area artificial",
    2: "Area descoberta",
    3: "Corpo d'agua",
    4: "Vegetacao campestre",
    5: "Vegetacao florestal",
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return read_json(path)


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def rel(path: Path) -> str:
    return path.relative_to(WEBSITE).as_posix()


def raster_sample(
    path: Path,
    *,
    max_size: int = 1300,
    nearest: bool = False,
    mask_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        scale = max(src.width / max_size, src.height / max_size, 1.0)
        width = max(1, int(round(src.width / scale)))
        height = max(1, int(round(src.height / scale)))
        data = src.read(
            1,
            out_shape=(height, width),
            masked=True,
            resampling=Resampling.nearest if nearest else Resampling.bilinear,
        )
    arr = np.asarray(data.astype(np.float32).filled(np.nan), dtype=np.float32)
    mask = ~np.ma.getmaskarray(data)
    if mask_path is not None:
        with rasterio.open(mask_path) as mask_src:
            mask_data = mask_src.read(
                1,
                out_shape=arr.shape,
                masked=True,
                resampling=Resampling.nearest,
            )
            mask_nodata = mask_src.nodata
        mask_arr = np.asarray(mask_data.astype(np.float32).filled(np.nan), dtype=np.float32)
        mask_valid = ~np.ma.getmaskarray(mask_data) & np.isfinite(mask_arr)
        if mask_nodata is not None:
            mask_valid &= mask_arr != float(mask_nodata)
        mask &= mask_valid
    return arr, mask


def continuous_rgb(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    cmap_name: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    valid = mask & np.isfinite(data)
    if vmin is None or vmax is None:
        if np.any(valid):
            lo, hi = np.percentile(data[valid], [2, 98])
            vmin = float(lo if vmin is None else vmin)
            vmax = float(hi if vmax is None else vmax)
        else:
            vmin, vmax = 0.0, 1.0
    if math.isclose(float(vmin), float(vmax)):
        vmax = float(vmin) + 1.0
    scaled = np.clip((data - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0)
    rgba = colormaps[cmap_name](np.nan_to_num(scaled, nan=0.0))
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb[~valid] = np.array([235, 238, 241], dtype=np.uint8)
    return rgb


def discrete_rgb(data: np.ndarray, mask: np.ndarray, palette: Mapping[int, Tuple[int, int, int]]) -> np.ndarray:
    rgb = np.full((*data.shape, 3), 235, dtype=np.uint8)
    rounded = np.rint(np.nan_to_num(data, nan=-999999.0)).astype(np.int32)
    for value, color in palette.items():
        rgb[(rounded == int(value)) & mask] = np.array(color, dtype=np.uint8)
    rgb[~mask] = np.array([235, 238, 241], dtype=np.uint8)
    return rgb


def save_jpeg(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, "JPEG", quality=88, optimize=True)


def save_png(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(path, "PNG", optimize=True)


def copy_asset(src: Path, filename: str) -> str:
    src = Path(src).expanduser().resolve()
    out = ASSETS / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)
    return rel(out)


def require_site_input(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} não encontrado: {path}")
    return path


def make_thumbnails() -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    specs = {
        "dtm": ("thumb_dtm.jpg", "gray", None, None, False),
        "slope": ("thumb_slope.jpg", "magma", 0.0, 10.0, False),
        "geology": ("thumb_geology.jpg", "viridis", 0.0, 10.0, True),
        "pedology": ("thumb_pedology.jpg", "viridis", 0.0, 10.0, True),
        "geomorphology": ("thumb_geomorphology.jpg", "viridis", 0.0, 10.0, True),
        "pluviosity": ("thumb_pluviosity.jpg", "viridis", 0.0, 10.0, False),
        "pluviosity_mm": ("thumb_pluviosity_mm.jpg", "Blues", None, None, False),
        "score": ("thumb_score.jpg", "viridis", 0.0, 10.0, False),
    }
    for key, (filename, cmap, vmin, vmax, nearest) in specs.items():
        data, mask = raster_sample(THEME_RASTERS[key], nearest=nearest)
        out = ASSETS / filename
        save_jpeg(out, continuous_rgb(data, mask, cmap_name=cmap, vmin=vmin, vmax=vmax))
        outputs[key] = rel(out)

    for key, filename in [("lulc", "thumb_lulc.jpg"), ("class5", "thumb_class5.jpg"), ("valid", "thumb_valid.jpg")]:
        data, mask = raster_sample(
            THEME_RASTERS[key],
            nearest=True,
            mask_path=FINAL_DTM if key == "lulc" else None,
        )
        out = ASSETS / filename
        save_jpeg(out, discrete_rgb(data, mask, DISCRETE_COLORS[key]))
        outputs[key] = rel(out)

    outputs["overview"] = outputs["score"]
    return outputs


def make_webviewer_asset() -> Dict[str, Any]:
    data, mask = raster_sample(THEME_RASTERS["score"], max_size=1800, nearest=False)
    valid = mask & np.isfinite(data) & (data != -9999)
    rgb = continuous_rgb(data, valid, cmap_name="viridis", vmin=0.0, vmax=10.0)
    alpha = np.where(valid, 215, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    image_path = ASSETS / "webviewer_score.png"
    save_png(image_path, rgba)

    with rasterio.open(THEME_RASTERS["score"]) as src:
        bounds = src.bounds
        xs = [bounds.left, bounds.right, bounds.right, bounds.left]
        ys = [bounds.top, bounds.top, bounds.bottom, bounds.bottom]
        lon, lat = transform(src.crs, "EPSG:4326", xs, ys)
    coords = [[float(x), float(y)] for x, y in zip(lon, lat)]
    meta = {
        "image": rel(image_path),
        "coordinates": coords,
        "bounds": [
            [min(c[0] for c in coords), min(c[1] for c in coords)],
            [max(c[0] for c in coords), max(c[1] for c in coords)],
        ],
        "center": [
            sum(c[0] for c in coords) / 4.0,
            sum(c[1] for c in coords) / 4.0,
        ],
    }
    (ASSETS / "webviewer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def make_dtm_sensitivity_assets() -> Dict[str, Any]:
    stats_path = require_site_input(
        DTM_SENSITIVITY_COMPARISON / "difference_stats.json",
        "estatísticas da comparação com DTM tendencioso",
    )
    payload = read_json(stats_path)
    if "dtm_difference_stats" not in payload or "dtm_difference_raster" not in payload:
        raise RuntimeError(
            "Os artefatos de diferença entre DTMs ainda não existem. Rode "
            "`.venv/bin/python -m src.ibge_dtm_sensitivity --comparison-only` antes de gerar o website."
        )

    final_dtm = require_site_input(Path(payload["final_dtm"]), "DTM final")
    alt_dtm = require_site_input(Path(payload["alternative_dtm"]), "DTM tendencioso")
    final_score = require_site_input(Path(payload["final_score"]), "score IBGE final")
    alt_score = require_site_input(Path(payload["alternative_score"]), "score IBGE alternativo")
    score_diff = require_site_input(Path(payload["difference_raster"]), "raster de diferença do score")
    dtm_diff = require_site_input(Path(payload["dtm_difference_raster"]), "raster de diferença entre DTMs")
    histogram_png = require_site_input(Path(payload["histogram_png"]), "histograma da diferença do score")
    comparison_report = require_site_input(
        DTM_SENSITIVITY_COMPARISON / "comparison_report.md",
        "relatório da comparação com DTM tendencioso",
    )
    class_transition_csv = require_site_input(Path(payload["class5_transition_csv"]), "matriz de transição de classes")
    histogram_csv = require_site_input(Path(payload["histogram_csv"]), "histograma CSV da diferença do score")
    dtm_histogram_csv = require_site_input(
        Path(payload["dtm_difference_histogram_csv"]), "histograma CSV da diferença entre DTMs"
    )
    dtm_stats_csv = require_site_input(Path(payload["dtm_difference_stats_csv"]), "estatísticas CSV da diferença entre DTMs")

    final_dtm_data, final_dtm_mask = raster_sample(final_dtm, max_size=1700, nearest=False)
    alt_dtm_data, alt_dtm_mask = raster_sample(alt_dtm, max_size=1700, nearest=False)
    combined_valid = final_dtm_mask & alt_dtm_mask & np.isfinite(final_dtm_data) & np.isfinite(alt_dtm_data)
    if np.any(combined_valid):
        combined = np.concatenate([final_dtm_data[combined_valid], alt_dtm_data[combined_valid]])
        dtm_vmin, dtm_vmax = np.percentile(combined, [2, 98])
        dtm_vmin = float(dtm_vmin)
        dtm_vmax = float(dtm_vmax)
    else:
        dtm_vmin, dtm_vmax = 0.0, 1.0

    assets: Dict[str, str] = {}
    out = ASSETS / "sensitivity_dtm_final_gray.jpg"
    save_jpeg(out, continuous_rgb(final_dtm_data, final_dtm_mask, cmap_name="gray", vmin=dtm_vmin, vmax=dtm_vmax))
    assets["dtm_final"] = rel(out)
    out = ASSETS / "sensitivity_dtm_biased_gray.jpg"
    save_jpeg(out, continuous_rgb(alt_dtm_data, alt_dtm_mask, cmap_name="gray", vmin=dtm_vmin, vmax=dtm_vmax))
    assets["dtm_biased"] = rel(out)

    final_score_data, final_score_mask = raster_sample(final_score, max_size=1700, nearest=False)
    alt_score_data, alt_score_mask = raster_sample(alt_score, max_size=1700, nearest=False)
    out = ASSETS / "sensitivity_score_final_viridis.jpg"
    save_jpeg(out, continuous_rgb(final_score_data, final_score_mask, cmap_name="viridis", vmin=0.0, vmax=10.0))
    assets["score_final"] = rel(out)
    out = ASSETS / "sensitivity_score_biased_viridis.jpg"
    save_jpeg(out, continuous_rgb(alt_score_data, alt_score_mask, cmap_name="viridis", vmin=0.0, vmax=10.0))
    assets["score_biased"] = rel(out)

    dtm_diff_data, dtm_diff_mask = raster_sample(dtm_diff, max_size=1700, nearest=False)
    dtm_diff_stats = payload["dtm_difference_stats"]
    dtm_limit = max(abs(float(dtm_diff_stats["min"])), abs(float(dtm_diff_stats["max"])))
    if math.isclose(dtm_limit, 0.0):
        dtm_limit = 1.0
    out = ASSETS / "sensitivity_dtm_difference_gray.jpg"
    save_jpeg(out, continuous_rgb(dtm_diff_data, dtm_diff_mask, cmap_name="gray", vmin=-dtm_limit, vmax=dtm_limit))
    assets["dtm_difference"] = rel(out)

    score_diff_data, score_diff_mask = raster_sample(score_diff, max_size=1700, nearest=False)
    score_diff_stats = payload["score_difference_stats"]
    out = ASSETS / "sensitivity_score_difference_viridis.jpg"
    save_jpeg(
        out,
        continuous_rgb(
            score_diff_data,
            score_diff_mask,
            cmap_name="viridis",
            vmin=float(score_diff_stats["min"]),
            vmax=float(score_diff_stats["max"]),
        ),
    )
    assets["score_difference"] = rel(out)

    assets["histogram"] = copy_asset(histogram_png, "sensitivity_difference_histogram.png")
    assets["stats_json"] = copy_asset(stats_path, "sensitivity_difference_stats.json")
    assets["report_md"] = copy_asset(comparison_report, "sensitivity_comparison_report.md")
    assets["class_transition_csv"] = copy_asset(class_transition_csv, "sensitivity_class5_transition_matrix.csv")
    assets["histogram_csv"] = copy_asset(histogram_csv, "sensitivity_difference_histogram.csv")
    assets["dtm_histogram_csv"] = copy_asset(dtm_histogram_csv, "sensitivity_dtm_difference_histogram.csv")
    assets["dtm_stats_csv"] = copy_asset(dtm_stats_csv, "sensitivity_dtm_difference_stats.csv")

    raster_paths = {
        "final_score": final_score,
        "alternative_score": alt_score,
        "score_difference": score_diff,
        "dtm_difference": dtm_diff,
    }
    raster_metadata = {
        key: {
            "path": str(path),
            "size_mb": path.stat().st_size / (1024 * 1024),
            "included_in_website": False,
        }
        for key, path in raster_paths.items()
    }

    result = {
        "payload": payload,
        "assets": assets,
        "display": {
            "dtm_vmin": dtm_vmin,
            "dtm_vmax": dtm_vmax,
            "dtm_difference_symmetric_limit": dtm_limit,
            "score_difference_vmin": float(score_diff_stats["min"]),
            "score_difference_vmax": float(score_diff_stats["max"]),
        },
        "raster_artifacts": raster_metadata,
        "asset_policy": {
            "web_assets_are_downsampled": True,
            "full_geotiffs_are_not_copied_to_website": True,
            "github_file_size_limit_mb": 100,
        },
    }
    (ASSETS / "dtm_sensitivity_assets.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def same_grid(left: rasterio.io.DatasetReader, right: rasterio.io.DatasetReader) -> bool:
    return (
        left.crs == right.crs
        and left.transform == right.transform
        and left.width == right.width
        and left.height == right.height
    )


def finite_score_mask(score: np.ndarray, valid: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    mask = (valid == 1) & np.isfinite(score)
    if nodata is not None and not math.isnan(float(nodata)):
        mask &= score != float(nodata)
    return mask


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return rel(path)


def make_lite_full_comparison_assets() -> Optional[Dict[str, Any]]:
    if WEBSITE_VARIANT != "lite":
        return None
    full_score_path = COMPLETE_IBGE_OUTPUTS / "ibge_susceptibility_score_1to10.tif"
    lite_score_path = OUTPUTS / "ibge_susceptibility_score_1to10.tif"
    full_valid_path = COMPLETE_IBGE_OUTPUTS / "ibge_valid_mask.tif"
    lite_valid_path = OUTPUTS / "ibge_valid_mask.tif"
    full_class_path = COMPLETE_IBGE_OUTPUTS / "ibge_class_map_5class.tif"
    lite_class_path = OUTPUTS / "ibge_class_map_5class.tif"
    required = [
        full_score_path,
        lite_score_path,
        full_valid_path,
        lite_valid_path,
        full_class_path,
        lite_class_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Não foi possível comparar lite vs completo; arquivos ausentes: {missing}")

    diffs: list[np.ndarray] = []
    transition_counts: Counter[tuple[int, int]] = Counter()
    accounting = Counter()
    with rasterio.open(full_score_path) as full_score, rasterio.open(lite_score_path) as lite_score, rasterio.open(
        full_valid_path
    ) as full_valid, rasterio.open(lite_valid_path) as lite_valid, rasterio.open(
        full_class_path
    ) as full_class, rasterio.open(lite_class_path) as lite_class:
        for label, dataset in [
            ("score lite", lite_score),
            ("valid completo", full_valid),
            ("valid lite", lite_valid),
            ("classe completo", full_class),
            ("classe lite", lite_class),
        ]:
            if not same_grid(full_score, dataset):
                raise RuntimeError(f"Grid incompatível para comparação lite vs completo: {label}")
        pixel_area = abs(float(full_score.transform.a * full_score.transform.e))
        for _, window in full_score.block_windows(1):
            full_arr = full_score.read(1, window=window)
            lite_arr = lite_score.read(1, window=window)
            full_valid_arr = full_valid.read(1, window=window)
            lite_valid_arr = lite_valid.read(1, window=window)
            full_mask = finite_score_mask(full_arr, full_valid_arr, full_score.nodata)
            lite_mask = finite_score_mask(lite_arr, lite_valid_arr, lite_score.nodata)
            common = full_mask & lite_mask
            accounting["full_valid_pixels"] += int(np.count_nonzero(full_mask))
            accounting["lite_valid_pixels"] += int(np.count_nonzero(lite_mask))
            accounting["common_valid_pixels"] += int(np.count_nonzero(common))
            accounting["full_only_valid_pixels"] += int(np.count_nonzero(full_mask & ~lite_mask))
            accounting["lite_only_valid_pixels"] += int(np.count_nonzero(lite_mask & ~full_mask))
            if not np.any(common):
                continue
            diff = (lite_arr[common].astype(np.float32) - full_arr[common].astype(np.float32)).astype(np.float32)
            diffs.append(diff)

            full_cls = full_class.read(1, window=window)
            lite_cls = lite_class.read(1, window=window)
            valid_class = common & (full_cls >= 1) & (full_cls <= 5) & (lite_cls >= 1) & (lite_cls <= 5)
            if np.any(valid_class):
                pairs = np.stack([full_cls[valid_class].astype(np.int16), lite_cls[valid_class].astype(np.int16)], axis=1)
                unique, counts = np.unique(pairs, axis=0, return_counts=True)
                for pair, count in zip(unique, counts):
                    transition_counts[(int(pair[0]), int(pair[1]))] += int(count)

    if not diffs:
        raise RuntimeError("Não há pixels válidos comuns para comparar o produto lite com o completo.")
    values = np.concatenate(diffs).astype(np.float32)
    abs_values = np.abs(values)
    percentiles = np.percentile(values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    stats = {
        "count": int(values.size),
        "area_m2": float(values.size * pixel_area),
        "mean": float(np.mean(values)),
        "median": float(percentiles[4]),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p10": float(percentiles[2]),
        "p25": float(percentiles[3]),
        "p75": float(percentiles[5]),
        "p90": float(percentiles[6]),
        "p95": float(percentiles[7]),
        "p99": float(percentiles[8]),
        "mae": float(np.mean(abs_values)),
        "median_absolute_difference": float(np.median(abs_values)),
        "rmse": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
        "positive_pixels": int(np.count_nonzero(values > 0)),
        "negative_pixels": int(np.count_nonzero(values < 0)),
        "zero_pixels_exact": int(np.count_nonzero(values == 0)),
        "positive_fraction": float(np.count_nonzero(values > 0) / values.size),
        "negative_fraction": float(np.count_nonzero(values < 0) / values.size),
        "within_abs_threshold_fraction": {
            f"{threshold:.2f}": float(np.count_nonzero(abs_values <= threshold) / values.size)
            for threshold in (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
        },
    }

    transition_total = sum(transition_counts.values())
    unchanged = sum(transition_counts[(value, value)] for value in range(1, 6))
    upward = sum(
        count for (full_value, lite_value), count in transition_counts.items() if lite_value > full_value
    )
    downward = sum(
        count for (full_value, lite_value), count in transition_counts.items() if lite_value < full_value
    )
    transition_rows = []
    for full_value in range(1, 6):
        for lite_value in range(1, 6):
            count = int(transition_counts[(full_value, lite_value)])
            transition_rows.append(
                {
                    "classe_completo": full_value,
                    "classe_lite": lite_value,
                    "pixels": count,
                    "area_m2": float(count * pixel_area),
                    "fraction": float(count / transition_total) if transition_total else 0.0,
                }
            )
    transition_summary = {
        "evaluated_pixels": int(transition_total),
        "unchanged_pixels": int(unchanged),
        "upward_pixels": int(upward),
        "downward_pixels": int(downward),
        "changed_pixels": int(transition_total - unchanged),
        "unchanged_fraction": float(unchanged / transition_total) if transition_total else 0.0,
        "upward_fraction": float(upward / transition_total) if transition_total else 0.0,
        "downward_fraction": float(downward / transition_total) if transition_total else 0.0,
        "changed_fraction": float((transition_total - unchanged) / transition_total) if transition_total else 0.0,
    }

    hist_counts, hist_edges = np.histogram(values, bins=100)
    hist_rows = [
        {
            "bin_left": float(hist_edges[idx]),
            "bin_right": float(hist_edges[idx + 1]),
            "count": int(count),
            "fraction": float(count / values.size),
        }
        for idx, count in enumerate(hist_counts)
    ]
    histogram_csv = write_rows_csv(
        ASSETS / "lite_full_score_difference_histogram.csv",
        hist_rows,
        ["bin_left", "bin_right", "count", "fraction"],
    )
    transition_csv = write_rows_csv(
        ASSETS / "lite_full_class5_transition_matrix.csv",
        transition_rows,
        ["classe_completo", "classe_lite", "pixels", "area_m2", "fraction"],
    )

    histogram_png_path = ASSETS / "lite_full_score_difference_histogram.png"
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.hist(values, bins=100, color="#2E6F95", edgecolor="white", linewidth=0.25)
    ax.axvline(0.0, color="#222222", linewidth=1.2, label="Sem diferença")
    ax.axvline(stats["mean"], color="#FDE725", linewidth=1.5, label=f"Média {stats['mean']:.4f}")
    ax.axvline(stats["median"], color="#35B779", linewidth=1.5, label=f"Mediana {stats['median']:.4f}")
    ax.set_title("Diferença do score IBGE 1-10: lite - completo")
    ax.set_xlabel("Diferença no score 1-10")
    ax.set_ylabel("Pixels válidos comuns")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(histogram_png_path)
    plt.close(fig)

    payload = {
        "comparison": "lite_minus_complete",
        "complete_score": str(full_score_path),
        "lite_score": str(lite_score_path),
        "complete_class5": str(full_class_path),
        "lite_class5": str(lite_class_path),
        "score_difference_stats": stats,
        "valid_pixel_accounting": {
            key: int(value) for key, value in accounting.items()
        }
        | {
            "common_valid_area_m2": float(accounting["common_valid_pixels"] * pixel_area),
            "full_only_valid_area_m2": float(accounting["full_only_valid_pixels"] * pixel_area),
            "lite_only_valid_area_m2": float(accounting["lite_only_valid_pixels"] * pixel_area),
        },
        "class5_transition_summary": transition_summary,
        "assets": {
            "histogram_png": rel(histogram_png_path),
            "histogram_csv": histogram_csv,
            "transition_csv": transition_csv,
        },
    }
    stats_json_path = ASSETS / "lite_full_comparison_stats.json"
    payload["assets"]["stats_json"] = rel(stats_json_path)
    stats_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return payload


def raster_stats(
    path: Path,
    *,
    discrete: bool = False,
    include_nodata: bool = False,
    mask_path: Optional[Path] = None,
) -> Dict[str, Any]:
    with rasterio.open(path) as src:
        mask_src = rasterio.open(mask_path) if mask_path is not None else None
        pixel_area = abs(float(src.transform.a * src.transform.e))
        counts: Counter[int] = Counter()
        total = 0
        total_sum = 0.0
        total_sumsq = 0.0
        min_value = math.inf
        max_value = -math.inf
        try:
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window)
                valid = np.isfinite(arr)
                if src.nodata is not None and not include_nodata:
                    valid &= arr != src.nodata
                if mask_src is not None:
                    mask_arr = mask_src.read(1, window=window)
                    mask_valid = np.isfinite(mask_arr)
                    if mask_src.nodata is not None:
                        mask_valid &= mask_arr != mask_src.nodata
                    valid &= mask_valid
                vals = arr[valid]
                if vals.size == 0:
                    continue
                if discrete:
                    unique, freq = np.unique(vals.astype(np.int64), return_counts=True)
                    for value, count in zip(unique, freq):
                        counts[int(value)] += int(count)
                else:
                    vals64 = vals.astype(np.float64)
                    total += int(vals64.size)
                    total_sum += float(vals64.sum())
                    total_sumsq += float(np.square(vals64).sum())
                    min_value = min(min_value, float(vals64.min()))
                    max_value = max(max_value, float(vals64.max()))
        finally:
            if mask_src is not None:
                mask_src.close()
        if discrete:
            total_count = sum(counts.values())
            return {
                "values": {
                    str(value): {
                        "pixels": count,
                        "area_m2": count * pixel_area,
                        "fraction": count / total_count if total_count else 0.0,
                    }
                    for value, count in sorted(counts.items())
                },
                "total_pixels": total_count,
            }
        mean = total_sum / total if total else 0.0
        variance = max(total_sumsq / total - mean * mean, 0.0) if total else 0.0
        return {
            "min": min_value if total else None,
            "max": max_value if total else None,
            "mean": mean,
            "std": math.sqrt(variance),
            "pixels": total,
        }


def mapping_table(rows: Iterable[Sequence[Any]], headers: Sequence[str]) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def fmt_number(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def fmt_int(value: float) -> str:
    return f"{int(round(float(value))):,}".replace(",", ".")


def fmt_percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%".replace(".", ",")


def fmt_area(value: float) -> str:
    return f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_optional_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "pendente"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "pendente"
    if not math.isfinite(numeric):
        return "pendente"
    return fmt_number(numeric, digits)


def fmt_optional_int(value: Any) -> str:
    if value is None:
        return "pendente"
    try:
        return fmt_int(float(value))
    except (TypeError, ValueError, OverflowError):
        return "pendente"


def pct_from_fraction(value: Any) -> str:
    if value is None:
        return "pendente"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "pendente"
    if not math.isfinite(numeric):
        return "pendente"
    return fmt_percent(numeric)


def best_available_lulc_output_dir() -> Path:
    for candidate in (LULC_FULLRES_OUTPUTS, LULC_STANDARD_OUTPUTS):
        if (
            (candidate / "selected_experiment.json").exists()
            or (candidate / "ensemble_results.json").exists()
            or (candidate / "sweep_results.json").exists()
        ):
            return candidate
    return LULC_FULLRES_OUTPUTS


def metric_from_run(row: Mapping[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def run_sort_value(row: Mapping[str, Any]) -> Tuple[float, float, str]:
    return (
        float(row.get("val_macro_iou", -1.0) or -1.0),
        float(row.get("test_macro_iou", -1.0) or -1.0),
        str(row.get("run_id", "")),
    )


def eligible_fullres_single_runs(runs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    eligible = []
    for row in runs:
        run_id = str(row.get("run_id", ""))
        if run_id.startswith("smoke_"):
            continue
        try:
            resolution = float(row.get("resolution", 0.0))
        except (TypeError, ValueError):
            continue
        if abs(resolution - 0.16) > 1e-9:
            continue
        eligible.append(row)
    return eligible


def select_best_single_lulc_run(runs: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    eligible = eligible_fullres_single_runs(runs)
    if not eligible:
        return None
    return max(eligible, key=run_sort_value)


def normalized_lulc_run(row: Mapping[str, Any]) -> Dict[str, Any]:
    config = row.get("experiment_config", {}) or {}
    model = config.get("model", {}) if isinstance(config, Mapping) else {}
    loss = config.get("loss", {}) if isinstance(config, Mapping) else {}
    return {
        "run_id": row.get("run_id"),
        "experiment": row.get("experiment"),
        "architecture": row.get("architecture") or model.get("architecture"),
        "encoder": row.get("encoder") or model.get("encoder"),
        "feature_set": row.get("feature_set") or config.get("feature_set"),
        "loss": row.get("loss") or loss.get("name"),
        "resolution": row.get("resolution") or config.get("resolution"),
        "seed": row.get("seed"),
        "epochs": row.get("epochs"),
        "best_epoch": row.get("best_epoch"),
        "tile_size": row.get("tile_size") or config.get("tile_size"),
        "stride": row.get("stride") or config.get("stride"),
        "val_macro_iou": metric_from_run(row, "val_macro_iou"),
        "val_macro_f1": metric_from_run(row, "val_macro_f1"),
        "val_overall_accuracy": metric_from_run(row, "val_overall_accuracy"),
        "test_macro_iou": metric_from_run(row, "test_macro_iou"),
        "test_macro_f1": metric_from_run(row, "test_macro_f1"),
        "test_overall_accuracy": metric_from_run(row, "test_overall_accuracy"),
        "output_dir": row.get("output_dir"),
    }


def architecture_label(row: Mapping[str, Any]) -> str:
    architecture = str(row.get("architecture") or "pendente")
    encoder = str(row.get("encoder") or "pendente")
    names = {
        "unet": "U-Net",
        "unetplusplus": "U-Net++",
        "deeplabv3plus": "DeepLabV3+",
        "fpn": "FPN",
    }
    return f"{names.get(architecture.lower(), architecture)} / {encoder}"


def load_lulc_method_report() -> Dict[str, Any]:
    output_dir = best_available_lulc_output_dir()
    selected = read_json_if_exists(output_dir / "selected_experiment.json")
    ensemble = read_json_if_exists(output_dir / "ensemble_results.json")
    sweep = read_json_if_exists(output_dir / "sweep_results.json") or {"runs": []}
    if LULC_REPORT_MODE != "best_single" and selected and selected.get("selection_type") == "ensemble":
        ensemble = selected

    runs = list(sweep.get("runs", []))
    candidate_runs = [normalized_lulc_run(row) for row in eligible_fullres_single_runs(runs)]
    candidate_runs.sort(key=run_sort_value, reverse=True)
    for index, row in enumerate(candidate_runs, start=1):
        row["model_order"] = index
        row["display_name"] = architecture_label(row)

    single_selection = read_json_if_exists(LULC_SINGLE_SELECTION_PATH)
    best_single_raw: Optional[Mapping[str, Any]]
    if single_selection and single_selection.get("selected_run"):
        best_single_raw = single_selection["selected_run"]
    else:
        best_single_raw = select_best_single_lulc_run(runs)

    if LULC_REPORT_MODE == "best_single":
        voters = [normalized_lulc_run(best_single_raw)] if best_single_raw else []
        ensemble_payload: Dict[str, Any] = {}
        selected_payload: Dict[str, Any] = dict(best_single_raw or {})
    else:
        voters_raw: Sequence[Mapping[str, Any]] = []
        if ensemble and ensemble.get("voters"):
            voters_raw = list(ensemble["voters"])
        elif ensemble and ensemble.get("voter_run_ids"):
            ids = {str(run_id) for run_id in ensemble["voter_run_ids"]}
            voters_raw = [row for row in runs if str(row.get("run_id")) in ids]
        else:
            voters_raw = [row for row in runs if float(row.get("resolution", 0.0) or 0.0) == 0.16][:5]
        voters = [normalized_lulc_run(row) for row in voters_raw]
        ensemble_payload = ensemble or {}
        selected_payload = selected or {}

    for index, row in enumerate(voters, start=1):
        row["model_order"] = index
        row["display_name"] = architecture_label(row)

    fullres_experiments = {
        name: payload
        for name, payload in lulc_inputs.experiment_grid.items()
        if name.startswith("fullres_")
    }
    if LULC_REPORT_MODE == "best_single":
        metrics_available = bool(voters) and all(
            voters[0].get(key) is not None
            for key in ("val_macro_iou", "test_macro_iou", "val_overall_accuracy", "test_overall_accuracy")
        )
    else:
        metrics_available = bool(ensemble_payload and ensemble_payload.get("metrics")) and all(
            row.get("val_macro_iou") is not None and row.get("test_macro_iou") is not None
            for row in voters
        )

    class_rows = [
        {
            "class_id": class_id,
            "name": payload["name"],
            "ibge_land_use_note": payload["ibge_land_use_note"],
        }
        for class_id, payload in sorted(lulc_inputs.class_definitions.items())
    ]
    feature_sets = {
        name: {"channels": list(payload.get("channels", []))}
        for name, payload in lulc_inputs.feature_sets.items()
    }
    report: Dict[str, Any] = {
        "source_dir": str(output_dir),
        "selected_experiment_path": str(output_dir / "selected_experiment.json"),
        "ensemble_results_path": str(output_dir / "ensemble_results.json"),
        "sweep_results_path": str(output_dir / "sweep_results.json"),
        "report_mode": LULC_REPORT_MODE,
        "single_selection_path": str(LULC_SINGLE_SELECTION_PATH),
        "metrics_available": metrics_available,
        "metrics_note": (
            "Métricas do modelo individual lidas dos artefatos locais em outputs_fullres/."
            if LULC_REPORT_MODE == "best_single" and metrics_available
            else "Métricas lidas dos artefatos locais em outputs_fullres/."
            if metrics_available
            else "Métricas pendentes: copie os JSONs de treinamento da máquina externa para preencher esta seção."
        ),
        "class_definitions": class_rows,
        "feature_sets": feature_sets,
        "split_constraints": dict(lulc_inputs.split_constraints),
        "sampler": dict(lulc_inputs.sampler),
        "ensemble_config": dict(lulc_inputs.ensemble),
        "params": {
            "polygons": lulc_inputs.input_polygons,
            "orthophoto": lulc_inputs.input_ortho,
            "class_text_field": lulc_inputs.lulc_params["class_text_field"],
            "class_value_field": lulc_inputs.lulc_params["class_value_field"],
            "output_nodata": lulc_inputs.lulc_params["output_nodata"],
            "ignore_index": lulc_inputs.lulc_params["ignore_index"],
            "train_percentage": lulc_inputs.lulc_params["train_percentage"],
            "validation_percentage": lulc_inputs.lulc_params["validation_percentage"],
            "test_percentage": lulc_inputs.lulc_params["test_percentage"],
            "training": dict(lulc_inputs.lulc_params["training"]),
            "augmentation": dict(lulc_inputs.lulc_params["augmentation"]),
            "inference": dict(lulc_inputs.lulc_params["inference"]),
            "postprocessing": dict(lulc_inputs.lulc_params["postprocessing"]),
            "loss": dict(lulc_inputs.lulc_params["loss"]),
        },
        "fullres_experiments": fullres_experiments,
        "candidate_runs": candidate_runs,
        "selected_single": voters[0] if LULC_REPORT_MODE == "best_single" and voters else {},
        "single_selection": single_selection or {},
        "voters": voters,
        "ensemble": ensemble_payload,
        "selected": selected_payload,
        "sweep_run_count": len(runs),
    }
    report["markdown"] = complete_lulc_method_markdown(report)
    return report


def lulc_model_narrative(row: Mapping[str, Any]) -> str:
    architecture = str(row.get("architecture") or "").lower()
    encoder = str(row.get("encoder") or "").lower()
    feature_set = str(row.get("feature_set") or "").lower()
    loss = str(row.get("loss") or "").lower()
    architecture_notes = {
        "unetplusplus": (
            "U-Net++ usa conexões de salto aninhadas entre encoder e decoder. A intenção é reduzir a lacuna "
            "semântica entre detalhes rasos da imagem e representações profundas, o que tende a ajudar em bordas "
            "finas entre telhados, solo exposto, vegetação e pequenos corpos d'água."
        ),
        "deeplabv3plus": (
            "DeepLabV3+ combina encoder convolucional com pirâmide espacial atrous. Ele foi incluído porque observa "
            "o contexto em múltiplas escalas sem perder completamente a resolução, uma propriedade útil quando a "
            "mesma classe aparece em manchas pequenas e grandes."
        ),
        "unet": (
            "U-Net é a arquitetura de referência para segmentação supervisionada com poucos rótulos. O encoder "
            f"`{encoder}` fornece as feições profundas e o decoder reconstrói a máscara na resolução do tile."
        ),
        "fpn": (
            "FPN agrega mapas de feições em diferentes níveis de escala. Ele funciona como um contraponto aos "
            "decoders U-Net porque enfatiza uma pirâmide explícita de feições, favorecendo a estabilidade em objetos "
            "de tamanhos variados."
        ),
    }
    encoder_notes = {
        "resnet18": (
            "O encoder ResNet-18 foi escolhido como opção leve e robusta, com boa relação entre capacidade e custo "
            "computacional. Em full-res, essa escolha é importante porque cada época processa muitos tiles."
        ),
        "resnet34": (
            "O encoder ResNet-34 aumenta a profundidade em relação ao ResNet-18. Ele foi usado para testar se mais "
            "capacidade convolucional melhora classes visualmente parecidas, como solo exposto e área artificial."
        ),
        "mit_b0": (
            "O encoder MiT-B0 introduz uma alternativa leve baseada em atenção/transformer. Ele não é apenas mais um "
            "ResNet: sua função no conjunto é trazer um viés arquitetural diferente, potencialmente mais sensível a "
            "relações espaciais amplas."
        ),
    }
    feature_notes = {
        "rgb": (
            "O feature set RGB usa exclusivamente os três primeiros canais da ortofoto. Ele testa quanto o problema "
            "pode ser resolvido apenas pela aparência direta dos pixels."
        ),
        "rgb_indices": (
            "O feature set RGB+índices adiciona HSV, brilho, excesso de verde e textura local. Isso torna explícitas "
            "pistas que o modelo poderia aprender sozinho, mas que ajudam quando há poucos polígonos rotulados."
        ),
    }
    loss_notes = {
        "focal_lovasz": (
            "A perda focal reduz a influência de pixels fáceis e coloca mais peso nos erros difíceis; Lovasz aproxima "
            "diretamente uma otimização orientada a IoU. A combinação é adequada quando a métrica de seleção é macro IoU."
        ),
        "focal_dice": (
            "Focal+dice combina foco em exemplos difíceis com sobreposição espacial. Ela é útil para classes pequenas "
            "porque a componente dice não deixa a otimização ser dominada apenas pela frequência dos pixels."
        ),
        "weighted_ce_lovasz": (
            "Cross-entropy ponderada preserva uma interpretação probabilística por pixel e compensa frequências de "
            "classe; Lovasz adiciona pressão direta sobre IoU. Essa combinação é um teste forte para classes raras."
        ),
        "weighted_ce_dice": (
            "Cross-entropy ponderada+dice foi a linha de base. Ela combina estabilidade de treino com uma componente "
            "de sobreposição espacial."
        ),
    }
    return " ".join(
        part
        for part in (
            architecture_notes.get(architecture),
            encoder_notes.get(encoder),
            feature_notes.get(feature_set),
            loss_notes.get(loss),
        )
        if part
    )


def complete_lulc_single_method_markdown(report: Mapping[str, Any]) -> str:
    params = report["params"]
    training = params["training"]
    augmentation = params["augmentation"]
    inference = params["inference"]
    selected = report.get("selected_single", {}) or {}
    candidate_runs = list(report.get("candidate_runs", []))
    class_lines = [
        f"- Classe {row['class_id']}: `{row['name']}`, nota IBGE USOVEG {fmt_optional_number(row['ibge_land_use_note'], 1)}."
        for row in report["class_definitions"]
    ]
    candidate_lines = []
    for idx, row in enumerate(candidate_runs, start=1):
        marker = "selecionado" if row.get("run_id") == selected.get("run_id") else "candidato"
        candidate_lines.append(
            f"- {idx}. `{row.get('run_id')}` ({marker}): {architecture_label(row)}, "
            f"features `{row.get('feature_set')}`, loss `{row.get('loss')}`, "
            f"val macro IoU {fmt_optional_number(row.get('val_macro_iou'))}, "
            f"test macro IoU {fmt_optional_number(row.get('test_macro_iou'))}."
        )
    if not candidate_lines:
        candidate_lines = ["- Métricas dos candidatos ainda não estão disponíveis nesta cópia local."]

    selected_name = architecture_label(selected) if selected else "pendente"
    selected_narrative = lulc_model_narrative(selected) if selected else ""
    lines = [
        "# Descrição completa do método LULC Lite",
        "",
        "Esta versão lite do relatório IBGE usa uma única rede LULC individual. Ela não usa média de probabilidades, voto majoritário, concordância entre modelos ou qualquer outro componente de ensemble. O objetivo é documentar uma alternativa mais simples e mais leve: selecionar o melhor modelo individual full-resolution e usar diretamente o seu raster LULC como USOVEG no IBGE adaptado.",
        "",
        "A página principal do projeto continua documentando o produto com ensemble. Esta página lite é paralela: seus outputs IBGE, thumbnails, webviewer, tabelas e textos são derivados de `outputs_lite/`, `reports_lite/` e `configs_lite/`. Isso evita misturar resultados do ensemble com resultados do modelo único.",
        "",
        "## Critério de seleção",
        "",
        "O critério de seleção é maior macro IoU de validação entre runs full-res, excluindo smoke runs e excluindo o próprio ensemble. Em caso de empate, a ordenação usa macro IoU de teste e depois `run_id`. O teste permanece como confirmação final; ele não é o critério principal porque isso contaminaria o holdout de teste com decisão de modelo.",
        "",
        f"Modelo selecionado: `{selected.get('run_id', 'pendente')}`.",
        f"Arquitetura selecionada: {selected_name}.",
        f"Feature set: `{selected.get('feature_set', 'pendente')}`.",
        f"Função de perda: `{selected.get('loss', 'pendente')}`.",
        f"Tile/stride: {selected.get('tile_size', 'pendente')} / {selected.get('stride', 'pendente')}.",
        f"Seed: {selected.get('seed', 'pendente')}.",
        f"Melhor época: {selected.get('best_epoch', 'pendente')}.",
        f"Validação: macro IoU {fmt_optional_number(selected.get('val_macro_iou'))}, macro F1 {fmt_optional_number(selected.get('val_macro_f1'))}, acurácia global {fmt_optional_number(selected.get('val_overall_accuracy'))}.",
        f"Teste final: macro IoU {fmt_optional_number(selected.get('test_macro_iou'))}, macro F1 {fmt_optional_number(selected.get('test_macro_f1'))}, acurácia global {fmt_optional_number(selected.get('test_overall_accuracy'))}.",
        "",
        "## Comparação dos candidatos",
        "",
        "Os cinco candidatos full-res foram treinados com resolução de 16 cm, tiles de 192 pixels e stride de 96 pixels. A tabela HTML abaixo apresenta as mesmas métricas em formato tabular; a lista aqui resume a justificativa de seleção.",
        "",
        *candidate_lines,
        "",
        "## Entradas e classes",
        "",
        f"- Polígonos de treinamento: `{params['polygons']}`.",
        f"- Ortofoto RGB/RGBA: `{params['orthophoto']}`.",
        f"- Campo textual: `{params['class_text_field']}`.",
        f"- Campo numérico: `{params['class_value_field']}`.",
        f"- Pixels ignorados no treinamento: `{params['ignore_index']}`.",
        f"- Nodata no raster LULC final: `{params['output_nodata']}`.",
        "",
        "O treinamento usa classes internas 0 a 4, mas o raster final usa os códigos originais 1 a 5. O valor 255 é reservado para pixels ignorados durante o treinamento; o valor 0 é nodata de saída. Essa separação evita confundir ausência de dado com uma classe semântica real.",
        "",
        *class_lines,
        "",
        "Para a álgebra IBGE, a conversão é direta: área artificial recebe nota 10, área descoberta nota 5, corpo d'água nota 0 e exclusão da máscara válida, vegetação campestre nota 2 e vegetação florestal nota 1.",
        "",
        "## Leitura técnica do modelo selecionado",
        "",
        selected_narrative or "Descrição técnica pendente porque a configuração do modelo selecionado não foi encontrada nos artefatos locais.",
        "",
        "O modelo selecionado combina U-Net com encoder ResNet-34, feature set RGB e perda `weighted_ce_lovasz`. Essa configuração foi a melhor por validação nos artefatos disponíveis. A escolha por validação preserva a função do teste como holdout final.",
        "",
        "A U-Net reconstrói a máscara de segmentação por um decoder que recebe conexões de salto do encoder. Essas conexões preservam detalhes espaciais de bordas e objetos pequenos, relevantes em ortofoto de 16 cm. O ResNet-34 adiciona mais profundidade que o ResNet-18, aumentando capacidade para distinguir padrões visualmente próximos, como área artificial, solo descoberto e vegetação baixa.",
        "",
        "A perda `weighted_ce_lovasz` combina cross-entropy ponderada por classe com Lovasz. A cross-entropy mantém estabilidade por pixel e compensa classes raras; Lovasz adiciona pressão direta sobre IoU. Essa combinação é coerente com o critério de seleção por macro IoU, pois o objetivo não é apenas acertar muitos pixels, mas equilibrar desempenho entre as cinco classes.",
        "",
        "## Treinamento e avaliação",
        "",
        f"- Otimizador: `{training['optimizer']}`.",
        f"- Scheduler: `{training['scheduler']}`.",
        f"- Learning rate: `{training['learning_rate']}`.",
        f"- Weight decay: `{training['weight_decay']}`.",
        f"- Batch size: `{training['batch_size']}`.",
        f"- Épocas configuradas: `{training['epochs']}`.",
        f"- Early stopping patience: `{training['early_stopping_patience']}`.",
        f"- Augmentation: flips {augmentation['flip_probability']}, rotate90 `{augmentation['rotate90']}`, brilho {augmentation['brightness']}, contraste {augmentation['contrast']}.",
        "",
        "A divisão treino/validação/teste usa blocos espaciais estratificados por classe. Essa estratégia reduz vazamento espacial e evita que classes pequenas desapareçam dos splits de validação e teste. A macro IoU é a métrica principal porque dá peso equivalente às cinco classes, mesmo quando a área de cada uma é muito diferente.",
        "",
        "## Inferência e uso no IBGE Lite",
        "",
        f"- Janela de inferência: `{inference['window_size']}` pixels.",
        f"- Sobreposição: `{inference['overlap']}` pixels.",
        f"- Batch de inferência: `{inference['batch_size']}`.",
        "",
        "O raster LULC individual selecionado é reamostrado para o grid do DTM por vizinho mais próximo, preservando classes categóricas. Em seguida, cada classe é convertida para a nota USOVEG e entra na álgebra IBGE com peso 0,10. Todos os demais temas permanecem iguais ao produto principal: DTM final, declividade derivada do drone, geologia, pedologia, geomorfologia e pluviosidade.",
        "",
        "## O que não existe nesta versão lite",
        "",
        "Esta versão não calcula agreement, confidence, margin ou entropy de ensemble. Esses diagnósticos dependem de múltiplos modelos e não fazem sentido para um único votante. A incerteza aqui deve ser inferida por inspeção visual, pelas métricas do holdout e pela comparação com o produto principal com ensemble.",
        "",
        "## Observação sobre artefatos",
        "",
        report["metrics_note"],
        "",
    ]
    return "\n".join(lines)


def complete_lulc_method_markdown(report: Mapping[str, Any]) -> str:
    if report.get("report_mode") == "best_single":
        return complete_lulc_single_method_markdown(report)

    params = report["params"]
    training = params["training"]
    augmentation = params["augmentation"]
    inference = params["inference"]
    ensemble = report.get("ensemble", {}) or {}
    ensemble_metrics = ensemble.get("metrics", {}) if isinstance(ensemble, Mapping) else {}
    agreement = ensemble.get("agreement_summary", {}) if isinstance(ensemble, Mapping) else {}
    voters = list(report.get("voters", []))
    class_lines = [
        f"- Classe {row['class_id']}: `{row['name']}`, nota IBGE USOVEG {fmt_optional_number(row['ibge_land_use_note'], 1)}."
        for row in report["class_definitions"]
    ]
    feature_lines = [
        f"- `{name}`: {', '.join(payload['channels'])}."
        for name, payload in report["feature_sets"].items()
    ]
    voter_lines = []
    for row in voters:
        narrative = lulc_model_narrative(row)
        voter_lines.extend(
            [
                f"- Modelo {row['model_order']}: {row['display_name']}.",
                f"- Identificador: `{row.get('run_id') or 'pendente'}`.",
                f"- Configuração: resolução {fmt_optional_number(row.get('resolution'), 2)} m; feature set `{row.get('feature_set') or 'pendente'}`; perda `{row.get('loss') or 'pendente'}`; tile {row.get('tile_size') or 'pendente'}; stride {row.get('stride') or 'pendente'}; seed {row.get('seed') or 'pendente'}.",
                f"- Treinamento: {row.get('epochs') or 'pendente'} épocas planejadas/executadas; melhor época {row.get('best_epoch') or 'pendente'}.",
                f"- Validação: macro IoU {fmt_optional_number(row.get('val_macro_iou'))}, macro F1 {fmt_optional_number(row.get('val_macro_f1'))}, acurácia global {fmt_optional_number(row.get('val_overall_accuracy'))}.",
                f"- Teste final: macro IoU {fmt_optional_number(row.get('test_macro_iou'))}, macro F1 {fmt_optional_number(row.get('test_macro_f1'))}, acurácia global {fmt_optional_number(row.get('test_overall_accuracy'))}.",
                f"- Leitura técnica: {narrative or 'descrição técnica pendente para esta configuração.'}",
                "",
            ]
        )
    if not voter_lines:
        voter_lines = [
            "- Os cinco modelos full-res ainda não têm métricas locais disponíveis nesta cópia do repositório.",
            "- Quando os arquivos `sweep_results.json`, `ensemble_results.json` e `selected_experiment.json` forem copiados da máquina de treinamento, esta aba passa a ser preenchida automaticamente.",
        ]

    val = ensemble_metrics.get("val", {}) if isinstance(ensemble_metrics, Mapping) else {}
    test = ensemble_metrics.get("test", {}) if isinstance(ensemble_metrics, Mapping) else {}
    lines = [
        "# Descrição completa do método LULC",
        "",
        "Esta aba documenta o método de geração da camada LULC usada como tema USOVEG no produto IBGE adaptado. O objetivo do LULC foi substituir uma base regional ou nacional de uso e cobertura da terra por um produto local treinado com polígonos interpretados sobre ortofoto de drone.",
        "",
        "A implementação está versionada no repositório, principalmente em `IBGE_method/own_LULC/lulc_inputs.py` e nos módulos de `IBGE_method/own_LULC/implementation/`. O arquivo `lulc_inputs.py` é a superfície única de hiperparâmetros: entradas, classes, resolução, divisão espacial, modelos, perdas, treino, inferência e ensemble ficam declarados ali. Isso é intencional: o relatório deve explicar uma configuração que pode ser reexecutada, e não uma sequência informal de decisões manuais.",
        "",
        "Este método LULC tem uma função específica dentro do produto final. Ele não estima suscetibilidade diretamente. Ele classifica cobertura e uso da terra em cinco classes locais; depois, essas classes são convertidas para a nota USOVEG da álgebra IBGE. Portanto, o LULC influencia o score final apenas pelo peso de USOVEG, mas influencia também a máscara válida porque corpos d'água são excluídos da suscetibilidade.",
        "",
        "## Leitura recomendada da aba",
        "",
        "A primeira parte descreve o dado de entrada, a codificação das classes e o desenho de avaliação. A segunda parte descreve as cinco redes individuais, porque cada uma foi incluída e como ler suas métricas. A terceira parte descreve o ensemble, que é o produto de produção usado pelo IBGE adaptado. As tabelas no final reproduzem os números dos arquivos JSON disponíveis localmente; se os artefatos da máquina externa forem substituídos depois, o site passa a refletir os novos números na próxima geração.",
        "",
        "## Entradas e codificação de classes",
        "",
        f"- Polígonos de treinamento: `{params['polygons']}`.",
        f"- Ortofoto RGB/RGBA: `{params['orthophoto']}`.",
        f"- Campo textual: `{params['class_text_field']}`.",
        f"- Campo numérico: `{params['class_value_field']}`.",
        f"- Pixels ignorados no treinamento: `{params['ignore_index']}`.",
        f"- Nodata no raster LULC final: `{params['output_nodata']}`.",
        "",
        "As classes usadas no raster final são os códigos originais 1 a 5. Internamente, durante o treinamento, elas são convertidas para índices 0 a 4 porque a função de perda multiclasse espera classes contíguas. Na escrita do GeoTIFF final, a codificação volta para 1 a 5. Esse detalhe é importante para auditoria: as métricas internas por classe normalmente aparecem como 0, 1, 2, 3 e 4, enquanto o GeoTIFF final e a álgebra IBGE usam 1, 2, 3, 4 e 5.",
        "",
        "O valor 255 é usado apenas para pixels ignorados no treinamento. Isso inclui pixels fora dos polígonos rotulados, pixels inválidos da ortofoto e regiões onde não se quer calcular perda. Esse valor nunca deve aparecer como classe LULC final. O valor 0 é reservado para nodata de saída, isto é, pixels onde o modelo não deve produzir uma classe válida.",
        "",
        *class_lines,
        "",
        "A interpretação semântica das classes foi mantida curta no raster para evitar ambiguidades: `artif` representa áreas artificiais, `descob` representa solo descoberto ou superfícies expostas não vegetadas, `corpo_agua` representa água, `veg_campestre` representa vegetação baixa/campestre e `veg_florestal` representa vegetação arbórea/florestal. Para o IBGE adaptado, essas classes não entram como nomes; entram como notas USOVEG: artificial recebe nota alta, solo descoberto recebe nota intermediária, vegetação recebe notas baixas e água é excluída.",
        "",
        "## Conjuntos de variáveis",
        "",
        "Foram previstos dois conjuntos de atributos derivados apenas da ortofoto, sem insumos externos adicionais:",
        "",
        *feature_lines,
        "",
        "O conjunto `rgb` preserva o problema na forma mais direta possível: o modelo vê vermelho, verde e azul e precisa aprender as separações a partir da aparência. Ele é simples, barato e reduz o risco de introduzir transformações artificiais que distorçam os dados.",
        "",
        "O conjunto `rgb_indices` amplia o RGB com HSV, brilho, excesso de verde e textura local. Ele foi incluído para ajudar a separar vegetação, solo exposto, água e superfícies artificiais quando a assinatura espectral RGB pura é ambígua. O excesso de verde favorece a separação de vegetação; HSV explicita matiz e saturação; brilho ajuda em sombras e superfícies claras; textura local ajuda a diferenciar telhados, copas, solo exposto e áreas homogêneas de água.",
        "",
        "A quarta banda da ortofoto, quando existe, não é usada como classe ou feição espectral principal. Ela pode atuar como máscara de validade, evitando que regiões sem imagem útil entrem no treinamento ou na inferência. A decisão de usar apenas os três primeiros canais como RGB reduz dependência de um canal alfa que pode variar entre exportações de ortofoto.",
        "",
        "## Divisão espacial e desenho de avaliação",
        "",
        f"- Estratégia de split: `{report['split_constraints']['strategy']}`.",
        f"- Percentuais por classe: treino {fmt_percent(params['train_percentage'])}, validação {fmt_percent(params['validation_percentage'])}, teste {fmt_percent(params['test_percentage'])}.",
        f"- Exigência de todas as classes em cada split: `{report['split_constraints']['require_all_classes_per_split']}`.",
        f"- Tentativas máximas de seeds para achar split viável: `{report['split_constraints']['max_seed_attempts']}`.",
        "",
        "A divisão não é uma amostragem aleatória simples de pixels. Ela usa blocos espaciais para reduzir vazamento espacial entre treino, validação e teste. Se pixels vizinhos quase idênticos fossem sorteados aleatoriamente para splits diferentes, a validação poderia medir memorização local da ortofoto, e não generalização para blocos realmente não vistos.",
        "",
        "Além disso, as metas de proporção são aplicadas por classe. Isso significa que a distribuição desejada de treino, validação e teste é verificada dentro de cada classe, não apenas no total de pixels. Essa escolha foi feita depois de observar que classes minoritárias podiam ficar com suporte muito baixo em validação ou teste. Sem suporte mínimo por classe, a macro IoU fica instável: uma classe pequena pode dominar a interpretação da métrica por acaso, ou simplesmente não ser avaliada de maneira útil.",
        "",
        "A validação é usada para seleção de modelos e escolha do ensemble. O teste é mantido como confirmação final. Essa separação é deliberada: olhar o teste para escolher modelo causaria viés de seleção. No relatório, por isso, a macro IoU de validação explica por que um modelo foi considerado forte, enquanto a macro IoU de teste indica se essa escolha se confirmou em blocos mantidos fora da seleção.",
        "",
        "A métrica principal é macro IoU em cinco classes. Macro IoU calcula IoU por classe e depois tira a média simples, dando o mesmo peso a classes grandes e pequenas. Isso é mais exigente do que acurácia global, porque um modelo não pode compensar desempenho ruim em `descob` ou `corpo_agua` acertando muitos pixels de vegetação.",
        "",
        "## Amostragem balanceada e aumento de dados",
        "",
        f"- Sampler: `{report['sampler']['strategy']}`.",
        f"- Potência para classes raras: `{report['sampler']['rare_class_power']}`.",
        f"- Peso máximo por tile: `{report['sampler']['max_weight']}`.",
        f"- Otimizador: `{training['optimizer']}`.",
        f"- Scheduler: `{training['scheduler']}`.",
        f"- Taxa de aprendizado: `{training['learning_rate']}`.",
        f"- Decaimento de peso: `{training['weight_decay']}`.",
        f"- Batch size: `{training['batch_size']}`.",
        f"- Épocas: `{training['epochs']}`.",
        f"- Early stopping patience: `{training['early_stopping_patience']}`.",
        f"- Aumentos: flips com probabilidade {augmentation['flip_probability']}, rotações de 90 graus `{augmentation['rotate90']}`, brilho {augmentation['brightness']}, contraste {augmentation['contrast']}.",
        "",
        "O sampler pondera tiles com maior presença de classes raras. Isso aumenta a frequência com que água, solo exposto ou outras classes de menor área aparecem nos batches de treinamento, sem alterar a avaliação. A validação e o teste continuam feitos nos blocos espaciais mantidos fora do treino.",
        "",
        "A ponderação por classe e o sampler balanceado atacam problemas diferentes. A ponderação na função de perda aumenta o custo de errar classes raras nos pixels rotulados. O sampler aumenta a chance de essas classes raras aparecerem nos batches. Em conjunto, eles evitam que o treinamento seja dominado pela classe espacialmente mais abundante.",
        "",
        "Os aumentos de dados foram mantidos conservadores. Flips e rotações de 90 graus são adequados porque a classe de uso e cobertura não depende da orientação absoluta da imagem. Pequenas mudanças de brilho e contraste simulam diferenças de iluminação e exposição. Borramento e ruído existem como hiperparâmetros, mas ficam desabilitados por padrão para não degradar bordas finas em uma ortofoto de 16 cm.",
        "",
        "O otimizador AdamW foi usado porque separa atualização de pesos e decaimento, sendo estável em redes de segmentação modernas. O scheduler cosseno reduz a taxa de aprendizado progressivamente, permitindo passos maiores no início e ajustes mais finos no final das 100 épocas.",
        "",
        "## Funções de perda testadas",
        "",
        "Foram usadas combinações de perdas complementares. Cross-entropy ponderada é uma perda por pixel estável e probabilística; focal loss reduz o peso de exemplos fáceis e enfatiza erros difíceis; dice loss favorece sobreposição espacial; Lovasz aproxima diretamente uma otimização ligada à IoU. Como o objetivo de seleção é macro IoU, Lovasz é especialmente útil nas configurações em que a prioridade é melhorar interseção sobre união, e não apenas acurácia por pixel.",
        "",
        "A combinação `weighted_ce_dice` foi usada como linha de base. `focal_dice` foi testada para tornar o treino mais sensível a exemplos difíceis sem abandonar a ideia de sobreposição. `weighted_ce_lovasz` e `focal_lovasz` foram incluídas para pressionar diretamente a métrica de IoU, o que faz sentido quando classes pequenas precisam aparecer bem na avaliação macro.",
        "",
        "## Inferência full-resolution",
        "",
        f"- Janela de inferência: `{inference['window_size']}` pixels.",
        f"- Sobreposição: `{inference['overlap']}` pixels.",
        f"- Batch de inferência: `{inference['batch_size']}`.",
        f"- CUDA habilitado quando disponível: `{inference['use_cuda']}`.",
        "",
        "A inferência percorre a ortofoto em janelas sobrepostas. Para cada pixel, as probabilidades acumuladas nas janelas que o cobrem são médias antes da decisão final. A sobreposição reduz artefatos de borda, porque pixels próximos às margens de uma janela também são vistos em outra janela, mais centralizada.",
        "",
        "O resultado individual de cada modelo inclui um raster de classes e um raster multibanda de probabilidades, com cinco bandas, uma por classe. O raster de classes é útil para inspeção direta. O raster de probabilidades é mais importante para o ensemble, porque preserva incerteza: um modelo que prevê classe 4 com probabilidade 0,51 carrega informação diferente de um modelo que prevê classe 4 com probabilidade 0,99.",
        "",
        "## Por que cinco modelos diferentes",
        "",
        "O ensemble não foi desenhado para juntar cinco cópias quase idênticas do mesmo treinamento. Ele combina arquiteturas, encoders, losses e feature sets diferentes. A motivação é reduzir dependência de um único viés de modelo. Quando modelos diferentes convergem para a mesma classe em um pixel, a confiança interpretativa aumenta. Quando divergem, os rasters de concordância, margem e entropia indicam onde o mapa merece inspeção humana.",
        "",
        "A diversidade arquitetural cobre quatro famílias: U-Net++, DeepLabV3+, U-Net e FPN. A diversidade de encoder cobre ResNet-18, ResNet-34 e MiT-B0. A diversidade de perda cobre focal, dice, Lovasz e cross-entropy ponderada. A diversidade de features aparece principalmente no modelo U-Net++ com `rgb_indices`, enquanto os demais mantêm RGB para testar robustez sem engenharia adicional de canais.",
        "",
        "## Os cinco modelos full-res",
        "",
        *voter_lines,
        "## Ensemble",
        "",
        f"- Tipo de seleção: `{ensemble.get('selection_type', 'pendente')}`.",
        f"- Estratégia: `{ensemble.get('strategy', report['ensemble_config'].get('strategy', 'pendente'))}`.",
        f"- Política de votantes: `{ensemble.get('voter_policy', report['ensemble_config'].get('voter_policy', 'pendente'))}`.",
        f"- Número de votantes: `{len(voters) if voters else 'pendente'}`.",
        f"- Validação do ensemble: macro IoU {fmt_optional_number(val.get('macro_iou'))}, macro F1 {fmt_optional_number(val.get('macro_f1'))}, acurácia global {fmt_optional_number(val.get('overall_accuracy'))}, pixels avaliados {fmt_optional_int(val.get('evaluated_pixels'))}.",
        f"- Teste final do ensemble: macro IoU {fmt_optional_number(test.get('macro_iou'))}, macro F1 {fmt_optional_number(test.get('macro_f1'))}, acurácia global {fmt_optional_number(test.get('overall_accuracy'))}, pixels avaliados {fmt_optional_int(test.get('evaluated_pixels'))}.",
        f"- Concordância média entre votos duros: {fmt_optional_number(agreement.get('mean_agreement'))}.",
        f"- Confiança média da probabilidade média: {fmt_optional_number(agreement.get('mean_confidence'))}.",
        f"- Margem média entre as duas maiores probabilidades: {fmt_optional_number(agreement.get('mean_margin'))}.",
        f"- Entropia média normalizada: {fmt_optional_number(agreement.get('mean_entropy'))}.",
        "",
        "O ensemble usa média de probabilidades. Esse método é preferido ao voto majoritário simples porque preserva a confiança de cada modelo. Em cada pixel, as cinco distribuições de probabilidade são somadas e normalizadas; a classe final é a classe com maior probabilidade média. A concordância por voto duro é gravada como diagnóstico separado, junto com confiança, margem e entropia.",
        "",
        "Na prática, a média de probabilidades responde a duas perguntas ao mesmo tempo. A primeira é qual classe recebeu maior suporte coletivo dos modelos. A segunda é quão concentrado foi esse suporte. Se todos os modelos favorecem a mesma classe com alta probabilidade, a confiança e a margem tendem a ser altas e a entropia tende a ser baixa. Se os modelos se dividem entre duas classes, a classe final ainda é definida, mas a margem diminui e a entropia aumenta.",
        "",
        "A concordância por voto duro é uma métrica diferente da confiança probabilística. Ela mede a fração de modelos cuja classe mais provável coincide com a classe final do ensemble. Uma concordância baixa pode ocorrer mesmo quando a confiança média é alta, especialmente em pixels onde um modelo diverge sistematicamente dos demais ou onde probabilidades muito fortes de alguns modelos superam votos mais fracos de outros. Por isso, concordância, confiança, margem e entropia devem ser lidas em conjunto, não isoladamente.",
        "",
        "A margem é a diferença entre a maior e a segunda maior probabilidade média. Ela é uma das métricas mais úteis para revisão visual: margens baixas indicam zonas de fronteira, mistura espectral ou ambiguidade semântica. A entropia normalizada mede dispersão geral entre as cinco classes. Entropia alta indica que o ensemble não concentrou probabilidade em uma única classe; nesses locais, a classificação final deve ser tratada como menos estável.",
        "",
        "## Interpretação das métricas",
        "",
        "A acurácia global mostra a fração de pixels avaliados corretamente, mas pode ser excessivamente otimista quando uma classe ocupa grande área. Macro F1 resume equilíbrio entre precisão e revocação por classe. Macro IoU é a métrica mais rígida e foi usada como principal critério porque penaliza simultaneamente falsos positivos e falsos negativos em cada classe.",
        "",
        "Resultados muito altos em validação e teste indicam que os polígonos rotulados, o grid full-res e o desenho de split produziram um problema separável para os blocos avaliados. Isso é bom, mas não elimina a necessidade de auditoria visual. A avaliação mede desempenho contra os rótulos disponíveis; se os rótulos tiverem bordas imprecisas, classes semanticamente misturadas ou lacunas fora dos polígonos, essas limitações não aparecem automaticamente nas métricas.",
        "",
        "No uso final, os rasters de diagnóstico do ensemble são tão importantes quanto o raster de classe. Áreas com baixa margem, baixa concordância ou entropia mais alta devem ser priorizadas em revisão humana, principalmente quando coincidem com transições entre vegetação campestre, vegetação florestal, solo exposto e áreas artificiais.",
        "",
        "## Relação com o IBGE adaptado",
        "",
        "O LULC entra no IBGE adaptado como USOVEG. A conversão é feita depois da classificação: classe 1 vira nota 10, classe 2 vira nota 5, classe 3 vira nota 0 e é excluída da máscara válida, classe 4 vira nota 2 e classe 5 vira nota 1. Essa transformação é deliberadamente simples para manter rastreabilidade entre o mapa de cobertura e a álgebra IBGE.",
        "",
        "Como o peso de USOVEG na álgebra final é 0,10, erros de LULC não dominam sozinhos o score IBGE. Ainda assim, podem alterar o resultado em zonas urbanizadas, solo exposto e bordas de água. A classe de água é especialmente sensível porque afeta a máscara válida, não apenas a nota ponderada.",
        "",
        "## Limitações e cuidados",
        "",
        "O treinamento depende dos polígonos disponíveis. Áreas não representadas nos polígonos podem ser classificadas com menor confiabilidade, mesmo que a métrica em validação e teste seja alta. A ortofoto também representa uma data específica; mudanças posteriores de uso do solo não são capturadas sem nova imagem ou nova inferência.",
        "",
        "O produto full-res tem granularidade de 16 cm, mas a semântica das classes não deve ser interpretada como verdade absoluta em cada pixel isolado. Em ortofotos de alta resolução, bordas de copa, sombras, transições solo-vegetação e superfícies parcialmente cobertas podem misturar respostas espectrais. Por isso, a leitura mais robusta é espacial: padrões contínuos e zonas de incerteza são mais informativos do que pixels isolados.",
        "",
        "## Reprodutibilidade",
        "",
        "A execução full-res foi preparada para máquina externa com CUDA. O script `IBGE_method/own_LULC/run_fullres_external.sh` recebe a ortofoto como argumento, exporta os caminhos necessários e executa a varredura full-res com retomada segura. O ensemble pode ser reconstruído a partir dos rasters de probabilidade já gerados, sem retreinar os modelos, desde que os artefatos de cada votante estejam presentes.",
        "",
        "Os principais arquivos de controle são `lulc_inputs.py`, `sweep_results.json`, `ensemble_results.json` e `selected_experiment.json`. O primeiro descreve o que deveria ser executado; os três últimos descrevem o que foi efetivamente executado e selecionado. Esta aba lê esses artefatos e evita copiar modelos ou GeoTIFFs para o website.",
        "",
        "## Observação sobre placeholders",
        "",
        report["metrics_note"],
        "",
    ]
    return "\n".join(lines)


def complete_method_markdown(config: Mapping[str, Any], summary: Mapping[str, Any], stats: Mapping[str, Any]) -> str:
    class5 = stats["class5"]["values"]
    class3 = summary.get("class_distribution", {}).get("classes", {})
    lulc = stats["lulc"]["values"]
    slope = stats["slope"]
    geom = stats["geomorphology"]
    geol = stats["geology"]
    ped = stats["pedology"]
    pluv = stats["pluviosity"]
    pluv_mm = stats["pluviosity_mm"]
    valid = stats["valid"]["values"]
    valid_pixels = int(valid["1"]["pixels"])
    invalid_pixels = int(valid["0"]["pixels"])
    ref = config["reference_grid"]
    pixel_area = abs(float(ref["resolution"][0]) * float(ref["resolution"][1]))
    total_grid_pixels = int(ref["width"]) * int(ref["height"])
    lulc_mapping = config["input_proxies"]["USOVEG"]["class_to_ibge_note"]
    lulc_note_mean = sum(float(lulc_mapping[str(value)]) * payload["fraction"] for value, payload in lulc.items())
    weighted_means = {
        "DECL": slope["mean"] * config["weights"]["slope"],
        "GEM": geom["mean"] * config["weights"]["geomorphology"],
        "GEO": geol["mean"] * config["weights"]["geology"],
        "PED": ped["mean"] * config["weights"]["pedology"],
        "USOVEG": lulc_note_mean * config["weights"]["land_use"],
        "PLUV": pluv["mean"] * config["weights"]["pluviosity"],
    }
    output_lines = [f"- `{Path(path).name}`." for path in summary.get("outputs", [])]
    lines = [
        "# Descrição completa do método",
        "",
        "Este texto documenta a geração do produto IBGE adaptado de suscetibilidade a movimentos de massa. Ele se baseia na revisão metodológica de `IBGE_method/IBGE_method_full_review.md` e nos números efetivamente gerados nesta execução.",
        "",
        "## Diferença entre o produto IBGE nacional e esta adaptação",
        "",
        "O método nacional do IBGE integra seis temas ambientais em uma escala comum de potencialidade de 1 a 10: Geologia, Geomorfologia, Pedologia, Cobertura e uso da terra com Vegetação, Declividade e Pluviosidade. No fluxo nacional estrito, esses temas são agregados à Grade Estatística de 1 km x 1 km por máxima sobreposição.",
        "",
        "Nesta adaptação, a grade de 1 km foi substituída pelo grid do DTM final do drone em 16 cm. A álgebra foi aplicada pixel a pixel, preservando pesos, sentido físico das notas e classes finais da metodologia IBGE.",
        "",
        f"O grid de referência foi `{ref['path']}`, em `{ref['crs']}`, com {fmt_int(ref['width'])} colunas por {fmt_int(ref['height'])} linhas. A resolução é {fmt_number(ref['resolution'][0], 2)} m por {fmt_number(ref['resolution'][1], 2)} m, ou {fmt_number(pixel_area, 4)} m² por pixel. O retângulo total tem {fmt_int(total_grid_pixels)} pixels.",
        "",
        "## Cadeia operacional executada",
        "",
        "- O DTM final foi usado como raster de referência para CRS, transform, shape e resolução.",
        f"- O processamento foi feito em tiles de {config['tile_processing']['tile_size_pixels']} pixels.",
        f"- A declividade foi calculada com halo de {config['tile_processing']['slope_halo_pixels']} pixel para reduzir artefatos de borda.",
        "- Camadas categóricas foram alinhadas por vizinho mais próximo.",
        "- Camadas contínuas, como pluviosidade antes da classificação, foram alinhadas por interpolação bilinear.",
        "- Pixels com nodata, água ou nota temática inválida foram removidos da máscara válida.",
        "",
        "## Insumos usados",
        "",
        f"- **Declividade / DECL:** `{config['input_proxies']['DECL']['path']}`.",
        f"- **Cobertura e uso / USOVEG:** `{config['input_proxies']['USOVEG'].get('path', '')}`.",
        f"- **Geologia / GEO:** `{SIG_PATHS['geology']}`, campo `SIGLA_UNID`.",
        f"- **Pedologia / PED:** `{SIG_PATHS['pedology']}`, campo `DESC_`.",
        f"- **Geomorfologia / GEM:** `{SIG_PATHS['relief']}`, campo `Classe`.",
        f"- **Pluviosidade / PLUV:** `{SIG_PATHS['rain_pma2']}`.",
        "",
        "## Declividade, DECL",
        "",
        f"A declividade foi derivada do DTM final do drone e classificada pelos limiares IBGE: 0-3% = 1, 3-8% = 3, 8-20% = 5, 20-45% = 8, 45-75% = 9 e acima de 75% = 10. No produto, DECL teve {fmt_int(slope['pixels'])} pixels avaliados, variou de {fmt_number(slope['min'])} a {fmt_number(slope['max'])}, com média {fmt_number(slope['mean'])} e desvio padrão {fmt_number(slope['std'])}. Sua contribuição média ponderada foi {fmt_number(weighted_means['DECL'])}.",
        "",
        "## Cobertura e uso da terra com vegetação, USOVEG",
        "",
        f"O LULC full-res por deep learning foi convertido para notas IBGE: área artificial = 10, área descoberta = 5, corpo d'água = 0 e exclusão, vegetação campestre = 2, vegetação florestal = 1. A distribuição no DTM válido foi: área artificial {fmt_percent(lulc['1']['fraction'])}, área descoberta {fmt_percent(lulc['2']['fraction'])}, corpo d'água {fmt_percent(lulc['3']['fraction'])}, vegetação campestre {fmt_percent(lulc['4']['fraction'])} e vegetação florestal {fmt_percent(lulc['5']['fraction'])}. A nota média USOVEG foi {fmt_number(lulc_note_mean)}.",
        "",
        "## Geologia, GEO",
        "",
        f"No método estrito, GEO é média de litologia, gênese, província estrutural e subprovíncia estrutural. Nesta adaptação foi usado o proxy local pelo campo `SIGLA_UNID`. A unidade no recorte válido foi `PRps`, nota 7,3. A camada teve {fmt_int(geol['pixels'])} pixels avaliados e ficou constante em {fmt_number(geol['mean'])}.",
        "",
        "## Pedologia, PED",
        "",
        f"No método estrito, PED é `max(PROF, TEXT, RELTEXT)` para o solo dominante. Nesta adaptação foi usado o proxy pelo campo `DESC_`: `PEe1` = 8 e `Rios` = 0/excluído. A camada teve {fmt_int(ped['pixels'])} pixels válidos, variando de {fmt_number(ped['min'])} a {fmt_number(ped['max'])}.",
        "",
        "## Geomorfologia, GEM",
        "",
        f"A geomorfologia usa padrões locais de relevo como proxy dos modelados BDIA. O mapeamento foi: Planícies e terraços fluviais = 1, Colinas = 3, Morros baixos = 7 e Morros altos = 9. GEM avaliou {fmt_int(geom['pixels'])} pixels, variou de {fmt_number(geom['min'])} a {fmt_number(geom['max'])}, com média {fmt_number(geom['mean'])}.",
        "",
        "## Pluviosidade, PLUV",
        "",
        f"A pluviosidade veio do Atlas Pluviométrico CPRM/SGB PMA. Nesta adaptação, a nota foi interpolada continuamente entre limiares IBGE para preservar variação local. A PMA variou de {fmt_number(pluv_mm['min'])} mm a {fmt_number(pluv_mm['max'])} mm, gerando PLUV entre {fmt_number(pluv['min'])} e {fmt_number(pluv['max'])}.",
        "",
        "## Álgebra final",
        "",
        "`S = 0,15*GEO + 0,20*GEM + 0,15*PED + 0,10*USOVEG + 0,35*DECL + 0,05*PLUV`",
        "",
        "Contribuições médias ponderadas nesta execução:",
        "",
        f"- DECL: {fmt_number(weighted_means['DECL'])}.",
        f"- GEM: {fmt_number(weighted_means['GEM'])}.",
        f"- GEO: {fmt_number(weighted_means['GEO'])}.",
        f"- PED: {fmt_number(weighted_means['PED'])}.",
        f"- USOVEG: {fmt_number(weighted_means['USOVEG'])}.",
        f"- PLUV: {fmt_number(weighted_means['PLUV'])}.",
        "",
        "## Resultado gerado",
        "",
        f"A máscara final contém {fmt_int(valid_pixels)} pixels válidos, equivalentes a {fmt_area(valid['1']['area_m2'])} m², e {fmt_int(invalid_pixels)} pixels inválidos, equivalentes a {fmt_area(valid['0']['area_m2'])} m². A fração final válida foi {fmt_percent(summary['valid_fraction'])}.",
        "",
        f"O score 1-10 variou de {fmt_number(summary['score_1_to_10_min'])} a {fmt_number(summary['score_1_to_10_max'])}. A distribuição das cinco classes finais foi:",
        "",
        f"- Muito baixa: {fmt_int(class5['1']['pixels'])} pixels, {fmt_area(class5['1']['area_m2'])} m², {fmt_percent(class5['1']['fraction'])}.",
        f"- Baixa: {fmt_int(class5['2']['pixels'])} pixels, {fmt_area(class5['2']['area_m2'])} m², {fmt_percent(class5['2']['fraction'])}.",
        f"- Média: {fmt_int(class5['3']['pixels'])} pixels, {fmt_area(class5['3']['area_m2'])} m², {fmt_percent(class5['3']['fraction'])}.",
        f"- Alta: {fmt_int(class5['4']['pixels'])} pixels, {fmt_area(class5['4']['area_m2'])} m², {fmt_percent(class5['4']['fraction'])}.",
        f"- Muito alta: {fmt_int(class5['5']['pixels'])} pixels, {fmt_area(class5['5']['area_m2'])} m², {fmt_percent(class5['5']['fraction'])}.",
        "",
        "A versão em três classes agregou o resultado em níveis gerais:",
        "",
        f"- Classe 1: {fmt_int(class3.get('1', {}).get('pixels', 0))} pixels, {fmt_area(class3.get('1', {}).get('area_m2', 0))} m², {fmt_percent(class3.get('1', {}).get('fraction', 0))}.",
        f"- Classe 2: {fmt_int(class3.get('2', {}).get('pixels', 0))} pixels, {fmt_area(class3.get('2', {}).get('area_m2', 0))} m², {fmt_percent(class3.get('2', {}).get('fraction', 0))}.",
        f"- Classe 3: {fmt_int(class3.get('3', {}).get('pixels', 0))} pixels, {fmt_area(class3.get('3', {}).get('area_m2', 0))} m², {fmt_percent(class3.get('3', {}).get('fraction', 0))}.",
        "",
        "## Arquivos produzidos",
        "",
        *output_lines,
        "",
        "## Interpretação",
        "",
        "O resultado deve ser interpretado como produto IBGE adaptado de alta resolução, adequado para inspeção local. Geologia, Pedologia e Geomorfologia são proxies locais compatíveis; eles preservam a escala e direção física da metodologia, mas não reproduzem integralmente as bases nacionais BDIA na grade de 1 km.",
        "",
    ]
    return "\n".join(lines)


def inline_markdown(text: str) -> str:
    escaped = escape(text)
    parts = escaped.split("`")
    for idx in range(1, len(parts), 2):
        parts[idx] = f"<code>{parts[idx]}</code>"
    text = "".join(parts)
    text = text.replace("**", "")
    return text


def markdown_to_html(markdown: str) -> str:
    html_parts = []
    in_list = False
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue
        if line.startswith("# "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h2>{escape(line[2:])}</h2>")
        elif line.startswith("## "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h3>{escape(line[3:])}</h3>")
        elif line.startswith("- "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{inline_markdown(line[2:])}</li>")
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<p>{inline_markdown(line)}</p>")
    if in_list:
        html_parts.append("</ul>")
    return "\n".join(html_parts)


def stats_table(stats: Mapping[str, Any], *, labels: Optional[Mapping[int, str]] = None) -> str:
    if "values" in stats:
        rows = []
        for value, payload in stats["values"].items():
            label = labels.get(int(value), "") if labels else ""
            rows.append(
                [
                    value,
                    label,
                    payload["pixels"],
                    f"{payload['area_m2']:.2f}",
                    f"{100.0 * payload['fraction']:.2f}%",
                ]
            )
        return mapping_table(rows, ["Valor", "Classe", "Pixels", "Area (m2)", "Fração"])
    return mapping_table(
        [
            ["Mínimo", f"{stats.get('min'):.4f}" if stats.get("min") is not None else "sem dado"],
            ["Máximo", f"{stats.get('max'):.4f}" if stats.get("max") is not None else "sem dado"],
            ["Média", f"{stats.get('mean'):.4f}"],
            ["Desvio padrão", f"{stats.get('std'):.4f}"],
            ["Pixels válidos", stats.get("pixels")],
        ],
        ["Métrica", "Valor"],
    )


def section(title: str, image: str, text: str, *tables: str) -> str:
    return (
        f"<div class=\"figure\"><img src=\"{escape(image)}\" alt=\"{escape(title)}\"></div>"
        f"<div class=\"copy\"><h2>{escape(title)}</h2>{text}{''.join(tables)}</div>"
    )


def metric_cards(rows: Sequence[Sequence[Any]]) -> str:
    cards = "".join(
        f"<div class=\"metric-card\"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>"
        for label, value in rows
    )
    return f"<div class=\"metric-grid\">{cards}</div>"


def swipe_compare(title: str, before: str, after: str, before_label: str, after_label: str, note: str) -> str:
    return (
        "<article class=\"swipe-card\">"
        f"<h3>{escape(title)}</h3>"
        f"<p>{escape(note)}</p>"
        "<div class=\"swipe-compare\" style=\"--pos:50%\">"
        f"<img class=\"swipe-img\" src=\"{escape(after)}\" alt=\"{escape(after_label)}\">"
        f"<img class=\"swipe-img swipe-before\" src=\"{escape(before)}\" alt=\"{escape(before_label)}\">"
        "<div class=\"swipe-handle\" aria-hidden=\"true\"></div>"
        "<input class=\"swipe-range\" type=\"range\" min=\"0\" max=\"100\" value=\"50\" aria-label=\"Arraste para comparar\">"
        f"<span class=\"swipe-label swipe-left\">{escape(before_label)}</span>"
        f"<span class=\"swipe-label swipe-right\">{escape(after_label)}</span>"
        "</div>"
        "</article>"
    )


def artifact_link(href: str, label: str) -> str:
    return f"<li><a href=\"{escape(href)}\">{escape(label)}</a></li>"


def dtm_sensitivity_section(sensitivity: Mapping[str, Any]) -> str:
    payload = sensitivity["payload"]
    assets = sensitivity["assets"]
    display = sensitivity["display"]
    score_stats = payload["score_difference_stats"]
    dtm_stats = payload["dtm_difference_stats"]
    valid = payload["valid_pixel_accounting"]
    transition = payload["class5_transition"]
    path_rows = [
        ["DTM final", payload["final_dtm"]],
        ["DTM tendencioso", payload["alternative_dtm"]],
        ["LULC mantido", payload["lulc"]],
        ["Score final", payload["final_score"]],
        ["Score alternativo", payload["alternative_score"]],
    ]
    raster_rows = [
        [label, meta["path"], f"{meta['size_mb']:.2f} MB", "não incluído no website"]
        for label, meta in sensitivity["raster_artifacts"].items()
    ]
    artifact_links = "".join(
        [
            artifact_link(assets["report_md"], "Relatório Markdown do experimento"),
            artifact_link(assets["stats_json"], "Estatísticas JSON"),
            artifact_link(assets["class_transition_csv"], "Matriz CSV de transição de classes"),
            artifact_link(assets["histogram_csv"], "Histograma CSV da diferença de score"),
            artifact_link(assets["dtm_stats_csv"], "Estatísticas CSV da diferença entre DTMs"),
            artifact_link(assets["dtm_histogram_csv"], "Histograma CSV da diferença entre DTMs"),
        ]
    )
    score_metric_cards = metric_cards(
        [
            ["Pixels comparados", fmt_int(valid["common_valid_pixels"])],
            ["Área comparada", f"{fmt_area(valid['common_valid_area_m2'])} m²"],
            ["Diferença média", fmt_number(score_stats["mean"])],
            ["MAE", fmt_number(score_stats["mean_absolute_error"])],
            ["RMSE", fmt_number(score_stats["rmse"])],
            ["Score aumentou", fmt_percent(score_stats["positive_fraction"])],
            ["Score reduziu", fmt_percent(score_stats["negative_fraction"])],
            ["Classe mudou", fmt_percent(transition["upward_class_fraction"] + transition["downward_class_fraction"])],
        ]
    )
    dtm_metric_cards = metric_cards(
        [
            ["Pixels DTM válidos", fmt_int(valid["dtm_common_valid_pixels"])],
            ["Diferença média", f"{fmt_number(dtm_stats['mean'])} m"],
            ["Mediana", f"{fmt_number(dtm_stats['median'])} m"],
            ["Desvio padrão", f"{fmt_number(dtm_stats['std'])} m"],
            ["Mínimo", f"{fmt_number(dtm_stats['min'])} m"],
            ["Máximo", f"{fmt_number(dtm_stats['max'])} m"],
            ["DTM alt. mais alto", fmt_percent(dtm_stats["positive_fraction"])],
            ["DTM alt. mais baixo", fmt_percent(dtm_stats["negative_fraction"])],
        ]
    )
    return (
        "<div class=\"sensitivity-page\">"
        "<section class=\"sensitivity-block\">"
        "<h2>Comparação com DTM tendencioso</h2>"
        "<p>Esta aba documenta o experimento em que o produto IBGE adaptado foi recalculado com "
        "<code>DTM_OTJC_3_1_16cm.tif</code>, um DTM afetado por erros sistemáticos. O produto oficial "
        "não foi regravado; ele foi apenas usado como referência para comparação pixel a pixel.</p>"
        "<p>A convenção adotada em todas as diferenças é <code>alternativo - final</code>. Valores positivos "
        "indicam aumento no DTM ou no score de suscetibilidade quando o DTM tendencioso é usado.</p>"
        f"{mapping_table(path_rows, ['Insumo', 'Caminho'])}"
        "</section>"
        "<section class=\"sensitivity-block\">"
        "<h2>Resumo estatístico</h2>"
        "<h3>Diferença entre produtos IBGE</h3>"
        f"{score_metric_cards}"
        "<h3>Diferença entre DTMs</h3>"
        f"{dtm_metric_cards}"
        "</section>"
        "<section class=\"sensitivity-block\">"
        "<h2>Sliders comparativos</h2>"
        "<div class=\"swipe-grid\">"
        f"{swipe_compare('DTM final vs DTM tendencioso', assets['dtm_final'], assets['dtm_biased'], 'DTM final', 'DTM tendencioso', 'Ambos estão em preto-e-branco com a mesma escala para preservar uma comparação visual justa.')}"
        f"{swipe_compare('Produto IBGE final vs produto alternativo', assets['score_final'], assets['score_biased'], 'Score final', 'Score alternativo', 'Ambos os scores estão em viridis fixo na escala 0-10.')}"
        "</div>"
        "</section>"
        "<section class=\"sensitivity-block\">"
        "<h2>Rasters de diferença</h2>"
        "<div class=\"difference-grid\">"
        "<figure>"
        f"<img src=\"{escape(assets['dtm_difference'])}\" alt=\"Diferença entre DTMs\">"
        "<figcaption>"
        "Diferença entre DTMs em escala de cinza centrada em zero. Preto indica DTM alternativo mais baixo; "
        "cinza médio indica ausência de diferença; branco indica DTM alternativo mais alto. "
        f"Limite simétrico exibido: ±{fmt_number(display['dtm_difference_symmetric_limit'])} m."
        "</figcaption>"
        "</figure>"
        "<figure>"
        f"<img src=\"{escape(assets['score_difference'])}\" alt=\"Diferença entre scores IBGE\">"
        "<figcaption>"
        "Diferença entre scores IBGE em viridis. Roxo/azul indica redução do score no produto alternativo; "
        "verde/amarelo indica aumento. "
        f"Intervalo exibido: {fmt_number(display['score_difference_vmin'])} a {fmt_number(display['score_difference_vmax'])}."
        "</figcaption>"
        "</figure>"
        "</div>"
        "</section>"
        "<section class=\"sensitivity-block\">"
        "<h2>Histograma e artefatos leves</h2>"
        "<div class=\"histogram-panel\">"
        f"<img src=\"{escape(assets['histogram'])}\" alt=\"Histograma das diferenças de score\">"
        "<div>"
        "<p>O histograma resume a distribuição de <code>score alternativo - score final</code> nos pixels válidos comuns.</p>"
        f"<ul class=\"artifact-list\">{artifact_links}</ul>"
        "</div>"
        "</div>"
        "</section>"
        "<section class=\"sensitivity-block\">"
        "<h2>Rasters completos fora do website</h2>"
        "<p>Os GeoTIFFs completos não são copiados nem linkados dentro de <code>IBGE_method/website/</code>, para respeitar o limite rígido de 100 MB do GitHub. A tabela abaixo registra os caminhos locais dos artefatos completos.</p>"
        f"{mapping_table(raster_rows, ['Artefato', 'Caminho local', 'Tamanho', 'Política web'])}"
        "</section>"
        "</div>"
    )


def lulc_metric_table(voters: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for row in voters:
        rows.append(
            [
                row.get("model_order"),
                row.get("display_name"),
                row.get("feature_set") or "pendente",
                row.get("loss") or "pendente",
                fmt_optional_number(row.get("resolution"), 2),
                row.get("best_epoch") or "pendente",
                fmt_optional_number(row.get("val_macro_iou")),
                fmt_optional_number(row.get("val_macro_f1")),
                fmt_optional_number(row.get("val_overall_accuracy")),
                fmt_optional_number(row.get("test_macro_iou")),
                fmt_optional_number(row.get("test_macro_f1")),
                fmt_optional_number(row.get("test_overall_accuracy")),
            ]
        )
    if not rows:
        rows.append(["pendente", "pendente", "pendente", "pendente", "pendente", "pendente", "pendente", "pendente", "pendente", "pendente", "pendente", "pendente"])
    return mapping_table(
        rows,
        [
            "#",
            "Modelo",
            "Features",
            "Loss",
            "Res. (m)",
            "Melhor época",
            "Val IoU",
            "Val F1",
            "Val Acc.",
            "Test IoU",
            "Test F1",
            "Test Acc.",
        ],
    )


def lulc_ensemble_metric_table(ensemble: Mapping[str, Any]) -> str:
    metrics = ensemble.get("metrics", {}) if isinstance(ensemble, Mapping) else {}
    rows = []
    for split_name, label in (("val", "Validação"), ("test", "Teste final")):
        payload = metrics.get(split_name, {}) if isinstance(metrics, Mapping) else {}
        rows.append(
            [
                label,
                fmt_optional_int(payload.get("evaluated_pixels")),
                fmt_optional_number(payload.get("macro_iou")),
                fmt_optional_number(payload.get("macro_f1")),
                fmt_optional_number(payload.get("overall_accuracy")),
            ]
        )
    return mapping_table(rows, ["Split", "Pixels avaliados", "Macro IoU", "Macro F1", "Acurácia global"])


def lulc_per_class_metric_table(ensemble: Mapping[str, Any], class_names: Mapping[int, str]) -> str:
    metrics = ensemble.get("metrics", {}) if isinstance(ensemble, Mapping) else {}
    rows = []
    for split_name, label in (("val", "Validação"), ("test", "Teste final")):
        payload = metrics.get(split_name, {}) if isinstance(metrics, Mapping) else {}
        per_class = payload.get("per_class", {}) if isinstance(payload, Mapping) else {}
        for internal_idx_str, class_metrics in sorted(per_class.items(), key=lambda item: int(item[0])):
            final_class = int(internal_idx_str) + 1
            rows.append(
                [
                    label,
                    f"{final_class} - {class_names.get(final_class, '')}",
                    fmt_optional_int(class_metrics.get("support_pixels")),
                    fmt_optional_int(class_metrics.get("predicted_pixels")),
                    fmt_optional_number(class_metrics.get("iou")),
                    fmt_optional_number(class_metrics.get("f1")),
                ]
            )
    if not rows:
        rows.append(["pendente", "pendente", "pendente", "pendente", "pendente", "pendente"])
    return mapping_table(rows, ["Split", "Classe final", "Suporte", "Preditos", "IoU", "F1"])


def lulc_agreement_table(ensemble: Mapping[str, Any], class_names: Mapping[int, str]) -> str:
    agreement = ensemble.get("agreement_summary", {}) if isinstance(ensemble, Mapping) else {}
    by_class = agreement.get("by_predicted_class", {}) if isinstance(agreement, Mapping) else {}
    rows = []
    for class_id_str, payload in sorted(by_class.items(), key=lambda item: int(item[0])):
        class_id = int(class_id_str)
        rows.append(
            [
                f"{class_id} - {class_names.get(class_id, '')}",
                fmt_optional_int(payload.get("pixels")),
                fmt_optional_number(payload.get("mean_agreement")),
                fmt_optional_number(payload.get("mean_confidence")),
                fmt_optional_number(payload.get("mean_margin")),
                fmt_optional_number(payload.get("mean_entropy")),
            ]
        )
    if not rows:
        rows.append(["pendente", "pendente", "pendente", "pendente", "pendente", "pendente"])
    return mapping_table(rows, ["Classe prevista", "Pixels", "Concordância", "Confiança", "Margem", "Entropia"])


def lulc_artifact_table(report: Mapping[str, Any]) -> str:
    ensemble = report.get("ensemble", {}) if isinstance(report.get("ensemble", {}), Mapping) else {}
    output_paths: Dict[str, Any] = {}
    for key in ("promoted_lulc", "promoted_probabilities"):
        if ensemble.get(key):
            output_paths[key] = ensemble[key]
    outputs = ensemble.get("outputs", {}) if isinstance(ensemble.get("outputs", {}), Mapping) else {}
    output_paths.update(outputs)
    output_paths["selected_experiment.json"] = report["selected_experiment_path"]
    output_paths["ensemble_results.json"] = report["ensemble_results_path"]
    output_paths["sweep_results.json"] = report["sweep_results_path"]

    rows = []
    for label, raw_path in output_paths.items():
        path = Path(str(raw_path))
        size = f"{path.stat().st_size / (1024 * 1024):.2f} MB" if path.exists() else "pendente"
        rows.append([label, str(raw_path), size, "não copiado para o website"])
    return mapping_table(rows, ["Artefato", "Caminho local", "Tamanho", "Política web"])


def single_lulc_artifact_table(report: Mapping[str, Any]) -> str:
    paths: Dict[str, Any] = {
        "sweep_results.json": report["sweep_results_path"],
        "single_model_selection.json": report["single_selection_path"],
    }
    selection = report.get("single_selection", {}) if isinstance(report.get("single_selection", {}), Mapping) else {}
    if selection.get("selected_lulc_path"):
        paths["selected_lulc"] = selection["selected_lulc_path"]
    selected = report.get("selected_single", {}) if isinstance(report.get("selected_single", {}), Mapping) else {}
    if selected.get("output_dir"):
        paths["selected_run_dir"] = selected["output_dir"]
    rows = []
    for label, raw_path in paths.items():
        path = Path(str(raw_path))
        if path.is_dir():
            size = "diretório"
        else:
            size = f"{path.stat().st_size / (1024 * 1024):.2f} MB" if path.exists() else "pendente"
        rows.append([label, str(raw_path), size, "não copiado para o website"])
    return mapping_table(rows, ["Artefato", "Caminho local", "Tamanho", "Política web"])


def single_lulc_method_section(report: Mapping[str, Any], lulc_thumbnail: str) -> str:
    selected = report.get("selected_single", {}) if isinstance(report.get("selected_single", {}), Mapping) else {}
    cards = metric_cards(
        [
            ["Modelo selecionado", selected.get("display_name", architecture_label(selected)) if selected else "pendente"],
            ["Run selecionado", selected.get("run_id", "pendente")],
            ["Val macro IoU", fmt_optional_number(selected.get("val_macro_iou"))],
            ["Test macro IoU", fmt_optional_number(selected.get("test_macro_iou"))],
            ["Melhor época", selected.get("best_epoch", "pendente")],
            ["Features", selected.get("feature_set", "pendente")],
            ["Loss", selected.get("loss", "pendente")],
            ["Ensemble", "não usado"],
        ]
    )
    class_rows = [
        [row["class_id"], row["name"], fmt_optional_number(row["ibge_land_use_note"], 1)]
        for row in report["class_definitions"]
    ]
    return (
        "<article class=\"method-description\">"
        f"{markdown_to_html(report['markdown'])}"
        "<h3>Resumo numérico do LULC Lite</h3>"
        f"{cards}"
        f"<div class=\"figure inline\"><img src=\"{escape(lulc_thumbnail)}\" alt=\"LULC Lite usado no método IBGE\"></div>"
        "<h3>Classes finais e notas USOVEG</h3>"
        f"{mapping_table(class_rows, ['Código final', 'Classe LULC', 'Nota IBGE USOVEG'])}"
        "<h3>Comparação dos cinco candidatos full-res</h3>"
        f"{lulc_metric_table(report.get('candidate_runs', []))}"
        "<h3>Artefatos locais</h3>"
        "<p>Esta aba registra os caminhos dos artefatos completos, mas não copia modelos, GeoTIFFs ou probabilidades para o website lite.</p>"
        f"{single_lulc_artifact_table(report)}"
        "<p><a href=\"descricao_completa_metodo_lulc.md\">Abrir esta descrição em Markdown</a></p>"
        "<p><a href=\"assets/lulc_method_report.json\">Abrir metadados LULC Lite em JSON</a></p>"
        "</article>"
    )


def lite_full_comparison_section(comparison: Mapping[str, Any]) -> str:
    stats = comparison["score_difference_stats"]
    accounting = comparison["valid_pixel_accounting"]
    transition = comparison["class5_transition_summary"]
    assets = comparison["assets"]
    cards = metric_cards(
        [
            ["Pixels comuns", fmt_int(accounting["common_valid_pixels"])],
            ["Área comum", f"{fmt_area(accounting['common_valid_area_m2'])} m²"],
            ["Diferença média", fmt_number(stats["mean"])],
            ["Mediana", fmt_number(stats["median"])],
            ["MAE", fmt_number(stats["mae"])],
            ["RMSE", fmt_number(stats["rmse"])],
            ["Lite maior", fmt_percent(stats["positive_fraction"])],
            ["Lite menor", fmt_percent(stats["negative_fraction"])],
            ["Classe mudou", fmt_percent(transition["changed_fraction"])],
            ["Classe aumentou", fmt_percent(transition["upward_fraction"])],
            ["Classe reduziu", fmt_percent(transition["downward_fraction"])],
            ["Classe igual", fmt_percent(transition["unchanged_fraction"])],
        ]
    )
    stats_rows = [
        ["Mínimo", fmt_number(stats["min"])],
        ["P01", fmt_number(stats["p01"])],
        ["P05", fmt_number(stats["p05"])],
        ["P10", fmt_number(stats["p10"])],
        ["P25", fmt_number(stats["p25"])],
        ["Mediana / P50", fmt_number(stats["median"])],
        ["P75", fmt_number(stats["p75"])],
        ["P90", fmt_number(stats["p90"])],
        ["P95", fmt_number(stats["p95"])],
        ["P99", fmt_number(stats["p99"])],
        ["Máximo", fmt_number(stats["max"])],
        ["Desvio padrão", fmt_number(stats["std"])],
        ["Diferença absoluta mediana", fmt_number(stats["median_absolute_difference"])],
    ]
    threshold_rows = [
        [f"<= {threshold}", fmt_percent(fraction)]
        for threshold, fraction in stats["within_abs_threshold_fraction"].items()
    ]
    accounting_rows = [
        ["Válidos no completo", fmt_int(accounting["full_valid_pixels"])],
        ["Válidos no lite", fmt_int(accounting["lite_valid_pixels"])],
        ["Válidos em ambos", fmt_int(accounting["common_valid_pixels"])],
        ["Válidos só no completo", fmt_int(accounting["full_only_valid_pixels"])],
        ["Válidos só no lite", fmt_int(accounting["lite_only_valid_pixels"])],
    ]
    artifact_links = "".join(
        [
            artifact_link(assets["stats_json"], "Estatísticas JSON"),
            artifact_link(assets["histogram_csv"], "Histograma CSV"),
            artifact_link(assets["transition_csv"], "Matriz de transição 5 classes CSV"),
        ]
    )
    return (
        "<div class=\"sensitivity-page\">"
        "<section class=\"sensitivity-block\">"
        "<h2>Comparação com o completo</h2>"
        "<p>Esta aba compara o produto IBGE lite, baseado no melhor modelo LULC individual, com o produto completo, baseado no ensemble LULC. A diferença numérica é sempre <code>lite - completo</code>. Valores positivos indicam que a versão lite aumentou o score de suscetibilidade; valores negativos indicam redução.</p>"
        f"{mapping_table([['Produto completo', comparison['complete_score']], ['Produto lite', comparison['lite_score']]], ['Produto', 'Raster de score 1-10'])}"
        "</section>"
        "<section class=\"sensitivity-block\">"
        "<h2>Resumo numérico</h2>"
        f"{cards}"
        "</section>"
        "<section class=\"sensitivity-block\">"
        "<h2>Estatísticas descritivas da diferença</h2>"
        f"{mapping_table(stats_rows, ['Métrica', 'lite - completo'])}"
        "<h3>Frações por limiar absoluto</h3>"
        f"{mapping_table(threshold_rows, ['|Diferença|', 'Fração dos pixels comuns'])}"
        "</section>"
        "<section class=\"sensitivity-block\">"
        "<h2>Máscara válida e classes finais</h2>"
        f"{mapping_table(accounting_rows, ['Conta', 'Pixels'])}"
        f"{mapping_table([['Classes iguais', fmt_int(transition['unchanged_pixels']), fmt_percent(transition['unchanged_fraction'])], ['Classes aumentaram no lite', fmt_int(transition['upward_pixels']), fmt_percent(transition['upward_fraction'])], ['Classes reduziram no lite', fmt_int(transition['downward_pixels']), fmt_percent(transition['downward_fraction'])], ['Classes mudaram', fmt_int(transition['changed_pixels']), fmt_percent(transition['changed_fraction'])]], ['Transição 5 classes', 'Pixels', 'Fração'])}"
        "</section>"
        "<section class=\"sensitivity-block\">"
        "<h2>Histograma</h2>"
        "<div class=\"histogram-panel\">"
        f"<img src=\"{escape(assets['histogram_png'])}\" alt=\"Histograma da diferença entre produto lite e completo\">"
        "<div>"
        "<p>O histograma mostra a distribuição de <code>score lite - score completo</code> nos pixels válidos comuns. A linha preta marca diferença zero; as linhas coloridas marcam média e mediana.</p>"
        f"<ul class=\"artifact-list\">{artifact_links}</ul>"
        "</div>"
        "</div>"
        "</section>"
        "</div>"
    )


def lulc_method_section(report: Mapping[str, Any], lulc_thumbnail: str) -> str:
    if report.get("report_mode") == "best_single":
        return single_lulc_method_section(report, lulc_thumbnail)

    class_names = {row["class_id"]: row["name"] for row in report["class_definitions"]}
    ensemble = report.get("ensemble", {}) if isinstance(report.get("ensemble", {}), Mapping) else {}
    agreement = ensemble.get("agreement_summary", {}) if isinstance(ensemble, Mapping) else {}
    cards = metric_cards(
        [
            ["Modelos no ensemble", fmt_optional_int(len(report.get("voters", [])))],
            ["Resolução do produto", f"{fmt_optional_number(ensemble.get('resolution'), 2)} m"],
            ["Val macro IoU ensemble", fmt_optional_number((ensemble.get("metrics", {}).get("val", {}) if isinstance(ensemble.get("metrics", {}), Mapping) else {}).get("macro_iou"))],
            ["Test macro IoU ensemble", fmt_optional_number((ensemble.get("metrics", {}).get("test", {}) if isinstance(ensemble.get("metrics", {}), Mapping) else {}).get("macro_iou"))],
            ["Confiança média", fmt_optional_number(agreement.get("mean_confidence"))],
            ["Margem média", fmt_optional_number(agreement.get("mean_margin"))],
            ["Pixels baixa margem", fmt_optional_int(agreement.get("low_margin_pixels"))],
            ["Pixels alta entropia", fmt_optional_int(agreement.get("high_entropy_pixels"))],
        ]
    )
    class_rows = [
        [row["class_id"], row["name"], fmt_optional_number(row["ibge_land_use_note"], 1)]
        for row in report["class_definitions"]
    ]
    feature_rows = [
        [name, ", ".join(payload["channels"])]
        for name, payload in report["feature_sets"].items()
    ]
    experiment_rows = [
        [
            name,
            payload["model"]["architecture"],
            payload["model"]["encoder"],
            payload["feature_set"],
            payload["loss"]["name"],
            payload["resolution"],
            payload["tile_size"],
            payload["stride"],
        ]
        for name, payload in report["fullres_experiments"].items()
    ]
    return (
        "<article class=\"method-description\">"
        f"{markdown_to_html(report['markdown'])}"
        "<h3>Resumo numérico do produto LULC full-res</h3>"
        f"{cards}"
        f"<div class=\"figure inline\"><img src=\"{escape(lulc_thumbnail)}\" alt=\"LULC usado no método IBGE\"></div>"
        "<h3>Classes finais e notas USOVEG</h3>"
        f"{mapping_table(class_rows, ['Código final', 'Classe LULC', 'Nota IBGE USOVEG'])}"
        "<h3>Feature sets disponíveis no código</h3>"
        f"{mapping_table(feature_rows, ['Feature set', 'Canais'])}"
        "<h3>Configurações full-res versionadas</h3>"
        f"{mapping_table(experiment_rows, ['Experimento', 'Arquitetura', 'Encoder', 'Features', 'Loss', 'Resolução', 'Tile', 'Stride'])}"
        "<h3>Métricas individuais dos cinco modelos</h3>"
        f"{lulc_metric_table(report.get('voters', []))}"
        "<h3>Métricas do ensemble</h3>"
        f"{lulc_ensemble_metric_table(ensemble)}"
        "<h3>Métricas do ensemble por classe</h3>"
        f"{lulc_per_class_metric_table(ensemble, class_names)}"
        "<h3>Diagnóstico de concordância do ensemble</h3>"
        f"{lulc_agreement_table(ensemble, class_names)}"
        "<h3>Artefatos locais</h3>"
        "<p>Esta aba registra os caminhos dos artefatos completos, mas não copia modelos, GeoTIFFs ou probabilidades para o website.</p>"
        f"{lulc_artifact_table(report)}"
        "<p><a href=\"descricao_completa_metodo_lulc.md\">Abrir esta descrição em Markdown</a></p>"
        "<p><a href=\"assets/lulc_method_report.json\">Abrir metadados LULC em JSON</a></p>"
        "</article>"
    )


def build_html(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    thumbs: Mapping[str, str],
    viewer: Mapping[str, Any],
    stats: Mapping[str, Any],
    sensitivity: Optional[Mapping[str, Any]],
    lulc_report: Mapping[str, Any],
    lite_full_comparison: Optional[Mapping[str, Any]],
) -> str:
    is_lite = WEBSITE_VARIANT == "lite"
    product_title = "Produto IBGE adaptado lite em 16 cm" if is_lite else "Produto IBGE adaptado em 16 cm"
    product_intro = (
        "Este relatório documenta a versão lite do produto IBGE adaptado de alta resolução. "
        "A álgebra e os proxies são os mesmos do produto principal, mas USOVEG usa apenas o "
        "melhor modelo LULC individual full-res, sem ensemble."
        if is_lite
        else "Este relatório documenta a geração do produto IBGE adaptado de alta resolução. "
        "A grade estatística nacional de 1 km foi substituída pelo grid do DTM final em 16 cm; "
        "os pesos, a escala de notas e as classes finais da metodologia IBGE foram preservados."
    )
    method_description_md = complete_method_markdown(config, summary, stats)
    method_description_html = markdown_to_html(method_description_md)
    lulc_mapping = custom_lulc_land_use_mapping()
    lulc_rows = [[k, LULC_NAMES.get(k, ""), v] for k, v in sorted(lulc_mapping.items())]
    slope_rows = [[lo, hi if hi is not None else ">=75", note] for lo, hi, note in config["slope_classes_percent_to_note"]]
    pluv_rows = [
        ["400-1000", 4],
        ["1000-1500", 6],
        ["1500-2000", 8],
        ["2000-2500", 9],
        ["2500-4300", 10],
        ["adaptação 16 cm", "interpolação contínua entre os limiares acima"],
    ]
    geom_rows = sorted(config["input_proxies"]["GEM"]["mapped_values"].items())
    geo_rows = sorted(config["input_proxies"]["GEO"]["mapped_values"].items())
    ped_rows = sorted(config["input_proxies"]["PED"]["mapped_values"].items())
    weights_rows = [[key, value] for key, value in IBGE_WEIGHTS.items()]

    tabs = [
        ("overview", "Visão Geral"),
        ("inputs", "Entradas"),
        ("lulc", "LULC / USOVEG"),
        ("slope", "Declividade"),
        ("geology", "Geologia"),
        ("pedology", "Pedologia"),
        ("geomorphology", "Geomorfologia"),
        ("pluviosity", "Pluviosidade"),
        ("algebra", "Álgebra IBGE"),
        ("results", "Resultados"),
        ("method_description", "Descrição completa do método"),
        (
            "lulc_method_description",
            "Descrição completa do método LULC Lite" if is_lite else "Descrição completa do método LULC",
        ),
    ]
    if sensitivity is not None:
        tabs.append(("dtm_sensitivity", "Comparação com DTM tendencioso"))
    if lite_full_comparison is not None:
        tabs.append(("lite_full_comparison", "Comparação com o completo"))
    tabs.append(("webviewer", "webviewer"))
    nav = "".join(
        f"<button class=\"tab-button{' active' if idx == 0 else ''}\" data-tab=\"{tab_id}\">{label}</button>"
        for idx, (tab_id, label) in enumerate(tabs)
    )

    tab_content = {
        "overview": section(
            product_title,
            thumbs["overview"],
            (
                f"<p>{escape(product_intro)}</p>"
                f"<p><strong>Fração válida:</strong> {summary['valid_fraction']:.6f}. "
                f"<strong>Score 1-10:</strong> {summary['score_1_to_10_min']:.4f} a {summary['score_1_to_10_max']:.4f}.</p>"
            ),
            mapping_table(weights_rows, ["Tema", "Peso"]),
        ),
        "inputs": section(
            "Entradas e proxies",
            thumbs["dtm"],
            (
                f"<p><strong>DTM:</strong> {escape(config['input_proxies']['DECL']['path'])}</p>"
                f"<p><strong>LULC:</strong> {escape(config['input_proxies']['USOVEG'].get('path', ''))}</p>"
                f"<p><strong>Geologia:</strong> {escape(SIG_PATHS['geology'])}</p>"
                f"<p><strong>Pedologia:</strong> {escape(SIG_PATHS['pedology'])}</p>"
                f"<p><strong>Geomorfologia:</strong> {escape(SIG_PATHS['relief'])}</p>"
                f"<p><strong>Pluviosidade:</strong> {escape(SIG_PATHS['rain_pma2'])}</p>"
            ),
        ),
        "lulc": section(
            "LULC / USOVEG",
            thumbs["lulc"],
            (
                "<p>Fonte: melhor modelo LULC individual full-res gerado por deep learning. "
                "A classe de água recebe nota 0 e é excluída dos pixels válidos. O ensemble não é usado nesta versão lite.</p>"
                if is_lite
                else "<p>Fonte: ensemble LULC full-res gerado por deep learning. A classe de água recebe nota 0 e é excluída dos pixels válidos.</p>"
            ),
            mapping_table(lulc_rows, ["Código LULC", "Classe", "Nota IBGE"]),
            stats_table(stats["lulc"], labels=LULC_NAMES),
        ),
        "slope": section(
            "Declividade",
            thumbs["slope"],
            "<p>Fonte: DTM final do drone. A declividade percentual é calculada no grid de 16 cm com halo de 1 pixel nos tiles.</p>",
            mapping_table(slope_rows, ["Declividade (%) - início", "Declividade (%) - fim", "Nota IBGE"]),
            stats_table(stats["slope"]),
        ),
        "geology": section(
            "Geologia",
            thumbs["geology"],
            (
                "<p>Proxy adaptado: a metodologia IBGE estrita calcula GEO a partir da média de "
                "litologia, gênese, província estrutural e subprovíncia estrutural. Nesta versão "
                "de alta resolução não há, no recorte operacional, a tabela BDIA completa com esses "
                "quatro componentes normalizados para cada polígono. Por isso, a camada local de "
                "geologia é usada como proxy compatível.</p>"
                "<p>O campo <code>SIGLA_UNID</code> identifica a unidade geológica dominante. Cada "
                "unidade foi convertida diretamente para uma nota GEO 1-10 já interpretada para "
                "potencialidade a movimentos de massa. No recorte atual, apenas a unidade "
                "<code>PRps</code> intercepta a área válida do DTM, resultando em nota geológica "
                "espacialmente constante. Isso não indica falha de rasterização; indica limitação "
                "da variabilidade geológica disponível dentro desta área de estudo.</p>"
            ),
            mapping_table(geo_rows, ["SIGLA_UNID", "Nota GEO"]),
            stats_table(stats["geology"]),
        ),
        "pedology": section(
            "Pedologia",
            thumbs["pedology"],
            (
                "<p>Proxy adaptado: na metodologia IBGE estrita, PED é calculado como "
                "<code>max(PROF, TEXT, RELTEXT)</code>, considerando o solo dominante e atributos "
                "pedológicos como profundidade, textura e relação/gradiente textural. A camada "
                "local disponível para esta área não expõe todos esses campos analíticos de forma "
                "compatível com a tabela BDIA/pedologia completa.</p>"
                "<p>Assim, o campo <code>DESC_</code> foi usado como legenda pedológica operacional. "
                "Cada legenda recebeu uma nota PED proxy. No recorte válido, a classe dominante é "
                "<code>PEe1</code>, mapeada para nota 8, enquanto <code>Rios</code> recebe 0 e é "
                "excluído da máscara válida. Portanto, a baixa variação pedológica no produto final "
                "decorre da resolução temática e da legenda disponível, não de simplificação no grid "
                "de 16 cm.</p>"
            ),
            mapping_table(ped_rows, ["DESC_", "Nota PED"]),
            stats_table(stats["pedology"]),
        ),
        "geomorphology": section(
            "Geomorfologia",
            thumbs["geomorphology"],
            (
                "<p>Proxy adaptado: a metodologia IBGE estrita usa modelados geomorfológicos BDIA, "
                "especialmente a quarta ordem taxonômica, para diferenciar acumulação, dissecação, "
                "dissolução e aplanamento. Nesta adaptação, a camada local de padrões de relevo foi "
                "usada como substituta de maior compatibilidade espacial com o estudo.</p>"
                "<p>O campo <code>Classe</code> representa formas de relevo como planícies, colinas, "
                "morros baixos e morros altos. A transformação segue a lógica geomorfológica do IBGE: "
                "ambientes deposicionais/planos recebem notas menores, formas intermediárias recebem "
                "notas médias e relevo mais dissecado/íngreme recebe notas altas. Diferentemente da "
                "geologia e da pedologia, esta camada apresenta variação espacial relevante dentro "
                "da área, contribuindo para contrastes locais no score final.</p>"
            ),
            mapping_table(geom_rows, ["Classe", "Nota GEM"]),
            stats_table(stats["geomorphology"]),
        ),
        "pluviosity": section(
            "Pluviosidade",
            thumbs["pluviosity"],
            "<p>Fonte: Atlas Pluviométrico CPRM/SGB PMA. Para a adaptação 16 cm, a nota é interpolada continuamente entre os limiares IBGE para preservar a variação local.</p>",
            mapping_table(pluv_rows, ["Precipitação média anual (mm)", "Nota PLUV"]),
            stats_table(stats["pluviosity"]),
            f"<h3>PMA em mm</h3>{stats_table(stats['pluviosity_mm'])}<div class=\"figure inline\"><img src=\"{thumbs['pluviosity_mm']}\" alt=\"PMA em mm\"></div>",
        ),
        "algebra": section(
            "Álgebra IBGE",
            thumbs["score"],
            (
                "<p>O score final é calculado por pixel:</p>"
                "<p><code>S = 0.15*GEO + 0.20*GEM + 0.15*PED + 0.10*USOVEG + 0.35*DECL + 0.05*PLUV</code></p>"
                "<p>Pixels com água, nodata ou nota temática inválida são excluídos da máscara válida.</p>"
            ),
            mapping_table(weights_rows, ["Tema", "Peso"]),
        ),
        "results": section(
            "Resultados",
            thumbs["class5"],
            (
                "<p>O score 1-10 é reclassificado em cinco classes: muito baixa, baixa, média, alta e muito alta.</p>"
                f"<p><strong>Score 1-10:</strong> {summary['score_1_to_10_min']:.4f} a {summary['score_1_to_10_max']:.4f}.</p>"
            ),
            stats_table(stats["class5"], labels={1: "Muito baixa", 2: "Baixa", 3: "Média", 4: "Alta", 5: "Muito alta", 255: "Sem dado"}),
            f"<h3>Máscara válida</h3>{stats_table(stats['valid'], labels={0: 'Inválido', 1: 'Válido'})}<div class=\"figure inline\"><img src=\"{thumbs['valid']}\" alt=\"Máscara válida\"></div>",
        ),
        "method_description": (
            "<article class=\"method-description\">"
            f"{method_description_html}"
            "<p><a href=\"descricao_completa_metodo.md\">Abrir esta descrição em Markdown</a></p>"
            "</article>"
        ),
        "lulc_method_description": lulc_method_section(lulc_report, thumbs["lulc"]),
        "webviewer": (
            "<div class=\"viewer-shell\">"
            "<aside class=\"viewer-panel\">"
            "<h2>webviewer</h2>"
            "<p>Score final 0-10 sobreposto ao mapa base. A imagem foi derivada de "
            "<code>ibge_susceptibility_score_1to10.tif</code> e está registrada em WGS84.</p>"
            "<h3>Mapa de fundo</h3>"
            "<div class=\"map-controls\" role=\"group\" aria-label=\"Mapa de fundo\">"
            "<button type=\"button\" class=\"base-button active\" data-base=\"osm\">OSM</button>"
            "<button type=\"button\" class=\"base-button\" data-base=\"bing\">Bing imagem</button>"
            "</div>"
            "<h3>Camada IBGE</h3>"
            "<label class=\"opacity-control\">Opacidade do score"
            "<input id=\"score-opacity\" type=\"range\" min=\"0\" max=\"1\" step=\"0.05\" value=\"0.88\">"
            "</label>"
            "<div class=\"legend\"><span>0</span><div class=\"ramp\"></div><span>10</span></div>"
            "<p class=\"viewer-note\">Viridis: valores menores em roxo/azul e maiores em verde/amarelo.</p>"
            "</aside>"
            "<div class=\"map-wrap\"><div id=\"map\"></div></div>"
            "</div>"
        ),
    }
    if sensitivity is not None:
        tab_content["dtm_sensitivity"] = dtm_sensitivity_section(sensitivity)
    if lite_full_comparison is not None:
        tab_content["lite_full_comparison"] = lite_full_comparison_section(lite_full_comparison)
    sections = "".join(
        f"<section id=\"{tab_id}\" class=\"tab-panel{' active' if idx == 0 else ''}\">{tab_content[tab_id]}</section>"
        for idx, (tab_id, _) in enumerate(tabs)
    )

    viewer_json = json.dumps(viewer, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(product_title)}</title>
  <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
  <style>
    :root {{ --ink:#18212b; --muted:#5b6673; --line:#d8dee6; --bg:#f6f8fb; --panel:#ffffff; --accent:#1f6f8b; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ padding:24px 32px 16px; background:#0f2430; color:white; }}
    header h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
    header p {{ margin:0; color:#d5e4ec; max-width:1050px; line-height:1.5; }}
    nav {{ display:flex; flex-wrap:wrap; gap:6px; padding:12px 32px; background:#e9eef4; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; }}
    button.tab-button {{ border:1px solid #c8d2dc; background:white; color:#1b2a36; padding:9px 12px; border-radius:6px; cursor:pointer; font-size:14px; }}
    button.tab-button.active {{ background:var(--accent); color:white; border-color:var(--accent); }}
    .map-controls {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:10px 0 12px; }}
    .base-button {{ border:1px solid #c8d2dc; background:white; color:#1b2a36; padding:8px 12px; border-radius:6px; cursor:pointer; font-size:14px; }}
    .base-button.active {{ background:#263b4a; color:white; border-color:#263b4a; }}
    main {{ padding:24px 32px 40px; }}
    .tab-panel {{ display:none; grid-template-columns:minmax(280px, 42%) minmax(320px, 1fr); gap:24px; align-items:start; }}
    .tab-panel.active {{ display:grid; }}
    .figure img {{ width:100%; max-height:720px; object-fit:contain; background:white; border:1px solid var(--line); border-radius:8px; }}
    .figure.inline img {{ max-width:680px; margin-top:12px; }}
    .copy, #webviewer {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:20px; }}
    h2 {{ margin:0 0 12px; font-size:22px; }}
    h3 {{ margin:20px 0 8px; font-size:17px; }}
    p {{ line-height:1.55; color:var(--muted); }}
    code {{ background:#eef2f6; padding:2px 5px; border-radius:4px; }}
    table {{ width:100%; border-collapse:collapse; margin:14px 0 18px; font-size:14px; }}
    th, td {{ text-align:left; border-bottom:1px solid var(--line); padding:8px 10px; vertical-align:top; }}
    th {{ background:#f0f4f8; color:#22313d; }}
    #webviewer, #method_description, #lulc_method_description, #dtm_sensitivity, #lite_full_comparison {{ grid-column:1 / -1; background:transparent; border:0; padding:0; }}
    .method-description {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:24px; max-width:1060px; }}
    .method-description h2 {{ margin-top:0; }}
    .method-description h3 {{ margin-top:24px; }}
    .method-description p, .method-description li {{ line-height:1.65; color:var(--muted); }}
    .method-description ul {{ padding-left:22px; }}
    .sensitivity-page {{ display:grid; gap:18px; }}
    .sensitivity-block {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:22px; }}
    .sensitivity-block h2 {{ margin-top:0; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:10px; margin:12px 0 6px; }}
    .metric-card {{ border:1px solid var(--line); border-radius:8px; padding:12px 14px; background:#f8fafc; }}
    .metric-card span {{ display:block; color:var(--muted); font-size:13px; line-height:1.35; }}
    .metric-card strong {{ display:block; color:var(--ink); font-size:20px; margin-top:4px; overflow-wrap:anywhere; }}
    .swipe-grid {{ display:grid; grid-template-columns:repeat(2, minmax(320px, 1fr)); gap:18px; align-items:start; }}
    .swipe-card {{ min-width:0; }}
    .swipe-card p {{ margin-top:0; }}
    .swipe-compare {{ --pos:50%; position:relative; overflow:hidden; border:1px solid var(--line); border-radius:8px; background:white; isolation:isolate; }}
    .swipe-img {{ display:block; width:100%; height:auto; max-height:760px; object-fit:contain; }}
    .swipe-before {{ position:absolute; inset:0; height:100%; clip-path:inset(0 calc(100% - var(--pos)) 0 0); }}
    .swipe-handle {{ position:absolute; top:0; bottom:0; left:var(--pos); width:2px; background:white; box-shadow:0 0 0 1px rgba(0,0,0,.22); transform:translateX(-1px); pointer-events:none; }}
    .swipe-handle::after {{ content:""; position:absolute; top:50%; left:50%; width:34px; height:34px; border-radius:50%; background:white; border:1px solid rgba(0,0,0,.25); transform:translate(-50%, -50%); box-shadow:0 2px 8px rgba(0,0,0,.22); }}
    .swipe-range {{ position:absolute; inset:0; width:100%; height:100%; opacity:0; cursor:ew-resize; }}
    .swipe-label {{ position:absolute; top:10px; padding:5px 8px; border-radius:6px; background:rgba(15,36,48,.82); color:white; font-size:12px; pointer-events:none; }}
    .swipe-left {{ left:10px; }}
    .swipe-right {{ right:10px; }}
    .difference-grid {{ display:grid; grid-template-columns:repeat(2, minmax(320px, 1fr)); gap:18px; }}
    .difference-grid figure {{ margin:0; }}
    .difference-grid img, .histogram-panel img {{ width:100%; border:1px solid var(--line); border-radius:8px; background:white; }}
    figcaption {{ color:var(--muted); line-height:1.5; font-size:14px; margin-top:8px; }}
    .histogram-panel {{ display:grid; grid-template-columns:minmax(320px, 54%) minmax(280px, 1fr); gap:18px; align-items:start; }}
    .artifact-list {{ columns:2; padding-left:20px; }}
    .artifact-list li {{ break-inside:avoid; margin-bottom:7px; }}
    .viewer-shell {{ display:grid; grid-template-columns:320px minmax(420px,1fr); gap:16px; align-items:stretch; }}
    .viewer-panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; }}
    .viewer-panel p {{ margin-top:0; }}
    .viewer-note {{ font-size:13px; }}
    .opacity-control {{ display:grid; gap:8px; color:var(--muted); font-size:14px; margin:8px 0 14px; }}
    .opacity-control input {{ width:100%; }}
    .map-wrap {{ background:white; border:1px solid var(--line); border-radius:8px; padding:8px; }}
    #map {{ width:100%; height:76vh; min-height:560px; border-radius:6px; }}
    .legend {{ display:flex; align-items:center; gap:10px; margin-top:10px; color:var(--muted); }}
    .ramp {{ height:14px; width:280px; border-radius:999px; border:1px solid #c5cbd3; background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725); }}
    @media (max-width:900px) {{ .tab-panel.active {{ display:block; }} .copy {{ margin-top:18px; }} .viewer-shell, .swipe-grid, .difference-grid, .histogram-panel {{ display:block; }} .map-wrap, .swipe-card + .swipe-card, .difference-grid figure + figure {{ margin-top:14px; }} .artifact-list {{ columns:1; }} nav, main, header {{ padding-left:16px; padding-right:16px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(product_title)}</h1>
    <p>{escape('Relatório técnico em português da geração do produto lite: melhor modelo LULC individual, proxies, mapeamentos, álgebra e visualização interativa do score de suscetibilidade.' if is_lite else 'Relatório técnico em português da geração do produto final: entradas, proxies, mapeamentos, álgebra e visualização interativa do score de suscetibilidade.')}</p>
  </header>
  <nav>{nav}</nav>
  <main>{sections}</main>
  <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
  <script>
    document.querySelectorAll('.tab-button').forEach((button) => {{
      button.addEventListener('click', () => {{
        document.querySelectorAll('.tab-button').forEach((b) => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
        button.classList.add('active');
        document.getElementById(button.dataset.tab).classList.add('active');
        if (button.dataset.tab === 'webviewer' && window.ibgeMap) {{
          setTimeout(() => window.ibgeMap.resize(), 50);
        }}
      }});
    }});
    document.querySelectorAll('.swipe-compare').forEach((swipe) => {{
      const range = swipe.querySelector('.swipe-range');
      const updateSwipe = () => swipe.style.setProperty('--pos', `${{range.value}}%`);
      range.addEventListener('input', updateSwipe);
      updateSwipe();
    }});
    const viewer = {viewer_json};
    window.ibgeMap = new maplibregl.Map({{
      container: 'map',
      style: {{
        version: 8,
        sources: {{
          osm: {{
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png'],
            tileSize: 256,
            attribution: '© OpenStreetMap contributors'
          }},
          bing: {{
            type: 'raster',
            tiles: [
              'https://ecn.t0.tiles.virtualearth.net/tiles/a{{quadkey}}.jpeg?g=129&mkt=pt-BR&n=z',
              'https://ecn.t1.tiles.virtualearth.net/tiles/a{{quadkey}}.jpeg?g=129&mkt=pt-BR&n=z',
              'https://ecn.t2.tiles.virtualearth.net/tiles/a{{quadkey}}.jpeg?g=129&mkt=pt-BR&n=z',
              'https://ecn.t3.tiles.virtualearth.net/tiles/a{{quadkey}}.jpeg?g=129&mkt=pt-BR&n=z'
            ],
            tileSize: 256,
            attribution: '© Microsoft Bing'
          }}
        }},
        layers: [
          {{ id: 'osm-base', type: 'raster', source: 'osm', layout: {{ visibility: 'visible' }} }},
          {{ id: 'bing-base', type: 'raster', source: 'bing', layout: {{ visibility: 'none' }} }}
        ]
      }},
      center: viewer.center,
      zoom: 14,
      attributionControl: false
    }});
    window.ibgeMap.addControl(new maplibregl.NavigationControl({{ visualizePitch: false }}), 'top-right');
    window.ibgeMap.on('load', () => {{
      window.ibgeMap.addSource('score', {{
        type: 'image',
        url: viewer.image,
        coordinates: viewer.coordinates
      }});
      window.ibgeMap.addLayer({{
        id: 'score-layer',
        type: 'raster',
        source: 'score',
        paint: {{ 'raster-opacity': 0.88 }}
      }});
      window.ibgeMap.fitBounds(viewer.bounds, {{ padding: 30, duration: 0 }});
    }});
    document.getElementById('score-opacity').addEventListener('input', (event) => {{
      window.ibgeMap.setPaintProperty('score-layer', 'raster-opacity', Number(event.target.value));
    }});
    document.querySelectorAll('.base-button').forEach((button) => {{
      button.addEventListener('click', () => {{
        document.querySelectorAll('.base-button').forEach((b) => b.classList.remove('active'));
        button.classList.add('active');
        const useOsm = button.dataset.base === 'osm';
        window.ibgeMap.setLayoutProperty('osm-base', 'visibility', useOsm ? 'visible' : 'none');
        window.ibgeMap.setLayoutProperty('bing-base', 'visibility', useOsm ? 'none' : 'visible');
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    config = read_json(CONFIGS / "method_config.json")
    summary = read_json(REPORTS / "summary.json")
    thumbnails = make_thumbnails()
    viewer = make_webviewer_asset()
    sensitivity = None if WEBSITE_VARIANT == "lite" else make_dtm_sensitivity_assets()
    lite_full_comparison = make_lite_full_comparison_assets() if WEBSITE_VARIANT == "lite" else None
    lulc_report = load_lulc_method_report()
    active_dtm = FINAL_DTM
    stats = {
        "lulc": raster_stats(THEME_RASTERS["lulc"], discrete=True, mask_path=active_dtm),
        "slope": raster_stats(THEME_RASTERS["slope"], mask_path=active_dtm),
        "geology": raster_stats(THEME_RASTERS["geology"], mask_path=active_dtm),
        "pedology": raster_stats(THEME_RASTERS["pedology"], mask_path=active_dtm),
        "geomorphology": raster_stats(THEME_RASTERS["geomorphology"], mask_path=active_dtm),
        "pluviosity": raster_stats(THEME_RASTERS["pluviosity"], mask_path=active_dtm),
        "pluviosity_mm": raster_stats(THEME_RASTERS["pluviosity_mm"], mask_path=active_dtm),
        "class5": raster_stats(THEME_RASTERS["class5"], discrete=True),
        "valid": raster_stats(THEME_RASTERS["valid"], discrete=True, include_nodata=True),
    }
    report_data = {
        "config": config,
        "summary": summary,
        "thumbnails": thumbnails,
        "webviewer": viewer,
        "dtm_sensitivity": sensitivity,
        "lite_full_comparison": lite_full_comparison,
        "website_variant": WEBSITE_VARIANT,
        "lulc_method": lulc_report,
        "stats": stats,
    }
    (ASSETS / "report_data.json").write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
    (ASSETS / "lulc_method_report.json").write_text(
        json.dumps(lulc_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (WEBSITE / "descricao_completa_metodo.md").write_text(
        complete_method_markdown(config, summary, stats),
        encoding="utf-8",
    )
    (WEBSITE / "descricao_completa_metodo_lulc.md").write_text(
        str(lulc_report["markdown"]),
        encoding="utf-8",
    )
    (WEBSITE / "index.html").write_text(
        build_html(config, summary, thumbnails, viewer, stats, sensitivity, lulc_report, lite_full_comparison),
        encoding="utf-8",
    )
    print(f"[ibge-website] Wrote {WEBSITE / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
