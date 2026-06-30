"""Build the Portuguese static report for the IBGE high-resolution product."""

from __future__ import annotations

import html
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
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


OUTPUTS = ROOT / "IBGE_method" / "outputs"
REPORTS = ROOT / "IBGE_method" / "reports"
CONFIGS = ROOT / "IBGE_method" / "configs"
WEBSITE = ROOT / "IBGE_method" / "website"
ASSETS = WEBSITE / "assets"
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
    if selected and selected.get("selection_type") == "ensemble":
        ensemble = selected

    runs = list(sweep.get("runs", []))
    voters_raw: Sequence[Mapping[str, Any]] = []
    if ensemble and ensemble.get("voters"):
        voters_raw = list(ensemble["voters"])
    elif ensemble and ensemble.get("voter_run_ids"):
        ids = {str(run_id) for run_id in ensemble["voter_run_ids"]}
        voters_raw = [row for row in runs if str(row.get("run_id")) in ids]
    else:
        voters_raw = [row for row in runs if float(row.get("resolution", 0.0) or 0.0) == 0.16][:5]

    voters = [normalized_lulc_run(row) for row in voters_raw]
    for index, row in enumerate(voters, start=1):
        row["model_order"] = index
        row["display_name"] = architecture_label(row)

    fullres_experiments = {
        name: payload
        for name, payload in lulc_inputs.experiment_grid.items()
        if name.startswith("fullres_")
    }
    metrics_available = bool(ensemble and ensemble.get("metrics")) and all(
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
        "metrics_available": metrics_available,
        "metrics_note": (
            "Métricas lidas dos artefatos locais em outputs_fullres/."
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
        "voters": voters,
        "ensemble": ensemble or {},
        "selected": selected or {},
        "sweep_run_count": len(runs),
    }
    report["markdown"] = complete_lulc_method_markdown(report)
    return report


def complete_lulc_method_markdown(report: Mapping[str, Any]) -> str:
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
        voter_lines.extend(
            [
                f"- Modelo {row['model_order']}: {row['display_name']}.",
                f"- Identificador: `{row.get('run_id') or 'pendente'}`.",
                f"- Configuração: resolução {fmt_optional_number(row.get('resolution'), 2)} m; feature set `{row.get('feature_set') or 'pendente'}`; perda `{row.get('loss') or 'pendente'}`; tile {row.get('tile_size') or 'pendente'}; stride {row.get('stride') or 'pendente'}; seed {row.get('seed') or 'pendente'}.",
                f"- Treinamento: {row.get('epochs') or 'pendente'} épocas planejadas/executadas; melhor época {row.get('best_epoch') or 'pendente'}.",
                f"- Validação: macro IoU {fmt_optional_number(row.get('val_macro_iou'))}, macro F1 {fmt_optional_number(row.get('val_macro_f1'))}, acurácia global {fmt_optional_number(row.get('val_overall_accuracy'))}.",
                f"- Teste final: macro IoU {fmt_optional_number(row.get('test_macro_iou'))}, macro F1 {fmt_optional_number(row.get('test_macro_f1'))}, acurácia global {fmt_optional_number(row.get('test_overall_accuracy'))}.",
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
        "A implementação está versionada no repositório, principalmente em `IBGE_method/own_LULC/lulc_inputs.py` e nos módulos de `IBGE_method/own_LULC/implementation/`. O arquivo `lulc_inputs.py` é a superfície única de hiperparâmetros: entradas, classes, resolução, divisão espacial, modelos, perdas, treino, inferência e ensemble ficam declarados ali.",
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
        "As classes usadas no raster final são os códigos originais 1 a 5. Internamente, durante o treinamento, elas são convertidas para índices 0 a 4 porque a função de perda multiclasse espera classes contíguas. Na escrita do GeoTIFF final, a codificação volta para 1 a 5.",
        "",
        *class_lines,
        "",
        "## Conjuntos de variáveis",
        "",
        "Foram previstos dois conjuntos de atributos derivados apenas da ortofoto, sem insumos externos adicionais:",
        "",
        *feature_lines,
        "",
        "O conjunto `rgb_indices` amplia o RGB com HSV, brilho, excesso de verde e textura local. Ele foi incluído para ajudar a separar vegetação, solo exposto, água e superfícies artificiais quando a assinatura espectral RGB pura é ambígua.",
        "",
        "## Divisão espacial e desenho de avaliação",
        "",
        f"- Estratégia de split: `{report['split_constraints']['strategy']}`.",
        f"- Percentuais por classe: treino {fmt_percent(params['train_percentage'])}, validação {fmt_percent(params['validation_percentage'])}, teste {fmt_percent(params['test_percentage'])}.",
        f"- Exigência de todas as classes em cada split: `{report['split_constraints']['require_all_classes_per_split']}`.",
        f"- Tentativas máximas de seeds para achar split viável: `{report['split_constraints']['max_seed_attempts']}`.",
        "",
        "A divisão não é uma amostragem aleatória simples de pixels. Ela usa blocos espaciais para reduzir vazamento espacial entre treino, validação e teste. Além disso, as metas de proporção são aplicadas por classe: a ideia é que cada classe tenha suporte próprio em treino, validação e teste, evitando avaliações artificialmente boas ou ruins por ausência de uma classe minoritária.",
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
        "## Inferência full-resolution",
        "",
        f"- Janela de inferência: `{inference['window_size']}` pixels.",
        f"- Sobreposição: `{inference['overlap']}` pixels.",
        f"- Batch de inferência: `{inference['batch_size']}`.",
        f"- CUDA habilitado quando disponível: `{inference['use_cuda']}`.",
        "",
        "A inferência percorre a ortofoto em janelas sobrepostas. Para cada pixel, as probabilidades acumuladas nas janelas que o cobrem são médias antes da decisão final. O resultado individual de cada modelo inclui um raster de classes e um raster multibanda de probabilidades, com cinco bandas, uma por classe.",
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


def lulc_method_section(report: Mapping[str, Any], lulc_thumbnail: str) -> str:
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
    sensitivity: Mapping[str, Any],
    lulc_report: Mapping[str, Any],
) -> str:
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
        ("lulc_method_description", "Descrição completa do método LULC"),
        ("dtm_sensitivity", "Comparação com DTM tendencioso"),
        ("webviewer", "webviewer"),
    ]
    nav = "".join(
        f"<button class=\"tab-button{' active' if idx == 0 else ''}\" data-tab=\"{tab_id}\">{label}</button>"
        for idx, (tab_id, label) in enumerate(tabs)
    )

    tab_content = {
        "overview": section(
            "Produto IBGE adaptado em 16 cm",
            thumbs["overview"],
            (
                "<p>Este relatório documenta a geração do produto IBGE adaptado de alta resolução. "
                "A grade estatística nacional de 1 km foi substituída pelo grid do DTM final em 16 cm; "
                "os pesos, a escala de notas e as classes finais da metodologia IBGE foram preservados.</p>"
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
            "<p>Fonte: ensemble LULC full-res gerado por deep learning. A classe de água recebe nota 0 e é excluída dos pixels válidos.</p>",
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
        "dtm_sensitivity": dtm_sensitivity_section(sensitivity),
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
  <title>Produto IBGE Adaptado 16 cm</title>
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
    #webviewer, #method_description, #lulc_method_description, #dtm_sensitivity {{ grid-column:1 / -1; background:transparent; border:0; padding:0; }}
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
    <h1>Produto IBGE adaptado em grade de 16 cm</h1>
    <p>Relatório técnico em português da geração do produto final: entradas, proxies, mapeamentos, álgebra e visualização interativa do score de suscetibilidade.</p>
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
    sensitivity = make_dtm_sensitivity_assets()
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
        build_html(config, summary, thumbnails, viewer, stats, sensitivity, lulc_report),
        encoding="utf-8",
    )
    print(f"[ibge-website] Wrote {WEBSITE / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
