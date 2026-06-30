"""Run and compare an IBGE adapted susceptibility experiment with an alternative DTM."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

from src.three_method_comparison import (
    IBGE_ADAPTED_TILE_SIZE,
    MethodDirs,
    ROOT,
    generate_ibge_method,
    iter_grid_windows,
    output_profile,
)


FINAL_OUTPUTS = ROOT / "IBGE_method" / "outputs"
DEFAULT_ALT_DTM = Path("/home/kaue/data/landslide/DTM_OTJC_3_1_16cm.tif")
DEFAULT_FINAL_DTM = Path("/home/kaue/data/landslide/dtm_final.tif")
DEFAULT_LULC = ROOT / "IBGE_method" / "own_LULC" / "outputs_fullres" / "lulc_custom_ensemble_16cm.tif"
DEFAULT_EXPERIMENT_ROOT = ROOT / "IBGE_method" / "dtm_sensitivity" / "DTM_OTJC_3_1_16cm"

SCORE_NAME = "ibge_susceptibility_score_1to10.tif"
VALID_NAME = "ibge_valid_mask.tif"
CLASS5_NAME = "ibge_class_map_5class.tif"
DIFF_NODATA = -9999.0
DTM_DIFF_NAME = "dtm_difference_alt_minus_final.tif"
DTM_DIFF_PREVIEW_NAME = "dtm_difference_gray.png"


@contextmanager
def temporary_env(key: str, value: str) -> Iterable[None]:
    previous = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_csv_rows(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def method_dirs_for(root: Path) -> MethodDirs:
    dirs = MethodDirs(root=root, outputs=root / "outputs", configs=root / "configs", reports=root / "reports")
    for path in (dirs.root, dirs.outputs, dirs.configs, dirs.reports):
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def same_transform(left: rasterio.Affine, right: rasterio.Affine, *, tolerance: float = 1e-9) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(tuple(left), tuple(right)))


def assert_matching_grid(reference: rasterio.DatasetReader, candidate: rasterio.DatasetReader, label: str) -> None:
    problems = []
    if reference.crs != candidate.crs:
        problems.append(f"CRS differs: {reference.crs} != {candidate.crs}")
    if reference.width != candidate.width or reference.height != candidate.height:
        problems.append(
            f"shape differs: {reference.width}x{reference.height} != {candidate.width}x{candidate.height}"
        )
    if not same_transform(reference.transform, candidate.transform):
        problems.append(f"transform differs: {reference.transform} != {candidate.transform}")
    if problems:
        joined = "; ".join(problems)
        raise RuntimeError(f"{label} is not on the final-product pixel grid: {joined}")


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def generate_alternative_product(
    *,
    alt_dtm: Path,
    lulc_path: Path,
    experiment_root: Path,
    force: bool,
) -> Dict[str, Any]:
    dirs = method_dirs_for(experiment_root)
    required = [dirs.outputs / SCORE_NAME, dirs.outputs / VALID_NAME, dirs.outputs / CLASS5_NAME]
    if not force and all(path.exists() for path in required):
        print(
            "[ibge-dtm-sensitivity] Alternative IBGE product already exists; "
            "use --force to regenerate it.",
            flush=True,
        )
        summary_path = dirs.reports / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        return {"outputs": [str(path) for path in required], "skipped_existing": True}

    with temporary_env("IBGE_CUSTOM_LULC_PATH", str(lulc_path)):
        with rasterio.open(alt_dtm) as reference:
            print(
                "[ibge-dtm-sensitivity] Generating alternative IBGE product from "
                f"{alt_dtm}",
                flush=True,
            )
            return generate_ibge_method(reference, dirs)


def finite_valid_scores(score: np.ndarray, valid: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    mask = valid == 1
    if nodata is not None and not math.isnan(float(nodata)):
        mask &= score != nodata
    return mask & np.isfinite(score)


def finite_valid_raster(values: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    mask = np.isfinite(values)
    if nodata is not None and not math.isnan(float(nodata)):
        mask &= values != nodata
    return mask


def descriptive_stats(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        raise RuntimeError("No valid pixels were available for the comparison.")
    percentiles = np.percentile(values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    abs_values = np.abs(values)
    thresholds = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    return {
        "count": int(values.size),
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
        "mean_absolute_error": float(np.mean(abs_values)),
        "median_absolute_difference": float(np.median(abs_values)),
        "rmse": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
        "positive_pixels": int(np.count_nonzero(values > 0)),
        "negative_pixels": int(np.count_nonzero(values < 0)),
        "zero_pixels_exact": int(np.count_nonzero(values == 0)),
        "positive_fraction": float(np.count_nonzero(values > 0) / values.size),
        "negative_fraction": float(np.count_nonzero(values < 0) / values.size),
        "within_abs_threshold_fraction": {
            f"{threshold:.2f}": float(np.count_nonzero(abs_values <= threshold) / values.size)
            for threshold in thresholds
        },
    }


def histogram_rows(values: np.ndarray, *, bins: int = 100) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    counts, edges = np.histogram(values, bins=bins)
    total = int(values.size)
    rows = []
    for idx, count in enumerate(counts):
        rows.append(
            {
                "bin_left": float(edges[idx]),
                "bin_right": float(edges[idx + 1]),
                "count": int(count),
                "fraction": float(count / total) if total else 0.0,
            }
        )
    return counts, edges, rows


def write_histogram_png(path: Path, values: np.ndarray, stats: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.hist(values, bins=100, color="#365C8D", edgecolor="white", linewidth=0.25)
    ax.axvline(0, color="#222222", linewidth=1.2, label="Sem diferença")
    ax.axvline(stats["mean"], color="#FDE725", linewidth=1.5, label=f"Média {stats['mean']:.4f}")
    ax.axvline(stats["median"], color="#35B779", linewidth=1.5, label=f"Mediana {stats['median']:.4f}")
    ax.set_title("Diferença do score IBGE 1-10: DTM alternativo - produto final")
    ax.set_xlabel("Diferença no score 1-10")
    ax.set_ylabel("Pixels válidos")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_centered_grayscale_preview(path: Path, raster_path: Path, stats: Mapping[str, Any]) -> None:
    with rasterio.open(raster_path) as src:
        scale = max(src.width / 1800, src.height / 1800, 1.0)
        width = max(1, int(round(src.width / scale)))
        height = max(1, int(round(src.height / scale)))
        data = src.read(1, out_shape=(height, width), masked=True, resampling=Resampling.bilinear)

    arr = np.asarray(data.astype(np.float32).filled(np.nan), dtype=np.float32)
    valid = ~np.ma.getmaskarray(data) & np.isfinite(arr)
    limit = max(abs(float(stats["min"])), abs(float(stats["max"])))
    if math.isclose(limit, 0.0):
        limit = 1.0
    scaled = np.clip((arr + limit) / (2.0 * limit), 0.0, 1.0)
    gray = (np.nan_to_num(scaled, nan=0.5) * 255).astype(np.uint8)
    rgb = np.dstack([gray, gray, gray])
    rgb[~valid] = np.array([235, 238, 241], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, "PNG", optimize=True)


def class_transition_rows(counter: Counter[tuple[int, int]]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    rows = []
    for final_class in range(1, 6):
        for alt_class in range(1, 6):
            count = int(counter[(final_class, alt_class)])
            rows.append(
                {
                    "final_class": final_class,
                    "alternative_class": alt_class,
                    "count": count,
                    "fraction": float(count / total) if total else 0.0,
                }
            )
    return rows


def compare_products(
    *,
    final_outputs: Path,
    experiment_root: Path,
    alt_dtm: Path,
    final_dtm: Path,
    lulc_path: Path,
    force: bool,
) -> Dict[str, Any]:
    final_score_path = require_file(final_outputs / SCORE_NAME, "final IBGE score")
    final_valid_path = require_file(final_outputs / VALID_NAME, "final IBGE valid mask")
    final_class5_path = require_file(final_outputs / CLASS5_NAME, "final IBGE 5-class map")
    alt_score_path = require_file(experiment_root / "outputs" / SCORE_NAME, "alternative IBGE score")
    alt_valid_path = require_file(experiment_root / "outputs" / VALID_NAME, "alternative IBGE valid mask")
    alt_class5_path = require_file(experiment_root / "outputs" / CLASS5_NAME, "alternative IBGE 5-class map")

    comparison_dir = experiment_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    diff_path = comparison_dir / "score_difference_alt_minus_final.tif"
    dtm_diff_path = comparison_dir / DTM_DIFF_NAME
    dtm_preview_path = comparison_dir / DTM_DIFF_PREVIEW_NAME
    mask_path = comparison_dir / "valid_comparison_mask.tif"
    stats_path = comparison_dir / "difference_stats.json"
    if (
        not force
        and diff_path.exists()
        and dtm_diff_path.exists()
        and dtm_preview_path.exists()
        and mask_path.exists()
        and stats_path.exists()
    ):
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        if "dtm_difference_stats" in payload:
            print("[ibge-dtm-sensitivity] Existing complete comparison found; use --force to recompute.", flush=True)
            return payload

    diff_chunks: list[np.ndarray] = []
    dtm_diff_chunks: list[np.ndarray] = []
    transition_counter: Counter[tuple[int, int]] = Counter()
    final_valid_pixels = 0
    alt_valid_pixels = 0
    intersection_pixels = 0
    final_only_pixels = 0
    alt_only_pixels = 0
    dtm_common_pixels = 0

    with (
        rasterio.open(final_score_path) as final_score,
        rasterio.open(alt_score_path) as alt_score,
        rasterio.open(final_valid_path) as final_valid,
        rasterio.open(alt_valid_path) as alt_valid,
        rasterio.open(final_class5_path) as final_class5,
        rasterio.open(alt_class5_path) as alt_class5,
        rasterio.open(final_dtm) as final_dtm_src,
        rasterio.open(alt_dtm) as alt_dtm_src,
    ):
        for label, dataset in (
            ("alternative score", alt_score),
            ("final valid mask", final_valid),
            ("alternative valid mask", alt_valid),
            ("final 5-class map", final_class5),
            ("alternative 5-class map", alt_class5),
            ("final DTM", final_dtm_src),
            ("alternative DTM", alt_dtm_src),
        ):
            assert_matching_grid(final_score, dataset, label)

        diff_profile = output_profile(final_score, count=1, dtype="float32", nodata=DIFF_NODATA)
        mask_profile = output_profile(final_score, count=1, dtype="uint8", nodata=0)

        with (
            rasterio.open(diff_path, "w", **diff_profile) as diff_dst,
            rasterio.open(dtm_diff_path, "w", **diff_profile) as dtm_diff_dst,
            rasterio.open(mask_path, "w", **mask_profile) as mask_dst,
        ):
            diff_dst.set_band_description(1, "Alternative IBGE score minus final IBGE score on valid common pixels")
            dtm_diff_dst.set_band_description(
                1, "Alternative DTM elevation minus final DTM elevation on common valid DTM pixels"
            )
            mask_dst.set_band_description(1, "Pixels valid in both final and alternative IBGE products")
            total_tiles = math.ceil(final_score.width / IBGE_ADAPTED_TILE_SIZE) * math.ceil(
                final_score.height / IBGE_ADAPTED_TILE_SIZE
            )
            for idx, window in enumerate(iter_grid_windows(final_score, IBGE_ADAPTED_TILE_SIZE), start=1):
                if idx == 1 or idx == total_tiles or idx % 20 == 0:
                    print(f"[ibge-dtm-sensitivity] Comparing tile {idx}/{total_tiles}", flush=True)

                final_score_arr = final_score.read(1, window=window)
                alt_score_arr = alt_score.read(1, window=window)
                final_valid_arr = final_valid.read(1, window=window)
                alt_valid_arr = alt_valid.read(1, window=window)
                final_dtm_arr = final_dtm_src.read(1, window=window)
                alt_dtm_arr = alt_dtm_src.read(1, window=window)

                final_mask = finite_valid_scores(final_score_arr, final_valid_arr, final_score.nodata)
                alt_mask = finite_valid_scores(alt_score_arr, alt_valid_arr, alt_score.nodata)
                common = final_mask & alt_mask
                dtm_common = (
                    finite_valid_raster(final_dtm_arr, final_dtm_src.nodata)
                    & finite_valid_raster(alt_dtm_arr, alt_dtm_src.nodata)
                )

                final_valid_pixels += int(np.count_nonzero(final_mask))
                alt_valid_pixels += int(np.count_nonzero(alt_mask))
                intersection_pixels += int(np.count_nonzero(common))
                final_only_pixels += int(np.count_nonzero(final_mask & ~alt_mask))
                alt_only_pixels += int(np.count_nonzero(alt_mask & ~final_mask))
                dtm_common_pixels += int(np.count_nonzero(dtm_common))

                diff = np.full(final_score_arr.shape, DIFF_NODATA, dtype=np.float32)
                if np.any(common):
                    valid_diff = (alt_score_arr[common] - final_score_arr[common]).astype(np.float32)
                    diff[common] = valid_diff
                    diff_chunks.append(valid_diff)

                    final_cls = final_class5.read(1, window=window)[common]
                    alt_cls = alt_class5.read(1, window=window)[common]
                    for final_value, alt_value in zip(final_cls, alt_cls):
                        if int(final_value) != 255 and int(alt_value) != 255:
                            transition_counter[(int(final_value), int(alt_value))] += 1

                dtm_diff = np.full(final_dtm_arr.shape, DIFF_NODATA, dtype=np.float32)
                if np.any(dtm_common):
                    valid_dtm_diff = (alt_dtm_arr[dtm_common] - final_dtm_arr[dtm_common]).astype(np.float32)
                    dtm_diff[dtm_common] = valid_dtm_diff
                    dtm_diff_chunks.append(valid_dtm_diff)

                diff_dst.write(diff, 1, window=window)
                dtm_diff_dst.write(dtm_diff, 1, window=window)
                mask_dst.write(common.astype("uint8"), 1, window=window)

        pixel_area = abs(float(final_score.transform.a * final_score.transform.e))
        reference_grid = {
            "crs": str(final_score.crs),
            "width": int(final_score.width),
            "height": int(final_score.height),
            "resolution": [float(final_score.res[0]), float(final_score.res[1])],
            "bounds": [float(v) for v in final_score.bounds],
            "pixel_area_m2": pixel_area,
        }

    differences = np.concatenate(diff_chunks) if diff_chunks else np.array([], dtype=np.float32)
    dtm_differences = np.concatenate(dtm_diff_chunks) if dtm_diff_chunks else np.array([], dtype=np.float32)
    stats = descriptive_stats(differences)
    dtm_stats = descriptive_stats(dtm_differences)
    _, _, hist_rows = histogram_rows(differences, bins=100)
    _, _, dtm_hist_rows = histogram_rows(dtm_differences, bins=100)
    hist_csv = comparison_dir / "difference_histogram.csv"
    hist_png = comparison_dir / "difference_histogram.png"
    stats_csv = comparison_dir / "difference_stats.csv"
    dtm_stats_csv = comparison_dir / "dtm_difference_stats.csv"
    dtm_hist_csv = comparison_dir / "dtm_difference_histogram.csv"
    transitions_csv = comparison_dir / "class5_transition_matrix.csv"
    report_path = comparison_dir / "comparison_report.md"

    class_rows = class_transition_rows(transition_counter)
    same_class_pixels = sum(row["count"] for row in class_rows if row["final_class"] == row["alternative_class"])
    upward_class_pixels = sum(row["count"] for row in class_rows if row["alternative_class"] > row["final_class"])
    downward_class_pixels = sum(row["count"] for row in class_rows if row["alternative_class"] < row["final_class"])

    payload: Dict[str, Any] = {
        "experiment": "IBGE DTM sensitivity: DTM_OTJC_3_1_16cm systematic-error DTM",
        "difference_definition": "alternative_score_1to10_minus_final_score_1to10",
        "dtm_difference_definition": "alternative_dtm_minus_final_dtm",
        "final_dtm": str(final_dtm),
        "alternative_dtm": str(alt_dtm),
        "lulc": str(lulc_path),
        "final_score": str(final_score_path),
        "alternative_score": str(alt_score_path),
        "difference_raster": str(diff_path),
        "dtm_difference_raster": str(dtm_diff_path),
        "dtm_difference_preview": str(dtm_preview_path),
        "valid_comparison_mask": str(mask_path),
        "histogram_png": str(hist_png),
        "histogram_csv": str(hist_csv),
        "dtm_difference_stats_csv": str(dtm_stats_csv),
        "dtm_difference_histogram_csv": str(dtm_hist_csv),
        "class5_transition_csv": str(transitions_csv),
        "reference_grid": reference_grid,
        "valid_pixel_accounting": {
            "final_valid_pixels": int(final_valid_pixels),
            "alternative_valid_pixels": int(alt_valid_pixels),
            "common_valid_pixels": int(intersection_pixels),
            "final_only_valid_pixels": int(final_only_pixels),
            "alternative_only_valid_pixels": int(alt_only_pixels),
            "common_valid_area_m2": float(intersection_pixels * pixel_area),
            "dtm_common_valid_pixels": int(dtm_common_pixels),
            "dtm_common_valid_area_m2": float(dtm_common_pixels * pixel_area),
        },
        "score_difference_stats": stats,
        "dtm_difference_stats": dtm_stats,
        "class5_transition": {
            "same_class_pixels": int(same_class_pixels),
            "upward_class_pixels": int(upward_class_pixels),
            "downward_class_pixels": int(downward_class_pixels),
            "same_class_fraction": float(same_class_pixels / intersection_pixels) if intersection_pixels else 0.0,
            "upward_class_fraction": float(upward_class_pixels / intersection_pixels) if intersection_pixels else 0.0,
            "downward_class_fraction": float(downward_class_pixels / intersection_pixels) if intersection_pixels else 0.0,
        },
    }

    write_json(stats_path, payload)
    write_csv_rows(hist_csv, hist_rows, ["bin_left", "bin_right", "count", "fraction"])
    write_csv_rows(dtm_hist_csv, dtm_hist_rows, ["bin_left", "bin_right", "count", "fraction"])
    write_histogram_png(hist_png, differences, stats)
    write_centered_grayscale_preview(dtm_preview_path, dtm_diff_path, dtm_stats)
    write_csv_rows(
        stats_csv,
        [
            {"metric": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value}
            for key, value in stats.items()
        ],
        ["metric", "value"],
    )
    write_csv_rows(
        dtm_stats_csv,
        [
            {"metric": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value}
            for key, value in dtm_stats.items()
        ],
        ["metric", "value"],
    )
    write_csv_rows(transitions_csv, class_rows, ["final_class", "alternative_class", "count", "fraction"])
    write_report(report_path, payload)
    return payload


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def fmt_int(value: int) -> str:
    return f"{int(value):,}".replace(",", ".")


def fmt_area(value: float) -> str:
    return f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    stats = payload["score_difference_stats"]
    dtm_stats = payload["dtm_difference_stats"]
    valid = payload["valid_pixel_accounting"]
    transition = payload["class5_transition"]
    content = "\n".join(
        [
            "# Experimento de sensibilidade ao DTM",
            "",
            "Este experimento compara o produto IBGE final com uma versão alternativa gerada a partir do DTM `DTM_OTJC_3_1_16cm.tif`, descrito como afetado por erros sistemáticos. O produto oficial não foi regenerado; ele foi usado apenas como referência de comparação.",
            "",
            "A diferença dos scores foi calculada como `score alternativo - score final`. A diferença de elevação foi calculada como `DTM alternativo - DTM final`.",
            "",
            "Valores positivos indicam aumento no produto derivado do DTM alternativo. Valores negativos indicam redução.",
            "",
            "## Entradas",
            "",
            f"- DTM final: `{payload['final_dtm']}`.",
            f"- DTM alternativo: `{payload['alternative_dtm']}`.",
            f"- LULC mantido: `{payload['lulc']}`.",
            f"- Score final: `{payload['final_score']}`.",
            f"- Score alternativo: `{payload['alternative_score']}`.",
            "",
            "## Pixels comparados",
            "",
            f"- Pixels válidos no produto final: {fmt_int(valid['final_valid_pixels'])}.",
            f"- Pixels válidos no produto alternativo: {fmt_int(valid['alternative_valid_pixels'])}.",
            f"- Pixels válidos comuns usados na comparação: {fmt_int(valid['common_valid_pixels'])}.",
            f"- Área comum comparada: {fmt_area(valid['common_valid_area_m2'])} m².",
            f"- Pixels válidos somente no produto final: {fmt_int(valid['final_only_valid_pixels'])}.",
            f"- Pixels válidos somente no produto alternativo: {fmt_int(valid['alternative_only_valid_pixels'])}.",
            f"- Pixels com DTM válido comum: {fmt_int(valid['dtm_common_valid_pixels'])}.",
            "",
            "## Estatísticas da diferença de score",
            "",
            f"- Média: {fmt(stats['mean'])}.",
            f"- Mediana: {fmt(stats['median'])}.",
            f"- Desvio padrão: {fmt(stats['std'])}.",
            f"- Mínimo: {fmt(stats['min'])}.",
            f"- Máximo: {fmt(stats['max'])}.",
            f"- P05: {fmt(stats['p05'])}.",
            f"- P25: {fmt(stats['p25'])}.",
            f"- P75: {fmt(stats['p75'])}.",
            f"- P95: {fmt(stats['p95'])}.",
            f"- Erro absoluto médio: {fmt(stats['mean_absolute_error'])}.",
            f"- RMSE: {fmt(stats['rmse'])}.",
            f"- Fração com aumento do score: {fmt(100 * stats['positive_fraction'], 2)}%.",
            f"- Fração com redução do score: {fmt(100 * stats['negative_fraction'], 2)}%.",
            "",
            "## Estatísticas da diferença entre DTMs",
            "",
            f"- Média: {fmt(dtm_stats['mean'])} m.",
            f"- Mediana: {fmt(dtm_stats['median'])} m.",
            f"- Desvio padrão: {fmt(dtm_stats['std'])} m.",
            f"- Mínimo: {fmt(dtm_stats['min'])} m.",
            f"- Máximo: {fmt(dtm_stats['max'])} m.",
            f"- P05: {fmt(dtm_stats['p05'])} m.",
            f"- P25: {fmt(dtm_stats['p25'])} m.",
            f"- P75: {fmt(dtm_stats['p75'])} m.",
            f"- P95: {fmt(dtm_stats['p95'])} m.",
            f"- Erro absoluto médio: {fmt(dtm_stats['mean_absolute_error'])} m.",
            f"- RMSE: {fmt(dtm_stats['rmse'])} m.",
            f"- Fração com DTM alternativo mais alto: {fmt(100 * dtm_stats['positive_fraction'], 2)}%.",
            f"- Fração com DTM alternativo mais baixo: {fmt(100 * dtm_stats['negative_fraction'], 2)}%.",
            "",
            "## Mudança de classe final",
            "",
            f"- Mesma classe 5 níveis: {fmt_int(transition['same_class_pixels'])} pixels, {fmt(100 * transition['same_class_fraction'], 2)}%.",
            f"- Classe aumentou no produto alternativo: {fmt_int(transition['upward_class_pixels'])} pixels, {fmt(100 * transition['upward_class_fraction'], 2)}%.",
            f"- Classe diminuiu no produto alternativo: {fmt_int(transition['downward_class_pixels'])} pixels, {fmt(100 * transition['downward_class_fraction'], 2)}%.",
            "",
            "## Artefatos",
            "",
            f"- Raster de diferença de score: `{payload['difference_raster']}`.",
            f"- Raster de diferença entre DTMs: `{payload['dtm_difference_raster']}`.",
            f"- Prévia da diferença entre DTMs: `{payload['dtm_difference_preview']}`.",
            f"- Máscara de comparação: `{payload['valid_comparison_mask']}`.",
            f"- Histograma PNG: `{payload['histogram_png']}`.",
            f"- Histograma CSV: `{payload['histogram_csv']}`.",
            f"- Estatísticas CSV da diferença entre DTMs: `{payload['dtm_difference_stats_csv']}`.",
            f"- Histograma CSV da diferença entre DTMs: `{payload['dtm_difference_histogram_csv']}`.",
            f"- Matriz de transição de classes: `{payload['class5_transition_csv']}`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an alternative IBGE adapted product from a systematic-error DTM and compare it to the final product."
    )
    parser.add_argument("--alt-dtm", type=Path, default=DEFAULT_ALT_DTM)
    parser.add_argument("--final-dtm", type=Path, default=DEFAULT_FINAL_DTM)
    parser.add_argument("--lulc", type=Path, default=DEFAULT_LULC)
    parser.add_argument("--final-outputs", type=Path, default=FINAL_OUTPUTS)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Do not generate the alternative product; compare existing outputs only.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate the alternative product and comparison artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alt_dtm = require_file(args.alt_dtm, "alternative DTM")
    final_dtm = require_file(args.final_dtm, "final DTM")
    lulc_path = require_file(args.lulc, "custom full-resolution LULC")
    final_outputs = args.final_outputs.expanduser().resolve()
    experiment_root = args.experiment_root.expanduser().resolve()

    with rasterio.open(final_outputs / SCORE_NAME) as final_score, rasterio.open(alt_dtm) as alt_src:
        assert_matching_grid(final_score, alt_src, "alternative DTM")
    with rasterio.open(final_outputs / SCORE_NAME) as final_score, rasterio.open(final_dtm) as final_src:
        assert_matching_grid(final_score, final_src, "final DTM")

    if not args.comparison_only:
        generate_alternative_product(
            alt_dtm=alt_dtm,
            lulc_path=lulc_path,
            experiment_root=experiment_root,
            force=args.force,
        )

    payload = compare_products(
        final_outputs=final_outputs,
        experiment_root=experiment_root,
        alt_dtm=alt_dtm,
        final_dtm=final_dtm,
        lulc_path=lulc_path,
        force=args.force,
    )
    stats = payload["score_difference_stats"]
    dtm_stats = payload["dtm_difference_stats"]
    print(
        "[ibge-dtm-sensitivity] Done. "
        f"Compared {stats['count']:,} valid score pixels; "
        f"score mean diff={stats['mean']:.6f}; "
        f"score MAE={stats['mean_absolute_error']:.6f}; "
        f"DTM mean diff={dtm_stats['mean']:.6f} m.",
        flush=True,
    )


if __name__ == "__main__":
    main()
