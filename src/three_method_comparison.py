"""Generate DL, IBGE-adapted, and SGB-style susceptibility outputs on one drone grid."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]
SIG_ROOT = Path(
    "/home/kaue/Downloads/05_Geospatial_Data_Maps/"
    "sig_cachoeirodoitapemirim_es_suscet"
)
DATA_ROOT = SIG_ROOT / "data_interest_area"
DRONE_DTM = Path("/home/kaue/data/landslide/feb26/DTM_4_GNSS-AAT_16cm.tif")
DRONE_ORTHO = Path("/home/kaue/data/landslide/feb26/Ortho_4_GNSS-AAT_16cm.tif")
GROUND_TRUTH_25M = Path("/home/kaue/data/landslide/test/Ground_truth_test.tif")

MAPBIOMAS_10M_2023_URL = (
    "https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/"
    "lulc_10m/collection_2/integration/"
    "mapbiomas_10m_collection2_integration_v1-classification_2023.tif"
)
CUSTOM_LULC_OUTPUT = ROOT / "IBGE_method" / "own_LULC" / "outputs" / "lulc_custom_10m.tif"

DL_SOURCE_OUTPUTS = {
    "susceptibility": ROOT / "outputs" / "test_susceptibility.tif",
    "susceptibility_high": ROOT / "outputs" / "test_susceptibility_high.tif",
    "class_probabilities": ROOT / "outputs" / "test_class_probabilities.tif",
    "uncertainty": ROOT / "outputs" / "test_uncertainty.tif",
    "class_map": ROOT / "outputs" / "test_class_map.tif",
    "valid_mask": ROOT / "outputs" / "test_valid_mask.tif",
}


@dataclass(frozen=True)
class MethodDirs:
    root: Path
    outputs: Path
    configs: Path
    reports: Path


def method_dirs(name: str) -> MethodDirs:
    root = ROOT / name
    return MethodDirs(root=root, outputs=root / "outputs", configs=root / "configs", reports=root / "reports")


def ensure_method_dirs() -> Dict[str, MethodDirs]:
    dirs = {
        "dl": method_dirs("DL_method"),
        "ibge": method_dirs("IBGE_method"),
        "sgb": method_dirs("SGB_method"),
    }
    for group in dirs.values():
        for path in (group.root, group.outputs, group.configs, group.reports):
            path.mkdir(parents=True, exist_ok=True)
    return dirs


def local_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


SIG_PATHS = {
    "relief": local_path("04.Padr\u00f5esDeRelevo", "Relevo.shp"),
    "features_poly": local_path("05.Fei\u00e7\u00f5es", "feicoes.shp"),
    "features_point": local_path("05.Fei\u00e7\u00f5es", "feicoes_pontuais.shp"),
    "geology": local_path("06.Geologia", "Geologia.shp"),
    "lineaments": local_path("06.Geologia", "Lineamentos.shp"),
    "pedology": local_path("07.Pedologia", "Pedologia.shp"),
    "rain_pma2": local_path(
        "08.Isoietas",
        "AtlasPluviometrico",
        "IsoietasAnuaisMedias",
        "pma2",
    ),
    "sgb_mass": local_path(
        "12.Suscetibilidade", "SuscetibilidadeMovimentoDeMassa.shp"
    ),
}


IBGE_WEIGHTS = {
    "slope": 0.35,
    "geomorphology": 0.20,
    "geology": 0.15,
    "pedology": 0.15,
    "land_use": 0.10,
    "pluviosity": 0.05,
}

RELIEF_IBGE_NOTES = {
    "Plan\u00edcies e terra\u00e7os fluviais": 1.0,
    "Colinas": 3.0,
    "Morrotes": 5.0,
    "Morrotes altos": 6.0,
    "Morros baixos": 7.0,
    "Morros altos": 9.0,
    "Serras": 10.0,
}

GEOLOGY_IBGE_NOTES = {
    "PRps": 7.3,
    "NP3a_gamma_1Iag": 6.75,
    "NP3a_gamma_1Ich": 6.75,
    "NP3a_gamma_1Imf": 6.75,
    "C_cortado_2a_gamma_5Isa": 6.0,
    "Q2a": 9.5,
    "Q2fl": 9.3,
}

PEDOLOGY_IBGE_NOTES = {
    "Ad2": 5.0,
    "BV1": 7.0,
    "BV2": 7.0,
    "Ca3": 9.0,
    "LVa2": 7.0,
    "PEe1": 8.0,
    "PEe3": 8.0,
    "PEe4": 8.0,
    "PEe11": 8.0,
    "Ra": 10.0,
    "Rde1": 10.0,
    "Rios": 0.0,
}

# Broad MapBiomas class-code mapping to the IBGE land-use/vegetation notes.
# Unknown nonzero classes are reported and treated as missing instead of filled.
MAPBIOMAS_IBGE_LAND_USE_NOTES = {
    3: 1.0,  # Forest formation
    4: 2.0,  # Savanna formation
    5: 1.0,  # Mangrove
    6: 1.0,  # Flooded forest / restinga arborea depending on collection
    9: 4.0,  # Silviculture
    11: 2.0,  # Wetland
    12: 2.0,  # Grassland
    15: 8.0,  # Pasture
    18: 9.0,  # Agriculture
    19: 9.0,
    20: 9.0,
    21: 6.0,  # Mosaic of uses
    24: 10.0,  # Urban area
    25: 5.0,  # Other non-vegetated area
    29: 5.0,  # Rocky outcrop
    30: 5.0,  # Mining
    32: 2.0,  # Salt flat / herbaceous restinga depending on collection
    33: 0.0,  # Water is excluded from the landslide score
    39: 9.0,
    40: 9.0,
    41: 9.0,
    46: 9.0,
    47: 9.0,
    48: 9.0,
    49: 9.0,
    50: 5.0,
    62: 9.0,
}

SGB_WEIGHTS = {
    "slope": 0.34,
    "curvature": 0.12,
    "drainage": 0.10,
    "lineament_density": 0.16,
    "relief": 0.13,
    "geology": 0.08,
    "pedology": 0.07,
}

RELIEF_SGB_SCORES = {
    "Plan\u00edcies e terra\u00e7os fluviais": 0.10,
    "Colinas": 0.30,
    "Morrotes": 0.50,
    "Morrotes altos": 0.65,
    "Morros baixos": 0.45,
    "Morros altos": 0.85,
    "Serras": 0.95,
}

PEDOLOGY_SGB_SCORES = {
    "Ad2": 0.35,
    "BV1": 0.55,
    "BV2": 0.55,
    "Ca3": 0.80,
    "LVa2": 0.40,
    "PEe1": 0.65,
    "PEe3": 0.65,
    "PEe4": 0.65,
    "PEe11": 0.65,
    "Ra": 0.95,
    "Rde1": 0.95,
    "Rios": 0.0,
}

GEOLOGY_SGB_SCORES = {
    "PRps": 0.75,
    "NP3a_gamma_1Iag": 0.60,
    "NP3a_gamma_1Ich": 0.60,
    "NP3a_gamma_1Imf": 0.60,
    "C_cortado_2a_gamma_5Isa": 0.55,
    "Q2a": 0.30,
    "Q2fl": 0.30,
}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def output_profile(
    reference: DatasetReader,
    *,
    count: int,
    dtype: str,
    nodata: Optional[float],
) -> Dict[str, Any]:
    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        count=count,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
        NUM_THREADS="ALL_CPUS",
    )
    if dtype.startswith("float"):
        profile["predictor"] = 2
    else:
        profile.pop("predictor", None)
    return profile


def write_single_band(
    path: Path,
    data: np.ndarray,
    reference: DatasetReader,
    *,
    dtype: str,
    nodata: Optional[float],
    description: Optional[str] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile(reference, count=1, dtype=dtype, nodata=nodata)) as dst:
        dst.write(data.astype(dtype), 1)
        if description:
            dst.set_band_description(1, description)


def write_multiband(
    path: Path,
    data: np.ndarray,
    reference: DatasetReader,
    *,
    dtype: str,
    nodata: Optional[float],
    descriptions: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile(reference, count=data.shape[0], dtype=dtype, nodata=nodata)) as dst:
        dst.write(data.astype(dtype))
        for idx, description in enumerate(descriptions, start=1):
            dst.set_band_description(idx, description)


def valid_mask_from_dtm(dtm: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    valid = np.isfinite(dtm)
    if nodata is not None:
        valid &= dtm != nodata
    return valid


def fill_invalid_nearest(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if bool(valid_mask.all()):
        return data.astype(np.float32, copy=False)
    if not bool(valid_mask.any()):
        raise RuntimeError("Reference DTM has no valid pixels.")
    invalid = ~valid_mask
    _, indices = distance_transform_edt(invalid, return_indices=True)
    filled = data.copy()
    filled[invalid] = data[tuple(indices[:, invalid])]
    return filled.astype(np.float32, copy=False)


def reference_valid_mask(reference: DatasetReader) -> np.ndarray:
    dtm = reference.read(1).astype(np.float32)
    return valid_mask_from_dtm(dtm, reference.nodata)


def reproject_dtm_to_resolution(
    reference: DatasetReader, resolution_m: float
) -> Tuple[np.ndarray, Dict[str, Any]]:
    profile = coarse_profile(reference, resolution_m)
    arr = np.full((profile["height"], profile["width"]), np.nan, dtype=np.float32)
    reproject(
        source=rasterio.band(reference, 1),
        destination=arr,
        src_transform=reference.transform,
        src_crs=reference.crs,
        src_nodata=reference.nodata,
        dst_transform=profile["transform"],
        dst_crs=profile["crs"],
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return arr, profile


def slope_arrays(dtm: np.ndarray, transform: Affine) -> Tuple[np.ndarray, np.ndarray]:
    y_res = abs(transform.e)
    x_res = abs(transform.a)
    dzdy, dzdx = np.gradient(dtm, y_res, x_res)
    gradient = np.sqrt(dzdx * dzdx + dzdy * dzdy).astype(np.float32)
    slope_deg = np.degrees(np.arctan(gradient)).astype(np.float32)
    slope_percent = (gradient * 100.0).astype(np.float32)
    return slope_deg, slope_percent


def terrain_slope_on_reference(
    reference: DatasetReader, resolution_m: float
) -> Tuple[np.ndarray, np.ndarray]:
    coarse_dtm, coarse = reproject_dtm_to_resolution(reference, resolution_m)
    slope_deg_coarse, slope_percent_coarse = slope_arrays(coarse_dtm, coarse["transform"])
    target = {
        "height": reference.height,
        "width": reference.width,
        "transform": reference.transform,
        "crs": reference.crs,
    }
    slope_deg = reproject_array_between_grids(
        slope_deg_coarse,
        coarse["transform"],
        coarse["crs"],
        target,
        resampling=Resampling.bilinear,
        dtype="float32",
        src_nodata=np.nan,
        dst_nodata=0.0,
    )
    slope_percent = reproject_array_between_grids(
        slope_percent_coarse,
        coarse["transform"],
        coarse["crs"],
        target,
        resampling=Resampling.bilinear,
        dtype="float32",
        src_nodata=np.nan,
        dst_nodata=0.0,
    )
    return slope_deg, slope_percent


def smoothed_dtm_and_slope(
    reference: DatasetReader, *, sigma_pixels: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtm = reference.read(1).astype(np.float32)
    valid = valid_mask_from_dtm(dtm, reference.nodata)
    filled = fill_invalid_nearest(dtm, valid)
    if sigma_pixels > 0:
        filled = gaussian_filter(filled, sigma=sigma_pixels).astype(np.float32)
    y_res = abs(reference.transform.e)
    x_res = abs(reference.transform.a)
    dzdy, dzdx = np.gradient(filled, y_res, x_res)
    gradient = np.sqrt(dzdx * dzdx + dzdy * dzdy).astype(np.float32)
    slope_deg = np.degrees(np.arctan(gradient)).astype(np.float32)
    slope_percent = (gradient * 100.0).astype(np.float32)
    return filled, valid, slope_deg, slope_percent


def rasterize_vector_values(
    vector_path: Path,
    reference: DatasetReader,
    *,
    field: str,
    mapping: Mapping[str, float],
    fill: float = 0.0,
    dtype: str = "float32",
    all_touched: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    gdf = gpd.read_file(vector_path)
    bbox_gdf = gpd.GeoDataFrame(geometry=[box(*reference.bounds)], crs=reference.crs)
    clipped = gpd.overlay(gdf, bbox_gdf.to_crs(gdf.crs), how="intersection")
    if clipped.empty:
        arr = np.full((reference.height, reference.width), fill, dtype=dtype)
        return arr, {"feature_count": 0, "values": []}

    clipped = clipped.to_crs(reference.crs)
    shapes = []
    values = []
    unknown = []
    for _, row in clipped.iterrows():
        key = str(row[field]).strip()
        if key not in mapping:
            unknown.append(key)
            value = fill
        else:
            value = mapping[key]
        shapes.append((row.geometry, value))
        values.append(key)
    arr = features.rasterize(
        shapes,
        out_shape=(reference.height, reference.width),
        transform=reference.transform,
        fill=fill,
        dtype=dtype,
        all_touched=all_touched,
    )
    return arr, {
        "feature_count": int(len(clipped)),
        "values": sorted(set(values)),
        "unknown_values": sorted(set(unknown)),
    }


def reproject_raster_to_reference(
    src_path_or_url: str | Path,
    reference: DatasetReader,
    *,
    resampling: Resampling,
    dtype: str,
    src_nodata: Optional[float] = None,
    dst_nodata: float = 0.0,
) -> np.ndarray:
    dst = np.full((reference.height, reference.width), dst_nodata, dtype=dtype)
    with rasterio.Env(CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with rasterio.open(src_path_or_url) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata if src_nodata is not None else src.nodata,
                dst_transform=reference.transform,
                dst_crs=reference.crs,
                dst_nodata=dst_nodata,
                resampling=resampling,
            )
    return dst


def reproject_multiband_to_reference(
    src_path: Path,
    reference: DatasetReader,
    *,
    resampling: Resampling,
    dtype: str,
    dst_nodata: float,
) -> np.ndarray:
    with rasterio.open(src_path) as src:
        dst = np.full((src.count, reference.height, reference.width), dst_nodata, dtype=dtype)
        for band_idx in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band_idx),
                destination=dst[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=reference.transform,
                dst_crs=reference.crs,
                dst_nodata=dst_nodata,
                resampling=resampling,
            )
    return dst


def classify_ibge_slope(slope_percent: np.ndarray, valid: np.ndarray) -> np.ndarray:
    note = np.zeros(slope_percent.shape, dtype=np.float32)
    note[(slope_percent >= 0) & (slope_percent < 3)] = 1.0
    note[(slope_percent >= 3) & (slope_percent < 8)] = 3.0
    note[(slope_percent >= 8) & (slope_percent < 20)] = 5.0
    note[(slope_percent >= 20) & (slope_percent < 45)] = 8.0
    note[(slope_percent >= 45) & (slope_percent < 75)] = 9.0
    note[slope_percent >= 75] = 10.0
    note[~valid] = 0.0
    return note


def classify_pluviosity(rain_mm: np.ndarray, valid: np.ndarray) -> np.ndarray:
    note = np.zeros(rain_mm.shape, dtype=np.float32)
    note[(rain_mm >= 400) & (rain_mm < 1000)] = 4.0
    note[(rain_mm >= 1000) & (rain_mm < 1500)] = 6.0
    note[(rain_mm >= 1500) & (rain_mm < 2000)] = 8.0
    note[(rain_mm >= 2000) & (rain_mm < 2500)] = 9.0
    note[(rain_mm >= 2500) & (rain_mm <= 4300)] = 10.0
    note[~valid] = 0.0
    return note


def map_mapbiomas_to_ibge_notes(raw: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    notes = np.zeros(raw.shape, dtype=np.float32)
    unknown = []
    for code in np.unique(raw):
        code_int = int(code)
        if code_int == 0:
            continue
        if code_int in MAPBIOMAS_IBGE_LAND_USE_NOTES:
            notes[raw == code_int] = MAPBIOMAS_IBGE_LAND_USE_NOTES[code_int]
        else:
            unknown.append(code_int)
    notes[~valid] = 0.0
    return notes, unknown


def custom_lulc_land_use_mapping() -> Dict[int, float]:
    from IBGE_method.own_LULC.lulc_inputs import class_definitions

    return {
        int(class_value): float(definition["ibge_land_use_note"])
        for class_value, definition in class_definitions.items()
    }


def custom_lulc_output_path() -> Path:
    selected_path = ROOT / "IBGE_method" / "own_LULC" / "outputs" / "selected_experiment.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        promoted = selected.get("promoted_lulc")
        if promoted:
            return Path(promoted)
    from IBGE_method.own_LULC.implementation.config import load_config, output_path

    config = load_config()
    return output_path(config, "lulc_filename")


def map_custom_lulc_to_ibge_notes(
    raw: np.ndarray, valid: np.ndarray, mapping: Mapping[int, float]
) -> Tuple[np.ndarray, List[int]]:
    notes = np.zeros(raw.shape, dtype=np.float32)
    unknown = []
    for code in np.unique(raw):
        code_int = int(code)
        if code_int == 0:
            continue
        if code_int in mapping:
            notes[raw == code_int] = float(mapping[code_int])
        else:
            unknown.append(code_int)
    notes[~valid] = 0.0
    return notes, unknown


def ibge_land_use_source_metadata() -> Dict[str, Any]:
    custom_lulc = custom_lulc_output_path()
    if custom_lulc.exists():
        mapping = custom_lulc_land_use_mapping()
        return {
            "source": "Custom polygon-trained LULC",
            "path": str(custom_lulc),
            "class_to_ibge_note": {str(k): v for k, v in sorted(mapping.items())},
        }
    return {
        "source": "MapBiomas 10m collection 2 integration 2023",
        "url": MAPBIOMAS_10M_2023_URL,
    }


def load_ibge_land_use(
    reference: DatasetReader, valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, Any], str, str]:
    custom_lulc = custom_lulc_output_path()
    if custom_lulc.exists():
        raw = reproject_raster_to_reference(
            custom_lulc,
            reference,
            resampling=Resampling.nearest,
            dtype="uint8",
            dst_nodata=0,
        )
        mapping = custom_lulc_land_use_mapping()
        notes, unknown = map_custom_lulc_to_ibge_notes(raw, valid, mapping)
        return (
            raw,
            notes,
            unknown,
            ibge_land_use_source_metadata(),
            "ibge_land_use_custom_lulc.tif",
            "Custom polygon-trained LULC reprojected by nearest neighbor",
        )

    raw = reproject_raster_to_reference(
        MAPBIOMAS_10M_2023_URL,
        reference,
        resampling=Resampling.nearest,
        dtype="uint8",
        dst_nodata=0,
    )
    notes, unknown = map_mapbiomas_to_ibge_notes(raw, valid)
    return (
        raw,
        notes,
        unknown,
        ibge_land_use_source_metadata(),
        "ibge_land_use_mapbiomas_2023.tif",
        "MapBiomas 10m 2023 class reprojected by nearest neighbor",
    )


def classify_ibge_final(score_1_to_10: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    class5 = np.full(score_1_to_10.shape, 255, dtype=np.uint8)
    class5[(score_1_to_10 >= 0.0) & (score_1_to_10 < 3.5) & valid] = 1
    class5[(score_1_to_10 >= 3.5) & (score_1_to_10 < 4.5) & valid] = 2
    class5[(score_1_to_10 >= 4.5) & (score_1_to_10 < 5.5) & valid] = 3
    class5[(score_1_to_10 >= 5.5) & (score_1_to_10 < 6.5) & valid] = 4
    class5[(score_1_to_10 >= 6.5) & valid] = 5

    class3 = np.full(score_1_to_10.shape, 255, dtype=np.uint8)
    class3[np.isin(class5, [1, 2])] = 1
    class3[class5 == 3] = 2
    class3[np.isin(class5, [4, 5])] = 3
    return class5, class3


def generate_ibge_method(reference: DatasetReader, dirs: MethodDirs) -> Dict[str, Any]:
    dtm_valid = reference_valid_mask(reference)
    _slope_deg, slope_percent = terrain_slope_on_reference(reference, resolution_m=10.0)
    slope_note = classify_ibge_slope(slope_percent, dtm_valid)
    del slope_percent

    relief_note, relief_meta = rasterize_vector_values(
        SIG_PATHS["relief"],
        reference,
        field="Classe",
        mapping=RELIEF_IBGE_NOTES,
    )
    geology_note, geology_meta = rasterize_vector_values(
        SIG_PATHS["geology"],
        reference,
        field="SIGLA_UNID",
        mapping=GEOLOGY_IBGE_NOTES,
    )
    pedology_note, pedology_meta = rasterize_vector_values(
        SIG_PATHS["pedology"],
        reference,
        field="DESC_",
        mapping=PEDOLOGY_IBGE_NOTES,
    )

    rain = reproject_raster_to_reference(
        SIG_PATHS["rain_pma2"],
        reference,
        resampling=Resampling.bilinear,
        dtype="float32",
        dst_nodata=0.0,
    )
    pluviosity_note = classify_pluviosity(rain, dtm_valid)
    del rain

    (
        land_use_raw,
        land_use_note,
        unknown_land_use,
        land_use_source,
        land_use_output_name,
        land_use_description,
    ) = load_ibge_land_use(reference, dtm_valid)
    if unknown_land_use:
        raise RuntimeError(
            "The selected IBGE land-use source returned unmapped classes inside "
            f"the drone bbox: {unknown_land_use}. Add real class mappings before "
            "generating IBGE output."
        )

    valid = (
        dtm_valid
        & (slope_note > 0)
        & (relief_note > 0)
        & (geology_note > 0)
        & (pedology_note > 0)
        & (land_use_note > 0)
        & (pluviosity_note > 0)
    )

    score = (
        slope_note * IBGE_WEIGHTS["slope"]
        + relief_note * IBGE_WEIGHTS["geomorphology"]
        + geology_note * IBGE_WEIGHTS["geology"]
        + pedology_note * IBGE_WEIGHTS["pedology"]
        + land_use_note * IBGE_WEIGHTS["land_use"]
        + pluviosity_note * IBGE_WEIGHTS["pluviosity"]
    ).astype(np.float32)
    score[~valid] = -9999.0
    susceptibility = np.where(valid, score / 10.0, -9999.0).astype(np.float32)
    class5, class3 = classify_ibge_final(score, valid)

    write_single_band(
        dirs.outputs / "ibge_susceptibility_score.tif",
        susceptibility,
        reference,
        dtype="float32",
        nodata=-9999.0,
        description="IBGE adapted weighted susceptibility score 0-1",
    )
    write_single_band(
        dirs.outputs / "ibge_class_map_5class.tif",
        class5,
        reference,
        dtype="uint8",
        nodata=255,
        description="IBGE adapted classes: 1 very low to 5 very high",
    )
    write_single_band(
        dirs.outputs / "ibge_class_map_3class.tif",
        class3,
        reference,
        dtype="uint8",
        nodata=255,
        description="IBGE adapted classes collapsed to 1 low, 2 medium, 3 high",
    )
    write_single_band(
        dirs.outputs / "ibge_valid_mask.tif",
        valid.astype(np.uint8),
        reference,
        dtype="uint8",
        nodata=0,
        description="Valid pixels for IBGE adapted method",
    )
    write_single_band(
        dirs.outputs / land_use_output_name,
        land_use_raw,
        reference,
        dtype="uint8",
        nodata=0,
        description=land_use_description,
    )

    config = {
        "method": "High-resolution local adaptation of IBGE weighted susceptibility",
        "reference_grid": reference_summary(reference),
        "weights": IBGE_WEIGHTS,
        "slope_classes_percent_to_note": [
            [0, 3, 1],
            [3, 8, 3],
            [8, 20, 5],
            [20, 45, 8],
            [45, 75, 9],
            [75, None, 10],
        ],
        "classification_breaks_1_to_10": [0, 3.5, 4.5, 5.5, 6.5, 10],
        "land_use_source": land_use_source,
        "terrain_analysis_resolution_m": 10.0,
        "note": (
            "This is not the strict national IBGE 1 km statistical-grid product; "
            "it applies the IBGE thematic scoring logic on the drone footprint. "
            "Slope is derived from the real drone DTM on a 10 m analysis grid to "
            "avoid centimeter-scale surface roughness dominating the IBGE weights."
        ),
    }
    write_json(dirs.configs / "method_config.json", config)

    summary = {
        "outputs": output_listing(dirs.outputs),
        "valid_fraction": fraction(valid),
        "theme_metadata": {
            "relief": relief_meta,
            "geology": geology_meta,
            "pedology": pedology_meta,
            "land_use": land_use_source,
            "land_use_classes": sorted(int(x) for x in np.unique(land_use_raw) if int(x) != 0),
        },
        "class_distribution": class_distribution(class3, reference),
    }
    write_json(dirs.reports / "summary.json", summary)
    write_method_report(
        dirs.reports / "method_report.md",
        "IBGE Adapted Method",
        config,
        summary,
    )
    return summary


def reference_summary(reference: DatasetReader) -> Dict[str, Any]:
    return {
        "path": str(reference.name),
        "crs": str(reference.crs),
        "width": int(reference.width),
        "height": int(reference.height),
        "resolution": [float(reference.res[0]), float(reference.res[1])],
        "bounds": [float(x) for x in reference.bounds],
        "area_km2": float((reference.bounds.right - reference.bounds.left) * (reference.bounds.top - reference.bounds.bottom) / 1_000_000.0),
    }


def fraction(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask) / mask.size)


def output_listing(outputs_dir: Path) -> List[str]:
    return sorted(str(path.relative_to(ROOT)) for path in outputs_dir.glob("*.tif"))


def area_per_pixel(reference: DatasetReader) -> float:
    return abs(float(reference.transform.a * reference.transform.e))


def class_distribution(class_map: np.ndarray, reference: DatasetReader) -> Dict[str, Any]:
    valid = class_map != 255
    pixel_area = area_per_pixel(reference)
    total_area = float(np.count_nonzero(valid) * pixel_area)
    result: Dict[str, Any] = {"total_valid_area_m2": total_area, "classes": {}}
    for cls in sorted(int(v) for v in np.unique(class_map[valid])) if np.any(valid) else []:
        area = float(np.count_nonzero(class_map == cls) * pixel_area)
        result["classes"][str(cls)] = {
            "pixels": int(np.count_nonzero(class_map == cls)),
            "area_m2": area,
            "fraction": float(area / total_area) if total_area else 0.0,
        }
    return result


def write_method_report(
    path: Path,
    title: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    content = [
        f"# {title}",
        "",
        "Generated by `.venv/bin/python manage.py three-methods`.",
        "",
        "## Important caveat",
        str(config.get("note", "No caveat recorded.")),
        "",
        "## Outputs",
    ]
    content.extend(f"- `{item}`" for item in summary.get("outputs", []))
    content.extend(
        [
            "",
            "## Class distribution",
            "See `summary.json` for exact areas and fractions.",
            "",
        ]
    )
    write_text(path, "\n".join(content))


def copy_or_reproject_dl_outputs(reference: DatasetReader, dirs: MethodDirs) -> Dict[str, Any]:
    missing = [str(path) for path in DL_SOURCE_OUTPUTS.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            "Existing DL inference products are missing; run the main pipeline first: "
            + ", ".join(missing)
        )

    products = {
        "susceptibility": ("dl_susceptibility.tif", Resampling.bilinear, "float32", 0.0),
        "susceptibility_high": ("dl_susceptibility_high.tif", Resampling.bilinear, "float32", 0.0),
        "uncertainty": ("dl_uncertainty.tif", Resampling.bilinear, "float32", 0.0),
        "valid_mask": ("dl_valid_mask.tif", Resampling.nearest, "uint8", 0),
        "class_map": ("dl_class_map_model_encoding.tif", Resampling.nearest, "uint8", 255),
    }
    for key, (name, resampling, dtype, nodata) in products.items():
        data = reproject_raster_to_reference(
            DL_SOURCE_OUTPUTS[key],
            reference,
            resampling=resampling,
            dtype=dtype,
            dst_nodata=nodata,
        )
        write_single_band(
            dirs.outputs / name,
            data,
            reference,
            dtype=dtype,
            nodata=nodata,
            description=f"DL {key} reprojected to drone reference grid",
        )
        if key == "class_map":
            class3 = np.full(data.shape, 255, dtype=np.uint8)
            class3[data == 0] = 1
            class3[data == 1] = 2
            class3[data == 2] = 3
            write_single_band(
                dirs.outputs / "dl_class_map_3class.tif",
                class3,
                reference,
                dtype="uint8",
                nodata=255,
                description="DL class map converted to 1 low, 2 medium, 3 high",
            )

    class_probs = reproject_multiband_to_reference(
        DL_SOURCE_OUTPUTS["class_probabilities"],
        reference,
        resampling=Resampling.bilinear,
        dtype="float32",
        dst_nodata=0.0,
    )
    write_multiband(
        dirs.outputs / "dl_class_probabilities.tif",
        class_probs,
        reference,
        dtype="float32",
        nodata=0.0,
        descriptions=["P(low)", "P(medium)", "P(high)"],
    )

    source_metadata = ROOT / "outputs" / "test_feature_metadata.json"
    if source_metadata.exists():
        shutil.copyfile(source_metadata, dirs.reports / "source_feature_metadata.json")

    config = {
        "method": "Existing deep-learning inference products reprojected to the exact drone DTM grid",
        "reference_grid": reference_summary(reference),
        "source_outputs": {key: str(path) for key, path in DL_SOURCE_OUTPUTS.items()},
        "note": (
            "This packages the existing DL inference products on the drone footprint. "
            "No retraining was run in this step; augmentation and SMOTE remain training-time choices."
        ),
    }
    write_json(dirs.configs / "method_config.json", config)

    with rasterio.open(dirs.outputs / "dl_class_map_3class.tif") as src:
        class3 = src.read(1)
    summary = {
        "outputs": output_listing(dirs.outputs),
        "class_distribution": class_distribution(class3, reference),
        "valid_fraction": float(np.mean(class3 != 255)),
    }
    write_json(dirs.reports / "summary.json", summary)
    write_method_report(dirs.reports / "method_report.md", "Deep Learning Method", config, summary)
    return summary


def coarse_profile(reference: DatasetReader, resolution_m: float) -> Dict[str, Any]:
    left, bottom, right, top = reference.bounds
    width = int(math.ceil((right - left) / resolution_m))
    height = int(math.ceil((top - bottom) / resolution_m))
    transform = from_bounds(left, bottom, right, top, width, height)
    return {
        "crs": reference.crs,
        "transform": transform,
        "width": width,
        "height": height,
    }


def reproject_array_between_grids(
    source: np.ndarray,
    src_transform: Affine,
    src_crs: Any,
    dst_profile: Mapping[str, Any],
    *,
    resampling: Resampling,
    dtype: str,
    src_nodata: Optional[float] = None,
    dst_nodata: float = 0.0,
) -> np.ndarray:
    dst = np.full((int(dst_profile["height"]), int(dst_profile["width"])), dst_nodata, dtype=dtype)
    reproject(
        source=source,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    return dst


def coarse_vector_score(
    vector_path: Path,
    coarse: Mapping[str, Any],
    *,
    field: str,
    mapping: Mapping[str, float],
) -> np.ndarray:
    gdf = gpd.read_file(vector_path).to_crs(coarse["crs"])
    left, bottom, right, top = rasterio.transform.array_bounds(
        int(coarse["height"]), int(coarse["width"]), coarse["transform"]
    )
    clipped = gpd.overlay(
        gdf,
        gpd.GeoDataFrame(geometry=[box(left, bottom, right, top)], crs=coarse["crs"]),
        how="intersection",
    )
    shapes = []
    for _, row in clipped.iterrows():
        value = mapping.get(str(row[field]).strip(), 0.0)
        shapes.append((row.geometry, value))
    if not shapes:
        return np.zeros((int(coarse["height"]), int(coarse["width"])), dtype=np.float32)
    return features.rasterize(
        shapes,
        out_shape=(int(coarse["height"]), int(coarse["width"])),
        transform=coarse["transform"],
        fill=0.0,
        dtype="float32",
        all_touched=True,
    )


def lineament_density_score(coarse: Mapping[str, Any]) -> np.ndarray:
    gdf = gpd.read_file(SIG_PATHS["lineaments"]).to_crs(coarse["crs"])
    left, bottom, right, top = rasterio.transform.array_bounds(
        int(coarse["height"]), int(coarse["width"]), coarse["transform"]
    )
    clipped = gpd.overlay(
        gdf,
        gpd.GeoDataFrame(geometry=[box(left, bottom, right, top)], crs=coarse["crs"]),
        how="intersection",
    )
    if clipped.empty:
        return np.full((int(coarse["height"]), int(coarse["width"])), 0.2, dtype=np.float32)
    line_pixels = features.rasterize(
        [(geom, 1) for geom in clipped.geometry],
        out_shape=(int(coarse["height"]), int(coarse["width"])),
        transform=coarse["transform"],
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    density = gaussian_filter(line_pixels.astype(np.float32), sigma=3.0)
    if float(density.max()) <= 0.0:
        return np.full(density.shape, 0.2, dtype=np.float32)
    q95 = np.percentile(density[density > 0], 95)
    scaled = np.clip(density / max(q95, 1e-6), 0.0, 1.0)
    return (0.2 + 0.8 * scaled).astype(np.float32)


def drainage_proximity_score(coarse_dtm: np.ndarray, valid: np.ndarray, resolution_m: float) -> np.ndarray:
    # A real-data drainage proxy from the drone DTM: negative TPI marks local valley axes.
    smoothed = gaussian_filter(coarse_dtm.astype(np.float32), sigma=1.0)
    mean = uniform_filter(smoothed, size=9, mode="nearest")
    tpi = smoothed - mean
    valley_threshold = np.percentile(tpi[valid], 20) if np.any(valid) else np.nan
    valley = (tpi <= valley_threshold) & valid
    if not np.any(valley):
        return np.full(coarse_dtm.shape, 0.2, dtype=np.float32)
    distance_m = distance_transform_edt(~valley) * resolution_m
    return np.clip(1.0 - (distance_m / 100.0), 0.2, 1.0).astype(np.float32)


def curvature_score_from_grid(dtm: np.ndarray, valid: np.ndarray, transform: Affine) -> np.ndarray:
    y_res = abs(transform.e)
    x_res = abs(transform.a)
    dzdy, dzdx = np.gradient(dtm, y_res, x_res)
    d2zdy2, _ = np.gradient(dzdy, y_res, x_res)
    _, d2zdx2 = np.gradient(dzdx, y_res, x_res)
    curvature = (d2zdx2 + d2zdy2).astype(np.float32)
    score = np.full(curvature.shape, 0.5, dtype=np.float32)
    if np.any(valid):
        lo, hi = np.percentile(curvature[valid], [33, 67])
        score[curvature <= lo] = 0.8
        score[curvature >= hi] = 0.35
    score[~valid] = 0.0
    return score


def sgb_slope_score(slope_deg: np.ndarray, valid: np.ndarray) -> np.ndarray:
    score = np.full(slope_deg.shape, 0.2, dtype=np.float32)
    score[(slope_deg >= 10) & (slope_deg < 25)] = 0.6
    score[(slope_deg >= 25) & (slope_deg < 50)] = 0.9
    score[slope_deg >= 50] = 1.0
    score[~valid] = 0.0
    return score


def classify_sgb_score(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(score.shape, 255, dtype=np.uint8)
    out[(score < 0.45) & valid] = 1
    out[(score >= 0.45) & (score < 0.68) & valid] = 2
    out[(score >= 0.68) & valid] = 3
    return out


def generate_sgb_method(reference: DatasetReader, dirs: MethodDirs) -> Dict[str, Any]:
    dtm_valid = reference_valid_mask(reference)
    slope_deg, _slope_percent = terrain_slope_on_reference(reference, resolution_m=5.0)
    score = sgb_slope_score(slope_deg, dtm_valid) * SGB_WEIGHTS["slope"]
    del slope_deg

    coarse_resolution = 5.0
    coarse_dtm, coarse = reproject_dtm_to_resolution(reference, coarse_resolution)
    coarse_valid = np.isfinite(coarse_dtm)
    coarse_curvature = curvature_score_from_grid(
        coarse_dtm, coarse_valid, coarse["transform"]
    )
    coarse_drainage = drainage_proximity_score(coarse_dtm, coarse_valid, coarse_resolution)
    coarse_lineament = lineament_density_score(coarse)
    coarse_relief = coarse_vector_score(
        SIG_PATHS["relief"], coarse, field="Classe", mapping=RELIEF_SGB_SCORES
    )
    coarse_geology = coarse_vector_score(
        SIG_PATHS["geology"], coarse, field="SIGLA_UNID", mapping=GEOLOGY_SGB_SCORES
    )
    coarse_pedology = coarse_vector_score(
        SIG_PATHS["pedology"], coarse, field="DESC_", mapping=PEDOLOGY_SGB_SCORES
    )

    curvature_score = reproject_array_between_grids(
        coarse_curvature,
        coarse["transform"],
        coarse["crs"],
        {"height": reference.height, "width": reference.width, "transform": reference.transform, "crs": reference.crs},
        resampling=Resampling.bilinear,
        dtype="float32",
    )
    score += curvature_score * SGB_WEIGHTS["curvature"]
    del curvature_score
    drainage_score = reproject_array_between_grids(
        coarse_drainage,
        coarse["transform"],
        coarse["crs"],
        {"height": reference.height, "width": reference.width, "transform": reference.transform, "crs": reference.crs},
        resampling=Resampling.bilinear,
        dtype="float32",
    )
    score += drainage_score * SGB_WEIGHTS["drainage"]
    del drainage_score
    lineament_score = reproject_array_between_grids(
        coarse_lineament,
        coarse["transform"],
        coarse["crs"],
        {"height": reference.height, "width": reference.width, "transform": reference.transform, "crs": reference.crs},
        resampling=Resampling.bilinear,
        dtype="float32",
    )
    score += lineament_score * SGB_WEIGHTS["lineament_density"]
    del lineament_score
    relief_score = reproject_array_between_grids(
        coarse_relief,
        coarse["transform"],
        coarse["crs"],
        {"height": reference.height, "width": reference.width, "transform": reference.transform, "crs": reference.crs},
        resampling=Resampling.nearest,
        dtype="float32",
    )
    score += relief_score * SGB_WEIGHTS["relief"]
    geology_score = reproject_array_between_grids(
        coarse_geology,
        coarse["transform"],
        coarse["crs"],
        {"height": reference.height, "width": reference.width, "transform": reference.transform, "crs": reference.crs},
        resampling=Resampling.nearest,
        dtype="float32",
    )
    score += geology_score * SGB_WEIGHTS["geology"]
    pedology_score = reproject_array_between_grids(
        coarse_pedology,
        coarse["transform"],
        coarse["crs"],
        {"height": reference.height, "width": reference.width, "transform": reference.transform, "crs": reference.crs},
        resampling=Resampling.nearest,
        dtype="float32",
    )
    score += pedology_score * SGB_WEIGHTS["pedology"]

    valid = (
        dtm_valid
        & (relief_score > 0)
        & (geology_score > 0)
        & (pedology_score > 0)
    )
    del relief_score, geology_score, pedology_score
    score = score.astype(np.float32)
    score[~valid] = -9999.0
    class_map = classify_sgb_score(score, valid)

    write_single_band(
        dirs.outputs / "sgb_susceptibility_score.tif",
        score,
        reference,
        dtype="float32",
        nodata=-9999.0,
        description="Deterministic SGB-style susceptibility score 0-1",
    )
    write_single_band(
        dirs.outputs / "sgb_class_map.tif",
        class_map,
        reference,
        dtype="uint8",
        nodata=255,
        description="SGB-style classes: 1 low, 2 medium, 3 high",
    )
    write_single_band(
        dirs.outputs / "sgb_valid_mask.tif",
        valid.astype(np.uint8),
        reference,
        dtype="uint8",
        nodata=0,
        description="Valid pixels for SGB-style method",
    )

    config = {
        "method": "Deterministic SGB-style reconstruction on the drone DTM bbox",
        "reference_grid": reference_summary(reference),
        "weights": SGB_WEIGHTS,
        "class_breaks": {"low_max": 0.45, "medium_max": 0.68},
        "coarse_factor_resolution_m": coarse_resolution,
        "slope_analysis_resolution_m": 5.0,
        "note": (
            "This is a deterministic reconstruction because no landslide-scar polygon "
            "inventory intersects the drone bbox for formal ISD calibration. Terrain "
            "factors are derived from the real drone DTM on a 5 m analysis grid."
        ),
    }
    write_json(dirs.configs / "method_config.json", config)

    summary = {
        "outputs": output_listing(dirs.outputs),
        "valid_fraction": fraction(valid),
        "lineaments_in_bbox": int(len(gpd.read_file(SIG_PATHS["lineaments"]).to_crs(reference.crs).clip(box(*reference.bounds)))),
        "class_distribution": class_distribution(class_map, reference),
    }
    write_json(dirs.reports / "summary.json", summary)
    write_method_report(dirs.reports / "method_report.md", "SGB-Style Method", config, summary)
    return summary


def rasterize_official_sgb_reference(reference: DatasetReader, dirs: MethodDirs) -> Path:
    class_map = {"Baixa": 1, "Media": 2, "Alta": 3}
    gdf = gpd.read_file(SIG_PATHS["sgb_mass"])
    bbox_gdf = gpd.GeoDataFrame(geometry=[box(*reference.bounds)], crs=reference.crs)
    clipped = gpd.overlay(gdf, bbox_gdf.to_crs(gdf.crs), how="intersection").to_crs(reference.crs)
    arr = features.rasterize(
        [(row.geometry, class_map[str(row["Classe"]).strip()]) for _, row in clipped.iterrows()],
        out_shape=(reference.height, reference.width),
        transform=reference.transform,
        fill=255,
        dtype="uint8",
        all_touched=True,
    )
    out = dirs.outputs / "reference_official_sgb_class_map.tif"
    write_single_band(
        out,
        arr,
        reference,
        dtype="uint8",
        nodata=255,
        description="Official SGB mass-movement susceptibility clipped to drone grid",
    )
    return out


def reproject_ground_truth_reference(reference: DatasetReader, dirs: MethodDirs) -> Path:
    arr = reproject_raster_to_reference(
        GROUND_TRUTH_25M,
        reference,
        resampling=Resampling.nearest,
        dtype="uint8",
        dst_nodata=0,
    )
    class_map = np.full(arr.shape, 255, dtype=np.uint8)
    class_map[arr == 1] = 1
    class_map[arr == 2] = 2
    class_map[arr == 3] = 3
    out = dirs.outputs / "reference_ground_truth_25m_class_map.tif"
    write_single_band(
        out,
        class_map,
        reference,
        dtype="uint8",
        nodata=255,
        description="25m ground truth clipped/reprojected by nearest neighbor to drone grid",
    )
    return out


def confusion_matrix(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    labels: Sequence[int] = (1, 2, 3),
) -> Dict[str, Any]:
    mask = (pred != 255) & (ref != 255)
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for label_pred in labels:
        for label_ref in labels:
            matrix[label_to_idx[label_ref], label_to_idx[label_pred]] = int(
                np.count_nonzero(mask & (ref == label_ref) & (pred == label_pred))
            )
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    return {
        "labels_order": list(labels),
        "rows_reference_columns_prediction": matrix.tolist(),
        "pixels_compared": total,
        "overall_accuracy": accuracy,
    }


def validate_methods(reference: DatasetReader, dirs: Dict[str, MethodDirs]) -> Dict[str, Any]:
    sgb_ref_path = rasterize_official_sgb_reference(reference, dirs["sgb"])
    gt_ref_path = reproject_ground_truth_reference(reference, dirs["sgb"])

    predictions = {
        "dl": dirs["dl"].outputs / "dl_class_map_3class.tif",
        "ibge": dirs["ibge"].outputs / "ibge_class_map_3class.tif",
        "sgb": dirs["sgb"].outputs / "sgb_class_map.tif",
    }
    with rasterio.open(sgb_ref_path) as src:
        sgb_ref = src.read(1)
    with rasterio.open(gt_ref_path) as src:
        gt_ref = src.read(1)

    report: Dict[str, Any] = {
        "reference_outputs": {
            "official_sgb": str(sgb_ref_path.relative_to(ROOT)),
            "ground_truth_25m": str(gt_ref_path.relative_to(ROOT)),
        },
        "methods": {},
    }
    for name, path in predictions.items():
        with rasterio.open(path) as src:
            pred = src.read(1)
        report["methods"][name] = {
            "prediction": str(path.relative_to(ROOT)),
            "class_distribution": class_distribution(pred, reference),
            "against_official_sgb": confusion_matrix(pred, sgb_ref),
            "against_ground_truth_25m": confusion_matrix(pred, gt_ref),
        }

    for group in dirs.values():
        write_json(group.reports / "validation.json", report)
    write_text(
        dirs["sgb"].reports / "comparison_report.md",
        comparison_report_markdown(report),
    )
    return report


def comparison_report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Three-Method Validation Summary",
        "",
        "Reference layers are real data clipped/reprojected to the drone grid.",
        "",
        "## References",
    ]
    refs = report.get("reference_outputs", {})
    for key, value in refs.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Overall accuracy against references"])
    for method, payload in report.get("methods", {}).items():
        sgb_acc = payload["against_official_sgb"]["overall_accuracy"]
        gt_acc = payload["against_ground_truth_25m"]["overall_accuracy"]
        lines.append(f"- {method}: SGB={sgb_acc:.4f}, GT25m={gt_acc:.4f}")
    lines.append("")
    lines.append("See `validation.json` for confusion matrices and class areas.")
    return "\n".join(lines)


def write_provenance(reference: DatasetReader, dirs: Dict[str, MethodDirs], validation: Mapping[str, Any]) -> None:
    provenance = {
        "reference_grid": reference_summary(reference),
        "drone_inputs": {
            "dtm": str(DRONE_DTM),
            "orthophoto": str(DRONE_ORTHO),
        },
        "municipal_sig_root": str(SIG_ROOT),
        "source_layers": {key: str(path) for key, path in SIG_PATHS.items()},
        "ground_truth_25m": str(GROUND_TRUTH_25M),
        "land_use": ibge_land_use_source_metadata(),
        "official_references": {
            "ibge_susceptibility": "https://biblioteca.ibge.gov.br/index.php/biblioteca-catalogo?id=2101684&view=detalhes",
            "ibge_pedology_complement": "https://biblioteca.ibge.gov.br/index.php/biblioteca-catalogo?id=2102076&view=detalhes",
            "sgb_technical_note": "https://rigeo.sgb.gov.br/handle/doc/16588",
            "sgb_cachoeiro_package": "https://rigeo.sgb.gov.br/handle/doc/14892",
            "mapbiomas_10m": "https://brasil.mapbiomas.org/mapbiomas-cobertura-10m/",
        },
        "validation": validation,
        "no_synthetic_source_data": True,
        "augmentation_policy": "Training augmentation/SMOTE is allowed; fabricated source layers are not.",
    }
    for group in dirs.values():
        write_json(group.reports / "provenance.json", provenance)


def run(force: bool = False) -> Dict[str, Any]:
    for path in (DRONE_DTM, DRONE_ORTHO, GROUND_TRUTH_25M):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in SIG_PATHS.values():
        if not path.exists():
            raise FileNotFoundError(path)

    dirs = ensure_method_dirs()
    with rasterio.open(DRONE_DTM) as reference:
        if not force and all((group.reports / "summary.json").exists() for group in dirs.values()):
            print("Three-method outputs already exist; use --force to regenerate.")
        else:
            print("[three-methods] Generating DL method package...")
            dl_summary = copy_or_reproject_dl_outputs(reference, dirs["dl"])
            print("[three-methods] Generating IBGE adapted method...")
            ibge_summary = generate_ibge_method(reference, dirs["ibge"])
            print("[three-methods] Generating SGB-style method...")
            sgb_summary = generate_sgb_method(reference, dirs["sgb"])
            print("[three-methods] Validating methods against real references...")
            validation = validate_methods(reference, dirs)
            write_provenance(reference, dirs, validation)
            return {
                "dl": dl_summary,
                "ibge": ibge_summary,
                "sgb": sgb_summary,
                "validation": validation,
            }

        validation_path = dirs["sgb"].reports / "validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
        return {"validation": validation}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate method outputs")
    args = parser.parse_args(argv)
    run(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
